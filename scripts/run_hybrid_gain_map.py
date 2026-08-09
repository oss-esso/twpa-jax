"""Run a compact pump/gain map with bounded HB -> TD recovery.

This is the map-level client for the validated hybrid column controller.  A TD
PERIOD_1 waveform is allowed to contribute a gain-map point through its exact
Fourier projection when production HB Newton cannot recover the root.  Such a
point is marked ``pump_status=TD_PERIOD1`` and is never presented as an HB
convergence.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map  # noqa: E402
from scripts import run_hybrid_column as hybrid  # noqa: E402
from twpa_solver.hybrid_column import ColumnBudget, ColumnController  # noqa: E402
from twpa_solver.map import run_isolated_jobs  # noqa: E402


def build_args(ns: argparse.Namespace) -> argparse.Namespace:
    argv = [
        "--circuit-dir", str(ns.circuit_dir.resolve()),
        "--outdir", str(ns.outdir.resolve()),
        "--executor", "inprocess", "--mode", "warmstart",
        "--n-power", str(ns.n_power), "--n-frequency", str(ns.n_frequency),
        "--pump-power-min-dbm", str(ns.power_min_dbm),
        "--pump-power-max-dbm", str(ns.power_max_dbm),
        "--pump-freq-min-ghz", str(ns.freq_min_ghz),
        "--pump-freq-max-ghz", str(ns.freq_max_ghz),
        "--attenuation-db", str(ns.attenuation_db),
        "--pump-mode-count", str(ns.pump_mode_count), "--nt", str(ns.nt),
        "--pump-mode-policy", ns.pump_mode_policy,
        "--mixing-order", str(ns.mixing_order), "--harmonics", str(ns.harmonics),
        "--signal-detuning-mhz", str(ns.signal_detuning_mhz),
        "--no-signal-spectrum",
        "--signal-offset-count-per-side", str(ns.signal_offset_count_per_side),
        "--signal-offset-step-mhz", str(ns.signal_offset_step_mhz),
        "--signal-workers", "1",
        "--inproc-pump-backend", ns.inproc_pump_backend,
        "--inproc-preconditioner", ns.inproc_preconditioner,
        "--inproc-solve-deadline-s", str(ns.inproc_solve_deadline_s),
        "--inproc-max-newton", str(ns.inproc_max_newton),
        "--inproc-fail-fast", "--overwrite", "--log-level", "WARNING",
    ]
    for flag, value in (("--pump-port", ns.pump_port),
                        ("--source-port", ns.source_port),
                        ("--out-port", ns.out_port)):
        if value is not None:
            argv.extend([flag, str(value)])
    if ns.dc_branch_flux_over_phi0 is not None:
        argv.extend(["--dc-branch-flux-over-phi0", str(ns.dc_branch_flux_over_phi0)])
    parsed = run_gain_map.parse_args(argv)
    runtime_circuit = run_gain_map.load_circuit(parsed.circuit_dir)
    roles = run_gain_map.resolve_port_roles(
        runtime_circuit,
        pump_port=parsed.pump_port,
        source_port=parsed.source_port,
        out_port=parsed.out_port,
    )
    for role, port in roles.items():
        setattr(parsed, role, port)
    parsed.mixing_order = run_gain_map.resolve_mixing_order(
        parsed.mixing_order,
        dc_current_a=parsed.dc_current_a,
        dc_branch_flux_over_phi0=parsed.dc_branch_flux_over_phi0,
        dc_solution=parsed.dc_solution,
        design_meta=runtime_circuit.summary,
    )
    if ns.signal_ghz is not None:
        parsed.signal_ghz = float(ns.signal_ghz)
    parsed.workflow_mode = "slow"
    return parsed


def _td_skip_row(point: Any, status: str) -> dict[str, Any]:
    return {
        "point_index": point.index,
        "i_power": point.i_power,
        "j_freq": point.j_freq,
        "pump_power_dbm": point.power_dbm,
        "pump_freq_ghz": point.pump_freq_ghz,
        "pump_current_peak_a": point.current_a,
        "status": status,
        "pump_status": status,
        "gain_status": status,
        "pump_failure_reason": "not evaluated after TD physical boundary",
        "pump_dir": "",
        "elapsed_s": 0.0,
    }


def _rows_for_column(
    points: list[Any], result: Any, backend: hybrid.ProductionPeriodicBackend,
) -> list[dict[str, Any]]:
    records = {int(record.target.index): record for record in result.records}
    rows: list[dict[str, Any]] = []
    outside_seen = False
    skip_status = (
        "SKIP_AFTER_PHYSICAL_BOUNDARY"
        if result.status.value == "PHYSICAL_BOUNDARY_FOUND"
        else "SKIP_AFTER_COLUMN_FAILURE"
    )
    for point in points:
        record = records.get(int(point.index))
        if record is None:
            rows.append(_td_skip_row(point, skip_status))
            continue
        row = dict(backend.map_rows.get(int(point.index), {}))
        route = record.route.value
        row.update({
            "point_index": point.index,
            "i_power": point.i_power,
            "j_freq": point.j_freq,
            "pump_power_dbm": point.power_dbm,
            "pump_freq_ghz": point.pump_freq_ghz,
            "pump_current_peak_a": point.current_a,
            "hybrid_route": route,
            "hybrid_state": record.state.value,
            "hybrid_classification": record.classification,
            "td_periods": record.td_periods,
            "td_d1": record.d1,
            "td_best_low_order_dn": record.best_low_order_dn,
            "td_r_j": record.r_j,
            "td_phase_winding": record.phase_winding,
        })
        if record.state.value == "PHYSICAL_BOUNDARY_FOUND":
            row["status"] = "PHYSICAL_BOUNDARY"
            row["pump_status"] = "PHYSICAL_BOUNDARY"
            row["gain_status"] = "NOT_RUN"
            outside_seen = True
        elif record.state.value in {"COLUMN_NUMERICAL_FAILURE", "COLUMN_UNRESOLVED_BUDGET"}:
            row["status"] = record.state.value
            row["pump_status"] = record.state.value
            row["gain_status"] = "NOT_RUN"
        rows.append(row)
        if outside_seen:
            break
    if outside_seen:
        done = {int(row["point_index"]) for row in rows}
        rows.extend(
            _td_skip_row(point, skip_status)
            for point in points if int(point.index) not in done
        )
    return sorted(rows, key=lambda row: int(row["point_index"]))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    int_fields = {"point_index", "i_power", "j_freq", "td_periods"}
    float_fields = {
        "pump_power_dbm", "pump_freq_ghz", "pump_current_peak_a", "gain_db",
        "gain_vs_off_db", "gain_vs_pumpdiag_db", "signal_ghz",
        "linear_rel_residual", "pump_coeff_rel", "pump_time_rel", "elapsed_s",
        "pump_runtime_s", "pump_wall_runtime_s", "gain_total_runtime_s",
        "td_d1", "td_best_low_order_dn", "td_r_j", "td_phase_winding",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    row[key] = None
                elif key in int_fields:
                    row[key] = int(float(value))
                elif key in float_fields:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _child_arguments(ns: argparse.Namespace, frequency: float, outdir: Path) -> list[str]:
    return [
        "--circuit-dir", str(ns.circuit_dir), "--outdir", str(outdir),
        "--n-power", str(ns.n_power), "--n-frequency", "1",
        "--power-min-dbm", str(ns.power_min_dbm),
        "--power-max-dbm", str(ns.power_max_dbm),
        "--freq-min-ghz", f"{frequency:.12g}",
        "--freq-max-ghz", f"{frequency:.12g}",
        "--pump-mode-count", str(ns.pump_mode_count), "--nt", str(ns.nt),
        "--signal-detuning-mhz", str(ns.signal_detuning_mhz),
        "--signal-offset-count-per-side", str(ns.signal_offset_count_per_side),
        "--signal-offset-step-mhz", str(ns.signal_offset_step_mhz),
        "--inproc-pump-backend", ns.inproc_pump_backend,
        "--inproc-preconditioner", ns.inproc_preconditioner,
        "--inproc-solve-deadline-s", str(ns.inproc_solve_deadline_s),
        "--inproc-max-newton", str(ns.inproc_max_newton),
        "--td-ramp-periods", str(ns.td_ramp_periods),
        "--td-hold-periods", str(ns.td_hold_periods),
        "--td-checkpoint-periods", str(ns.td_checkpoint_periods),
        "--max-td-bridges", str(ns.max_td_bridges),
        "--_single-column",
    ]


def run_isolated_columns(args: argparse.Namespace) -> int:
    """Run each frequency in a fresh process to bound allocator/cache growth."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    frequencies = np.linspace(args.freq_min_ghz, args.freq_max_ghz, args.n_frequency)
    powers = np.linspace(args.power_min_dbm, args.power_max_dbm, args.n_power)
    global_points, _, _ = run_gain_map.build_points(build_args(args))
    global_by_col = {
        int(point.j_freq): sorted(
            [p for p in global_points if int(p.j_freq) == int(point.j_freq)],
            key=lambda p: p.i_power,
        )
        for point in global_points
    }
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    worker_peak_rss: list[int] = []
    child_elapsed_total = 0.0
    started = time.perf_counter()
    def run_child(item: tuple[int, float]) -> tuple[int, float, Path, int]:
        j, frequency = item
        child_dir = args.outdir / "columns" / f"f_{j:03d}_{frequency:.6f}ghz"
        child_dir.mkdir(parents=True, exist_ok=True)
        child_csv = child_dir / "map_points.csv"
        child_summary = child_dir / "map_summary.json"
        complete = False
        if child_csv.exists() and child_summary.exists():
            try:
                complete = len(_read_csv_rows(child_csv)) == int(args.n_power)
            except (OSError, csv.Error, ValueError):
                complete = False
        if not complete:
            log = (child_dir / "run.log").open("w", encoding="utf-8")
            err = (child_dir / "run.err.log").open("w", encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve()), *_child_arguments(args, float(frequency), child_dir)]
            proc = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=err, check=False)
            log.close(); err.close()
            if proc.returncode != 0 or not child_csv.exists():
                (child_dir / "worker_error.json").write_text(
                    json.dumps({"returncode": proc.returncode}), encoding="utf-8"
                )
        return j, float(frequency), child_dir, int(child_csv.exists())

    jobs = list(enumerate(frequencies))
    workers = max(1, min(int(args.frequency_workers), len(jobs)))
    completed: list[tuple[int, float, Path, int]] = run_isolated_jobs(
        jobs, run_child, workers
    )
    completed.sort(key=lambda item: item[0])

    for j, frequency, child_dir, has_csv in completed:
        child_csv = child_dir / "map_points.csv"
        if not has_csv:
            raise RuntimeError(
                f"hybrid frequency column {j} failed; see {child_dir / 'run.err.log'}"
            )
        child_rows = _read_csv_rows(child_csv)
        targets = global_by_col[j]
        if len(child_rows) != len(targets):
            raise RuntimeError(f"column {j} returned {len(child_rows)} rows, expected {len(targets)}")
        for row, target in zip(child_rows, targets):
            row["point_index"] = int(target.index)
            row["i_power"] = int(target.i_power)
            row["j_freq"] = int(target.j_freq)
            row["pump_power_dbm"] = float(target.power_dbm)
            row["pump_freq_ghz"] = float(target.pump_freq_ghz)
            row["pump_current_peak_a"] = float(target.current_a)
        all_rows.extend(child_rows)
        child_summary = child_dir / "hybrid_columns.json"
        if child_summary.exists():
            child_data = json.loads(child_summary.read_text(encoding="utf-8"))
            for summary in child_data:
                summary = dict(summary)
                summary["j_freq"] = j
                for key in ("last_working", "first_outside"):
                    if summary.get(key) is not None:
                        summary[key] = int(targets[summary[key] % len(targets)].index)
                for record in summary.get("records", []):
                    record["point_index"] = int(
                        targets[int(record["point_index"]) % len(targets)].index
                    )
                summaries.append(summary)
            child_status = child_data[0].get("status") if child_data else None
            child_records = {
                int(record["point_index"]): record
                for record in (child_data[0].get("records", []) if child_data else [])
            }
            for row in child_rows:
                local = child_records.get(int(row["point_index"]))
                if local is not None:
                    row["hybrid_route"] = local.get("route")
                    row["hybrid_state"] = local.get("state")
                    row["hybrid_classification"] = local.get("classification")
                    row["td_periods"] = local.get("td_periods", 0)
                if (
                    child_status != "PHYSICAL_BOUNDARY_FOUND"
                    and row.get("status") == "SKIP_AFTER_PHYSICAL_BOUNDARY"
                ):
                    row["status"] = "SKIP_AFTER_COLUMN_FAILURE"
                    row["pump_status"] = "SKIP_AFTER_COLUMN_FAILURE"
                    row["gain_status"] = "SKIP_AFTER_COLUMN_FAILURE"
        child_map_summary = child_dir / "map_summary.json"
        if child_map_summary.exists():
            child_summary_payload = json.loads(child_map_summary.read_text(encoding="utf-8"))
            child_elapsed_total += float(child_summary_payload.get("elapsed_s", 0.0))
            if child_summary_payload.get("peak_rss_bytes") is not None:
                worker_peak_rss.append(int(child_summary_payload["peak_rss_bytes"]))
        print(json.dumps({"column": j, "frequency_ghz": float(frequency), "status": "complete"}), flush=True)

    all_rows.sort(key=lambda row: int(row["point_index"]))
    gain_args = build_args(args)
    gain_args.peak_rss_bytes = max(worker_peak_rss) if worker_peak_rss else None
    run_gain_map.write_points_csv(args.outdir / "map_points.csv", all_rows)
    gain = run_gain_map.gain_grid(all_rows, args.n_power, args.n_frequency)
    run_gain_map.write_arrays(
        args.outdir / "map_arrays.npz",
        np.asarray(powers), np.asarray(frequencies), {"gain_db": gain},
    )
    gate = run_gain_map.GateResult(
        evaluated=False, passed=False, reasons=["hybrid map gate not applicable"]
    )
    run_gain_map.write_summary(
        args.outdir, gain_args, [], all_rows, gate,
        child_elapsed_total or (time.perf_counter() - started),
    )
    (args.outdir / "hybrid_columns.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status_counts": {
            status: sum(1 for row in all_rows if row.get("status") == status)
            for status in sorted({row.get("status") for row in all_rows})
        },
        "pass_count": int(np.isfinite(gain).sum()),
        "elapsed_s": child_elapsed_total or (time.perf_counter() - started),
        "outdir": str(args.outdir),
    }, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-power", type=int, default=20)
    parser.add_argument("--n-frequency", type=int, default=20)
    parser.add_argument("--power-min-dbm", type=float, default=-26.0)
    parser.add_argument("--power-max-dbm", type=float, default=-16.0)
    parser.add_argument("--freq-min-ghz", type=float, default=7.6)
    parser.add_argument("--freq-max-ghz", type=float, default=7.85)
    parser.add_argument("--attenuation-db", type=float, default=0.0)
    parser.add_argument("--pump-mode-count", type=int, default=10)
    parser.add_argument("--pump-mode-policy", default="positive_odd_jc")
    parser.add_argument("--mixing-order", choices=("auto", "3", "4"), default="auto")
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--pump-port", type=int, default=None)
    parser.add_argument("--source-port", type=int, default=None)
    parser.add_argument("--out-port", type=int, default=None)
    parser.add_argument("--dc-branch-flux-over-phi0", type=float, default=None)
    parser.add_argument("--signal-ghz", type=float, default=None)
    parser.add_argument("--nt", type=int, default=40)
    parser.add_argument("--signal-detuning-mhz", type=float, default=500.0)
    parser.add_argument("--signal-offset-count-per-side", type=int, default=5)
    parser.add_argument("--signal-offset-step-mhz", type=float, default=500.0)
    parser.add_argument("--inproc-pump-backend", choices=["full", "schur_cpu_mt"], default="full")
    parser.add_argument("--inproc-preconditioner", default="real_coupled")
    parser.add_argument("--inproc-solve-deadline-s", type=float, default=14.0)
    parser.add_argument("--inproc-max-newton", type=int, default=16)
    parser.add_argument("--td-ramp-periods", type=int, default=10)
    parser.add_argument("--td-hold-periods", type=int, default=40)
    parser.add_argument("--td-checkpoint-periods", type=int, default=10)
    parser.add_argument("--max-td-bridges", type=int, default=2)
    parser.add_argument(
        "--frequency-workers", type=int, default=1,
        help="Independent frequency-column child processes (slow mode).",
    )
    parser.add_argument("--no-isolate-columns", dest="isolate_columns", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--_single-column", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(isolate_columns=True)
    args = parser.parse_args(argv)
    if args.isolate_columns and not args._single_column and args.n_frequency > 1:
        return run_isolated_columns(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    gain_args = build_args(args)
    # TD validation and branch transfer require the same float64 checkpoint
    # contract used by the validated hybrid-column runner.  The production map
    # can use float32 for ordinary HB-only storage, but not for a transient
    # anchor: quantization can turn a 1e-12 HB residual into a false fixture
    # failure.
    gain_args.pump_solution_dtype = "float64"
    gain_args.hybrid_compact_storage = True
    points, powers, freqs = run_gain_map.build_points(gain_args)
    by_freq: dict[int, list[Any]] = {}
    for point in points:
        by_freq.setdefault(int(point.j_freq), []).append(point)

    all_rows: list[dict[str, Any]] = []
    column_summaries: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for j in sorted(by_freq):
        column_points = sorted(by_freq[j], key=lambda point: point.power_dbm)
        freq = float(column_points[0].pump_freq_ghz)
        column_dir = args.outdir / "columns" / f"f_{j:03d}_{freq:.6f}ghz"
        column_dir.mkdir(parents=True, exist_ok=True)
        backend = hybrid.ProductionPeriodicBackend(
            gain_args, points, column_dir
        )
        h1_args = hybrid.h1.parse_args([])
        h1_args.pump_port = int(gain_args.pump_port)
        h1_args.dc_flux_over_phi0 = float(
            gain_args.dc_branch_flux_over_phi0
            if gain_args.dc_branch_flux_over_phi0 is not None else 0.0
        )
        dynamic = hybrid.H1DynamicBackend(
            args.circuit_dir, freq, h1_args,
            gain_args.pump_current_jc_scale,
            args.td_ramp_periods, args.td_hold_periods,
            args.td_checkpoint_periods,
        )
        result = ColumnController(
            backend, dynamic, column_dir,
            ColumnBudget(max_td_bridges=args.max_td_bridges),
        ).run(column_points)
        rows = _rows_for_column(column_points, result, backend)
        all_rows.extend(rows)
        column_summaries.append({
            "j_freq": j,
            "pump_freq_ghz": freq,
            "status": result.status.value,
            "td_bridges": result.td_bridges,
            "td_periods": result.td_periods,
            "hb_restart_successes": result.hb_restart_successes,
            "last_working": (
                int(result.last_working.target.index)
                if result.last_working is not None else None
            ),
            "first_outside": (
                int(result.first_outside.target.index)
                if result.first_outside is not None else None
            ),
            "physical_boundary_bracket": {
                "lower": (
                    float(result.last_working.target.power_dbm)
                    if result.last_working is not None else None
                ),
                "upper": (
                    float(result.first_outside.target.power_dbm)
                    if result.first_outside is not None else None
                ),
            },
            "records": [
                {
                    "point_index": int(record.target.index),
                    "route": record.route.value,
                    "state": record.state.value,
                    "classification": record.classification,
                    "td_periods": record.td_periods,
                }
                for record in result.records
            ],
        })
        print(json.dumps(column_summaries[-1]), flush=True)
        del backend, dynamic
        gc.collect()

    all_rows.sort(key=lambda row: int(row["point_index"]))
    run_gain_map.write_points_csv(args.outdir / "map_points.csv", all_rows)
    gain = run_gain_map.gain_grid(all_rows, args.n_power, args.n_frequency)
    run_gain_map.write_arrays(
        args.outdir / "map_arrays.npz", powers, freqs, {"gain_db": gain}
    )
    gate = run_gain_map.GateResult(evaluated=False, passed=False, reasons=["hybrid map gate not applicable"])
    run_gain_map.write_summary(
        args.outdir, gain_args, [], all_rows, gate,
        time.perf_counter() - total_started,
    )
    (args.outdir / "hybrid_columns.json").write_text(
        json.dumps(column_summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status_counts": {status: sum(1 for row in all_rows if row.get("status") == status)
                          for status in sorted({row.get("status") for row in all_rows})},
        "pass_count": int(np.isfinite(gain).sum()),
        "elapsed_s": time.perf_counter() - total_started,
        "outdir": str(args.outdir),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
