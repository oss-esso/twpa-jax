"""Experiment 16: IPM pump-power / pump-frequency gain map.

This is a thin orchestration layer around exp08 (pump HB solve) and exp09
(linearized gain from the solved pump).  The map axes are external pump power
and pump frequency; the injected pump current is computed with the old-IPM
source convention:

    source_power_dbm = external_power_dbm - attenuation_db
    I_peak = sqrt(2 * P_source_W / Z0)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def dbm_to_watts(power_dbm: float) -> float:
    return 1.0e-3 * 10.0 ** (float(power_dbm) / 10.0)


def old_ipm_peak_current_from_external_dbm(
    external_power_dbm: float,
    *,
    attenuation_db: float,
    z0_ohm: float,
) -> tuple[float, float]:
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    source_power_dbm = float(external_power_dbm) - float(attenuation_db)
    current = math.sqrt(2.0 * dbm_to_watts(source_power_dbm) / z0_ohm)
    return source_power_dbm, current


def slug_float(value: float) -> str:
    text = f"{value:.9g}".replace("-", "m").replace(".", "p")
    return text


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_command(cmd: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True)
    return int(proc.returncode)


def signal_ghz_for(pump_freq_ghz: float, args: argparse.Namespace) -> float:
    """Readout signal frequency: ws = wp - detuning (default 100 MHz) per cell,
    unless an explicit fixed --signal-ghz overrides it."""
    if getattr(args, "signal_ghz", None) is not None:
        return float(args.signal_ghz)
    return float(pump_freq_ghz) - float(args.signal_detuning_mhz) / 1000.0


def best_gain_row(gain_csv: Path) -> dict[str, Any] | None:
    if not gain_csv.exists():
        return None
    with gain_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = []
    for row in rows:
        try:
            row["_gain_db"] = float(row["gain_db"])
            valid.append(row)
        except Exception:
            pass
    if not valid:
        return None
    return max(valid, key=lambda r: r["_gain_db"])


def sum_report_field(report: dict[str, Any], key: str) -> float:
    return float(sum(float(r.get(key, 0.0) or 0.0) for r in report.get("reports", [])))


def point_status(pump_report: dict[str, Any] | None, gain_report: dict[str, Any] | None) -> tuple[str, str, str]:
    pump_status = "MISSING"
    if pump_report is not None:
        pump_status = str(pump_report.get("final_status", "UNKNOWN"))
    gain_status = "MISSING"
    if gain_report is not None:
        results = gain_report.get("results", [])
        if results:
            gain_status = "VALID_SOLVED" if all(r.get("status") == "VALID_SOLVED" for r in results) else "PARTIAL"
        else:
            gain_status = "EMPTY"
    ok = pump_status == "VALID_CONVERGED" and gain_status == "VALID_SOLVED"
    return ("VALID_CONVERGED" if ok else "ERROR", pump_status, gain_status)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=os.path.join("outputs", "ipm_gain_map_5x5_power_freq"))
    p.add_argument("--ipm-dir", default=os.path.join("outputs", "ipm_python_design"))
    p.add_argument("--n-power", type=int, default=5)
    p.add_argument("--n-frequency", type=int, default=5)
    p.add_argument("--pump-power-min-dbm", type=float, default=-30.0)
    p.add_argument("--pump-power-max-dbm", type=float, default=-20.0)
    p.add_argument("--pump-freq-min-ghz", type=float, default=6.0)
    p.add_argument("--pump-freq-max-ghz", type=float, default=8.0)
    p.add_argument("--attenuation-db", type=float, default=35.0)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    # Default: signal tracks the pump at ws = wp - 100 MHz per cell. Pass
    # --signal-ghz to force a fixed absolute signal.
    p.add_argument("--signal-ghz", type=float, default=None,
                   help="Fixed absolute signal frequency (GHz). If omitted, the "
                   "signal tracks each cell's pump at ws = wp - --signal-detuning-mhz.")
    p.add_argument("--signal-detuning-mhz", type=float, default=100.0,
                   help="Signal detuning below the pump when --signal-ghz is not "
                   "set: ws = wp - detuning (default 100 MHz).")
    p.add_argument("--gain-sweep", action="store_true", help="Run exp09 signal sweep instead of one signal point.")
    p.add_argument("--signal-start-ghz", type=float, default=4.0)
    p.add_argument("--signal-stop-ghz", type=float, default=8.0)
    p.add_argument("--signal-points", type=int, default=21)
    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--nt", type=int, default=32)
    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--gmres-rtol", type=float, default=1e-7)
    p.add_argument("--gmres-maxiter", type=int, default=80)
    p.add_argument("--sidebands", type=int, default=2)
    p.add_argument("--gamma-nt", type=int, default=128)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    if args.overwrite and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    powers = np.linspace(args.pump_power_min_dbm, args.pump_power_max_dbm, args.n_power)
    freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.n_frequency)

    rows: list[dict[str, Any]] = []
    gain_grid = np.full((args.n_power, args.n_frequency), np.nan)
    convergence_grid = np.zeros((args.n_power, args.n_frequency), dtype=bool)
    status_grid = np.empty((args.n_power, args.n_frequency), dtype=object)

    t_all = time.perf_counter()
    point_index = 0
    for i, power_dbm in enumerate(powers):
        for j, pump_freq_ghz in enumerate(freqs):
            source_power_dbm, pump_current_a = old_ipm_peak_current_from_external_dbm(
                float(power_dbm),
                attenuation_db=args.attenuation_db,
                z0_ohm=args.z0_ohm,
            )
            point_name = (
                f"point_{point_index:04d}_p_{slug_float(float(power_dbm))}dbm_"
                f"fp_{slug_float(float(pump_freq_ghz))}ghz"
            )
            point_dir = outdir / "points" / point_name
            pump_dir = point_dir / "pump"
            gain_dir = point_dir / "gain"

            pump_report_path = pump_dir / "pump_report.json"
            gain_report_path = gain_dir / "gain_report.json"
            can_resume = args.resume and pump_report_path.exists() and gain_report_path.exists()

            point_start = time.perf_counter()
            pump_rc = 0
            gain_rc = 0
            message = ""

            if not can_resume:
                pump_cmd = [
                    sys.executable,
                    "experiments/exp08_full_ipm_pump_solve.py",
                    "--ipm-dir",
                    args.ipm_dir,
                    "--outdir",
                    str(pump_dir),
                    "--pump-port",
                    str(args.pump_port),
                    "--pump-freq-ghz",
                    f"{float(pump_freq_ghz):.12g}",
                    "--pump-current-a",
                    f"{pump_current_a:.17g}",
                    "--harmonics",
                    str(args.harmonics),
                    "--nt",
                    str(args.nt),
                    "--continuation-steps",
                    str(args.continuation_steps),
                    "--newton-tol",
                    str(args.newton_tol),
                    "--gmres-rtol",
                    str(args.gmres_rtol),
                    "--gmres-maxiter",
                    str(args.gmres_maxiter),
                    "--quiet",
                ]
                pump_rc = run_command(pump_cmd, point_dir / "pump_stdout.txt", point_dir / "pump_stderr.txt")

                gain_cmd = [
                    sys.executable,
                    "experiments/exp09_full_ipm_gain_from_pump.py",
                    "--ipm-dir",
                    args.ipm_dir,
                    "--pump-dir",
                    str(pump_dir),
                    "--outdir",
                    str(gain_dir),
                    "--z0-ohm",
                    str(args.z0_ohm),
                    "--source-port",
                    str(args.source_port),
                    "--out-port",
                    str(args.out_port),
                    "--sidebands",
                    str(args.sidebands),
                    "--gamma-nt",
                    str(args.gamma_nt),
                    "--fallback-pump-freq-ghz",
                    f"{float(pump_freq_ghz):.12g}",
                ]
                if args.gain_sweep:
                    gain_cmd.extend([
                        "--sweep",
                        "--signal-start-ghz",
                        str(args.signal_start_ghz),
                        "--signal-stop-ghz",
                        str(args.signal_stop_ghz),
                        "--points",
                        str(args.signal_points),
                    ])
                else:
                    gain_cmd.extend(["--signal-ghz",
                                     f"{signal_ghz_for(pump_freq_ghz, args):.12g}"])
                gain_rc = run_command(gain_cmd, point_dir / "gain_stdout.txt", point_dir / "gain_stderr.txt")

            pump_report = load_json(pump_report_path) if pump_report_path.exists() else None
            gain_report = load_json(gain_report_path) if gain_report_path.exists() else None
            status, pump_status, gain_status = point_status(pump_report, gain_report)
            if pump_rc != 0 or gain_rc != 0:
                status = "ERROR"
                message = f"pump_rc={pump_rc}; gain_rc={gain_rc}"

            best = best_gain_row(gain_dir / "gain_sweep.csv")
            gain_db = float(best["_gain_db"]) if best is not None else math.nan
            signal_ghz = float(best.get("signal_ghz", "nan")) if best is not None else math.nan
            gain_grid[i, j] = gain_db
            convergence_grid[i, j] = status == "VALID_CONVERGED"
            status_grid[i, j] = status

            pump_runtime_s = None
            pump_factor_runtime_s = None
            pump_coeff_rel = None
            pump_time_rel = None
            pump_current_over_ic = None
            if pump_report is not None:
                pump_runtime_s = sum_report_field(pump_report, "runtime_s")
                pump_factor_runtime_s = sum_report_field(pump_report, "factor_runtime_s")
                reports = pump_report.get("reports", [])
                if reports:
                    pump_coeff_rel = reports[-1].get("coeff_rel")
                    pump_time_rel = reports[-1].get("time_rel")
                pump_current_over_ic = pump_report.get("metadata", {}).get("pump_current_ratio_ic_median")

            gain_total_runtime_s = None
            gain_factor_solve_runtime_s = None
            linear_rel_residual = None
            if gain_report is not None:
                gain_total_runtime_s = gain_report.get("metadata", {}).get("total_runtime_s")
                results = gain_report.get("results", [])
                gain_factor_solve_runtime_s = sum(float(r.get("factor_solve_runtime_s", 0.0) or 0.0) for r in results)
                if best is not None:
                    linear_rel_residual = best.get("linear_rel_residual")

            row = {
                "point_index": point_index,
                "pump_power_dbm": float(power_dbm),
                "source_power_dbm": source_power_dbm,
                "attenuation_db": args.attenuation_db,
                "pump_freq_ghz": float(pump_freq_ghz),
                "pump_current_peak_a": pump_current_a,
                "status": status,
                "pump_status": pump_status,
                "gain_status": gain_status,
                "gain_db": gain_db,
                "signal_ghz": signal_ghz,
                "linear_rel_residual": linear_rel_residual,
                "pump_runtime_s": pump_runtime_s,
                "pump_factor_runtime_s": pump_factor_runtime_s,
                "gain_total_runtime_s": gain_total_runtime_s,
                "gain_factor_solve_runtime_s": gain_factor_solve_runtime_s,
                "elapsed_s": time.perf_counter() - point_start,
                "pump_current_over_ic_median": pump_current_over_ic,
                "pump_coeff_rel": pump_coeff_rel,
                "pump_time_rel": pump_time_rel,
                "point_dir": str(point_dir),
                "message": message,
            }
            rows.append(row)
            print(
                f"{point_index + 1}/{args.n_power * args.n_frequency} "
                f"Pext={float(power_dbm):.3g} dBm Psrc={source_power_dbm:.3g} dBm "
                f"fp={float(pump_freq_ghz):.4g} GHz I={pump_current_a:.3e} A "
                f"status={status} gain={gain_db:.4g} dB",
                flush=True,
            )

            if args.stop_on_error and status != "VALID_CONVERGED":
                write_outputs(outdir, rows, gain_grid, convergence_grid, status_grid, powers, freqs, args, t_all)
                raise SystemExit(1)

            point_index += 1

    write_outputs(outdir, rows, gain_grid, convergence_grid, status_grid, powers, freqs, args, t_all)


def write_outputs(
    outdir: Path,
    rows: list[dict[str, Any]],
    gain_grid: np.ndarray,
    convergence_grid: np.ndarray,
    status_grid: np.ndarray,
    powers: np.ndarray,
    freqs: np.ndarray,
    args: argparse.Namespace,
    t_all: float,
) -> None:
    points_csv = outdir / "gain_map_points.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with points_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    np.savez(
        outdir / "gain_map_arrays.npz",
        pump_power_dbm=powers,
        pump_frequency_ghz=freqs,
        gain_db_grid=gain_grid,
        convergence_grid=convergence_grid,
        status_grid=status_grid.astype(str),
    )
    np.savetxt(outdir / "gain_db_grid.csv", gain_grid, delimiter=",")
    np.savetxt(outdir / "convergence_mask_grid.csv", convergence_grid.astype(int), delimiter=",", fmt="%d")

    status_counts = dict(Counter(r["status"] for r in rows))
    valid_rows = [r for r in rows if r["status"] == "VALID_CONVERGED" and np.isfinite(float(r["gain_db"]))]
    best = max(valid_rows, key=lambda r: float(r["gain_db"])) if valid_rows else None
    elapsed = time.perf_counter() - t_all

    summary = {
        "output_dir": str(outdir),
        "n_power": int(args.n_power),
        "n_frequency": int(args.n_frequency),
        "n_points": len(rows),
        "status_counts": status_counts,
        "all_pass": len(rows) == int(args.n_power) * int(args.n_frequency) and len(valid_rows) == len(rows),
        "elapsed_s": elapsed,
        "pump_power_dbm_min": float(args.pump_power_min_dbm),
        "pump_power_dbm_max": float(args.pump_power_max_dbm),
        "pump_freq_ghz_min": float(args.pump_freq_min_ghz),
        "pump_freq_ghz_max": float(args.pump_freq_max_ghz),
        "attenuation_db": float(args.attenuation_db),
        "source_power_dbm_min": float(args.pump_power_min_dbm - args.attenuation_db),
        "source_power_dbm_max": float(args.pump_power_max_dbm - args.attenuation_db),
        "z0_ohm": float(args.z0_ohm),
        "current_convention": "I_peak = sqrt(2 * P_source_W / Z0), P_source_dBm = P_external_dBm - attenuation_dB",
        "gain_sweep": bool(args.gain_sweep),
        "signal_ghz": args.signal_ghz,
        "signal_detuning_mhz": args.signal_detuning_mhz,
        "signal_convention": ("sweep" if args.gain_sweep
                              else "fixed" if args.signal_ghz is not None
                              else f"ws = wp - {args.signal_detuning_mhz} MHz"),
        "best": best,
    }
    with (outdir / "gain_map_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# IPM exp08/exp09 Gain Map",
        "",
        f"- grid: `{args.n_power} x {args.n_frequency}`",
        f"- external pump power: `{args.pump_power_min_dbm}` to `{args.pump_power_max_dbm}` dBm",
        f"- attenuation: `{args.attenuation_db}` dB",
        f"- source pump power: `{args.pump_power_min_dbm - args.attenuation_db}` to `{args.pump_power_max_dbm - args.attenuation_db}` dBm",
        f"- pump frequency: `{args.pump_freq_min_ghz}` to `{args.pump_freq_max_ghz}` GHz",
        f"- signal readout: `{'sweep max' if args.gain_sweep else ('%g GHz' % args.signal_ghz) if args.signal_ghz is not None else ('ws = wp - %g MHz' % args.signal_detuning_mhz)}`",
        "- current convention: `I_peak = sqrt(2 * P_source_W / Z0)`",
        f"- elapsed: `{elapsed:.3f}` s",
        f"- status counts: `{status_counts}`",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                "## Best Valid Point",
                "",
                f"- gain: `{float(best['gain_db']):.6f}` dB",
                f"- external pump power: `{float(best['pump_power_dbm']):.6g}` dBm",
                f"- source pump power: `{float(best['source_power_dbm']):.6g}` dBm",
                f"- pump frequency: `{float(best['pump_freq_ghz']):.6g}` GHz",
                f"- pump current: `{float(best['pump_current_peak_a']):.6e}` A",
                f"- signal frequency: `{float(best['signal_ghz']):.6g}` GHz",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `gain_map_points.csv`",
            "- `gain_db_grid.csv`",
            "- `convergence_mask_grid.csv`",
            "- `gain_map_arrays.npz`",
            "- `gain_map_summary.json`",
        ]
    )
    (outdir / "gain_map_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
