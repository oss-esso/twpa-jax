"""Run smoke map calculations from an exported Harmonia old-IPM circuit JSON."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver_old.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver_old.model.units import dbm_to_old_julia_peak_current
from twpa_solver_old.residuals.linear import solve_linear_sparameters


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    imported = import_julia_circuit_json(args.circuit_json)
    import_runtime = time.perf_counter() - t0
    model = imported.model

    config = vars(args).copy()
    config["import_runtime_s"] = import_runtime
    config["model_metadata"] = model.metadata
    config["node_count"] = model.num_nodes
    config["element_count"] = model.metadata["element_count"]
    config["josephson_junction_count"] = model.metadata["josephson_junction_count"]
    config["mutual_coupling_count"] = model.metadata["mutual_coupling_count"]
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    pump_freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.points)
    external_powers = np.linspace(args.pump_power_min_dbm, args.pump_power_max_dbm, args.points)
    rows: list[dict[str, Any]] = []
    for pext in external_powers:
        for fp in pump_freqs:
            rows.append(_run_row(imported, fp, pext, args))

    _write_rows(outdir / "report_old_ipm_python_backend_rows.csv", rows)
    _write_grid(outdir / "raw_gain_max_db_grid.csv", rows, "gain_db_max", args.points)
    _write_grid(outdir / "convergence_mask_grid.csv", rows, "valid_converged", args.points)
    _write_status_grid(outdir / "status_grid.csv", rows, args.points)
    _write_report(outdir / "report.md", imported, rows, args)
    print(outdir)


def _run_row(imported, pump_frequency_ghz: float, external_power_dbm: float, args) -> dict[str, Any]:
    model = imported.model
    source_power_dbm = float(external_power_dbm) - float(args.power_offset_db)
    pump_current_a = dbm_to_old_julia_peak_current(source_power_dbm)
    row: dict[str, Any] = {
        "pump_frequency_ghz": float(pump_frequency_ghz),
        "external_power_dbm": float(external_power_dbm),
        "source_power_dbm": source_power_dbm,
        "pump_current_a": pump_current_a,
        "pump_current_ua": pump_current_a * 1e6,
        "backend": "python-exported-netlist",
        "solver": args.solver,
        "node_count": model.num_nodes,
        "element_count": model.metadata["element_count"],
        "josephson_junction_count": model.metadata["josephson_junction_count"],
        "mutual_coupling_count": model.metadata["mutual_coupling_count"],
        "input_port": 1,
        "output_port": 2,
        "pump_port": 4,
        "valid_converged": False,
        "failure_reason": "",
        "runtime_s": 0.0,
        "gain_db_max": np.nan,
    }
    if model.num_nodes > args.max_linear_nodes:
        row["status"] = "IMPORTED_ASSEMBLED_SOLVE_SKIPPED_DENSE_SCALE"
        row["failure_reason"] = (
            f"linear dense smoke skipped: node_count={model.num_nodes} exceeds "
            f"--max-linear-nodes={args.max_linear_nodes}; pump HB backend not implemented yet"
        )
        return row
    t0 = time.perf_counter()
    try:
        sparams = solve_linear_sparameters(model, pump_frequency_ghz * 1e9)
        port_names = [p.name for p in model.ports]
        i_in = port_names.index("P1")
        i_out = port_names.index("P2")
        s21 = sparams.s[i_out, i_in]
        row["gain_db_max"] = float(10.0 * np.log10(np.abs(s21) ** 2))
        row["status"] = "LINEAR_IMPORTED_SMOKE_OK_HB_NOT_IMPLEMENTED"
        row["failure_reason"] = "full pump-only HB solve for exported old-IPM netlist is not implemented in this recovery pass"
    except Exception as exc:  # pragma: no cover - exercised by full-scale smoke if numerically singular.
        row["status"] = "IMPORTED_LINEAR_SMOKE_FAILED"
        row["failure_reason"] = repr(exc)
    row["runtime_s"] = time.perf_counter() - t0
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_grid(path: Path, rows: list[dict[str, Any]], key: str, points: int) -> None:
    values = np.asarray([row[key] for row in rows], dtype=float).reshape(points, points)
    np.savetxt(path, values, delimiter=",")


def _write_status_grid(path: Path, rows: list[dict[str, Any]], points: int) -> None:
    values = np.asarray([row["status"] for row in rows], dtype=object).reshape(points, points)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(values)


def _write_report(path: Path, imported, rows: list[dict[str, Any]], args) -> None:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    lines = [
        "# Exported Julia Netlist Python Backend Smoke",
        "",
        "This run consumes the exact JSON exported from `build_old_ipm_circuit()`.",
        "It does not use the reduced Python IPM surrogate topologies.",
        "",
        f"- circuit_json: `{args.circuit_json}`",
        f"- node_count: {imported.model.num_nodes}",
        f"- element_count: {imported.model.metadata['element_count']}",
        f"- josephson_junction_count: {imported.model.metadata['josephson_junction_count']}",
        f"- mutual_coupling_count: {imported.model.metadata['mutual_coupling_count']}",
        f"- points: {args.points}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(statuses.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "The current independent backend imports and assembles the full old-IPM netlist and can run a linear S-parameter smoke when the dense solve is feasible. Full pump-only AFT/HB for the exported 2508-junction old-IPM circuit is not implemented in this recovery pass.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit-json", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--pump-freq-min-ghz", type=float, default=6.0)
    parser.add_argument("--pump-freq-max-ghz", type=float, default=8.0)
    parser.add_argument("--pump-power-min-dbm", type=float, default=-28.0)
    parser.add_argument("--pump-power-max-dbm", type=float, default=-19.0)
    parser.add_argument("--power-offset-db", type=float, default=32.0)
    parser.add_argument("--pump-harmonics", type=int, default=10)
    parser.add_argument("--modulation-harmonics", type=int, default=5)
    parser.add_argument("--solver", default="scipy-least-squares")
    parser.add_argument("--max-linear-nodes", type=int, default=5000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
