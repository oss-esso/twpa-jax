"""Run an IPM pump/conversion gain map with convergence artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver_old.experiments.plot_ipm_gain_map import plot_gain_map
from twpa_solver_old.model.ipm import IPMConfig, build_ipm_topology
from twpa_solver_old.model.units import (
    current_peak_to_dbm,
    dbm_to_current_peak,
    dbm_to_old_julia_peak_current,
)
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver_old.residuals.conversion import build_conversion_sparameters
from twpa_solver_old.solvers.continuation import snake_grid_indices
from twpa_solver_old.solvers.jax_dense_newton import solve_jax_dense_newton
from twpa_solver_old.solvers.jax_newton_krylov import solve_jax_newton_krylov
from twpa_solver_old.solvers.pseudo_transient import solve_pseudo_transient
from twpa_solver_old.solvers.scipy_least_squares import solve_least_squares
from twpa_solver_old.solvers.scipy_root import solve_newton_krylov, solve_root

REQUIRED_PARITY_ROW_FIELDS = (
    "topology",
    "geometry_profile",
    "coupler_model",
    "old_julia_parity_mode",
    "external_power_dbm",
    "power_offset_db",
    "source_power_dbm",
    "pump_current_a",
    "pump_port_equivalent",
    "input_port_equivalent",
    "output_port_equivalent",
    "cells_per_line",
    "historical_target_cells_or_junctions",
    "pump_harmonics",
    "sidebands",
    "solver",
    "status",
    "residual_norm_l2",
    "residual_norm_inf",
    "signal_gain_db",
    "idler_gain_db",
)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    config = vars(args).copy()
    config["started_at_unix"] = time.time()
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = build_ipm_topology(
        args.topology,
        IPMConfig(
            cells_per_line=args.cells_per_line,
            critical_current_a=args.critical_current_a,
            shunt_capacitance_f=args.shunt_capacitance_f,
            z0_ohm=args.z0,
            coupler_inductance_top_h=args.coupler_inductance_top_h,
            coupler_inductance_bottom_h=args.coupler_inductance_bottom_h,
            coupler_k=args.coupler_k,
            coupler_shunt_capacitance_f=args.coupler_shunt_capacitance_f,
            coupler_mutual_capacitance_f=args.coupler_mutual_capacitance_f,
        )
    )
    config["model_metadata"] = model.metadata
    config["old_julia_parity_mode_effective"] = _is_old_julia_parity(args)
    (outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    pump_freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.points) * 1e9
    powers = np.linspace(args.pump_power_min_dbm, args.pump_power_max_dbm, args.points)
    rows: list[dict[str, Any]] = []
    last_solution: np.ndarray | None = None
    cold_solution = None
    order = snake_grid_indices(args.points, args.points) if args.continuation == "snake" else [
        (i, j) for i in range(args.points) for j in range(args.points)
    ]
    for freq_idx, power_idx in order:
        freq = float(pump_freqs[freq_idx])
        power = float(powers[power_idx])
        drive = _compute_drive_convention(power, args)
        external_current = drive["external_current_peak_a"]
        current = drive["pump_current_a"]
        source_power_dbm = drive["source_power_dbm"]
        residual = PumpAFTResidual(
            model,
            PumpAFTConfig(
                pump_frequency_hz=freq,
                harmonics=args.pump_harmonics,
                source_current_peak_a=current,
                residual_scale_a=args.residual_scale_a,
            ),
        )
        x0 = last_solution if last_solution is not None else residual.initial_guess()
        result = _solve(args.solver, residual, x0, args.max_nfev, args.optimizer_tolerance)
        result = _apply_acceptance_tolerance(result, args.solver_tolerance)
        if result.success:
            last_solution = result.solution
            if cold_solution is None:
                cold_solution = result.solution
        conversion = None
        signal_gain_db = np.nan
        idler_gain_db = np.nan
        if args.compute_conversion_sparams.lower() == "true":
            conversion = build_conversion_sparameters(
                model,
                residual,
                result.solution,
                args.signal_frequency_ghz * 1e9,
                args.sidebands,
                pump_success=result.success,
                pump_status=result.status,
            )
            if result.success:
                signal_gain_db = conversion.signal_gain_db
                idler_gain_db = conversion.idler_gain_db
        row = _row_from_solution_status(
            {
                "pump_frequency_hz": freq,
                "pump_frequency_ghz": freq / 1e9,
                "pump_power_dbm": power,
                "external_power_dbm": power,
                "pump_power_dbm_external": power,
                "source_power_dbm": source_power_dbm,
                "power_offset_db": args.power_offset_db if _is_old_julia_parity(args) else "",
                "pump_current_peak_a_external_norton": external_current,
                "pump_current_coupling": args.pump_current_coupling,
                "pump_current_a": current,
                "pump_current_peak_a_internal": current,
                "topology": args.topology,
                "geometry_profile": model.metadata.get("geometry_profile", args.topology),
                "coupler_model": model.metadata.get("coupler_model", ""),
                "old_julia_parity_mode": _is_old_julia_parity(args),
                "pump_port_equivalent": 4 if _old_port_convention(args) else "",
                "input_port_equivalent": 1 if _old_port_convention(args) else "",
                "output_port_equivalent": 2 if _old_port_convention(args) else "",
                "cells_per_line": args.cells_per_line,
                "historical_target_cells_or_junctions": model.metadata.get(
                    "historical_target_cells_or_junctions",
                    "",
                ),
                "pump_harmonics": args.pump_harmonics,
                "sidebands": args.sidebands,
                "pump_source_nodes": ";".join(str(node) for node in model.pump_nodes),
                "freq_index": freq_idx,
                "power_index": power_idx,
                "status": result.status,
                "success": result.success,
                "solver": result.solver_name,
                "residual_norm_l2": result.residual_norm_l2,
                "residual_norm_inf": result.residual_norm_inf,
                "num_iterations": result.num_iterations,
                "num_residual_evals": result.num_residual_evals,
                "num_jacobian_evals": result.num_jacobian_evals,
                "num_linear_solves": result.num_linear_solves,
                "runtime_s": result.runtime_s,
                "signal_gain_db": signal_gain_db,
                "idler_gain_db": idler_gain_db,
                "pump_status_for_sparams": conversion.pump_status if conversion else result.status,
            },
            success=result.success,
            status=result.status,
        )
        rows.append(row)
    _write_outputs(outdir, rows, args.points)
    _write_solver_summary(outdir, rows)
    plot_gain_map(outdir)
    _write_report(outdir, config, rows, model.num_nodes)


def _solve(
    name: str,
    residual: PumpAFTResidual,
    x0: np.ndarray,
    max_nfev: int | None,
    tolerance: float,
):
    if name == "scipy-least-squares":
        return solve_least_squares(residual, x0, tolerance=tolerance, max_nfev=max_nfev)
    if name == "scipy-root":
        return solve_root(residual, x0, tolerance=tolerance)
    if name == "scipy-newton-krylov":
        return solve_newton_krylov(residual, x0, tolerance=tolerance)
    if name == "pseudo-transient":
        return solve_pseudo_transient(residual, x0, tolerance=tolerance)
    if name == "jax-dense-newton":
        import jax.numpy as jnp

        def r_jax(x):
            return jnp.asarray(residual(np.asarray(x)))

        return solve_jax_dense_newton(r_jax, x0, tolerance=tolerance)
    if name == "jax-newton-krylov":
        import jax.numpy as jnp

        def r_jax(x):
            return jnp.asarray(residual(np.asarray(x)))

        return solve_jax_newton_krylov(r_jax, x0, tolerance=tolerance)
    raise ValueError(f"unknown solver {name}")


def _write_outputs(outdir: Path, rows: list[dict[str, Any]], points: int) -> None:
    with (outdir / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status_fields = [
        "freq_index",
        "power_index",
        "status",
        "success",
        "residual_norm_inf",
        "runtime_s",
    ]
    with (outdir / "pump_solution_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=status_fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in status_fields} for row in rows])
    for field, filename in [
        ("signal_gain_db", "gain_signal_db_grid.csv"),
        ("idler_gain_db", "idler_gain_db_grid.csv"),
        ("residual_norm_inf", "residual_norm_grid.csv"),
        ("runtime_s", "runtime_grid.csv"),
    ]:
        np.savetxt(outdir / filename, _grid(rows, points, field), delimiter=",")
    np.savetxt(
        outdir / "convergence_mask_grid.csv",
        _grid(rows, points, "success").astype(int),
        delimiter=",",
        fmt="%d",
    )


def _apply_acceptance_tolerance(result, tolerance: float):
    if result.residual_norm_inf <= tolerance and np.isfinite(result.residual_norm_inf):
        return replace(result, status="converged", success=True)
    if np.isfinite(result.residual_norm_inf):
        return replace(result, status="diagnostic", success=False)
    return replace(result, status="failed", success=False)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _is_old_julia_parity(args: argparse.Namespace) -> bool:
    return args.topology == "ipm_jtwpa_old_julia_parity" or _parse_bool(
        args.use_old_julia_power_offset
    )


def _old_port_convention(args: argparse.Namespace) -> bool:
    return args.topology == "ipm_jtwpa_old_julia_parity" or _parse_bool(
        args.old_port_convention
    )


def _compute_drive_convention(
    external_power_dbm: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    if _is_old_julia_parity(args):
        source_power_dbm = float(external_power_dbm) - float(args.power_offset_db)
        current = dbm_to_old_julia_peak_current(source_power_dbm, args.z0)
        return {
            "external_current_peak_a": dbm_to_old_julia_peak_current(
                external_power_dbm,
                args.z0,
            ),
            "source_power_dbm": source_power_dbm,
            "pump_current_a": current,
        }
    external_current = dbm_to_current_peak(external_power_dbm, args.z0)
    current = args.pump_current_coupling * external_current
    return {
        "external_current_peak_a": external_current,
        "source_power_dbm": current_peak_to_dbm(current, args.z0),
        "pump_current_a": current,
    }


def _row_from_solution_status(
    base: dict[str, Any],
    *,
    success: bool,
    status: str,
) -> dict[str, Any]:
    """Mask S-parameter gains when pump status is not cleanly converged."""
    row = dict(base)
    row["status"] = status
    row["success"] = bool(success)
    if not success:
        row["signal_gain_db"] = np.nan
        row["idler_gain_db"] = np.nan
    return row


def _write_solver_summary(outdir: Path, rows: list[dict[str, Any]]) -> None:
    success = [row for row in rows if row["success"]]
    summary = [
        {
            "solver": rows[0]["solver"],
            "cells": len(rows),
            "success_rate": len(success) / len(rows),
            "mean_residual_inf": float(np.mean([row["residual_norm_inf"] for row in rows])),
            "median_runtime_s": float(np.median([row["runtime_s"] for row in rows])),
            "invalid_cells": len(rows) - len(success),
            "status": "implemented and passed" if success else "implemented but no converged cells",
        }
    ]
    with (outdir / "solver_comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _grid(rows: list[dict[str, Any]], points: int, field: str) -> np.ndarray:
    grid = np.full((points, points), np.nan)
    for row in rows:
        grid[int(row["freq_index"]), int(row["power_index"])] = row[field]
    return grid


def _write_report(outdir: Path, config: dict[str, Any], rows: list[dict[str, Any]], nodes: int) -> None:
    success_count = sum(bool(row["success"]) for row in rows)
    text = f"""# New TWPA Solver IPM Gain Map

JosephsonCircuits.jl is no longer the production solver backend.
The new solver treats the TWPA as a modular nonlinear dynamical system.
Harmonic balance is implemented as a Fourier pseudo-spectral residual.
S-parameters are computed by pump-only HB followed by linearized conversion-matrix analysis.
Alternative nonlinear solvers are implemented through a shared residual interface.

## Run

- topology: {config['topology']}
- nodes: {nodes}
- cells_per_line: {config['cells_per_line']}
- grid: {config['points']} x {config['points']}
- pump harmonics: {config['pump_harmonics']}
- sidebands: {config['sidebands']}
- solver: {config['solver']}
- converged cells: {success_count} / {len(rows)}

## Outputs

Raw rows, pump status, gain grids, convergence masks, residual/runtime grids, and plots are in this folder.
Nonconverged cells are retained as diagnostic rows and masked from clean gain grids.
"""
    (outdir / "report.md").write_text(text, encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--topology", default="ipm_jtwpa_reduced_marker")
    parser.add_argument("--cells-per-line", type=int, default=4)
    parser.add_argument("--critical-current-a", type=float, default=8e-6)
    parser.add_argument("--shunt-capacitance-f", type=float, default=70e-15)
    parser.add_argument("--z0", type=float, default=50.0)
    parser.add_argument("--coupler-inductance-top-h", type=float, default=41.3e-12)
    parser.add_argument("--coupler-inductance-bottom-h", type=float, default=41.3e-12)
    parser.add_argument("--coupler-k", type=float, default=0.25)
    parser.add_argument("--coupler-shunt-capacitance-f", type=float, default=17.3e-15)
    parser.add_argument("--coupler-mutual-capacitance-f", type=float, default=1e-15)
    parser.add_argument("--pump-freq-min-ghz", type=float, default=6.0)
    parser.add_argument("--pump-freq-max-ghz", type=float, default=8.0)
    parser.add_argument("--pump-power-min-dbm", type=float, default=-28.0)
    parser.add_argument("--pump-power-max-dbm", type=float, default=-19.0)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--pump-harmonics", type=int, default=2)
    parser.add_argument("--sidebands", type=int, default=1)
    parser.add_argument("--signal-frequency-ghz", type=float, default=5.0)
    parser.add_argument("--pump-current-coupling", type=float, default=1e-3)
    parser.add_argument("--use-old-julia-power-offset", default="false")
    parser.add_argument("--power-offset-db", type=float, default=32.0)
    parser.add_argument("--old-port-convention", default="false")
    parser.add_argument("--solver", default="scipy-least-squares")
    parser.add_argument("--continuation", default="snake")
    parser.add_argument("--compute-conversion-sparams", default="true")
    parser.add_argument("--residual-scale-a", type=float, default=1e-6)
    parser.add_argument("--solver-tolerance", type=float, default=1.0)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-nfev", type=int, default=80)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
