"""Scaling benchmark for reduced-marker and physical-coupler IPM topologies."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import numpy as np

from twpa_solver.model.ipm import IPMConfig, build_ipm_topology
from twpa_solver.model.units import dbm_to_current_peak
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver.residuals.jax_aft_hb import JaxPumpAFTResidual
from twpa_solver.solvers.jax_dense_newton import solve_jax_dense_newton
from twpa_solver.solvers.jax_newton_krylov import solve_jax_newton_krylov
from twpa_solver.solvers.preconditioners import build_linear_passive_preconditioner
from twpa_solver.solvers.scipy_least_squares import solve_least_squares


def main(argv: list[str] | None = None) -> None:
    jax.config.update("jax_enable_x64", True)
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    rows = _run_benchmark(args)
    _write_rows(outdir / "scaling_rows.csv", rows)
    _write_summary(outdir / "scaling_summary.md", rows)
    _plot(rows, outdir / "plots")


def _run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topology in ("ipm_jtwpa_reduced_marker", "ipm_jtwpa_physical_coupler"):
        for cells in (4, 8, 16, 32):
            for harmonics in (1, 3, 5):
                for sidebands in (1, 3):
                    rows.append(
                        _run_case(
                            topology=topology,
                            cells=cells,
                            harmonics=harmonics,
                            sidebands=sidebands,
                            solver="scipy_least_squares",
                            args=args,
                        )
                    )
                    if cells <= 4 and harmonics == 1 and sidebands == 1:
                        rows.append(
                            _run_case(
                                topology=topology,
                                cells=cells,
                                harmonics=harmonics,
                                sidebands=sidebands,
                                solver="jax_dense_newton",
                                args=args,
                            )
                        )
                        rows.append(
                            _run_case(
                                topology=topology,
                                cells=cells,
                                harmonics=harmonics,
                                sidebands=sidebands,
                                solver="jax_newton_krylov_preconditioned",
                                args=args,
                            )
                        )
    return rows


def _run_case(
    *,
    topology: str,
    cells: int,
    harmonics: int,
    sidebands: int,
    solver: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = build_ipm_topology(topology, IPMConfig(cells_per_line=cells))
    current = args.pump_current_coupling * dbm_to_current_peak(args.pump_power_dbm)
    residual = PumpAFTResidual(
        model,
        PumpAFTConfig(
            pump_frequency_hz=args.pump_frequency_ghz * 1e9,
            harmonics=harmonics,
            source_current_peak_a=current,
            residual_scale_a=args.residual_scale_a,
        ),
    )
    x0 = np.zeros(residual.size)
    started = time.perf_counter()
    try:
        if solver == "scipy_least_squares":
            result = solve_least_squares(
                residual,
                x0,
                tolerance=args.optimizer_tolerance,
                max_nfev=args.max_nfev,
            )
        elif solver == "jax_dense_newton":
            jax_residual = JaxPumpAFTResidual(model, residual.config)
            result = solve_jax_dense_newton(jax_residual, x0, tolerance=args.solver_tolerance)
        elif solver == "jax_newton_krylov_preconditioned":
            jax_residual = JaxPumpAFTResidual(model, residual.config)
            preconditioner = build_linear_passive_preconditioner(residual)
            result = solve_jax_newton_krylov(
                jax_residual,
                x0,
                tolerance=args.solver_tolerance,
                max_iterations=4,
                preconditioner=preconditioner,
            )
        else:
            raise ValueError(solver)
        status = "converged" if result.residual_norm_inf <= args.solver_tolerance else "diagnostic"
        success = status == "converged"
        message = result.message
        runtime = result.runtime_s
        residual_inf = result.residual_norm_inf
        linear_solves = result.num_linear_solves
    except Exception as exc:
        status = "failed"
        success = False
        message = str(exc)
        runtime = time.perf_counter() - started
        residual_inf = np.nan
        linear_solves = 0
    return {
        "topology": topology,
        "coupler_model": model.metadata.get("coupler_model", ""),
        "cells_per_line": cells,
        "nodes": model.num_nodes,
        "pump_harmonics": harmonics,
        "sidebands": sidebands,
        "solver": solver,
        "status": status,
        "success": success,
        "residual_norm_inf": residual_inf,
        "runtime_s": runtime,
        "num_linear_solves": linear_solves,
        "message": message,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    full = [
        row
        for row in rows
        if int(row["cells_per_line"]) == 32
        and int(row["pump_harmonics"]) == 5
        and int(row["sidebands"]) == 3
        and row["solver"] == "scipy_least_squares"
    ]
    feasible = all(row["status"] != "failed" for row in full) and len(full) > 0
    text = [
        "# New TWPA Solver Scaling Benchmark",
        "",
        f"- total rows: {len(rows)}",
        f"- full 32-cell / 5-harmonic / 3-sideband scipy feasibility: {feasible}",
        "",
        "This benchmark is a one-case feasibility screen, not a production gain map.",
        "Rows are in `scaling_rows.csv`; plots are in `plots/`.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _plot(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    for metric, filename, ylabel in [
        ("runtime_s", "runtime_vs_cells.png", "runtime_s"),
        ("residual_norm_inf", "residual_vs_cells.png", "residual_norm_inf"),
    ]:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        for topology in sorted(set(row["topology"] for row in rows)):
            selected = [
                row
                for row in rows
                if row["topology"] == topology
                and row["solver"] == "scipy_least_squares"
                and int(row["pump_harmonics"]) == 1
                and int(row["sidebands"]) == 1
            ]
            ax.plot(
                [int(row["cells_per_line"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                label=topology,
            )
        ax.set_xlabel("cells_per_line")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    for topology in sorted(set(row["topology"] for row in rows)):
        xs = []
        ys = []
        for cells in sorted(set(int(row["cells_per_line"]) for row in rows)):
            selected = [row for row in rows if row["topology"] == topology and int(row["cells_per_line"]) == cells]
            if selected:
                xs.append(cells)
                ys.append(sum(row["success"] in {True, "True", "true"} for row in selected) / len(selected))
        ax.plot(xs, ys, marker="o", label=topology)
    ax.set_xlabel("cells_per_line")
    ax.set_ylabel("success_rate")
    ax.legend()
    fig.savefig(plot_dir / "success_rate_vs_cells.png", dpi=160)
    plt.close(fig)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pump-frequency-ghz", type=float, default=6.5)
    parser.add_argument("--pump-power-dbm", type=float, default=-24.0)
    parser.add_argument("--pump-current-coupling", type=float, default=1e-3)
    parser.add_argument("--residual-scale-a", type=float, default=1e-6)
    parser.add_argument("--solver-tolerance", type=float, default=1.0)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-nfev", type=int, default=30)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
