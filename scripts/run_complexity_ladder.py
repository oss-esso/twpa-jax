"""Build and probe the controlled Josephson circuit-complexity ladder.

This runner is intentionally small: it builds one rung, obtains a low-drive
periodic HB seed with the production solver, and optionally hands that seed to
the validated H3 transient diagnostic.  It does not alter the production IPM.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.h1_transient_branch_transfer import run_experiment  # noqa: E402
from twpa_solver.builders.complexity_ladder import (  # noqa: E402
    LadderParameters,
    build_and_save_ladder,
)
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump import HarmonicGrid, HarmonicNewtonKrylovSolver, NewtonKrylovSettings  # noqa: E402
from twpa_solver.pump.problem import FullPumpProblem  # noqa: E402
from twpa_solver.pump.validation import validate_production_hb_state  # noqa: E402


def _solver() -> HarmonicNewtonKrylovSolver:
    return HarmonicNewtonKrylovSolver(NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=15, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=40, gmres_maxiter=60, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=True,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
        stall_ratio=0.8, stall_patience=4, solve_deadline_s=180.0,
    ))


def _checkpoint(circuit_dir: Path, circuit: object, outdir: Path, current_a: float, frequency_hz: float) -> Path:
    from twpa_solver.core import load_circuit
    loaded = load_circuit(circuit_dir)
    branch = make_branch_law(loaded)
    modes = np.arange(1, 11, dtype=int)
    grid = HarmonicGrid(modes=modes, nt=40, omega=2.0 * math.pi * frequency_hz)
    problem = FullPumpProblem(
        loaded.C, loaded.G, loaded.K, loaded.Bphi,
        branch, grid, loaded.port_to_index[1], current_a,
    )
    solver = _solver()
    state, report = solver.solve_one(problem, problem.zeros(), 1.0)
    route = "DIRECT"
    reports = [report]
    if not report.converged:
        state, reports = solver.solve_continuation(
            problem, continuation_steps=20, x_init=problem.zeros(),
        )
        report = reports[-1] if reports else report
        route = "POWER_SUBSTEP"
    if not report.converged:
        raise RuntimeError(f"production HB solve failed via {route}: {report}")
    validation = validate_production_hb_state(
        loaded, branch, frequency_hz=frequency_hz, pump_port=1,
        pump_current_a=current_a, modes=modes, state=state, nt=40,
        metadata={"topology": circuit.metadata.get("topology")},
    )
    if not validation["checkpoint_validated"]:
        raise RuntimeError(
            "low-drive HB seed reported convergence but failed residual validation: "
            f"coeff_rel={validation.get('production_hb_coeff_rel')}, "
            f"time_rel={validation.get('production_hb_time_rel')}"
        )
    checkpoint = outdir / "hb_checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint / "pump_solution.npz",
        X_real=state.real, X_imag=state.imag, pump_modes=modes,
    )
    (checkpoint / "pump_report.json").write_text(json.dumps({
        "final_status": "VALID_CONVERGED",
        "metadata": {
            "pump_current_a": current_a,
            "pump_modes": modes.tolist(),
            **validation,
            "newton_iterations": report.newton_iterations,
            "runtime_s": report.runtime_s,
            "solver_route": route,
            "reports": [item.__dict__ for item in reports],
        },
    }, indent=2), encoding="utf-8")
    return checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=("linear", "single_jj", "jtl", "ipm_section"), default="single_jj")
    parser.add_argument("--n-cells", type=int, default=8)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "complexity_ladder" / "single_jj")
    parser.add_argument("--start-fraction", type=float, default=0.05)
    parser.add_argument("--target-fraction", type=float, default=1.0)
    parser.add_argument("--ramp-periods", type=int, default=10)
    parser.add_argument("--hold-periods", type=int, default=10)
    parser.add_argument("--max-step", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    params = LadderParameters()
    args.outdir.mkdir(parents=True, exist_ok=True)
    circuit_dir = args.outdir / "circuit"
    circuit = build_and_save_ladder(
        circuit_dir, args.rung, n_cells=args.n_cells if args.rung in ("jtl", "ipm_section") else None,
    )
    summary = {
        "rung": args.rung, "n_cells": args.n_cells if args.rung in ("jtl", "ipm_section") else None,
        "circuit_dir": str(circuit_dir), "metadata": circuit.metadata,
    }
    if args.build_only:
        (args.outdir / "ladder_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        return 0

    frequency_hz = params.frequency_hz
    current_start = args.start_fraction * params.ic_a
    current_target = args.target_fraction * params.ic_a
    checkpoint = _checkpoint(circuit_dir, circuit, args.outdir, current_start, frequency_hz)
    transient_args = argparse.Namespace(
        circuit_dir=circuit_dir, checkpoint=checkpoint, outdir=args.outdir / "transient",
        freq_ghz=frequency_hz / 1e9, pump_port=1, target_current_a=current_target,
        ramp_periods=args.ramp_periods, hold_periods=args.hold_periods,
        samples_per_period=16, rtol=2e-5, atol=args.atol, max_step=args.max_step,
        method="implicit_trapezoid", max_newton=args.max_newton,
    )
    result = run_experiment(transient_args)
    summary.update({
        "start_current_a": current_start, "target_current_a": current_target,
        "transient": result,
    })
    (args.outdir / "ladder_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "rung": args.rung, "classification": result["classification"],
        "final_status": result["final_status"], "target_current_a": current_target,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
