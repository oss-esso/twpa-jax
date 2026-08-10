"""Correct and continue a Floquet-seeded period-doubled pump branch.

This is an opt-in research workflow.  It refuses stale or incomplete pump
checkpoints, uses the production HB equations for correction, and stops only
on a validated junction-utilization threshold, a continuation result, or an
explicit numerical blocker.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump import (  # noqa: E402
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
    build_period_doubled_problem,
    continue_until_utilization,
    correct_period_doubled_seed,
    period_doubled_basis,
)
from twpa_solver.pump.diagnostics import branch_stress_metrics  # noqa: E402
from twpa_solver.pump.io import summarize_solution, write_results  # noqa: E402
from twpa_solver.pump.validation import validate_production_hb_state  # noqa: E402
from twpa_solver.signal import load_pump  # noqa: E402
from twpa_solver.signal.period_doubled import solve_period_doubled_gain  # noqa: E402
from twpa_solver.signal.floquet import sideband_list  # noqa: E402
from twpa_solver.signal.gamma import build_khat, compute_gamma_hat  # noqa: E402
from twpa_solver.signal.stability import (  # noqa: E402
    classify_floquet_resonance,
    refine_complex_resonance,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--pump-port", type=int, required=True)
    parser.add_argument("--freq-ghz", type=float, required=True)
    parser.add_argument("--sidebands", type=int, default=6)
    parser.add_argument("--gamma-nt", type=int, default=512)
    parser.add_argument("--hb-nt", type=int, default=None)
    parser.add_argument("--full-residual-threshold", type=float, default=1.0e-8)
    parser.add_argument("--residual-threshold", type=float, default=1.0e-8)
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--gmres-maxiter", type=int, default=400)
    parser.add_argument("--floquet-iters", type=int, default=20)
    parser.add_argument("--floquet-tol", type=float, default=1.0e-8)
    parser.add_argument("--target-mu", type=float, default=2.0)
    parser.add_argument("--mu-step", type=float, default=0.02)
    parser.add_argument("--break-threshold", type=float, default=0.999999)
    parser.add_argument("--max-wall-s", type=float, default=0.0)
    parser.add_argument("--signal-ghz", type=float, default=None)
    parser.add_argument("--output-port", type=int, default=None)
    parser.add_argument("--gain-sidebands", type=int, default=6)
    return parser.parse_args(argv)


def _dc_flux(metadata: dict[str, Any], branch_count: int) -> np.ndarray:
    value = metadata.get("dc_branch_flux", metadata.get("dc_branch_flux_wb"))
    if value is None:
        return np.zeros(branch_count, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(branch_count, float(array[0]), dtype=float)
    if array.size != branch_count:
        raise ValueError("checkpoint dc_branch_flux has incompatible length")
    return array


def _solver(args: argparse.Namespace) -> HarmonicNewtonKrylovSolver:
    settings = NewtonKrylovSettings(
        max_newton=args.max_newton,
        gmres_maxiter=args.gmres_maxiter,
        compute_time_residual=True,
        verbose=False,
    )
    return HarmonicNewtonKrylovSolver(settings)


def run(args: argparse.Namespace) -> dict[str, Any]:
    circuit = load_circuit(args.circuit_dir)
    branch = make_branch_law(circuit)
    pump = load_pump(args.checkpoint, fallback_pump_freq_ghz=args.freq_ghz)
    metadata = dict(pump.metadata)
    current_a = float(metadata.get("pump_current_a", float("nan")))
    if not np.isfinite(current_a) or current_a <= 0.0:
        raise ValueError("checkpoint lacks a positive pump_current_a")
    dc_flux = _dc_flux(metadata, circuit.branch_count)

    initial_validation = validate_production_hb_state(
        circuit,
        branch,
        frequency_hz=pump.pump_freq_ghz * 1.0e9,
        pump_port=args.pump_port,
        pump_current_a=current_a,
        modes=np.asarray(pump.modes),
        state=pump.X,
        nt=max(40, 2 * max(pump.modes) + 2),
        residual_threshold=args.residual_threshold,
        full_residual_threshold=args.full_residual_threshold,
        metadata=metadata,
        dc_branch_flux=dc_flux,
        source_mode=pump.basis.source_mode,
    )
    if not initial_validation["checkpoint_validated"]:
        return {
            "status": "INVALID_PERIOD1_CHECKPOINT",
            "initial_validation": initial_validation,
        }

    ms = sideband_list(args.sidebands)
    max_ell = max(abs(m - q) for m in ms for q in ms)
    gamma_hat = compute_gamma_hat(
        circuit,
        pump,
        max_ell=max_ell,
        gamma_nt=max(args.gamma_nt, 2 * max(pump.modes) + 4),
        dc_branch_flux=dc_flux,
    )
    khat = build_khat(circuit.Bphi, gamma_hat, drop_tol=0.0)
    floquet = refine_complex_resonance(
        circuit,
        khat,
        pump.omega_p,
        ms,
        signal_ghz_guess=0.5 * pump.pump_freq_ghz,
        max_iters=args.floquet_iters,
        tol=args.floquet_tol,
    )
    classification = classify_floquet_resonance(floquet, pump.omega_p)
    floquet_record = {
        "omega_real_hz": float(floquet.omega.real / (2.0 * math.pi)),
        "omega_imag_hz": float(floquet.omega.imag / (2.0 * math.pi)),
        "growth_rate_per_s": floquet.growth_rate_per_s,
        "converged": floquet.converged,
        "residual": floquet.residual,
        "classification": asdict(classification),
        "sidebands": ms,
    }
    if not floquet.converged or classification.kind != "PERIOD_DOUBLING_CANDIDATE":
        result = {
            "status": "NO_VALID_PERIOD_DOUBLING_CANDIDATE",
            "checkpoint": str(args.checkpoint),
            "initial_validation": initial_validation,
            "floquet": floquet_record,
        }
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "period_doubled_summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return result
    if floquet.mode_vector is None:
        raise RuntimeError("Floquet refinement returned no mode vector")

    target_basis = period_doubled_basis(pump.basis)
    problem = build_period_doubled_problem(
        circuit,
        branch,
        target_basis,
        pump_current_a=current_a,
        pump_port=args.pump_port,
        dc_branch_flux=dc_flux,
        nt=args.hb_nt,
    )
    solver = _solver(args)
    correction = correct_period_doubled_seed(
        solver,
        problem,
        pump.X,
        pump.basis,
        floquet.mode_vector,
        ms,
        target_basis,
    )
    result: dict[str, Any] = {
        "status": "PERIOD_DOUBLED_CORRECTION_FAILED",
        "checkpoint": str(args.checkpoint),
        "initial_validation": initial_validation,
        "floquet": floquet_record,
        "target_basis": target_basis.to_metadata(),
    }
    if correction is None:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "period_doubled_summary.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return result

    corrected_validation = validate_production_hb_state(
        circuit,
        branch,
        frequency_hz=target_basis.omega_p / (2.0 * math.pi),
        pump_port=args.pump_port,
        pump_current_a=current_a,
        modes=np.asarray(target_basis.modes),
        state=correction.state,
        nt=problem.grid.nt,
        residual_threshold=args.residual_threshold,
        full_residual_threshold=args.full_residual_threshold,
        metadata=target_basis.to_metadata(),
        dc_branch_flux=dc_flux,
        source_mode=target_basis.source_mode,
    )
    result.update({
        "correction": {
            "converged": correction.report.converged,
            "coeff_rel": correction.report.coeff_rel,
            "time_rel": correction.report.time_rel,
            "newton_iterations": correction.report.newton_iterations,
            "gmres_iterations_total": correction.report.gmres_iterations_total,
            "perturbation_amplitude": correction.perturbation_amplitude,
            "perturbation_sign": correction.perturbation_sign,
            "attempted": correction.attempted,
        },
        "corrected_validation": corrected_validation,
        "corrected_stress": branch_stress_metrics(problem, correction.state),
    })
    if not corrected_validation["checkpoint_validated"]:
        result["status"] = "PERIOD_DOUBLED_CORRECTION_RESIDUAL_FAILED"
    else:
        continuation = continue_until_utilization(
            solver,
            problem,
            correction,
            i_ref_a=current_a,
            mu_max=args.target_mu,
            mu_step=args.mu_step,
            break_threshold=args.break_threshold,
            max_wall_s=args.max_wall_s,
        )
        result.update({
            "status": "PERIOD_DOUBLED_BRANCH_COMPLETE",
            "continuation": {
                "mu": continuation.mu,
                "info": continuation.info,
            },
        })
        if args.signal_ghz is not None and args.output_port is not None:
            gain = solve_period_doubled_gain(
                circuit,
                continuation.state,
                target_basis,
                signal_ghz=args.signal_ghz,
                source_port=args.pump_port,
                out_port=args.output_port,
                sidebands=args.gain_sidebands,
                dc_branch_flux=dc_flux,
            )
            result["gain"] = asdict(gain)
        metadata_out = {
            **target_basis.to_metadata(),
            "pump_solution_dtype": "float64",
            "pump_current_a": float(current_a * continuation.mu),
            "pump_freq_ghz": float(target_basis.omega_p / (2.0 * math.pi * 1.0e9)),
            "physical_pump_freq_ghz": float(pump.pump_freq_ghz),
            "physical_pump_mode": 2,
            "dc_branch_flux": dc_flux.tolist(),
            "pump_validation_status": "VALID_CONVERGED",
            "branch_route": "floquet_period_doubled",
        }
        write_results(
            args.outdir,
            continuation.state,
            [],
            summarize_solution(problem, continuation.state),
            metadata_out,
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "period_doubled_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({
        "status": result.get("status"),
        "floquet": result.get("floquet", {}).get("classification"),
        "corrected_validation": result.get("corrected_validation"),
        "corrected_stress": result.get("corrected_stress"),
        "continuation": result.get("continuation", {}).get("info"),
    }, indent=2, default=str))
    return 0 if result.get("status") == "PERIOD_DOUBLED_BRANCH_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
