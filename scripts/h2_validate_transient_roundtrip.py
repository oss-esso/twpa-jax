"""Milestone H2: validate HB -> time domain -> FFT -> HB at fixed drive.

This script intentionally does not ramp the pump or inspect post-fold physics.
It exercises the existing H1 DAE reduction from two saved, converged 7.9 GHz
HB checkpoints and reports integration/refinement errors against the known
analytic HB waveform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from h1_transient_branch_transfer import (
    ROOT,
    build_system,
    implicit_euler_ramp,
    implicit_trapezoid_ramp,
    load_hb_initial,
)
from twpa_solver.core.constants import PHI0_REDUCED
from twpa_solver.pump.hb import FullPumpProblem, HarmonicGrid
from twpa_solver.pump.solver import HarmonicNewtonKrylovSolver, NewtonKrylovSettings


def checkpoint_x(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    data = np.load(checkpoint / "pump_solution.npz")
    X = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(data["X_imag"], dtype=float)
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    return X, np.asarray(data["pump_modes"], dtype=int), float(report["metadata"]["pump_current_a"]), report


def hb_waveform(X: np.ndarray, modes: np.ndarray, theta: np.ndarray) -> np.ndarray:
    basis = np.exp(1j * theta[:, None] * modes[None, :])
    return 2.0 * np.real(basis @ X)


def hb_velocity(X: np.ndarray, modes: np.ndarray, omega: float, theta: np.ndarray) -> np.ndarray:
    basis = np.exp(1j * theta[:, None] * modes[None, :])
    return 2.0 * np.real(basis @ ((1j * modes * omega)[:, None] * X))


def interpolate_states(theta: np.ndarray, states: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(query, theta, row) for row in states])


def evaluate_case(
    system: Any, checkpoint: Path, periods: int, step_theta: float,
    *, run_roundtrip: bool, integrator_name: str, plot_dir: Path | None = None,
) -> dict[str, Any]:
    X, modes, current, report = checkpoint_x(checkpoint)
    x0, w0, _current, _ = load_hb_initial(checkpoint, system.circuit, system.omega)
    y0 = system.pack(x0 / PHI0_REDUCED, w0 / PHI0_REDUCED)
    integrate = (
        implicit_trapezoid_ramp if integrator_name == "trapezoid"
        else implicit_euler_ramp
    )
    theta, states, integration = integrate(
        system, y0, current, current, 2.0 * math.pi * periods, 0.0,
        step_theta,
        newton_tol=1e-6 if integrator_name == "trapezoid" else 1e-3,
        max_newton=12,
    )
    qd = np.asarray([system.unpack(states[:, i])[0] for i in range(states.shape[1])])
    pd_rows = []
    for i, angle in enumerate(theta):
        q, p = system.unpack(states[:, i])
        p[system.algebraic] = system.algebraic_velocity(
            q, p[system.differential], system.source(angle, current, current, 0.0)
        )
        pd_rows.append(p)
    pd = np.asarray(pd_rows)
    x_td = PHI0_REDUCED * qd
    v_td = system.omega * PHI0_REDUCED * pd
    x_hb = hb_waveform(X, modes, theta)
    v_hb = hb_velocity(X, modes, system.omega, theta)
    x_scale = max(float(np.linalg.norm(x_hb) / math.sqrt(x_hb.size)), 1e-300)
    v_scale = max(float(np.linalg.norm(v_hb) / math.sqrt(v_hb.size)), 1e-300)
    e_x = np.linalg.norm(x_td - x_hb, axis=1) / np.maximum(np.linalg.norm(x_hb, axis=1), x_scale)
    e_v = np.linalg.norm(v_td - v_hb, axis=1) / v_scale
    strobe_theta = 2.0 * math.pi * np.arange(periods + 1)
    strobe_states = interpolate_states(theta, states, strobe_theta)
    strobe_x = PHI0_REDUCED * np.asarray([
        system.unpack(strobe_states[:, i])[0] for i in range(strobe_states.shape[1])
    ])
    strobe_scale = max(float(np.linalg.norm(x_hb[0])), x_scale)
    d_strobe = np.linalg.norm(strobe_x - strobe_x[0], axis=1) / strobe_scale
    final_start = max(0.0, theta[-1] - 2.0 * math.pi)
    fft_theta = np.linspace(final_start, theta[-1], 40, endpoint=False)
    fft_states = interpolate_states(theta, states, fft_theta)
    fft_x = PHI0_REDUCED * np.asarray([
        system.unpack(fft_states[:, i])[0] for i in range(fft_states.shape[1])
    ])
    X_td = np.exp(-1j * fft_theta[:, None] * modes[None, :]).T @ fft_x / fft_x.shape[0]
    X_td *= 1.0
    e_X = float(np.linalg.norm(X_td - X) / max(np.linalg.norm(X), 1e-300))
    phi_td = (system.circuit.Bphi.T @ qd.T).T
    phi_hb = (system.circuit.Bphi.T @ (x_hb.T)).T / PHI0_REDUCED
    obs = {
        "td_rj": float(np.max(np.abs(np.sin(phi_td)))),
        "hb_rj": float(np.max(np.abs(np.sin(phi_hb)))),
        "td_max_phi": float(np.max(np.abs(phi_td))),
        "hb_max_phi": float(np.max(np.abs(phi_hb))),
        "td_min_cos": float(np.min(np.cos(phi_td))),
        "hb_min_cos": float(np.min(np.cos(phi_hb))),
    }
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        representative = int(np.argmax(np.max(np.abs(phi_hb), axis=0)))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(theta / (2.0 * math.pi), x_td[:, representative], label="TD")
        ax.plot(theta / (2.0 * math.pi), x_hb[:, representative], "--", label="HB")
        ax.set(xlabel="pump periods", ylabel="representative flux state")
        ax.legend()
        fig.tight_layout(); fig.savefig(plot_dir / "transient_vs_hb_waveform.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(theta / (2.0 * math.pi), np.max(np.abs(np.sin(phi_td)), axis=1))
        ax.set(xlabel="pump periods", ylabel="max |sin(phi)|")
        fig.tight_layout(); fig.savefig(plot_dir / "junction_observable.png", dpi=150); plt.close(fig)
    algebraic = system.algebraic
    if algebraic.size:
        alg_residuals = []
        for i, angle in enumerate(theta):
            q, p = system.unpack(states[:, i])
            p[algebraic] = system.algebraic_velocity(
                q, p[system.differential], system.source(angle, current, current, 0.0)
            )
            flux = PHI0_REDUCED * (system.circuit.Bphi.T @ q)
            currents = np.asarray(system.branch.current(flux[None, :]))[0]
            source = system.source(angle, current, current, 0.0)
            residual = system.circuit.G[algebraic] @ (system.omega * PHI0_REDUCED * p)
            residual = residual + system.circuit.K[algebraic] @ (PHI0_REDUCED * q)
            residual = residual + system.circuit.Bphi[algebraic] @ currents - source[algebraic]
            alg_residuals.append(float(np.linalg.norm(residual) / max(abs(current), 1e-300)))
        dae_error = float(max(alg_residuals))
    else:
        dae_error = 0.0
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint), "current_a": current, "step_theta": step_theta,
        "integrator_name": integrator_name,
        "periods": periods, "integration": integration,
        "e_x_rms": float(np.sqrt(np.mean(e_x**2))), "e_x_max": float(np.max(e_x)),
        "e_v_rms": float(np.sqrt(np.mean(e_v**2))),
        "e_X": e_X, "strobe_max": float(np.max(d_strobe)),
        "strobe_periods": list(range(periods + 1)),
        "strobe_errors": d_strobe.tolist(),
        "dae_constraint_max": dae_error, "observables": obs,
    }
    if run_roundtrip:
        grid = HarmonicGrid(modes=modes, nt=40, omega=system.omega)
        problem = FullPumpProblem(
            system.circuit.C, system.circuit.G, system.circuit.K,
            system.circuit.Bphi, system.branch, grid, system.pump_node, current,
        )
        solver = HarmonicNewtonKrylovSolver(NewtonKrylovSettings(
            newton_tol=1e-9, max_newton=15, gmres_rtol=1e-7, gmres_atol=0.0,
            gmres_restart=60, gmres_maxiter=80, min_alpha=1.0 / 1024.0,
            preconditioner="real_coupled", compute_time_residual=True,
            verbose=False, continuation_predictor="none", jvp_mode="aft",
            stall_ratio=0.8, stall_patience=4, solve_deadline_s=180.0,
        ))
        X_restart, hb_report = solver.solve_one(problem, X_td, 1.0)
        result["roundtrip"] = {
            "converged": bool(hb_report.converged),
            "coeff_rel": float(hb_report.coeff_rel),
            "restart_error": float(np.linalg.norm(X_restart - X) / max(np.linalg.norm(X), 1e-300)),
            "runtime_s": hb_report.runtime_s,
        }
    return result


def write_plots(outdir: Path, case: dict[str, Any], refinement: list[dict[str, Any]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row["step_theta"] for row in refinement], [row["e_X"] for row in refinement], "o-")
    ax.set(xlabel="step theta", ylabel="Fourier coefficient error")
    ax.set_xscale("log"); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(outdir / "fourier_error_vs_resolution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(case["strobe_periods"], case["strobe_errors"], "o-")
    ax.set(xlabel="pump period", ylabel="stroboscopic state error")
    fig.tight_layout(); fig.savefig(outdir / "stroboscopic_error.png", dpi=150); plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "h2_validation")
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--periods", type=int, default=20)
    parser.add_argument("--refinement-periods", type=int, default=20)
    parser.add_argument("--steps", default="0.2,0.1,0.05")
    parser.add_argument("--no-roundtrip", action="store_true")
    parser.add_argument("--plot-waveforms", action="store_true")
    parser.add_argument("--integrator", choices=("euler", "trapezoid"), default="trapezoid")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoints = [
        ROOT / "g1_current_79" / "pass" / "points" / "point_0004_p_m23p8947dbm_fp_7p9ghz" / "pump",
        ROOT / "g1_current_79" / "pass" / "points" / "point_0012_p_m19p6842dbm_fp_7p9ghz" / "pump",
    ]
    system = build_system(args.circuit_dir, 7.9, 4)
    steps = [float(value) for value in args.steps.split(",")]
    cases = []
    for checkpoint in checkpoints:
        cases.append(evaluate_case(
            system, checkpoint, args.periods, steps[-1],
            run_roundtrip=not args.no_roundtrip,
            integrator_name=args.integrator,
            plot_dir=args.outdir if args.plot_waveforms and checkpoint == checkpoints[0] else None,
        ))
    refinement = [evaluate_case(
        system, checkpoints[1], args.refinement_periods, step,
        run_roundtrip=False, integrator_name=args.integrator,
    ) for step in steps]
    summary = {"cases": cases, "refinement": refinement}
    refinement_converges = (
        len(refinement) < 2
        or refinement[-1]["e_X"] < refinement[0]["e_X"]
    )
    summary["validated"] = bool(
        all(case["e_X"] <= 1e-3 and case["dae_constraint_max"] < 1e-8 for case in cases)
        and refinement[-1]["e_X"] <= 1e-3
        and refinement_converges
        and all(case.get("roundtrip", {}).get("converged", False) for case in cases)
    )
    (args.outdir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.outdir / "validation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint", "step_theta", "e_X", "strobe_max",
                "dae_constraint_max", "e_x_rms", "e_x_max",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(row for row in refinement)
    write_plots(args.outdir, cases[-1], refinement)
    print(json.dumps({"validated": summary["validated"], "cases": cases, "refinement": refinement}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
