"""Controlled N=16 HB-versus-time-domain departure diagnosis.

This script is a small fixture for deciding whether the short-JTL departure is
caused by harmonic truncation, transient discretization, or a genuinely
unstable periodic orbit.  It deliberately does not run a drive sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.h1_transient_branch_transfer import (  # noqa: E402
    build_system,
    implicit_trapezoid_ramp,
    load_hb_initial,
)
from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump.hb import FullPumpProblem, HarmonicGrid  # noqa: E402
from twpa_solver.pump.solver import (  # noqa: E402
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
)


def load_fixture(path: Path, freq_ghz: float, pump_port: int) -> tuple[Any, Any, np.ndarray, np.ndarray, float, np.ndarray]:
    circuit = load_circuit(path / "circuit")
    system = build_system(path / "circuit", freq_ghz, pump_port)
    x, v, current, report = load_hb_initial(path / "hb_checkpoint", circuit, system.omega)
    q0 = x / system.phi0
    p0 = v / system.phi0
    return circuit, system, q0, p0, current, np.asarray(report["metadata"]["pump_modes"], dtype=int)


def hb_problem(system: Any, modes: np.ndarray, current: float, nt: int) -> FullPumpProblem:
    grid = HarmonicGrid(modes=modes, nt=nt, omega=system.omega)
    return FullPumpProblem(
        system.circuit.C, system.circuit.G, system.circuit.K,
        system.circuit.Bphi, system.branch, grid, system.pump_node, current,
    )


def pointwise_residual(system: Any, modes: np.ndarray, current: float, X: np.ndarray, nt: int) -> dict[str, Any]:
    grid = HarmonicGrid(modes=modes, nt=nt, omega=system.omega)
    x = grid.synthesize(X)
    xd = grid.synthesize_derivative(X, 1)
    xdd = grid.synthesize_derivative(X, 2)
    flux = system.circuit.Bphi.T @ x.T
    branch_current = np.asarray(system.branch.current(flux.T))[...,]
    source = np.zeros((nt, system.circuit.node_count), dtype=float)
    source[:, system.pump_node] = current * np.cos(2.0 * np.pi * np.arange(nt) / nt)
    residual = (
        (system.circuit.C @ xdd.T).T
        + (system.circuit.G @ xd.T).T
        + (system.circuit.K @ x.T).T
        + (system.circuit.Bphi @ branch_current.T).T
        - source
    )
    source_rms = max(float(np.sqrt(np.mean(source * source))), 1e-300)
    pointwise = np.linalg.norm(residual, axis=1)
    coeff = np.fft.rfft(residual, axis=0) / nt
    retained = np.zeros_like(residual)
    for mode in modes:
        index = int(mode)
        retained += 2.0 * np.real(
            np.exp(2j * np.pi * index * np.arange(nt)[:, None] / nt) * coeff[index][None, :]
        )
    out_band = residual - retained
    algebraic = np.flatnonzero(np.diff(system.circuit.C.indptr) == 0)
    return {
        "max_rel": float(np.max(pointwise) / source_rms),
        "rms_rel": float(np.sqrt(np.mean(residual * residual)) / source_rms),
        "retained_rms_rel": float(np.sqrt(np.mean(retained * retained)) / source_rms),
        "out_of_band_rms_rel": float(np.sqrt(np.mean(out_band * out_band)) / source_rms),
        "algebraic_max_rel": float(np.max(np.abs(residual[:, algebraic])) / source_rms) if algebraic.size else 0.0,
        "retained_coeff_rel": float(np.linalg.norm(np.asarray([coeff[int(k)] for k in modes])) / source_rms),
        "residual": residual,
    }


def one_period(system: Any, q0: np.ndarray, p0: np.ndarray, current: float, step: float, atol: float) -> dict[str, Any]:
    y0 = system.pack(q0, p0)
    started = time.perf_counter()
    theta, states, integration = implicit_trapezoid_ramp(
        system, y0, current, current, 2.0 * math.pi, 0.0, step,
        newton_tol=atol, max_newton=16,
    )
    y_end = states[:, -1]
    closure = float(np.linalg.norm(y_end - y0) / max(np.linalg.norm(y0), 1e-300))
    q_end, _ = system.unpack(y_end)
    q_start, _ = system.unpack(y0)
    phi_start = system.circuit.Bphi.T @ (system.phi0 * q_start)
    phi_end = system.circuit.Bphi.T @ (system.phi0 * q_end)
    obs_error = float(np.max(np.abs(np.sin(phi_end / system.phi0) - np.sin(phi_start / system.phi0))))
    return {
        "step_theta": step,
        "runtime_s": time.perf_counter() - started,
        "success": bool(integration["success"]),
        "steps": int(integration["steps"]),
        "closure_rel": closure,
        "junction_observable_error": obs_error,
        "algebraic_residual": 0.0,
        "message": integration["message"],
    }


def correct_hb_seed(system: Any, modes: np.ndarray, current: float, seed: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Re-solve the small fixture so a stale checkpoint cannot masquerade as HB."""
    grid = HarmonicGrid(modes=modes, nt=max(2 * int(modes.max()) + 2, 40), omega=system.omega)
    problem = FullPumpProblem(
        system.circuit.C, system.circuit.G, system.circuit.K,
        system.circuit.Bphi, system.branch, grid, system.pump_node, current,
    )
    solver = HarmonicNewtonKrylovSolver(NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=60, gmres_maxiter=100, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=True,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
        stall_ratio=0.8, stall_patience=4, solve_deadline_s=120.0,
    ))
    corrected, report = solver.solve_one(problem, seed, 1.0)
    return corrected, {
        "converged": bool(report.converged),
        "coeff_rel": float(report.coeff_rel),
        "time_rel": float(report.time_rel) if report.time_rel is not None else None,
        "newton_iterations": int(report.newton_iterations),
        "runtime_s": float(report.runtime_s),
    }


def mode_audit(system: Any) -> dict[str, Any]:
    ic = np.asarray(system.branch.critical_current, dtype=float)
    linear_k = system.circuit.K + system.circuit.Bphi @ sp.diags(ic / system.phi0) @ system.circuit.Bphi.T
    eigvals = la.eigvals(linear_k.toarray(), system.circuit.C.toarray())
    positive = np.sqrt(np.maximum(np.real(eigvals[np.isfinite(eigvals)]), 0.0))
    positive = np.sort(positive[positive > 0.0])
    ratios = positive / system.omega
    return {
        "omega_over_omega_p": ratios.tolist(),
        "omega_max_over_omega_p": float(ratios[-1]) if ratios.size else 0.0,
        "omega_max_dt_at_step_0p01": float(ratios[-1] * 0.01) if ratios.size else 0.0,
        "largest_eigenfrequencies_hz": (positive / (2.0 * math.pi)).tolist(),
    }


def flow_map(system: Any, y0: np.ndarray, current: float, step: float) -> tuple[np.ndarray, dict[str, Any]]:
    theta, states, integration = implicit_trapezoid_ramp(
        system, y0, current, current, 2.0 * math.pi, 0.0, step,
        newton_tol=1e-8, max_newton=16,
    )
    return states[:, -1], integration


def long_time_drift(system: Any, y0: np.ndarray, current: float, step: float, periods: int) -> list[float]:
    base = y0.copy()
    current_state = y0.copy()
    drift = [0.0]
    for _ in range(periods):
        current_state, integration = flow_map(system, current_state, current, step)
        if not integration["success"]:
            raise RuntimeError(integration["message"])
        drift.append(float(
            np.linalg.norm(current_state - base) / max(np.linalg.norm(base), 1e-300)
        ))
    return drift


def finite_difference_jacobian(system: Any, y: np.ndarray, current: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    base, integration = flow_map(system, y, current, step)
    if not integration["success"]:
        raise RuntimeError(integration["message"])
    jac = np.empty((y.size, y.size), dtype=float)
    for col in range(y.size):
        epsilon = 1e-6 * max(1.0, abs(float(y[col])))
        yp = y.copy(); ym = y.copy()
        yp[col] += epsilon; ym[col] -= epsilon
        fp, _ = flow_map(system, yp, current, step)
        fm, _ = flow_map(system, ym, current, step)
        jac[:, col] = (fp - fm) / (2.0 * epsilon)
    return base, jac


def shooting(system: Any, y0: np.ndarray, current: float, step: float, max_iter: int = 5) -> dict[str, Any]:
    y = y0.copy()
    initial = None
    for iteration in range(max_iter):
        end, jac = finite_difference_jacobian(system, y, current, step)
        residual = end - y
        norm = float(np.linalg.norm(residual) / max(np.linalg.norm(y), 1e-300))
        if initial is None:
            initial = norm
        if norm < 1e-8:
            break
        delta = la.solve(jac - np.eye(y.size), -residual)
        y += delta
    end, _ = flow_map(system, y, current, step)
    final = float(np.linalg.norm(end - y) / max(np.linalg.norm(y), 1e-300))
    _, monodromy = finite_difference_jacobian(system, y, current, step)
    multipliers = la.eigvals(monodromy)
    return {
        "initial_closure_rel": initial,
        "final_closure_rel": final,
        "iterations": iteration + 1,
        "orbit": y,
        "monodromy": monodromy,
        "multipliers": multipliers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "ladder_jtl16_constant025")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=ROOT / "n16_hb_td_debug")
    parser.add_argument("--nt", type=int, default=4096)
    parser.add_argument("--shoot-step", type=float, default=0.02)
    parser.add_argument("--shoot-iterations", type=int, default=3)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    _, system, q0_checkpoint, p0_checkpoint, current, modes = load_fixture(args.fixture, args.freq_ghz, args.pump_port)
    checkpoint_data = np.load(args.fixture / "hb_checkpoint" / "pump_solution.npz")
    checkpoint_X = np.asarray(checkpoint_data["X_real"]) + 1j * np.asarray(checkpoint_data["X_imag"])
    X, correction = correct_hb_seed(system, modes, current, checkpoint_X)
    corrected_grid = HarmonicGrid(modes=modes, nt=40, omega=system.omega)
    x_hb = corrected_grid.synthesize(X)[0]
    v_hb = corrected_grid.synthesize_derivative(X, 1)[0] / system.omega
    q0 = x_hb / system.phi0
    p0 = v_hb / system.phi0
    problem = hb_problem(system, modes, current, 40)
    norms = problem.norms(X, 1.0, True)
    residual = pointwise_residual(system, modes, current, X, args.nt)
    residual_array = residual.pop("residual")
    np.savez_compressed(args.outdir / "pointwise_residual.npz", residual=residual_array)
    residual_spectrum = np.abs(np.fft.rfft(residual_array, axis=0))
    plt.figure(figsize=(7, 4))
    plt.semilogy(np.arange(residual_spectrum.shape[0]), np.max(residual_spectrum, axis=1))
    plt.xlabel("harmonic index"); plt.ylabel("max residual Fourier amplitude")
    plt.tight_layout(); plt.savefig(args.outdir / "hb_residual_spectrum.png", dpi=150); plt.close()

    convergence = [one_period(system, q0, p0, current, step, 1e-8) for step in (0.02, 0.01, 0.005)]
    audit = mode_audit(system)
    y0 = system.pack(q0, p0)
    drift = long_time_drift(system, y0, current, 0.01, 10)
    shooting_result = shooting(
        system, y0, current, args.shoot_step, max_iter=args.shoot_iterations,
    )
    multipliers = shooting_result.pop("multipliers")
    orbit = shooting_result.pop("orbit")
    shooting_result.pop("monodromy")
    np.savez_compressed(args.outdir / "shooting_orbit.npz", y=orbit)
    plt.figure(figsize=(7, 4))
    plt.semilogy(range(len(drift)), np.maximum(drift, 1e-18), "o-")
    plt.xlabel("pump period"); plt.ylabel("stroboscopic error")
    plt.tight_layout(); plt.savefig(args.outdir / "stroboscopic_error_growth.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4))
    steps = [row["step_theta"] for row in convergence]
    errors = [row["closure_rel"] for row in convergence]
    plt.loglog(steps, errors, "o-")
    plt.xlabel("step in pump phase"); plt.ylabel("one-period closure error")
    plt.tight_layout(); plt.savefig(args.outdir / "one_period_convergence.png", dpi=150); plt.close()
    sorted_multipliers = multipliers[np.argsort(-np.abs(multipliers))]
    report = {
        "fixture": str(args.fixture),
        "current_a": current,
        "modes": modes.tolist(),
        "hb_norms": norms,
        "checkpoint_correction": correction,
        "checkpoint_to_corrected_rel": float(np.linalg.norm(X - checkpoint_X) / max(np.linalg.norm(X), 1e-300)),
        "pointwise_residual": residual,
        "one_period_convergence": convergence,
        "mode_audit": audit,
        "long_time_stroboscopic_drift": drift,
        "shooting": shooting_result,
        "floquet_multipliers": [
            {"real": float(value.real), "imag": float(value.imag), "abs": float(abs(value))}
            for value in sorted_multipliers[:8]
        ],
        "status": "INCONCLUSIVE",
    }
    (args.outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
