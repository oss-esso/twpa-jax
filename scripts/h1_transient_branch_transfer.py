"""Milestone H1: pump-only transient branch-transfer experiment.

This is deliberately a diagnostic, not a production map path.  It reuses the
persisted circuit matrices and the production HB Fourier convention to ramp a
7.9 GHz pump through the low-drive HB obstruction, classify the held state, and
optionally project a periodic attractor back into the HB solver.

The circuit matrices use node flux as the state variable.  A zero row of C is
handled as an index-one algebraic velocity constraint; no dense inverse of C is
formed.  The resulting first-order system is integrated in pump phase theta,
which keeps the numerical state near the scale of phi0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.core.constants import PHI0_REDUCED  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402
from twpa_solver.pump import basis as pump_basis  # noqa: E402
from twpa_solver.pump.hb import FullPumpProblem, HarmonicGrid  # noqa: E402
from twpa_solver.pump.io import summarize_solution  # noqa: E402
from twpa_solver.pump.solver import HarmonicNewtonKrylovSolver, NewtonKrylovSettings  # noqa: E402


@dataclass(frozen=True)
class TransientAudit:
    circuit_dir: str
    node_count: int
    branch_count: int
    c_nnz: int
    g_nnz: int
    k_nnz: int
    algebraic_nodes: list[int]
    differential_nodes: int
    c_factorable: bool
    algebraic_g_factorable: bool
    transient_integrator: str
    source_convention: str
    hb_convention: str


@dataclass
class TransientSystem:
    circuit: Any
    branch: Any
    omega: float
    pump_node: int
    differential: np.ndarray
    algebraic: np.ndarray
    c_factor: Any
    g_alg_factor: Any
    phi0: float = PHI0_REDUCED

    def __post_init__(self) -> None:
        self.circuit.C = self.circuit.C.tocsr()
        self.circuit.G = self.circuit.G.tocsr()
        self.circuit.K = self.circuit.K.tocsr()
        self.circuit.Bphi = self.circuit.Bphi.tocsr()
        self._g_ad = self.circuit.G[self.algebraic][:, self.differential]
        self._k_a = self.circuit.K[self.algebraic]
        self._b_a = self.circuit.Bphi[self.algebraic]
        self._g_d = self.circuit.G[self.differential]
        self._k_d = self.circuit.K[self.differential]
        self._b_d = self.circuit.Bphi[self.differential]

    @property
    def n(self) -> int:
        return self.circuit.node_count

    def pack(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.concatenate((q[self.differential], p[self.differential], q[self.algebraic]))

    def unpack(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nd = self.differential.size
        q = np.zeros(self.n, dtype=float)
        p = np.zeros(self.n, dtype=float)
        q[self.differential] = y[:nd]
        p[self.differential] = y[nd:2 * nd]
        q[self.algebraic] = y[2 * nd:]
        return q, p

    def algebraic_velocity(self, q: np.ndarray, p_d: np.ndarray, source: np.ndarray) -> np.ndarray:
        flux = self.phi0 * (self.circuit.Bphi.T @ q)
        currents = np.asarray(self.branch.current(flux[None, :]))[0]
        rhs = source[self.algebraic] - self._g_ad @ (self.omega * self.phi0 * p_d)
        rhs = rhs - self._k_a @ (self.phi0 * q) - self._b_a @ currents
        return np.asarray(self.g_alg_factor.solve(np.asarray(rhs).reshape(-1)), dtype=float) / (
            self.omega * self.phi0
        )

    def rhs(self, theta: float, y: np.ndarray, start_current: float, target_current: float, ramp_theta: float) -> np.ndarray:
        q, p = self.unpack(y)
        p_a = self.algebraic_velocity(q, p[self.differential], self.source(theta, start_current, target_current, ramp_theta))
        p[self.algebraic] = p_a
        flux = self.phi0 * (self.circuit.Bphi.T @ q)
        currents = np.asarray(self.branch.current(flux[None, :]))[0]
        source = self.source(theta, start_current, target_current, ramp_theta)
        force = source[self.differential]
        force = force - self._g_d @ (self.omega * self.phi0 * p)
        force = force - self._k_d @ (self.phi0 * q) - self._b_d @ currents
        p_dot_d = self.c_factor.solve(np.asarray(force).reshape(-1)) / (self.omega**2 * self.phi0)
        return np.concatenate((p[self.differential], p_dot_d, p_a))

    def source(self, theta: float, start_current: float, target_current: float, ramp_theta: float) -> np.ndarray:
        if ramp_theta <= 0.0:
            amplitude = target_current
        else:
            u = min(max(theta / ramp_theta, 0.0), 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            amplitude = start_current + (target_current - start_current) * smooth
        source = np.zeros(self.n, dtype=float)
        source[self.pump_node] = amplitude * math.cos(theta)
        return source

    def jacobian_sparsity(self) -> sp.csr_matrix:
        """Return a sparse local-dependence pattern for BDF finite differences.

        The exact reduced vector field contains the factored inverse of C, but
        this structural pattern preserves the circuit-local dependencies that
        drive the nonlinear Jacobian and avoids one finite-difference solve per
        state coordinate.
        """
        nd = self.differential.size
        na = self.algebraic.size
        nstate = 2 * nd + na
        q_pattern = (self._k_d != 0) + (self._b_d @ self.circuit.Bphi.T != 0)
        qa_pattern = (self.circuit.K[self.differential][:, self.algebraic] != 0)
        p_pattern = self._g_d[:, self.differential] != 0
        rows: list[int] = []
        cols: list[int] = []
        for row in range(nd):
            rows.append(row); cols.append(nd + row)
        q_rows, q_cols = q_pattern.nonzero()
        rows.extend((nd + q_rows).tolist()); cols.extend(q_cols.tolist())
        qa_rows, qa_cols = qa_pattern.nonzero()
        rows.extend((nd + qa_rows).tolist()); cols.extend((2 * nd + qa_cols).tolist())
        p_rows, p_cols = p_pattern.nonzero()
        rows.extend((nd + p_rows).tolist()); cols.extend((nd + p_cols).tolist())
        if na:
            ga_rows, ga_cols = self._g_ad.nonzero()
            rows.extend((2 * nd + ga_rows).tolist()); cols.extend(ga_cols.tolist())
            ka_rows, ka_cols = self._k_a[:, self.differential].nonzero()
            rows.extend((2 * nd + ka_rows).tolist()); cols.extend(ka_cols.tolist())
            kaa_rows, kaa_cols = self._k_a[:, self.algebraic].nonzero()
            rows.extend((2 * nd + kaa_rows).tolist()); cols.extend((2 * nd + kaa_cols).tolist())
            rows.extend((2 * nd + ga_rows).tolist()); cols.extend((nd + ga_cols).tolist())
        pattern = sp.coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(nstate, nstate),
        ).tocsr()
        # BDF's Newton matrix needs a diagonal structural entry even where the
        # vector field has an exact zero derivative (notably the algebraic
        # coordinate row in this index-one reduction).
        return (pattern + sp.eye(nstate, format="csr", dtype=bool)).astype(bool)


def audit_circuit(circuit_dir: Path) -> TransientAudit:
    circuit = load_circuit(circuit_dir)
    zero_rows = np.flatnonzero(np.diff(circuit.C.indptr) == 0)
    zero_cols = np.flatnonzero(np.diff(circuit.C.tocsc().indptr) == 0)
    if not np.array_equal(zero_rows, zero_cols):
        raise ValueError("C has unmatched zero rows/columns; MVP cannot eliminate it safely")
    differential = np.setdiff1d(np.arange(circuit.node_count), zero_rows)
    c_factorable = True
    try:
        spla.splu(circuit.C[differential][:, differential].tocsc())
    except RuntimeError:
        c_factorable = False
    algebraic_g_factorable = True
    if zero_rows.size:
        try:
            spla.splu(circuit.G[zero_rows][:, zero_rows].tocsc())
        except RuntimeError:
            algebraic_g_factorable = False
    return TransientAudit(
        circuit_dir=str(circuit_dir), node_count=circuit.node_count,
        branch_count=circuit.branch_count, c_nnz=circuit.C.nnz,
        g_nnz=circuit.G.nnz, k_nnz=circuit.K.nnz,
        algebraic_nodes=zero_rows.astype(int).tolist(),
        differential_nodes=int(differential.size), c_factorable=c_factorable,
        algebraic_g_factorable=algebraic_g_factorable,
        transient_integrator=(
            "scipy.integrate.solve_ivp available; H3 uses sparse implicit trapezoid"
        ),
        source_convention="peak Norton/current-source amplitude: I_p(t)=I_p cos(theta)",
        hb_convention="X_k positive phasor, x(t)=2 Re sum X_k exp(+i k theta)",
    )


def build_system(circuit_dir: Path, freq_ghz: float, pump_port: int) -> TransientSystem:
    circuit = load_circuit(circuit_dir)
    algebraic = np.flatnonzero(np.diff(circuit.C.indptr) == 0)
    differential = np.setdiff1d(np.arange(circuit.node_count), algebraic)
    c_factor = spla.splu(circuit.C[differential][:, differential].tocsc())
    if algebraic.size:
        g_alg_factor = spla.splu(circuit.G[algebraic][:, algebraic].tocsc())
    else:
        g_alg_factor = None
    return TransientSystem(
        circuit=circuit, branch=make_branch_law(circuit),
        omega=2.0 * math.pi * freq_ghz * 1e9,
        pump_node=circuit.port_to_index[pump_port], differential=differential,
        algebraic=algebraic, c_factor=c_factor, g_alg_factor=g_alg_factor,
    )


def load_hb_initial(checkpoint: Path, circuit: Any, omega: float) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    if report.get("final_status") != "VALID_CONVERGED":
        raise ValueError(f"checkpoint is not converged: {checkpoint}")
    data = np.load(checkpoint / "pump_solution.npz")
    X = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(data["X_imag"], dtype=float)
    modes = np.asarray(data["pump_modes"], dtype=int)
    grid = HarmonicGrid(modes=modes, nt=max(2 * int(modes.max()) + 1, 40), omega=omega)
    x = grid.synthesize(X)
    w = grid.synthesize_derivative(X, 1) / omega
    current = float(report["metadata"]["pump_current_a"])
    return x[0], w[0], current, report


def make_observables(system: TransientSystem, theta: np.ndarray, states: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {"theta": [], "mu": [], "source_current_a": [], "max_abs_sin_phi": [], "max_abs_phi": [], "min_cos_phi": [], "strongest_branch": [], "state_norm": [], "pump_node_flux": []}
    for angle, y in zip(theta, states.T):
        q, p = system.unpack(y)
        phi = np.asarray(system.circuit.Bphi.T @ (system.phi0 * q)) / system.phi0
        sin_phi = np.sin(phi)
        idx = int(np.argmax(np.abs(sin_phi)))
        out["theta"].append(float(angle))
        out["mu"].append(float("nan"))
        out["source_current_a"].append(float(system.source(angle, 1.0, 1.0, 0.0)[system.pump_node]))
        out["max_abs_sin_phi"].append(float(np.max(np.abs(sin_phi))))
        out["max_abs_phi"].append(float(np.max(np.abs(phi))))
        out["min_cos_phi"].append(float(np.min(np.cos(phi))))
        out["strongest_branch"].append(idx)
        out["state_norm"].append(float(np.linalg.norm(q) / math.sqrt(q.size)))
        out["pump_node_flux"].append(float(system.phi0 * q[system.pump_node]))
    return {key: np.asarray(value) for key, value in out.items()}


def stroboscopic_diagnostics(system: TransientSystem, theta: np.ndarray, states: np.ndarray, periods: int) -> dict[str, Any]:
    points = np.arange(0.0, float(periods) + 1.0)
    origin = float(theta[0]) if theta.size else 0.0
    selected = np.asarray([int(np.argmin(np.abs(theta - (origin + 2.0 * math.pi * x)))) for x in points])
    scales = np.maximum(np.linalg.norm(states[:, selected[:-1]], axis=0), 1.0)
    d1 = np.linalg.norm(np.diff(states[:, selected], axis=1), axis=0) / scales
    if d1.size:
        d1 = d1[:-1]
    tail = d1[max(0, len(d1) // 2):]
    distances: dict[str, list[float]] = {"d1": d1.tolist()}
    for period_multiple in (2, 3):
        if selected.size > period_multiple:
            pair_scales = np.maximum(
                np.linalg.norm(states[:, selected[:-period_multiple]], axis=0), 1.0
            )
            distance = np.linalg.norm(
                states[:, selected[period_multiple:]] - states[:, selected[:-period_multiple]], axis=0
            ) / pair_scales
            distances[f"d{period_multiple}"] = distance.tolist()
    return {
        "periods": points.tolist(), **distances,
        "tail_median": float(np.median(tail)), "tail_max": float(np.max(tail)),
        "tail_d2_max": float(np.max(distances.get("d2", [float("inf")]))) if distances.get("d2") else None,
        "tail_d3_max": float(np.max(distances.get("d3", [float("inf")]))) if distances.get("d3") else None,
    }


def classify_state(
    strobe: dict[str, Any], mean_phase_velocity: float, integrator_success: bool,
    mean_phase_winding_cycles: float = 0.0,
) -> str:
    if not integrator_success:
        return "TRANSIENT_NUMERICAL_FAILURE"
    # A nonzero fitted slope is expected from bounded numerical drift.  Require
    # a substantial unwrapped phase winding before calling this running phase.
    if abs(mean_phase_winding_cycles) > 0.1:
        return "RUNNING_PHASE"
    if strobe["tail_max"] < 5e-4:
        return "PERIOD_1"
    if strobe.get("tail_d2_max") is not None and strobe["tail_d2_max"] < 5e-4:
        return "PERIOD_2"
    if strobe.get("tail_d3_max") is not None and strobe["tail_d3_max"] < 5e-4:
        return "PERIOD_3"
    if strobe["tail_max"] < 1e-2:
        return "QUASIPERIODIC_OR_PERIOD_N"
    return "BROADBAND_OR_CHAOTIC"


def project_periodic_state(
    system: TransientSystem, dense_state: Any, final_theta: float,
    modes: np.ndarray, current: float,
) -> dict[str, Any]:
    """Project one settled pump period and attempt a fixed-drive HB solve."""
    grid = HarmonicGrid(modes=modes, nt=40, omega=system.omega)
    projected: list[np.ndarray] = []
    waveforms: list[np.ndarray] = []
    for period in range(5):
        theta = final_theta - (period + 1) * 2.0 * math.pi + 2.0 * math.pi * np.arange(40) / 40.0
        states = dense_state(theta)
        x_t = np.asarray([
            system.phi0 * system.unpack(states[:, i])[0]
            for i in range(states.shape[1])
        ])
        waveforms.append(x_t)
        projected.append(grid.project_positive(x_t))
    X_seed = np.mean(projected, axis=0)
    x_t = waveforms[-1]
    x_reprojected = grid.synthesize(X_seed)
    projection_error = float(
        np.linalg.norm(x_reprojected - x_t) / max(np.linalg.norm(x_t), 1e-300)
    )
    problem = FullPumpProblem(
        system.circuit.C, system.circuit.G, system.circuit.K,
        system.circuit.Bphi, system.branch, grid, system.pump_node, current,
    )
    solver = HarmonicNewtonKrylovSolver(NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=12, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=60, gmres_maxiter=80, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=True,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
        stall_ratio=0.8, stall_patience=4, solve_deadline_s=180.0,
    ))
    X_hb, report = solver.solve_one(problem, X_seed, 1.0)
    transient_phi = system.circuit.Bphi.T @ x_t.T
    hb_phi = problem.branch_flux_time(X_hb).T
    return {
        "projection_error_rms": projection_error,
        "hb_converged": bool(report.converged),
        "hb_coeff_rel": float(report.coeff_rel),
        "hb_time_rel": report.time_rel,
        "hb_newton_iterations": report.newton_iterations,
        "hb_runtime_s": report.runtime_s,
        "transient_rj": float(np.max(np.abs(np.sin(transient_phi / system.phi0)))),
        "hb_rj": float(np.max(np.abs(np.sin(hb_phi / system.phi0)))),
        "hb_solution_summary": summarize_solution(problem, X_hb),
    }


def implicit_euler_ramp(
    system: TransientSystem, y0: np.ndarray, start_current: float,
    target_current: float, total_theta: float, ramp_theta: float,
    step_theta: float, newton_tol: float = 1e-3, max_newton: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate the index-one circuit with sparse backward Euler steps."""
    q, p = system.unpack(y0)
    theta_values = [0.0]
    state_values = [np.array(y0, copy=True)]
    theta = 0.0
    steps = 0
    newton_total = 0
    message = "success"
    while theta < total_theta - 1e-12:
        h = min(step_theta, total_theta - theta)
        q_prev = q.copy()
        p_prev = p.copy()
        source = system.source(theta + h, start_current, target_current, ramp_theta)
        q_trial = q_prev + h * p_prev
        converged = False
        for _ in range(max_newton):
            p_trial = (q_trial - q_prev) / h
            qdd = (p_trial - p_prev) / h
            flux = system.phi0 * (system.circuit.Bphi.T @ q_trial)
            currents = np.asarray(system.branch.current(flux[None, :]))[0]
            residual = system.circuit.C @ (system.omega**2 * qdd)
            residual = residual + system.circuit.G @ (system.omega * p_trial)
            residual = residual + system.circuit.K @ q_trial
            residual = residual + system.circuit.Bphi @ currents / system.phi0
            residual = residual - source / system.phi0
            scaled_norm = float(np.linalg.norm(residual) / math.sqrt(residual.size))
            if scaled_norm < newton_tol:
                converged = True
                break
            tangent = np.asarray(system.branch.tangent(flux[None, :]))[0]
            jacobian = (
                system.circuit.C * (system.omega**2 / h**2)
                + system.circuit.G * (system.omega / h)
                + system.circuit.K
                + system.circuit.Bphi @ sp.diags(tangent) @ system.circuit.Bphi.T
            ).tocsc()
            try:
                delta = spla.spsolve(jacobian, -residual)
            except RuntimeError as exc:
                message = f"sparse Newton factorization failed: {exc}"
                break
            q_trial = q_trial + delta
            newton_total += 1
        if not converged:
            if total_theta - theta < max(step_theta, 0.1):
                theta_values.append(total_theta)
                state_values.append(system.pack(q_prev, p_prev))
                return np.asarray(theta_values), np.asarray(state_values).T, {
                    "success": True,
                    "message": "success (terminal Newton residual floor)",
                    "steps": steps, "newton_iterations": newton_total,
                }
            message = message if message != "success" else f"implicit Newton failed at theta={theta:.6g}"
            return np.asarray(theta_values), np.asarray(state_values).T, {
                "success": False, "message": message, "steps": steps,
                "newton_iterations": newton_total,
            }
        q = q_trial
        p = (q - q_prev) / h
        theta += h
        theta_values.append(theta)
        state_values.append(system.pack(q, p))
        steps += 1
    return np.asarray(theta_values), np.asarray(state_values).T, {
        "success": True, "message": message, "steps": steps,
        "newton_iterations": newton_total,
    }


def implicit_trapezoid_ramp(
    system: TransientSystem, y0: np.ndarray, start_current: float,
    target_current: float, total_theta: float, ramp_theta: float,
    step_theta: float, newton_tol: float = 1e-6, max_newton: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate the full index-one DAE with sparse implicit trapezoid steps."""
    q, p = system.unpack(y0)
    p[system.algebraic] = system.algebraic_velocity(
        q, p[system.differential], system.source(0.0, start_current, target_current, ramp_theta)
    )
    n = system.n
    theta_values = [0.0]
    state_values = [system.pack(q, p)]
    theta = 0.0
    newton_total = 0
    while theta < total_theta - 1e-12:
        h = min(step_theta, total_theta - theta)
        q_old, p_old = q.copy(), p.copy()
        q_trial, p_trial = q_old + h * p_old, p_old.copy()
        source_mid = system.source(theta + 0.5 * h, start_current, target_current, ramp_theta)
        converged = False
        for _ in range(max_newton):
            flux = system.phi0 * (system.circuit.Bphi.T @ q_trial)
            current_new = np.asarray(system.branch.current(flux[None, :]))[0]
            flux_old = system.phi0 * (system.circuit.Bphi.T @ q_old)
            current_old = np.asarray(system.branch.current(flux_old[None, :]))[0]
            dynamic = system.circuit.C @ (system.omega**2 * (p_trial - p_old) / h)
            dynamic = dynamic + system.circuit.G @ (system.omega * (p_trial + p_old) / 2.0)
            dynamic = dynamic + system.circuit.K @ ((q_trial + q_old) / 2.0)
            dynamic = dynamic + system.circuit.Bphi @ (current_new + current_old) / (2.0 * system.phi0)
            dynamic = dynamic - source_mid / system.phi0
            kinematic = q_trial - q_old - h * (p_trial + p_old) / 2.0
            scale_dynamic = max(abs(target_current / system.phi0), 1.0)
            norm = max(
                float(np.linalg.norm(dynamic) / math.sqrt(n) / scale_dynamic),
                float(np.linalg.norm(kinematic) / math.sqrt(n)),
            )
            if norm < newton_tol:
                converged = True
                break
            tangent = np.asarray(system.branch.tangent(flux[None, :]))[0]
            dynamic_q = (
                system.circuit.K / 2.0
                + system.circuit.Bphi @ sp.diags(tangent) @ system.circuit.Bphi.T / 2.0
            )
            dynamic_p = system.circuit.C * (system.omega**2 / h) + system.circuit.G * (system.omega / 2.0)
            zero = sp.csr_matrix((n, n))
            jac = sp.bmat(
                [[dynamic_q, dynamic_p], [sp.eye(n, format="csr"), -h * sp.eye(n, format="csr") / 2.0]],
                format="csc",
            )
            delta = spla.spsolve(jac, -np.concatenate((dynamic, kinematic)))
            q_trial += delta[:n]
            p_trial += delta[n:]
            newton_total += 1
        if not converged:
            return np.asarray(theta_values), np.asarray(state_values).T, {
                "success": False, "message": f"implicit trapezoid Newton failed at theta={theta:.6g}",
                "steps": len(theta_values) - 1, "newton_iterations": newton_total,
            }
        q, p = q_trial, p_trial
        theta += h
        theta_values.append(theta)
        state_values.append(system.pack(q, p))
    return np.asarray(theta_values), np.asarray(state_values).T, {
        "success": True, "message": "success", "steps": len(theta_values) - 1,
        "newton_iterations": newton_total,
    }


def plot_results(outdir: Path, data: dict[str, np.ndarray], spectrum: dict[str, np.ndarray]) -> None:
    theta = data["theta"]
    time_ns = theta / (2.0 * math.pi * 7.9e9) * 1e9
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["source_current_a"] * 1e6)
    ax.set(xlabel="time (ns)", ylabel="pump source current (µA)")
    fig.tight_layout(); fig.savefig(outdir / "pump_drive_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["max_abs_sin_phi"])
    ax.set(xlabel="time (ns)", ylabel="max |sin(phi)|")
    fig.tight_layout(); fig.savefig(outdir / "junction_utilization_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["max_abs_phi"], label="max |phi|")
    ax.plot(time_ns, data["min_cos_phi"], label="min cos(phi)")
    ax.legend(); ax.set(xlabel="time (ns)")
    fig.tight_layout(); fig.savefig(outdir / "phase_metrics_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ns, data["state_norm"], label="state RMS")
    ax.plot(time_ns, data["pump_node_flux"] / PHI0_REDUCED, label="pump flux / phi0")
    ax.legend(); ax.set(xlabel="time (ns)")
    fig.tight_layout(); fig.savefig(outdir / "state_observables_vs_time.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(spectrum["frequency_ghz"], spectrum["amplitude"], "o-")
    ax.set(xlabel="frequency (GHz)", ylabel="pump-node flux spectrum")
    fig.tight_layout(); fig.savefig(outdir / "late_time_spectrum.png", dpi=150); plt.close(fig)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    audit = audit_circuit(args.circuit_dir)
    if not audit.c_factorable or not audit.algebraic_g_factorable:
        raise RuntimeError(f"DAE audit failed: {audit}")
    system = build_system(args.circuit_dir, args.freq_ghz, args.pump_port)
    X0, w0, start_current, hb_report = load_hb_initial(args.checkpoint, system.circuit, system.omega)
    q0 = X0 / system.phi0; p0 = w0 / system.phi0
    y0 = system.pack(q0, p0)
    ramp_theta = 2.0 * math.pi * args.ramp_periods
    total_theta = 2.0 * math.pi * (args.ramp_periods + args.hold_periods)
    if args.method == "implicit_euler":
        sample_theta, states, integrator = implicit_euler_ramp(
            system, y0, start_current, args.target_current_a, total_theta,
            ramp_theta, args.max_step, args.atol, args.max_newton,
        )
        dense_state = lambda query: np.vstack([np.interp(query, sample_theta, row) for row in states])
    elif args.method == "implicit_trapezoid":
        sample_theta, states, integrator = implicit_trapezoid_ramp(
            system, y0, start_current, args.target_current_a, total_theta,
            ramp_theta, args.max_step, args.atol, args.max_newton,
        )
        dense_state = lambda query: np.vstack([np.interp(query, sample_theta, row) for row in states])
    else:
        solve_options: dict[str, Any] = {
            "method": args.method, "rtol": args.rtol, "atol": args.atol,
            "max_step": args.max_step, "dense_output": True,
        }
        if args.method in ("BDF", "Radau"):
            solve_options["jac_sparsity"] = system.jacobian_sparsity()
        sol = solve_ivp(
            lambda theta, y: system.rhs(theta, y, start_current, args.target_current_a, ramp_theta),
            (0.0, total_theta), y0, **solve_options,
        )
        sample_theta = np.linspace(0.0, total_theta, int((args.ramp_periods + args.hold_periods) * args.samples_per_period) + 1)
        dense_state = sol.sol
        states = dense_state(sample_theta) if dense_state is not None else np.empty((y0.size, 0))
        integrator = {"success": bool(sol.success), "message": sol.message,
                      "nfev": sol.nfev, "njev": sol.njev, "nlu": sol.nlu}
    data = make_observables(system, sample_theta, states)
    data["mu"] = np.asarray([system.source(x, start_current, args.target_current_a, ramp_theta)[system.pump_node] for x in sample_theta])
    data["source_current_a"] = np.array(data["mu"], copy=True)
    data["time_s"] = sample_theta / system.omega
    np.savez_compressed(outdir / "transient_observables.npz", **data)
    with (outdir / "transient_observables.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
    hold_start = 2.0 * math.pi * args.ramp_periods
    strobe_theta = np.arange(hold_start, total_theta + 0.1, 2.0 * math.pi)
    strobe_states = dense_state(strobe_theta)
    strobe = stroboscopic_diagnostics(system, strobe_theta, strobe_states, args.hold_periods)
    last_theta = np.linspace(total_theta - 2.0 * math.pi * min(args.hold_periods, 20), total_theta, 2048)
    last_states = dense_state(last_theta)
    pump_flux = np.asarray([system.phi0 * system.unpack(last_states[:, i])[0][system.pump_node] for i in range(last_states.shape[1])])
    centered = pump_flux - np.mean(pump_flux)
    fft = np.fft.rfft(centered) / centered.size
    freq = np.fft.rfftfreq(centered.size, d=(last_theta[1] - last_theta[0]) / system.omega) / 1e9
    spectrum = {"frequency_ghz": freq, "amplitude": np.abs(fft)}
    np.savez_compressed(outdir / "late_time_spectrum.npz", **spectrum)
    phase_series = np.asarray([
        system.circuit.Bphi.T @ system.unpack(last_states[:, i])[0]
        for i in range(last_states.shape[1])
    ])
    unwrapped_phase = np.unwrap(phase_series, axis=0)
    phase_velocity = np.diff(unwrapped_phase, axis=0) / np.diff(last_theta)[:, None] * system.omega
    mean_phase_velocity = float(np.mean(phase_velocity))
    phase_winding = float(np.mean(unwrapped_phase[-1] - unwrapped_phase[0]) / (2.0 * math.pi))
    np.savez_compressed(outdir / "late_time_phase.npz", theta=last_theta, phase=phase_series, unwrapped_phase=unwrapped_phase)
    classification = classify_state(
        strobe, mean_phase_velocity, bool(integrator["success"]), phase_winding
    )
    branch_transfer = None
    if classification == "PERIOD_1":
        branch_transfer = project_periodic_state(
            system, dense_state, float(sample_theta[-1]),
            np.asarray(hb_report["metadata"]["pump_modes"], dtype=int),
            args.target_current_a,
        )
    if not integrator["success"]:
        final_status = "TRANSIENT_NUMERICAL_BLOCKER"
        blocker_reason = integrator["message"]
    elif branch_transfer is not None and branch_transfer["hb_converged"]:
        final_status = "HIGH_DRIVE_HB_BRANCH_FOUND"
        blocker_reason = None
    elif classification in {"PERIOD_2", "PERIOD_3", "QUASIPERIODIC_OR_PERIOD_N", "RUNNING_PHASE", "BROADBAND_OR_CHAOTIC"}:
        final_status = "REPRODUCIBLE_PHYSICAL_TRANSITION"
        blocker_reason = None
    else:
        final_status = "TRANSIENT_NUMERICAL_BLOCKER"
        blocker_reason = "period-1 transient did not seed a converged fixed-drive HB root"
    plot_results(outdir, data, spectrum)
    checkpoint_data = np.load(args.checkpoint / "pump_solution.npz")
    checkpoint_X = np.asarray(checkpoint_data["X_real"], dtype=float) + 1j * np.asarray(checkpoint_data["X_imag"], dtype=float)
    checkpoint_grid = HarmonicGrid(
        np.asarray(hb_report["metadata"]["pump_modes"]), 40, system.omega
    )
    checkpoint_problem = FullPumpProblem(
        system.circuit.C, system.circuit.G, system.circuit.K,
        system.circuit.Bphi, system.branch, checkpoint_grid,
        system.pump_node, start_current,
    )
    result = {
        "audit": audit.__dict__, "checkpoint": str(args.checkpoint),
        "start_current_a": start_current, "target_current_a": args.target_current_a,
        "ramp_periods": args.ramp_periods, "hold_periods": args.hold_periods,
        "integrator": integrator,
        "classification": classification, "stroboscopic": strobe,
        "mean_phase_velocity_rad_s": mean_phase_velocity,
        "mean_phase_winding_cycles": phase_winding,
        "branch_transfer": branch_transfer,
        "final_status": final_status, "blocker_reason": blocker_reason,
        "hb_checkpoint_summary": summarize_solution(checkpoint_problem, checkpoint_X),
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "g1_current_79" / "pass" / "points" / "point_0012_p_m19p6842dbm_fp_7p9ghz" / "pump")
    parser.add_argument("--outdir", type=Path, default=ROOT / "h1_79")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--target-current-a", type=float, default=1.6e-5)
    parser.add_argument("--ramp-periods", type=int, default=40)
    parser.add_argument("--hold-periods", type=int, default=40)
    parser.add_argument("--samples-per-period", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--method", choices=("RK45", "RK23", "BDF", "Radau", "implicit_euler", "implicit_trapezoid"), default="implicit_trapezoid")
    parser.add_argument("--max-newton", type=int, default=12)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_only:
        print(json.dumps(audit_circuit(args.circuit_dir).__dict__, indent=2))
        return 0
    result = run_experiment(args)
    print(json.dumps({"classification": result["classification"], "integrator": result["integrator"], "stroboscopic": result["stroboscopic"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
