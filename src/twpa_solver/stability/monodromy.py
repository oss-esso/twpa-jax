"""Matrix-free tangent integration for the validated implicit trapezoid step.

The state used here is the same reduced index-one state as the production
transient driver: ``(q_d, p_d, q_a)``, where ``q`` is node flux divided by
``phi0``, ``p=dq/dtheta``, ``d`` indexes differential nodes, and ``a`` indexes
algebraic nodes.  Algebraic velocity perturbations are reconstructed from the
linearized algebraic constraint; arbitrary inconsistent DAE perturbations are
therefore never passed to Arnoldi.

One tangent step eliminates the kinematic equation analytically and factors
only the resulting sparse node-flux Schur matrix.  The factors are retained
for the phase grid and reused by every monodromy matvec.  The monodromy itself
is exposed only as a ``scipy.sparse.linalg.LinearOperator``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.pump.basis import load_pump_basis_from_solution
from twpa_solver.pump.problem import HarmonicGrid


def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} has size {result.size}; expected {size}")
    return result


@dataclass(frozen=True)
class HBPeriodicOrbit:
    """Periodic HB waveform sampled on the tangent integration phase grid."""

    theta: np.ndarray
    q: np.ndarray
    p: np.ndarray
    omega: float
    modes: tuple[int, ...]
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float).reshape(-1)
        q = np.asarray(self.q, dtype=float)
        p = np.asarray(self.p, dtype=float)
        if theta.size < 2 or q.shape != p.shape or q.shape[0] != theta.size:
            raise ValueError("orbit arrays must have shape (phase, node)")
        if not np.isclose(theta[0], 0.0) or not np.isclose(
            theta[-1], 2.0 * math.pi
        ):
            raise ValueError("orbit phase grid must cover exactly one period")
        if np.any(np.diff(theta) <= 0.0):
            raise ValueError("orbit phase grid must be strictly increasing")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "p", p)
        object.__setattr__(
            self, "coefficients", np.asarray(self.coefficients, dtype=np.complex128)
        )

    @property
    def node_count(self) -> int:
        return int(self.q.shape[1])

    def initial_state(self, system: Any) -> np.ndarray:
        """Pack the phase-zero orbit state in the system's transient layout."""
        return np.asarray(system.pack(self.q[0], self.p[0]), dtype=float)


def build_hb_periodic_orbit(
    coefficients: np.ndarray,
    modes: list[int] | np.ndarray,
    omega: float,
    phi0: float,
    *,
    steps_per_period: int,
) -> HBPeriodicOrbit:
    """Reconstruct an HB orbit and its phase derivative on a closed grid.

    ``coefficients`` follow the production positive-phasor convention
    ``x=2 Re sum(X_k exp(+i k theta))``.  The returned ``p`` is ``dq/dtheta``.
    """
    if omega <= 0.0 or phi0 <= 0.0:
        raise ValueError("omega and phi0 must be positive")
    if steps_per_period < 2:
        raise ValueError("steps_per_period must be at least two")
    X = np.asarray(coefficients, dtype=np.complex128)
    mode_array = np.asarray(modes, dtype=int).reshape(-1)
    if X.ndim != 2 or X.shape[0] != mode_array.size:
        raise ValueError("coefficients shape does not match modes")
    grid = HarmonicGrid(
        modes=mode_array,
        nt=int(steps_per_period),
        omega=omega,
    )
    x = grid.synthesize(X)
    dx_dtheta = grid.synthesize_derivative(X, 1) / omega
    theta = np.linspace(0.0, 2.0 * math.pi, steps_per_period + 1)
    q = np.vstack((x, x[0])) / phi0
    p = np.vstack((dx_dtheta, dx_dtheta[0])) / phi0
    return HBPeriodicOrbit(
        theta=theta,
        q=q,
        p=p,
        omega=omega,
        modes=tuple(int(mode) for mode in mode_array),
        coefficients=X,
    )


def load_hb_periodic_orbit(
    checkpoint: str | Path,
    omega: float,
    phi0: float,
    *,
    steps_per_period: int,
) -> tuple[HBPeriodicOrbit, dict[str, Any]]:
    """Load a validated persisted HB checkpoint and reconstruct its orbit."""
    directory = Path(checkpoint)
    report = json.loads((directory / "pump_report.json").read_text())
    if report.get("final_status") != "VALID_CONVERGED":
        raise ValueError(f"HB checkpoint is not converged: {directory}")
    X, basis = load_pump_basis_from_solution(directory, fallback_omega_p=omega)
    if not np.isclose(basis.omega_p, omega, rtol=1e-10, atol=0.0):
        raise ValueError("HB checkpoint pump frequency does not match the transient system")
    return (
        build_hb_periodic_orbit(
            X,
            basis.modes,
            omega,
            phi0,
            steps_per_period=steps_per_period,
        ),
        report,
    )


@dataclass
class _ConsistentStateSpace:
    system: Any

    def __post_init__(self) -> None:
        if self.system.full_state:
            raise NotImplementedError(
                "Floquet tangent reduction requires a factorable algebraic "
                "velocity block; full-state index-one fallback is not implemented"
            )
        self.differential = np.asarray(self.system.differential, dtype=int)
        self.algebraic = np.asarray(self.system.algebraic, dtype=int)
        self.n = int(self.system.n)
        self.nd = int(self.differential.size)
        self.na = int(self.algebraic.size)

    @property
    def size(self) -> int:
        return 2 * self.nd + self.na

    def unpack(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = _as_vector(vector, self.size, "tangent vector")
        dq = np.zeros(self.n, dtype=float)
        dp = np.zeros(self.n, dtype=float)
        dq[self.differential] = value[: self.nd]
        dp[self.differential] = value[self.nd : 2 * self.nd]
        dq[self.algebraic] = value[2 * self.nd :]
        return dq, dp

    def pack(self, dq: np.ndarray, dp: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(dq, dtype=float)[self.differential],
                np.asarray(dp, dtype=float)[self.differential],
                np.asarray(dq, dtype=float)[self.algebraic],
            )
        )

    def _constraint_jacobian(self, q: np.ndarray) -> sp.csr_matrix:
        flux = self.system.phi0 * (self.system.circuit.Bphi.T @ q)
        tangent = np.asarray(self.system.branch.tangent(flux[None, :]))[0]
        return (
            self.system.circuit.K[self.algebraic]
            + self.system._b_a @ sp.diags(tangent) @ self.system.circuit.Bphi.T
        ).tocsr()

    def project_q(self, dq: np.ndarray, q: np.ndarray) -> np.ndarray:
        if not self.na:
            return np.asarray(dq, dtype=float)
        jac = self._constraint_jacobian(q)
        result = np.asarray(dq, dtype=float).copy()
        rhs = -jac[:, self.differential] @ result[self.differential]
        result[self.algebraic] = spla.spsolve(
            jac[:, self.algebraic].tocsc(), rhs
        )
        return result

    def reconstruct_p_algebraic(
        self, dq: np.ndarray, q: np.ndarray, dp_d: np.ndarray
    ) -> np.ndarray:
        result = np.zeros(self.n, dtype=float)
        result[self.differential] = dp_d
        if not self.na:
            return result
        jac = self._constraint_jacobian(q)
        rhs = -self.system._g_ad @ (
            self.system.omega * self.system.phi0 * dp_d
        ) - self.system.phi0 * (jac @ dq)
        result[self.algebraic] = self.system.g_alg_factor.solve(rhs) / (
            self.system.omega * self.system.phi0
        )
        return result

    def make_consistent(self, vector: np.ndarray, q: np.ndarray) -> np.ndarray:
        dq, dp_d = self.unpack(vector)
        dq = self.project_q(dq, q)
        dp = self.reconstruct_p_algebraic(dq, q, dp_d[self.differential])
        return self.pack(dq, dp)


@dataclass
class _TangentStep:
    factor: Any
    rhs_q: sp.csc_matrix
    rhs_p: sp.csc_matrix
    h: float

    def apply(self, dq0: np.ndarray, dp0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dq1 = self.factor.solve(self.rhs_q @ dq0 + self.rhs_p @ dp0)
        dp1 = (2.0 / self.h) * (dq1 - dq0) - dp0
        return np.asarray(dq1, dtype=float), np.asarray(dp1, dtype=float)


@dataclass
class MonodromyOperator:
    """Matrix-free one-period tangent operator."""

    system: Any
    orbit: HBPeriodicOrbit
    steps: tuple[_TangentStep, ...]
    state_space: _ConsistentStateSpace
    matvec_count: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.state_space.size, self.state_space.size)

    @property
    def period_theta(self) -> float:
        return 2.0 * math.pi

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        self.matvec_count += 1
        current = self.state_space.make_consistent(vector, self.orbit.q[0])
        for index, step in enumerate(self.steps):
            dq, dp = self.state_space.unpack(current)
            dq, dp = step.apply(dq, dp)
            dq = self.state_space.project_q(dq, self.orbit.q[index + 1])
            current = self.state_space.pack(dq, dp)
            current = self.state_space.make_consistent(
                current, self.orbit.q[index + 1]
            )
        return current

    def as_linear_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(
            self.shape,
            matvec=self.matvec,
            dtype=np.float64,
        )


def _build_step(system: Any, q0: np.ndarray, q1: np.ndarray, h: float) -> _TangentStep:
    circuit = system.circuit
    flux0 = system.phi0 * (circuit.Bphi.T @ q0)
    flux1 = system.phi0 * (circuit.Bphi.T @ q1)
    tangent0 = np.asarray(system.branch.tangent(flux0[None, :]))[0]
    tangent1 = np.asarray(system.branch.tangent(flux1[None, :]))[0]
    nonlinear0 = circuit.Bphi @ sp.diags(tangent0) @ circuit.Bphi.T
    nonlinear1 = circuit.Bphi @ sp.diags(tangent1) @ circuit.Bphi.T
    dynamic_q0 = (circuit.K + nonlinear0) / 2.0
    dynamic_q1 = (circuit.K + nonlinear1) / 2.0
    dynamic_p1 = circuit.C * (system.omega**2 / h) + circuit.G * (system.omega / 2.0)
    dynamic_p0 = -circuit.C * (system.omega**2 / h) + circuit.G * (system.omega / 2.0)
    schur = (dynamic_q1 + (2.0 / h) * dynamic_p1).tocsc()
    factor = spla.splu(schur)
    rhs_q = (-dynamic_q0 + (2.0 / h) * dynamic_p1).tocsc()
    rhs_p = (-dynamic_p0 + dynamic_p1).tocsc()
    return _TangentStep(factor=factor, rhs_q=rhs_q, rhs_p=rhs_p, h=h)


def build_monodromy_operator(
    system: Any,
    orbit: HBPeriodicOrbit,
    *,
    max_step_theta: float = 0.05,
) -> MonodromyOperator:
    """Build the one-period tangent map around a reconstructed HB orbit."""
    if max_step_theta <= 0.0:
        raise ValueError("max_step_theta must be positive")
    n_steps = int(orbit.theta.size - 1)
    if np.max(np.diff(orbit.theta)) > max_step_theta * (1.0 + 1.0e-12):
        raise ValueError("orbit grid contains a step larger than max_step_theta")
    state_space = _ConsistentStateSpace(system)
    steps = tuple(
        _build_step(system, orbit.q[index], orbit.q[index + 1], orbit.theta[index + 1] - orbit.theta[index])
        for index in range(n_steps)
    )
    return MonodromyOperator(
        system=system,
        orbit=orbit,
        steps=steps,
        state_space=state_space,
    )
