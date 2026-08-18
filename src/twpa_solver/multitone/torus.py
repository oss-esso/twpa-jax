"""Autonomous two-frequency harmonic-balance problem and bordered solve."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from twpa_solver.multitone.basis import (
    MultiToneBasis,
    ToneIndex,
    build_autonomous_torus_basis,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.schur import (
    SchurMultiToneProblem,
    build_multitone_schur_problem,
)
from twpa_solver.pump.backends.fast_coupled import FastCoupledPreconditioner
from twpa_solver.pump.problem import pack_complex, unpack_complex
from twpa_solver.pump.solver import bordered_solve_refined


@dataclass
class TorusProblem:
    """Wrap a multitone problem with the unknown generator frequency.

    The autonomous phase has one neutral direction.  The bordered system adds
    the phase anchor ``Im X[(0, 1), node_ref] = 0`` to the full residual, so
    the extra frequency unknown and the extra scalar equation keep the system
    square.  Newton linear systems are solved by block elimination against
    the existing real-coupled fast preconditioner; the bordered matrix is not
    assembled.
    """

    base_problem: FullMultiToneProblem | SchurMultiToneProblem
    pump_modes: tuple[int, ...]
    q_max: int
    omega_a: float
    node_ref: int = 0
    source_tau: float = 1.0
    omega_fd_relative_step: float = 1.0e-6
    factor_backend: str = "pardiso"
    precond_reuse: int = 1
    max_step_over_state: float = 2.0
    _problem_caches: dict[float, dict[object, object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.omega_a <= 0.0:
            raise ValueError("omega_a must be positive")
        if self.q_max < 1:
            raise ValueError("q_max must be >= 1")
        if not self.pump_modes:
            raise ValueError("pump_modes must not be empty")
        if not 0 <= self.node_ref < self.base_problem.n:
            raise ValueError("node_ref is outside the circuit node range")
        if self.omega_fd_relative_step <= 0.0:
            raise ValueError("omega_fd_relative_step must be positive")
        if self.factor_backend not in {"pardiso", "banded", "superlu"}:
            raise ValueError(
                "factor_backend must be one of 'pardiso', 'banded', or 'superlu'"
            )
        if self.precond_reuse < 1:
            raise ValueError("precond_reuse must be >= 1")
        if self.base_problem.basis.require_signal_sector:
            raise ValueError(
                "base_problem must use an autonomous torus basis, not a signal sector"
            )
        if self.is_schur:
            cache = self._problem_caches.setdefault(self.omega_a, {})
            cache["torus_schur_problem"] = self.base_problem

    @property
    def is_schur(self) -> bool:
        """Return whether the torus problem uses retained-node coordinates."""
        return isinstance(self.base_problem, SchurMultiToneProblem)

    @property
    def anchor_full_node(self) -> int:
        """Return the corresponding full-node index for the phase anchor."""
        if not self.is_schur:
            return self.node_ref
        retained = self.base_problem.partition.retained
        return int(retained[self.node_ref])

    @property
    def basis(self) -> MultiToneBasis:
        """Return the basis at the current generator frequency."""
        return build_autonomous_torus_basis(
            self.base_problem.basis.omega_p,
            self.omega_a,
            self.pump_modes,
            self.q_max,
            n_p=self.base_problem.basis.n_p,
            n_delta=self.base_problem.basis.n_delta,
        )

    @property
    def unknown_size(self) -> int:
        return 2 * self.basis.n_tones * self.base_problem.n + 1

    def with_omega_a(self, omega_a: float) -> "TorusProblem":
        """Return the same physical problem at another generator frequency."""
        result = replace(self, omega_a=float(omega_a))
        result._problem_caches = self._problem_caches
        return result

    def full_problem(
        self, omega_a: float | None = None
    ) -> FullMultiToneProblem | SchurMultiToneProblem:
        """Build the coordinate problem at ``omega_a`` with a local cache.

        Linear blocks and the fast-preconditioner pattern are valid only for a
        fixed frequency lattice.  Caches are therefore separated by generator
        frequency rather than shared across Newton iterates with different
        bases.
        """
        current = self.omega_a if omega_a is None else float(omega_a)
        basis = build_autonomous_torus_basis(
            self.base_problem.basis.omega_p,
            current,
            self.pump_modes,
            self.q_max,
            n_p=self.base_problem.basis.n_p,
            n_delta=self.base_problem.basis.n_delta,
        )
        cache = self._problem_caches.setdefault(current, {})
        if not self.is_schur:
            return replace(self.base_problem, basis=basis, cache=cache)

        base = self.base_problem
        full = replace(base.full, basis=basis, cache=cache)
        problem_cache = cache.get("torus_schur_problem")
        if problem_cache is None:
            port_indices = list(base.full.circuit.port_to_index.values())
            problem_cache = build_multitone_schur_problem(
                full,
                port_indices,
                linear_apply_mode=base.linear_apply_mode,
                preconditioner=base.preconditioner,
            )
            cache["torus_schur_problem"] = problem_cache
        return problem_cache

    def anchor(self, X: np.ndarray) -> float:
        """Return the real phase anchor for the generator coefficient."""
        generator = ToneIndex(0, 1)
        basis = self.full_problem().basis
        if generator not in basis.tones:
            raise ValueError("autonomous basis does not contain the generator tone")
        return float(np.imag(X[basis.index_of(generator), self.node_ref]))

    def residual_coeffs(
        self, X: np.ndarray, omega_a: float | None = None
    ) -> np.ndarray:
        """Return the full complex residual at the requested generator tone."""
        return self.full_problem(omega_a).residual_coeffs(X, self.source_tau)

    def residual_vector(
        self, X: np.ndarray, omega_a: float | None = None
    ) -> np.ndarray:
        """Return the full residual plus the scalar phase anchor."""
        packed = pack_complex(self.residual_coeffs(X, omega_a))
        return np.concatenate((packed, np.asarray([self.anchor(X)])))

    def jacobian(
        self, X: np.ndarray, omega_a: float | None = None
    ) -> sp.csc_matrix:
        """Return the unbordered real Jacobian for diagnostics.

        Newton updates use :func:`bordered_solve_refined` and never assemble a
        bordered sparse matrix.  This method remains as a diagnostic API for
        callers that need the unbordered exact real-coupled Jacobian.
        """
        current = self.omega_a if omega_a is None else float(omega_a)
        problem = self.full_problem(current)
        tangent = problem.tangent_state(X)
        spectral = problem.spectral_tangent_state(tangent)
        if not hasattr(problem, "real_coupled_matrix"):
            raise TypeError(
                "exact sparse Jacobian diagnostics are unavailable for Schur "
                "coordinates; use the fast preconditioner path"
            )
        return problem.real_coupled_matrix(spectral).tocsc()

    def _factor_options(self) -> tuple[bool, bool]:
        """Return ``use_pardiso`` and ``use_banded`` for the fast backend."""
        if self.factor_backend == "banded":
            return False, True
        if self.factor_backend == "superlu":
            return False, False
        return True, False

    def _preconditioner(
        self,
        problem: Any,
        tangent: Any,
    ) -> FastCoupledPreconditioner:
        """Return a cached fast preconditioner and refactor its numeric values."""
        cache = getattr(problem, "cache", None)
        if cache is None:
            cache = getattr(problem.part, "_torus_cache", None)
            if cache is None:
                cache = {}
                problem.part._torus_cache = cache
        key = ("torus_fast_coupled", self.factor_backend)
        count_key = ("torus_fast_coupled_refactors", self.factor_backend)
        preconditioner = cache.get(key)
        if preconditioner is None:
            use_pardiso, use_banded = self._factor_options()
            preconditioner = FastCoupledPreconditioner(
                problem,
                use_pardiso=use_pardiso,
                use_banded=use_banded,
            )
            cache[key] = preconditioner
            cache[count_key] = 0
        refactor_count = int(cache[count_key])
        if refactor_count % self.precond_reuse == 0:
            preconditioner.refactor(tangent)
        cache[count_key] = refactor_count + 1
        return preconditioner

    def _linearization(
        self,
        problem: Any,
        X: np.ndarray,
    ) -> tuple[Callable[[np.ndarray], np.ndarray], FastCoupledPreconditioner]:
        """Build the exact JVP and the production preconditioner for one step."""
        tangent = problem.tangent_state(X)
        spectral = problem.spectral_tangent_state(tangent)

        def matvec(vector: np.ndarray) -> np.ndarray:
            variation = unpack_complex(vector, X.shape)
            applied = problem.jvp_coeffs_with_spectral_tangent(variation, spectral)
            return pack_complex(applied)

        return matvec, self._preconditioner(problem, tangent)

    def generator_rows(self) -> list[int]:
        """Return basis rows carrying autonomous (``q != 0``) content."""
        return [
            row for row, tone in enumerate(self.basis.tones) if tone.q != 0
        ]

    def generator_norm(self, X: np.ndarray) -> float:
        """Return the Euclidean norm of the ``q != 0`` sector."""
        rows = self.generator_rows()
        if not rows:
            raise ValueError("basis has no autonomous q != 0 sector")
        return float(np.linalg.norm(np.asarray(X)[rows]))

    def _generator_mask(self, n_tones: int, n_nodes: int) -> np.ndarray:
        """Return a real-packed mask selecting the ``q != 0`` sector."""
        mask = np.zeros(2 * n_tones * n_nodes, dtype=bool)
        for row in self.generator_rows():
            start = row * n_nodes
            mask[start : start + n_nodes] = True
            offset = n_tones * n_nodes + start
            mask[offset : offset + n_nodes] = True
        return mask

    def amplitude_residual(self, X: np.ndarray, amplitude: float) -> float:
        """Return the amplitude-normalization residual.

        The autonomous phase anchor alone does **not** exclude the period-1
        solution: with the ``q != 0`` sector exactly zero the anchor is
        satisfied identically, the residual carries no ``q != 0`` content, and
        ``dR/domega_a`` vanishes, so the period-1 state is an exact root of the
        phase-anchored bordered system for *any* generator frequency.  Measured
        on ``ipm_2c_fixed``: Newton converged in one iteration to residual
        ``6.7e-20`` and returned the seeded ``omega_a`` unchanged to sixteen
        significant figures, at every seed amplitude tried.

        Prescribing ``||X_{q != 0}|| = amplitude`` removes that branch by
        construction, because a zero sector cannot satisfy it.  The pump scale
        becomes an unknown in exchange, so the solved branch is reported as the
        drive at which a torus of the requested amplitude exists.
        """
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        return self.generator_norm(X) - float(amplitude)

    def solve_newton_amplitude(
        self,
        X0: np.ndarray,
        amplitude: float,
        *,
        omega_a0: float | None = None,
        source_tau0: float | None = None,
        max_newton: int = 30,
        residual_tol: float = 1.0e-9,
        min_alpha: float = 1.0 / 1024.0,
        tau_fd_relative_step: float = 1.0e-6,
    ) -> tuple[np.ndarray, float, float, dict[str, Any]]:
        """Solve the amplitude-parameterized autonomous system.

        Unknowns are ``(X, omega_a, source_tau)`` and equations are the
        coefficient residual, the generator phase anchor, and the amplitude
        normalization.  The system stays square, and the period-1 branch is not
        a solution of it.

        Each Newton step takes three linear solves against the *unbordered*
        Jacobian through one shared numeric factorization, then closes a dense
        2x2 system on the two scalar unknowns.
        """
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        X = np.asarray(X0, dtype=np.complex128).copy()
        omega = self.omega_a if omega_a0 is None else float(omega_a0)
        tau = self.source_tau if source_tau0 is None else float(source_tau0)
        if X.shape != (self.basis.n_tones, self.base_problem.n):
            raise ValueError("X0 shape does not match the autonomous basis")
        if self.generator_norm(X) <= 0.0:
            raise ValueError(
                "X0 has an empty q != 0 sector; the amplitude-parameterized "
                "solve cannot start on the period-1 branch"
            )

        history: list[float] = []

        def scalars(state: np.ndarray, omega_a: float, source_tau: float):
            problem = replace(self.with_omega_a(omega_a), source_tau=source_tau)
            coefficients = problem.full_problem().residual_coeffs(
                state, source_tau
            )
            return problem, coefficients

        for iteration in range(max_newton + 1):
            current = replace(self.with_omega_a(omega), source_tau=tau)
            problem = current.full_problem()
            coefficients = problem.residual_coeffs(X, tau)
            anchor_value = current.anchor(X)
            amplitude_value = current.amplitude_residual(X, amplitude)
            residual = np.concatenate((
                pack_complex(coefficients),
                np.asarray([anchor_value, amplitude_value]),
            ))
            norm = float(np.linalg.norm(residual))
            history.append(norm)
            if norm <= residual_tol:
                return X, omega, tau, {
                    "converged": True,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "generator_norm": current.generator_norm(X),
                    "amplitude": float(amplitude),
                }
            if iteration == max_newton:
                break

            matvec, preconditioner = current._linearization(problem, X)
            packed = pack_complex(coefficients)

            omega_step = current.omega_fd_relative_step * max(abs(omega), 1.0)
            b_omega = (
                pack_complex(scalars(X, omega + omega_step, tau)[1])
                - pack_complex(scalars(X, omega - omega_step, tau)[1])
            ) / (2.0 * omega_step)
            tau_step = tau_fd_relative_step * max(abs(tau), 1.0)
            b_tau = (
                pack_complex(scalars(X, omega, tau + tau_step)[1])
                - pack_complex(scalars(X, omega, tau - tau_step)[1])
            ) / (2.0 * tau_step)

            generator_index = problem.basis.index_of(ToneIndex(0, 1)) * problem.n
            anchor_position = problem.H * problem.n + generator_index + self.node_ref
            mask = current._generator_mask(problem.H, problem.n)
            packed_state = pack_complex(X)
            sector_norm = float(np.linalg.norm(packed_state[mask]))
            if sector_norm <= 0.0:
                return X, omega, tau, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "failure_reason": "iterate collapsed onto the period-1 branch",
                }
            amplitude_gradient = np.zeros_like(packed_state)
            amplitude_gradient[mask] = packed_state[mask] / sector_norm

            y0 = preconditioner.solve(-packed)
            z_omega = preconditioner.solve(b_omega)
            z_tau = preconditioner.solve(b_tau)

            # Nondimensionalize before closing the 2x2.  The unknowns span
            # twenty-two orders of magnitude on a real device -- node flux is
            # ~1e-13 Wb, source_tau is ~1, omega_a is ~5e9 rad/s -- so an
            # unscaled closure produces a delta_tau of ~1e15 and a step 1e16
            # times the state, which no line search can accept.  This is the
            # same failure the pseudo-arclength metric had before it was given
            # a state-scale metric; see pump/solver.py::solve_arclength.
            omega_scale = float(self.basis.omega_p)
            tau_scale = max(abs(tau), 1.0)
            state_scale = max(float(np.linalg.norm(packed_state)), 1.0e-300)
            matrix = np.array([
                [
                    z_omega[anchor_position] * omega_scale / state_scale,
                    z_tau[anchor_position] * tau_scale / state_scale,
                ],
                [
                    float(amplitude_gradient @ z_omega) * omega_scale / state_scale,
                    float(amplitude_gradient @ z_tau) * tau_scale / state_scale,
                ],
            ])
            rhs = np.array([
                (y0[anchor_position] + anchor_value) / state_scale,
                (float(amplitude_gradient @ y0) + amplitude_value) / state_scale,
            ])
            condition = np.linalg.cond(matrix)
            if not np.isfinite(condition) or condition > 1.0e14:
                return X, omega, tau, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "bordered_condition": float(condition),
                    "failure_reason": "bordered 2x2 system degenerated",
                }
            scaled_omega, scaled_tau = np.linalg.solve(matrix, rhs)
            delta_omega = float(scaled_omega) * omega_scale
            delta_tau = float(scaled_tau) * tau_scale
            step = y0 - delta_omega * z_omega - delta_tau * z_tau

            # Trust region on the state, so one bad closure cannot throw the
            # iterate 1e16 times its own size and force the line search to
            # spend every halving on an unusable direction.
            step_norm = float(np.linalg.norm(step))
            limit = self.max_step_over_state * state_scale
            if step_norm > limit:
                shrink = limit / step_norm
                step = step * shrink
                delta_omega *= shrink
                delta_tau *= shrink
            delta_x = unpack_complex(step, X.shape)

            alpha = 1.0
            accepted = False
            while alpha >= min_alpha:
                trial_x = X + alpha * delta_x
                trial_omega = omega + alpha * float(delta_omega)
                trial_tau = tau + alpha * float(delta_tau)
                if trial_omega <= 0.0 or trial_tau <= 0.0:
                    alpha *= 0.5
                    continue
                trial = replace(
                    self.with_omega_a(trial_omega), source_tau=trial_tau
                )
                trial_residual = np.concatenate((
                    pack_complex(
                        trial.full_problem().residual_coeffs(trial_x, trial_tau)
                    ),
                    np.asarray([
                        trial.anchor(trial_x),
                        trial.amplitude_residual(trial_x, amplitude),
                    ]),
                ))
                if np.linalg.norm(trial_residual) < norm:
                    X, omega, tau = trial_x, trial_omega, trial_tau
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return X, omega, tau, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "failure_reason": "line search failed",
                }

        return X, omega, tau, {
            "converged": False,
            "iterations": max_newton,
            "residual_norm": float(history[-1]),
            "residual_history": history,
            "failure_reason": "maximum Newton iterations reached",
        }

    def solve_newton(
        self,
        X0: np.ndarray,
        *,
        omega_a0: float | None = None,
        max_newton: int = 20,
        residual_tol: float = 1.0e-9,
        min_alpha: float = 1.0 / 1024.0,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Solve the phase-anchored autonomous system with line-searched Newton.

        WARNING: this formulation admits the period-1 branch as an exact
        solution for any ``omega_a`` -- see :meth:`amplitude_residual`.  It is
        retained for diagnostics and for below-onset control runs, where
        returning period-1 is the correct answer.  Use
        :meth:`solve_newton_amplitude` to find a torus.
        """
        X = np.asarray(X0, dtype=np.complex128).copy()
        omega = self.omega_a if omega_a0 is None else float(omega_a0)
        if X.shape != (self.basis.n_tones, self.base_problem.n):
            raise ValueError("X0 shape does not match the autonomous basis")
        residual_history: list[float] = []
        factor_backends: list[str] = []

        for iteration in range(max_newton + 1):
            current = self.with_omega_a(omega)
            problem = current.full_problem()
            coefficient_residual = problem.residual_coeffs(X, current.source_tau)
            residual = np.concatenate(
                (pack_complex(coefficient_residual), np.asarray([current.anchor(X)]))
            )
            norm = float(np.linalg.norm(residual))
            residual_history.append(norm)
            if norm <= residual_tol:
                return X, omega, {
                    "converged": True,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": residual_history,
                    "factor_backend": factor_backends,
                    "precond_reuse": self.precond_reuse,
                }
            if iteration == max_newton:
                break

            matvec, preconditioner = current._linearization(problem, X)
            step_size = current.omega_fd_relative_step * max(abs(omega), 1.0)
            plus = pack_complex(current.residual_coeffs(X, omega + step_size))
            minus = pack_complex(current.residual_coeffs(X, omega - step_size))
            d_omega = (plus - minus) / (2.0 * step_size)
            generator = ToneIndex(0, 1)
            generator_index = problem.basis.index_of(generator) * problem.n
            imag_offset = problem.H * problem.n

            def anchor_dot(vector: np.ndarray) -> float:
                return float(vector[imag_offset + generator_index + self.node_ref])

            update = bordered_solve_refined(
                matvec,
                preconditioner.solve,
                pack_complex(coefficient_residual),
                current.anchor(X),
                -d_omega,
                anchor_dot,
                0.0,
            )
            factor_backends.append(preconditioner.last_factor_backend)
            if update is None:
                return X, omega, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": residual_history,
                    "factor_backend": factor_backends,
                    "precond_reuse": self.precond_reuse,
                    "failure_reason": "bordered scalar denominator degenerated",
                }
            step, delta_omega = update
            delta_x = unpack_complex(step, X.shape)
            alpha = 1.0
            accepted = False
            while alpha >= min_alpha:
                trial_x = X + alpha * delta_x
                trial_omega = omega + alpha * delta_omega
                if trial_omega <= 0.0:
                    alpha *= 0.5
                    continue
                trial_residual = current.residual_vector(trial_x, trial_omega)
                if np.linalg.norm(trial_residual) < norm:
                    X, omega = trial_x, trial_omega
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return X, omega, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": residual_history,
                    "factor_backend": factor_backends,
                    "precond_reuse": self.precond_reuse,
                    "failure_reason": "line search failed",
                }
        return X, omega, {
            "converged": False,
            "iterations": max_newton,
            "residual_norm": float(residual_history[-1]),
            "residual_history": residual_history,
            "factor_backend": factor_backends,
            "precond_reuse": self.precond_reuse,
            "failure_reason": "maximum Newton iterations reached",
        }
