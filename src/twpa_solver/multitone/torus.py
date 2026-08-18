"""Autonomous two-frequency harmonic-balance problem and continuation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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
from twpa_solver.multitone.source import AffineSourcePath
from twpa_solver.pump.backends.fast_coupled import FastCoupledPreconditioner
from twpa_solver.pump.problem import pack_complex, unpack_complex
from twpa_solver.pump.solver import bordered_solve_refined


@dataclass
class TorusProblem:
    """Wrap a multitone problem with the unknown generator frequency.

    The autonomous phase has one neutral direction.  The diagnostic Newton
    system adds the phase anchor ``Im X[(0, 1), node_ref] = 0`` to the full
    residual.  Production torus continuation uses the matrix-free augmented
    PALC system in :meth:`solve_torus_arclength`, which also carries the source
    scale and an explicit branch tangent.
    """

    base_problem: FullMultiToneProblem | SchurMultiToneProblem
    pump_modes: tuple[int, ...]
    q_max: int
    omega_a: float
    sideband_harmonics: int | None = None
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
        if self.sideband_harmonics is not None and self.sideband_harmonics < 1:
            raise ValueError("sideband_harmonics must be >= 1")
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
            sideband_harmonics=self.sideband_harmonics,
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
            sideband_harmonics=self.sideband_harmonics,
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

    def omitted_q_residual(
        self, X: np.ndarray, evaluation_q_max: int
    ) -> dict[str, float]:
        """Evaluate residual content outside the retained generator order.

        The state is zero-padded into a larger autonomous lattice and the
        nonlinear residual is evaluated once.  No larger Newton factorization
        is formed, so this is a cheap truncation diagnostic rather than a
        higher-order torus solve.
        """
        if evaluation_q_max <= self.q_max:
            raise ValueError("evaluation_q_max must exceed q_max")
        evaluation_basis = build_autonomous_torus_basis(
            self.base_problem.basis.omega_p,
            self.omega_a,
            self.pump_modes,
            evaluation_q_max,
            sideband_harmonics=self.sideband_harmonics,
        )
        evaluation_state = np.zeros(
            (evaluation_basis.n_tones, self.base_problem.n), dtype=np.complex128
        )
        for row, tone in enumerate(self.basis.tones):
            if tone in evaluation_basis.tones:
                evaluation_state[evaluation_basis.index_of(tone)] = X[row]
        source_path = self.base_problem.source_path
        evaluation_start = np.zeros(
            (evaluation_basis.n_tones, self.base_problem.n), dtype=np.complex128
        )
        evaluation_delta = np.zeros_like(evaluation_start)
        for row, tone in enumerate(self.base_problem.basis.tones):
            target_row = evaluation_basis.index_of(tone)
            evaluation_start[target_row] = source_path.source_start[row]
            evaluation_delta[target_row] = source_path.source_delta[row]
        evaluation_source = AffineSourcePath(
            evaluation_start, evaluation_delta
        )
        if self.is_schur:
            base = self.base_problem
            full = replace(
                base.full,
                basis=evaluation_basis,
                source_path=evaluation_source,
                cache={},
            )
            port_indices = list(full.circuit.port_to_index.values())
            evaluation_problem = build_multitone_schur_problem(
                full,
                port_indices,
                linear_apply_mode=base.linear_apply_mode,
                preconditioner=base.preconditioner,
            )
        else:
            evaluation_problem = replace(
                self.base_problem,
                basis=evaluation_basis,
                source_path=evaluation_source,
                cache={},
            )
        residual = evaluation_problem.residual_coeffs(
            evaluation_state, self.source_tau
        )
        omitted_rows = [
            row
            for row, tone in enumerate(evaluation_basis.tones)
            if abs(tone.q) > self.q_max
        ]
        omitted = pack_complex(residual[omitted_rows])
        source = pack_complex(
            evaluation_problem.source_coeffs(self.source_tau)
        )
        absolute = float(
            np.linalg.norm(omitted) / max(np.sqrt(omitted.size), 1.0)
        )
        scale = max(
            float(np.linalg.norm(source) / max(np.sqrt(source.size), 1.0)),
            1e-30,
        )
        return {
            "omitted_q_residual_abs": absolute,
            "omitted_q_residual_rel": absolute / scale,
            "omitted_q_max": float(evaluation_q_max),
        }

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

    def _normalized_merit(
        self,
        problem: FullMultiToneProblem | SchurMultiToneProblem,
        X: np.ndarray,
        coefficients: np.ndarray,
        anchor_value: float,
        amplitude_value: float,
        amplitude: float,
        source_tau: float,
    ) -> tuple[float, dict[str, float]]:
        """Return a dimensionless merit and its component diagnostics."""
        packed = pack_complex(coefficients)
        source = pack_complex(problem.source_coeffs(source_tau))
        coefficient_scale = max(
            float(np.linalg.norm(source) / max(np.sqrt(source.size), 1.0)),
            1.0e-30,
        )
        state_scale = max(
            float(np.linalg.norm(pack_complex(X)) / max(np.sqrt(packed.size), 1.0)),
            1.0e-30,
        )
        coefficient_relative = float(
            np.linalg.norm(packed) / max(np.sqrt(packed.size), 1.0)
            / coefficient_scale
        )
        anchor_relative = abs(float(anchor_value)) / state_scale
        amplitude_relative = abs(float(amplitude_value)) / max(amplitude, 1.0e-30)
        components = {
            "coefficient_relative": coefficient_relative,
            "anchor_relative": anchor_relative,
            "amplitude_relative": amplitude_relative,
        }
        return max(components.values()), components

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
        """Diagnose the amplitude-parameterized autonomous system.

        Unknowns are ``(X, omega_a, source_tau)`` and equations are the
        coefficient residual, the generator phase anchor, and the amplitude
        normalization.  The system stays square, and the period-1 branch is
        not a solution of it.  This method is retained as a diagnostic for the
        degenerate closure; production continuation uses NS branch switching
        and pseudo-arclength instead.

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

        def scalar_coefficients(
            state: np.ndarray, omega_a: float, source_tau: float
        ) -> np.ndarray:
            problem = replace(self.with_omega_a(omega_a), source_tau=source_tau)
            return problem.full_problem().residual_coeffs(
                state, source_tau
            )

        for iteration in range(max_newton + 1):
            current = replace(self.with_omega_a(omega), source_tau=tau)
            problem = current.full_problem()
            coefficients = problem.residual_coeffs(X, tau)
            anchor_value = current.anchor(X)
            amplitude_value = current.amplitude_residual(X, amplitude)
            norm, components = current._normalized_merit(
                problem,
                X,
                coefficients,
                anchor_value,
                amplitude_value,
                amplitude,
                tau,
            )
            history.append(norm)
            if norm <= residual_tol:
                return X, omega, tau, {
                    "converged": True,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "generator_norm": current.generator_norm(X),
                    "amplitude": float(amplitude),
                    **components,
                }
            if iteration == max_newton:
                break

            matvec, preconditioner = current._linearization(problem, X)
            packed = pack_complex(coefficients)

            omega_step = current.omega_fd_relative_step * max(abs(omega), 1.0)
            b_omega = (
                pack_complex(scalar_coefficients(X, omega + omega_step, tau))
                - pack_complex(scalar_coefficients(X, omega - omega_step, tau))
            ) / (2.0 * omega_step)
            tau_step = tau_fd_relative_step * max(abs(tau), 1.0)
            b_tau = (
                pack_complex(scalar_coefficients(X, omega, tau + tau_step))
                - pack_complex(scalar_coefficients(X, omega, tau - tau_step))
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
                trial_problem = trial.full_problem()
                trial_coefficients = trial_problem.residual_coeffs(
                    trial_x, trial_tau
                )
                trial_norm, _ = trial._normalized_merit(
                    trial_problem,
                    trial_x,
                    trial_coefficients,
                    trial.anchor(trial_x),
                    trial.amplitude_residual(trial_x, amplitude),
                    amplitude,
                    trial_tau,
                )
                if trial_norm < norm:
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
                    **components,
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
        :meth:`solve_torus_branch_switch` to start a torus branch and
        :meth:`solve_torus_arclength` to continue it.
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

    def _phase_constraint(
        self,
        X: np.ndarray,
        state_scale: float,
        phase_reference: np.ndarray | None,
    ) -> tuple[float, np.ndarray]:
        """Return a normalized torus phase condition and its state row."""
        packed_size = 2 * X.size
        row = np.zeros(packed_size, dtype=float)
        if phase_reference is None:
            generator = ToneIndex(0, 1)
            index = self.basis.index_of(generator) * self.base_problem.n
            position = X.size + index + self.node_ref
            row[position] = 1.0 / state_scale
            return self.anchor(X) / state_scale, row

        reference = np.asarray(phase_reference, dtype=np.complex128)
        if reference.shape != X.shape:
            raise ValueError("phase_reference shape does not match torus state")
        rows = [
            index
            for index, tone in enumerate(self.basis.tones)
            if tone.q == 1
        ]
        if not rows:
            raise ValueError("phase_reference has no q=+1 sector")
        flat_reference = reference.reshape(-1)
        selected = np.concatenate(
            [flat_reference[index * X.shape[1] : (index + 1) * X.shape[1]]
             for index in rows]
        )
        reference_norm = float(np.linalg.norm(selected))
        if reference_norm <= 0.0:
            raise ValueError("phase_reference has zero q=+1 norm")
        for index in rows:
            start = index * X.shape[1]
            stop = start + X.shape[1]
            values = flat_reference[start:stop]
            row[start:stop] = -values.imag / (reference_norm * state_scale)
            row[X.size + start:X.size + stop] = (
                values.real / (reference_norm * state_scale)
            )
        return float(row @ pack_complex(X)), row

    def solve_torus_arclength(
        self,
        X0: np.ndarray,
        *,
        previous_X: np.ndarray,
        previous_omega_a: float,
        previous_source_tau: float,
        tangent: np.ndarray,
        step_size: float,
        omega_a0: float | None = None,
        source_tau0: float | None = None,
        phase_reference: np.ndarray | None = None,
        max_newton: int = 20,
        residual_tol: float = 1.0e-9,
        min_alpha: float = 1.0 / 1024.0,
        gmres_rtol: float = 1.0e-8,
        gmres_maxiter: int = 80,
        gmres_restart: int = 60,
    ) -> tuple[np.ndarray, float, float, dict[str, Any], np.ndarray | None]:
        """Correct one torus point with a matrix-free PALC system.

        The augmented unknown is ``(X, omega_a, source_tau)``.  The equations
        are the multitone residual, one torus phase condition, and an explicit
        pseudo-arclength condition.  The state Jacobian is used only as the
        state block of a preconditioner; it is not required to be invertible.
        This is the branch-switch corrector at an NS point and the ordinary
        pseudo-arclength corrector after the first nonzero torus point.

        ``tangent`` is expressed in normalized coordinates: packed state
        divided by the previous-state norm, frequency divided by
        ``omega_p``, and source scale divided by the previous source scale.
        """
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        X = np.asarray(X0, dtype=np.complex128).copy()
        previous = np.asarray(previous_X, dtype=np.complex128)
        if X.shape != previous.shape:
            raise ValueError("X0 and previous_X shapes do not match")
        expected_shape = (self.basis.n_tones, self.base_problem.n)
        if X.shape != expected_shape:
            raise ValueError("X0 shape does not match the torus basis")
        tangent_value = np.asarray(tangent, dtype=float).reshape(-1)
        state_size = 2 * X.size
        if tangent_value.size != state_size + 2:
            raise ValueError("tangent size does not match the augmented state")

        state_scale = max(float(np.linalg.norm(pack_complex(previous))), 1e-300)
        omega_scale = max(abs(previous_omega_a), self.basis.omega_p, 1.0)
        tau_scale = max(abs(previous_source_tau), 1.0)
        tangent_value = tangent_value.copy()
        tangent_norm = float(np.linalg.norm(tangent_value))
        if tangent_norm <= 0.0:
            raise ValueError("tangent must be nonzero")
        tangent_value /= tangent_norm

        omega = self.omega_a if omega_a0 is None else float(omega_a0)
        tau = self.source_tau if source_tau0 is None else float(source_tau0)
        if omega <= 0.0 or tau <= 0.0:
            raise ValueError("omega_a0 and source_tau0 must be positive")
        previous_packed = pack_complex(previous)
        previous_augmented = np.concatenate((
            previous_packed / state_scale,
            np.asarray([
                previous_omega_a / omega_scale,
                previous_source_tau / tau_scale,
            ]),
        ))
        history: list[float] = []
        gmres_history: list[float] = []
        factor_backends: list[str] = []

        def evaluate(
            state: np.ndarray, frequency: float, source_tau: float
        ) -> tuple[Any, np.ndarray, float, np.ndarray, float, float]:
            current = replace(
                self.with_omega_a(frequency), source_tau=source_tau
            )
            problem = current.full_problem()
            coefficients = problem.residual_coeffs(state, source_tau)
            source = pack_complex(problem.source_coeffs(source_tau))
            residual_scale = max(
                float(np.linalg.norm(source) / max(np.sqrt(source.size), 1.0)),
                1e-30,
            )
            phase_value, phase_row = current._phase_constraint(
                state, state_scale, phase_reference
            )
            scaled = np.concatenate((
                pack_complex(state) / state_scale,
                np.asarray([frequency / omega_scale, source_tau / tau_scale]),
            ))
            arclength = float(
                tangent_value @ (scaled - previous_augmented) - step_size
            )
            return (
                current,
                coefficients,
                residual_scale,
                phase_row,
                phase_value,
                arclength,
            )

        for iteration in range(max_newton + 1):
            (
                current,
                coefficients,
                residual_scale,
                phase_row,
                phase_value,
                arclength,
            ) = evaluate(X, omega, tau)
            residual = np.concatenate((
                pack_complex(coefficients) / residual_scale,
                np.asarray([phase_value, arclength]),
            ))
            norm = float(np.linalg.norm(residual) / np.sqrt(residual.size))
            history.append(norm)
            if norm <= residual_tol:
                scaled = np.concatenate((
                    pack_complex(X) / state_scale,
                    np.asarray([omega / omega_scale, tau / tau_scale]),
                ))
                secant = scaled - previous_augmented
                secant_norm = float(np.linalg.norm(secant))
                new_tangent = (
                    secant / secant_norm if secant_norm > 0.0 else None
                )
                return X, omega, tau, {
                    "converged": True,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "gmres_residual_history": gmres_history,
                    "factor_backend": factor_backends,
                    "source_tau": tau,
                    "arclength_step": step_size,
                    "phase_constraint": phase_value,
                }, new_tangent
            if iteration == max_newton:
                break

            matvec, preconditioner = current._linearization(
                current.full_problem(), X
            )
            factor_backends.append(preconditioner.last_factor_backend)
            omega_step = current.omega_fd_relative_step * max(abs(omega), 1.0)
            plus_problem = replace(
                current.with_omega_a(omega + omega_step), source_tau=tau
            ).full_problem()
            minus_problem = replace(
                current.with_omega_a(omega - omega_step), source_tau=tau
            ).full_problem()
            b_omega = (
                pack_complex(plus_problem.residual_coeffs(X, tau))
                - pack_complex(minus_problem.residual_coeffs(X, tau))
            ) / (2.0 * omega_step)
            b_tau = -pack_complex(
                current.full_problem().source_delta_coeffs()
            )

            def augmented_matvec(vector: np.ndarray) -> np.ndarray:
                delta_state = vector[:state_size]
                delta_omega = vector[state_size]
                delta_tau = vector[state_size + 1]
                state_part = (
                    matvec(delta_state)
                    + b_omega * omega_scale * delta_omega
                    + b_tau * tau_scale * delta_tau
                ) / residual_scale
                phase_part = float(phase_row @ delta_state)
                tangent_state = float(
                    tangent_value[:state_size] @ (delta_state / state_scale)
                )
                arclength_part = (
                    tangent_state
                    + tangent_value[state_size] * delta_omega
                    + tangent_value[state_size + 1] * delta_tau
                )
                return np.concatenate((
                    state_part,
                    np.asarray([phase_part, arclength_part]),
                ))

            def augmented_preconditioner(vector: np.ndarray) -> np.ndarray:
                result = np.zeros_like(vector)
                result[:state_size] = residual_scale * preconditioner.solve(
                    vector[:state_size]
                )
                result[state_size:] = vector[state_size:]
                return result

            operator = spla.LinearOperator(
                shape=(state_size + 2, state_size + 2),
                matvec=augmented_matvec,
                dtype=float,
            )
            preconditioner_operator = spla.LinearOperator(
                shape=(state_size + 2, state_size + 2),
                matvec=augmented_preconditioner,
                dtype=float,
            )
            gmres_history.clear()
            try:
                update, gmres_info = spla.gmres(
                    operator,
                    -residual,
                    M=preconditioner_operator,
                    rtol=gmres_rtol,
                    atol=0.0,
                    restart=gmres_restart,
                    maxiter=gmres_maxiter,
                    callback=gmres_history.append,
                    callback_type="pr_norm",
                )
            except TypeError:
                update, gmres_info = spla.gmres(
                    operator,
                    -residual,
                    M=preconditioner_operator,
                    tol=gmres_rtol,
                    restart=gmres_restart,
                    maxiter=gmres_maxiter,
                    callback=gmres_history.append,
                )
            if gmres_info != 0:
                return X, omega, tau, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "gmres_residual_history": gmres_history,
                    "gmres_info": int(gmres_info),
                    "factor_backend": factor_backends,
                    "failure_reason": "augmented GMRES failed",
                }, None

            state_step = update[:state_size]
            step_norm = float(np.linalg.norm(state_step))
            max_step = self.max_step_over_state * state_scale
            if step_norm > max_step:
                update = update * (max_step / step_norm)
            delta_state = unpack_complex(update[:state_size], X.shape)
            delta_omega = float(update[state_size] * omega_scale)
            delta_tau = float(update[state_size + 1] * tau_scale)
            alpha = 1.0
            accepted = False
            while alpha >= min_alpha:
                trial_X = X + alpha * delta_state
                trial_omega = omega + alpha * delta_omega
                trial_tau = tau + alpha * delta_tau
                if trial_omega <= 0.0 or trial_tau <= 0.0:
                    alpha *= 0.5
                    continue
                trial_data = evaluate(trial_X, trial_omega, trial_tau)
                trial_residual = np.concatenate((
                    pack_complex(trial_data[1]) / trial_data[2],
                    np.asarray([trial_data[4], trial_data[5]]),
                ))
                trial_norm = float(
                    np.linalg.norm(trial_residual) / np.sqrt(trial_residual.size)
                )
                if trial_norm < norm:
                    X, omega, tau = trial_X, trial_omega, trial_tau
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return X, omega, tau, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "gmres_residual_history": gmres_history,
                    "factor_backend": factor_backends,
                    "failure_reason": "line search failed",
                }, None

        return X, omega, tau, {
            "converged": False,
            "iterations": max_newton,
            "residual_norm": float(history[-1]),
            "residual_history": history,
            "gmres_residual_history": gmres_history,
            "factor_backend": factor_backends,
            "failure_reason": "maximum Newton iterations reached",
        }, None

    def predict_torus_arclength(
        self,
        previous_X: np.ndarray,
        previous_omega_a: float,
        previous_source_tau: float,
        tangent: np.ndarray,
        step_size: float,
    ) -> tuple[np.ndarray, float, float]:
        """Return a predictor in the normalized PALC direction."""
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        state = np.asarray(previous_X, dtype=np.complex128)
        state_scale = max(float(np.linalg.norm(pack_complex(state))), 1e-300)
        omega_scale = max(abs(previous_omega_a), self.basis.omega_p, 1.0)
        tau_scale = max(abs(previous_source_tau), 1.0)
        tangent_value = np.asarray(tangent, dtype=float).reshape(-1)
        expected = 2 * state.size + 2
        if tangent_value.size != expected:
            raise ValueError("tangent size does not match the augmented state")
        tangent_norm = float(np.linalg.norm(tangent_value))
        if tangent_norm <= 0.0:
            raise ValueError("tangent must be nonzero")
        tangent_value = tangent_value / tangent_norm
        previous_augmented = np.concatenate((
            pack_complex(state) / state_scale,
            np.asarray([
                previous_omega_a / omega_scale,
                previous_source_tau / tau_scale,
            ]),
        ))
        predictor = previous_augmented + step_size * tangent_value
        return (
            unpack_complex(
                predictor[:2 * state.size] * state_scale, state.shape
            ),
            float(predictor[-2] * omega_scale),
            float(predictor[-1] * tau_scale),
        )

    def solve_torus_branch_switch(
        self,
        X_ns: np.ndarray,
        *,
        omega_a_ns: float,
        source_tau_ns: float,
        perturbation: np.ndarray,
        step_size: float = 1.0e-2,
        **kwargs: Any,
    ) -> tuple[np.ndarray, float, float, dict[str, Any], np.ndarray | None]:
        """Switch from a period-1 NS point into the nonzero torus branch."""
        state = np.asarray(X_ns, dtype=np.complex128)
        mode = np.asarray(perturbation, dtype=np.complex128)
        if state.shape != mode.shape:
            raise ValueError("perturbation shape does not match X_ns")
        if self.generator_norm(mode) <= 0.0:
            raise ValueError("perturbation has no q != 0 content")
        state_scale = max(
            float(np.linalg.norm(pack_complex(state))),
            float(np.linalg.norm(pack_complex(mode))),
            1e-300,
        )
        direction = pack_complex(mode) / state_scale
        tangent = np.concatenate((direction, np.asarray([0.0, 0.0])))
        tangent /= max(float(np.linalg.norm(tangent)), 1e-300)
        predictor_X, predictor_omega, predictor_tau = (
            self.predict_torus_arclength(
                state,
                omega_a_ns,
                source_tau_ns,
                tangent,
                step_size,
            )
        )
        return self.solve_torus_arclength(
            predictor_X,
            previous_X=state,
            previous_omega_a=omega_a_ns,
            previous_source_tau=source_tau_ns,
            tangent=tangent,
            step_size=step_size,
            omega_a0=predictor_omega,
            source_tau0=predictor_tau,
            phase_reference=mode,
            **kwargs,
        )
