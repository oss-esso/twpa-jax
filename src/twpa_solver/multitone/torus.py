"""Autonomous two-frequency harmonic-balance problem and continuation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
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


def apply_border_aware_preconditioner(
    state_solve: Callable[[np.ndarray], np.ndarray],
    state_rhs: np.ndarray,
    border_rhs: np.ndarray,
    border_columns: np.ndarray,
    constraint_rows: np.ndarray,
    border_matrix: np.ndarray,
) -> np.ndarray:
    """Apply an exact two-row bordered preconditioner.

    ``state_solve`` may be an approximate solve.  The two scalar unknowns are
    eliminated through the Schur complement of that approximate state block.
    Keeping this operation separate makes the bordered algebra testable with a
    dense state solve before it is used by matrix-free GMRES.
    """
    rhs = np.asarray(state_rhs, dtype=float)
    border = np.asarray(border_rhs, dtype=float).reshape(2)
    columns = np.asarray(border_columns, dtype=float)
    rows = np.asarray(constraint_rows, dtype=float)
    small = np.asarray(border_matrix, dtype=float).reshape(2, 2)
    y = np.asarray(state_solve(rhs), dtype=float)
    solved_columns = np.column_stack(
        [np.asarray(state_solve(columns[:, index]), dtype=float)
         for index in range(2)]
    )
    schur = small - rows @ solved_columns
    scalars = np.linalg.solve(schur, border - rows @ y)
    state = y - solved_columns @ scalars
    return np.concatenate((state, scalars))


def apply_one_border_preconditioner(
    state_solve: Callable[[np.ndarray], np.ndarray],
    state_rhs: np.ndarray,
    border_rhs: float,
    border_column: np.ndarray,
    constraint_row: np.ndarray,
    border_scalar: float,
) -> np.ndarray:
    """Apply a state preconditioner with one scalar border unknown."""
    state_result = np.asarray(state_solve(state_rhs), dtype=float)
    solved_column = np.asarray(state_solve(border_column), dtype=float)
    schur = float(border_scalar - constraint_row @ solved_column)
    if abs(schur) <= 1.0e-15:
        raise np.linalg.LinAlgError("one-border Schur complement is singular")
    scalar = float((border_rhs - constraint_row @ state_result) / schur)
    return np.concatenate((state_result - solved_column * scalar, [scalar]))


def _append_timing_event(
    path: Path | None,
    stage: str,
    status: str,
    iteration: int | None = None,
    **fields: Any,
) -> None:
    """Append one flushed stage event to an optional JSONL telemetry file."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    event: dict[str, Any] = {
        "timestamp": time.time(),
        "stage": stage,
        "status": status,
    }
    if iteration is not None:
        event["iteration"] = iteration
    event.update(fields)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()


def _lattice_key_payload(key: object) -> list[int] | str:
    """Serialize a mode key without reducing it to a physical frequency."""
    if isinstance(key, ToneIndex):
        return [int(key.h), int(key.q)]
    return str(key)


def lattice_label_audit(
    problem: Any,
    preconditioner: FastCoupledPreconditioner,
) -> dict[str, Any]:
    """Audit exact lattice labels against scalar-frequency rounding.

    The torus JVP and its fast preconditioner must use ``(h, q)`` keys.  The
    physical frequencies are intentionally included as a contrast because
    rounding ``(h * omega_p + q * omega_a) / omega_p`` collapses nearby
    generator sectors.
    """
    mode_keys = list(problem.mode_keys)
    expected_difference = {
        left - right for left in mode_keys for right in mode_keys
    }
    expected_sum = {left + right for left in mode_keys for right in mode_keys}
    preconditioner_keys = list(preconditioner._ells)
    preconditioner_difference = {
        preconditioner_keys[int(index)]
        for index in np.asarray(preconditioner._ell_diff).reshape(-1)
    }
    preconditioner_sum = {
        preconditioner_keys[int(index)]
        for index in np.asarray(preconditioner._ell_sum).reshape(-1)
    }
    normalized_frequencies = [
        float(tone.omega(problem.basis.omega_p, problem.basis.delta)
              / problem.basis.omega_p)
        for tone in problem.basis.tones
    ]
    rounded = [int(round(value)) for value in normalized_frequencies]
    rounded_collisions = len(rounded) - len(set(rounded))
    return {
        "jvp_mode_keys": [_lattice_key_payload(key) for key in mode_keys],
        "preconditioner_mode_keys": [
            _lattice_key_payload(key) for key in preconditioner.modes
        ],
        "jvp_difference_keys": [
            _lattice_key_payload(key) for key in sorted(expected_difference)
        ],
        "jvp_sum_keys": [
            _lattice_key_payload(key) for key in sorted(expected_sum)
        ],
        "preconditioner_difference_keys": [
            _lattice_key_payload(key) for key in sorted(preconditioner_difference)
        ],
        "preconditioner_sum_keys": [
            _lattice_key_payload(key) for key in sorted(preconditioner_sum)
        ],
        "preconditioner_uses_exact_lattice_keys": all(
            isinstance(key, ToneIndex) for key in preconditioner_keys
        ),
        "jvp_difference_keys_match": preconditioner_difference
        == expected_difference,
        "jvp_sum_keys_match": preconditioner_sum == expected_sum,
        "scalar_frequency_rounding": rounded,
        "scalar_frequency_rounding_collision_count": rounded_collisions,
        "scalar_frequency_rounding_would_collapse": rounded_collisions > 0,
    }


def _short_gmres(
    operator: spla.LinearOperator,
    rhs: np.ndarray,
    preconditioner: spla.LinearOperator,
    *,
    rtol: float,
    maxiter: int,
    restart: int,
) -> dict[str, Any]:
    """Run a bounded GMRES probe and retain its convergence history."""
    _, report = _gmres_solve(
        operator,
        rhs,
        preconditioner,
        rtol=rtol,
        maxiter=maxiter,
        restart=restart,
    )
    return report


def _gmres_solve(
    operator: spla.LinearOperator,
    rhs: np.ndarray,
    preconditioner: spla.LinearOperator,
    *,
    rtol: float,
    maxiter: int,
    restart: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run GMRES and return both the update and convergence telemetry."""
    history: list[float] = []
    try:
        update, info = spla.gmres(
            operator,
            rhs,
            M=preconditioner,
            rtol=rtol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            callback=history.append,
            callback_type="pr_norm",
        )
    except TypeError:
        update, info = spla.gmres(
            operator,
            rhs,
            M=preconditioner,
            tol=rtol,
            restart=restart,
            maxiter=maxiter,
            callback=history.append,
        )
    return update, {
        "info": int(info),
        "iterations": len(history),
        "residual_history": history,
    }


@dataclass
class BranchLockGeometry:
    """Normalized affine row used to retain a finite torus branch."""

    predictor_packed: np.ndarray
    state_scale: float
    radial_unit: np.ndarray
    phase_unit: np.ndarray
    constraint_row: np.ndarray
    beta: float

    @property
    def row_physical(self) -> np.ndarray:
        """Return the row acting on unscaled packed state coordinates."""
        return self.constraint_row / self.state_scale

    @property
    def radial_projection(self) -> float:
        """Return the lock-row projection on the radial direction."""
        return float(self.constraint_row @ self.radial_unit)

    @property
    def phase_projection(self) -> float:
        """Return the lock-row projection on the phase direction."""
        return float(self.constraint_row @ self.phase_unit)

    def value(self, state: np.ndarray) -> float:
        """Evaluate the affine lock condition at a complex state."""
        packed = pack_complex(np.asarray(state, dtype=np.complex128))
        return float(
            self.constraint_row @ (packed - self.predictor_packed)
            / self.state_scale
        )


def build_branch_lock_geometry(
    predictor: np.ndarray,
    q_values: np.ndarray,
    *,
    beta: float = 1.0,
) -> BranchLockGeometry:
    """Build an oblique normalized row from q-sector radial and phase modes."""
    state = np.asarray(predictor, dtype=np.complex128)
    q_array = np.asarray(q_values, dtype=int).reshape(-1)
    if state.ndim != 2 or q_array.size != state.shape[0]:
        raise ValueError("q_values must contain one entry per torus tone")
    if beta == 0.0:
        raise ValueError("beta must be nonzero")
    radial = state * (q_array[:, None] != 0)
    phase = 1j * q_array[:, None] * state
    radial_packed = pack_complex(radial)
    phase_packed = pack_complex(phase)
    radial_norm = float(np.linalg.norm(radial_packed))
    phase_norm = float(np.linalg.norm(phase_packed))
    if radial_norm <= 0.0:
        raise ValueError("predictor has no q != 0 content")
    if phase_norm <= 0.0:
        raise ValueError("predictor has no autonomous phase direction")
    radial_unit = radial_packed / radial_norm
    phase_unit = phase_packed / phase_norm
    constraint = phase_unit + float(beta) * radial_unit
    constraint /= max(float(np.linalg.norm(constraint)), 1.0e-300)
    state_scale = max(float(np.linalg.norm(pack_complex(state))), 1.0e-300)
    return BranchLockGeometry(
        predictor_packed=pack_complex(state),
        state_scale=state_scale,
        radial_unit=radial_unit,
        phase_unit=phase_unit,
        constraint_row=constraint,
        beta=float(beta),
    )


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

    def linear_fidelity_report(
        self,
        X: np.ndarray,
        *,
        omega_a: float,
        source_tau: float,
        previous_X: np.ndarray,
        previous_omega_a: float,
        previous_source_tau: float,
        tangent: np.ndarray,
        phase_reference: np.ndarray | None = None,
        gmres_rtol: float = 1.0e-8,
        gmres_maxiter: int = 2,
        gmres_restart: int = 2,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        """Measure state and bordered preconditioner fidelity at one point.

        This is a bounded diagnostic.  It does not update the torus state and
        does not alter the corrector's GMRES settings.  The state-only and
        augmented probes use the same JVP, normalization, border columns, and
        preconditioner as :meth:`solve_torus_arclength`.
        """
        state = np.asarray(X, dtype=np.complex128)
        previous = np.asarray(previous_X, dtype=np.complex128)
        if state.shape != previous.shape:
            raise ValueError("X and previous_X shapes do not match")
        expected_shape = (self.basis.n_tones, self.base_problem.n)
        if state.shape != expected_shape:
            raise ValueError("X shape does not match the torus basis")
        if omega_a <= 0.0 or source_tau <= 0.0:
            raise ValueError("omega_a and source_tau must be positive")
        tangent_value = np.asarray(tangent, dtype=float).reshape(-1)
        state_size = 2 * state.size
        if tangent_value.size != state_size + 2:
            raise ValueError("tangent size does not match the augmented state")

        state_scale = max(float(np.linalg.norm(pack_complex(previous))), 1e-300)
        omega_scale = max(abs(previous_omega_a), self.basis.omega_p, 1.0)
        tau_scale = max(abs(previous_source_tau), 1.0)
        current = replace(self.with_omega_a(omega_a), source_tau=source_tau)
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
        matvec, preconditioner = current._linearization(problem, state)

        omega_step = current.omega_fd_relative_step * max(abs(omega_a), 1.0)
        plus_problem = replace(
            current.with_omega_a(omega_a + omega_step), source_tau=source_tau
        ).full_problem()
        minus_problem = replace(
            current.with_omega_a(omega_a - omega_step), source_tau=source_tau
        ).full_problem()
        b_omega = (
            pack_complex(plus_problem.residual_coeffs(state, source_tau))
            - pack_complex(minus_problem.residual_coeffs(state, source_tau))
        ) / (2.0 * omega_step)
        b_tau_raw = -pack_complex(problem.source_delta_coeffs())
        tau_fd_step = current.omega_fd_relative_step * max(abs(source_tau), 1.0)
        plus_tau_problem = replace(
            current, source_tau=source_tau + tau_fd_step
        ).full_problem()
        minus_tau_problem = replace(
            current, source_tau=source_tau - tau_fd_step
        ).full_problem()
        plus_source = pack_complex(
            plus_tau_problem.source_coeffs(source_tau + tau_fd_step)
        )
        minus_source = pack_complex(
            minus_tau_problem.source_coeffs(source_tau - tau_fd_step)
        )
        residual_scale_tau = (
            float(np.linalg.norm(plus_source))
            - float(np.linalg.norm(minus_source))
        ) / (2.0 * tau_fd_step * max(np.sqrt(plus_source.size), 1.0))
        coefficient_vector = pack_complex(coefficients)
        b_tau = (
            b_tau_raw * residual_scale - coefficient_vector * residual_scale_tau
        ) / residual_scale**2

        tangent_value = tangent_value / max(float(np.linalg.norm(tangent_value)), 1e-300)

        def augmented_matvec(vector: np.ndarray) -> np.ndarray:
            delta_state = vector[:state_size]
            delta_omega = vector[state_size]
            delta_tau = vector[state_size + 1]
            physical_delta_state = delta_state * state_scale
            state_part = (
                matvec(physical_delta_state)
                + b_omega * omega_scale * delta_omega
            ) / residual_scale + b_tau * tau_scale * delta_tau
            phase_part = float(phase_row @ physical_delta_state)
            arclength_part = (
                float(tangent_value[:state_size] @ delta_state)
                + tangent_value[state_size] * delta_omega
                + tangent_value[state_size + 1] * delta_tau
            )
            return np.concatenate((
                state_part,
                np.asarray([phase_part, arclength_part]),
            ))

        border_columns = np.column_stack((
            b_omega * omega_scale / residual_scale,
            b_tau * tau_scale,
        ))
        constraint_rows = np.vstack((
            phase_row * state_scale,
            tangent_value[:state_size],
        ))
        border_matrix = np.array(
            [
                [0.0, 0.0],
                [tangent_value[state_size], tangent_value[state_size + 1]],
            ],
            dtype=float,
        )

        def state_preconditioner_solve(vector: np.ndarray) -> np.ndarray:
            return (
                residual_scale * np.asarray(preconditioner.solve(vector), dtype=float)
                / state_scale
            )

        preconditioned_columns = np.column_stack(
            [state_preconditioner_solve(border_columns[:, index]) for index in range(2)]
        )
        border_schur = border_matrix - constraint_rows @ preconditioned_columns
        augmented_preconditioner = lambda vector: apply_border_aware_preconditioner(
            state_preconditioner_solve,
            vector[:state_size],
            vector[state_size:],
            border_columns,
            constraint_rows,
            border_matrix,
        )
        phase_frequency_column = border_columns[:, 0]
        phase_frequency_row = constraint_rows[0]
        phase_frequency_column_solved = state_preconditioner_solve(
            phase_frequency_column
        )
        phase_frequency_schur = float(
            -phase_frequency_row @ phase_frequency_column_solved
        )

        def phase_frequency_matvec(vector: np.ndarray) -> np.ndarray:
            augmented = np.concatenate((vector, np.asarray([0.0])))
            applied = augmented_matvec(augmented)
            return np.concatenate((applied[:state_size], applied[-2:-1]))

        def phase_frequency_preconditioner(vector: np.ndarray) -> np.ndarray:
            return apply_one_border_preconditioner(
                state_preconditioner_solve,
                vector[:state_size],
                float(vector[state_size]),
                phase_frequency_column,
                phase_frequency_row,
                0.0,
            )

        def direct_phase_frequency_solve(vector: np.ndarray) -> np.ndarray:
            """Solve the phase-frequency border with two state solves."""
            state_rhs = np.asarray(vector[:state_size], dtype=float)
            phase_rhs = float(vector[state_size])
            y = state_preconditioner_solve(state_rhs)
            z = state_preconditioner_solve(phase_frequency_column)
            eta = (phase_rhs - phase_frequency_row @ y) / phase_frequency_schur
            return np.concatenate((y - z * eta, np.asarray([eta])))

        def diagonal_augmented_preconditioner(vector: np.ndarray) -> np.ndarray:
            return np.concatenate((
                state_preconditioner_solve(vector[:state_size]),
                vector[state_size:],
            ))

        state_operator = spla.LinearOperator(
            shape=(state_size, state_size),
            matvec=matvec,
            dtype=float,
        )
        state_preconditioner_operator = spla.LinearOperator(
            shape=(state_size, state_size),
            matvec=preconditioner.solve,
            dtype=float,
        )
        augmented_operator = spla.LinearOperator(
            shape=(state_size + 2, state_size + 2),
            matvec=augmented_matvec,
            dtype=float,
        )
        augmented_preconditioner_operator = spla.LinearOperator(
            shape=(state_size + 2, state_size + 2),
            matvec=augmented_preconditioner,
            dtype=float,
        )
        phase_frequency_operator = spla.LinearOperator(
            shape=(state_size + 1, state_size + 1),
            matvec=phase_frequency_matvec,
            dtype=float,
        )
        phase_frequency_preconditioner_operator = spla.LinearOperator(
            shape=(state_size + 1, state_size + 1),
            matvec=phase_frequency_preconditioner,
            dtype=float,
        )
        diagonal_augmented_preconditioner_operator = spla.LinearOperator(
            shape=(state_size + 2, state_size + 2),
            matvec=diagonal_augmented_preconditioner,
            dtype=float,
        )

        rng = np.random.default_rng(random_seed)
        random_state = rng.normal(size=state_size)
        random_state /= max(float(np.linalg.norm(random_state)), 1e-300)
        state_vectors = {"random": random_state}
        q_plus = np.zeros_like(state)
        for index, tone in enumerate(self.basis.tones):
            if tone.q == 1:
                q_plus[index] = state[index]
        q_plus_vector = pack_complex(q_plus)
        if np.linalg.norm(q_plus_vector) > 0.0:
            state_vectors["q_plus_1"] = q_plus_vector / np.linalg.norm(q_plus_vector)

        phase_mode = np.zeros_like(state)
        for index, tone in enumerate(self.basis.tones):
            if tone.q != 0:
                phase_mode[index] = 1j * tone.q * state[index]
        phase_mode_packed = pack_complex(phase_mode)

        q_residual = np.zeros_like(coefficients)
        for index, tone in enumerate(self.basis.tones):
            if tone.q != 0:
                q_residual[index] = 1j * tone.q * coefficients[index]
        phase_equivariance_relative = float(
            np.linalg.norm(matvec(phase_mode_packed) - pack_complex(q_residual))
            / max(
                np.linalg.norm(matvec(phase_mode_packed))
                + np.linalg.norm(pack_complex(q_residual)),
                1e-30,
            )
        )

        def phase_frequency_residual(
            trial_state: np.ndarray,
            trial_omega_a: float,
        ) -> np.ndarray:
            trial_problem = replace(
                current.with_omega_a(trial_omega_a),
                source_tau=source_tau,
            ).full_problem()
            trial_coefficients = trial_problem.residual_coeffs(
                trial_state, source_tau
            )
            trial_phase, _ = current._phase_constraint(
                trial_state, state_scale, phase_reference
            )
            return np.concatenate((
                pack_complex(trial_coefficients) / residual_scale,
                np.asarray([trial_phase]),
            ))

        phase_frequency_base = phase_frequency_residual(state, omega_a)
        fd_step = max(float(current.omega_fd_relative_step), 1e-7)
        fd_directions: dict[str, np.ndarray] = {"random": rng.normal(
            size=state_size + 1
        )}
        fd_directions["omega"] = np.zeros(state_size + 1)
        fd_directions["omega"][state_size] = 1.0
        if np.linalg.norm(phase_mode_packed) > 0.0:
            fd_directions["phase_mode"] = np.concatenate((
                phase_mode_packed / state_scale,
                np.asarray([0.0]),
            ))
        for name, direction in fd_directions.items():
            fd_directions[name] = direction / max(
                float(np.linalg.norm(direction)), 1e-300
            )

        phase_frequency_fd_errors: dict[str, float] = {}
        for name, direction in fd_directions.items():
            trial_state = state + fd_step * state_scale * unpack_complex(
                direction[:state_size], state.shape
            )
            trial_omega_a = omega_a + fd_step * omega_scale * direction[state_size]
            finite_difference = (
                phase_frequency_residual(trial_state, trial_omega_a)
                - phase_frequency_base
            ) / fd_step
            analytic = phase_frequency_matvec(direction)
            phase_frequency_fd_errors[name] = float(
                np.linalg.norm(finite_difference - analytic)
                / max(np.linalg.norm(analytic), 1e-30)
            )

        state_fidelity: dict[str, float] = {}
        for name, rhs in state_vectors.items():
            solved = np.asarray(preconditioner.solve(rhs), dtype=float)
            state_fidelity[name] = float(
                np.linalg.norm(matvec(solved) - rhs) / max(np.linalg.norm(rhs), 1e-30)
            )

        state_rhs = -pack_complex(coefficients) / residual_scale
        augmented_rhs = np.concatenate((state_rhs, np.asarray([-phase_value, 0.0])))
        state_gmres = _short_gmres(
            state_operator,
            state_rhs,
            state_preconditioner_operator,
            rtol=gmres_rtol,
            maxiter=gmres_maxiter,
            restart=gmres_restart,
        )
        augmented_gmres = _short_gmres(
            augmented_operator,
            augmented_rhs,
            augmented_preconditioner_operator,
            rtol=gmres_rtol,
            maxiter=gmres_maxiter,
            restart=gmres_restart,
        )
        phase_frequency_rhs = np.concatenate((
            state_rhs,
            np.asarray([-phase_value]),
        ))
        phase_frequency_gmres = _short_gmres(
            phase_frequency_operator,
            phase_frequency_rhs,
            phase_frequency_preconditioner_operator,
            rtol=gmres_rtol,
            maxiter=gmres_maxiter,
            restart=gmres_restart,
        )
        diagonal_augmented_gmres = _short_gmres(
            augmented_operator,
            augmented_rhs,
            diagonal_augmented_preconditioner_operator,
            rtol=gmres_rtol,
            maxiter=gmres_maxiter,
            restart=gmres_restart,
        )
        augmented_random = rng.normal(size=state_size + 2)
        augmented_random /= max(float(np.linalg.norm(augmented_random)), 1e-300)
        augmented_solved = augmented_preconditioner(augmented_random)
        augmented_fidelity = float(
            np.linalg.norm(augmented_matvec(augmented_solved) - augmented_random)
            / max(np.linalg.norm(augmented_random), 1e-30)
        )
        phase_null_action = float(
            np.linalg.norm(matvec(phase_mode_packed))
            / max(np.linalg.norm(phase_mode_packed), 1e-30)
        )
        phase_frequency_rhs_cases = {
            "random": rng.normal(size=state_size + 1),
            "actual_newton": phase_frequency_rhs,
            "pure_phase": np.concatenate((
                np.zeros(state_size),
                np.asarray([1.0]),
            )),
            "omega_column": np.concatenate((
                phase_frequency_column,
                np.asarray([0.0]),
            )),
            "phase_mode": np.concatenate((
                phase_mode_packed / max(state_scale, 1e-300),
                np.asarray([0.0]),
            )),
        }
        phase_frequency_preconditioner_fidelity: dict[str, float] = {}
        direct_bordered_residual: dict[str, float] = {}
        for name, rhs in phase_frequency_rhs_cases.items():
            rhs = np.asarray(rhs, dtype=float)
            rhs /= max(float(np.linalg.norm(rhs)), 1e-300)
            preconditioned = phase_frequency_preconditioner(rhs)
            phase_frequency_preconditioner_fidelity[name] = float(
                np.linalg.norm(phase_frequency_matvec(preconditioned) - rhs)
                / max(np.linalg.norm(rhs), 1e-30)
            )
            direct_solution = direct_phase_frequency_solve(rhs)
            direct_bordered_residual[name] = float(
                np.linalg.norm(phase_frequency_matvec(direct_solution) - rhs)
                / max(np.linalg.norm(rhs), 1e-30)
            )
        border_singular_values = np.linalg.svd(
            border_schur, compute_uv=False
        )
        return {
            "state_scale": state_scale,
            "omega_scale": omega_scale,
            "tau_scale": tau_scale,
            "residual_scale": residual_scale,
            "state_preconditioner_fidelity": state_fidelity,
            "augmented_preconditioner_fidelity": augmented_fidelity,
            "state_only_gmres": state_gmres,
            "augmented_gmres": augmented_gmres,
            "phase_frequency_gmres": phase_frequency_gmres,
            "diagonal_augmented_gmres": diagonal_augmented_gmres,
            "phase_null_action_relative": phase_null_action,
            "phase_equivariance_relative": phase_equivariance_relative,
            "phase_frequency_fd_errors": phase_frequency_fd_errors,
            "phase_frequency_preconditioner_fidelity": (
                phase_frequency_preconditioner_fidelity
            ),
            "direct_bordered_residual": direct_bordered_residual,
            "border_schur_matrix": border_schur.tolist(),
            "border_schur_singular_values": border_singular_values.tolist(),
            "border_schur_condition": float(np.linalg.cond(border_schur)),
            "phase_frequency_schur": phase_frequency_schur,
            "phase_frequency_schur_abs": abs(phase_frequency_schur),
            "phase_residual": float(phase_value),
            "state_residual_norm": float(np.linalg.norm(state_rhs)),
            "lattice_labels": lattice_label_audit(problem, preconditioner),
            "factor_backend": preconditioner.last_factor_backend,
            "factor_phase": preconditioner.last_factor_phase,
        }

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

    def branch_lock_geometry(
        self,
        predictor: np.ndarray,
        *,
        beta: float = 1.0,
    ) -> "BranchLockGeometry":
        """Build an oblique phase and branch-lock row at ``predictor``.

        The row is expressed in normalized real state coordinates.  Its two
        components are the radial q != 0 direction and the autonomous phase
        direction.  The affine row therefore fixes the torus gauge while
        excluding the period-1 state from the fixed-drive corrector.
        """
        state = np.asarray(predictor, dtype=np.complex128)
        expected_shape = (self.basis.n_tones, self.base_problem.n)
        if state.shape != expected_shape:
            raise ValueError("predictor shape does not match the torus basis")
        q_values = np.asarray(
            [tone.q for tone in self.basis.tones], dtype=int
        )
        return build_branch_lock_geometry(state, q_values, beta=beta)

    def solve_newton_branch_locked(
        self,
        X0: np.ndarray,
        *,
        predictor_X: np.ndarray | None = None,
        omega_a0: float | None = None,
        beta: float = 1.0,
        branch_collapse_fraction: float = 0.25,
        max_newton: int = 20,
        residual_tol: float = 1.0e-9,
        min_alpha: float = 1.0 / 1024.0,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Correct a finite torus at fixed source drive with a branch lock.

        This is deliberately separate from :meth:`solve_newton`.  The latter
        retains the phase-anchor formulation for diagnostics and period-1
        control runs.  Here the affine oblique row is fixed at a finite torus
        predictor, so a Newton line search cannot accept the coexisting
        period-1 root merely because its q != 0 phase anchor vanishes.
        """
        if not 0.0 < branch_collapse_fraction < 1.0:
            raise ValueError("branch_collapse_fraction must lie in (0, 1)")
        if beta == 0.0:
            raise ValueError("beta must be nonzero")
        X = np.asarray(X0, dtype=np.complex128).copy()
        predictor = X.copy() if predictor_X is None else np.asarray(
            predictor_X, dtype=np.complex128
        )
        if X.shape != predictor.shape:
            raise ValueError("X0 and predictor_X shapes do not match")
        expected_shape = (self.basis.n_tones, self.base_problem.n)
        if X.shape != expected_shape:
            raise ValueError("X0 shape does not match the torus basis")
        geometry = self.branch_lock_geometry(predictor, beta=beta)
        predictor_generator_norm = self.generator_norm(predictor)
        if predictor_generator_norm <= 0.0:
            raise ValueError("predictor has no q != 0 content")
        omega = self.omega_a if omega_a0 is None else float(omega_a0)
        residual_history: list[float] = []
        merit_history: list[float] = []
        factor_backends: list[str] = []
        lock_history: list[float] = []
        collapse_history: list[float] = []

        for iteration in range(max_newton + 1):
            current = self.with_omega_a(omega)
            problem = current.full_problem()
            coefficients = problem.residual_coeffs(X, current.source_tau)
            packed = pack_complex(coefficients)
            source = pack_complex(problem.source_coeffs(current.source_tau))
            coefficient_scale = max(
                float(np.linalg.norm(source) / max(np.sqrt(source.size), 1.0)),
                1.0e-30,
            )
            coefficient_relative = float(
                np.linalg.norm(packed)
                / max(np.sqrt(packed.size), 1.0)
                / coefficient_scale
            )
            lock_value = geometry.value(X)
            merit = max(coefficient_relative, abs(lock_value))
            generator_ratio = self.generator_norm(X) / predictor_generator_norm
            residual_history.append(float(np.linalg.norm(packed)))
            merit_history.append(merit)
            lock_history.append(lock_value)
            collapse_history.append(generator_ratio)
            if coefficient_relative <= residual_tol and abs(lock_value) <= residual_tol:
                return X, omega, {
                    "converged": True,
                    "iterations": iteration,
                    "residual_norm": float(np.linalg.norm(packed)),
                    "coefficient_relative": coefficient_relative,
                    "lock_value": lock_value,
                    "generator_norm_relative": generator_ratio,
                    "residual_history": residual_history,
                    "merit_history": merit_history,
                    "lock_history": lock_history,
                    "generator_norm_history": collapse_history,
                    "factor_backend": factor_backends,
                    "branch_lock_beta": float(beta),
                    "branch_lock_phase_projection": (
                        geometry.phase_projection
                    ),
                    "branch_lock_radial_projection": (
                        geometry.radial_projection
                    ),
                    "branch_lock_pump_value": None,
                    "precond_reuse": self.precond_reuse,
                }
            if iteration == max_newton:
                break

            matvec, preconditioner = current._linearization(problem, X)
            step_size = current.omega_fd_relative_step * max(abs(omega), 1.0)
            plus = pack_complex(current.residual_coeffs(X, omega + step_size))
            minus = pack_complex(current.residual_coeffs(X, omega - step_size))
            d_omega = (plus - minus) / (2.0 * step_size)

            def lock_dot(vector: np.ndarray) -> float:
                return float(geometry.row_physical @ vector)

            update = bordered_solve_refined(
                matvec,
                preconditioner.solve,
                packed,
                lock_value,
                -d_omega,
                lock_dot,
                0.0,
            )
            factor_backends.append(preconditioner.last_factor_backend)
            if update is None:
                return X, omega, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": float(np.linalg.norm(packed)),
                    "coefficient_relative": coefficient_relative,
                    "lock_value": lock_value,
                    "generator_norm_relative": generator_ratio,
                    "residual_history": residual_history,
                    "merit_history": merit_history,
                    "lock_history": lock_history,
                    "generator_norm_history": collapse_history,
                    "factor_backend": factor_backends,
                    "branch_lock_beta": float(beta),
                    "branch_lock_phase_projection": geometry.phase_projection,
                    "branch_lock_radial_projection": geometry.radial_projection,
                    "branch_lock_pump_value": None,
                    "precond_reuse": self.precond_reuse,
                    "failure_reason": "branch-lock bordered denominator degenerated",
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
                trial_ratio = (
                    current.generator_norm(trial_x)
                    / predictor_generator_norm
                )
                if trial_ratio < branch_collapse_fraction:
                    alpha *= 0.5
                    continue
                trial_problem = current.with_omega_a(trial_omega).full_problem()
                trial_coefficients = trial_problem.residual_coeffs(
                    trial_x, current.source_tau
                )
                trial_packed = pack_complex(trial_coefficients)
                trial_relative = float(
                    np.linalg.norm(trial_packed)
                    / max(np.sqrt(trial_packed.size), 1.0)
                    / coefficient_scale
                )
                trial_lock = geometry.value(trial_x)
                trial_merit = max(trial_relative, abs(trial_lock))
                if trial_merit < merit:
                    X, omega = trial_x, trial_omega
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return X, omega, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": float(np.linalg.norm(packed)),
                    "coefficient_relative": coefficient_relative,
                    "lock_value": lock_value,
                    "generator_norm_relative": generator_ratio,
                    "residual_history": residual_history,
                    "merit_history": merit_history,
                    "lock_history": lock_history,
                    "generator_norm_history": collapse_history,
                    "factor_backend": factor_backends,
                    "branch_lock_beta": float(beta),
                    "branch_lock_phase_projection": geometry.phase_projection,
                    "branch_lock_radial_projection": geometry.radial_projection,
                    "branch_lock_pump_value": None,
                    "precond_reuse": self.precond_reuse,
                    "failure_reason": "branch-lock line search failed",
                }

        return X, omega, {
            "converged": False,
            "iterations": max_newton,
            "residual_norm": float(residual_history[-1]),
            "coefficient_relative": float(
                merit_history[-1] if merit_history else np.inf
            ),
            "lock_value": float(lock_history[-1]) if lock_history else np.inf,
            "generator_norm_relative": (
                float(collapse_history[-1]) if collapse_history else 0.0
            ),
            "residual_history": residual_history,
            "merit_history": merit_history,
            "lock_history": lock_history,
            "generator_norm_history": collapse_history,
            "factor_backend": factor_backends,
            "branch_lock_beta": float(beta),
            "branch_lock_phase_projection": geometry.phase_projection,
            "branch_lock_radial_projection": geometry.radial_projection,
            "branch_lock_pump_value": None,
            "precond_reuse": self.precond_reuse,
            "failure_reason": "maximum Newton iterations reached",
        }

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
        if self.is_schur:
            full_base = self.base_problem.full
            source_start = np.asarray(full_base.source_path.source_start)
            source_delta = np.asarray(full_base.source_path.source_delta)
            source_basis = full_base.basis
            source_nodes = full_base.circuit.node_count
        else:
            source_path = self.base_problem.source_path
            source_start = np.asarray(source_path.source_start)
            source_delta = np.asarray(source_path.source_delta)
            source_basis = self.base_problem.basis
            source_nodes = self.base_problem.n
        evaluation_start = np.zeros(
            (evaluation_basis.n_tones, source_nodes), dtype=np.complex128
        )
        evaluation_delta = np.zeros_like(evaluation_start)
        for row, tone in enumerate(source_basis.tones):
            target_row = evaluation_basis.index_of(tone)
            evaluation_start[target_row] = source_start[row]
            evaluation_delta[target_row] = source_delta[row]
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
        linear_debug: bool = False,
        linear_debug_fd_step: float = 1.0e-6,
        timing_path: Path | None = None,
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
        if linear_debug_fd_step <= 0.0:
            raise ValueError("linear_debug_fd_step must be positive")
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
        gmres_history_by_newton: list[list[float]] = []
        factor_backends: list[str] = []
        border_condition_history: list[float] = []
        linear_debug_report: dict[str, Any] | None = None
        preconditioner_solve_count = 0

        _append_timing_event(
            timing_path,
            "torus_corrector",
            "start",
            source_tau=tau,
            omega_a=omega,
            state_size=state_size,
        )

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

        def residual_from_evaluation(
            data: tuple[Any, np.ndarray, float, np.ndarray, float, float],
        ) -> np.ndarray:
            return np.concatenate((
                pack_complex(data[1]) / data[2],
                np.asarray([data[4], data[5]]),
            ))

        for iteration in range(max_newton + 1):
            residual_start = time.perf_counter()
            (
                current,
                coefficients,
                residual_scale,
                phase_row,
                phase_value,
                arclength,
            ) = evaluate(X, omega, tau)
            _append_timing_event(
                timing_path,
                "residual_evaluation",
                "after",
                iteration=iteration,
                runtime_s=time.perf_counter() - residual_start,
                residual_norm=float(
                    np.linalg.norm(pack_complex(coefficients))
                ),
            )
            residual = residual_from_evaluation((
                current,
                coefficients,
                residual_scale,
                phase_row,
                phase_value,
                arclength,
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
                    "gmres_history_by_newton": gmres_history_by_newton,
                    "gmres_iterations_total": sum(
                        len(item) for item in gmres_history_by_newton
                    ),
                    "factor_backend": factor_backends,
                    "border_schur_condition_history": border_condition_history,
                    "source_tau": tau,
                    "arclength_step": step_size,
                    "phase_constraint": phase_value,
                    "arclength_residual": arclength,
                    "linear_debug": linear_debug_report,
                }, new_tangent
            if iteration == max_newton:
                break

            linearization_start = time.perf_counter()
            matvec, preconditioner = current._linearization(
                current.full_problem(), X
            )
            _append_timing_event(
                timing_path,
                "linearization_and_preconditioner",
                "after",
                iteration=iteration,
                runtime_s=time.perf_counter() - linearization_start,
                assembly_runtime_s=preconditioner.last_assembly_runtime_s,
                factor_runtime_s=preconditioner.last_factor_runtime_s,
                factor_phase=preconditioner.last_factor_phase,
                factor_backend=preconditioner.last_factor_backend,
                pardiso_analyzed=bool(getattr(preconditioner, "_analyzed", False)),
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
            b_tau_raw = -pack_complex(
                current.full_problem().source_delta_coeffs()
            )
            tau_fd_step = current.omega_fd_relative_step * max(abs(tau), 1.0)
            plus_tau_problem = replace(
                current, source_tau=tau + tau_fd_step
            ).full_problem()
            minus_tau_problem = replace(
                current, source_tau=tau - tau_fd_step
            ).full_problem()
            plus_source = pack_complex(
                plus_tau_problem.source_coeffs(tau + tau_fd_step)
            )
            minus_source = pack_complex(
                minus_tau_problem.source_coeffs(tau - tau_fd_step)
            )
            residual_scale_tau = (
                float(np.linalg.norm(plus_source))
                - float(np.linalg.norm(minus_source))
            ) / (2.0 * tau_fd_step * max(np.sqrt(plus_source.size), 1.0))
            coefficient_vector = pack_complex(coefficients)
            b_tau = (
                b_tau_raw * residual_scale
                - coefficient_vector * residual_scale_tau
            ) / residual_scale**2

            def augmented_matvec(vector: np.ndarray) -> np.ndarray:
                delta_state = vector[:state_size]
                delta_omega = vector[state_size]
                delta_tau = vector[state_size + 1]
                physical_delta_state = delta_state * state_scale
                state_part = (
                    matvec(physical_delta_state)
                    + b_omega * omega_scale * delta_omega
                ) / residual_scale + b_tau * tau_scale * delta_tau
                phase_part = float(phase_row @ physical_delta_state)
                tangent_state = float(tangent_value[:state_size] @ delta_state)
                arclength_part = (
                    tangent_state
                    + tangent_value[state_size] * delta_omega
                    + tangent_value[state_size + 1] * delta_tau
                )
                return np.concatenate((
                    state_part,
                    np.asarray([phase_part, arclength_part]),
                ))

            border_start = time.perf_counter()
            border_columns = np.column_stack((
                b_omega * omega_scale / residual_scale,
                b_tau * tau_scale,
            ))
            constraint_rows = np.vstack((
                phase_row * state_scale,
                tangent_value[:state_size],
            ))
            border_matrix = np.array(
                [
                    [0.0, 0.0],
                    [
                        tangent_value[state_size],
                        tangent_value[state_size + 1],
                    ],
                ],
                dtype=float,
            )

            def state_preconditioner_solve(vector: np.ndarray) -> np.ndarray:
                nonlocal preconditioner_solve_count
                preconditioner_solve_count += 1
                solve_start = time.perf_counter()
                physical = residual_scale * np.asarray(
                    preconditioner.solve(vector), dtype=float
                )
                _append_timing_event(
                    timing_path,
                    "preconditioner_solve",
                    "after",
                    iteration=iteration,
                    call_index=preconditioner_solve_count,
                    runtime_s=time.perf_counter() - solve_start,
                )
                return physical / state_scale

            preconditioned_columns = np.column_stack(
                [state_preconditioner_solve(border_columns[:, index])
                 for index in range(2)]
            )
            border_schur = border_matrix - constraint_rows @ (
                preconditioned_columns
            )
            border_schur_condition = float(np.linalg.cond(border_schur))
            border_condition_history.append(border_schur_condition)
            _append_timing_event(
                timing_path,
                "augmented_border_setup",
                "after",
                iteration=iteration,
                runtime_s=time.perf_counter() - border_start,
                schur_condition=border_schur_condition,
            )

            def augmented_preconditioner(vector: np.ndarray) -> np.ndarray:
                state_result = state_preconditioner_solve(vector[:state_size])
                scalar_rhs = (
                    vector[state_size:] - constraint_rows @ state_result
                )
                scalars = np.linalg.solve(border_schur, scalar_rhs)
                return np.concatenate((
                    state_result - preconditioned_columns @ scalars,
                    scalars,
                ))

            if linear_debug and linear_debug_report is None:
                rng = np.random.default_rng(0)
                critical = np.zeros_like(pack_complex(X))
                if phase_reference is not None:
                    critical = pack_complex(phase_reference)
                phase_mode = np.zeros_like(X)
                for index, tone in enumerate(self.basis.tones):
                    if tone.q != 0:
                        phase_mode[index] = 1j * tone.q * X[index]

                def normalized_direction(
                    state_direction: np.ndarray,
                    omega_direction: float = 0.0,
                    tau_direction: float = 0.0,
                ) -> np.ndarray:
                    direction = np.concatenate((
                        pack_complex(state_direction) / state_scale,
                        np.asarray([omega_direction, tau_direction]),
                    ))
                    return direction / max(float(np.linalg.norm(direction)), 1e-300)

                random_state = unpack_complex(
                    rng.normal(size=state_size), X.shape
                )
                directions = {
                    "random": normalized_direction(random_state),
                    "critical_mode": normalized_direction(
                        unpack_complex(critical, X.shape)
                    ),
                    "phase_mode": normalized_direction(phase_mode),
                    "pure_omega": normalized_direction(
                        np.zeros_like(X), 1.0, 0.0
                    ),
                    "pure_tau": normalized_direction(
                        np.zeros_like(X), 0.0, 1.0
                    ),
                }

                def augmented_residual_at(
                    state: np.ndarray,
                    frequency: float,
                    source_tau: float,
                ) -> np.ndarray:
                    return residual_from_evaluation(
                        evaluate(state, frequency, source_tau)
                    )

                finite_difference_errors: dict[str, float] = {}
                for name, direction in directions.items():
                    state_delta = unpack_complex(
                        direction[:state_size] * state_scale, X.shape
                    )
                    frequency_delta = direction[state_size] * omega_scale
                    tau_delta = direction[state_size + 1] * tau_scale
                    trial = augmented_residual_at(
                        X + linear_debug_fd_step * state_delta,
                        omega + linear_debug_fd_step * frequency_delta,
                        tau + linear_debug_fd_step * tau_delta,
                    )
                    finite_difference = (trial - residual) / linear_debug_fd_step
                    analytic = augmented_matvec(direction)
                    finite_difference_errors[name] = float(
                        np.linalg.norm(finite_difference - analytic)
                        / max(np.linalg.norm(analytic), 1e-30)
                    )

                critical_norm = max(float(np.linalg.norm(critical)), 1e-300)
                phase_packed = pack_complex(phase_mode)
                phase_norm = max(float(np.linalg.norm(phase_packed)), 1e-300)
                linear_debug_report = {
                    "residual_coeff_norm": float(
                        np.linalg.norm(pack_complex(coefficients))
                    ),
                    "residual_norm": norm,
                    "omega_column_norm": float(
                        np.linalg.norm(b_omega * omega_scale)
                    ),
                    "tau_column_norm": float(
                        np.linalg.norm(b_tau * tau_scale)
                    ),
                    "phase_row_norm": float(np.linalg.norm(phase_row)),
                    "tangent_state_row_norm": float(
                        np.linalg.norm(tangent_value[:state_size])
                    ),
                    "j_critical_mode_norm": float(
                        np.linalg.norm(matvec(critical)) / critical_norm
                    ),
                    "j_phase_mode_norm": float(
                        np.linalg.norm(matvec(phase_packed)) / phase_norm
                    ),
                    "phase_on_critical": float(phase_row @ critical),
                    "phase_on_phase_mode": float(phase_row @ phase_packed),
                    "tangent_on_critical": float(
                        tangent_value[:state_size] @ (critical / state_scale)
                    ),
                    "tangent_on_phase_mode": float(
                        tangent_value[:state_size] @ (phase_packed / state_scale)
                    ),
                    "pure_omega_action_norm": float(
                        np.linalg.norm(augmented_matvec(directions["pure_omega"]))
                    ),
                    "pure_tau_action_norm": float(
                        np.linalg.norm(augmented_matvec(directions["pure_tau"]))
                    ),
                    "finite_difference_relative_errors": finite_difference_errors,
                    "border_schur_matrix": border_schur.tolist(),
                    "border_schur_condition": border_schur_condition,
                    "state_scale": state_scale,
                    "omega_scale": omega_scale,
                    "tau_scale": tau_scale,
                }

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
            gmres_start = time.perf_counter()
            _append_timing_event(
                timing_path,
                "augmented_gmres",
                "before",
                iteration=iteration,
                residual_norm=norm,
            )
            update, gmres_report = _gmres_solve(
                operator,
                -residual,
                preconditioner_operator,
                rtol=gmres_rtol,
                maxiter=gmres_maxiter,
                restart=gmres_restart,
            )
            gmres_history.extend(gmres_report["residual_history"])
            gmres_history_by_newton.append(list(gmres_history))
            gmres_info = int(gmres_report["info"])
            _append_timing_event(
                timing_path,
                "augmented_gmres",
                "after",
                iteration=iteration,
                runtime_s=time.perf_counter() - gmres_start,
                info=gmres_info,
                callback_iterations=len(gmres_history),
            )
            if gmres_info != 0:
                return X, omega, tau, {
                    "converged": False,
                    "iterations": iteration,
                    "residual_norm": norm,
                    "residual_history": history,
                    "gmres_residual_history": gmres_history,
                    "gmres_history_by_newton": gmres_history_by_newton,
                    "gmres_iterations_total": sum(
                        len(item) for item in gmres_history_by_newton
                    ),
                    "gmres_info": gmres_info,
                    "factor_backend": factor_backends,
                    "linear_debug": linear_debug_report,
                    "border_schur_matrix": border_schur.tolist(),
                    "border_schur_condition": border_schur_condition,
                    "border_schur_condition_history": border_condition_history,
                    "phase_constraint": phase_value,
                    "arclength_residual": arclength,
                    "failure_reason": "augmented GMRES failed",
                }, None

            state_step = update[:state_size]
            step_norm = float(np.linalg.norm(state_step) * state_scale)
            max_step = self.max_step_over_state * state_scale
            if step_norm > max_step:
                update = update * (max_step / step_norm)
            delta_state = unpack_complex(
                update[:state_size] * state_scale, X.shape
            )
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
                line_search_start = time.perf_counter()
                trial_data = evaluate(trial_X, trial_omega, trial_tau)
                _append_timing_event(
                    timing_path,
                    "line_search_residual",
                    "after",
                    iteration=iteration,
                    alpha=alpha,
                    runtime_s=time.perf_counter() - line_search_start,
                )
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
                    "gmres_history_by_newton": gmres_history_by_newton,
                    "gmres_iterations_total": sum(
                        len(item) for item in gmres_history_by_newton
                    ),
                    "factor_backend": factor_backends,
                    "linear_debug": linear_debug_report,
                    "border_schur_matrix": border_schur.tolist(),
                    "border_schur_condition": border_schur_condition,
                    "border_schur_condition_history": border_condition_history,
                    "phase_constraint": phase_value,
                    "arclength_residual": arclength,
                    "failure_reason": "line search failed",
                }, None

        return X, omega, tau, {
            "converged": False,
            "iterations": max_newton,
            "residual_norm": float(history[-1]),
            "residual_history": history,
            "gmres_residual_history": gmres_history,
            "gmres_history_by_newton": gmres_history_by_newton,
            "gmres_iterations_total": sum(
                len(item) for item in gmres_history_by_newton
            ),
            "factor_backend": factor_backends,
            "border_schur_condition_history": border_condition_history,
            "linear_debug": linear_debug_report,
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
