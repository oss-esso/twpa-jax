"""Tests for the Jacobian singularity test functions
(``twpa_solver.pump.singularity``): a fold is a zero eigenvalue of the exact
real-packed pump Jacobian, and this module is the first place in
``twpa_solver.pump`` that measures it directly instead of inferring it from
solver behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from twpa_solver.pump.singularity import (  # noqa: E402
    _permutation_parity,
    jacobian_det_signature,
    jacobian_min_eigenvalue,
)


class _DenseStubProblem:
    """Minimal duck-typed stand-in exposing exactly what both singularity
    functions need: ``tangent_state``/``spectral_tangent_state`` are pass-
    throughs, and ``assemble_real_coupled_preconditioner`` wraps an arbitrary
    dense real matrix as a real ``spla.SuperLU`` factor. This exercises the
    exact same dispatch path (no ``assemble_real_coupled_fast`` attribute, so
    both functions fall back to the plain SuperLU branch) with a fully known,
    hand-picked spectrum -- the real Josephson toy problem's spectrum is not
    directly controllable, which this needs to be.
    """

    def tangent_state(self, X: np.ndarray) -> np.ndarray:
        return X

    def spectral_tangent_state(self, tangent: np.ndarray) -> np.ndarray:
        return tangent

    def assemble_real_coupled_preconditioner(self, spectral: np.ndarray) -> spla.SuperLU:
        return spla.splu(sp.csc_matrix(spectral))


def _symmetric_matrix_with_spectrum(eigs: np.ndarray, *, seed: int = 7) -> np.ndarray:
    n = eigs.size
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q * eigs) @ Q.T


def test_min_eigenvalue_matches_known_spectrum() -> None:
    eigs = np.array([-3.0, -1.0, 0.001, 2.0, 4.0, 5.0])
    A = _symmetric_matrix_with_spectrum(eigs)
    problem = _DenseStubProblem()

    estimate = jacobian_min_eigenvalue(problem, A, 0.0, iters=40)
    true_min = eigs[np.argmin(np.abs(eigs))]
    assert estimate == pytest.approx(true_min, rel=1e-6)


def test_det_signature_matches_known_spectrum() -> None:
    eigs = np.array([-3.0, -1.0, 2.0, 4.0, 5.0])
    A = _symmetric_matrix_with_spectrum(eigs)
    problem = _DenseStubProblem()

    sign, log_abs_det = jacobian_det_signature(problem, A, 0.0)
    true_sign, true_log_abs_det = np.linalg.slogdet(A)
    assert sign == int(true_sign)
    assert log_abs_det == pytest.approx(true_log_abs_det, rel=1e-9)


def test_det_signature_flips_sign_across_a_turning_point() -> None:
    # One eigenvalue crosses zero at lam=0.5; the other four stay fixed and
    # bounded away from zero, matching a genuine simple fold's spectrum
    # (exactly one eigenvalue crosses, not the whole spectrum drifting).
    n = 6
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    problem = _DenseStubProblem()

    def matrix_at(lam: float) -> np.ndarray:
        eigs = np.array([-3.0, -1.0, lam - 0.5, 2.0, 4.0, 5.0])
        return (Q * eigs) @ Q.T

    sign_before, _ = jacobian_det_signature(problem, matrix_at(0.3), 0.3)
    sign_after, _ = jacobian_det_signature(problem, matrix_at(0.7), 0.7)
    assert sign_before == -1
    assert sign_after == 1
    assert sign_before != sign_after

    # min_eigenvalue should track the crossing eigenvalue directly and
    # change sign at the same point.
    eig_before = jacobian_min_eigenvalue(problem, matrix_at(0.3), 0.3, iters=40)
    eig_after = jacobian_min_eigenvalue(problem, matrix_at(0.7), 0.7, iters=40)
    assert eig_before < 0.0 < eig_after


def test_permutation_parity_matches_transposition_count() -> None:
    assert _permutation_parity(np.arange(5)) == 1  # identity: even (0 transpositions)
    assert _permutation_parity(np.array([1, 0, 2, 3, 4])) == -1  # one swap: odd
    assert _permutation_parity(np.array([1, 0, 3, 2])) == 1  # two swaps: even
    assert _permutation_parity(np.array([2, 0, 1])) == 1  # one 3-cycle: even (2 transpositions)


def test_min_eigenvalue_invariant_under_state_rescaling() -> None:
    # Same trap Phase 1 closed for the arclength metric: X is node flux
    # (webers, ~1e-13 on a real device), so any function of X that is
    # accidentally unit-dependent would silently misreport on production
    # scales despite passing on an O(1) toy fixture. The residual
    # D(w)X + Bphi*Ic*sin(psi/phi0) - S is exactly covariant under
    # X -> s*X when phi0 -> s*phi0, Ic -> s*Ic, pump_current -> s*pump_current
    # (C, G, K unchanged), so the Jacobian dR/dX -- and therefore its
    # eigenvalues -- must come out IDENTICAL (not merely proportional) at
    # s=1 and s=1e-15, evaluated at the correspondingly-scaled X.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_advanced_continuation import _build_scaled_problem, _solver  # noqa: E402

    solver = _solver()
    problem_unit = _build_scaled_problem(0.6, 1.0)
    X_unit, rep_unit = solver.solve_one(problem_unit, problem_unit.zeros(), 0.5)
    assert rep_unit.converged

    s = 1e-15
    problem_scaled = _build_scaled_problem(0.6, s)
    X_scaled, rep_scaled = solver.solve_one(problem_scaled, problem_scaled.zeros(), 0.5)
    assert rep_scaled.converged

    eig_unit = jacobian_min_eigenvalue(problem_unit, X_unit, 0.5, iters=40)
    eig_scaled = jacobian_min_eigenvalue(problem_scaled, X_scaled, 0.5, iters=40)
    assert eig_scaled == pytest.approx(eig_unit, rel=1e-6)
