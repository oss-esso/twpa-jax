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
    _assembled_jacobian_matrix,
    _permutation_parity,
    jacobian_det_signature,
    jacobian_min_eigenvalue,
    smallest_singular_triplets,
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


class _CachedMutableFactor:
    """Mimics the production ``FastCoupledPreconditioner``: one object,
    mutated in place on every ``refactor``/assemble call, exactly the
    pattern that caused the aliasing bug this test guards against."""

    def __init__(self) -> None:
        self.M = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))

    def solve(self, b: np.ndarray) -> np.ndarray:
        return b / self.M[0, 0]


class _FastCachedStubProblem:
    """Duck-typed stand-in for ``SchurReducedProblem``: exposes
    ``assemble_real_coupled_fast`` returning the SAME cached factor object
    every call, mutated to reflect the tangent passed in -- the real
    caching behavior ``_assembled_jacobian_matrix`` must not be fooled by."""

    def __init__(self) -> None:
        self._factor = _CachedMutableFactor()

    def tangent_state(self, X: np.ndarray) -> np.ndarray:
        return X

    def assemble_real_coupled_fast(self, tangent: np.ndarray) -> _CachedMutableFactor:
        self._factor.M = sp.csr_matrix(np.array([[float(tangent[0, 0])]], dtype=np.float64))
        return self._factor


def test_assembled_jacobian_matrix_is_not_aliased_across_calls() -> None:
    # Regression for the Milestone F.5 bug: _assembled_jacobian_matrix used
    # to return a direct reference to the production preconditioner's
    # cached, in-place-mutated buffer, so two snapshots at different X
    # silently became the SAME object once the second call ran -- any
    # finite-difference-style comparison saw an exact-zero difference
    # regardless of the true derivative. This is exactly what happened to
    # the finite-difference alpha at mu~0.5253 on designs/ipm_2c_fixed
    # (bit-identical "difference" that was really `Ma is Mb`), while the
    # unrelated exact AFT alpha (which never calls this function) found the
    # true value to be hugely nonzero.
    problem = _FastCachedStubProblem()
    Ma = _assembled_jacobian_matrix(problem, np.array([[1.0]]))
    Mb = _assembled_jacobian_matrix(problem, np.array([[2.0]]))

    assert Ma is not Mb
    assert Ma[0, 0] == pytest.approx(1.0)
    assert Mb[0, 0] == pytest.approx(2.0)


def _folding_branch():
    """Shared real-solver fixture for the smallest_singular_triplets tests
    below: the same 1-DOF near-resonant fold used throughout
    test_advanced_continuation.py (genuine fold at lambda~0.7808)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_advanced_continuation import _build_folding_problem, _solver  # noqa: PLC0415

    from twpa_solver.pump.solver import trace_branch  # noqa: PLC0415

    solver = _solver()
    problem = _build_folding_problem(pump_current=1.0)
    branch = trace_branch(
        solver, problem, i_ref=1.0, mu0=0.0, mu_max=1.0, ds=0.05, max_steps=250,
    )
    return problem, branch


def test_smallest_singular_triplets_shrinks_near_a_known_fold() -> None:
    # Milestone F's core premise: sigma1(J) should fall toward zero as the
    # branch approaches a genuine fold, and stay comparatively large away
    # from it -- this is the real-solver analogue of the eigenvalue check
    # test_det_signature_flips_sign_across_a_turning_point already runs on a
    # hand-built spectrum, but on the actual Jacobian this repo assembles.
    problem, branch = _folding_branch()
    near_fold_pt = min(branch.points, key=lambda p: abs(p.mu - 0.7808))
    far_pt = min(branch.points, key=lambda p: abs(p.mu - 0.1))

    triplet_near = smallest_singular_triplets(problem, near_fold_pt.X, k=2, iters=40)
    triplet_far = smallest_singular_triplets(problem, far_pt.X, k=2, iters=40)

    assert triplet_near.converged and triplet_near.estimator == "augmented_shift_invert"
    assert triplet_far.converged and triplet_far.estimator == "augmented_shift_invert"
    assert triplet_near.sigma[0] < triplet_far.sigma[0]

    hat_near = triplet_near.sigma[0] / triplet_near.sigma_ref
    hat_far = triplet_far.sigma[0] / triplet_far.sigma_ref
    assert hat_near < hat_far


def test_smallest_singular_triplets_vectors_are_unit_norm_and_finite() -> None:
    problem, branch = _folding_branch()
    near_fold_pt = min(branch.points, key=lambda p: abs(p.mu - 0.7808))

    triplet = smallest_singular_triplets(problem, near_fold_pt.X, k=2, iters=40)

    assert triplet.converged
    assert len(triplet.sigma) == 2
    assert triplet.sigma[0] <= triplet.sigma[1]
    for u, v in zip(triplet.u, triplet.v):
        assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
        assert np.linalg.norm(u) == pytest.approx(1.0, rel=1e-6)
        assert np.linalg.norm(v) == pytest.approx(1.0, rel=1e-6)
