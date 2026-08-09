"""Tests for the Govaerts-Pryce refinement pass on the arclength bordered
system (``solver.py::bordered_solve_refined``) and the least-squares
singular-factor fallback (``fast_coupled.py::FastCoupledPreconditioner``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from twpa_solver.pump.backends.fast_coupled import FastCoupledPreconditioner  # noqa: E402
from twpa_solver.pump.solver import _bordered_block_step, bordered_solve_refined  # noqa: E402


def _simple_fold_system(smallest_sv: float, *, seed: int = 3, n: int = 10):
    """A generic real matrix with exactly one near-zero singular value and
    all others O(1) -- the "simple fold" spectrum a genuine turning point
    produces, as opposed to a matrix that is merely globally ill-conditioned.

    The border (``c``, ``S``) is aligned with the fold's right/left
    near-null singular vectors, matching how the real arclength tangent
    ``Xdot ~ J^-1 S`` and the metric row ``c = Xdot`` behave near an actual
    turning point. With this alignment the bordered ``(n+1)x(n+1)`` matrix
    stays well-conditioned (cond ~ 1.5-1.7) no matter how singular ``A``
    itself becomes -- the whole premise of pseudo-arclength continuation.
    """
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv = np.ones(n)
    sv[-1] = smallest_sv
    A = (Q1 * sv) @ Q2.T
    v = Q2[:, -1]  # right near-null vector: A v ~ smallest_sv * w
    w = Q1[:, -1]  # left near-null vector: w^T A ~ smallest_sv * v^T
    S = w + 0.1 * rng.standard_normal(n)  # generic: w^T S != 0 (transversality)
    R = rng.standard_normal(n)
    target = 0.123
    lam_dot = 0.37
    c = v
    return A, S, R, target, lam_dot, c


def _dense_bordered_reference(A, S, R, target, lam_dot, c):
    n = A.shape[0]
    B = np.zeros((n + 1, n + 1))
    B[:n, :n] = A
    B[:n, n] = -S
    B[n, :n] = c
    B[n, n] = lam_dot
    rhs = np.concatenate([-R, [-target]])
    sol = np.linalg.solve(B, rhs)
    return sol[:n], sol[n]


def test_refined_bordered_solve_matches_dense_reference_near_singular():
    # cond(A) = 1e13: a direct (LU-based, roundoff-limited) linsolve of A
    # alone picks up a relative error of order cond(A) * machine_eps ~ 1e-3,
    # which plain block elimination (Keller 1977) amplifies into the output.
    A, S, R, target, lam_dot, c = _simple_fold_system(smallest_sv=1e-13)

    def linsolve(rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(A, rhs)

    def matvec(v: np.ndarray) -> np.ndarray:
        return A @ v

    def c_dot(v: np.ndarray) -> float:
        return float(c @ v)

    dX_ref, dlam_ref = _dense_bordered_reference(A, S, R, target, lam_dot, c)
    ref_scale = max(np.linalg.norm(dX_ref), abs(dlam_ref))

    plain = _bordered_block_step(linsolve, -R, -target, S, c_dot, lam_dot)
    assert plain is not None
    d_X_plain, d_lam_plain, _b = plain
    err_plain = max(
        np.max(np.abs(d_X_plain - dX_ref)), abs(d_lam_plain - dlam_ref)
    ) / ref_scale

    refined = bordered_solve_refined(matvec, linsolve, R, target, S, c_dot, lam_dot)
    assert refined is not None
    d_X_ref_step, d_lam_ref_step = refined
    err_refined = max(
        np.max(np.abs(d_X_ref_step - dX_ref)), abs(d_lam_ref_step - dlam_ref)
    ) / ref_scale

    # Plain block elimination is measurably degraded by A's conditioning;
    # one refinement pass restores it to within a couple orders of machine
    # epsilon, and is markedly (>1000x) more accurate than the plain result.
    assert err_plain > 1e-6
    assert err_refined < 1e-12
    assert err_refined < err_plain / 1000.0


def test_refined_bordered_solve_degrades_gracefully_as_singularity_tightens():
    # The plain-elimination error should grow with cond(A); the refined
    # error should stay pinned near machine precision throughout.
    errs_plain = []
    errs_refined = []
    for smallest_sv in (1e-8, 1e-10, 1e-12, 1e-14):
        A, S, R, target, lam_dot, c = _simple_fold_system(smallest_sv=smallest_sv)

        def linsolve(rhs: np.ndarray, A=A) -> np.ndarray:
            return np.linalg.solve(A, rhs)

        def matvec(v: np.ndarray, A=A) -> np.ndarray:
            return A @ v

        def c_dot(v: np.ndarray, c=c) -> float:
            return float(c @ v)

        dX_ref, dlam_ref = _dense_bordered_reference(A, S, R, target, lam_dot, c)
        ref_scale = max(np.linalg.norm(dX_ref), abs(dlam_ref))

        plain = _bordered_block_step(linsolve, -R, -target, S, c_dot, lam_dot)
        d_X_plain, d_lam_plain, _b = plain
        errs_plain.append(
            max(np.max(np.abs(d_X_plain - dX_ref)), abs(d_lam_plain - dlam_ref))
            / ref_scale
        )

        refined = bordered_solve_refined(matvec, linsolve, R, target, S, c_dot, lam_dot)
        d_X_r, d_lam_r = refined
        errs_refined.append(
            max(np.max(np.abs(d_X_r - dX_ref)), abs(d_lam_r - dlam_ref)) / ref_scale
        )

    assert errs_plain[-1] > errs_plain[0]  # plain degrades with cond(A)
    assert all(e < 1e-12 for e in errs_refined)  # refined stays pinned


def _make_bare_preconditioner(M: sp.csr_matrix, *, use_pardiso: bool) -> FastCoupledPreconditioner:
    """Build a ``FastCoupledPreconditioner`` bypassing ``__init__`` entirely.

    ``__init__`` requires a full circuit problem (Schur-reduced pump basis,
    branch law, etc.) just to fix ``M``'s sparsity pattern; the singular
    factor/solve fallback under test here depends only on ``M`` and a handful
    of scalar flags, so construct the object directly with ``__new__`` and
    set exactly what ``_factor``/``solve``/``_lsq_solve`` read.
    """
    obj = FastCoupledPreconditioner.__new__(FastCoupledPreconditioner)
    obj.M = M.tocsr()
    obj.use_banded = False
    obj.use_pardiso = use_pardiso
    obj._pardiso = None
    obj._lu = None
    obj._analyzed = False
    obj._singular_fallback = False
    obj.last_assembly_runtime_s = 0.0
    obj.last_factor_runtime_s = 0.0
    obj.last_pardiso_error = ""
    obj.last_factor_backend = ""
    obj.pardiso_strict = False
    return obj


def test_exactly_singular_matrix_falls_back_to_least_squares() -> None:
    n = 8
    A = sp.identity(n, format="lil", dtype=np.float64)
    A[-1, -1] = 0.0  # exactly singular: last row/col is all zero
    M = A.tocsr()

    precond = _make_bare_preconditioner(M, use_pardiso=False)
    precond._factor()

    assert precond.last_factor_backend == "lsq_singular_fallback"
    assert precond._singular_fallback is True

    rhs = np.ones(n)
    x = precond.solve(rhs)
    assert np.all(np.isfinite(x))
    assert precond.last_factor_backend == "lsq_singular_fallback"
    # Least-squares minimum-norm solution: for this diagonal system the
    # first n-1 unknowns are exactly determined (identity block) and the
    # last is driven to 0 by minimum-norm (its column is entirely zero, so
    # it contributes nothing to the residual either way).
    np.testing.assert_allclose(x[:-1], rhs[:-1], atol=1e-8)
    assert abs(x[-1]) < 1e-6


def test_singular_fallback_does_not_raise_on_repeated_solve() -> None:
    n = 6
    A = sp.identity(n, format="lil", dtype=np.float64)
    A[2, 2] = 0.0
    A[5, 5] = 0.0
    M = A.tocsr()

    precond = _make_bare_preconditioner(M, use_pardiso=False)
    precond._factor()
    assert precond.last_factor_backend == "lsq_singular_fallback"

    for _ in range(3):
        x = precond.solve(np.arange(n, dtype=np.float64))
        assert np.all(np.isfinite(x))
