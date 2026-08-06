"""Singularity test functions for the harmonic-balance pump Jacobian.

Nothing in ``twpa_solver.pump`` previously tracked a singularity measure:
"the solver stopped converging" and "the branch genuinely turned" were
indistinguishable without one (see
``docs/development/arclength_metric_fix_and_fold_test_function_plan.md``,
Phase 4). Both functions here reuse the same exact real-packed Jacobian the
Newton-Krylov solver already assembles/factors every step, so they are cheap
diagnostics, not a new physics model.

Reading the two together at adjacent converged points along a branch:

* ``jacobian_min_eigenvalue`` -> 0 smoothly as the drive approaches a wall
  means a genuine fold, and gives the fold location.
* ``jacobian_min_eigenvalue`` staying at its low-drive magnitude while Newton
  fails to converge means the wall is numerical, not physical.
* A sign change of ``jacobian_det_signature`` between two adjacent converged
  points means an odd number of eigenvalues crossed zero between them -- a
  genuine simple singularity was passed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _permutation_parity(perm: np.ndarray) -> int:
    """Sign (+1/-1) of a permutation array, via cycle decomposition."""
    perm = np.asarray(perm, dtype=np.int64)
    n = perm.size
    visited = np.zeros(n, dtype=bool)
    parity = 1
    for i in range(n):
        if visited[i]:
            continue
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = int(perm[j])
            cycle_len += 1
        if cycle_len % 2 == 0:
            parity = -parity
    return parity


def _real_coupled_factor(problem: Any, tangent: Any):
    """Dispatch to the Schur ``FastCoupledPreconditioner`` when available,
    else the plain dense-preconditioner path -- the same dispatch
    ``solver.py::_linear_solver`` uses, so this measures the identical
    Jacobian the Newton-Krylov corrector actually solves against.
    """
    if hasattr(problem, "assemble_real_coupled_fast"):
        return problem.assemble_real_coupled_fast(tangent)
    spectral = problem.spectral_tangent_state(tangent)
    return problem.assemble_real_coupled_preconditioner(spectral)


def jacobian_min_eigenvalue(
    problem: Any,
    X: np.ndarray,
    lam: float,
    *,
    iters: int = 20,
    seed: int = 0,
) -> float:
    """Smallest-magnitude eigenvalue of the exact real-packed pump Jacobian.

    Inverse power iteration (``v <- M^-1 v / ||M^-1 v||``) using only the
    already-factored solver's ``.solve`` -- no transpose needed, so this runs
    identically on the Schur ``FastCoupledPreconditioner`` (PARDISO/SuperLU/
    lsqr-fallback) and the plain ``spla.SuperLU`` path. A fold is a zero
    *eigenvalue* of the Jacobian, not merely a small singular value: this
    repo's device Jacobians are not symmetric, so eigenvalues and singular
    values differ, and the eigenvalue is the physically meaningful one (it is
    what crosses zero at a turning point of the underlying nonlinear map).

    ``lam`` is accepted for API symmetry with :func:`jacobian_det_signature`
    and future callers that key results by ``(X, lam)``; the Jacobian itself
    only depends on ``X`` through ``tangent_state``.

    Returns the eigenvalue estimate as a signed real float (the Rayleigh
    quotient of ``M^-1``, inverted); ``iters=20`` is ~20 cheap back-solves
    reusing one cached factor, seconds per point on a production circuit.
    """
    del lam  # Jacobian depends on X only; kept for a uniform (problem, X, lam) call signature.
    tangent = problem.tangent_state(X)
    factor = _real_coupled_factor(problem, tangent)
    dim = factor.M.shape[0] if hasattr(factor, "M") else factor.shape[0]

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)

    inv_eig = 0.0
    for _ in range(int(iters)):
        w = factor.solve(v)
        if not np.all(np.isfinite(w)):
            return float("nan")
        norm_w = float(np.linalg.norm(w))
        if norm_w < 1e-300:
            return float("inf")  # M^-1 v collapsed to zero: no small eigenvalue here.
        inv_eig = float(v @ w)  # Rayleigh quotient of M^-1 at the current iterate.
        v = w / norm_w

    if inv_eig == 0.0 or not math.isfinite(inv_eig):
        return float("inf")
    return 1.0 / inv_eig


def jacobian_det_signature(
    problem: Any, X: np.ndarray, lam: float,
) -> tuple[int, float]:
    """``(sign, log|det|)`` of the exact real-packed pump Jacobian.

    Requires an actual ``spla.SuperLU`` factor (``L``, ``U``, ``perm_r``,
    ``perm_c``), so this always goes through
    ``assemble_real_coupled_preconditioner`` -- a separate, diagnostic-only
    factorization that does not disturb the hot path's cached PARDISO
    factor. SuperLU's ``L`` has unit diagonal by construction, so
    ``det(A) = sign(perm_r) * sign(perm_c) * prod(diag(U))``; only ``U``'s
    diagonal contributes to the magnitude.

    An exactly singular assembled Jacobian makes ``spla.splu`` raise rather
    than return a factor with a zero pivot; that failure is itself the
    signature this function exists to report, so it is caught and mapped to
    ``(0, -inf)`` (det exactly zero) rather than propagating.
    """
    del lam  # kept for API symmetry with jacobian_min_eigenvalue.
    tangent = problem.tangent_state(X)
    spectral = problem.spectral_tangent_state(tangent)
    try:
        factor = problem.assemble_real_coupled_preconditioner(spectral)
    except RuntimeError:
        return 0, float("-inf")

    diag_u = np.asarray(factor.U.diagonal(), dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_abs_det = float(np.sum(np.log(np.abs(diag_u))))
    sign_u = int(np.prod(np.sign(diag_u)))  # 0 if any pivot is exactly zero

    sign = sign_u * _permutation_parity(factor.perm_r) * _permutation_parity(factor.perm_c)
    return sign, log_abs_det
