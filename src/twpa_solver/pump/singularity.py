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
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla

from twpa_solver.pump.problem import pack_complex, unpack_complex
from twpa_solver.pump.solver import _bordered_block_step, _real_dot


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


def _inverse_power_min_eigenvalue(
    factor: Any, dim: int, *, iters: int, seed: int,
) -> float:
    """Original estimator: inverse power iteration on ``factor.solve``.

    Kept as the fallback when shift-invert Arnoldi fails to converge (e.g. on
    a dimension too small for ARPACK's ``ncv`` bookkeeping, or a genuinely
    pathological operator).
    """
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


def _jacobian_min_eigenvalue_impl(
    problem: Any,
    X: np.ndarray,
    lam: float,
    *,
    iters: int,
    seed: int,
) -> tuple[float, str]:
    """Shared implementation: returns ``(eigenvalue, estimator_name)``.

    ``estimator_name`` is ``"shift_invert"`` (``scipy.sparse.linalg.eigs``,
    shift-invert Arnoldi around ``sigma=0`` -- finds the eigenvalue nearest
    zero directly) or ``"inverse_power"`` (the original 20-iteration inverse
    power/Rayleigh-quotient loop, used only when shift-invert raises, e.g.
    ``ArpackNoConvergence`` or a dimension too small for ARPACK).
    """
    del lam  # Jacobian depends on X only; kept for a uniform (problem, X, lam) call signature.
    tangent = problem.tangent_state(X)
    factor = _real_coupled_factor(problem, tangent)
    dim = factor.M.shape[0] if hasattr(factor, "M") else factor.shape[0]
    shape = X.shape

    def op_inv(v: np.ndarray) -> np.ndarray:
        return factor.solve(v)

    def op_forward(v: np.ndarray) -> np.ndarray:
        V = unpack_complex(v, shape)
        JV = problem.jvp_coeffs_with_tangent(V, tangent)
        return pack_complex(JV)

    OPinv = spla.LinearOperator((dim, dim), matvec=op_inv, dtype=np.float64)
    A = spla.LinearOperator((dim, dim), matvec=op_forward, dtype=np.float64)

    try:
        eigvals = spla.eigs(
            A, k=1, sigma=0.0, which="LM", OPinv=OPinv,
            maxiter=max(50, int(iters) * 5), return_eigenvectors=False,
        )
        val = complex(eigvals[0])
        if not np.isfinite(val):
            return float("nan"), "shift_invert"
        return float(val.real), "shift_invert"
    except (spla.ArpackNoConvergence, RuntimeError, ValueError, FloatingPointError):
        pass

    return _inverse_power_min_eigenvalue(factor, dim, iters=iters, seed=seed), "inverse_power"


def jacobian_min_eigenvalue(
    problem: Any,
    X: np.ndarray,
    lam: float,
    *,
    iters: int = 20,
    seed: int = 0,
) -> float:
    """Smallest-magnitude eigenvalue of the exact real-packed pump Jacobian.

    Shift-invert Arnoldi (``scipy.sparse.linalg.eigs``, ``sigma=0``) around
    the already-factored solver's ``.solve`` as ``OPinv`` -- no transpose
    needed, so this runs identically on the Schur ``FastCoupledPreconditioner``
    (PARDISO/SuperLU/lsqr-fallback) and the plain ``spla.SuperLU`` path. Falls
    back to inverse power iteration if shift-invert fails to converge. A fold
    is a zero *eigenvalue* of the Jacobian, not merely a small singular value:
    this repo's device Jacobians are not symmetric, so eigenvalues and
    singular values differ, and the eigenvalue is the physically meaningful
    one (it is what crosses zero at a turning point of the underlying
    nonlinear map).

    ``lam`` is accepted for API symmetry with :func:`jacobian_det_signature`
    and future callers that key results by ``(X, lam)``; the Jacobian itself
    only depends on ``X`` through ``tangent_state``.

    Returns the eigenvalue estimate as a signed real float. Use
    :func:`jacobian_min_eigenvalue_with_estimator` to also learn which
    estimator produced it.
    """
    value, _estimator = _jacobian_min_eigenvalue_impl(problem, X, lam, iters=iters, seed=seed)
    return value


def jacobian_min_eigenvalue_with_estimator(
    problem: Any,
    X: np.ndarray,
    lam: float,
    *,
    iters: int = 20,
    seed: int = 0,
) -> tuple[float, str]:
    """Same as :func:`jacobian_min_eigenvalue`, also returning which
    estimator produced the value (``"shift_invert"`` or ``"inverse_power"``).
    """
    return _jacobian_min_eigenvalue_impl(problem, X, lam, iters=iters, seed=seed)


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


def bordered_conditioning(
    problem: Any,
    X: np.ndarray,
    lam: float,
    Xdot: np.ndarray,
    lam_dot: float,
    state_scale: float,
    *,
    iters: int = 20,
    seed: int = 0,
) -> float:
    """Condition-number estimate of the pseudo-arclength bordered system.

    The direct fold-vs-branch-point discriminator: a genuine fold leaves the
    *bordered* system well-conditioned even though ``J`` itself is singular
    there (bordering exists precisely to regularize that rank-1 degeneracy,
    Keller 1977). A branch point -- two solution branches crossing -- is a
    rank-2 degeneracy that one border cannot regularize, so the bordered
    system goes ill-conditioned too. ``jacobian_min_eigenvalue`` alone cannot
    tell these apart; this function can.

    Mirrors ``solve_arclength``'s own corrector exactly: the inverse action
    reuses ``solver.py``'s bordering primitive (``_bordered_block_step``)
    with the real-coupled factor's ``.solve`` as the approximate ``J^-1`` --
    the same near-exact factor ``jacobian_min_eigenvalue`` uses, not GMRES
    (this is a diagnostic, not the production corrector). The forward action
    is the augmented operator ``A(dX, dlam) = (J dX - dlam*S, metric_x(Xdot,
    dX) + lam_dot*dlam)`` applied via a cheap JVP (no solve).

    Power iteration (same cost/style as ``jacobian_min_eigenvalue``, no
    dense reconstruction): forward iteration estimates ``sigma_max(A)``,
    inverse iteration (via bordering) estimates ``1/sigma_min(A)``. Returns
    their product, or ``inf`` if either iteration collapses/diverges
    (measurably degenerate bordered system).
    """
    del lam  # kept for API symmetry with jacobian_min_eigenvalue/jacobian_det_signature.
    tangent = problem.tangent_state(X)
    factor = _real_coupled_factor(problem, tangent)
    shape = X.shape
    S = problem.source_coeffs(1.0)

    def linsolve(v: np.ndarray) -> np.ndarray:
        return unpack_complex(factor.solve(pack_complex(v)), shape)

    def c_dot(v: np.ndarray) -> float:
        return _real_dot(Xdot, v) / (state_scale * state_scale)

    def combined_norm(dX: np.ndarray, dlam: float) -> float:
        return math.sqrt(_real_dot(dX, dX) / (state_scale * state_scale) + dlam * dlam)

    rng = np.random.default_rng(seed)
    dX = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    dlam = float(rng.standard_normal())
    nrm0 = combined_norm(dX, dlam)
    if not math.isfinite(nrm0) or nrm0 < 1e-300:
        return float("nan")
    dX, dlam = dX / nrm0, dlam / nrm0

    # Forward power iteration: estimates sigma_max(A) via cheap JVP, no solve.
    sigma_max = 0.0
    fX, flam = dX, dlam
    for _ in range(int(iters)):
        JV = problem.jvp_coeffs_with_tangent(fX, tangent)
        r_a = JV - flam * S
        r_c = c_dot(fX) + lam_dot * flam
        nrm = combined_norm(r_a, r_c)
        if not math.isfinite(nrm):
            return float("nan")
        if nrm < 1e-300:
            return float("inf")  # A collapses a generic direction to zero.
        sigma_max = nrm
        fX, flam = r_a / nrm, r_c / nrm

    # Inverse power iteration: estimates 1/sigma_min(A) via the bordering
    # primitive, reusing linsolve as the approximate J^-1. J b = S is
    # constant across every iteration (X, tangent never change here), so it
    # is solved once and passed in rather than re-solved every call.
    b = linsolve(S)
    inv_growth = 0.0
    iX, ilam = dX, dlam
    for _ in range(int(iters)):
        step = _bordered_block_step(linsolve, iX, ilam, S, c_dot, lam_dot, b=b)
        if step is None:
            return float("inf")  # bordering denominator degenerate.
        new_X, new_lam, _b = step
        nrm = combined_norm(new_X, new_lam)
        if not math.isfinite(nrm):
            return float("nan")
        if nrm < 1e-300:
            return float("inf")
        inv_growth = nrm
        iX, ilam = new_X / nrm, new_lam / nrm

    return sigma_max * inv_growth


# Milestone F (fold_plan.md): smallest singular values/vectors of the exact
# real-packed Jacobian, plus a reference (largest) singular scale for
# normalization. ``jacobian_min_eigenvalue`` above answers "is J singular"
# via an EIGENvalue (the physically correct test for a fold, a zero
# eigenvalue of the nonsymmetric device Jacobian). This answers the
# complementary linear-algebra question Milestone F needs: J's local
# nullity/corank and its left/right near-null vectors (w, v), via SINGULAR
# values -- ARPACK's `eigsh` needs a symmetric operator, and J itself is not
# symmetric, so this is not a drop-in replacement, it is additional
# information (SVD, not eigendecomposition).


@dataclass
class SingularTriplet:
    """Up to ``k`` smallest singular values/vectors of the real-packed
    pump Jacobian ``J``, ascending by ``sigma``.

    ``u``/``v`` are the left/right singular vectors in the SAME real-packed
    coordinates as ``pack_complex(X)`` (dimension ``2*H*n``), each
    normalized to unit Euclidean norm -- NOT the PALC corrector's own
    state-scaled metric. ``sigma_ref`` is a cheap largest-singular-value
    estimate (power iteration on ``J^T J``, no solves) for normalizing
    "small": ``sigma[0] / sigma_ref`` is dimensionless regardless of the
    problem's absolute residual/state scaling.

    ``converged=False`` means the estimator itself failed (ARPACK
    non-convergence, a singular ``J^T`` factorization, or too few positive
    eigenvalues recovered from the augmented system) -- ``sigma``/``u``/``v``
    are empty and callers must treat this as a reduced-confidence
    diagnostic, never as evidence for or against a fold.
    """

    sigma: list[float]
    u: list[np.ndarray]
    v: list[np.ndarray]
    sigma_ref: float
    converged: bool
    estimator: str


def _reference_singular_scale(M: Any, *, iters: int, seed: int) -> float:
    """Largest singular value of ``M`` via power iteration on ``M^T M``.

    No linear solves -- two sparse matvecs per iteration -- so this is cheap
    relative to the smallest-singular-value estimate below, which needs two
    fresh LU factorizations.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[1])
    nrm0 = float(np.linalg.norm(v))
    if nrm0 < 1e-300:
        return float("nan")
    v /= nrm0
    for _ in range(int(iters)):
        w = M.T @ (M @ v)
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-300:
            return 0.0
        v = w / nrm
    return float(np.linalg.norm(M @ v))


def _assembled_jacobian_matrix_and_solve(problem: Any, X: np.ndarray):
    """The exact real-packed Jacobian as an explicit sparse matrix, plus the
    production factor's own ``.solve`` when one is already available.

    Two problem types reach this: the plain ``FullPumpProblem`` test fixture
    (only ``real_coupled_jacobian`` -- a diagnostic-only assembly, same
    pattern as ``jacobian_det_signature`` above) and the production
    ``SchurReducedProblem`` (``assemble_real_coupled_fast`` -- always
    available on that class -- which both assembles the matrix as ``.M``
    AND already factors it, PARDISO/banded-backed). Reusing that ``.solve``
    instead of a second from-scratch ``spla.splu`` of the same matrix is the
    difference between this being cheap and this being a second full
    production factorization on every call.

    ``factor.M`` on the fast path is a single buffer the production
    ``FastCoupledPreconditioner`` caches on the partition and mutates IN
    PLACE (``.refactor(tangent)`` rewrites ``.M.data`` and re-factors,
    reusing the same object across calls -- correct and cheap for
    production callers, which always solve immediately). Returning that
    reference directly bit us: a caller taking two snapshots at different
    ``X`` (e.g. a finite-difference sweep) got the SAME mutable object both
    times (``Mp is Mm``), so any later comparison saw an exact-zero
    difference regardless of the true derivative -- not a physics finding,
    an aliasing bug (fold_plan.md Milestone F.5, caught when the exact AFT
    alpha disagreed with the finite-difference alpha by 21 orders of
    magnitude at mu~0.5253 on designs/ipm_2c_fixed). ``.copy()`` makes each
    call's return value an independent snapshot; ``solve`` remains the live
    bound method (single-call use only, e.g. inside
    :func:`smallest_singular_triplets`'s own synchronous ARPACK loop, never
    held across a second ``_assembled_jacobian_matrix*`` call).
    """
    tangent = problem.tangent_state(X)
    if hasattr(problem, "assemble_real_coupled_fast"):
        factor = problem.assemble_real_coupled_fast(tangent)
        return factor.M.copy(), factor.solve
    spectral = problem.spectral_tangent_state(tangent)
    return problem.real_coupled_jacobian(spectral), None


def _assembled_jacobian_matrix(problem: Any, X: np.ndarray):
    """The exact real-packed Jacobian as an explicit sparse matrix only
    (forward action, no solve) -- used where only a JVP is needed, e.g.
    the quadratic fold coefficient's finite difference."""
    M, _solve = _assembled_jacobian_matrix_and_solve(problem, X)
    return M


def smallest_singular_triplets(
    problem: Any,
    X: np.ndarray,
    *,
    k: int = 2,
    iters: int = 30,
    seed: int = 0,
) -> SingularTriplet:
    """The ``k`` smallest singular values/vectors of the pump Jacobian at ``X``.

    Standard trick: the symmetric augmented matrix ``[[0, J],[J^T, 0]]`` has
    eigenvalues ``+-sigma_i(J)``, with eigenvector ``[u_i; v_i]`` for
    ``+sigma_i`` giving exactly the left/right singular vectors. ARPACK's
    shift-invert ``eigsh`` around ``sigma=0`` finds those nearest zero
    directly, needing only:

    - forward action: two sparse matvecs (``J`` and ``J^T``, both free once
      ``J`` is assembled as an explicit matrix);
    - inverse action ``J^-1``: the production factor's own ``.solve`` when
      one is already available (``SchurReducedProblem``, PARDISO/banded-
      backed), else a fresh ``spla.splu`` of ``J`` (the plain test-fixture
      path);
    - inverse action ``J^-T``: always a fresh ``spla.splu`` of ``J^T`` --
      no backend exposes a transpose-solve, so this is the one genuinely
      new cost, and only at detected fold candidates, never per
      continuation step.

    Returns a :class:`SingularTriplet` with ``converged=False`` (empty
    ``sigma``/``u``/``v``) rather than raising on any ARPACK or
    factorization failure -- per Milestone F's instruction not to fake a
    left null vector or claim a simple fold without real evidence.
    """
    M, fast_solve = _assembled_jacobian_matrix_and_solve(problem, X)
    dim = M.shape[0]

    try:
        j_solve = fast_solve if fast_solve is not None else spla.splu(M.tocsc()).solve
        jt_factor = spla.splu(M.T.tocsc())
    except (RuntimeError, ValueError):
        return SingularTriplet([], [], [], float("nan"), False, "transpose_factor_failed")

    def op_inv(z: np.ndarray) -> np.ndarray:
        a, b = z[:dim], z[dim:]
        x = jt_factor.solve(b)
        y = j_solve(a)
        return np.concatenate([x, y])

    def op_fwd(z: np.ndarray) -> np.ndarray:
        a, b = z[:dim], z[dim:]
        return np.concatenate([M @ b, M.T @ a])

    A = spla.LinearOperator((2 * dim, 2 * dim), matvec=op_fwd, dtype=np.float64)
    OPinv = spla.LinearOperator((2 * dim, 2 * dim), matvec=op_inv, dtype=np.float64)
    n_want = max(2 * int(k), 4)
    if n_want >= 2 * dim:
        n_want = 2 * dim - 1  # ARPACK requires k < N

    try:
        eigvals, eigvecs = spla.eigsh(
            A, k=n_want, sigma=0.0, which="LM", OPinv=OPinv,
            maxiter=max(200, int(iters) * 10), tol=0.0,
        )
    except (spla.ArpackNoConvergence, RuntimeError, ValueError, FloatingPointError):
        return SingularTriplet([], [], [], float("nan"), False, "augmented_eigsh_failed")

    positive = sorted((i for i in range(len(eigvals)) if eigvals[i] > 0), key=lambda i: eigvals[i])
    if len(positive) < k:
        return SingularTriplet([], [], [], float("nan"), False, "insufficient_positive_eigenvalues")

    sigmas: list[float] = []
    us: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    for idx in positive[:k]:
        vec = eigvecs[:, idx]
        u_part, v_part = vec[:dim], vec[dim:]
        u_norm = float(np.linalg.norm(u_part))
        v_norm = float(np.linalg.norm(v_part))
        if u_norm < 1e-12 or v_norm < 1e-12:
            return SingularTriplet([], [], [], float("nan"), False, "degenerate_augmented_eigenvector")
        sigmas.append(float(eigvals[idx]))
        us.append(u_part / u_norm)
        vs.append(v_part / v_norm)

    sigma_ref = _reference_singular_scale(M, iters=iters, seed=seed)
    return SingularTriplet(sigmas, us, vs, sigma_ref, True, "augmented_shift_invert")


def singular_triplet_residuals(
    problem: Any, X: np.ndarray, triplet: SingularTriplet, *, index: int = 0,
) -> dict[str, float | None]:
    """Normalized residuals of one singular pair: ``r_right=|Jv-sigma w|/sigma_ref``,
    ``r_left=|J^Tw-sigma v|/sigma_ref``.

    Milestone F step (user request, 2026-08-08): before trusting
    ``tau=w^T F_mu`` to rule out a branch point, quantify how good ``(w,v)``
    actually are. Does not change the estimator -- purely a reported
    diagnostic, cheap relative to :func:`smallest_singular_triplets` itself
    (one matrix assembly, two matvecs, no new factorization).
    """
    if not triplet.converged or index >= len(triplet.sigma):
        return {"r_right": None, "r_left": None}
    sigma = triplet.sigma[index]
    w = triplet.u[index]
    v = triplet.v[index]
    M = _assembled_jacobian_matrix(problem, X)
    r_right = float(np.linalg.norm(M @ v - sigma * w))
    r_left = float(np.linalg.norm(M.T @ w - sigma * v))
    scale = triplet.sigma_ref if triplet.sigma_ref and triplet.sigma_ref > 1e-300 else 1.0
    return {"r_right": r_right / scale, "r_left": r_left / scale}
