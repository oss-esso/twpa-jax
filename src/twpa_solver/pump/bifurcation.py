"""Milestone F (fold_plan.md): Jacobian-level bifurcation classification.

Milestone E (``solver.py``) answers a purely kinematic question about a
traced curve: did ``t_mu`` change sign? That is a fact about the trace, not
about the underlying harmonic-balance equations ``F(X,mu)=0``. This module
answers the algebraic question: does an actual singularity of ``J=F_X``
exist at a ``FoldCandidate``, what is its local nullity, and does it satisfy
the two generic simple-fold conditions (parameter transversality and the
quadratic fold coefficient)?

Consumes the post-hoc ``FoldCandidate``/``ContinuationBranch`` objects
Milestone E already produces. Does NOT modify ``solve_arclength``, does NOT
stop continuation, and is NOT wired into ``run_gain_map.py`` -- until this
module's classifications have been validated against real folds, nothing
here may gate a production map cell.

Promotion hierarchy this module completes (started in ``solver.py``):

    tangent sign flip -> fold candidate -> [THIS MODULE] -> classified event
        SIMPLE_FOLD | POSSIBLE_BRANCH_POINT | POSSIBLE_HIGHER_DEGENERACY
        | NONSINGULAR_TANGENT_FLIP | SINGULAR_UNCLASSIFIED | DIAGNOSTIC_FAILURE
        | QUADRATICALLY_DEGENERATE_SINGULARITY (F.5, alpha exactly ~0)

F.5 (fold_plan.md, 2026-08-08) replaced the finite-difference quadratic
coefficient with an EXACT one: ``exact_quadratic_fold_coefficient``/
``exact_cubic_fold_coefficient`` differentiate the same closed-form AFT
nonlinearity ``jvp_coeffs_with_tangent`` already uses, analytically, with
no ``eps`` anywhere. ``quadratic_fold_coefficient_fd`` (the original
finite-difference version) is kept only as a validation cross-check.

Every tolerance below is a config default, not a locked production
criterion -- report the raw normalized diagnostics and calibrate later
against real controls (fold_plan.md Milestone F: "we need to see their
actual scale on this problem before locking production criteria").
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from twpa_solver.pump.problem import pack_complex, unpack_complex
from twpa_solver.pump.singularity import (
    SingularTriplet,
    _assembled_jacobian_matrix,
    bordered_conditioning,
    singular_triplet_residuals,
    smallest_singular_triplets,
)
from twpa_solver.pump.solver import (
    ContinuationBranch,
    FoldCandidate,
    HarmonicNewtonKrylovSolver,
    _real_dot,
    find_fold_candidates,
)

FOLD_CLASS_SIMPLE_FOLD = "SIMPLE_FOLD"
FOLD_CLASS_POSSIBLE_BRANCH_POINT = "POSSIBLE_BRANCH_POINT"
FOLD_CLASS_POSSIBLE_HIGHER_DEGENERACY = "POSSIBLE_HIGHER_DEGENERACY"
FOLD_CLASS_NONSINGULAR_TANGENT_FLIP = "NONSINGULAR_TANGENT_FLIP"
FOLD_CLASS_SINGULAR_UNCLASSIFIED = "SINGULAR_UNCLASSIFIED"
FOLD_CLASS_DIAGNOSTIC_FAILURE = "DIAGNOSTIC_FAILURE"
FOLD_CLASS_QUADRATICALLY_DEGENERATE_SINGULARITY = "QUADRATICALLY_DEGENERATE_SINGULARITY"
"""alpha (exact) ~ 0 to machine/projected precision, but the singular
triplet itself is clean (well-conditioned residuals, tau clearly nonzero)
and the cubic coefficient gamma is clearly nonzero. Deliberately NOT named
anything with "cusp" in it -- cusp classification is a two-parameter
question (fold_plan.md Milestone F.5); this only reports the LOCAL
one-parameter evidence that would be consistent with one."""

_HALF_WINDOW_DEFAULT = 2


@dataclass
class BifurcationTolerances:
    """Provisional classification thresholds.

    All dimensionless (ratios of normalized quantities), so a single default
    is at least unit-consistent across problems of different absolute scale
    -- but none of these values are locked; report the raw diagnostics
    alongside the label they produced.
    """

    sigma1_hat_max: float = 1e-2
    """sigma[0]/sigma_ref below this is "small" (near-singular)."""

    sigma_gap_max: float = 1e-1
    """sigma[0]/sigma[1] below this is "well separated" (corank likely 1)."""

    tau_min: float = 1e-3
    """Transversality tau=|w.S|/|S| at/above this is "clearly nonzero"."""

    alpha_hat_min: float = 1e-6
    """alpha_hat = alpha_exact/sigma_ref (dimensionless, same normalization
    as sigma1_hat) at/above this is "clearly nonzero" -- the primary alpha
    gate now that alpha is computed exactly (F.5), not from a finite
    difference. Also used for gamma_hat (the cubic coefficient), which
    shares the same natural scale."""

    residual_max: float = 1e-6
    """Normalized singular-vector residuals (r_right, r_left from
    ``singular_triplet_residuals``) must be below this, AND comfortably
    below the measured tau, before a small alpha is promoted to
    QUADRATICALLY_DEGENERATE_SINGULARITY rather than left SINGULAR_UNCLASSIFIED."""

    alpha_snr_min: float = 3.0
    """Ratio of the plateaued alpha's mean magnitude to its own spread
    across the eps sweep, required to call the FINITE-DIFFERENCE alpha
    (validation/cross-check path only, see ``quadratic_fold_coefficient_fd``)
    "clearly nonzero" -- compares the signal to its own measured noise floor
    instead of an absolute magic number, since alpha carries no natural unit."""

    alpha_plateau_rel_tol: float = 0.3
    """Relative spread (std/mean|.|) within the middle eps window below
    which the finite-difference sweep is considered to have reached a
    plateau at all (validation path only)."""

    alpha_eps_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    """Geometric sequence of eps multipliers for the finite-difference
    cross-check of the quadratic coefficient (validation path only)."""

    eps_base: float = 1e-6
    """Base RELATIVE finite-difference step fraction for the validation
    cross-check, scaled by ``alpha_eps_scales`` and by
    ``max(|X|, state_scale)`` (the perturbation direction v is already
    unit-norm, so the absolute step is
    ``eps_base * scale * max(|X|, state_scale)``)."""

    singular_triplet_iters: int = 30
    singular_triplet_seed: int = 0


@dataclass
class QuadraticFoldResult:
    alpha: float | None
    resolved: bool
    values: list[float] = field(default_factory=list)
    eps_values: list[float] = field(default_factory=list)
    snr: float | None = None


@dataclass
class TransversalityResult:
    beta: float
    tau: float
    nonzero: bool


@dataclass
class SingularityProfilePoint:
    index: int
    s: float
    mu: float
    t_mu: float
    sigma1: float | None
    sigma2: float | None
    sigma_ref: float | None
    sigma1_hat: float | None
    r_gap: float | None
    estimator: str


@dataclass
class SingularRegion:
    """One or more adjacent fold candidates sharing a single sigma1(s) trough."""

    candidate_ids: list[int]
    branch_point_index: int  # index into branch.points of the sigma1 minimum
    window: tuple[int, int]  # (lo, hi) point-index span covering the group


def transversality(
    problem: Any,
    w_packed: np.ndarray,
    *,
    tolerances: BifurcationTolerances | None = None,
) -> TransversalityResult:
    """beta = w^T F_mu = -w^T S_ref; tau = |beta| / (|w| |S_ref|).

    ``w_packed`` must already be unit-norm (as :func:`smallest_singular_triplets`
    returns it), so ``tau = |beta| / |S_ref|``.
    """
    tol = tolerances or BifurcationTolerances()
    S_packed = pack_complex(problem.source_coeffs(1.0))
    s_norm = float(np.linalg.norm(S_packed))
    if s_norm < 1e-300:
        return TransversalityResult(0.0, 0.0, False)
    beta = -float(np.dot(w_packed, S_packed))
    tau = abs(beta) / s_norm
    return TransversalityResult(beta, tau, tau >= tol.tau_min)


def _bphi_bphit(problem: Any) -> tuple[Any, Any]:
    """(Bphi, BphiT) for either problem type -- ``SchurReducedProblem``
    exposes the retained-node-reduced pair as ``Bphi_r``/``BphiT_r``, the
    plain ``FullPumpProblem`` exposes ``Bphi``/``BphiT`` directly."""
    if hasattr(problem, "Bphi_r"):
        return problem.Bphi_r, problem.BphiT_r
    return problem.Bphi, problem.BphiT


def _branch_flux_time(problem: Any, X: np.ndarray) -> np.ndarray:
    _, BphiT = _bphi_bphit(problem)
    x_t = problem.grid.synthesize(X)
    return (BphiT @ x_t.T).T


def _psi_total_time(problem: Any, X: np.ndarray) -> np.ndarray:
    return _branch_flux_time(problem, X) + problem.dc_branch_flux[None, :]


def _ic_phi0(problem: Any) -> tuple[np.ndarray, float]:
    """(critical_current, phi0) for either branch-law class in use here.

    The toy test fixture (``JosephsonBranchArray``) exposes ``.Ic``; the
    production branch (``JosephsonBranchLaw`` from ``make_branch_law``,
    ``critical_current`` field) is the SAME ``Ic*sin(psi/phi0)`` formula
    under a different attribute name -- both are covered by this lookup.
    ``EffectiveSnailBranchLaw``/``CompositeBranchLaw`` (other designs, e.g.
    the Le Gal benchmark) are a genuinely different nonlinearity and are
    NOT covered by ``d2n_coeffs``/``d3n_coeffs``'s hardcoded sin/cos
    formula -- this raises rather than silently assuming the wrong branch
    law.
    """
    branch = problem.branch
    ic = getattr(branch, "Ic", None)
    if ic is None:
        ic = getattr(branch, "critical_current", None)
    if ic is None:
        raise AttributeError(
            f"d2n_coeffs/d3n_coeffs assume a plain Josephson sin(psi/phi0) branch law "
            f"(Ic or critical_current + phi0); got {type(branch).__name__}, which has neither. "
            "Not supported by the exact AFT F.5 formulas -- use quadratic_fold_coefficient_fd instead."
        )
    return ic, branch.phi0


def d2n_coeffs(problem: Any, X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Exact ``D^2N[X][v,v]``, the analytic second directional derivative of
    the AFT nonlinear current (fold_plan.md Milestone F.5).

    ``N(X) = Bphi[Ic sin(psi/phi0)]``, ``psi(X+eps v)(t) = psi(X)(t) +
    eps*dpsi_v(t)`` (linear in ``v``), so ``d^2/deps^2`` at ``eps=0`` is
    ``Bphi[-(Ic/phi0^2) sin(psi/phi0) (dpsi_v)^2]`` -- derived directly by
    differentiating the same closed form ``jvp_coeffs_with_tangent`` already
    uses for the first derivative, not a finite difference. ``F_XX[v,v] ==
    D2N[X][v,v]`` exactly, since the linear circuit part of ``F`` does not
    depend on ``X`` at all (zero second derivative). Never builds a dense
    Hessian/tensor -- only the same AFT projection (synthesize -> branch
    incidence -> nonlinearity -> incidence -> project) the residual and JVP
    already use, evaluated at the perturbation direction ``V`` twice.
    """
    Bphi, _ = _bphi_bphit(problem)
    Ic, phi0 = _ic_phi0(problem)
    psi_total_t = _psi_total_time(problem, X)
    dpsi_v_t = _branch_flux_time(problem, V)
    d2_t = -(Ic[None, :] / phi0 ** 2) * np.sin(psi_total_t / phi0) * dpsi_v_t ** 2
    dn_t = (Bphi @ d2_t.T).T
    return problem.grid.project_positive(dn_t)


def d3n_coeffs(problem: Any, X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Exact ``D^3N[X][v,v,v]``, the analytic third directional derivative,
    one more differentiation of :func:`d2n_coeffs`'s closed form."""
    Bphi, _ = _bphi_bphit(problem)
    Ic, phi0 = _ic_phi0(problem)
    psi_total_t = _psi_total_time(problem, X)
    dpsi_v_t = _branch_flux_time(problem, V)
    d3_t = -(Ic[None, :] / phi0 ** 3) * np.cos(psi_total_t / phi0) * dpsi_v_t ** 3
    dn_t = (Bphi @ d3_t.T).T
    return problem.grid.project_positive(dn_t)


def exact_quadratic_fold_coefficient(
    problem: Any, X: np.ndarray, v_packed: np.ndarray, w_packed: np.ndarray,
) -> float:
    """alpha = w^T F_XX[v,v], computed exactly via :func:`d2n_coeffs` --
    the primary alpha path (replaces the finite difference entirely, no
    eps anywhere)."""
    V = unpack_complex(v_packed, X.shape)
    D2N = d2n_coeffs(problem, X, V)
    return float(np.dot(w_packed, pack_complex(D2N)))


def exact_cubic_fold_coefficient(
    problem: Any, X: np.ndarray, v_packed: np.ndarray, w_packed: np.ndarray,
) -> float:
    """gamma = w^T F_XXX[v,v,v], computed exactly via :func:`d3n_coeffs`."""
    V = unpack_complex(v_packed, X.shape)
    D3N = d3n_coeffs(problem, X, V)
    return float(np.dot(w_packed, pack_complex(D3N)))


def quadratic_fold_coefficient_fd(
    problem: Any,
    X: np.ndarray,
    v_packed: np.ndarray,
    w_packed: np.ndarray,
    *,
    state_scale: float = 1.0,
    tolerances: BifurcationTolerances | None = None,
) -> QuadraticFoldResult:
    """Finite-difference alpha -- validation/cross-check tool only.

    ``exact_quadratic_fold_coefficient`` is the primary path used by
    ``classify_region``; this centered-difference version (originally the
    only implementation) is kept to cross-validate the exact AFT formula
    against an independent numerical estimate on toy/nonsingular real
    states, per fold_plan.md Milestone F.5's explicit validation request.
    Never builds ``F_XX`` as a tensor -- only assembles ``J`` (a matrix) at
    two nearby points along ``v`` per eps value. Sweeps a geometric eps
    sequence; requires the middle values to plateau (relative spread below
    ``tolerances.alpha_plateau_rel_tol``) before reporting a resolved alpha.
    """
    tol = tolerances or BifurcationTolerances()
    shape = X.shape
    V = unpack_complex(v_packed, shape)
    x_norm = float(np.linalg.norm(pack_complex(X)))
    # v_packed is already unit-norm, so the absolute step must scale WITH
    # the state's own magnitude, not against it -- eps_base is a relative
    # step fraction (default 1e-6, standard central-difference range), and
    # the perturbation eps*V is that fraction of max(|X|, state_scale). An
    # earlier version divided by this scale instead of multiplying: on this
    # device X is node flux at ~1e-13 Wb, so dividing produced eps ~1e6-1e8
    # -- so large that X +- eps*V is dominated entirely by the eps*V term,
    # and since the branch nonlinearity is cos(psi/phi0) (exactly even),
    # J(X+eps V) and J(X-eps V) became numerically IDENTICAL and every
    # alpha_value silently came out exactly 0.0 (not noise -- a real bug,
    # caught by exactly this symptom on designs/ipm_2c_fixed).
    scale_ref = max(x_norm, state_scale, 1e-300)

    values: list[float] = []
    eps_values: list[float] = []
    for scale in tol.alpha_eps_scales:
        eps = tol.eps_base * scale * scale_ref
        if not math.isfinite(eps) or eps <= 0.0:
            continue
        M_plus = _assembled_jacobian_matrix(problem, X + eps * V)
        M_minus = _assembled_jacobian_matrix(problem, X - eps * V)
        Jv_plus = M_plus @ v_packed
        Jv_minus = M_minus @ v_packed
        q = (Jv_plus - Jv_minus) / (2.0 * eps)
        values.append(float(np.dot(w_packed, q)))
        eps_values.append(eps)

    if len(values) < 3:
        return QuadraticFoldResult(None, False, values, eps_values, None)

    mid = len(values) // 2
    window = values[mid - 1:mid + 2]
    mean_abs = float(np.mean(np.abs(window)))
    spread = float(np.std(window))
    resolved = mean_abs > 1e-300 and (spread / mean_abs) < tol.alpha_plateau_rel_tol
    if not resolved:
        return QuadraticFoldResult(None, False, values, eps_values, None)
    alpha = float(np.mean(window))
    snr = (mean_abs / spread) if spread > 1e-300 else float("inf")
    return QuadraticFoldResult(alpha, True, values, eps_values, snr)


def _tangent_at(
    solver: HarmonicNewtonKrylovSolver,
    problem: Any,
    X: np.ndarray,
    S: np.ndarray,
    state_scale: float,
) -> tuple[np.ndarray, float] | None:
    """Best-effort unit tangent (Xdot, lam_dot) at a stored branch point.

    Reuses the same tangent-solve construction ``solve_arclength``'s own
    corrector and ``_refine_fold`` already perform -- not the PALC state
    itself (branch points do not persist Xdot), so this is an approximation
    good enough for the ``bordered_conditioning`` comparison diagnostic,
    not for anything that gates a classification.
    """
    try:
        lin = solver._linear_solver(problem, X, deadline_s=0.0, t0=time.perf_counter())
        Xdot = lin(S)
    except (RuntimeError, FloatingPointError, ValueError, OverflowError):
        return None
    lam_dot = 1.0
    nrm = math.sqrt(_real_dot(Xdot, Xdot) / (state_scale * state_scale) + lam_dot * lam_dot)
    if not math.isfinite(nrm) or nrm <= 1e-300:
        return None
    return Xdot / nrm, lam_dot / nrm


def _bordered_conditioning_best_effort(
    solver: HarmonicNewtonKrylovSolver,
    problem: Any,
    X: np.ndarray,
    t_mu: float,
    state_scale: float,
) -> float | None:
    """``bordered_conditioning`` (singularity.py), for comparison only.

    Never gates a classification (fold_plan.md: "Keep the existing ...
    bordered-conditioning diagnostics for comparison"); returns ``None`` on
    any failure rather than raising.
    """
    S = problem.source_coeffs(1.0)
    tangent = _tangent_at(solver, problem, X, S, state_scale)
    if tangent is None:
        return None
    Xdot, lam_dot = tangent
    if lam_dot * t_mu < 0.0:
        Xdot, lam_dot = -Xdot, -lam_dot
    try:
        value = bordered_conditioning(problem, X, 0.0, Xdot, lam_dot, state_scale)
    except (RuntimeError, FloatingPointError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _candidate_window(
    branch: ContinuationBranch, candidate: FoldCandidate, half_window: int,
) -> tuple[int, int]:
    n = len(branch.points)
    lo = max(0, candidate.index - 1 - half_window)
    hi = min(n - 1, candidate.index + half_window)
    return lo, hi


def _build_singularity_profile(
    problem: Any,
    branch: ContinuationBranch,
    point_indices: list[int],
    tolerances: BifurcationTolerances,
) -> tuple[dict[int, SingularityProfilePoint], dict[int, SingularTriplet]]:
    """Compute the cheap sigma1/sigma2 profile at every requested point.

    ``smallest_singular_triplets`` is called once per point (this is the
    genuinely expensive part -- two fresh LU factorizations each); the
    returned :class:`SingularTriplet` objects are kept so the eventual
    argmin point's ``w``/``v`` do not need a second, redundant call.
    """
    pts = branch.points
    profiles: dict[int, SingularityProfilePoint] = {}
    triplets: dict[int, SingularTriplet] = {}
    for j in point_indices:
        pt = pts[j]
        triplet = smallest_singular_triplets(
            problem, pt.X, k=2,
            iters=tolerances.singular_triplet_iters,
            seed=tolerances.singular_triplet_seed,
        )
        triplets[j] = triplet
        sigma1 = triplet.sigma[0] if triplet.converged and triplet.sigma else None
        sigma2 = triplet.sigma[1] if triplet.converged and len(triplet.sigma) > 1 else None
        sigma_ref = triplet.sigma_ref if triplet.converged else None
        sigma1_hat = (
            sigma1 / sigma_ref
            if sigma1 is not None and sigma_ref is not None and sigma_ref > 1e-300
            else None
        )
        r_gap = sigma1 / sigma2 if sigma1 is not None and sigma2 and sigma2 > 1e-300 else None
        profiles[j] = SingularityProfilePoint(
            index=j, s=pt.s, mu=pt.mu, t_mu=pt.t_mu,
            sigma1=sigma1, sigma2=sigma2, sigma_ref=sigma_ref,
            sigma1_hat=sigma1_hat, r_gap=r_gap, estimator=triplet.estimator,
        )
    return profiles, triplets


def cluster_candidates_into_regions(
    branch: ContinuationBranch,
    candidates: list[FoldCandidate],
    profiles: dict[int, SingularityProfilePoint],
    *,
    tolerances: BifurcationTolerances | None = None,
    half_window: int = _HALF_WINDOW_DEFAULT,
) -> list[SingularRegion]:
    """Group adjacent candidates sharing one local minimum of sigma1(s).

    fold_plan.md Milestone F step 7: do not merge candidates just because
    they are close in ``s``. Two adjacent candidates stay in the SAME region
    unless some profiled point strictly between their stencil windows has
    ``sigma1_hat >= tolerances.sigma1_hat_max`` -- i.e. the profile visibly
    recovers to "not singular" before the next candidate. This reuses the
    same threshold ``classify_region`` uses for "is this point singular at
    all", rather than introducing a second, unrelated magic number, and it
    naturally merges a razor-thin candidate pair with no profiled points
    between their windows (nothing to prove they are separate).
    """
    tol = tolerances or BifurcationTolerances()
    if not candidates:
        return []

    windows = [_candidate_window(branch, c, half_window) for c in candidates]
    groups: list[list[int]] = [[0]]
    for i in range(1, len(candidates)):
        prev_hi = windows[i - 1][1]
        cur_lo = windows[i][0]
        between = range(prev_hi + 1, cur_lo)
        recovers = any(
            j in profiles and profiles[j].sigma1_hat is not None
            and profiles[j].sigma1_hat >= tol.sigma1_hat_max
            for j in between
        )
        if recovers:
            groups.append([i])
        else:
            groups[-1].append(i)

    regions: list[SingularRegion] = []
    for group in groups:
        lo = min(windows[i][0] for i in group)
        hi = max(windows[i][1] for i in group)
        candidates_with_sigma = [
            j for j in range(lo, hi + 1) if j in profiles and profiles[j].sigma1 is not None
        ]
        best_idx = (
            min(candidates_with_sigma, key=lambda j: profiles[j].sigma1)
            if candidates_with_sigma else lo
        )
        regions.append(SingularRegion(candidate_ids=list(group), branch_point_index=best_idx, window=(lo, hi)))
    return regions


def classify_region(
    solver: HarmonicNewtonKrylovSolver,
    problem: Any,
    branch: ContinuationBranch,
    region: SingularRegion,
    triplet: SingularTriplet,
    *,
    tolerances: BifurcationTolerances | None = None,
) -> dict:
    """Classify one :class:`SingularRegion` into one of the seven F outcomes.

    Evaluates diagnostics in increasing cost order, short-circuiting as soon
    as one test settles the classification (fold_plan.md step 8's
    conservative semantics, extended by F.5): sigma1_hat -> sigma_gap ->
    tau -> alpha (exact) -> [gamma + residuals, only if alpha ~ 0]. Every
    branch records its raw diagnostic values and a plain-language reason,
    even when it stops early -- later stages are simply left ``None``.
    """
    tol = tolerances or BifurcationTolerances()
    pt = branch.points[region.branch_point_index]
    row: dict[str, Any] = {
        "candidate_ids": list(region.candidate_ids),
        "branch_point_index": region.branch_point_index,
        "mu": pt.mu, "s": pt.s, "t_mu": pt.t_mu,
        "estimator": triplet.estimator, "triplet_converged": triplet.converged,
        "sigma1": None, "sigma2": None, "sigma_ref": None, "sigma1_hat": None, "sigma_gap": None,
        "beta": None, "tau": None,
        "alpha": None, "alpha_hat": None,
        "gamma": None, "gamma_hat": None,
        "singular_residual_right": None, "singular_residual_left": None,
        "bordered_conditioning": None,
    }

    if not triplet.converged or not triplet.sigma:
        row["class"] = FOLD_CLASS_DIAGNOSTIC_FAILURE
        row["reason"] = f"singular triplet estimator failed ({triplet.estimator})"
        return row

    sigma1 = triplet.sigma[0]
    sigma2 = triplet.sigma[1] if len(triplet.sigma) > 1 else None
    sigma_ref = triplet.sigma_ref
    sigma1_hat = sigma1 / sigma_ref if sigma_ref and sigma_ref > 1e-300 else None
    sigma_gap = (sigma1 / sigma2) if sigma2 and sigma2 > 1e-300 else None
    row.update(sigma1=sigma1, sigma2=sigma2, sigma_ref=sigma_ref, sigma1_hat=sigma1_hat, sigma_gap=sigma_gap)

    if sigma1_hat is None or sigma1_hat >= tol.sigma1_hat_max:
        row["class"] = FOLD_CLASS_NONSINGULAR_TANGENT_FLIP
        row["reason"] = f"sigma1_hat={sigma1_hat} not small (threshold {tol.sigma1_hat_max:.0e})"
        return row

    if sigma_gap is None or sigma_gap >= tol.sigma_gap_max:
        row["class"] = FOLD_CLASS_POSSIBLE_HIGHER_DEGENERACY
        row["reason"] = f"sigma1/sigma2={sigma_gap} not well separated (corank>=2 suspected)"
        return row

    w_packed, v_packed = triplet.u[0], triplet.v[0]
    trans = transversality(problem, w_packed, tolerances=tol)
    row.update(beta=trans.beta, tau=trans.tau)

    resid = singular_triplet_residuals(problem, pt.X, triplet, index=0)
    row.update(singular_residual_right=resid["r_right"], singular_residual_left=resid["r_left"])

    if not trans.nonzero:
        row["class"] = FOLD_CLASS_POSSIBLE_BRANCH_POINT
        row["reason"] = f"tau={trans.tau:.3e} near zero (F_mu in Range(J))"
        return row

    state_scale = branch.info.get("state_scale") or 1.0
    alpha_exact = exact_quadratic_fold_coefficient(problem, pt.X, v_packed, w_packed)
    alpha_hat = alpha_exact / sigma_ref if sigma_ref and sigma_ref > 1e-300 else None
    row.update(alpha=alpha_exact, alpha_hat=alpha_hat)
    row["bordered_conditioning"] = _bordered_conditioning_best_effort(
        solver, problem, pt.X, pt.t_mu, state_scale,
    )

    alpha_nonzero = alpha_hat is not None and abs(alpha_hat) >= tol.alpha_hat_min
    if alpha_nonzero:
        row["class"] = FOLD_CLASS_SIMPLE_FOLD
        row["reason"] = "sigma1 small & separated, tau and alpha (exact AFT) both clearly nonzero"
        return row

    # alpha ~ 0 (to the tol.alpha_hat_min floor): the quadratic fold
    # condition fails. Per fold_plan.md F.5, check the cubic coefficient
    # AND the singular-vector residuals before calling this anything more
    # specific than "unclassified" -- a small alpha next to noisy (w,v) is
    # not evidence of anything.
    gamma_exact = exact_cubic_fold_coefficient(problem, pt.X, v_packed, w_packed)
    gamma_hat = gamma_exact / sigma_ref if sigma_ref and sigma_ref > 1e-300 else None
    row.update(gamma=gamma_exact, gamma_hat=gamma_hat)

    residuals_clean = (
        resid["r_right"] is not None and resid["r_left"] is not None
        and resid["r_right"] < tol.residual_max and resid["r_left"] < tol.residual_max
    )
    gamma_nonzero = gamma_hat is not None and abs(gamma_hat) >= tol.alpha_hat_min

    if residuals_clean and gamma_nonzero:
        row["class"] = FOLD_CLASS_QUADRATICALLY_DEGENERATE_SINGULARITY
        row["reason"] = (
            "alpha (exact) ~ 0 to the alpha_hat_min floor, singular-vector residuals clean, "
            "cubic coefficient gamma clearly nonzero -- NOT a cusp claim (that needs the "
            "two-parameter continuation), only that the quadratic fold condition genuinely fails here"
        )
        return row

    row["class"] = FOLD_CLASS_SINGULAR_UNCLASSIFIED
    if not residuals_clean:
        row["reason"] = (
            f"alpha ~ 0 but singular-vector residuals too large to trust it "
            f"(r_right={resid['r_right']}, r_left={resid['r_left']})"
        )
    else:
        row["reason"] = f"alpha ~ 0 and cubic coefficient also not clearly nonzero (gamma_hat={gamma_hat})"
    return row


def _healthy_reference_indices(
    branch: ContinuationBranch, window: tuple[int, int], *, offset: int = 15,
) -> list[int]:
    """Up to two branch-point indices well outside ``window`` (one before,
    one after), for the local-contrast metric below. Clipped to the branch's
    own extent; returns fewer than two near either end of a short branch."""
    n = len(branch.points)
    lo, hi = window
    out = []
    before = lo - offset
    after = hi + offset
    if before >= 0:
        out.append(before)
    if after < n:
        out.append(after)
    return out


def _local_sigma1_contrast(
    problem: Any,
    branch: ContinuationBranch,
    window: tuple[int, int],
    sigma1_candidate: float | None,
    healthy_cache: dict[int, float | None],
    tolerances: BifurcationTolerances,
) -> dict[str, Any]:
    """C_sigma = sigma1(candidate) / median(sigma1 at nearby healthy points).

    Reported for interpretation only -- fold_plan.md F.5 explicitly asked
    NOT to recalibrate ``sigma1_hat_max`` from one device's data (risks
    overfitting the classifier to designs/ipm_2c_fixed); this instead
    quantifies "how much more singular than its own local background" a
    candidate is, without pretending there is a universal absolute
    threshold. ``healthy_cache`` is shared across regions in one
    ``classify_fold_candidates`` call so two regions sharing a reference
    point only pay for it once.
    """
    if sigma1_candidate is None:
        return {"sigma1_local_contrast": None, "sigma1_healthy_reference": []}
    indices = _healthy_reference_indices(branch, window)
    values: list[float] = []
    for j in indices:
        if j not in healthy_cache:
            t = smallest_singular_triplets(
                problem, branch.points[j].X, k=1,
                iters=tolerances.singular_triplet_iters, seed=tolerances.singular_triplet_seed,
            )
            healthy_cache[j] = t.sigma[0] if t.converged and t.sigma else None
        if healthy_cache[j] is not None:
            values.append(healthy_cache[j])
    if not values:
        return {"sigma1_local_contrast": None, "sigma1_healthy_reference": []}
    contrast = sigma1_candidate / float(np.median(values))
    return {"sigma1_local_contrast": contrast, "sigma1_healthy_reference": values}


def classify_fold_candidates(
    solver: HarmonicNewtonKrylovSolver,
    problem: Any,
    branch: ContinuationBranch,
    candidates: list[FoldCandidate] | None = None,
    *,
    tolerances: BifurcationTolerances | None = None,
    half_window: int = _HALF_WINDOW_DEFAULT,
) -> list[dict]:
    """Milestone F: classify every fold candidate (or candidate cluster) on ``branch``.

    Event-driven: if there are no candidates, this returns ``[]`` without
    calling :func:`smallest_singular_triplets` even once (fold_plan.md step
    6: "this tests that F remains event-driven rather than turning
    continuation into an SVD at every point"). Never modifies ``branch``,
    never touches ``solve_arclength``, never stops continuation -- purely a
    post-hoc report.
    """
    tol = tolerances or BifurcationTolerances()
    cands = candidates if candidates is not None else find_fold_candidates(branch)
    if not cands:
        return []

    windows = [_candidate_window(branch, c, half_window) for c in cands]
    point_indices = sorted({j for lo, hi in windows for j in range(lo, hi + 1)})
    profiles, triplets = _build_singularity_profile(problem, branch, point_indices, tol)

    regions = cluster_candidates_into_regions(
        branch, cands, profiles, tolerances=tol, half_window=half_window,
    )

    healthy_cache: dict[int, float | None] = {}
    rows = []
    for region in regions:
        triplet = triplets.get(region.branch_point_index)
        if triplet is None:
            triplet = SingularTriplet([], [], [], float("nan"), False, "argmin_point_not_profiled")
        row = classify_region(solver, problem, branch, region, triplet, tolerances=tol)
        row.update(_local_sigma1_contrast(
            problem, branch, region.window, row.get("sigma1"), healthy_cache, tol,
        ))
        lo, hi = region.window
        row["profile"] = [
            {
                "index": j, "s": profiles[j].s, "mu": profiles[j].mu, "t_mu": profiles[j].t_mu,
                "sigma1": profiles[j].sigma1, "sigma2": profiles[j].sigma2,
                "sigma1_hat": profiles[j].sigma1_hat, "estimator": profiles[j].estimator,
            }
            for j in range(lo, hi + 1) if j in profiles
        ]
        rows.append(row)
    return rows
