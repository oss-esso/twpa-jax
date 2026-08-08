"""Tests for Milestone F (``twpa_solver.pump.bifurcation``): the Jacobian-
level bifurcation classifier built on top of Milestone E's ``FoldCandidate``
objects.

Two layers are tested separately, matching how the module is actually used:

- The six-way ``classify_region`` decision tree is pure control flow once
  given a :class:`SingularTriplet` and (for the tau/alpha branches)
  ``transversality``/``quadratic_fold_coefficient`` results -- tested with
  hand-built stub triplets and monkeypatched sub-diagnostics, no real linear
  algebra needed, one test per branch of the tree.
- ``cluster_candidates_into_regions`` and the event-driven guard in
  ``classify_fold_candidates`` are tested with hand-built branches/profiles.
- ``smallest_singular_triplets`` itself (the real linear algebra) is already
  validated against a genuine fold in ``test_singularity.py``; this file
  does not re-derive that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from twpa_solver.pump import bifurcation  # noqa: E402
from twpa_solver.pump.bifurcation import (  # noqa: E402
    FOLD_CLASS_DIAGNOSTIC_FAILURE,
    FOLD_CLASS_NONSINGULAR_TANGENT_FLIP,
    FOLD_CLASS_POSSIBLE_BRANCH_POINT,
    FOLD_CLASS_POSSIBLE_HIGHER_DEGENERACY,
    FOLD_CLASS_QUADRATICALLY_DEGENERATE_SINGULARITY,
    FOLD_CLASS_SIMPLE_FOLD,
    FOLD_CLASS_SINGULAR_UNCLASSIFIED,
    SingularityProfilePoint,
    SingularRegion,
    TransversalityResult,
    classify_fold_candidates,
    classify_region,
    cluster_candidates_into_regions,
)
from twpa_solver.pump.singularity import SingularTriplet  # noqa: E402
from twpa_solver.pump.solver import BranchPoint, ContinuationBranch, FoldCandidate  # noqa: E402


def _stub_branch_point(mu: float, t_mu: float = 1.0) -> BranchPoint:
    return BranchPoint(s=mu, mu=mu, X=np.zeros((1, 1), dtype=complex), t_mu=t_mu, step_size=0.1)


def _stub_branch(n: int = 3) -> ContinuationBranch:
    return ContinuationBranch(
        i_ref=1.0, points=[_stub_branch_point(float(i)) for i in range(n)], info={},
    )


def _stub_region(index: int = 0) -> SingularRegion:
    return SingularRegion(candidate_ids=[0], branch_point_index=index, window=(index, index))


def _stub_triplet(sigma: tuple[float, float] = (1e-8, 1.0)) -> SingularTriplet:
    # sigma_ref=1.0 so sigma1_hat == sigma[0] directly -- keeps test math obvious.
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    return SingularTriplet(list(sigma), [u, u], [v, v], sigma_ref=1.0, converged=True, estimator="test")


class _StubSolverF:
    """``_linear_solver`` fails like a genuine tangent-solve failure would --
    exercises ``_bordered_conditioning_best_effort``'s graceful-None path
    instead of needing a real Newton-Krylov solver in these unit tests."""

    def _linear_solver(self, problem, X, *, deadline_s=0.0, t0=None):
        raise RuntimeError("stub: no real linear solver in this test")


class _StubProblemF:
    def source_coeffs(self, scale: float) -> np.ndarray:
        return np.array([[1.0 + 0j]])


# --- classify_region: one test per branch of the decision tree ------------


def test_classify_region_diagnostic_failure_when_triplet_did_not_converge() -> None:
    triplet = SingularTriplet([], [], [], float("nan"), False, "augmented_eigsh_failed")
    row = classify_region(None, None, _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_DIAGNOSTIC_FAILURE


def test_classify_region_nonsingular_when_sigma1_hat_not_small() -> None:
    triplet = _stub_triplet(sigma=(0.5, 1.0))  # sigma1_hat=0.5 >> default 1e-2
    row = classify_region(None, None, _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_NONSINGULAR_TANGENT_FLIP
    assert row["sigma1_hat"] == 0.5


def test_classify_region_higher_degeneracy_when_sigma_gap_not_separated() -> None:
    triplet = _stub_triplet(sigma=(1e-6, 1e-6))  # sigma1_hat tiny, sigma1/sigma2=1.0 >> 0.1
    row = classify_region(None, None, _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_POSSIBLE_HIGHER_DEGENERACY
    assert "corank" in row["reason"]


def _patch_residuals(monkeypatch, r_right: float, r_left: float) -> None:
    monkeypatch.setattr(
        bifurcation, "singular_triplet_residuals",
        lambda problem, X, triplet, *, index=0: {"r_right": r_right, "r_left": r_left},
    )


def test_classify_region_branch_point_when_tau_near_zero(monkeypatch) -> None:
    triplet = _stub_triplet(sigma=(1e-8, 1.0))
    monkeypatch.setattr(
        bifurcation, "transversality",
        lambda problem, w, *, tolerances=None: TransversalityResult(0.0, 0.0, False),
    )
    _patch_residuals(monkeypatch, 1e-9, 1e-9)
    row = classify_region(None, _StubProblemF(), _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_POSSIBLE_BRANCH_POINT
    assert row["tau"] == 0.0
    assert row["singular_residual_right"] == 1e-9  # residuals reported even here, per user request


def test_classify_region_quadratically_degenerate_when_alpha_zero_gamma_nonzero(monkeypatch) -> None:
    # sigma_ref=1.0 on the stub triplet, so alpha_hat == alpha directly.
    triplet = _stub_triplet(sigma=(1e-8, 1.0))
    monkeypatch.setattr(
        bifurcation, "transversality",
        lambda problem, w, *, tolerances=None: TransversalityResult(1.0, 1.0, True),
    )
    _patch_residuals(monkeypatch, 1e-9, 1e-9)  # comfortably below residual_max=1e-6
    monkeypatch.setattr(bifurcation, "exact_quadratic_fold_coefficient", lambda *a, **k: 0.0)
    monkeypatch.setattr(bifurcation, "exact_cubic_fold_coefficient", lambda *a, **k: 5.0)
    row = classify_region(_StubSolverF(), _StubProblemF(), _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_QUADRATICALLY_DEGENERATE_SINGULARITY
    assert row["alpha"] == 0.0
    assert row["gamma"] == 5.0
    assert row["bordered_conditioning"] is None  # stub solver's tangent solve fails gracefully


def test_classify_region_unclassified_when_alpha_zero_and_residuals_dirty(monkeypatch) -> None:
    triplet = _stub_triplet(sigma=(1e-8, 1.0))
    monkeypatch.setattr(
        bifurcation, "transversality",
        lambda problem, w, *, tolerances=None: TransversalityResult(1.0, 1.0, True),
    )
    _patch_residuals(monkeypatch, 0.5, 0.5)  # far above residual_max -- (w,v) not trustworthy
    monkeypatch.setattr(bifurcation, "exact_quadratic_fold_coefficient", lambda *a, **k: 0.0)
    monkeypatch.setattr(bifurcation, "exact_cubic_fold_coefficient", lambda *a, **k: 5.0)
    row = classify_region(_StubSolverF(), _StubProblemF(), _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_SINGULAR_UNCLASSIFIED
    assert "residuals too large" in row["reason"]


def test_classify_region_unclassified_when_alpha_and_gamma_both_zero(monkeypatch) -> None:
    triplet = _stub_triplet(sigma=(1e-8, 1.0))
    monkeypatch.setattr(
        bifurcation, "transversality",
        lambda problem, w, *, tolerances=None: TransversalityResult(1.0, 1.0, True),
    )
    _patch_residuals(monkeypatch, 1e-9, 1e-9)
    monkeypatch.setattr(bifurcation, "exact_quadratic_fold_coefficient", lambda *a, **k: 0.0)
    monkeypatch.setattr(bifurcation, "exact_cubic_fold_coefficient", lambda *a, **k: 0.0)
    row = classify_region(_StubSolverF(), _StubProblemF(), _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_SINGULAR_UNCLASSIFIED
    assert "gamma_hat" in row["reason"]


def test_classify_region_simple_fold_when_all_conditions_hold(monkeypatch) -> None:
    triplet = _stub_triplet(sigma=(1e-8, 1.0))
    monkeypatch.setattr(
        bifurcation, "transversality",
        lambda problem, w, *, tolerances=None: TransversalityResult(1.0, 1.0, True),
    )
    _patch_residuals(monkeypatch, 1e-9, 1e-9)
    monkeypatch.setattr(bifurcation, "exact_quadratic_fold_coefficient", lambda *a, **k: 5.0)
    row = classify_region(_StubSolverF(), _StubProblemF(), _stub_branch(), _stub_region(), triplet)
    assert row["class"] == FOLD_CLASS_SIMPLE_FOLD
    assert row["alpha"] == 5.0
    assert row["alpha_hat"] == 5.0  # sigma_ref=1.0 on the stub triplet
    assert row["gamma"] is None  # cubic coefficient not computed when alpha is already nonzero


# --- cluster_candidates_into_regions ---------------------------------------


def _profile_point(index: int, sigma1_hat: float) -> SingularityProfilePoint:
    return SingularityProfilePoint(
        index=index, s=float(index), mu=float(index), t_mu=1.0,
        sigma1=sigma1_hat, sigma2=1.0, sigma_ref=1.0, sigma1_hat=sigma1_hat,
        r_gap=sigma1_hat, estimator="test",
    )


def test_cluster_merges_adjacent_candidates_with_no_intervening_points() -> None:
    # Mirrors the real mu~0.5253 E0/E1 case: consecutive branch points, no
    # room between the two candidates' windows to prove they are separate.
    branch = ContinuationBranch(
        i_ref=1.0, points=[_stub_branch_point(float(i)) for i in range(6)], info={},
    )
    c0 = FoldCandidate(index=2, mu_before=1.0, mu_after=2.0, s_before=1.0, s_after=2.0)
    c1 = FoldCandidate(index=3, mu_before=2.0, mu_after=3.0, s_before=2.0, s_after=3.0)
    profiles = {i: _profile_point(i, sigma1_hat=1e-6) for i in range(6)}

    regions = cluster_candidates_into_regions(branch, [c0, c1], profiles, half_window=0)

    assert len(regions) == 1
    assert regions[0].candidate_ids == [0, 1]


def test_cluster_separates_candidates_with_a_clear_recovery_between_them() -> None:
    branch = ContinuationBranch(
        i_ref=1.0, points=[_stub_branch_point(float(i)) for i in range(10)], info={},
    )
    c0 = FoldCandidate(index=1, mu_before=0.0, mu_after=1.0, s_before=0.0, s_after=1.0)
    c1 = FoldCandidate(index=8, mu_before=7.0, mu_after=8.0, s_before=7.0, s_after=8.0)
    profiles = {i: _profile_point(i, sigma1_hat=1e-6) for i in range(10)}
    profiles[4] = _profile_point(4, sigma1_hat=0.5)  # clear recovery, well above default 1e-2

    regions = cluster_candidates_into_regions(branch, [c0, c1], profiles, half_window=0)

    assert [r.candidate_ids for r in regions] == [[0], [1]]


def test_cluster_picks_the_sigma1_minimum_as_the_region_anchor() -> None:
    branch = ContinuationBranch(
        i_ref=1.0, points=[_stub_branch_point(float(i)) for i in range(5)], info={},
    )
    c0 = FoldCandidate(index=2, mu_before=1.0, mu_after=2.0, s_before=1.0, s_after=2.0)
    profiles = {i: _profile_point(i, sigma1_hat=1.0) for i in range(5)}
    profiles[2] = _profile_point(2, sigma1_hat=1e-9)  # the true minimum

    regions = cluster_candidates_into_regions(branch, [c0], profiles, half_window=1)

    assert len(regions) == 1
    assert regions[0].branch_point_index == 2


# --- F.5: exact AFT alpha validated against the finite-difference version -


def test_exact_alpha_matches_finite_difference_at_the_known_fold() -> None:
    # fold_plan.md Milestone F.5's explicit validation request: the exact
    # AFT quadratic coefficient must agree with the (now secondary,
    # validation-only) finite-difference version wherever the latter has a
    # stable plateau. Uses the same real fold (lambda~0.7808) as the
    # end-to-end SIMPLE_FOLD test above.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_advanced_continuation import _build_folding_problem, _solver  # noqa: PLC0415

    from twpa_solver.pump.singularity import smallest_singular_triplets  # noqa: PLC0415
    from twpa_solver.pump.solver import trace_branch  # noqa: PLC0415

    solver = _solver()
    problem = _build_folding_problem(pump_current=1.0)
    branch = trace_branch(
        solver, problem, i_ref=1.0, mu0=0.0, mu_max=1.0, ds=0.05, max_steps=250,
    )
    near_fold_pt = min(branch.points, key=lambda p: abs(p.mu - 0.7808))
    triplet = smallest_singular_triplets(problem, near_fold_pt.X, k=2, iters=40)
    assert triplet.converged
    w_packed, v_packed = triplet.u[0], triplet.v[0]

    alpha_exact = bifurcation.exact_quadratic_fold_coefficient(
        problem, near_fold_pt.X, v_packed, w_packed,
    )
    fd_result = bifurcation.quadratic_fold_coefficient_fd(
        problem, near_fold_pt.X, v_packed, w_packed, state_scale=1.0,
    )

    assert fd_result.resolved
    assert alpha_exact == pytest.approx(fd_result.alpha, rel=1e-3)


def test_exact_alpha_matches_finite_difference_at_a_nonsingular_point() -> None:
    # Away from any fold too -- the exact formula must not merely happen to
    # agree near a singularity.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_advanced_continuation import _build_folding_problem, _solver  # noqa: PLC0415

    from twpa_solver.pump.singularity import smallest_singular_triplets  # noqa: PLC0415
    from twpa_solver.pump.solver import trace_branch  # noqa: PLC0415

    solver = _solver()
    problem = _build_folding_problem(pump_current=1.0)
    branch = trace_branch(
        solver, problem, i_ref=1.0, mu0=0.0, mu_max=1.0, ds=0.05, max_steps=250,
    )
    far_pt = min(branch.points, key=lambda p: abs(p.mu - 0.2))
    triplet = smallest_singular_triplets(problem, far_pt.X, k=2, iters=40)
    assert triplet.converged
    w_packed, v_packed = triplet.u[0], triplet.v[0]

    alpha_exact = bifurcation.exact_quadratic_fold_coefficient(
        problem, far_pt.X, v_packed, w_packed,
    )
    fd_result = bifurcation.quadratic_fold_coefficient_fd(
        problem, far_pt.X, v_packed, w_packed, state_scale=1.0,
    )

    assert fd_result.resolved
    assert alpha_exact == pytest.approx(fd_result.alpha, rel=1e-3)


# --- classify_fold_candidates: event-driven guard --------------------------


def test_classify_fold_candidates_on_a_real_solver_finds_a_simple_fold() -> None:
    # End-to-end positive control on the same 1-DOF near-resonant branch
    # used throughout test_advanced_continuation.py and test_singularity.py
    # (genuine S-shaped fold at lambda~0.7808). If F cannot classify this
    # reproducible, hand-verified fold as SIMPLE_FOLD, it is not ready to be
    # trusted on the real device branches, regardless of what it says about
    # any ambiguous region (fold_plan.md Milestone F).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_advanced_continuation import _build_folding_problem, _solver  # noqa: PLC0415

    from twpa_solver.pump.solver import trace_branch  # noqa: PLC0415

    solver = _solver()
    problem = _build_folding_problem(pump_current=1.0)
    branch = trace_branch(
        solver, problem, i_ref=1.0, mu0=0.0, mu_max=1.0, ds=0.05, max_steps=250,
    )

    rows = classify_fold_candidates(solver, problem, branch)

    assert len(rows) == 1
    row = rows[0]
    assert row["class"] == FOLD_CLASS_SIMPLE_FOLD
    assert row["mu"] == pytest.approx(0.78, abs=0.1)
    assert row["sigma1_hat"] < 1e-2
    assert row["sigma_gap"] < 1e-1
    assert row["tau"] > 0.5
    assert abs(row["alpha"]) > 0.0
    assert row["singular_residual_right"] is not None and row["singular_residual_right"] < 1e-6
    assert row["singular_residual_left"] is not None and row["singular_residual_left"] < 1e-6


def test_classify_fold_candidates_does_nothing_when_there_are_no_candidates(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("smallest_singular_triplets must not run with zero candidates")

    monkeypatch.setattr(bifurcation, "smallest_singular_triplets", _boom)
    branch = ContinuationBranch(
        i_ref=1.0,
        points=[_stub_branch_point(0.0, t_mu=1.0), _stub_branch_point(0.1, t_mu=1.0)],
        info={},
    )

    rows = classify_fold_candidates(None, None, branch)

    assert rows == []
