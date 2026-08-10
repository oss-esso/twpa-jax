"""Tests for the advanced intra-cell continuation methods added to the pump
solver: the tangent (Euler) predictor, pseudo-transient continuation, and
pseudo-arclength continuation.

A tiny one-node LC + Josephson problem (shared with
``test_exp08_seed_adaptive_warmstart``) runs the full Newton-Krylov stack in
milliseconds while exercising the real residual / JVP / real-coupled
preconditioner / arclength code paths.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from twpa_solver.pump import hb as exp08  # noqa: E402


def _build_problem(pump_current: float, *, omega: float = 0.37):
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    G = sp.csr_matrix(np.array([[0.01]], dtype=np.complex128))
    K = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = exp08.HarmonicGrid(modes=np.array([1, 2, 3]), nt=16, omega=omega)
    branch = exp08.JosephsonBranchArray(Ic=np.array([1.0], dtype=np.float64), phi0=1.0)
    return exp08.FullIPMPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi, branch=branch, grid=grid,
        pump_node_index=0, pump_current_a=pump_current,
    )


def _build_folding_problem(
    pump_current: float, *, Ic: float = 0.3, G: float = 0.001, omega: float = 0.9,
):
    # Lightly damped, near-resonant 1-DOF branch (small G, omega close to the
    # plasma resonance sqrt(K/C)=1): genuinely S-shaped/multivalued in lambda
    # over part of [0, 1], unlike _build_problem's well-conditioned default.
    # (omega=0.9, pump_current=1.0, Ic=0.3) reaches target_lam=1.0 via arclength
    # with a real fold at lambda~0.7808, needing ~166 total steps (fold hit
    # around step ~90) -- found by sweeping (omega, pump_current, Ic) for a
    # case with both a real fold_lambda and reached_target=True.
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Gm = sp.csr_matrix(np.array([[G]], dtype=np.complex128))
    Km = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = exp08.HarmonicGrid(modes=np.array([1, 2, 3]), nt=32, omega=omega)
    branch = exp08.JosephsonBranchArray(Ic=np.array([Ic], dtype=np.float64), phi0=1.0)
    return exp08.FullIPMPumpProblem(
        C=C, G=Gm, K=Km, Bphi=Bphi, branch=branch, grid=grid,
        pump_node_index=0, pump_current_a=pump_current,
    )


def _settings() -> exp08.NewtonKrylovSettings:
    return exp08.NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=30, gmres_rtol=1e-9, gmres_atol=0.0,
        gmres_restart=40, gmres_maxiter=200, min_alpha=1.0 / 1024.0,
        preconditioner="mean_tangent", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
    )


def _solver() -> exp08.HarmonicNewtonKrylovSolver:
    return exp08.HarmonicNewtonKrylovSolver(_settings())


def test_tangent_predictor_beats_copy_near_the_branch() -> None:
    # Converge at lambda=0.5, then predict lambda=0.6. The exact tangent should
    # give a lower residual at the new lambda than simply copying the state.
    problem = _build_problem(pump_current=0.5)
    solver = _solver()
    X_half, rep = solver.solve_one(problem, problem.zeros(), 0.5)
    assert rep.converged
    d_lambda = 0.1
    pred = solver.tangent_predictor(problem, X_half, d_lambda)
    r_copy = problem.norms(X_half, 0.6, False)["coeff_rel"]
    r_tan = problem.norms(pred, 0.6, False)["coeff_rel"]
    assert r_tan < r_copy


def test_pseudo_transient_converges_from_zero() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X, reports = solver.solve_pseudo_transient(problem, problem.zeros(), delta0=1.0)
    assert reports[-1].converged
    assert problem.norms(X, 1.0, False)["coeff_rel"] < 1e-7


def test_residual_homotopy_polishes_a_periodic_seed_to_the_physical_root() -> None:
    """The final homotopy problem must be the unchanged production HB problem."""
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    seed, seed_report = solver.solve_one(problem, problem.zeros(), 0.3 / 0.6)
    assert seed_report.converged
    assert problem.norms(seed, 1.0, False)["coeff_rel"] > 1e-4

    X, reports, trace = solver.solve_residual_homotopy(
        problem,
        seed,
        initial_step=0.2,
        min_step=1.0 / 64.0,
        max_step=0.25,
        max_steps=32,
    )

    assert trace.reached_target
    assert trace.final_eta == 1.0
    assert reports
    assert reports[-1].converged
    assert problem.norms(X, 1.0, False)["coeff_rel"] < 1e-8


def test_residual_homotopy_rejects_nonfinite_seed() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    seed = problem.zeros()
    seed[0, 0] = np.nan
    with np.testing.assert_raises(ValueError):
        solver.solve_residual_homotopy(problem, seed)


def test_arclength_reaches_target_lambda() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80)
    assert info["reached_target"]
    assert abs(lam - 1.0) < 1e-9
    # The target endpoint is linearly interpolated between the two straddling
    # arclength points, so its residual is small but not at Newton tolerance
    # (it is consumed as a warm guess for a final target solve downstream).
    assert problem.norms(X, 1.0, False)["coeff_rel"] < 1e-3


def test_arclength_matches_direct_solution() -> None:
    # The arclength solution at lambda=1 must equal the ordinary solve at full
    # drive (same branch, easy current).
    problem = _build_problem(pump_current=0.4)
    solver = _solver()
    X_direct, rep = solver.solve_one(problem, problem.zeros(), 1.0)
    assert rep.converged
    # ds=0.1 leaves ~7e-7 of linear-interpolation truncation at the target
    # crossing (interpolated between the two straddling arclength points, not
    # itself Newton-polished); ds=0.02 shrinks that below 1e-7 while still
    # converging in well under max_steps.
    X_arc, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0, max_steps=200)
    assert info["reached_target"]
    np.testing.assert_allclose(X_arc, X_direct, atol=1e-7)


def test_solve_arclength_mu_k1_matches_solve_arclength() -> None:
    # i_ref == problem.pump_current_a -> k=1 -> must be bit-identical to
    # calling solve_arclength directly (docs/development/fold_plan.md
    # Milestone A's "prove easy points match existing results").
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X_lam, lam, info_lam = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80)
    X_mu, mu, info_mu = solver.solve_arclength_mu(
        problem, problem.zeros(), 0.0, i_ref=0.6,
        ds=0.1, target_mu=1.0, max_steps=80)
    assert mu == lam
    np.testing.assert_array_equal(X_mu, X_lam)
    assert info_mu["reached_target"] == info_lam["reached_target"]
    assert info_mu["steps"] == info_lam["steps"]
    assert info_mu["mu_ref_current_a"] == 0.6


def test_solve_arclength_mu_rescales_target_and_step_by_reference_current() -> None:
    # i_ref = 2x problem.pump_current_a -> k=2 -> target_mu=0.5 must reach the
    # exact same physical point (and state) as the k=1 case's target_lam=1.0.
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X_ref, mu_ref, info_ref = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80)
    X_mu, mu, info_mu = solver.solve_arclength_mu(
        problem, problem.zeros(), 0.0, i_ref=1.2,
        ds=0.05, target_mu=0.5, max_steps=80)
    assert info_mu["reached_target"]
    assert abs(mu - 0.5) < 1e-9
    np.testing.assert_allclose(X_mu, X_ref, atol=1e-9)


def test_solve_arclength_mu_fold_mu_scales_with_reference_current() -> None:
    # Known fold at lambda~0.7808 for pump_current=1.0 (see
    # _build_folding_problem's own docstring). i_ref = 2x that current halves
    # the reported fold_mu relative to fold_lambda, since mu = lambda / k and
    # k = i_ref / pump_current_a = 2.
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    _X_lam, _lam, info_lam = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.05, target_lam=1.0, max_steps=250)
    fold_lambda = info_lam["fold_lambda"]
    assert fold_lambda is not None

    _X_mu, _mu, info_mu = solver.solve_arclength_mu(
        problem, problem.zeros(), 0.0, i_ref=2.0,
        ds=0.025, target_mu=0.5, max_steps=250)
    assert info_mu["fold_mu"] is not None
    assert abs(info_mu["fold_mu"] - fold_lambda / 2.0) < 1e-6


def _build_scaled_problem(pump_current: float, s: float, *, omega: float = 0.37):
    # Residual D(w)X + Bphi*Ic*sin(psi/phi0) - S is exactly covariant under
    # X -> s*X when phi0 -> s*phi0, Ic -> s*Ic, pump_current -> s*pump_current
    # (C, G, K, lambda unchanged): psi/phi0 is invariant so the nonlinear term
    # scales by s via Ic, the linear term scales by s via X, and the source
    # scales by s via pump_current_a. So the solution scales by exactly s and
    # every accepted lambda along the branch is identical to the s=1 problem.
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    G = sp.csr_matrix(np.array([[0.01]], dtype=np.complex128))
    K = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = exp08.HarmonicGrid(modes=np.array([1, 2, 3]), nt=16, omega=omega)
    branch = exp08.JosephsonBranchArray(Ic=np.array([s], dtype=np.float64), phi0=s)
    return exp08.FullIPMPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi, branch=branch, grid=grid,
        pump_node_index=0, pump_current_a=s * pump_current,
    )


def test_arclength_is_invariant_under_state_rescaling() -> None:
    solver = _solver()
    problem_unit = _build_scaled_problem(0.6, 1.0)
    X_unit, lam_unit, info_unit = solver.solve_arclength(
        problem_unit, problem_unit.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )

    s = 1e-15
    problem_scaled = _build_scaled_problem(0.6, s)
    X_scaled, lam_scaled, info_scaled = solver.solve_arclength(
        problem_scaled, problem_scaled.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )

    assert info_unit["reached_target"] == info_scaled["reached_target"]
    assert info_unit["reached_target"]
    assert abs(lam_scaled - lam_unit) < 1e-9
    np.testing.assert_allclose(X_scaled, s * X_unit, rtol=1e-4, atol=s * 1e-8)


def test_arclength_reports_state_scale() -> None:
    solver = _solver()
    problem = _build_scaled_problem(0.6, 1e-15)
    _X, _lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )
    assert info["state_scale"] is not None
    assert math.isfinite(info["state_scale"])
    assert info["state_scale"] > 0.0
    # The scale must actually track the state's units, not sit pinned at 1.0.
    assert info["state_scale"] < 1e-10


def test_relative_step_floor_scales_with_ds() -> None:
    # newton_max=0 makes the inner Newton loop's range(1, 1) empty, so every
    # outer step fails unconditionally without ever evaluating the residual
    # tolerance (which this fixture reaches too easily via the tol floor
    # `max(newton_tol*10, 1e-8)` to force a realistic corrector failure).
    # Halving continues until step_size < ds*1e-6, so the number of halvings
    # to reach the floor is the SAME for any ds (log2(1e6) ~ 20) -- the old
    # absolute 1e-4 floor would instead give a ds-dependent step count (1
    # halving for ds=1e-4, ~10 for ds=0.1).
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    steps_by_ds = {}
    for ds in (1e-4, 0.1):
        X, lam, info = solver.solve_arclength(
            problem, problem.zeros(), 0.0, ds=ds, target_lam=1.0,
            max_steps=200, newton_max=0,
        )
        assert info["terminal_reason"] == "minimum_step"
        assert lam == 0.0  # no step was ever accepted
        steps_by_ds[ds] = info["steps"]
    assert steps_by_ds[1e-4] == steps_by_ds[0.1]


def test_rescale_every_zero_is_regression_safe() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X_default, lam_default, info_default = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )
    X_explicit, lam_explicit, info_explicit = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
        rescale_every=0,
    )
    assert info_default["rescale_count"] == 0
    assert info_explicit["rescale_count"] == 0
    np.testing.assert_array_equal(X_default, X_explicit)
    assert lam_default == lam_explicit
    assert info_default["steps"] == info_explicit["steps"]


def test_rescale_every_fires_and_reaches_same_solution() -> None:
    # This fixture's 1-DOF branch never gets stiff enough to demonstrate a
    # rescale_every=0 FAILURE that rescaling fixes (see
    # docs/development/arclength_fold_resolution_plan.md Phase 1 -- that
    # comparison needs the real 2c device, exercised in Phase 3). This test
    # instead verifies the mechanism itself is correct and safe: rescaling
    # actually fires, and produces the same converged solution as the
    # unrescaled path (rescaling only changes the metric used to get there,
    # never the accepted X/lambda history).
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X0, lam0, info0 = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
        rescale_every=0,
    )
    X5, lam5, info5 = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
        rescale_every=5,
    )
    assert info0["reached_target"] and info5["reached_target"]
    assert info5["rescale_count"] > 0
    assert lam0 == lam5 == 1.0
    np.testing.assert_allclose(X5, X0, atol=1e-6)


def test_metric_rescale_preserves_physical_predictor() -> None:
    from twpa_solver.pump.solver import _rescale_arclength_tangent

    Xdot = np.array([3.0 + 4.0j, -2.0j], dtype=np.complex128)
    lam_dot = 0.17
    ds = 0.31

    transformed = _rescale_arclength_tangent(Xdot, lam_dot, 2.5, ds)

    assert transformed is not None
    Xdot_new, lam_dot_new, ds_new, _q = transformed
    np.testing.assert_allclose(ds_new * Xdot_new, ds * Xdot, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(ds_new * lam_dot_new, ds * lam_dot, rtol=0.0, atol=1e-15)


def test_adaptive_metric_rescale_tracks_state_amplitude_near_fold() -> None:
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    telemetry: list[dict] = []

    _X, _lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=float("inf"),
        max_steps=120, rescale_every=5, step_telemetry=telemetry.append,
    )

    assert info["rescale_count"] > 0
    assert len(info["rescale_events"]) == info["rescale_count"]
    assert all(event["predictor_error"] < 1e-12 for event in info["rescale_events"])
    # The tangent remains free to approach zero in its lambda component; a
    # rescale must not algebraically reset it to 1/sqrt(2).
    assert any(abs(entry["t_lam_own"]) < 0.2 for entry in telemetry)


def test_max_steps_after_fold_none_pins_current_behavior() -> None:
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=120, newton_max=15,
    )
    assert info["fold_lambda"] is not None
    assert not info["reached_target"]
    assert info["terminal_reason"] == "max_steps"
    assert info["steps"] == 120


def test_max_steps_after_fold_zero_matches_none() -> None:
    # max_steps_after_fold=0 must behave exactly like the disabled default
    # (None) -- the fold in this fixture is detected well before step 120,
    # so a budget of "fold_step + 0" is not larger than the original 120 and
    # extends nothing.
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=120, newton_max=15, max_steps_after_fold=0,
    )
    assert not info["reached_target"]
    assert info["terminal_reason"] == "max_steps"
    assert info["steps"] == 120


def test_max_steps_after_fold_rounds_the_fold() -> None:
    # With enough post-fold budget, the same run that failed at max_steps=120
    # above continues onto the returning branch and reaches target_lam=1.0.
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=120, newton_max=15, max_steps_after_fold=90,
    )
    assert info["fold_lambda"] is not None
    assert info["reached_target"]
    assert abs(lam - 1.0) < 1e-9

    # Must match the same fold/target run allowed a large max_steps up front.
    X_ref, lam_ref, info_ref = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=200, newton_max=15,
    )
    assert info_ref["reached_target"]
    np.testing.assert_allclose(X, X_ref, atol=1e-6)


def test_fold_power_rescale_every_is_forwarded() -> None:
    # fold_power is a thin wrapper around solve_arclength with
    # target_lam=inf; this only checks rescale_every reaches the inner call
    # and doesn't change the located fold on a fixture that doesn't need
    # rescaling (see docs/development/arclength_fold_resolution_plan.md
    # Phase 4 -- run_fold_follow now forwards this so a metric-mistuning
    # false negative can't silently read as "no fold in range" again).
    from twpa_solver.pump.solver import fold_power

    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    lam0 = fold_power(solver, problem, ds=0.02, max_steps=120)
    lam5 = fold_power(solver, problem, ds=0.02, max_steps=120, rescale_every=5)
    assert lam0 is not None
    assert lam5 is not None
    # Rescaling nudges exactly which accepted step first shows the lam_dot
    # sign flip, so the two fold locations are close but not bit-identical.
    assert abs(lam0 - lam5) < 1e-2


def test_trace_branch_stores_points_and_resolve_matches_direct_solve() -> None:
    # Milestone B (docs/development/fold_plan.md Section 20/23): one branch
    # trace, then bracket+resolve a target off it, must match a fresh
    # solve_one at the same physical point exactly (same converged basin).
    from twpa_solver.pump.solver import resolve_target_on_branch, trace_branch

    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    branch = trace_branch(solver, problem, i_ref=0.6, mu_max=1.0, ds=0.1, max_steps=80)
    assert branch.info["reached_target"]
    assert len(branch.points) >= 2
    # Monotonic up to the last on_step sample: it can overshoot target_lam
    # before the same-step interpolation appends the exact target point
    # after it (test_singularity_measurement.py's on_step tests document the
    # same overshoot-then-interpolate ordering).
    mus = [p.mu for p in branch.points[:-1]]
    assert mus == sorted(mus)

    result = resolve_target_on_branch(solver, problem, branch, 0.5)
    assert result is not None
    X_resolved, report = result
    assert report.converged
    X_direct, rep_direct = solver.solve_one(problem, problem.zeros(), 0.5)
    assert rep_direct.converged
    np.testing.assert_allclose(X_resolved, X_direct, atol=1e-8)


def test_bracket_target_returns_none_beyond_traced_extent() -> None:
    from twpa_solver.pump.solver import resolve_target_on_branch, trace_branch

    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    branch = trace_branch(solver, problem, i_ref=0.6, mu_max=1.0, ds=0.1, max_steps=80)
    assert resolve_target_on_branch(solver, problem, branch, 5.0) is None


def test_trace_branch_records_fold_and_multiple_targets_reuse_one_trace() -> None:
    # Same physical branch traced once (i_ref=1.0); two different targets
    # below the fold both resolved off it -- the point of Milestone B is
    # that neither target reruns its own lambda:0->1 continuation problem.
    from twpa_solver.pump.solver import resolve_target_on_branch, trace_branch

    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    branch = trace_branch(
        solver, problem, i_ref=1.0, mu0=0.0, mu_max=1.0, ds=0.05, max_steps=250,
    )
    assert branch.info["fold_mu"] is not None
    assert abs(branch.info["fold_mu"] - 0.7808) < 5e-3

    for mu_target in (0.3, 0.6):
        result = resolve_target_on_branch(solver, problem, branch, mu_target)
        assert result is not None
        X, report = result
        assert report.converged
        assert problem.norms(X, mu_target, False)["coeff_rel"] < 1e-7


def test_step_telemetry_default_none_leaves_solve_arclength_unaffected() -> None:
    # step_telemetry=None (default) must be bit-identical to before this
    # parameter's payload was extended with residuals/GMRES counts.
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80)
    assert info["reached_target"]
    assert abs(lam - 1.0) < 1e-9


def test_step_telemetry_reports_residuals_and_gmres_count_per_accepted_step() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    telemetry: list[dict] = []
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
        step_telemetry=telemetry.append,
    )
    assert info["reached_target"]
    assert len(telemetry) == info["steps"]
    for entry in telemetry:
        for key in (
            "step", "lam", "step_size", "used_newton", "theta_deg",
            "state_scale", "rejected_steps", "hb_residual_rel",
            "palc_residual", "gmres_iterations", "t_lam_pred", "t_lam_own",
            "rescale_event", "rescale_norm", "rescale_step_size_before",
            "rescale_step_size_after", "rescale_predictor_error",
        ):
            assert key in entry
        # HB residual at each accepted step must itself be converged (that
        # is what "accepted" means); PALC constraint residual likewise.
        assert entry["hb_residual_rel"] < 1e-6
        assert abs(entry["palc_residual"]) < 1e-6
        assert entry["gmres_iterations"] >= 1


def test_step_telemetry_via_solve_arclength_mu_converts_lam_to_mu() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    telemetry: list[dict] = []
    X, mu, info = solver.solve_arclength_mu(
        problem, problem.zeros(), 0.0, i_ref=1.2,
        ds=0.05, target_mu=0.5, max_steps=80,
        step_telemetry=telemetry.append,
    )
    assert info["reached_target"]
    assert len(telemetry) == info["steps"]
    for entry in telemetry:
        assert "mu" in entry and "lam" not in entry
    # Accepted steps can overshoot target_mu before the final interpolated
    # crossing (same overshoot-then-interpolate ordering as trace_branch's
    # on_step) -- only the returned mu is guaranteed to equal the target.
    assert abs(mu - 0.5) < 1e-9


def test_step_telemetry_forwarded_through_trace_branch() -> None:
    from twpa_solver.pump.solver import trace_branch

    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    telemetry: list[dict] = []
    branch = trace_branch(
        solver, problem, i_ref=1.0, mu0=0.0, mu_max=1.0, ds=0.05, max_steps=250,
        step_telemetry=telemetry.append,
    )
    assert branch.info["fold_mu"] is not None
    assert len(telemetry) == branch.info["steps"]
    # Telemetry's own t_lam_own sign change must land at the same fold
    # already found via branch.points' t_mu (independent cross-check that
    # the new telemetry path observes the same physical event).
    signs = [e["t_lam_own"] >= 0 for e in telemetry]
    flips = [i for i in range(1, len(signs)) if signs[i] != signs[i - 1]]
    assert len(flips) >= 1
    flip_mu = telemetry[flips[0]]["mu"]
    assert abs(flip_mu - branch.info["fold_mu"]) < 5e-2


def test_scaled_two_point_arclength_reaches_higher_drive() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X0, rep0 = solver.solve_one(problem, problem.zeros(), 0.2)
    X1, rep1 = solver.solve_one(problem, X0, 0.3)
    assert rep0.converged and rep1.converged

    points, info = solver.trace_arclength_from_two_points(
        problem, X0, 0.2, X1, 0.3, ds=0.05, max_steps=30,
    )

    assert len(points) > 2
    assert max(lam for _X, lam in points) >= 1.0
    assert info["state_scale"] > 0.0


# ---------------------------------------------------------------------------
# Milestone C: adaptive ds (fold_plan.md Sections 10-11)
# ---------------------------------------------------------------------------


def test_adaptive_step_size_grows_on_easy_corrector_and_shrinks_on_hard() -> None:
    from twpa_solver.pump.solver import _adaptive_step_size

    kwargs = dict(
        newton_target=4, growth_exponent=0.5, growth_clamp=(0.5, 1.5),
        ds_max=1.0, ds_min=1e-9,
    )
    # 1 Newton iteration (well under target 4) -> grow, clamped at 1.5x.
    easy = _adaptive_step_size(0.1, 1, 0.0, **kwargs)
    assert easy == 0.15000000000000002 or abs(easy - 0.1 * 1.5) < 1e-12
    # 16 Newton iterations (far over target 4) -> shrink, clamped at 0.5x.
    hard = _adaptive_step_size(0.1, 16, 0.0, **kwargs)
    assert abs(hard - 0.1 * 0.5) < 1e-12
    # Exactly at target -> no change from the Newton-effort factor.
    at_target = _adaptive_step_size(0.1, 4, 0.0, **kwargs)
    assert abs(at_target - 0.1) < 1e-12


def test_adaptive_step_size_curvature_caps_growth_near_a_fold() -> None:
    from twpa_solver.pump.solver import _adaptive_step_size

    kwargs = dict(
        newton_target=4, growth_exponent=0.5, growth_clamp=(0.5, 1.5),
        ds_max=1.0, ds_min=1e-9,
    )
    # Cheap corrector (would grow to 0.15) but a sharply bending branch
    # (60 deg tangent swing) must still shrink -- curvature overrides a good
    # Newton-effort factor, exactly the "fold produces smaller steps with no
    # fold-specific hardcoded ds" property fold_plan.md Section 11 asks for.
    straight = _adaptive_step_size(0.1, 1, math.radians(2.0), **kwargs)
    bending = _adaptive_step_size(0.1, 1, math.radians(60.0), **kwargs)
    assert straight > 0.1
    assert bending < 0.1
    assert bending <= 0.1 * 0.5 + 1e-12


def test_adaptive_step_size_respects_ds_min_and_ds_max() -> None:
    from twpa_solver.pump.solver import _adaptive_step_size

    grown = _adaptive_step_size(
        0.1, 1, 0.0, newton_target=4, growth_exponent=0.5,
        growth_clamp=(0.5, 1.5), ds_max=0.12, ds_min=1e-9,
    )
    assert grown == 0.12
    shrunk = _adaptive_step_size(
        0.1, 100, math.radians(90.0), newton_target=4, growth_exponent=0.5,
        growth_clamp=(0.5, 1.5), ds_max=1.0, ds_min=0.09,
    )
    assert shrunk == 0.09


def test_solve_arclength_adaptive_reaches_same_solution_as_legacy() -> None:
    # step_control="adaptive" changes the step-size path, not the physics --
    # both modes must converge to the same target point.
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X_legacy, lam_legacy, info_legacy = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, max_steps=80, target_lam=1.0,
    )
    X_adaptive, lam_adaptive, info_adaptive = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, max_steps=80, target_lam=1.0,
        step_control="adaptive",
    )
    assert info_legacy["reached_target"]
    assert info_adaptive["reached_target"]
    assert lam_legacy == lam_adaptive == 1.0
    np.testing.assert_allclose(X_adaptive, X_legacy, atol=1e-6)


def test_solve_arclength_adaptive_ds_max_allows_larger_steps_than_ds_initial() -> None:
    # Legacy mode's growth is hard-capped at ds_initial; adaptive mode's
    # ds_max is a separate, independently settable ceiling.
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    seen_step_sizes: list[float] = []

    def _collect(Xc, lamc, step_size, Xdot, lam_dot, state_scale) -> None:
        seen_step_sizes.append(step_size)

    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.05, max_steps=80, target_lam=1.0,
        step_control="adaptive", ds_max=0.5, on_step=_collect,
    )
    assert info["reached_target"]
    assert max(seen_step_sizes) > 0.05


def test_solve_arclength_default_step_control_is_legacy() -> None:
    # Regression pin: omitting step_control must be bit-identical to
    # step_control="legacy", not a silent default change to "adaptive".
    problem = _build_problem(pump_current=0.6)
    X_default, lam_default, _info_default = _solver().solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, max_steps=80, target_lam=1.0,
    )
    X_legacy, lam_legacy, _info_legacy = _solver().solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, max_steps=80, target_lam=1.0,
        step_control="legacy",
    )
    assert lam_default == lam_legacy
    np.testing.assert_array_equal(X_default, X_legacy)


# ---------------------------------------------------------------------------
# Milestone D: fold events (fold_plan.md Section 16)
# ---------------------------------------------------------------------------


def test_refine_fold_disabled_by_default_leaves_fold_refined_none() -> None:
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    _X, _lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=120, newton_max=15,
    )
    assert info["fold_lambda"] is not None
    assert info["fold_refined"] is None


def test_refine_fold_narrows_bracket_below_the_step_grid() -> None:
    # The bracketing accepted points around the sign flip are ds=0.02 apart;
    # a converged refinement must land far tighter than that grid spacing,
    # and near the ~0.7808 fold this fixture is known to have (matches the
    # value pinned by test_trace_branch_records_fold_and_multiple_targets_
    # reuse_one_trace at a coarser ds).
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    _X, _lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=120, newton_max=15, refine_fold=True,
    )
    refined = info["fold_refined"]
    assert refined is not None
    assert refined["converged"]
    assert refined["lam"] is not None
    assert abs(refined["lam"] - 0.7808) < 5e-3
    assert refined["bracket_width"] < 0.02
    assert abs(refined["lam_dot"]) < 1e-6 or refined["bracket_width"] < 1e-6


def test_refine_fold_tolerances_are_configurable() -> None:
    # A loose fold_t_tol should converge in fewer secant iterations than a
    # tight one, on the same bracketed fold.
    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    _X, _lam, loose = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=120, newton_max=15, refine_fold=True, fold_t_tol=1e-2,
    )
    _X, _lam, tight = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.02, target_lam=1.0,
        max_steps=120, newton_max=15, refine_fold=True, fold_t_tol=1e-10,
        fold_lambda_tol=1e-12,
    )
    assert loose["fold_refined"]["iterations"] <= tight["fold_refined"]["iterations"]


def test_solve_arclength_mu_refine_fold_converts_units() -> None:
    from twpa_solver.pump.solver import trace_branch

    problem = _build_folding_problem(pump_current=1.0)
    solver = _solver()
    i_ref = 2.0 * problem.pump_current_a  # k = 2.0
    X, mu, info = solver.solve_arclength_mu(
        problem, problem.zeros(), 0.0, i_ref=i_ref, target_mu=2.0, ds=0.01,
        max_steps=120, newton_max=15, refine_fold=True,
    )
    assert info["fold_mu"] is not None
    refined = info["fold_refined"]
    assert refined is not None
    assert refined["converged"]
    assert refined["mu"] is not None
    # Refined mu must sit within the coarse bracket's own tolerance of the
    # raw (unrefined) bracket-point fold_mu -- same physical fold, tighter
    # localization.
    assert abs(refined["mu"] - info["fold_mu"]) < 0.01

    # Same knobs also reach trace_branch (Milestone B) unchanged.
    branch = trace_branch(
        solver, problem, i_ref=i_ref, mu0=0.0, mu_max=2.0, ds=0.01,
        max_steps=120, refine_fold=True,
    )
    assert branch.info["fold_refined"] is not None
    assert branch.info["fold_refined"]["converged"]


# --- Milestone E: production reachability semantics (fold_plan.md) ---
#
# classify_fold_pair/classify_adjacent_fold_candidate_pairs call
# solver.solve_one at caller-chosen seeds; the thing under test is the
# orchestration (segment selection, mu_star placement, pairwise-distance
# gating), not whether a real device happens to have a local S-bend at hand.
# A stub solver that echoes its seed back as an already-converged "root"
# gives full, deterministic control over root distinctness via the X values
# placed on synthetic BranchPoints -- mocking at the solve_one boundary,
# matching how the other unit is exercised (real solves are covered by
# test_arclength_*/test_refine_fold_*).


class _StubReport:
    def __init__(self, converged: bool, coeff_rel: float = 1e-13) -> None:
        self.converged = converged
        self.coeff_rel = coeff_rel


class _StubProblem:
    pump_current_a = 1.0


class _EchoSolver:
    """solve_one that always converges and returns the seed X unchanged."""

    def solve_one(self, problem, X_guess, mu):
        return np.array(X_guess, copy=True), _StubReport(True)


class _NthCallFailsSolver:
    """solve_one that fails on the ``fail_at`` 0-indexed call, else echoes."""

    def __init__(self, fail_at: int) -> None:
        self._fail_at = fail_at
        self._calls = 0

    def solve_one(self, problem, X_guess, mu):
        converged = self._calls != self._fail_at
        self._calls += 1
        return np.array(X_guess, copy=True), _StubReport(converged)


def _branch_point(mu: float, x: float, t_mu: float, s: float) -> "BranchPoint":  # noqa: F821
    from twpa_solver.pump.solver import BranchPoint

    return BranchPoint(s=s, mu=mu, X=np.array([x], dtype=np.complex128), t_mu=t_mu, step_size=0.01)


def test_find_fold_candidates_detects_all_sign_changes() -> None:
    from twpa_solver.pump.solver import ContinuationBranch, find_fold_candidates

    # t_mu: +, +, -, -, +, + -> two sign changes, at index 2 and index 4.
    points = [
        _branch_point(mu=0.1, x=0.1, t_mu=1.0, s=0.1),
        _branch_point(mu=0.2, x=0.2, t_mu=1.0, s=0.2),
        _branch_point(mu=0.19, x=0.21, t_mu=-1.0, s=0.3),
        _branch_point(mu=0.18, x=0.22, t_mu=-1.0, s=0.4),
        _branch_point(mu=0.25, x=0.23, t_mu=1.0, s=0.5),
        _branch_point(mu=0.30, x=0.24, t_mu=1.0, s=0.6),
    ]
    branch = ContinuationBranch(i_ref=1.0, points=points, info={})
    candidates = find_fold_candidates(branch)
    assert [c.index for c in candidates] == [2, 4]
    assert candidates[0].mu_before == 0.2 and candidates[0].mu_after == 0.19
    assert candidates[1].mu_before == 0.18 and candidates[1].mu_after == 0.25
    # Placeholder promotion fields (Milestone F territory), present but inert.
    assert all(c.kind == "TANGENT_SIGN_FLIP" for c in candidates)
    assert all(c.validation == "UNVALIDATED" for c in candidates)


def test_find_fold_candidates_monotone_branch_has_none() -> None:
    from twpa_solver.pump.solver import ContinuationBranch, find_fold_candidates

    points = [_branch_point(mu=0.1 * i, x=0.1 * i, t_mu=1.0, s=0.1 * i) for i in range(5)]
    branch = ContinuationBranch(i_ref=1.0, points=points, info={})
    assert find_fold_candidates(branch) == []


def _pair_branch(x_before: float, x_inside: float, x_after: float) -> "ContinuationBranch":  # noqa: F821
    from twpa_solver.pump.solver import ContinuationBranch

    # General triple-interval overlap [max(min_i), min(max_i)] over the three
    # segments: before=[0.40,0.50], inside=[0.42,0.44], after=[0.43,0.55] ->
    # overlap = [max(0.40,0.42,0.43), min(0.50,0.44,0.55)] = [0.43, 0.44].
    points = [
        _branch_point(mu=0.40, x=x_before, t_mu=1.0, s=0.0),
        _branch_point(mu=0.50, x=x_before, t_mu=1.0, s=0.1),   # candidate_a: + -> - at index 2
        _branch_point(mu=0.44, x=x_inside, t_mu=-1.0, s=0.2),
        _branch_point(mu=0.42, x=x_inside, t_mu=-1.0, s=0.3),  # candidate_b: - -> + at index 4
        _branch_point(mu=0.43, x=x_after, t_mu=1.0, s=0.4),
        _branch_point(mu=0.55, x=x_after, t_mu=1.0, s=0.5),
    ]
    return ContinuationBranch(i_ref=1.0, points=points, info={})


def test_classify_fold_pair_distinct_roots_is_local_fold_pair() -> None:
    from twpa_solver.pump.solver import (
        FOLD_TOPOLOGY_LOCAL_PAIR,
        classify_fold_pair,
        find_fold_candidates,
    )

    branch = _pair_branch(x_before=1.0, x_inside=2.0, x_after=3.0)
    candidate_a, candidate_b = find_fold_candidates(branch)
    result = classify_fold_pair(_EchoSolver(), _StubProblem(), branch, candidate_a, candidate_b)
    assert result["status"] == FOLD_TOPOLOGY_LOCAL_PAIR
    assert all(d > 1e-6 for d in result["pairwise"].values())


def test_classify_fold_pair_collapsed_roots_is_unresolved() -> None:
    # fold_plan.md Milestone D.5's actual measurement on ipm_2c_fixed 7.9 GHz
    # mu~0.5253: "inside" and "after" converged to the same root despite a
    # clean second sign flip. Reproduced here with the stub: inside == after.
    # UNRESOLVED_NEAR_FOLD reports insufficient/inconsistent evidence -- it is
    # NOT a claim of a specific mathematical degeneracy (that is Milestone F).
    from twpa_solver.pump.solver import (
        FOLD_TOPOLOGY_UNRESOLVED,
        classify_fold_pair,
        find_fold_candidates,
    )

    branch = _pair_branch(x_before=1.0, x_inside=2.0, x_after=2.0)
    candidate_a, candidate_b = find_fold_candidates(branch)
    result = classify_fold_pair(_EchoSolver(), _StubProblem(), branch, candidate_a, candidate_b)
    assert result["status"] == FOLD_TOPOLOGY_UNRESOLVED
    assert result["pairwise"]["inside_after"] < 1e-6


def test_classify_fold_pair_reports_unresolved_when_a_seed_fails_to_converge() -> None:
    from twpa_solver.pump.solver import FOLD_TOPOLOGY_UNRESOLVED, classify_fold_pair, find_fold_candidates

    branch = _pair_branch(x_before=1.0, x_inside=2.0, x_after=3.0)
    candidate_a, candidate_b = find_fold_candidates(branch)
    # dict insertion order in classify_fold_pair is before, inside, after ->
    # fail the 2nd call ("inside").
    result = classify_fold_pair(
        _NthCallFailsSolver(fail_at=1), _StubProblem(), branch, candidate_a, candidate_b,
    )
    assert result["status"] == FOLD_TOPOLOGY_UNRESOLVED
    assert result["reason"] == "a segment seed failed to converge"


def test_classify_fold_pair_reports_unresolved_when_no_mu_overlap() -> None:
    from twpa_solver.pump.solver import (
        BranchPoint,
        ContinuationBranch,
        FOLD_TOPOLOGY_UNRESOLVED,
        classify_fold_pair,
        find_fold_candidates,
    )

    # "before" segment's peak mu (0.06) never reaches up to where "inside"
    # ever gets (>= 0.10), so there is no mu at which all three segments
    # coexist under the general triple-interval overlap
    # [max(min_i), min(max_i)] -- max(mins)=0.11 > min(maxs)=0.06.
    points = [
        _branch_point(mu=0.05, x=1.0, t_mu=1.0, s=0.0),
        _branch_point(mu=0.06, x=1.0, t_mu=1.0, s=0.1),
        _branch_point(mu=0.20, x=2.0, t_mu=-1.0, s=0.2),
        _branch_point(mu=0.10, x=2.0, t_mu=-1.0, s=0.3),
        _branch_point(mu=0.11, x=3.0, t_mu=1.0, s=0.4),
        _branch_point(mu=0.55, x=3.0, t_mu=1.0, s=0.5),
    ]
    branch = ContinuationBranch(i_ref=1.0, points=points, info={})
    candidate_a, candidate_b = find_fold_candidates(branch)
    result = classify_fold_pair(_EchoSolver(), _StubProblem(), branch, candidate_a, candidate_b)
    assert result["status"] == FOLD_TOPOLOGY_UNRESOLVED
    assert "no mu overlap" in result["reason"]


def test_classify_adjacent_fold_candidate_pairs_tests_every_consecutive_pair() -> None:
    # 3 candidates (E0, E1, E2) -> adjacent pairing tests (E0,E1) and (E1,E2),
    # NOT a disjoint (E0,E1) with E2 left trailing -- the correction from
    # fold_plan.md's original disjoint two-at-a-time design, which could
    # silently miss the meaningful pair when an early candidate is noise.
    from twpa_solver.pump.solver import ContinuationBranch, classify_adjacent_fold_candidate_pairs

    points = [
        _branch_point(mu=0.10, x=1.0, t_mu=1.0, s=0.0),
        _branch_point(mu=0.20, x=1.0, t_mu=1.0, s=0.1),   # E0: + -> -
        _branch_point(mu=0.14, x=2.0, t_mu=-1.0, s=0.2),
        _branch_point(mu=0.12, x=2.0, t_mu=-1.0, s=0.3),  # E1: - -> +
        _branch_point(mu=0.13, x=3.0, t_mu=1.0, s=0.4),
        _branch_point(mu=0.40, x=3.0, t_mu=1.0, s=0.5),   # E2: + -> -
        _branch_point(mu=0.30, x=4.0, t_mu=-1.0, s=0.6),
    ]
    branch = ContinuationBranch(i_ref=1.0, points=points, info={})
    results = classify_adjacent_fold_candidate_pairs(_EchoSolver(), _StubProblem(), branch)
    assert len(results) == 2  # (E0,E1) and (E1,E2), not a disjoint 1-pair grouping
    assert [(r["candidate_a_id"], r["candidate_b_id"]) for r in results] == [(0, 1), (1, 2)]


def test_classify_adjacent_fold_candidate_pairs_empty_when_fewer_than_two_candidates() -> None:
    from twpa_solver.pump.solver import ContinuationBranch, classify_adjacent_fold_candidate_pairs

    # 0 candidates.
    points = [_branch_point(mu=0.1 * i, x=0.1 * i, t_mu=1.0, s=0.1 * i) for i in range(5)]
    branch0 = ContinuationBranch(i_ref=1.0, points=points, info={})
    assert classify_adjacent_fold_candidate_pairs(_EchoSolver(), _StubProblem(), branch0) == []

    # 1 candidate (a lone, unrecovered sign flip) -- no adjacent partner to pair with.
    points1 = [
        _branch_point(mu=0.10, x=1.0, t_mu=1.0, s=0.0),
        _branch_point(mu=0.60, x=1.0, t_mu=1.0, s=0.5),
        _branch_point(mu=0.59, x=2.0, t_mu=-1.0, s=0.6),
    ]
    branch1 = ContinuationBranch(i_ref=1.0, points=points1, info={})
    assert classify_adjacent_fold_candidate_pairs(_EchoSolver(), _StubProblem(), branch1) == []
