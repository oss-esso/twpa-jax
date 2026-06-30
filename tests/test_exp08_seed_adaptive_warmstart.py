"""Focused tests for the exp08 pump-solve speedup paths.

These cover the opt-in machinery added on top of the legacy cold/fixed solve:
the ``linear_phasor`` initial guess, the adaptive continuation traversal, and
the warm-start direct solve (``promote-from-pump-dir`` style). A tiny one-node
LC + Josephson problem is used so the full Newton-Krylov stack runs in
milliseconds while still exercising the real residual, JVP, preconditioner and
continuation code paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXPERIMENTS))

import exp08_full_ipm_pump_solve as exp08  # noqa: E402


def _build_problem(
    pump_current: float, *, omega: float = 0.37
) -> exp08.FullIPMPumpProblem:
    """One node, one Josephson branch to ground: C x'' + G x' + K x + Ic sin(x)."""
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    G = sp.csr_matrix(np.array([[0.01]], dtype=np.complex128))
    K = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = exp08.HarmonicGrid(modes=np.array([1, 2, 3]), nt=16, omega=omega)
    branch = exp08.JosephsonBranchArray(Ic=np.array([1.0], dtype=np.float64), phi0=1.0)
    return exp08.FullIPMPumpProblem(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        branch=branch,
        grid=grid,
        pump_node_index=0,
        pump_current_a=pump_current,
    )


def _settings(newton_tol: float = 1e-9) -> exp08.NewtonKrylovSettings:
    return exp08.NewtonKrylovSettings(
        newton_tol=newton_tol,
        max_newton=30,
        gmres_rtol=1e-9,
        gmres_atol=0.0,
        gmres_restart=40,
        gmres_maxiter=200,
        min_alpha=1.0 / 1024.0,
        preconditioner="mean_tangent",
        compute_time_residual=True,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


@pytest.fixture
def problem() -> exp08.FullIPMPumpProblem:
    return _build_problem(pump_current=0.3)


@pytest.fixture
def solver() -> exp08.HarmonicNewtonKrylovSolver:
    return exp08.HarmonicNewtonKrylovSolver(_settings())


# ---------------------------------------------------------------------------
# linear_phasor seed
# ---------------------------------------------------------------------------

def test_linear_phasor_seed_solves_fundamental_block(
    problem: exp08.FullIPMPumpProblem,
) -> None:
    X_seed, meta = exp08.build_linear_phasor_seed(
        problem, source_scale=1.0, method="gmres", rtol=1e-10, maxiter=200, restart=40
    )

    assert X_seed.shape == (problem.H, problem.n)
    assert np.all(np.isfinite(X_seed))
    # Only the source (fundamental) mode is seeded; higher modes stay zero.
    assert np.allclose(X_seed[1:], 0.0)
    assert np.linalg.norm(X_seed[problem.source_row]) > 0.0
    # D(omega_p) X1 = S1 is solved tightly and the source convention matches.
    assert meta["linear_seed_linear_residual_rel"] < 1e-8
    assert meta["linear_seed_source_error_rel"] < 1e-10
    assert meta["initial_guess"] == "linear_phasor"


def test_linear_phasor_seed_direct_matches_gmres(
    problem: exp08.FullIPMPumpProblem,
) -> None:
    X_gmres, _ = exp08.build_linear_phasor_seed(problem, method="gmres", rtol=1e-12)
    X_direct, meta = exp08.build_linear_phasor_seed(problem, method="direct")
    assert meta["linear_seed_method"] == "direct"
    assert np.allclose(X_gmres, X_direct, atol=1e-9)


# ---------------------------------------------------------------------------
# adaptive continuation
# ---------------------------------------------------------------------------

def test_adaptive_continuation_matches_fixed_continuation(
    problem: exp08.FullIPMPumpProblem,
    solver: exp08.HarmonicNewtonKrylovSolver,
) -> None:
    X_fixed, reports_fixed = solver.solve_continuation(
        problem, continuation_steps=20, x_init=None
    )
    assert reports_fixed[-1].converged

    X_seed, _ = exp08.build_linear_phasor_seed(problem, rtol=1e-10)
    X_adapt, reports_adapt, trace = solver.solve_adaptive_continuation(
        problem,
        X_seed,
        initial_step=1.0,
        min_step=0.05,
        growth=1.5,
        shrink=0.5,
        fallback_fixed_steps=20,
    )

    assert reports_adapt[-1].converged
    assert trace.mode == "adaptive"
    assert trace.accepted_lambdas[-1] == pytest.approx(1.0)
    assert not trace.fallback_used
    # Both paths land on the same harmonic-balance fixed point.
    assert np.max(np.abs(X_fixed - X_adapt)) < 1e-9
    # The seeded adaptive path takes far fewer total Newton steps than fixed.
    assert sum(r.newton_iterations for r in reports_adapt) < sum(
        r.newton_iterations for r in reports_fixed
    )


def test_adaptive_continuation_shrinks_then_falls_back() -> None:
    # Cap Newton iterations so a single full-scale step at a stiff (near-fold)
    # drive cannot converge, forcing the adaptive loop to shrink the step. With
    # min_step=0.6 the first shrink (1.0 -> 0.5) underflows, so it falls back to
    # graded fixed continuation, which does reach the solution.
    stiff_settings = _settings()
    stiff_settings.max_newton = 4
    solver = exp08.HarmonicNewtonKrylovSolver(stiff_settings)
    problem = _build_problem(pump_current=2.0)

    X, reports, trace = solver.solve_adaptive_continuation(
        problem,
        None,
        initial_step=1.0,
        min_step=0.6,
        growth=1.5,
        shrink=0.5,
        fallback_fixed_steps=20,
    )
    assert trace.failed_attempts >= 1
    assert trace.fallback_used
    assert reports[-1].converged  # fixed fallback recovers the solution


# ---------------------------------------------------------------------------
# warm-start direct solve (promote-from-pump-dir traversal)
# ---------------------------------------------------------------------------

def test_warm_start_direct_beats_cold_continuation(
    solver: exp08.HarmonicNewtonKrylovSolver,
) -> None:
    base = _build_problem(pump_current=0.30)
    X_base, base_reports = solver.solve_continuation(base, continuation_steps=20)
    assert base_reports[-1].converged

    # Neighboring map point at slightly higher pump power.
    nxt = _build_problem(pump_current=0.33)

    X_warm, warm_reports = solver.solve_direct(nxt, X_base)
    X_cold, cold_reports = solver.solve_continuation(nxt, continuation_steps=20)

    assert warm_reports[-1].converged
    assert cold_reports[-1].converged
    # Same solution either way.
    assert np.max(np.abs(X_warm - X_cold)) < 1e-9
    # Warm start is a single full-scale solve and uses fewer Newton steps.
    assert len(warm_reports) == 1
    assert warm_reports[-1].newton_iterations < sum(
        r.newton_iterations for r in cold_reports
    )


def test_warm_start_traversal_chains_across_power_steps(
    solver: exp08.HarmonicNewtonKrylovSolver,
) -> None:
    # Emulate a one-column map traversal: seed the first point, then warm-start
    # each subsequent (higher-power) point from the previous solution.
    currents = [0.20, 0.25, 0.30, 0.35]
    X_prev: np.ndarray | None = None
    newton_per_point: list[int] = []

    for idx, current in enumerate(currents):
        prob = _build_problem(pump_current=current)
        if X_prev is None:
            X_seed, _ = exp08.build_linear_phasor_seed(prob, rtol=1e-10)
            X_prev, reports, _ = solver.solve_adaptive_continuation(
                prob,
                X_seed,
                initial_step=1.0,
                min_step=0.05,
                growth=1.5,
                shrink=0.5,
                fallback_fixed_steps=20,
            )
        else:
            X_prev, reports = solver.solve_direct(prob, X_prev)
        assert reports[-1].converged
        newton_per_point.append(sum(r.newton_iterations for r in reports))

    # Warm-started points (after the first) stay cheap as the map advances.
    assert all(n <= newton_per_point[0] for n in newton_per_point[1:])
