"""Regression test for the adaptive-continuation fixed-step fallback.

``solve_adaptive_continuation`` (src/twpa_solver/pump/solver.py) bisects the
source-scale lambda toward 1.0; when the step shrinks below ``min_step`` it
falls back to a fixed-step ladder via ``solve_continuation``. That fallback
used to pass the ORIGINAL seed (``x_init``) and implicitly restart the ladder
at lambda=0, discarding every converged state the adaptive phase had already
reached. Debugging a real map column (fp=7.329 GHz, -28.25 dBm,
outputs/measurement_match_debug_01/column_debug_col3_trim) showed this
concretely: adaptive bisection reached lambda=0.9375 well-converged, then the
old fallback restarted the fixed ladder at lambda=0.05 and burned the wall-time
budget re-deriving cheap low-lambda states before it could get back near the
fold. The fix resumes the ladder from (X_current, lambda_current) instead.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.pump.problem import (
    FullPumpProblem,
    HarmonicGrid,
    JosephsonBranchArray,
)
from twpa_solver.pump.solver import HarmonicNewtonKrylovSolver, NewtonKrylovSettings


def _build_problem(pump_current: float, *, omega: float = 0.37) -> FullPumpProblem:
    """One node, one Josephson branch to ground: C x'' + G x' + K x + Ic sin(x)."""
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    G = sp.csr_matrix(np.array([[0.01]], dtype=np.complex128))
    K = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = HarmonicGrid(modes=np.array([1, 2, 3]), nt=16, omega=omega)
    branch = JosephsonBranchArray(Ic=np.array([1.0], dtype=np.float64), phi0=1.0)
    return FullPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi, branch=branch, grid=grid,
        pump_node_index=0, pump_current_a=pump_current,
    )


def _settings(max_newton: int = 30) -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-9,
        max_newton=max_newton,
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


def test_solve_continuation_lambda_start_resumes_span() -> None:
    """``lambda_start`` should march only the remaining span from a given X."""
    problem = _build_problem(pump_current=0.30)
    solver = HarmonicNewtonKrylovSolver(_settings())

    X_full, reports_full = solver.solve_continuation(problem, continuation_steps=20)
    assert reports_full[-1].converged

    # Converge directly to lambda=0.5, then resume 0.5 -> 1.0 in 4 steps and
    # check it lands on the same fixed point as the full 0 -> 1 ladder.
    X_at_half, report_half = solver.solve_one(problem, problem.zeros(), 0.5)
    assert report_half.converged
    X_final, reports_final = solver.solve_continuation(
        problem, continuation_steps=4, x_init=X_at_half, lambda_start=0.5,
    )
    assert reports_final[-1].converged
    assert reports_final[-1].source_scale == pytest.approx(1.0)
    assert all(r.source_scale > 0.5 for r in reports_final)
    assert np.max(np.abs(X_final - X_full)) < 1e-8


def test_adaptive_continuation_fallback_resumes_not_restarts() -> None:
    """The fixed-step fallback must resume from lambda_current, not lambda=0.

    Cap Newton iterations so the ladder makes real partial progress (some
    accepted lambda > 0) before the remaining steps become too stiff and the
    bisection underflows min_step, forcing the fallback.
    """
    stiff_settings = _settings(max_newton=3)
    solver = HarmonicNewtonKrylovSolver(stiff_settings)
    problem = _build_problem(pump_current=2.0)

    seen_calls: list[dict] = []
    original = solver.solve_continuation

    def spy(*args, **kwargs):
        seen_calls.append(kwargs)
        return original(*args, **kwargs)

    solver.solve_continuation = spy  # type: ignore[method-assign]

    X, reports, trace = solver.solve_adaptive_continuation(
        problem, None,
        initial_step=1.0, min_step=0.1, growth=1.5, shrink=0.5,
        fallback_fixed_steps=20,
    )

    assert trace.fallback_used
    assert seen_calls, "fallback must call solve_continuation"
    fallback_kwargs = seen_calls[-1]

    last_adaptive_lambda = trace.accepted_lambdas[-1] if trace.accepted_lambdas else 0.0
    # The bug: fallback resumed from lambda=0 (x_init=None/original seed)
    # regardless of adaptive progress. The fix: it resumes from exactly the
    # best lambda the adaptive phase reached.
    assert fallback_kwargs["lambda_start"] == pytest.approx(last_adaptive_lambda)
    if last_adaptive_lambda > 0.0:
        # Genuine progress was made before falling back -- the regression
        # this test exists to catch requires lambda_start > 0.
        assert fallback_kwargs["lambda_start"] > 0.0
        assert fallback_kwargs["x_init"] is not None
        assert not np.allclose(fallback_kwargs["x_init"], problem.zeros())
    # The fallback ladder must only be sized for the remaining span, not the
    # full fallback_fixed_steps count computed from lambda=0.
    assert fallback_kwargs["continuation_steps"] <= 20
    if last_adaptive_lambda > 0.0:
        assert fallback_kwargs["continuation_steps"] < 20
