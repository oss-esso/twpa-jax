"""Tests for the correctly-sited singularity measurement added in
``docs/development/arclength_fold_resolution_plan.md`` Phase 0: the
``solve_arclength`` ``on_step`` hook, the shift-invert eigenvalue estimator,
and the fold-vs-branch-point ``bordered_conditioning`` diagnostic.

Reuses the tiny one-node LC + Josephson fixture from
``test_advanced_continuation.py`` so the full Newton-Krylov/arclength stack
runs in milliseconds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.pump.singularity import (  # noqa: E402
    bordered_conditioning,
    jacobian_min_eigenvalue_with_estimator,
)


def _build_problem(pump_current: float, *, modes=(1, 2, 3), omega: float = 0.37):
    C = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    G = sp.csr_matrix(np.array([[0.01]], dtype=np.complex128))
    K = sp.csr_matrix(np.array([[1.0]], dtype=np.complex128))
    Bphi = sp.csr_matrix(np.array([[1.0]], dtype=np.float64))
    grid = exp08.HarmonicGrid(modes=np.array(list(modes)), nt=32, omega=omega)
    branch = exp08.JosephsonBranchArray(Ic=np.array([1.0], dtype=np.float64), phi0=1.0)
    return exp08.FullIPMPumpProblem(
        C=C, G=G, K=K, Bphi=Bphi, branch=branch, grid=grid,
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


def test_on_step_fires_only_on_accepted_steps() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    calls: list[tuple[float, float]] = []

    def on_step(Xc, lamc, step_size, Xdot, lam_dot, state_scale):
        calls.append((float(lamc), float(step_size)))

    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
        on_step=on_step,
    )
    assert info["reached_target"]
    assert len(calls) > 0
    # info["steps"] counts every outer loop pass, including halved retries;
    # on_step only fires on accepted ones, so it can never exceed that count.
    assert len(calls) <= info["steps"]
    # This fixture's branch is smooth (no fold) at pump_current=0.6, so every
    # accepted lambda strictly increases; the last accepted point is the one
    # that crosses (or lands on) target_lam=1.0, which the corrector then
    # interpolates back from.
    lambdas = [c[0] for c in calls]
    assert all(b > a for a, b in zip(lambdas, lambdas[1:]))
    assert lambdas[-1] >= 1.0
    if len(lambdas) > 1:
        assert lambdas[-2] < 1.0


def test_on_step_disabled_by_default() -> None:
    # Existing callers that never pass on_step must see no behavior change.
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
    )
    assert info["reached_target"]
    assert abs(lam - 1.0) < 1e-9


def test_shift_invert_matches_dense_reference() -> None:
    # A wider mode set (dim_real=2*5*1=10) gives ARPACK's shift-invert eigs a
    # comfortable margin over its ncv bookkeeping so it actually takes the
    # shift_invert branch rather than falling back.
    problem = _build_problem(pump_current=0.4, modes=(1, 2, 3, 4, 5))
    solver = _solver()
    X, rep = solver.solve_one(problem, problem.zeros(), 0.7)
    assert rep.converged

    tangent = problem.tangent_state(X)
    spectral = problem.spectral_tangent_state(tangent)
    dense = problem.real_coupled_jacobian(spectral).toarray()
    eigvals = np.linalg.eigvals(dense)
    ref = eigvals[np.argmin(np.abs(eigvals))]

    value, estimator = jacobian_min_eigenvalue_with_estimator(problem, X, 0.7, iters=30)
    assert estimator == "shift_invert"
    assert math.isclose(value, ref.real, rel_tol=1e-6, abs_tol=1e-8)


def test_bordered_conditioning_finite_at_converged_point() -> None:
    problem = _build_problem(pump_current=0.6)
    solver = _solver()
    captured: dict[str, Any] = {}

    def on_step(Xc, lamc, step_size, Xdot, lam_dot, state_scale):
        captured["Xc"] = Xc
        captured["lamc"] = lamc
        captured["Xdot"] = Xdot
        captured["lam_dot"] = lam_dot
        captured["state_scale"] = state_scale

    X, lam, info = solver.solve_arclength(
        problem, problem.zeros(), 0.0, ds=0.1, target_lam=1.0, max_steps=80,
        on_step=on_step,
    )
    assert info["reached_target"]
    assert captured

    cond = bordered_conditioning(
        problem, captured["Xc"], captured["lamc"], captured["Xdot"],
        captured["lam_dot"], captured["state_scale"], iters=20,
    )
    assert math.isfinite(cond)
    assert cond > 0.0
