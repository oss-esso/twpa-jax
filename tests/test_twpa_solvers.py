from __future__ import annotations

import numpy as np

from twpa_solver.residuals.two_tone import make_two_tone_grid
from twpa_solver.solvers.anderson import anderson_accelerate
from twpa_solver.solvers.arclength import ArclengthPoint, trace_scalar_branch
from twpa_solver.solvers.deflation import cluster_solutions
from twpa_solver.solvers.mor import fit_reduced_model, evaluate_reduced_model
from twpa_solver.solvers.scipy_least_squares import solve_least_squares
from twpa_solver.solvers.shooting import solve_periodic_shooting


def test_arclength_traces_toy_fold_scaffold() -> None:
    points = trace_scalar_branch(
        lambda u, lam: u**3 - lam * u + 0.1,
        ArclengthPoint(-0.1, 0.0, 0.099),
        (1.0, 0.0),
        0.02,
        2,
    )
    assert len(points) == 3
    assert all(np.isfinite(point.residual) for point in points)


def test_shooting_for_stationary_periodic_state() -> None:
    state = solve_periodic_shooting(lambda _t, x: -x, np.asarray([0.1]), 1.0)
    assert abs(state[0]) < 1e-8


def test_anderson_accelerates_simple_fixed_point() -> None:
    sol, history = anderson_accelerate(lambda x: np.cos(x), np.asarray([0.5]), max_iterations=20)
    assert abs(sol[0] - np.cos(sol[0])) < 1e-6
    assert history


def test_deflation_clusters_multistart_solutions() -> None:
    result = solve_least_squares(lambda x: np.asarray([x[0] - 1.0]), np.asarray([0.0]))
    clusters = cluster_solutions([result, result])
    assert len(clusters) == 1
    assert clusters[0].members == 2


def test_mor_placeholder_roundtrip_and_two_tone_grid() -> None:
    model = fit_reduced_model(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]))
    np.testing.assert_allclose(evaluate_reduced_model(model, np.asarray([1.2])), [4.0])
    grid = make_two_tone_grid(6e9, 5e9, 1)
    assert len(grid) == 9
