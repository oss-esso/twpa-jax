from __future__ import annotations

import numpy as np

from twpa_solver.solvers.scipy_least_squares import solve_least_squares
from twpa_solver.solvers.scipy_root import solve_root


def test_scipy_least_squares_solves_toy_nonlinear_problem() -> None:
    result = solve_least_squares(lambda x: np.asarray([x[0] ** 2 - 4.0]), np.asarray([1.0]))
    assert result.success
    assert abs(abs(result.solution[0]) - 2.0) < 1e-6


def test_scipy_root_solves_toy_problem() -> None:
    result = solve_root(lambda x: np.asarray([x[0] - 2.0]), np.asarray([0.0]))
    assert result.success
    assert abs(result.solution[0] - 2.0) < 1e-6
