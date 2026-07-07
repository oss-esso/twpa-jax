from __future__ import annotations

import numpy as np

from twpa_solver_old.solvers.continuation import continue_parameter, snake_grid_indices
from twpa_solver_old.solvers.scipy_least_squares import solve_least_squares


def test_continuation_reaches_parameter_path() -> None:
    results = continue_parameter(
        [1.0, 4.0],
        lambda value: lambda x: np.asarray([x[0] ** 2 - value]),
        np.asarray([0.8]),
        lambda residual, x0: solve_least_squares(residual, x0),
    )
    assert all(result.success for result in results)
    assert abs(results[-1].solution[0] - 2.0) < 1e-6


def test_snake_grid_indices() -> None:
    assert snake_grid_indices(2, 3) == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
