from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from twpa_solver.solvers.jax_dense_newton import solve_jax_dense_newton


def test_jax_dense_newton_solves_tiny_problem() -> None:
    def residual(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([x[0] ** 2 - 9.0])

    result = solve_jax_dense_newton(residual, np.asarray([2.0]))
    assert result.success
    assert abs(result.solution[0] - 3.0) < 1e-6
