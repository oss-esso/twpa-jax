"""Dense JAX Newton solver for small residual systems."""

from __future__ import annotations

import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from twpa_solver_old.solvers.base import SolverResult, result_from_residual


def solve_jax_dense_newton(
    residual: Callable[[jnp.ndarray], jnp.ndarray],
    x0: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 25,
) -> SolverResult:
    """Dense Newton method using jax.jacfwd and damped backtracking."""
    start = time.perf_counter()
    x = jnp.asarray(x0, dtype=jnp.float64)
    jac = jax.jacfwd(residual)
    linear_solves = 0
    iterations = 0
    message = "maximum iterations reached"
    for iteration in range(max_iterations):
        iterations = iteration + 1
        r = residual(x)
        if float(jnp.max(jnp.abs(r))) <= tolerance:
            message = "converged"
            break
        j = jac(x)
        step = jnp.linalg.solve(j, -r)
        linear_solves += 1
        old_norm = float(jnp.linalg.norm(r))
        damping = 1.0
        accepted = False
        for _ in range(12):
            trial = x + damping * step
            if float(jnp.linalg.norm(residual(trial))) < old_norm:
                x = trial
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            x = x + 1e-3 * step
    runtime = time.perf_counter() - start
    final = np.asarray(residual(x), dtype=float)
    return result_from_residual(
        solution=np.asarray(x, dtype=float),
        residual=final,
        success=float(np.max(np.abs(final))) <= tolerance,
        message=message,
        solver_name="jax-dense-newton",
        runtime_s=runtime,
        tolerance=tolerance,
        num_iterations=iterations,
        num_jacobian_evals=iterations,
        num_linear_solves=linear_solves,
    )
