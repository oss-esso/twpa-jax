"""Matrix-free Newton-Krylov scaffold using JAX JVPs and SciPy GMRES."""

from __future__ import annotations

import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from twpa_solver_old.solvers.base import SolverResult, result_from_residual


def solve_jax_newton_krylov(
    residual: Callable[[jnp.ndarray], jnp.ndarray],
    x0: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 20,
    gmres_rtol: float = 1e-6,
    preconditioner: LinearOperator | None = None,
) -> SolverResult:
    """Newton-Krylov solve without explicitly forming a dense Jacobian."""
    start = time.perf_counter()
    x = jnp.asarray(x0, dtype=jnp.float64)
    linear_solves = 0
    message = "maximum iterations reached"
    gmres_iterations = 0
    for iteration in range(max_iterations):
        r = residual(x)
        if float(jnp.max(jnp.abs(r))) <= tolerance:
            message = "converged"
            break
        r_np = np.asarray(r, dtype=float)

        def matvec(v: np.ndarray) -> np.ndarray:
            _, jvp = jax.jvp(residual, (x,), (jnp.asarray(v, dtype=jnp.float64),))
            return np.array(jvp, dtype=float, copy=True)

        op = LinearOperator((r_np.size, r_np.size), matvec=matvec, dtype=float)
        def _count_gmres(_residual_norm: float) -> None:
            nonlocal gmres_iterations
            gmres_iterations += 1

        step, info = gmres(
            op,
            -r_np,
            rtol=gmres_rtol,
            atol=0.0,
            M=preconditioner,
            callback=_count_gmres,
            callback_type="pr_norm",
        )
        linear_solves += 1
        if info != 0:
            message = f"gmres did not converge: info={info}"
            break
        damping = 1.0
        old = float(jnp.linalg.norm(r))
        for _ in range(10):
            trial = x + damping * jnp.asarray(step)
            if float(jnp.linalg.norm(residual(trial))) < old:
                x = trial
                break
            damping *= 0.5
    runtime = time.perf_counter() - start
    final = np.asarray(residual(x), dtype=float)
    return result_from_residual(
        solution=np.asarray(x, dtype=float),
        residual=final,
        success=float(np.max(np.abs(final))) <= tolerance,
        message=message,
        solver_name="jax-newton-krylov-jvp-gmres",
        runtime_s=runtime,
        tolerance=tolerance,
        num_iterations=iteration + 1,
        num_linear_solves=linear_solves,
        metadata={
            "gmres_rtol": gmres_rtol,
            "preconditioned": preconditioner is not None,
            "last_gmres_iterations": gmres_iterations,
        },
    )
