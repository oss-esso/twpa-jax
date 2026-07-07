"""SciPy root and Newton-Krylov wrappers."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.optimize import newton_krylov, root

from twpa_solver_old.solvers.base import SolverResult, result_from_residual


def solve_root(
    residual: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    method: str = "hybr",
    tolerance: float = 1e-8,
) -> SolverResult:
    """Solve with scipy.optimize.root."""
    start = time.perf_counter()
    out = root(residual, np.asarray(x0, dtype=float), method=method, tol=tolerance)
    runtime = time.perf_counter() - start
    return result_from_residual(
        solution=out.x,
        residual=residual(out.x),
        success=bool(out.success),
        message=str(out.message),
        solver_name=f"scipy-root:{method}",
        runtime_s=runtime,
        tolerance=tolerance,
        num_residual_evals=int(getattr(out, "nfev", 0)),
        num_jacobian_evals=int(getattr(out, "njev", 0) or 0),
    )


def solve_newton_krylov(
    residual: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    tolerance: float = 1e-8,
    maxiter: int = 50,
) -> SolverResult:
    """Solve with scipy.optimize.newton_krylov as a CPU matrix-free baseline."""
    start = time.perf_counter()
    try:
        sol = newton_krylov(residual, np.asarray(x0, dtype=float), f_tol=tolerance, maxiter=maxiter)
        success = True
        message = "newton_krylov converged"
    except Exception as exc:
        sol = np.asarray(x0, dtype=float)
        success = False
        message = f"newton_krylov failed: {exc}"
    runtime = time.perf_counter() - start
    return result_from_residual(
        solution=sol,
        residual=residual(sol),
        success=success,
        message=message,
        solver_name="scipy-newton-krylov",
        runtime_s=runtime,
        tolerance=tolerance,
        num_iterations=maxiter if not success else 0,
    )
