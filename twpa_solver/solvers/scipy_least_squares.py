"""SciPy least-squares nonlinear solver baseline."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from twpa_solver.solvers.base import SolverResult, result_from_residual


def solve_least_squares(
    residual: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    method: str = "trf",
    tolerance: float = 1e-7,
    max_nfev: int | None = None,
) -> SolverResult:
    """Solve R(x)=0 using scipy.optimize.least_squares."""
    start = time.perf_counter()
    scipy_result = least_squares(
        residual,
        np.asarray(x0, dtype=float),
        method=method,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_nfev,
    )
    runtime = time.perf_counter() - start
    return result_from_residual(
        solution=scipy_result.x,
        residual=residual(scipy_result.x),
        success=bool(scipy_result.success),
        message=str(scipy_result.message),
        solver_name=f"scipy-least-squares:{method}",
        runtime_s=runtime,
        tolerance=tolerance,
        num_iterations=int(getattr(scipy_result, "njev", 0) or 0),
        num_residual_evals=int(scipy_result.nfev),
        num_jacobian_evals=int(getattr(scipy_result, "njev", 0) or 0),
        metadata={"cost": float(scipy_result.cost), "optimality": float(scipy_result.optimality)},
    )
