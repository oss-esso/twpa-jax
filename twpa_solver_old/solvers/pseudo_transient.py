"""Pseudo-transient continuation globalizer."""

from __future__ import annotations

from typing import Callable

import numpy as np

from twpa_solver_old.solvers.base import SolverResult
from twpa_solver_old.solvers.scipy_least_squares import solve_least_squares


def solve_pseudo_transient(
    residual: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    tolerance: float = 1e-7,
    max_steps: int = 8,
) -> SolverResult:
    """Simple pseudo-transient wrapper that damps the initial guess before LS."""
    x = np.asarray(x0, dtype=float).copy()
    history = []
    dt = 1e-2
    for _ in range(max_steps):
        r = residual(x)
        norm = float(np.linalg.norm(r))
        history.append({"dt": dt, "residual_l2": norm})
        if np.max(np.abs(r)) <= 10.0 * tolerance:
            break
        x = x - dt * r[: x.size]
        dt = min(dt * 1.5, 1.0)
    result = solve_least_squares(residual, x, tolerance=tolerance)
    result.metadata["pseudo_transient_history"] = history
    return result
