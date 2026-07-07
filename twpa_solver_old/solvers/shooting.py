"""Small-system periodic shooting validation solver."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


def solve_periodic_shooting(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0_guess: np.ndarray,
    period_s: float,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Solve x(T; x0) - x0 = 0 for a small periodic orbit."""

    def residual(x0: np.ndarray) -> np.ndarray:
        sol = solve_ivp(rhs, (0.0, period_s), x0, rtol=1e-8, atol=1e-10)
        return sol.y[:, -1] - x0

    out = root(residual, np.asarray(x0_guess, dtype=float), tol=tolerance)
    if not out.success:
        raise RuntimeError(str(out.message))
    return np.asarray(out.x, dtype=float)
