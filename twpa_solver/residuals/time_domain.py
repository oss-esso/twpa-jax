"""Time-domain residual helpers for validation solvers."""

from __future__ import annotations

import numpy as np


def second_order_state_rhs(
    x: np.ndarray,
    v: np.ndarray,
    mass: np.ndarray,
    force: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first-order RHS for M xddot + force = 0."""
    return v, -np.linalg.solve(mass, force)
