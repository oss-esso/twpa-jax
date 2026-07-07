"""Basic Anderson acceleration for fixed-point iterations."""

from __future__ import annotations

from typing import Callable

import numpy as np


def anderson_accelerate(
    fixed_point: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    memory: int = 3,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, list[float]]:
    """Run a small Anderson-accelerated fixed-point iteration."""
    x = np.asarray(x0, dtype=float)
    xs: list[np.ndarray] = []
    fs: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    history: list[float] = []
    for _ in range(max_iterations):
        g = fixed_point(x)
        f = g - x
        norm = float(np.linalg.norm(f))
        history.append(norm)
        if norm <= tolerance:
            return g, history
        xs.append(x)
        fs.append(f)
        gs.append(g)
        xs = xs[-memory:]
        fs = fs[-memory:]
        gs = gs[-memory:]
        if len(fs) == 1:
            x = g
            continue
        fmat = np.column_stack(fs)
        gram = fmat.T @ fmat
        ones = np.ones((len(fs), 1))
        kkt = np.block([[gram, ones], [ones.T, np.zeros((1, 1))]])
        rhs = np.zeros(len(fs) + 1)
        rhs[-1] = 1.0
        try:
            coeff = np.linalg.solve(kkt, rhs)[:-1]
            x = sum(weight * value for weight, value in zip(coeff, gs))
        except np.linalg.LinAlgError:
            x = g
    return x, history
