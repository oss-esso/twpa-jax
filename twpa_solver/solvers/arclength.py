"""Pseudo-arclength continuation scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class ArclengthPoint:
    u: float
    parameter: float
    residual: float


def trace_scalar_branch(
    residual: Callable[[float, float], float],
    start: ArclengthPoint,
    tangent: tuple[float, float],
    step: float,
    count: int,
) -> list[ArclengthPoint]:
    """Predictor-corrector pseudo-arclength continuation for scalar equations."""
    points = [start]
    du, dlambda = tangent
    tangent_norm = float(np.hypot(du, dlambda))
    du /= tangent_norm
    dlambda /= tangent_norm
    for _ in range(count):
        prev = points[-1]
        pred_u = prev.u + step * du
        pred_l = prev.parameter + step * dlambda

        def augmented(z: np.ndarray) -> np.ndarray:
            u, lam = z
            arc = (u - pred_u) * du + (lam - pred_l) * dlambda
            return np.asarray([residual(float(u), float(lam)), arc])

        out = root(augmented, np.asarray([pred_u, pred_l]))
        u_new, l_new = out.x
        points.append(ArclengthPoint(float(u_new), float(l_new), float(residual(u_new, l_new))))
    return points
