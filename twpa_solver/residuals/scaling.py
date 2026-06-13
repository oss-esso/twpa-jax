"""Residual scaling helpers."""

from __future__ import annotations

import numpy as np


def safe_scale(values: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
    """Divide by a positive scale without hiding invalid scales."""
    scale_arr = np.asarray(scale, dtype=float)
    if np.any(scale_arr <= 0.0):
        raise ValueError("scale must be positive")
    return np.asarray(values, dtype=float) / scale_arr
