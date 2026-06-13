"""Gain conversion helpers."""

from __future__ import annotations

import numpy as np


def power_gain_db(s_value: complex) -> float:
    """Return 10 log10 |S|^2."""
    return float(10.0 * np.log10(max(abs(s_value) ** 2, 1e-300)))
