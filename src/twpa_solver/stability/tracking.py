"""Continuous matching of Floquet multipliers between pump powers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def track_multiplier_branches(
    previous: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """Return current multipliers ordered by nearest complex-plane branches.

    The matching cost is normalized by the local distance to the unit circle,
    which prevents a large-magnitude mode from automatically displacing a
    near-unit branch.  Eigenvector-overlap matching can be added by callers
    when vectors are available; this value-only matcher is deterministic and
    is the fallback used by the scan CLI.
    """
    old = np.asarray(previous, dtype=np.complex128).reshape(-1)
    new = np.asarray(current, dtype=np.complex128).reshape(-1)
    if old.size == 0:
        return new.copy()
    if new.size < old.size:
        raise ValueError("current multiplier set is smaller than previous set")
    scale = 1.0 + np.abs(old)[:, None] + np.abs(new)[None, :]
    cost = np.abs(old[:, None] - new[None, :]) / scale
    rows, columns = linear_sum_assignment(cost)
    ordered = np.empty_like(new)
    ordered[: old.size] = new[columns[np.argsort(rows)]]
    remaining = [index for index in range(new.size) if index not in set(columns)]
    if remaining:
        ordered[old.size :] = new[remaining]
    return ordered
