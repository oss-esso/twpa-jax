"""Model-order-reduction hooks for later conversion sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReducedModelPlaceholder:
    frequencies_hz: np.ndarray
    response: np.ndarray
    status: str = "scaffolded_only"


def build_full_order_conversion_system(matrix: np.ndarray) -> np.ndarray:
    """Return a full-order matrix for MOR sampling."""
    return np.asarray(matrix)


def sample_transfer_function(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Sample a transfer function by solving a full-order linear system."""
    return np.linalg.solve(matrix, rhs)


def fit_reduced_model(frequencies_hz: np.ndarray, response: np.ndarray) -> ReducedModelPlaceholder:
    """Store samples for a future vector-fitting/reduced-basis implementation."""
    return ReducedModelPlaceholder(np.asarray(frequencies_hz), np.asarray(response))


def evaluate_reduced_model(model: ReducedModelPlaceholder, frequencies_hz: np.ndarray) -> np.ndarray:
    """Nearest-neighbor placeholder evaluation for tests and API readiness."""
    idx = np.searchsorted(model.frequencies_hz, frequencies_hz, side="left")
    idx = np.clip(idx, 0, len(model.frequencies_hz) - 1)
    return model.response[idx]
