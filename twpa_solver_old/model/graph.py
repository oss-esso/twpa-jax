"""Graph helpers for node-flux circuit models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUND = -1


@dataclass(frozen=True)
class Branch:
    """Directed branch between two non-ground nodes or ground."""

    start: int
    end: int
    label: str = ""


def incidence_matrix(num_nodes: int, branches: list[Branch]) -> np.ndarray:
    """Build a reduced node incidence matrix with ground represented by -1."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    matrix = np.zeros((num_nodes, len(branches)), dtype=float)
    for col, branch in enumerate(branches):
        for node in (branch.start, branch.end):
            if node != GROUND and not 0 <= node < num_nodes:
                raise ValueError(f"branch node {node} outside 0..{num_nodes - 1}")
        if branch.start != GROUND:
            matrix[branch.start, col] += 1.0
        if branch.end != GROUND:
            matrix[branch.end, col] -= 1.0
    return matrix


def branch_fluxes(node_flux: np.ndarray, incidence: np.ndarray) -> np.ndarray:
    """Return branch flux differences D^T phi."""
    phi = np.asarray(node_flux, dtype=float)
    return incidence.T @ phi
