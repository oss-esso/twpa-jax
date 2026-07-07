"""Parameter continuation utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from twpa_solver_old.solvers.base import SolverResult


def continue_parameter(
    values: list[float],
    make_residual: Callable[[float], Callable[[np.ndarray], np.ndarray]],
    x0: np.ndarray,
    solve: Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray], SolverResult],
    *,
    csv_path: Path | None = None,
) -> list[SolverResult]:
    """Warm-start solves across a scalar parameter path."""
    x = np.asarray(x0, dtype=float)
    results: list[SolverResult] = []
    rows = []
    for value in values:
        result = solve(make_residual(float(value)), x)
        results.append(result)
        rows.append(
            {
                "parameter": value,
                "status": result.status,
                "residual_norm_inf": result.residual_norm_inf,
                "runtime_s": result.runtime_s,
            }
        )
        if result.success:
            x = result.solution
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return results


def snake_grid_indices(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Return row-wise snake traversal indices."""
    indices: list[tuple[int, int]] = []
    for row in range(n_rows):
        cols = range(n_cols) if row % 2 == 0 else range(n_cols - 1, -1, -1)
        indices.extend((row, col) for col in cols)
    return indices
