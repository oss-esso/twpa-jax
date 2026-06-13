"""Multistart and branch clustering hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from twpa_solver.solvers.base import SolverResult


@dataclass(frozen=True)
class BranchCluster:
    representative: np.ndarray
    members: int
    best_residual_inf: float


def cluster_solutions(
    results: list[SolverResult],
    *,
    relative_tolerance: float = 1e-5,
) -> list[BranchCluster]:
    """Cluster successful multistart solutions by relative Euclidean distance."""
    clusters: list[BranchCluster] = []
    for result in results:
        if not result.success:
            continue
        sol = result.solution
        matched = False
        for idx, cluster in enumerate(clusters):
            scale = max(np.linalg.norm(cluster.representative), 1.0)
            if np.linalg.norm(sol - cluster.representative) / scale <= relative_tolerance:
                clusters[idx] = BranchCluster(
                    cluster.representative,
                    cluster.members + 1,
                    min(cluster.best_residual_inf, result.residual_norm_inf),
                )
                matched = True
                break
        if not matched:
            clusters.append(BranchCluster(sol, 1, result.residual_norm_inf))
    return clusters


def run_multistart(
    residual: Callable[[np.ndarray], np.ndarray],
    guesses: list[np.ndarray],
    solve: Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray], SolverResult],
) -> tuple[list[SolverResult], list[BranchCluster]]:
    """Run several initial guesses and cluster converged branches."""
    results = [solve(residual, guess) for guess in guesses]
    return results, cluster_solutions(results)
