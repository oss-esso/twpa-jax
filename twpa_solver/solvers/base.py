"""Shared solver result containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SolverResult:
    status: str
    success: bool
    message: str
    solution: np.ndarray
    residual_norm_l2: float
    residual_norm_inf: float
    num_iterations: int
    num_residual_evals: int
    num_jacobian_evals: int
    num_linear_solves: int
    runtime_s: float
    solver_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["solution"] = np.asarray(self.solution).tolist()
        return data


def classify_residual(norm_inf: float, success: bool, tolerance: float) -> str:
    """Map solver flags and residuals to a convergence class."""
    if success and norm_inf <= tolerance:
        return "converged"
    if np.isfinite(norm_inf):
        return "diagnostic"
    return "failed"


def result_from_residual(
    *,
    solution: np.ndarray,
    residual: np.ndarray,
    success: bool,
    message: str,
    solver_name: str,
    runtime_s: float,
    tolerance: float,
    num_iterations: int = 0,
    num_residual_evals: int = 0,
    num_jacobian_evals: int = 0,
    num_linear_solves: int = 0,
    metadata: dict[str, Any] | None = None,
) -> SolverResult:
    """Build a SolverResult from a final residual vector."""
    res = np.asarray(residual, dtype=float)
    norm_l2 = float(np.linalg.norm(res))
    norm_inf = float(np.max(np.abs(res))) if res.size else 0.0
    status = classify_residual(norm_inf, success, tolerance)
    return SolverResult(
        status=status,
        success=bool(status == "converged"),
        message=message,
        solution=np.asarray(solution, dtype=float),
        residual_norm_l2=norm_l2,
        residual_norm_inf=norm_inf,
        num_iterations=int(num_iterations),
        num_residual_evals=int(num_residual_evals),
        num_jacobian_evals=int(num_jacobian_evals),
        num_linear_solves=int(num_linear_solves),
        runtime_s=float(runtime_s),
        solver_name=solver_name,
        metadata=dict(metadata or {}),
    )
