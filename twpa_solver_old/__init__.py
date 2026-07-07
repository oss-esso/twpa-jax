"""First-principles modular TWPA solver package."""

from __future__ import annotations

from twpa_solver_old.model.ipm import (
    IPMConfig,
    build_ipm_jtwpa,
    build_ipm_jtwpa_old_constants_compact_surrogate,
    build_ipm_jtwpa_old_julia_parity,
    build_ipm_jtwpa_physical_coupler,
    build_ipm_jtwpa_reduced_marker,
)
from twpa_solver_old.solvers.base import SolverResult

__all__ = [
    "IPMConfig",
    "SolverResult",
    "build_ipm_jtwpa",
    "build_ipm_jtwpa_old_constants_compact_surrogate",
    "build_ipm_jtwpa_old_julia_parity",
    "build_ipm_jtwpa_physical_coupler",
    "build_ipm_jtwpa_reduced_marker",
]
