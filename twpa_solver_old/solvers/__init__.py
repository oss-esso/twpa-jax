"""Nonlinear solver wrappers."""

from __future__ import annotations

from twpa_solver_old.solvers.base import SolverResult
from twpa_solver_old.solvers.preconditioners import build_linear_passive_preconditioner
from twpa_solver_old.solvers.scipy_least_squares import solve_least_squares

__all__ = ["SolverResult", "build_linear_passive_preconditioner", "solve_least_squares"]
