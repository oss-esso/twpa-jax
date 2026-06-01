"""
twpa.solvers
============

Numerical solvers for the JAX-backed TWPA simulator.

This package contains:

    hb_solver
        Dense Newton, residual packing, and dense linear-solve helpers used by
        the reference harmonic-balance backend.

    continuation
        Parameter-continuation utilities for pump power, pump current, and
        other operating-point sweeps.

    newton_krylov
        Matrix-free Newton-Krylov backend for large structured HB systems.

    linear_solvers
        Shared direct/iterative linear solver wrappers.

    preconditioners
        Structured preconditioners for block-banded and ladder-like systems.

    block_banded
        Block-banded matrix containers and operations for industrial-scale
        100 mm / 20,000-cell simulations.

Imports are intentionally lazy so importing ``twpa.solvers`` does not trigger
heavy JAX compilation or optional sparse-solver imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "hb_solver": "twpa.solvers.hb_solver",
    "continuation": "twpa.solvers.continuation",
    "newton_krylov": "twpa.solvers.newton_krylov",
    "linear_solvers": "twpa.solvers.linear_solvers",
    "preconditioners": "twpa.solvers.preconditioners",
    "block_banded": "twpa.solvers.block_banded",
}


_LAZY_SYMBOLS = {
    # hb_solver.py
    "NewtonStatus": ("twpa.solvers.hb_solver", "NewtonStatus"),
    "LinearSolveMethod": ("twpa.solvers.hb_solver", "LinearSolveMethod"),
    "DenseNewtonConfig": ("twpa.solvers.hb_solver", "DenseNewtonConfig"),
    "NewtonIterationRecord": ("twpa.solvers.hb_solver", "NewtonIterationRecord"),
    "NewtonSolveReport": ("twpa.solvers.hb_solver", "NewtonSolveReport"),
    "DenseLinearSolveResult": ("twpa.solvers.hb_solver", "DenseLinearSolveResult"),
    "HBSolverResult": ("twpa.solvers.hb_solver", "HBSolverResult"),
    "dense_linear_solve": ("twpa.solvers.hb_solver", "dense_linear_solve"),
    "dense_newton_solve": ("twpa.solvers.hb_solver", "dense_newton_solve"),
    "pack_unknown_tree": ("twpa.solvers.hb_solver", "pack_unknown_tree"),
    "pack_residual_tree": ("twpa.solvers.hb_solver", "pack_residual_tree"),

    # continuation.py
    "ContinuationStatus": ("twpa.solvers.continuation", "ContinuationStatus"),
    "ContinuationStepReport": ("twpa.solvers.continuation", "ContinuationStepReport"),
    "ContinuationResult": ("twpa.solvers.continuation", "ContinuationResult"),
    "ContinuationSolverConfig": (
        "twpa.solvers.continuation",
        "ContinuationSolverConfig",
    ),
    "ContinuationSchedule": ("twpa.solvers.continuation", "ContinuationSchedule"),
    "make_continuation_schedule": (
        "twpa.solvers.continuation",
        "make_continuation_schedule",
    ),
    "solve_continuation": ("twpa.solvers.continuation", "solve_continuation"),

    # newton_krylov.py
    "NewtonKrylovConfig": ("twpa.solvers.newton_krylov", "NewtonKrylovConfig"),
    "NewtonKrylovResult": ("twpa.solvers.newton_krylov", "NewtonKrylovResult"),
    "newton_krylov_solve": ("twpa.solvers.newton_krylov", "newton_krylov_solve"),

    # linear_solvers.py
    "LinearOperator": ("twpa.solvers.linear_solvers", "LinearOperator"),
    "IterativeLinearSolveConfig": (
        "twpa.solvers.linear_solvers",
        "IterativeLinearSolveConfig",
    ),
    "IterativeLinearSolveResult": (
        "twpa.solvers.linear_solvers",
        "IterativeLinearSolveResult",
    ),
    "solve_linear_system": ("twpa.solvers.linear_solvers", "solve_linear_system"),

    # preconditioners.py
    "PreconditionerKind": ("twpa.solvers.preconditioners", "PreconditionerKind"),
    "PreconditionerConfig": ("twpa.solvers.preconditioners", "PreconditionerConfig"),
    "Preconditioner": ("twpa.solvers.preconditioners", "Preconditioner"),
    "build_preconditioner": ("twpa.solvers.preconditioners", "build_preconditioner"),

    # block_banded.py
    "BlockBandedMatrix": ("twpa.solvers.block_banded", "BlockBandedMatrix"),
    "BlockBandedConfig": ("twpa.solvers.block_banded", "BlockBandedConfig"),
    "build_block_banded_from_dense": (
        "twpa.solvers.block_banded",
        "build_block_banded_from_dense",
    ),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import solver submodules and commonly used symbols.
    """
    if name in _LAZY_SUBMODULES:
        module = import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module

    if name in _LAZY_SYMBOLS:
        module_name, symbol_name = _LAZY_SYMBOLS[name]
        module = import_module(module_name)
        symbol = getattr(module, symbol_name)
        globals()[name] = symbol
        return symbol

    raise AttributeError(f"module 'twpa.solvers' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_SYMBOLS))


__all__ = [
    "hb_solver",
    "continuation",
    "newton_krylov",
    "linear_solvers",
    "preconditioners",
    "block_banded",
    *_LAZY_SYMBOLS.keys(),
]