"""
twpa
====

JAX-backed simulation toolkit for superconducting traveling-wave parametric
amplifiers.

The package is organized in layers:

    twpa.core
        Frequency plans, layouts, harmonic-balance projection utilities,
        physical parameter containers.

    twpa.linear
        Pump-off linear cell/cascade/MNA/dispersion/coarsening tools.

    twpa.nonlinear
        Nonlinear kinetic-inductance models, harmonic balance, pump solves,
        small-signal linearization, and gain calculations.

    twpa.solvers
        Dense Newton, continuation, and linear-solve utilities.

    twpa.workflows
        Production workflows for industrial 100 mm simulations, calibration,
        and synthetic benchmarks.

This top-level __init__ intentionally keeps imports light. Heavy numerical
objects are lazy-loaded through __getattr__.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from twpa.core.numerics import enable_x64


enable_x64()

__version__ = "0.1.0"


_LAZY_SUBMODULES = {
    "core": "twpa.core",
    "linear": "twpa.linear",
    "nonlinear": "twpa.nonlinear",
    "solvers": "twpa.solvers",
    "workflows": "twpa.workflows",
}


_LAZY_SYMBOLS = {
    # Core
    "FrequencyPlan": ("twpa.core.frequency_plan", "FrequencyPlan"),
    "LineLayout": ("twpa.core.layout", "LineLayout"),
    "NonlinearParams": ("twpa.core.params", "NonlinearParams"),
    "SolverConfig": ("twpa.core.params", "SolverConfig"),

    # Nonlinear pump/gain
    "PumpDriveConfig": ("twpa.nonlinear.pump_hb_ladder", "PumpDriveConfig"),
    "PumpHBLadderConfig": ("twpa.nonlinear.pump_hb_ladder", "PumpHBLadderConfig"),
    "PumpHBLadderResult": ("twpa.nonlinear.pump_hb_ladder", "PumpHBLadderResult"),
    "solve_pump_hb_ladder": ("twpa.nonlinear.pump_hb_ladder", "solve_pump_hb_ladder"),

    "GainSolveConfig": ("twpa.nonlinear.gain", "GainSolveConfig"),
    "GainSweepConfig": ("twpa.nonlinear.gain", "GainSweepConfig"),
    "GainSweepResult": ("twpa.nonlinear.gain", "GainSweepResult"),
    "solve_gain_sweep_from_pump": ("twpa.nonlinear.gain", "solve_gain_sweep_from_pump"),

    # Workflows
    "Industrial100mmWorkflowConfig": (
        "twpa.workflows.industrial_100mm",
        "Industrial100mmWorkflowConfig",
    ),
    "run_industrial_100mm_workflow": (
        "twpa.workflows.industrial_100mm",
        "run_industrial_100mm_workflow",
    ),
    "SyntheticBenchmarkConfig": (
        "twpa.workflows.synthetic_benchmarks",
        "SyntheticBenchmarkConfig",
    ),
    "run_synthetic_benchmarks": (
        "twpa.workflows.synthetic_benchmarks",
        "run_synthetic_benchmarks",
    ),
}


def __getattr__(name: str) -> Any:
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

    raise AttributeError(f"module 'twpa' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(_LAZY_SUBMODULES)
        | set(_LAZY_SYMBOLS)
    )


__all__ = [
    "__version__",
    "core",
    "linear",
    "nonlinear",
    "solvers",
    "workflows",
    "FrequencyPlan",
    "LineLayout",
    "NonlinearParams",
    "SolverConfig",
    "PumpDriveConfig",
    "PumpHBLadderConfig",
    "PumpHBLadderResult",
    "solve_pump_hb_ladder",
    "GainSolveConfig",
    "GainSweepConfig",
    "GainSweepResult",
    "solve_gain_sweep_from_pump",
    "Industrial100mmWorkflowConfig",
    "run_industrial_100mm_workflow",
    "SyntheticBenchmarkConfig",
    "run_synthetic_benchmarks",
]
