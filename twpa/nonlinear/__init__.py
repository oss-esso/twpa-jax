"""
twpa.nonlinear
==============

Nonlinear harmonic-balance and gain tools for KI-TWPA simulation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "one_node": "twpa.nonlinear.one_node",
    "kinetic_inductance": "twpa.nonlinear.kinetic_inductance",
    "distributed_hb": "twpa.nonlinear.distributed_hb",
    "pump_hb_ladder": "twpa.nonlinear.pump_hb_ladder",
    "linearization": "twpa.nonlinear.linearization",
    "gain": "twpa.nonlinear.gain",
}


_LAZY_SYMBOLS = {
    # Kinetic inductance
    "KineticInductanceModel": (
        "twpa.nonlinear.kinetic_inductance",
        "KineticInductanceModel",
    ),

    # Distributed HB
    "DistributedHBConfig": ("twpa.nonlinear.distributed_hb", "DistributedHBConfig"),
    "DistributedHBState": ("twpa.nonlinear.distributed_hb", "DistributedHBState"),
    "DistributedHBResidual": ("twpa.nonlinear.distributed_hb", "DistributedHBResidual"),
    "DistributedHBSolveResult": (
        "twpa.nonlinear.distributed_hb",
        "DistributedHBSolveResult",
    ),
    "solve_distributed_hb": ("twpa.nonlinear.distributed_hb", "solve_distributed_hb"),
    "solve_distributed_pump_current_hb": (
        "twpa.nonlinear.distributed_hb",
        "solve_distributed_pump_current_hb",
    ),

    # Pump HB
    "PumpDriveConfig": ("twpa.nonlinear.pump_hb_ladder", "PumpDriveConfig"),
    "PumpHBLadderConfig": ("twpa.nonlinear.pump_hb_ladder", "PumpHBLadderConfig"),
    "PumpHBLadderResult": ("twpa.nonlinear.pump_hb_ladder", "PumpHBLadderResult"),
    "PumpContinuationResult": (
        "twpa.nonlinear.pump_hb_ladder",
        "PumpContinuationResult",
    ),
    "solve_pump_hb_ladder": (
        "twpa.nonlinear.pump_hb_ladder",
        "solve_pump_hb_ladder",
    ),
    "solve_pump_hb_ladder_from_power_dbm": (
        "twpa.nonlinear.pump_hb_ladder",
        "solve_pump_hb_ladder_from_power_dbm",
    ),
    "solve_pump_hb_ladder_from_current_rms": (
        "twpa.nonlinear.pump_hb_ladder",
        "solve_pump_hb_ladder_from_current_rms",
    ),

    # Linearization
    "SmallSignalLinearizationConfig": (
        "twpa.nonlinear.linearization",
        "SmallSignalLinearizationConfig",
    ),
    "SmallSignalSource": ("twpa.nonlinear.linearization", "SmallSignalSource"),
    "SmallSignalState": ("twpa.nonlinear.linearization", "SmallSignalState"),
    "SmallSignalSolveResult": (
        "twpa.nonlinear.linearization",
        "SmallSignalSolveResult",
    ),
    "DistributedHBLinearization": (
        "twpa.nonlinear.linearization",
        "DistributedHBLinearization",
    ),
    "build_linearization_from_pump_result": (
        "twpa.nonlinear.linearization",
        "build_linearization_from_pump_result",
    ),
    "solve_linearized_small_signal": (
        "twpa.nonlinear.linearization",
        "solve_linearized_small_signal",
    ),

    # Gain
    "GainSolveConfig": ("twpa.nonlinear.gain", "GainSolveConfig"),
    "GainSweepConfig": ("twpa.nonlinear.gain", "GainSweepConfig"),
    "GainPointResult": ("twpa.nonlinear.gain", "GainPointResult"),
    "GainSweepResult": ("twpa.nonlinear.gain", "GainSweepResult"),
    "GainOperatingMap": ("twpa.nonlinear.gain", "GainOperatingMap"),
    "build_gain_linearization_from_pump": (
        "twpa.nonlinear.gain",
        "build_gain_linearization_from_pump",
    ),
    "solve_gain_point": ("twpa.nonlinear.gain", "solve_gain_point"),
    "solve_gain_sweep": ("twpa.nonlinear.gain", "solve_gain_sweep"),
    "solve_gain_sweep_from_pump": (
        "twpa.nonlinear.gain",
        "solve_gain_sweep_from_pump",
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

    raise AttributeError(f"module 'twpa.nonlinear' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(_LAZY_SUBMODULES)
        | set(_LAZY_SYMBOLS)
    )


__all__ = [
    "one_node",
    "kinetic_inductance",
    "distributed_hb",
    "pump_hb_ladder",
    "linearization",
    "gain",
    *_LAZY_SYMBOLS.keys(),
]