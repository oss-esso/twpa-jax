"""
twpa.core
=========

Core data structures and numerical utilities for the JAX-backed TWPA simulator.

This package contains the low-level objects used by both the linear and
nonlinear simulation stacks:

    units
        Unit conversion helpers.

    params
        Physical and solver parameter containers.

    layout
        Distributed transmission-line / lumped-cell layout containers.

    frequency_plan
        Harmonic-balance tone and label management.

    harmonics
        Complex Fourier coefficient helpers.

    hb_fft
        Time/frequency projection utilities for harmonic balance.

    disorder
        Fabrication-disorder and parameter-perturbation helpers.

Imports are intentionally lazy so importing ``twpa.core`` stays lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "units": "twpa.core.units",
    "params": "twpa.core.params",
    "layout": "twpa.core.layout",
    "frequency_plan": "twpa.core.frequency_plan",
    "harmonics": "twpa.core.harmonics",
    "hb_fft": "twpa.core.hb_fft",
    "disorder": "twpa.core.disorder",
}


_LAZY_SYMBOLS = {
    # params.py
    "NonlinearParams": ("twpa.core.params", "NonlinearParams"),
    "SolverConfig": ("twpa.core.params", "SolverConfig"),

    # layout.py
    "LineLayout": ("twpa.core.layout", "LineLayout"),
    "make_layout_from_arrays": ("twpa.core.layout", "make_layout_from_arrays"),

    # frequency_plan.py
    "FrequencyPlan": ("twpa.core.frequency_plan", "FrequencyPlan"),
    "make_pump_only_plan": ("twpa.core.frequency_plan", "make_pump_only_plan"),

    # hb_fft.py
    "HBProjectionConfig": ("twpa.core.hb_fft", "HBProjectionConfig"),
    "HBProjectionGrid": ("twpa.core.hb_fft", "HBProjectionGrid"),
    "make_projection_grid_from_plan": (
        "twpa.core.hb_fft",
        "make_projection_grid_from_plan",
    ),

    # harmonics.py
    "zeros_for_plan": ("twpa.core.harmonics", "zeros_for_plan"),
    "coefficient_power_summary": (
        "twpa.core.harmonics",
        "coefficient_power_summary",
    ),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import core submodules and commonly used symbols.
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

    raise AttributeError(f"module 'twpa.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_SYMBOLS))


__all__ = [
    "units",
    "params",
    "layout",
    "frequency_plan",
    "harmonics",
    "hb_fft",
    "disorder",
    *_LAZY_SYMBOLS.keys(),
]