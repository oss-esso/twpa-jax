"""
twpa.plotting
=============

Plotting utilities for TWPA simulation diagnostics and gain maps.

This package contains optional visualization helpers. The core simulator does
not depend on plotting, so these imports are intentionally lazy.

Modules
-------
diagnostics
    Linear response, dispersion, pump profile, residual-history, and recovery
    diagnostic plots.

gain_maps
    Gain-map and operating-map plotting helpers for pump/signal sweeps.

Notes
-----
The plotting functions use matplotlib and return ``(fig, ax)`` objects so that
scripts and notebooks can further customize or save the figures.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "diagnostics": "twpa.plotting.diagnostics",
    "gain_maps": "twpa.plotting.gain_maps",
}


_LAZY_SYMBOLS = {
    # diagnostics.py
    "PlotConfig": ("twpa.plotting.diagnostics", "PlotConfig"),
    "save_figure": ("twpa.plotting.diagnostics", "save_figure"),
    "plot_s21": ("twpa.plotting.diagnostics", "plot_s21"),
    "plot_dispersion": ("twpa.plotting.diagnostics", "plot_dispersion"),
    "plot_stopbands": ("twpa.plotting.diagnostics", "plot_stopbands"),
    "plot_pump_profile": ("twpa.plotting.diagnostics", "plot_pump_profile"),
    "plot_newton_history": ("twpa.plotting.diagnostics", "plot_newton_history"),
    "plot_fit_history": ("twpa.plotting.diagnostics", "plot_fit_history"),
    "plot_recovery_truth_vs_fit": (
        "twpa.plotting.diagnostics",
        "plot_recovery_truth_vs_fit",
    ),

    # gain_maps.py
    "GainMapPlotConfig": ("twpa.plotting.gain_maps", "GainMapPlotConfig"),
    "plot_gain_sweep": ("twpa.plotting.gain_maps", "plot_gain_sweep"),
    "plot_gain_map": ("twpa.plotting.gain_maps", "plot_gain_map"),
    "plot_operating_map": ("twpa.plotting.gain_maps", "plot_operating_map"),
    "plot_compression_sweep": (
        "twpa.plotting.gain_maps",
        "plot_compression_sweep",
    ),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import plotting submodules and common symbols.
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

    raise AttributeError(f"module 'twpa.plotting' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_SYMBOLS))


__all__ = [
    "diagnostics",
    "gain_maps",
    *_LAZY_SYMBOLS.keys(),
]