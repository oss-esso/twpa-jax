"""
twpa.linear
===========

Pump-off linear microwave simulation tools for the JAX-backed TWPA simulator.

This package contains:

    rf_networks
        ABCD/S/Y/Z network utilities and two-port conversions.

    cells
        Lumped cell models and per-cell validation helpers.

    cascade
        Full-line ABCD cascade, S-parameter scans, and validation reports.

    ladder_mna
        Modified-nodal-analysis reference solver for small ladder validation.

    dispersion
        Effective propagation constant, Bloch dispersion, stopband detection,
        and phase-matching diagnostics.

    coarsening
        Effective-cell and reduced-layout construction for nonlinear studies.

Imports are intentionally lazy so importing ``twpa.linear`` remains lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "rf_networks": "twpa.linear.rf_networks",
    "cells": "twpa.linear.cells",
    "cascade": "twpa.linear.cascade",
    "ladder_mna": "twpa.linear.ladder_mna",
    "dispersion": "twpa.linear.dispersion",
    "coarsening": "twpa.linear.coarsening",
}


_LAZY_SYMBOLS = {
    # cells.py
    "CellModelKind": ("twpa.linear.cells", "CellModelKind"),
    "CellModelConfig": ("twpa.linear.cells", "CellModelConfig"),
    "validate_layout_cells": ("twpa.linear.cells", "validate_layout_cells"),
    "layout_cell_parameter_summary": (
        "twpa.linear.cells",
        "layout_cell_parameter_summary",
    ),

    # cascade.py
    "CascadeStrategy": ("twpa.linear.cascade", "CascadeStrategy"),
    "CascadeConfig": ("twpa.linear.cascade", "CascadeConfig"),
    "LinearScanResult": ("twpa.linear.cascade", "LinearScanResult"),
    "run_linear_scan": ("twpa.linear.cascade", "run_linear_scan"),
    "validate_cascade": ("twpa.linear.cascade", "validate_cascade"),
    "compare_layout_to_uniform_rlgc_line": (
        "twpa.linear.cascade",
        "compare_layout_to_uniform_rlgc_line",
    ),

    # ladder_mna.py
    "LadderMNAConfig": ("twpa.linear.ladder_mna", "LadderMNAConfig"),
    "validate_ladder_mna": ("twpa.linear.ladder_mna", "validate_ladder_mna"),
    "compare_ladder_mna_to_abcd": (
        "twpa.linear.ladder_mna",
        "compare_ladder_mna_to_abcd",
    ),

    # dispersion.py
    "DispersionExtractionMethod": (
        "twpa.linear.dispersion",
        "DispersionExtractionMethod",
    ),
    "DispersionConfig": ("twpa.linear.dispersion", "DispersionConfig"),
    "DispersionResult": ("twpa.linear.dispersion", "DispersionResult"),
    "StopbandMetric": ("twpa.linear.dispersion", "StopbandMetric"),
    "extract_layout_dispersion": (
        "twpa.linear.dispersion",
        "extract_layout_dispersion",
    ),
    "validate_dispersion_result": (
        "twpa.linear.dispersion",
        "validate_dispersion_result",
    ),
    "detect_stopbands": ("twpa.linear.dispersion", "detect_stopbands"),
    "compute_dp4wm_phase_matching": (
        "twpa.linear.dispersion",
        "compute_dp4wm_phase_matching",
    ),
    "nonlinear_delta_beta_dp4wm_simple": (
        "twpa.linear.dispersion",
        "nonlinear_delta_beta_dp4wm_simple",
    ),

    # coarsening.py
    "CoarseningMethod": ("twpa.linear.coarsening", "CoarseningMethod"),
    "CoarseningConfig": ("twpa.linear.coarsening", "CoarseningConfig"),
    "CoarseningHierarchyConfig": (
        "twpa.linear.coarsening",
        "CoarseningHierarchyConfig",
    ),
    "CoarseningHierarchy": ("twpa.linear.coarsening", "CoarseningHierarchy"),
    "coarsen_layout": ("twpa.linear.coarsening", "coarsen_layout"),
    "generate_coarsening_hierarchy": (
        "twpa.linear.coarsening",
        "generate_coarsening_hierarchy",
    ),
    "compare_hierarchy_dispersion": (
        "twpa.linear.coarsening",
        "compare_hierarchy_dispersion",
    ),
    "make_uniform_surrogate_layout": (
        "twpa.linear.coarsening",
        "make_uniform_surrogate_layout",
    ),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import linear submodules and commonly used symbols.
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

    raise AttributeError(f"module 'twpa.linear' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_SYMBOLS))


__all__ = [
    "rf_networks",
    "cells",
    "cascade",
    "ladder_mna",
    "dispersion",
    "coarsening",
    *_LAZY_SYMBOLS.keys(),
]