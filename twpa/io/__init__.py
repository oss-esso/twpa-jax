"""
twpa.io
=======

Input/output utilities for the JAX-backed TWPA simulator.

This package contains practical bridge code for moving between simulator-native
objects and external artifacts:

    measurement
        Measurement dataset loaders and normalizers for S-parameters, gain
        traces, CSV/NPZ files, and calibration inputs.

    netlist
        Lightweight circuit/netlist import-export helpers for one-period and
        distributed-ladder models.

    reports
        JSON, Markdown, and table-report helpers for simulation runs.

    checkpoints
        Checkpoint save/load utilities for long simulations and calibration
        workflows.

The package is intentionally lightweight and does not import heavy simulator
modules until a symbol is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "measurement": "twpa.io.measurement",
    "netlist": "twpa.io.netlist",
    "reports": "twpa.io.reports",
    "checkpoints": "twpa.io.checkpoints",
}


_LAZY_SYMBOLS = {
    # measurement.py
    "MeasurementKind": ("twpa.io.measurement", "MeasurementKind"),
    "MeasurementLoadConfig": ("twpa.io.measurement", "MeasurementLoadConfig"),
    "SParameterMeasurement": ("twpa.io.measurement", "SParameterMeasurement"),
    "GainMeasurement": ("twpa.io.measurement", "GainMeasurement"),
    "load_sparameter_measurement": (
        "twpa.io.measurement",
        "load_sparameter_measurement",
    ),
    "load_gain_measurement": (
        "twpa.io.measurement",
        "load_gain_measurement",
    ),

    # netlist.py
    "NetlistFormat": ("twpa.io.netlist", "NetlistFormat"),
    "NetlistExportConfig": ("twpa.io.netlist", "NetlistExportConfig"),
    "export_layout_to_spice_subckt": (
        "twpa.io.netlist",
        "export_layout_to_spice_subckt",
    ),
    "write_spice_subckt": ("twpa.io.netlist", "write_spice_subckt"),

    # reports.py
    "ReportFormat": ("twpa.io.reports", "ReportFormat"),
    "RunReport": ("twpa.io.reports", "RunReport"),
    "write_json_report": ("twpa.io.reports", "write_json_report"),
    "write_markdown_report": ("twpa.io.reports", "write_markdown_report"),
    "write_run_report_bundle": (
        "twpa.io.reports",
        "write_run_report_bundle",
    ),

    # checkpoints.py
    "CheckpointKind": ("twpa.io.checkpoints", "CheckpointKind"),
    "CheckpointMetadata": ("twpa.io.checkpoints", "CheckpointMetadata"),
    "save_checkpoint": ("twpa.io.checkpoints", "save_checkpoint"),
    "load_checkpoint": ("twpa.io.checkpoints", "load_checkpoint"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import IO submodules and commonly used symbols.
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

    raise AttributeError(f"module 'twpa.io' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_SYMBOLS))


__all__ = [
    "measurement",
    "netlist",
    "reports",
    "checkpoints",
    *_LAZY_SYMBOLS.keys(),
]