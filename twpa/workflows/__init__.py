"""
twpa.workflows
==============

Production workflows for TWPA simulation, calibration, and benchmarks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "industrial_100mm": "twpa.workflows.industrial_100mm",
    "calibration": "twpa.workflows.calibration",
    "gain_map": "twpa.workflows.gain_map",
    "synthetic_benchmarks": "twpa.workflows.synthetic_benchmarks",
}


_LAZY_SYMBOLS = {
    # Industrial workflow
    "IndustrialStageStatus": (
        "twpa.workflows.industrial_100mm",
        "IndustrialStageStatus",
    ),
    "IndustrialRunMode": (
        "twpa.workflows.industrial_100mm",
        "IndustrialRunMode",
    ),
    "IndustrialLayoutSpec": (
        "twpa.workflows.industrial_100mm",
        "IndustrialLayoutSpec",
    ),
    "Industrial100mmWorkflowConfig": (
        "twpa.workflows.industrial_100mm",
        "Industrial100mmWorkflowConfig",
    ),
    "Industrial100mmWorkflowResult": (
        "twpa.workflows.industrial_100mm",
        "Industrial100mmWorkflowResult",
    ),
    "build_industrial_layout": (
        "twpa.workflows.industrial_100mm",
        "build_industrial_layout",
    ),
    "run_industrial_100mm_workflow": (
        "twpa.workflows.industrial_100mm",
        "run_industrial_100mm_workflow",
    ),
    "summarize_workflow_markdown": (
        "twpa.workflows.industrial_100mm",
        "summarize_workflow_markdown",
    ),

    # Calibration
    "CalibrationParameterSpec": (
        "twpa.workflows.calibration",
        "CalibrationParameterSpec",
    ),
    "CalibrationVectorSpec": (
        "twpa.workflows.calibration",
        "CalibrationVectorSpec",
    ),
    "CalibrationTarget": (
        "twpa.workflows.calibration",
        "CalibrationTarget",
    ),
    "SParameterCalibrationData": (
        "twpa.workflows.calibration",
        "SParameterCalibrationData",
    ),
    "GainCalibrationData": (
        "twpa.workflows.calibration",
        "GainCalibrationData",
    ),
    "CalibrationOptimizerConfig": (
        "twpa.workflows.calibration",
        "CalibrationOptimizerConfig",
    ),
    "CalibrationResult": (
        "twpa.workflows.calibration",
        "CalibrationResult",
    ),
    "calibrate": (
        "twpa.workflows.calibration",
        "calibrate",
    ),
    "export_calibration_artifacts": (
        "twpa.workflows.calibration",
        "export_calibration_artifacts",
    ),

    # Native gain map
    "NativeGainMapResult": (
        "twpa.workflows.gain_map",
        "NativeGainMapResult",
    ),
    "solve_native_gain_map": (
        "twpa.workflows.gain_map",
        "solve_native_gain_map",
    ),
    "export_native_gain_map_artifacts": (
        "twpa.workflows.gain_map",
        "export_native_gain_map_artifacts",
    ),

    # Synthetic benchmarks
    "BenchmarkStatus": (
        "twpa.workflows.synthetic_benchmarks",
        "BenchmarkStatus",
    ),
    "SyntheticLayoutKind": (
        "twpa.workflows.synthetic_benchmarks",
        "SyntheticLayoutKind",
    ),
    "SyntheticLayoutSpec": (
        "twpa.workflows.synthetic_benchmarks",
        "SyntheticLayoutSpec",
    ),
    "SyntheticBenchmarkConfig": (
        "twpa.workflows.synthetic_benchmarks",
        "SyntheticBenchmarkConfig",
    ),
    "SyntheticBenchmarkSuiteResult": (
        "twpa.workflows.synthetic_benchmarks",
        "SyntheticBenchmarkSuiteResult",
    ),
    "build_synthetic_layout": (
        "twpa.workflows.synthetic_benchmarks",
        "build_synthetic_layout",
    ),
    "run_synthetic_benchmarks": (
        "twpa.workflows.synthetic_benchmarks",
        "run_synthetic_benchmarks",
    ),
    "make_fast_linear_synthetic_config": (
        "twpa.workflows.synthetic_benchmarks",
        "make_fast_linear_synthetic_config",
    ),
    "make_small_nonlinear_synthetic_config": (
        "twpa.workflows.synthetic_benchmarks",
        "make_small_nonlinear_synthetic_config",
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

    raise AttributeError(f"module 'twpa.workflows' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(_LAZY_SUBMODULES)
        | set(_LAZY_SYMBOLS)
    )


__all__ = [
    "industrial_100mm",
    "calibration",
    "gain_map",
    "synthetic_benchmarks",
    *_LAZY_SYMBOLS.keys(),
]
