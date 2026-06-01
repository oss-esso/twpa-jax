"""
twpa.inference
==============

Inference and parameter-recovery tools for the JAX-backed TWPA simulator.

This package contains higher-level utilities for turning simulator runs into
parameter-estimation workflows:

    priors
        Prior distributions, bounds, transforms, and parameter-vector helpers.

    synthetic
        Synthetic measurement generation for recovery studies and regression
        tests.

    fitting
        Optimizer-facing objective functions and fitting orchestration.

    recovery
        End-to-end synthetic recovery experiments and reporting utilities.

The lower-level production calibration workflow lives in:

    twpa.workflows.calibration

The modules in ``twpa.inference`` are intentionally more experiment-oriented:
they are useful for notebooks, identifiability studies, synthetic benchmarks,
and thesis figures.

Imports are lazy so importing ``twpa.inference`` stays lightweight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_SUBMODULES = {
    "priors": "twpa.inference.priors",
    "synthetic": "twpa.inference.synthetic",
    "fitting": "twpa.inference.fitting",
    "recovery": "twpa.inference.recovery",
}


_LAZY_SYMBOLS = {
    # priors.py
    "PriorKind": ("twpa.inference.priors", "PriorKind"),
    "ParameterPrior": ("twpa.inference.priors", "ParameterPrior"),
    "PriorSet": ("twpa.inference.priors", "PriorSet"),
    "ParameterSample": ("twpa.inference.priors", "ParameterSample"),
    "make_scale_prior": ("twpa.inference.priors", "make_scale_prior"),
    "make_positive_prior": ("twpa.inference.priors", "make_positive_prior"),

    # synthetic.py
    "SyntheticMeasurementKind": (
        "twpa.inference.synthetic",
        "SyntheticMeasurementKind",
    ),
    "SyntheticNoiseConfig": (
        "twpa.inference.synthetic",
        "SyntheticNoiseConfig",
    ),
    "SyntheticSParameterDataset": (
        "twpa.inference.synthetic",
        "SyntheticSParameterDataset",
    ),
    "SyntheticGainDataset": (
        "twpa.inference.synthetic",
        "SyntheticGainDataset",
    ),
    "generate_synthetic_sparameters": (
        "twpa.inference.synthetic",
        "generate_synthetic_sparameters",
    ),
    "generate_synthetic_gain_data": (
        "twpa.inference.synthetic",
        "generate_synthetic_gain_data",
    ),

    # fitting.py
    "FitStatus": ("twpa.inference.fitting", "FitStatus"),
    "FitConfig": ("twpa.inference.fitting", "FitConfig"),
    "FitResult": ("twpa.inference.fitting", "FitResult"),
    "run_parameter_fit": ("twpa.inference.fitting", "run_parameter_fit"),

    # recovery.py
    "RecoveryExperimentConfig": (
        "twpa.inference.recovery",
        "RecoveryExperimentConfig",
    ),
    "RecoveryExperimentResult": (
        "twpa.inference.recovery",
        "RecoveryExperimentResult",
    ),
    "run_synthetic_recovery_experiment": (
        "twpa.inference.recovery",
        "run_synthetic_recovery_experiment",
    ),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import inference submodules and common symbols.
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

    raise AttributeError(f"module 'twpa.inference' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_SYMBOLS))


__all__ = [
    "priors",
    "synthetic",
    "fitting",
    "recovery",
    *_LAZY_SYMBOLS.keys(),
]