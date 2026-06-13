"""SI constants and RF source conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    h: float = 6.62607015e-34
    e: float = 1.602176634e-19

    @property
    def phi0(self) -> float:
        return self.h / (2.0 * self.e)

    @property
    def reduced_phi0(self) -> float:
        return self.phi0 / (2.0 * np.pi)


CONSTANTS = PhysicalConstants()


def dbm_to_watts(power_dbm: float | np.ndarray) -> float | np.ndarray:
    """Convert available RF power in dBm to watts."""
    return 1e-3 * np.power(10.0, np.asarray(power_dbm) / 10.0)


def watts_to_dbm(power_w: float | np.ndarray, floor_w: float = 1e-300) -> float | np.ndarray:
    """Convert watts to dBm with a finite logarithm floor."""
    return 10.0 * np.log10(np.maximum(np.asarray(power_w), floor_w) / 1e-3)


def dbm_to_norton_current_rms(power_dbm: float, z0_ohm: float = 50.0) -> float:
    """Return Norton RMS current for available source power into z0."""
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    return float(2.0 * np.sqrt(dbm_to_watts(power_dbm) / z0_ohm))


def dbm_to_current_peak(power_dbm: float, z0_ohm: float = 50.0) -> float:
    """Return sinusoidal peak Norton current for available source power."""
    return float(np.sqrt(2.0) * dbm_to_norton_current_rms(power_dbm, z0_ohm))


def dbm_to_old_julia_peak_current(power_dbm: float, z0_ohm: float = 50.0) -> float:
    """Return the peak current convention used by old Julia IPM maps."""
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    return float(np.sqrt(2.0 * dbm_to_watts(power_dbm) / z0_ohm))


def current_peak_to_dbm(current_peak_a: float, z0_ohm: float = 50.0) -> float:
    """Return available source power in dBm from peak Norton current."""
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    current_rms = float(current_peak_a) / np.sqrt(2.0)
    return float(watts_to_dbm(current_rms**2 * z0_ohm / 4.0))


def old_julia_peak_current_to_dbm(current_peak_a: float, z0_ohm: float = 50.0) -> float:
    """Invert dbm_to_old_julia_peak_current."""
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    return float(watts_to_dbm(float(current_peak_a) ** 2 * z0_ohm / 2.0))
