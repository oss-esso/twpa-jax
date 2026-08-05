"""Frequency-dependent passive port environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortEnvironment:
    """Standing-wave impedance correction to an already-stamped 50-ohm port."""

    z0_ohm: float = 50.0
    z1_ohm: float = 14.2
    tau1_s: float = 10.5e-9
    phi1_rad: float = -0.7 * np.pi
    z2_ohm: float = 1.9
    tau2_s: float = 121.0e-9
    phi2_rad: float = 0.0
    check_frequencies_hz: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.z0_ohm <= 0.0 or self.tau1_s < 0.0 or self.tau2_s < 0.0:
            raise ValueError("z0_ohm must be positive and delays must be non-negative")
        if self.check_frequencies_hz is None:
            frequencies = np.linspace(0.0, 30.0e9, 4097)
        else:
            frequencies = np.asarray(self.check_frequencies_hz, dtype=float).reshape(-1)
            if np.any(frequencies < 0.0):
                raise ValueError("check_frequencies_hz must be non-negative")
        impedance = self.impedance(2.0 * np.pi * frequencies)
        if np.any(np.real(impedance) <= 0.0):
            raise ValueError("PortEnvironment must be passive: Re Z_env(omega) must be positive")

    def impedance(self, omega: float | np.ndarray) -> np.ndarray:
        w = np.asarray(omega, dtype=float)
        return (
            self.z0_ohm
            + self.z1_ohm * np.exp(1j * (w * self.tau1_s + self.phi1_rad))
            + self.z2_ohm * np.exp(1j * (w * self.tau2_s + self.phi2_rad))
        )

    def admittance(self, omega: float | np.ndarray) -> complex | np.ndarray:
        value = 1.0 / self.impedance(omega) - 1.0 / self.z0_ohm
        return complex(value) if np.ndim(omega) == 0 else value
