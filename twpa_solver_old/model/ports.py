"""Port definitions and wave normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Port:
    name: str
    node: int
    z0_ohm: float = 50.0

    def __post_init__(self) -> None:
        if self.node < 0:
            raise ValueError("port node must be non-negative")
        if self.z0_ohm <= 0.0:
            raise ValueError("z0_ohm must be positive")


def voltage_current_to_waves(
    voltage_v: complex | np.ndarray,
    current_a: complex | np.ndarray,
    z0_ohm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert port voltage/current to power-wave amplitudes."""
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    root_z = np.sqrt(z0_ohm)
    v = np.asarray(voltage_v, dtype=complex)
    i = np.asarray(current_a, dtype=complex)
    return (v + z0_ohm * i) / (2.0 * root_z), (v - z0_ohm * i) / (2.0 * root_z)
