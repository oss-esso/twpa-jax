"""Explicit nonlinear branch laws independent of solver code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from twpa_solver.model.units import CONSTANTS


class Nonlinearity(Protocol):
    """Interface for a scalar or vector branch current law."""

    def current(self, branch_flux: np.ndarray) -> np.ndarray:
        """Evaluate branch current."""

    def derivative(self, branch_flux: np.ndarray) -> np.ndarray:
        """Evaluate dI/dpsi."""


@dataclass(frozen=True)
class JosephsonNonlinearity:
    """Josephson current law I = Ic sin(psi / varphi0)."""

    critical_current_a: np.ndarray
    reduced_flux_quantum_wb: float = CONSTANTS.reduced_phi0

    def current(self, branch_flux: np.ndarray) -> np.ndarray:
        psi = np.asarray(branch_flux, dtype=float)
        return self.critical_current_a * np.sin(psi / self.reduced_flux_quantum_wb)

    def derivative(self, branch_flux: np.ndarray) -> np.ndarray:
        psi = np.asarray(branch_flux, dtype=float)
        return (
            self.critical_current_a
            / self.reduced_flux_quantum_wb
            * np.cos(psi / self.reduced_flux_quantum_wb)
        )


@dataclass(frozen=True)
class RFSQUIDNonlinearity:
    """RF-SQUID placeholder branch law using an effective Josephson element."""

    critical_current_a: np.ndarray
    flux_bias_wb: float = 0.0
    reduced_flux_quantum_wb: float = CONSTANTS.reduced_phi0

    def current(self, branch_flux: np.ndarray) -> np.ndarray:
        psi = np.asarray(branch_flux, dtype=float) + self.flux_bias_wb
        return self.critical_current_a * np.sin(psi / self.reduced_flux_quantum_wb)

    def derivative(self, branch_flux: np.ndarray) -> np.ndarray:
        psi = np.asarray(branch_flux, dtype=float) + self.flux_bias_wb
        return (
            self.critical_current_a
            / self.reduced_flux_quantum_wb
            * np.cos(psi / self.reduced_flux_quantum_wb)
        )


@dataclass(frozen=True)
class KineticInductanceNonlinearity:
    """Weak cubic kinetic-inductance-like branch current scaffold."""

    inverse_inductance_h_inv: np.ndarray
    cubic_scale_wb: float

    def current(self, branch_flux: np.ndarray) -> np.ndarray:
        psi = np.asarray(branch_flux, dtype=float)
        return self.inverse_inductance_h_inv * psi * (
            1.0 - (psi / self.cubic_scale_wb) ** 2
        )

    def derivative(self, branch_flux: np.ndarray) -> np.ndarray:
        psi = np.asarray(branch_flux, dtype=float)
        return self.inverse_inductance_h_inv * (
            1.0 - 3.0 * (psi / self.cubic_scale_wb) ** 2
        )
