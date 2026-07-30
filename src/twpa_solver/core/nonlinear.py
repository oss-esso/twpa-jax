"""Vectorized nonlinear branch laws shared by solver and observables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class BranchLaw(Protocol):
    def current(self, flux: np.ndarray) -> np.ndarray: ...
    def tangent(self, flux: np.ndarray) -> np.ndarray: ...
    @property
    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class JosephsonBranchLaw:
    critical_current: np.ndarray
    phi0: float

    def current(self, flux: np.ndarray) -> np.ndarray:
        return self.critical_current[None, :] * np.sin(flux / self.phi0)

    def tangent(self, flux: np.ndarray) -> np.ndarray:
        return self.critical_current[None, :] * np.cos(flux / self.phi0) / self.phi0

    def gamma(self, flux: np.ndarray) -> np.ndarray:
        return self.tangent(flux)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"type": "josephson", "phi0": self.phi0}


@dataclass(frozen=True)
class EffectiveSnailBranchLaw:
    critical_current: np.ndarray
    ratio: np.ndarray
    phi_ext: np.ndarray
    phi0: float
    equilibrium_flux: np.ndarray | None = None
    external_flux_on_small_junction: bool = False

    def _absolute_flux(self, flux: np.ndarray) -> np.ndarray:
        equilibrium = self.equilibrium_flux
        if equilibrium is None:
            return flux
        return flux + equilibrium[None, :]

    def current(self, flux: np.ndarray) -> np.ndarray:
        absolute_flux = self._absolute_flux(flux)
        return self.critical_current[None, :] * (
            (
                self.ratio[None, :]
                * np.sin(
                    (absolute_flux - self.phi_ext[None, :]) / self.phi0
                )
                + np.sin(absolute_flux / (3.0 * self.phi0))
            )
            if self.external_flux_on_small_junction
            else (
                self.ratio[None, :] * np.sin(absolute_flux / self.phi0)
                + np.sin(
                    (absolute_flux - self.phi_ext[None, :])
                    / (3.0 * self.phi0)
                )
            )
        )

    def tangent(self, flux: np.ndarray) -> np.ndarray:
        absolute_flux = self._absolute_flux(flux)
        return self.critical_current[None, :] / self.phi0 * (
            (
                self.ratio[None, :]
                * np.cos(
                    (absolute_flux - self.phi_ext[None, :]) / self.phi0
                )
                + np.cos(absolute_flux / (3.0 * self.phi0)) / 3.0
            )
            if self.external_flux_on_small_junction
            else (
                self.ratio[None, :] * np.cos(absolute_flux / self.phi0)
                + np.cos(
                    (absolute_flux - self.phi_ext[None, :])
                    / (3.0 * self.phi0)
                )
                / 3.0
            )
        )

    def gamma(self, flux: np.ndarray) -> np.ndarray:
        return self.tangent(flux)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "type": "effective_snail",
            "phi0": self.phi0,
            "equilibrium_flux": None
            if self.equilibrium_flux is None
            else self.equilibrium_flux.tolist(),
            "external_flux_on_small_junction": self.external_flux_on_small_junction,
        }


def make_branch_law(circuit: Any) -> BranchLaw:
    return getattr(circuit, "branch_law", None) or JosephsonBranchLaw(
        circuit.Ic, circuit.phi0
    )
