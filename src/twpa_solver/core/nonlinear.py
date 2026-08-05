"""Vectorized nonlinear branch laws shared by solver and observables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.optimize import brentq

from twpa_solver.core.constants import PHI0_REDUCED


def snail_taylor_coefficients(
    ratio: float, flux_over_flux0: float, *, phi0: float = PHI0_REDUCED
) -> dict[str, float]:
    """Return analytic shifted-SNAIL Taylor coefficients in ``Ic`` units.

    The equilibrium solves ``r sin(x) + sin((x-2 pi f)/3) = 0``.  The
    derivatives are evaluated analytically at the stable root, selected by
    ``g1 > 0``; ``g3`` is the third derivative divided by ``3!``.
    """
    external = 2.0 * np.pi * flux_over_flux0

    def current(x: float) -> float:
        return ratio * np.sin(x) + np.sin((x - external) / 3.0)

    grid = np.linspace(external - 3.0 * np.pi, external + 3.0 * np.pi, 2001)
    roots: list[float] = []
    for left, right in zip(grid[:-1], grid[1:]):
        fl, fr = current(float(left)), current(float(right))
        if fl == 0.0:
            roots.append(float(left))
        elif fl * fr < 0.0:
            roots.append(float(brentq(current, left, right)))
    for equilibrium in roots:
        phase_small = equilibrium
        phase_large = (equilibrium - external) / 3.0
        g1 = ratio * np.cos(phase_small) + np.cos(phase_large) / 3.0
        if g1 <= 0.0:
            continue
        g2 = -ratio * np.sin(phase_small) - np.sin(phase_large) / 9.0
        g3 = (-ratio * np.cos(phase_small) - np.cos(phase_large) / 27.0) / 6.0
        return {
            "g1": float(g1),
            "g2": float(g2),
            "g3": float(g3),
            "g3_over_g1": float(g3 / g1),
            "equilibrium_flux_rad": float(equilibrium),
        }
    raise ValueError(
        f"no stable SNAIL equilibrium for ratio={ratio}, "
        f"flux_over_flux0={flux_over_flux0}"
    )


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


@dataclass(frozen=True)
class CompositeBranchLaw:
    """Dispatch several branch laws onto disjoint columns of one branch axis."""

    laws: tuple[Any, ...]
    columns: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.laws) != len(self.columns) or not self.laws:
            raise ValueError("laws and columns must be non-empty and have equal length")
        normalized = tuple(np.asarray(column, dtype=int) for column in self.columns)
        if any(column.ndim != 1 or column.size == 0 for column in normalized):
            raise ValueError("each composite column set must be a non-empty one-dimensional array")
        all_columns = np.concatenate(normalized)
        nbranch = int(all_columns.max()) + 1
        if np.any(all_columns < 0) or np.unique(all_columns).size != all_columns.size or not np.array_equal(
            np.sort(all_columns), np.arange(nbranch)
        ):
            raise ValueError("composite columns must partition range(nbranch) exactly")
        for law, column in zip(self.laws, normalized):
            if np.asarray(law.current(np.zeros((1, column.size)))).shape != (1, column.size):
                raise ValueError("each law must accept exactly its assigned number of branches")
        object.__setattr__(self, "columns", normalized)

    @property
    def nbranch(self) -> int:
        return int(sum(column.size for column in self.columns))

    def _dispatch(self, name: str, flux: np.ndarray) -> np.ndarray:
        value = np.asarray(flux, dtype=float)
        if value.ndim == 0 or value.shape[-1] != self.nbranch:
            raise ValueError("flux must have branches on its final axis")
        result = np.empty_like(value, dtype=float)
        for law, column in zip(self.laws, self.columns):
            result[..., column] = getattr(law, name)(value[..., column])
        return result

    def current(self, flux: np.ndarray) -> np.ndarray:
        return self._dispatch("current", flux)

    def tangent(self, flux: np.ndarray) -> np.ndarray:
        return self._dispatch("tangent", flux)

    def gamma(self, flux: np.ndarray) -> np.ndarray:
        return self.tangent(flux)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"type": "composite", "parts": [law.metadata for law in self.laws]}


def make_branch_law(circuit: Any) -> BranchLaw:
    return getattr(circuit, "branch_law", None) or JosephsonBranchLaw(
        circuit.Ic, circuit.phi0
    )
