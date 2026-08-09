"""Lumped kinetic-inductance branch laws.

The polynomial model is expressed in the flux domain used by the solver, but
is parameterized by the physically useful current scales.  It is deliberately
finite at the critical current: crossing that boundary is reported by
``kinetic_validity`` rather than enforced by the constitutive law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import betainc, betaincinv


KI_MODEL_PRESETS: dict[str, dict[str, float]] = {
    "hung_2025": {
        # Hung 2025: Ic = 1.15 mA, I*2 = 3.25 mA, I*4 = 1.70 mA.
        "istar2_over_ic": 3.25e-3 / 1.15e-3,
        "istar4_over_ic": 1.70e-3 / 1.15e-3,
    },
    "hung_2025_alt": {
        # Alternative static fit: I** = 1.65 mA and n = 2.21 at Ic=1.15 mA.
        "istarstar_over_ic": 1.65e-3 / 1.15e-3,
        "n": 2.21,
    },
}


def resolve_ki_model(
    name: str, critical_current_a: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a named model's nonlinear current scales per branch."""
    try:
        preset = KI_MODEL_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(KI_MODEL_PRESETS))
        raise ValueError(f"unknown kinetic-inductance model {name!r}; valid names: {valid}") from exc
    ic = np.asarray(critical_current_a, dtype=float)
    if np.any(~np.isfinite(ic)) or np.any(ic <= 0.0):
        raise ValueError("critical_current_a must be finite and strictly positive")
    return (
        ic * preset["istar2_over_ic"],
        ic * preset["istar4_over_ic"],
    )


@dataclass(frozen=True)
class KineticInductorBranchLaw:
    """Polynomial kinetic-inductance law with optional quartic correction."""

    kinetic_inductance_h: np.ndarray
    critical_current_a: np.ndarray
    istar2_a: np.ndarray
    istar4_a: np.ndarray | None
    model: str = "hung_2025"
    newton_max_iter: int = 20
    newton_rtol: float = 1e-14

    def __post_init__(self) -> None:
        lk = np.asarray(self.kinetic_inductance_h, dtype=float)
        ic = np.asarray(self.critical_current_a, dtype=float)
        i2 = np.asarray(self.istar2_a, dtype=float)
        if lk.ndim != 1 or ic.shape != lk.shape or i2.shape != lk.shape:
            raise ValueError("kinetic-inductance parameters must be one-dimensional arrays of equal length")
        if np.any(~np.isfinite(lk)) or np.any(lk <= 0) or np.any(~np.isfinite(ic)) or np.any(ic <= 0):
            raise ValueError("kinetic_inductance_h and critical_current_a must be finite and positive")
        if np.any(~np.isfinite(i2)) or np.any(i2 <= 0):
            raise ValueError("istar2_a must be finite and positive")
        if self.istar4_a is not None:
            i4 = np.asarray(self.istar4_a, dtype=float)
            if i4.shape != lk.shape or np.any(~np.isfinite(i4)) or np.any(i4 <= 0):
                raise ValueError("istar4_a must be None or a finite positive array matching the branches")
        if self.newton_max_iter <= 0 or self.newton_rtol <= 0:
            raise ValueError("newton_max_iter and newton_rtol must be positive")
        object.__setattr__(self, "kinetic_inductance_h", lk)
        object.__setattr__(self, "critical_current_a", ic)
        object.__setattr__(self, "istar2_a", i2)
        if self.istar4_a is not None:
            object.__setattr__(self, "istar4_a", np.asarray(self.istar4_a, dtype=float))

    def _check_flux(self, flux: np.ndarray) -> np.ndarray:
        value = np.asarray(flux, dtype=float)
        if value.ndim == 0 or value.shape[-1] != self.kinetic_inductance_h.size:
            raise ValueError("flux must have branches on its final axis")
        return value

    def flux(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, dtype=float)
        if current.ndim == 0 or current.shape[-1] != self.kinetic_inductance_h.size:
            raise ValueError("current must have branches on its final axis")
        result = self.kinetic_inductance_h * (
            current
            + current**3 / (3.0 * self.istar2_a**2)
        )
        if self.istar4_a is not None:
            result = result + self.kinetic_inductance_h * current**5 / (5.0 * self.istar4_a**4)
        return result

    def differential_inductance(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, dtype=float)
        if current.ndim == 0 or current.shape[-1] != self.kinetic_inductance_h.size:
            raise ValueError("current must have branches on its final axis")
        result = self.kinetic_inductance_h * (1.0 + (current / self.istar2_a) ** 2)
        if self.istar4_a is not None:
            result = result + self.kinetic_inductance_h * (current / self.istar4_a) ** 4
        return result

    def current(self, flux: np.ndarray) -> np.ndarray:
        target = self._check_flux(flux)
        # Stable exact inversion of the cubic part:
        # u^3 + 3u = 3 Phi/(Lk I*2), u = 2 sinh(asinh(x)/3).
        scale = self.kinetic_inductance_h * self.istar2_a
        result = 2.0 * self.istar2_a * np.sinh(
            np.arcsinh(1.5 * target / scale) / 3.0
        )
        if self.istar4_a is None:
            return result

        scale_residual = np.maximum(np.abs(target), scale * 1e-30)
        for _ in range(self.newton_max_iter):
            residual = self.flux(result) - target
            if np.all(np.abs(residual) <= self.newton_rtol * scale_residual):
                return result
            step = residual / self.differential_inductance(result)
            candidate = result - step
            old_norm = np.abs(residual)
            new_residual = self.flux(candidate) - target
            damping = np.ones_like(candidate)
            bad = np.abs(new_residual) > old_norm
            while np.any(bad):
                damping = np.where(bad, damping * 0.5, damping)
                candidate = result - damping * step
                new_residual = self.flux(candidate) - target
                bad = (np.abs(new_residual) > old_norm) & (damping > 2.0**-20)
                if not np.any(bad):
                    break
            result = candidate
        residual = self.flux(result) - target
        worst = float(np.max(np.abs(residual) / scale_residual))
        raise RuntimeError(f"kinetic-inductance inversion did not converge; worst relative residual={worst:.3e}")

    def tangent(self, flux: np.ndarray) -> np.ndarray:
        current = self.current(flux)
        return 1.0 / self.differential_inductance(current)

    def gamma(self, flux: np.ndarray) -> np.ndarray:
        return self.tangent(flux)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "type": "kinetic_inductor",
            "model": self.model,
            "quartic": self.istar4_a is not None,
            "Lk_h": self.kinetic_inductance_h.tolist(),
            "Ic_a": self.critical_current_a.tolist(),
            "Istar2_a": self.istar2_a.tolist(),
            "Istar4_a": None if self.istar4_a is None else self.istar4_a.tolist(),
        }


def kinetic_validity(
    law: KineticInductorBranchLaw,
    branch_flux_time: np.ndarray,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Report peak total current and critical-current status per branch."""
    flux = np.asarray(branch_flux_time, dtype=float)
    dc = np.zeros(law.critical_current_a.size) if dc_branch_flux is None else np.asarray(dc_branch_flux, dtype=float)
    if flux.ndim != 2 or flux.shape[1] != dc.size or dc.shape != (dc.size,):
        raise ValueError("branch_flux_time must be (time, branch) and dc_branch_flux must be (branch,)")
    total_current = law.current(flux + dc[None, :])
    maximum = np.max(np.abs(total_current), axis=0)
    ratio = maximum / law.critical_current_a
    status = np.where(ratio < 1.0, "SUPERCONDUCTING", "THRESHOLD_CROSSED")
    return {"max_abs_current_a": maximum, "current_over_ic": ratio, "status": status}


def kinetic_dc_branch_flux(circuit: Any, dc_current_a: float | np.ndarray) -> np.ndarray:
    """Return branch flux offsets for a uniform KI DC current."""
    from twpa_solver.core.nonlinear import CompositeBranchLaw

    branch_count = int(circuit.Bphi.shape[1])
    result = np.zeros(branch_count, dtype=float)
    law = getattr(circuit, "branch_law", None)
    if law is None:
        return result
    parts = [(law, np.arange(branch_count))]
    if isinstance(law, CompositeBranchLaw):
        parts = list(zip(law.laws, law.columns))
    kinetic_parts = [
        (part, np.asarray(columns, dtype=int))
        for part, columns in parts
        if "kinetic" in part.metadata.get("type", "")
    ]
    if not kinetic_parts:
        return result
    n_kinetic = sum(columns.size for _, columns in kinetic_parts)
    currents = np.asarray(dc_current_a, dtype=float).reshape(-1)
    if currents.size == 1:
        currents = np.full(n_kinetic, currents[0])
    if currents.size != n_kinetic:
        raise ValueError(f"dc_current_a must be scalar or contain {n_kinetic} kinetic-branch values")
    offset = 0
    for part, columns in kinetic_parts:
        values = currents[offset:offset + columns.size]
        result[columns] = part.flux(values[None, :])[0]
        offset += columns.size
    return result


@dataclass
class KineticInductorAltBranchLaw:
    """Alternative finite-flux model with a divergent inductance at ``I**``.

    The constitutive relation is evaluated in closed form with the incomplete
    beta function.  Outside the mathematical domain it is continued with a
    monotone finite-slope extension and counted diagnostically, so trial solver
    steps cannot turn into hard failures.
    """

    kinetic_inductance_h: np.ndarray
    critical_current_a: np.ndarray
    istarstar_a: np.ndarray
    n: float = 2.21
    out_of_domain_samples: int = 0

    def __post_init__(self) -> None:
        self.kinetic_inductance_h = np.asarray(self.kinetic_inductance_h, dtype=float).reshape(-1)
        self.critical_current_a = np.asarray(self.critical_current_a, dtype=float).reshape(-1)
        self.istarstar_a = np.asarray(self.istarstar_a, dtype=float).reshape(-1)
        if self.critical_current_a.shape != self.kinetic_inductance_h.shape or self.istarstar_a.shape != self.kinetic_inductance_h.shape:
            raise ValueError("alternative kinetic-inductance parameters must have equal branch lengths")
        if self.n <= 1 or np.any(self.kinetic_inductance_h <= 0) or np.any(self.istarstar_a <= 0):
            raise ValueError("alternative law requires n > 1 and positive parameters")

    @property
    def _beta_a(self) -> float:
        return 1.0 / self.n

    @property
    def _beta_b(self) -> float:
        return 1.0 - 1.0 / self.n

    @property
    def _beta_total(self) -> float:
        return float(np.pi / np.sin(np.pi / self.n))

    def _phi_max(self) -> np.ndarray:
        return self.kinetic_inductance_h * self.istarstar_a * self._beta_total / self.n

    def flux(self, current: np.ndarray) -> np.ndarray:
        value = np.asarray(current, dtype=float)
        if value.ndim == 0 or value.shape[-1] != self.kinetic_inductance_h.size:
            raise ValueError("current must have branches on its final axis")
        magnitude = np.abs(value)
        ratio = magnitude / self.istarstar_a
        inside = ratio < 1.0
        w = np.minimum(ratio, 1.0) ** self.n
        result = self._phi_max() * betainc(self._beta_a, self._beta_b, w)
        if np.any(~inside):
            self.out_of_domain_samples += int(np.count_nonzero(~inside))
            # Use a finite slope sampled just inside the boundary.
            near = 1.0 - 1e-12
            slope = self.kinetic_inductance_h / (1.0 - near**self.n) ** (1.0 / self.n)
            result = np.where(inside, result, self._phi_max() + slope * (magnitude - self.istarstar_a))
        return np.sign(value) * result

    def current(self, flux: np.ndarray) -> np.ndarray:
        value = np.asarray(flux, dtype=float)
        if value.ndim == 0 or value.shape[-1] != self.kinetic_inductance_h.size:
            raise ValueError("flux must have branches on its final axis")
        magnitude = np.abs(value)
        maximum = self._phi_max()
        inside = magnitude <= maximum
        w = betaincinv(self._beta_a, self._beta_b, np.minimum(magnitude / maximum, 1.0))
        result = self.istarstar_a * w ** (1.0 / self.n)
        if np.any(~inside):
            self.out_of_domain_samples += int(np.count_nonzero(~inside))
            near = 1.0 - 1e-12
            slope = self.kinetic_inductance_h / (1.0 - near**self.n) ** (1.0 / self.n)
            result = np.where(inside, result, self.istarstar_a + (magnitude - maximum) / slope)
        return np.sign(value) * result

    def differential_inductance(self, current: np.ndarray) -> np.ndarray:
        value = np.asarray(current, dtype=float)
        ratio = np.abs(value) / self.istarstar_a
        return self.kinetic_inductance_h / (1.0 - np.minimum(ratio, 1.0 - 1e-15) ** self.n) ** (1.0 / self.n)

    def tangent(self, flux: np.ndarray) -> np.ndarray:
        return 1.0 / self.differential_inductance(self.current(flux))

    def gamma(self, flux: np.ndarray) -> np.ndarray:
        return self.tangent(flux)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"type": "kinetic_inductor_alt", "model": "hung_2025_alt", "n": self.n}
