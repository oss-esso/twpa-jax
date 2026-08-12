"""RCSJ junction shunt damping utilities.

The shunt is deliberately represented as a real conductance in ``G``.  It is
an analysis regularizer for the present JTWPA work, not a claim about the
physical subgap resistance of an aluminium junction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
import scipy.sparse as sp

from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.core.constants import PHI0


ELEMENTARY_CHARGE_C = 1.602176634e-19
DEFAULT_AL_DELTA_EV = 180.0e-6


@dataclass(frozen=True)
class RCSJParameters:
    """Per-junction RCSJ parameters for one resistance-ratio setting."""

    resistance_ratio: float
    delta_ev: float
    critical_current_a: np.ndarray
    junction_capacitance_f: np.ndarray
    rn_ohm: np.ndarray
    resistance_ohm: np.ndarray
    beta_c: np.ndarray
    plasma_omega_rad_s: np.ndarray
    quality_factor: np.ndarray
    damping_per_pump_period: np.ndarray

    @property
    def summary(self) -> dict[str, Any]:
        def values(array: np.ndarray) -> list[float]:
            return [float(x) for x in np.asarray(array).reshape(-1)]

        return {
            "resistance_ratio": float(self.resistance_ratio),
            "delta_ev": float(self.delta_ev),
            "rn_ohm": values(self.rn_ohm),
            "resistance_ohm": values(self.resistance_ohm),
            "beta_c": values(self.beta_c),
            "plasma_omega_rad_s": values(self.plasma_omega_rad_s),
            "quality_factor": values(self.quality_factor),
            "damping_per_pump_period": values(self.damping_per_pump_period),
            "junction_capacitance_f": values(self.junction_capacitance_f),
        }


def _as_branch_array(value: float | np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.full(count, float(array[0]), dtype=float)
    if array.size != count:
        raise ValueError(f"{name} must have length 1 or {count}, got {array.size}")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return array


def rcsj_parameters(
    critical_current_a: np.ndarray,
    junction_capacitance_f: float | np.ndarray,
    resistance_ratio: float,
    *,
    delta_ev: float = DEFAULT_AL_DELTA_EV,
    pump_frequency_hz: float = 7.12e9,
) -> RCSJParameters:
    """Return RCSJ parameters using the Ambegaokar--Baratoff relation.

    ``delta_ev`` is the superconducting gap expressed in electron-volts, so
    ``Ic * Rn = pi * delta_ev / 2`` has units of volts.  The reported damping
    is ``T_p / (R Cj)`` and is therefore directly comparable to a per-pump-
    period dielectric damping number.
    """
    ratio = float(resistance_ratio)
    if ratio <= 0.0 or math.isnan(ratio):
        raise ValueError("resistance_ratio must be positive or infinity")
    delta = float(delta_ev)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_ev must be finite and strictly positive")
    if not math.isfinite(pump_frequency_hz) or pump_frequency_hz <= 0.0:
        raise ValueError("pump_frequency_hz must be finite and strictly positive")

    ic = np.asarray(critical_current_a, dtype=float).reshape(-1)
    if ic.size == 0 or np.any(~np.isfinite(ic)) or np.any(ic <= 0.0):
        raise ValueError("critical_current_a must contain positive finite values")
    cj = _as_branch_array(junction_capacitance_f, ic.size, "junction_capacitance_f")
    ic_rn_v = math.pi * delta / 2.0
    rn = ic_rn_v / ic
    resistance = rn * ratio
    omega_p = np.sqrt(2.0 * math.pi * ic / (PHI0 * cj))
    beta_c = 2.0 * math.pi * ic * resistance**2 * cj / PHI0
    quality = omega_p * resistance * cj
    damping = 1.0 / (pump_frequency_hz * resistance * cj)
    return RCSJParameters(
        resistance_ratio=ratio,
        delta_ev=delta,
        critical_current_a=ic,
        junction_capacitance_f=cj,
        rn_ohm=rn,
        resistance_ohm=resistance,
        beta_c=beta_c,
        plasma_omega_rad_s=omega_p,
        quality_factor=quality,
        damping_per_pump_period=damping,
    )


def _metadata_junction_capacitance(circuit: CircuitMatrices) -> float | np.ndarray | None:
    metadata = circuit.metadata if isinstance(circuit.metadata, dict) else {}
    candidates = [metadata.get("junction_capacitance_f"), metadata.get("jj_capacitance_f")]
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        candidates.extend([nested.get("junction_capacitance_f"), nested.get("jj_capacitance_f")])
        derived = nested.get("derived")
        if isinstance(derived, dict):
            candidates.append(derived.get("jj_capacitance_f"))
    for value in candidates:
        if value is not None:
            return value
    return None


def stamp_rcsj_shunt(
    circuit: CircuitMatrices,
    resistance_ratio: float,
    *,
    junction_capacitance_f: float | np.ndarray | None = None,
    delta_ev: float = DEFAULT_AL_DELTA_EV,
    pump_frequency_hz: float = 7.12e9,
) -> tuple[CircuitMatrices, RCSJParameters]:
    """Stamp ``Bphi @ diag(1/Rj) @ Bphi.T`` into ``G``.

    Infinity is an exact control: the original ``CircuitMatrices`` object is
    returned without arithmetic, preserving every existing matrix and solver
    path bit-for-bit.  Finite settings return a separate circuit object.
    """
    ratio = float(resistance_ratio)
    if math.isinf(ratio) and ratio > 0.0:
        cj = junction_capacitance_f
        if cj is None:
            cj = _metadata_junction_capacitance(circuit)
        if cj is None:
            raise ValueError("junction_capacitance_f is required for RCSJ parameters")
        return circuit, rcsj_parameters(
            circuit.Ic, cj, ratio, delta_ev=delta_ev, pump_frequency_hz=pump_frequency_hz
        )

    if junction_capacitance_f is None:
        junction_capacitance_f = _metadata_junction_capacitance(circuit)
    if junction_capacitance_f is None:
        raise ValueError("junction_capacitance_f is required for RCSJ stamping")
    params = rcsj_parameters(
        circuit.Ic, junction_capacitance_f, ratio,
        delta_ev=delta_ev, pump_frequency_hz=pump_frequency_hz,
    )
    conductance = sp.diags(1.0 / params.resistance_ohm, format="csr")
    stamp = (circuit.Bphi @ conductance @ circuit.Bphi.T).tocsr()
    metadata = dict(circuit.metadata) if isinstance(circuit.metadata, dict) else {}
    metadata["rcsj"] = params.summary
    metadata["rcsj"]["stamp"] = "Bphi @ diag(1 / R_j) @ Bphi.T"
    return replace(circuit, G=(circuit.G + stamp).tocsr(), metadata=metadata), params


def rcsj_capacitance_from_metadata(circuit: CircuitMatrices) -> float | np.ndarray | None:
    """Public metadata lookup used by campaign/reporting code."""
    return _metadata_junction_capacitance(circuit)
