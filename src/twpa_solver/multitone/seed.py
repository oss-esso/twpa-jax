"""Warm starts for finite-signal multitone solves."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex, canonicalize
from twpa_solver.pump.basis import PumpBasis


def promote_pump_solution(
    X_pump: np.ndarray,
    pump_basis: PumpBasis,
    multitone_basis: MultiToneBasis,
) -> np.ndarray:
    """Embed positive pump coefficients into the ``q=0`` multitone sector."""
    source = np.asarray(X_pump, dtype=np.complex128)
    if source.shape[0] != pump_basis.n_modes:
        raise ValueError("pump solution row count does not match pump basis")
    result = np.zeros((multitone_basis.n_tones, source.shape[1]), dtype=np.complex128)
    for row, mode in enumerate(pump_basis.modes):
        tone = ToneIndex(int(mode), 0)
        if tone in multitone_basis.tones:
            result[multitone_basis.index_of(tone)] = source[row]
    return result


def seed_from_floquet(
    basis: MultiToneBasis,
    sideband_coeffs: Mapping[int, np.ndarray],
    *,
    omega_p: float,
    omega_s: float,
    signal_current_a: float,
    reference_current_a: float = 1.0,
) -> np.ndarray:
    """Map unit-current Floquet sidebands onto the multitone lattice."""
    if reference_current_a == 0.0:
        raise ValueError("reference_current_a must be nonzero")
    result: np.ndarray | None = None
    for sideband, coefficient in sideband_coeffs.items():
        frequency = omega_s + int(sideband) * omega_p
        delta = omega_p - omega_s
        tone, conjugated = canonicalize(
            ToneIndex(int(sideband) + 1, -1), omega_p, delta
        )
        if abs(tone.omega(omega_p, delta) - abs(frequency)) > 1e-9 * max(abs(frequency), 1.0):
            raise ValueError(f"Floquet sideband {sideband} does not map to an integer torus tone")
        if tone not in basis.tones:
            raise ValueError(f"Floquet sideband {sideband} maps to missing tone {tone}")
        values = np.asarray(coefficient, dtype=np.complex128).reshape(-1)
        if result is None:
            result = np.zeros((basis.n_tones, values.size), dtype=np.complex128)
        if values.size != result.shape[1]:
            raise ValueError("Floquet sidebands have inconsistent node dimensions")
        if conjugated:
            values = np.conj(values)
        result[basis.index_of(tone)] += values * (signal_current_a / reference_current_a)
    if result is None:
        raise ValueError("sideband_coeffs must contain at least one sideband")
    return result


def pump_plus_floquet_seed(
    X_pump: np.ndarray,
    pump_basis: PumpBasis,
    basis: MultiToneBasis,
    sideband_coeffs: Mapping[int, np.ndarray],
    *,
    omega_p: float,
    omega_s: float,
    signal_current_a: float,
    reference_current_a: float = 1.0,
) -> np.ndarray:
    """Combine a promoted pump state with a scaled Floquet signal seed."""
    return promote_pump_solution(X_pump, pump_basis, basis) + seed_from_floquet(
        basis,
        sideband_coeffs,
        omega_p=omega_p,
        omega_s=omega_s,
        signal_current_a=signal_current_a,
        reference_current_a=reference_current_a,
    )
