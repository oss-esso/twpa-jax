"""Warm starts for finite-signal multitone solves."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from twpa_solver.multitone.basis import (
    MultiToneBasis,
    ToneIndex,
    canonicalize,
)
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
        tone = ToneIndex(int(mode) * multitone_basis.pump_tone.h, 0)
        if tone in multitone_basis.tones:
            result[multitone_basis.index_of(tone)] = source[row]
    return result


def seed_torus_from_pump(
    X_pump: np.ndarray,
    pump_basis: PumpBasis,
    basis: MultiToneBasis,
    *,
    amplitude: float = 1.0e-6,
    node_ref: int = 0,
) -> np.ndarray:
    """Promote a pump state and add a deterministic autonomous perturbation.

    The perturbation occupies the represented ``q = +/-1`` sectors and fixes
    the generator phase at ``node_ref``.  Its scale is relative to the pump
    coefficient norm, so the same amplitude sweep is meaningful across
    circuits and pump powers.
    """
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    source = promote_pump_solution(X_pump, pump_basis, basis)
    if not 0 <= node_ref < source.shape[1]:
        raise ValueError("node_ref is outside the pump state")
    perturbation = np.zeros_like(source)
    scale = float(np.linalg.norm(source))
    if scale == 0.0:
        scale = 1.0
    for row, tone in enumerate(basis.tones):
        if tone.q == 1:
            perturbation[row, node_ref] = scale * amplitude
        elif tone.q == -1:
            perturbation[row, node_ref] = 0.5 * scale * amplitude
    if not np.any(perturbation):
        raise ValueError("basis has no represented q=+/-1 sector")
    return source + perturbation


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


def seed_torus_from_floquet(
    X_pump: np.ndarray,
    pump_basis: PumpBasis,
    basis: MultiToneBasis,
    floquet_vector: np.ndarray,
    floquet_sidebands: list[int] | np.ndarray,
    *,
    omega_p: float,
    omega_a: float,
    perturbation_amplitude: float = 1.0e-4,
    node_ref: int = 0,
) -> np.ndarray:
    """Embed a Floquet eigenvector into the autonomous ``q=1`` sector.

    A Floquet block ``m`` maps to the autonomous tone ``(m, 1)`` at physical
    frequency ``m * omega_p + omega_a``.  The real reconstruction supplies
    the conjugate ``q=-1`` sector.  The generator phase is rotated so the
    reference-node coefficient is real, matching the torus phase anchor.
    """
    if omega_p <= 0.0 or omega_a <= 0.0:
        raise ValueError("omega_p and omega_a must be positive")
    if perturbation_amplitude <= 0.0:
        raise ValueError("perturbation_amplitude must be positive")
    if not 0 <= node_ref < X_pump.shape[1]:
        raise ValueError("node_ref is outside the pump state")
    source = promote_pump_solution(X_pump, pump_basis, basis)
    vector = np.asarray(floquet_vector, dtype=np.complex128).reshape(-1)
    sidebands = [int(value) for value in np.asarray(floquet_sidebands).reshape(-1)]
    n = source.shape[1]
    if vector.size != len(sidebands) * n:
        raise ValueError("floquet_vector size does not match sidebands and nodes")
    perturbation = np.zeros_like(source)
    for block, sideband in enumerate(sidebands):
        values = vector[block * n : (block + 1) * n]
        tone, conjugated = canonicalize(
            ToneIndex(sideband, 1), omega_p, omega_a
        )
        if tone not in basis.tones:
            raise ValueError(f"Floquet sideband {sideband} maps to missing tone {tone}")
        if conjugated:
            values = np.conj(values)
        perturbation[basis.index_of(tone)] += values
    norm = float(np.linalg.norm(perturbation))
    if norm == 0.0:
        raise ValueError("floquet_vector has no representable torus content")
    anchor_value = perturbation[basis.index_of(ToneIndex(0, 1)), node_ref]
    if abs(anchor_value) > 0.0:
        perturbation *= np.exp(-1j * np.angle(anchor_value))
    scale = float(np.linalg.norm(source))
    if scale == 0.0:
        scale = 1.0
    return source + perturbation_amplitude * scale * perturbation / norm
