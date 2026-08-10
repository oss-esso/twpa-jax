"""Explicit pump-basis utilities for Floquet branch diagnostics.

These helpers do not assert that a period-doubled state exists.  They provide
the representation and seed required after an independently refined Hill root
has identified a candidate at ``-1``.
"""

from __future__ import annotations

import numpy as np

from twpa_solver.pump.basis import PumpBasis


def period_doubled_basis(
    basis: PumpBasis,
    *,
    max_mode: int | None = None,
) -> PumpBasis:
    """Return a dense half-pump basis with the physical pump at mode two.

    ``basis.omega_p`` is the physical pump frequency.  The returned basis has
    fundamental ``basis.omega_p / 2`` and includes every mode from DC through
    ``2 * max(basis.modes)``.  This is deliberately dense: omitting odd modes
    would make a period-doubled solution impossible to represent.
    """
    source_max = max(int(mode) for mode in basis.modes)
    limit = 2 * source_max if max_mode is None else int(max_mode)
    if limit < 2 or limit % 2:
        raise ValueError("period-doubled max_mode must be even and >= 2")
    if basis.omega_p <= 0.0:
        raise ValueError("basis.omega_p must be positive")
    return PumpBasis(
        modes=list(range(limit + 1)),
        policy="period_doubled_half_pump",
        omega_p=basis.omega_p / 2.0,
        basis=basis.basis,
        real_reconstruction_factor=basis.real_reconstruction_factor,
        phase_convention=basis.phase_convention,
        source_mode=2,
    )


def build_period_doubled_seed(
    pump_state: np.ndarray,
    pump_basis: PumpBasis,
    floquet_vector: np.ndarray,
    floquet_sidebands: list[int] | np.ndarray,
    target_basis: PumpBasis,
    *,
    perturbation_amplitude: float = 1.0e-4,
    perturbation_sign: float = 1.0,
) -> np.ndarray:
    """Embed a half-pump Hill eigenvector into a nonlinear HB seed.

    The Hill vector contains blocks at frequencies
    ``omega_p/2 + m*omega_p``.  Their half-pump indices are therefore
    ``1 + 2*m``.  Negative-frequency blocks are conjugated into the positive
    phasor representation.  The physical period-1 pump is copied to even
    half-pump modes and a normalized perturbation is added.

    This function only creates an initial guess.  The returned state must pass
    the ordinary production residual and provenance validation before it can be
    used for gain.
    """
    if perturbation_amplitude <= 0.0:
        raise ValueError("perturbation_amplitude must be positive")
    if perturbation_sign == 0.0 or not np.isfinite(perturbation_sign):
        raise ValueError("perturbation_sign must be finite and nonzero")
    source = np.asarray(pump_state, dtype=np.complex128)
    if source.ndim != 2 or source.shape[0] != pump_basis.n_modes:
        raise ValueError("pump_state shape does not match pump_basis")
    vector = np.asarray(floquet_vector, dtype=np.complex128).reshape(-1)
    sidebands = [int(m) for m in np.asarray(floquet_sidebands).reshape(-1)]
    n = source.shape[1]
    if vector.size != len(sidebands) * n:
        raise ValueError("floquet_vector size does not match sidebands and nodes")
    if target_basis.source_mode != 2 or not np.isclose(
        target_basis.omega_p, pump_basis.omega_p / 2.0
    ):
        raise ValueError("target_basis is not a half-pump basis for pump_basis")

    seed = np.zeros((target_basis.n_modes, n), dtype=np.complex128)
    source_rows = {int(mode): row for row, mode in enumerate(pump_basis.modes)}
    for mode, row in source_rows.items():
        target_mode = 2 * mode
        if target_mode in target_basis.modes:
            seed[target_basis.modes.index(target_mode)] = source[row]

    perturbation = np.zeros_like(seed)
    for block, sideband in enumerate(sidebands):
        half_mode = 1 + 2 * sideband
        values = vector[block * n : (block + 1) * n]
        if half_mode > 0 and half_mode in target_basis.modes:
            perturbation[target_basis.modes.index(half_mode)] += values
        elif half_mode < 0 and -half_mode in target_basis.modes:
            perturbation[target_basis.modes.index(-half_mode)] += np.conj(values)

    perturbation_norm = float(np.linalg.norm(perturbation))
    if perturbation_norm == 0.0:
        raise ValueError("floquet_vector has no representable half-pump content")
    reference_norm = max(float(np.linalg.norm(seed)), 1.0)
    seed += perturbation * (
        perturbation_sign * perturbation_amplitude * reference_norm / perturbation_norm
    )
    return seed
