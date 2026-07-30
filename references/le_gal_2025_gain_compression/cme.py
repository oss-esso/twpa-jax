"""Independent three-envelope coupled-mode oracle.

This module intentionally has no imports from :mod:`twpa_solver`.  It is a
small reference integration of the pump, signal, and idler envelopes and is
used to establish morphology before a harmonic-balance comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CMEParameters:
    """Three-envelope equations with optional Appendix-C-style Kerr terms.

    The first six fields preserve the normalized oracle contract.  The Kerr
    and physical-unit fields make power-dependent phase matching explicit;
    coefficients are supplied by the benchmark calibration rather than by
    the HB implementation.
    """

    length: float = 1.0
    coupling: float = 1.0
    phase_mismatch: float = 0.0
    loss_p: float = 0.0
    loss_s: float = 0.0
    loss_i: float = 0.0
    self_phase_p: float = 0.0
    self_phase_s: float = 0.0
    self_phase_i: float = 0.0
    cross_phase_ps: float = 0.0
    cross_phase_pi: float = 0.0
    cross_phase_si: float = 0.0
    omega_p: float = 2.0 * np.pi * 7.5e9
    omega_s: float = 2.0 * np.pi * 6.0e9
    omega_i: float = 2.0 * np.pi * 9.0e9
    z0: float = 62.3765


def published_cme_parameters(
    signal_frequency_hz: float,
    *,
    pump_frequency_hz: float = 7.5e9,
    pump_power_w: float = 10.0 ** ((-78.4 - 30.0) / 10.0),
    cells: int = 700,
    cell_length_m: float = 8.7e-6,
    critical_current_a: float = 1.4e-6,
    ratio: float = 0.062,
    inductance_h: float = 869.6e-12,
    ground_capacitance_f: float = 223.5e-15,
    snail_capacitance_f: float = 31e-15,
) -> CMEParameters:
    """Derive a coefficient-complete normalized CME from paper parameters.

    The derivation uses the small-signal branch slope and cubic Taylor
    coefficient of the half-flux SNAIL, and the exact discrete ladder
    dispersion ``2 asin(omega sqrt(L C)/2) / dx``.  The resulting coefficients
    are inferred model quantities, not digitized paper data.
    """
    phi0 = 2.067833848e-15 / (2.0 * np.pi)
    omega_p = 2.0 * np.pi * pump_frequency_hz
    omega_s = 2.0 * np.pi * signal_frequency_hz
    omega_i = 2.0 * omega_p - omega_s
    slope_factor = ratio * (-1.0) + 1.0 / 3.0
    slope = critical_current_a * slope_factor / phi0
    linear_l = 1.0 / slope
    cubic_current = critical_current_a * (ratio / 6.0 - 1.0 / 162.0) / phi0**3
    z0 = float(np.sqrt(inductance_h / ground_capacitance_f))
    def wave_number(omega: float) -> float:
        argument = omega * np.sqrt((inductance_h + snail_capacitance_f / (ground_capacitance_f / inductance_h)) * ground_capacitance_f)
        argument = np.clip(argument * 0.5, -0.999999999, 0.999999999)
        return 2.0 * np.arcsin(argument) / cell_length_m
    mismatch = wave_number(omega_s) + wave_number(omega_i) - 2.0 * wave_number(omega_p)
    k_p = wave_number(omega_p)
    pump_envelope = np.sqrt(pump_power_w * z0) / omega_p
    # Projection of a real cubic waveform onto its positive-frequency
    # envelope contributes the standard 1/8 Fourier factor.
    gamma = (cubic_current / slope) * k_p / 8.0
    # ``_rhs`` multiplies coupling by two envelope amplitudes, so this is the
    # physical coefficient in 1/(m Wb^2), not the already pump-scaled gain.
    coupling = abs(gamma)
    # SPM/XPM coefficients follow the same Taylor phase-shift coefficient;
    # the factors are the Appendix-C three-wave degeneracy factors.
    return CMEParameters(
        length=cells * cell_length_m,
        coupling=float(coupling),
        phase_mismatch=float(mismatch),
        self_phase_p=float(gamma),
        self_phase_s=float(gamma),
        self_phase_i=float(gamma),
        cross_phase_ps=float(2.0 * gamma),
        cross_phase_pi=float(2.0 * gamma),
        cross_phase_si=float(2.0 * gamma),
        omega_p=omega_p,
        omega_s=omega_s,
        omega_i=omega_i,
        z0=z0,
    )


def _rhs(z: float, y: np.ndarray, p: CMEParameters) -> np.ndarray:
    ap, ass, ai = y[:3] + 1j * y[3:]
    phase = np.exp(1j * p.phase_mismatch * z)
    coupling = 1j * p.coupling
    derivative = np.array(
        [
            -0.5 * p.loss_p * ap
            + 1j * (p.self_phase_p * abs(ap) ** 2 + p.cross_phase_ps * abs(ass) ** 2 + p.cross_phase_pi * abs(ai) ** 2) * ap
            + 2.0 * coupling * ass * ai * np.conj(phase),
            -0.5 * p.loss_s * ass
            + 1j * (p.self_phase_s * abs(ass) ** 2 + p.cross_phase_ps * abs(ap) ** 2 + p.cross_phase_si * abs(ai) ** 2) * ass
            + coupling * ap * np.conj(ai) * phase,
            -0.5 * p.loss_i * ai
            + 1j * (p.self_phase_i * abs(ai) ** 2 + p.cross_phase_pi * abs(ap) ** 2 + p.cross_phase_si * abs(ass) ** 2) * ai
            + coupling * ap * np.conj(ass) * phase,
        ],
        dtype=np.complex128,
    )
    return np.concatenate((derivative.real, derivative.imag))


def integrate_cme(
    initial: tuple[complex, complex, complex],
    parameters: CMEParameters = CMEParameters(),
    *,
    points: int = 401,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the independent CME and return ``(z, envelopes)``."""
    if points < 2:
        raise ValueError("points must be at least 2")
    initial_array = np.asarray(initial, dtype=np.complex128)
    y0 = np.concatenate((initial_array.real, initial_array.imag))
    z = np.linspace(0.0, parameters.length, points)
    result = solve_ivp(
        lambda coordinate, state: _rhs(coordinate, state, parameters),
        (0.0, parameters.length),
        y0,
        t_eval=z,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        method="DOP853",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return z, result.y[:3] + 1j * result.y[3:]


def envelopes_from_powers(
    pump_power_w: float, signal_power_w: float, parameters: CMEParameters
) -> tuple[complex, complex, complex]:
    """Apply the paper's physical input convention, with an empty idler."""
    if min(pump_power_w, signal_power_w) < 0.0:
        raise ValueError("powers must be nonnegative")
    return (
        np.sqrt(pump_power_w * parameters.z0) / parameters.omega_p,
        np.sqrt(signal_power_w * parameters.z0) / parameters.omega_s,
        0.0j,
    )


def photon_flux(envelopes: np.ndarray) -> np.ndarray:
    """Return the normalized three-envelope photon invariant."""
    if envelopes.shape[0] != 3:
        raise ValueError("envelopes must have three rows")
    return np.sum(np.abs(envelopes) ** 2, axis=0)


def depletion_only_gain(gain_linear: float, signal_power: float, pump_power: float) -> float:
    """Return the paper's simple depletion model in linear power units."""
    if min(gain_linear, signal_power, pump_power) <= 0.0:
        raise ValueError("powers and gain must be positive")
    return gain_linear / (1.0 + 2.0 * gain_linear * signal_power / pump_power)
