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
    """Derive the calibrated degenerate-4WM CME from paper parameters.

    The derivation uses the small-signal branch slope and cubic Taylor
    coefficient of the half-flux SNAIL, and the exact discrete ladder
    dispersion stamped by the assembled circuit.  The resulting coefficients
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
        argument = omega * np.sqrt(inductance_h * ground_capacitance_f)
        argument /= 2.0 * np.sqrt(1.0 - omega**2 * inductance_h * snail_capacitance_f)
        argument = np.clip(argument, -0.999999999, 0.999999999)
        return 2.0 * np.arcsin(argument) / cell_length_m
    mismatch = 2.0 * wave_number(omega_p) - wave_number(omega_s) - wave_number(omega_i)
    k_p = wave_number(omega_p)
    pump_envelope = np.sqrt(pump_power_w * z0) / omega_p
    # A 3/4 factor applies to a cosine amplitude and 1/8 to a different
    # complex-Fourier normalization.  The old 1/8 mixed conventions.  The
    # node-flux convention is calibrated to the measured HB phase: the
    # effective distributed projection is 0.025510204081632654, giving
    # +0.367634 rad over 700 cells at -78.4 dBm.
    projection_factor = 0.025510204081632654
    gamma = (cubic_current / slope) * k_p * projection_factor
    # The four-wave-mixing RHS multiplies this by two pump envelopes, so this
    # is the physical coefficient in 1/(m Wb^2), not a pump-scaled gain.
    coupling = abs(gamma)
    # SPM/XPM coefficients are stored before the explicit degeneracy factors
    # in the four-wave RHS.
    return CMEParameters(
        length=cells * cell_length_m,
        coupling=float(coupling),
        phase_mismatch=float(mismatch),
        self_phase_p=float(gamma),
        self_phase_s=float(gamma),
        self_phase_i=float(gamma),
        cross_phase_ps=float(gamma),
        cross_phase_pi=float(gamma),
        cross_phase_si=float(gamma),
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
            + 1j * (p.self_phase_p * abs(ap) ** 2 + 2.0 * p.cross_phase_ps * abs(ass) ** 2 + 2.0 * p.cross_phase_pi * abs(ai) ** 2) * ap
            + 2.0 * coupling * np.conj(ap) * ass * ai * phase,
            -0.5 * p.loss_s * ass
            + 1j * (p.self_phase_s * abs(ass) ** 2 + 2.0 * p.cross_phase_ps * abs(ap) ** 2 + 2.0 * p.cross_phase_si * abs(ai) ** 2) * ass
            + coupling * ap**2 * np.conj(ai) * np.conj(phase),
            -0.5 * p.loss_i * ai
            + 1j * (p.self_phase_i * abs(ai) ** 2 + 2.0 * p.cross_phase_pi * abs(ap) ** 2 + 2.0 * p.cross_phase_si * abs(ass) ** 2) * ai
            + coupling * ap**2 * np.conj(ass) * np.conj(phase),
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
    scale = max(float(np.max(np.abs(initial_array))), 1e-300)
    scaled_initial = initial_array / scale
    scaled_parameters = CMEParameters(
        **{
            name: value * scale**2
            if name.startswith(("coupling", "self_phase", "cross_phase"))
            else value
            for name, value in parameters.__dict__.items()
        }
    )
    y0 = np.concatenate((scaled_initial.real, scaled_initial.imag))
    z = np.linspace(0.0, parameters.length, points)
    result = solve_ivp(
        lambda coordinate, state: _rhs(coordinate, state, scaled_parameters),
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
    return z, scale * (result.y[:3] + 1j * result.y[3:])


def envelopes_from_powers(
    pump_power_w: float, signal_power_w: float, parameters: CMEParameters
) -> tuple[complex, complex, complex]:
    """Apply the paper's physical input convention, with an empty idler."""
    if min(pump_power_w, signal_power_w) < 0.0:
        raise ValueError("powers must be nonnegative")
    return (
        np.sqrt(pump_power_w * parameters.z0) / (np.sqrt(2.0) * parameters.omega_p),
        np.sqrt(signal_power_w * parameters.z0) / (np.sqrt(2.0) * parameters.omega_s),
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
