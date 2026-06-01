"""
twpa.core.units
===============

Unit constants and RF power/voltage/current conversion utilities.

Design rules
------------
1. All internal simulator quantities are SI.
2. Public helpers may accept engineering units, but must return SI unless
   explicitly named otherwise.
3. Functions are JAX-compatible and work with Python scalars or JAX arrays.
4. No hidden global state is used.
5. dBm helpers assume real RF power delivered to a reference impedance.
6. Voltage/current helpers distinguish RMS and peak amplitudes explicitly.

This module is intentionally small and dependency-light because every later
layer depends on it: parameter objects, RF networks, harmonic balance residuals,
pump-power initialization, gain extraction, and fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


ArrayLike = Any


# ---------------------------------------------------------------------------
# SI prefixes
# ---------------------------------------------------------------------------

YOTTA: float = 1e24
ZETTA: float = 1e21
EXA: float = 1e18
PETA: float = 1e15
TERA: float = 1e12
GIGA: float = 1e9
MEGA: float = 1e6
KILO: float = 1e3
HECTO: float = 1e2
DECA: float = 1e1

DECI: float = 1e-1
CENTI: float = 1e-2
MILLI: float = 1e-3
MICRO: float = 1e-6
NANO: float = 1e-9
PICO: float = 1e-12
FEMTO: float = 1e-15
ATTO: float = 1e-18
ZEPTO: float = 1e-21
YOCTO: float = 1e-24


# Common short aliases used in circuit code.
GHz: float = GIGA
MHz: float = MEGA
kHz: float = KILO

mm: float = MILLI
um: float = MICRO
nm: float = NANO

uA: float = MICRO
mA: float = MILLI

nH: float = NANO
pH: float = PICO

pF: float = PICO
fF: float = FEMTO


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalConstants:
    """
    Physical constants in SI units.

    Values are CODATA exact or standard SI definitions where applicable.
    """

    c0: float = 299_792_458.0
    mu0: float = 1.25663706212e-6
    eps0: float = 8.8541878128e-12
    h: float = 6.62607015e-34
    hbar: float = 1.054571817e-34
    e: float = 1.602176634e-19
    k_B: float = 1.380649e-23

    @property
    def phi0(self) -> float:
        """Magnetic flux quantum Phi_0 = h / (2e), in Wb."""
        return self.h / (2.0 * self.e)

    @property
    def reduced_phi0(self) -> float:
        """Reduced flux quantum phi_0 = Phi_0 / (2π), in Wb/rad."""
        return self.phi0 / (2.0 * float(jnp.pi))

    @property
    def z_vacuum(self) -> float:
        """Vacuum impedance sqrt(mu0 / eps0), in ohm."""
        return float(jnp.sqrt(self.mu0 / self.eps0))


CONSTANTS = PhysicalConstants()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def asarray_si(x: ArrayLike, dtype: Any | None = None) -> jax.Array:
    """
    Convert input to a JAX array without changing units.

    Parameters
    ----------
    x:
        Scalar, list, NumPy array, or JAX array.
    dtype:
        Optional dtype. If omitted, JAX default dtype is used. In the thesis
        simulator, x64 should be enabled globally by the environment/bootstrap
        script, not here.

    Returns
    -------
    jax.Array
        JAX array representation of x.
    """
    if dtype is None:
        return jnp.asarray(x)
    return jnp.asarray(x, dtype=dtype)


def _safe_positive(x: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """
    Clip a quantity to a small positive floor for logarithms/divisions.

    This is only used in conversion helpers where log(0) would otherwise
    produce -inf. Solver residuals should not silently clip physical variables.
    """
    return jnp.maximum(asarray_si(x), floor)


# ---------------------------------------------------------------------------
# Frequency / angular-frequency conversions
# ---------------------------------------------------------------------------

def hz(value: ArrayLike) -> jax.Array:
    """Return frequency in Hz from a value already expressed in Hz."""
    return asarray_si(value)


def ghz(value: ArrayLike) -> jax.Array:
    """Convert GHz to Hz."""
    return asarray_si(value) * GHz


def mhz(value: ArrayLike) -> jax.Array:
    """Convert MHz to Hz."""
    return asarray_si(value) * MHz


def khz(value: ArrayLike) -> jax.Array:
    """Convert kHz to Hz."""
    return asarray_si(value) * kHz


def angular_frequency(freq_hz: ArrayLike) -> jax.Array:
    """
    Convert frequency f in Hz to angular frequency omega in rad/s.

    omega = 2πf
    """
    return 2.0 * jnp.pi * asarray_si(freq_hz)


def frequency_from_angular(omega_rad_s: ArrayLike) -> jax.Array:
    """
    Convert angular frequency omega in rad/s to frequency f in Hz.

    f = omega / (2π)
    """
    return asarray_si(omega_rad_s) / (2.0 * jnp.pi)


# ---------------------------------------------------------------------------
# dB / magnitude conversions
# ---------------------------------------------------------------------------

def db_to_power_ratio(db: ArrayLike) -> jax.Array:
    """Convert dB to power ratio."""
    return 10.0 ** (asarray_si(db) / 10.0)


def power_ratio_to_db(ratio: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """Convert power ratio to dB."""
    return 10.0 * jnp.log10(_safe_positive(ratio, floor=floor))


def db_to_amplitude_ratio(db: ArrayLike) -> jax.Array:
    """Convert dB to voltage/current amplitude ratio."""
    return 10.0 ** (asarray_si(db) / 20.0)


def amplitude_ratio_to_db(ratio: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """Convert voltage/current amplitude ratio to dB."""
    return 20.0 * jnp.log10(_safe_positive(ratio, floor=floor))


# ---------------------------------------------------------------------------
# RF power conversions
# ---------------------------------------------------------------------------

def dbm_to_watts(dbm: ArrayLike) -> jax.Array:
    """
    Convert power in dBm to watts.

    P[W] = 1e-3 * 10^(P[dBm]/10)
    """
    return 1e-3 * 10.0 ** (asarray_si(dbm) / 10.0)


def watts_to_dbm(power_w: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """
    Convert power in watts to dBm.

    P[dBm] = 10 log10(P[W] / 1mW)
    """
    return 10.0 * jnp.log10(_safe_positive(power_w, floor=floor) / 1e-3)


def dbw_to_watts(dbw: ArrayLike) -> jax.Array:
    """Convert power in dBW to watts."""
    return 10.0 ** (asarray_si(dbw) / 10.0)


def watts_to_dbw(power_w: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """Convert power in watts to dBW."""
    return 10.0 * jnp.log10(_safe_positive(power_w, floor=floor))


# ---------------------------------------------------------------------------
# RMS / peak conversions
# ---------------------------------------------------------------------------

def rms_to_peak(x_rms: ArrayLike) -> jax.Array:
    """Convert sinusoidal RMS amplitude to peak amplitude."""
    return jnp.sqrt(2.0) * asarray_si(x_rms)


def peak_to_rms(x_peak: ArrayLike) -> jax.Array:
    """Convert sinusoidal peak amplitude to RMS amplitude."""
    return asarray_si(x_peak) / jnp.sqrt(2.0)


def peak_to_peak_to_peak(x_pp: ArrayLike) -> jax.Array:
    """Convert peak-to-peak sinusoidal amplitude to peak amplitude."""
    return asarray_si(x_pp) / 2.0


def peak_to_peak_to_rms(x_pp: ArrayLike) -> jax.Array:
    """Convert peak-to-peak sinusoidal amplitude to RMS amplitude."""
    return asarray_si(x_pp) / (2.0 * jnp.sqrt(2.0))


def peak_to_peak_from_peak(x_peak: ArrayLike) -> jax.Array:
    """Convert peak sinusoidal amplitude to peak-to-peak amplitude."""
    return 2.0 * asarray_si(x_peak)


def peak_to_peak_from_rms(x_rms: ArrayLike) -> jax.Array:
    """Convert RMS sinusoidal amplitude to peak-to-peak amplitude."""
    return 2.0 * jnp.sqrt(2.0) * asarray_si(x_rms)


# ---------------------------------------------------------------------------
# Power, voltage, current and impedance
# ---------------------------------------------------------------------------

def watts_to_vrms(power_w: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """
    Convert delivered RF power to RMS voltage across a real impedance.

    P = V_rms^2 / Z
    """
    return jnp.sqrt(asarray_si(power_w) * asarray_si(z_ohm))


def watts_to_irms(power_w: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """
    Convert delivered RF power to RMS current through a real impedance.

    P = I_rms^2 Z
    """
    return jnp.sqrt(asarray_si(power_w) / asarray_si(z_ohm))


def watts_to_vpeak(power_w: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert delivered RF power to peak sinusoidal voltage."""
    return rms_to_peak(watts_to_vrms(power_w, z_ohm))


def watts_to_ipeak(power_w: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert delivered RF power to peak sinusoidal current."""
    return rms_to_peak(watts_to_irms(power_w, z_ohm))


def dbm_to_vrms(dbm: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert dBm to RMS voltage across a real impedance."""
    return watts_to_vrms(dbm_to_watts(dbm), z_ohm)


def dbm_to_irms(dbm: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert dBm to RMS current through a real impedance."""
    return watts_to_irms(dbm_to_watts(dbm), z_ohm)


def dbm_to_vpeak(dbm: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert dBm to peak sinusoidal voltage across a real impedance."""
    return watts_to_vpeak(dbm_to_watts(dbm), z_ohm)


def dbm_to_ipeak(dbm: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert dBm to peak sinusoidal current through a real impedance."""
    return watts_to_ipeak(dbm_to_watts(dbm), z_ohm)


def vrms_to_watts(v_rms: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert RMS voltage across a real impedance to delivered power."""
    return asarray_si(v_rms) ** 2 / asarray_si(z_ohm)


def irms_to_watts(i_rms: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert RMS current through a real impedance to delivered power."""
    return asarray_si(i_rms) ** 2 * asarray_si(z_ohm)


def vpeak_to_watts(v_peak: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert peak sinusoidal voltage to delivered power."""
    return vrms_to_watts(peak_to_rms(v_peak), z_ohm)


def ipeak_to_watts(i_peak: ArrayLike, z_ohm: ArrayLike = 50.0) -> jax.Array:
    """Convert peak sinusoidal current to delivered power."""
    return irms_to_watts(peak_to_rms(i_peak), z_ohm)


def voltage_current_to_power(v_rms: ArrayLike, i_rms: ArrayLike) -> jax.Array:
    """
    Convert RMS voltage/current to average real power for matched phase.

    For complex phasors use complex_power().
    """
    return asarray_si(v_rms) * asarray_si(i_rms)


def complex_power(v_rms_phasor: ArrayLike, i_rms_phasor: ArrayLike) -> jax.Array:
    """
    Complex power S = V_rms * conj(I_rms).

    Returns complex apparent power in VA. The real part is average power.
    """
    return asarray_si(v_rms_phasor) * jnp.conj(asarray_si(i_rms_phasor))


# ---------------------------------------------------------------------------
# Traveling-wave / RF power-wave variables
# ---------------------------------------------------------------------------

def voltage_current_to_power_waves(
    voltage: ArrayLike,
    current: ArrayLike,
    z0: ArrayLike = 50.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Convert port voltage/current phasors to incident/reflected power waves.

    Convention
    ----------
    a = (V + Z0 I) / (2 sqrt(Z0))
    b = (V - Z0 I) / (2 sqrt(Z0))

    This is the convention used by the later conversion-matrix layer.

    Parameters
    ----------
    voltage:
        Complex voltage phasor.
    current:
        Complex current phasor entering the port.
    z0:
        Real reference impedance.

    Returns
    -------
    a, b:
        Incident and reflected/scattered power-wave phasors. |a|^2 and |b|^2
        have units of watts if V and I are RMS phasors.
    """
    v = asarray_si(voltage)
    i = asarray_si(current)
    z = asarray_si(z0)
    denom = 2.0 * jnp.sqrt(z)
    a = (v + z * i) / denom
    b = (v - z * i) / denom
    return a, b


def power_waves_to_voltage_current(
    a: ArrayLike,
    b: ArrayLike,
    z0: ArrayLike = 50.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Convert incident/reflected power waves to port voltage/current phasors.

    Inverse of voltage_current_to_power_waves().

    V = sqrt(Z0) (a + b)
    I = (a - b) / sqrt(Z0)
    """
    aa = asarray_si(a)
    bb = asarray_si(b)
    z = asarray_si(z0)
    sqrt_z = jnp.sqrt(z)
    voltage = sqrt_z * (aa + bb)
    current = (aa - bb) / sqrt_z
    return voltage, current


def available_power_to_power_wave(power_w: ArrayLike) -> jax.Array:
    """
    Return incident power-wave magnitude for available power P.

    Since |a|^2 = P, the RMS power-wave amplitude is sqrt(P).
    """
    return jnp.sqrt(asarray_si(power_w))


def dbm_to_power_wave(dbm: ArrayLike) -> jax.Array:
    """Convert dBm to incident power-wave magnitude sqrt(W)."""
    return available_power_to_power_wave(dbm_to_watts(dbm))


# ---------------------------------------------------------------------------
# Gain helpers
# ---------------------------------------------------------------------------

def voltage_gain_db(v_out: ArrayLike, v_in: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """Voltage gain in dB: 20 log10(|Vout/Vin|)."""
    ratio = jnp.abs(asarray_si(v_out)) / _safe_positive(jnp.abs(asarray_si(v_in)), floor)
    return amplitude_ratio_to_db(ratio, floor=floor)


def power_gain_db(p_out: ArrayLike, p_in: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """Power gain in dB: 10 log10(Pout/Pin)."""
    ratio = _safe_positive(p_out, floor) / _safe_positive(p_in, floor)
    return power_ratio_to_db(ratio, floor=floor)


def sparam_gain_db(s_complex: ArrayLike, floor: float = 1e-300) -> jax.Array:
    """
    Convert complex S-parameter magnitude to dB.

    For matched equal-reference impedance ports, power gain from S21 is |S21|^2,
    so the dB gain is 20 log10 |S21|.
    """
    return amplitude_ratio_to_db(jnp.abs(asarray_si(s_complex)), floor=floor)


# ---------------------------------------------------------------------------
# Engineering formatting helpers
# ---------------------------------------------------------------------------

def to_float(x: ArrayLike) -> float:
    """
    Convert scalar-like value to Python float.

    Intended only for reporting, JSON serialization, and logging.
    Do not use inside JIT-compiled code.
    """
    return float(jnp.asarray(x))


def scalar_summary(x: ArrayLike) -> dict[str, float]:
    """
    Return basic scalar/array summary useful for JSON reports.

    Do not use inside JIT-compiled code.
    """
    arr = jnp.asarray(x)
    abs_arr = jnp.abs(arr)
    return {
        "min": float(jnp.min(abs_arr)),
        "max": float(jnp.max(abs_arr)),
        "mean": float(jnp.mean(abs_arr)),
        "shape_size": float(arr.size),
    }


__all__ = [
    "ArrayLike",
    "PhysicalConstants",
    "CONSTANTS",
    "YOTTA",
    "ZETTA",
    "EXA",
    "PETA",
    "TERA",
    "GIGA",
    "MEGA",
    "KILO",
    "HECTO",
    "DECA",
    "DECI",
    "CENTI",
    "MILLI",
    "MICRO",
    "NANO",
    "PICO",
    "FEMTO",
    "ATTO",
    "ZEPTO",
    "YOCTO",
    "GHz",
    "MHz",
    "kHz",
    "mm",
    "um",
    "nm",
    "uA",
    "mA",
    "nH",
    "pH",
    "pF",
    "fF",
    "asarray_si",
    "hz",
    "ghz",
    "mhz",
    "khz",
    "angular_frequency",
    "frequency_from_angular",
    "db_to_power_ratio",
    "power_ratio_to_db",
    "db_to_amplitude_ratio",
    "amplitude_ratio_to_db",
    "dbm_to_watts",
    "watts_to_dbm",
    "dbw_to_watts",
    "watts_to_dbw",
    "rms_to_peak",
    "peak_to_rms",
    "peak_to_peak_to_peak",
    "peak_to_peak_to_rms",
    "peak_to_peak_from_peak",
    "peak_to_peak_from_rms",
    "watts_to_vrms",
    "watts_to_irms",
    "watts_to_vpeak",
    "watts_to_ipeak",
    "dbm_to_vrms",
    "dbm_to_irms",
    "dbm_to_vpeak",
    "dbm_to_ipeak",
    "vrms_to_watts",
    "irms_to_watts",
    "vpeak_to_watts",
    "ipeak_to_watts",
    "voltage_current_to_power",
    "complex_power",
    "voltage_current_to_power_waves",
    "power_waves_to_voltage_current",
    "available_power_to_power_wave",
    "dbm_to_power_wave",
    "voltage_gain_db",
    "power_gain_db",
    "sparam_gain_db",
    "to_float",
    "scalar_summary",
]