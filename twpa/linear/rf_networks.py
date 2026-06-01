"""
twpa.linear.rf_networks
=======================

Low-level RF two-port network utilities.

This module provides ABCD matrices, S-parameter conversion, lumped element
two-ports, and passive network diagnostics. It is the foundation for the linear
100 mm / 20,000-cell simulator.

Conventions
-----------
ABCD / transmission matrix convention:

    [V1]   [A B] [V2]
    [I1] = [C D] [I2]

where currents are defined entering the ports under the standard RF two-port
convention used by the ABCD-to-S formulas below.

For equal real reference impedance Z0:

    denom = A + B/Z0 + C Z0 + D

    S11 = (A + B/Z0 - C Z0 - D) / denom
    S21 = 2 / denom
    S12 = 2 (A D - B C) / denom
    S22 = (-A + B/Z0 - C Z0 + D) / denom

For reciprocal passive lumped networks, det(ABCD) = A D - B C ≈ 1, so
S12 = S21.

Shape convention
----------------
All network matrices use trailing shape (..., 2, 2). Usually the leading axis
is frequency:

    abcd.shape = (F, 2, 2)
    s.shape    = (F, 2, 2)

Scalar-frequency inputs also return a batch dimension of length 1 unless the
caller squeezes manually. This keeps downstream code simpler.

This file intentionally avoids topology-specific long-line cascade logic. That
comes later in:

    twpa.linear.cells
    twpa.linear.cascade
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from twpa.core.units import angular_frequency, sparam_gain_db


ArrayLike = Any


# ---------------------------------------------------------------------------
# Basic array helpers
# ---------------------------------------------------------------------------

def _as_frequency_array(frequency_hz: ArrayLike) -> jax.Array:
    """
    Convert scalar or vector frequency input to a 1D float array.
    """
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if f.ndim == 0:
        f = f.reshape((1,))
    if f.ndim != 1:
        raise ValueError(f"frequency_hz must be scalar or 1D, got shape {f.shape}")
    if bool(jnp.any(f < 0.0)):
        raise ValueError("frequency_hz must be non-negative for passive RF networks")
    return f


def _as_complex(x: ArrayLike) -> jax.Array:
    arr = jnp.asarray(x)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    return arr


def _as_real(x: ArrayLike) -> jax.Array:
    arr = jnp.asarray(x)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        arr = arr.astype(jnp.float64)
    return arr


def _broadcast_to_frequency(name: str, value: ArrayLike, f: jax.Array) -> jax.Array:
    """
    Broadcast scalar or length-F array to frequency shape (F,).
    """
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return jnp.full_like(f, arr, dtype=arr.dtype)
    if arr.ndim == 1 and arr.shape[0] == f.shape[0]:
        return arr
    try:
        return jnp.broadcast_to(arr, f.shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {arr.shape} cannot broadcast to frequency shape {f.shape}"
        ) from exc


def _check_abcd_shape(name: str, matrix: ArrayLike) -> jax.Array:
    m = _as_complex(matrix)
    if m.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have trailing shape (2, 2), got {m.shape}")
    return m


def _check_s_shape(name: str, matrix: ArrayLike) -> jax.Array:
    m = _as_complex(matrix)
    if m.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have trailing shape (2, 2), got {m.shape}")
    return m


def _identity_like_batch(batch_shape: tuple[int, ...], dtype: Any = jnp.complex128) -> jax.Array:
    eye = jnp.eye(2, dtype=dtype)
    return jnp.broadcast_to(eye, batch_shape + (2, 2))


# ---------------------------------------------------------------------------
# ABCD construction
# ---------------------------------------------------------------------------

def abcd_identity(
    frequency_hz: ArrayLike | None = None,
    *,
    n: int | None = None,
    dtype: Any = jnp.complex128,
) -> jax.Array:
    """
    Return identity ABCD matrix/matrices.

    Either pass frequency_hz or n.
    """
    if frequency_hz is None and n is None:
        return jnp.eye(2, dtype=dtype).reshape((1, 2, 2))
    if frequency_hz is not None:
        f = _as_frequency_array(frequency_hz)
        return _identity_like_batch((f.shape[0],), dtype=dtype)
    if n is None or int(n) <= 0:
        raise ValueError("n must be positive")
    return _identity_like_batch((int(n),), dtype=dtype)


def abcd_series_impedance(z_ohm: ArrayLike) -> jax.Array:
    """
    ABCD matrix for a series impedance.

        [1 Z]
        [0 1]

    z_ohm can be scalar or shape (F,).
    """
    z = _as_complex(z_ohm)
    if z.ndim == 0:
        z = z.reshape((1,))
    if z.ndim != 1:
        raise ValueError(f"z_ohm must be scalar or 1D, got shape {z.shape}")

    zeros = jnp.zeros_like(z)
    ones = jnp.ones_like(z)
    return jnp.stack(
        [
            jnp.stack([ones, z], axis=-1),
            jnp.stack([zeros, ones], axis=-1),
        ],
        axis=-2,
    )


def abcd_shunt_admittance(y_S: ArrayLike) -> jax.Array:
    """
    ABCD matrix for a shunt admittance.

        [1 0]
        [Y 1]

    y_S can be scalar or shape (F,).
    """
    y = _as_complex(y_S)
    if y.ndim == 0:
        y = y.reshape((1,))
    if y.ndim != 1:
        raise ValueError(f"y_S must be scalar or 1D, got shape {y.shape}")

    zeros = jnp.zeros_like(y)
    ones = jnp.ones_like(y)
    return jnp.stack(
        [
            jnp.stack([ones, zeros], axis=-1),
            jnp.stack([y, ones], axis=-1),
        ],
        axis=-2,
    )


def abcd_series_resistor(
    frequency_hz: ArrayLike,
    *,
    R_ohm: ArrayLike,
) -> jax.Array:
    """ABCD matrix for a series resistor."""
    f = _as_frequency_array(frequency_hz)
    R = _broadcast_to_frequency("R_ohm", R_ohm, f)
    return abcd_series_impedance(R.astype(jnp.complex128))


def abcd_series_inductor(
    frequency_hz: ArrayLike,
    *,
    L_H: ArrayLike,
    R_ohm: ArrayLike = 0.0,
) -> jax.Array:
    """
    ABCD matrix for a series inductor with optional series resistance.

        Z = R + j omega L
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)
    L = _broadcast_to_frequency("L_H", L_H, f)
    R = _broadcast_to_frequency("R_ohm", R_ohm, f)
    z = R + 1j * omega * L
    return abcd_series_impedance(z)


def abcd_shunt_capacitor(
    frequency_hz: ArrayLike,
    *,
    C_F: ArrayLike,
    G_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    ABCD matrix for a shunt capacitor with optional conductance.

        Y = G + j omega C
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)
    C = _broadcast_to_frequency("C_F", C_F, f)
    G = _broadcast_to_frequency("G_S", G_S, f)
    y = G + 1j * omega * C
    return abcd_shunt_admittance(y)


def abcd_shunt_resistor(
    frequency_hz: ArrayLike,
    *,
    R_ohm: ArrayLike,
) -> jax.Array:
    """
    ABCD matrix for a shunt resistor.

        Y = 1/R
    """
    f = _as_frequency_array(frequency_hz)
    R = _broadcast_to_frequency("R_ohm", R_ohm, f)
    if bool(jnp.any(R <= 0.0)):
        raise ValueError("R_ohm must be positive for shunt resistor")
    y = 1.0 / R
    return abcd_shunt_admittance(y.astype(jnp.complex128))


def abcd_shunt_parallel_lc(
    frequency_hz: ArrayLike,
    *,
    L_H: ArrayLike,
    C_F: ArrayLike,
    G_S: ArrayLike = 0.0,
    R_parallel_ohm: ArrayLike | None = None,
) -> jax.Array:
    """
    ABCD matrix for a shunt parallel LC resonator.

        Y = G + 1/Rp + j omega C + 1/(j omega L)

    At DC, the inductor admittance is singular. Use this only for positive
    RF frequencies.
    """
    f = _as_frequency_array(frequency_hz)
    if bool(jnp.any(f == 0.0)):
        raise ValueError("Parallel LC shunt resonator is singular at DC")
    omega = angular_frequency(f)

    L = _broadcast_to_frequency("L_H", L_H, f)
    C = _broadcast_to_frequency("C_F", C_F, f)
    G = _broadcast_to_frequency("G_S", G_S, f)

    y = G + 1j * omega * C + 1.0 / (1j * omega * L)

    if R_parallel_ohm is not None:
        Rp = _broadcast_to_frequency("R_parallel_ohm", R_parallel_ohm, f)
        if bool(jnp.any(Rp <= 0.0)):
            raise ValueError("R_parallel_ohm must be positive")
        y = y + 1.0 / Rp

    return abcd_shunt_admittance(y)


def abcd_series_lc(
    frequency_hz: ArrayLike,
    *,
    L_H: ArrayLike,
    C_F: ArrayLike,
    R_ohm: ArrayLike = 0.0,
) -> jax.Array:
    """
    ABCD matrix for a series LC branch.

        Z = R + j omega L + 1/(j omega C)

    At DC, the capacitor impedance is singular. Use positive RF frequencies.
    """
    f = _as_frequency_array(frequency_hz)
    if bool(jnp.any(f == 0.0)):
        raise ValueError("Series LC branch is singular at DC")
    omega = angular_frequency(f)

    L = _broadcast_to_frequency("L_H", L_H, f)
    C = _broadcast_to_frequency("C_F", C_F, f)
    R = _broadcast_to_frequency("R_ohm", R_ohm, f)

    z = R + 1j * omega * L + 1.0 / (1j * omega * C)
    return abcd_series_impedance(z)


# ---------------------------------------------------------------------------
# Lumped unit-cell ABCD matrices
# ---------------------------------------------------------------------------

def abcd_t_cell(
    frequency_hz: ArrayLike,
    *,
    L_series_H: ArrayLike,
    C_shunt_F: ArrayLike,
    R_series_ohm: ArrayLike = 0.0,
    G_shunt_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    ABCD matrix for a symmetric T-cell.

        series Z/2  ->  shunt Y  ->  series Z/2

    where

        Z = R + j omega L
        Y = G + j omega C

    This is the standard lumped transmission-line cell used by the early
    KI-TWPA simulator.
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    L = _broadcast_to_frequency("L_series_H", L_series_H, f)
    C = _broadcast_to_frequency("C_shunt_F", C_shunt_F, f)
    R = _broadcast_to_frequency("R_series_ohm", R_series_ohm, f)
    G = _broadcast_to_frequency("G_shunt_S", G_shunt_S, f)

    z = R + 1j * omega * L
    y = G + 1j * omega * C

    left = abcd_series_impedance(0.5 * z)
    shunt = abcd_shunt_admittance(y)
    right = abcd_series_impedance(0.5 * z)

    return cascade_abcd(left, shunt, right)


def abcd_pi_cell(
    frequency_hz: ArrayLike,
    *,
    L_series_H: ArrayLike,
    C_shunt_F: ArrayLike,
    R_series_ohm: ArrayLike = 0.0,
    G_shunt_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    ABCD matrix for a symmetric pi-cell.

        shunt Y/2  ->  series Z  ->  shunt Y/2

    This is an alternative discretization to the T-cell.
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    L = _broadcast_to_frequency("L_series_H", L_series_H, f)
    C = _broadcast_to_frequency("C_shunt_F", C_shunt_F, f)
    R = _broadcast_to_frequency("R_series_ohm", R_series_ohm, f)
    G = _broadcast_to_frequency("G_shunt_S", G_shunt_S, f)

    z = R + 1j * omega * L
    y = G + 1j * omega * C

    left = abcd_shunt_admittance(0.5 * y)
    series = abcd_series_impedance(z)
    right = abcd_shunt_admittance(0.5 * y)

    return cascade_abcd(left, series, right)


def abcd_loaded_t_cell(
    frequency_hz: ArrayLike,
    *,
    L_series_H: ArrayLike,
    C_shunt_F: ArrayLike,
    C_stub_F: ArrayLike = 0.0,
    R_series_ohm: ArrayLike = 0.0,
    G_shunt_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    ABCD matrix for a T-cell with extra shunt stub capacitance.

        C_total = C_shunt + C_stub

    This is the first-order model for capacitive-stub-loaded KI-TWPA cells.
    """
    f = _as_frequency_array(frequency_hz)
    C = _broadcast_to_frequency("C_shunt_F", C_shunt_F, f)
    C_stub = _broadcast_to_frequency("C_stub_F", C_stub_F, f)
    return abcd_t_cell(
        f,
        L_series_H=L_series_H,
        C_shunt_F=C + C_stub,
        R_series_ohm=R_series_ohm,
        G_shunt_S=G_shunt_S,
    )


# ---------------------------------------------------------------------------
# Cascading and matrix powers
# ---------------------------------------------------------------------------

def cascade_two_abcd(a: ArrayLike, b: ArrayLike) -> jax.Array:
    """
    Cascade two ABCD matrices.

        M_total = A @ B

    Supports broadcasting over leading dimensions.
    """
    aa = _check_abcd_shape("a", a)
    bb = _check_abcd_shape("b", b)
    return jnp.matmul(aa, bb)


def cascade_abcd(*matrices: ArrayLike) -> jax.Array:
    """
    Cascade multiple ABCD matrices in order.

    Example:
        cascade_abcd(series_left, shunt, series_right)
    """
    if len(matrices) == 0:
        raise ValueError("At least one matrix is required")
    out = _check_abcd_shape("matrices[0]", matrices[0])
    for i, matrix in enumerate(matrices[1:], start=1):
        out = cascade_two_abcd(out, _check_abcd_shape(f"matrices[{i}]", matrix))
    return out


def abcd_power(matrix: ArrayLike, n: int) -> jax.Array:
    """
    Raise an ABCD matrix batch to an integer power using binary exponentiation.

    Parameters
    ----------
    matrix:
        Shape (..., 2, 2).
    n:
        Non-negative integer.

    Returns
    -------
    Shape (..., 2, 2).
    """
    m = _check_abcd_shape("matrix", matrix)
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return _identity_like_batch(m.shape[:-2], dtype=m.dtype)

    result = _identity_like_batch(m.shape[:-2], dtype=m.dtype)
    base = m
    exponent = n

    while exponent > 0:
        if exponent % 2 == 1:
            result = jnp.matmul(result, base)
        base = jnp.matmul(base, base)
        exponent //= 2

    return result


def cascade_scan_abcd(matrices: ArrayLike) -> jax.Array:
    """
    Cascade a sequence of ABCD matrices with shape (N, F, 2, 2).

    Returns final cascaded matrix with shape (F, 2, 2).

    This is a simple scan over cells. Long-line optimized versions live in
    twpa.linear.cascade.
    """
    m = _check_abcd_shape("matrices", matrices)
    if m.ndim != 4:
        raise ValueError(f"matrices must have shape (N, F, 2, 2), got {m.shape}")

    f_count = m.shape[1]
    init = _identity_like_batch((f_count,), dtype=m.dtype)

    def step(carry: jax.Array, cell: jax.Array) -> tuple[jax.Array, None]:
        return jnp.matmul(carry, cell), None

    final, _ = jax.lax.scan(step, init, m)
    return final


# ---------------------------------------------------------------------------
# ABCD <-> S-parameters
# ---------------------------------------------------------------------------

def abcd_to_s(
    abcd: ArrayLike,
    *,
    z0_ohm: ArrayLike = 50.0,
) -> jax.Array:
    """
    Convert ABCD matrix/matrices to S-parameters.

    Assumes equal real reference impedance at both ports.
    """
    m = _check_abcd_shape("abcd", abcd)
    z0 = jnp.asarray(z0_ohm, dtype=jnp.float64)
    if bool(jnp.any(z0 <= 0.0)):
        raise ValueError("z0_ohm must be positive")

    A = m[..., 0, 0]
    B = m[..., 0, 1]
    C = m[..., 1, 0]
    D = m[..., 1, 1]

    denom = A + B / z0 + C * z0 + D
    det = A * D - B * C

    S11 = (A + B / z0 - C * z0 - D) / denom
    S21 = 2.0 / denom
    S12 = 2.0 * det / denom
    S22 = (-A + B / z0 - C * z0 + D) / denom

    return jnp.stack(
        [
            jnp.stack([S11, S12], axis=-1),
            jnp.stack([S21, S22], axis=-1),
        ],
        axis=-2,
    )


def s_to_abcd(
    s: ArrayLike,
    *,
    z0_ohm: ArrayLike = 50.0,
) -> jax.Array:
    """
    Convert S-parameters to ABCD matrices for equal real Z0.

    Requires S21 != 0.
    """
    sm = _check_s_shape("s", s)
    z0 = jnp.asarray(z0_ohm, dtype=jnp.float64)
    if bool(jnp.any(z0 <= 0.0)):
        raise ValueError("z0_ohm must be positive")

    S11 = sm[..., 0, 0]
    S12 = sm[..., 0, 1]
    S21 = sm[..., 1, 0]
    S22 = sm[..., 1, 1]

    if bool(jnp.any(jnp.abs(S21) == 0.0)):
        raise ValueError("S21 contains zero; S-to-ABCD conversion is singular")

    A = ((1.0 + S11) * (1.0 - S22) + S12 * S21) / (2.0 * S21)
    B = z0 * ((1.0 + S11) * (1.0 + S22) - S12 * S21) / (2.0 * S21)
    C = ((1.0 - S11) * (1.0 - S22) - S12 * S21) / (2.0 * z0 * S21)
    D = ((1.0 - S11) * (1.0 + S22) + S12 * S21) / (2.0 * S21)

    return jnp.stack(
        [
            jnp.stack([A, B], axis=-1),
            jnp.stack([C, D], axis=-1),
        ],
        axis=-2,
    )


def s_to_db(s: ArrayLike, *, floor: float = 1e-300) -> jax.Array:
    """
    Convert complex S-parameters to magnitude in dB.
    """
    return sparam_gain_db(s, floor=floor)


def s21(s: ArrayLike) -> jax.Array:
    """Extract S21 from S-matrix batch."""
    sm = _check_s_shape("s", s)
    return sm[..., 1, 0]


def s11(s: ArrayLike) -> jax.Array:
    """Extract S11 from S-matrix batch."""
    sm = _check_s_shape("s", s)
    return sm[..., 0, 0]


def s12(s: ArrayLike) -> jax.Array:
    """Extract S12 from S-matrix batch."""
    sm = _check_s_shape("s", s)
    return sm[..., 0, 1]


def s22(s: ArrayLike) -> jax.Array:
    """Extract S22 from S-matrix batch."""
    sm = _check_s_shape("s", s)
    return sm[..., 1, 1]


# ---------------------------------------------------------------------------
# Impedance/admittance derived quantities
# ---------------------------------------------------------------------------

def abcd_input_impedance(
    abcd: ArrayLike,
    *,
    load_impedance_ohm: ArrayLike = 50.0,
) -> jax.Array:
    """
    Input impedance looking into ABCD network terminated by ZL.

        Zin = (A ZL + B) / (C ZL + D)
    """
    m = _check_abcd_shape("abcd", abcd)
    zl = jnp.asarray(load_impedance_ohm, dtype=jnp.complex128)

    A = m[..., 0, 0]
    B = m[..., 0, 1]
    C = m[..., 1, 0]
    D = m[..., 1, 1]

    return (A * zl + B) / (C * zl + D)


def abcd_output_impedance(
    abcd: ArrayLike,
    *,
    source_impedance_ohm: ArrayLike = 50.0,
) -> jax.Array:
    """
    Output impedance looking into port 2 with source impedance ZS at port 1.

        Zout = (D ZS + B) / (C ZS + A)
    """
    m = _check_abcd_shape("abcd", abcd)
    zs = jnp.asarray(source_impedance_ohm, dtype=jnp.complex128)

    A = m[..., 0, 0]
    B = m[..., 0, 1]
    C = m[..., 1, 0]
    D = m[..., 1, 1]

    return (D * zs + B) / (C * zs + A)


def reflection_coefficient_from_impedance(
    z_ohm: ArrayLike,
    *,
    z0_ohm: ArrayLike = 50.0,
) -> jax.Array:
    """
    Reflection coefficient Γ = (Z - Z0)/(Z + Z0).
    """
    z = jnp.asarray(z_ohm, dtype=jnp.complex128)
    z0 = jnp.asarray(z0_ohm, dtype=jnp.float64)
    return (z - z0) / (z + z0)


def impedance_from_reflection_coefficient(
    gamma: ArrayLike,
    *,
    z0_ohm: ArrayLike = 50.0,
) -> jax.Array:
    """
    Inverse of reflection_coefficient_from_impedance.
    """
    g = jnp.asarray(gamma, dtype=jnp.complex128)
    z0 = jnp.asarray(z0_ohm, dtype=jnp.float64)
    return z0 * (1.0 + g) / (1.0 - g)


# ---------------------------------------------------------------------------
# Transmission-line analytic comparator
# ---------------------------------------------------------------------------

def ideal_lossless_line_abcd(
    frequency_hz: ArrayLike,
    *,
    length_m: float,
    L_per_m_H: float,
    C_per_m_F: float,
) -> jax.Array:
    """
    ABCD matrix of an ideal lossless transmission line.

        gamma = j beta
        beta = omega sqrt(L C)
        Zc = sqrt(L/C)

        A = D = cos(beta l)
        B = j Zc sin(beta l)
        C = j sin(beta l) / Zc

    This is a comparator for discrete LC ladders away from cutoff.
    """
    if length_m < 0.0:
        raise ValueError("length_m must be non-negative")
    if L_per_m_H <= 0.0:
        raise ValueError("L_per_m_H must be positive")
    if C_per_m_F <= 0.0:
        raise ValueError("C_per_m_F must be positive")

    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)
    beta = omega * jnp.sqrt(L_per_m_H * C_per_m_F)
    zc = jnp.sqrt(L_per_m_H / C_per_m_F)
    theta = beta * length_m

    A = jnp.cos(theta)
    D = A
    B = 1j * zc * jnp.sin(theta)
    C = 1j * jnp.sin(theta) / zc

    return jnp.stack(
        [
            jnp.stack([A, B], axis=-1),
            jnp.stack([C, D], axis=-1),
        ],
        axis=-2,
    )


def lossy_line_abcd_rlgc(
    frequency_hz: ArrayLike,
    *,
    length_m: float,
    R_per_m_ohm: float,
    L_per_m_H: float,
    G_per_m_S: float,
    C_per_m_F: float,
) -> jax.Array:
    """
    ABCD matrix of a uniform lossy RLGC transmission line.

        gamma = sqrt((R + jωL)(G + jωC))
        Zc = sqrt((R + jωL)/(G + jωC))

        A = D = cosh(gamma l)
        B = Zc sinh(gamma l)
        C = sinh(gamma l)/Zc
    """
    if length_m < 0.0:
        raise ValueError("length_m must be non-negative")
    if L_per_m_H <= 0.0:
        raise ValueError("L_per_m_H must be positive")
    if C_per_m_F <= 0.0:
        raise ValueError("C_per_m_F must be positive")
    if R_per_m_ohm < 0.0:
        raise ValueError("R_per_m_ohm must be non-negative")
    if G_per_m_S < 0.0:
        raise ValueError("G_per_m_S must be non-negative")

    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    z_per_m = R_per_m_ohm + 1j * omega * L_per_m_H
    y_per_m = G_per_m_S + 1j * omega * C_per_m_F

    gamma = jnp.sqrt(z_per_m * y_per_m)
    zc = jnp.sqrt(z_per_m / y_per_m)

    gl = gamma * length_m

    A = jnp.cosh(gl)
    D = A
    B = zc * jnp.sinh(gl)
    C = jnp.sinh(gl) / zc

    return jnp.stack(
        [
            jnp.stack([A, B], axis=-1),
            jnp.stack([C, D], axis=-1),
        ],
        axis=-2,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkDiagnostics:
    """
    Summary diagnostics for a two-port network.
    """

    det_abcd_max_abs_error: float
    reciprocity_s_max_abs_error: float
    passivity_max_singular_value: float
    passivity_violation: float
    s21_db_min: float
    s21_db_max: float
    metadata: Mapping[str, Any] | None = None

    @property
    def reciprocal_passed(self) -> bool:
        return self.reciprocity_s_max_abs_error < 1e-9

    @property
    def passive_passed(self) -> bool:
        return self.passivity_violation <= 1e-9

    def to_dict(self) -> dict[str, Any]:
        return {
            "det_abcd_max_abs_error": self.det_abcd_max_abs_error,
            "reciprocity_s_max_abs_error": self.reciprocity_s_max_abs_error,
            "passivity_max_singular_value": self.passivity_max_singular_value,
            "passivity_violation": self.passivity_violation,
            "s21_db_min": self.s21_db_min,
            "s21_db_max": self.s21_db_max,
            "reciprocal_passed": self.reciprocal_passed,
            "passive_passed": self.passive_passed,
            "metadata": dict(self.metadata or {}),
        }


def abcd_determinant(abcd: ArrayLike) -> jax.Array:
    """Return determinant A D - B C of ABCD matrices."""
    m = _check_abcd_shape("abcd", abcd)
    return m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]


def reciprocal_error_s(s: ArrayLike) -> jax.Array:
    """
    Return |S21 - S12| for each matrix.
    """
    sm = _check_s_shape("s", s)
    return jnp.abs(sm[..., 1, 0] - sm[..., 0, 1])


def passivity_singular_values(s: ArrayLike) -> jax.Array:
    """
    Singular values of S-matrix batch.

    For a passive network, the largest singular value should be <= 1.
    """
    sm = _check_s_shape("s", s)
    return jnp.linalg.svd(sm, compute_uv=False)


def diagnose_two_port(
    abcd: ArrayLike,
    *,
    z0_ohm: float = 50.0,
    metadata: Mapping[str, Any] | None = None,
) -> NetworkDiagnostics:
    """
    Compute basic two-port diagnostics.
    """
    m = _check_abcd_shape("abcd", abcd)
    sm = abcd_to_s(m, z0_ohm=z0_ohm)

    det_err = jnp.max(jnp.abs(abcd_determinant(m) - 1.0))
    rec_err = jnp.max(reciprocal_error_s(sm))
    sv = passivity_singular_values(sm)
    max_sv = jnp.max(sv)
    violation = jnp.maximum(max_sv - 1.0, 0.0)
    s21_db = s_to_db(s21(sm))

    return NetworkDiagnostics(
        det_abcd_max_abs_error=float(det_err),
        reciprocity_s_max_abs_error=float(rec_err),
        passivity_max_singular_value=float(max_sv),
        passivity_violation=float(violation),
        s21_db_min=float(jnp.min(s21_db)),
        s21_db_max=float(jnp.max(s21_db)),
        metadata=dict(metadata or {}),
    )


def compare_sparameters(
    s_a: ArrayLike,
    s_b: ArrayLike,
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """
    Compare two S-parameter arrays.
    """
    a = _check_s_shape("s_a", s_a)
    b = _check_s_shape("s_b", s_b)
    if a.shape != b.shape:
        raise ValueError(f"S-parameter arrays must have same shape, got {a.shape} and {b.shape}")

    diff = a - b
    return {
        "label_a": label_a,
        "label_b": label_b,
        "shape": tuple(int(v) for v in a.shape),
        "max_abs_diff": float(jnp.max(jnp.abs(diff))),
        "rms_abs_diff": float(jnp.sqrt(jnp.mean(jnp.abs(diff) ** 2))),
        "s21_max_abs_diff": float(jnp.max(jnp.abs(s21(a) - s21(b)))),
        "s21_db_max_abs_diff": float(
            jnp.max(jnp.abs(s_to_db(s21(a)) - s_to_db(s21(b))))
        ),
    }


def unwrap_phase(x: ArrayLike) -> jax.Array:
    """
    JAX-compatible 1D phase unwrap.

    jnp.unwrap exists in recent JAX versions, but this helper keeps the call
    localized and easy to replace if needed.
    """
    return jnp.unwrap(jnp.angle(jnp.asarray(x)))


def group_delay_from_s21(
    frequency_hz: ArrayLike,
    s21_values: ArrayLike,
) -> jax.Array:
    """
    Estimate group delay from S21 phase.

        tau_g = - d phi / d omega
    """
    f = _as_frequency_array(frequency_hz)
    y = jnp.asarray(s21_values)
    if y.shape[0] != f.shape[0]:
        raise ValueError("s21_values first dimension must match frequency length")

    phase = unwrap_phase(y)
    omega = angular_frequency(f)
    dphi = jnp.gradient(phase, omega)
    return -dphi


def effective_beta_from_s21(
    frequency_hz: ArrayLike,
    s21_values: ArrayLike,
    *,
    length_m: float,
    sign: Literal["positive", "negative"] = "positive",
) -> jax.Array:
    """
    Estimate effective propagation constant beta from unwrapped S21 phase.

    For a through line, approximately:

        S21 ~ exp(-j beta l)

    so

        beta ~ -unwrap(angle(S21)) / l

    Depending on convention, use sign="negative" to flip.
    """
    if length_m <= 0.0:
        raise ValueError("length_m must be positive")
    _ = _as_frequency_array(frequency_hz)
    phase = unwrap_phase(s21_values)
    beta = -phase / length_m
    if sign == "positive":
        return beta
    if sign == "negative":
        return -beta
    raise ValueError(f"Unsupported sign {sign!r}")


__all__ = [
    "abcd_identity",
    "abcd_series_impedance",
    "abcd_shunt_admittance",
    "abcd_series_resistor",
    "abcd_series_inductor",
    "abcd_shunt_capacitor",
    "abcd_shunt_resistor",
    "abcd_shunt_parallel_lc",
    "abcd_series_lc",
    "abcd_t_cell",
    "abcd_pi_cell",
    "abcd_loaded_t_cell",
    "cascade_two_abcd",
    "cascade_abcd",
    "abcd_power",
    "cascade_scan_abcd",
    "abcd_to_s",
    "s_to_abcd",
    "s_to_db",
    "s21",
    "s11",
    "s12",
    "s22",
    "abcd_input_impedance",
    "abcd_output_impedance",
    "reflection_coefficient_from_impedance",
    "impedance_from_reflection_coefficient",
    "ideal_lossless_line_abcd",
    "lossy_line_abcd_rlgc",
    "NetworkDiagnostics",
    "abcd_determinant",
    "reciprocal_error_s",
    "passivity_singular_values",
    "diagnose_two_port",
    "compare_sparameters",
    "unwrap_phase",
    "group_delay_from_s21",
    "effective_beta_from_s21",
]


# Compatibility surface used by foundation tests. These wrappers keep scalar
# two-port helpers scalar while preserving batched behavior for array inputs.
def _maybe_squeeze_twoport(x: jax.Array) -> jax.Array:
    return x[0] if x.ndim == 3 and x.shape[0] == 1 else x


def abcd_identity(
    frequency_hz: ArrayLike | None = None,
    *,
    n: int | None = None,
    dtype: Any = jnp.complex128,
) -> jax.Array:
    if frequency_hz is None and n is None:
        return jnp.eye(2, dtype=dtype)
    if frequency_hz is not None:
        f = _as_frequency_array(frequency_hz)
        return _identity_like_batch((f.shape[0],), dtype=dtype)
    if n is None or int(n) <= 0:
        raise ValueError("n must be positive")
    return _identity_like_batch((int(n),), dtype=dtype)


def series_impedance_abcd(Z: ArrayLike) -> jax.Array:
    return _maybe_squeeze_twoport(abcd_series_impedance(Z))


def shunt_admittance_abcd(Y: ArrayLike) -> jax.Array:
    return _maybe_squeeze_twoport(abcd_shunt_admittance(Y))


def cascade_abcd(*matrices: ArrayLike) -> jax.Array:
    if len(matrices) == 0:
        raise ValueError("At least one matrix is required")
    out = _check_abcd_shape("matrices[0]", matrices[0])
    for i, matrix in enumerate(matrices[1:], start=1):
        out = jnp.matmul(out, _check_abcd_shape(f"matrices[{i}]", matrix))
    return out


def abcd_to_s(abcd: ArrayLike, *, z0_ohm: ArrayLike = 50.0) -> jax.Array:
    m = _check_abcd_shape("abcd", abcd)
    z0 = jnp.asarray(z0_ohm, dtype=jnp.float64)
    if bool(jnp.any(z0 <= 0.0)):
        raise ValueError("z0_ohm must be positive")
    A = m[..., 0, 0]
    B = m[..., 0, 1]
    C = m[..., 1, 0]
    D = m[..., 1, 1]
    denom = A + B / z0 + C * z0 + D
    det = A * D - B * C
    S11 = (A + B / z0 - C * z0 - D) / denom
    S21 = 2.0 / denom
    S12 = 2.0 * det / denom
    S22 = (-A + B / z0 - C * z0 + D) / denom
    return jnp.stack(
        [jnp.stack([S11, S12], axis=-1), jnp.stack([S21, S22], axis=-1)],
        axis=-2,
    )


def s_to_abcd(s: ArrayLike, *, z0_ohm: ArrayLike = 50.0) -> jax.Array:
    sm = _check_s_shape("s", s)
    z0 = jnp.asarray(z0_ohm, dtype=jnp.float64)
    if bool(jnp.any(z0 <= 0.0)):
        raise ValueError("z0_ohm must be positive")
    S11 = sm[..., 0, 0]
    S12 = sm[..., 0, 1]
    S21 = sm[..., 1, 0]
    S22 = sm[..., 1, 1]
    if bool(jnp.any(jnp.abs(S21) == 0.0)):
        raise ValueError("S21 contains zero; S-to-ABCD conversion is singular")
    A = ((1.0 + S11) * (1.0 - S22) + S12 * S21) / (2.0 * S21)
    B = z0 * ((1.0 + S11) * (1.0 + S22) - S12 * S21) / (2.0 * S21)
    C = ((1.0 - S11) * (1.0 - S22) - S12 * S21) / (2.0 * z0 * S21)
    D = ((1.0 - S11) * (1.0 + S22) + S12 * S21) / (2.0 * S21)
    return jnp.stack(
        [jnp.stack([A, B], axis=-1), jnp.stack([C, D], axis=-1)],
        axis=-2,
    )


def transmission_line_abcd(
    gamma: ArrayLike,
    length_m: float,
    z0_ohm: float,
) -> jax.Array:
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    g = jnp.asarray(gamma, dtype=jnp.complex128)
    gl = g * float(length_m)
    A = jnp.cosh(gl)
    B = z0_ohm * jnp.sinh(gl)
    C = jnp.sinh(gl) / z0_ohm
    D = A
    out = jnp.stack(
        [jnp.stack([A, B], axis=-1), jnp.stack([C, D], axis=-1)],
        axis=-2,
    )
    return out
