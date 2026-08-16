#!/usr/bin/env python3
"""Finite-difference time-domain solver for the 990-cell rf-SQUID JTWPA model
of Guarcello et al., arXiv:2406.01185v2 (2024), Appendix A.

The implementation follows Eqs. (A14)-(A60): centered finite differences in
physical time and an implicit tridiagonal solve for the Josephson phases at every
step.  The literal centered input/load-current recurrences are available through
--port-update paper-centered.  The default --port-update stable integrates the
same boundary ODEs with an integrating-factor step to suppress the parasitic
leapfrog mode of a centered discretization of a dissipative first-order ODE.

Default device parameters reproduce the paper's stated circuit:
    N=990, Ic=2 uA, Cj=200 fF, Rj=20 kohm,
    Lg=120 pH, Cg=24 fF, Ri=Rl=50 ohm, Ci=24 fF, Cl=1 nF.

The paper uses normalized dt=0.01 and tmax=20000 (normalization to the JJ
plasma angular frequency), i.e. 2,000,000 time steps.  This code converts those
values to physical seconds and executes the same finite-difference equations.

Power convention
----------------
The paper explicitly states Psign=-100 dBm corresponds to Vsign=0.032 mV.
That is not the standard 50-ohm peak-voltage conversion.  To make the ambiguity explicit, --power-convention paper scales all source
voltage amplitudes from that stated reference.  The default is --power-convention
50ohm, using Vpk=sqrt(2 Z P), because the paper's stated -100 dBm -> 0.032 mV
conversion is 20 dB larger in power than the standard 50-ohm convention and a
literal use strongly overdrives the translated model.

Examples
--------
Quick smoke test:
    python guarcello_jtwpa_fdtd.py point --quick

Paper-scale single point (2,000,000 steps):
    python guarcello_jtwpa_fdtd.py point --pump-dbm -55 --signal-ghz 6.42

Pump scan analogous to Fig. 2(a):
    python guarcello_jtwpa_fdtd.py sweep-pump --start -70 --stop -45 --num 101 --workers 8

Signal-frequency scan analogous to Fig. 2(b):
    python guarcello_jtwpa_fdtd.py sweep-signal --start 4 --stop 10 --num 121 --pump-dbm -55 --workers 8

Bias-current scan analogous to Fig. 2(c):
    python guarcello_jtwpa_fdtd.py sweep-bias --start 0 --stop 34 --num 137 --pump-dbm -55 --workers 8

Pump/bias map analogous to Fig. 3:
    python guarcello_jtwpa_fdtd.py map-pump-bias --pump-num 81 --bias-num 61 --workers 8

Notes
-----
* The paper does not state the exact number of sweep points used in its figures,
  nor the precise transient-discard fraction used for FFT/Poincare processing.
  Those are exposed as CLI options rather than silently guessed.
* The optional non-sinusoidal CPR uses Eq. (6) of the paper.  Set --tau to a
  value in (0,1]; omit it for the sinusoidal CPR used in Figs. 2-4.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numba import njit

HBAR = 1.054_571_817e-34
E_CHARGE = 1.602_176_634e-19
PHI0_REDUCED = HBAR / (2.0 * E_CHARGE)  # Phi0/(2*pi)


@dataclass(frozen=True)
class Device:
    n_cells: int = 990
    ic_a: float = 2.0e-6
    cj_f: float = 200.0e-15
    rj_ohm: float = 20.0e3
    lg_h: float = 120.0e-12
    cg_f: float = 24.0e-15
    ri_ohm: float = 50.0
    ci_f: float = 24.0e-15
    rl_ohm: float = 50.0
    cl_f: float = 1.0e-9

    @property
    def omega_plasma(self) -> float:
        return math.sqrt(self.ic_a / (PHI0_REDUCED * self.cj_f))

    @property
    def f_plasma_hz(self) -> float:
        return self.omega_plasma / (2.0 * math.pi)


@dataclass(frozen=True)
class RunConfig:
    dt_norm: float = 0.01
    tmax_norm: float = 20_000.0
    transient_fraction: float = 0.5
    record_stride: int = 20
    pump_ghz: float = 7.0
    signal_ghz: float = 6.42
    pump_dbm: float = -55.0
    signal_dbm: float = -100.0
    bias_ua: float = 0.0
    tau: float = -1.0  # <=0 means sinusoidal CPR
    power_convention: str = "50ohm"
    port_update: str = "stable"  # stable | paper-centered


def dbm_to_vpk(dbm: float, convention: str = "paper", z_ohm: float = 50.0) -> float:
    """Convert dBm to source peak voltage.

    'paper' anchors P=-100 dBm to the paper's explicit V=0.032 mV statement.
    '50ohm' uses the standard sinusoidal peak-voltage relation.
    """
    if convention == "paper":
        return 0.032e-3 * 10.0 ** ((dbm + 100.0) / 20.0)
    if convention == "50ohm":
        p_w = 1.0e-3 * 10.0 ** (dbm / 10.0)
        return math.sqrt(2.0 * z_ohm * p_w)
    raise ValueError(f"unknown power convention: {convention}")


def _make_cap_array(device: Device) -> np.ndarray:
    # C_0 = C_i in Appendix A, and C_1...C_N are the cell shunt capacitances.
    c = np.empty(device.n_cells + 1, dtype=np.float64)
    c[0] = device.ci_f
    c[1:] = device.cg_f
    return c


def build_coefficients(device: Device, dt_s: float) -> dict[str, np.ndarray | float]:
    """Build the time-independent coefficients of Eqs. (A19)-(A26), (A43), (A56)."""
    n = device.n_cells
    c = _make_cap_array(device)
    cminus = c[1:] / c[:-1]  # C_n^- = C_n / C_{n-1}, n=1..N

    cjt = device.cj_f * PHI0_REDUCED
    inv_rjt = (1.0 / device.rj_ohm) * PHI0_REDUCED
    inv_lgt = (1.0 / device.lg_h) * PHI0_REDUCED

    alpha_p = np.full(n, cjt / dt_s**2 + inv_rjt / (2.0 * dt_s), dtype=np.float64)
    alpha_m = np.full(n, cjt / dt_s**2 - inv_rjt / (2.0 * dt_s), dtype=np.float64)

    # Eq. (A23): C~"_n = phi0 * [Cj + C_{n-1}C_n/(C_{n-1}+C_n)]
    ceff = PHI0_REDUCED * (
        device.cj_f + (c[:-1] * c[1:]) / (c[:-1] + c[1:])
    )
    at_p = ceff / dt_s**2 + inv_rjt / (2.0 * dt_s)
    at_m = ceff / dt_s**2 - inv_rjt / (2.0 * dt_s)

    fdiag = inv_lgt - 2.0 * cjt / dt_s**2
    ft_diag = inv_lgt - 2.0 * ceff / dt_s**2

    # Tridiagonal A of Eq. (A57).  0-based row i corresponds to n=i+1.
    lower = np.zeros(n - 1, dtype=np.float64)
    diag = (1.0 + cminus) * at_p
    upper = np.zeros(n - 1, dtype=np.float64)
    for i in range(1, n):
        lower[i - 1] = -cminus[i] * alpha_p[i - 1]
    for i in range(0, n - 1):
        upper[i] = -alpha_p[i + 1]

    # A43/A56 simply remove the missing outside coefficients; the arrays above
    # already have no element outside the [1,N] chain.

    # Pre-factor the constant tridiagonal matrix once (Thomas LU).
    lu_diag = diag.copy()
    lu_mult = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        mult = lower[i - 1] / lu_diag[i - 1]
        lu_mult[i - 1] = mult
        lu_diag[i] -= mult * upper[i - 1]

    return {
        "cminus": cminus,
        "alpha_p": alpha_p,
        "alpha_m": alpha_m,
        "at_p": at_p,
        "at_m": at_m,
        "fdiag": float(fdiag),
        "ft_diag": ft_diag,
        "lower": lower,
        "diag": diag,
        "upper": upper,
        "lu_diag": lu_diag,
        "lu_mult": lu_mult,
    }


@njit(cache=True, fastmath=True, nogil=True)
def _cpr_value(phi: float, ic: float, tau: float) -> float:
    if tau <= 0.0:
        return ic * math.sin(phi)
    # Eq. (6): normalized short-junction CPR.  The tau->0 limit is sin(phi).
    s = math.sin(0.5 * phi)
    den = 2.0 * math.sqrt(max(1.0 - tau * s * s, 1.0e-15))
    norm = 1.0 - math.sqrt(max(1.0 - tau, 0.0))
    return ic * math.sin(phi) * norm / den


@njit(cache=True, fastmath=True, nogil=True)
def _integrate_numba(
    n_steps: int,
    record_stride: int,
    dt_s: float,
    pump_w: float,
    signal_w: float,
    pump_vpk: float,
    signal_vpk: float,
    bias_a: float,
    ic_a: float,
    ci_f: float,
    rl_ohm: float,
    cl_f: float,
    cg_f: float,
    ri_ohm: float,
    lg_h: float,
    tau: float,
    stable_ports: int,
    cminus: np.ndarray,
    alpha_p: np.ndarray,
    alpha_m: np.ndarray,
    at_p: np.ndarray,
    at_m: np.ndarray,
    fdiag: float,
    ft_diag: np.ndarray,
    upper: np.ndarray,
    lu_diag: np.ndarray,
    lu_mult: np.ndarray,
):
    """Time-march Eqs. (A57), (A59), (A60)."""
    n = cminus.shape[0]
    phi_prev = np.zeros(n, dtype=np.float64)
    phi_cur = np.zeros(n, dtype=np.float64)
    phi_next = np.zeros(n, dtype=np.float64)
    f = np.zeros(n, dtype=np.float64)
    ft = np.zeros(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    ii_prev = 0.0
    ii_cur = 0.0
    il_prev = 0.0
    il_cur = 0.0

    omega_i = 1.0 / (ri_ohm * ci_f)
    out_count = n_steps // record_stride + 1
    t_out = np.empty(out_count, dtype=np.float64)
    v_out = np.empty(out_count, dtype=np.float64)
    phi_last_out = np.empty(out_count, dtype=np.float64)
    beta_out = np.empty(out_count, dtype=np.float64)
    gamma_out = np.empty(out_count, dtype=np.float64)
    rec = 0
    t_out[rec] = 0.0
    v_out[rec] = 0.0
    phi_last_out[rec] = 0.0
    beta_l = lg_h * ic_a / PHI0_REDUCED
    beta_out[rec] = 0.0
    gamma_out[rec] = beta_l / 6.0
    rec += 1

    for m in range(1, n_steps + 1):
        t = m * dt_s

        # Current source functions f_n^m and f~_n^m, Eqs. (A20)/(A22).
        for i in range(n):
            ij = _cpr_value(phi_cur[i], ic_a, tau)
            f[i] = fdiag * phi_cur[i] + ij
            ft[i] = ft_diag[i] * phi_cur[i] + ij

        # RHS A_n of Eq. (A58), with A42/A55 at the two boundaries.
        # n=1 left boundary
        if n == 1:
            rhs[0] = cminus[0] * ii_cur - (1.0 + cminus[0]) * ft[0]
            rhs[0] += -(1.0 + cminus[0]) * at_m[0] * phi_prev[0]
            rhs[0] += cminus[0] * bias_a
            # right boundary also applies; for N=1 use I_l contribution.
            rhs[0] += il_cur + bias_a
        else:
            rhs[0] = cminus[0] * ii_cur - (1.0 + cminus[0]) * ft[0] + f[1]
            rhs[0] += -(1.0 + cminus[0]) * at_m[0] * phi_prev[0]
            rhs[0] += alpha_m[1] * phi_prev[1] + cminus[0] * bias_a

            # interior rows n=2..N-1
            for i in range(1, n - 1):
                cm = cminus[i]
                rhs[i] = cm * f[i - 1] - (1.0 + cm) * ft[i] + f[i + 1]
                rhs[i] += cm * alpha_m[i - 1] * phi_prev[i - 1]
                rhs[i] += -(1.0 + cm) * at_m[i] * phi_prev[i]
                rhs[i] += alpha_m[i + 1] * phi_prev[i + 1]

            # n=N right boundary
            i = n - 1
            cm = cminus[i]
            rhs[i] = cm * f[i - 1] - (1.0 + cm) * ft[i] + il_cur
            rhs[i] += cm * alpha_m[i - 1] * phi_prev[i - 1]
            rhs[i] += -(1.0 + cm) * at_m[i] * phi_prev[i] + bias_a

        # Solve the prefactored tridiagonal system A phi^{m+1}=rhs.
        y[0] = rhs[0]
        for i in range(1, n):
            y[i] = rhs[i] - lu_mult[i - 1] * y[i - 1]
        phi_next[n - 1] = y[n - 1] / lu_diag[n - 1]
        for i in range(n - 2, -1, -1):
            phi_next[i] = (y[i] - upper[i] * phi_next[i + 1]) / lu_diag[i]

        # Analytic derivative of Vi = Vp sin(wp t)+Vs sin(ws t).
        vdot_i = pump_vpk * pump_w * math.cos(pump_w * t)
        vdot_i += signal_vpk * signal_w * math.cos(signal_w * t)

        # Branch currents I_1^m and I_N^m from Eq. (A18), now that phi^{m+1} is known.
        i1_m = alpha_p[0] * phi_next[0] + f[0] + alpha_m[0] * phi_prev[0]
        i = n - 1
        in_m = alpha_p[i] * phi_next[i] + f[i] + alpha_m[i] * phi_prev[i]

        if stable_ports == 0:
            # Literal centered first-derivative recurrences, Eqs. (A59)/(A60).
            # These are retained for Appendix-A parity.  Centered differencing of
            # a dissipative first-order ODE has a parasitic leapfrog mode, so the
            # default CLI uses the stable one-step update below.
            ii_next = ii_prev + (
                i1_m - ii_cur + ci_f * vdot_i - bias_a
            ) * (2.0 * dt_s * omega_i)
            il_next = il_prev + (
                in_m - (1.0 + cg_f / cl_f) * il_cur - bias_a
            ) * (2.0 * dt_s / (cg_f * rl_ohm))
        else:
            # Stable integrating-factor update of the same boundary ODEs (A29)/(A45),
            # freezing I_1^m / I_N^m and Vdot over one extremely small timestep.
            # This removes the parasitic centered-difference mode without changing
            # the circuit equations or the implicit tridiagonal phase solve.
            ei = math.exp(-omega_i * dt_s)
            target_i = i1_m + ci_f * vdot_i - bias_a
            ii_next = ei * ii_cur + (1.0 - ei) * target_i

            a_l = (1.0 + cg_f / cl_f) / (cg_f * rl_ohm)
            el = math.exp(-a_l * dt_s)
            target_l = (in_m - bias_a) / (1.0 + cg_f / cl_f)
            il_next = el * il_cur + (1.0 - el) * target_l

        if m % record_stride == 0:
            t_out[rec] = t
            v_out[rec] = il_next * rl_ohm
            phi_last_out[rec] = phi_next[n - 1]
            # Sec. IV: Phi_dc = Lg I_circ, I_circ = I_L - I_J = 2 I_L - I_N.
            # I_N^m is centered at time m; I_L^m = phi0/Lg * phi_N^m.
            i_ind_m = (PHI0_REDUCED / lg_h) * phi_cur[n - 1]
            i_circ_m = 2.0 * i_ind_m - in_m
            phi_dc_m = lg_h * i_circ_m / PHI0_REDUCED
            beta_out[rec] = 0.5 * beta_l * math.sin(phi_dc_m)
            gamma_out[rec] = (beta_l / 6.0) * math.cos(phi_dc_m)
            rec += 1

        # Rotate three-level time state.
        tmp = phi_prev
        phi_prev = phi_cur
        phi_cur = phi_next
        phi_next = tmp
        ii_prev, ii_cur = ii_cur, ii_next
        il_prev, il_cur = il_cur, il_next

    return t_out[:rec], v_out[:rec], phi_last_out[:rec], beta_out[:rec], gamma_out[:rec], phi_cur, ii_cur, il_cur


def simulate(device: Device, cfg: RunConfig):
    omega_p = device.omega_plasma
    dt_s = cfg.dt_norm / omega_p
    n_steps = int(round(cfg.tmax_norm / cfg.dt_norm))
    coeff = build_coefficients(device, dt_s)
    pump_vpk = dbm_to_vpk(cfg.pump_dbm, cfg.power_convention, device.ri_ohm)
    signal_vpk = dbm_to_vpk(cfg.signal_dbm, cfg.power_convention, device.ri_ohm)

    args = (
        n_steps,
        cfg.record_stride,
        dt_s,
        2.0 * math.pi * cfg.pump_ghz * 1e9,
        2.0 * math.pi * cfg.signal_ghz * 1e9,
        pump_vpk,
        signal_vpk,
        cfg.bias_ua * 1e-6,
        device.ic_a,
        device.ci_f,
        device.rl_ohm,
        device.cl_f,
        device.cg_f,
        device.ri_ohm,
        device.lg_h,
        cfg.tau,
        1 if cfg.port_update == "stable" else 0,
        coeff["cminus"], coeff["alpha_p"], coeff["alpha_m"], coeff["at_p"], coeff["at_m"],
        coeff["fdiag"], coeff["ft_diag"], coeff["upper"], coeff["lu_diag"], coeff["lu_mult"],
    )
    t0 = time.perf_counter()
    out = _integrate_numba(*args)
    runtime = time.perf_counter() - t0
    return out, runtime


def exact_tone_amplitude(t: np.ndarray, v: np.ndarray, freq_hz: float) -> float:
    """Least-squares sinusoid amplitude at an exact frequency."""
    w = 2.0 * np.pi * freq_hz
    c = np.cos(w * t)
    s = np.sin(w * t)
    # Solve 2x2 normal equations; robust to a non-integer number of periods.
    cc = float(c @ c); ss = float(s @ s); cs = float(c @ s)
    vc = float(v @ c); vs = float(v @ s)
    det = cc * ss - cs * cs
    if abs(det) < 1e-300:
        return float("nan")
    a = (vc * ss - vs * cs) / det
    b = (vs * cc - vc * cs) / det
    return math.hypot(a, b)


def _multitone_basis_frequencies(
    t: np.ndarray, signal_hz: float, pump_hz: float,
) -> tuple[list[float], list[float]]:
    """Return retained and rejected frequencies for the multi-tone fit."""
    dt = float(np.mean(np.diff(t)))
    nyquist_hz = 0.5 / dt
    resolution_hz = 1.0 / (t[-1] - t[0])
    requested = [
        signal_hz,
        *(harmonic * pump_hz for harmonic in range(1, 6)),
        pump_hz - signal_hz,
        pump_hz + signal_hz,
        2.0 * pump_hz - signal_hz,
        2.0 * pump_hz + signal_hz,
    ]
    retained: list[float] = []
    dropped: list[float] = []
    for frequency in requested:
        if frequency <= 0.0 or frequency >= nyquist_hz:
            dropped.append(frequency)
        elif any(abs(frequency - other) <= resolution_hz for other in retained):
            dropped.append(frequency)
        else:
            retained.append(frequency)
    return retained, dropped


def multitone_amplitude(
    t: np.ndarray, v: np.ndarray, signal_hz: float, pump_hz: float,
) -> float:
    """Fit DC, pump harmonics, and pump-signal intermods simultaneously."""
    retained, _ = _multitone_basis_frequencies(t, signal_hz, pump_hz)
    frequencies = [0.0, *retained]
    columns = [np.ones_like(t)]
    for frequency in frequencies[1:]:
        angle = 2.0 * np.pi * frequency * t
        columns.extend((np.cos(angle), np.sin(angle)))
    coefficients, _, _, _ = np.linalg.lstsq(np.column_stack(columns), v, rcond=None)
    signal_index = 1 + retained.index(signal_hz) * 2
    return float(math.hypot(coefficients[signal_index], coefficients[signal_index + 1]))


def multitone_fit_metadata(
    t: np.ndarray, signal_hz: float, pump_hz: float,
) -> dict[str, object]:
    """Describe the retained and rejected frequencies in the simultaneous fit."""
    retained, dropped = _multitone_basis_frequencies(t, signal_hz, pump_hz)
    return {
        "basis_frequencies_hz": [0.0, *retained],
        "dropped_basis_frequencies_hz": dropped,
        "record_resolution_hz": 1.0 / (t[-1] - t[0]),
    }


def poincare_crossings(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    if len(v) < 3:
        return np.empty(0)
    dv = np.gradient(v, t)
    idx = np.nonzero(v[:-1] * v[1:] <= 0.0)[0]
    vals = np.empty(idx.size, dtype=np.float64)
    for j, i in enumerate(idx):
        den = v[i + 1] - v[i]
        a = 0.0 if den == 0 else -v[i] / den
        vals[j] = dv[i] + a * (dv[i + 1] - dv[i])
    return vals


def poincare_crossing_branches(t: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Return both-sign and directional Poincare derivative branches."""
    values = poincare_crossings(t, v)
    return {
        "upward": values[values > 0.0],
        "downward": values[values < 0.0],
        "both": values,
    }


def spectrum_dbm(t: np.ndarray, v: np.ndarray, r_ohm: float, fmax_ghz: float = 30.0):
    if len(v) < 4:
        return np.empty(0), np.empty(0)
    dt = float(np.mean(np.diff(t)))
    vv = v - np.mean(v)
    win = np.hanning(len(vv))
    coherent_gain = float(np.mean(win))
    spec = np.fft.rfft(vv * win)
    f = np.fft.rfftfreq(len(vv), dt)
    # single-sided peak amplitude, then convert to RMS power into R.
    amp_pk = 2.0 * np.abs(spec) / (len(vv) * coherent_gain)
    if amp_pk.size:
        amp_pk[0] *= 0.5
    p_w = (amp_pk / math.sqrt(2.0)) ** 2 / r_ohm
    dbm = 10.0 * np.log10(np.maximum(p_w, 1e-300) / 1e-3)
    mask = f <= fmax_ghz * 1e9
    return f[mask] / 1e9, dbm[mask]


def wideband_gain(
    t: np.ndarray, v: np.ndarray, signal_hz: float, pump_hz: float,
    input_vpk: float, device: Device,
) -> tuple[float, dict[str, float]]:
    """Measure output power around the signal with pump lines notched."""
    frequencies_ghz, spectrum = spectrum_dbm(t, v, device.rl_ohm)
    frequencies_hz = frequencies_ghz * 1e9
    resolution_hz = 1.0 / (t[-1] - t[0])
    half_bandwidth_hz = 0.5e9
    notch_width_hz = 2.0 * resolution_hz
    in_band = np.abs(frequencies_hz - signal_hz) <= half_bandwidth_hz
    for harmonic in range(1, 6):
        in_band &= np.abs(frequencies_hz - harmonic * pump_hz) > notch_width_hz
    output_power_w = float(np.sum(1e-3 * 10.0 ** (spectrum[in_band] / 10.0)))
    input_power_w = input_vpk**2 / (2.0 * device.ri_ohm)
    gain_db = 10.0 * math.log10(max(output_power_w, 1e-300) / input_power_w)
    return gain_db, {
        "wideband_half_bandwidth_hz": half_bandwidth_hz,
        "pump_notch_half_width_hz": notch_width_hz,
        "wideband_output_power_w": output_power_w,
    }


def analyze(device: Device, cfg: RunConfig, out) -> dict:
    t, vout, phi_last, beta_t, gamma_t, phi_final, ii_final, il_final = out
    start = int(max(0, min(len(t) - 2, round(cfg.transient_fraction * len(t)))))
    ts = t[start:]
    vs = vout[start:]
    signal_hz = cfg.signal_ghz * 1e9
    pump_hz = cfg.pump_ghz * 1e9
    signal_amp = multitone_amplitude(ts, vs, signal_hz, pump_hz)
    vin = dbm_to_vpk(cfg.signal_dbm, cfg.power_convention, device.ri_ohm)
    gain_db = 20.0 * math.log10(max(signal_amp, 1e-300) / vin)
    gain_wideband_db, wideband_meta = wideband_gain(
        ts, vs, signal_hz, pump_hz, vin, device,
    )
    branches_phys = poincare_crossing_branches(ts, vs)
    # The paper uses a derivative with respect to normalized time.  Convert
    # dV/dt_phys -> dV/d(t*omega_p) and express it in mV for plotting.
    branches = {
        name: values / device.omega_plasma * 1e3
        for name, values in branches_phys.items()
    }
    ps = branches["upward"]
    beta_ss = beta_t[start:]
    gamma_ss = gamma_t[start:]
    return {
        "gain_db": gain_db,
        "gain_wideband_db": gain_wideband_db,
        **wideband_meta,
        **multitone_fit_metadata(ts, signal_hz, pump_hz),
        "signal_vout_peak_v": signal_amp,
        "poincare_std_mV_per_tnorm": float(np.std(ps)) if ps.size else float("nan"),
        "poincare_std_mV_per_tnorm_upward": float(np.std(ps)) if ps.size else float("nan"),
        "poincare_std_mV_per_tnorm_downward": float(np.std(branches["downward"])) if branches["downward"].size else float("nan"),
        "poincare_std_mV_per_tnorm_both": float(np.std(branches["both"])) if branches["both"].size else float("nan"),
        "poincare_count": int(ps.size),
        "branch_mean": float(np.mean(ps)) if ps.size else float("nan"),
        "branch_min": float(np.min(ps)) if ps.size else float("nan"),
        "branch_max": float(np.max(ps)) if ps.size else float("nan"),
        "branch_p01": float(np.percentile(ps, 1)) if ps.size else float("nan"),
        "branch_p99": float(np.percentile(ps, 99)) if ps.size else float("nan"),
        "beta_mean": float(np.mean(beta_ss)) if beta_ss.size else float("nan"),
        "beta_std": float(np.std(beta_ss)) if beta_ss.size else float("nan"),
        "gamma_mean": float(np.mean(gamma_ss)) if gamma_ss.size else float("nan"),
        "gamma_std": float(np.std(gamma_ss)) if gamma_ss.size else float("nan"),
        "max_abs_phi_last_recorded": float(np.max(np.abs(phi_last[start:]))) if len(phi_last[start:]) else 0.0,
        "final_input_current_a": float(ii_final),
        "final_load_current_a": float(il_final),
        "recorded_samples": int(len(t)),
        "steady_samples": int(len(ts)),
    }


def save_point(path: Path, device: Device, cfg: RunConfig, out, runtime: float, analysis: dict):
    path.mkdir(parents=True, exist_ok=True)
    t, vout, phi_last, beta_t, gamma_t, phi_final, ii_final, il_final = out
    np.savez_compressed(path / "timeseries.npz", t_s=t, vout_v=vout, phi_last=phi_last, phi_final=phi_final)
    meta = {
        "device": asdict(device),
        "run": asdict(cfg),
        "runtime_s": runtime,
        "analysis": analysis,
        "f_plasma_ghz": device.f_plasma_hz / 1e9,
        "dt_physical_ps": cfg.dt_norm / device.omega_plasma * 1e12,
        "n_steps": int(round(cfg.tmax_norm / cfg.dt_norm)),
    }
    (path / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    start = int(round(cfg.transient_fraction * len(t)))
    fghz, dbm = spectrum_dbm(t[start:], vout[start:], device.rl_ohm)
    np.savez_compressed(path / "spectrum.npz", frequency_ghz=fghz, spectrum_dbm=dbm)
    branches = poincare_crossing_branches(t[start:], vout[start:])
    np.save(path / "poincare.npy", branches["upward"] / device.omega_plasma * 1e3)
    np.savez_compressed(
        path / "poincare_branches.npz",
        upward=branches["upward"] / device.omega_plasma * 1e3,
        downward=branches["downward"] / device.omega_plasma * 1e3,
        both=branches["both"] / device.omega_plasma * 1e3,
    )


def _point_job(device: Device, cfg: RunConfig):
    out, runtime = simulate(device, cfg)
    a = analyze(device, cfg, out)
    return a, runtime, out


def run_sweep(device: Device, cfgs: list[RunConfig], xvals: np.ndarray, xname: str,
              output: Path, workers: int, keep_timeseries: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    rows = [None] * len(cfgs)
    spectra = [None] * len(cfgs)
    ps_values = [None] * len(cfgs)

    def write_rows() -> None:
        completed = [row for row in rows if row is not None]
        if not completed:
            return
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(completed[0].keys()))
            writer.writeheader()
            writer.writerows(completed)

    def consume(i, cfg, result):
        a, rt, out = result
        t, v, *_ = out
        start = int(round(cfg.transient_fraction * len(t)))
        f, dbm = spectrum_dbm(t[start:], v[start:], device.rl_ohm)
        branches = poincare_crossing_branches(t[start:], v[start:])
        ps = branches["upward"] / device.omega_plasma * 1e3
        rows[i] = {xname: float(xvals[i]), "runtime_s": float(rt), **a}
        spectra[i] = (f, dbm)
        ps_values[i] = ps
        point_path = output / f"point_{i:05d}"
        point_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            point_path / "poincare_branches.npz",
            upward=branches["upward"] / device.omega_plasma * 1e3,
            downward=branches["downward"] / device.omega_plasma * 1e3,
            both=branches["both"] / device.omega_plasma * 1e3,
        )
        if keep_timeseries:
            save_point(point_path, device, cfg, out, rt, a)
        write_rows()

    if workers <= 1:
        for i, cfg in enumerate(cfgs):
            print(f"[{i+1}/{len(cfgs)}] {xname}={xvals[i]:.8g}", flush=True)
            consume(i, cfg, _point_job(device, cfg))
    else:
        # Numba kernel is nogil=True, so threads avoid process-local JIT recompilation.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_point_job, device, cfg): i for i, cfg in enumerate(cfgs)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                consume(i, cfgs[i], fut.result())
                done += 1
                print(f"[{done}/{len(cfgs)}] finished {xname}={xvals[i]:.8g}", flush=True)

    write_rows()

    # Common frequency grid is identical when dt/recording are identical.
    if spectra and spectra[0] is not None:
        f = spectra[0][0]
        mat = np.vstack([s[1] for s in spectra])
        np.savez_compressed(output / "spectra_map.npz", x=xvals, frequency_ghz=f, spectrum_dbm=mat)

    # Ragged Poincare values, stored as object array.
    np.savez_compressed(output / "poincare_map.npz", x=xvals,
                        values=np.asarray(ps_values, dtype=object))
    (output / "config.json").write_text(json.dumps({"device": asdict(device), "base_run": asdict(cfgs[0])}, indent=2), encoding="utf-8")
    plot_sweep_overview(output, xvals, xname, rows, spectra, ps_values)
    if xname in ("pump_dbm", "bias_ua"):
        plot_nonlinearity_overview(output, xvals, xname, rows)
    return rows



def plot_sweep_overview(output: Path, xvals: np.ndarray, xname: str, rows, spectra, ps_values):
    """Write a Fig.-2-style gain / Poincare / Fourier overview."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped (matplotlib unavailable): {exc}", flush=True)
        return
    xlabel = {"pump_dbm": "Pump power (dBm)", "signal_ghz": "Signal frequency (GHz)", "bias_ua": "Bias current (uA)"}.get(xname, xname)
    gain = np.asarray([r["gain_db"] for r in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True, constrained_layout=True)
    axes[0].plot(xvals, gain)
    axes[0].set_ylabel("Gain (dB)")
    for xv, vals in zip(xvals, ps_values):
        if vals is not None and len(vals):
            axes[1].scatter(np.full(len(vals), xv), vals, s=1.5, rasterized=True)
    axes[1].set_ylabel(r"$V'_{PS}$ (mV / normalized time)")
    if spectra and spectra[0] is not None:
        f = spectra[0][0]
        mat = np.vstack([sp[1] for sp in spectra]).T
        mesh = axes[2].pcolormesh(xvals, f, mat, shading="auto")
        axes[2].set_ylim(0.0, min(30.0, float(np.max(f))))
        axes[2].set_ylabel("Fourier frequency (GHz)")
        fig.colorbar(mesh, ax=axes[2], label="FT (dBm)")
    axes[2].set_xlabel(xlabel)
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)


def plot_map_overview(output: Path, pumps: np.ndarray, biases: np.ndarray, gain: np.ndarray, psstd: np.ndarray):
    """Write a Fig.-3-style gain / Poincare-standard-deviation map."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped (matplotlib unavailable): {exc}", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    m0 = axes[0].pcolormesh(biases, pumps, gain, shading="auto")
    axes[0].set_xlabel("Bias current (uA)"); axes[0].set_ylabel("Pump power (dBm)"); axes[0].set_title("Gain (dB)")
    fig.colorbar(m0, ax=axes[0], label="Gain (dB)")
    m1 = axes[1].pcolormesh(biases, pumps, psstd, shading="auto")
    axes[1].set_xlabel("Bias current (uA)"); axes[1].set_ylabel("Pump power (dBm)"); axes[1].set_title(r"$\sigma_{V'_{PS}}$")
    fig.colorbar(m1, ax=axes[1], label=r"$\sigma_{V'_{PS}}$")
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)


def plot_nonlinearity_overview(output: Path, xvals: np.ndarray, xname: str, rows):
    """Write a Fig.-4-style mean/std plot of beta(t) and gamma(t)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    xlabel = {"pump_dbm": "Pump power (dBm)", "bias_ua": "Bias current (uA)"}.get(xname, xname)
    bmean = np.asarray([r["beta_mean"] for r in rows]); bstd = np.asarray([r["beta_std"] for r in rows])
    gmean = np.asarray([r["gamma_mean"] for r in rows]); gstd = np.asarray([r["gamma_std"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True, constrained_layout=True)
    ax = axes[0]; ax2 = ax.twinx(); ax.plot(xvals, bmean, label="mean beta"); ax2.plot(xvals, bstd, linestyle="--", label="std beta"); ax.set_ylabel("mean beta"); ax2.set_ylabel("std beta")
    ax = axes[1]; ax2 = ax.twinx(); ax.plot(xvals, gmean, label="mean gamma"); ax2.plot(xvals, gstd, linestyle="--", label="std gamma"); ax.set_ylabel("mean gamma"); ax2.set_ylabel("std gamma"); ax.set_xlabel(xlabel)
    fig.savefig(output / "nonlinear_coefficients.png", dpi=180)
    plt.close(fig)

def add_common(p: argparse.ArgumentParser):
    p.add_argument("--n-cells", type=int, default=990)
    p.add_argument("--dt-norm", type=float, default=0.01)
    p.add_argument("--tmax-norm", type=float, default=20_000.0)
    p.add_argument("--transient-fraction", type=float, default=0.5)
    p.add_argument("--record-stride", type=int, default=20)
    p.add_argument("--pump-ghz", type=float, default=7.0)
    p.add_argument("--signal-ghz", type=float, default=6.42)
    p.add_argument("--pump-dbm", type=float, default=-55.0)
    p.add_argument("--signal-dbm", type=float, default=-100.0)
    p.add_argument("--bias-ua", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=-1.0, help="<=0: sinusoidal CPR; otherwise transparency in (0,1]")
    p.add_argument("--power-convention", choices=["paper", "50ohm"], default="50ohm")
    p.add_argument("--port-update", choices=["stable", "paper-centered"], default="stable", help="stable integrates the same boundary ODEs without the leapfrog parasitic mode")
    p.add_argument("--quick", action="store_true", help="use tmax_norm=20 for a fast smoke test")


def make_device(args) -> Device:
    return Device(n_cells=args.n_cells)


def make_cfg(args, **updates) -> RunConfig:
    d = dict(
        dt_norm=args.dt_norm,
        tmax_norm=(20.0 if args.quick else args.tmax_norm),
        transient_fraction=args.transient_fraction,
        record_stride=args.record_stride,
        pump_ghz=args.pump_ghz,
        signal_ghz=args.signal_ghz,
        pump_dbm=args.pump_dbm,
        signal_dbm=args.signal_dbm,
        bias_ua=args.bias_ua,
        tau=args.tau,
        power_convention=args.power_convention,
        port_update=args.port_update,
    )
    d.update(updates)
    return RunConfig(**d)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("point", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p); p.add_argument("--output", type=Path, default=Path("outputs/guarcello_point"))

    p = sub.add_parser("benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p); p.add_argument("--bench-steps", type=int, default=50_000)

    p = sub.add_parser("sweep-pump", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p); p.add_argument("--start", type=float, default=-70.0); p.add_argument("--stop", type=float, default=-45.0); p.add_argument("--num", type=int, default=101)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--output", type=Path, default=Path("outputs/guarcello_fig2a")); p.add_argument("--keep-timeseries", action="store_true")

    p = sub.add_parser("sweep-signal", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p); p.add_argument("--start", type=float, default=4.0); p.add_argument("--stop", type=float, default=10.0); p.add_argument("--num", type=int, default=121)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--output", type=Path, default=Path("outputs/guarcello_fig2b")); p.add_argument("--keep-timeseries", action="store_true")

    p = sub.add_parser("sweep-bias", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p); p.add_argument("--start", type=float, default=0.0); p.add_argument("--stop", type=float, default=34.0); p.add_argument("--num", type=int, default=137)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--output", type=Path, default=Path("outputs/guarcello_fig2c")); p.add_argument("--keep-timeseries", action="store_true")

    p = sub.add_parser("map-pump-bias", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common(p); p.add_argument("--pump-start", type=float, default=-61.0); p.add_argument("--pump-stop", type=float, default=-53.0); p.add_argument("--pump-num", type=int, default=81)
    p.add_argument("--bias-start", type=float, default=0.0); p.add_argument("--bias-stop", type=float, default=6.0); p.add_argument("--bias-num", type=int, default=61)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--output", type=Path, default=Path("outputs/guarcello_fig3"))

    args = ap.parse_args()
    device = make_device(args)

    if args.cmd == "point":
        cfg = make_cfg(args)
        # Warm JIT with a tiny run so reported runtime excludes compilation.
        warm = RunConfig(**{**asdict(cfg), "tmax_norm": 0.05, "record_stride": 1})
        simulate(device, warm)
        out, runtime = simulate(device, cfg)
        a = analyze(device, cfg, out)
        save_point(args.output, device, cfg, out, runtime, a)
        print(json.dumps({
            "runtime_s": runtime,
            "analysis": a,
            "f_plasma_ghz": device.f_plasma_hz / 1e9,
            "dt_ps": cfg.dt_norm / device.omega_plasma * 1e12,
            "steps": int(round(cfg.tmax_norm / cfg.dt_norm)),
            "output": str(args.output),
        }, indent=2))
        return

    if args.cmd == "benchmark":
        cfg = make_cfg(args, tmax_norm=args.bench_steps * args.dt_norm, record_stride=max(1, args.bench_steps // 1000))
        warm = RunConfig(**{**asdict(cfg), "tmax_norm": 0.05, "record_stride": 1})
        simulate(device, warm)
        _, runtime = simulate(device, cfg)
        full_steps = int(round(args.tmax_norm / args.dt_norm))
        per_step = runtime / args.bench_steps
        print(json.dumps({
            "n_cells": device.n_cells,
            "bench_steps": args.bench_steps,
            "benchmark_runtime_s": runtime,
            "us_per_step": 1e6 * per_step,
            "estimated_full_steps": full_steps,
            "estimated_full_runtime_s": per_step * full_steps,
            "estimated_full_runtime_min": per_step * full_steps / 60.0,
        }, indent=2))
        return

    if args.cmd in ("sweep-pump", "sweep-signal", "sweep-bias"):
        base = make_cfg(args)
        if args.cmd == "sweep-pump":
            x = np.linspace(args.start, args.stop, args.num)
            cfgs = [RunConfig(**{**asdict(base), "pump_dbm": float(v)}) for v in x]
            xname = "pump_dbm"
        elif args.cmd == "sweep-signal":
            x = np.linspace(args.start, args.stop, args.num)
            cfgs = [RunConfig(**{**asdict(base), "signal_ghz": float(v)}) for v in x]
            xname = "signal_ghz"
        else:
            x = np.linspace(args.start, args.stop, args.num)
            cfgs = [RunConfig(**{**asdict(base), "bias_ua": float(v)}) for v in x]
            xname = "bias_ua"
        # JIT warmup in the parent thread.
        warm = RunConfig(**{**asdict(base), "tmax_norm": 0.05, "record_stride": 1})
        simulate(device, warm)
        rows = run_sweep(device, cfgs, x, xname, args.output, args.workers, args.keep_timeseries)
        print(f"wrote {len(rows)} points to {args.output}")
        return

    if args.cmd == "map-pump-bias":
        base = make_cfg(args)
        pumps = np.linspace(args.pump_start, args.pump_stop, args.pump_num)
        biases = np.linspace(args.bias_start, args.bias_stop, args.bias_num)
        pts = [(p, b) for p in pumps for b in biases]
        cfgs = [RunConfig(**{**asdict(base), "pump_dbm": float(p), "bias_ua": float(b)}) for p, b in pts]
        args.output.mkdir(parents=True, exist_ok=True)
        warm = RunConfig(**{**asdict(base), "tmax_norm": 0.05, "record_stride": 1})
        simulate(device, warm)
        gain = np.empty((len(pumps), len(biases)))
        psstd = np.empty_like(gain)
        runtime = np.empty_like(gain)
        def do_one(i):
            a, rt, _ = _point_job(device, cfgs[i])
            return i, a, rt
        if args.workers <= 1:
            iterator = map(do_one, range(len(cfgs)))
        else:
            ex = ThreadPoolExecutor(max_workers=args.workers)
            iterator = (f.result() for f in as_completed([ex.submit(do_one, i) for i in range(len(cfgs))]))
        count = 0
        for i, a, rt in iterator:
            ip = i // len(biases); ib = i % len(biases)
            gain[ip, ib] = a["gain_db"]
            psstd[ip, ib] = a["poincare_std_mV_per_tnorm"]
            runtime[ip, ib] = rt
            count += 1
            if count % max(1, len(cfgs)//100) == 0:
                print(f"[{count}/{len(cfgs)}]", flush=True)
        if args.workers > 1:
            ex.shutdown()
        np.savez_compressed(args.output / "map.npz", pump_dbm=pumps, bias_ua=biases, gain_db=gain, poincare_std=psstd, runtime_s=runtime)
        (args.output / "config.json").write_text(json.dumps({"device": asdict(device), "base_run": asdict(base)}, indent=2), encoding="utf-8")
        plot_map_overview(args.output, pumps, biases, gain, psstd)
        print(f"wrote {len(cfgs)} points to {args.output}")


if __name__ == "__main__":
    main()
