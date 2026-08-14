#!/usr/bin/env python3
"""Attractor-continuation diagnostics and bifurcation classification.

The module keeps the geometric classifier independent from the large transient
solver.  A transient campaign can therefore test the classifier on saved
traces, while ``continue_attractor`` supplies the required carry-forward
protocol to any integrator with a small callback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp

PERIOD_DOUBLING = "PERIOD_DOUBLING"
PERIOD_DOUBLING_ONSET = "PERIOD_DOUBLING_ONSET"
PITCHFORK_CANDIDATE = "PITCHFORK_CANDIDATE"
FOLD_CANDIDATE = "FOLD_CANDIDATE"
NEIMARK_SACKER = "NEIMARK_SACKER"
CHAOS_NO_CLEAN_BIFURCATION = "CHAOS_NO_CLEAN_BIFURCATION"
NO_BIFURCATION_FOUND = "NO_BIFURCATION_FOUND"
MIN_SAMPLES_PER_PERIOD = 4.0
MAX_PERIOD_MULTIPLE = 8


def poincare_crossing_branches(t: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Return upward, downward, and legacy two-sign Poincare branches."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or t.size != v.size:
        raise ValueError("t and v must be one-dimensional arrays of equal length")
    if v.size < 3:
        empty = np.empty(0, dtype=float)
        return {"upward": empty, "downward": empty, "both": empty}
    dv = np.gradient(v, t)
    indices = np.nonzero(v[:-1] * v[1:] <= 0.0)[0]
    values = np.empty(indices.size, dtype=float)
    for j, i in enumerate(indices):
        denominator = v[i + 1] - v[i]
        fraction = 0.0 if denominator == 0.0 else -v[i] / denominator
        values[j] = dv[i] + fraction * (dv[i + 1] - dv[i])
    return {
        "upward": values[values > 0.0],
        "downward": values[values < 0.0],
        "both": values,
    }


def poincare_crossings(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the upward ``dV/dt`` branch used by the sigma statistic."""
    return poincare_crossing_branches(t, v)["upward"]


def sigma_vprime_ps(points: np.ndarray, *, ddof: int = 0) -> float:
    """Scalar Guarcello chaos indicator ``sigma(V'_PS)``."""
    values = np.asarray(points, dtype=float).reshape(-1)
    if values.size <= ddof:
        return float("nan")
    return float(np.std(values, ddof=ddof))


def _harmonic_residual(frequencies: np.ndarray, drive_hz: float, *, tol: float) -> float:
    if frequencies.size == 0 or drive_hz <= 0.0:
        return float("inf")
    distances = np.abs(frequencies[:, None] - drive_hz * np.arange(1, 20)[None, :])
    return float(np.min(distances) / drive_hz) if distances.size else float("inf")


def _peak_frequencies(t: np.ndarray, v: np.ndarray, *, fmax_hz: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    if len(v) < 8:
        return np.empty(0), np.empty(0)
    dt = float(np.mean(np.diff(t)))
    if dt <= 0.0:
        raise ValueError("time samples must increase")
    centered = np.asarray(v, dtype=float) - np.mean(v)
    window = np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(centered * window))
    frequencies = np.fft.rfftfreq(centered.size, dt)
    if spectrum.size:
        spectrum[0] = 0.0
    if fmax_hz is not None:
        keep = frequencies <= fmax_hz
        frequencies, spectrum = frequencies[keep], spectrum[keep]
    if spectrum.size < 3:
        return frequencies, spectrum
    threshold = max(float(np.max(spectrum)) * 0.05, np.finfo(float).eps)
    peaks = np.flatnonzero((spectrum[1:-1] > spectrum[:-2]) &
                           (spectrum[1:-1] >= spectrum[2:])) + 1
    peaks = peaks[spectrum[peaks] >= threshold]
    order = peaks[np.argsort(spectrum[peaks])[::-1]]
    return frequencies[order], spectrum[order]


def _period_clusters(
    points: np.ndarray, *, tolerance: float, tolerance_decay: float = 1.0,
) -> tuple[int, np.ndarray]:
    values = np.asarray(points, dtype=float).reshape(-1)
    if values.size == 0:
        return 0, np.empty(0)
    if tolerance_decay <= 0.0:
        raise ValueError("tolerance_decay must be positive")
    scale = max(float(np.ptp(values)), abs(float(np.median(values))), 1.0e-15)
    bins = np.sort(values)
    groups = [[bins[0]]]
    for value in bins[1:]:
        admitted_tolerance = tolerance / tolerance_decay ** (len(groups) - 1)
        if value - groups[-1][-1] <= admitted_tolerance * scale:
            groups[-1].append(value)
        else:
            groups.append([value])
    centers = np.array([np.mean(group) for group in groups])
    return len(groups), centers


@dataclass(frozen=True)
class Classification:
    verdict: str
    sigma_vprime_ps: float
    poincare_points: int
    poincare_clusters: int
    dominant_frequencies_hz: tuple[float, ...]
    reason: str
    sigma_deep_stable: float | None = None
    sigma_ratio: float | None = None
    ratio_threshold: float | None = None
    period_multiple: int | None = None
    q_even: float | None = None
    q_dc: float | None = None
    spectral_period_doubling: bool = False
    spectral_period_disagreement: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_attractor(
    poincare: np.ndarray, *, spectrum_frequencies_hz: np.ndarray | None = None,
    drive_hz: float | None = None, sigma_threshold: float = 0.1,
    cluster_tolerance: float = 0.03,
    cluster_tolerance_decay: float = 1.0,
    sigma_deep_stable: float | None = None,
    ratio_threshold: float = 40.0,
    sweep_controls: np.ndarray | None = None,
    sweep_sigmas: np.ndarray | None = None,
    period_multiple_value: int | None = None,
    q_even: float | None = None,
    q_dc: float | None = None,
    baseline_q_even: float = 0.0,
    baseline_q_dc: float = 0.0,
    symmetry_floor_factor: float = 5.0,
    half_integer_line_db: float | None = None,
    half_integer_floor_db: float | None = None,
    half_integer_gate_db: float = 20.0,
    branch_terminated: bool = False,
    incommensurate: bool = False,
) -> str:
    """Classify an attractor from Poincare geometry and optional spectrum.

    The return value is intentionally one of the four route-plan verdicts.
    Use ``classify_details`` when the evidence/reason is also needed.
    """
    return classify_details(
        poincare, spectrum_frequencies_hz=spectrum_frequencies_hz,
        drive_hz=drive_hz, sigma_threshold=sigma_threshold,
        cluster_tolerance=cluster_tolerance, cluster_tolerance_decay=cluster_tolerance_decay,
        sigma_deep_stable=sigma_deep_stable,
        ratio_threshold=ratio_threshold, sweep_controls=sweep_controls,
        sweep_sigmas=sweep_sigmas,
        period_multiple_value=period_multiple_value, q_even=q_even, q_dc=q_dc,
        baseline_q_even=baseline_q_even, baseline_q_dc=baseline_q_dc,
        symmetry_floor_factor=symmetry_floor_factor,
        half_integer_line_db=half_integer_line_db,
        half_integer_floor_db=half_integer_floor_db,
        half_integer_gate_db=half_integer_gate_db,
        branch_terminated=branch_terminated, incommensurate=incommensurate,
    ).verdict


def classify_details(
    poincare: np.ndarray, *, spectrum_frequencies_hz: np.ndarray | None = None,
    drive_hz: float | None = None, sigma_threshold: float = 0.1,
    cluster_tolerance: float = 0.03,
    cluster_tolerance_decay: float = 1.0,
    sigma_deep_stable: float | None = None,
    ratio_threshold: float = 40.0,
    sweep_controls: np.ndarray | None = None,
    sweep_sigmas: np.ndarray | None = None,
    period_multiple_value: int | None = None,
    q_even: float | None = None,
    q_dc: float | None = None,
    baseline_q_even: float = 0.0,
    baseline_q_dc: float = 0.0,
    symmetry_floor_factor: float = 5.0,
    half_integer_line_db: float | None = None,
    half_integer_floor_db: float | None = None,
    half_integer_gate_db: float = 20.0,
    branch_terminated: bool = False,
    incommensurate: bool = False,
) -> Classification:
    """Return a verdict with enough evidence to audit a campaign result."""
    values = np.asarray(poincare, dtype=float)
    if values.ndim == 1:
        scalar = values
        dimensions = 1
    elif values.ndim == 2 and values.shape[1] >= 1:
        scalar = values[:, 0]
        dimensions = values.shape[1]
    else:
        raise ValueError("poincare must be a vector or an N x D array")
    scalar = scalar[np.isfinite(scalar)]
    sigma = sigma_vprime_ps(scalar)
    if sigma_deep_stable is not None and sigma_deep_stable <= 0.0:
        raise ValueError("sigma_deep_stable must be positive")
    ratio = None if sigma_deep_stable is None else sigma / sigma_deep_stable
    clusters, centers = _period_clusters(
        scalar, tolerance=cluster_tolerance, tolerance_decay=cluster_tolerance_decay
    )
    frequencies = np.asarray(spectrum_frequencies_hz if spectrum_frequencies_hz is not None else [], dtype=float)
    frequencies = frequencies[np.isfinite(frequencies)]
    dominant = tuple(float(x) for x in frequencies[:8])
    half_harmonic = False
    if drive_hz and drive_hz > 0 and frequencies.size:
        half_harmonic = bool(np.min(np.abs(frequencies - drive_hz / 2.0)) < 0.03 * drive_hz or
                             np.min(np.abs(frequencies - 3.0 * drive_hz / 2.0)) < 0.03 * drive_hz)
    monotone = is_smooth_monotone_rise(sweep_controls, sweep_sigmas)
    if symmetry_floor_factor <= 1.0:
        raise ValueError("symmetry_floor_factor must exceed one")
    if half_integer_gate_db <= 0.0:
        raise ValueError("half_integer_gate_db must be positive")
    spectral_period_doubling = bool(
        half_integer_line_db is not None and half_integer_floor_db is not None
        and np.isfinite(half_integer_line_db) and np.isfinite(half_integer_floor_db)
        and half_integer_line_db - half_integer_floor_db >= half_integer_gate_db
    )
    spectral_period_disagreement = bool(
        spectral_period_doubling and period_multiple_value not in (2, 4, 8)
    )
    new_evidence = period_multiple_value is not None or q_even is not None or q_dc is not None
    if spectral_period_doubling and clusters > MAX_PERIOD_MULTIPLE:
        verdict, reason = CHAOS_NO_CLEAN_BIFURCATION, (
            "half-integer spectrum is present, but the Poincare section has "
            f"{clusters} clusters, exceeding the maximum tested period "
            f"multiple {MAX_PERIOD_MULTIPLE}"
        )
    elif spectral_period_doubling and period_multiple_value in (2, 4, 8):
        verdict, reason = PERIOD_DOUBLING, (
            f"pump-referred half-integer spectrum exceeds its floor by "
            f"{half_integer_line_db - half_integer_floor_db:.3g} dB"
        )
    elif spectral_period_doubling:
        verdict, reason = PERIOD_DOUBLING_ONSET, (
            "half-integer spectrum is present, but the time-domain period "
            f"test returned {period_multiple_value}; no clean period-{period_multiple_value} closure"
        )
    elif new_evidence and period_multiple_value is not None and period_multiple_value >= 2 and half_harmonic:
        verdict, reason = PERIOD_DOUBLING, (
            f"period multiple {period_multiple_value} with half-integer pump line"
        )
    elif new_evidence and period_multiple_value == 0:
        if clusters <= 1:
            verdict, reason = NO_BIFURCATION_FOUND, (
                "the period test found no closure, but the directional Poincare "
                "section remains one compact cluster"
            )
        else:
            verdict, reason = CHAOS_NO_CLEAN_BIFURCATION, "no tested integer pump period matches the waveform"
    elif new_evidence and period_multiple_value == 1 and (
        (q_even is not None and q_even > max(abs(baseline_q_even), 1.0e-12) * symmetry_floor_factor)
        or (q_dc is not None and q_dc > max(abs(baseline_q_dc), 1.0e-12) * symmetry_floor_factor)
    ):
        verdict, reason = PITCHFORK_CANDIDATE, "period unchanged with broken pump symmetry"
    elif new_evidence and period_multiple_value == 1 and branch_terminated:
        verdict, reason = FOLD_CANDIDATE, "period and symmetry unchanged at branch termination"
    elif new_evidence and period_multiple_value == 1:
        verdict, reason = NO_BIFURCATION_FOUND, "period and symmetry remain unchanged"
    elif new_evidence and incommensurate:
        verdict, reason = NEIMARK_SACKER, "incommensurate sideband lattice"
    elif monotone:
        verdict, reason = NO_BIFURCATION_FOUND, (
            "sigma rises smoothly and monotonically with the drive; no bifurcation shape"
        )
    elif scalar.size < 4:
        verdict, reason = NO_BIFURCATION_FOUND, "too few post-transient Poincare points"
    elif clusters == 1 and not (
        frequencies.size and drive_hz and
        _harmonic_residual(frequencies[:4], drive_hz, tol=0.03) > 0.03
    ):
        verdict, reason = NO_BIFURCATION_FOUND, "one compact directional Poincare cluster"
    elif clusters == 2 and (half_harmonic or not np.isclose(centers[0] + centers[1], 0.0,
                                                              atol=cluster_tolerance * max(np.ptp(scalar), 1e-15))):
        verdict, reason = PERIOD_DOUBLING, "two resolved Poincare clusters"
    elif clusters == 2 and np.isclose(centers[0] + centers[1], 0.0,
                                      atol=cluster_tolerance * max(np.ptp(scalar), 1e-15)):
        verdict, reason = NO_BIFURCATION_FOUND, "compact symmetric crossing pair at the drive frequency"
    elif dimensions >= 2 and scalar.size >= 12 and np.ptp(values[:, 1]) > 1e-10:
        verdict, reason = NEIMARK_SACKER, "Poincare samples occupy a two-dimensional closed set"
    elif sigma_deep_stable is not None and ratio is not None and ratio < ratio_threshold:
        verdict, reason = NO_BIFURCATION_FOUND, (
            f"within-sweep sigma ratio {ratio:.3g} is below {ratio_threshold:.3g}"
        )
    elif sigma < sigma_threshold and clusters <= 3:
        verdict, reason = NO_BIFURCATION_FOUND, "compact Poincare set with discrete response"
    elif frequencies.size and drive_hz and _harmonic_residual(frequencies[:4], drive_hz, tol=0.03) > 0.03:
        # A sideband alone is not a Neimark--Sacker verdict.  The route plan
        # makes Poincare geometry primary; a scalar V'_PS trace cannot show a
        # closed invariant curve.  Keep the observation as corroborating
        # evidence for a non-clean boundary instead of over-classifying it.
        verdict, reason = CHAOS_NO_CLEAN_BIFURCATION, (
            "non-harmonic sideband without two-dimensional Poincare geometry"
        )
    else:
        verdict, reason = CHAOS_NO_CLEAN_BIFURCATION, "broad/scattered Poincare response without a resolved two-point route"
    return Classification(
        verdict, sigma, int(scalar.size), clusters, dominant, reason,
        sigma_deep_stable, ratio, ratio_threshold,
        period_multiple_value, q_even, q_dc,
        spectral_period_doubling, spectral_period_disagreement,
    )


def sigma_ratio(sigma_max: float, sigma_deep_stable: float) -> float:
    """Return the dimensionless within-sweep sigma ratio."""
    if sigma_deep_stable <= 0.0:
        raise ValueError("sigma_deep_stable must be positive")
    return float(sigma_max / sigma_deep_stable)


def is_smooth_monotone_rise(
    controls: np.ndarray | None, sigmas: np.ndarray | None,
    *, max_relative_step: float = 0.35,
) -> bool:
    """Identify drive-tracking sigma ramps that are not bifurcations."""
    if controls is None or sigmas is None:
        return False
    x = np.asarray(controls, dtype=float).reshape(-1)
    y = np.asarray(sigmas, dtype=float).reshape(-1)
    if x.size < 4 or x.size != y.size or not np.all(np.isfinite(y)):
        return False
    order = np.argsort(x, kind="stable")
    values = y[order]
    differences = np.diff(values)
    if np.count_nonzero(differences >= -np.finfo(float).eps) < values.size - 2:
        return False
    scale = max(float(np.max(values)), np.finfo(float).eps)
    return bool(np.max(np.abs(np.diff(differences))) / scale <= max_relative_step)


def classify_sweep(
    controls: np.ndarray, sigmas: np.ndarray, *, ratio_threshold: float = 40.0,
) -> list[str]:
    """Classify scalar sweep points, rejecting smooth monotone sigma ramps."""
    controls = np.asarray(controls, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if controls.shape != sigmas.shape:
        raise ValueError("controls and sigmas must have equal shapes")
    stable = float(np.min(sigmas))
    guard = is_smooth_monotone_rise(controls, sigmas)
    if guard:
        return [NO_BIFURCATION_FOUND] * sigmas.size
    return [
        (CHAOS_NO_CLEAN_BIFURCATION if sigma_ratio(float(value), stable) >= ratio_threshold
         else NO_BIFURCATION_FOUND)
        for value in sigmas
    ]


def classify_trace(
    time: np.ndarray, output_voltage: np.ndarray, *, drive_hz: float,
    sigma_threshold: float = 0.1, baseline_q_even: float = 0.0,
    baseline_q_dc: float = 0.0, branch_terminated: bool = False,
    symmetry_floor_factor: float = 5.0,
) -> Classification:
    """Extract Guarcello crossings and FFT peaks from one steady-state trace."""
    points = poincare_crossings(time, output_voltage)
    frequencies, _ = _peak_frequencies(time, output_voltage)
    orders = symmetry_order_parameters(time, output_voltage, drive_hz)
    multiple = period_multiple(time, output_voltage, drive_hz)
    return classify_details(points, spectrum_frequencies_hz=frequencies,
                            drive_hz=drive_hz, sigma_threshold=sigma_threshold,
                            period_multiple_value=multiple,
                            q_even=float(orders["q_even"]),
                            q_dc=float(orders["q_dc"]),
                            baseline_q_even=baseline_q_even,
                            baseline_q_dc=baseline_q_dc,
                            symmetry_floor_factor=symmetry_floor_factor,
                            branch_terminated=branch_terminated)


@dataclass(frozen=True)
class ContinuationPoint:
    control: float
    direction: str
    classification: dict[str, Any]
    state_norm: float
    transient_decay: float


def largest_lyapunov_map(
    step_map: Callable[[Any], Any], initial_state: Any, *, steps: int,
    epsilon: float = 1e-8,
) -> float:
    """Estimate the largest exponent of a sampled-period map.

    This finite-difference map form is the circuit analogue of the Phase-1
    tangent estimator: it does not mistake a diverging transient for an
    attractor exponent, and it works with the existing transient integrator
    without requiring a dense circuit Jacobian.
    """
    x = np.asarray(initial_state, dtype=float).copy()
    direction = np.ones_like(x) / max(np.linalg.norm(x), 1.0)
    total = 0.0
    used = 0
    for _ in range(int(steps)):
        y = np.asarray(step_map(x), dtype=float)
        z = np.asarray(step_map(x + epsilon * direction), dtype=float)
        delta = z - y
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(norm) or norm == 0.0:
            return float("nan")
        total += math.log(norm / epsilon)
        direction = delta / norm
        x = y
        used += 1
    return float(total / used) if used else float("nan")


def fourier_map(
    time: np.ndarray,
    voltage: np.ndarray,
    *,
    fmax_hz: float | None = None,
    keep_dc: bool = False,
) -> dict[str, np.ndarray]:
    """Return a compact FT map row for a continuation point."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    if t.size < 8 or t.size != v.size:
        empty = np.empty(0)
        return {"frequency_hz": empty, "amplitude": empty,
                "complex_amplitude": empty.astype(complex)}
    dt = float(np.mean(np.diff(t)))
    window = np.hanning(v.size)
    frequency = np.fft.rfftfreq(v.size, dt)
    signal = v if keep_dc else v - np.mean(v)
    complex_amplitude = 2.0 * np.fft.rfft(signal * window) / max(np.sum(window), 1e-30)
    amplitude = np.abs(complex_amplitude)
    if fmax_hz is not None:
        keep = frequency <= fmax_hz
        frequency = frequency[keep]
        amplitude = amplitude[keep]
        complex_amplitude = complex_amplitude[keep]
    return {"frequency_hz": frequency, "amplitude": amplitude,
            "complex_amplitude": complex_amplitude}


def _exact_tone_coefficients(
    time: np.ndarray, voltage: np.ndarray, frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Return least-squares cosine/sine coefficients for exact tones."""
    t = np.asarray(time, dtype=float).reshape(-1)
    v = np.asarray(voltage, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if t.size != v.size or t.size < 4:
        raise ValueError("time and voltage must have at least four equal samples")
    columns = [np.ones(t.size)]
    for frequency in frequencies:
        angle = 2.0 * np.pi * frequency * t
        columns.extend((np.cos(angle), np.sin(angle)))
    coefficients, *_ = np.linalg.lstsq(np.column_stack(columns), v, rcond=None)
    return coefficients


def _bandlimited_interpolate(
    time: np.ndarray, values: np.ndarray, queries: np.ndarray,
    *, kernel_half_width: int = 12,
) -> np.ndarray:
    """Interpolate a uniformly sampled trace with a local sinc kernel."""
    step = float(np.mean(np.diff(time)))
    output = np.empty(queries.size, dtype=float)
    for index, query in enumerate(queries):
        sample = (query - time[0]) / step
        center = int(round(sample))
        left = max(0, center - kernel_half_width)
        right = min(values.size, center + kernel_half_width + 1)
        output[index] = float(np.sum(
            values[left:right] * np.sinc(sample - np.arange(left, right))
        ))
    return output


def _fractional_delay(values: np.ndarray, shift_samples: float) -> np.ndarray:
    """Resample a uniform trace at a fractional sample delay by Fourier phase."""
    frequency = np.fft.fftfreq(values.size)
    spectrum = np.fft.fft(values)
    return np.fft.ifft(
        spectrum * np.exp(2.0j * np.pi * frequency * shift_samples)
    ).real


def symmetry_order_parameters(
    time: np.ndarray, voltage: np.ndarray, pump_hz: float,
) -> dict[str, np.ndarray | float]:
    """Measure DC and pump-harmonic symmetry using exact-tone least squares."""
    if pump_hz <= 0.0:
        raise ValueError("pump_hz must be positive")
    coefficients = _exact_tone_coefficients(
        time, voltage, pump_hz * np.arange(1.0, 7.0),
    )
    dc = abs(float(coefficients[0]))
    magnitudes = np.array([
        float(np.hypot(coefficients[2 * index - 1], coefficients[2 * index]))
        for index in range(1, 7)
    ])
    fundamental = max(magnitudes[0], np.finfo(float).eps)
    return {
        "q_even": float(magnitudes[1] / fundamental),
        "q_dc": float(dc / fundamental),
        "harmonic_magnitudes": magnitudes,
    }


def period_multiple(
    time: np.ndarray, voltage: np.ndarray, pump_hz: float, max_n: int = 8,
) -> int:
    """Return the smallest resolved integer multiple of the pump period."""
    t = np.asarray(time, dtype=float).reshape(-1)
    v = np.asarray(voltage, dtype=float).reshape(-1)
    if pump_hz <= 0.0 or max_n < 1 or t.size != v.size or t.size < 8:
        raise ValueError("invalid period-test inputs")
    period = 1.0 / pump_hz
    samples_per_period = 1.0 / (float(np.mean(np.diff(t))) * pump_hz)
    if samples_per_period < MIN_SAMPLES_PER_PERIOD:
        raise ValueError(
            f"period test requires at least {MIN_SAMPLES_PER_PERIOD:g} samples "
            f"per pump period; got {samples_per_period:.3g}"
        )
    centered_norm = np.linalg.norm(v - np.mean(v))
    if centered_norm == 0.0:
        return 1
    tolerance = 2.0e-3
    for multiple in range(1, max_n + 1):
        shift = multiple * period
        shift_samples = shift / float(np.mean(np.diff(t)))
        shifted = _fractional_delay(v, shift_samples)
        margin = int(np.ceil(shift_samples)) + 4
        if v.size - 2 * margin < 32:
            continue
        valid = slice(margin, -margin)
        residual = np.linalg.norm(shifted[valid] - v[valid]) / centered_norm
        if residual <= tolerance:
            return multiple
    return 0


@dataclass
class H1AttractorAdapter:
    """Small bridge from the existing H1 transient model to continuation."""
    module: Any
    system: Any
    state: np.ndarray
    current_a: float
    out_port: int = 2
    periods: int = 40
    samples_per_period: int = 64
    rtol: float = 2e-5
    atol: float = 1e-3
    max_step: float = 0.5
    reference_current_a: float | None = None
    reference_power_dbm: float | None = None
    ramp_periods: int = 10

    def integrate(self, target_current_a: float, previous_state: np.ndarray | None = None):
        initial = self.state if previous_state is None else np.asarray(previous_state, dtype=float)
        start_current = float(self.current_a)
        ramp_theta = 2.0 * math.pi * max(0, int(self.ramp_periods))
        total_theta = ramp_theta + 2.0 * math.pi * self.periods
        theta_all, states_all, integrator = self.module.implicit_trapezoid_ramp(
            self.system, initial, start_current, float(target_current_a),
            total_theta, ramp_theta, self.max_step, newton_tol=1e-6,
            max_newton=12, min_step_theta=1.0 / 32.0,
        )
        if not integrator.get("success"):
            raise RuntimeError(f"H1 attractor integration failed: {integrator.get('message')}")
        theta = np.asarray(theta_all)
        states = np.asarray(states_all)
        hold = theta >= ramp_theta
        theta = theta[hold] - ramp_theta
        states = states[:, hold]
        self.state = states[:, -1]
        self.current_a = float(target_current_a)
        observables = self.module.make_observables(
            self.system, theta, states, out_port=self.out_port,
            start_current=float(target_current_a), target_current=float(target_current_a), ramp_theta=0.0,
        )
        return self.state, theta, observables["output_voltage_v"], states

    def current_for_power_dbm(self, power_dbm: float) -> float:
        """Scale the authoritative checkpoint current in its HB power convention."""
        reference_current = self.reference_current_a or self.current_a
        reference_power = self.reference_power_dbm
        if reference_power is None:
            raise ValueError("checkpoint power is unavailable for power continuation")
        return float(reference_current * 10.0 ** ((float(power_dbm) - reference_power) / 20.0))


def make_h1_attractor_adapter(
    circuit_dir: Path, checkpoint: Path, *, freq_ghz: float = 7.9,
    pump_port: int = 4, out_port: int = 2, periods: int = 40,
    ramp_periods: int = 10, reference_power_dbm: float | None = None,
) -> H1AttractorAdapter:
    """Load an authoritative HB checkpoint for carry-forward continuation."""
    source = Path(__file__).resolve().parents[1] / "h1_transient_branch_transfer.py"
    spec = importlib.util.spec_from_file_location("h1_transient_branch_transfer", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load transient solver from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    circuit = module.load_circuit(circuit_dir)
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    dc_flux = module.checkpoint_dc_flux(report, circuit, checkpoint=checkpoint)
    system = module.build_system(circuit_dir, freq_ghz, pump_port, dc_flux)
    q0, p0, current, report = module.load_hb_initial(checkpoint, system.circuit, system.omega)
    metadata = report.get("metadata", {})
    power = metadata.get("pump_power_dbm_requested", metadata.get("pump_power_dbm"))
    q0 = q0 / system.phi0
    p0 = p0 / system.phi0
    if system.g_alg_factor is None:
        system.project_algebraic_state(
            q0, p0, system.source(0.0, current, current, 0.0)
        )
    return H1AttractorAdapter(
        module, system, system.pack(q0, p0), current,
        out_port, periods, reference_current_a=current, ramp_periods=ramp_periods,
        reference_power_dbm=(float(reference_power_dbm) if reference_power_dbm is not None
                             else (float(power) if power is not None else None)),
    )


def envelope_decay(time: np.ndarray, signal: np.ndarray, *, period: float) -> float:
    """Estimate the normalized late-envelope slope used by the route protocol."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time.size < 8 or period <= 0.0:
        return float("inf")
    window = max(2, int(round(period / np.mean(np.diff(time)))))
    count = signal.size // window
    if count < 4:
        return float("inf")
    envelope = np.array([np.std(signal[i * window:(i + 1) * window]) for i in range(count)])
    x = np.arange(count, dtype=float)
    return float(abs(np.polyfit(x[-max(3, count // 2):], envelope[-max(3, count // 2):], 1)[0]))


def continue_attractor(
    controls: Iterable[float], *, direction: str,
    integrate: Callable[[float, Any], tuple[Any, np.ndarray, np.ndarray, np.ndarray]],
    initial_state: Any, drive_hz: float, decay_limit: float = 1e-5,
) -> list[ContinuationPoint]:
    """Run carry-forward continuation over a control sequence.

    ``integrate(control, state)`` returns ``(final_state, time, voltage,
    state_trace)``.  The callback owns the circuit-specific transient solver;
    this function owns ordering, decay gating, diagnostics, and state reuse.
    """
    state = initial_state
    points: list[ContinuationPoint] = []
    period = 1.0 / drive_hz
    for control in controls:
        state, time, voltage, states = integrate(float(control), state)
        decay = envelope_decay(time, voltage, period=period)
        result = classify_trace(time, voltage, drive_hz=drive_hz)
        if decay > decay_limit:
            result = Classification(result.verdict, result.sigma_vprime_ps,
                                    result.poincare_points, result.poincare_clusters,
                                    result.dominant_frequencies_hz,
                                    result.reason + "; envelope slope did not meet decay gate")
        points.append(ContinuationPoint(float(control), direction, result.as_dict(),
                                        float(np.linalg.norm(np.asarray(state))), decay))
    return points


def _load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    time_key = "theta" if "theta" in keys else "t"
    voltage_key = "output_voltage_v" if "output_voltage_v" in keys else "vout_v"
    if time_key not in keys or voltage_key not in keys:
        raise ValueError(f"{path} must contain ({time_key}, {voltage_key})")
    return np.asarray(data[time_key]), np.asarray(data[voltage_key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="NPZ trace with theta/t and output_voltage_v/vout_v")
    parser.add_argument("--drive-ghz", type=float, default=7.9)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    time, voltage = _load_trace(args.trace)
    result = classify_trace(time, voltage, drive_hz=args.drive_ghz * 1e9)
    payload = result.as_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
