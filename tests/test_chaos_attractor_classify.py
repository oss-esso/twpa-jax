from __future__ import annotations

import numpy as np
import pytest
import importlib.util
from pathlib import Path
import sys

from scripts.chaos.attractor_classify import (
    CHAOS_NO_CLEAN_BIFURCATION,
    NEIMARK_SACKER,
    NO_BIFURCATION_FOUND,
    PERIOD_DOUBLING,
    PERIOD_DOUBLING_ONSET,
    PITCHFORK_CANDIDATE,
    FOLD_CANDIDATE,
    classify_attractor,
    classify_details,
    classify_trace,
    largest_lyapunov_map,
    poincare_crossings,
    poincare_crossing_branches,
    sigma_vprime_ps,
    classify_sweep,
    sigma_ratio,
    is_smooth_monotone_rise,
    _period_clusters,
    fourier_map,
    period_multiple,
    symmetry_order_parameters,
)
from scripts.chaos.run_attractor_campaign import dbm_to_peak_current, parse_ratios
from scripts.chaos.run_guarcello_repro import gain_vs_off_db, add_gain_vs_off


_FDTD_SPEC = importlib.util.spec_from_file_location(
    "guarcello_jtwpa_fdtd_test",
    Path(__file__).parents[1] / "docs" / "development" / "chaos_papers" /
    "guarcello_jtwpa_fdtd.py",
)
assert _FDTD_SPEC is not None and _FDTD_SPEC.loader is not None
_FDTD = importlib.util.module_from_spec(_FDTD_SPEC)
sys.modules[_FDTD_SPEC.name] = _FDTD
_FDTD_SPEC.loader.exec_module(_FDTD)


def test_crossings_and_sigma_match_reference_formula() -> None:
    t = np.linspace(0.0, 20.0, 2001)
    v = np.sin(t) + 0.1 * np.sin(3.0 * t)
    expected_dv = np.gradient(v, t)
    indices = np.flatnonzero(v[:-1] * v[1:] <= 0.0)
    expected = []
    for i in indices:
        a = 0.0 if v[i + 1] == v[i] else -v[i] / (v[i + 1] - v[i])
        expected.append(expected_dv[i] + a * (expected_dv[i + 1] - expected_dv[i]))
    branches = poincare_crossing_branches(t, v)
    assert np.allclose(branches["both"], expected)
    assert np.isclose(sigma_vprime_ps(branches["both"]), np.std(expected))


def test_clean_sinusoid_uses_one_direction_for_sigma() -> None:
    t = np.linspace(0.0, 40.0 * np.pi, 40001)
    amplitude = 0.7
    omega = 1.3
    values = amplitude * np.sin(omega * t)
    crossings = poincare_crossings(t, values)
    assert crossings.size > 4
    assert np.std(crossings) < 1.0e-4
    assert np.isclose(np.mean(crossings), amplitude * omega, rtol=2.0e-3)


def test_synthetic_verdicts() -> None:
    assert classify_attractor(np.ones(80) * 0.02) == NO_BIFURCATION_FOUND
    assert classify_attractor(np.tile([0.02, 1.0], 40),
                              spectrum_frequencies_hz=np.array([3.95e9]),
                              drive_hz=7.9e9) == PERIOD_DOUBLING
    angle = np.linspace(0.0, 10.0 * np.pi, 100)
    torus = np.column_stack((np.cos(angle), np.sin(angle)))
    assert classify_attractor(torus) == NEIMARK_SACKER
    rng = np.random.default_rng(4)
    assert classify_attractor(rng.normal(size=100)) == CHAOS_NO_CLEAN_BIFURCATION


def test_trace_classifier_uses_voltage_crossings() -> None:
    t = np.linspace(0.0, 100e-9, 20_001)
    v = np.sin(2.0 * np.pi * 7.9e9 * t)
    result = classify_trace(t, v, drive_hz=7.9e9)
    assert result.verdict == NO_BIFURCATION_FOUND
    assert result.poincare_points > 100


def test_scalar_sideband_does_not_claim_neimark_sacker() -> None:
    result = classify_attractor(
        np.sin(np.linspace(0.0, 80.0 * np.pi, 240)),
        spectrum_frequencies_hz=np.array([1.37]), drive_hz=1.0,
    )
    assert result == CHAOS_NO_CLEAN_BIFURCATION


def test_campaign_power_and_resistance_parsers() -> None:
    ratios = parse_ratios("inf, 1e6, 1e2")
    assert np.isinf(ratios[0]) and ratios[1:] == (1.0e6, 1.0e2)
    assert np.isclose(dbm_to_peak_current(-30.0), np.sqrt(2.0e-6 / 50.0))


def test_guarcello_gain_is_normalized_to_pump_off() -> None:
    assert np.isclose(gain_vs_off_db(2.0, 1.0), 20.0 * np.log10(2.0))


def test_guarcello_reduction_adds_reference_gain_without_overwriting_absolute() -> None:
    rows = [{"signal_vout_peak_v": 2.0, "gain_db": -7.0}]
    result = add_gain_vs_off(rows, unpumped_signal_v=1.0)
    assert result[0]["gain_db"] == -7.0
    assert result[0]["gain_vs_off_db"] == pytest.approx(6.0205999)


def test_multitone_estimator_rejects_large_pump_leakage() -> None:
    t = np.arange(4097, dtype=float) * 2.0e-11
    signal_hz = 6.42e9 + 0.137e9
    pump_hz = 7.0e9 + 0.271e9
    expected = 1.0e-6
    voltage = (
        expected * np.sin(2.0 * np.pi * signal_hz * t + 0.37)
        + 10 ** (43.0 / 20.0) * expected *
        np.sin(2.0 * np.pi * pump_hz * t + 1.11)
    )
    old = _FDTD.exact_tone_amplitude(t, voltage, signal_hz)
    new = _FDTD.multitone_amplitude(t, voltage, signal_hz, pump_hz)
    assert abs(old - expected) / expected > 1.0e-3
    assert new == pytest.approx(expected, rel=1.0e-9)


def test_sampled_map_lyapunov_matches_known_expansion() -> None:
    value = largest_lyapunov_map(lambda x: 2.0 * np.asarray(x), np.array([1.0]), steps=12)
    assert np.isclose(value, np.log(2.0), atol=1e-10)


def test_ratio_statistic_replaces_absolute_sigma_cut() -> None:
    assert np.isclose(sigma_ratio(0.1, 0.0025), 40.0)
    controls = np.arange(6.0)
    sigmas = np.array([0.002, 0.0024, 0.003, 0.004, 0.005, 0.006])
    assert is_smooth_monotone_rise(controls, sigmas)
    assert all(value == NO_BIFURCATION_FOUND for value in classify_sweep(controls, sigmas))


def test_ratio_sweep_can_report_a_discontinuous_broadening() -> None:
    controls = np.arange(6.0)
    sigmas = np.array([0.002, 0.0021, 0.0022, 0.1, 0.11, 0.1])
    assert not is_smooth_monotone_rise(controls, sigmas)
    assert classify_sweep(controls, sigmas)[3] == CHAOS_NO_CLEAN_BIFURCATION


def test_period_multiple_resolves_period_one_two_and_four() -> None:
    pump_hz = 1.0
    t = np.linspace(0.0, 64.0, 64 * 256 + 1)
    for multiple in (1, 2, 4):
        voltage = np.sin(2.0 * np.pi * t / multiple)
        assert period_multiple(t, voltage, pump_hz) == multiple


def test_symmetry_order_parameters_measure_even_harmonic() -> None:
    pump_hz = 1.0
    t = np.linspace(0.0, 64.0, 64 * 256 + 1)
    odd = np.sin(2.0 * np.pi * t) + 0.2 * np.sin(6.0 * np.pi * t)
    broken = odd + 0.03 * np.sin(4.0 * np.pi * t)
    assert symmetry_order_parameters(t, odd, pump_hz)["q_even"] < 1.0e-12
    assert symmetry_order_parameters(t, broken, pump_hz)["q_even"] == pytest.approx(
        0.03, rel=0.01
    )


def test_fourier_map_default_preserves_legacy_output() -> None:
    t = np.linspace(0.0, 3.7, 257)
    v = 0.4 + np.sin(2.1 * t) + 0.2 * np.cos(4.2 * t)
    expected = np.fft.rfftfreq(v.size, np.mean(np.diff(t)))
    window = np.hanning(v.size)
    expected_amplitude = 2.0 * np.abs(
        np.fft.rfft((v - np.mean(v)) * window)
    ) / max(np.sum(window), 1.0e-30)
    result = fourier_map(t, v)
    assert np.array_equal(result["frequency_hz"], expected)
    assert np.array_equal(result["amplitude"], expected_amplitude)


def test_period_clusters_decay_resolves_feigenbaum_scaled_period_eight() -> None:
    points = np.array([0.0, 0.2, 0.27, 0.298, 0.309, 0.3134, 0.3152, 0.316])
    without_decay, _ = _period_clusters(points, tolerance=0.03, tolerance_decay=1.0)
    with_decay, _ = _period_clusters(points, tolerance=0.03, tolerance_decay=2.503)
    assert without_decay < 8
    assert with_decay == 8


def test_period_clusters_decay_one_matches_legacy_formula_exactly() -> None:
    for points in (
        np.array([0.0, 1.0, 1.030927835051546]),
        np.array([-2.0, -1.99, -1.2, 0.0, 0.01, 1.0]),
        np.array([0.0]),
    ):
        scale = max(float(np.ptp(points)), abs(float(np.median(points))), 1.0e-15)
        bins = np.sort(points)
        groups = [[bins[0]]]
        for value in bins[1:]:
            if value - groups[-1][-1] <= 0.03 * scale:
                groups[-1].append(value)
            else:
                groups.append([value])
        expected_centers = np.array([np.mean(group) for group in groups])
        expected = len(groups), expected_centers
        actual = _period_clusters(points, tolerance=0.03, tolerance_decay=1.0)
        assert actual[0] == expected[0]
        assert np.array_equal(actual[1], expected[1])


def test_pitchfork_candidate_requires_unchanged_period_and_broken_symmetry() -> None:
    t = np.linspace(0.0, 64.0, 64 * 256 + 1)
    voltage = np.sin(2.0 * np.pi * t) + 0.03 * np.sin(4.0 * np.pi * t)
    result = classify_trace(
        t, voltage, drive_hz=1.0, baseline_q_even=0.0, baseline_q_dc=0.0,
    )
    assert result.period_multiple == 1
    assert result.q_even == pytest.approx(0.03, rel=0.01)
    assert result.verdict == PITCHFORK_CANDIDATE


def test_fold_candidate_requires_termination_without_symmetry_change() -> None:
    result = classify_details(
        np.ones(64), period_multiple_value=1, q_even=0.0, q_dc=0.0,
        branch_terminated=True,
    )
    assert result.verdict == FOLD_CANDIDATE


def test_period_doubling_precedes_pitchfork_candidate() -> None:
    result = classify_details(
        np.array([0.2, 0.8] * 32),
        spectrum_frequencies_hz=np.array([0.5]), drive_hz=1.0,
        period_multiple_value=2, q_even=0.5, q_dc=0.5,
    )
    assert result.verdict == PERIOD_DOUBLING


def test_period_multiple_handles_6_2_samples_per_pump_period() -> None:
    pump_hz = 1.0
    t = np.arange(0.0, 64.0, 1.0 / 6.2)
    voltage = np.sin(2.0 * np.pi * pump_hz * t)
    assert period_multiple(t, voltage, pump_hz) == 1


def test_period_multiple_returns_zero_for_nonperiodic_trace() -> None:
    t = np.linspace(0.0, 64.0, 64 * 32 + 1)
    rng = np.random.default_rng(19)
    voltage = rng.normal(size=t.size)
    assert period_multiple(t, voltage, 1.0) == 0


def test_symmetry_gate_requires_a_factor_above_the_measured_floor() -> None:
    result = classify_details(
        np.ones(64), period_multiple_value=1, q_even=4.0e-5, q_dc=8.118e-4,
        baseline_q_even=1.0e-5, baseline_q_dc=8.0e-4,
        symmetry_floor_factor=5.0,
    )
    assert result.verdict == NO_BIFURCATION_FOUND


def test_spectral_half_harmonic_gate_overrides_missing_period_match() -> None:
    result = classify_details(
        np.ones(64), period_multiple_value=0, q_even=0.0, q_dc=0.0,
        half_integer_line_db=-70.0, half_integer_floor_db=-180.0,
        half_integer_gate_db=20.0,
    )
    assert result.verdict == PERIOD_DOUBLING_ONSET
    assert result.spectral_period_doubling is True


def test_broad_poincare_cloud_overrides_spectral_half_harmonic_for_clean_verdict() -> None:
    result = classify_details(
        np.arange(26, dtype=float) * 0.04, period_multiple_value=0,
        spectrum_frequencies_hz=np.array([0.5]), drive_hz=1.0,
        half_integer_line_db=-70.0, half_integer_floor_db=-180.0,
        half_integer_gate_db=20.0,
    )
    assert result.poincare_clusters > 8
    assert result.spectral_period_doubling is True
    assert result.verdict == CHAOS_NO_CLEAN_BIFURCATION
