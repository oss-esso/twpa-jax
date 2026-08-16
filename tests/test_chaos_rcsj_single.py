from __future__ import annotations

import numpy as np

from scripts.chaos.rcsj_single_junction import (
    _continuation,
    RCSJParameters,
    drive_phase_poincare,
    extract_period_doubling_sequence,
    feigenbaum_ratios,
    integrate_rk4,
    lyapunov_exponents,
    poincare_period_count,
    staircase_box_counting_dimension,
    stroboscopic_period_count,
)


def test_fixed_rk4_matches_exponential_relaxation() -> None:
    times, states = integrate_rk4(lambda _t, y: -y, np.array([1.0]), h=1 / 32, steps=320)
    assert np.isclose(states[-1, 0], np.exp(-10.0), rtol=2e-6)
    assert np.isclose(times[-1], 10.0)


def test_lyapunov_qr_matches_linear_exponents() -> None:
    matrix = np.diag([-0.3, -1.1])
    rhs = lambda _t, y: matrix @ y
    jac = lambda _t, _y: matrix
    values = lyapunov_exponents(rhs, jac, np.array([1.0, 1.0]), h=0.01,
                                steps=20_000, renormalize_every=10)
    assert np.allclose(np.sort(values), [-1.1, -0.3], atol=2e-4)


def test_drive_phase_poincare_has_expected_period_three() -> None:
    omega = 2.0 * np.pi
    times = np.linspace(0.0, 12.0, 12001)
    phase = 0.25 * np.sin(omega * times / 3.0)
    points = drive_phase_poincare(times, np.column_stack((phase, phase * 0.0)), omega)
    assert points.shape[0] == 12
    assert np.ptp(points[:, 0]) > 0.4
    assert poincare_period_count(points) == 3


def test_poincare_period_count_ignores_unwrapped_phase() -> None:
    points = np.column_stack((np.arange(12) * 4.0 * np.pi / 3.0, np.ones(12)))
    assert poincare_period_count(points) == 3


def test_stroboscopic_period_count_reads_scalar_period_three_sequence() -> None:
    values = np.sin(2.0 * np.pi * np.arange(18) / 3.0)
    assert stroboscopic_period_count(values) == 3


def test_feigenbaum_ratio_definition() -> None:
    assert np.allclose(feigenbaum_ratios([0.0, 1.0, 1.25, 1.3125]), [4.0, 4.0])


def test_period_doubling_extraction_orders_controls_and_brackets_transitions() -> None:
    result = extract_period_doubling_sequence(
        [0.4, 0.1, 0.5, 0.3, 0.2], [4, 1, 8, 2, 1],
    )
    assert result["status"] == "sequence_detected"
    assert result["periods_sorted"] == [1, 1, 2, 4, 8]
    assert np.allclose(result["bifurcation_parameters"], [0.25, 0.35, 0.45])
    assert len(result["feigenbaum_ratios"]) == 1


def test_period_doubling_extraction_rejects_mismatched_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        extract_period_doubling_sequence([0.1, 0.2], [1])


def test_staircase_dimension_uses_current_voltage_geometry() -> None:
    currents = np.linspace(0.0, 1.0, 128)
    voltages = currents.copy()
    dimension = staircase_box_counting_dimension(currents, voltages, scales=[4, 8, 16, 32])
    assert 0.8 < dimension < 1.2


def test_continuation_accepts_explicit_published_currents(tmp_path) -> None:
    args = type("Args", (), {
        "beta": 0.3, "omega": 0.5, "amplitude": 0.8, "step": 1 / 32,
        "currents": "0.26078606,0.26078594,0.26078102",
        "start_current": 1.2, "stop_current": 0.0, "num": 121,
        "transient": 0.0, "averaging": 0.0, "record_stride": 1.0,
        "lyapunov_steps": 0,
    })()
    result = _continuation(args)
    assert [row["I_dc"] for row in result["rows"]] == [0.26078606, 0.26078594, 0.26078102]


def test_fast_cvc_matches_generic_mean_voltage() -> None:
    base = {
        "beta": 0.3, "omega": 0.5, "amplitude": 0.8, "step": 1 / 32,
        "currents": "0.8,0.7,0.6", "start_current": 1.2,
        "stop_current": 0.0, "num": 121, "transient": 10.0,
        "averaging": 20.0, "record_stride": 1.0, "lyapunov_steps": 0,
    }
    generic = _continuation(type("Args", (), {**base, "fast_cvc": False})())
    fast = _continuation(type("Args", (), {**base, "fast_cvc": True})())
    assert np.allclose(
        [row["final_phase"] for row in generic["rows"]],
        [row["final_phase"] for row in fast["rows"]],
        atol=1e-10,
    )
    assert np.allclose(
        [row["final_voltage"] for row in generic["rows"]],
        [row["final_voltage"] for row in fast["rows"]],
        atol=1e-10,
    )
    # The generic path averages recorded samples; the fast path averages every
    # RK4 step, so their finite-window means need only agree at this scale.
    assert np.allclose(
        [row["mean_voltage"] for row in generic["rows"]],
        [row["mean_voltage"] for row in fast["rows"]],
        atol=5e-2,
    )
    assert all(row["poincare_period_count"] in {None, 1, 2, 3, 4, 5, 6, 12} for row in fast["rows"])


def test_fast_cvc_scalar_periods_match_generic_full_step_sections() -> None:
    base = {
        "beta": 0.3, "omega": 0.5, "amplitude": 0.8, "step": 1 / 32,
        "currents": "0.25756981,0.26025001,0.26067300",
        "start_current": 1.2, "stop_current": 0.0, "num": 121,
        "current_step": None, "transient": 100.0, "averaging": 500.0,
        "record_stride": 1 / 32, "lyapunov_steps": 0,
    }
    generic = _continuation(type("Args", (), {**base, "fast_cvc": False})())
    fast = _continuation(type("Args", (), {**base, "fast_cvc": True})())
    assert [row["poincare_period_count"] for row in generic["rows"]] == [3, 6, 6]
    assert [row["poincare_period_count"] for row in fast["rows"]] == [3, 6, 6]


def test_continuation_current_step_builds_descending_grid() -> None:
    args = type("Args", (), {
        "beta": 0.3, "omega": 0.5, "amplitude": 0.8, "step": 1 / 32,
        "currents": None, "current_step": 0.1, "start_current": 0.3,
        "stop_current": 0.1, "num": 121, "transient": 0.0,
        "averaging": 0.0, "record_stride": 1.0, "lyapunov_steps": 0,
        "fast_cvc": False,
    })()
    result = _continuation(args)
    assert [row["I_dc"] for row in result["rows"]] == [0.3, 0.2, 0.1]
