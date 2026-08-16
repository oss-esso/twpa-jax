from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_td_mode_gate import analyze_trace


def test_mode_gate_reports_half_pump_bin_and_sideband_peaks() -> None:
    pump_frequency = 7.9
    periods = 512
    samples_per_period = 128
    theta = np.arange(periods * samples_per_period) * 2.0 * np.pi / samples_per_period
    modulation = 0.06
    voltage = np.cos(theta) + 0.2 * np.cos(
        (1.0 + modulation / pump_frequency) * theta
    )

    result = analyze_trace(theta, voltage, pump_frequency)

    assert result["sample_count"] == periods * samples_per_period
    assert result["frequency_resolution_ghz"] == pytest.approx(
        pump_frequency / periods
    )
    assert result["half_pump_frequency_ghz"] == pytest.approx(3.95)
    assert result["voltage_peak_v"] > 1.0
    assert result["usable"] is True
    assert result["peaks"]


def test_mode_gate_rejects_an_empty_voltage_trace() -> None:
    theta = np.arange(32, dtype=float) * 2.0 * np.pi / 128.0

    result = analyze_trace(theta, np.zeros_like(theta), 7.9)

    assert result["usable"] is False
    assert result["peaks"] == []
