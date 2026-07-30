from __future__ import annotations

import pytest

from references.le_gal_2025_gain_compression.cme import depletion_only_gain


def test_paper_depletion_model_p1db() -> None:
    gain_linear = 100.0
    pump_dbm = -78.4
    pump_w = 10.0 ** ((pump_dbm - 30.0) / 10.0)
    expected_dbm = pump_dbm + 10.0 * __import__("math").log10(
        (10.0**0.1 - 1.0) / (2.0 * gain_linear)
    )
    signal_dbm = -107.3
    signal_w = 10.0 ** ((signal_dbm - 30.0) / 10.0)
    gain_at_point = depletion_only_gain(gain_linear, signal_w, pump_w)
    assert expected_dbm == pytest.approx(-107.3, abs=0.2)
    assert 10.0 * __import__("math").log10(gain_at_point / gain_linear) == pytest.approx(-1.0, abs=0.15)
