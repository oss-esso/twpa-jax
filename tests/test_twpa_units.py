from __future__ import annotations

import numpy as np

from twpa_solver_old.model.ports import voltage_current_to_waves
from twpa_solver_old.model.units import current_peak_to_dbm, dbm_to_current_peak


def test_dbm_current_roundtrip() -> None:
    current = dbm_to_current_peak(-20.0, 50.0)
    assert current > 0.0
    np.testing.assert_allclose(current_peak_to_dbm(current, 50.0), -20.0, atol=1e-12)


def test_wave_normalization_matched_load_has_no_reflection() -> None:
    a, b = voltage_current_to_waves(1.0 + 0j, 1.0 / 50.0 + 0j, 50.0)
    assert abs(a) > 0.0
    assert abs(b) < 1e-12
