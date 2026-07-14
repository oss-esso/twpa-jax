from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from twpa_solver import InsertionLossModel, default_loss_model
from twpa_solver.loss import LOSS_A10_A_DB, LOSS_A10_B_DB, LOSS_A10_C_DB

CSV_PATH = Path(__file__).resolve().parents[1] / "docs" / "loss_A10.csv"


def test_frozen_coeffs_match_csv_refit() -> None:
    fitted = InsertionLossModel.fit_csv(CSV_PATH)
    assert fitted.c_db == pytest.approx(LOSS_A10_C_DB, abs=1e-6)
    assert fitted.a_db == pytest.approx(LOSS_A10_A_DB, abs=1e-6)
    assert fitted.b_db == pytest.approx(LOSS_A10_B_DB, abs=1e-6)


def test_fit_quality_within_tolerance() -> None:
    raw = np.genfromtxt(str(CSV_PATH), delimiter=",", names=True)
    freq = np.asarray(raw["Frequency_GHz"], dtype=float)
    att = -np.asarray(raw["Insertion_Loss_dB"], dtype=float)
    predicted = default_loss_model().attenuation_db(freq)
    rms = float(np.sqrt(np.mean((predicted - att) ** 2)))
    assert rms < 0.5  # measured fit RMS ~0.37 dB


def test_dc_value_is_offset() -> None:
    model = default_loss_model()
    assert model.attenuation_db(0.0) == pytest.approx(LOSS_A10_C_DB)


def test_pump_band_matches_old_flat_35db() -> None:
    # Old flat attenuation was calibrated in the ~8 GHz pump band.
    assert default_loss_model().attenuation_db(8.0) == pytest.approx(35.4, abs=0.2)


def test_attenuation_is_monotonic_increasing() -> None:
    freqs = np.linspace(0.0, 20.0, 200)
    att = default_loss_model().attenuation_db(freqs)
    assert np.all(np.diff(att) > 0.0)


def test_scalar_returns_float_array_returns_array() -> None:
    model = default_loss_model()
    assert isinstance(model.attenuation_db(5.0), float)
    out = model.attenuation_db(np.array([1.0, 2.0, 3.0]))
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_dbm_to_peak_current_applies_frequency_loss() -> None:
    model = default_loss_model()
    # Higher frequency => more attenuation => smaller on-chip current.
    i_low = model.dbm_to_peak_current_a(0.0, 4.0)
    i_high = model.dbm_to_peak_current_a(0.0, 12.0)
    assert i_low > i_high > 0.0

    # Matches the explicit sqrt(2 P / Z0) convention at a given frequency.
    freq_ghz = 8.0
    att_db = model.attenuation_db(freq_ghz)
    power_w = 1.0e-3 * 10.0 ** ((0.0 - att_db) / 10.0)
    expected = math.sqrt(2.0 * power_w / 50.0)
    assert model.dbm_to_peak_current_a(0.0, freq_ghz) == pytest.approx(expected)


def test_negative_frequency_rejected() -> None:
    with pytest.raises(ValueError):
        default_loss_model().attenuation_db(-1.0)
