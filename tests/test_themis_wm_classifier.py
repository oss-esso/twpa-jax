from __future__ import annotations

import numpy as np

from scripts.chaos.themis_wm_classifier import (
    classify_track, fit_gain_divergence, load_cube, peak_track,
)


def test_cube_loader_and_peak_tracking(tmp_path) -> None:
    frequency = np.linspace(4e9, 12e9, 2001)
    powers = np.linspace(-30.0, -20.0, 11)
    response = np.full((powers.size, frequency.size), -20.0)
    target = int(np.argmin(abs(frequency - 11.85e9)))
    for row in range(8):
        response[row, target] = 10.0 + row
    response[8:, target] = 2.0
    path = tmp_path / "7.9GHz_response.npy"
    np.save(path, {"Frequency": frequency, "Response": response,
                   "PumpPower": powers, "SignalPower": -100.0})
    cube = load_cube(path)
    track = peak_track(cube)
    assert track.collapse_index == 7
    assert np.allclose(track.peak_frequency_hz, frequency[target])
    assert classify_track(track) == "PERIOD_DOUBLING"


def test_gain_divergence_fit_recovers_inverse_square_scaling(tmp_path) -> None:
    frequency = np.linspace(4e9, 12e9, 101)
    powers = np.linspace(-10.0, -1.0, 10)
    response = np.full((powers.size, frequency.size), -30.0)
    target = 50
    distance = -1.0 - powers
    response[:, target] = 10.0 * np.log10(100.0 / np.maximum(distance, 0.1)**2)
    path = tmp_path / "7.9GHz_scaling.npy"
    np.save(path, {"Frequency": frequency, "Response": response,
                   "PumpPower": powers, "SignalPower": -30.0})
    cube = load_cube(path)
    fit = fit_gain_divergence(cube)
    assert fit.status == "OK"
    assert np.isclose(fit.exponent, 2.0, atol=1e-10)
    assert fit.r_squared > 0.999999
