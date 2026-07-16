"""Tests for scripts/align_map_to_measurement.py.

Verify the (df, dP, dG) shift fit recovers a known synthetic offset and that the
ROI weighting / NaN masking behave. Uses synthetic maps built from a smooth
Gaussian gain ridge so the fit has a well-defined minimum.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import align_map_to_measurement as amm  # noqa: E402


def _ridge(ff: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """A smooth 2-D gain ridge: peak on a diagonal band in (freq, power)."""
    return 20.0 * np.exp(-((ff - 7.0) ** 2) / 0.4 - ((pp + 24.0) ** 2) / 6.0)


def _make_maps(df_true: float, dP_true: float, dG_true: float):
    """Sim on a fine grid; measurement = sim shifted by (df, dP) + dG."""
    sim_f = np.linspace(6.0, 8.5, 60)
    sim_p = np.linspace(-32.0, -19.0, 80)
    sf, sp = np.meshgrid(sim_f, sim_p, indexing="ij")
    sim_g = _ridge(sf, sp)

    meas_f = np.linspace(6.2, 8.0, 40)
    meas_p = np.linspace(-30.0, -20.0, 30)
    mf, mp = np.meshgrid(meas_f, meas_p, indexing="ij")
    # G_meas(f,P) = G_sim(f - df, P - dP) + dG  -> ridge sampled at shifted coords.
    meas_g = _ridge(mf - df_true, mp - dP_true) + dG_true

    sim = {"pump_freq_ghz": sim_f, "pump_power_dbm": sim_p, "gain_db": sim_g}
    meas = {"pump_freq_ghz": meas_f, "pump_power_dbm": meas_p,
            "peak_gain_db": meas_g}
    return meas, sim


def test_recovers_known_shift_l2() -> None:
    df_true, dP_true, dG_true = 0.30, -2.0, 1.5
    meas, sim = _make_maps(df_true, dP_true, dG_true)
    fit = amm.align_maps(meas, sim, coarse_freq_step=0.05, coarse_power_step=0.25)
    assert abs(fit["freq_shift_ghz"] - df_true) < 0.06
    assert abs(fit["power_shift_db"] - dP_true) < 0.3
    assert abs(fit["gain_offset_db"] - dG_true) < 0.3
    assert fit["rmse_db"] < 0.5


def test_recovers_zero_shift() -> None:
    meas, sim = _make_maps(0.0, 0.0, 0.0)
    fit = amm.align_maps(meas, sim, coarse_freq_step=0.05, coarse_power_step=0.25)
    assert abs(fit["freq_shift_ghz"]) < 0.06
    assert abs(fit["power_shift_db"]) < 0.3
    assert fit["rmse_db"] < 0.5


def test_huber_matches_l2_on_clean_data() -> None:
    df_true, dP_true = 0.20, -1.0
    meas, sim = _make_maps(df_true, dP_true, 0.0)
    fit = amm.align_maps(meas, sim, loss="huber", huber_delta=2.0,
                         coarse_freq_step=0.05, coarse_power_step=0.25)
    assert abs(fit["freq_shift_ghz"] - df_true) < 0.06
    assert abs(fit["power_shift_db"] - dP_true) < 0.3


def test_huber_robust_to_outliers() -> None:
    df_true, dP_true = 0.20, -1.0
    meas, sim = _make_maps(df_true, dP_true, 0.0)
    # Inject a few large measurement glitches in the ROI.
    g = meas["peak_gain_db"]
    g[10, 12] += 60.0
    g[20, 8] -= 50.0
    fit_l2 = amm.align_maps(meas, sim, loss="l2",
                            coarse_freq_step=0.05, coarse_power_step=0.25)
    fit_h = amm.align_maps(meas, sim, loss="huber", huber_delta=2.0,
                           coarse_freq_step=0.05, coarse_power_step=0.25)
    err_l2 = abs(fit_l2["freq_shift_ghz"] - df_true) + \
        abs(fit_l2["power_shift_db"] - dP_true) / 10.0
    err_h = abs(fit_h["freq_shift_ghz"] - df_true) + \
        abs(fit_h["power_shift_db"] - dP_true) / 10.0
    assert err_h <= err_l2 + 1e-9


def test_nan_sim_cells_are_masked() -> None:
    meas, sim = _make_maps(0.1, -0.5, 0.0)
    sim["gain_db"][:, :20] = np.nan  # kill low-power sim cells
    fit = amm.align_maps(meas, sim, coarse_freq_step=0.05, coarse_power_step=0.25)
    assert fit["overlap_cells"] > 0
    assert np.isfinite(fit["rmse_db"])
    # Residual is NaN exactly where the fit had no valid overlap.
    assert np.isnan(fit["residual_db"]).any()


def test_roi_floor_downweights_background() -> None:
    meas, sim = _make_maps(0.25, -1.5, 0.0)
    # A large flat mismatch far outside the ridge should barely move the fit
    # because the background weight is tiny.
    meas["peak_gain_db"][0, :] += 5.0  # low-freq edge, ~zero-gain background
    fit = amm.align_maps(meas, sim, roi_floor=0.01,
                         coarse_freq_step=0.05, coarse_power_step=0.25)
    assert abs(fit["freq_shift_ghz"] - 0.25) < 0.1
    assert abs(fit["power_shift_db"] - (-1.5)) < 0.5


def test_fit_window_restricts_and_recovers() -> None:
    df_true, dP_true = 0.30, -2.0
    meas, sim = _make_maps(df_true, dP_true, 0.0)
    # Fit only a freq+power window around the ridge; still recovers the shift.
    fit = amm.align_maps(
        meas, sim, fit_freq_range=(6.6, 7.4), fit_power_range=(-27.0, -21.0),
        coarse_freq_step=0.05, coarse_power_step=0.25,
    )
    assert abs(fit["freq_shift_ghz"] - df_true) < 0.1
    assert abs(fit["power_shift_db"] - dP_true) < 0.5
    # Cells outside the window contribute no residual (weight 0 -> masked NaN).
    f = meas["pump_freq_ghz"]
    outside = (f < 6.6) | (f > 7.4)
    assert np.isnan(fit["residual_db"][outside, :]).all()


def test_fit_window_too_small_raises() -> None:
    meas, sim = _make_maps(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="fewer than 8"):
        amm.align_maps(meas, sim, fit_freq_range=(6.99, 7.0),
                       fit_power_range=(-24.01, -24.0))


def test_overlap_guard_rejects_tiny_corner() -> None:
    # Sim covers only a narrow high-freq strip; a shift that slides the fit
    # window onto that strip has tiny overlap and must not win.
    meas, sim = _make_maps(0.0, 0.0, 0.0)
    sim["gain_db"][:55, :] = np.nan  # keep only the top ~5 freq rows finite
    fit = amm.align_maps(meas, sim, fit_freq_range=(6.4, 7.6),
                         min_overlap_frac=0.5,
                         coarse_freq_step=0.05, coarse_power_step=0.25)
    # With almost no overlap available, no shift clears the 50% floor -> inf.
    assert not np.isfinite(fit["score"])


def test_load_sim_map_transposes(tmp_path: Path) -> None:
    d = tmp_path / "map"
    d.mkdir()
    power = np.linspace(-32, -19, 5)
    freq = np.linspace(6.0, 8.5, 4)
    gain = np.arange(20, dtype=float).reshape(5, 4)  # (power, freq)
    np.savez(d / "map_arrays.npz", pump_power_dbm=power,
             pump_frequency_ghz=freq, gain_db_warm=gain)
    out = amm.load_sim_map(d)
    assert out["gain_db"].shape == (4, 5)  # (freq, power)
    assert np.allclose(out["gain_db"], gain.T)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
