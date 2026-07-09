import numpy as np

from twpa_solver.plotting.metrics import SpectrumFit, SpectrumFitMetrics
from twpa_solver.plotting.spectrum import _fit_error, _symmetric_error_edges


def test_fit_error_uses_all_finite_signal_samples():
    metrics = SpectrumFitMetrics(
        point_index=0,
        pump_power_dbm=-30.0,
        pump_freq_ghz=8.0,
        status="PASS",
        peak_gain_db_fit=1.0,
        peak_signal_freq_ghz_fit=7.9,
        band_left_ghz_fit=7.8,
        band_right_ghz_fit=8.0,
        bandwidth_ghz_fit=0.2,
        gbp_ghz_fit=0.2,
        gbp_dbghz_fit=0.0,
        ripple_db_fit=0.0,
        smoothness_rms_curvature_fit=0.0,
        smoothness_norm_fit=0.0,
        mean_gain_db_fit=0.0,
        median_gain_db_fit=0.0,
        min_gain_db_fit=0.0,
        score_fit=0.0,
    )
    fit = SpectrumFit(
        freq_ghz=np.array([7.5, 7.75, 8.0, 8.25]),
        gain_db_raw=np.array([0.1, np.nan, -0.1, 0.2]),
        gain_db_smooth=np.zeros(4),
        f_dense_ghz=np.array([7.5, 8.25]),
        g_dense_db=np.array([0.0, 0.0]),
        band_mask_dense=np.array([True, True]),
        metrics=metrics,
    )

    assert _fit_error(fit).size == 3


def test_symmetric_error_edges_use_quarter_db_bins():
    edges = _symmetric_error_edges(np.array([-0.31, 0.12]))

    assert np.isclose(edges[0], -0.5)
    assert np.isclose(edges[-1], 0.5)
    assert np.allclose(np.diff(edges), 0.25)
