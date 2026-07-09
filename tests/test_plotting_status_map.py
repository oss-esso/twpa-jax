"""Status-map classification tests for saved gain-map outputs."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from twpa_solver.plotting.candidates import compute_all_fit_metrics
from twpa_solver.plotting.data import MapData
from twpa_solver.plotting.maps import status_label_for_row


def test_status_label_uses_pump_and_gain_diagnostics() -> None:
    assert status_label_for_row({"status": "PASS"}) == "PASS"
    assert status_label_for_row({"status": "ERROR", "pump_status": "FAIL", "gain_status": "ERROR"}) == "PUMP_FAILED"
    assert (
        status_label_for_row(
            {"status": "ERROR", "pump_status": "VALID_CONVERGED", "gain_status": "ERROR"}
        )
        == "GAIN_FAILED"
    )
    assert status_label_for_row({"status": "SKIP_PAST_FOLD"}) == "FOLD_SKIPPED"
    assert status_label_for_row({"status": "INVALID_GAIN:not enough samples"}) == "INVALID_GAIN"


def test_fit_metrics_preserve_solver_status_diagnostics(tmp_path) -> None:
    points = pd.DataFrame(
        [
            {
                "point_index": 0,
                "pump_power_dbm": -28.0,
                "pump_freq_ghz": 8.0,
                "status": "ERROR",
                "pump_status": "FAIL",
                "gain_status": "ERROR",
                "pump_failure_reason": "stalled at Newton 4",
            }
        ]
    )
    data = MapData(run_dir=tmp_path, points=points, arrays={}, spectrum={})
    config = SimpleNamespace(operation_drop_db=3.0, n_dense=200, window_frac=0.05, polyorder=3)

    metrics = compute_all_fit_metrics(data, config)

    assert bool(metrics.loc[0, "valid_fit"]) is False
    assert metrics.loc[0, "pump_status"] == "FAIL"
    assert metrics.loc[0, "gain_status"] == "ERROR"
    assert metrics.loc[0, "pump_failure_reason"] == "stalled at Newton 4"
    assert np.isnan(metrics.loc[0, "peak_gain_db_fit"])
