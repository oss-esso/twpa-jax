from __future__ import annotations

import numpy as np

from twpa_solver.experiments.run_ipm_25x25_gain_map import _row_from_solution_status


def test_nonconverged_pump_status_masks_gain_row() -> None:
    row = _row_from_solution_status(
        base={
            "signal_gain_db": 1.0,
            "idler_gain_db": 2.0,
        },
        success=False,
        status="diagnostic",
    )
    assert row["status"] == "diagnostic"
    assert np.isnan(row["signal_gain_db"])
    assert np.isnan(row["idler_gain_db"])
