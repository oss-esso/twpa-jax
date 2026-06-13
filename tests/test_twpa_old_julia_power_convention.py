from __future__ import annotations

from argparse import Namespace

import numpy as np

from twpa_solver.experiments.run_ipm_25x25_gain_map import _compute_drive_convention
from twpa_solver.model.units import dbm_to_old_julia_peak_current


def test_old_julia_offset_sets_source_power_and_peak_current() -> None:
    args = Namespace(
        topology="ipm_jtwpa_old_julia_parity",
        use_old_julia_power_offset="true",
        power_offset_db=32.0,
        old_port_convention="true",
        z0=50.0,
        pump_current_coupling=1e-3,
    )
    drive = _compute_drive_convention(-28.0, args)
    assert drive["source_power_dbm"] == -60.0
    np.testing.assert_allclose(
        drive["pump_current_a"],
        dbm_to_old_julia_peak_current(-60.0, 50.0),
    )
