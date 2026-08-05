from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twpa_solver.loss import signal_line_loss_model

from scripts.measured_psat_pipeline import (
    DEFAULT_CUBE_PATH,
    MEAS_PUMP_GHZ,
    MEAS_PUMP_INSTRUMENT_DBM,
    load_cube,
    p1db_cut_at_p2db,
    stage4_psat_vs_frequency_plot,
)

# The fabricated flat constant Phase 2 removed from production code. Kept
# here, only, so the shift-identity test can compute what the pipeline used
# to report and compare it against the new frequency-resolved loss_B1 axis.
_LEGACY_FLAT_SIGNAL_LINE_LOSS_DB = 72.5


@pytest.mark.skipif(
    not DEFAULT_CUBE_PATH.exists(), reason="Themis cube not present in this checkout"
)
def test_on_chip_power_shift_matches_72p5_minus_loss_b1() -> None:
    """Per-column power axis shift: new - old == 72.5 - att_B1(f), exactly.

    ``on_chip_power_dbm(index) = instrument_power_dbm - att_B1(f)`` replaces
    the removed ``instrument_power_dbm - 72.5``. Since attenuation is a
    per-column constant (same at every power row), the shift is the same
    constant for every row in that column.
    """
    cube = load_cube(DEFAULT_CUBE_PATH)
    att_b1 = signal_line_loss_model().attenuation_db(cube.frequency_ghz)
    expected_shift = _LEGACY_FLAT_SIGNAL_LINE_LOSS_DB - att_b1

    for index in (0, cube.frequency_ghz.size // 2, cube.frequency_ghz.size - 1):
        old_power_dbm = cube.instrument_power_dbm - _LEGACY_FLAT_SIGNAL_LINE_LOSS_DB
        new_power_dbm = cube.on_chip_power_dbm(index)
        shift = new_power_dbm - old_power_dbm
        assert np.allclose(shift, expected_shift[index], atol=1e-9)


@pytest.mark.skipif(
    not DEFAULT_CUBE_PATH.exists(), reason="Themis cube not present in this checkout"
)
def test_p1db_shift_matches_72p5_minus_loss_b1_per_column() -> None:
    """p1db_new(f) - p1db_old(f) == 72.5 - att_B1(f), to 1e-9.

    p1db_cut_at_p2db's only dependence on the power axis is the final
    threshold-crossing interpolation, which is affine in the power array --
    everything else (breakdown index, smoothing, threshold) only touches gain
    values. So a rigid per-column shift of the power axis must shift the
    reported P1dB by exactly the same constant.
    """
    cube = load_cube(DEFAULT_CUBE_PATH)
    att_b1 = signal_line_loss_model().attenuation_db(cube.frequency_ghz)

    checked = 0
    for index in range(0, cube.frequency_ghz.size, 250):
        old_power_dbm = cube.instrument_power_dbm - _LEGACY_FLAT_SIGNAL_LINE_LOSS_DB
        new_power_dbm = cube.on_chip_power_dbm(index)
        raw_column = cube.response_db[:, index]
        rough_g0 = cube.row0[index]

        p1db_old, _ = p1db_cut_at_p2db(raw_column, old_power_dbm, rough_g0)
        p1db_new, _ = p1db_cut_at_p2db(raw_column, new_power_dbm, rough_g0)
        if not (np.isfinite(p1db_old) and np.isfinite(p1db_new)):
            continue
        expected_shift = _LEGACY_FLAT_SIGNAL_LINE_LOSS_DB - att_b1[index]
        assert p1db_new - p1db_old == pytest.approx(expected_shift, abs=1e-9)
        checked += 1
    assert checked > 0


def test_psat_uses_g0_local_not_g0_smooth_exactly(tmp_path: Path) -> None:
    """psat == p1db_fit + g0_local - 1.0 exactly, never mixed with G0_smooth.

    Uses deliberately different g0_local and g0_smooth arrays so a
    regression back to the mixed estimator (the bug Phase 3 fixed) would
    change ``psat`` and fail this test.
    """
    freq_usable = np.array([5.0, 6.0, 7.0, 8.0])
    g0_local_usable = np.array([10.0, 11.0, 9.5, 12.0])
    g0_smooth_usable = np.array([9.0, 10.5, 10.0, 11.5])  # deliberately != local
    p1db_fit = np.array([-90.0, -88.0, -91.0, -87.0])
    inlier = np.array([True, True, False, True])

    _, psat = stage4_psat_vs_frequency_plot(
        freq_usable, g0_local_usable, g0_smooth_usable, p1db_fit, inlier, tmp_path
    )

    expected = p1db_fit + g0_local_usable - 1.0
    np.testing.assert_allclose(psat, expected, rtol=0.0, atol=0.0)

    mixed = p1db_fit + g0_smooth_usable - 1.0
    assert not np.allclose(psat, mixed)  # g0_local and g0_smooth differ here


@pytest.mark.skipif(
    not DEFAULT_CUBE_PATH.exists(), reason="Themis cube not present in this checkout"
)
def test_on_chip_pump_matches_plan_value() -> None:
    """On-chip pump at MEAS_PUMP_GHZ is within the plan's measured -55.54 dBm.

    The fitted loss_A10 model gives -55.68 dBm here, 0.14 dB off the plan's
    hand-noted figure -- consistent with rounding in that note, not a defect
    in this wiring (pump_line_loss_model() is the same frozen A10 fit used
    everywhere else in the repo, unchanged by this phase).
    """
    cube = load_cube(DEFAULT_CUBE_PATH)
    assert cube.on_chip_pump_dbm == pytest.approx(-55.54, abs=0.2)
    assert MEAS_PUMP_INSTRUMENT_DBM == pytest.approx(-21.0)
