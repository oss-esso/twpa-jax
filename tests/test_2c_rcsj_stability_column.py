"""Contracts for the matched-circuit 2c RCSJ stability column."""

from __future__ import annotations

import math

from scripts import run_2c_rcsj_stability_column as campaign


def test_continuation_grid_preserves_targets_and_limits_power_steps() -> None:
    points = campaign.continuation_currents()
    retained = tuple(current for current, is_target in points if is_target)

    assert retained == campaign.TARGET_CURRENTS_A
    assert campaign.max_continuation_step_db(points) <= 0.25


def test_failed_hb_row_carries_no_physical_diagnostics() -> None:
    row = {
        "status": "ERROR",
        "pump_current_peak_a": campaign.TARGET_CURRENTS_A[0],
        "pump_power_dbm": -26.371,
        "gain_vs_off_db": 99.0,
        "pump_branch_current_max_over_ic": 99.0,
    }

    result = campaign._hb_record(
        "C_HB_RCSJ",
        1.0e5,
        8.2e-5,
        "identity",
        row,
        1.0,
    )

    assert result["status"] == "ERROR"
    assert result["gain_vs_off_db"] is None
    assert result["r_j"] is None
    assert result["production_hb_full_residual_rel"] is None


def test_missing_fixture_sidecar_is_blocking(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    record = {
        "checkpoint_path": str(checkpoint),
        "resistance_ratio": 1.0e5,
    }

    result = campaign.fixture_integrity(record, tmp_path / "circuit")

    assert result == {
        "passed": False,
        "reason": "missing checkpoint or fixture sidecar",
    }


def test_primary_ratio_is_inside_the_prescribed_regularizer_ladder() -> None:
    assert campaign.PRIMARY_RATIO in campaign.RATIOS
    assert all(1.0e4 <= ratio <= 1.0e6 for ratio in campaign.RATIOS)
    assert not any(math.isinf(ratio) for ratio in campaign.RATIOS)
