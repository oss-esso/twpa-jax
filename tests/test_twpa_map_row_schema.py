from __future__ import annotations

from twpa_solver_old.experiments.run_ipm_25x25_gain_map import REQUIRED_PARITY_ROW_FIELDS


def test_parity_row_schema_contains_required_fields() -> None:
    required = set(REQUIRED_PARITY_ROW_FIELDS)
    assert "old_julia_parity_mode" in required
    assert "external_power_dbm" in required
    assert "source_power_dbm" in required
    assert "pump_current_a" in required
    assert "historical_target_cells_or_junctions" in required
