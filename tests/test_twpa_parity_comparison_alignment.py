from __future__ import annotations

from twpa_solver.experiments.compare_python_to_old_julia_map import compare_rows


def test_parity_comparison_aligns_by_frequency_and_external_power() -> None:
    old_rows = [
        {
            "pump_frequency_ghz": "6.0",
            "external_power_dbm": "-28.0",
            "gain_db_max": "1.5",
            "status": "valid_converged",
            "source_power_dbm": "-60.0",
            "pump_current_ua": "6.0",
        }
    ]
    python_rows = [
        {
            "pump_frequency_ghz": "6.0",
            "external_power_dbm": "-28.0",
            "signal_gain_db": "1.0",
            "status": "converged",
            "success": "True",
            "source_power_dbm": "-60.0",
            "pump_current_a": "0.000006",
        }
    ]
    rows, summary = compare_rows(old_rows, python_rows)
    assert summary["aligned_cells"] == 1
    assert rows[0]["gain_difference_db"] == -0.5
    assert summary["source_power_mismatch_max_db"] == 0.0
