import json
import math

from twpa_solver_old.model.units import dbm_to_old_julia_peak_current


def test_old_power_convention_from_exported_json(tmp_path):
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "source_convention": {
                    "power_offset_db": 32.0,
                    "source_power_dbm_formula": "external_power_dbm - power_offset_db",
                    "pump_current_a_formula": "sqrt(2 * source_power_W / 50)",
                }
            }
        ),
        encoding="utf-8",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    source_power_dbm = -28.0 - data["source_convention"]["power_offset_db"]
    current = dbm_to_old_julia_peak_current(source_power_dbm)
    expected = math.sqrt(2.0 * (1e-3 * 10 ** (-60.0 / 10.0)) / 50.0)
    assert source_power_dbm == -60.0
    assert math.isclose(current, expected)
