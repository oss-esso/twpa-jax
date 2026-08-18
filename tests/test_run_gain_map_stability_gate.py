from __future__ import annotations

import math

from scripts.run_gain_map import apply_stability_gate


def test_stability_gate_is_opt_in_and_labels_only_tracked_unstable_rows() -> None:
    rows = [
        {
            "pump_power_dbm": -29.0,
            "status": "PASS",
            "gain_status": "PASS",
            "gain_db": 20.0,
            "gain_vs_off_db": 20.0,
        },
        {
            "pump_power_dbm": -28.0,
            "status": "PASS",
            "gain_status": "PASS",
            "gain_db": 19.0,
            "gain_vs_off_db": 19.0,
        },
    ]
    payload = {
        "points": [
            {"parameter": -29.0, "stability_verdict": "STABLE"},
            {"parameter": -28.0, "stability_verdict": "UNSTABLE_NS"},
        ]
    }

    gated = apply_stability_gate(rows, payload)

    assert gated[0]["status"] == "PASS"
    assert gated[1]["status"] == "PAST_NS"
    assert math.isnan(gated[1]["gain_db"])
