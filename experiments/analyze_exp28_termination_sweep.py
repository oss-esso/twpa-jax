"""Fit the corrected-termination 2c pump-swept slope once the sweep completes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORRECTED = ROOT / "outputs" / "exp28_2c_termination_43ohm_pump_sweep"
ORIGINAL = ROOT / "outputs" / "exp28_controlled_pump_sweep" / "2c_7p4ghz"


def fit(summary: dict[str, object]) -> dict[str, object]:
    rows = [
        row for row in summary.get("results", [])
        if row.get("status") == "VALID_SOLVED"
        and row.get("p1db_input_dbm") is not None
        and row.get("small_signal_gain_db") is not None
    ]
    gain = np.asarray([float(row["small_signal_gain_db"]) for row in rows])
    p1db = np.asarray([float(row["p1db_input_dbm"]) for row in rows])
    if len(rows) < 3:
        return {"n_valid": len(rows), "status": "INSUFFICIENT_POINTS"}
    slope, intercept = np.polyfit(gain, p1db, 1)
    residual = p1db - (slope * gain + intercept)
    se = math.sqrt(float(np.sum(residual**2) / (len(rows) - 2)) / float(np.sum((gain - np.mean(gain)) ** 2)))
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum((p1db - np.mean(p1db)) ** 2))
    return {
        "status": "VALID_FIT",
        "n_valid": len(rows),
        "gain_span_db": float(np.ptp(gain)),
        "slope_db_per_db": float(slope),
        "slope_se_db_per_db": float(se),
        "r_squared": r_squared,
        "gain_db": gain.tolist(),
        "p1db_input_dbm": p1db.tolist(),
    }


def fit_gain_window(summary: dict[str, object], minimum_gain_db: float) -> dict[str, object]:
    filtered = {
        **summary,
        "results": [
            row
            for row in summary.get("results", [])
            if row.get("small_signal_gain_db") is not None
            and float(row["small_signal_gain_db"]) > minimum_gain_db
        ],
    }
    return fit(filtered)


def main() -> None:
    corrected_summary = json.loads((CORRECTED / "pump_sweep_summary.json").read_text(encoding="utf-8"))
    original_summary = json.loads((ORIGINAL / "pump_sweep_summary.json").read_text(encoding="utf-8"))
    corrected_fit = fit(corrected_summary)
    original_fit = fit(original_summary)
    report = {
        "device": "2c",
        "signal_ghz": 7.4,
        "termination_ohm": 43.32750543560899,
        "corrected": corrected_fit,
        "original_termination_label": "production 50 ohm",
        "original_circuit_source": "outputs\\ipm_python_design",
        "original": original_fit,
        "matched_gain_window_db": ">11",
        "corrected_matched_gain_window": fit_gain_window(corrected_summary, 11.0),
        "original_matched_gain_window": fit_gain_window(original_summary, 11.0),
    }
    (CORRECTED / "termination_slope_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (CORRECTED / "pump_swept_slope.json").write_text(json.dumps({**corrected_fit, "device": "2c", "signal_ghz": 7.4, "pump_swept": True}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
