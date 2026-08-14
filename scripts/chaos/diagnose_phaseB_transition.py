"""Write residual diagnostics for the saved Guarcello transition traces."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.chaos.attractor_classify import _fractional_delay


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "chaos" / "phaseB" / "guarcello"
OUT = RUN / "transition_residual_diagnostics.json"


def residuals(trace_path: Path, pump_hz: float = 7.0e9) -> list[float]:
    data = np.load(trace_path, allow_pickle=False)
    t = np.asarray(data["t"], dtype=float)
    v = np.asarray(data["v_out"], dtype=float)
    start = v.size // 2
    t, v = t[start:], v[start:]
    step = float(np.mean(np.diff(t)))
    denominator = max(float(np.linalg.norm(v - np.mean(v))), np.finfo(float).tiny)
    values = []
    for multiple in range(1, 13):
        shift_samples = multiple / pump_hz / step
        shifted = _fractional_delay(v, shift_samples)
        margin = int(np.ceil(shift_samples)) + 4
        values.append(float(np.linalg.norm(shifted[margin:-margin] - v[margin:-margin]) / denominator))
    return values


def spectral_lines(trace_path: Path, pump_hz: float = 7.0e9) -> dict[str, float]:
    data = np.load(trace_path.parent / "spectrum.npz", allow_pickle=False)
    frequency = np.asarray(data["frequency_hz"], dtype=float)
    level = np.asarray(data["spectrum_db_relative_pump"], dtype=float)
    return {
        "f_p_over_2_db": float(level[np.abs(frequency - 0.5 * pump_hz) <= 80e6].max()),
        "three_f_p_over_2_db": float(level[np.abs(frequency - 1.5 * pump_hz) <= 80e6].max()),
    }


def main() -> int:
    with (RUN / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for target in (-53.553, -52.895):
        row = min(rows, key=lambda item: abs(float(item["pump_power_dbm"]) - target))
        trace = ROOT / row["trace_path"]
        selected[str(target)] = {
            "measured_power_dbm": float(row["pump_power_dbm"]),
            "period_multiple": int(row["period_multiple"]),
            "poincare_clusters": int(row["poincare_clusters"]),
            "verdict": row["verdict"],
            "pump_referred_half_integer_db": spectral_lines(trace),
            "residual_n_1_to_12": residuals(trace),
        }
    payload = {
        "traces": selected,
        "alias_check": {
            "status": "NOT_AVAILABLE",
            "reason": "Phase B persisted only record_stride=20 traces; no smaller-stride raw trace exists",
            "sample_rate_ghz": 43.6,
            "nyquist_ghz": 21.8,
            "four_fp_alias_ghz": 15.6,
            "five_fp_alias_ghz": 8.6,
            "six_fp_alias_ghz": 1.6,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
