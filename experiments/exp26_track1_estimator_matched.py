"""Compare refined and measurement-style P1dB estimators on exp24b curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.signal import savgol_filter


FREQUENCIES = (5.8, 6.03, 6.257, 6.49, 6.71, 6.943, 7.29, 7.629)


def crossing(power: np.ndarray, gain: np.ndarray, window: int) -> tuple[float | None, float]:
    """Return the last measured-style threshold crossing and G0."""
    smooth = savgol_filter(gain, window, 2)
    plateau = smooth[power < -120.0]
    if plateau.size == 0:
        plateau = smooth[: min(7, smooth.size)]
    g0 = float(np.median(plateau))
    above = np.flatnonzero(smooth >= g0 - 1.0)
    if above.size == 0 or int(above[-1]) >= smooth.size - 1:
        return None, g0
    index = int(above[-1])
    if smooth[index + 1] == smooth[index]:
        return None, g0
    result = power[index] + (g0 - 1.0 - smooth[index]) * (
        power[index + 1] - power[index]
    ) / (smooth[index + 1] - smooth[index])
    return float(result), g0


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Fit a slope and its standard error."""
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    sxx = float(np.sum((x - x.mean()) ** 2))
    standard_error = math.sqrt(float(np.sum(residual**2)) / max(x.size - 2, 1) / sxx)
    return {
        "n": int(x.size),
        "slope_db_per_db": float(slope),
        "slope_se_db_per_db": standard_error,
        "intercept_dbm": float(intercept),
        "fit_rms_db": float(np.sqrt(np.mean(residual**2))),
    }


def read_case(root: Path, frequency: float, window: int) -> dict[str, object]:
    """Read one curve and extract the measurement-style crossing."""
    case = root / "q01" / f"frequency_{frequency:.6f}ghz"
    with (case / "compression_points.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    power = np.asarray([float(row["signal_power_dbm"]) for row in rows])
    gain = np.asarray([float(row["gain_db"]) for row in rows])
    measured_p1db, g0 = crossing(power, gain, window)
    summary = json.loads((case / "compression_summary.json").read_text(encoding="utf-8"))
    return {
        "signal_ghz": frequency,
        "window": window,
        "g0_measurement_pipeline_db": g0,
        "p1db_refined_dbm": float(summary["p1db_input_dbm"]),
        "p1db_measurement_pipeline_dbm": measured_p1db,
        "difference_measurement_minus_refined_db": (
            None if measured_p1db is None else measured_p1db - float(summary["p1db_input_dbm"])
        ),
        "gain_refined_db": float(summary["small_signal_gain_vs_off_db"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/exp24b_q_axis_slope"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp26_track1_estimator_matched"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for window in (11, 3):
        rows.extend(read_case(args.input_dir, frequency, window) for frequency in FREQUENCIES)
    with (args.output_dir / "estimator_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fits: dict[str, object] = {}
    refined = [row for row in rows if int(row["window"]) == 11]
    x_refined = np.asarray([float(row["gain_refined_db"]) for row in refined])
    y_refined = np.asarray([float(row["p1db_refined_dbm"]) for row in refined])
    fits["refined"] = fit(x_refined, y_refined)
    for window in (11, 3):
        selected = [row for row in rows if int(row["window"]) == window and row["p1db_measurement_pipeline_dbm"] is not None]
        x = np.asarray([float(row["g0_measurement_pipeline_db"]) for row in selected])
        y = np.asarray([float(row["p1db_measurement_pipeline_dbm"]) for row in selected])
        fits[f"measurement_pipeline_sg{window}"] = fit(x, y)
    report = {
        "fits": fits,
        "grid_step_db": 16.0 / 3.0,
        "sg11_span_db": 10.0 * 16.0 / 3.0,
        "note": "SG(11) spans about 53.3 dB on the model grid; SG(3) is also reported as a local-knee comparison.",
    }
    (args.output_dir / "estimator_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
