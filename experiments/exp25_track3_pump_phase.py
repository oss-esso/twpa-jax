"""Measure pump phase motion along the exp24b q<=1 compression curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FREQUENCIES = (5.8, 6.03, 6.257, 6.49, 6.71, 6.943, 7.29, 7.629)


def read_rows(root: Path, frequency: float) -> tuple[list[dict[str, str]], dict[str, object]]:
    case = root / "q01" / f"frequency_{frequency:.6f}ghz"
    with (case / "compression_points.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((case / "compression_summary.json").read_text(encoding="utf-8"))
    return rows, summary


def phase_rows(root: Path, frequency: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return per-power phase/compression rows and the P1dB phase metric."""
    rows, summary = read_rows(root, frequency)
    pump = np.asarray(
        [complex(float(row["pump_s21_real"]), float(row["pump_s21_imag"])) for row in rows]
    )
    phase = np.unwrap(np.angle(pump))
    phase -= phase[0]
    output: list[dict[str, object]] = []
    for row, shift in zip(rows, phase):
        output.append({
            "signal_ghz": frequency,
            "signal_power_dbm": float(row["signal_power_dbm"]),
            "compression_db": float(row["compression_db"]),
            "phase_shift_rad": float(shift),
            "point_status": row["status"],
        })
    powers = np.asarray([float(row["signal_power_dbm"]) for row in rows])
    p1db = float(summary["p1db_input_dbm"])
    p1_phase = float(np.interp(p1db, powers, phase))
    compression = np.asarray([float(row["compression_db"]) for row in rows])
    early = np.abs(phase) > 0.1
    early &= compression < 0.1
    metric = {
        "signal_ghz": frequency,
        "small_signal_gain_db": float(summary["small_signal_gain_vs_off_db"]),
        "p1db_input_dbm": p1db,
        "phase_shift_at_p1db_rad": p1_phase,
        "max_abs_phase_before_0p1db_rad": float(np.max(np.abs(phase[compression < 0.1]))),
        "phase_leads_compression": bool(np.any(early)),
    }
    return output, metric


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/exp24b_q_axis_slope"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp25_track3_pump_phase"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=False)
    for axis, frequency in zip(axes.flat, FREQUENCIES):
        rows, metric = phase_rows(args.input_dir, frequency)
        all_rows.extend(rows)
        metrics.append(metric)
        power = [float(row["signal_power_dbm"]) for row in rows]
        compression = [float(row["compression_db"]) for row in rows]
        phase = [float(row["phase_shift_rad"]) for row in rows]
        axis.plot(power, phase, "o-", label="pump phase shift")
        axis.plot(power, compression, "s--", label="compression (dB)")
        axis.axhline(0.1, color="tab:red", lw=0.8, ls=":")
        axis.set_title(f"{frequency:.3f} GHz")
        axis.set_xlabel("signal power (dBm)")
        axis.set_ylabel("rad / dB")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Exp24b q<=1: pump phase motion versus compression")
    figure.tight_layout()
    figure.savefig(args.output_dir / "pump_phase_vs_compression.png", dpi=170)
    plt.close(figure)
    with (args.output_dir / "pump_phase_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "pump_phase_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    gain = np.asarray([float(row["small_signal_gain_db"]) for row in metrics])
    p1phase = np.asarray([float(row["phase_shift_at_p1db_rad"]) for row in metrics])
    correlation = float(np.corrcoef(gain, p1phase)[0, 1])
    report = {
        "metrics": metrics,
        "p1db_phase_gain_correlation": correlation,
        "any_phase_leads_compression": bool(any(row["phase_leads_compression"] for row in metrics)),
        "lead_threshold_rad": 0.1,
    }
    (args.output_dir / "pump_phase_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
