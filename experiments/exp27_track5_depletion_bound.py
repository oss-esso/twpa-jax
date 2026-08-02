"""Compare observed pump-port depletion with the simple depletion requirement."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "exp24b_q_axis_slope" / "q01"
OUTPUT = ROOT / "outputs" / "exp27_track5_depletion_bound"
Z0 = 50.0


def read_frequency(path: Path) -> tuple[dict[str, object], list[dict[str, float | str]]]:
    summary = json.loads((path / "compression_summary.json").read_text(encoding="utf-8"))
    with (path / "compression_points.csv").open(newline="", encoding="utf-8") as handle:
        points = [
            {
                key: value if key in {"status", "recovery_rung"} else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]
    return summary, points


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    p1_rows = []
    for path in sorted(INPUT.glob("frequency_*/")):
        summary, points = read_frequency(path)
        pump_input = float(summary["pump_current_a"]) ** 2 * Z0 / 2.0
        pump_reference_s = complex(
            float(summary["pump_reference_s21_real"]),
            float(summary["pump_reference_s21_imag"]),
        )
        pump_port_reference = abs(pump_reference_s) ** 2 * pump_input
        gain_linear = 10.0 ** (float(summary["small_signal_gain_vs_off_db"]) / 10.0)
        frequency = float(summary["signal_ghz"])
        for point in points:
            signal_power = float(point["signal_current_a"]) ** 2 * Z0 / 2.0
            required = 10.0 * math.log10(
                1.0 + 2.0 * gain_linear * signal_power / max(pump_port_reference, 1e-300)
            )
            observed = float(point["pump_depletion_db"])
            row = {
                "frequency_ghz": frequency,
                "signal_power_dbm": float(point["signal_power_dbm"]),
                "gain_db": float(point["gain_vs_off_db"]),
                "compression_db": float(point["compression_db"]),
                "observed_pump_depletion_db_port2": observed,
                "required_depletion_db": required,
                "observed_minus_required_db": observed - required,
                "converted_fraction_relative_port2_pump": 1.0 - 10.0 ** (observed / 10.0),
                "pump_port_reference_power_w": pump_port_reference,
                "status": str(point["status"]),
            }
            rows.append(row)
            if row["compression_db"] >= 1.0 and not any(
                abs(float(existing["frequency_ghz"]) - frequency) < 1e-9 for existing in p1_rows
            ):
                p1_rows.append(row)
    rows.sort(key=lambda row: (row["frequency_ghz"], row["signal_power_dbm"]))
    p1_rows.sort(key=lambda row: row["frequency_ghz"])

    with (OUTPUT / "depletion_points.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "pump_measurement_port": 2,
        "pump_port_fraction_of_total_from_prior_circuit_check": 0.091,
        "p1db_grid_rows": p1_rows,
        "shortfall_db_min": float(min(row["observed_minus_required_db"] for row in rows)),
        "shortfall_db_max": float(max(row["observed_minus_required_db"] for row in rows)),
        "shortfall_db_at_p1db_mean": float(np.mean([row["observed_minus_required_db"] for row in p1_rows])),
        "shortfall_db_at_p1db_by_frequency": {
            f"{row['frequency_ghz']:.3f}": row["observed_minus_required_db"] for row in p1_rows
        },
        "interpretation": (
            "The available exp24b pump depletion is measured at port 2, which carries "
            "only 9.1 percent of the pump. Total-pump depletion cannot be inferred "
            "from these artifacts without rerunning the observable at port 3."
        ),
    }
    (OUTPUT / "depletion_bound_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    gain = np.asarray([row["gain_db"] for row in rows])
    shortfall = np.asarray([row["observed_minus_required_db"] for row in rows])
    compression = np.asarray([row["compression_db"] for row in rows])
    axes[0].scatter(gain, shortfall, c=compression, s=10, cmap="viridis")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Observed gain (dB)", ylabel="Observed − required depletion (dB)")
    axes[1].scatter(compression, shortfall, c=gain, s=10, cmap="plasma")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Compression (dB)", ylabel="Observed − required depletion (dB)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle("2c depletion requirement versus port-2 observation")
    figure.tight_layout()
    figure.savefig(OUTPUT / "depletion_bound.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
