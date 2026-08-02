"""Emit the Task B gain-ridge CSV and heatmap from a serial HB JSON run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    rows = []
    for source in sorted(Path("references").glob("le_gal_2025_gain_ridge_m*.json")):
        rows.extend(json.loads(source.read_text(encoding="utf-8")))
    fields = ["pump_dBm", "signal_GHz", "gain_vs_off_db", "status", "residual_norm", "nonlinear_pump_phase_rad"]
    with Path("references/le_gal_2025_gain_ridge.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "TIMEOUT") for field in fields})
    pumps = sorted({float(row["pump_dBm"]) for row in rows})
    frequencies = sorted({float(row["signal_GHz"]) for row in rows})
    matrix = []
    for pump in pumps:
        matrix.append([
            next((float(row["gain_vs_off_db"]) for row in rows
                  if float(row["pump_dBm"]) == pump and float(row["signal_GHz"]) == frequency
                  and row["status"] == "SOLVED"), float("nan"))
            for frequency in frequencies
        ])
    figure, axis = plt.subplots(figsize=(10, 4.8))
    image = axis.imshow(matrix, aspect="auto", origin="lower", extent=(frequencies[0] - 0.05, frequencies[-1] + 0.05, pumps[0] - 0.5, pumps[-1] + 0.5))
    axis.set(xlabel="Signal frequency (GHz)", ylabel="Pump power (dBm)", title="Le Gal HB gain ridge")
    figure.colorbar(image, ax=axis, label="Gain vs off (dB)")
    figure.tight_layout()
    figure.savefig("references/le_gal_2025_gain_ridge.png", dpi=160)


if __name__ == "__main__":
    main()
