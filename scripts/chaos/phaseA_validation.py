"""Validate the Phase A classifier on synthetic traces and saved Phase 2 data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.chaos.attractor_classify import (
    classify_trace,
    period_multiple,
    poincare_crossings,
    _period_clusters,
    fourier_map,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "chaos" / "phaseA"


def synthetic_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    pump_hz = 1.0
    time = np.linspace(0.0, 64.0, 64 * 256 + 1)
    cases = []
    for multiple in (1, 2, 4, 8):
        voltage = np.sin(2.0 * np.pi * time / multiple)
        if multiple > 1:
            voltage += 0.15 * np.sin(2.0 * np.pi * time)
        cases.append((f"period-{multiple}", time, voltage))
    return cases


def make_validation_plot() -> Path:
    cases = synthetic_cases()
    figure, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    for column, (name, time, voltage) in enumerate(cases):
        classification = classify_trace(time, voltage, drive_hz=1.0)
        spectrum = fourier_map(time, voltage, fmax_hz=4.0)
        points = poincare_crossings(time, voltage)
        for period_line in np.arange(0.5, 4.01, 0.5):
            axes[1, column].axvline(
                period_line, color="0.65", linewidth=0.6, linestyle=":"
            )
        display = time <= 4.0
        axes[0, column].plot(time[display], voltage[display], linewidth=0.8)
        axes[0, column].set_title(
            f"{name}\nperiod_multiple={classification.period_multiple}, "
            f"{classification.verdict}"
        )
        axes[0, column].set_ylabel("Voltage (normalised)")
        axes[0, column].set_xlabel("Time (pump periods)")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].plot(
            spectrum["frequency_hz"], spectrum["amplitude"], linewidth=0.8
        )
        axes[1, column].set_xlim(0.0, 4.0)
        axes[1, column].set_ylabel("Amplitude (normalised)")
        axes[1, column].set_xlabel("Frequency / pump frequency")
        axes[1, column].grid(alpha=0.2)
        axes[2, column].plot(
            np.arange(points.size), points, ".", markersize=2.5,
        )
        axes[2, column].set_ylabel("Poincare $V'_{PS}$")
        axes[2, column].set_xlabel("Crossing index")
        axes[2, column].grid(alpha=0.2)

    period8_points = np.array([0.0, 0.2, 0.27, 0.298, 0.309, 0.3134, 0.3152, 0.316])
    without_decay, centers_without = _period_clusters(
        period8_points, tolerance=0.03, tolerance_decay=1.0
    )
    with_decay, centers_with = _period_clusters(
        period8_points, tolerance=0.03, tolerance_decay=2.503
    )
    axes[2, 3].plot(
        np.arange(centers_without.size), centers_without, "x",
        color="tab:red", markersize=7, label=f"without decay: {without_decay}"
    )
    axes[2, 3].plot(
        np.arange(centers_with.size), centers_with, "o", fillstyle="none",
        color="tab:green", markersize=6, label=f"with decay: {with_decay}"
    )
    axes[2, 3].legend(fontsize=8)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "classifier_validation.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def reduce_saved_traces() -> Path:
    rows: list[dict[str, object]] = []
    phase2 = ROOT / "outputs" / "chaos" / "phase2"
    for path in sorted(phase2.rglob("timeseries.npz")):
        data = np.load(path, allow_pickle=False)
        keys = set(data.files)
        time_key = "t" if "t" in keys else "theta" if "theta" in keys else "t_s"
        time = np.asarray(data[time_key], dtype=float)
        voltage_key = "vout_v" if "vout_v" in keys else "output_voltage_v"
        if voltage_key not in keys:
            continue
        voltage = np.asarray(data[voltage_key], dtype=float)
        result = classify_trace(time, voltage, drive_hz=7.0e9)
        rows.append({"trace": str(path.relative_to(ROOT)), **result.as_dict()})
    payload = {"trace_count": len(rows), "rows": rows}
    path = OUT / "phase2_trace_classification.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    figure = make_validation_plot()
    report = reduce_saved_traces()
    print(json.dumps({"figure": str(figure), "report": str(report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
