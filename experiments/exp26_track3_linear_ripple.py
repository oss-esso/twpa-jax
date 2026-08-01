"""Measure the pump-off linear S21 ripple of the 2c circuit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from twpa_solver.signal.passive import db20, passive_s_matrix


ROOT = Path(__file__).resolve().parents[1]
CIRCUIT = ROOT / "outputs" / "ipm_python_design"
OLD_OUTPUT = ROOT / "outputs" / "exp24b_q_axis_slope" / "q01"
OUTPUT = ROOT / "outputs" / "exp26_track3_linear_ripple"
TARGETS_GHZ = np.array([5.800, 6.030, 6.257, 6.490, 6.710, 6.943, 7.290, 7.629])


def load_exp24_gain() -> np.ndarray:
    values = []
    for summary_path in sorted(OLD_OUTPUT.glob("frequency_*/compression_summary.json")):
        summary = json.loads(summary_path.read_text())
        values.append((float(summary["signal_ghz"]), float(summary["small_signal_gain_vs_off_db"])))
    values.sort()
    return np.asarray(values, dtype=float)


def dominant_period_ghz(freq_ghz: np.ndarray, response_db: np.ndarray) -> float:
    centered = response_db - np.mean(response_db)
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(centered.size, d=float(freq_ghz[1] - freq_ghz[0]))
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    return float(1.0 / frequencies[index]) if index else float("nan")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freq_ghz = np.arange(5.5, 9.0001, 0.002)
    s_matrix = passive_s_matrix(CIRCUIT, freq_ghz * 1e9, ports=(1, 2))
    s21_db = db20(s_matrix[:, 1, 0])
    exp24_gain = load_exp24_gain()

    peaks, _ = find_peaks(s21_db, prominence=0.01)
    troughs, _ = find_peaks(-s21_db, prominence=0.01)
    dominant_peaks = peaks[np.argsort(s21_db[peaks])[-10:]] if peaks.size else np.array([], dtype=int)
    dominant_troughs = troughs[np.argsort(s21_db[troughs])[:10]] if troughs.size else np.array([], dtype=int)

    rows = []
    for frequency_ghz in TARGETS_GHZ:
        index = int(np.argmin(np.abs(freq_ghz - frequency_ghz)))
        all_extrema = np.concatenate([dominant_peaks, dominant_troughs])
        nearest = int(all_extrema[np.argmin(np.abs(freq_ghz[all_extrema] - frequency_ghz))]) if all_extrema.size else index
        nearest_kind = "peak" if nearest in set(dominant_peaks.tolist()) else "trough"
        percentile = 100.0 * float(np.mean(s21_db <= s21_db[index]))
        rows.append(
            {
                "frequency_ghz": float(frequency_ghz),
                "s21_pump_off_db": float(s21_db[index]),
                "exp24b_gain_vs_off_db": float(exp24_gain[np.argmin(np.abs(exp24_gain[:, 0] - frequency_ghz)), 1]),
                "ripple_percentile": percentile,
                "nearest_extremum": nearest_kind,
                "nearest_extremum_frequency_ghz": float(freq_ghz[nearest]),
            }
        )

    np.savez_compressed(OUTPUT / "pump_off_s21.npz", frequency_ghz=freq_ghz, s21_db=s21_db)
    with (OUTPUT / "target_locations.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    report = {
        "frequency_start_ghz": float(freq_ghz[0]),
        "frequency_stop_ghz": float(freq_ghz[-1]),
        "step_ghz": float(freq_ghz[1] - freq_ghz[0]),
        "n_points": int(freq_ghz.size),
        "s21_min_db": float(np.min(s21_db)),
        "s21_max_db": float(np.max(s21_db)),
        "s21_peak_to_peak_db": float(np.ptp(s21_db)),
        "dominant_ripple_period_ghz": dominant_period_ghz(freq_ghz, s21_db),
        "target_locations": rows,
    }
    with (OUTPUT / "linear_ripple_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(freq_ghz, s21_db, color="tab:blue", linewidth=0.8, label="pump-off |S21|")
    target_data = np.asarray([(row["frequency_ghz"], row["exp24b_gain_vs_off_db"]) for row in rows])
    axis.scatter(target_data[:, 0], target_data[:, 1], color="tab:red", zorder=4, label="exp24b q≤1 G0")
    for frequency_ghz, gain_db in target_data:
        axis.annotate(f"{frequency_ghz:.3f}", (frequency_ghz, gain_db), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    axis.set(xlabel="Frequency (GHz)", ylabel="Magnitude (dB)", title="2c pump-off linear ripple and exp24b gain samples")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "linear_ripple.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
