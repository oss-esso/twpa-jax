"""Rebuild 2c with prescribed matched port shunts and rerun exp24b q<=1."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "ipm_python_design"
OUTPUT = ROOT / "outputs" / "exp27_track2_matched_ports_corrected"
CIRCUIT = OUTPUT / "ipm_python_design_matched"
Z_MATCH = 84.6
FREQUENCIES = np.array([5.800, 6.030, 6.257, 6.490, 6.710, 6.943, 7.290, 7.629])


def build_circuit() -> None:
    if not CIRCUIT.exists():
        shutil.copytree(SOURCE, CIRCUIT)
    conductance = load_npz(CIRCUIT / "G.npz").tolil()
    with (CIRCUIT / "ipm_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    port_map = summary.get("matrices", {}).get("port_vectors", {})
    if isinstance(port_map, dict) and all(isinstance(value, list) for value in port_map.values()):
        indices = [int(value[0]) for value in port_map.values()]
    else:
        indices = [int(value) for value in port_map.values()]
    g_match = 1.0 / Z_MATCH
    for index in indices:
        conductance[index, index] = g_match
    save_npz(CIRCUIT / "G.npz", conductance.tocsr())
    (CIRCUIT / "exp27_match_metadata.json").write_text(
        json.dumps({"prescribed_z_ohm": Z_MATCH, "port_indices": indices, "g_match_s": g_match}, indent=2),
        encoding="utf-8",
    )


def measure_ripple() -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from twpa_solver.signal.passive import db20, passive_s_matrix

    frequency_ghz = np.arange(5.5, 9.0001, 0.002)
    s_matrix = passive_s_matrix(CIRCUIT, frequency_ghz * 1e9, ports=(1, 2))
    s21_db = db20(s_matrix[:, 1, 0])
    peaks, _ = find_peaks(s21_db, prominence=0.01)
    spectrum = np.abs(np.fft.rfft(s21_db - np.mean(s21_db)))
    spectrum[0] = 0.0
    fft_freq = np.fft.rfftfreq(s21_db.size, frequency_ghz[1] - frequency_ghz[0])
    dominant = int(np.argmax(spectrum))
    report = {
        "frequency_start_ghz": float(frequency_ghz[0]),
        "frequency_stop_ghz": float(frequency_ghz[-1]),
        "step_ghz": float(frequency_ghz[1] - frequency_ghz[0]),
        "s21_min_db": float(np.min(s21_db)),
        "s21_max_db": float(np.max(s21_db)),
        "s21_peak_to_peak_db": float(np.ptp(s21_db)),
        "dominant_ripple_period_ghz": float(1.0 / fft_freq[dominant]),
        "n_local_peaks": int(peaks.size),
    }
    np.savez_compressed(OUTPUT / "matched_pump_off_s21.npz", frequency_ghz=frequency_ghz, s21_db=s21_db)
    return report


def run_frequency(frequency_ghz: float) -> dict[str, object]:
    label = f"frequency_{frequency_ghz:.3f}".replace(".", "p")
    output_dir = OUTPUT / "q01" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", str(CIRCUIT),
        "--pump-freq-ghz", "7.540816326531111",
        "--signal-ghz", f"{frequency_ghz:.3f}",
        "--pump-current-a", "7.231074707853736e-06",
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "dense_real",
        "--pump-harmonics", "6",
        "--pump-nt", "40",
        "--multitone-basis", "lattice",
        "--multitone-sidebands", "1",
        "--source-port", "1",
        "--pump-port", "4",
        "--out-port", "2",
        "--attenuation-db", "0",
        "--factor-backend", "pardiso",
        "--n-signal-power", "16",
        "--signal-current-min-a", "1e-10",
        "--signal-current-max-a", "1e-6",
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600",
        "--allow-memory-overcommit",
    ]
    if abs(frequency_ghz - 7.629) < 1e-6:
        command.extend(["--spatial-profiles", "--save-states", "selected"])
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (output_dir / "run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary_path = output_dir / "compression_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"signal_ghz": float(frequency_ghz), "status": "SUBPROCESS_FAILED", "returncode": completed.returncode}


def fit_slope(summaries: list[dict[str, object]]) -> dict[str, object]:
    valid = [
        summary for summary in summaries
        if summary.get("status") == "VALID_SOLVED"
        and np.isfinite(float(summary.get("small_signal_gain_vs_off_db", np.nan)))
        and np.isfinite(float(summary.get("p1db_input_dbm", np.nan)))
    ]
    gain = np.asarray([float(summary["small_signal_gain_vs_off_db"]) for summary in valid])
    p1db = np.asarray([float(summary["p1db_input_dbm"]) for summary in valid])
    if len(valid) < 2:
        return {"n": len(valid), "slope_db_per_db": None}
    slope, intercept = np.polyfit(gain, p1db, 1)
    residual = p1db - (slope * gain + intercept)
    se = math.sqrt(float(np.sum(residual**2) / (len(valid) - 2)) / float(np.sum((gain - np.mean(gain)) ** 2))) if len(valid) > 2 else None
    return {
        "n": len(valid),
        "slope_db_per_db": float(slope),
        "slope_se_db_per_db": se,
        "intercept_dbm": float(intercept),
        "fit_rms_db": float(np.sqrt(np.mean(residual**2))),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_circuit()
    ripple = measure_ripple()
    summaries = [run_frequency(float(frequency)) for frequency in FREQUENCIES]
    report = {"matched_z_ohm": Z_MATCH, "ripple": ripple, "slope_fit": fit_slope(summaries), "summaries": summaries}
    (OUTPUT / "matched_ports_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    points = [
        (summary.get("small_signal_gain_vs_off_db"), summary.get("p1db_input_dbm"), summary.get("signal_ghz"))
        for summary in summaries
        if summary.get("status") == "VALID_SOLVED"
    ]
    if points:
        gain, p1db, frequency = np.asarray(points, dtype=float).T
        slope = report["slope_fit"].get("slope_db_per_db")
        intercept = report["slope_fit"].get("intercept_dbm")
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter(gain, p1db, label="matched-port q≤1")
        if slope is not None:
            grid = np.linspace(np.min(gain), np.max(gain), 100)
            axis.plot(grid, slope * grid + intercept, label=f"fit {slope:.3f}")
        for x, y, f in zip(gain, p1db, frequency):
            axis.annotate(f"{f:.3f}", (x, y), fontsize=7)
        axis.set(xlabel="Small-signal gain (dB)", ylabel="P1dB input (dBm)", title="Matched-port 2c q≤1 compression")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(OUTPUT / "matched_compression_slope.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
