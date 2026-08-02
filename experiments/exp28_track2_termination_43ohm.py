"""Passive 2c S21 check with the corrected branch-node termination."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "ipm_python_design"
OUTPUT = ROOT / "docs" / "development" / "exp28_2c_termination_43ohm"
CIRCUIT = Path("D:/tmp/exp28_2c_termination_43ohm/ipm_python_design_terminated")
Z_MATCH_OHM = 43.32750543560899


def build_circuit() -> list[int]:
    if not CIRCUIT.exists():
        shutil.copytree(SOURCE, CIRCUIT)
    conductance = load_npz(CIRCUIT / "G.npz").tolil()
    summary = json.loads((CIRCUIT / "ipm_summary.json").read_text(encoding="utf-8"))
    port_map = summary.get("matrices", {}).get("port_vectors", {})
    indices = [int(value[0]) if isinstance(value, list) else int(value) for value in port_map.values()]
    g_match = 1.0 / Z_MATCH_OHM
    for index in indices:
        conductance[index, index] = g_match
    save_npz(CIRCUIT / "G.npz", conductance.tocsr())
    (CIRCUIT / "termination_metadata.json").write_text(
        json.dumps({"termination_ohm": Z_MATCH_OHM, "port_indices": indices, "conductance_s": g_match}, indent=2),
        encoding="utf-8",
    )
    return indices


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    indices = build_circuit()
    from twpa_solver.signal.passive import db20, passive_s_matrix

    frequencies = np.arange(5.5, 9.0001, 0.002)
    s_matrix = passive_s_matrix(CIRCUIT, frequencies * 1e9, ports=(1, 2))
    s21_db = db20(s_matrix[:, 1, 0])
    peaks, _ = find_peaks(s21_db, prominence=0.01)
    spectrum = np.abs(np.fft.rfft(s21_db - np.mean(s21_db)))
    spectrum[0] = 0.0
    fft_frequency = np.fft.rfftfreq(s21_db.size, frequencies[1] - frequencies[0])
    dominant = int(np.argmax(spectrum))
    report = {
        "termination_ohm": Z_MATCH_OHM,
        "port_indices": indices,
        "frequency_start_ghz": float(frequencies[0]),
        "frequency_stop_ghz": float(frequencies[-1]),
        "frequency_step_ghz": float(frequencies[1] - frequencies[0]),
        "s21_min_db": float(np.min(s21_db)),
        "s21_max_db": float(np.max(s21_db)),
        "s21_peak_to_peak_db": float(np.ptp(s21_db)),
        "dominant_ripple_period_ghz": float(1.0 / fft_frequency[dominant]),
        "n_local_peaks": int(peaks.size),
    }
    np.savez_compressed(OUTPUT / "pump_off_s21.npz", frequency_ghz=frequencies, s21_db=s21_db)
    (OUTPUT / "passive_termination_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
