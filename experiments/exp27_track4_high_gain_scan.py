"""Scan 2c pump current for the attainable small-signal gain ceiling."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "exp27_track4_high_gain_scan"
FREQUENCIES = [6.943, 7.290, 7.629]
PUMP_CURRENTS = [
    1.0e-6,
    3.0e-6,
    5.0e-6,
    7.231074707853736e-6,
    9.0e-6,
    1.1e-5,
    1.3e-5,
    1.5e-5,
    1.8e-5,
    2.2e-5,
]


def run_point(frequency_ghz: float, pump_current_a: float) -> dict[str, object]:
    label = f"f{frequency_ghz:.3f}_i{pump_current_a:.3e}".replace(".", "p").replace("-", "m")
    output_dir = OUTPUT / "scan" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", "designs/ipm_2c_fixed",
        "--pump-freq-ghz", "7.540816326531111",
        "--signal-ghz", f"{frequency_ghz:.3f}",
        "--pump-current-a", f"{pump_current_a:.16e}",
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
        "--n-signal-power", "1",
        "--signal-current-min-a", "1e-10",
        "--signal-current-max-a", "1e-10",
        "--p1db-power-tol-db", "0",
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "120",
        "--allow-memory-overcommit",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (output_dir / "run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary_path = output_dir / "compression_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "frequency_ghz": frequency_ghz,
            "pump_current_a": pump_current_a,
            "status": summary.get("status"),
            "gain_db": summary.get("small_signal_gain_vs_off_db"),
            "power_balance_rel_err": summary.get("max_power_balance_rel_err"),
            "path": str(output_dir),
        }
    return {
        "frequency_ghz": frequency_ghz,
        "pump_current_a": pump_current_a,
        "status": "SUBPROCESS_FAILED",
        "returncode": completed.returncode,
        "path": str(output_dir),
    }


def run_high_gain_compression(point: dict[str, object]) -> dict[str, object] | None:
    gain = point.get("gain_db")
    if gain is None or not np.isfinite(float(gain)) or float(gain) <= 20.0:
        return None
    output_dir = OUTPUT / "high_gain_compression"
    command = [
        sys.executable,
        "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", "designs/ipm_2c_fixed",
        "--pump-freq-ghz", "7.540816326531111",
        "--signal-ghz", f"{float(point['frequency_ghz']):.3f}",
        "--pump-current-a", f"{float(point['pump_current_a']):.16e}",
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "dense_real",
        "--pump-harmonics", "6", "--pump-nt", "40",
        "--multitone-basis", "lattice", "--multitone-sidebands", "1",
        "--source-port", "1", "--pump-port", "4", "--out-port", "2",
        "--attenuation-db", "0", "--factor-backend", "pardiso",
        "--n-signal-power", "16", "--signal-current-min-a", "1e-10",
        "--signal-current-max-a", "1e-6", "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600", "--allow-memory-overcommit",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (output_dir / "run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    return {"path": str(output_dir), "returncode": completed.returncode}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [
        run_point(frequency, current)
        for frequency in FREQUENCIES
        for current in PUMP_CURRENTS
    ]
    finite = [row for row in rows if row.get("status") == "VALID_SOLVED" and row.get("gain_db") is not None]
    ceiling = max(finite, key=lambda row: float(row["gain_db"])) if finite else None
    report = {
        "frequencies_ghz": FREQUENCIES,
        "pump_currents_a": PUMP_CURRENTS,
        "rows": rows,
        "gain_ceiling": ceiling,
        "high_gain_compression": run_high_gain_compression(ceiling) if ceiling else None,
    }
    (OUTPUT / "high_gain_scan_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for frequency in FREQUENCIES:
        selected = [row for row in finite if abs(float(row["frequency_ghz"]) - frequency) < 1e-9]
        selected.sort(key=lambda row: float(row["pump_current_a"]))
        axis.plot(
            [float(row["pump_current_a"]) * 1e6 for row in selected],
            [float(row["gain_db"]) for row in selected],
            marker="o",
            label=f"{frequency:.3f} GHz",
        )
    axis.set(xlabel="Pump current (µA)", ylabel="Small-signal gain (dB)", title="2c pump-current gain scan")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "high_gain_scan.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
