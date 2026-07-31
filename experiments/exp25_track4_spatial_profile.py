"""Measure internal signal peaking at P1dB for three exp24b frequencies."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = (5.8, 6.49, 7.629)


def command(output_dir: Path, frequency: float) -> list[str]:
    """Build the q<=1 exp24b command with spatial profiles enabled."""
    return [
        sys.executable, "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", "outputs/ipm_python_design",
        "--pump-freq-ghz", "7.540816326531111",
        "--pump-current-a", "7.231074707853736e-06",
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "dense_real", "--pump-harmonics", "6",
        "--pump-nt", "40", "--multitone-basis", "lattice",
        "--multitone-sidebands", "1", "--source-port", "1",
        "--pump-port", "4", "--out-port", "2", "--attenuation-db", "0",
        "--factor-backend", "pardiso", "--n-signal-power", "16",
        "--signal-current-min-a", "1e-10", "--signal-current-max-a", "1e-6",
        "--recovery", "ladder", "--signal-continuation-deadline-s", "600",
        "--signal-workers", "1", "--signal-ghz", str(frequency),
        "--allow-memory-overcommit", "--spatial-profiles",
    ]


def metric(output_dir: Path, frequency: float) -> dict[str, object]:
    """Extract P1dB spatial metrics from one completed run."""
    summary = json.loads((output_dir / "compression_summary.json").read_text(encoding="utf-8"))
    with (output_dir / "spatial_profiles.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["operating_point"] == "p1db"]
    rows.sort(key=lambda row: int(row["branch_index"]))
    signal = np.asarray([float(row["signal_flux_abs"]) for row in rows])
    pump = np.asarray([float(row["pump_flux_abs"]) for row in rows])
    delta_k = np.asarray([float(row["delta_k_eff_rad_per_cell"]) for row in rows])
    gradient = np.diff(signal)
    gradient = gradient[np.abs(gradient) > max(np.max(np.abs(signal)) * 1e-14, 1e-300)]
    sign_changes = int(np.count_nonzero(np.signbit(gradient[1:]) != np.signbit(gradient[:-1])))
    return {
        "signal_ghz": frequency,
        "status": summary.get("status"),
        "small_signal_gain_db": summary.get("small_signal_gain_vs_off_db"),
        "p1db_input_dbm": summary.get("p1db_input_dbm"),
        "branch_count": int(signal.size),
        "internal_peaking_factor": float(np.max(signal) / signal[0]),
        "gradient_sign_changes": sign_changes,
        "profile_monotone": sign_changes == 0,
        "max_signal_over_max_pump": float(np.max(signal) / np.max(pump)),
        "delta_k_mean_rad_per_cell": float(np.mean(delta_k)),
        "delta_k_spread_rad_per_cell": float(np.max(delta_k) - np.min(delta_k)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp25_track4_spatial_profile"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for frequency in CASES:
        case_dir = args.output_dir / f"frequency_{frequency:.6f}ghz"
        case_dir.mkdir(parents=True, exist_ok=True)
        summary_path = case_dir / "compression_summary.json"
        if not summary_path.exists():
            result = subprocess.run(command(case_dir, frequency), check=False)
            if result.returncode:
                metrics.append({"signal_ghz": frequency, "status": "SUBPROCESS_FAILED", "returncode": result.returncode})
                continue
        metrics.append(metric(case_dir, frequency))
        with (case_dir / "spatial_profiles.csv").open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["operating_point"] == "p1db"]
        rows.sort(key=lambda row: int(row["branch_index"]))
        signal = np.asarray([float(row["signal_flux_abs"]) for row in rows])
        axis.plot(np.arange(signal.size), signal / signal[0], label=f"{frequency:.3f} GHz")
    axis.set_xlabel("branch index")
    axis.set_ylabel("P1dB signal flux / input-branch flux")
    axis.set_title("Internal signal profile at P1dB")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "signal_profiles_p1db.png", dpi=180)
    plt.close(figure)
    (args.output_dir / "spatial_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "spatial_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in metrics for key in row}))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
