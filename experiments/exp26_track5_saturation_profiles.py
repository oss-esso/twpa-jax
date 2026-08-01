"""Analyze the four selected spatial profiles from an exp26 q<=1 run."""

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
RUN = ROOT / "outputs" / "exp26_track5_profile_7p629_q01"


def dbm_from_current(current_a: float) -> float:
    return 10.0 * math.log10((current_a * current_a * 50.0 / 2.0) / 1e-3)


def load_rows() -> list[dict[str, float | str]]:
    with (RUN / "spatial_profiles.csv").open(newline="", encoding="utf-8") as handle:
        return [
            {
                key: (value if key == "operating_point" else float(value))
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
            if row.get("selected_signal_current_a")
        ]


def main() -> None:
    rows = load_rows()
    metrics = []
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["operating_point"]), []).append(row)
    order = ["smallest", "decade_below_p1db", "p1db", "largest_converged"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    colors = {label: color for label, color in zip(order, ("tab:blue", "tab:orange", "tab:red", "tab:green"))}

    for label in order:
        profile = sorted(grouped[label], key=lambda row: int(row["branch_index"]))
        branch = np.asarray([row["branch_index"] for row in profile], dtype=float)
        signal = np.asarray([row["signal_flux_abs"] for row in profile], dtype=float)
        pump = np.asarray([row["pump_flux_abs"] for row in profile], dtype=float)
        delta_k = np.asarray([row["delta_k_eff_rad_per_cell"] for row in profile], dtype=float)
        local_gain = np.diff(np.log(np.maximum(signal, 1e-300)))
        max_index = int(np.argmax(local_gain))
        negative = np.flatnonzero(local_gain < 0.0)
        pump_drop = pump[:-1] - pump[1:]
        max_drop_index = int(np.argmax(pump_drop)) if pump_drop.size else 0
        selected_current = float(profile[0]["selected_signal_current_a"])
        metrics.append(
            {
                "operating_point": label,
                "selected_signal_current_a": selected_current,
                "selected_signal_power_dbm": dbm_from_current(selected_current),
                "target_signal_current_a": float(profile[0]["target_signal_current_a"]),
                "n_branches": int(branch.size),
                "internal_peaking_factor": float(np.max(signal) / max(signal[0], 1e-300)),
                "max_signal_to_max_pump": float(np.max(signal) / max(np.max(pump), 1e-300)),
                "max_local_log_gain_per_cell": float(local_gain[max_index]) if local_gain.size else None,
                "max_local_gain_after_branch": int(branch[max_index + 1]) if local_gain.size else None,
                "first_negative_gain_after_branch": int(branch[negative[0] + 1]) if negative.size else None,
                "n_negative_local_gain_steps": int(negative.size),
                "delta_k_mean_rad_per_cell": float(np.mean(delta_k)),
                "delta_k_std_rad_per_cell": float(np.std(delta_k)),
                "delta_k_min_rad_per_cell": float(np.min(delta_k)),
                "delta_k_max_rad_per_cell": float(np.max(delta_k)),
                "pump_start_flux_abs": float(pump[0]),
                "pump_end_flux_abs": float(pump[-1]),
                "pump_end_over_start": float(pump[-1] / max(pump[0], 1e-300)),
                "pump_drop_fraction": float((pump[0] - pump[-1]) / max(pump[0], 1e-300)),
                "largest_pump_drop_after_branch": int(branch[max_drop_index + 1]) if pump_drop.size else None,
            }
        )
        normalized_signal = signal / max(signal[0], 1e-300)
        normalized_pump = pump / max(pump[0], 1e-300)
        axes[0].plot(branch, normalized_signal, color=colors[label], label=label)
        axes[1].plot(branch, normalized_pump, color=colors[label], label=label)

    axes[0].set_ylabel("Signal flux / input flux")
    axes[0].set_title("2c q≤1 spatial profiles at four signal powers, 7.629 GHz")
    axes[1].set_xlabel("Branch index")
    axes[1].set_ylabel("Pump flux / input flux")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(RUN / "saturation_profiles.png", dpi=180)
    plt.close(figure)

    report = {"frequency_ghz": 7.629, "profiles": metrics}
    with (RUN / "saturation_profile_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
