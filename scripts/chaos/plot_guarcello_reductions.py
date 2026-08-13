#!/usr/bin/env python3
"""Plot corrected Guarcello Phase-2 reductions from saved JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / "outputs" / "chaos" / "phase2"
OUT = PHASE2 / "plots"


def load_rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def fig2a() -> None:
    rows = load_rows(PHASE2 / "fig2a_50ohm_full" / "directional_sigma_reduction.json")
    x = np.array([float(row["control"]) for row in rows])
    absolute = np.array([float(row["gain_db"]) for row in rows])
    relative = np.array([float(row["gain_vs_off_db"]) for row in rows])
    sigma_up = np.array([float(row["sigma_upward"]) for row in rows])
    sigma_both = np.array([float(row["sigma_both"]) for row in rows])
    target_x = np.array([-70, -65, -62, -60, -58, -57, -56, -55, -54, -53.5])
    target_y = np.array([0, 0.6, 1.4, 2, 3.4, 4.5, 6, 7.5, 9, 12])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(x, absolute, "o-", label="absolute gain")
    axes[0].plot(x, relative, "o-", label="pump-off normalized gain")
    axes[0].plot(target_x, target_y, "k--", alpha=0.7, label="digitized paper")
    axes[0].axvspan(-54, -53.5, color="tab:red", alpha=0.12, label="transition band")
    axes[0].set(xlabel="Pump power (dBm)", ylabel="Gain (dB)", title="Fig. 2(a): gain")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].semilogy(x, sigma_up, "o-", label=r"$\sigma_{up}$")
    axes[1].semilogy(x, sigma_both, "--", alpha=0.7, label=r"$\sigma_{both}$ (legacy)")
    axes[1].axvspan(-54, -53.5, color="tab:red", alpha=0.12)
    axes[1].set(xlabel="Pump power (dBm)", ylabel=r"$\sigma(V'_{PS})$", title="Fig. 2(a): Poincaré spread")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "fig2a_corrected.png", dpi=180)
    plt.close(fig)


def fig2b() -> None:
    paths = [
        PHASE2 / "fig2b_50ohm_full" / "directional_sigma_reduction.json",
        PHASE2 / "fig2b_50ohm_pump_m57_full" / "directional_sigma_reduction.json",
    ]
    labels = ["-54.5 dBm", "-57 dBm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for path, label in zip(paths, labels):
        rows = load_rows(path)
        x = np.array([float(row["control"]) for row in rows])
        gain = np.array([float(row["gain_vs_off_db"]) for row in rows])
        sigma = np.array([float(row["sigma_upward"]) for row in rows])
        keep = ~np.isclose(x, 7.0)
        axes[0].plot(x[keep], gain[keep], "o-", label=label)
        axes[1].semilogy(x[keep], sigma[keep], "o-", label=label)
    axes[0].axvline(7.0, color="k", linestyle=":", label="masked pump frequency")
    axes[0].set(xlabel="Signal frequency (GHz)", ylabel="Gain vs pump-off (dB)", title="Fig. 2(b): corrected gain")
    axes[1].axvline(7.0, color="k", linestyle=":")
    axes[1].set(xlabel="Signal frequency (GHz)", ylabel=r"$\sigma_{up}$", title="Fig. 2(b): directional spread")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    fig.savefig(OUT / "fig2b_corrected_comparison.png", dpi=180)
    plt.close(fig)


def port_audit() -> None:
    data = json.loads((PHASE2 / "port_update_boundary_audit.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    x = [row["pump_dbm"] for row in rows]
    stable = [row["stable_gain_db"] for row in rows]
    centered = [row["paper_centered_gain_db"] for row in rows]
    fig, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    axis.plot(x, stable, "o-", label="stable update")
    axis.plot(x, centered, "x--", label="paper-centered (NaN)")
    axis.set(xlabel="Pump power (dBm)", ylabel="Absolute gain (dB)", title="Port-update boundary audit")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(OUT / "port_update_audit.png", dpi=180)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig2a()
    fig2b()
    port_audit()
    print(f"Wrote plots to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
