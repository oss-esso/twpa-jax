#!/usr/bin/env python3
"""Plot the multi-tone Guarcello reductions and paper-aligned diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / "outputs" / "chaos" / "phase2"
FIG2A = PHASE2 / "fig2a_50ohm_mtls" / "run"
FIG2B = PHASE2 / "fig2b_50ohm_pump_m55_mtls" / "run"
OUT = PHASE2 / "plots_mtls"


def rows(path: Path) -> list[dict[str, str]]:
    with (path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_fig2a() -> None:
    data = rows(FIG2A)
    reference = json.loads((PHASE2 / "pump_off_reference_50ohm.json").read_text())
    reference_v = float(reference["analysis"]["signal_vout_peak_v"])
    x = np.asarray([float(row["pump_dbm"]) for row in data])
    narrow_absolute = np.asarray([float(row["gain_db"]) for row in data])
    narrow = 20.0 * np.log10(
        np.asarray([float(row["signal_vout_peak_v"]) for row in data]) / reference_v
    )
    wide = narrow + np.asarray([float(row["gain_wideband_db"]) for row in data]) - narrow_absolute
    reference = {-70: 0.0, -65: 0.6, -62: 1.4, -60: 2.0, -58: 3.4,
                 -57: 4.5, -56: 6.0, -55: 7.5, -54: 9.0, -53.5: 12.0}
    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    axes[0].plot(x, narrow, "o-", label="narrowband multi-tone")
    axes[0].plot(x, wide, "o-", label="wideband")
    axes[0].plot(list(reference), list(reference.values()), "k--", label="digitized paper")
    axes[0].set_ylabel("Gain (dB)")
    axes[0].set_xlim(-70, -45)
    mean_runtime = np.mean([float(row["runtime_s"]) for row in data])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for index, row in enumerate(data):
        branch = np.load(FIG2A / f"point_{index:05d}" / "poincare_branches.npz")
        axes[1].scatter(np.full(branch["upward"].size, x[index]), branch["upward"],
                        s=2, alpha=0.25, color="tab:blue")
    axes[1].set_ylabel("$V'_{PS}$ (paper units)")
    axes[1].set_xlim(-70, -45)
    axes[1].set_ylim(0.0, 0.4)
    axes[1].grid(True, alpha=0.3)
    spectra = np.load(FIG2A / "spectra_map.npz")
    mesh = axes[2].pcolormesh(
        spectra["x"], spectra["frequency_ghz"], spectra["spectrum_dbm"].T,
        shading="auto", cmap="Blues", vmin=-165.0, vmax=-70.0,
    )
    colorbar = fig.colorbar(mesh, ax=axes[2], pad=0.01)
    colorbar.set_label("Fourier amplitude (dBm)")
    for frequency, label in ((6.42, "$f_s$"), (7.0, "$f_p$"),
                             (3.5, "$f_p/2$"), (10.5, "$3f_p/2$"), (14.0, "$2f_p$")):
        axes[2].axhline(frequency, color="k", linestyle=":", alpha=0.5)
        axes[2].text(-69.5, frequency, label, va="bottom", fontsize=8)
    axes[2].set_xlim(-70, -45)
    axes[2].set_ylim(0, 20)
    axes[2].set_ylabel("Frequency (GHz)")
    axes[2].set_xlabel("Pump power (dBm)")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Guarcello Fig. 2(a): multi-tone gain, bifurcation points, spectrum")
    fig.text(
        0.5, 0.008, f"Average runtime per point: {mean_runtime:.2f} s",
        ha="center", va="bottom", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig2a_mtls_three_panel.png", dpi=180)
    plt.close(fig)


def plot_fig2b() -> None:
    data = rows(FIG2B)
    reference = json.loads((PHASE2 / "pump_off_reference_50ohm.json").read_text())
    reference_v = float(reference["analysis"]["signal_vout_peak_v"])
    x = np.asarray([float(row["signal_ghz"]) for row in data])
    narrow_absolute = np.asarray([float(row["gain_db"]) for row in data])
    narrow = 20.0 * np.log10(
        np.asarray([float(row["signal_vout_peak_v"]) for row in data]) / reference_v
    )
    wide = narrow + np.asarray([float(row["gain_wideband_db"]) for row in data]) - narrow_absolute
    mask = ~np.isclose(x, 7.0)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x[mask], narrow[mask], "o-", label="narrowband multi-tone")
    axes[0].plot(x[mask], wide[mask], "o-", label="wideband")
    axes[0].axvline(7.0, color="k", linestyle=":", label="masked $f_s=f_p$")
    axes[0].set_ylabel("Gain (dB)")
    mean_runtime = np.mean([float(row["runtime_s"]) for row in data])
    axes[0].text(
        0.01, 0.02, f"Ppump = -55 dBm; mean runtime/point: {mean_runtime:.2f} s",
        transform=axes[0].transAxes, fontsize=9, va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    sigma = np.asarray([float(row["poincare_std_mV_per_tnorm_upward"]) for row in data])
    axes[1].plot(x[mask], sigma[mask], "o-")
    axes[1].axvline(7.0, color="k", linestyle=":")
    axes[1].set_xlabel("Signal frequency (GHz)")
    axes[1].set_ylabel(r"$\sigma_{up}$ (diagnostic)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Guarcello Fig. 2(b): multi-tone estimator")
    fig.tight_layout()
    fig.savefig(OUT / "fig2b_mtls.png", dpi=180)
    plt.close(fig)


def plot_fig4_pump_panels() -> None:
    """Plot Fig. 4 pump panels and bias panels when the sweep is available."""
    data = rows(FIG2A)
    pump = np.asarray([float(row["pump_dbm"]) for row in data])
    beta_mean = np.asarray([float(row["beta_mean"]) for row in data])
    beta_std = np.asarray([float(row["beta_std"]) for row in data])
    gamma_mean = np.asarray([float(row["gamma_mean"]) for row in data])
    gamma_std = np.asarray([float(row["gamma_std"]) for row in data])

    bias_path = PHASE2 / "fig4_bias_0_34_m55" / "summary.csv"
    has_bias = bias_path.exists()
    fig, axes = plt.subplots(2, 2 if has_bias else 1, figsize=(12 if has_bias else 9, 8),
                             sharex="col", squeeze=False)
    axes = axes if has_bias else axes[:, :1]
    series = (
        (axes[0, 0], beta_mean, beta_std, r"$\beta$", "tab:red", "goldenrod", "Pump power (dBm)"),
        (axes[1, 0], gamma_mean, gamma_std, r"$\gamma$", "navy", "cyan", "Pump power (dBm)"),
    )
    if has_bias:
        with bias_path.open(newline="", encoding="utf-8") as handle:
            bias_rows = list(csv.DictReader(handle))
        bias = np.asarray([float(row["bias_ua"]) for row in bias_rows])
        series += (
            (axes[0, 1], np.asarray([float(row["beta_mean"]) for row in bias_rows]),
             np.asarray([float(row["beta_std"]) for row in bias_rows]), r"$\beta$",
             "tab:red", "goldenrod", "Bias current (µA)"),
            (axes[1, 1], np.asarray([float(row["gamma_mean"]) for row in bias_rows]),
             np.asarray([float(row["gamma_std"]) for row in bias_rows]), r"$\gamma$",
             "navy", "cyan", "Bias current (µA)"),
        )
    for axis, mean, spread, label, color, spread_color, xlabel in series:
        twin = axis.twinx()
        horizontal = pump if xlabel.startswith("Pump") else bias
        axis.plot(horizontal, mean, color=color, linewidth=2, label=f"mean {label}")
        twin.plot(horizontal, spread, color=spread_color, linewidth=1.5,
                  label=f"std {label}")
        axis.set_ylabel(f"Mean {label}", color=color)
        twin.set_ylabel(f"Std. dev. {label}", color=spread_color)
        axis.grid(True, alpha=0.3)
        lines, labels = axis.get_legend_handles_labels()
        twin_lines, twin_labels = twin.get_legend_handles_labels()
        axis.legend(lines + twin_lines, labels + twin_labels, loc="upper left")
        axis.set_xlabel(xlabel)
    axes[0, 0].set_title("Pump sweep")
    axes[1, 0].set_xlim(-70, -45)
    if has_bias:
        axes[0, 1].set_title("Bias sweep at -55 dBm")
        axes[1, 1].set_xlim(0, 34)
    else:
        fig.text(0.99, 0.01, "Bias-sweep panels unavailable: no artifact exists.",
                 ha="right", va="bottom", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig4_nonlinearity_pump_panels.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_fig2a()
    plot_fig2b()
    plot_fig4_pump_panels()
    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
