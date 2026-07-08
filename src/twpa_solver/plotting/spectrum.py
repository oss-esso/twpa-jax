"""Detailed candidate spectrum plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from twpa_solver.plotting.metrics import SpectrumFit
from twpa_solver.plotting.style import THESIS_FIGSIZE_SPECTRUM, save_figure


def _band_raw_gain(fit: SpectrumFit) -> np.ndarray:
    m = fit.metrics
    mask = (
        (fit.freq_ghz >= m.band_left_ghz_fit)
        & (fit.freq_ghz <= m.band_right_ghz_fit)
        & np.isfinite(fit.gain_db_raw)
    )
    return fit.gain_db_raw[mask]


def plot_candidate_spectrum(
    fit: SpectrumFit,
    outpath: Path | str,
    title: str | None = None,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    """Plot raw gain, fitted envelope, operation band, and band histogram."""
    fig = plt.figure(figsize=THESIS_FIGSIZE_SPECTRUM)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.4], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1], sharey=ax)

    m = fit.metrics
    ax.plot(fit.freq_ghz, fit.gain_db_raw, color="#1f77b4", lw=1.5, label="Gain")
    ax.plot(fit.f_dense_ghz, fit.g_dense_db, color="#ff7f0e", ls="--", lw=1.8, label="Envelope")
    if np.isfinite(m.band_left_ghz_fit) and np.isfinite(m.band_right_ghz_fit):
        ax.axvspan(
            m.band_left_ghz_fit,
            m.band_right_ghz_fit,
            color="#2ca25f",
            alpha=0.18,
            label="Operation band",
        )
    ax.set_xlabel("Signal frequency fs / GHz")
    ax.set_ylabel("Gain / dB")
    ax.set_title(title or f"Candidate point {m.point_index}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    annotation = (
        f"Pp = {m.pump_power_dbm:.3f} dBm\n"
        f"wp/2pi = {m.pump_freq_ghz:.3f} GHz\n\n"
        f"Gmax = {m.peak_gain_db_fit:.2f} dB\n"
        f"GBP = {m.gbp_ghz_fit:.3g} GHz\n"
        f"Ripple = {m.ripple_db_fit:.2f} dB\n"
        f"Smoothness = {m.smoothness_norm_fit:.3g}\n"
        f"BW = {m.bandwidth_ghz_fit:.3g} GHz"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "alpha": 0.85},
    )

    band_gain = _band_raw_gain(fit)
    if band_gain.size:
        hist, edges = np.histogram(band_gain, bins=min(20, max(3, band_gain.size)), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axh.barh(centers, hist, height=np.diff(edges), alpha=0.35, color="#2ca25f")
        if band_gain.size >= 3 and np.nanstd(band_gain) > 0.0:
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(band_gain)
                y = np.linspace(float(np.min(band_gain)), float(np.max(band_gain)), 300)
                axh.plot(kde(y), y, color="black", lw=1.2)
            except (ImportError, ValueError, np.linalg.LinAlgError):
                pass
        mean = float(np.mean(band_gain))
        sigma = float(np.std(band_gain))
        axh.axhline(mean, color="black", ls="--", lw=1.0)
        axh.axhline(mean + sigma, color="black", ls=":", lw=0.9)
        axh.axhline(mean - sigma, color="black", ls=":", lw=0.9)
        axh.text(
            0.97,
            0.97,
            f"sigma = {sigma:.3g} dB\nGmax = {m.peak_gain_db_fit:.2f} dB",
            transform=axh.transAxes,
            va="top",
            ha="right",
            fontsize=9,
        )
    axh.set_xlabel("Density")
    axh.tick_params(axis="y", labelleft=False)
    axh.grid(alpha=0.2)
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)
