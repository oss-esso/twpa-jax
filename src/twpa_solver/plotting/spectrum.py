"""Detailed candidate spectrum plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from twpa_solver.plotting.metrics import SpectrumFit
from twpa_solver.plotting.style import THESIS_FIGSIZE_SPECTRUM, save_figure

ERROR_BIN_WIDTH_DB = 0.25


def _fit_error(fit: SpectrumFit) -> np.ndarray:
    mask = np.isfinite(fit.freq_ghz) & np.isfinite(fit.gain_db_raw)
    if not np.any(mask):
        return np.asarray([], dtype=float)
    envelope_at_samples = np.interp(fit.freq_ghz[mask], fit.f_dense_ghz, fit.g_dense_db)
    return fit.gain_db_raw[mask] - envelope_at_samples


def _symmetric_error_edges(values: np.ndarray, *, bin_width: float = ERROR_BIN_WIDTH_DB) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        limit = bin_width
    else:
        limit = max(float(np.nanmax(np.abs(finite))), bin_width)
        limit = np.ceil(limit / bin_width) * bin_width
    return np.arange(-limit, limit + 0.5 * bin_width, bin_width)


def plot_candidate_s21_bandwidth(
    freq_ghz: np.ndarray,
    gain_db: np.ndarray,
    meta: dict,
    band: dict | None,
    outpath: Path | str,
    title: str | None = None,
    *,
    drop_db: float = 3.0,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    """Plot a candidate's swept S21 with its -3 dB band and pump condition.

    ``band`` is the dict from ``metrics.minus3db_band`` (or None). The annotation
    reports the map's own trailing gain (``meta['map_gain_db']``) next to the
    swept peak so a needle that the sweep under-resolves is still obvious.
    """
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_SPECTRUM)
    ax.plot(freq_ghz, gain_db, color="#1f77b4", lw=1.6, label="S21 gain")

    if band is not None:
        peak = band["peak_gain_db"]
        ax.axvspan(band["band_left_ghz"], band["band_right_ghz"],
                   color="#2ca25f", alpha=0.18, label=f"-{drop_db:g} dB band")
        ax.axhline(peak - drop_db, color="#2ca25f", ls=":", lw=1.2)
        for edge in (band["band_left_ghz"], band["band_right_ghz"]):
            ax.axvline(edge, color="#2ca25f", ls="--", lw=1.0, alpha=0.7)
        ax.plot([band["peak_freq_ghz"]], [peak], "o", color="#d62728", ms=6, label="peak")

    ax.set_xlabel("Signal frequency fs / GHz")
    ax.set_ylabel("S21 gain / dB")
    ax.set_title(title or f"Candidate point {meta.get('point_index')}")
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.5, linewidth=1.2)
    ax.grid(which="minor", alpha=0.25, linewidth=0.6)
    ax.legend(loc="best")

    lines = [
        f"Pump: Pp = {meta['pump_power_dbm']:.3f} dBm",
        f"      fp = {meta['pump_freq_ghz']:.4f} GHz",
    ]
    if meta.get("map_gain_db") is not None:
        lines.append(f"map gain @ ws = {meta['map_gain_db']:.2f} dB")
    if band is not None:
        lines += [
            "",
            f"Gmax(swept) = {band['peak_gain_db']:.2f} dB @ {band['peak_freq_ghz']:.4f} GHz",
            f"BW(-{drop_db:g} dB) = {band['bandwidth_ghz'] * 1e3:.1f} MHz"
            + ("  [clipped]" if band.get("band_clipped") else ""),
            f"GBP = {band['gbp_ghz']:.3g} GHz",
        ]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox={"boxstyle": "round,pad=0.35", "fc": "white", "alpha": 0.85})
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)


def plot_candidate_spectrum(
    fit: SpectrumFit,
    outpath: Path | str,
    title: str | None = None,
    *,
    save_pdf: bool = False,
    save_svg: bool = False,
) -> None:
    """Plot raw gain, fitted envelope, operation band, and band histogram."""
    fig = plt.figure(figsize=THESIS_FIGSIZE_SPECTRUM, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.4], wspace=0.04)

    ax = fig.add_subplot(gs[:, 0])
    axh = fig.add_subplot(gs[:, 1])

    axh.yaxis.tick_right()
    axh.yaxis.set_label_position("right")
    axh.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    axh.spines["left"].set_visible(False)
    axh.spines["right"].set_visible(True)

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

    fit_error = _fit_error(fit)
    if fit_error.size:
        edges = _symmetric_error_edges(fit_error)
        hist, edges = np.histogram(
            fit_error,
            bins=edges,
            density=False,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        axh.barh(centers, hist, height=np.diff(edges), alpha=0.35, color="#2ca25f")
        if fit_error.size >= 3 and np.nanstd(fit_error) > 0.0:
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(fit_error)
                y = np.linspace(float(edges[0]), float(edges[-1]), 300)
                bin_width = float(np.mean(np.diff(edges)))
                axh.plot(kde(y) * fit_error.size * bin_width, y, color="black", lw=1.2)
            except (ImportError, ValueError, np.linalg.LinAlgError):
                pass
        mean = float(np.mean(fit_error))
        sigma = float(np.std(fit_error))
        axh.axhline(mean, color="black", ls="--", lw=1.0)
        axh.axhline(mean + sigma, color="black", ls=":", lw=0.9)
        axh.axhline(mean - sigma, color="black", ls=":", lw=0.9)
        axh.set_ylim(float(edges[0]), float(edges[-1]))
        axh.text(
            0.97,
            0.97,
            f"mean = {mean:.3g} dB\nsigma = {sigma:.3g} dB",
            transform=axh.transAxes,
            va="top",
            ha="right",
            fontsize=9,
        )
    axh.set_xlabel("Count")
    axh.set_ylabel("Gain - envelope / dB")
    axh.grid(alpha=0.2)
    save_figure(fig, outpath, save_pdf=save_pdf, save_svg=save_svg)
