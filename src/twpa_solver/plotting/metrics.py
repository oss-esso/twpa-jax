"""Fit-based spectrum metrics for gain-map plotting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


@dataclass
class SpectrumFitMetrics:
    point_index: int
    pump_power_dbm: float
    pump_freq_ghz: float
    status: str
    peak_gain_db_fit: float
    peak_signal_freq_ghz_fit: float
    band_left_ghz_fit: float
    band_right_ghz_fit: float
    bandwidth_ghz_fit: float
    gbp_ghz_fit: float
    gbp_dbghz_fit: float
    ripple_db_fit: float
    smoothness_rms_curvature_fit: float
    smoothness_norm_fit: float
    mean_gain_db_fit: float
    median_gain_db_fit: float
    min_gain_db_fit: float
    score_fit: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpectrumFit:
    freq_ghz: np.ndarray
    gain_db_raw: np.ndarray
    gain_db_smooth: np.ndarray
    f_dense_ghz: np.ndarray
    g_dense_db: np.ndarray
    band_mask_dense: np.ndarray
    metrics: SpectrumFitMetrics


def auto_savgol_window(n: int, window_frac: float, polyorder: int) -> int:
    """Choose an odd Savitzky-Golay window that is valid for the trace length."""
    if n <= 2:
        return 0
    requested = max(11, int(round(float(window_frac) * n)))
    window = min(requested, n - 1 if n % 2 == 0 else n)
    if window % 2 == 0:
        window -= 1
    min_window = int(polyorder) + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        window = min_window if min_window <= n else 0
    return int(window if window >= 3 else 0)


def _finite_sorted_xy(
    freq_ghz: np.ndarray,
    gain_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(freq_ghz, dtype=float).reshape(-1)
    g = np.asarray(gain_db, dtype=float).reshape(-1)
    mask = np.isfinite(f) & np.isfinite(g)
    f = f[mask]
    g = g[mask]
    if f.size == 0:
        return f, g
    order = np.argsort(f)
    f = f[order]
    g = g[order]
    unique_f, idx = np.unique(f, return_index=True)
    return unique_f, g[idx]


def fit_spectrum(
    freq_ghz: np.ndarray,
    gain_db: np.ndarray,
    *,
    window_frac: float = 0.05,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, CubicSpline]:
    """Smooth a spectrum and fit a cubic spline envelope."""
    f, g = _finite_sorted_xy(freq_ghz, gain_db)
    if f.size < 2:
        raise ValueError("at least two finite spectrum samples are required")
    effective_poly = max(0, min(int(polyorder), f.size - 1))
    window = auto_savgol_window(f.size, window_frac, effective_poly)
    if window > effective_poly and window >= 3:
        smooth = savgol_filter(g, window_length=window, polyorder=effective_poly, mode="interp")
    else:
        smooth = g.copy()
    spline = CubicSpline(f, smooth)
    return f, smooth, spline


def dense_fit_curve(
    freq_ghz: np.ndarray,
    gain_db: np.ndarray,
    *,
    n_dense: int = 2000,
    window_frac: float = 0.05,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CubicSpline]:
    """Return sorted raw frequencies, smoothed samples, and dense spline values."""
    f, smooth, spline = fit_spectrum(
        freq_ghz,
        gain_db,
        window_frac=window_frac,
        polyorder=polyorder,
    )
    f_dense = np.linspace(float(f.min()), float(f.max()), int(max(2, n_dense)))
    g_dense = np.asarray(spline(f_dense), dtype=float)
    return f, smooth, f_dense, g_dense, spline


def find_operation_band_from_fit(
    f_dense: np.ndarray,
    g_dense: np.ndarray,
    *,
    peak_index: int | None = None,
    drop_db: float = 3.0,
) -> tuple[float, float, np.ndarray]:
    """Find the contiguous fitted operation band around the fitted peak."""
    f = np.asarray(f_dense, dtype=float)
    g = np.asarray(g_dense, dtype=float)
    if f.size < 2 or f.size != g.size or not np.all(np.isfinite(g)):
        return np.nan, np.nan, np.zeros_like(g, dtype=bool)
    peak = int(np.argmax(g) if peak_index is None else peak_index)
    threshold = float(g[peak] - drop_db)
    left = peak
    while left > 0 and g[left - 1] >= threshold:
        left -= 1
    right = peak
    while right < g.size - 1 and g[right + 1] >= threshold:
        right += 1
    mask = np.zeros_like(g, dtype=bool)
    mask[left : right + 1] = True
    return float(f[left]), float(f[right]), mask


def minus3db_band(
    freq_ghz: np.ndarray,
    gain_db: np.ndarray,
    *,
    drop_db: float = 3.0,
) -> dict[str, Any] | None:
    """Measure the operation band directly from the raw sweep (no smoothing).

    Walks out from the global-max sample until the gain drops ``drop_db`` below
    the peak, linearly interpolating each crossing. Unlike the spline-envelope
    fit this preserves razor-thin near-fold gain peaks, so the reported -3 dB
    bandwidth is the real one rather than a smoothed-over overestimate. Returns
    ``None`` for <2 finite samples. ``band_clipped`` flags an edge that ran into
    the sweep boundary (widen the sweep to resolve it).
    """
    f = np.asarray(freq_ghz, dtype=float).reshape(-1)
    g = np.asarray(gain_db, dtype=float).reshape(-1)
    mask = np.isfinite(f) & np.isfinite(g)
    f, g = f[mask], g[mask]
    if f.size < 2:
        return None
    order = np.argsort(f)
    f, g = f[order], g[order]
    peak_i = int(np.argmax(g))
    peak = float(g[peak_i])
    threshold = peak - float(drop_db)

    clipped = False
    left = float(f[0])
    for j in range(peak_i, 0, -1):
        if g[j - 1] < threshold:
            left = float(np.interp(threshold, [g[j - 1], g[j]], [f[j - 1], f[j]]))
            break
    else:
        clipped = True
    right = float(f[-1])
    for j in range(peak_i, f.size - 1):
        if g[j + 1] < threshold:
            right = float(np.interp(threshold, [g[j + 1], g[j]], [f[j + 1], f[j]]))
            break
    else:
        clipped = True

    bandwidth = float(right - left)
    gbp = float(10.0 ** (peak / 10.0) * bandwidth)
    return {
        "peak_gain_db": peak,
        "peak_freq_ghz": float(f[peak_i]),
        "band_left_ghz": left,
        "band_right_ghz": right,
        "bandwidth_ghz": bandwidth,
        "gbp_ghz": gbp,
        "band_clipped": clipped,
    }


def _invalid_metrics(metadata: dict[str, Any], status: str) -> SpectrumFitMetrics:
    nan = float("nan")
    return SpectrumFitMetrics(
        point_index=int(metadata.get("point_index", -1)),
        pump_power_dbm=float(metadata.get("pump_power_dbm", nan)),
        pump_freq_ghz=float(metadata.get("pump_freq_ghz", nan)),
        status=status,
        peak_gain_db_fit=nan,
        peak_signal_freq_ghz_fit=nan,
        band_left_ghz_fit=nan,
        band_right_ghz_fit=nan,
        bandwidth_ghz_fit=nan,
        gbp_ghz_fit=nan,
        gbp_dbghz_fit=nan,
        ripple_db_fit=nan,
        smoothness_rms_curvature_fit=nan,
        smoothness_norm_fit=nan,
        mean_gain_db_fit=nan,
        median_gain_db_fit=nan,
        min_gain_db_fit=nan,
        score_fit=nan,
    )


def compute_fit_metrics(
    freq_ghz: np.ndarray,
    gain_db: np.ndarray,
    metadata: dict[str, Any],
    *,
    drop_db: float = 3.0,
    n_dense: int = 2000,
    window_frac: float = 0.05,
    polyorder: int = 3,
) -> SpectrumFit:
    """Compute all standard fitted metrics and return plot intermediates."""
    raw_f, raw_g = _finite_sorted_xy(freq_ghz, gain_db)
    try:
        f_sample, smooth, f_dense, g_dense, spline = dense_fit_curve(
            raw_f,
            raw_g,
            n_dense=n_dense,
            window_frac=window_frac,
            polyorder=polyorder,
        )
        peak_index = int(np.argmax(g_dense))
        left, right, band_mask = find_operation_band_from_fit(
            f_dense,
            g_dense,
            peak_index=peak_index,
            drop_db=drop_db,
        )
        bandwidth = float(right - left)
        if bandwidth <= 0.0 or int(np.count_nonzero(band_mask)) < 2:
            raise ValueError("invalid fitted operation band")
        g_band = g_dense[band_mask]
        g2_band = np.asarray(spline(f_dense, 2), dtype=float)[band_mask]
        peak_gain = float(g_dense[peak_index])
        g_peak_lin = float(10.0 ** (peak_gain / 10.0))
        gbp = float(g_peak_lin * bandwidth)
        smooth_rms = float(np.sqrt(np.mean(g2_band**2)))
        smooth_norm = float(smooth_rms * bandwidth**2)
        ripple = float(np.max(g_band) - np.min(g_band))
        score = float(
            peak_gain
            + 5.0 * np.log10(max(gbp, 1e-12))
            - ripple
            - smooth_norm
        )
        metrics = SpectrumFitMetrics(
            point_index=int(metadata.get("point_index", -1)),
            pump_power_dbm=float(metadata.get("pump_power_dbm", np.nan)),
            pump_freq_ghz=float(metadata.get("pump_freq_ghz", np.nan)),
            status=str(metadata.get("status", "UNKNOWN")),
            peak_gain_db_fit=peak_gain,
            peak_signal_freq_ghz_fit=float(f_dense[peak_index]),
            band_left_ghz_fit=left,
            band_right_ghz_fit=right,
            bandwidth_ghz_fit=bandwidth,
            gbp_ghz_fit=gbp,
            gbp_dbghz_fit=float(peak_gain + 10.0 * np.log10(max(bandwidth, 1e-12))),
            ripple_db_fit=ripple,
            smoothness_rms_curvature_fit=smooth_rms,
            smoothness_norm_fit=smooth_norm,
            mean_gain_db_fit=float(np.mean(g_band)),
            median_gain_db_fit=float(np.median(g_band)),
            min_gain_db_fit=float(np.min(g_band)),
            score_fit=score,
        )
        return SpectrumFit(raw_f, raw_g, smooth, f_dense, g_dense, band_mask, metrics)
    except (ValueError, FloatingPointError) as exc:
        metrics = _invalid_metrics(metadata, f"INVALID_GAIN:{exc}")
        empty_mask = np.zeros(max(int(n_dense), 2), dtype=bool)
        f_dense = np.linspace(0.0, 1.0, empty_mask.size)
        g_dense = np.full_like(f_dense, np.nan)
        return SpectrumFit(raw_f, raw_g, raw_g.copy(), f_dense, g_dense, empty_mask, metrics)
