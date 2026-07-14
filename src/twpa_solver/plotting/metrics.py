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


def _threshold_crossing(
    f0: float, g0: float, f1: float, g1: float, threshold: float,
) -> float:
    """Linear-interpolate the frequency where gain crosses ``threshold``."""
    if g1 == g0:
        return f1
    return f0 + (threshold - g0) * (f1 - f0) / (g1 - g0)


def _walk_band_edge(
    f: np.ndarray, g: np.ndarray, start: int, step: int, threshold: float,
    *, bridge_ghz: float = 0.0,
) -> tuple[float, bool]:
    """Walk from ``start`` in direction ``step`` to the -3 dB crossing.

    With ``bridge_ghz > 0`` a sub-threshold gap narrower than ``bridge_ghz`` (the
    degenerate self-pumping notch at fs ~ fp) is skipped over rather than ending
    the band, so the returned edge sits past the notch on the far lobe. Returns
    ``(edge_ghz, clipped)`` where ``clipped`` means the sweep boundary was hit
    while still above threshold.
    """
    n = f.size
    boundary = float(f[0] if step < 0 else f[-1])
    j = start
    while 0 <= j + step <= n - 1:
        nxt = j + step
        if g[nxt] >= threshold:
            j = nxt
            continue
        crossing = _threshold_crossing(f[j], g[j], f[nxt], g[nxt], threshold)
        if bridge_ghz > 0.0:
            k = nxt
            while (0 <= k <= n - 1 and g[k] < threshold
                   and abs(f[k] - crossing) <= bridge_ghz):
                k += step
            if 0 <= k <= n - 1 and g[k] >= threshold and abs(f[k] - crossing) <= bridge_ghz:
                j = k
                continue
        return float(crossing), False
    return boundary, True


def minus3db_band(
    freq_ghz: np.ndarray,
    gain_db: np.ndarray,
    *,
    drop_db: float = 3.0,
    smooth_window_frac: float | None = None,
    polyorder: int = 3,
    bridge_ghz: float = 0.0,
) -> dict[str, Any] | None:
    """Measure the -3 dB operation band around the peak of the smoothed sweep.

    The band peak is always the maximum of the (heavily) Savitzky-Golay smoothed
    curve; the walk drops ``drop_db`` below that peak, linearly interpolating each
    crossing. The raw global-max sample is still reported separately as
    ``window_max_db`` so a razor near-fold needle stays visible without hijacking
    the band.

    ``smooth_window_frac`` (fraction of the sweep length) sets the Savitzky-Golay
    window used for the band/peak/GBP; larger fractions flatten ripple and needles
    so the band tracks the broadband envelope. The fitted curve is returned under
    the plot-only keys ``_fit_freq_ghz`` / ``_fit_gain_db``. With
    ``smooth_window_frac=None`` the raw sweep is used.

    Returns ``None`` for <2 finite samples. ``band_clipped`` flags an edge that
    ran into the sweep boundary (widen the sweep to resolve it).
    """
    f = np.asarray(freq_ghz, dtype=float).reshape(-1)
    g = np.asarray(gain_db, dtype=float).reshape(-1)
    mask = np.isfinite(f) & np.isfinite(g)
    f, g = f[mask], g[mask]
    if f.size < 2:
        return None
    order = np.argsort(f)
    f, g = f[order], g[order]
    gmax_i = int(np.argmax(g))  # raw needle, reported regardless of smoothing

    smoothed = smooth_window_frac is not None
    if smoothed:
        window = auto_savgol_window(f.size, float(smooth_window_frac), int(polyorder))
        if window >= 3:
            g_band = savgol_filter(
                g, window_length=window,
                polyorder=min(int(polyorder), window - 1), mode="interp",
            )
        else:
            g_band = g.astype(float).copy()
    else:
        g_band = g

    peak_i = int(np.argmax(g_band))  # peak of the smoothed curve
    peak = float(g_band[peak_i])
    threshold = peak - float(drop_db)
    peak_lin = 10.0 ** (peak / 10.0)

    left, clip_l = _walk_band_edge(f, g_band, peak_i, -1, threshold)
    right, clip_r = _walk_band_edge(f, g_band, peak_i, +1, threshold)
    bandwidth = float(right - left)
    result: dict[str, Any] = {
        "peak_gain_db": peak,
        "peak_freq_ghz": float(f[peak_i]),
        "band_left_ghz": left,
        "band_right_ghz": right,
        "bandwidth_ghz": bandwidth,
        "gbp_ghz": float(peak_lin * bandwidth),
        "band_clipped": clip_l or clip_r,
        "smoothed": smoothed,
        "window_max_db": float(g[gmax_i]),
        "window_max_freq_ghz": float(f[gmax_i]),
    }
    if bridge_ghz > 0.0:
        left_b, clip_lb = _walk_band_edge(f, g_band, peak_i, -1, threshold,
                                          bridge_ghz=bridge_ghz)
        right_b, clip_rb = _walk_band_edge(f, g_band, peak_i, +1, threshold,
                                           bridge_ghz=bridge_ghz)
        bandwidth_b = float(right_b - left_b)
        result.update({
            "bridged_band_left_ghz": left_b,
            "bridged_band_right_ghz": right_b,
            "bridged_bandwidth_ghz": bandwidth_b,
            "bridged_gbp_ghz": float(peak_lin * bandwidth_b),
            "bridged_band_clipped": clip_lb or clip_rb,
            "bridged": bandwidth_b > bandwidth + 1e-9,
        })
    if smoothed:
        result["_fit_freq_ghz"] = f
        result["_fit_gain_db"] = g_band
    return result


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
