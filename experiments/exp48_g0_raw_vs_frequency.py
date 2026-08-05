"""Raw G0 vs signal frequency, straight from the Themis cube.

No smoothing, no plateau median. G0 here is literally the single lowest-power
row of ``Response`` at each frequency column -- the most direct read of gain
the cube offers. Plotted against the plateau-median rule exp46 uses
(``median(response[:10, :])``) so the two definitions can be compared by eye
rather than argued about.

    python experiments/exp48_g0_raw_vs_frequency.py

A second plot overlays the ten operating-point candidates already produced by
the gain-map campaigns (five from ``outputs/Inosuisse/2c_reg``, five from
``outputs/campaign_diss/2c_base``, each its own pump frequency and design) and
ranks them by RMS difference to the measured raw G0 over their frequency
overlap. The best-RMS candidate is the one to carry into the saturation
distributions -- picking an operating point by matching G0(f), not by
assuming the exp45 pump frequency was already the right one.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp46_psat_distributions import measured_table  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from twpa_solver.plotting.metrics import auto_savgol_window  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

G0_SMOOTH_WINDOW_FRAC = 0.35
G0_SMOOTH_POLYORDER = 3
RESPONSE_SAVGOL_WINDOW = 11
RESPONSE_SAVGOL_ORDER = 2
MIN_SMOOTHED_G0_DB = 3.0

CUBE = ROOT / (
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
MEAS_PUMP_GHZ = 7.256
PUMP_EXCLUSION_GHZ = 0.15

# Two points with a hand-extracted reference G0, from
# docs/development/.../Images/Saturation_*.png.
REFERENCE = {6.571: 12.8, 6.932: 13.8}

MEAS_COLOUR = "#d1495b"

CANDIDATE_DIRS = [
    ROOT / "outputs" / "Inosuisse" / "2c_reg" / "plots" / "candidate_sweeps",
    ROOT / "outputs" / "campaign_diss" / "2c_base" / "plots" / "candidate_sweeps",
]


def load_candidates() -> list[dict]:
    """Each candidate's (frequency GHz, gain_vs_off_db, pump_ghz, label)."""
    candidates = []
    for root in CANDIDATE_DIRS:
        for run_dir in sorted(root.glob("rank_*")):
            sweep_path = run_dir / "gain_sweep.csv"
            report_path = run_dir / "gain_report.json"
            if not sweep_path.exists() or not report_path.exists():
                continue
            rows = list(csv.DictReader(sweep_path.open(encoding="utf-8")))
            frequency = np.array([float(r["signal_ghz"]) for r in rows])
            gain = np.array([float(r["gain_vs_off_db"]) for r in rows])
            pump_ghz = json.loads(report_path.read_text(encoding="utf-8"))[
                "metadata"
            ]["pump_freq_ghz"]
            candidates.append({
                "label": f"{root.parent.parent.name}/{run_dir.name}",
                "frequency_ghz": frequency,
                "gain_vs_off_db": gain,
                "pump_ghz": float(pump_ghz),
            })
    return candidates


def rank_candidates(
    candidates: list[dict], measured_ghz: np.ndarray, measured_g0: np.ndarray
) -> list[dict]:
    """RMS(candidate G0 - measured raw G0) over each candidate's own frequency
    grid, interpolating the measurement onto it -- the candidate grids differ
    (each is a 501-point sweep centred on its own pump), so a shared grid
    would need its own resampling choice for no benefit.
    """
    order = np.argsort(measured_ghz)
    meas_ghz_sorted = measured_ghz[order]
    meas_g0_sorted = measured_g0[order]

    ranked = []
    for candidate in candidates:
        freq = candidate["frequency_ghz"]
        pump = candidate["pump_ghz"]
        keep = (
            (freq >= meas_ghz_sorted[0])
            & (freq <= meas_ghz_sorted[-1])
            & (np.abs(freq - pump) > PUMP_EXCLUSION_GHZ)
            & (np.abs(freq - MEAS_PUMP_GHZ) > PUMP_EXCLUSION_GHZ)
        )
        if keep.sum() < 10:
            rms = float("nan")
        else:
            meas_on_grid = np.interp(
                freq[keep], meas_ghz_sorted, meas_g0_sorted
            )
            rms = float(
                np.sqrt(np.mean((candidate["gain_vs_off_db"][keep] - meas_on_grid) ** 2))
            )
        ranked.append({**candidate, "rms_db": rms, "n_compared": int(keep.sum())})
    ranked.sort(key=lambda c: (np.isnan(c["rms_db"]), c["rms_db"]))
    return ranked


P2DB_SMOOTH_WINDOW_FRAC = 0.35


def p1db_cut_at_p2db(
    raw: np.ndarray, power: np.ndarray, rough_g0: float,
    window_frac: float = P2DB_SMOOTH_WINDOW_FRAC,
) -> tuple[float, float]:
    """P1dB via hard savgol (order 3) on the trace truncated at P2dB, with a
    self-consistent local G0 -- returns (p1db, g0_local).

    Neither a cross-column frequency-smoothed G0 nor a single row0 sample is a
    reliable per-column threshold on its own: the frequency-smoothed version
    (window=699) can sit 1+ dB above a column's real local plateau (verified
    at f=5.7 GHz: g0_smooth=8.59 dB vs an actual ~7.4-7.5 dB plateau), which
    makes the 1 dB threshold unreachable and a persistent-crossing rule then
    locks onto the very first noisy point instead of real compression. A bare
    row0 sample is occasionally itself the noisy one (f=5.016 GHz: row0=5.77
    vs a smoothed local plateau of ~4.5-4.6). Fixed by decoupling the two uses
    of G0: ``rough_g0`` (row0 is fine here) only locates the coarse P2dB
    truncation boundary -- breakdown is a sharp crash, so a ~1 dB error there
    does not move the boundary appreciably -- then the real G0 used for the
    1 dB threshold is read off the max of the low-power third of the
    already-smoothed, already-truncated segment: local to this column, and
    averaged over the same ~30+ power points as the hard savgol window, so it
    is far more robust than either single-statistic version. Combined with
    persistent-crossing (last point above threshold, never recovering --
    matches breakdown never un-crashing in this data), this cuts
    P1dB<-110 dBm spurious columns from 99/924 (naive first-crossing) to
    15/1061 across 4-9.5 GHz.
    """
    above2 = np.flatnonzero(raw >= rough_g0 - 2.0)
    if above2.size == 0:
        return float("nan"), float("nan")
    k2 = int(above2[-1])
    seg = raw[: k2 + 1]
    seg_power = power[: k2 + 1]
    seg_window = auto_savgol_window(seg.size, window_frac, G0_SMOOTH_POLYORDER)
    if seg_window < G0_SMOOTH_POLYORDER + 2:
        return float("nan"), float("nan")
    smooth = savgol_filter(seg, seg_window, G0_SMOOTH_POLYORDER, mode="interp")
    g0_local = float(np.max(smooth[: max(5, smooth.size // 3)]))
    threshold1 = g0_local - 1.0
    above = np.flatnonzero(smooth >= threshold1)
    if above.size == 0 or above[-1] >= smooth.size - 1:
        return float("nan"), g0_local
    knee = int(above[-1])
    p1db = float(np.interp(
        threshold1, [smooth[knee + 1], smooth[knee]], [seg_power[knee + 1], seg_power[knee]]
    ))
    return p1db, g0_local


def heavily_smoothed_g0_and_p1db() -> Path:
    """G0, heavily smoothed like a gain spectrum, and the P1dB it drives.

    G0 is smoothed the same way ``scripts/plot_gain_map.py`` smooths gain
    spectra: Savitzky-Golay with a window at 35% of the trace length
    (``auto_savgol_window``, the production default), which suppresses ripple
    hard enough that the pump-exclusion gap (a downward spike to -30 dB) would
    otherwise drag the whole curve down across most of its 699-point window --
    so the gap is linearly bridged before smoothing, then re-excised from the
    plotted curve.

    The smoothed G0 gates which columns count as usable (>3 dB) -- robust
    against the noise-floor false positives a single row0 sample gives (47
    columns in the 9.6-12 GHz floor pass row0>3 dB on pure noise; zero pass on
    the smoothed curve). But it is NOT the per-column P1dB threshold: see
    ``p1db_cut_at_p2db``, which derives a self-consistent local G0 instead.
    The P1dB curve is plotted raw -- no further smoothing on top of it.
    """
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - 72.5
    response = np.asarray(data["Response"], dtype=float)
    row0 = response[0, :]

    pump_gap = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    bridged = row0.copy()
    keep = ~pump_gap
    bridged[pump_gap] = np.interp(
        frequency[pump_gap], frequency[keep], row0[keep]
    )

    window = auto_savgol_window(frequency.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER)
    g0_smooth = savgol_filter(bridged, window, G0_SMOOTH_POLYORDER, mode="interp")

    p1db = np.full(frequency.size, np.nan)
    for index in range(frequency.size):
        p1db[index], _ = p1db_cut_at_p2db(response[:, index], power, row0[index])

    usable = keep & (g0_smooth > MIN_SMOOTHED_G0_DB) & np.isfinite(p1db)

    figure, (axis_g0, axis_p1db) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axis_g0.plot(frequency, row0, color="0.75", linewidth=0.6, label="raw G0 (row 0)")
    axis_g0.plot(
        frequency[keep], g0_smooth[keep], color="#1f6feb", linewidth=2.0,
        label=f"G0, savgol window_frac={G0_SMOOTH_WINDOW_FRAC} (n={window}), order {G0_SMOOTH_POLYORDER}",
    )
    axis_g0.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis_g0.axhline(MIN_SMOOTHED_G0_DB, color="0.4", linestyle=":", linewidth=1.0,
                     label=f"usable gate ({MIN_SMOOTHED_G0_DB} dB)")
    for target, ref in REFERENCE.items():
        axis_g0.plot(target, ref, "k*", markersize=13, zorder=5)
    axis_g0.set_ylim(-15, 20)
    axis_g0.set_ylabel("G0 (dB)")
    axis_g0.set_title("Heavily smoothed G0 (gain-spectrum rule, window=35% of trace)")
    axis_g0.legend(fontsize=8, loc="upper right")
    axis_g0.grid(alpha=0.25, linestyle=":")

    axis_p1db.plot(
        frequency[usable], p1db[usable], color="#d1495b", linewidth=0.9,
        marker=".", markersize=2,
    )
    axis_p1db.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis_p1db.set_xlabel("signal frequency (GHz)")
    axis_p1db.set_ylabel("$P_{1dB}$ input (dBm)")
    axis_p1db.set_title(
        f"P1dB vs frequency, cut at P2dB + hard savgol (window_frac={P2DB_SMOOTH_WINDOW_FRAC}) "
        f"(n={int(usable.sum())} usable columns)"
    )
    axis_p1db.grid(alpha=0.25, linestyle=":")

    figure.tight_layout()
    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "g0_smoothed_and_p1db_vs_frequency.png"
    figure.savefig(png, dpi=180)
    plt.close(figure)

    print(f"\nG0 smoothing: window={window} of {frequency.size} points (frac={G0_SMOOTH_WINDOW_FRAC})")
    for target, ref in REFERENCE.items():
        i = int(np.abs(frequency - target).argmin())
        print(f"  f={frequency[i]:.3f} GHz  G0_smooth={g0_smooth[i]:6.2f}  reference={ref}")
    print(f"P1dB: n_usable={int(usable.sum())} of {frequency.size}")
    print(f"wrote {png}")
    return png


def robust_savgol_fit(
    y: np.ndarray, window_frac: float = 0.15, order: int = 3,
    n_iter: int = 6, sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative sigma-clipped Savitzky-Golay fit. Returns (fit, inlier_mask).

    Each iteration replaces the current outliers with the running fit before
    refitting (not dropped, since savgol needs a gap-free uniform sequence),
    then re-clips at ``sigma`` robust-sigma (MAD-based) residual. Outliers
    cannot drag the fit through themselves this way -- an isolated -130 dBm
    spike is replaced by its neighbourhood's fit value before the next pass
    ever sees it.
    """
    window = auto_savgol_window(y.size, window_frac, order)
    working = y.copy()
    mask = np.isfinite(y)
    working[~mask] = np.interp(
        np.flatnonzero(~mask), np.flatnonzero(mask), y[mask]
    )
    fit = savgol_filter(working, window, order, mode="interp")
    for _ in range(n_iter):
        resid = working - fit
        med = np.median(resid[mask])
        mad = np.median(np.abs(resid[mask] - med))
        robust_std = 1.4826 * mad
        new_mask = mask & (np.abs(resid - med) < sigma * max(robust_std, 1e-6))
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
        working = y.copy()
        working[~mask] = fit[~mask]
        fit = savgol_filter(working, window, order, mode="interp")
    return fit, mask


def plot_p1db_robust_fit() -> Path:
    """Robust-fit the P1dB(f) trace to knock out the remaining outliers from
    ``heavily_smoothed_g0_and_p1db``'s per-column extraction.
    """
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - 72.5
    response = np.asarray(data["Response"], dtype=float)
    row0 = response[0, :]

    pump_gap = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    bridged = row0.copy()
    keep = ~pump_gap
    bridged[pump_gap] = np.interp(frequency[pump_gap], frequency[keep], row0[keep])
    freq_window = auto_savgol_window(frequency.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER)
    g0_smooth = savgol_filter(bridged, freq_window, G0_SMOOTH_POLYORDER, mode="interp")

    p1db = np.full(frequency.size, np.nan)
    for index in range(frequency.size):
        p1db[index], _ = p1db_cut_at_p2db(response[:, index], power, row0[index])
    usable = keep & (g0_smooth > MIN_SMOOTHED_G0_DB) & np.isfinite(p1db)

    freq_u = frequency[usable]
    p1db_u = p1db[usable]
    fit, inlier = robust_savgol_fit(p1db_u)

    figure, axis = plt.subplots(figsize=(13, 6.5))
    axis.plot(freq_u, p1db_u, color="0.75", linewidth=0.7, label="P1dB, per-column (before fit)")
    axis.plot(
        freq_u[~inlier], p1db_u[~inlier], "x", color="#d1495b", markersize=5,
        markeredgewidth=1.2, label=f"rejected outliers (n={int((~inlier).sum())})",
    )
    axis.plot(freq_u, fit, color="#1f6feb", linewidth=1.8, label="robust savgol fit")
    axis.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis.set_xlabel("signal frequency (GHz)")
    axis.set_ylabel("$P_{1dB}$ input (dBm)")
    axis.set_title(
        f"P1dB vs frequency -- robust sigma-clipped savgol fit "
        f"(n={freq_u.size}, {int((~inlier).sum())} rejected)"
    )
    axis.legend(fontsize=8.5, loc="lower right")
    axis.grid(alpha=0.25, linestyle=":")

    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "p1db_robust_fit.png"
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)

    print(
        f"\nrobust P1dB fit: n={freq_u.size}  rejected={int((~inlier).sum())}  "
        f"({100*(~inlier).sum()/freq_u.size:.1f}%)"
    )
    print(f"wrote {png}")
    return png


def plot_psat_smoothed() -> Path:
    """P_sat = P1dB + G0 - 1 (output-referred 1 dB compression), both terms
    fully smoothed: G0 from the frequency-domain hard savgol, P1dB from its
    robust sigma-clipped fit.
    """
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - 72.5
    response = np.asarray(data["Response"], dtype=float)
    row0 = response[0, :]

    pump_gap = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    bridged = row0.copy()
    keep = ~pump_gap
    bridged[pump_gap] = np.interp(frequency[pump_gap], frequency[keep], row0[keep])
    freq_window = auto_savgol_window(frequency.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER)
    g0_smooth = savgol_filter(bridged, freq_window, G0_SMOOTH_POLYORDER, mode="interp")

    p1db = np.full(frequency.size, np.nan)
    for index in range(frequency.size):
        p1db[index], _ = p1db_cut_at_p2db(response[:, index], power, row0[index])
    usable = keep & (g0_smooth > MIN_SMOOTHED_G0_DB) & np.isfinite(p1db)

    freq_u = frequency[usable]
    g0_u = g0_smooth[usable]
    p1db_fit, inlier = robust_savgol_fit(p1db[usable])
    psat = p1db_fit + g0_u - 1.0

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(freq_u, psat, color="#2a9d8f", linewidth=1.8)
    axis.plot(
        freq_u[~inlier], psat[~inlier], "x", color="0.6", markersize=4,
        markeredgewidth=1.0, label=f"P1dB was an outlier here (n={int((~inlier).sum())})",
    )
    axis.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis.axhline(
        float(np.median(psat)), color="0.3", linestyle="--", linewidth=1.2,
        label=f"median={np.median(psat):.2f} dBm",
    )
    axis.set_xlabel("signal frequency (GHz)")
    axis.set_ylabel("$P_{sat}$ output (dBm)")
    axis.set_title(
        r"$P_{sat}=P_{1dB}+G_0-1$, smoothed G0 (window_frac=0.35) + robust-fit P1dB"
        f"  (n={freq_u.size})"
    )
    axis.legend(fontsize=8.5, loc="lower right")
    axis.grid(alpha=0.25, linestyle=":")

    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "psat_smoothed_vs_frequency.png"
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)

    print(
        f"\nPsat (smoothed G0 + fit P1dB): n={freq_u.size}  median={np.median(psat):.2f}  "
        f"mean={psat.mean():.2f}  std={psat.std(ddof=1):.2f}  "
        f"p5/p95={np.percentile(psat,[5,95])}"
    )
    print(f"wrote {png}")
    return png


def plot_psat_smoothed_distribution() -> Path:
    """Distribution + fit of the smoothed-G0/robust-fit-P1dB P_sat trace."""
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - 72.5
    response = np.asarray(data["Response"], dtype=float)
    row0 = response[0, :]

    pump_gap = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    bridged = row0.copy()
    keep = ~pump_gap
    bridged[pump_gap] = np.interp(frequency[pump_gap], frequency[keep], row0[keep])
    freq_window = auto_savgol_window(frequency.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER)
    g0_smooth = savgol_filter(bridged, freq_window, G0_SMOOTH_POLYORDER, mode="interp")

    p1db = np.full(frequency.size, np.nan)
    for index in range(frequency.size):
        p1db[index], _ = p1db_cut_at_p2db(response[:, index], power, row0[index])
    usable = keep & (g0_smooth > MIN_SMOOTHED_G0_DB) & np.isfinite(p1db)

    p1db_fit, _ = robust_savgol_fit(p1db[usable])
    psat = p1db_fit + g0_smooth[usable] - 1.0

    figure, axis = plt.subplots(figsize=(9, 6))
    q25, q75 = np.percentile(psat, [25, 75])
    span = max(q75 - q25, 1e-9)
    lo = max(float(psat.min()), np.median(psat) - 6.0 * span)
    hi = min(float(psat.max()), np.median(psat) + 6.0 * span)
    bins = np.linspace(lo, hi, 60)
    axis.hist(psat, bins=bins, density=True, alpha=0.5, color="#2a9d8f",
              edgecolor="none", label=f"data (n={psat.size})")

    grid = np.linspace(lo, hi, 400)
    norm_params = stats.norm.fit(psat)
    axis.plot(grid, stats.norm.pdf(grid, *norm_params), color="0.15", linewidth=1.8,
              label=f"normal  $\\mu$={norm_params[0]:.2f}  $\\sigma$={norm_params[1]:.2f}")
    skew_params = stats.skewnorm.fit(psat)
    axis.plot(grid, stats.skewnorm.pdf(grid, *skew_params), color="#d1495b",
              linewidth=1.8, linestyle="--", label=f"skew-normal  a={skew_params[0]:.2f}")
    axis.set_xlim(lo, hi)
    axis.set_xlabel("$P_{sat}$ output (dBm)")
    axis.set_ylabel("density")
    axis.set_title(
        f"Distribution of smoothed $P_{{sat}}$ (n={psat.size})\n"
        f"median={np.median(psat):.2f}  mean={psat.mean():.2f}  std={psat.std(ddof=1):.2f}"
    )
    axis.legend(fontsize=8.5)
    axis.grid(alpha=0.25, linestyle=":")

    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "psat_smoothed_distribution.png"
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)

    ks_norm = stats.kstest(psat, lambda x: stats.norm.cdf(x, *norm_params))
    ks_skew = stats.kstest(psat, lambda x: stats.skewnorm.cdf(x, *skew_params))
    print(
        f"\nsmoothed Psat distribution: n={psat.size}  median={np.median(psat):.2f}  "
        f"mean={psat.mean():.2f}  std={psat.std(ddof=1):.2f}  skew={stats.skew(psat):.3f}"
    )
    print(f"  normal fit      mu={norm_params[0]:7.2f} sigma={norm_params[1]:6.2f}  "
          f"KS stat={ks_norm.statistic:.4f} p={ks_norm.pvalue:.2e}")
    print(f"  skew-normal fit a={skew_params[0]:7.2f} loc={skew_params[1]:7.2f} "
          f"scale={skew_params[2]:6.2f}  KS stat={ks_skew.statistic:.4f} p={ks_skew.pvalue:.2e}")
    print(f"wrote {png}")
    return png


def plot_gain_curves_in_window(centre_ghz: float, half_width_ghz: float) -> Path:
    """Gain vs input power for every column in a frequency window, P1dB dotted."""
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - 72.5
    response = np.asarray(data["Response"], dtype=float)
    row0 = response[0, :]

    pump_gap = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    bridged = row0.copy()
    keep = ~pump_gap
    bridged[pump_gap] = np.interp(frequency[pump_gap], frequency[keep], row0[keep])
    window = auto_savgol_window(frequency.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER)
    g0_smooth = savgol_filter(bridged, window, G0_SMOOTH_POLYORDER, mode="interp")

    indices = np.flatnonzero(np.abs(frequency - centre_ghz) <= half_width_ghz)

    figure, (axis, axis_zoom) = plt.subplots(1, 2, figsize=(18, 7))
    colours = plt.cm.viridis(np.linspace(0, 1, indices.size))
    p1db_points: list[tuple[float, float]] = []
    for colour, index in zip(colours, indices):
        smooth_col = savgol_filter(
            response[:, index], RESPONSE_SAVGOL_WINDOW, RESPONSE_SAVGOL_ORDER
        )
        threshold = g0_smooth[index] - 1.0
        crossings = np.flatnonzero(
            (smooth_col[:-1] >= threshold) & (smooth_col[1:] < threshold)
        )
        for ax in (axis, axis_zoom):
            ax.plot(power, smooth_col, color=colour, linewidth=1.0, alpha=0.85)
        if crossings.size:
            knee = int(crossings[0])
            p1db = np.interp(
                threshold,
                [smooth_col[knee + 1], smooth_col[knee]],
                [power[knee + 1], power[knee]],
            )
            p1db_points.append((p1db, threshold))
            for ax in (axis, axis_zoom):
                ax.plot(p1db, threshold, "o", color=colour, markersize=4,
                        markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(frequency[indices].min(), frequency[indices].max()),
    )
    figure.colorbar(sm, ax=axis, label="signal frequency (GHz)")
    axis.set_xlabel("input signal power (dBm)")
    axis.set_ylabel("gain (dB)")
    axis.set_title(
        f"Gain vs power, {frequency[indices].min():.3f}-{frequency[indices].max():.3f} GHz "
        f"(n={indices.size} columns)"
    )
    axis.grid(alpha=0.25, linestyle=":")

    p1db_arr = np.array(p1db_points)
    pad_x = 0.15 * max(np.ptp(p1db_arr[:, 0]), 1e-6)
    pad_y = 0.6 * max(np.ptp(p1db_arr[:, 1]), 1e-6)
    axis_zoom.set_xlim(p1db_arr[:, 0].min() - pad_x, p1db_arr[:, 0].max() + pad_x)
    axis_zoom.set_ylim(p1db_arr[:, 1].min() - pad_y, p1db_arr[:, 1].max() + pad_y)
    axis_zoom.set_xlabel("input signal power (dBm)")
    axis_zoom.set_ylabel("gain (dB)")
    axis_zoom.set_title(f"Zoom on P1dB cluster (n={len(p1db_points)} dots)")
    axis_zoom.grid(alpha=0.25, linestyle=":")

    figure.suptitle("dots = P1dB (first crossing, smoothed-G0 threshold)", fontsize=10)
    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "gain_curves_ripple_window.png"
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)
    print(f"wrote {png}")
    return png


def plot_smoothing_check(centre_ghz: float, half_width_ghz: float) -> Path:
    """Before/after: does power-axis smoothing still misfire once post-P2dB
    (device-breakdown) data is excluded from the fit?

    P2dB (threshold = G0_smooth - 2) is located on the RAW column by the
    persistent rule (last index still above threshold, never recovering after)
    -- the collapse in this data never un-crashes, so "persistent" is the
    physical breakdown point. Everything after it is truncated before the
    savgol fit runs, so a fit window can no longer straddle the crash. The
    "before" curve is the full-range fit already used elsewhere in this file.
    """
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - 72.5
    response = np.asarray(data["Response"], dtype=float)
    row0 = response[0, :]

    pump_gap = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    bridged = row0.copy()
    keep = ~pump_gap
    bridged[pump_gap] = np.interp(frequency[pump_gap], frequency[keep], row0[keep])
    window = auto_savgol_window(frequency.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER)
    g0_smooth = savgol_filter(bridged, window, G0_SMOOTH_POLYORDER, mode="interp")

    indices = np.flatnonzero(np.abs(frequency - centre_ghz) <= half_width_ghz)

    figure, (axis_before, axis_after) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    colours = plt.cm.viridis(np.linspace(0, 1, indices.size))
    before_pts: list[tuple[float, float]] = []
    after_pts: list[tuple[float, float]] = []

    for colour, index in zip(colours, indices):
        raw = response[:, index]
        threshold1 = g0_smooth[index] - 1.0
        threshold2 = g0_smooth[index] - 2.0

        # before: fit the full 121-point trace, as elsewhere in this file.
        smooth_full = savgol_filter(raw, RESPONSE_SAVGOL_WINDOW, RESPONSE_SAVGOL_ORDER)
        crossings = np.flatnonzero(
            (smooth_full[:-1] >= threshold1) & (smooth_full[1:] < threshold1)
        )
        axis_before.plot(power, smooth_full, color=colour, linewidth=1.0, alpha=0.85)
        if crossings.size:
            knee = int(crossings[0])
            p1db = np.interp(
                threshold1, [smooth_full[knee + 1], smooth_full[knee]],
                [power[knee + 1], power[knee]],
            )
            before_pts.append((p1db, threshold1))
            axis_before.plot(p1db, threshold1, "o", color=colour, markersize=4,
                              markeredgecolor="black", markeredgewidth=0.4, zorder=5)

        # after: locate P2dB on raw (persistent rule), truncate, refit only the
        # pre-breakdown segment.
        above2 = np.flatnonzero(raw >= threshold2)
        if above2.size == 0:
            continue
        k2 = int(above2[-1])
        domain = slice(0, k2 + 1)
        seg = raw[domain]
        seg_window = min(RESPONSE_SAVGOL_WINDOW, seg.size if seg.size % 2 else seg.size - 1)
        if seg_window < RESPONSE_SAVGOL_ORDER + 2:
            continue
        smooth_trunc = savgol_filter(seg, seg_window, RESPONSE_SAVGOL_ORDER)
        crossings2 = np.flatnonzero(
            (smooth_trunc[:-1] >= threshold1) & (smooth_trunc[1:] < threshold1)
        )
        axis_after.plot(power[domain], smooth_trunc, color=colour, linewidth=1.0, alpha=0.85)
        if crossings2.size:
            knee = int(crossings2[0])
            p1db = np.interp(
                threshold1, [smooth_trunc[knee + 1], smooth_trunc[knee]],
                [power[domain][knee + 1], power[domain][knee]],
            )
            after_pts.append((p1db, threshold1))
            axis_after.plot(p1db, threshold1, "o", color=colour, markersize=4,
                             markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    for ax, title in ((axis_before, "before: full-range fit"),
                       (axis_after, "after: truncated at P2dB, refit")):
        ax.set_xlabel("input signal power (dBm)")
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle=":")
    axis_before.set_ylabel("gain (dB)")

    before_arr = np.array(before_pts)
    after_arr = np.array(after_pts) if after_pts else before_arr
    lo = min(before_arr[:, 0].min(), after_arr[:, 0].min())
    hi = max(before_arr[:, 0].max(), after_arr[:, 0].max())
    pad = 0.15 * max(hi - lo, 1e-6)
    ylo = min(before_arr[:, 1].min(), after_arr[:, 1].min())
    yhi = max(before_arr[:, 1].max(), after_arr[:, 1].max())
    ypad = 0.6 * max(yhi - ylo, 1e-6)
    for ax in (axis_before, axis_after):
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(ylo - ypad, yhi + ypad)

    figure.suptitle(
        f"Power-axis smoothing check, {frequency[indices].min():.3f}-"
        f"{frequency[indices].max():.3f} GHz -- P1dB dots "
        f"(n_before={len(before_pts)}, n_after={len(after_pts)})",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "smoothing_check_before_after.png"
    figure.savefig(png, dpi=180)
    plt.close(figure)

    print(
        f"\nsmoothing check: before spread={before_arr[:,0].max()-before_arr[:,0].min():.2f} dB  "
        f"after spread={after_arr[:,0].max()-after_arr[:,0].min():.2f} dB"
    )
    print(f"wrote {png}")
    return png


def plot_hard_smoothing_check(
    centre_ghz: float, half_width_ghz: float, window_frac: float = G0_SMOOTH_WINDOW_FRAC
) -> Path:
    """Each curve cut at P2dB, then refit with the same hard savgol rule used
    for G0 (window_frac=0.35, order 3) instead of the weak window=11 rule.
    """
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - 72.5
    response = np.asarray(data["Response"], dtype=float)
    row0 = response[0, :]

    pump_gap = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    bridged = row0.copy()
    keep = ~pump_gap
    bridged[pump_gap] = np.interp(frequency[pump_gap], frequency[keep], row0[keep])
    freq_window = auto_savgol_window(frequency.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER)
    g0_smooth = savgol_filter(bridged, freq_window, G0_SMOOTH_POLYORDER, mode="interp")

    indices = np.flatnonzero(np.abs(frequency - centre_ghz) <= half_width_ghz)

    figure, axis = plt.subplots(figsize=(13, 8))
    colours = plt.cm.viridis(np.linspace(0, 1, indices.size))
    p1db_pts: list[tuple[float, float]] = []

    for colour, index in zip(colours, indices):
        raw = response[:, index]
        threshold1 = g0_smooth[index] - 1.0
        threshold2 = g0_smooth[index] - 2.0

        above2 = np.flatnonzero(raw >= threshold2)
        if above2.size == 0:
            continue
        k2 = int(above2[-1])
        seg = raw[: k2 + 1]
        seg_power = power[: k2 + 1]

        seg_window = auto_savgol_window(seg.size, window_frac, G0_SMOOTH_POLYORDER)
        if seg_window < G0_SMOOTH_POLYORDER + 2:
            continue
        smooth_hard = savgol_filter(seg, seg_window, G0_SMOOTH_POLYORDER, mode="interp")

        axis.plot(seg_power, smooth_hard, color=colour, linewidth=1.2, alpha=0.85)
        crossings = np.flatnonzero(
            (smooth_hard[:-1] >= threshold1) & (smooth_hard[1:] < threshold1)
        )
        if crossings.size:
            knee = int(crossings[0])
            p1db = np.interp(
                threshold1, [smooth_hard[knee + 1], smooth_hard[knee]],
                [seg_power[knee + 1], seg_power[knee]],
            )
            p1db_pts.append((p1db, threshold1))
            axis.plot(p1db, threshold1, "o", color=colour, markersize=5,
                      markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    p1db_arr = np.array(p1db_pts)
    axis.set_xlim(p1db_arr[:, 0].min() - 6.0, p1db_arr[:, 0].max() + 6.0)
    axis.set_ylim(p1db_arr[:, 1].min() - 4.0, p1db_arr[:, 1].max() + 6.0)
    axis.set_xlabel("input signal power (dBm)")
    axis.set_ylabel("gain (dB)")
    axis.set_title(
        f"Hard savgol (window_frac={window_frac}, order {G0_SMOOTH_POLYORDER}) "
        f"on data cut at P2dB, {frequency[indices].min():.3f}-{frequency[indices].max():.3f} GHz  "
        f"(n={len(p1db_pts)}, P1dB spread={p1db_arr[:,0].max()-p1db_arr[:,0].min():.2f} dB)"
    )
    axis.grid(alpha=0.25, linestyle=":")

    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"hard_smoothing_cut_at_p2db_frac{window_frac:.2f}.png"
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)
    print(f"\nhard smoothing frac={window_frac}, cut at P2dB: n={len(p1db_pts)}  spread={p1db_arr[:,0].max()-p1db_arr[:,0].min():.2f} dB")
    print(f"wrote {png}")
    return png


def plot_measured_psat() -> Path:
    """Measured P_sat vs signal frequency, plus its fitted distribution.

    Uses exp46's corrected extraction (raw row-0 G0, first-crossing P1dB), so
    this reflects the G0 fix rather than the biased median-of-10 version.
    P_sat = P1dB_input + G0 - 1 (output-referred 1 dB compression), matching
    ``scripts/run_compression.py:956``.
    """
    frequency, g0, p1db, usable = measured_table("first")
    frequency, g0, p1db = frequency[usable], g0[usable], p1db[usable]
    psat = p1db + g0 - 1.0

    figure, (axis_freq, axis_hist) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    order = np.argsort(frequency)
    axis_freq.plot(
        frequency[order], psat[order], color=MEAS_COLOUR, linewidth=0.9, alpha=0.85,
    )
    axis_freq.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis_freq.axhline(
        float(np.median(psat)), color="0.3", linestyle="--", linewidth=1.2,
        label=f"median={np.median(psat):.2f} dBm",
    )
    axis_freq.set_xlabel("signal frequency (GHz)")
    axis_freq.set_ylabel("$P_{sat}$ output (dBm)")
    axis_freq.set_title(f"Measured $P_{{sat}}$ vs frequency (n={psat.size})")
    axis_freq.legend(fontsize=9)
    axis_freq.grid(alpha=0.25, linestyle=":")

    q25, q75 = np.percentile(psat, [25, 75])
    span = max(q75 - q25, 1e-9)
    lo = max(float(psat.min()), np.median(psat) - 4.0 * span)
    hi = min(float(psat.max()), np.median(psat) + 4.0 * span)
    bins = np.linspace(lo, hi, 50)
    axis_hist.hist(
        psat, bins=bins, density=True, alpha=0.5, color=MEAS_COLOUR,
        edgecolor="none", label=f"data (n={psat.size})",
    )

    grid = np.linspace(lo, hi, 400)
    norm_params = stats.norm.fit(psat)
    axis_hist.plot(
        grid, stats.norm.pdf(grid, *norm_params), color="0.15", linewidth=1.8,
        label=f"normal  $\\mu$={norm_params[0]:.2f}  $\\sigma$={norm_params[1]:.2f}",
    )
    skew_params = stats.skewnorm.fit(psat)
    axis_hist.plot(
        grid, stats.skewnorm.pdf(grid, *skew_params), color="#2a9d8f",
        linewidth=1.8, linestyle="--",
        label=f"skew-normal  a={skew_params[0]:.2f}",
    )
    axis_hist.set_xlim(lo, hi)
    axis_hist.set_xlabel("$P_{sat}$ output (dBm)")
    axis_hist.set_ylabel("density")
    axis_hist.set_title("Distribution + fit")
    axis_hist.legend(fontsize=8)
    axis_hist.grid(alpha=0.25, linestyle=":")

    figure.suptitle(
        f"Measured saturation power -- {CUBE.name}, pump {MEAS_PUMP_GHZ} GHz "
        "(corrected raw-G0 extraction)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "psat_measured_vs_frequency.png"
    figure.savefig(png, dpi=180)
    plt.close(figure)

    ks_norm = stats.kstest(psat, lambda x: stats.norm.cdf(x, *norm_params))
    ks_skew = stats.kstest(psat, lambda x: stats.skewnorm.cdf(x, *skew_params))
    print(
        f"\nmeasured P_sat: n={psat.size}  median={np.median(psat):.2f}  "
        f"mean={psat.mean():.2f}  std={psat.std(ddof=1):.2f}  "
        f"skew={stats.skew(psat):.3f}"
    )
    print(
        f"  normal fit      mu={norm_params[0]:7.2f} sigma={norm_params[1]:6.2f}  "
        f"KS stat={ks_norm.statistic:.4f} p={ks_norm.pvalue:.2e}"
    )
    print(
        f"  skew-normal fit a={skew_params[0]:7.2f} loc={skew_params[1]:7.2f} "
        f"scale={skew_params[2]:6.2f}  KS stat={ks_skew.statistic:.4f} p={ks_skew.pvalue:.2e}"
    )
    return png


def main() -> int:
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    response = np.asarray(data["Response"], dtype=float)

    g0_lowest_row = response[0, :]
    g0_median10 = np.median(response[:10, :], axis=0)

    exclude = np.abs(frequency - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ

    figure, axis = plt.subplots(figsize=(13, 5.5))
    axis.plot(
        frequency, g0_lowest_row, color="#d1495b", linewidth=0.8, alpha=0.85,
        label="raw, lowest signal power (row 0)",
    )
    axis.plot(
        frequency, g0_median10, color="#1f6feb", linewidth=1.3,
        label="median of 10 lowest rows (exp46 rule)",
    )
    axis.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15, label=f"pump exclusion (+-{PUMP_EXCLUSION_GHZ} GHz)",
    )
    for target, ref_g0 in REFERENCE.items():
        axis.plot(target, ref_g0, "k*", markersize=14, zorder=5)
        axis.annotate(
            f"ref {ref_g0:.1f} dB", (target, ref_g0),
            textcoords="offset points", xytext=(6, 8), fontsize=8,
        )

    axis.set_xlim(frequency.min(), frequency.max())
    axis.set_ylim(-15, 20)
    axis.set_xlabel("signal frequency (GHz)")
    axis.set_ylabel("gain (dB)")
    axis.set_title(
        f"Raw G0 vs signal frequency -- {CUBE.name}, pump {MEAS_PUMP_GHZ} GHz"
    )
    axis.legend(fontsize=9, loc="upper right")
    axis.grid(alpha=0.25, linestyle=":")

    outdir = ROOT / "outputs" / "presentation"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "g0_raw_vs_frequency.png"
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)

    print(f"n columns: {frequency.size}  (pump-excluded: {int(exclude.sum())})")
    print(
        f"raw row0   G0: min={g0_lowest_row.min():6.2f} "
        f"max={g0_lowest_row.max():6.2f} median={np.median(g0_lowest_row):6.2f}"
    )
    print(
        f"median10   G0: min={g0_median10.min():6.2f} "
        f"max={g0_median10.max():6.2f} median={np.median(g0_median10):6.2f}"
    )
    for target, ref_g0 in REFERENCE.items():
        i = int(np.abs(frequency - target).argmin())
        print(
            f"f={frequency[i]:.3f} GHz  raw_row0={g0_lowest_row[i]:6.2f}  "
            f"median10={g0_median10[i]:6.2f}  reference={ref_g0:6.2f}"
        )
    print(f"\nwrote {png}")

    smoothed_png = heavily_smoothed_g0_and_p1db()

    robust_png = plot_p1db_robust_fit()

    psat_smoothed_png = plot_psat_smoothed()

    psat_dist_png = plot_psat_smoothed_distribution()

    ripple_png = plot_gain_curves_in_window(9.2, 0.05)

    check_png = plot_smoothing_check(9.2, 0.05)

    plot_hard_smoothing_check(9.2, 0.05, window_frac=P2DB_SMOOTH_WINDOW_FRAC)

    psat_png = plot_measured_psat()
    print(f"wrote {psat_png}")

    candidates = load_candidates()
    if not candidates:
        print("\nno candidate_sweeps found; skipping candidate comparison")
        return 0

    ranked = rank_candidates(candidates, frequency, g0_lowest_row)

    figure2, axis2 = plt.subplots(figsize=(13, 6))
    axis2.plot(
        frequency, g0_lowest_row, color="0.55", linewidth=0.7, alpha=0.7,
        label="measured, raw G0 (row 0)", zorder=1,
    )
    colours = plt.cm.tab10(np.linspace(0, 1, len(ranked)))
    for colour, candidate in zip(colours, ranked):
        style = "-" if candidate["rms_db"] == ranked[0]["rms_db"] else "--"
        width = 2.2 if candidate["rms_db"] == ranked[0]["rms_db"] else 1.1
        axis2.plot(
            candidate["frequency_ghz"], candidate["gain_vs_off_db"],
            style, color=colour, linewidth=width,
            label=f"{candidate['label']}  RMS={candidate['rms_db']:.2f} dB",
        )
    axis2.set_xlim(4.0, 11.0)
    axis2.set_ylim(-15, 20)
    axis2.set_xlabel("signal frequency (GHz)")
    axis2.set_ylabel("gain (dB)")
    axis2.set_title(
        "Candidate operating points vs measured raw G0 -- "
        f"best: {ranked[0]['label']} (RMS={ranked[0]['rms_db']:.2f} dB)"
    )
    axis2.legend(fontsize=7.5, loc="upper left", ncol=2)
    axis2.grid(alpha=0.25, linestyle=":")

    png2 = outdir / "g0_candidate_ranking.png"
    figure2.tight_layout()
    figure2.savefig(png2, dpi=180)
    plt.close(figure2)

    print("\ncandidate ranking (RMS vs measured raw G0, pump-excluded both sides):")
    for candidate in ranked:
        print(
            f"  {candidate['label']:<40} pump={candidate['pump_ghz']:7.4f} GHz  "
            f"RMS={candidate['rms_db']:6.3f} dB  n={candidate['n_compared']}"
        )
    best = ranked[0]
    print(
        f"\nbest operating point: {best['label']} "
        f"(pump {best['pump_ghz']:.4f} GHz, RMS {best['rms_db']:.3f} dB)"
    )
    print(f"wrote {png2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
