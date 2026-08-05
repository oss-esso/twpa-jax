"""Measured saturation-power pipeline: Themis cube -> G0 -> P1dB -> P_sat.

Standalone script -- does NOT import anything from ``experiments/`` (that
folder is scheduled for removal). It is the durable, documented replacement
for the exploratory work done interactively against
``experiments/exp46_psat_distributions.py`` and
``experiments/exp48_g0_raw_vs_frequency.py``; every threshold and window size
below came from real sweeps run against this data, not guesses -- see each
function's docstring for the specific numbers that justified it.

Source data is a single-pump-frequency Themis transmission cube: one pump
frequency and power, 2001 signal-frequency columns (4-12 GHz, 4 MHz step),
121 signal-power rows per column. ``Response`` is gain in dB.

Pipeline (each stage writes one PNG to ``--outdir``, default
``outputs/presentation``):

  1. Raw G0 vs frequency               -> g0_raw_vs_frequency.png
  2. Heavily smoothed G0 + P1dB(f)     -> g0_smoothed_and_p1db_vs_frequency.png
  3. Robust sigma-clipped P1dB fit     -> p1db_robust_fit.png
  4. Smoothed P_sat vs frequency       -> psat_smoothed_vs_frequency.png
  5. Smoothed P_sat distribution + fit -> psat_smoothed_distribution.png

Two diagnostic plots from the same pipeline, useful for re-validating the
P1dB extraction on a small frequency window rather than trusting the full
band blind:

  6. Gain-vs-power curves in a ripple window -> gain_curves_ripple_window.png
  7. Hard-savgol-cut-at-P2dB validation       -> hard_smoothing_cut_at_p2db_frac0.35.png

Model-side pipeline (only runs when ``--model-dir`` is given -- a
``scripts/run_compression.py --n-signal-freq`` sweep, e.g.
``outputs/exp45_2c_p1db_vs_frequency_op7p379``, at the map-matched operating
point whose small-signal gain lines up with the measured peak G0):

  8. Model P_sat vs frequency               -> psat_model_vs_frequency.png
  9. Model P_sat distribution + fit         -> psat_model_distribution.png
  10. Measured vs model, fitted curves only,
      two panels (distribution / vs freq)   -> psat_measured_vs_model_distribution.png

Usage:

    python scripts/measured_psat_pipeline.py
    python scripts/measured_psat_pipeline.py --model-dir outputs/exp45_2c_p1db_vs_frequency_op7p379
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

from twpa_solver.loss import pump_line_loss_model, signal_line_loss_model
from twpa_solver.ports import port_available_power_w

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CUBE_PATH = ROOT / (
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
DEFAULT_OUTDIR = ROOT / "outputs" / "presentation"

# On-chip signal power = instrument SignalPower - signal_line_loss_model()
# evaluated at the column's own frequency (docs/development/loss_B1.csv, RMS
# 2.80e-5 dB). Replaces a fabricated flat 72.5 dB that erased a real 5.95 dB
# tilt across the band and was off by +9.4 to +15.3 dB across it (see
# docs/development/psat_comparison_fix_plan.md Phase 2).
#
# The pump line uses loss_A10 (signal_line_loss_model's sibling on the pump
# feedline), not loss_B1 -- forced by energy conservation, not a fit choice:
# P_sat + 3 dB <= P_pump must hold, and only loss_A10 (on-chip pump
# -55.54 dBm at MEAS_PUMP_GHZ) satisfies it.
# Cube's own pump frequency; the +-0.15 GHz window around it is excluded from
# every stage (the VNA response there is dominated by pump leakage, not gain).
MEAS_PUMP_GHZ = 7.256
PUMP_EXCLUSION_GHZ = 0.15
# Instrument-side pump drive power (dBm), read off the cube's own "PumpPower"
# field (-21 for 105C5_7.256GHz.npy). On-chip pump power is this minus
# pump_line_loss_model() at MEAS_PUMP_GHZ.
MEAS_PUMP_INSTRUMENT_DBM = -21.0
# A column only counts toward P1dB/P_sat once its smoothed G0 clears this --
# below it the "1 dB compression point" isn't a meaningful concept.
MIN_SMOOTHED_G0_DB = 3.0

# Frequency-domain G0 smoothing: matches scripts/plot_gain_map.py's
# --sweep-smooth-frac default (0.35) -- the repo's established convention for
# suppressing ripple in a gain spectrum. polyorder=3 so the smoothing doesn't
# flatten the compression knee shape along with the ripple.
G0_SMOOTH_WINDOW_FRAC = 0.35
G0_SMOOTH_POLYORDER = 3

# Power-domain smoothing for the per-column P1dB extraction, applied to the
# trace AFTER it is truncated at P2dB (see p1db_cut_at_p2db). Found by
# sweeping window_frac in {0.10, 0.15, 0.20, 0.25, 0.35, 0.5, 0.7} against a
# 25-column test window (9.152-9.248 GHz) and measuring the P1dB spread across
# those 25 columns: 0.25 gave the numerical minimum (1.73 dB) but a visible
# ripple artifact still survived in the lightest-gain curves there; 0.35 is
# the smallest window that removes the ripple outright (1.82 dB) rather than
# one that merely dodges it in that one metric. 0.5+ overshoots and bends the
# real knee shape -- spread climbs back to 3.3-5.1 dB.
P2DB_SMOOTH_WINDOW_FRAC = 0.35

# Robust iterative fit of P1dB(f), to knock out the columns where the
# per-column extraction above still lands on a spurious dip (concentrated
# near the pump-exclusion edge and near the G0 gate boundary -- physically
# marginal SNR regions, not a bug worth chasing further at the per-column
# level). Measured on the full ~1022-usable-column trace: rejects 116 points
# (11.4%).
ROBUST_FIT_WINDOW_FRAC = 0.15
ROBUST_FIT_ORDER = 3
ROBUST_FIT_N_ITER = 6
ROBUST_FIT_SIGMA = 3.0

# Diagnostic-plot-only smoothing (stage 6's "before" curve): the original,
# too-weak fixed-window rule this pipeline replaced. Kept only so the
# diagnostic plots can show what changed.
WEAK_RESPONSE_SAVGOL_WINDOW = 11
WEAK_RESPONSE_SAVGOL_ORDER = 2

# Two points with a hand-extracted reference G0, read directly off
# docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK/Images/
# Saturation_6.571GHz.png and Saturation_6.932GHz.png. Used only to sanity
# check the raw-row0 G0 extraction in stage 1 (reproduces both to <0.2 dB; a
# median-of-the-10-lowest-power-rows rule, an earlier and now-superseded
# approach, read ~0.2-0.3 dB low at these two points).
REFERENCE_POINTS: dict[float, float] = {6.571: 12.8, 6.932: 13.8}

MEASURED_COLOUR = "#d1495b"
FIT_COLOUR = "#1f6feb"
PSAT_COLOUR = "#2a9d8f"
MODEL_COLOUR = "#8a4fc9"

# Port impedance for the Norton on-chip power conversion (port_available_power_w),
# used by load_model_sweep_from_points -- see docs/development/psat_comparison_fix_plan.md
# Phase 1: designs/ipm_2c_fixed's G matrix has G[port,port] == 1/Z0_OHM at all
# four ports, exactly.
Z0_OHM = 50.0
# Ratio-of-outgoing-pump-power disagreement beyond which the energy-accounted
# depletion estimate and the solver's own pump_depletion_all_port_db field are
# treated as inconsistent (Phase 7 cross-check).
DEPLETION_CROSS_CHECK_RATIO = 2.0


def auto_savgol_window(n: int, window_frac: float, polyorder: int) -> int:
    """Choose an odd Savitzky-Golay window valid for a trace of length n.

    Reimplemented locally (matches twpa_solver.plotting.metrics.auto_savgol_window)
    so this script has no dependency on the production package staying stable --
    it is meant to survive on its own after experiments/ is deleted.
    """
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


@dataclass
class MeasurementCube:
    """Raw arrays loaded from the Themis .npy cube.

    The signal line's attenuation is frequency-dependent (``loss_B1``), so
    on-chip signal power is NOT a single (n_power,) axis shared by every
    frequency column -- it is ``instrument_power_dbm - signal_attenuation_db
    [column]``, a per-column rigid shift of the instrument power axis. Use
    ``on_chip_power_dbm(column_index)`` rather than reading a flat power axis.
    """

    frequency_ghz: np.ndarray  # (n_freq,)
    instrument_power_dbm: np.ndarray  # (n_power,), pre-attenuation
    signal_attenuation_db: np.ndarray  # (n_freq,), loss_B1 at each column
    response_db: np.ndarray  # (n_power, n_freq), gain
    row0: np.ndarray  # (n_freq,), response_db[0, :]
    on_chip_pump_dbm: float  # instrument pump power - pump_line_loss_model()

    def on_chip_power_dbm(self, column_index: int) -> np.ndarray:
        """On-chip signal power (dBm) for one frequency column, (n_power,)."""
        return self.instrument_power_dbm - self.signal_attenuation_db[column_index]


def load_cube(cube_path: Path) -> MeasurementCube:
    data = np.load(cube_path, allow_pickle=True).item()
    frequency_ghz = np.asarray(data["Frequency"], dtype=float) / 1e9
    instrument_power_dbm = np.asarray(data["SignalPower"], dtype=float)
    signal_attenuation_db = np.asarray(
        signal_line_loss_model().attenuation_db(frequency_ghz), dtype=float
    )
    response_db = np.asarray(data["Response"], dtype=float)
    on_chip_pump_dbm = MEAS_PUMP_INSTRUMENT_DBM - float(
        pump_line_loss_model().attenuation_db(MEAS_PUMP_GHZ)
    )
    return MeasurementCube(
        frequency_ghz=frequency_ghz,
        instrument_power_dbm=instrument_power_dbm,
        signal_attenuation_db=signal_attenuation_db,
        response_db=response_db,
        row0=response_db[0, :],
        on_chip_pump_dbm=on_chip_pump_dbm,
    )


@dataclass
class SmoothedG0:
    """Frequency-domain smoothed G0 and the mask of non-pump-gap columns."""

    values: np.ndarray  # (n_freq,), only valid where keep is True
    keep: np.ndarray  # (n_freq,) bool, False inside the pump-exclusion gap
    window: int


def compute_smoothed_g0(cube: MeasurementCube) -> SmoothedG0:
    """G0 smoothed the same way a gain spectrum is smoothed elsewhere in this
    repo (see G0_SMOOTH_WINDOW_FRAC docstring above).

    The pump-exclusion gap is linearly bridged before smoothing -- otherwise
    the -30 dB dip there would drag the whole ~700-point window down across
    most of the band -- then re-excised via the returned ``keep`` mask.
    """
    pump_gap = np.abs(cube.frequency_ghz - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ
    keep = ~pump_gap
    bridged = cube.row0.copy()
    bridged[pump_gap] = np.interp(
        cube.frequency_ghz[pump_gap], cube.frequency_ghz[keep], cube.row0[keep]
    )
    window = auto_savgol_window(
        cube.frequency_ghz.size, G0_SMOOTH_WINDOW_FRAC, G0_SMOOTH_POLYORDER
    )
    values = savgol_filter(bridged, window, G0_SMOOTH_POLYORDER, mode="interp")
    return SmoothedG0(values=values, keep=keep, window=window)


def p1db_cut_at_p2db(
    raw_column: np.ndarray,
    power_dbm: np.ndarray,
    rough_g0: float,
    window_frac: float = P2DB_SMOOTH_WINDOW_FRAC,
) -> tuple[float, float]:
    """P1dB for one frequency column. Returns (p1db_dbm, g0_local_db).

    Both NaN if the column never reaches ``rough_g0 - 2`` or the truncated
    segment is too short to smooth.

    Three ideas fixed the false-crossing spikes this session, each verified
    against a specific failing column:

    1. Truncate at P2dB before fitting. Locate the breakdown boundary on the
       RAW column as the last index where ``raw >= rough_g0 - 2`` (breakdown
       never un-crashes in this data, so "last point still above" is the true
       collapse point); drop everything after it. The amplifier model is not
       valid past breakdown, and a fit window straddling the crash corrupts
       the knee shape on the good side of it.

    2. Hard-smooth the truncated segment (window_frac=0.35, order 3) instead
       of a weak fixed window. A small fixed window (e.g. 11 of ~121 points)
       leaves enough residual ripple that a threshold sitting near the flat
       low-power plateau gets crossed by noise, not real compression.

    3. Derive G0 for the -1 dB threshold SELF-CONSISTENTLY from this same
       smoothed, truncated segment (max of its low-power third) rather than
       from ``rough_g0`` or the cross-column frequency-smoothed G0. Neither of
       those is reliable as a per-column threshold on its own:
       - the frequency-smoothed G0 can sit over 1 dB above a column's real
         local plateau (verified failure at f=5.7 GHz: g0_smooth=8.59 dB vs
         an actual ~7.4-7.5 dB plateau -- the 1 dB threshold then becomes
         unreachable and any crossing rule locks onto the first noisy point);
       - a bare single row0 sample is occasionally itself the noisy outlier
         (verified failure at f=5.016 GHz: row0=5.77 dB vs a smoothed local
         plateau of ~4.5-4.6 dB).
       ``rough_g0`` (row0 is fine here) is only used for step 1's coarse
       truncation boundary -- a ~1 dB error there does not move a sharp
       breakdown crash appreciably, so it does not need to be precise.

    P1dB itself is read by the PERSISTENT crossing rule (last index still at
    or above the threshold, with recovery after it): real compression never
    un-crashes in this data, so first-crossing spuriously locks onto ripple
    in the flat plateau whenever gain sits only marginally above the gate;
    persistent structurally cannot land on a transient dip.

    Combined, these three fixes cut the count of P1dB < -110 dBm spurious
    columns (a clear tell that the extraction hit noise, not physics) from
    99 of 924 (naive first-crossing on a weakly smoothed trace) to 15 of 1061
    across the well-populated 4-9.5 GHz range.
    """
    above_p2db = np.flatnonzero(raw_column >= rough_g0 - 2.0)
    if above_p2db.size == 0:
        return float("nan"), float("nan")
    breakdown_index = int(above_p2db[-1])

    segment = raw_column[: breakdown_index + 1]
    segment_power = power_dbm[: breakdown_index + 1]
    segment_window = auto_savgol_window(
        segment.size, window_frac, G0_SMOOTH_POLYORDER
    )
    if segment_window < G0_SMOOTH_POLYORDER + 2:
        return float("nan"), float("nan")
    smooth = savgol_filter(segment, segment_window, G0_SMOOTH_POLYORDER, mode="interp")

    g0_local = float(np.max(smooth[: max(5, smooth.size // 3)]))
    threshold_1db = g0_local - 1.0

    above_threshold = np.flatnonzero(smooth >= threshold_1db)
    if above_threshold.size == 0 or above_threshold[-1] >= smooth.size - 1:
        return float("nan"), g0_local
    knee = int(above_threshold[-1])
    p1db = float(np.interp(
        threshold_1db,
        [smooth[knee + 1], smooth[knee]],
        [segment_power[knee + 1], segment_power[knee]],
    ))
    return p1db, g0_local


@dataclass
class P1dBResult:
    """Per-column P1dB and the usable-column mask, aligned to cube frequency."""

    p1db_dbm: np.ndarray  # (n_freq,), NaN where not extractable
    g0_local_db: np.ndarray  # (n_freq,), NaN where not extractable
    usable: np.ndarray  # (n_freq,) bool


def compute_p1db_per_column(cube: MeasurementCube, g0: SmoothedG0) -> P1dBResult:
    n_freq = cube.frequency_ghz.size
    p1db_dbm = np.full(n_freq, np.nan)
    g0_local_db = np.full(n_freq, np.nan)
    for index in range(n_freq):
        p1db_dbm[index], g0_local_db[index] = p1db_cut_at_p2db(
            cube.response_db[:, index],
            cube.on_chip_power_dbm(index),
            cube.row0[index],
        )
    usable = g0.keep & (g0.values > MIN_SMOOTHED_G0_DB) & np.isfinite(p1db_dbm)
    return P1dBResult(p1db_dbm=p1db_dbm, g0_local_db=g0_local_db, usable=usable)


def robust_savgol_fit(
    y: np.ndarray,
    window_frac: float = ROBUST_FIT_WINDOW_FRAC,
    order: int = ROBUST_FIT_ORDER,
    n_iter: int = ROBUST_FIT_N_ITER,
    sigma: float = ROBUST_FIT_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative sigma-clipped Savitzky-Golay fit. Returns (fit, inlier_mask).

    Each iteration replaces the current outliers' values with the running
    fit's value at that index before refitting -- points are never dropped,
    since savgol needs a gap-free uniformly sampled sequence -- then re-clips
    at ``sigma`` robust-sigma (MAD-based, so a handful of -130 dBm spikes
    cannot inflate the threshold that is supposed to catch them). Stops early
    if the inlier mask stops changing between iterations.
    """
    window = auto_savgol_window(y.size, window_frac, order)
    working = y.copy()
    finite = np.isfinite(y)
    working[~finite] = np.interp(
        np.flatnonzero(~finite), np.flatnonzero(finite), y[finite]
    )
    fit = savgol_filter(working, window, order, mode="interp")
    mask = finite
    for _ in range(n_iter):
        residual = working - fit
        median_residual = np.median(residual[mask])
        mad = np.median(np.abs(residual[mask] - median_residual))
        robust_sigma = 1.4826 * mad
        new_mask = mask & (
            np.abs(residual - median_residual) < sigma * max(robust_sigma, 1e-6)
        )
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
        working = y.copy()
        working[~mask] = fit[~mask]
        fit = savgol_filter(working, window, order, mode="interp")
    return fit, mask


def _save_figure(figure: plt.Figure, outdir: Path, filename: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / filename
    figure.tight_layout()
    figure.savefig(png, dpi=180)
    plt.close(figure)
    print(f"wrote {png}")
    return png


def stage1_raw_g0_plot(cube: MeasurementCube, outdir: Path) -> Path:
    """Raw G0 vs frequency: response[0, :], validated against REFERENCE_POINTS."""
    figure, axis = plt.subplots(figsize=(13, 5.5))
    axis.plot(
        cube.frequency_ghz, cube.row0, color=MEASURED_COLOUR, linewidth=0.8,
        alpha=0.85, label="raw G0 (row 0, lowest signal power)",
    )
    axis.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15, label=f"pump exclusion (+-{PUMP_EXCLUSION_GHZ} GHz)",
    )
    for target_ghz, reference_g0 in REFERENCE_POINTS.items():
        axis.plot(target_ghz, reference_g0, "k*", markersize=14, zorder=5)
        axis.annotate(
            f"ref {reference_g0:.1f} dB", (target_ghz, reference_g0),
            textcoords="offset points", xytext=(6, 8), fontsize=8,
        )
    axis.set_xlim(cube.frequency_ghz.min(), cube.frequency_ghz.max())
    axis.set_ylim(-15, 20)
    axis.set_xlabel("signal frequency (GHz)")
    axis.set_ylabel("gain (dB)")
    axis.set_title(f"Raw G0 vs signal frequency -- pump {MEAS_PUMP_GHZ} GHz")
    axis.legend(fontsize=9, loc="upper right")
    axis.grid(alpha=0.25, linestyle=":")

    print(f"raw G0: min={cube.row0.min():.2f} max={cube.row0.max():.2f} "
          f"median={np.median(cube.row0):.2f}")
    for target_ghz, reference_g0 in REFERENCE_POINTS.items():
        index = int(np.abs(cube.frequency_ghz - target_ghz).argmin())
        print(f"  f={cube.frequency_ghz[index]:.3f} GHz  "
              f"raw_row0={cube.row0[index]:.2f}  reference={reference_g0:.2f}")

    return _save_figure(figure, outdir, "g0_raw_vs_frequency.png")


def stage2_smoothed_g0_and_p1db_plot(
    cube: MeasurementCube, g0: SmoothedG0, p1db: P1dBResult, outdir: Path
) -> Path:
    """Two-panel: heavily smoothed G0 (gates usability), P1dB(f) it drives."""
    figure, (axis_g0, axis_p1db) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axis_g0.plot(cube.frequency_ghz, cube.row0, color="0.75", linewidth=0.6,
                 label="raw G0 (row 0)")
    axis_g0.plot(
        cube.frequency_ghz[g0.keep], g0.values[g0.keep], color=FIT_COLOUR,
        linewidth=2.0,
        label=f"G0, savgol window_frac={G0_SMOOTH_WINDOW_FRAC} "
              f"(n={g0.window}), order {G0_SMOOTH_POLYORDER}",
    )
    axis_g0.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis_g0.axhline(
        MIN_SMOOTHED_G0_DB, color="0.4", linestyle=":", linewidth=1.0,
        label=f"usable gate ({MIN_SMOOTHED_G0_DB} dB)",
    )
    for target_ghz, reference_g0 in REFERENCE_POINTS.items():
        axis_g0.plot(target_ghz, reference_g0, "k*", markersize=13, zorder=5)
    axis_g0.set_ylim(-15, 20)
    axis_g0.set_ylabel("G0 (dB)")
    axis_g0.set_title("Heavily smoothed G0 (gain-spectrum rule, window=35% of trace)")
    axis_g0.legend(fontsize=8, loc="upper right")
    axis_g0.grid(alpha=0.25, linestyle=":")

    axis_p1db.plot(
        cube.frequency_ghz[p1db.usable], p1db.p1db_dbm[p1db.usable],
        color=MEASURED_COLOUR, linewidth=0.9, marker=".", markersize=2,
    )
    axis_p1db.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis_p1db.set_xlabel("signal frequency (GHz)")
    axis_p1db.set_ylabel("$P_{1dB}$ input (dBm)")
    axis_p1db.set_title(
        f"P1dB vs frequency, cut at P2dB + hard savgol "
        f"(window_frac={P2DB_SMOOTH_WINDOW_FRAC}) "
        f"(n={int(p1db.usable.sum())} usable columns)"
    )
    axis_p1db.grid(alpha=0.25, linestyle=":")

    print(f"G0 smoothing window={g0.window} of {cube.frequency_ghz.size} points "
          f"(frac={G0_SMOOTH_WINDOW_FRAC})")
    for target_ghz, reference_g0 in REFERENCE_POINTS.items():
        index = int(np.abs(cube.frequency_ghz - target_ghz).argmin())
        print(f"  f={cube.frequency_ghz[index]:.3f} GHz  "
              f"G0_smooth={g0.values[index]:.2f}  reference={reference_g0:.2f}")
    print(f"P1dB: n_usable={int(p1db.usable.sum())} of {cube.frequency_ghz.size}")

    return _save_figure(figure, outdir, "g0_smoothed_and_p1db_vs_frequency.png")


def stage3_robust_p1db_fit_plot(
    cube: MeasurementCube, p1db: P1dBResult, outdir: Path
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Robust-fit P1dB(f). Returns (png, freq_usable, p1db_fit, inlier_mask)."""
    freq_usable = cube.frequency_ghz[p1db.usable]
    p1db_usable = p1db.p1db_dbm[p1db.usable]
    fit, inlier = robust_savgol_fit(p1db_usable)

    figure, axis = plt.subplots(figsize=(13, 6.5))
    axis.plot(freq_usable, p1db_usable, color="0.75", linewidth=0.7,
              label="P1dB, per-column (before fit)")
    axis.plot(
        freq_usable[~inlier], p1db_usable[~inlier], "x", color=MEASURED_COLOUR,
        markersize=5, markeredgewidth=1.2,
        label=f"rejected outliers (n={int((~inlier).sum())})",
    )
    axis.plot(freq_usable, fit, color=FIT_COLOUR, linewidth=1.8,
              label="robust savgol fit")
    axis.axvspan(
        MEAS_PUMP_GHZ - PUMP_EXCLUSION_GHZ, MEAS_PUMP_GHZ + PUMP_EXCLUSION_GHZ,
        color="0.5", alpha=0.15,
    )
    axis.set_xlabel("signal frequency (GHz)")
    axis.set_ylabel("$P_{1dB}$ input (dBm)")
    axis.set_title(
        f"P1dB vs frequency -- robust sigma-clipped savgol fit "
        f"(n={freq_usable.size}, {int((~inlier).sum())} rejected)"
    )
    axis.legend(fontsize=8.5, loc="lower right")
    axis.grid(alpha=0.25, linestyle=":")

    n_rejected = int((~inlier).sum())
    print(f"robust P1dB fit: n={freq_usable.size}  rejected={n_rejected}  "
          f"({100 * n_rejected / freq_usable.size:.1f}%)")

    png = _save_figure(figure, outdir, "p1db_robust_fit.png")
    return png, freq_usable, fit, inlier


def stage4_psat_vs_frequency_plot(
    freq_usable: np.ndarray,
    g0_local_usable: np.ndarray,
    g0_smooth_usable: np.ndarray,
    p1db_fit: np.ndarray,
    inlier: np.ndarray,
    outdir: Path,
) -> tuple[Path, np.ndarray]:
    """P_sat = P1dB_fit + g0_local - 1 vs frequency. Returns (png, psat).

    P1dB is DEFINED by ``G(P1dB) = g0_local - 1`` (p1db_cut_at_p2db's own
    per-column threshold), so the output-referred point is
    ``P1dB + g0_local - 1`` identically -- using ``G0_smooth`` (the
    frequency-smoothed spectrum) here instead mixes two different gain
    estimates, so the "added gain" in the sum is not the gain the compression
    point was actually measured against. ``G0_smooth`` keeps exactly one role
    elsewhere: the ``MIN_SMOOTHED_G0_DB`` usability gate in
    ``compute_p1db_per_column``. Both variants are printed once, below, so the
    size of the mixing bug this replaces is on record.
    """
    psat = p1db_fit + g0_local_usable - 1.0
    psat_mixed = p1db_fit + g0_smooth_usable - 1.0
    mixing_delta = psat - psat_mixed

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(freq_usable, psat, color=PSAT_COLOUR, linewidth=1.8)
    axis.plot(
        freq_usable[~inlier], psat[~inlier], "x", color="0.6", markersize=4,
        markeredgewidth=1.0,
        label=f"P1dB was an outlier here (n={int((~inlier).sum())})",
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
        r"$P_{sat}=P_{1dB}+g_{0,local}-1$, self-consistent per-column G0 "
        f"+ robust-fit P1dB  (n={freq_usable.size})"
    )
    axis.legend(fontsize=8.5, loc="lower right")
    axis.grid(alpha=0.25, linestyle=":")

    print(f"Psat (g0_local + fit P1dB, self-consistent): n={freq_usable.size}  "
          f"median={np.median(psat):.2f}  mean={psat.mean():.2f}  "
          f"std={psat.std(ddof=1):.2f}  "
          f"p5/p95={np.percentile(psat, [5, 95])}")
    print(f"Psat (G0_smooth + fit P1dB, mixed-estimator -- superseded): "
          f"median={np.median(psat_mixed):.2f}  "
          f"delta vs self-consistent: median={np.median(mixing_delta):+.2f}  "
          f"min={mixing_delta.min():+.2f}  max={mixing_delta.max():+.2f} dB")

    png = _save_figure(figure, outdir, "psat_smoothed_vs_frequency.png")
    return png, psat


def stage5_psat_distribution_plot(psat: np.ndarray, outdir: Path) -> Path:
    """Distribution + normal/skew-normal fit of the smoothed P_sat trace.

    KS-rejects both fits on the full cube (p ~ 1e-7 to 1e-9): the histogram
    has a comb/spike structure because both G0 and P1dB are smoothed in
    frequency, so neighbouring columns are highly correlated, not independent
    draws. That is reported honestly below rather than claimed as a clean fit
    -- the KS test is doing exactly what it should here.
    """
    figure, axis = plt.subplots(figsize=(9, 6))
    q25, q75 = np.percentile(psat, [25, 75])
    span = max(q75 - q25, 1e-9)
    lo = max(float(psat.min()), float(np.median(psat) - 6.0 * span))
    hi = min(float(psat.max()), float(np.median(psat) + 6.0 * span))
    bins = np.linspace(lo, hi, 60)
    axis.hist(psat, bins=bins, density=True, alpha=0.5, color=PSAT_COLOUR,
              edgecolor="none", label=f"data (n={psat.size})")

    grid = np.linspace(lo, hi, 400)
    normal_params = stats.norm.fit(psat)
    axis.plot(
        grid, stats.norm.pdf(grid, *normal_params), color="0.15", linewidth=1.8,
        label=f"normal  $\\mu$={normal_params[0]:.2f}  $\\sigma$={normal_params[1]:.2f}",
    )
    skew_params = stats.skewnorm.fit(psat)
    axis.plot(
        grid, stats.skewnorm.pdf(grid, *skew_params), color=MEASURED_COLOUR,
        linewidth=1.8, linestyle="--", label=f"skew-normal  a={skew_params[0]:.2f}",
    )
    axis.set_xlim(lo, hi)
    axis.set_xlabel("$P_{sat}$ output (dBm)")
    axis.set_ylabel("density")
    axis.set_title(
        f"Distribution of smoothed $P_{{sat}}$ (n={psat.size})\n"
        f"median={np.median(psat):.2f}  mean={psat.mean():.2f}  "
        f"std={psat.std(ddof=1):.2f}"
    )
    axis.legend(fontsize=8.5)
    axis.grid(alpha=0.25, linestyle=":")

    ks_normal = stats.kstest(psat, lambda x: stats.norm.cdf(x, *normal_params))
    ks_skew = stats.kstest(psat, lambda x: stats.skewnorm.cdf(x, *skew_params))
    print(f"smoothed Psat distribution: n={psat.size}  median={np.median(psat):.2f}  "
          f"mean={psat.mean():.2f}  std={psat.std(ddof=1):.2f}  "
          f"skew={stats.skew(psat):.3f}")
    print(f"  normal fit      mu={normal_params[0]:7.2f} sigma={normal_params[1]:6.2f}  "
          f"KS stat={ks_normal.statistic:.4f} p={ks_normal.pvalue:.2e}")
    print(f"  skew-normal fit a={skew_params[0]:7.2f} loc={skew_params[1]:7.2f} "
          f"scale={skew_params[2]:6.2f}  KS stat={ks_skew.statistic:.4f} "
          f"p={ks_skew.pvalue:.2e}")

    return _save_figure(figure, outdir, "psat_smoothed_distribution.png")


def diagnostic6_gain_curves_in_window_plot(
    cube: MeasurementCube,
    g0: SmoothedG0,
    outdir: Path,
    centre_ghz: float = 9.2,
    half_width_ghz: float = 0.05,
) -> Path:
    """Gain-vs-power curves for every column in a frequency window, P1dB
    marked as a dot on each -- lets a reader re-validate the per-column
    extraction on a small, inspectable slice instead of trusting the full
    2001-column band blind.
    """
    indices = np.flatnonzero(
        np.abs(cube.frequency_ghz - centre_ghz) <= half_width_ghz
    )

    figure, (axis_full, axis_zoom) = plt.subplots(1, 2, figsize=(18, 7))
    colours = plt.cm.viridis(np.linspace(0, 1, indices.size))
    p1db_points: list[tuple[float, float]] = []
    for colour, index in zip(colours, indices):
        column_power_dbm = cube.on_chip_power_dbm(index)
        p1db, g0_local = p1db_cut_at_p2db(
            cube.response_db[:, index], column_power_dbm, cube.row0[index]
        )
        smooth_col = savgol_filter(
            cube.response_db[:, index], WEAK_RESPONSE_SAVGOL_WINDOW,
            WEAK_RESPONSE_SAVGOL_ORDER,
        )
        for axis in (axis_full, axis_zoom):
            axis.plot(column_power_dbm, smooth_col, color=colour, linewidth=1.0,
                      alpha=0.85)
        if np.isfinite(p1db):
            threshold = g0_local - 1.0
            p1db_points.append((p1db, threshold))
            for axis in (axis_full, axis_zoom):
                axis.plot(p1db, threshold, "o", color=colour, markersize=4,
                          markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    scalar_mappable = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(
            cube.frequency_ghz[indices].min(), cube.frequency_ghz[indices].max()
        ),
    )
    figure.colorbar(scalar_mappable, ax=axis_full, label="signal frequency (GHz)")
    axis_full.set_xlabel("input signal power (dBm)")
    axis_full.set_ylabel("gain (dB)")
    axis_full.set_title(
        f"Gain vs power, {cube.frequency_ghz[indices].min():.3f}-"
        f"{cube.frequency_ghz[indices].max():.3f} GHz (n={indices.size} columns)"
    )
    axis_full.grid(alpha=0.25, linestyle=":")

    p1db_array = np.array(p1db_points)
    pad_x = 0.15 * max(np.ptp(p1db_array[:, 0]), 1e-6)
    pad_y = 0.6 * max(np.ptp(p1db_array[:, 1]), 1e-6)
    axis_zoom.set_xlim(p1db_array[:, 0].min() - pad_x, p1db_array[:, 0].max() + pad_x)
    axis_zoom.set_ylim(p1db_array[:, 1].min() - pad_y, p1db_array[:, 1].max() + pad_y)
    axis_zoom.set_xlabel("input signal power (dBm)")
    axis_zoom.set_ylabel("gain (dB)")
    axis_zoom.set_title(f"Zoom on P1dB cluster (n={len(p1db_points)} dots)")
    axis_zoom.grid(alpha=0.25, linestyle=":")

    figure.suptitle(
        "dots = P1dB (p1db_cut_at_p2db, self-consistent local G0)", fontsize=10
    )
    return _save_figure(figure, outdir, "gain_curves_ripple_window.png")


def diagnostic7_hard_smoothing_validation_plot(
    cube: MeasurementCube,
    g0: SmoothedG0,
    outdir: Path,
    centre_ghz: float = 9.2,
    half_width_ghz: float = 0.05,
    window_frac: float = P2DB_SMOOTH_WINDOW_FRAC,
) -> Path:
    """Each column in the window cut at P2dB and refit with the hard-savgol
    rule; de-zoomed to show the family of curves converging on a tight P1dB
    cluster. This is the plot that motivated picking window_frac=0.35 (P1dB
    spread across the window dropped to 1.82 dB here; see the module-level
    docstring on P2DB_SMOOTH_WINDOW_FRAC for the full sweep).

    Deliberately uses the cross-column frequency-smoothed G0
    (``g0.values[index]``) for both the P2dB truncation and the P1dB
    threshold, NOT the self-consistent local G0 from ``p1db_cut_at_p2db`` --
    this plot is isolating and validating the smoothing-window choice alone,
    the same way it was run this session before the self-consistent-G0 fix
    (stage 2's actual pipeline) was found. Mixing the two methods here would
    invalidate the comparison this plot exists to show.
    """
    indices = np.flatnonzero(
        np.abs(cube.frequency_ghz - centre_ghz) <= half_width_ghz
    )

    figure, axis = plt.subplots(figsize=(13, 8))
    colours = plt.cm.viridis(np.linspace(0, 1, indices.size))
    p1db_points: list[tuple[float, float]] = []

    for colour, index in zip(colours, indices):
        raw_column = cube.response_db[:, index]
        threshold = g0.values[index] - 1.0
        threshold2 = g0.values[index] - 2.0
        above_p2db = np.flatnonzero(raw_column >= threshold2)
        if above_p2db.size == 0:
            continue
        breakdown_index = int(above_p2db[-1])
        segment = raw_column[: breakdown_index + 1]
        segment_power = cube.on_chip_power_dbm(index)[: breakdown_index + 1]
        segment_window = auto_savgol_window(
            segment.size, window_frac, G0_SMOOTH_POLYORDER
        )
        if segment_window < G0_SMOOTH_POLYORDER + 2:
            continue
        smooth = savgol_filter(
            segment, segment_window, G0_SMOOTH_POLYORDER, mode="interp"
        )

        axis.plot(segment_power, smooth, color=colour, linewidth=1.2, alpha=0.85)
        above_threshold = np.flatnonzero(smooth >= threshold)
        if above_threshold.size and above_threshold[-1] < smooth.size - 1:
            knee = int(above_threshold[-1])
            p1db = np.interp(
                threshold, [smooth[knee + 1], smooth[knee]],
                [segment_power[knee + 1], segment_power[knee]],
            )
            p1db_points.append((p1db, threshold))
            axis.plot(p1db, threshold, "o", color=colour, markersize=5,
                      markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    p1db_array = np.array(p1db_points)
    spread = p1db_array[:, 0].max() - p1db_array[:, 0].min()
    axis.set_xlim(p1db_array[:, 0].min() - 6.0, p1db_array[:, 0].max() + 6.0)
    axis.set_ylim(p1db_array[:, 1].min() - 4.0, p1db_array[:, 1].max() + 6.0)
    axis.set_xlabel("input signal power (dBm)")
    axis.set_ylabel("gain (dB)")
    axis.set_title(
        f"Hard savgol (window_frac={window_frac}, order {G0_SMOOTH_POLYORDER}) "
        f"on data cut at P2dB, {cube.frequency_ghz[indices].min():.3f}-"
        f"{cube.frequency_ghz[indices].max():.3f} GHz  "
        f"(n={len(p1db_points)}, P1dB spread={spread:.2f} dB)"
    )
    axis.grid(alpha=0.25, linestyle=":")

    print(f"hard smoothing validation, window_frac={window_frac}: "
          f"n={len(p1db_points)}  spread={spread:.2f} dB")

    return _save_figure(
        figure, outdir, f"hard_smoothing_cut_at_p2db_frac{window_frac:.2f}.png"
    )


@dataclass
class ModelSweep:
    """Per-frequency compression results read from a run_compression.py
    ``--n-signal-freq`` sweep, restricted to points that actually compressed.
    """

    frequency_ghz: np.ndarray  # (n,), VALID_SOLVED points only
    gain_db: np.ndarray  # (n,), small_signal_gain_vs_off_db
    psat_dbm: np.ndarray  # (n,), p1db_output_dbm
    p1db_input_dbm: np.ndarray  # (n,)
    n_total: int  # frequency_* directories found, including NO_GAIN ones
    # pump_depletion_all_port_db at the P1dB current -- NaN where the source
    # sweep didn't carry the field (e.g. an older compression_summary.json).
    # See src/twpa_solver/multitone/observables.py::power_balance's docstring:
    # 10*log10(outgoing pump power WITH signal / outgoing pump power pump-only),
    # so depletion fraction eta = 1 - 10**(this/10).
    pump_depletion_all_port_db: np.ndarray = field(default_factory=lambda: np.array([]))


def load_model_sweep(model_dir: Path) -> ModelSweep:
    """Read ``model_dir/frequency_*/compression_summary.json``.

    ``p1db_output_dbm`` is already P_sat by construction --
    ``scripts/run_compression.py`` computes it as
    ``p1db_input_dbm + small_signal_gain_db - 1.0``, the exact same
    output-referred-1dB-compression formula this pipeline uses for the
    measured side (``stage4_psat_vs_frequency_plot``). No G0 smoothing or
    robust P1dB fitting is needed here: each point is one clean
    nonlinear-solver result at its own frequency, not a noisy VNA sweep with
    correlated neighbouring columns, so there is no ripple to smooth out.
    Points with ``status != "VALID_SOLVED"`` (below the small-signal-gain
    gate, e.g. the band edges where the device has no gain) are dropped.
    """
    directories = sorted(model_dir.glob("frequency_*"))
    if not directories:
        raise FileNotFoundError(f"no frequency_* directories under {model_dir}")
    frequency_ghz: list[float] = []
    gain_db: list[float] = []
    psat_dbm: list[float] = []
    p1db_input_dbm: list[float] = []
    pump_depletion_all_port_db: list[float] = []
    n_total = 0
    for directory in directories:
        summary_path = directory / "compression_summary.json"
        if not summary_path.exists():
            continue
        n_total += 1
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "VALID_SOLVED":
            continue
        if summary.get("p1db_output_dbm") is None:
            continue
        frequency_ghz.append(float(summary["signal_ghz"]))
        gain_db.append(float(summary["small_signal_gain_vs_off_db"]))
        psat_dbm.append(float(summary["p1db_output_dbm"]))
        p1db_input_dbm.append(float(summary["p1db_input_dbm"]))
        depletion = summary.get("p1db_pump_depletion_all_port_db")
        pump_depletion_all_port_db.append(
            float(depletion) if depletion is not None else float("nan")
        )
    return ModelSweep(
        frequency_ghz=np.asarray(frequency_ghz),
        gain_db=np.asarray(gain_db),
        psat_dbm=np.asarray(psat_dbm),
        p1db_input_dbm=np.asarray(p1db_input_dbm),
        n_total=n_total,
        pump_depletion_all_port_db=np.asarray(pump_depletion_all_port_db),
    )


def load_model_sweep_from_points(run_dir: Path) -> tuple[ModelSweep, list[dict[str, object]]]:
    """Read a raw/merged ``frequency_*/compression_points.csv`` sweep directly
    -- no ``compression_summary.json`` required, so this also reads sweeps
    merged at the CSV level (e.g. a low-current and a high-current
    ``run_compression.py`` run concatenated per frequency and sorted by
    ``signal_current_a``; see psat_comparison_fix_plan.md Phase 5/7).

    On-chip power is computed directly from ``signal_current_a`` (Norton,
    zero attenuation) -- the model current already IS the on-chip current, no
    cable exists in the simulation. P1dB/G0 use the same persistent-crossing
    rule as the measured side's ``p1db_cut_at_p2db``: G0 = first
    (lowest-current) row's gain, threshold = G0 - 1, P1dB = last index still
    at/above threshold, linearly interpolated in on-chip dBm. Pump depletion
    at P1dB is ``pump_depletion_all_port_db`` interpolated the same way.

    Returns ``(summary, curves)``: ``summary`` is a scalar-per-frequency
    ``ModelSweep`` (compatible with the JSON loader's output, so Phase 6's
    decomposition code works unchanged); ``curves`` is the per-frequency raw
    (power_dbm, gain_db) arrays Phase 7's compression-shape overlay needs and
    ``ModelSweep``'s scalar fields cannot carry.
    """
    directories = sorted(run_dir.glob("frequency_*"))
    if not directories:
        raise FileNotFoundError(f"no frequency_* directories under {run_dir}")
    frequency_ghz: list[float] = []
    gain_db: list[float] = []
    psat_dbm: list[float] = []
    p1db_input_dbm: list[float] = []
    pump_depletion_all_port_db: list[float] = []
    curves: list[dict[str, object]] = []
    n_total = 0
    for directory in directories:
        points_path = directory / "compression_points.csv"
        if not points_path.exists():
            continue
        n_total += 1
        match = re.search(r"_(\d+\.\d+)ghz", directory.name)
        if match is None:
            continue
        freq = float(match.group(1))
        with points_path.open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r["status"] == "VALID_SOLVED"]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: float(r["signal_current_a"]))
        current = np.array([float(r["signal_current_a"]) for r in rows])
        power = np.array([
            10.0 * np.log10(port_available_power_w(i, Z0_OHM, convention="norton") / 1.0e-3)
            for i in current
        ])
        gain = np.array([float(r["gain_vs_off_db"]) for r in rows])
        depletion = np.array([float(r["pump_depletion_all_port_db"]) for r in rows])
        g0 = float(gain[0])
        threshold = g0 - 1.0
        above = np.flatnonzero(gain >= threshold)
        p1db = float("nan")
        if above.size and above[-1] < gain.size - 1:
            knee = int(above[-1])
            p1db = float(np.interp(
                threshold, [gain[knee + 1], gain[knee]], [power[knee + 1], power[knee]],
            ))
            depletion_at_p1db = float(np.interp(
                threshold, [gain[knee + 1], gain[knee]],
                [depletion[knee + 1], depletion[knee]],
            ))
            frequency_ghz.append(freq)
            gain_db.append(g0)
            p1db_input_dbm.append(p1db)
            psat_dbm.append(p1db + g0 - 1.0)
            pump_depletion_all_port_db.append(depletion_at_p1db)
        curves.append({
            "freq_ghz": freq, "power_dbm": power, "gain_db": gain,
            "g0_db": g0, "p1db_dbm": p1db,
        })

    if not frequency_ghz:
        raise ValueError(f"no frequency reached 1 dB compression under {run_dir}")

    model = ModelSweep(
        frequency_ghz=np.asarray(frequency_ghz),
        gain_db=np.asarray(gain_db),
        psat_dbm=np.asarray(psat_dbm),
        p1db_input_dbm=np.asarray(p1db_input_dbm),
        n_total=n_total,
        pump_depletion_all_port_db=np.asarray(pump_depletion_all_port_db),
    )
    return model, curves


def model_psat_vs_frequency_plot(model: ModelSweep, outdir: Path) -> Path:
    """P_sat vs frequency for the model sweep -- one point per solved
    frequency, no smoothing (see load_model_sweep for why none is needed).
    """
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(
        model.frequency_ghz, model.psat_dbm, "o-", color=MODEL_COLOUR,
        linewidth=1.4, markersize=5, label="model $P_{sat}$ (per frequency)",
    )
    axis.axhline(
        float(np.median(model.psat_dbm)), color="0.3", linestyle="--",
        linewidth=1.2, label=f"median={np.median(model.psat_dbm):.2f} dBm",
    )
    axis.set_xlabel("signal frequency (GHz)")
    axis.set_ylabel("$P_{sat}$ output (dBm)")
    axis.set_title(
        f"Model $P_{{sat}}$ vs frequency (n={model.psat_dbm.size} of "
        f"{model.n_total} points solved)"
    )
    axis.legend(fontsize=8.5)
    axis.grid(alpha=0.25, linestyle=":")

    print(
        f"model Psat: n={model.psat_dbm.size} of {model.n_total} attempted  "
        f"median={np.median(model.psat_dbm):.2f}  mean={model.psat_dbm.mean():.2f}  "
        f"std={model.psat_dbm.std(ddof=1):.2f}"
    )

    return _save_figure(figure, outdir, "psat_model_vs_frequency.png")


def model_psat_distribution_plot(model: ModelSweep, outdir: Path) -> Path:
    """Distribution + normal fit of the model P_sat values.

    Skips the skew-normal fit the measured side also reports: skew-normal
    has 3 free parameters, and at n~20-30 points that fit is not meaningfully
    constrained. The KS test is reported for completeness but is weak-powered
    this small -- read it qualitatively (does the data look normal-ish), not
    as a pass/fail gate the way it might be at n>100.
    """
    psat = model.psat_dbm
    figure, axis = plt.subplots(figsize=(9, 6))
    pad = max(0.5 * float(psat.std(ddof=1)), 0.5)
    lo, hi = float(psat.min()) - pad, float(psat.max()) + pad
    n_bins = int(np.clip(psat.size // 2, 5, 15))
    bins = np.linspace(lo, hi, n_bins)
    axis.hist(
        psat, bins=bins, density=True, alpha=0.55, color=MODEL_COLOUR,
        edgecolor="none", label=f"data (n={psat.size})",
    )

    grid = np.linspace(lo, hi, 400)
    normal_params = stats.norm.fit(psat)
    axis.plot(
        grid, stats.norm.pdf(grid, *normal_params), color="0.15", linewidth=1.8,
        label=f"normal  $\\mu$={normal_params[0]:.2f}  $\\sigma$={normal_params[1]:.2f}",
    )
    axis.set_xlim(lo, hi)
    axis.set_xlabel("$P_{sat}$ output (dBm)")
    axis.set_ylabel("density")
    axis.set_title(
        f"Distribution of model $P_{{sat}}$ (n={psat.size})\n"
        f"median={np.median(psat):.2f}  mean={psat.mean():.2f}  "
        f"std={psat.std(ddof=1):.2f}"
    )
    axis.legend(fontsize=8.5)
    axis.grid(alpha=0.25, linestyle=":")

    ks_normal = stats.kstest(psat, lambda x: stats.norm.cdf(x, *normal_params))
    print(
        f"model Psat distribution: n={psat.size}  median={np.median(psat):.2f}  "
        f"mean={psat.mean():.2f}  std={psat.std(ddof=1):.2f}  "
        f"skew={stats.skew(psat):.3f}"
    )
    print(
        f"  normal fit  mu={normal_params[0]:7.2f}  sigma={normal_params[1]:6.2f}  "
        f"KS stat={ks_normal.statistic:.4f} p={ks_normal.pvalue:.2e}  "
        f"(n={psat.size} -- weak-powered this small, read qualitatively)"
    )

    return _save_figure(figure, outdir, "psat_model_distribution.png")


def comparison_plot(
    measured_freq: np.ndarray,
    measured_psat: np.ndarray,
    model: ModelSweep,
    outdir: Path,
) -> Path:
    """Two-panel measured-vs-model comparison, fitted curves only (no raw
    histograms or per-point scatter): left panel overlays the fitted P_sat
    distributions (measured normal + skew-normal from stage5, model normal
    from model_psat_distribution_plot); right panel overlays the fitted
    P_sat-vs-frequency curves (measured's robust-fit trace from stage4,
    model's per-frequency line from model_psat_vs_frequency_plot).

    Refits the same normal/skew-normal parameters those standalone plots
    already computed rather than threading them through as extra return
    values -- both fits are deterministic and cheap, so this stays
    guaranteed-consistent with the individual plots without widening every
    stage's signature.

    Not a formal two-sample test (no KS-2samp): the two samples are not
    comparable draws. The measured side is 1000+ frequency-smoothed Themis
    columns at one fixed device/pump, with strong correlation between
    neighbouring columns (both G0 and P1dB are smoothed in frequency there --
    see stage5's docstring). The model side is ~20-30 independent nonlinear
    solves on designs/ipm_2c_fixed at pump_freq_ghz=7.37931034483 GHz (the
    map-derived operating point whose small-signal gain matched the measured
    peak G0). This is a qualitative read -- same range, same rough shape --
    not a statistical verdict.
    """
    figure, (axis_dist, axis_freq) = plt.subplots(1, 2, figsize=(16, 6.5))

    q25, q75 = np.percentile(measured_psat, [25, 75])
    span = max(q75 - q25, 1e-9)
    lo = min(float(measured_psat.min()), float(model.psat_dbm.min()))
    lo = max(lo, float(np.median(measured_psat) - 6.0 * span))
    hi = max(float(measured_psat.max()), float(model.psat_dbm.max()))
    grid = np.linspace(lo, hi, 400)

    measured_normal = stats.norm.fit(measured_psat)
    measured_skew = stats.skewnorm.fit(measured_psat)
    model_normal = stats.norm.fit(model.psat_dbm)

    axis_dist.plot(
        grid, stats.norm.pdf(grid, *measured_normal), color=PSAT_COLOUR,
        linewidth=2.0,
        label=f"measured normal  $\\mu$={measured_normal[0]:.2f}",
    )
    axis_dist.plot(
        grid, stats.skewnorm.pdf(grid, *measured_skew), color=PSAT_COLOUR,
        linewidth=1.6, linestyle="--",
        label=f"measured skew-normal  a={measured_skew[0]:.2f}",
    )
    axis_dist.plot(
        grid, stats.norm.pdf(grid, *model_normal), color=MODEL_COLOUR,
        linewidth=2.0, label=f"model normal  $\\mu$={model_normal[0]:.2f}",
    )
    axis_dist.set_xlim(lo, hi)
    axis_dist.set_xlabel("$P_{sat}$ output (dBm)")
    axis_dist.set_ylabel("density")
    axis_dist.set_title("Fitted $P_{sat}$ distributions")
    axis_dist.legend(fontsize=8)
    axis_dist.grid(alpha=0.25, linestyle=":")

    axis_freq.plot(
        measured_freq, measured_psat, color=PSAT_COLOUR, linewidth=1.8,
        label=f"measured, robust-fit (n={measured_psat.size})",
    )
    axis_freq.plot(
        model.frequency_ghz, model.psat_dbm, "o-", color=MODEL_COLOUR,
        linewidth=1.6, markersize=4, label=f"model (n={model.psat_dbm.size})",
    )
    axis_freq.set_xlabel("signal frequency (GHz)")
    axis_freq.set_ylabel("$P_{sat}$ output (dBm)")
    axis_freq.set_title("Fitted $P_{sat}$ vs frequency")
    axis_freq.legend(fontsize=8)
    axis_freq.grid(alpha=0.25, linestyle=":")

    gap = float(np.median(model.psat_dbm) - np.median(measured_psat))
    print(
        f"measured vs model median Psat gap: {gap:+.2f} dB (model - measured), "
        f"qualitative comparison only -- see comparison_plot docstring for why "
        f"no two-sample test is reported"
    )

    return _save_figure(figure, outdir, "psat_measured_vs_model_distribution.png")


@dataclass
class Decomposition:
    """Pointwise G0/P1dB/P_sat deltas on the MODEL frequency grid (Phase 6).

    Both sides are gated identically by ``MIN_SMOOTHED_G0_DB`` before any
    delta is formed -- previously the gate only ran on the measured side,
    which silently kept low-gain model points with no measured counterpart.
    ``delta_psat == delta_p1db + delta_g0`` is asserted to 1e-9 in
    ``compute_decomposition`` itself, not left for a caller to check.
    """

    freq_ghz: np.ndarray
    model_g0_db: np.ndarray
    measured_g0_db: np.ndarray  # envelope (g0_local), same quantity psat was built from
    measured_g0_raw_db: np.ndarray  # row0, unsmoothed -- diagnostic only
    model_p1db_dbm: np.ndarray
    measured_p1db_dbm: np.ndarray
    model_psat_dbm: np.ndarray
    measured_psat_dbm: np.ndarray
    delta_g0_db: np.ndarray  # model - measured (envelope)
    delta_g0_raw_db: np.ndarray  # model - measured (raw row0)
    delta_p1db_db: np.ndarray  # model - measured
    delta_psat_db: np.ndarray  # model - measured


def _interp_to_model_grid(
    model_freq_ghz: np.ndarray, measured_freq_ghz: np.ndarray, measured_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate MEASURED values onto the MODEL grid (never the reverse --
    upsampling the sparse model onto 2001 measured columns would invent
    ripple structure the model never resolved). Returns (interpolated,
    coverage_mask); coverage is False outside the measured frequency range,
    where np.interp would silently clamp to the nearest measured value.
    """
    interpolated = np.interp(model_freq_ghz, measured_freq_ghz, measured_values)
    coverage = (model_freq_ghz >= measured_freq_ghz.min()) & (
        model_freq_ghz <= measured_freq_ghz.max()
    )
    return interpolated, coverage


def compute_decomposition(
    model: ModelSweep,
    measured_freq_ghz: np.ndarray,
    measured_g0_db: np.ndarray,
    measured_g0_raw_db: np.ndarray,
    measured_p1db_dbm: np.ndarray,
    measured_psat_dbm: np.ndarray,
) -> Decomposition:
    g0_meas, coverage = _interp_to_model_grid(model.frequency_ghz, measured_freq_ghz, measured_g0_db)
    g0_meas_raw, _ = _interp_to_model_grid(model.frequency_ghz, measured_freq_ghz, measured_g0_raw_db)
    p1db_meas, _ = _interp_to_model_grid(model.frequency_ghz, measured_freq_ghz, measured_p1db_dbm)
    psat_meas, _ = _interp_to_model_grid(model.frequency_ghz, measured_freq_ghz, measured_psat_dbm)

    gate = coverage & (model.gain_db > MIN_SMOOTHED_G0_DB) & (g0_meas > MIN_SMOOTHED_G0_DB)

    freq = model.frequency_ghz[gate]
    model_g0 = model.gain_db[gate]
    model_p1db = model.p1db_input_dbm[gate]
    model_psat = model.psat_dbm[gate]
    g0_meas = g0_meas[gate]
    g0_meas_raw = g0_meas_raw[gate]
    p1db_meas = p1db_meas[gate]
    psat_meas = psat_meas[gate]

    delta_g0 = model_g0 - g0_meas
    delta_g0_raw = model_g0 - g0_meas_raw
    delta_p1db = model_p1db - p1db_meas
    delta_psat = model_psat - psat_meas

    np.testing.assert_allclose(delta_psat, delta_p1db + delta_g0, rtol=0.0, atol=1e-9)

    return Decomposition(
        freq_ghz=freq,
        model_g0_db=model_g0, measured_g0_db=g0_meas, measured_g0_raw_db=g0_meas_raw,
        model_p1db_dbm=model_p1db, measured_p1db_dbm=p1db_meas,
        model_psat_dbm=model_psat, measured_psat_dbm=psat_meas,
        delta_g0_db=delta_g0, delta_g0_raw_db=delta_g0_raw,
        delta_p1db_db=delta_p1db, delta_psat_db=delta_psat,
    )


def g0_model_vs_measured_plot(decomposition: Decomposition, outdir: Path) -> Path:
    d = decomposition
    figure, (axis_curve, axis_delta) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]},
    )
    axis_curve.plot(d.freq_ghz, d.measured_g0_raw_db, color="0.75", linewidth=0.7,
                     label="measured G0, raw (row0)")
    axis_curve.plot(d.freq_ghz, d.measured_g0_db, color=MEASURED_COLOUR, linewidth=1.6,
                     label="measured G0, envelope (g0_local)")
    axis_curve.plot(d.freq_ghz, d.model_g0_db, "o-", color=MODEL_COLOUR, linewidth=1.6,
                     markersize=4, label="model G0")
    axis_curve.set_ylabel("G0 (dB)")
    axis_curve.set_title(f"G0(f), model vs measured (n={d.freq_ghz.size})")
    axis_curve.legend(fontsize=8.5)
    axis_curve.grid(alpha=0.25, linestyle=":")

    axis_delta.axhline(0.0, color="0.3", linewidth=1.0)
    axis_delta.plot(d.freq_ghz, d.delta_g0_raw_db, color="0.75", linewidth=0.7,
                     label="$\\Delta G_0$ raw")
    axis_delta.plot(d.freq_ghz, d.delta_g0_db, color=MEASURED_COLOUR, linewidth=1.6,
                     label="$\\Delta G_0$ envelope")
    axis_delta.set_xlabel("signal frequency (GHz)")
    axis_delta.set_ylabel("$\\Delta G_0$ (dB)\nmodel - measured")
    axis_delta.legend(fontsize=8.5)
    axis_delta.grid(alpha=0.25, linestyle=":")

    print(
        f"delta_G0 (envelope): median={np.median(d.delta_g0_db):+.2f}  "
        f"mean={d.delta_g0_db.mean():+.2f}  std={d.delta_g0_db.std(ddof=1):.2f}  "
        f"(raw: median={np.median(d.delta_g0_raw_db):+.2f})"
    )
    return _save_figure(figure, outdir, "g0_model_vs_measured.png")


def p1db_model_vs_measured_plot(decomposition: Decomposition, outdir: Path) -> Path:
    d = decomposition
    figure, (axis_curve, axis_delta) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]},
    )
    axis_curve.plot(d.freq_ghz, d.measured_p1db_dbm, color=MEASURED_COLOUR, linewidth=1.6,
                     label="measured P1dB input (robust fit)")
    axis_curve.plot(d.freq_ghz, d.model_p1db_dbm, "o-", color=MODEL_COLOUR, linewidth=1.6,
                     markersize=4, label="model P1dB input")
    axis_curve.set_ylabel("$P_{1dB}$ input (dBm)")
    axis_curve.set_title(f"P1dB(f), model vs measured (n={d.freq_ghz.size})")
    axis_curve.legend(fontsize=8.5)
    axis_curve.grid(alpha=0.25, linestyle=":")

    axis_delta.axhline(0.0, color="0.3", linewidth=1.0)
    axis_delta.plot(d.freq_ghz, d.delta_p1db_db, color=FIT_COLOUR, linewidth=1.6)
    axis_delta.set_xlabel("signal frequency (GHz)")
    axis_delta.set_ylabel("$\\Delta P_{1dB}$ (dB)\nmodel - measured")
    axis_delta.grid(alpha=0.25, linestyle=":")

    print(
        f"delta_P1dB: median={np.median(d.delta_p1db_db):+.2f}  "
        f"mean={d.delta_p1db_db.mean():+.2f}  std={d.delta_p1db_db.std(ddof=1):.2f}"
    )
    return _save_figure(figure, outdir, "p1db_model_vs_measured.png")


def psat_decomposition_plot_and_csv(decomposition: Decomposition, outdir: Path) -> tuple[Path, Path]:
    """Stacked delta_P1dB / delta_G0 / delta_Psat panels plus the per-frequency
    CSV. The masked fraction ``1 - |delta_Psat|/|delta_P1dB|`` is the amount of
    the P1dB defect that summing into P_sat hides (Phase 6 cancellation metric).
    """
    d = decomposition
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for axis, values, label, colour in (
        (axes[0], d.delta_p1db_db, "$\\Delta P_{1dB}$", FIT_COLOUR),
        (axes[1], d.delta_g0_db, "$\\Delta G_0$", MEASURED_COLOUR),
        (axes[2], d.delta_psat_db, "$\\Delta P_{sat}$", PSAT_COLOUR),
    ):
        axis.axhline(0.0, color="0.3", linewidth=1.0)
        axis.plot(d.freq_ghz, values, color=colour, linewidth=1.6)
        axis.axhline(float(np.median(values)), color=colour, linestyle="--", linewidth=1.0,
                     label=f"median={np.median(values):+.2f} dB")
        axis.set_ylabel(f"{label} (dB)")
        axis.legend(fontsize=8.5, loc="upper right")
        axis.grid(alpha=0.25, linestyle=":")
    axes[-1].set_xlabel("signal frequency (GHz)")
    axes[0].set_title(
        f"$P_{{sat}}=P_{{1dB}}+G_0-1$ pointwise decomposition, model - measured "
        f"(n={d.freq_ghz.size})"
    )
    png = _save_figure(figure, outdir, "psat_decomposition.png")

    masked_fraction = 1.0 - np.abs(d.delta_psat_db) / np.maximum(np.abs(d.delta_p1db_db), 1e-12)
    print(
        f"delta_Psat: median={np.median(d.delta_psat_db):+.2f}  "
        f"mean={d.delta_psat_db.mean():+.2f}  std={d.delta_psat_db.std(ddof=1):.2f}"
    )
    print(
        f"cancellation: median masked fraction 1-|dPsat|/|dP1dB| = "
        f"{np.median(masked_fraction):.3f} "
        f"({100 * np.median(masked_fraction):.1f}% of the P1dB defect is hidden "
        f"by summing into P_sat)"
    )
    if d.freq_ghz.size > 1:
        model_spacing_ghz = float(np.median(np.diff(d.freq_ghz)))
        print(
            f"decomposition n_eff: {d.freq_ghz.size} == n (Phase 8) -- model "
            f"frequencies are independent nonlinear solves spaced "
            f"~{model_spacing_ghz:.3f} GHz apart, far beyond the measured "
            f"~4-25 MHz correlation length, so each delta_G0/delta_P1dB/"
            f"delta_Psat point is its own independent sample"
        )

    csv_path = outdir / "psat_decomposition.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    header = (
        "freq_ghz,model_g0_db,measured_g0_db,model_p1db_dbm,measured_p1db_dbm,"
        "model_psat_dbm,measured_psat_dbm,delta_g0_db,delta_p1db_db,delta_psat_db,"
        "masked_fraction"
    )
    rows = np.column_stack([
        d.freq_ghz, d.model_g0_db, d.measured_g0_db, d.model_p1db_dbm, d.measured_p1db_dbm,
        d.model_psat_dbm, d.measured_psat_dbm, d.delta_g0_db, d.delta_p1db_db, d.delta_psat_db,
        masked_fraction,
    ])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"wrote {csv_path}")
    return png, csv_path


def autocorrelation_length(residual: np.ndarray, max_lag: int = 400) -> int:
    """Smallest lag (samples) where the mean-subtracted residual's
    autocorrelation first drops to ``<= 1/e`` -- converts a frequency
    trace's raw sample count into a correlation length in samples (Phase 8).
    Falls back to 1 (no correlation measurable) on a degenerate input.
    """
    residual = np.asarray(residual, dtype=float)
    residual = residual - residual.mean()
    denom = float(np.dot(residual, residual))
    if denom <= 0.0 or residual.size < 3:
        return 1
    limit = min(max_lag, residual.size - 1)
    for lag in range(1, limit + 1):
        rho = float(np.dot(residual[:-lag], residual[lag:])) / denom
        if rho <= 1.0 / np.e:
            return lag
    return max(limit, 1)


def block_bootstrap_ci(
    values: np.ndarray, block_length: int, n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """Moving-block bootstrap 90% CI on the median (Phase 8): block length is
    set by the trace's own measured correlation length, not treated as if
    every column were an independent draw.
    """
    rng = np.random.default_rng(seed)
    n = values.size
    block_length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / block_length))
    starts = np.arange(n - block_length + 1)
    medians = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([values[s:s + block_length] for s in picks])[:n]
        medians[b] = np.median(sample)
    return float(np.percentile(medians, 5)), float(np.percentile(medians, 95))


def phase8_statistics_report(
    cube: MeasurementCube,
    g0: SmoothedG0,
    psat: np.ndarray,
    p1db_raw_usable: np.ndarray,
    g0_raw_usable: np.ndarray,
) -> None:
    """Report n_eff and a smoothing-sensitivity check instead of treating the
    measured P_sat trace as an i.i.d. sample (Phase 8).

    The measured "distribution" is the histogram of ONE fitted curve --
    neighbouring frequency columns are highly correlated by construction
    (both G0 and P1dB are smoothed in frequency), so the raw column count
    overstates the independent information by orders of magnitude. G0's
    699-column (~2.80 GHz) smoothing window means ~3 independent G0 values
    span the whole 8 GHz usable band; the P1dB robust-fit residual, by
    contrast, decorrelates in ~1 column (~4 MHz) -- it is smoothing genuine
    per-column spread, not correlated noise, which is exactly why removing
    all smoothing moves the median only slightly but blows up std/skew (see
    the printed comparison below).
    """
    spacing_ghz = float(np.median(np.diff(cube.frequency_ghz)))

    g0_values = g0.values[g0.keep]
    g0_corr_samples = autocorrelation_length(g0_values)
    g0_n_eff = max(1, int(g0_values.size / g0_corr_samples))
    print(
        f"G0 n_eff: {g0_n_eff} (n={g0_values.size}, correlation length="
        f"{g0_corr_samples} columns={g0_corr_samples * spacing_ghz:.2f} GHz, "
        f"set by the {G0_SMOOTH_WINDOW_FRAC}-fraction smoothing window)"
    )

    psat_corr_samples = autocorrelation_length(psat)
    psat_n_eff = max(1, int(psat.size / psat_corr_samples))
    print(
        f"Psat n_eff: {psat_n_eff} (n={psat.size}, correlation length="
        f"{psat_corr_samples} columns={psat_corr_samples * spacing_ghz:.2f} GHz)"
    )

    block = max(psat_corr_samples, g0_corr_samples)
    ci_lo, ci_hi = block_bootstrap_ci(psat, block)
    print(
        f"Psat median block-bootstrap 90% CI (block={block} columns="
        f"{block * spacing_ghz:.2f} GHz, n_eff={psat_n_eff}): "
        f"[{ci_lo:.2f}, {ci_hi:.2f}] dBm, point estimate {np.median(psat):.2f} dBm"
    )

    psat_unsmoothed = p1db_raw_usable + g0_raw_usable - 1.0
    finite = np.isfinite(psat_unsmoothed)
    print(
        f"smoothing sensitivity: Psat WITH smoothing median={np.median(psat):.2f} "
        f"std={psat.std(ddof=1):.2f} skew={stats.skew(psat):+.2f}  |  "
        f"WITHOUT smoothing (raw per-column P1dB + raw row0 G0, n={int(finite.sum())}) "
        f"median={np.median(psat_unsmoothed[finite]):.2f} "
        f"std={psat_unsmoothed[finite].std(ddof=1):.2f} "
        f"skew={stats.skew(psat_unsmoothed[finite]):+.2f}"
    )
    print(
        "KS tests against normal/skew-normal (stage5/model distribution "
        "plots) are a labelled diagnostic only, not a distributional "
        "verdict -- both reject on comb structure the smoothing itself "
        "creates, not on any physical departure from Gaussianity."
    )


def energy_conservation_gate(
    label: str, psat_dbm: np.ndarray, pump_dbm: float, tolerance_db: float = 1e-6
) -> None:
    """Assert ``P_sat + 3 dB <= P_pump`` pointwise (Phase 7).

    Signal + idler output at compression cannot exceed the pump; violating
    this is not a subtle discrepancy, it is impossible. This is the exact
    test that ruled out an inferred 45.7 dB pump-line-loss figure during
    planning (see psat_comparison_fix_plan.md, "Verified inputs": at
    loss=45.0 dB, ``P_sat - P_pump = +1.57 dB``, which fails this gate).
    Raises with the offending numbers instead of silently plotting a
    physically impossible ``P_sat``.
    """
    violation = psat_dbm + 3.0 - tolerance_db > pump_dbm
    if np.any(violation):
        worst = float(np.max(psat_dbm[violation]))
        raise ValueError(
            f"{label}: energy conservation violated at "
            f"{int(violation.sum())} of {psat_dbm.size} point(s) -- "
            f"worst P_sat={worst:.2f} dBm (P_sat+3dB={worst + 3.0:.2f} dBm) "
            f"exceeds on-chip pump {pump_dbm:.2f} dBm. Check the pump-line "
            "loss model / on-chip pump figure, not the solver."
        )


def model_depletion_cross_check(
    model: ModelSweep, model_pump_dbm: float
) -> np.ndarray:
    """Compare energy-accounted depletion against the solver's own
    ``pump_depletion_all_port_db`` field, per model frequency (Phase 7).

    ``eta_energy`` assumes signal + idler output power together equal
    ``P_sat + 3 dB`` (both tones roughly comparable at 4WM degenerate-pump
    compression -- the same assumption used to derive the pump-line loss in
    planning). ``eta_solver`` comes directly from the solver's own all-port
    pump-power accounting (``power_balance``'s ``pump_depletion_all_port_db``),
    with no assumption about the idler at all. Agreement between the two
    validates the +3 dB assumption; the reference case (7.086 GHz) agreed to
    -0.058 dB solver vs ~1.1% energy-accounted -- both ~1.3%. Returns
    ``eta_solver`` per frequency; raises nothing, since this is a diagnostic
    not a gate -- flags disagreement beyond ``DEPLETION_CROSS_CHECK_RATIO``
    with a printed warning instead.
    """
    eta_energy = 10.0 ** ((model.psat_dbm + 3.0 - model_pump_dbm) / 10.0)
    eta_solver = 1.0 - 10.0 ** (model.pump_depletion_all_port_db / 10.0)
    print(f'{"freq_ghz":>10} {"eta_energy":>11} {"eta_solver":>11} {"ratio":>8}')
    for freq, energy, solver in zip(model.frequency_ghz, eta_energy, eta_solver):
        ratio = energy / solver if solver > 0 else float("nan")
        flag = (
            " <-- DISAGREE" if np.isfinite(ratio) and (
                ratio > DEPLETION_CROSS_CHECK_RATIO
                or ratio < 1.0 / DEPLETION_CROSS_CHECK_RATIO
            ) else ""
        )
        print(f"{freq:10.3f} {100 * energy:10.2f}% {100 * solver:10.2f}% "
              f"{ratio:8.2f}{flag}")
    return eta_solver


def pump_referred_comparison_plot(
    model: ModelSweep,
    curves: list[dict[str, object]],
    model_pump_dbm: float,
    cube: MeasurementCube,
    p1db: P1dBResult,
    measured_freq: np.ndarray,
    measured_psat: np.ndarray,
    outdir: Path,
) -> Path:
    """Four panels ordered by how many calibration constants they depend on
    (Phase 7): (1) G0(f), none; (2) compression-curve shape vs
    ``P_in - P1dB``, none; (3) ``P_sat - P_pump``, one loss model per side,
    each internally self-consistent; (4) absolute ``P_sat(f)``, both line
    losses. Put the calibration-free panels first so a reader forms a
    judgement before the panel that can be moved by a wrong constant.
    """
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_g0, ax_shape, ax_gap, ax_abs = axes.ravel()

    ax_g0.plot(model.frequency_ghz, model.gain_db, "o-", color=MODEL_COLOUR,
               linewidth=1.6, markersize=4, label="model G0")
    ax_g0.plot(measured_freq, p1db.g0_local_db[p1db.usable], color=MEASURED_COLOUR,
               linewidth=1.2, alpha=0.85, label="measured G0 (envelope)")
    ax_g0.set_xlabel("signal frequency (GHz)")
    ax_g0.set_ylabel("G0 (dB)")
    ax_g0.set_title("Panel 1: $G_0(f)$ -- no calibration constant")
    ax_g0.legend(fontsize=8.5)
    ax_g0.grid(alpha=0.25, linestyle=":")

    colormap = plt.get_cmap("viridis")
    freqs = model.frequency_ghz
    norm = plt.Normalize(freqs.min(), freqs.max()) if freqs.size else plt.Normalize(0, 1)
    for curve in curves:
        if not np.isfinite(curve["p1db_dbm"]):
            continue
        colour = colormap(norm(curve["freq_ghz"]))
        ax_shape.plot(
            curve["power_dbm"] - curve["p1db_dbm"], curve["gain_db"] - curve["g0_db"],
            color=colour, linewidth=1.4, marker="o", markersize=3, mfc="white",
        )
        index = int(np.argmin(np.abs(cube.frequency_ghz - curve["freq_ghz"])))
        column_p1db, column_g0 = p1db.p1db_dbm[index], p1db.g0_local_db[index]
        if np.isfinite(column_p1db):
            smooth_col = savgol_filter(
                cube.response_db[:, index], WEAK_RESPONSE_SAVGOL_WINDOW,
                WEAK_RESPONSE_SAVGOL_ORDER,
            )
            ax_shape.plot(
                cube.on_chip_power_dbm(index) - column_p1db, smooth_col - column_g0,
                color=colour, linewidth=1.0, linestyle="--", alpha=0.7,
            )
    ax_shape.axvline(0.0, color="0.3", linewidth=1.0, linestyle=":")
    ax_shape.axhline(-1.0, color="0.3", linewidth=1.0, linestyle=":")
    ax_shape.set_xlim(-25, 10)
    ax_shape.set_ylim(-15, 2)
    ax_shape.set_xlabel("$P_{in} - P_{1dB}$ (dB)")
    ax_shape.set_ylabel("gain $- G_0$ (dB)")
    ax_shape.set_title(
        "Panel 2: compression-curve shape -- no calibration constant\n"
        "solid = model, dashed = measured (same colour = matched frequency)"
    )
    ax_shape.grid(alpha=0.25, linestyle=":")

    measured_pump_dbm = cube.on_chip_pump_dbm
    model_gap = model.psat_dbm - model_pump_dbm
    measured_gap = measured_psat - measured_pump_dbm
    ax_gap.plot(model.frequency_ghz, model_gap, "o-", color=MODEL_COLOUR,
                linewidth=1.6, markersize=4,
                label=f"model, median={np.median(model_gap):+.2f} dB")
    ax_gap.plot(measured_freq, measured_gap, color=MEASURED_COLOUR, linewidth=1.2,
                alpha=0.85, label=f"measured, median={np.median(measured_gap):+.2f} dB")
    ax_gap.set_xlabel("signal frequency (GHz)")
    ax_gap.set_ylabel("$P_{sat} - P_{pump}$ (dB)")
    ax_gap.set_title(
        "Panel 3: pump-referred gap -- one loss model per side, "
        "each self-consistent"
    )
    ax_gap.legend(fontsize=8.5)
    ax_gap.grid(alpha=0.25, linestyle=":")

    ax_abs.plot(model.frequency_ghz, model.psat_dbm, "o-", color=MODEL_COLOUR,
                linewidth=1.6, markersize=4, label="model $P_{sat}$")
    ax_abs.plot(measured_freq, measured_psat, color=MEASURED_COLOUR, linewidth=1.2,
                alpha=0.85, label="measured $P_{sat}$")
    ax_abs.set_xlabel("signal frequency (GHz)")
    ax_abs.set_ylabel("$P_{sat}$ output (dBm)")
    ax_abs.set_title(
        "Panel 4: absolute $P_{sat}(f)$ -- both line losses "
        "(model: on-chip current, no cable; measured: loss_B1/loss_A10)"
    )
    ax_abs.legend(fontsize=8.5)
    ax_abs.grid(alpha=0.25, linestyle=":")

    figure.suptitle(
        f"Pump-referred comparison, ordered by calibration dependence -- "
        f"model pump {model_pump_dbm:.2f} dBm, measured pump "
        f"{measured_pump_dbm:.2f} dBm", fontsize=12,
    )
    print(f"P_sat - P_pump: model median={np.median(model_gap):+.2f} dB, "
          f"measured median={np.median(measured_gap):+.2f} dB, "
          f"gap={np.median(model_gap) - np.median(measured_gap):+.2f} dB")
    return _save_figure(figure, outdir, "pump_referred_comparison.png")


def run_model_side(
    model_dir: Path | None,
    outdir: Path,
    measured_freq: np.ndarray | None = None,
    measured_psat: np.ndarray | None = None,
    measured_g0_local: np.ndarray | None = None,
    measured_g0_raw: np.ndarray | None = None,
    measured_p1db: np.ndarray | None = None,
    model_points_dir: Path | None = None,
    model_pump_dbm: float | None = None,
    cube: MeasurementCube | None = None,
    p1db_result: P1dBResult | None = None,
) -> None:
    """Model-side P_sat pipeline: read a run_compression.py frequency sweep
    (``model_dir/frequency_*/compression_summary.json``, produced with
    ``--n-signal-freq`` and, ideally, ``--stop-after-p1db``) and plot P_sat
    vs frequency plus its distribution. Overlays the fitted measured and
    model curves (``comparison_plot``) when both ``measured_freq`` and
    ``measured_psat`` (the measured-side stage 3/4 output) are given, and runs
    the Phase 6 pointwise G0/P1dB/P_sat decomposition when the additional
    ``measured_g0_local``/``measured_g0_raw``/``measured_p1db`` arrays
    (aligned to ``measured_freq``) are also given.

    No-ops with a message if neither ``model_dir`` nor ``model_points_dir`` is
    given -- this stays optional since the model sweep is a separate,
    expensive campaign (``scripts/run_compression.py``), not something this
    script can produce itself.

    ``model_points_dir`` (Phase 7) is a second, independent model source: a
    raw or CSV-merged ``compression_points.csv`` sweep read via
    ``load_model_sweep_from_points`` rather than ``compression_summary.json``
    -- used when the model sweep was assembled by concatenating two
    independently-run current ranges rather than one
    ``run_compression.py --n-signal-freq`` invocation. Requires
    ``model_pump_dbm``, ``cube`` and ``p1db_result``; runs the energy-
    conservation gate, the depletion cross-check, and the pump-referred
    4-panel comparison. Independent of ``model_dir`` -- both may run in the
    same invocation against different directories.
    """
    if model_dir is not None:
        model = load_model_sweep(model_dir)
        print(
            f"loaded {model_dir}: {model.n_total} frequency points attempted, "
            f"{model.psat_dbm.size} VALID_SOLVED (usable)"
        )
        print()
        model_psat_vs_frequency_plot(model, outdir)
        print()
        model_psat_distribution_plot(model, outdir)
        if measured_freq is not None and measured_psat is not None:
            print()
            comparison_plot(measured_freq, measured_psat, model, outdir)
        if (
            measured_freq is not None and measured_g0_local is not None
            and measured_g0_raw is not None and measured_p1db is not None
            and measured_psat is not None
        ):
            print()
            decomposition = compute_decomposition(
                model, measured_freq, measured_g0_local, measured_g0_raw,
                measured_p1db, measured_psat,
            )
            g0_model_vs_measured_plot(decomposition, outdir)
            print()
            p1db_model_vs_measured_plot(decomposition, outdir)
            print()
            psat_decomposition_plot_and_csv(decomposition, outdir)
    elif model_points_dir is None:
        print("--model-dir/--model-points-dir not given; skipping the model-side pipeline.")

    if model_points_dir is not None:
        if model_pump_dbm is None or cube is None or p1db_result is None:
            raise ValueError(
                "--model-points-dir requires model_pump_dbm, cube and p1db_result"
            )
        points_model, curves = load_model_sweep_from_points(model_points_dir)
        print(
            f"loaded {model_points_dir}: {points_model.n_total} frequency "
            f"directories, {points_model.psat_dbm.size} crossed 1 dB compression"
        )
        print()
        energy_conservation_gate("model", points_model.psat_dbm, model_pump_dbm)
        energy_conservation_gate(
            "measured", measured_psat, cube.on_chip_pump_dbm
        )
        print("energy-conservation gate: PASS (both sides)")
        print()
        model_depletion_cross_check(points_model, model_pump_dbm)
        print()
        pump_referred_comparison_plot(
            points_model, curves, model_pump_dbm, cube, p1db_result,
            measured_freq, measured_psat, outdir,
        )
        if (
            measured_g0_local is not None and measured_g0_raw is not None
            and measured_p1db is not None
        ):
            # Phase 6 on the same merged-points model data used by Phase 7
            # above, so both phases' numbers come from one source rather
            # than silently diverging if --model-dir points elsewhere.
            print()
            decomposition = compute_decomposition(
                points_model, measured_freq, measured_g0_local, measured_g0_raw,
                measured_p1db, measured_psat,
            )
            g0_model_vs_measured_plot(decomposition, outdir)
            print()
            p1db_model_vs_measured_plot(decomposition, outdir)
            print()
            psat_decomposition_plot_and_csv(decomposition, outdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-path", type=Path, default=DEFAULT_CUBE_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Model compression-sweep directory "
             "(frequency_*/compression_summary.json, from scripts/run_compression.py "
             "--n-signal-freq). Omit to skip the model-side pipeline.",
    )
    parser.add_argument(
        "--model-points-dir", type=Path, default=None,
        help="Phase 7: model compression-sweep directory read from raw/merged "
             "frequency_*/compression_points.csv (no compression_summary.json "
             "required) -- runs the energy-conservation gate, the depletion "
             "cross-check, and the pump-referred 4-panel comparison. "
             "Independent of --model-dir.",
    )
    parser.add_argument(
        "--model-operating-point-json", type=Path,
        default=ROOT / "outputs" / "fit_operating_point_phase4" / "fit_operating_point.json",
        help="JSON with an 'on_chip_pump_dbm_norton' key, read when "
             "--model-points-dir is given (the model sweep's fixed pump has "
             "no per-frequency dependence, so one scalar covers the sweep).",
    )
    args = parser.parse_args()

    if not args.cube_path.exists():
        raise FileNotFoundError(args.cube_path)

    cube = load_cube(args.cube_path)
    pump_excluded = int(
        (np.abs(cube.frequency_ghz - MEAS_PUMP_GHZ) <= PUMP_EXCLUSION_GHZ).sum()
    )
    print(f"loaded {args.cube_path.name}: {cube.frequency_ghz.size} frequency "
          f"columns, {cube.instrument_power_dbm.size} power rows, "
          f"{pump_excluded} excluded by the pump gap")
    print(
        f"signal line: loss_B1, {cube.signal_attenuation_db.min():.2f}-"
        f"{cube.signal_attenuation_db.max():.2f} dB across the band; "
        f"pump line: loss_A10, on-chip pump {cube.on_chip_pump_dbm:.2f} dBm "
        f"(instrument {MEAS_PUMP_INSTRUMENT_DBM:.1f} dBm at {MEAS_PUMP_GHZ} GHz)"
    )

    g0 = compute_smoothed_g0(cube)
    p1db = compute_p1db_per_column(cube, g0)

    print()
    stage1_raw_g0_plot(cube, args.outdir)
    print()
    stage2_smoothed_g0_and_p1db_plot(cube, g0, p1db, args.outdir)
    print()
    _, freq_usable, p1db_fit, inlier = stage3_robust_p1db_fit_plot(
        cube, p1db, args.outdir
    )
    print()
    _, psat = stage4_psat_vs_frequency_plot(
        freq_usable,
        p1db.g0_local_db[p1db.usable],
        g0.values[p1db.usable],
        p1db_fit,
        inlier,
        args.outdir,
    )
    print()
    stage5_psat_distribution_plot(psat, args.outdir)
    print()
    phase8_statistics_report(
        cube, g0, psat, p1db.p1db_dbm[p1db.usable], cube.row0[p1db.usable],
    )
    print()
    diagnostic6_gain_curves_in_window_plot(cube, g0, args.outdir)
    print()
    diagnostic7_hard_smoothing_validation_plot(cube, g0, args.outdir)
    print()
    model_pump_dbm = None
    if args.model_points_dir is not None:
        if not args.model_operating_point_json.exists():
            raise FileNotFoundError(args.model_operating_point_json)
        operating_point = json.loads(
            args.model_operating_point_json.read_text(encoding="utf-8")
        )
        model_pump_dbm = float(operating_point["on_chip_pump_dbm_norton"])
        print(
            f"model operating point: {args.model_operating_point_json} "
            f"(status={operating_point.get('status')}), "
            f"on-chip pump {model_pump_dbm:.2f} dBm"
        )
        print()
    run_model_side(
        args.model_dir, args.outdir,
        measured_freq=freq_usable, measured_psat=psat,
        measured_g0_local=p1db.g0_local_db[p1db.usable],
        measured_g0_raw=cube.row0[p1db.usable],
        measured_p1db=p1db_fit,
        model_points_dir=args.model_points_dir,
        model_pump_dbm=model_pump_dbm,
        cube=cube,
        p1db_result=p1db,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
