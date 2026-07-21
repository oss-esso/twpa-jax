"""Normalize measurement vs each sim to [-1,1], align on first peak, subtract.

For each of the two (measurement, sim) pairs:
  1. Take gain-vs-pump-freq at each source's own (min + span/3) power (same
     curves as plot_gain_vs_freq_third_power.py).
  2. Min-max normalize each curve to [-1, 1].
  3. Find each curve's first (lowest-frequency) peak; shift the sim's
     frequency axis so its first peak lines up with measurement's first peak.
  4. Resample both onto a common grid over the post-shift overlap and
     subtract: residual = measurement_norm - sim_norm.

Usage:
    python scripts/plot_gain_residual_aligned.py \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_123p9_cg66_halfcurrent_run_gain_map \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_recovered79_cg33_run_gain_map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import align_map_to_measurement as amm  # noqa: E402


def third_up_power(powers: np.ndarray) -> float:
    lo, hi = float(np.min(powers)), float(np.max(powers))
    return lo + (hi - lo) / 3.0


def nearest_row(freq: np.ndarray, gain_freq_power: np.ndarray, powers: np.ndarray, target_dbm: float):
    idx = int(np.argmin(np.abs(powers - target_dbm)))
    order = np.argsort(freq)
    return freq[order], gain_freq_power[order, idx], float(powers[idx])


def normalize_minmax(y: np.ndarray) -> np.ndarray:
    finite = np.isfinite(y)
    lo, hi = np.nanmin(y[finite]), np.nanmax(y[finite])
    return 2.0 * (y - lo) / (hi - lo) - 1.0


def first_peak_freq(freq: np.ndarray, y_norm: np.ndarray, prominence: float = 0.2) -> float:
    peaks, _ = find_peaks(y_norm, prominence=prominence)
    if peaks.size == 0:
        raise ValueError("no peaks found -- lower --peak-prominence")
    return float(freq[peaks[0]])


def align_and_subtract(
    meas_freq: np.ndarray, meas_norm: np.ndarray,
    sim_freq: np.ndarray, sim_norm: np.ndarray,
    prominence: float, grid_points: int,
) -> dict:
    meas_peak = first_peak_freq(meas_freq, meas_norm, prominence)
    sim_peak = first_peak_freq(sim_freq, sim_norm, prominence)
    shift = meas_peak - sim_peak
    sim_freq_shifted = sim_freq + shift

    lo = max(meas_freq.min(), sim_freq_shifted.min())
    hi = min(meas_freq.max(), sim_freq_shifted.max())
    grid = np.linspace(lo, hi, grid_points)

    meas_on_grid = np.interp(grid, meas_freq, meas_norm)
    sim_on_grid = np.interp(grid, sim_freq_shifted, sim_norm)
    residual = meas_on_grid - sim_on_grid

    return {
        "meas_peak_ghz": meas_peak,
        "sim_peak_ghz": sim_peak,
        "shift_ghz": shift,
        "grid": grid,
        "meas_on_grid": meas_on_grid,
        "sim_on_grid": sim_on_grid,
        "residual": residual,
        "rms_residual": float(np.sqrt(np.nanmean(residual ** 2))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", type=Path,
                    default=Path("docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"))
    p.add_argument("--map-dir", type=Path, action="append", required=True)
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--frequency-shift-ghz", type=float, default=0.99,
                    help="Calibration shift applied to measurement pump-freq axis before anything else.")
    p.add_argument("--freq-range-ghz", type=float, nargs=2, default=(7.0, 8.0))
    p.add_argument("--peak-prominence", type=float, default=0.2)
    p.add_argument("--grid-points", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("plots/gain_residual_aligned.png"))
    args = p.parse_args()

    lo, hi = args.freq_range_ghz

    meas = amm.load_measurement_map(args.measurement_dir, tuple(args.signal_band_ghz))
    target = third_up_power(meas["pump_power_dbm"])
    mf, mg, mp = nearest_row(meas["pump_freq_ghz"], meas["peak_gain_db"], meas["pump_power_dbm"], target)
    mf = mf + args.frequency_shift_ghz
    band = (mf >= lo) & (mf <= hi)
    mf, mg = mf[band], mg[band]
    mg_norm = normalize_minmax(mg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for map_dir in args.map_dir:
        sim = amm.load_sim_map(map_dir)
        target = third_up_power(sim["pump_power_dbm"])
        sf, sg, sp = nearest_row(sim["pump_freq_ghz"], sim["gain_db"], sim["pump_power_dbm"], target)
        band = (sf >= lo) & (sf <= hi)
        sf, sg = sf[band], sg[band]
        sg_norm = normalize_minmax(sg)

        r = align_and_subtract(mf, mg_norm, sf, sg_norm, args.peak_prominence, args.grid_points)

        label = map_dir.name
        print(
            f"{label}: measurement first peak={r['meas_peak_ghz']:.4f} GHz, "
            f"sim first peak={r['sim_peak_ghz']:.4f} GHz, shift applied={r['shift_ghz']:+.4f} GHz, "
            f"RMS residual={r['rms_residual']:.4f}"
        )

        axes[0].plot(r["grid"], r["meas_on_grid"], "-", lw=1.3, color="tab:blue",
                     label="measurement (normalized)" if map_dir is args.map_dir[0] else None)
        axes[0].plot(r["grid"], r["sim_on_grid"], "-", lw=1.3, label=f"{label} (normalized, shifted {r['shift_ghz']:+.3f} GHz)")
        axes[1].plot(r["grid"], r["residual"], "-", lw=1.3,
                     label=f"measurement - {label}  (RMS={r['rms_residual']:.3f})")

    axes[0].set_ylabel("normalized gain [-1, 1]")
    axes[0].set_title("normalized, first-peak-aligned curves")
    axes[1].set_ylabel("residual (measurement - sim)")
    axes[1].set_xlabel("pump frequency (GHz)")
    axes[1].axhline(0, color="grey", lw=0.8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
