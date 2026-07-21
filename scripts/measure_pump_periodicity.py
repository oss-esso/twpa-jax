"""Quantify pump-frequency comb periodicity for the measurement and sim maps.

Reuses ``plot_lj_periodicity.load_measurement`` / ``load_simulation`` for the
curve extraction, then reports two independent periodicity estimates per
curve so they cross-check each other:

  * peak count: local maxima (scipy.signal.find_peaks, prominence-gated) on
    the mean-centered curve -> average spacing = span / (n_peaks - 1).
  * FFT: dominant non-DC frequency bin of the mean-centered curve, resampled
    onto a common uniform grid -> period = 1 / dominant_frequency.

All curves are restricted to the overlapping pump-frequency window (default
7-8 GHz, the sim's native coverage) before comparison, with the measurement
shifted by --frequency-shift-ghz to match the existing lj-periodicity
calibration convention.

Usage:
    python scripts/measure_pump_periodicity.py \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_123p9_cg66_halfcurrent_run_gain_map \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_recovered79_cg33_run_gain_map \
        --target-power-dbm -20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_lj_periodicity as ljp  # noqa: E402


def resample_common_grid(
    freq: np.ndarray, gain: np.ndarray, lo: float, hi: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(freq)
    freq, gain = freq[order], gain[order]
    finite = np.isfinite(gain)
    freq, gain = freq[finite], gain[finite]
    if freq.size < 2:
        raise ValueError("fewer than 2 finite points in this curve/window -- pick a lower power")
    grid = np.linspace(lo, hi, n)
    return grid, np.interp(grid, freq, gain)


def periodicity_metrics(
    grid: np.ndarray, gain: np.ndarray, peak_prominence_db: float
) -> dict:
    centered = gain - np.nanmean(gain)
    span = float(grid[-1] - grid[0])

    peaks, _ = find_peaks(centered, prominence=peak_prominence_db)
    n_peaks = int(peaks.size)
    peak_spacing_ghz = float(span / (n_peaks - 1)) if n_peaks >= 2 else float("nan")

    # Require at least 2 full cycles inside the window before trusting a
    # period estimate -- otherwise the "dominant" FFT bin is just the
    # window's own slow envelope/trend, not real comb periodicity.
    n = grid.size
    dx = float(grid[1] - grid[0])
    spectrum = np.abs(np.fft.rfft(centered))
    freqs_per_ghz = np.fft.rfftfreq(n, d=dx)
    min_cycles_per_ghz = 2.0 / span
    valid = freqs_per_ghz >= min_cycles_per_ghz
    if np.any(valid):
        candidate_idx = np.flatnonzero(valid)
        dominant_idx = candidate_idx[np.argmax(spectrum[valid])]
        dominant_cycles_per_ghz = float(freqs_per_ghz[dominant_idx])
        fft_period_ghz = float(1.0 / dominant_cycles_per_ghz)
    else:
        fft_period_ghz = float("nan")

    return {
        "n_peaks": n_peaks,
        "peak_spacing_ghz": peak_spacing_ghz,
        "fft_period_ghz": fft_period_ghz,
        "span_ghz": span,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", type=Path,
                    default=Path("docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"))
    p.add_argument("--map-dir", type=Path, action="append", required=True)
    p.add_argument("--target-power-dbm", type=float, default=-20.0)
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--frequency-shift-ghz", type=float, default=0.99)
    p.add_argument("--freq-window-ghz", type=float, nargs=2, default=(7.0, 8.0))
    p.add_argument("--grid-points", type=int, default=400)
    p.add_argument("--peak-prominence-db", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("plots/pump_periodicity_metrics.json"))
    args = p.parse_args()

    lo, hi = args.freq_window_ghz
    results: dict[str, dict] = {}

    meas_freq, meas_gain, meas_power = ljp.load_measurement(
        args.measurement_dir, args.target_power_dbm, tuple(args.signal_band_ghz)
    )
    grid, gain = resample_common_grid(
        meas_freq + args.frequency_shift_ghz, meas_gain, lo, hi, args.grid_points
    )
    results[f"measurement ({args.measurement_dir.name}, {meas_power:.2f} dBm)"] = {
        **periodicity_metrics(grid, gain, args.peak_prominence_db),
        "selected_power_dbm": meas_power,
    }

    for map_dir in args.map_dir:
        sim_freq, sim_gain, sim_power = ljp.load_simulation(map_dir, args.target_power_dbm)
        grid, gain = resample_common_grid(sim_freq, sim_gain, lo, hi, args.grid_points)
        results[f"{map_dir.name} ({sim_power:.2f} dBm)"] = {
            **periodicity_metrics(grid, gain, args.peak_prominence_db),
            "selected_power_dbm": sim_power,
        }

    print(f"frequency window: {lo}-{hi} GHz  (measurement shifted +{args.frequency_shift_ghz} GHz)")
    print(f"{'source':70s} {'n_peaks':>8} {'peak_spacing(GHz)':>18} {'fft_period(GHz)':>17}")
    for name, r in results.items():
        print(f"{name:70s} {r['n_peaks']:8d} {r['peak_spacing_ghz']:18.4f} {r['fft_period_ghz']:17.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
