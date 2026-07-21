"""Count diagonal high-gain/low-gain streak pairs in a 2-D gain map.

The gain map (pump freq x pump power) shows diagonal ridges of high gain
separated by troughs -- a fixed-power horizontal slice cuts through
different ridges at different points depending on which row you pick (very
sensitive to the exact power chosen, see measure_pump_periodicity.py).
Collapsing to the per-frequency ENVELOPE -- the best (max) gain achievable
at each pump frequency, over ALL powers -- traces the top of each diagonal
ridge regardless of which power it peaks at, and is far more robust.

One high+low streak pair = one period. Reported two ways:

  * peak count on the envelope curve (integer-ish, prominence-gated).
  * FFT dominant period of the envelope -> fractional period count
    (span / period), matching the "5.5 periods" kind of by-eye count.

Usage:
    python scripts/measure_map_envelope_periodicity.py \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_123p9_cg66_halfcurrent_run_gain_map \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_recovered79_cg33_run_gain_map \
        --measurement-dir docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK
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

import align_map_to_measurement as amm  # noqa: E402


def envelope_over_power(freq: np.ndarray, gain_freq_power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(freq)
    freq = freq[order]
    gain_freq_power = gain_freq_power[order]
    envelope = np.nanmax(gain_freq_power, axis=1)
    return freq, envelope


def resample_common_grid(
    freq: np.ndarray, envelope: np.ndarray, lo: float, hi: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(envelope)
    freq, envelope = freq[finite], envelope[finite]
    if freq.size < 2:
        raise ValueError("fewer than 2 finite envelope points in this window")
    grid = np.linspace(lo, hi, n)
    return grid, np.interp(grid, freq, envelope)


def periodicity_metrics(grid: np.ndarray, envelope: np.ndarray, peak_prominence_db: float) -> dict:
    centered = envelope - np.nanmean(envelope)
    span = float(grid[-1] - grid[0])

    peaks, _ = find_peaks(centered, prominence=peak_prominence_db)
    n_peaks_streaks = int(peaks.size)

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
        n_periods_fft = float(span * dominant_cycles_per_ghz)
    else:
        fft_period_ghz = float("nan")
        n_periods_fft = float("nan")

    return {
        "n_high_gain_streaks": n_peaks_streaks,
        "fft_period_ghz": fft_period_ghz,
        "n_periods_fft": n_periods_fft,
        "span_ghz": span,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measurement-dir", type=Path,
                    default=Path("docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"))
    p.add_argument("--map-dir", type=Path, action="append", required=True)
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--frequency-shift-ghz", type=float, default=0.99)
    p.add_argument("--freq-window-ghz", type=float, nargs=2, default=(7.0, 8.0))
    p.add_argument("--grid-points", type=int, default=400)
    p.add_argument("--peak-prominence-db", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("plots/map_envelope_periodicity_metrics.json"))
    p.add_argument("--plot", type=Path, default=Path("plots/map_envelope_periodicity.png"))
    args = p.parse_args()

    lo, hi = args.freq_window_ghz
    results: dict[str, dict] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    meas = amm.load_measurement_map(args.measurement_dir, tuple(args.signal_band_ghz))
    mf, menv = envelope_over_power(meas["pump_freq_ghz"], meas["peak_gain_db"])
    grid, env = resample_common_grid(mf + args.frequency_shift_ghz, menv, lo, hi, args.grid_points)
    name = f"measurement ({args.measurement_dir.name})"
    results[name] = periodicity_metrics(grid, env, args.peak_prominence_db)
    curves[name] = (grid, env)

    for map_dir in args.map_dir:
        sim = amm.load_sim_map(map_dir)
        sf, senv = envelope_over_power(sim["pump_freq_ghz"], sim["gain_db"])
        grid, env = resample_common_grid(sf, senv, lo, hi, args.grid_points)
        results[map_dir.name] = periodicity_metrics(grid, env, args.peak_prominence_db)
        curves[map_dir.name] = (grid, env)

    print(f"frequency window: {lo}-{hi} GHz  (measurement shifted +{args.frequency_shift_ghz} GHz)")
    print(f"{'source':70s} {'n_streaks':>10} {'fft_period(GHz)':>16} {'n_periods_fft':>14}")
    for name, r in results.items():
        print(
            f"{name:70s} {r['n_high_gain_streaks']:10d} "
            f"{r['fft_period_ghz']:16.4f} {r['n_periods_fft']:14.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, (grid, env) in curves.items():
        ax.plot(grid, env - np.nanmean(env), "-", lw=1.5, label=name)
    ax.set_xlabel("pump frequency (GHz)")
    ax.set_ylabel("envelope gain, mean-centered (dB)")
    ax.set_title("per-frequency envelope (max gain over all powers) -- mean-centered")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
