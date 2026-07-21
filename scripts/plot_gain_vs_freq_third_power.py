"""Plot gain vs pump frequency for measurement + N sim maps, each at its own
power = min_power + (max_power - min_power) / 3, on one shared plot.

Usage:
    python scripts/plot_gain_vs_freq_third_power.py \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_123p9_cg66_halfcurrent_run_gain_map \
        --map-dir outputs/exp10_pump_map_trailing_50x50_m30_m20_recovered79_cg33_run_gain_map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path,
                    default=Path("docs/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"))
    p.add_argument("--map-dir", type=Path, action="append", required=True)
    p.add_argument("--signal-band-ghz", type=float, nargs=2, default=(4.0, 12.0))
    p.add_argument("--frequency-shift-ghz", type=float, default=0.99)
    p.add_argument("--freq-range-ghz", type=float, nargs=2, default=(7.0, 8.0))
    p.add_argument("--out", type=Path, default=Path("plots/gain_vs_freq_third_power.png"))
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    meas = amm.load_measurement_map(args.measurement_dir, tuple(args.signal_band_ghz))
    target = third_up_power(meas["pump_power_dbm"])
    mf, mg, mp = nearest_row(meas["pump_freq_ghz"], meas["peak_gain_db"], meas["pump_power_dbm"], target)
    ax.plot(mf + args.frequency_shift_ghz, mg, "o-", ms=3, lw=1.3,
            label=f"measurement ({mp:.2f} dBm, target {target:.2f})")

    for map_dir in args.map_dir:
        sim = amm.load_sim_map(map_dir)
        target = third_up_power(sim["pump_power_dbm"])
        sf, sg, sp = nearest_row(sim["pump_freq_ghz"], sim["gain_db"], sim["pump_power_dbm"], target)
        ax.plot(sf, sg, "-", lw=1.5, label=f"{map_dir.name} ({sp:.2f} dBm, target {target:.2f})")

    ax.set_xlim(*args.freq_range_ghz)
    ax.set_xlabel("pump frequency (GHz)")
    ax.set_ylabel("peak signal gain (dB)")
    ax.set_title("gain vs pump frequency, each source at its own (min + span/3) power")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
