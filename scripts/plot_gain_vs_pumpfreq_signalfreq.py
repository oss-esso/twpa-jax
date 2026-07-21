"""Plot S21 gain (dB) vs pump frequency and signal frequency, fixed pump power.

Reads map_spectrum.npz written by run_gain_map.py (--n-power 1, signal
spectrum enabled) and pcolormeshes gain_db over the (pump_freq, signal_freq)
plane. signal_ghz is per-offset-per-pump-freq (fp+offset), so the mesh uses
the 2D signal_ghz array directly rather than a fixed offset axis.

Usage:
    python scripts/plot_gain_vs_pumpfreq_signalfreq.py <run-dir> <title> <out-png>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# shared colormap range across the pump-freq-axis trio (this + the two
# lj79_cg33_scale2/lj123p9_cg66_scale1 sim maps + the Themis measurement map)
VMIN = -3.0
VMAX = 13.0


def plot_one(run_dir: Path, title: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(run_dir / "map_spectrum.npz")
    freqs = d["pump_frequency_ghz"]  # (n_freq,)
    gain = d["gain_spectrum_db"][0]  # (n_freq, n_off)
    signal_ghz = d["signal_ghz"]  # (n_off, n_freq)
    pump_power_dbm = float(d["pump_power_dbm"][0])

    X = np.tile(freqs[None, :], (signal_ghz.shape[0], 1))  # (n_off, n_freq)
    Z = gain.T  # (n_off, n_freq)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    mesh = ax.pcolormesh(X, signal_ghz, Z, shading="nearest", cmap="viridis",
                          vmin=VMIN, vmax=VMAX)
    ax.plot(freqs, freqs, color="white", lw=0.8, linestyle=":", alpha=0.6,
            label="signal = pump")
    fig.colorbar(mesh, ax=ax, label="gain (dB)")
    ax.set_xlabel("pump frequency (GHz)")
    ax.set_ylabel("signal frequency (GHz)")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{title}  (pp={pump_power_dbm:.2f} dBm)", fontsize=11)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: plot_gain_vs_pumpfreq_signalfreq.py <run-dir> <title> <out-png>")
        return 1
    plot_one(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
