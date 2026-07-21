"""Plot S21 gain (dB) vs pump power and signal frequency, fixed pump frequency.

Sibling of plot_gain_vs_pumpfreq_signalfreq.py -- reads map_spectrum.npz
written by run_gain_map.py (--n-frequency 1, signal spectrum enabled) and
pcolormeshes gain_db over the (pump_power, signal_freq) plane.

Usage:
    python scripts/plot_gain_vs_pumppower_signalfreq.py <run-dir> <title> <out-png>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# shared colormap range across the pump-power-axis trio (this + the two
# lj79_cg33_scale2/lj123p9_cg66_scale1 sim maps + the Themis measurement map)
VMIN = -5.0
VMAX = 17.0


def plot_one(run_dir: Path, title: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(run_dir / "map_spectrum.npz")
    powers = d["pump_power_dbm"]  # (n_power,)
    gain = d["gain_spectrum_db"][:, 0, :]  # (n_power, n_off)
    signal_ghz = d["signal_ghz"][:, 0]  # (n_off,) -- single pump freq column
    pump_freq_ghz = float(d["pump_frequency_ghz"][0])

    X, Y = np.meshgrid(powers, signal_ghz, indexing="xy")
    Z = gain.T  # (n_off, n_power)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    mesh = ax.pcolormesh(X, Y, Z, shading="nearest", cmap="viridis",
                          vmin=VMIN, vmax=VMAX)
    ax.axhline(pump_freq_ghz, color="white", lw=0.8, linestyle=":", alpha=0.6,
               label=f"signal = pump ({pump_freq_ghz:.4f} GHz)")
    fig.colorbar(mesh, ax=ax, label="gain (dB)")
    ax.set_xlabel("pump power (dBm)")
    ax.set_ylabel("signal frequency (GHz)")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{title}  (fp={pump_freq_ghz:.4f} GHz)", fontsize=11)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: plot_gain_vs_pumppower_signalfreq.py <run-dir> <title> <out-png>")
        return 1
    plot_one(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
