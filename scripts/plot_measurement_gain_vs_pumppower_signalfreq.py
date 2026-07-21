"""Plot raw Themis transmission vs pump power and signal frequency, at a fixed
pump-frequency file. Response is already a (n_power, n_sig) cube -- this is a
direct plot, no stacking across files needed (cf. the pump-freq-axis script).

Usage:
    python scripts/plot_measurement_gain_vs_pumppower_signalfreq.py [npy_name]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_DIR = ROOT / "docs" / "14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"

DEFAULT_NPY = "105C5_7.593GHz.npy"
SIGNAL_GHZ_MIN = 5.0
SIGNAL_GHZ_MAX = 10.0

# shared colormap range across the pump-power-axis trio, see
# plot_gain_vs_pumppower_signalfreq.py
VMIN = -5.0
VMAX = 17.0


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npy_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPY
    d = np.load(MEASUREMENT_DIR / npy_name, allow_pickle=True).item()
    freq_ghz = np.asarray(d["Frequency"], dtype=float) / 1e9
    resp = np.asarray(d["Response"], dtype=float)  # (n_power, n_sig)
    powers = np.asarray(d["PumpPower"], dtype=float)

    mask = (freq_ghz >= SIGNAL_GHZ_MIN) & (freq_ghz <= SIGNAL_GHZ_MAX)
    freq_ghz = freq_ghz[mask]
    Z = resp[:, mask].T  # (n_sig, n_power)

    nominal_fp = npy_name.replace("105C5_", "").replace("GHz.npy", "")

    X, Y = np.meshgrid(powers, freq_ghz, indexing="xy")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    mesh = ax.pcolormesh(X, Y, Z, shading="nearest", cmap="viridis", vmin=VMIN, vmax=VMAX)
    ax.axhline(float(nominal_fp), color="white", lw=0.8, linestyle=":", alpha=0.6,
               label=f"signal = pump ({nominal_fp} GHz)")
    fig.colorbar(mesh, ax=ax, label="response (dB, raw/uncalibrated)")
    ax.set_xlabel("pump power (dBm)")
    ax.set_ylabel("signal frequency (GHz)")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Themis measurement 105C5, {npy_name} (raw/uncalibrated)", fontsize=11)
    fig.tight_layout()

    out = ROOT / "plots" / f"measurement_gain_vs_pumppower_signalfreq_{nominal_fp}GHz.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
