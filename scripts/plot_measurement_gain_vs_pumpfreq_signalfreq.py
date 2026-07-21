"""Plot raw Themis transmission vs pump frequency and signal frequency, at the
PumpPower row nearest a fixed target, using the per-pump-frequency .npy files.

Companion to plot_gain_vs_pumpfreq_signalfreq.py (sim side): same axes/style,
built from measurement instead of a run_gain_map.py spectrum cube. ``Response``
is treated as raw/uncalibrated gain_db, same convention as
align_map_to_measurement.py and plot_spectrum_compare.py (no pump-off baseline
subtraction, no df/dP/dG fit).

Usage:
    python scripts/plot_measurement_gain_vs_pumpfreq_signalfreq.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_DIR = ROOT / "docs" / "14.18.08_Themis_SetupAug25_noVTS_transmission_15mK"

TARGET_PUMP_POWER_DBM = -25.0
SIGNAL_GHZ_MIN = 5.0
SIGNAL_GHZ_MAX = 10.0

# shared colormap range across the pump-freq-axis trio, see
# plot_gain_vs_pumpfreq_signalfreq.py
VMIN = -3.0
VMAX = 13.0

FNAME_RE = re.compile(r"105C5_([0-9.]+)GHz\.npy")


def load_all() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    files = sorted(MEASUREMENT_DIR.glob("105C5_*GHz.npy"))
    pump_freqs = []
    rows = []
    used_power = None
    sig_ghz = None
    for f in files:
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        nominal_fp = float(m.group(1))
        d = np.load(f, allow_pickle=True).item()
        freq_ghz = np.asarray(d["Frequency"], dtype=float) / 1e9
        resp = np.asarray(d["Response"], dtype=float)
        powers = np.asarray(d["PumpPower"], dtype=float)
        idx = int(np.argmin(np.abs(powers - TARGET_PUMP_POWER_DBM)))
        if sig_ghz is None:
            sig_ghz = freq_ghz
        elif not np.array_equal(sig_ghz, freq_ghz):
            raise ValueError(f"{f.name}: Frequency axis differs from other files")
        if used_power is None:
            used_power = float(powers[idx])
        pump_freqs.append(nominal_fp)
        rows.append(resp[idx, :])

    pump_freqs = np.asarray(pump_freqs)
    order = np.argsort(pump_freqs)
    pump_freqs = pump_freqs[order]
    cube = np.asarray(rows)[order]  # (n_pump_freq, n_sig)
    mask = (sig_ghz >= SIGNAL_GHZ_MIN) & (sig_ghz <= SIGNAL_GHZ_MAX)
    return pump_freqs, sig_ghz[mask], cube[:, mask], used_power


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pump_freqs, sig_ghz, cube, used_power = load_all()
    print(f"loaded {len(pump_freqs)} pump-frequency files, "
          f"pump range {pump_freqs.min():.3f}-{pump_freqs.max():.3f} GHz, "
          f"nearest pump power to {TARGET_PUMP_POWER_DBM} dBm target = {used_power:.3f} dBm")

    X, Y = np.meshgrid(pump_freqs, sig_ghz, indexing="xy")
    Z = cube.T  # (n_sig, n_pump_freq)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    mesh = ax.pcolormesh(X, Y, Z, shading="nearest", cmap="viridis", vmin=VMIN, vmax=VMAX)
    ax.plot(pump_freqs, pump_freqs, color="white", lw=0.8, linestyle=":", alpha=0.6,
            label="signal = pump")
    fig.colorbar(mesh, ax=ax, label="response (dB, raw/uncalibrated)")
    ax.set_xlabel("pump frequency (GHz)")
    ax.set_ylabel("signal frequency (GHz)")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Themis measurement 105C5 (pp={used_power:.2f} dBm, raw/uncalibrated)",
                 fontsize=11)
    fig.tight_layout()

    out = ROOT / "plots" / "measurement_gain_vs_pumpfreq_signalfreq.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
