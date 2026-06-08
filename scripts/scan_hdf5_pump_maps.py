from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

ROOTS = [
    Path(r"D:\Projects\Thesis\Harmonia"),
    Path(r"D:\Projects\Thesis\Harmonia.jl"),
    Path(r"D:\Projects\Thesis\outputs"),
]

def find_axis(f, names):
    for name in names:
        if name in f:
            return np.array(f[name]), name
    return None, None

for root in ROOTS:
    if not root.exists():
        continue

    for path in sorted(root.rglob("*.h5")):
        try:
            with h5py.File(path, "r") as f:
                powers, pkey = find_axis(f, [
                    "simulation_info/pump_powers_dBm_axis",
                    "simulation_info/pump_powers_dBm",
                    "pump_powers_dBm_axis",
                    "pump_powers_dBm",
                ])
                freqs, fkey = find_axis(f, [
                    "simulation_info/pump_freqs_GHz_axis",
                    "simulation_info/pump_freqs_GHz",
                    "pump_freqs_GHz_axis",
                    "pump_freqs_GHz",
                ])
                gain, gkey = find_axis(f, [
                    "simulation_info/gain_2D_matrix",
                    "gain_2D_matrix",
                    "gain_2D",
                ])

                print("\n===", path, "===")
                if powers is not None:
                    powers = powers.astype(float)
                    print("power:", pkey, powers.shape, float(np.min(powers)), float(np.max(powers)))
                else:
                    print("power: MISSING")

                if freqs is not None:
                    freqs = freqs.astype(float)
                    print("freq :", fkey, freqs.shape, float(np.min(freqs)), float(np.max(freqs)))
                else:
                    print("freq : MISSING")

                if gain is not None:
                    print("gain :", gkey, gain.shape, "finite", int(np.isfinite(gain).sum()), "/", gain.size)
                else:
                    print("gain : MISSING")

        except Exception as exc:
            print("\n===", path, "===")
            print("ERROR:", repr(exc))
