"""Plot the late-window TD signal input and output waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--signal-current-a", type=float, default=1e-10)
    p.add_argument("--signal-ghz", type=float, default=7.4)
    p.add_argument("--pump-ghz", type=float, default=7.9)
    args = p.parse_args()
    data = np.load(args.run / "signal_late_window.npz")
    theta = np.asarray(data["theta"], dtype=float)
    vout = np.asarray(data["output_voltage_v"], dtype=float)
    time_ns = theta / (2.0 * np.pi * args.pump_ghz)  # theta is pump phase
    time_ns *= 1e-0  # GHz and ns cancel
    iin = args.signal_current_a * np.cos(2.0 * np.pi * args.signal_ghz / args.pump_ghz * theta / (2.0 * np.pi))

    if "max_abs_i_over_ic" in data:
        metric = np.asarray(data["max_abs_i_over_ic"], dtype=float)
        phase = np.mod(theta / (2.0 * np.pi), 1.0)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for period in np.unique(np.floor(theta / (2.0 * np.pi)).astype(int)):
            mask = np.floor(theta / (2.0 * np.pi)).astype(int) == period
            ax2.plot(phase[mask], metric[mask], color="tab:red", alpha=0.35, lw=0.8)
        ax2.set(xlabel="pump-period phase", ylabel="max junction |I/Ic|",
                title="Junction current stress, periods aligned")
        ax2.grid(alpha=0.25)
        out2 = args.run / "i_over_ic_aligned_periods.png"
        fig2.tight_layout()
        fig2.savefig(out2, dpi=160)
        plt.close(fig2)
        print(out2)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_ns - time_ns[0], iin * 1e9, color="tab:blue", lw=1.0)
    axes[0].set_ylabel("input current (nA peak)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(time_ns - time_ns[0], vout * 1e6, color="tab:orange", lw=1.0)
    axes[1].set_ylabel("output voltage (µV)")
    axes[1].set_xlabel("time within late window (ns)")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"TD signal injection: {args.signal_ghz:.3f} GHz")
    fig.tight_layout()
    out = args.run / "signal_input_output_vs_time.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
