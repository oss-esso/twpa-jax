"""Overlay every solved 2c model compression curve on the Themis cuts.

Supersedes the three-frequency figure from `exp32_overlay_cuts.py`.  Model
points are gathered from every run made against `designs/ipm_2c_fixed` at the
exp31 operating point: the long exp32 sweeps at 5.296, 6.540 and 7.052 GHz, the
short exp33 probes that added 6.300, 6.440, 6.640 and 6.800 GHz, and the exp33
deep run that carried 6.540 GHz past the continuation wall to -85 dBm.

Runs at the same frequency are merged and de-duplicated by signal power, keeping
the point from the longer sweep where they overlap.  Only S=10 runs are used;
the S=6 probe is a basis-convergence check, not a production curve, and mixing
bases in one trace would hide that.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

CUBE = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
MEAS_PUMP_GHZ = 7.256
MODEL_PUMP_GHZ = 7.100
SIGNAL_LINE_LOSS_DB = 72.5

# frequency -> run directories, longest sweep first so it wins on overlap
RUNS: dict[float, tuple[str, ...]] = {
    5.296: ("outputs/exp32_themis_curve_match/fs_5.296ghz",),
    6.300: ("outputs/exp33_wall_probe/fs_6.300ghz_s10",),
    6.440: ("outputs/exp33_wall_probe/fs_6.440ghz_s10",),
    6.540: (
        "outputs/exp32_themis_curve_match/fs_6.540ghz",
        "outputs/exp33_wall_probe/fs_6.540ghz_s10",
        "outputs/exp33_wall_probe/fs_6.540ghz_s10_deep",
    ),
    6.640: ("outputs/exp33_wall_probe/fs_6.640ghz_s10",),
    6.800: ("outputs/exp33_wall_probe/fs_6.800ghz_s10",),
    7.052: ("outputs/exp32_themis_curve_match/fs_7.052ghz",),
}
COLORS = (
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:cyan",
)


def measured_cut(frequency_ghz: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Device-referred signal power, raw gain, and G0 for one column."""
    data = np.load(CUBE, allow_pickle=True).item()
    freq = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - SIGNAL_LINE_LOSS_DB
    column = np.asarray(data["Response"], dtype=float)[
        :, int(np.argmin(np.abs(freq - frequency_ghz)))
    ]
    return power, column, float(np.median(column[:10]))


def model_curve(directories: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Merged (power, gain) across runs, first directory winning on overlap."""
    merged: dict[float, float] = {}
    for directory in reversed(directories):
        path = Path(directory) / "compression_points.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["status"] != "VALID_SOLVED":
                    continue
                gain = float(row["gain_vs_off_db"])
                if np.isfinite(gain):
                    merged[round(float(row["signal_power_dbm"]), 3)] = gain
    power = np.array(sorted(merged))
    return power, np.array([merged[p] for p in power])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/presentation/2c_compression_vs_themis.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    table: list[dict[str, object]] = []
    for (signal_ghz, directories), color in zip(sorted(RUNS.items()), COLORS):
        power, raw, g0 = measured_cut(signal_ghz)
        smooth = savgol_filter(raw, 11, 2)
        model_power, model_gain = model_curve(directories)
        model_g0 = float(model_gain[0]) if model_gain.size else float("nan")

        for ax in axes:
            ax.plot(power, smooth, lw=1.9, color=color, alpha=0.75)
            ax.plot(model_power, model_gain, lw=1.5, color=color, ls="--",
                    marker="o", ms=4.5, mfc="white", mew=1.3)
        # One legend entry per frequency: the measured and model traces share a
        # colour, so two entries each would double the legend for no gain.
        axes[0].plot([], [], lw=1.9, color=color,
                     label=f"{signal_ghz:.3f} GHz — meas {g0:.2f} / "
                           f"model {model_g0:.2f} dB")
        table.append({
            "signal_ghz": signal_ghz,
            "measured_g0_db": g0,
            "model_g0_db": model_g0,
            "n_model_points": int(model_power.size),
            "model_max_power_dbm": (
                float(model_power.max()) if model_power.size else None
            ),
        })

    axes[0].plot([], [], lw=1.9, color="0.35", label="solid = Themis measurement")
    axes[0].plot([], [], lw=1.5, color="0.35", ls="--", marker="o", ms=4.5,
                 mfc="white", label="dashed = multitone HB model")
    for ax, (lo, hi), ylim in zip(
        axes, ((-135.0, -66.0), (-115.0, -72.0)), (None, (-6.0, 18.0))
    ):
        ax.set_xlim(lo, hi)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("signal power at device (dBm)")
        ax.set_ylabel("gain (dB)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title(
        f"2c — Themis 105C5 (pump {MEAS_PUMP_GHZ} GHz) vs model "
        f"(pump {MODEL_PUMP_GHZ} GHz, S=10)"
    )
    axes[1].set_title("compression region")
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    plt.close(fig)

    print(f"wrote {args.output}")
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
