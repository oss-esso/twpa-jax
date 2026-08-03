"""Plot the 2c compression curves paired with measurement two ways.

Left panel pairs each model curve with the measured column at the **same signal
frequency**. Right panel pairs it with the measured column whose **small-signal
gain matches**, which generally sits at a different frequency because the
model's gain band is offset and rippled relative to the device.

The same-frequency pairing charges the model for its gain-band error twice:
P1dB depends on gain, so comparing a 9.5 dB model trace against an 11.8 dB
measured trace reports a saturation difference that is really a gain
difference. The gain-matched pairing removes that, and is the comparison to
read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter

CUBE = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
SIGNAL_LINE_LOSS_DB = 72.5
MEAS_PUMP_GHZ = 7.256
MODEL_PUMP_GHZ = 7.100
PUMP_EXCLUSION_GHZ = 0.15
POWER_LIMITS = (-112.0, -74.0)


def measured_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (frequency GHz, device power dBm, smoothed response, G0)."""
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - SIGNAL_LINE_LOSS_DB
    response = np.asarray(data["Response"], dtype=float)
    smooth = savgol_filter(response, 11, 2, axis=0)
    return frequency, power, smooth, np.median(response[:10, :], axis=0)


def model_curves(run_dir: Path) -> list[dict[str, object]]:
    """Solved (frequency, power, gain, G0) per model frequency."""
    curves: list[dict[str, object]] = []
    for directory in sorted(run_dir.glob("frequency_*")):
        points = directory / "compression_points.csv"
        if not points.exists():
            continue
        match = re.search(r"_(\d+\.\d+)ghz", directory.name)
        if match is None:
            continue
        with points.open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r["status"] == "VALID_SOLVED"]
        if not rows:
            continue
        rows.sort(key=lambda r: float(r["signal_power_dbm"]))
        power = np.array([float(r["signal_power_dbm"]) for r in rows])
        gain = np.array([float(r["gain_vs_off_db"]) for r in rows])
        curves.append({
            "signal_ghz": float(match.group(1)),
            "power_dbm": power,
            "gain_db": gain,
            "g0_db": float(gain[0]),
        })
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("outputs/exp45_2c_p1db_vs_frequency"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/presentation/2c_curves_two_way.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    freq, power, smooth, g0 = measured_cube()
    usable = np.abs(freq - MEAS_PUMP_GHZ) > PUMP_EXCLUSION_GHZ
    curves = model_curves(args.run_dir)
    if not curves:
        raise SystemExit(f"no model curves under {args.run_dir}")

    ghz = np.array([c["signal_ghz"] for c in curves])
    norm = Normalize(vmin=ghz.min(), vmax=ghz.max())
    colormap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.4), sharey=True)
    pairing: list[dict[str, float]] = []
    for curve in curves:
        color = colormap(norm(curve["signal_ghz"]))
        same_index = int(np.argmin(np.abs(freq - curve["signal_ghz"])))
        # Gain match is taken over columns away from the pump notch, where G0 is
        # not a parametric gain at all.
        candidates = np.where(usable, np.abs(g0 - curve["g0_db"]), np.inf)
        gain_index = int(np.argmin(candidates))

        for ax, index in ((axes[0], same_index), (axes[1], gain_index)):
            ax.plot(power, smooth[:, index], lw=1.6, color=color, alpha=0.85)
            ax.plot(curve["power_dbm"], curve["gain_db"], lw=1.4, ls="--",
                    color=color, marker="o", ms=3.5, mfc="white", mew=1.0)

        pairing.append({
            "model_ghz": float(curve["signal_ghz"]),
            "model_g0_db": float(curve["g0_db"]),
            "same_frequency_meas_g0_db": float(g0[same_index]),
            "gain_matched_meas_ghz": float(freq[gain_index]),
            "gain_matched_meas_g0_db": float(g0[gain_index]),
            "gain_match_residual_db": float(
                abs(g0[gain_index] - curve["g0_db"])
            ),
        })

    mean_same = float(np.mean([
        p["same_frequency_meas_g0_db"] - p["model_g0_db"] for p in pairing
    ]))
    mean_matched = float(np.mean([
        p["gain_match_residual_db"] for p in pairing
    ]))
    axes[0].set_title(
        "paired at the SAME FREQUENCY\n"
        f"measured $G_0$ exceeds model by {mean_same:.2f} dB on average — "
        "the curves start apart"
    )
    axes[1].set_title(
        "paired at MATCHED $G_0$\n"
        f"gain matched to {mean_matched:.3f} dB — "
        "the comparison that isolates saturation"
    )
    for ax in axes:
        ax.set_xlim(*POWER_LIMITS)
        ax.set_xlabel("signal power at device (dBm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("gain (dB)")
    axes[0].set_ylim(0.0, 14.0)

    solid = plt.Line2D([], [], color="0.3", lw=1.6, label="Themis measurement")
    dashed = plt.Line2D([], [], color="0.3", lw=1.4, ls="--", marker="o",
                        ms=3.5, mfc="white", label="multitone HB model")
    axes[0].legend(handles=[solid, dashed], fontsize=9, loc="lower left")

    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=colormap), ax=axes, fraction=0.025, pad=0.015
    )
    bar.set_label("model signal frequency (GHz)")
    fig.suptitle(
        f"2c compression curves — model (pump {MODEL_PUMP_GHZ} GHz) vs "
        f"Themis 105C5 (pump {MEAS_PUMP_GHZ} GHz), {len(curves)} frequencies",
        fontsize=12,
    )
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.output}")
    print(f'{"model GHz":>10} {"model G0":>9} {"same-f G0":>10} '
          f'{"matched GHz":>12} {"matched G0":>11} {"resid":>7}')
    for row in pairing:
        print(
            f'{row["model_ghz"]:10.2f} {row["model_g0_db"]:9.2f} '
            f'{row["same_frequency_meas_g0_db"]:10.2f} '
            f'{row["gain_matched_meas_ghz"]:12.3f} '
            f'{row["gain_matched_meas_g0_db"]:11.2f} '
            f'{row["gain_match_residual_db"]:7.3f}'
        )
    args.output.with_suffix(".json").write_text(
        json.dumps({
            "mean_same_frequency_g0_gap_db": mean_same,
            "mean_gain_match_residual_db": mean_matched,
            "pairing": pairing,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
