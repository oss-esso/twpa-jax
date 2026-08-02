"""Overlay the exp32 model compression curves on the exp30 measured cuts.

Reproduces the two-panel layout of
``outputs/exp30_themis_pump_inference/themis_gain_vs_power_cuts.png`` -- raw and
Savitzky-Golay-smoothed measured traces, the ``G0 - 1 dB`` threshold, and the
measured P1dB marker -- and adds the model curve solved at the exp31 operating
point (pump 7.100 GHz, 7.2311 uA on ``designs/ipm_2c_fixed``) for the same three
signal frequencies.

Measured P1dB uses the exp30 convention (last upward crossing of ``G0 - 1``);
the model curves are monotone, so their first and last crossings coincide.
"""

from __future__ import annotations

import argparse
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
SIGNAL_GHZ = (5.296, 6.540, 7.052)
COLORS = ("tab:blue", "tab:orange", "tab:green")
PANELS = (
    ("full sweep", (-133.0, -66.0), None),
    ("compression region", (-115.0, -70.0), (-8.0, 14.5)),
)


def measured_cut(frequency_ghz: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (device signal power dBm, raw gain dB, G0) for one column."""
    data = np.load(CUBE, allow_pickle=True).item()
    freq = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - SIGNAL_LINE_LOSS_DB
    column = np.asarray(data["Response"], dtype=float)[
        :, int(np.argmin(np.abs(freq - frequency_ghz)))
    ]
    return power, column, float(np.median(column[:10]))


def last_crossing(power: np.ndarray, gain: np.ndarray, threshold: float) -> float:
    """Power at the highest-power upward crossing of ``threshold``."""
    above = np.flatnonzero(gain >= threshold)
    if above.size == 0 or above.max() + 1 >= gain.size:
        return float("nan")
    k = int(above.max())
    return float(
        np.interp(threshold, [gain[k + 1], gain[k]], [power[k + 1], power[k]])
    )


def model_cut(root: Path, frequency_ghz: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (power dBm, gain dB, summary) for one solved model sweep."""
    run_dir = root / f"fs_{frequency_ghz:.3f}ghz"
    rows = np.genfromtxt(
        run_dir / "compression_points.csv", delimiter=",", names=True
    )
    power = np.atleast_1d(rows["signal_power_dbm"])
    gain = np.atleast_1d(rows["gain_vs_off_db"])
    keep = np.isfinite(power) & np.isfinite(gain)
    summary = json.loads(
        (run_dir / "compression_summary.json").read_text(encoding="utf-8")
    )
    return power[keep], gain[keep], summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("outputs/exp32_themis_curve_match")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/exp32_themis_curve_match/themis_vs_model_cuts.png"),
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))
    table: list[str] = []
    for signal_ghz, color in zip(SIGNAL_GHZ, COLORS):
        power, raw, g0 = measured_cut(signal_ghz)
        smooth = savgol_filter(raw, 11, 2)
        p1db_meas = last_crossing(power, smooth, g0 - 1.0)
        model_power, model_gain, summary = model_cut(args.model_dir, signal_ghz)
        model_g0 = float(summary["small_signal_gain_vs_off_db"])
        p1db_model = summary.get("p1db")

        for ax in axes:
            ax.plot(power, raw, lw=0.8, color=color, alpha=0.30)
            ax.plot(
                power, smooth, lw=2.2, color=color,
                label=f"meas {signal_ghz:.3f} GHz  ($G_0$={g0:.2f} dB)",
            )
            ax.axhline(g0 - 1.0, color=color, ls=":", lw=0.9, alpha=0.8)
            ax.axvline(p1db_meas, color=color, ls="--", lw=1.2, alpha=0.9)
            ax.plot(
                model_power, model_gain, lw=1.8, color=color, ls="--",
                marker="o", ms=4.0, mfc="white", mew=1.2,
                label=f"model {signal_ghz:.3f} GHz  ($G_0$={model_g0:.2f} dB)",
            )
            if p1db_model is not None:
                ax.axvline(
                    float(p1db_model), color=color, ls="-.", lw=1.4, alpha=0.9
                )

        model_text = "wall" if p1db_model is None else f"{float(p1db_model):.2f}"
        table.append(
            f"{signal_ghz:.3f} GHz | meas {g0:5.2f} dB @ {p1db_meas:7.2f} dBm"
            f" | model {model_g0:5.2f} dB @ {model_text} dBm"
        )

    for ax, (title, xlim, ylim) in zip(axes, PANELS):
        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("signal power at device (dBm)")
        ax.set_ylabel("gain (dB)")
        ax.grid(alpha=0.3)
        ax.set_title(f"Themis 105C5 vs model — {title}")
        ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle(
        f"measured pump {MEAS_PUMP_GHZ} GHz  vs  model pump {MODEL_PUMP_GHZ} GHz "
        "(designs/ipm_2c_fixed, S=10)   "
        "dashed vline = measured P1dB, dash-dot = model P1dB",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)

    print(f"wrote {args.output}")
    for line in table:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
