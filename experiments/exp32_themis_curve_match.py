"""Compare model compression curves against the Themis 105C5 cuts.

The exp31 scan located the operating point that reproduces the measured
small-signal gain at three frequencies simultaneously: pump 7.100 GHz at
7.2311e-06 A on ``designs/ipm_2c_fixed``.  This runs the full signal-power sweep
there and overlays the result on the measurement, so the comparison is of curve
*shape* -- where the knee sits, how it bends, and where the device collapses --
rather than of a single extracted number.

The measurement pumps at 7.256 GHz, so the pump frequencies differ by 0.156 GHz;
the signal frequencies are the measured ones, not detuning-matched.  Both
choices are recorded in the summary.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

CIRCUIT_DIR = "designs/ipm_2c_fixed"
PUMP_FREQ_GHZ = 7.100
PUMP_CURRENT_A = 7.231074707853736e-06
SIGNAL_GHZ = (5.296, 6.540, 7.052)
CUBE = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
MEAS_PUMP_GHZ = 7.256
SIGNAL_LINE_LOSS_DB = 72.5
COLORS = ("tab:blue", "tab:orange", "tab:green")


def command(output_dir: Path, signal_ghz: float, sidebands: int) -> list[str]:
    """Production compression command for one signal frequency."""
    return [
        sys.executable, "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", CIRCUIT_DIR,
        "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
        "--pump-current-a", str(PUMP_CURRENT_A),
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", "10",
        "--pump-nt", "40",
        "--multitone-basis", "matched",
        "--multitone-sidebands", str(sidebands),
        "--source-port", "1", "--pump-port", "4", "--out-port", "2",
        "--attenuation-db", "0",
        "--factor-backend", "pardiso",
        "--n-signal-power", "28",
        "--signal-current-min-a", "1e-09",
        "--signal-current-max-a", "3e-06",
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "900",
        "--signal-workers", "1",
        "--signal-ghz", str(signal_ghz),
        "--allow-memory-overcommit",
    ]


def measured_cut(frequency_ghz: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (device signal power dBm, gain dB, G0) for one measured column."""
    data = np.load(CUBE, allow_pickle=True).item()
    freq = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - SIGNAL_LINE_LOSS_DB
    column = np.asarray(data["Response"], dtype=float)[
        :, int(np.argmin(np.abs(freq - frequency_ghz)))
    ]
    return power, column, float(np.median(column[:10]))


def plot(root: Path, sidebands: int) -> None:
    """Overlay every solved model curve on its measured counterpart."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for signal_ghz, color in zip(SIGNAL_GHZ, COLORS):
        power, gain, g0 = measured_cut(signal_ghz)
        smooth = savgol_filter(gain, 11, 2)
        for ax in axes:
            ax.plot(power, smooth, lw=2.0, color=color, alpha=0.55,
                    label=f"meas {signal_ghz:.3f} GHz  $G_0$={g0:.2f}")

        summary_path = root / f"fs_{signal_ghz:.3f}ghz" / "compression_summary.json"
        points_path = root / f"fs_{signal_ghz:.3f}ghz" / "compression_points.csv"
        if not points_path.exists():
            continue
        rows = np.genfromtxt(points_path, delimiter=",", names=True)
        model_power = np.atleast_1d(rows["signal_power_dbm"])
        model_gain = np.atleast_1d(rows["gain_vs_off_db"])
        keep = np.isfinite(model_power) & np.isfinite(model_gain)
        label = f"model {signal_ghz:.3f} GHz"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            label += f"  $G_0$={summary['small_signal_gain_vs_off_db']:.2f}"
        for ax in axes:
            ax.plot(model_power[keep], model_gain[keep], lw=1.6, color=color,
                    ls="--", marker="o", ms=3, label=label)

    for ax, (lo, hi) in zip(axes, ((-135.0, -62.0), (-110.0, -70.0))):
        ax.set_xlim(lo, hi)
        ax.set_xlabel("signal power at device (dBm)")
        ax.set_ylabel("gain (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_title(
        f"Themis 105C5 (pump {MEAS_PUMP_GHZ} GHz) vs model "
        f"(pump {PUMP_FREQ_GHZ} GHz, S={sidebands})"
    )
    axes[1].set_ylim(-10.0, 20.0)
    axes[1].set_title("compression region")
    fig.tight_layout()
    fig.savefig(root / "themis_vs_model_curves.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp32_themis_curve_match")
    )
    parser.add_argument("--sidebands", type=int, default=10)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for signal_ghz in SIGNAL_GHZ:
        run_dir = args.output_dir / f"fs_{signal_ghz:.3f}ghz"
        summary_path = run_dir / "compression_summary.json"
        if not args.plot_only and not summary_path.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = command(run_dir, signal_ghz, args.sidebands)
            print("run " + subprocess.list2cmdline(cmd), flush=True)
            completed = subprocess.run(cmd, check=False)
            if completed.returncode:
                print(f"FAILED fs={signal_ghz} rc={completed.returncode}", flush=True)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _, _, g0_meas = measured_cut(signal_ghz)
            results.append({
                "signal_ghz": signal_ghz,
                "status": summary.get("status"),
                "model_g0_db": summary.get("small_signal_gain_vs_off_db"),
                "measured_g0_db": g0_meas,
                "model_p1db_dbm": summary.get("p1db"),
                "n_failed_power_points": summary.get("n_failed_power_points"),
                "p1db_degraded": summary.get("p1db_degraded"),
                "max_power_balance_rel_err": summary.get("max_power_balance_rel_err"),
            })
            print(json.dumps(results[-1], indent=2), flush=True)

    plot(args.output_dir, args.sidebands)
    (args.output_dir / "curve_match_summary.json").write_text(
        json.dumps(
            {
                "circuit_dir": CIRCUIT_DIR,
                "model_pump_ghz": PUMP_FREQ_GHZ,
                "measured_pump_ghz": MEAS_PUMP_GHZ,
                "pump_current_a": PUMP_CURRENT_A,
                "pump_current_dbm_on_chip": 10.0 * math.log10(
                    PUMP_CURRENT_A**2 * 50.0 / 2.0 / 1e-3
                ),
                "sidebands": args.sidebands,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
