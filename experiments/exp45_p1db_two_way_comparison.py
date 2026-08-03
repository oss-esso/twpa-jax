"""Compare 2c model P1dB against measurement two ways: same frequency, same gain.

The frequency-by-frequency comparison is the obvious one but it is confounded:
the model's gain band is offset and rippled relative to the device, so at a given
frequency the two are rarely at the same gain, and P1dB depends strongly on gain.

The gain-matched comparison removes that. For each model frequency it finds the
measured columns whose small-signal gain matches to within a tolerance and
compares P1dB against those. If the model's saturation physics is right, the two
clouds should lie on top of each other in the P1dB-versus-gain plane even where
the frequency-by-frequency curves disagree.

The pure pump-depletion line ``P1dB = Pp + 10log10[(10^0.1-1)/(2 G_lin)]`` is
drawn for orientation only, never as a bound: it is one mechanism among several.
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
from scipy.signal import savgol_filter

CUBE = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
SIGNAL_LINE_LOSS_DB = 72.5
MEAS_PUMP_GHZ = 7.256
MEAS_PUMP_ON_CHIP_DBM = -65.98
MODEL_PUMP_GHZ = 7.100
PUMP_EXCLUSION_GHZ = 0.15
GAIN_MATCH_TOL_DB = 0.25
MIN_MEASURED_G0_DB = 3.0


def measured_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column measured (frequency GHz, G0 dB, P1dB dBm)."""
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    power = np.asarray(data["SignalPower"], dtype=float) - SIGNAL_LINE_LOSS_DB
    response = np.asarray(data["Response"], dtype=float)
    g0 = np.median(response[:10, :], axis=0)
    p1db = np.full(frequency.size, np.nan)
    for index in range(frequency.size):
        smooth = savgol_filter(response[:, index], 11, 2)
        threshold = g0[index] - 1.0
        below = np.flatnonzero(
            (smooth[:-1] >= threshold) & (smooth[1:] < threshold)
        )
        if below.size:
            first = int(below[0])
            p1db[index] = np.interp(
                threshold,
                [smooth[first + 1], smooth[first]],
                [power[first + 1], power[first]],
            )
    usable = (
        (np.abs(frequency - MEAS_PUMP_GHZ) > PUMP_EXCLUSION_GHZ)
        & (g0 > MIN_MEASURED_G0_DB)
        & np.isfinite(p1db)
    )
    return frequency[usable], g0[usable], p1db[usable]


def model_table(run_dir: Path) -> list[dict[str, float]]:
    """Per-frequency model (frequency GHz, G0 dB, P1dB dBm)."""
    rows: list[dict[str, float]] = []
    for directory in sorted(run_dir.glob("frequency_*")):
        summary = json.loads(
            (directory / "compression_summary.json").read_text(encoding="utf-8")
        )
        p1db = summary.get("p1db")
        gain = summary.get("small_signal_gain_vs_off_db")
        if p1db is None or gain is None:
            continue
        match = re.search(r"_(\d+\.\d+)ghz", directory.name)
        if match is None:
            continue
        rows.append({
            "signal_ghz": float(match.group(1)),
            "model_g0_db": float(gain),
            "model_p1db_dbm": float(p1db),
            "n_failed_power_points": float(
                summary.get("n_failed_power_points") or 0
            ),
        })
    return rows


def depletion_reference_dbm(gain_db: np.ndarray) -> np.ndarray:
    """P1dB if pump depletion were the only saturation mechanism."""
    return MEAS_PUMP_ON_CHIP_DBM + 10.0 * np.log10(
        (10.0**0.1 - 1.0) / (2.0 * 10.0 ** (gain_db / 10.0))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("outputs/exp45_2c_p1db_vs_frequency"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/presentation")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meas_ghz, meas_g0, meas_p1db = measured_table()
    rows = model_table(args.run_dir)
    if not rows:
        raise SystemExit(f"no model frequencies found under {args.run_dir}")

    for row in rows:
        # Same frequency: interpolate the measured columns onto the model grid.
        row["meas_g0_at_same_ghz_db"] = float(
            np.interp(row["signal_ghz"], meas_ghz, meas_g0)
        )
        row["meas_p1db_at_same_ghz_dbm"] = float(
            np.interp(row["signal_ghz"], meas_ghz, meas_p1db)
        )
        # Same gain: every measured column within tolerance of the model's G0,
        # regardless of where in the band it sits.
        matched = np.abs(meas_g0 - row["model_g0_db"]) <= GAIN_MATCH_TOL_DB
        row["n_gain_matched_columns"] = float(np.count_nonzero(matched))
        row["meas_p1db_at_same_gain_dbm"] = (
            float(np.median(meas_p1db[matched])) if matched.any() else float("nan")
        )
        row["meas_p1db_at_same_gain_spread_db"] = (
            float(np.percentile(meas_p1db[matched], 84)
                  - np.percentile(meas_p1db[matched], 16))
            if np.count_nonzero(matched) > 2 else float("nan")
        )
        row["delta_same_frequency_db"] = (
            row["model_p1db_dbm"] - row["meas_p1db_at_same_ghz_dbm"]
        )
        row["delta_same_gain_db"] = (
            row["model_p1db_dbm"] - row["meas_p1db_at_same_gain_dbm"]
        )

    with (args.output_dir / "2c_p1db_two_way.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ghz = np.array([r["signal_ghz"] for r in rows])
    model_g0 = np.array([r["model_g0_db"] for r in rows])
    model_p1db = np.array([r["model_p1db_dbm"] for r in rows])
    same_f = np.array([r["meas_p1db_at_same_ghz_dbm"] for r in rows])
    same_g = np.array([r["meas_p1db_at_same_gain_dbm"] for r in rows])
    meas_g0_at_f = np.array([r["meas_g0_at_same_ghz_db"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))

    axes[0].plot(meas_ghz, meas_g0, lw=1.2, color="tab:orange", alpha=0.55,
                 label="measured $G_0$")
    axes[0].plot(ghz, model_g0, lw=2.0, color="tab:blue", marker="o", ms=5,
                 label="model $G_0$")
    axes[0].axvline(MODEL_PUMP_GHZ, color="tab:blue", ls=":", lw=1.1)
    axes[0].axvline(MEAS_PUMP_GHZ, color="tab:orange", ls=":", lw=1.1)
    axes[0].set_xlabel("signal frequency (GHz)")
    axes[0].set_ylabel("small-signal gain (dB)")
    axes[0].set_title("gain band (pumps marked)")

    axes[1].plot(meas_ghz, meas_p1db, lw=1.2, color="tab:orange", alpha=0.55,
                 label="measured")
    axes[1].plot(ghz, model_p1db, lw=2.0, color="tab:blue", marker="o", ms=5,
                 label="model")
    axes[1].set_xlabel("signal frequency (GHz)")
    axes[1].set_ylabel("$P_{1dB}$ (dBm)")
    axes[1].set_title("same frequency — confounded by band offset")

    order = np.argsort(meas_g0)
    axes[2].plot(meas_g0[order], meas_p1db[order], lw=0.0, marker=".", ms=3,
                 color="tab:orange", alpha=0.35, label="measured columns")
    axes[2].plot(model_g0, model_p1db, lw=0.0, marker="o", ms=7,
                 color="tab:blue", mfc="none", mew=1.8, label="model")
    gain_axis = np.linspace(
        min(model_g0.min(), meas_g0.min()), max(model_g0.max(), meas_g0.max()), 50
    )
    axes[2].plot(gain_axis, depletion_reference_dbm(gain_axis), lw=1.4,
                 color="0.4", ls="-.", label="pure depletion reference")
    axes[2].set_xlabel("small-signal gain (dB)")
    axes[2].set_ylabel("$P_{1dB}$ (dBm)")
    axes[2].set_title("same gain — the comparison that is not confounded")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.suptitle(
        f"2c — model (pump {MODEL_PUMP_GHZ} GHz) vs Themis 105C5 "
        f"(pump {MEAS_PUMP_GHZ} GHz), {len(rows)} model frequencies",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(args.output_dir / "2c_p1db_two_way.png", dpi=140)
    plt.close(fig)

    finite_f = np.isfinite(same_f)
    finite_g = np.isfinite(same_g)
    summary = {
        "n_model_frequencies": len(rows),
        "n_measured_columns": int(meas_ghz.size),
        "gain_match_tolerance_db": GAIN_MATCH_TOL_DB,
        "model_gain_span_db": float(model_g0.max() - model_g0.min()),
        "model_p1db_span_db": float(model_p1db.max() - model_p1db.min()),
        "measured_gain_span_db": float(meas_g0.max() - meas_g0.min()),
        "measured_p1db_span_db": float(meas_p1db.max() - meas_p1db.min()),
        "same_frequency_mean_delta_db": float(
            np.mean(model_p1db[finite_f] - same_f[finite_f])
        ),
        "same_frequency_rms_delta_db": float(
            np.sqrt(np.mean((model_p1db[finite_f] - same_f[finite_f]) ** 2))
        ),
        "same_gain_mean_delta_db": float(
            np.mean(model_p1db[finite_g] - same_g[finite_g])
        ),
        "same_gain_rms_delta_db": float(
            np.sqrt(np.mean((model_p1db[finite_g] - same_g[finite_g]) ** 2))
        ),
        "mean_model_minus_measured_g0_db": float(
            np.mean(model_g0 - meas_g0_at_f)
        ),
    }
    (args.output_dir / "2c_p1db_two_way.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f'{"fs":>6} {"mG0":>7} {"eG0":>7} {"mP1":>8} {"eP1@f":>8} '
          f'{"d@f":>7} {"eP1@G":>8} {"d@G":>7} {"n":>5}')
    for row in rows:
        print(
            f'{row["signal_ghz"]:6.2f} {row["model_g0_db"]:7.2f} '
            f'{row["meas_g0_at_same_ghz_db"]:7.2f} {row["model_p1db_dbm"]:8.2f} '
            f'{row["meas_p1db_at_same_ghz_dbm"]:8.2f} '
            f'{row["delta_same_frequency_db"]:+7.2f} '
            f'{row["meas_p1db_at_same_gain_dbm"]:8.2f} '
            f'{row["delta_same_gain_db"]:+7.2f} '
            f'{int(row["n_gain_matched_columns"]):5d}'
        )
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
