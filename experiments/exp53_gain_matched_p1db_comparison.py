"""Compare 2c model P1dB against measurement at matched gain, calibration-correct.

Corrected version of exp45_p1db_two_way_comparison.py -- that script never
got updated for either of the two 2026-08-05 calibration fixes
([[loss-b1-fabricated-72p5-fixed]], [[pump-power-norton-6db]]), so its
"+0.24 dB match" headline was retracted (see [[2c-model-compresses-early-confirmed]],
memory [[live-2c-model-compresses-late]]). This script applies both:

  - measured side: signal_line_loss_model() (loss_B1) at each column's own
    frequency, not the fabricated flat 72.5 dB constant.
  - model side: reads each summary's own recorded ``power_convention`` and
    corrects to Norton if it was written under ``legacy_traveling_wave``,
    instead of assuming one convention for the whole run.

Same methodology otherwise: for each model frequency, find measured columns
whose small-signal gain matches within +-0.25 dB (any frequency), and compare
P1dB against the median of those. This is the only P1dB comparison validated
as not confounded by this device's gain-vs-frequency ripple
(same-frequency comparison mixes in the ripple offset).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.loss import signal_line_loss_model  # noqa: E402

CUBE = ROOT / (
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
MEAS_PUMP_GHZ = 7.256
PUMP_EXCLUSION_GHZ = 0.15
GAIN_MATCH_TOL_DB = 0.25
MIN_MEASURED_G0_DB = 3.0
NORTON_OFFSET_DB = 10.0 * np.log10(4.0)  # 6.0206 dB, norton -> legacy_traveling_wave


def measured_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column measured (frequency GHz, G0 dB, P1dB dBm), on-chip via loss_B1."""
    data = np.load(CUBE, allow_pickle=True).item()
    frequency = np.asarray(data["Frequency"], dtype=float) / 1e9
    instrument_power = np.asarray(data["SignalPower"], dtype=float)  # (121,)
    loss_model = signal_line_loss_model()
    atten = np.array([loss_model.attenuation_db(f) for f in frequency])  # (2001,)
    response = np.asarray(data["Response"], dtype=float)
    g0 = np.median(response[:10, :], axis=0)
    p1db = np.full(frequency.size, np.nan)
    for index in range(frequency.size):
        power = instrument_power - atten[index]
        smooth = savgol_filter(response[:, index], 11, 2)
        threshold = g0[index] - 1.0
        below = np.flatnonzero((smooth[:-1] >= threshold) & (smooth[1:] < threshold))
        if below.size:
            first = int(below[0])
            p1db[index] = np.interp(
                threshold, [smooth[first + 1], smooth[first]],
                [power[first + 1], power[first]],
            )
    usable = (
        (np.abs(frequency - MEAS_PUMP_GHZ) > PUMP_EXCLUSION_GHZ)
        & (g0 > MIN_MEASURED_G0_DB) & np.isfinite(p1db)
    )
    return frequency[usable], g0[usable], p1db[usable]


def model_table(run_dir: Path) -> list[dict[str, float]]:
    """Per-frequency model (frequency GHz, G0 dB, P1dB dBm), corrected to Norton."""
    rows: list[dict[str, float]] = []
    for directory in sorted(run_dir.glob("frequency_*")):
        if not directory.is_dir():
            continue
        summary_path = directory / "compression_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        p1db = summary.get("p1db")
        gain = summary.get("small_signal_gain_vs_off_db")
        if p1db is None or gain is None:
            continue
        match = re.search(r"_(\d+\.\d+)ghz", directory.name)
        if match is None:
            continue
        convention = summary.get("power_convention", "legacy_traveling_wave")
        # "p1db"/"p1db_input_dbm" is EXTERNAL/instrument-referred: run_compression's
        # _current_to_dbm() adds the run's own attenuation_db on top of the true
        # on-chip power (see run_compression.py:444-448). measured_table() below
        # produces ON-CHIP power (instrument - loss_B1). Comparing the two directly
        # without undoing that offset is a convention mismatch that inflated an
        # earlier version of this script's gap to +41 dB (see
        # [[2c-model-compresses-early-confirmed]] for the corrected number) --
        # subtract the run's own recorded attenuation_db to get back on-chip first.
        on_chip_dbm = float(p1db) - float(summary.get("attenuation_db", 0.0))
        # Target convention is legacy_traveling_wave (design intent confirmed
        # 2026-08-06, CLAUDE.md "Port power convention"). A run recorded under
        # norton reads 6.0206 dB low against that target and needs it added back.
        p1db_reference = on_chip_dbm + (
            NORTON_OFFSET_DB if convention == "norton" else 0.0
        )
        rows.append({
            "signal_ghz": float(match.group(1)),
            "model_g0_db": float(gain),
            "model_p1db_dbm": p1db_reference,
            "power_convention": convention,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meas_ghz, meas_g0, meas_p1db = measured_table()
    rows = model_table(args.run_dir)
    if not rows:
        raise SystemExit(f"no model frequencies found under {args.run_dir}")

    for row in rows:
        row["meas_g0_at_same_ghz_db"] = float(np.interp(row["signal_ghz"], meas_ghz, meas_g0))
        row["meas_p1db_at_same_ghz_dbm"] = float(np.interp(row["signal_ghz"], meas_ghz, meas_p1db))
        matched = np.abs(meas_g0 - row["model_g0_db"]) <= GAIN_MATCH_TOL_DB
        row["n_gain_matched_columns"] = int(np.count_nonzero(matched))
        row["meas_p1db_at_same_gain_dbm"] = (
            float(np.median(meas_p1db[matched])) if matched.any() else float("nan")
        )
        row["delta_same_frequency_db"] = row["model_p1db_dbm"] - row["meas_p1db_at_same_ghz_dbm"]
        row["delta_same_gain_db"] = row["model_p1db_dbm"] - row["meas_p1db_at_same_gain_dbm"]

    with (args.output_dir / "gain_matched_p1db.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    same_gain = np.array([r["delta_same_gain_db"] for r in rows])
    finite = np.isfinite(same_gain)
    n_matched = np.array([r["n_gain_matched_columns"] for r in rows])
    well_sampled = finite & (n_matched >= 5)

    summary = {
        "run_dir": str(args.run_dir),
        "n_model_frequencies": len(rows),
        "n_measured_columns": int(meas_ghz.size),
        "gain_match_tolerance_db": GAIN_MATCH_TOL_DB,
        "same_gain_mean_delta_db": float(np.mean(same_gain[finite])),
        "same_gain_median_delta_db": float(np.median(same_gain[finite])),
        "same_gain_rms_delta_db": float(np.sqrt(np.mean(same_gain[finite] ** 2))),
        "same_gain_mean_delta_db_well_sampled_ge5cols": (
            float(np.mean(same_gain[well_sampled])) if well_sampled.any() else None
        ),
        "n_well_sampled": int(np.count_nonzero(well_sampled)),
    }
    (args.output_dir / "gain_matched_p1db_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    ghz = np.array([r["signal_ghz"] for r in rows])
    model_g0 = np.array([r["model_g0_db"] for r in rows])
    model_p1db = np.array([r["model_p1db_dbm"] for r in rows])

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    order = np.argsort(meas_g0)
    axes[0].plot(meas_g0[order], meas_p1db[order], lw=0.0, marker=".", ms=3,
                 color="tab:orange", alpha=0.35, label="measured columns")
    axes[0].plot(model_g0, model_p1db, lw=0.0, marker="o", ms=7,
                 color="tab:blue", mfc="none", mew=1.8, label="model")
    axes[0].set_xlabel("small-signal gain (dB)")
    axes[0].set_ylabel("$P_{1dB}$ (dBm, Norton on-chip)")
    axes[0].set_title("same gain -- calibration-corrected")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.0, color="0.3", lw=1.0)
    axes[1].plot(ghz, same_gain, "o-", color="tab:red")
    axes[1].set_xlabel("model signal frequency (GHz)")
    axes[1].set_ylabel("$\\Delta P_{1dB}$ = model - measured (dB)")
    axes[1].set_title(
        f"mean={summary['same_gain_mean_delta_db']:+.2f} dB  "
        f"median={summary['same_gain_median_delta_db']:+.2f} dB"
    )
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "gain_matched_p1db.png", dpi=140)
    plt.close(figure)

    print(f'{"fs":>7} {"mG0":>7} {"eP1@G":>8} {"mP1":>8} {"d@G":>7} {"n":>4} {"conv":>20}')
    for row in rows:
        print(
            f'{row["signal_ghz"]:7.3f} {row["model_g0_db"]:7.3f} '
            f'{row["meas_p1db_at_same_gain_dbm"]:8.2f} {row["model_p1db_dbm"]:8.2f} '
            f'{row["delta_same_gain_db"]:+7.2f} {row["n_gain_matched_columns"]:4d} '
            f'{row["power_convention"]:>20}'
        )
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
