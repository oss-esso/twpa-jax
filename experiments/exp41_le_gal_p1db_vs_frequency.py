"""Extract P1dB versus frequency and test the paper's central claim.

Le Gal 2025 reports that the lowest 1 dB compression power does **not** occur at
the frequencies of maximum linear gain, and that a pure pump-depletion model
misses part of that structure.  That is a shape statement: it needs no digitized
reference curve and no absolute power calibration, so it can be tested even
though this benchmark's absolute gain still sits below the published value.

Reads the frequency x signal-power sweep from ``scripts/run_le_gal_2025_hb.py``,
interpolates the 1 dB crossing inside its bracket rather than taking the first
grid point past it, and reports where the P1dB minimum sits relative to the gain
maximum.  The pure-depletion prediction is drawn for reference only; it is one
mechanism among several, so the measured crossing is expected at or below it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PUMP_POWER_DBM = -78.4
PUMP_GHZ = 7.5


def depletion_only_p1db_dbm(gain_db: float, pump_dbm: float) -> float:
    """Input P1dB if pump depletion were the only saturation mechanism."""
    gain_linear = 10.0 ** (gain_db / 10.0)
    return pump_dbm + 10.0 * math.log10(
        (10.0**0.1 - 1.0) / (2.0 * gain_linear)
    )


def crossing_dbm(
    power_dbm: np.ndarray, compression_db: np.ndarray, target_db: float = 1.0
) -> float:
    """Interpolated first upward crossing of ``target_db`` of compression."""
    above = np.flatnonzero(compression_db >= target_db)
    if above.size == 0 or above[0] == 0:
        return float("nan")
    index = int(above[0])
    return float(
        np.interp(
            target_db,
            [compression_db[index - 1], compression_db[index]],
            [power_dbm[index - 1], power_dbm[index]],
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("outputs/exp41_le_gal_p1db_vs_freq/fig3a.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    by_frequency: dict[float, list[dict[str, object]]] = {}
    for row in json.loads(args.input.read_text(encoding="utf-8")):
        if row.get("status") == "SOLVED":
            by_frequency.setdefault(float(row["signal_GHz"]), []).append(row)

    records: list[dict[str, float]] = []
    for frequency in sorted(by_frequency):
        points = sorted(
            by_frequency[frequency], key=lambda row: float(row["signal_dBm"])
        )
        power = np.array([float(row["signal_dBm"]) for row in points])
        gain = np.array([float(row["gain_vs_off_db"]) for row in points])
        depletion = np.array([float(row["pump_depletion_db"]) for row in points])
        small_signal = float(gain[0])
        p1db = crossing_dbm(power, small_signal - gain)
        records.append({
            "signal_ghz": frequency,
            "small_signal_gain_db": small_signal,
            "p1db_dbm": p1db,
            "depletion_only_p1db_dbm": depletion_only_p1db_dbm(
                small_signal, PUMP_POWER_DBM
            ),
            "pump_depletion_at_p1db_db": (
                float("nan") if math.isnan(p1db)
                else float(np.interp(p1db, power, depletion))
            ),
            "max_compression_db": float(small_signal - gain[-1]),
        })

    with (output_dir / "p1db_vs_frequency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    frequency = np.array([r["signal_ghz"] for r in records])
    gain = np.array([r["small_signal_gain_db"] for r in records])
    p1db = np.array([r["p1db_dbm"] for r in records])
    reference = np.array([r["depletion_only_p1db_dbm"] for r in records])
    valid = np.isfinite(p1db)

    gain_peak_ghz = float(frequency[int(np.argmax(gain))])
    p1db_min_ghz = float(frequency[valid][int(np.argmin(p1db[valid]))])
    # If saturation were pure pump depletion, P1dB would be a strictly
    # decreasing function of gain, so its minimum would sit on the gain peak.
    offset_ghz = p1db_min_ghz - gain_peak_ghz
    correlation = float(np.corrcoef(gain[valid], p1db[valid])[0, 1])

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)
    axes[0].plot(frequency, gain, lw=2.0, color="tab:blue", marker="o", ms=4)
    axes[0].axvline(gain_peak_ghz, color="tab:blue", ls="--", lw=1.3,
                    label=f"gain maximum {gain_peak_ghz:.2f} GHz")
    axes[0].axvline(p1db_min_ghz, color="tab:red", ls="--", lw=1.3,
                    label=f"$P_{{1dB}}$ minimum {p1db_min_ghz:.2f} GHz")
    axes[0].set_ylabel("small-signal gain (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_title(
        f"Le Gal 2025 benchmark, 700 cells — pump {PUMP_GHZ} GHz "
        f"@ {PUMP_POWER_DBM} dBm"
    )

    axes[1].plot(frequency[valid], p1db[valid], lw=2.0, color="tab:red",
                 marker="o", ms=4, label="HB $P_{1dB}$")
    axes[1].plot(frequency, reference, lw=1.5, color="0.45", ls="-.",
                 label="pure pump-depletion reference")
    axes[1].axvline(gain_peak_ghz, color="tab:blue", ls="--", lw=1.3)
    axes[1].axvline(p1db_min_ghz, color="tab:red", ls="--", lw=1.3)
    axes[1].set_xlabel("signal frequency (GHz)")
    axes[1].set_ylabel("$P_{1dB}$ (dBm)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "p1db_vs_frequency.png", dpi=140)
    plt.close(fig)

    summary = {
        "gain_maximum_ghz": gain_peak_ghz,
        "gain_maximum_db": float(np.max(gain)),
        "p1db_minimum_ghz": p1db_min_ghz,
        "p1db_minimum_dbm": float(np.min(p1db[valid])),
        "offset_ghz": offset_ghz,
        "p1db_at_gain_maximum_dbm": float(
            p1db[int(np.argmax(gain))]
        ),
        "gain_p1db_correlation": correlation,
        "frequencies_without_p1db": [
            r["signal_ghz"] for r in records if math.isnan(r["p1db_dbm"])
        ],
        "mean_p1db_minus_depletion_reference_db": float(
            np.mean(p1db[valid] - reference[valid])
        ),
        "paper_claim_p1db_minimum_offset_from_gain_maximum": bool(
            abs(offset_ghz) > 1e-9
        ),
    }
    (output_dir / "p1db_vs_frequency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f'{"fs GHz":>7} {"G0 dB":>8} {"P1dB dBm":>10} {"depl ref":>10} '
          f'{"delta":>8} {"depl@P1dB":>10}')
    for record in records:
        print(
            f'{record["signal_ghz"]:>7.2f} {record["small_signal_gain_db"]:>8.3f} '
            f'{record["p1db_dbm"]:>10.3f} {record["depletion_only_p1db_dbm"]:>10.3f} '
            f'{record["p1db_dbm"] - record["depletion_only_p1db_dbm"]:>8.3f} '
            f'{record["pump_depletion_at_p1db_db"]:>10.4f}'
        )
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
