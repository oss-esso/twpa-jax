"""Build the presentation figure set and number table for 2c and jtwpa.

Two devices, two different kinds of evidence:

* **2c** (`designs/ipm_2c_fixed`) is a fabricated chip with a measured Themis
  dataset, so it carries a model-versus-measurement comparison.  Its figures
  already exist from exp30/exp32/exp34 and are only referenced here.
* **jtwpa** is a JosephsonCircuits.jl documentation design with no measurement,
  so it carries internal physics instead: a compression curve read against pump
  depletion, and P1dB across frequency at nearly constant gain.

The pure-depletion reference ``P1dB = Pp + 10 log10[(10^0.1 - 1) / (2 G_lin)]``
is drawn throughout as a reference line, never as an acceptance bound: it is one
saturation mechanism among several, so a device is expected to compress at or
before it.  Where gain is flat and P1dB is not, depletion alone cannot be the
whole story -- that is the point the jtwpa panels are there to make.
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

JTWPA_RUN = Path("outputs/exp20_multitone_compression_converged/jtwpa/s10")
JTWPA_FREQ_DIR = Path("outputs/exp21_p1db_vs_frequency_converged/jtwpa")
Z0_OHM = 50.0


def pump_power_dbm(run_dir: Path) -> float:
    """On-chip pump power from the run's own drive current."""
    summary = json.loads(
        (run_dir / "compression_summary.json").read_text(encoding="utf-8")
    )
    current = float(summary["pump_current_a"])
    return 10.0 * math.log10(current**2 * Z0_OHM / 2.0 / 1e-3)


def read_points(path: Path) -> list[dict[str, str]]:
    """Rows of a compression_points.csv as plain string dicts."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def depletion_only_p1db_dbm(gain_db: float, pump_dbm: float) -> float:
    """Input P1dB if pump depletion were the only saturation mechanism."""
    return pump_dbm + 10.0 * math.log10(
        (10.0**0.1 - 1.0) / (2.0 * 10.0 ** (gain_db / 10.0))
    )


def crossing_dbm(power: np.ndarray, compression: np.ndarray) -> float:
    """Interpolated first upward crossing of 1 dB compression."""
    above = np.flatnonzero(compression >= 1.0)
    if above.size == 0 or above[0] == 0:
        return float("nan")
    index = int(above[0])
    return float(
        np.interp(
            1.0,
            [compression[index - 1], compression[index]],
            [power[index - 1], power[index]],
        )
    )


def compression_figure(output: Path) -> dict[str, float]:
    """Gain and pump transmission versus signal power for jtwpa."""
    rows = [r for r in read_points(JTWPA_RUN / "compression_points.csv")
            if r["status"] == "VALID_SOLVED"]
    rows.sort(key=lambda r: float(r["signal_power_dbm"]))
    power = np.array([float(r["signal_power_dbm"]) for r in rows])
    gain = np.array([float(r["gain_vs_off_db"]) for r in rows])
    depletion = np.array([float(r["pump_depletion_db"]) for r in rows])
    small_signal = float(gain[0])
    p1db = crossing_dbm(power, small_signal - gain)
    depletion_at_p1db = float(np.interp(p1db, power, depletion))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(power, gain, lw=2.2, color="tab:blue", marker="o", ms=4,
            label=f"gain  ($G_0$ = {small_signal:.2f} dB)")
    ax.axhline(small_signal - 1.0, color="tab:blue", ls=":", lw=1.0)
    ax.axvline(p1db, color="tab:blue", ls="--", lw=1.4,
               label=f"$P_{{1dB}}$ = {p1db:.2f} dBm")
    ax.set_xlabel("signal power at device input (dBm)")
    ax.set_ylabel("gain (dB)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.3)
    # The sweep spans ~110 dB but compression lives in its last 30; showing all
    # of it compresses the interesting region into a few pixels.
    ax.set_xlim(p1db - 22.0, float(power.max()) + 2.0)

    twin = ax.twinx()
    twin.plot(power, depletion, lw=1.8, color="tab:red", marker="s", ms=3.5,
              label="pump transmission")
    twin.plot([p1db], [depletion_at_p1db], ls="none", marker="*", ms=15,
              color="tab:red",
              label=f"at $P_{{1dB}}$: {depletion_at_p1db:.3f} dB")
    twin.set_ylabel("pump transmission (dB)", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")

    handles, labels = ax.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    ax.legend(handles + twin_handles, labels + twin_labels,
              fontsize=9, loc="lower left")
    ax.set_title("JTWPA (JC design) — gain compression and pump depletion, S=10")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return {
        "small_signal_gain_db": small_signal,
        "p1db_dbm": p1db,
        "pump_depletion_at_p1db_db": depletion_at_p1db,
        "max_pump_depletion_db": float(depletion.min()),
    }


def frequency_figure(output: Path) -> dict[str, object]:
    """P1dB and gain versus frequency, against the depletion reference."""
    records: list[dict[str, float]] = []
    for directory in sorted(JTWPA_FREQ_DIR.glob("frequency_*")):
        summary_path = directory / "compression_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        p1db = summary.get("p1db")
        gain = summary.get("small_signal_gain_vs_off_db")
        if p1db is None or gain is None:
            continue
        records.append({
            "signal_ghz": float(directory.name.split("_")[-1].rstrip("ghz")),
            "gain_db": float(gain),
            "p1db_dbm": float(p1db),
            "depletion_reference_dbm": depletion_only_p1db_dbm(
                float(gain), pump_power_dbm(directory)
            ),
        })
    frequency = np.array([r["signal_ghz"] for r in records])
    gain = np.array([r["gain_db"] for r in records])
    p1db = np.array([r["p1db_dbm"] for r in records])

    # Under pure pump depletion P1dB is a function of gain alone, so the P1dB
    # spread can never exceed the gain spread. Comparing the two is the whole
    # test, and it needs no reference data.
    gain_span = float(gain.max() - gain.min())
    p1db_span = float(p1db.max() - p1db.min())

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    axes[0].plot(frequency, gain, lw=2.2, color="tab:red", marker="s", ms=4.5)
    axes[0].set_ylabel("small-signal gain (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(
        "JTWPA — gain is flat, $P_{1dB}$ is not: "
        f"{gain_span:.2f} dB of gain spans {p1db_span:.2f} dB of $P_{{1dB}}$"
    )
    axes[1].plot(frequency, p1db, lw=2.2, color="tab:blue", marker="o", ms=4.5,
                 label="$P_{1dB}$ (multitone HB)")
    axes[1].set_xlabel("signal frequency (GHz)")
    axes[1].set_ylabel("$P_{1dB}$ (dBm)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return {
        "n_frequencies": len(records),
        "gain_span_db": gain_span,
        "p1db_span_db": p1db_span,
        "depletion_allows_db": gain_span,
        "excess_over_depletion": (
            p1db_span / gain_span if gain_span > 0 else float("nan")
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/presentation")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compression = compression_figure(
        args.output_dir / "jtwpa_compression.png"
    )
    frequency = frequency_figure(
        args.output_dir / "jtwpa_p1db_vs_frequency.png"
    )
    summary = {"jtwpa_compression": compression, "jtwpa_frequency": frequency}
    (args.output_dir / "jtwpa_numbers.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "jtwpa_frequency"}
        | {"jtwpa_frequency": {
            k: v for k, v in frequency.items() if k != "records"
        }},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
