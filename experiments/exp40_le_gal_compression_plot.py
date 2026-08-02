"""Plot the Le Gal compression curve and its pump depletion.

Renders the signal-power sweep produced by ``scripts/run_le_gal_2025_hb.py``
in the layout of the paper's Fig. 1(c): gain against signal power, with the
1 dB compression point marked, and the pump transmission on a twin axis so the
two can be read together.

Pump depletion is what makes the compression physical rather than an amplitude
limiter, so it is plotted alongside rather than in a separate figure.  The
pure-depletion prediction ``P1dB = Pp + 10 log10[(10^0.1 - 1) / (2 G_lin)]`` is
drawn as a reference line, not as an acceptance bound: it is one mechanism
among several, so the true crossing is expected at or below it.
"""

from __future__ import annotations

import argparse
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
        default=Path("outputs/exp40_le_gal_compression/fig1c_fs6.0.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_name("fig1c_compression.png")

    rows = [
        row for row in json.loads(args.input.read_text(encoding="utf-8"))
        if row.get("status") == "SOLVED"
    ]
    if not rows:
        raise SystemExit(f"no SOLVED points in {args.input}")
    rows.sort(key=lambda row: float(row["signal_dBm"]))

    power = np.array([float(row["signal_dBm"]) for row in rows])
    gain = np.array([float(row["gain_vs_off_db"]) for row in rows])
    compression = np.array([float(row["compression_db"]) for row in rows])
    depletion = np.array([float(row["pump_depletion_db"]) for row in rows])
    signal_ghz = float(rows[0]["signal_GHz"])
    cells = int(rows[0]["cells"])

    small_signal = float(gain[0])
    p1db = crossing_dbm(power, compression)
    depletion_at_p1db = float(np.interp(p1db, power, depletion))
    depletion_reference = depletion_only_p1db_dbm(small_signal, PUMP_POWER_DBM)
    converted_percent = 100.0 * (1.0 - 10.0 ** (depletion_at_p1db / 10.0))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(power, gain, lw=2.0, color="tab:blue", marker="o", ms=4,
            label=f"HB gain  ($G_0$={small_signal:.2f} dB)")
    ax.axhline(small_signal - 1.0, color="tab:blue", ls=":", lw=1.0)
    ax.axvline(p1db, color="tab:blue", ls="--", lw=1.4,
               label=f"$P_{{1dB}}$ = {p1db:.2f} dBm")
    ax.axvline(depletion_reference, color="0.45", ls="-.", lw=1.2,
               label=f"depletion-only reference = {depletion_reference:.2f} dBm")
    ax.set_xlabel("signal power at device input (dBm)")
    ax.set_ylabel("gain (dB)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    twin.plot(power, depletion, lw=1.8, color="tab:red", marker="s", ms=3.5,
              label="pump transmission")
    twin.plot([p1db], [depletion_at_p1db], marker="*", ms=15,
              color="tab:red", ls="none",
              label=(
                  f"at $P_{{1dB}}$: {depletion_at_p1db:.3f} dB "
                  f"({converted_percent:.1f}% converted)"
              ))
    twin.set_ylabel("pump transmission (dB)", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")

    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels,
              fontsize=8.5, loc="lower left")
    ax.set_title(
        f"Le Gal 2025 benchmark, {cells} cells — compression at "
        f"$f_s$={signal_ghz} GHz, pump {PUMP_GHZ} GHz @ {PUMP_POWER_DBM} dBm"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)

    print(json.dumps({
        "input": str(args.input),
        "output": str(output),
        "signal_ghz": signal_ghz,
        "cells": cells,
        "small_signal_gain_db": small_signal,
        "p1db_dbm": p1db,
        "depletion_only_reference_dbm": depletion_reference,
        "p1db_minus_reference_db": p1db - depletion_reference,
        "pump_depletion_at_p1db_db": depletion_at_p1db,
        "pump_converted_percent_at_p1db": converted_percent,
        "max_pump_depletion_db": float(depletion.min()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
