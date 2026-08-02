"""Compare simulated 2c P1dB against the measured Themis saturation cube.

The model has no amplifying operating point at the measured pump frequency
(1.318 dB at 7.256 GHz / -66.7 dBm, measured by ``exp23_2c_measured_saturation``
``literal`` arm), so simulation and measurement cannot be compared at a common
pump setting. What *is* comparable is P1dB **at equal small-signal gain**: the
depletion bound and every competing compression mechanism scale with gain, so
gain is the meaningful abscissa.

Reads the ``matched`` arm's ``p1db_vs_frequency.csv`` and the measurement cube,
and reports simulated P1dB against the measured P1dB at the frequency whose
measured gain matches.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

MEASUREMENT = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
INPUT_LOSS_DB = 72.5
PUMP_DEVICE_DBM = -66.7
PUMP_GHZ = 7.256
PUMP_GUARD_GHZ = 0.15
DEPLETION_CONST_DB = 10.0 * math.log10((10.0**0.1 - 1.0) / 2.0)


def measured_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freq_ghz, G0_db, p1db_in_dbm) for every usable column."""
    data = np.load(MEASUREMENT, allow_pickle=True).item()
    freq = data["Frequency"] / 1e9
    pin = data["SignalPower"] - INPUT_LOSS_DB
    response = data["Response"]
    freqs, gains, p1dbs = [], [], []
    for j in range(freq.size):
        if abs(freq[j] - PUMP_GHZ) < PUMP_GUARD_GHZ:
            continue
        smooth = savgol_filter(response[:, j], 11, 2)
        g0 = float(np.median(smooth[pin < -120.0]))
        target = g0 - 1.0
        above = np.nonzero(smooth >= target)[0]
        if above.size == 0 or above[-1] == smooth.size - 1:
            continue
        k = int(above[-1])
        if smooth[k + 1] == smooth[k]:
            continue
        p1 = pin[k] + (target - smooth[k]) * (pin[k + 1] - pin[k]) / (
            smooth[k + 1] - smooth[k]
        )
        freqs.append(float(freq[j]))
        gains.append(g0)
        p1dbs.append(float(p1))
    return np.array(freqs), np.array(gains), np.array(p1dbs)


def simulated_rows(csv_path: Path) -> list[tuple[float, float, float]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("p1db_input_dbm"):
                continue
            rows.append(
                (
                    float(row["signal_ghz"]),
                    float(row["small_signal_gain_db"]),
                    float(row["p1db_input_dbm"]),
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-csv",
        type=Path,
        default=Path(
            "outputs/exp23_2c_measured_saturation/matched/p1db_vs_frequency.csv"
        ),
    )
    parser.add_argument("--gain-tol-db", type=float, default=0.5)
    args = parser.parse_args()

    m_freq, m_gain, m_p1db = measured_curves()
    print(f"measured: {m_freq.size} usable columns, "
          f"{m_freq.min():.3f}-{m_freq.max():.3f} GHz, "
          f"G0 {m_gain.min():.2f}-{m_gain.max():.2f} dB, "
          f"P1dB {m_p1db.min():.2f}..{m_p1db.max():.2f} dBm")
    print(f"measured pump {PUMP_GHZ} GHz at {PUMP_DEVICE_DBM} dBm\n")

    rows = simulated_rows(args.sim_csv)
    if not rows:
        print(f"no simulated P1dB rows in {args.sim_csv}")
        return 1

    print(f"{'fs sim':>7s} {'G sim':>7s} {'P1 sim':>9s} | "
          f"{'n match':>7s} {'G meas':>7s} {'P1 meas':>9s} {'sim-meas':>9s} | "
          f"{'rule':>9s} {'sim-rule':>9s}")
    deltas = []
    for fs, gain, p1 in rows:
        sel = np.abs(m_gain - gain) <= args.gain_tol_db
        rule = PUMP_DEVICE_DBM - gain + DEPLETION_CONST_DB
        if not sel.any():
            print(f"{fs:7.3f} {gain:7.3f} {p1:9.3f} | "
                  f"{'0':>7s} {'-':>7s} {'-':>9s} {'-':>9s} | "
                  f"{rule:9.3f} {p1 - rule:+9.3f}")
            continue
        g_ref = float(np.mean(m_gain[sel]))
        p_ref = float(np.median(m_p1db[sel]))
        deltas.append(p1 - p_ref)
        print(f"{fs:7.3f} {gain:7.3f} {p1:9.3f} | "
              f"{int(sel.sum()):7d} {g_ref:7.3f} {p_ref:9.3f} {p1 - p_ref:+9.3f} | "
              f"{rule:9.3f} {p1 - rule:+9.3f}")

    if deltas:
        arr = np.array(deltas)
        print(f"\nsim - measured at equal gain: n={arr.size}  "
              f"mean {arr.mean():+.3f} dB  sd {arr.std():.3f} dB  "
              f"range {arr.min():+.3f} .. {arr.max():+.3f}")
        print("Positive means the model compresses LATER than the hardware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
