"""Plot simulated vs measured 2c P1dB against small-signal gain.

Gain is the abscissa because simulation and measurement have no common pump
setting: the model does not amplify at the measured pump frequency (1.318 dB at
7.256 GHz / -66.7 dBm). Both the pump-depletion bound and every competing
compression mechanism scale with gain, so equal gain is the meaningful pairing.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

MEASUREMENT = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
INPUT_LOSS_DB = 72.5
MEAS_PUMP_DBM = -66.7
SIM_PUMP_DBM = 10.0 * math.log10(7.231074707853736e-06**2 * 50.0 / 2.0 / 1e-3)
PUMP_GHZ = 7.256
DEPLETION_CONST_DB = 10.0 * math.log10((10.0**0.1 - 1.0) / 2.0)


def measured() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(MEASUREMENT, allow_pickle=True).item()
    freq = data["Frequency"] / 1e9
    pin = data["SignalPower"] - INPUT_LOSS_DB
    response = data["Response"]
    gains, p1dbs = [], []
    for j in range(freq.size):
        if abs(freq[j] - PUMP_GHZ) < 0.15:
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
        gains.append(g0)
        p1dbs.append(
            float(pin[k] + (target - smooth[k]) * (pin[k + 1] - pin[k])
                  / (smooth[k + 1] - smooth[k]))
        )
    return np.array(gains), np.array(p1dbs)


def simulated(path: Path) -> tuple[np.ndarray, np.ndarray]:
    gains, p1dbs = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("p1db_input_dbm"):
                continue
            gains.append(float(row["small_signal_gain_db"]))
            p1dbs.append(float(row["p1db_input_dbm"]))
    return np.array(gains), np.array(p1dbs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-csv",
        type=Path,
        default=Path(
            "outputs/exp23_2c_measured_saturation/matched/p1db_vs_frequency.csv"
        ),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/exp23_2c_measured_saturation/p1db_vs_gain.png"),
    )
    args = parser.parse_args()

    m_gain, m_p1db = measured()
    s_gain, s_p1db = simulated(args.sim_csv)
    window = (m_gain >= s_gain.min()) & (m_gain <= s_gain.max())

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(m_gain[window], m_p1db[window], s=9, alpha=0.35,
               color="#2b6cb0", label=f"measured, {int(window.sum())} columns")
    ax.scatter(s_gain, s_p1db, s=70, color="#c53030", marker="o",
               edgecolor="black", zorder=5,
               label=f"simulated, {s_gain.size} frequencies")

    grid = np.linspace(s_gain.min(), s_gain.max(), 50)
    for gain, p1db, colour, tag in (
        (m_gain[window], m_p1db[window], "#2b6cb0", "measured"),
        (s_gain, s_p1db, "#c53030", "simulated"),
    ):
        slope, intercept = np.polyfit(gain, p1db, 1)
        ax.plot(grid, slope * grid + intercept, colour, lw=2,
                label=f"{tag} fit: {slope:+.2f} dB/dB")

    ax.plot(grid, MEAS_PUMP_DBM - grid + DEPLETION_CONST_DB, "k--", lw=1.5,
            label=f"depletion bound, Pp={MEAS_PUMP_DBM:.1f} dBm (slope -1)")
    ax.plot(grid, SIM_PUMP_DBM - grid + DEPLETION_CONST_DB, "k:", lw=1.5,
            label=f"depletion bound, Pp={SIM_PUMP_DBM:.1f} dBm (slope -1)")

    ax.set_xlabel("small-signal gain $G_0$ (dB)")
    ax.set_ylabel("input $P_{1dB}$ (dBm)")
    ax.set_title("2c gain compression: model vs measurement, paired at equal gain")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower left", fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
