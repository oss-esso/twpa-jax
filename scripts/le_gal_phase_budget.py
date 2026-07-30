"""Emit the signed discrete-ladder phase budget.

The nonlinear term uses the measured CME magnitude 60.4 rad/m and the
positive published ``g3/g1`` sign.  Thus ``dk_nl`` is positive.  The linear
term is ``2*k_p-k_s-k_i`` from the exact ladder dispersion and retains its
signed value.
"""

import csv
from pathlib import Path

import numpy as np

from twpa_solver.builders.le_gal_2025 import ladder_dispersion


def main() -> None:
    output = Path("references/le_gal_2025_gain_compression/phase_budget.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    length = 700 * 8.7e-6
    frequencies = np.arange(4.0, 11.0 + 0.05, 0.1)
    pump = 7.5e9
    def k(frequency: np.ndarray) -> np.ndarray:
        return ladder_dispersion(
            2 * np.pi * frequency,
            inductance_h=866.372e-12,
            snail_capacitance_f=31e-15,
            ground_capacitance_f=223.5e-15,
            cell_length_m=8.7e-6,
        )
    kp = float(k(np.array([pump]))[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["f_s_GHz", "f_i_GHz", "k_p", "k_s", "k_i", "dk_lin", "dk_nl", "dk_total", "regime"])
        for signal in frequencies:
            idler = 15.0 - signal
            ks, ki = k(np.array([signal, idler]) * 1e9)
            dk_lin = 2 * kp - ks - ki
            dk_nl = 60.4
            dk_total = dk_lin + dk_nl
            regime = "PHASE_MATCHED" if abs(dk_total) * length < 1.0 else "MISMATCHED"
            writer.writerow([signal, idler, kp, ks, ki, dk_lin, dk_nl, dk_total, regime])


if __name__ == "__main__":
    main()
