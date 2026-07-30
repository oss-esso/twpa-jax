"""Scan the published physical degenerate-4WM CME oracle.

The phase mismatch is recomputed for every signal frequency using the exact
discrete ladder dispersion.  The nonlinear contribution is +591.34 rad/m,
from ``(g3/g1) * k_p * |A_p|^2 / 8`` with the calibrated pump amplitude and
the positive published ``g3/g1`` sign.  The emitted gain is the signal power
gain, and ``photon_flux_rel_err`` is checked against the lossless invariant.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from references.le_gal_2025_gain_compression.cme import (
    envelopes_from_powers,
    integrate_cme,
    photon_flux,
    published_cme_parameters,
)
from twpa_solver.builders.le_gal_2025 import ladder_dispersion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "references/le_gal_2025_gain_compression/cme_gain_vs_frequency.csv"
    ))
    parser.add_argument("--signal-points", type=int, default=71)
    args = parser.parse_args()

    frequencies = np.linspace(4.0, 11.0, args.signal_points)
    pump_hz = 7.5e9
    pump_power_w = 10.0 ** ((-78.4 - 30.0) / 10.0)
    signal_power_w = 10.0 ** ((-115.0 - 30.0) / 10.0)
    nonlinear_mismatch = 591.34
    rows: list[dict[str, object]] = []
    for signal_ghz in frequencies:
        signal_hz = float(signal_ghz * 1e9)
        params = published_cme_parameters(signal_hz)
        k = lambda hz: ladder_dispersion(
            2.0 * np.pi * np.asarray(hz),
            inductance_h=866.372e-12,
            snail_capacitance_f=31e-15,
            ground_capacitance_f=223.5e-15,
            cell_length_m=8.7e-6,
        )
        kp = float(k(np.array([pump_hz]))[0])
        ks, ki = k(np.array([signal_hz, 15.0e9 - signal_hz]))
        dk_lin = 2.0 * kp - float(ks) - float(ki)
        dk_total = dk_lin + nonlinear_mismatch
        # ``dk_total`` already carries the calibrated pump Kerr phase shift;
        # leave the envelope SPM/XPM slots off here to avoid counting it twice.
        params = replace(
            params, phase_mismatch=dk_total,
            self_phase_p=0.0, self_phase_s=0.0, self_phase_i=0.0,
            cross_phase_ps=0.0, cross_phase_pi=0.0, cross_phase_si=0.0,
        )
        initial = envelopes_from_powers(pump_power_w, signal_power_w, params)
        try:
            _, envelopes = integrate_cme(initial, params, points=401)
            flux = photon_flux(envelopes)
            flux_error = float(np.max(np.abs(flux - flux[0])) / flux[0])
            if flux_error >= 1e-7:
                raise AssertionError(f"photon flux relative error {flux_error}")
            gain_db = float(10.0 * np.log10(abs(envelopes[1, -1] / initial[1]) ** 2))
            status = "CONSERVATIVE"
        except (AssertionError, RuntimeError, FloatingPointError, ValueError) as error:
            gain_db = ""
            flux_error = ""
            status = f"FAILED: {error}"
        rows.append({
            "f_s_GHz": float(signal_ghz), "gain_dB": gain_db,
            "photon_flux_rel_err": flux_error, "status": status,
            "dk_lin_rad_per_m": dk_lin, "dk_nl_rad_per_m": nonlinear_mismatch,
            "dk_total_rad_per_m": dk_total,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
