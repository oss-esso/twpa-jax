"""Scan the published physical degenerate-4WM CME oracle.

The phase mismatch and nonlinear coefficients come from
``published_cme_parameters`` with exp38's validated projection factor. The
emitted gain is the signal power gain, and ``photon_flux_rel_err`` checks the
lossless invariant.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from references.le_gal_2025_gain_compression.cme import (
    envelopes_from_powers,
    integrate_cme,
    photon_flux,
    published_cme_parameters,
)
def main() -> int:
    """Scan and write the validated CME oracle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(
        "references/le_gal_2025_gain_compression/cme_gain_vs_frequency.csv"
    ))
    parser.add_argument("--signal-points", type=int, default=71)
    args = parser.parse_args()

    frequencies = np.linspace(4.0, 11.0, args.signal_points)
    pump_power_w = 10.0 ** ((-78.4 - 30.0) / 10.0)
    signal_power_w = 10.0 ** ((-115.0 - 30.0) / 10.0)
    verdict = json.loads(
        Path(
            "references/le_gal_2025_gain_compression/exp38_kerr_verdict.json"
        ).read_text(encoding="utf-8")
    )
    nonlinear_mismatch = float(verdict["exp37_dk_nl_rad_per_m"])
    rows: list[dict[str, object]] = []
    for signal_ghz in frequencies:
        signal_hz = float(signal_ghz * 1e9)
        params = published_cme_parameters(signal_hz)
        dk_lin = float(params.phase_mismatch)
        dk_total = dk_lin + nonlinear_mismatch
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
