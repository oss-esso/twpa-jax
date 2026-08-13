#!/usr/bin/env python3
"""Identify a wideband gain half-bandwidth using only the stable-regime gate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / "outputs" / "chaos" / "phase2"
REFERENCE = json.loads((PHASE2 / "pump_off_reference_50ohm.json").read_text())
REFERENCE_GAIN = float(REFERENCE["analysis"]["gain_db"])
BANDWIDTHS_MHZ = (25, 50, 100, 200, 300, 500)


def reduce_run(run: Path, stable_filter: bool) -> dict[str, object]:
    with (run / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    spectra = np.load(run / "spectra_map.npz")
    frequencies = spectra["frequency_ghz"] * 1e9
    values = spectra["spectrum_dbm"]
    result = []
    for index, row in enumerate(rows):
        signal_hz = float(row.get("signal_ghz", 6.42)) * 1e9
        pump_hz = 7.0e9
        narrow = float(row.get("gain_db", "nan")) - REFERENCE_GAIN
        entries = {"control": float(row.get("pump_dbm", row.get("signal_ghz"))),
                   "narrowband_gain_vs_off_db": narrow}
        for bandwidth_mhz in BANDWIDTHS_MHZ:
            mask = np.abs(frequencies - signal_hz) <= bandwidth_mhz * 1e6
            for harmonic in range(1, 6):
                mask &= np.abs(frequencies - harmonic * pump_hz) > 2.0 * (frequencies[1] - frequencies[0])
            power_w = np.sum(1e-3 * 10.0 ** (values[index, mask] / 10.0))
            input_power_w = 1.0e-3 * 10.0 ** (-100.0 / 10.0)
            wide_raw = 10.0 * np.log10(max(power_w, 1e-300) / input_power_w)
            entries[f"wide_{bandwidth_mhz}_mhz_db"] = wide_raw - REFERENCE_GAIN
        result.append(entries)
    stable = [row for row in result if row["control"] <= -54.5] if stable_filter else [
        row for row in result if not np.isclose(row["control"], 7.0)
    ]
    gate = {}
    for bandwidth_mhz in BANDWIDTHS_MHZ:
        deltas = [abs(row[f"wide_{bandwidth_mhz}_mhz_db"] - row["narrowband_gain_vs_off_db"])
                  for row in stable]
        gate[str(bandwidth_mhz)] = {
            "max_delta_db_below_transition": float(max(deltas)),
            "pass": bool(max(deltas) <= 0.3),
        }
    return {"source": str(run), "bandwidths_mhz": BANDWIDTHS_MHZ,
            "gate_g6": gate, "rows": result}


def main() -> None:
    outputs = {
        "fig2a": reduce_run(PHASE2 / "fig2a_50ohm_mtls" / "run", True),
        "fig2b_m57": reduce_run(PHASE2 / "fig2b_50ohm_pump_m57_mtls" / "run", False),
        "fig2b_m55": reduce_run(PHASE2 / "fig2b_50ohm_pump_m55_mtls" / "run", False),
    }
    out = PHASE2 / "bandwidth_identification.json"
    out.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
