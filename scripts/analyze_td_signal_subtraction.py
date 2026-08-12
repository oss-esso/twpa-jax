"""Subtract a same-state pump-only TD control before signal demodulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def signal_design(theta: np.ndarray, pump_ghz: float, signal_ghz: float) -> np.ndarray:
    frequencies = [signal_ghz, pump_ghz, 2.0 * pump_ghz, 3.0 * pump_ghz]
    columns = [np.ones_like(theta)]
    for frequency in frequencies:
        phase = frequency / pump_ghz * theta
        columns.extend((np.cos(phase), np.sin(phase)))
    return np.column_stack(columns)


def fit_signal(theta: np.ndarray, voltage: np.ndarray, pump_ghz: float, signal_ghz: float) -> complex:
    coeff = np.linalg.lstsq(signal_design(theta, pump_ghz, signal_ghz), voltage, rcond=None)[0]
    return complex(coeff[1], -coeff[2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--on", type=Path, required=True)
    p.add_argument("--off", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--signal-current-a", type=float, required=True)
    p.add_argument("--hb-reference-db", type=float, required=True)
    p.add_argument("--hb-reference-source", type=str, required=True)
    p.add_argument("--pump-ghz", type=float, default=7.9)
    p.add_argument("--signal-ghz", type=float, default=7.4)
    args = p.parse_args()
    on = np.load(args.on / "signal_late_window.npz")
    off = np.load(args.off / "signal_late_window.npz")
    if on["theta"].shape != off["theta"].shape or not np.array_equal(on["theta"], off["theta"]):
        raise ValueError("on/off late windows must use the same theta grid")
    design = signal_design(on["theta"], args.pump_ghz, args.signal_ghz)
    coefficients = np.linalg.lstsq(
        design,
        np.column_stack((on["output_voltage_v"], off["output_voltage_v"])),
        rcond=None,
    )[0]
    c_on = complex(coefficients[1, 0], -coefficients[2, 0])
    c_off = complex(coefficients[1, 1], -coefficients[2, 1])
    c_delta = c_on - c_off
    v_peak = abs(c_delta)
    cancellation_ratio = float(abs(c_off) / max(v_peak, np.finfo(float).tiny))
    # Match production port_s_from_unit_current_response(): S=2V/(Z0 I).
    gain_db = 20.0 * math.log10(2.0 * v_peak / (50.0 * args.signal_current_a))
    result = {
        "method": "same_state_pump_only_control_subtraction",
        "signal_frequency_ghz": args.signal_ghz,
        "signal_current_a_peak": args.signal_current_a,
        "on_signal_coefficient_peak_v": abs(c_on),
        "off_signal_leakage_coefficient_peak_v": abs(c_off),
        "subtracted_signal_coefficient_peak_v": v_peak,
        "gain_db_50ohm": gain_db,
        "hb_reference_gain_vs_off_db": float(args.hb_reference_db),
        "hb_reference_source": args.hb_reference_source,
        "cancellation_ratio": cancellation_ratio,
        "cancellation_warning": bool(cancellation_ratio > 50.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
