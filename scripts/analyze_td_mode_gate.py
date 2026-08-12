"""Extract modulation sidebands from a retained transient output-voltage trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_trace(
    theta: np.ndarray,
    voltage: np.ndarray,
    pump_frequency_ghz: float,
    *,
    peak_count: int = 12,
) -> dict[str, object]:
    """Return FFT peaks and the distance to the pump half-frequency."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    voltage = np.asarray(voltage, dtype=float).reshape(-1)
    if theta.size != voltage.size or theta.size < 16:
        raise ValueError("theta and voltage must have at least 16 matching samples")
    spacing = np.diff(theta)
    if not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
        raise ValueError("the retained voltage trace must be uniformly sampled")

    centered = voltage - np.mean(voltage)
    voltage_peak = float(np.max(np.abs(voltage)))
    if voltage_peak == 0.0:
        return {
            "sample_count": int(theta.size),
            "frequency_resolution_ghz": float(
                pump_frequency_ghz / (theta.size * spacing[0] / (2.0 * np.pi))
            ),
            "voltage_peak_v": 0.0,
            "usable": False,
            "reason": "retained output-voltage trace is identically zero",
            "half_pump_frequency_ghz": pump_frequency_ghz / 2.0,
            "peaks": [],
        }
    window = np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(centered * window))
    frequencies = np.fft.rfftfreq(
        centered.size, d=float(spacing[0]) / (2.0 * np.pi * pump_frequency_ghz)
    )
    spectrum[0] = 0.0
    peak_indices = np.argsort(spectrum)[-peak_count:][::-1]
    peaks = [
        {"frequency_ghz": float(frequencies[i]), "amplitude": float(spectrum[i])}
        for i in peak_indices
    ]
    half_pump = pump_frequency_ghz / 2.0
    half_index = int(np.argmin(np.abs(frequencies - half_pump)))
    resolution = float(frequencies[1] - frequencies[0])
    return {
        "sample_count": int(theta.size),
        "frequency_resolution_ghz": resolution,
        "voltage_peak_v": voltage_peak,
        "usable": True,
        "half_pump_frequency_ghz": half_pump,
        "half_pump_nearest_bin_ghz": float(frequencies[half_index]),
        "half_pump_amplitude": float(spectrum[half_index]),
        "peaks": peaks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pump-frequency-ghz", type=float, default=7.9)
    parser.add_argument("--peak-count", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.run_dir / "td_compact.npz")
    result = analyze_trace(
        data["output_voltage_theta"],
        data["output_voltage_trace_v"],
        args.pump_frequency_ghz,
        peak_count=args.peak_count,
    )
    text = json.dumps(result, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
