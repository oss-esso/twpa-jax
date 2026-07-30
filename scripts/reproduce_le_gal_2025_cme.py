"""Generate inexpensive independent CME benchmark arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from references.le_gal_2025_gain_compression.cme import (
    CMEParameters,
    integrate_cme,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-points", type=int, default=31)
    parser.add_argument("--power-points", type=int, default=41)
    args = parser.parse_args()
    frequencies = np.linspace(4.0, 11.0, args.signal_points)
    powers = np.linspace(-115.0, -94.0, args.power_points)
    gains = np.empty((frequencies.size, powers.size))
    pump = np.empty_like(gains)
    for fi, _frequency in enumerate(frequencies):
        for pi, power in enumerate(powers):
            signal = 10.0 ** ((power + 115.0) / 20.0) * 1e-3
            _, envelopes = integrate_cme(
                (1.0, signal, 0.0),
                CMEParameters(length=1.0, coupling=4.0),
                points=101,
            )
            gains[fi, pi] = 20.0 * np.log10(max(abs(envelopes[1, -1] / signal), 1e-300))
            pump[fi, pi] = 20.0 * np.log10(max(abs(envelopes[0, -1]), 1e-300))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, frequency_GHz=frequencies, power_dBm=powers, gain_dB=gains, pump_dB=pump)
    args.output.with_suffix(".json").write_text(json.dumps({"model": "independent_cme", "status": "generated"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
