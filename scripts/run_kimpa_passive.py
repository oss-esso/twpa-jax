"""Sweep pump-off one-port S11 for a KIMPA fixture."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from twpa_solver.builders.kimpa import KIMPA_FIXTURES, build_kimpa
from twpa_solver.core import save_circuit
from twpa_solver.core.kinetic import kinetic_dc_branch_flux
from twpa_solver.signal.passive import passive_s_matrix


def _parabolic_minimum(freq: np.ndarray, value: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= len(freq) - 1:
        return float(freq[index]), float(value[index])
    x = freq[index - 1:index + 2]
    y = value[index - 1:index + 2]
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] <= 0:
        return float(freq[index]), float(value[index])
    vertex = float(-coefficients[1] / (2.0 * coefficients[0]))
    return vertex, float(np.polyval(coefficients, vertex))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=tuple(KIMPA_FIXTURES), default="kimpa_fabricated_nominal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-ghz", type=float, default=6.0)
    parser.add_argument("--stop-ghz", type=float, default=12.0)
    parser.add_argument("--points", type=int, default=1201)
    parser.add_argument("--dc-current-a", type=float, action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = args.output_dir / "circuit"
    circuit = build_kimpa(args.fixture)
    save_circuit(circuit, circuit_dir)
    frequencies = np.linspace(args.start_ghz, args.stop_ghz, args.points)
    traces = []
    rows = []
    for dc_current in args.dc_current_a or [0.0]:
        dc_flux = kinetic_dc_branch_flux(circuit, dc_current)
        scattering = passive_s_matrix(
            circuit_dir, frequencies * 1e9, ports=(1,), dc_branch_flux=dc_flux
        )[:, 0, 0]
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(scattering), 1e-300))
        traces.append((dc_current, scattering, magnitude_db))
        rows.extend([
            [float(dc_current), float(f), float(s.real), float(s.imag), float(db), float(np.angle(s))]
            for f, s, db in zip(frequencies, scattering, magnitude_db)
        ])
    csv_path = args.output_dir / "kimpa_passive.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dc_current_a", "freq_ghz", "s11_re", "s11_im", "s11_db", "s11_phase_rad"])
        writer.writerows(rows)
    print(f"wrote={csv_path}")
    for dc_current, _scattering, magnitude_db in traces:
        for index in range(1, len(frequencies) - 1):
            if magnitude_db[index] <= magnitude_db[index - 1] and magnitude_db[index] <= magnitude_db[index + 1]:
                frequency, depth = _parabolic_minimum(frequencies, magnitude_db, index)
                print(f"dc_current_a={dc_current:.9g} resonance_ghz={frequency:.9f} s11_db={depth:.6f}")
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 4))
        for dc_current, _scattering, magnitude_db in traces:
            axis.plot(frequencies, magnitude_db, label=f"I_dc={dc_current:g} A")
        axis.set(xlabel="Frequency (GHz)", ylabel="S11 (dB)", title=args.fixture)
        axis.grid(True, alpha=0.3)
        if len(traces) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "kimpa_passive.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
