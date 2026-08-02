"""Measure Le Gal dispersion from the assembled circuit and phase budget."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from twpa_solver.builders.le_gal_2025 import (
    build_effective_snail_line,
    ladder_dispersion,
)
from twpa_solver.core import CircuitMatrices


def bloch_wavenumber(
    circuit: CircuitMatrices, frequencies_hz: np.ndarray, cell_length_m: float
) -> np.ndarray:
    """Extract k from the assembled residual's linearized interior cell."""
    c_matrix = circuit.C.tocsr()
    branch_tangent = circuit.branch_law.tangent(
        np.zeros((1, circuit.branch_count), dtype=float)
    )[0]
    k_matrix = (
        circuit.K
        + circuit.Bphi @ sp.diags(branch_tangent) @ circuit.Bphi.T
    ).tocsr()
    diagonal_c = float(c_matrix[10, 10])
    off_diagonal_c = float(c_matrix[10, 11])
    diagonal_k = float(k_matrix[10, 10])
    off_diagonal_k = float(k_matrix[10, 11])
    values: list[float] = []
    for frequency_hz in frequencies_hz:
        omega = 2.0 * np.pi * float(frequency_hz)
        cosine = (
            diagonal_k - omega**2 * diagonal_c
        ) / (2.0 * (omega**2 * off_diagonal_c - off_diagonal_k))
        values.append(
            float(np.arccos(np.clip(cosine, -1.0, 1.0)) / cell_length_m)
        )
    return np.asarray(values)


def _relative_errors(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float, float]:
    relative = np.abs(candidate - reference) / np.maximum(np.abs(reference), 1e-30)
    return float(np.max(relative)), float(np.sqrt(np.mean(relative**2)))


def main() -> None:
    parameters = json.loads(
        Path("references/le_gal_2025_gain_compression/parameters.json").read_text()
    )["published_values"]
    circuit = build_effective_snail_line(
        cells=int(parameters["cells"]),
        cell_length_m=float(parameters["cell_length_m"]),
        critical_current_a=float(parameters["large_junction_critical_current_A"]),
        ratio=float(parameters["snail_ratio"]),
        snail_capacitance_f=float(parameters["snail_capacitance_F"]),
        ground_capacitance_f=float(parameters["ground_capacitance_F"]),
        flux_over_flux0=float(parameters["external_flux_over_flux0"]),
        port_impedance_ohm=float(np.sqrt(
            parameters["snail_inductance_H"] / parameters["ground_capacitance_F"]
        )),
    )
    frequencies_ghz = np.linspace(4.0, 11.0, 141)
    frequencies_hz = frequencies_ghz * 1e9
    measured = bloch_wavenumber(circuit, frequencies_hz, parameters["cell_length_m"])
    builder = ladder_dispersion(
        2.0 * np.pi * frequencies_hz,
        inductance_h=float(circuit.metadata["linear_inductance_h"]),
        snail_capacitance_f=float(parameters["snail_capacitance_F"]),
        ground_capacitance_f=float(parameters["ground_capacitance_F"]),
        cell_length_m=float(parameters["cell_length_m"]),
    )
    cme_k = np.asarray([
        2.0 * np.arcsin(
            np.clip(
                2.0 * np.pi * frequency * 1e9
                * np.sqrt(
                    float(parameters["snail_inductance_H"])
                    * (float(parameters["ground_capacitance_F"])
                       + float(parameters["snail_capacitance_F"]))
                ) / 2.0,
                -0.999999999,
                0.999999999,
            )
        ) / float(parameters["cell_length_m"])
        for frequency in frequencies_ghz
    ])
    builder_errors = _relative_errors(measured, builder)
    cme_errors = _relative_errors(measured, cme_k)
    output = Path("references/le_gal_2025_gain_compression/exp35_dispersion.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["f_GHz", "k_measured", "k_builder", "k_cme"])
        writer.writerows(zip(frequencies_ghz, measured, builder, cme_k))

    summary = {
        "method": "Bloch extraction from the assembled residual linearization",
        "builder_max_relative_error": builder_errors[0],
        "builder_rms_relative_error": builder_errors[1],
        "cme_max_relative_error": cme_errors[0],
        "cme_rms_relative_error": cme_errors[1],
    }
    Path("references/le_gal_2025_gain_compression/exp35_dispersion.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
