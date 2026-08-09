"""Validate the HB Kerr shift against the effective-SNAIL branch law."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp35_le_gal_dispersion import bloch_wavenumber
from experiments.exp36_le_gal_kerr_phase import (
    CELL_LENGTH_M,
    PUMP_GHZ,
    REFERENCE_PUMP_DBM,
    solve_pump,
)
from twpa_solver.builders.le_gal_2025 import build_effective_snail_line
from twpa_solver.core import CircuitMatrices
from twpa_solver.pump import FullPumpProblem, measure_pump_nonlinear_wavenumber

NOMINAL_PUMP_DBM = -78.4


def branch_law_prediction(
    circuit: CircuitMatrices,
    problem: FullPumpProblem,
    state: np.ndarray,
    wavenumber_rad_per_m: float,
) -> dict[str, float]:
    """Return the cubic branch-law prediction in the solver phasor convention."""
    source_row = int(problem.source_row)
    branch_coefficients = np.asarray(problem.BphiT @ state[source_row]).ravel()
    peak_amplitude = 2.0 * np.abs(branch_coefficients)
    mean_peak_squared = float(np.mean(peak_amplitude**2))
    law = circuit.branch_law
    equilibrium = np.asarray(law.equilibrium_flux, dtype=float)
    phase_small = equilibrium / float(law.phi0)
    phase_large = (equilibrium - law.phi_ext) / (3.0 * float(law.phi0))
    g1 = float(
        np.mean(
            law.critical_current
            / law.phi0
            * (law.ratio * np.cos(phase_small) + np.cos(phase_large) / 3.0)
        )
    )
    g3 = float(
        np.mean(
            law.critical_current
            / law.phi0**3
            * (-law.ratio * np.cos(phase_small) - np.cos(phase_large) / 27.0)
            / 6.0
        )
    )
    predicted = 3.0 / 8.0 * (g3 / g1) * mean_peak_squared * wavenumber_rad_per_m
    return {
        "g1_a_per_wb": g1,
        "g3_a_per_wb3": g3,
        "g3_over_g1_per_wb2": g3 / g1,
        "mean_peak_branch_flux_squared_wb2": mean_peak_squared,
        "dk_nl_analytic_rad_per_m": predicted,
    }


def main() -> int:
    """Run the nominal-power self-consistency comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=700)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exp37_le_gal_kerr_selfconsistency"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    circuit = build_effective_snail_line(
        cells=args.cells,
        port_impedance_ohm=62.4,
        external_flux_on_small_junction=False,
    )
    reference_problem, reference_state = solve_pump(circuit, REFERENCE_PUMP_DBM)
    _, pumped_state = solve_pump(circuit, NOMINAL_PUMP_DBM)
    measurement = measure_pump_nonlinear_wavenumber(
        reference_problem,
        reference_state,
        pumped_state,
        cell_length_m=CELL_LENGTH_M,
    )
    bloch = float(
        bloch_wavenumber(
            circuit,
            np.asarray([PUMP_GHZ * 1e9]),
            CELL_LENGTH_M,
        )[0]
    )
    analytic = branch_law_prediction(
        circuit,
        reference_problem,
        pumped_state,
        float(measurement["linear"]["wavenumber_rad_per_m"]),
    )
    measured = float(measurement["dk_nl_rad_per_m"])
    predicted = float(analytic["dk_nl_analytic_rad_per_m"])
    ratio = measured / predicted
    pump_power_w = 10.0 ** ((NOMINAL_PUMP_DBM - 30.0) / 10.0)
    cme_z0_ohm = math.sqrt(869.6e-12 / 223.5e-15)
    cme_pump_coefficient_wb = math.sqrt(pump_power_w * cme_z0_ohm) / (
        math.sqrt(2.0) * 2.0 * math.pi * PUMP_GHZ * 1e9
    )
    projection_denominator = (
        float(analytic["g3_over_g1_per_wb2"])
        * float(measurement["linear"]["wavenumber_rad_per_m"])
        * cme_pump_coefficient_wb**2
    )
    implied_factor = measured / projection_denominator
    unit_coefficients = np.zeros((3, 1), dtype=np.complex128)
    unit_coefficients[reference_problem.source_row, 0] = 1.0
    synthesized_peak = float(
        np.max(np.abs(reference_problem.grid.synthesize(unit_coefficients)))
    )
    summary = {
        "cells": args.cells,
        "pump_dbm": NOMINAL_PUMP_DBM,
        "pump_ghz": PUMP_GHZ,
        "bloch_wavenumber_rad_per_m": bloch,
        "driven_linear_wavenumber_rad_per_m": float(
            measurement["linear"]["wavenumber_rad_per_m"]
        ),
        "bloch_driven_relative_difference": abs(
            float(measurement["linear"]["wavenumber_rad_per_m"]) - bloch
        )
        / bloch,
        "measurement": measurement,
        "analytic": analytic,
        "measured_over_analytic": ratio,
        "phasor_reconstruction": {
            "convention": "x(t) = 2 Re sum_k X_k exp(+i k omega t)",
            "unit_fundamental_coefficient_peak": synthesized_peak,
            "branch_peak_amplitude_is": "2 * abs(branch_fundamental_coefficient)",
            "cme_input_pump_coefficient_wb": cme_pump_coefficient_wb,
        },
        "hb_implied_projection_factor": implied_factor,
        "projection_factor_over_one_eighth": implied_factor / (1.0 / 8.0),
        "projection_factor_over_committed_old": (
            implied_factor / 0.025510204081632654
        ),
        "agreement_percent": 100.0 * abs(measured - predicted) / abs(predicted),
    }
    (args.output_dir / "kerr_selfconsistency.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
