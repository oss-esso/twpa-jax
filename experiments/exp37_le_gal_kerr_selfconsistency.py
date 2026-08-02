"""Test whether the HB pump self-phase is consistent with its own waveform.

exp36 measured the pump self-phase on the Le Gal line as 0.18067 rad
(dk_nl = 29.67 rad/m) at -78.4 dBm, and a hand estimate
``dk_nl = (3/8)(g3/g1)<A^2>k_p`` predicted 151.7 rad/m -- a factor 5.11, close
to the 4.90 by which the CME oracle's coupling was reduced to match the solver.
That hand estimate carries assumptions about the phasor normalisation, so it
cannot settle anything on its own.

This replaces it with a check that is internal to the solver.  It takes the
converged pump state, reconstructs the branch flux and branch current with the
solver's *own* AFT synthesis and projection, forms the effective per-branch
inductance the fundamental actually sees,

    g_eff = I_1 / Psi_1,        L_eff = 1 / g_eff,

and integrates the discrete-ladder wavenumber built from ``L_eff`` along the
line.  The accumulated phase difference against a deeply linear pump is then
compared with the phase exp36 measured directly at the ports.

Both numbers come from the same converged state, so agreement means the solver
is self-consistent and the hand estimate was wrong; disagreement localises the
defect to the phase the solve actually accumulates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from twpa_solver.builders.le_gal_2025 import (
    build_effective_snail_line,
    ladder_dispersion,
)
from twpa_solver.pump import (
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
)
from twpa_solver.pump.problem import FullPumpProblem

PUMP_GHZ = 7.5
PUMP_MODES = (1, 3, 5)
NOMINAL_PUMP_DBM = -78.4
REFERENCE_PUMP_DBM = -118.4
CELL_LENGTH_M = 8.7e-6
GROUND_CAPACITANCE_F = 223.5e-15
SNAIL_CAPACITANCE_F = 31e-15
MEASURED_KERR_PHASE_RAD = 0.18066957963645902


def settings() -> NewtonKrylovSettings:
    """Solver settings copied from `scripts/run_le_gal_2025_hb.py`."""
    return NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=25, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=30, gmres_maxiter=50, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
    )


def solve_pump(circuit, pump_dbm: float) -> tuple[FullPumpProblem, np.ndarray]:
    """Converged pump state at one drive power, with its problem."""
    z0_ohm = float(circuit.metadata["port_impedance_ohm"])
    omega_p = 2.0 * math.pi * PUMP_GHZ * 1e9
    problem = FullPumpProblem(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        branch=circuit.branch_law,
        grid=HarmonicGrid(
            np.array(PUMP_MODES), nt=max(16, 2 * max(PUMP_MODES) + 2), omega=omega_p
        ),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=math.sqrt(2.0 * 10.0 ** ((pump_dbm - 30.0) / 10.0) / z0_ohm),
    )
    state, reports = HarmonicNewtonKrylovSolver(settings()).solve_continuation(
        problem, continuation_steps=8
    )
    if not reports[-1].converged:
        raise SystemExit(f"pump solve failed at {pump_dbm} dBm")
    return problem, np.asarray(state)


def effective_inductance(problem: FullPumpProblem, state: np.ndarray) -> np.ndarray:
    """Per-branch inductance seen by the pump fundamental.

    Uses the solver's own synthesis and positive-frequency projection, so no
    phasor-normalisation convention is introduced here.
    """
    psi_t = problem.branch_flux_time(state)
    total_t = psi_t + problem.dc_branch_flux[None, :]
    current_t = problem.branch.current(total_t) - problem.branch.current(
        problem.dc_branch_flux[None, :]
    )
    psi_coeffs = problem.grid.project_positive(psi_t)
    current_coeffs = problem.grid.project_positive(current_t)
    slope = current_coeffs[0] / psi_coeffs[0]
    return 1.0 / slope.real


def accumulated_phase(inductance: np.ndarray) -> float:
    """Total pump phase along the line for a per-branch inductance profile."""
    wavenumber = ladder_dispersion(
        2.0 * math.pi * PUMP_GHZ * 1e9,
        inductance_h=inductance,
        snail_capacitance_f=SNAIL_CAPACITANCE_F,
        ground_capacitance_f=GROUND_CAPACITANCE_F,
        cell_length_m=CELL_LENGTH_M,
    )
    return float(np.sum(np.asarray(wavenumber) * CELL_LENGTH_M))


def main() -> int:
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
        cells=args.cells, port_impedance_ohm=62.4,
        external_flux_on_small_junction=False,
    )
    length_m = args.cells * CELL_LENGTH_M

    problem_lo, state_lo = solve_pump(circuit, REFERENCE_PUMP_DBM)
    problem_hi, state_hi = solve_pump(circuit, NOMINAL_PUMP_DBM)
    inductance_lo = effective_inductance(problem_lo, state_lo)
    inductance_hi = effective_inductance(problem_hi, state_hi)

    phase_lo = accumulated_phase(inductance_lo)
    phase_hi = accumulated_phase(inductance_hi)
    predicted = phase_hi - phase_lo

    relative_inductance_shift = float(
        np.mean(inductance_hi / inductance_lo) - 1.0
    )
    summary = {
        "cells": args.cells,
        "length_m": length_m,
        "linear_inductance_h_mean": float(np.mean(inductance_lo)),
        "saturated_inductance_h_mean": float(np.mean(inductance_hi)),
        "relative_inductance_shift": relative_inductance_shift,
        "linear_phase_rad": phase_lo,
        "saturated_phase_rad": phase_hi,
        "predicted_kerr_phase_rad": predicted,
        "predicted_dk_nl_rad_per_m": predicted / length_m,
        "measured_kerr_phase_rad": MEASURED_KERR_PHASE_RAD,
        "measured_dk_nl_rad_per_m": MEASURED_KERR_PHASE_RAD / length_m,
        "ratio_predicted_over_measured": predicted / MEASURED_KERR_PHASE_RAD,
    }
    (args.output_dir / "kerr_selfconsistency.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
