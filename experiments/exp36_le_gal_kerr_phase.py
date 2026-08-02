"""Measure Le Gal pump Kerr shift with the shared propagation primitive."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from twpa_solver.builders.le_gal_2025 import build_effective_snail_line
from twpa_solver.core import CircuitMatrices
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
    measure_pump_nonlinear_wavenumber,
)

PUMP_GHZ = 7.5
PUMP_MODES = (1, 3, 5)
REFERENCE_PUMP_DBM = -118.4
CELL_LENGTH_M = 8.7e-6


def settings() -> NewtonKrylovSettings:
    """Return the benchmark pump-solver settings."""
    return NewtonKrylovSettings(
        newton_tol=1e-9,
        max_newton=25,
        gmres_rtol=1e-7,
        gmres_atol=0.0,
        gmres_restart=30,
        gmres_maxiter=50,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


def solve_pump(
    circuit: CircuitMatrices, pump_dbm: float
) -> tuple[FullPumpProblem, np.ndarray]:
    """Solve one pump power and return its problem and converged state."""
    z0_ohm = float(circuit.metadata["port_impedance_ohm"])
    omega_p = 2.0 * math.pi * PUMP_GHZ * 1e9
    problem = FullPumpProblem(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        branch=circuit.branch_law,
        grid=HarmonicGrid(np.asarray(PUMP_MODES), nt=16, omega=omega_p),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=math.sqrt(
            2.0 * 10.0 ** ((pump_dbm - 30.0) / 10.0) / z0_ohm
        ),
    )
    state, reports = HarmonicNewtonKrylovSolver(settings()).solve_continuation(
        problem, continuation_steps=8
    )
    if not reports[-1].converged:
        raise RuntimeError(f"pump solve failed at {pump_dbm} dBm")
    return problem, np.asarray(state)


def main() -> int:
    """Run the pump-power scaling measurement and write its artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=700)
    parser.add_argument(
        "--pump-dbm",
        type=float,
        nargs="+",
        default=[-108.4, -98.4, -88.4, -83.4, -78.4],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exp36_le_gal_kerr_phase"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    circuit = build_effective_snail_line(
        cells=args.cells,
        port_impedance_ohm=62.4,
        external_flux_on_small_junction=False,
    )
    reference_problem, reference_state = solve_pump(circuit, REFERENCE_PUMP_DBM)
    rows: list[dict[str, float | bool]] = []
    for pump_dbm in args.pump_dbm:
        _, pumped_state = solve_pump(circuit, float(pump_dbm))
        measurement = measure_pump_nonlinear_wavenumber(
            reference_problem,
            reference_state,
            pumped_state,
            cell_length_m=CELL_LENGTH_M,
        )
        rows.append(
            {
                "pump_dbm": float(pump_dbm),
                "pump_power_w": 10.0 ** ((float(pump_dbm) - 30.0) / 10.0),
                "dk_nl_rad_per_m": float(measurement["dk_nl_rad_per_m"]),
                "nonlinear_phase_rad": float(measurement["nonlinear_phase_rad"]),
                "linear_k_rad_per_m": float(
                    measurement["linear"]["wavenumber_rad_per_m"]
                ),
                "pumped_k_rad_per_m": float(
                    measurement["pumped"]["wavenumber_rad_per_m"]
                ),
                "pumped_recurrence_relative_residual": float(
                    measurement["pumped"]["recurrence_relative_residual"]
                ),
                "pump_converged": True,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    with (args.output_dir / "kerr_phase.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positive = [row for row in rows if float(row["dk_nl_rad_per_m"]) > 0.0]
    slope = float("nan")
    if len(positive) >= 2:
        slope = float(
            np.polyfit(
                np.log10([float(row["pump_power_w"]) for row in positive]),
                np.log10([float(row["dk_nl_rad_per_m"]) for row in positive]),
                1,
            )[0]
        )
    nominal = next(
        row for row in rows if abs(float(row["pump_dbm"]) + 78.4) < 1e-9
    )
    summary = {
        "measurement_primitive": (
            "twpa_solver.pump.wavenumber.measure_pump_nonlinear_wavenumber"
        ),
        "cells": args.cells,
        "pump_modes": list(PUMP_MODES),
        "source_mode": int(reference_problem.source_mode),
        "source_row": int(reference_problem.source_row),
        "power_law_slope": slope,
        "measured_dk_nl_rad_per_m_at_nominal": float(
            nominal["dk_nl_rad_per_m"]
        ),
    }
    (args.output_dir / "kerr_phase_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
