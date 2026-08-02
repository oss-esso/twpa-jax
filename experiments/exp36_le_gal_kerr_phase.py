"""Measure the HB pump self-phase on the Le Gal effective-SNAIL line.

The CME oracle's ``projection_factor`` was set to 0.025510204081632654 by
calibrating it to reproduce "+0.367634 rad over 700 cells at -78.4 dBm", i.e.
``dk_nl = 60.4 rad/m``, taken from this solver.  That makes the oracle
dependent on the solver it is supposed to test, so the calibration target has
to be measured independently before it can be trusted or rejected.

This solves the pump alone at a ladder of pump powers and reads the accumulated
phase of the fundamental across the line, referenced to a deeply linear pump.
Two things are checked:

* the phase must grow **linearly in pump power** -- that is what makes it a Kerr
  phase rather than an artefact;
* its magnitude at -78.4 dBm is the number the CME was calibrated against.

No measurement and no published figure is used as a target here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from twpa_solver.builders.le_gal_2025 import build_effective_snail_line
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
CELLS = 700
CELL_LENGTH_M = 8.7e-6
CALIBRATED_DK_NL_RAD_PER_M = 60.4


def settings() -> NewtonKrylovSettings:
    """Solver settings copied from `scripts/run_le_gal_2025_hb.py`."""
    return NewtonKrylovSettings(
        newton_tol=1e-9, max_newton=25, gmres_rtol=1e-7, gmres_atol=0.0,
        gmres_restart=30, gmres_maxiter=50, min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled", compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
    )


def current_from_dbm(power_dbm: float, z0_ohm: float) -> float:
    """Peak drive current for a power in dBm into ``z0_ohm``."""
    return math.sqrt(2.0 * 10.0 ** ((power_dbm - 30.0) / 10.0) / z0_ohm)


def transmitted_phase(circuit, pump_dbm: float, cells: int) -> tuple[float, bool]:
    """Fundamental-mode phase from input to output node, and convergence."""
    z0_ohm = float(circuit.metadata["port_impedance_ohm"])
    omega_p = 2.0 * math.pi * PUMP_GHZ * 1e9
    problem = FullPumpProblem(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        branch=circuit.branch_law,
        grid=HarmonicGrid(
            np.array(PUMP_MODES), nt=max(16, 2 * max(PUMP_MODES) + 2), omega=omega_p
        ),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=current_from_dbm(pump_dbm, z0_ohm),
    )
    state, reports = HarmonicNewtonKrylovSolver(settings()).solve_continuation(
        problem, continuation_steps=8
    )
    fundamental = np.asarray(state)[0]
    source = fundamental[circuit.port_to_index[1]]
    load = fundamental[circuit.port_to_index[2]]
    return float(np.angle(load) - np.angle(source)), bool(reports[-1].converged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=CELLS)
    parser.add_argument(
        "--pump-dbm", type=float, nargs="+",
        default=[-108.4, -98.4, -93.4, -88.4, -83.4, -78.4, -73.4],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp36_le_gal_kerr_phase")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    circuit = build_effective_snail_line(
        cells=args.cells, port_impedance_ohm=62.4,
        external_flux_on_small_junction=False,
    )
    length_m = args.cells * CELL_LENGTH_M

    reference, ref_converged = transmitted_phase(
        circuit, REFERENCE_PUMP_DBM, args.cells
    )
    if not ref_converged:
        raise SystemExit(
            f"linear reference pump at {REFERENCE_PUMP_DBM} dBm did not converge; "
            "every phase below would be referenced to garbage"
        )
    print(
        f"linear reference: {REFERENCE_PUMP_DBM} dBm -> "
        f"transmitted phase {reference:+.6f} rad\n",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    print(
        f'{"Pp dBm":>8} {"phase rad":>11} {"Kerr rad":>10} '
        f'{"dk_nl 1/m":>11} {"rad/mW":>12} {"conv":>5}'
    )
    for pump_dbm in args.pump_dbm:
        phase, converged = transmitted_phase(circuit, float(pump_dbm), args.cells)
        kerr = phase - reference
        power_w = 10.0 ** ((float(pump_dbm) - 30.0) / 10.0)
        rows.append({
            "pump_dbm": float(pump_dbm),
            "pump_power_w": power_w,
            "transmitted_phase_rad": phase,
            "kerr_phase_rad": kerr,
            "dk_nl_rad_per_m": kerr / length_m,
            "kerr_per_watt_rad_per_w": kerr / power_w,
            "pump_converged": converged,
        })
        print(
            f"{pump_dbm:>8.1f} {phase:>11.6f} {kerr:>10.6f} "
            f"{kerr / length_m:>11.3f} {kerr / power_w * 1e-3:>12.4e} "
            f"{str(converged):>5}",
            flush=True,
        )

    with (args.output_dir / "kerr_phase.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A Kerr phase is linear in pump power; fitting log|phase| against log P
    # turns that into a slope of exactly 1 and exposes any other scaling.
    good = [
        r for r in rows
        if r["pump_converged"] and abs(float(r["kerr_phase_rad"])) > 1e-12
    ]
    slope = float("nan")
    if len(good) >= 2:
        log_power = np.log10([float(r["pump_power_w"]) for r in good])
        log_phase = np.log10([abs(float(r["kerr_phase_rad"])) for r in good])
        slope = float(np.polyfit(log_power, log_phase, 1)[0])

    nominal = next(
        (r for r in rows if abs(float(r["pump_dbm"]) - NOMINAL_PUMP_DBM) < 1e-9),
        None,
    )
    summary = {
        "cells": args.cells,
        "length_m": length_m,
        "pump_ghz": PUMP_GHZ,
        "pump_modes": list(PUMP_MODES),
        "reference_pump_dbm": REFERENCE_PUMP_DBM,
        "power_law_slope": slope,
        "power_law_slope_expected": 1.0,
        "measured_kerr_phase_rad_at_nominal": (
            None if nominal is None else float(nominal["kerr_phase_rad"])
        ),
        "measured_dk_nl_rad_per_m_at_nominal": (
            None if nominal is None else float(nominal["dk_nl_rad_per_m"])
        ),
        "calibrated_dk_nl_rad_per_m": CALIBRATED_DK_NL_RAD_PER_M,
        "ratio_measured_over_calibrated": (
            None if nominal is None
            else float(nominal["dk_nl_rad_per_m"]) / CALIBRATED_DK_NL_RAD_PER_M
        ),
    }
    (args.output_dir / "kerr_phase_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
