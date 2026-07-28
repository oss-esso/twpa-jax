"""Single-operating-point multitone compression CLI scaffold."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from twpa_solver.builders.jc_doc import build_jpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.observables import tone_s21
from twpa_solver.multitone.preconditioners import (
    MULTITONE_PRECONDITIONERS,
    resolve_multitone_preconditioner,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.multitone.io import write_compression_outputs
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import PumpBasis
from twpa_solver.multitone.seed import promote_pump_solution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-signal-power", type=int, default=5)
    parser.add_argument("--multitone-basis", choices=("three_tone", "lattice"), default="three_tone")
    parser.add_argument("--resource-budget-gb", type=float, default=8.0)
    parser.add_argument("--save-states", choices=("none", "last", "selected", "all"), default="none")
    parser.add_argument("--signal-ghz-min", type=float)
    parser.add_argument("--signal-ghz-max", type=float)
    parser.add_argument("--n-signal-freq", type=int, default=1)
    parser.add_argument("--signal-workers", type=int, default=1)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--fixture", choices=("jpa",), default="jpa")
    parser.add_argument("--pump-freq-ghz", type=float, default=4.75001)
    parser.add_argument("--signal-ghz", type=float, default=4.5)
    parser.add_argument("--pump-current-a", type=float)
    parser.add_argument("--signal-current-min-a", type=float, default=1e-12)
    parser.add_argument("--signal-current-max-a", type=float, default=1e-9)
    parser.add_argument(
        "--multitone-preconditioner",
        choices=MULTITONE_PRECONDITIONERS,
        default="real_coupled_fast",
    )
    return parser


def _solve_jpa(args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, np.ndarray], dict[str, object]]:
    builder, metadata = build_jpa()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"], G=arrays["G"], K=arrays["K"], Bphi=arrays["Bphi"],
        Ic=arrays["Ic"], port_to_index=arrays["ports"],
    )
    pump_current = float(args.pump_current_a or metadata["pump_sources"][0]["current_a"])
    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9
    pump_problem = FullPumpProblem(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
        grid=HarmonicGrid(np.asarray([1]), nt=8, omega=omega_p),
        pump_node_index=circuit.port_to_index[1], pump_current_a=pump_current,
    )
    selected_preconditioner = resolve_multitone_preconditioner(
        args.multitone_preconditioner
    )
    solver_preconditioner = (
        "real_coupled_fast"
        if selected_preconditioner == "floquet_sector"
        else selected_preconditioner
    )
    settings = NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=20, gmres_maxiter=40, min_alpha=1.0 / 1024.0,
        preconditioner=solver_preconditioner,
        compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft",
    )
    pump_settings = replace(settings, preconditioner="real_coupled")
    pump_state, pump_reports = HarmonicNewtonKrylovSolver(
        pump_settings
    ).solve_continuation(pump_problem, continuation_steps=4)
    basis = build_three_tone_basis(omega_p, omega_p - 2.0 * math.pi * args.signal_ghz * 1e9)
    pump_basis = PumpBasis([1], "dense_real", omega_p)
    pump_seed = promote_pump_solution(pump_state, pump_basis, basis)
    pump_source = MultiToneDrive(basis.pump_tone, circuit.port_to_index[1], pump_current).to_coeffs(
        basis, circuit.node_count
    )
    signal_unit = MultiToneDrive(basis.signal_tone, circuit.port_to_index[1], 1.0).to_coeffs(
        basis, circuit.node_count
    )
    currents = np.geomspace(args.signal_current_min_a, args.signal_current_max_a, max(args.n_signal_power, 1))
    states: dict[str, np.ndarray] = {}
    points: list[dict[str, float]] = []
    previous = pump_seed
    reference_gain = None
    for index, current in enumerate(currents):
        path = AffineSourcePath.signal_turn_on(pump_source, signal_unit * float(current))
        problem = FullMultiToneProblem(
            circuit,
            basis,
            path,
            preconditioner=selected_preconditioner,
        )
        state, report = HarmonicNewtonKrylovSolver(settings).solve_one(problem, previous, 1.0)
        previous = state
        if report.converged:
            s21 = tone_s21(
                state, basis, circuit, signal_tone=basis.signal_tone,
                source_port=1, out_port=1, source_current_a=float(current),
            )
            gain_db = float(20.0 * np.log10(max(abs(s21), 1e-300)))
            reference_gain = gain_db if reference_gain is None else reference_gain
            status = "VALID_SOLVED"
        else:
            gain_db = float("nan")
            status = "SIGNAL_CONTINUATION_FAILED"
        points.append({
            "signal_current_a": float(current),
            "gain_db": gain_db,
            "compression_db": float(reference_gain - gain_db) if reference_gain is not None and np.isfinite(gain_db) else float("nan"),
            "status": status,
        })
        if index == len(currents) - 1:
            states["last"] = state
    summary = {
        "status": "VALID_SOLVED" if points and all(point["status"] == "VALID_SOLVED" for point in points) else "CHECK",
        "stability_status": "NOT_CHECKED",
        "fixture": args.fixture,
        "pump_freq_ghz": args.pump_freq_ghz,
        "signal_ghz": args.signal_ghz,
        "pump_current_a": pump_current,
        "pump_converged": bool(pump_reports[-1].converged),
        "multitone_preconditioner": selected_preconditioner,
        "basis": basis.to_metadata(),
    }
    return points, states, summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points, states, summary = _solve_jpa(args)
    summary.update({
        "multitone_basis": args.multitone_basis,
        "n_signal_power": args.n_signal_power,
        "resource_budget_gb": args.resource_budget_gb,
        "signal_frequency_range_ghz": [args.signal_ghz_min, args.signal_ghz_max],
        "n_signal_freq": args.n_signal_freq,
        "signal_workers": args.signal_workers,
    })
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_compression_outputs(args.output_dir, points, summary=summary, states=states, save_states=args.save_states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
