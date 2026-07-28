"""Run a finite-signal compression sweep at one pump operating point."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from twpa_solver import default_loss_model
from twpa_solver.builders.jc_doc import build_fqjtwpa, build_jpa, build_jtwpa
from twpa_solver.core import CircuitMatrices, load_circuit
from twpa_solver.multitone.basis import (
    MultiToneBasis,
    build_lattice_basis,
    build_sideband_matched_basis,
    build_three_tone_basis,
)
from twpa_solver.multitone.compression import solve_signal_power_point
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
from twpa_solver.pump.basis import (
    load_pump_basis_from_solution,
    resolve_pump_basis,
)
from twpa_solver.multitone.seed import promote_pump_solution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-signal-power", type=int, default=5)
    parser.add_argument(
        "--multitone-basis",
        choices=("matched", "three_tone", "lattice"),
        default="matched",
    )
    parser.add_argument("--multitone-sidebands", type=int, default=2)
    parser.add_argument("--resource-budget-gb", type=float, default=8.0)
    parser.add_argument("--save-states", choices=("none", "last", "selected", "all"), default="none")
    parser.add_argument("--signal-ghz-min", type=float)
    parser.add_argument("--signal-ghz-max", type=float)
    parser.add_argument("--n-signal-freq", type=int, default=1)
    parser.add_argument("--signal-workers", type=int, default=1)
    parser.add_argument("--summary-json", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixture", choices=("jpa", "jtwpa", "fqjtwpa"))
    source.add_argument("--circuit-dir", type=Path)
    parser.add_argument("--pump-solution-dir", type=Path)
    parser.add_argument("--pump-freq-ghz", type=float, default=4.75001)
    parser.add_argument(
        "--signal-ghz",
        type=float,
        required=True,
        help="Signal frequency in GHz.",
    )
    parser.add_argument("--pump-current-a", type=float)
    parser.add_argument(
        "--pump-current-jc-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the pump current (default: 1.0, the "
            "validated convention in docs/pump_current_conversions.tex)."
        ),
    )
    parser.add_argument("--pump-mode-policy", default="positive_odd_jc")
    parser.add_argument("--pump-mode-count", type=int, default=10)
    parser.add_argument("--pump-modes")
    parser.add_argument("--pump-harmonics", type=int, default=10)
    parser.add_argument("--pump-nt", type=int, default=40)
    parser.add_argument("--source-port", type=int)
    parser.add_argument("--out-port", type=int)
    parser.add_argument("--diagnostic-port", type=int)
    parser.add_argument("--attenuation-db", type=float, default=None)
    parser.add_argument("--signal-current-min-a", type=float, default=1e-12)
    parser.add_argument("--signal-current-max-a", type=float, default=1e-9)
    parser.add_argument("--recovery", choices=("plain", "ladder"), default="ladder")
    parser.add_argument("--signal-substep-init-db", type=float, default=0.5)
    parser.add_argument("--signal-substep-min-db", type=float, default=0.01)
    parser.add_argument("--signal-continuation-deadline-s", type=float, default=0.0)
    parser.add_argument("--signal-arclength-recovery", action="store_true")
    parser.add_argument(
        "--multitone-preconditioner",
        choices=MULTITONE_PRECONDITIONERS,
        default="real_coupled_fast",
    )
    return parser


def _fixture_circuit(name: str) -> tuple[CircuitMatrices, dict[str, object]]:
    builders = {"jpa": build_jpa, "jtwpa": build_jtwpa, "fqjtwpa": build_fqjtwpa}
    builder, metadata = builders[name]()
    arrays = builder.assemble()
    return CircuitMatrices(
        C=arrays["C"],
        G=arrays["G"],
        K=arrays["K"],
        Bphi=arrays["Bphi"],
        Ic=arrays["Ic"],
        port_to_index=arrays["ports"],
        metadata=metadata,
    ), metadata


def _load_source(args: argparse.Namespace) -> tuple[CircuitMatrices, dict[str, object], str]:
    if args.circuit_dir is not None:
        circuit = load_circuit(args.circuit_dir)
        return circuit, circuit.metadata, str(args.circuit_dir)
    fixture = args.fixture or "jpa"
    circuit, metadata = _fixture_circuit(fixture)
    return circuit, metadata, fixture


def _resolve_attenuation(args: argparse.Namespace) -> tuple[float, str]:
    if args.attenuation_db is not None:
        return float(args.attenuation_db), "explicit"
    if args.circuit_dir is None:
        return 0.0, "fixture_default_zero"
    return (
        float(default_loss_model().attenuation_db(args.pump_freq_ghz)),
        "themis_default_loss_model",
    )


def _build_multitone_basis(
    args: argparse.Namespace,
    pump_modes: list[int],
    omega_p: float,
    delta: float,
) -> MultiToneBasis:
    omega_max = omega_p * (max(pump_modes) + 1.0)
    if args.multitone_basis == "matched":
        basis = build_sideband_matched_basis(
            pump_modes,
            args.multitone_sidebands,
            omega_p,
            delta,
            omega_max,
        )
    elif args.multitone_basis == "lattice":
        basis = build_lattice_basis(
            pump_modes,
            args.multitone_sidebands,
            omega_p,
            delta,
            omega_max,
        )
    else:
        basis = build_three_tone_basis(omega_p, delta)
    represented_modes = {tone.h for tone in basis.tones if tone.q == 0}
    missing_modes = sorted(set(pump_modes) - represented_modes)
    if missing_modes:
        raise ValueError(
            "multitone basis silently truncates pump harmonics: "
            f"pump_modes={list(pump_modes)}, "
            f"multitone_q0_modes={sorted(represented_modes)}, "
            f"missing={missing_modes}"
        )
    return basis


def _solve_compression(args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, np.ndarray], dict[str, object]]:
    circuit, metadata, circuit_source = _load_source(args)
    attenuation_db, attenuation_source = _resolve_attenuation(args)
    source_port = int(args.source_port or 1)
    out_port = int(args.out_port or (1 if circuit_source == "jpa" else 2))
    diagnostic_port = int(args.diagnostic_port or out_port)
    for label, port in (("source", source_port), ("output", out_port), ("diagnostic", diagnostic_port)):
        if port not in circuit.port_to_index:
            raise ValueError(f"{label} port {port} is absent from circuit ports {sorted(circuit.port_to_index)}")
    pump_sources = metadata.get("pump_sources", [])
    default_current = pump_sources[0].get("current_a") if pump_sources else None
    if args.pump_current_a is None and default_current is None:
        raise ValueError("--pump-current-a is required when circuit metadata has no pump source")
    pump_current = float(args.pump_current_a or default_current) * float(args.pump_current_jc_scale)
    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9
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
    if args.pump_solution_dir is not None:
        pump_state, pump_basis = load_pump_basis_from_solution(
            args.pump_solution_dir,
            fallback_omega_p=omega_p,
        )
        pump_reports: list[object] = []
        pump_converged = True
    else:
        pump_basis = resolve_pump_basis(
            policy=args.pump_mode_policy,
            omega_p=omega_p,
            harmonics=args.pump_harmonics,
            mode_count=args.pump_mode_count,
            explicit_modes=args.pump_modes,
            design_meta=metadata,
        )
        pump_problem = FullPumpProblem(
            C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
            branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
            grid=HarmonicGrid(np.asarray(pump_basis.modes), nt=args.pump_nt, omega=omega_p),
            pump_node_index=circuit.port_to_index[source_port], pump_current_a=pump_current,
        )
        pump_settings = replace(settings, preconditioner="real_coupled")
        pump_state, pump_reports = HarmonicNewtonKrylovSolver(
            pump_settings
        ).solve_continuation(pump_problem, continuation_steps=4)
        pump_converged = bool(pump_reports[-1].converged)
    delta = omega_p - 2.0 * math.pi * args.signal_ghz * 1e9
    basis = _build_multitone_basis(args, pump_basis.modes, omega_p, delta)
    pump_seed = promote_pump_solution(pump_state, pump_basis, basis)
    pump_source = MultiToneDrive(basis.pump_tone, circuit.port_to_index[source_port], pump_current).to_coeffs(
        basis, circuit.node_count
    )
    signal_unit = MultiToneDrive(basis.signal_tone, circuit.port_to_index[source_port], 1.0).to_coeffs(
        basis, circuit.node_count
    )
    currents = np.geomspace(args.signal_current_min_a, args.signal_current_max_a, max(args.n_signal_power, 1))
    pump_off_path = AffineSourcePath.signal_turn_on(
        np.zeros_like(pump_source),
        signal_unit * float(currents[0]),
    )
    pump_off_problem = FullMultiToneProblem(
        circuit,
        basis,
        pump_off_path,
        preconditioner=selected_preconditioner,
    )
    pump_off_state, pump_off_report = HarmonicNewtonKrylovSolver(settings).solve_one(
        pump_off_problem,
        np.zeros_like(pump_seed),
        1.0,
    )
    if not pump_off_report.converged:
        raise RuntimeError("pump-off small-signal reference did not converge")
    pump_off_s21 = tone_s21(
        pump_off_state,
        basis,
        circuit,
        signal_tone=basis.signal_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=float(currents[0]),
    )
    pump_off_gain_db = float(20.0 * np.log10(max(abs(pump_off_s21), 1e-300)))
    states: dict[str, np.ndarray] = {}
    points: list[dict[str, float]] = []
    previous = pump_seed
    previous_previous = None
    previous_current = 0.0
    previous_previous_current = 0.0
    reference_gain = None
    for index, current in enumerate(currents):
        base_problem = FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.pump_turn_on(pump_source),
            preconditioner=selected_preconditioner,
        )
        solved = solve_signal_power_point(
            base_problem,
            previous,
            previous_previous,
            float(current),
            pump_source=pump_source,
            signal_source=signal_unit,
            solver=HarmonicNewtonKrylovSolver(settings),
            signal_current_prev_a=previous_current,
            signal_current_prevprev_a=previous_previous_current,
            recovery=args.recovery,
            pump_seed=pump_seed,
            signal_substep_init_db=args.signal_substep_init_db,
            signal_substep_min_db=args.signal_substep_min_db,
            continuation_deadline_s=args.signal_continuation_deadline_s,
            arclength_recovery=args.signal_arclength_recovery,
        )
        state = solved.state
        if solved.status == "VALID_SOLVED":
            s21 = tone_s21(
                state, basis, circuit, signal_tone=basis.signal_tone,
                source_port=source_port, out_port=out_port, source_current_a=float(current),
            )
            gain_db = float(20.0 * np.log10(max(abs(s21), 1e-300)))
            reference_gain = gain_db if reference_gain is None else reference_gain
            previous_previous = previous
            previous_previous_current = previous_current
            previous = state
            previous_current = float(current)
        else:
            gain_db = float("nan")
        points.append({
            "signal_current_a": float(current),
            "gain_db": gain_db,
            "gain_vs_off_db": gain_db - pump_off_gain_db,
            "compression_db": float(reference_gain - gain_db) if reference_gain is not None and np.isfinite(gain_db) else float("nan"),
            "status": solved.status,
            "recovery_rung": solved.used_recovery,
            "last_converged_signal_current_a": solved.last_converged_signal_current_a,
        })
        if index == len(currents) - 1:
            states["last"] = state
    small_signal_gain_db = float(points[0]["gain_db"]) if points else float("nan")
    no_gain = not np.isfinite(small_signal_gain_db) or small_signal_gain_db < 3.0
    if no_gain:
        for point in points:
            point["compression_db"] = float("nan")
    summary_status = (
        "NO_GAIN_AT_OPERATING_POINT"
        if no_gain
        else "VALID_SOLVED"
        if all(point["status"] == "VALID_SOLVED" for point in points)
        else "CHECK"
    )
    summary = {
        "status": summary_status,
        "stability_status": "NOT_CHECKED",
        "circuit_source": circuit_source,
        "pump_freq_ghz": args.pump_freq_ghz,
        "signal_ghz": args.signal_ghz,
        "pump_current_a": pump_current,
        "pump_converged": pump_converged,
        "pump_current_jc_scale": args.pump_current_jc_scale,
        "pump_solution_dir": str(args.pump_solution_dir) if args.pump_solution_dir else None,
        "source_port": source_port,
        "out_port": out_port,
        "diagnostic_port": diagnostic_port,
        "attenuation_db": attenuation_db,
        "attenuation_source": attenuation_source,
        "small_signal_gain_db": small_signal_gain_db,
        "small_signal_gain_vs_off_db": float(points[0]["gain_vs_off_db"]),
        "pump_off_gain_db": pump_off_gain_db,
        "p1db": None,
        "message": (
            f"Small-signal gain {small_signal_gain_db:.6g} dB is below the "
            "3 dB compression-study threshold; P1dB is not reported."
            if no_gain
            else None
        ),
        "multitone_preconditioner": selected_preconditioner,
        "recovery": args.recovery,
        "basis": basis.to_metadata(),
    }
    return points, states, summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points, states, summary = _solve_compression(args)
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
