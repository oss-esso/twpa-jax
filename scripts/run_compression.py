"""Run a finite-signal compression sweep at one pump operating point."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
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
from twpa_solver.multitone.observables import spatial_profiles, tone_s21
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
        help="Signal frequency in GHz; optional for a frequency-range sweep.",
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
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--signal-current-min-a", type=float, default=1e-12)
    parser.add_argument("--signal-current-max-a", type=float, default=1e-9)
    parser.add_argument("--recovery", choices=("plain", "ladder"), default="ladder")
    parser.add_argument("--signal-substep-init-db", type=float, default=0.5)
    parser.add_argument("--signal-substep-min-db", type=float, default=0.01)
    parser.add_argument("--signal-continuation-deadline-s", type=float, default=0.0)
    parser.add_argument("--signal-arclength-recovery", action="store_true")
    parser.add_argument(
        "--spatial-profiles",
        action="store_true",
        help="Write branch profiles at small, mid, and near-P1dB signal powers.",
    )
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


def _current_to_dbm(current_a: float, z0_ohm: float, attenuation_db: float) -> float:
    power_w = current_a * current_a * z0_ohm / 2.0
    return 10.0 * math.log10(power_w / 1.0e-3) + attenuation_db


def _interpolate_p1db_current(points: list[dict[str, object]]) -> float | None:
    valid = [
        point
        for point in points
        if point["status"] == "VALID_SOLVED"
        and np.isfinite(float(point["compression_db"]))
    ]
    for left, right in zip(valid, valid[1:]):
        c_left = float(left["compression_db"])
        c_right = float(right["compression_db"])
        if c_left <= 1.0 <= c_right and c_right > c_left:
            fraction = (1.0 - c_left) / (c_right - c_left)
            log_left = math.log10(float(left["signal_current_a"]))
            log_right = math.log10(float(right["signal_current_a"]))
            return 10.0 ** (log_left + fraction * (log_right - log_left))
    return None


def _p1db_nearest_point(
    points: list[dict[str, object]], p1db_current_a: float | None
) -> dict[str, object] | None:
    if p1db_current_a is None:
        return None
    valid = [point for point in points if point["status"] == "VALID_SOLVED"]
    return min(
        valid,
        key=lambda point: abs(
            math.log(float(point["signal_current_a"]) / p1db_current_a)
        ),
    )


def _solve_compression(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, float]],
    dict[str, np.ndarray],
    dict[str, object],
    list[dict[str, object]],
]:
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
        if not pump_converged:
            final = pump_reports[-1]
            raise RuntimeError(
                "pump continuation failed before the multitone solve: "
                f"source_scale={final.source_scale}, "
                f"coeff_rel={final.coeff_rel:.6g}, "
                f"reason={final.failure_reason}"
            )
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
    pump_only_problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(pump_source),
        preconditioner=selected_preconditioner,
    )
    pump_only_state, pump_only_report = HarmonicNewtonKrylovSolver(settings).solve_one(
        pump_only_problem,
        pump_seed,
        1.0,
    )
    if not pump_only_report.converged:
        pump_only_state, pump_only_reports, pump_only_trace = (
            HarmonicNewtonKrylovSolver(settings).solve_adaptive_continuation(
                pump_only_problem,
                None,
                initial_step=0.25,
                min_step=0.01,
                growth=1.5,
                shrink=0.5,
                fallback_fixed_steps=20,
                max_wall_s=args.signal_continuation_deadline_s,
            )
        )
        if (
            not pump_only_reports
            or not pump_only_reports[-1].converged
            or not pump_only_trace.accepted_lambdas
            or pump_only_trace.accepted_lambdas[-1] < 1.0 - 1e-12
        ):
            reason = (
                pump_only_reports[-1].failure_reason
                if pump_only_reports
                else pump_only_trace.failure_reason
            )
            raise RuntimeError(
                "pump-on signal-off adaptive reference did not converge: "
                f"{reason}"
            )
    pump_reference_s21 = tone_s21(
        pump_only_state,
        basis,
        circuit,
        signal_tone=basis.pump_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=pump_current,
    )
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
            signal_s21 = tone_s21(
                state, basis, circuit, signal_tone=basis.signal_tone,
                source_port=source_port, out_port=out_port, source_current_a=float(current),
            )
            pump_s21 = tone_s21(
                state,
                basis,
                circuit,
                signal_tone=basis.pump_tone,
                source_port=source_port,
                out_port=out_port,
                source_current_a=pump_current,
            )
            idler_s21 = tone_s21(
                state,
                basis,
                circuit,
                signal_tone=basis.idler_tone,
                source_port=source_port,
                out_port=out_port,
                source_current_a=float(current),
            )
            gain_db = float(20.0 * np.log10(max(abs(signal_s21), 1e-300)))
            pump_depletion_db = float(
                20.0 * np.log10(max(abs(pump_s21), 1e-300))
                - 20.0 * np.log10(max(abs(pump_reference_s21), 1e-300))
            )
            reference_gain = gain_db if reference_gain is None else reference_gain
            previous_previous = previous
            previous_previous_current = previous_current
            previous = state
            previous_current = float(current)
        else:
            gain_db = float("nan")
            signal_s21 = pump_s21 = idler_s21 = complex(float("nan"), float("nan"))
            pump_depletion_db = float("nan")
        points.append({
            "signal_current_a": float(current),
            "signal_power_dbm": _current_to_dbm(
                float(current), args.z0_ohm, attenuation_db
            ),
            "gain_db": gain_db,
            "gain_vs_off_db": gain_db - pump_off_gain_db,
            "pump_depletion_db": pump_depletion_db,
            "signal_s21_real": float(np.real(signal_s21)),
            "signal_s21_imag": float(np.imag(signal_s21)),
            "pump_s21_real": float(np.real(pump_s21)),
            "pump_s21_imag": float(np.imag(pump_s21)),
            "idler_s21_real": float(np.real(idler_s21)),
            "idler_s21_imag": float(np.imag(idler_s21)),
            "compression_db": float(reference_gain - gain_db) if reference_gain is not None and np.isfinite(gain_db) else float("nan"),
            "status": solved.status,
            "recovery_rung": solved.used_recovery,
            "last_converged_signal_current_a": solved.last_converged_signal_current_a,
        })
        if index == len(currents) - 1:
            states["last"] = state
        if solved.status == "VALID_SOLVED":
            if index == 0:
                states["zero_signal"] = state
            if index == len(currents) // 2:
                states["mid"] = state
            if (
                "p1db" not in states
                and np.isfinite(points[-1]["compression_db"])
                and points[-1]["compression_db"] >= 1.0
            ):
                states["p1db"] = state
    small_signal_gain_db = float(points[0]["gain_db"]) if points else float("nan")
    no_gain = not np.isfinite(small_signal_gain_db) or small_signal_gain_db < 3.0
    if no_gain:
        for point in points:
            point["compression_db"] = float("nan")
    p1db_current_a = None if no_gain else _interpolate_p1db_current(points)
    p1db_dbm = (
        _current_to_dbm(p1db_current_a, args.z0_ohm, attenuation_db)
        if p1db_current_a is not None
        else None
    )
    p1db_point = _p1db_nearest_point(points, p1db_current_a)
    p1db_output_dbm = (
        p1db_dbm + small_signal_gain_db - 1.0
        if p1db_dbm is not None
        else None
    )
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
        "pump_reference_s21_real": float(np.real(pump_reference_s21)),
        "pump_reference_s21_imag": float(np.imag(pump_reference_s21)),
        "p1db": p1db_dbm,
        "p1db_signal_current_a": p1db_current_a,
        "p1db_input_dbm": p1db_dbm,
        "p1db_output_dbm": p1db_output_dbm,
        "p1db_pump_depletion_db": (
            float(p1db_point["pump_depletion_db"])
            if p1db_point is not None
            else None
        ),
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
    spatial_rows: list[dict[str, object]] = []
    if args.spatial_profiles:
        for label in ("zero_signal", "mid", "p1db"):
            if label not in states:
                continue
            for row in spatial_profiles(states[label], basis, circuit):
                spatial_rows.append({"operating_point": label, **row})
    return points, states, summary, spatial_rows


def _write_one_result(args: argparse.Namespace) -> dict[str, object]:
    points, states, summary, spatial_rows = _solve_compression(args)
    summary.update({
        "multitone_basis": args.multitone_basis,
        "multitone_sidebands": args.multitone_sidebands,
        "n_signal_power": args.n_signal_power,
        "resource_budget_gb": args.resource_budget_gb,
    })
    write_compression_outputs(
        args.output_dir,
        points,
        summary=summary,
        states=states,
        save_states=args.save_states,
        spatial_rows=spatial_rows if args.spatial_profiles else None,
    )
    return summary


def _frequency_worker(payload: tuple[dict[str, object], float, str]) -> dict[str, object]:
    values, frequency_ghz, output_dir = payload
    args = argparse.Namespace(**values)
    args.signal_ghz = frequency_ghz
    args.output_dir = Path(output_dir)
    args.summary_json = None
    return _write_one_result(args)


def _run_frequency_sweep(args: argparse.Namespace) -> dict[str, object]:
    if args.signal_ghz_min is None or args.signal_ghz_max is None:
        raise ValueError(
            "--n-signal-freq > 1 requires --signal-ghz-min and --signal-ghz-max"
        )
    frequencies = np.linspace(
        args.signal_ghz_min,
        args.signal_ghz_max,
        args.n_signal_freq,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    values = vars(args).copy()
    for index, frequency in enumerate(frequencies):
        subdir = args.output_dir / f"frequency_{index:03d}_{frequency:.6f}ghz"
        payloads.append((values, float(frequency), str(subdir)))
    workers = max(1, min(int(args.signal_workers), len(payloads)))
    if workers == 1:
        summaries = [_frequency_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            summaries = list(executor.map(_frequency_worker, payloads))
    rows = [
        {
            "signal_ghz": summary["signal_ghz"],
            "status": summary["status"],
            "small_signal_gain_db": summary["small_signal_gain_db"],
            "small_signal_gain_vs_off_db": summary["small_signal_gain_vs_off_db"],
            "p1db_input_dbm": summary["p1db_input_dbm"],
            "p1db_output_dbm": summary["p1db_output_dbm"],
            "p1db_pump_depletion_db": summary["p1db_pump_depletion_db"],
        }
        for summary in summaries
    ]
    with (args.output_dir / "p1db_vs_frequency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt

        figure, axis_p1db = plt.subplots(figsize=(7.0, 4.2))
        x = [float(row["signal_ghz"]) for row in rows]
        p1db = [float(row["p1db_input_dbm"]) for row in rows]
        gain = [float(row["small_signal_gain_vs_off_db"]) for row in rows]
        axis_p1db.plot(x, p1db, "o-", color="tab:blue", label="P1dB input")
        axis_p1db.set_xlabel("Signal frequency (GHz)")
        axis_p1db.set_ylabel("P1dB input (dBm)", color="tab:blue")
        axis_gain = axis_p1db.twinx()
        axis_gain.plot(x, gain, "s--", color="tab:red", label="Small-signal gain")
        axis_gain.set_ylabel("Gain vs pump-off (dB)", color="tab:red")
        axis_p1db.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.output_dir / "p1db_vs_frequency.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    summary = {
        "status": (
            "VALID_SOLVED"
            if all(row["status"] == "VALID_SOLVED" for row in rows)
            else "CHECK"
        ),
        "stability_status": "NOT_CHECKED",
        "n_signal_freq": len(rows),
        "signal_workers": workers,
        "frequencies_ghz": [float(value) for value in frequencies],
    }
    (args.output_dir / "frequency_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_signal_freq == 1 and args.signal_ghz is None:
        parser.error("--signal-ghz is required for a single-frequency run")
    if args.n_signal_freq > 1:
        summary = _run_frequency_sweep(args)
        if args.summary_json:
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0
    summary = _write_one_result(args)
    summary.update({
        "signal_frequency_range_ghz": [args.signal_ghz, args.signal_ghz],
        "n_signal_freq": 1,
        "signal_workers": 1,
    })
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "compression_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
