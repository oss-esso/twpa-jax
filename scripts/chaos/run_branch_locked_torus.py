"""Continue a finite autonomous torus at fixed, re-solved pump drives."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_torus_branch as branch_runner  # noqa: E402
from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.multitone.basis import build_autonomous_torus_basis  # noqa: E402
from twpa_solver.multitone.problem import FullMultiToneProblem  # noqa: E402
from twpa_solver.multitone.seed import promote_pump_solution  # noqa: E402
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive  # noqa: E402
from twpa_solver.multitone.torus import TorusProblem  # noqa: E402
from twpa_solver.multitone.torus_signal import (  # noqa: E402
    solve_torus_signal_response,
    torus_junction_utilization,
)
from twpa_solver.signal.io import load_pump  # noqa: E402
from twpa_solver.signal.stability import audit_loss_convention  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-drive continuation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument(
        "--pump-solution-dirs", type=Path, nargs="+", required=True
    )
    parser.add_argument("--drive-dbms", type=float, nargs="+", required=True)
    parser.add_argument("--initial-drive-dbm", type=float, required=True)
    parser.add_argument("--initial-state-npz", type=Path, required=True)
    parser.add_argument(
        "--prior-drive-dbm",
        type=float,
        default=None,
        help="Optional accepted drive immediately before --initial-drive-dbm.",
    )
    parser.add_argument(
        "--prior-state-npz",
        type=Path,
        default=None,
        help="Optional state checkpoint matching --prior-drive-dbm.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--sideband-harmonics", type=int, default=None)
    parser.add_argument("--factor-backend", default="pardiso")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--branch-collapse-fraction", type=float, default=0.25)
    parser.add_argument("--max-newton", type=int, default=60)
    parser.add_argument("--residual-tol", type=float, default=1.0e-9)
    parser.add_argument("--signal-ghz", type=float, default=7.4)
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--omitted-q-max", type=int, default=3)
    parser.add_argument(
        "--min-drive-step-db",
        type=float,
        default=0.01,
        help="Minimum available drive step before adaptive continuation stops.",
    )
    parser.add_argument(
        "--drive-growth",
        type=float,
        default=1.5,
        help="Growth factor for the requested continuation step after success.",
    )
    parser.add_argument(
        "--max-runtime-s",
        type=float,
        default=7200.0,
        help="Hard wall-clock limit for the complete march.",
    )
    parser.add_argument(
        "--omitted-q-quotable-limit",
        type=float,
        default=0.10,
        help="Flag accepted Q=1 rows above this omitted-sector residual.",
    )
    parser.add_argument(
        "--stop-r-j",
        type=float,
        default=0.95,
        help="Stop after recording an accepted row at or above this r_J.",
    )
    return parser.parse_args(argv)


def _pump_current(pump: Any) -> float:
    """Read the achieved pump current from a pump solution."""
    for key in ("achieved_on_chip_current_a", "pump_current_a", "current_a"):
        value = pump.metadata.get(key)
        if value is not None:
            return float(value)
    raise KeyError("pump metadata has no current field")


def _radius_squared(state: np.ndarray, basis: Any) -> float:
    """Return q=+/-1 power relative to the q=0 power."""
    plus = [i for i, tone in enumerate(basis.tones) if tone.q == 1]
    minus = [i for i, tone in enumerate(basis.tones) if tone.q == -1]
    zero = [i for i, tone in enumerate(basis.tones) if tone.q == 0]
    numerator = float(np.linalg.norm(state[plus]) ** 2)
    numerator += float(np.linalg.norm(state[minus]) ** 2)
    denominator = float(np.linalg.norm(state[zero]) ** 2)
    return numerator / max(denominator, 1.0e-300)


def _q_indices(basis: Any, nonzero: bool) -> list[int]:
    """Return the generator-sector rows selected by ``nonzero``."""
    return [
        index
        for index, tone in enumerate(basis.tones)
        if (tone.q != 0) is nonzero
    ]


def _fit_normal_form(
    accepted_points: list[tuple[float, float]],
) -> dict[str, Any]:
    """Fit ``r² = a * (P - Pc)`` to accepted fixed-drive points."""
    if len(accepted_points) < 2:
        return {
            "normal_form_slope": None,
            "normal_form_pc_dbm": None,
            "normal_form_r2_fit_r2": None,
        }
    drives = np.asarray([point[0] for point in accepted_points], dtype=float)
    radii = np.asarray([point[1] for point in accepted_points], dtype=float)
    design = np.column_stack((drives, np.ones(drives.size)))
    slope, intercept = np.linalg.lstsq(design, radii, rcond=None)[0]
    fitted = slope * drives + intercept
    residual = radii - fitted
    total = radii - float(np.mean(radii))
    fit_r2 = 1.0 - float(np.dot(residual, residual)) / max(
        float(np.dot(total, total)), 1.0e-300
    )
    pc_dbm = -intercept / slope if slope > 0.0 else None
    return {
        "normal_form_slope": float(slope),
        "normal_form_pc_dbm": None if pc_dbm is None else float(pc_dbm),
        "normal_form_r2_fit_r2": fit_r2,
    }


def _linear_sector_extrapolation(
    previous_state: np.ndarray,
    prior_state: np.ndarray | None,
    previous_drive: float,
    prior_drive: float | None,
    next_drive: float,
    basis: Any,
) -> np.ndarray:
    """Extrapolate the q=0 sector linearly in drive."""
    result = np.array(previous_state, dtype=np.complex128, copy=True)
    if prior_state is None or prior_drive is None:
        return result
    denominator = previous_drive - prior_drive
    if denominator == 0.0:
        raise ValueError("accepted drive spacing is zero; cannot form a secant")
    ratio = (next_drive - previous_drive) / denominator
    zero_rows = _q_indices(basis, nonzero=False)
    result[zero_rows] = previous_state[zero_rows] + ratio * (
        previous_state[zero_rows] - prior_state[zero_rows]
    )
    return result


def _normal_form_predictor(
    previous_state: np.ndarray,
    prior_state: np.ndarray | None,
    previous_drive: float,
    prior_drive: float | None,
    next_drive: float,
    basis: Any,
    accepted_points: list[tuple[float, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict a torus state by scaling its nonzero sectors with fitted r²."""
    fit = _fit_normal_form(accepted_points)
    slope = fit["normal_form_slope"]
    pc_dbm = fit["normal_form_pc_dbm"]
    if slope is None or pc_dbm is None or slope <= 0.0:
        raise ValueError("normal-form fit has no positive slope")
    previous_r2 = _radius_squared(previous_state, basis)
    predicted_r2 = float(slope * (next_drive - pc_dbm))
    if previous_r2 <= 0.0 or predicted_r2 <= 0.0:
        raise ValueError("normal-form prediction has a non-positive radius")
    result = _linear_sector_extrapolation(
        previous_state,
        prior_state,
        previous_drive,
        prior_drive,
        next_drive,
        basis,
    )
    ratio = float(np.sqrt(predicted_r2 / previous_r2))
    nonzero_rows = _q_indices(basis, nonzero=True)
    result[nonzero_rows] = previous_state[nonzero_rows] * ratio
    fit["normal_form_predicted_r2"] = predicted_r2
    fit["normal_form_predicted_radius_ratio"] = ratio
    return result, fit


def _drive_ratio(
    previous_drive: float,
    prior_drive: float,
    next_drive: float,
) -> float:
    """Return a secant extrapolation ratio for three accepted drives."""
    denominator = previous_drive - prior_drive
    if denominator == 0.0:
        raise ValueError("accepted drive spacing is zero; cannot form a secant")
    return (next_drive - previous_drive) / denominator


def _write_row(path: Path, row: dict[str, Any]) -> None:
    """Append one flushed CSV row with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = list(row)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _save_state(
    path: Path,
    state: np.ndarray,
    omega: float,
    sideband_harmonics: int,
) -> None:
    """Write a recoverable fixed-drive state checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            state=state,
            tangent=np.zeros(2 * state.size + 2),
            omega_a=float(omega),
            source_tau=1.0,
            sideband_harmonics=int(sideband_harmonics),
        )
    temporary.replace(path)


def _release_torus_resources(torus: TorusProblem) -> None:
    """Release native factorization memory before constructing the next point."""
    caches = list(torus._problem_caches.values())
    base_cache = getattr(torus.base_problem, "cache", None)
    if base_cache is not None:
        caches.append(base_cache)
    for cache in caches:
        for resource in list(cache.values()):
            release = getattr(resource, "release", None)
            if callable(release):
                release()


def _build_problem(
    circuit: Any,
    pump: Any,
    omega_a: float,
    args: argparse.Namespace,
) -> tuple[TorusProblem, FullMultiToneProblem]:
    """Build the full-node K=5 torus problem for one pump solution."""
    basis = build_autonomous_torus_basis(
        pump.omega_p,
        omega_a,
        tuple(int(mode) for mode in pump.modes),
        args.q_max,
        sideband_harmonics=args.sideband_harmonics,
    )
    drive = MultiToneDrive(
        basis.pump_tone,
        circuit.port_to_index[4],
        _pump_current(pump),
    ).to_coeffs(basis, circuit.node_count)
    full = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(drive),
        loss_model=args.loss_model,
    )
    torus = TorusProblem(
        full,
        tuple(int(mode) for mode in pump.modes),
        args.q_max,
        omega_a,
        sideband_harmonics=args.sideband_harmonics,
        factor_backend=args.factor_backend,
        precond_reuse=1,
    )
    return torus, full


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded branch-locked fixed-drive continuation."""
    if len(args.pump_solution_dirs) != len(args.drive_dbms):
        raise ValueError("pump directories and drive values must have equal length")
    if any(
        right <= left
        for left, right in zip(args.drive_dbms[:-1], args.drive_dbms[1:])
    ):
        raise ValueError("drive values must be strictly increasing")
    if args.min_drive_step_db <= 0.0:
        raise ValueError("--min-drive-step-db must be positive")
    if args.drive_growth <= 1.0:
        raise ValueError("--drive-growth must exceed one")
    circuit = load_circuit(args.circuit_dir)
    loss_audit = audit_loss_convention(circuit, args.loss_model)
    checkpoint = branch_runner._load_state_checkpoint(args.initial_state_npz)
    if args.sideband_harmonics is None:
        args.sideband_harmonics = int(checkpoint["sideband_harmonics"])
    previous_state = checkpoint["state"]
    previous_omega = float(checkpoint["omega_a"])
    previous_drive = float(args.initial_drive_dbm)
    if (args.prior_state_npz is None) != (args.prior_drive_dbm is None):
        raise ValueError(
            "--prior-state-npz and --prior-drive-dbm must be supplied together"
        )
    prior_checkpoint = (
        branch_runner._load_state_checkpoint(args.prior_state_npz)
        if args.prior_state_npz is not None
        else None
    )
    prior_state = None if prior_checkpoint is None else prior_checkpoint["state"]
    prior_omega = (
        None if prior_checkpoint is None else float(prior_checkpoint["omega_a"])
    )
    prior_drive = (
        None if args.prior_drive_dbm is None else float(args.prior_drive_dbm)
    )
    accepted_points: list[tuple[float, float]] = []
    rows: list[dict[str, Any]] = []
    attempt_index = 0
    target_index = 0
    drive_step_db = abs(args.drive_dbms[0] - previous_drive)
    started_march = time.perf_counter()
    stop_reason = "ladder exhausted"

    while target_index < len(args.drive_dbms):
        if time.perf_counter() - started_march >= args.max_runtime_s:
            stop_reason = "maximum march runtime reached"
            break
        index = target_index
        pump_dir = args.pump_solution_dirs[index]
        drive_dbm = float(args.drive_dbms[index])
        started = time.perf_counter()
        pump = load_pump(pump_dir, fallback_pump_freq_ghz=7.9)
        fit = _fit_normal_form(accepted_points)
        torus, full = _build_problem(circuit, pump, previous_omega, args)
        source_basis = build_autonomous_torus_basis(
            pump.omega_p,
            previous_omega,
            tuple(int(mode) for mode in pump.modes),
            args.q_max,
            sideband_harmonics=args.sideband_harmonics,
        )
        previous_state = branch_runner._remap_state_basis(
            previous_state, source_basis, torus.basis
        )
        if prior_state is not None and prior_omega is not None:
            prior_basis = build_autonomous_torus_basis(
                pump.omega_p,
                prior_omega,
                tuple(int(mode) for mode in pump.modes),
                args.q_max,
                sideband_harmonics=args.sideband_harmonics,
            )
            prior_state = branch_runner._remap_state_basis(
                prior_state, prior_basis, torus.basis
            )
        if not accepted_points:
            if prior_state is not None and prior_drive is not None:
                accepted_points.append(
                    (prior_drive, _radius_squared(prior_state, torus.basis))
                )
            accepted_points.append(
                (previous_drive, _radius_squared(previous_state, torus.basis))
            )
        pump_state = promote_pump_solution(pump.X, pump.basis, torus.basis)
        predictor_candidates: list[tuple[str, np.ndarray, float, dict[str, Any]]]
        predictor_candidates = []
        if len(accepted_points) >= 2:
            try:
                normal_state, normal_fit = _normal_form_predictor(
                    previous_state,
                    prior_state,
                    previous_drive,
                    prior_drive,
                    drive_dbm,
                    torus.basis,
                    accepted_points,
                )
                normal_omega = previous_omega
                if prior_drive is not None and prior_omega is not None:
                    normal_ratio = _drive_ratio(
                        previous_drive, prior_drive, drive_dbm
                    )
                    normal_omega = previous_omega + normal_ratio * (
                        previous_omega - prior_omega
                    )
                predictor_candidates.append(
                    ("normal_form", normal_state, normal_omega, normal_fit)
                )
                fit = normal_fit
            except ValueError as error:
                fit = {**fit, "normal_form_error": str(error)}
        if prior_state is not None and prior_omega is not None:
            try:
                secant_ratio = _drive_ratio(
                    previous_drive, prior_drive, drive_dbm
                )
                secant_state = previous_state + secant_ratio * (
                    previous_state - prior_state
                )
                secant_omega = previous_omega + secant_ratio * (
                    previous_omega - prior_omega
                )
                predictor_candidates.append(
                    ("secant", secant_state, secant_omega, fit)
                )
            except ValueError as error:
                fit = {**fit, "secant_error": str(error)}
        predictor_candidates.append(("warm", previous_state, previous_omega, fit))

        accepted = False
        for route, candidate, candidate_omega, candidate_fit in predictor_candidates:
            attempt_started = time.perf_counter()
            state, omega, report = torus.solve_newton_branch_locked(
                candidate,
                predictor_X=candidate,
                omega_a0=candidate_omega,
                beta=args.beta,
                branch_collapse_fraction=args.branch_collapse_fraction,
                max_newton=args.max_newton,
                residual_tol=args.residual_tol,
            )
            geometry = torus.branch_lock_geometry(candidate, beta=args.beta)
            row: dict[str, Any] = {
                "attempt_index": attempt_index,
                "point_index": index,
                "drive_dbm": drive_dbm,
                "pump_solution_dir": str(pump_dir),
                "pump_current_a": _pump_current(pump),
                "accepted": bool(report.get("converged")),
                "predictor_route": route,
                "drive_step_db": drive_step_db,
                "normal_form_slope": candidate_fit.get("normal_form_slope"),
                "normal_form_pc_dbm": candidate_fit.get("normal_form_pc_dbm"),
                "normal_form_r2_fit_r2": candidate_fit.get(
                    "normal_form_r2_fit_r2"
                ),
                "normal_form_predicted_r2": candidate_fit.get(
                    "normal_form_predicted_r2"
                ),
                "normal_form_predicted_radius_ratio": candidate_fit.get(
                    "normal_form_predicted_radius_ratio"
                ),
                "loss_audit": json.dumps(loss_audit, default=str),
                "solver_runtime_s": time.perf_counter() - attempt_started,
                "march_runtime_s": time.perf_counter() - started_march,
                "lock_phase_projection": geometry.phase_projection,
                "lock_radial_projection": geometry.radial_projection,
                "lock_pump_value": geometry.value(pump_state),
                "omega_a_over_omega_p": float(omega / pump.omega_p),
                "omega_a_ghz": float(omega / (2.0 * np.pi * 1.0e9)),
                **report,
            }
            if "normal_form_error" in fit:
                row["normal_form_error"] = fit["normal_form_error"]
            if "secant_error" in fit:
                row["secant_error"] = fit["secant_error"]
            attempt_index += 1
            if report.get("converged"):
                response_started = time.perf_counter()
                full_at_omega = torus.full_problem(omega)
                radius_squared = _radius_squared(state, torus.basis)
                r_j = torus_junction_utilization(full_at_omega, state)
                row["radius_squared"] = radius_squared
                row["r_j"] = r_j
                if args.omitted_q_max > args.q_max:
                    row.update(torus.omitted_q_residual(state, args.omitted_q_max))
                omitted_rel = float(row.get("omitted_q_residual_rel", 0.0))
                row["omitted_q_quotable"] = omitted_rel < args.omitted_q_quotable_limit
                row["omitted_q_flag"] = not row["omitted_q_quotable"]
                response = solve_torus_signal_response(
                    full_at_omega,
                    state,
                    signal_ghz=args.signal_ghz,
                    source_port=args.source_port,
                    out_port=args.out_port,
                    loss_model=args.loss_model,
                )
                row["gain_vs_off_db"] = response.gain_vs_off_db
                row["signal_residual_rel"] = response.residual_rel
                row["signal_runtime_s"] = time.perf_counter() - response_started
                row["solver_runtime_s"] = time.perf_counter() - attempt_started
                state_path = args.state_dir / f"point_{len(accepted_points):03d}.npz"
                sideband_harmonics = (
                    args.sideband_harmonics
                    if args.sideband_harmonics is not None
                    else max(abs(int(mode)) for mode in pump.modes)
                )
                _save_state(state_path, state, omega, sideband_harmonics)
                row["state_checkpoint"] = str(state_path)
                _write_row(args.out, row)
                rows.append(row)
                print(json.dumps(row, default=str), flush=True)
                accepted = True
                prior_state = previous_state
                prior_omega = previous_omega
                prior_drive = previous_drive
                previous_state = state
                previous_omega = float(omega)
                previous_drive = drive_dbm
                accepted_points.append((drive_dbm, radius_squared))
                target_index += 1
                drive_step_db = min(
                    max(drive_step_db * args.drive_growth, args.min_drive_step_db),
                    max(args.drive_dbms[-1] - previous_drive, args.min_drive_step_db),
                )
                if r_j >= args.stop_r_j:
                    stop_reason = f"junction utilization reached r_J >= {args.stop_r_j}"
                    target_index = len(args.drive_dbms)
                break
            row["failure_reason"] = report.get("failure_reason")
            _write_row(args.out, row)
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
        _release_torus_resources(torus)
        del torus, full, pump, pump_state
        gc.collect()
        if accepted:
            continue
        last_step = drive_step_db * 0.5
        if last_step < args.min_drive_step_db:
            stop_reason = (
                "adaptive drive step fell below "
                f"--min-drive-step-db ({args.min_drive_step_db})"
            )
            break
        available = [
            (candidate_index, candidate_drive)
            for candidate_index, candidate_drive in enumerate(args.drive_dbms)
            if candidate_index > (target_index - 1)
            and previous_drive < candidate_drive <= previous_drive + last_step
        ]
        if not available:
            stop_reason = (
                "corrector failed and no re-solved pump checkpoint exists "
                f"within the halved step {last_step:.6g} dB"
            )
            break
        target_index = min(available, key=lambda item: item[1])[0]
        drive_step_db = last_step

    summary = {
        "loss_audit": loss_audit,
        "rows": rows,
        "accepted_points": accepted_points,
        "stop_reason": stop_reason,
        "attempt_count": attempt_index,
        "accepted_count": sum(bool(row.get("accepted")) for row in rows),
    }
    print(json.dumps({"stop_reason": stop_reason}, default=str), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
