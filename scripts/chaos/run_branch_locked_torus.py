"""Continue a finite autonomous torus at fixed, re-solved pump drives."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--sideband-harmonics", type=int, default=None)
    parser.add_argument("--factor-backend", default="pardiso")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--branch-collapse-fraction", type=float, default=0.25)
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--residual-tol", type=float, default=1.0e-9)
    parser.add_argument("--signal-ghz", type=float, default=7.4)
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--omitted-q-max", type=int, default=3)
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
    circuit = load_circuit(args.circuit_dir)
    loss_audit = audit_loss_convention(circuit, args.loss_model)
    checkpoint = branch_runner._load_state_checkpoint(args.initial_state_npz)
    previous_state = checkpoint["state"]
    previous_omega = float(checkpoint["omega_a"])
    previous_drive = float(args.initial_drive_dbm)
    prior_state: np.ndarray | None = None
    prior_omega: float | None = None
    rows: list[dict[str, Any]] = []

    for index, (pump_dir, drive_dbm) in enumerate(
        zip(args.pump_solution_dirs, args.drive_dbms)
    ):
        started = time.perf_counter()
        pump = load_pump(pump_dir, fallback_pump_freq_ghz=7.9)
        predictor = previous_state
        predictor_omega = previous_omega
        predictor_candidates: list[tuple[str, np.ndarray, float]] = []
        if prior_state is not None and prior_omega is not None:
            ratio = (drive_dbm - previous_drive) / max(
                previous_drive - float(args.initial_drive_dbm), 1.0e-300
            )
            predictor = previous_state + ratio * (previous_state - prior_state)
            predictor_omega = previous_omega + ratio * (previous_omega - prior_omega)
            predictor_candidates.append(("secant", predictor, predictor_omega))
        predictor_candidates.append(("warm", previous_state, previous_omega))
        torus, full = _build_problem(circuit, pump, predictor_omega, args)
        source_basis = build_autonomous_torus_basis(
            pump.omega_p,
            checkpoint["omega_a"] if index == 0 else previous_omega,
            tuple(int(mode) for mode in pump.modes),
            args.q_max,
            sideband_harmonics=args.sideband_harmonics,
        )
        if predictor.shape != (torus.basis.n_tones, torus.base_problem.n):
            predictor = branch_runner._remap_state_basis(
                predictor, source_basis, torus.basis
            )
        pump_state = promote_pump_solution(pump.X, pump.basis, torus.basis)
        state = previous_state
        omega = previous_omega
        report: dict[str, Any] = {"converged": False}
        predictor_route = "none"
        failed_attempts: list[dict[str, Any]] = []
        for route, candidate, candidate_omega in predictor_candidates:
            if candidate.shape != (torus.basis.n_tones, torus.base_problem.n):
                candidate = branch_runner._remap_state_basis(
                    candidate, source_basis, torus.basis
                )
            state, omega, report = torus.solve_newton_branch_locked(
                candidate,
                predictor_X=candidate,
                omega_a0=candidate_omega,
                beta=args.beta,
                branch_collapse_fraction=args.branch_collapse_fraction,
                max_newton=args.max_newton,
                residual_tol=args.residual_tol,
            )
            if report.get("converged"):
                predictor = candidate
                predictor_omega = candidate_omega
                predictor_route = route
                break
            failed_attempts.append(
                {
                    "route": route,
                    "failure_reason": report.get("failure_reason"),
                    "coefficient_relative": report.get("coefficient_relative"),
                }
            )
            predictor_route = route
        geometry = torus.branch_lock_geometry(predictor, beta=args.beta)
        lock_pump = geometry.value(pump_state)
        row: dict[str, Any] = {
            "point_index": index,
            "drive_dbm": float(drive_dbm),
            "pump_solution_dir": str(pump_dir),
            "pump_current_a": _pump_current(pump),
            "omega_a_over_omega_p": float(omega / pump.omega_p),
            "omega_a_ghz": float(omega / (2.0 * np.pi * 1.0e9)),
            "lock_phase_projection": geometry.phase_projection,
            "lock_radial_projection": geometry.radial_projection,
            "lock_pump_value": lock_pump,
            "predictor_route": predictor_route,
            "failed_predictor_attempts": json.dumps(failed_attempts),
            "loss_audit": json.dumps(loss_audit, default=str),
            "solver_runtime_s": time.perf_counter() - started,
            **report,
        }
        if report.get("converged"):
            response_started = time.perf_counter()
            full_at_omega = torus.full_problem(omega)
            row["radius_squared"] = _radius_squared(state, torus.basis)
            row["r_j"] = torus_junction_utilization(full_at_omega, state)
            if args.omitted_q_max > args.q_max:
                row.update(torus.omitted_q_residual(state, args.omitted_q_max))
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
            state_path = args.state_dir / f"point_{index + 1:03d}.npz"
            sideband_harmonics = (
                args.sideband_harmonics
                if args.sideband_harmonics is not None
                else max(abs(int(mode)) for mode in pump.modes)
            )
            _save_state(state_path, state, omega, sideband_harmonics)
            row["state_checkpoint"] = str(state_path)
            prior_state = previous_state
            prior_omega = previous_omega
            previous_state = state
            previous_omega = float(omega)
            previous_drive = float(drive_dbm)
        _write_row(args.out, row)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
        if not report.get("converged"):
            break
    return {"loss_audit": loss_audit, "rows": rows}


if __name__ == "__main__":
    run(parse_args())
