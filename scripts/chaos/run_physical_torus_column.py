"""Run a physical pump and Q=1 torus column without pump artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "chaos"))

import track_critical_root as tracker  # noqa: E402
from twpa_solver.multitone.basis import build_autonomous_torus_basis  # noqa: E402
from twpa_solver.multitone.problem import FullMultiToneProblem  # noqa: E402
from twpa_solver.multitone.schur import build_multitone_schur_problem  # noqa: E402
from twpa_solver.multitone.seed import (  # noqa: E402
    promote_pump_solution,
    seed_torus_from_floquet,
)
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive  # noqa: E402
from twpa_solver.multitone.torus import TorusProblem  # noqa: E402
from twpa_solver.signal import (  # noqa: E402
    audit_loss_convention,
    classify_floquet_resonance,
    local_minima,
    refine_complex_resonance,
    sideband_list,
    sweep_sigma_min,
)
from twpa_solver.signal.branch_tracking import stability_verdict  # noqa: E402


@dataclass
class Branch:
    """One candidate torus branch and its latest converged state."""

    candidate_index: int
    signal_ghz: complex
    mode_vector: np.ndarray
    sidebands: list[int]
    growth_rate: float
    multiplier: complex
    state: np.ndarray | None = None
    omega_a: float | None = None
    torus_started: bool = False
    ns_crossed: bool = False
    mode_overlap: float | None = None
    discontinuity: bool = False
    stability_verdict: str = "UNDECIDED"
    floquet_kind: str = ""
    multiplier_phase_rad: float | None = None
    floquet_converged: bool = True
    floquet_iterations: int = 0
    floquet_residual: float | None = None


def _pump_dc_flux(pump: Any, circuit: Any) -> np.ndarray | None:
    """Read DC-flux metadata from the freshly solved pump, if available."""
    value = pump.metadata.get("dc_branch_flux", pump.metadata.get("dc_branch_flux_wb"))
    if value is None:
        return None
    flux = np.asarray(value, dtype=float).reshape(-1)
    if flux.size == 1:
        flux = np.full(circuit.branch_count, float(flux[0]))
    if flux.size != circuit.branch_count:
        raise ValueError("fresh pump DC flux does not match circuit branches")
    return flux


def _parse_dbms(value: str) -> list[float]:
    """Parse a strictly increasing comma-separated source-power ladder."""
    values = [float(token.strip()) for token in value.split(",")]
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("--drive-dbms must contain finite values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("--drive-dbms must be strictly increasing")
    return values


def _parse_candidate_signals(value: str) -> list[float]:
    """Parse optional lowest-drive Floquet frequency seeds."""
    if not value.strip():
        return []
    values = [float(token.strip()) for token in value.split(",")]
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError("--candidate-signal-ghz must contain positive finite values")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the physical-column command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument(
        "--pump-template-dir",
        type=Path,
        required=True,
        help="Metadata template only; its pump waveform is never used.",
    )
    parser.add_argument("--drive-dbms", required=True)
    parser.add_argument("--omega-a-ratio", type=float, required=True)
    parser.add_argument("--sidebands", type=int, required=True)
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--floquet-only",
        action="store_true",
        help="Track Floquet branches without invoking the torus corrector.",
    )
    parser.add_argument("--pump-port", type=int, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--scan-min-ghz", type=float, default=0.4)
    parser.add_argument("--scan-max-ghz", type=float, default=1.4)
    parser.add_argument("--scan-points", type=int, default=21)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument(
        "--candidate-signal-ghz",
        default="",
        help="Optional lowest-drive Floquet frequency seeds, in GHz.",
    )
    parser.add_argument(
        "--untracked-scan-every",
        type=int,
        default=0,
        help="Run a diagnostic untracked candidate scan every N points.",
    )
    parser.add_argument("--untracked-scan-points", type=int, default=21)
    parser.add_argument("--gamma-nt", type=int, default=4096)
    parser.add_argument("--schur", action="store_true")
    parser.add_argument(
        "--factor-backend",
        choices=("pardiso", "banded", "superlu"),
        default="pardiso",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode-checkpoint-dir",
        type=Path,
        default=None,
        help="Write the current Floquet mode vectors as per-point NPZ files.",
    )
    parser.add_argument(
        "--repro-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Write each newly solved period-1 pump as a reusable checkpoint "
            "for the torus branch runner."
        ),
    )
    parser.add_argument("--pump-initial-step", type=float, default=0.25)
    parser.add_argument("--pump-min-step", type=float, default=1.0e-3)
    parser.add_argument("--pump-max-newton", type=int, default=32)
    parser.add_argument("--pump-solve-deadline-s", type=float, default=0.0)
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--residual-tol", type=float, default=1.0e-9)
    parser.add_argument("--branch-step", type=float, default=1.0e-2)
    parser.add_argument("--gmres-rtol", type=float, default=1.0e-8)
    parser.add_argument("--gmres-maxiter", type=int, default=240)
    parser.add_argument("--gmres-restart", type=int, default=80)
    parser.add_argument(
        "--linear-debug",
        action="store_true",
        help="Record augmented JVP, border, and GMRES diagnostics.",
    )
    parser.add_argument("--linear-debug-fd-step", type=float, default=1.0e-6)
    parser.add_argument("--node-ref", type=int, default=0)
    return parser.parse_args(argv)


def _build_torus(
    circuit: Any,
    pump: Any,
    omega_a: float,
    args: argparse.Namespace,
) -> tuple[TorusProblem, Any, np.ndarray]:
    """Build a torus problem from the currently converged pump only."""
    basis = build_autonomous_torus_basis(
        pump.omega_p,
        omega_a,
        tuple(int(mode) for mode in pump.modes),
        args.q_max,
        sideband_harmonics=args.k,
    )
    drive = MultiToneDrive(
        basis.pump_tone,
        circuit.port_to_index[args.pump_port],
        float(pump.metadata["pump_current_a"]),
    ).to_coeffs(basis, circuit.node_count)
    full = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(drive),
        loss_model=args.loss_model,
    )
    retained_index: int | None = None
    base: Any = full
    if args.schur:
        base = build_multitone_schur_problem(
            full,
            list(circuit.port_to_index.values()),
            preconditioner="real_coupled_fast",
        )
        retained_index = int(base.partition.retained_pos[args.node_ref])
        if retained_index < 0:
            raise ValueError("node-ref is eliminated by the Schur partition")
        node_ref = retained_index
    else:
        node_ref = args.node_ref
    torus = TorusProblem(
        base,
        tuple(int(mode) for mode in pump.modes),
        args.q_max,
        omega_a,
        node_ref=node_ref,
        sideband_harmonics=args.k,
        factor_backend=args.factor_backend,
        precond_reuse=1,
    )
    pump_state = promote_pump_solution(pump.X, pump.basis, basis)
    if args.schur:
        pump_state = pump_state[:, base.partition.retained]
    metadata = {
        "anchor_full_node": args.node_ref,
        "anchor_retained_node": retained_index,
    }
    return torus, metadata, pump_state


def _enumerate_candidates(
    circuit: Any,
    pump: Any,
    args: argparse.Namespace,
) -> list[Branch]:
    """Enumerate and refine Hill candidates at the first solved drive only."""
    modes = sideband_list(args.sidebands)
    dc_flux = _pump_dc_flux(pump, circuit)
    khat, khat_base = tracker._build_hill_operator(
        circuit, pump, modes, args.gamma_nt, dc_flux
    )
    grid = np.linspace(args.scan_min_ghz, args.scan_max_ghz, args.scan_points)
    estimates = sweep_sigma_min(
        circuit=circuit,
        khat=khat,
        khat_base=khat_base,
        omega_p=pump.omega_p,
        signal_ghz_grid=[float(value) for value in grid],
        ms=modes,
        loss_model=args.loss_model,
        iters=8,
    )
    sigma = [float(item.sigma_min) for item in estimates]
    branches: list[Branch] = []
    guesses = _parse_candidate_signals(args.candidate_signal_ghz)
    for grid_index in local_minima(
        sigma, k=max(0, args.max_candidates - len(guesses))
    ):
        guesses.append(float(grid[grid_index]))
    for signal_guess in guesses[:args.max_candidates]:
        resonance = refine_complex_resonance(
            circuit=circuit,
            khat=khat,
            khat_base=khat_base,
            omega_p=pump.omega_p,
            ms=modes,
            signal_ghz_guess=signal_guess,
            loss_model=args.loss_model,
        )
        if not resonance.converged or resonance.mode_vector is None:
            continue
        classification = classify_floquet_resonance(resonance, pump.omega_p)
        branches.append(
            Branch(
                candidate_index=len(branches),
                signal_ghz=complex(resonance.signal_ghz),
                mode_vector=np.asarray(resonance.mode_vector),
                sidebands=modes,
                growth_rate=float(resonance.growth_rate_per_s),
                multiplier=complex(classification.multiplier),
                stability_verdict=stability_verdict(
                    classification, resonance.converged
                ),
                floquet_kind=classification.kind,
                multiplier_phase_rad=float(classification.phase_rad),
                floquet_converged=bool(resonance.converged),
                floquet_iterations=int(resonance.iterations),
                floquet_residual=(
                    None
                    if resonance.residual is None
                    else float(resonance.residual)
                ),
            )
        )
    return branches


def _solve_branch(
    branch: Branch,
    circuit: Any,
    pump: Any,
    args: argparse.Namespace,
    first: bool,
) -> dict[str, Any]:
    """Branch-switch once, then warm-start fixed-drive torus Newton."""
    omega_a = (
        2.0 * math.pi * branch.signal_ghz.real * 1.0e9
        if first
        else float(branch.omega_a)
    )
    torus, metadata, pump_state = _build_torus(circuit, pump, omega_a, args)
    if first:
        seeded = seed_torus_from_floquet(
            pump.X,
            pump.basis,
            torus.basis,
            branch.mode_vector,
            branch.sidebands,
            omega_p=pump.omega_p,
            omega_a=omega_a,
            perturbation_amplitude=1.0,
            node_ref=metadata["anchor_full_node"],
        )
        state = seeded
        if args.schur:
            state = state[:, torus.base_problem.partition.retained]
        state, omega, tau, report, _tangent = (
            torus.solve_torus_branch_switch(
                pump_state,
                omega_a_ns=omega_a,
                source_tau_ns=1.0,
                perturbation=state - pump_state,
                step_size=args.branch_step,
                max_newton=args.max_newton,
                residual_tol=args.residual_tol,
                gmres_rtol=args.gmres_rtol,
                gmres_maxiter=args.gmres_maxiter,
                gmres_restart=args.gmres_restart,
                linear_debug=args.linear_debug,
                linear_debug_fd_step=args.linear_debug_fd_step,
            )
        )
        route = "floquet_branch_switch"
    else:
        if branch.state is None or branch.omega_a is None:
            raise ValueError("warm branch state is missing")
        state, omega, report = torus.solve_newton(
            branch.state,
            omega_a0=branch.omega_a,
            max_newton=args.max_newton,
            residual_tol=args.residual_tol,
        )
        tau = 1.0
        route = "warm_fixed_drive_newton"
    generator_norm = torus.generator_norm(state)
    pump_norm = max(float(np.linalg.norm(pump_state)), 1.0e-300)
    converged = bool(report.get("converged") and generator_norm > 1.0e-8 * pump_norm)
    if converged:
        branch.state = state
        branch.omega_a = float(omega)
        branch.torus_started = True
    return {
        "candidate_index": branch.candidate_index,
        "candidate_signal_real_ghz": branch.signal_ghz.real,
        "candidate_signal_imag_ghz": branch.signal_ghz.imag,
        "growth_rate_per_s": branch.growth_rate,
        "multiplier_magnitude": abs(branch.multiplier),
        "multiplier_phase_rad": branch.multiplier_phase_rad,
        "floquet_kind": branch.floquet_kind,
        "mode_overlap": branch.mode_overlap,
        "discontinuity": branch.discontinuity,
        "stability_verdict": branch.stability_verdict,
        "floquet_converged": branch.floquet_converged,
        "floquet_iterations": branch.floquet_iterations,
        "floquet_residual": branch.floquet_residual,
        "route": route,
        "converged": converged,
        "solver_converged": bool(report.get("converged")),
        "generator_norm_relative": generator_norm / pump_norm,
        "omega_a_over_omega_p": float(omega / pump.omega_p),
        "source_tau": float(tau),
        "iterations": int(report.get("iterations", 0)),
        "residual_norm": report.get("residual_norm"),
        "failure_reason": report.get("failure_reason"),
    }


def _mode_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return normalized complex-vector overlap."""
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(abs(np.vdot(left.reshape(-1), right.reshape(-1))) /
                 (left_norm * right_norm))


def _best_untracked_candidate(
    circuit: Any,
    pump: Any,
    branches: list[Branch],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Probe for a stronger untracked Hill root without changing branches."""
    modes = sideband_list(args.sidebands)
    khat, khat_base = tracker._build_hill_operator(
        circuit,
        pump,
        modes,
        args.gamma_nt,
        _pump_dc_flux(pump, circuit),
    )
    grid = np.linspace(args.scan_min_ghz, args.scan_max_ghz,
                       args.untracked_scan_points)
    estimates = sweep_sigma_min(
        circuit=circuit,
        khat=khat,
        khat_base=khat_base,
        omega_p=pump.omega_p,
        signal_ghz_grid=[float(value) for value in grid],
        ms=modes,
        loss_model=args.loss_model,
        iters=6,
    )
    sigma = [float(item.sigma_min) for item in estimates]
    probe_limit = max(2, 2 * args.max_candidates)
    candidates: list[dict[str, Any]] = []
    for grid_index in local_minima(sigma, k=probe_limit):
        resonance = refine_complex_resonance(
            circuit=circuit,
            khat=khat,
            khat_base=khat_base,
            omega_p=pump.omega_p,
            ms=modes,
            signal_ghz_guess=float(grid[grid_index]),
            loss_model=args.loss_model,
            max_iters=12,
        )
        if not resonance.converged or resonance.mode_vector is None:
            continue
        matched = any(
            _mode_overlap(branch.mode_vector, resonance.mode_vector) >= 0.8
            for branch in branches
        )
        if not matched:
            candidates.append(
                {
                    "growth_rate_per_s": float(resonance.growth_rate_per_s),
                    "signal_ghz": float(resonance.signal_ghz.real),
                }
            )
    if not candidates:
        return {
            "best_untracked_growth_per_s": None,
            "best_untracked_signal_ghz": None,
        }
    best = max(candidates, key=lambda item: item["growth_rate_per_s"])
    return {
        "best_untracked_growth_per_s": best["growth_rate_per_s"],
        "best_untracked_signal_ghz": best["signal_ghz"],
    }


def _write_branch_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one flat row per branch and source-drive point."""
    flat: list[dict[str, Any]] = []
    for row in rows:
        for branch in row.get("branches", []):
            flat.append(
                {
                    "point_index": row["point_index"],
                    "drive_dbm": row["drive_dbm"],
                    **branch,
                }
            )
    if not flat:
        return
    fieldnames: list[str] = []
    for row in flat:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
        handle.flush()
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the current partial column atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _write_mode_checkpoint(
    directory: Path,
    point_index: int,
    drive_dbm: float,
    branches: list[Branch],
) -> None:
    """Write current mode vectors without buffering the continuation."""
    directory.mkdir(parents=True, exist_ok=True)
    valid = [branch for branch in branches if branch.mode_vector.size > 0]
    if not valid:
        return
    vectors = np.stack([branch.mode_vector for branch in valid])
    path = directory / f"modes_{point_index:03d}_{drive_dbm:.5f}dbm.npz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            point_index=np.asarray(point_index, dtype=np.int64),
            drive_dbm=np.asarray(drive_dbm, dtype=float),
            candidate_index=np.asarray(
                [branch.candidate_index for branch in valid], dtype=np.int64
            ),
            vectors=vectors,
        )
    temporary.replace(path)


def _write_period1_checkpoint(
    directory: Path,
    point_index: int,
    drive_dbm: float,
    pump_step: Any,
    args: argparse.Namespace,
    loss_audit: dict[str, object],
) -> Path:
    """Persist the converged period-1 state used by the Floquet calculation."""
    if pump_step.pump is None or pump_step.full_state is None:
        raise ValueError("cannot checkpoint a non-converged pump step")
    checkpoint = (
        directory / f"point_{point_index:03d}_{drive_dbm:.5f}dbm" / "pump"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint / "pump_solution.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X_real=np.asarray(pump_step.full_state.real, dtype=np.float64),
            X_imag=np.asarray(pump_step.full_state.imag, dtype=np.float64),
            pump_modes=np.asarray(pump_step.pump.modes, dtype=np.int64),
        )
    temporary.replace(checkpoint / "pump_solution.npz")
    metadata = dict(pump_step.pump.metadata)
    metadata.update(
        {
            "checkpoint_kind": "period1_reproducibility",
            "requested_drive_dbm": float(drive_dbm),
            "requested_source_current_a": float(pump_step.source_current_a),
            "achieved_on_chip_current_a": float(pump_step.achieved_current_a),
            "pump_modes": [int(mode) for mode in pump_step.pump.modes],
            "pump_mode_policy": pump_step.pump.basis.policy,
            "pump_source_mode": int(pump_step.pump.basis.source_mode),
            "omega_p": float(pump_step.pump.omega_p),
            "pump_freq_ghz": float(pump_step.pump.pump_freq_ghz),
            "nt": int(pump_step.pump.nt_original),
            "pump_port": args.pump_port,
            "loss_audit": loss_audit,
            "pump_iterations": int(pump_step.iterations),
            "pump_coeff_rel": float(pump_step.coeff_rel),
            "pump_time_rel": pump_step.time_rel,
        }
    )
    report = {
        "final_status": "VALID_CONVERGED",
        "metadata": metadata,
    }
    report_path = checkpoint / "pump_report.json"
    report_temporary = report_path.with_suffix(".json.tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    report_temporary.replace(report_path)
    return checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the integrated physical pump and torus column."""
    drives = _parse_dbms(args.drive_dbms)
    circuit = tracker.load_circuit(args.circuit_dir)
    loss_audit = audit_loss_convention(circuit, args.loss_model)
    template = tracker.load_pump(args.pump_template_dir, fallback_pump_freq_ghz=7.9)
    tracker_args = tracker.parse_args(
        [
            "--circuit-dir", str(args.circuit_dir),
            "--pump-dir", str(args.pump_template_dir),
            "--drive-dbms", str(drives[0]),
            "--sidebands", str(args.sidebands),
            "--initial-signal-ghz", str(args.omega_a_ratio * template.pump_freq_ghz),
            "--loss-model", args.loss_model,
            "--out-csv", str(args.out.with_suffix(".pump.csv")),
            "--pump-port", str(args.pump_port),
            "--pump-initial-step", str(args.pump_initial_step),
            "--pump-min-step", str(args.pump_min_step),
            "--pump-max-newton", str(args.pump_max_newton),
            "--pump-solve-deadline-s", str(args.pump_solve_deadline_s),
        ]
    )
    continuation = tracker.PumpContinuation(tracker_args, circuit, template)
    modes = sideband_list(args.sidebands)
    rows: list[dict[str, Any]] = []
    branches: list[Branch] = []
    warm_state: np.ndarray | None = None
    base_current: float | None = None
    for point_index, drive_dbm in enumerate(drives):
        pump_step = continuation.solve(drive_dbm, warm_state)
        base_row: dict[str, Any] = {
            "point_index": point_index,
            "drive_dbm": drive_dbm,
            "pump_converged": pump_step.converged,
            "pump_current_a": pump_step.source_current_a,
            "pump_iterations": pump_step.iterations,
            "pump_coeff_rel": pump_step.coeff_rel,
            "pump_time_rel": pump_step.time_rel,
            "pump_failure_reason": pump_step.failure_reason,
            "best_untracked_growth_per_s": None,
            "best_untracked_signal_ghz": None,
        }
        if not pump_step.converged or pump_step.pump is None:
            base_row["status"] = "pump_failure"
            rows.append(base_row)
            _write_rows(args.out, rows)
            break
        if args.repro_checkpoint_dir is not None:
            _write_period1_checkpoint(
                args.repro_checkpoint_dir,
                point_index,
                drive_dbm,
                pump_step,
                args,
                loss_audit,
            )
        if not branches:
            branches = _enumerate_candidates(circuit, pump_step.pump, args)
            if not branches:
                base_row["status"] = "no_floquet_candidates"
                rows.append(base_row)
                _write_rows(args.out, rows)
                break
            base_current = pump_step.source_current_a
        else:
            khat, khat_base = tracker._build_hill_operator(
                circuit,
                pump_step.pump,
                modes,
                args.gamma_nt,
                _pump_dc_flux(pump_step.pump, circuit),
            )
            parameter = pump_step.source_current_a / max(base_current or 1.0, 1e-300)
            for branch in branches:
                previous_growth = branch.growth_rate
                previous_multiplier = branch.multiplier
                point = tracker.track_floquet_point(
                    circuit=circuit,
                    khat=khat,
                    khat_base=khat_base,
                    parameter=parameter,
                    omega_p=pump_step.pump.omega_p,
                    ms=modes,
                    seed_signal_ghz=branch.signal_ghz,
                    seed_mode_vector=branch.mode_vector,
                    previous_multiplier=previous_multiplier,
                    loss_model=args.loss_model,
                    max_iters=30,
                    tol=1.0e-9,
                )
                if (
                    point.resonance.converged
                    and point.resonance.mode_vector is not None
                ):
                    branch.ns_crossed = branch.ns_crossed or (
                        previous_growth < 0.0
                        <= point.resonance.growth_rate_per_s
                    )
                    branch.signal_ghz = complex(point.resonance.signal_ghz)
                    branch.mode_vector = np.asarray(point.resonance.mode_vector)
                    branch.growth_rate = float(point.resonance.growth_rate_per_s)
                    branch.multiplier = complex(point.classification.multiplier)
                branch.mode_overlap = point.mode_overlap
                branch.discontinuity = point.discontinuity
                branch.stability_verdict = point.stability_verdict
                branch.floquet_kind = point.classification.kind
                branch.multiplier_phase_rad = float(
                    point.classification.phase_rad
                )
                branch.floquet_converged = bool(point.resonance.converged)
                branch.floquet_iterations = int(point.resonance.iterations)
                branch.floquet_residual = (
                    None
                    if point.resonance.residual is None
                    else float(point.resonance.residual)
                )
        branch_rows = []
        if args.mode_checkpoint_dir is not None:
            _write_mode_checkpoint(
                args.mode_checkpoint_dir, point_index, drive_dbm, branches
            )
        for branch in branches:
            crosses = branch.ns_crossed or (
                point_index == 0 and branch.growth_rate >= 0.0
            )
            if not args.floquet_only and not branch.torus_started and crosses:
                branch_rows.append(
                    _solve_branch(branch, circuit, pump_step.pump, args, first=True)
                )
            elif branch.torus_started:
                branch_rows.append(
                    _solve_branch(branch, circuit, pump_step.pump, args, first=False)
                )
            else:
                branch_rows.append(
                    {
                        "candidate_index": branch.candidate_index,
                        "candidate_signal_real_ghz": branch.signal_ghz.real,
                        "candidate_signal_imag_ghz": branch.signal_ghz.imag,
                        "growth_rate_per_s": branch.growth_rate,
                        "multiplier_magnitude": abs(branch.multiplier),
                        "multiplier_phase_rad": branch.multiplier_phase_rad,
                        "floquet_kind": branch.floquet_kind,
                        "mode_overlap": branch.mode_overlap,
                        "discontinuity": branch.discontinuity,
                        "stability_verdict": branch.stability_verdict,
                        "floquet_converged": branch.floquet_converged,
                        "floquet_iterations": branch.floquet_iterations,
                        "floquet_residual": branch.floquet_residual,
                        "route": "floquet_tracking_only",
                        "converged": branch.floquet_converged,
                        "solver_converged": branch.floquet_converged,
                        "generator_norm_relative": 0.0,
                        "omega_a_over_omega_p": None,
                        "source_tau": 1.0,
                        "iterations": 0,
                        "residual_norm": None,
                        "failure_reason": "below NS branch switch",
                    }
                )
        if (
            args.untracked_scan_every > 0
            and point_index % args.untracked_scan_every == 0
        ):
            base_row.update(
                _best_untracked_candidate(
                    circuit, pump_step.pump, branches, args
                )
            )
        base_row["status"] = "PASS"
        base_row["branches"] = branch_rows
        rows.append(base_row)
        _write_rows(args.out, rows)
        _write_branch_rows(
            args.out.with_name(args.out.stem + ".branches.csv"), rows
        )
        warm_state = pump_step.full_state
    result = {
        "device": args.device,
        "loss_model": args.loss_model,
        "loss_audit": loss_audit,
        "pump_template_dir_metadata_only": str(args.pump_template_dir),
        "sidebands": args.sidebands,
        "q_max": args.q_max,
        "k": args.k,
        "floquet_only": args.floquet_only,
        "branch_step": args.branch_step,
        "gmres_maxiter": args.gmres_maxiter,
        "gmres_restart": args.gmres_restart,
        "untracked_scan_every": args.untracked_scan_every,
        "untracked_scan_points": args.untracked_scan_points,
        "candidate_signal_ghz": _parse_candidate_signals(
            args.candidate_signal_ghz
        ),
        "mode_checkpoint_dir": (
            None
            if args.mode_checkpoint_dir is None
            else str(args.mode_checkpoint_dir)
        ),
        "repro_checkpoint_dir": (
            None
            if args.repro_checkpoint_dir is None
            else str(args.repro_checkpoint_dir)
        ),
        "rows": rows,
    }
    temporary = args.out.with_suffix(args.out.suffix + ".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    temporary.replace(args.out.with_suffix(".json"))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the physical column and print its final report."""
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["rows"] and result["rows"][-1]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
