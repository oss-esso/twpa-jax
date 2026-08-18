"""Continue an autonomous two-frequency HB branch from pump solutions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.multitone.basis import (  # noqa: E402
    ToneIndex,
    build_autonomous_torus_basis,
)
from twpa_solver.multitone.problem import FullMultiToneProblem  # noqa: E402
from twpa_solver.multitone.schur import (  # noqa: E402
    build_multitone_schur_problem,
)
from twpa_solver.multitone.seed import (  # noqa: E402
    promote_pump_solution,
    seed_torus_from_floquet,
)
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive  # noqa: E402
from twpa_solver.multitone.torus import TorusProblem  # noqa: E402
from twpa_solver.pump.problem import pack_complex, unpack_complex  # noqa: E402
from twpa_solver.signal.io import load_pump  # noqa: E402
from twpa_solver.signal.stability import audit_loss_convention  # noqa: E402


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument(
        "--pump-solution-dir",
        "--pump-dir",
        dest="pump_solution_dirs",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--control",
        type=float,
        nargs="+",
        default=None,
        help="Control values aligned with --pump-solution-dir.",
    )
    parser.add_argument("--omega-a-ratio", type=float, required=True)
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--node-ref", type=int, default=0)
    parser.add_argument(
        "--schur",
        action="store_true",
        help="Solve on the circuit's retained-node Schur coordinates.",
    )
    parser.add_argument("--loss-model", default=None)
    parser.add_argument(
        "--floquet-seed-npz",
        type=Path,
        default=None,
        help="Critical Hill seed with vector and sidebands arrays.",
    )
    parser.add_argument(
        "--sideband-harmonics",
        type=int,
        default=None,
        help="Override the h-sideband half-width used by the torus basis.",
    )
    parser.add_argument(
        "--factor-backend",
        choices=("pardiso", "banded", "superlu"),
        default="pardiso",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--state-checkpoint-dir",
        type=Path,
        default=None,
        help="Write converged torus states and tangents for basis promotion.",
    )
    parser.add_argument(
        "--initial-state-npz",
        type=Path,
        default=None,
        help="Warm-start PALC from a saved torus state checkpoint.",
    )
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--residual-tol", type=float, default=1e-9)
    parser.add_argument(
        "--branch-step",
        type=float,
        default=0.05,
        help="Initial normalized pseudo-arclength step.",
    )
    parser.add_argument("--gmres-rtol", type=float, default=1.0e-8)
    parser.add_argument("--gmres-maxiter", type=int, default=240)
    parser.add_argument("--gmres-restart", type=int, default=80)
    parser.add_argument(
        "--linear-debug",
        action="store_true",
        help="Record augmented JVP, border, and GMRES diagnostics.",
    )
    parser.add_argument("--linear-debug-fd-step", type=float, default=1.0e-6)
    parser.add_argument(
        "--omitted-q-max",
        type=int,
        default=None,
        help="Evaluate residual content through this larger generator order.",
    )
    parser.add_argument(
        "--min-off-comb-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum q != 0 norm fraction for physical-torus acceptance."
        ),
    )
    return parser.parse_args(argv)


def _pump_current(pump: Any) -> float:
    for key in ("pump_current_a", "pump_current_peak_a", "current_a"):
        value = pump.metadata.get(key)
        if value is not None:
            return float(value)
    raise KeyError("pump metadata has no pump current field")


def _pump_port(pump: Any, circuit: Any) -> int:
    value = pump.metadata.get("pump_port")
    if value is not None:
        return int(value)
    if not circuit.port_to_index:
        raise KeyError("circuit has no port and pump metadata has no pump_port")
    return int(next(iter(circuit.port_to_index)))


def _load_seed(path: Path) -> tuple[np.ndarray, list[int]]:
    with np.load(path) as data:
        if "vector" not in data or "sidebands" not in data:
            raise ValueError("Floquet seed must contain vector and sidebands arrays")
        return np.asarray(data["vector"], dtype=np.complex128), [
        int(value) for value in np.asarray(data["sidebands"]).reshape(-1)
    ]


def _load_state_checkpoint(path: Path) -> dict[str, Any]:
    """Load a saved torus state and its continuation metadata."""
    with np.load(path) as data:
        required = {"state", "tangent", "omega_a", "source_tau"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"state checkpoint is missing fields: {sorted(missing)}"
            )
        return {
            "state": np.asarray(data["state"], dtype=np.complex128),
            "tangent": np.asarray(data["tangent"], dtype=float),
            "omega_a": float(data["omega_a"]),
            "source_tau": float(data["source_tau"]),
            "sideband_harmonics": int(data["sideband_harmonics"]),
        }


def _remap_state_basis(
    state: np.ndarray,
    source_basis: Any,
    target_basis: Any,
) -> np.ndarray:
    """Promote a torus state by matching common ``(h, q)`` tones."""
    source_rows = {
        ToneIndex(tone.h, tone.q): index
        for index, tone in enumerate(source_basis.tones)
    }
    promoted = np.zeros(
        (target_basis.n_tones, state.shape[1]), dtype=np.complex128
    )
    for target_index, tone in enumerate(target_basis.tones):
        source_index = source_rows.get(ToneIndex(tone.h, tone.q))
        if source_index is not None:
            promoted[target_index] = state[source_index]
    return promoted


def _remap_tangent(
    tangent: np.ndarray,
    source_basis: Any,
    target_basis: Any,
    source_state: np.ndarray,
    state_scale: float,
) -> np.ndarray:
    """Promote the packed state portion of a normalized tangent."""
    source_size = 2 * source_state.size
    tangent_state = unpack_complex(
        tangent[:source_size], source_state.shape
    )
    promoted_tangent = _remap_state_basis(
        tangent_state, source_basis, target_basis
    )
    target_size = 2 * promoted_tangent.size
    target_tangent = np.concatenate((
        pack_complex(promoted_tangent) / max(state_scale, 1.0e-300),
        tangent[source_size:],
    ))
    if target_tangent.size != target_size + 2:
        raise ValueError("promoted tangent has an invalid augmented size")
    norm = float(np.linalg.norm(target_tangent))
    return target_tangent / max(norm, 1.0e-300)


def _controls(args: argparse.Namespace) -> list[float | None]:
    if args.control is None:
        return [None] * len(args.pump_solution_dirs)
    if len(args.control) != len(args.pump_solution_dirs):
        raise ValueError("--control must have one value per pump solution directory")
    return [float(value) for value in args.control]


def _q_norms(state: np.ndarray, basis: Any) -> tuple[float, float]:
    q0 = [index for index, tone in enumerate(basis.tones) if tone.q == 0]
    q1 = [index for index, tone in enumerate(basis.tones) if tone.q != 0]
    q0_norm = float(np.linalg.norm(state[q0])) if q0 else 0.0
    q1_norm = float(np.linalg.norm(state[q1])) if q1 else 0.0
    return q0_norm, q1_norm


def _off_comb_fraction(state: np.ndarray, basis: Any) -> float:
    """Return the norm fraction outside the pump comb."""
    q0_norm, q1_norm = _q_norms(state, basis)
    return q1_norm / max(q0_norm + q1_norm, 1.0e-300)


def _torus_radius_squared(state: np.ndarray, basis: Any) -> float:
    """Return the q=+/-1 power relative to the q=0 power."""
    plus = [index for index, tone in enumerate(basis.tones) if tone.q == 1]
    minus = [index for index, tone in enumerate(basis.tones) if tone.q == -1]
    zero = [index for index, tone in enumerate(basis.tones) if tone.q == 0]
    numerator = float(np.linalg.norm(state[plus]) ** 2)
    numerator += float(np.linalg.norm(state[minus]) ** 2)
    denominator = float(np.linalg.norm(state[zero]) ** 2)
    return numerator / max(denominator, 1.0e-300)


def _sector_overlap(
    state: np.ndarray,
    reference: np.ndarray | None,
    basis: Any,
    q: int = 1,
) -> float | None:
    """Return the phase-invariant overlap in one generator sector."""
    if reference is None:
        return None
    rows = [index for index, tone in enumerate(basis.tones) if tone.q == q]
    if not rows:
        return None
    left = state[rows].reshape(-1)
    right = reference[rows].reshape(-1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(abs(np.vdot(left, right)) / (left_norm * right_norm))


def _vector_overlap(
    left: np.ndarray | None, right: np.ndarray | None
) -> float | None:
    """Return the phase-invariant overlap of two normalized coordinates."""
    if left is None or right is None:
        return None
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 0.0:
        return 0.0
    return float(abs(np.vdot(left_flat, right_flat)) / denominator)


def _tangent_angle(
    previous: np.ndarray | None, current: np.ndarray | None
) -> float | None:
    """Return the unsigned angle between consecutive normalized tangents."""
    if previous is None or current is None:
        return None
    denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
    if denominator <= 0.0:
        return None
    cosine = float(np.dot(previous, current) / denominator)
    return float(np.arccos(np.clip(abs(cosine), -1.0, 1.0)))


def _effective_drive_dbm(control: float | None, source_tau: float) -> float | None:
    """Convert a source-scale continuation point to an equivalent dBm value."""
    if control is None or source_tau <= 0.0:
        return None
    return float(control + 20.0 * np.log10(source_tau))


def _to_problem_nodes(seed: np.ndarray, torus: TorusProblem) -> np.ndarray:
    """Restrict a full-node seed to the coordinates used by the torus problem."""
    if not torus.is_schur:
        return seed
    return seed[:, torus.base_problem.partition.retained]


def _pump_coordinate(pump: Any, basis: Any, torus: TorusProblem) -> np.ndarray:
    """Promote and restrict the pump using the torus solver coordinates."""
    return _to_problem_nodes(
        promote_pump_solution(pump.X, pump.basis, basis), torus
    )


def _solve_seed(
    torus: TorusProblem,
    pump: Any,
    basis: Any,
    args: argparse.Namespace,
    previous_state: np.ndarray | None,
    previous_omega: float | None,
    previous_source_tau: float | None,
    previous_tangent: np.ndarray | None,
    floquet_seed: tuple[np.ndarray, list[int]] | None,
) -> tuple[np.ndarray, float, dict[str, Any], str]:
    report: dict[str, Any]
    if previous_state is None:
        pump_state = _pump_coordinate(pump, basis, torus)
        if floquet_seed is None:
            return (
                pump_state,
                args.omega_a_ratio * pump.omega_p,
                {
                    "converged": False,
                    "source_tau": 1.0,
                    "physical_torus_gate_passed": False,
                    "failure_reason": (
                        "Floquet seed is required for NS branch switch"
                    ),
                },
                "missing_floquet_seed",
            )
        vector, sidebands = floquet_seed
        seeded = seed_torus_from_floquet(
            pump.X,
            pump.basis,
            basis,
            vector,
            sidebands,
            omega_p=pump.omega_p,
            omega_a=args.omega_a_ratio * pump.omega_p,
            perturbation_amplitude=1.0,
            node_ref=args.node_ref,
        )
        mode = _to_problem_nodes(seeded, torus) - pump_state
        state, omega, tau, report, tangent = torus.solve_torus_branch_switch(
            pump_state,
            omega_a_ns=args.omega_a_ratio * pump.omega_p,
            source_tau_ns=1.0,
            perturbation=mode,
            step_size=args.branch_step,
            max_newton=args.max_newton,
            residual_tol=args.residual_tol,
            gmres_rtol=args.gmres_rtol,
            gmres_maxiter=args.gmres_maxiter,
            gmres_restart=args.gmres_restart,
            linear_debug=args.linear_debug,
            linear_debug_fd_step=args.linear_debug_fd_step,
        )
        route = "floquet_branch_switch"
    else:
        if (
            previous_omega is None
            or previous_source_tau is None
            or previous_tangent is None
        ):
            raise ValueError("a converged PALC point requires source and tangent")
        predictor = torus.predict_torus_arclength(
            previous_state,
            previous_omega,
            previous_source_tau,
            previous_tangent,
            args.branch_step,
        )
        state, omega, tau, report, tangent = torus.solve_torus_arclength(
            predictor[0],
            previous_X=previous_state,
            previous_omega_a=previous_omega,
            previous_source_tau=previous_source_tau,
            tangent=previous_tangent,
            step_size=args.branch_step,
            omega_a0=predictor[1],
            source_tau0=predictor[2],
            phase_reference=previous_state,
            max_newton=args.max_newton,
            residual_tol=args.residual_tol,
            gmres_rtol=args.gmres_rtol,
            gmres_maxiter=args.gmres_maxiter,
            gmres_restart=args.gmres_restart,
            linear_debug=args.linear_debug,
            linear_debug_fd_step=args.linear_debug_fd_step,
        )
        route = "palc"

    report = dict(report)
    report["source_tau"] = tau
    fraction = _off_comb_fraction(state, torus.basis)
    report["off_comb_norm_fraction"] = fraction
    if args.omitted_q_max is not None and args.omitted_q_max > args.q_max:
        report.update(torus.omitted_q_residual(state, args.omitted_q_max))
    report["physical_torus_gate_passed"] = bool(
        report.get("converged")
        and fraction >= args.min_off_comb_fraction
    )
    if tangent is not None:
        report["_tangent"] = tangent
    return state, omega, report, route


def run_branch(args: argparse.Namespace) -> dict[str, Any]:
    if args.q_max < 1:
        raise ValueError("--q-max must be >= 1")
    if args.omega_a_ratio <= 0.0:
        raise ValueError("--omega-a-ratio must be positive")
    if args.branch_step <= 0.0:
        raise ValueError("--branch-step must be positive")
    circuit = load_circuit(args.circuit_dir)
    loss_audit = audit_loss_convention(circuit, args.loss_model)
    controls = _controls(args)
    floquet_seed = _load_seed(args.floquet_seed_npz) if args.floquet_seed_npz else None
    initial_checkpoint = (
        _load_state_checkpoint(args.initial_state_npz)
        if args.initial_state_npz is not None
        else None
    )
    sideband_harmonics = args.sideband_harmonics
    if floquet_seed is not None:
        seed_harmonics = max(abs(value) for value in floquet_seed[1])
        if sideband_harmonics is None:
            sideband_harmonics = seed_harmonics
    rows: list[dict[str, Any]] = []
    previous_state: np.ndarray | None = None
    previous_omega: float | None = None
    previous_source_tau: float | None = None
    previous_tangent: np.ndarray | None = None
    ns_control = controls[0] if controls else None
    checkpoint_applied = False

    for index, (pump_dir, control) in enumerate(
        zip(args.pump_solution_dirs, controls)
    ):
        pump = load_pump(pump_dir, fallback_pump_freq_ghz=7.0)
        omega_a = args.omega_a_ratio * pump.omega_p
        basis = build_autonomous_torus_basis(
            pump.omega_p,
            omega_a,
            pump.modes,
            args.q_max,
            sideband_harmonics=sideband_harmonics,
        )
        pump_port = _pump_port(pump, circuit)
        drive = MultiToneDrive(
            basis.pump_tone,
            circuit.port_to_index[pump_port],
            _pump_current(pump),
        ).to_coeffs(basis, circuit.C.shape[0])
        full_base = FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.pump_turn_on(drive),
            loss_model=args.loss_model,
        )
        if args.schur:
            port_indices = list(circuit.port_to_index.values())
            base = build_multitone_schur_problem(
                full_base,
                port_indices,
                preconditioner="real_coupled_fast",
            )
            full_node_ref = args.node_ref
            retained_pos = int(base.partition.retained_pos[full_node_ref])
            if retained_pos < 0:
                raise ValueError(
                    f"node_ref {full_node_ref} is not retained by the Schur "
                    "partition"
                )
            node_ref = retained_pos
        else:
            base = full_base
            full_node_ref = args.node_ref
            node_ref = args.node_ref
        torus = TorusProblem(
            base,
            tuple(pump.modes),
            args.q_max,
            omega_a,
            node_ref=node_ref,
            sideband_harmonics=sideband_harmonics,
            factor_backend=args.factor_backend,
        )
        if initial_checkpoint is not None and not checkpoint_applied:
            source_basis = build_autonomous_torus_basis(
                pump.omega_p,
                initial_checkpoint["omega_a"],
                pump.modes,
                args.q_max,
                sideband_harmonics=initial_checkpoint["sideband_harmonics"],
            )
            source_state = initial_checkpoint["state"]
            source_tangent = initial_checkpoint["tangent"]
            if args.schur:
                retained = np.asarray(
                    torus.base_problem.partition.retained, dtype=int
                )
                source_state_size = 2 * source_state.size
                tangent_state = unpack_complex(
                    source_tangent[:source_state_size], source_state.shape
                )
                source_state = source_state[:, retained]
                source_tangent = np.concatenate((
                    pack_complex(tangent_state[:, retained]),
                    source_tangent[source_state_size:],
                ))
            previous_state = _remap_state_basis(
                source_state, source_basis, torus.basis
            )
            previous_omega = initial_checkpoint["omega_a"]
            previous_source_tau = initial_checkpoint["source_tau"]
            previous_tangent = _remap_tangent(
                source_tangent,
                source_basis,
                torus.basis,
                source_state,
                max(float(np.linalg.norm(pack_complex(previous_state))), 1e-300),
            )
            checkpoint_applied = True
        pump_state = _pump_coordinate(pump, basis, torus)
        critical_reference: np.ndarray | None = None
        if floquet_seed is not None:
            vector, seed_sidebands = floquet_seed
            seeded = seed_torus_from_floquet(
                pump.X,
                pump.basis,
                basis,
                vector,
                seed_sidebands,
                omega_p=pump.omega_p,
                omega_a=omega_a,
                perturbation_amplitude=1.0,
                node_ref=full_node_ref,
            )
            critical_reference = _to_problem_nodes(seeded, torus) - pump_state
        prior_state = previous_state
        prior_tangent = previous_tangent
        state, omega, report, seed_route = _solve_seed(
            torus,
            pump,
            basis,
            args,
            previous_state,
            previous_omega,
            previous_source_tau,
            previous_tangent,
            floquet_seed,
        )
        continuation_tangent = report.pop("_tangent", None)
        q0_norm, q1_norm = _q_norms(state, torus.basis)
        off_comb_fraction = q1_norm / max(q0_norm + q1_norm, 1.0e-300)
        source_tau = float(report.get("source_tau", 1.0))
        effective_drive_dbm = _effective_drive_dbm(control, source_tau)
        radius_squared = _torus_radius_squared(state, torus.basis)
        previous_state_overlap = _vector_overlap(state, prior_state)
        critical_mode_overlap = _sector_overlap(
            state, critical_reference, torus.basis
        )
        tangent_angle = _tangent_angle(prior_tangent, continuation_tangent)
        rows.append(
            {
                "point_index": index,
                "control": control,
                "effective_drive_dbm": effective_drive_dbm,
                "drive_delta_from_ns_db": (
                    None
                    if effective_drive_dbm is None or ns_control is None
                    else effective_drive_dbm - ns_control
                ),
                "pump_dir": str(pump_dir),
                "pump_freq_ghz": pump.pump_freq_ghz,
                "omega_a_rad_s": omega,
                "omega_a_over_omega_p": omega / pump.omega_p,
                "off_comb_norm_fraction": off_comb_fraction,
                "torus_radius_squared": radius_squared,
                "previous_state_overlap": previous_state_overlap,
                "critical_mode_overlap_q_plus_1": critical_mode_overlap,
                "tangent_angle_rad": tangent_angle,
                "q_max": args.q_max,
                "anchor_full_node": full_node_ref,
                "anchor_retained_index": node_ref if args.schur else None,
                "min_off_comb_fraction": args.min_off_comb_fraction,
                "seed_route": seed_route,
                **report,
            }
        )
        if (
            args.state_checkpoint_dir is not None
            and report.get("converged")
            and continuation_tangent is not None
        ):
            checkpoint_path = args.state_checkpoint_dir / (
                f"point_{index:03d}.npz"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                checkpoint_path,
                state=state,
                tangent=continuation_tangent,
                omega_a=omega,
                source_tau=source_tau,
                sideband_harmonics=sideband_harmonics,
            )
            rows[-1]["state_checkpoint"] = str(checkpoint_path)
        _atomic_write(
            args.out.with_name(f"{args.out.stem}.point_{index:03d}.json"),
            rows[-1],
        )
        if report.get("converged"):
            previous_state = state
            previous_omega = omega
            previous_source_tau = float(report["source_tau"])
            previous_tangent = continuation_tangent
        else:
            previous_state = None
            previous_omega = None
            previous_source_tau = None
            previous_tangent = None

    return {
            "metadata": {
            "device": args.device,
            "circuit_dir": str(args.circuit_dir),
            "q_max": args.q_max,
            "omega_a_ratio_seed": args.omega_a_ratio,
            "factor_backend": args.factor_backend,
            "schur": args.schur,
            "anchor_node_ref": args.node_ref,
            "anchor_retained_index": node_ref if args.schur else None,
            "anchor_full_node": full_node_ref,
            "seed_source": (
                str(args.floquet_seed_npz) if args.floquet_seed_npz else None
            ),
            "phase_condition": (
                "critical q=+1 reference for branch switch and PALC"
            ),
            "branch_corrector": "matrix_free_augmented_palc",
            "loss_audit": loss_audit,
        },
        "points": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_branch(args)
    _atomic_write(args.out, result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
