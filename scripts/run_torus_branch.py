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
from twpa_solver.multitone.basis import build_autonomous_torus_basis  # noqa: E402
from twpa_solver.multitone.problem import FullMultiToneProblem  # noqa: E402
from twpa_solver.multitone.schur import (  # noqa: E402
    build_multitone_schur_problem,
)
from twpa_solver.multitone.seed import (  # noqa: E402
    seed_torus_from_floquet,
    seed_torus_from_pump,
)
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive  # noqa: E402
from twpa_solver.multitone.torus import TorusProblem  # noqa: E402
from twpa_solver.signal.io import load_pump  # noqa: E402


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
        help="Optional Phase-6 seed with vector and sidebands arrays.",
    )
    parser.add_argument(
        "--factor-backend",
        choices=("pardiso", "banded", "superlu"),
        default="pardiso",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--residual-tol", type=float, default=1e-9)
    parser.add_argument(
        "--seed-amplitudes",
        type=float,
        nargs="+",
        default=None,
        help="Relative perturbations to try when the direct seed fails.",
    )
    parser.add_argument(
        "--min-off-comb-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum q != 0 norm fraction for physical-torus acceptance. "
            "Numerically converged floor solutions are rejected and the "
            "remaining seed amplitudes are tried."
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


def _to_problem_nodes(seed: np.ndarray, torus: TorusProblem) -> np.ndarray:
    """Restrict a full-node seed to the coordinates used by the torus problem."""
    if not torus.is_schur:
        return seed
    return seed[:, torus.base_problem.partition.retained]


def _solve_seed(
    torus: TorusProblem,
    pump: Any,
    basis: Any,
    args: argparse.Namespace,
    previous_state: np.ndarray | None,
    floquet_seed: tuple[np.ndarray, list[int]] | None,
) -> tuple[np.ndarray, float, dict[str, Any], str]:
    if previous_state is not None:
        state, omega, report = torus.solve_newton(
            previous_state,
            omega_a0=args.omega_a_ratio * pump.omega_p,
            max_newton=args.max_newton,
            residual_tol=args.residual_tol,
        )
        report = dict(report)
        fraction = _off_comb_fraction(state, torus.basis)
        report["off_comb_norm_fraction"] = fraction
        report["physical_torus_gate_passed"] = bool(
            report.get("converged")
            and fraction >= args.min_off_comb_fraction
        )
        return state, omega, report, "warm_start"

    amplitudes = args.seed_amplitudes
    if amplitudes is None:
        amplitudes = [1.0e-6]
    attempts: list[dict[str, Any]] = []
    seed_routes = [(float(amplitudes[0]), "pump_plus_perturbation")]
    seed_routes.extend(
        (float(value), "pump_amplitude_sweep") for value in amplitudes[1:]
    )
    last_state: np.ndarray | None = None
    last_omega = args.omega_a_ratio * pump.omega_p
    last_report: dict[str, Any] = {}
    last_converged_report: dict[str, Any] | None = None
    for amplitude, route in seed_routes:
        seed = seed_torus_from_pump(
            pump.X,
            pump.basis,
            basis,
            amplitude=amplitude,
            node_ref=args.node_ref,
        )
        seed = _to_problem_nodes(seed, torus)
        state, omega, report = torus.solve_newton(
            seed,
            omega_a0=args.omega_a_ratio * pump.omega_p,
            max_newton=args.max_newton,
            residual_tol=args.residual_tol,
        )
        last_state = state
        last_omega = omega
        last_report = dict(report)
        fraction = _off_comb_fraction(state, torus.basis)
        gate_passed = bool(
            report.get("converged")
            and fraction >= args.min_off_comb_fraction
        )
        attempts.append(
            {
                "route": route,
                "amplitude": amplitude,
                "off_comb_norm_fraction": fraction,
                "physical_torus_gate_passed": gate_passed,
                **report,
            }
        )
        if gate_passed:
            report = dict(report)
            report["off_comb_norm_fraction"] = fraction
            report["physical_torus_gate_passed"] = True
            report["seed_attempts"] = attempts
            return state, omega, report, route

        if report.get("converged"):
            last_converged_report = dict(report)
            last_converged_report["off_comb_norm_fraction"] = fraction

    if floquet_seed is not None:
        vector, sidebands = floquet_seed
        seed = seed_torus_from_floquet(
            pump.X,
            pump.basis,
            basis,
            vector,
            sidebands,
            omega_p=pump.omega_p,
            omega_a=args.omega_a_ratio * pump.omega_p,
            node_ref=args.node_ref,
        )
        seed = _to_problem_nodes(seed, torus)
        state, omega, report = torus.solve_newton(
            seed,
            omega_a0=args.omega_a_ratio * pump.omega_p,
            max_newton=args.max_newton,
            residual_tol=args.residual_tol,
        )
        last_state = state
        last_omega = omega
        last_report = dict(report)
        fraction = _off_comb_fraction(state, torus.basis)
        gate_passed = bool(
            report.get("converged")
            and fraction >= args.min_off_comb_fraction
        )
        attempts.append(
            {
                "route": "optional_floquet",
                "off_comb_norm_fraction": fraction,
                "physical_torus_gate_passed": gate_passed,
                **report,
            }
        )
        if gate_passed:
            report = dict(report)
            report["off_comb_norm_fraction"] = fraction
            report["physical_torus_gate_passed"] = True
            report["seed_attempts"] = attempts
            return state, omega, report, "optional_floquet"
        if report.get("converged"):
            last_converged_report = dict(report)
            last_converged_report["off_comb_norm_fraction"] = fraction

    if last_state is None:
        raise RuntimeError("no torus seed attempt was executed")
    final_report = dict(last_converged_report or last_report)
    final_report["solver_converged"] = bool(last_converged_report)
    final_report["converged"] = False
    final_report["physical_torus_gate_passed"] = False
    final_report["gate_reason"] = (
        "all seed attempts either failed Newton convergence or remained below "
        "--min-off-comb-fraction"
    )
    final_report["seed_attempts"] = attempts
    return last_state, last_omega, final_report, "physical_gate_rejected"


def run_branch(args: argparse.Namespace) -> dict[str, Any]:
    if args.q_max < 1:
        raise ValueError("--q-max must be >= 1")
    if args.omega_a_ratio <= 0.0:
        raise ValueError("--omega-a-ratio must be positive")
    circuit = load_circuit(args.circuit_dir)
    controls = _controls(args)
    floquet_seed = _load_seed(args.floquet_seed_npz) if args.floquet_seed_npz else None
    rows: list[dict[str, Any]] = []
    previous_state: np.ndarray | None = None

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
            factor_backend=args.factor_backend,
        )
        state, omega, report, seed_route = _solve_seed(
            torus,
            pump,
            basis,
            args,
            previous_state,
            floquet_seed,
        )
        q0_norm, q1_norm = _q_norms(state, torus.basis)
        off_comb_fraction = q1_norm / max(q0_norm + q1_norm, 1.0e-300)
        rows.append(
            {
                "point_index": index,
                "control": control,
                "pump_dir": str(pump_dir),
                "pump_freq_ghz": pump.pump_freq_ghz,
                "omega_a_rad_s": omega,
                "omega_a_over_omega_p": omega / pump.omega_p,
                "off_comb_norm_fraction": off_comb_fraction,
                "q_max": args.q_max,
                "anchor_full_node": full_node_ref,
                "anchor_retained_index": node_ref if args.schur else None,
                "min_off_comb_fraction": args.min_off_comb_fraction,
                "seed_route": seed_route,
                **report,
            }
        )
        _atomic_write(
            args.out.with_name(f"{args.out.stem}.point_{index:03d}.json"),
            rows[-1],
        )
        previous_state = state if report.get("converged") else None

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
            "phase_anchor": "Im X[(0,1), node_ref] = 0",
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
