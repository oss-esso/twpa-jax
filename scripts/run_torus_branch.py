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
    promote_pump_solution,
    seed_torus_from_floquet,
)
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive  # noqa: E402
from twpa_solver.multitone.torus import TorusProblem  # noqa: E402
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
        "--factor-backend",
        choices=("pardiso", "banded", "superlu"),
        default="pardiso",
    )
    parser.add_argument("--out", type=Path, required=True)
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
    sideband_harmonics = None
    if floquet_seed is not None:
        sideband_harmonics = max(
            abs(value) for value in floquet_seed[1]
        )
    rows: list[dict[str, Any]] = []
    previous_state: np.ndarray | None = None
    previous_omega: float | None = None
    previous_source_tau: float | None = None
    previous_tangent: np.ndarray | None = None

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
