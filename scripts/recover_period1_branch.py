"""Recover a PERIOD1 HB branch beyond a fixed-power Newton obstruction.

This is a bounded experiment driver.  It first attempts finer natural
continuation from a validated checkpoint and then falls back to the existing
pseudo-arclength corrector.  It never changes the HB basis and never labels a
failed target as a physical state.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.h1_transient_branch_transfer import build_system  # noqa: E402
from twpa_solver.pump.hb import FullPumpProblem, HarmonicGrid  # noqa: E402
from twpa_solver.pump.io import summarize_solution, write_results  # noqa: E402
from twpa_solver.pump.solver import (  # noqa: E402
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
)
from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.core.nonlinear import make_branch_law  # noqa: E402


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss / (1024.0**2))


def _load_checkpoint(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    report = json.loads((checkpoint / "pump_report.json").read_text(encoding="utf-8"))
    if report.get("final_status") != "VALID_CONVERGED":
        raise ValueError(f"checkpoint is not VALID_CONVERGED: {checkpoint}")
    data = np.load(checkpoint / "pump_solution.npz")
    X = np.asarray(data["X_real"], dtype=float) + 1j * np.asarray(data["X_imag"], dtype=float)
    modes = np.asarray(data["pump_modes"], dtype=int).reshape(-1)
    current = float(report["metadata"]["pump_current_a"])
    return X, modes, current, report


def _current_for_dbm(reference_current: float, reference_dbm: float, target_dbm: float) -> float:
    return reference_current * 10.0 ** ((float(target_dbm) - float(reference_dbm)) / 20.0)


def _problem(
    system: Any,
    modes: np.ndarray,
    current: float,
    dc_branch_flux: np.ndarray,
) -> FullPumpProblem:
    grid = HarmonicGrid(
        modes=modes,
        nt=max(2 * int(np.max(np.abs(modes))) + 1, 40),
        omega=system.omega,
    )
    return FullPumpProblem(
        system.circuit.C,
        system.circuit.G,
        system.circuit.K,
        system.circuit.Bphi,
        make_branch_law(system.circuit),
        grid,
        system.pump_node,
        current,
        dc_branch_flux=dc_branch_flux,
    )


def _settings(args: argparse.Namespace) -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=args.newton_tol,
        max_newton=args.newton_max,
        gmres_rtol=1.0e-7,
        gmres_atol=0.0,
        gmres_restart=60,
        gmres_maxiter=args.gmres_maxiter,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=True,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


def _write_checkpoint(
    output: Path,
    target_dbm: float,
    current: float,
    X: np.ndarray,
    modes: np.ndarray,
    problem: FullPumpProblem,
    source_report: dict[str, Any],
    recovery: dict[str, Any],
) -> Path:
    point = output / f"point_{target_dbm:+.6f}" / "pump"
    metadata = dict(source_report.get("metadata", {}))
    metadata.update(
        {
            "pump_power_dbm_requested": float(target_dbm),
            "pump_current_a": float(current),
            "pump_modes": modes.tolist(),
            "pump_validation_status": "VALID_CONVERGED",
            "period1_recovery": recovery,
        }
    )
    write_results(point, X, [], summarize_solution(problem, X), metadata)
    return point


def recover(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    X, modes, reference_current, source_report = _load_checkpoint(checkpoint)
    reference_dbm = float(source_report["metadata"]["pump_power_dbm_requested"])
    circuit = load_circuit(args.circuit_dir)
    metadata = source_report.get("metadata", {})
    dc_value = metadata.get("dc_branch_flux", metadata.get("dc_branch_flux_wb", 0.0))
    dc = np.asarray(dc_value, dtype=float).reshape(-1)
    if dc.size == 1:
        dc = np.full(circuit.branch_count, float(dc[0]))
    if dc.size != circuit.branch_count:
        raise ValueError("checkpoint DC flux has an incompatible branch count")
    system = build_system(args.circuit_dir, args.freq_ghz, args.pump_port, dc)
    solver = HarmonicNewtonKrylovSolver(_settings(args))
    targets = sorted(float(value) for value in args.target_dbm if float(value) > reference_dbm)
    if not targets:
        raise ValueError("all target powers must exceed the checkpoint power")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    current = reference_current
    previous_dbm = reference_dbm
    for target_dbm in targets:
        target_current = _current_for_dbm(current, previous_dbm, target_dbm)
        started = time.perf_counter()
        natural_problem = _problem(system, modes, target_current, dc)
        natural_X, natural_report = solver.solve_one(natural_problem, X, 1.0)
        method = "natural"
        report = natural_report
        recovered_X = natural_X
        palc_info: dict[str, Any] | None = None
        if not report.converged and args.enable_palc:
            method = "palc"
            palc_X, _lambda, palc_info = solver.solve_arclength(
                natural_problem,
                X,
                current / target_current,
                ds=args.palc_ds,
                max_steps=args.palc_max_steps,
                target_lam=1.0,
                max_wall_s=args.palc_deadline_s,
                rescale_every=5,
                max_steps_after_fold=args.palc_steps_after_fold,
                step_control="adaptive",
                refine_fold=True,
            )
            if palc_info.get("reached_target"):
                recovered_X = palc_X
                report = solver.solve_one(natural_problem, palc_X, 1.0)[1]
        converged = bool(report.converged)
        row: dict[str, Any] = {
            "power_dbm": target_dbm,
            "current_a": target_current,
            "method": method,
            "converged": converged,
            "newton": report.__dict__,
            "palc": palc_info,
            "rss_mb": _rss_mb(),
            "runtime_s": time.perf_counter() - started,
        }
        if converged:
            recovery = {
                "method": method,
                "source_checkpoint": str(checkpoint),
                "natural_report": report.__dict__,
                "palc": palc_info,
            }
            path = _write_checkpoint(
                args.output,
                target_dbm,
                target_current,
                recovered_X,
                modes,
                natural_problem,
                source_report,
                recovery,
            )
            row["checkpoint"] = str(path)
            X = recovered_X
            current = target_current
            previous_dbm = target_dbm
        rows.append(row)
        print(
            f"{target_dbm:+.6f} dBm: method={method} converged={converged} "
            f"rss_mb={row['rss_mb']} runtime={row['runtime_s']:.1f}s",
            flush=True,
        )
        if not converged:
            break
        gc.collect()
    payload = {
        "metadata": {
            "reference_checkpoint": str(checkpoint),
            "reference_power_dbm": reference_dbm,
            "targets_dbm": targets,
            "basis_modes": modes.tolist(),
            "recovery_order": ["natural", "palc"],
            "no_new_hb_ansatz": True,
        },
        "results": rows,
    }
    (args.output / "period1_recovery.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-dbm", type=float, nargs="+", required=True)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--output", type=Path, default=ROOT / ".hybrid_outputs" / "period1_recovery_7p9_2c_v1")
    parser.add_argument("--freq-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--newton-tol", type=float, default=1.0e-9)
    parser.add_argument("--newton-max", type=int, default=16)
    parser.add_argument("--gmres-maxiter", type=int, default=80)
    parser.add_argument("--enable-palc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--palc-ds", type=float, default=0.01)
    parser.add_argument("--palc-max-steps", type=int, default=150)
    parser.add_argument("--palc-steps-after-fold", type=int, default=30)
    parser.add_argument("--palc-deadline-s", type=float, default=300.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    recover(parse_args())
