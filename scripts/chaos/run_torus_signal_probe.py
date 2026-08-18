"""Measure the finite-signal HB probe along a torus pump column."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.multitone.basis import (  # noqa: E402
    build_sideband_matched_basis,
)
from twpa_solver.multitone.observables import tone_s21  # noqa: E402
from twpa_solver.multitone.problem import FullMultiToneProblem  # noqa: E402
from twpa_solver.multitone.schur import (  # noqa: E402
    SchurMultiToneProblem,
    build_multitone_schur_problem,
)
from twpa_solver.multitone.seed import promote_pump_solution  # noqa: E402
from twpa_solver.multitone.source import (  # noqa: E402
    AffineSourcePath,
    MultiToneDrive,
)
from twpa_solver.pump import (  # noqa: E402
    HarmonicNewtonKrylovSolver,
    NewtonKrylovSettings,
)
from twpa_solver.signal.io import load_pump  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the finite-signal probe arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--pump-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--signal-ghz", type=float, required=True)
    parser.add_argument("--signal-current-a", type=float, default=1.0e-12)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--sidebands", type=int, default=2)
    parser.add_argument("--schur", action="store_true")
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _settings() -> NewtonKrylovSettings:
    """Return the production multitone Newton settings."""
    return NewtonKrylovSettings(
        newton_tol=1.0e-10,
        max_newton=20,
        gmres_rtol=1.0e-8,
        gmres_atol=0.0,
        gmres_restart=20,
        gmres_maxiter=40,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled_fast",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


def _full_state(problem: Any, state: np.ndarray) -> np.ndarray:
    """Reconstruct a full-node state after a Schur solve."""
    if isinstance(problem, SchurMultiToneProblem):
        return problem.reconstruct_full(state)
    return state


def gain_vs_off_db(on: complex, off: complex) -> float:
    """Return signal gain relative to the identical pump-off response."""
    return float(20.0 * math.log10(max(abs(on), 1.0e-300) / max(abs(off), 1.0e-300)))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the current probe rows atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run pump-off and pump-on signal solves for each pump checkpoint."""
    if args.signal_current_a <= 0.0:
        raise ValueError("--signal-current-a must be positive")
    if args.signal_ghz <= 0.0 or args.sidebands < 2:
        raise ValueError("invalid signal frequency or sideband count")
    circuit = load_circuit(args.circuit_dir)
    settings = _settings()
    rows: list[dict[str, object]] = []
    off_state: np.ndarray | None = None
    off_problem: Any | None = None
    off_s21: complex | None = None
    cached_partition: Any | None = None
    cached_basis: Any | None = None

    for index, pump_dir in enumerate(args.pump_dir):
        started = time.perf_counter()
        pump = load_pump(pump_dir, fallback_pump_freq_ghz=7.9)
        delta = pump.omega_p - 2.0 * math.pi * args.signal_ghz * 1.0e9
        basis = build_sideband_matched_basis(
            pump.modes,
            args.sidebands,
            pump.omega_p,
            delta,
            pump.omega_p * (max(pump.modes) + 2),
        )
        pump_current = float(pump.metadata["pump_current_a"])
        pump_source = MultiToneDrive(
            basis.pump_tone,
            circuit.port_to_index[args.pump_port],
            pump_current,
        ).to_coeffs(basis, circuit.node_count)
        signal_source = MultiToneDrive(
            basis.signal_tone,
            circuit.port_to_index[args.source_port],
            args.signal_current_a,
        ).to_coeffs(basis, circuit.node_count)

        def make_problem(path: AffineSourcePath) -> Any:
            nonlocal cached_partition
            full = FullMultiToneProblem(
                circuit,
                basis,
                path,
                loss_model=args.loss_model,
                preconditioner="real_coupled_fast",
            )
            if not args.schur:
                return full
            if cached_partition is None:
                cached_partition = build_multitone_schur_problem(
                    full,
                    list(circuit.port_to_index.values()),
                    preconditioner="real_coupled_fast",
                ).partition
            return SchurMultiToneProblem(
                full,
                cached_partition,
                preconditioner="real_coupled_fast",
            )

        if cached_basis is None:
            cached_basis = basis
            off_problem = make_problem(
                AffineSourcePath.signal_turn_on(
                    np.zeros_like(signal_source), signal_source
                )
            )
            off_state, off_report = HarmonicNewtonKrylovSolver(settings).solve_one(
                off_problem,
                off_problem.zeros(),
                1.0,
            )
            if not off_report.converged:
                raise RuntimeError("pump-off signal probe did not converge")
            off_s21 = tone_s21(
                _full_state(off_problem, off_state),
                basis,
                circuit,
                signal_tone=basis.signal_tone,
                source_port=args.source_port,
                out_port=args.out_port,
                source_current_a=args.signal_current_a,
            )
        elif basis.to_metadata() != cached_basis.to_metadata():
            raise ValueError("pump checkpoints do not share a signal basis")

        pump_seed = promote_pump_solution(pump.X, pump.basis, basis)
        pump_problem = make_problem(
            AffineSourcePath.signal_turn_on(pump_source, signal_source)
        )
        if args.schur:
            pump_seed = pump_seed[:, cached_partition.retained]
        on_state, on_report = HarmonicNewtonKrylovSolver(settings).solve_one(
            pump_problem,
            pump_seed,
            1.0,
        )
        on_s21 = tone_s21(
            _full_state(pump_problem, on_state),
            basis,
            circuit,
            signal_tone=basis.signal_tone,
            source_port=args.source_port,
            out_port=args.out_port,
            source_current_a=args.signal_current_a,
        )
        row: dict[str, object] = {
            "point_index": index,
            "pump_dir": str(pump_dir),
            "pump_current_a": pump_current,
            "signal_ghz": args.signal_ghz,
            "signal_current_a": args.signal_current_a,
            "basis_tones": basis.n_tones,
            "pump_converged": bool(on_report.converged),
            "gain_vs_off_db": gain_vs_off_db(on_s21, off_s21),
            "runtime_s": time.perf_counter() - started,
            "status": "PASS" if on_report.converged else "FAIL",
            "failure_reason": on_report.failure_reason,
        }
        rows.append(row)
        _write_rows(args.out, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the probe and print its final rows."""
    rows = run(parse_args(argv))
    print(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
