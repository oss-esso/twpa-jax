"""Measure a 7.4-GHz linear response about a converged torus state."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver.core import load_circuit
from twpa_solver.multitone.basis import build_autonomous_torus_basis
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.multitone.torus_signal import (
    solve_torus_signal_response,
    torus_junction_utilization,
)
from twpa_solver.signal.io import load_pump


def _pump_current(pump: Any) -> float:
    """Return the achieved source current from pump metadata."""
    for key in ("pump_current_a", "pump_current_peak_a", "current_a"):
        value = pump.metadata.get(key)
        if value is not None:
            return float(value)
    raise KeyError("pump metadata has no achieved pump current")


def _load_state(path: Path) -> tuple[np.ndarray, float, float, int]:
    """Load a torus state checkpoint."""
    with np.load(path) as data:
        required = {"state", "omega_a", "source_tau", "sideband_harmonics"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"state checkpoint is missing {sorted(missing)}")
        return (
            np.asarray(data["state"], dtype=np.complex128),
            float(data["omega_a"]),
            float(data["source_tau"]),
            int(data["sideband_harmonics"]),
        )


def _row(
    *,
    pump_dir: Path,
    state_path: Path,
    pump_power_dbm: float | None,
    pump_current_a: float,
    source_tau: float,
    omega_a: float,
    pump_frequency_ghz: float,
    utilization: float,
    residual: float,
    result: Any,
    runtime_s: float,
) -> dict[str, Any]:
    """Build one incrementally serializable result row."""
    return {
        "status": "PASS",
        "pump_dir": str(pump_dir),
        "state_path": str(state_path),
        "pump_power_dbm": pump_power_dbm,
        "pump_current_a": pump_current_a,
        "source_tau": source_tau,
        "achieved_pump_current_a": pump_current_a * source_tau,
        "pump_frequency_ghz": pump_frequency_ghz,
        "omega_a_over_omega_p": omega_a / (2.0 * np.pi * pump_frequency_ghz * 1e9),
        "torus_residual_rel": residual,
        "r_j": utilization,
        "signal_ghz": result.signal_ghz,
        "gain_vs_off_db": result.gain_vs_off_db,
        "signal_residual_rel": result.residual_rel,
        "response_unknowns": result.response_unknowns,
        "response_matrix_nnz": result.matrix_nnz,
        "response_assemble_runtime_s": result.assemble_runtime_s,
        "response_factor_solve_runtime_s": result.factor_solve_runtime_s,
        "torus_runtime_s": runtime_s,
        "response_tones": [
            {"h": tone.h, "q": tone.q} for tone in result.response_tones
        ],
    }


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    """Append one flushed CSV row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the response runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, required=True)
    parser.add_argument("--pump-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--signal-ghz", type=float, default=7.4)
    parser.add_argument("--source-port", type=int, default=1)
    parser.add_argument("--out-port", type=int, default=2)
    parser.add_argument("--pump-port", type=int, default=4)
    parser.add_argument("--source-current-a", type=float, default=1.0e-12)
    parser.add_argument("--loss-model", default="current_complex_c")
    parser.add_argument("--linear-solver", choices=("pardiso", "superlu"), default="pardiso")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one torus response and append its result."""
    args = parse_args(argv)
    circuit = load_circuit(args.circuit_dir)
    pump = load_pump(args.pump_dir, fallback_pump_freq_ghz=7.9)
    state, omega_a, source_tau, sideband_harmonics = _load_state(args.state)
    basis = build_autonomous_torus_basis(
        pump.omega_p,
        omega_a,
        pump.modes,
        q_max=1,
        sideband_harmonics=sideband_harmonics,
    )
    drive = MultiToneDrive(
        basis.pump_tone,
        circuit.port_to_index[args.pump_port],
        _pump_current(pump),
    ).to_coeffs(basis, circuit.node_count)
    problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(drive),
        loss_model=args.loss_model,
    )
    if state.shape != (basis.n_tones, circuit.node_count):
        raise ValueError(
            f"state shape {state.shape} does not match basis/circuit "
            f"{(basis.n_tones, circuit.node_count)}"
        )
    started = time.perf_counter()
    residual = problem.norms(state, source_tau, compute_time_residual=False)["coeff_rel"]
    utilization = torus_junction_utilization(problem, state)
    response = solve_torus_signal_response(
        problem,
        state,
        signal_ghz=args.signal_ghz,
        source_port=args.source_port,
        out_port=args.out_port,
        source_current_a=args.source_current_a,
        loss_model=args.loss_model,
        linear_solver=args.linear_solver,
    )
    runtime = time.perf_counter() - started
    power = pump.metadata.get("pump_power_dbm_requested")
    row = _row(
        pump_dir=args.pump_dir,
        state_path=args.state,
        pump_power_dbm=None if power is None else float(power),
        pump_current_a=_pump_current(pump),
        source_tau=source_tau,
        omega_a=omega_a,
        pump_frequency_ghz=pump.pump_freq_ghz,
        utilization=utilization,
        residual=float(residual),
        result=response,
        runtime_s=runtime,
    )
    fieldnames = list(row)
    _append_csv(args.out, row, fieldnames)
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
