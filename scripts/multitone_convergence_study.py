"""Run nonlinear JPA P1dB convergence sweeps over the multitone basis."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twpa_solver.builders.jc_doc import build_fqjtwpa, build_jpa, build_jtwpa
from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import (
    MultiToneBasis,
    build_lattice_basis,
    build_three_tone_basis,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.compression_curve import refine_p1db
from twpa_solver.multitone.compression import solve_signal_power_point
from twpa_solver.multitone.seed import promote_pump_solution
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import PumpBasis


Z0_OHM = 50.0


@dataclass(frozen=True)
class ConvergenceSetting:
    basis_kind: str
    signal_order_max: int
    pump_policy: str
    pump_order: int
    torus_scale: int


@dataclass(frozen=True)
class ConvergenceResult:
    setting: ConvergenceSetting
    p1db_dbm: float
    small_signal_gain_db: float
    n_tones: int
    n_p: int
    n_delta: int
    even_pump_fraction: float
    runtime_s: float
    p1db_method: str


def _settings() -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10,
        max_newton=30,
        gmres_rtol=1e-8,
        gmres_atol=0.0,
        gmres_restart=40,
        gmres_maxiter=60,
        min_alpha=1.0 / 1024.0,
        preconditioner="real_coupled",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
    )


def _circuit(device: str = "jpa") -> tuple[CircuitMatrices, float, dict[str, object]]:
    builders = {"jpa": build_jpa, "jtwpa": build_jtwpa, "fqjtwpa": build_fqjtwpa}
    builder, metadata = builders[device]()
    arrays = builder.assemble()
    circuit = CircuitMatrices(
        C=arrays["C"],
        G=arrays["G"],
        K=arrays["K"],
        Bphi=arrays["Bphi"],
        Ic=arrays["Ic"],
        port_to_index=arrays["ports"],
    )
    return circuit, float(metadata["pump_sources"][0]["current_a"]), metadata


def _pump_modes(policy: str, order: int) -> list[int]:
    if order < 1:
        raise ValueError("pump_order must be positive")
    if policy == "odd":
        return list(range(1, 2 * order, 2))
    if policy == "dense":
        return list(range(1, order + 1))
    raise ValueError(f"unknown pump policy {policy!r}")


def _basis(
    setting: ConvergenceSetting,
    pump_modes: list[int],
    omega_p: float,
    delta: float,
) -> MultiToneBasis:
    if setting.basis_kind == "three_tone":
        base = build_three_tone_basis(omega_p, delta)
    elif setting.basis_kind == "lattice":
        base = build_lattice_basis(
            pump_modes,
            setting.signal_order_max,
            omega_p,
            delta,
            omega_p * (max(pump_modes) + 2),
        )
    else:
        raise ValueError(f"unknown basis kind {setting.basis_kind!r}")
    if setting.torus_scale == 1:
        return base
    return MultiToneBasis(
        list(base.tones),
        omega_p,
        delta,
        n_p=int(base.n_p) * setting.torus_scale,
        n_delta=int(base.n_delta) * setting.torus_scale,
    )


def _current_to_dbm(current_a: float) -> float:
    power_w = current_a * current_a * Z0_OHM / 2.0
    return 10.0 * math.log10(power_w / 1e-3)


def _first_p1db(
    currents_a: list[float], gains_db: list[float]
) -> float:
    target = gains_db[0] - 1.0
    for index in range(1, len(gains_db)):
        previous = gains_db[index - 1]
        current = gains_db[index]
        if previous > target and current <= target:
            p0 = _current_to_dbm(currents_a[index - 1])
            p1 = _current_to_dbm(currents_a[index])
            fraction = (target - previous) / (current - previous)
            return p0 + fraction * (p1 - p0)
    raise RuntimeError(
        "signal sweep did not cross 1 dB compression: "
        f"last_current={currents_a[-1]:.6e}, last_gain={gains_db[-1]:.6f}, "
        f"small_gain={gains_db[0]:.6f}, min_gain={min(gains_db):.6f}"
    )


def solve_jpa_p1db(
    setting: ConvergenceSetting,
    *,
    device: str = "jpa",
    pump_scale: float | None = None,
    n_signal_power: int = 41,
    signal_current_min_a: float = 1e-12,
    signal_current_max_a: float = 2e-7,
) -> ConvergenceResult:
    """Solve one real nonlinear P1dB point for a convergence setting."""
    started = time.perf_counter()
    circuit, fixture_pump_current, metadata = _circuit(device)
    if pump_scale is None:
        pump_scale = {"jpa": 2.2, "jtwpa": 2.0, "fqjtwpa": 1.0}[device]
    pump_current = pump_scale * fixture_pump_current
    omega_p = 2.0 * math.pi * float(metadata["pump_freqs_ghz"][0]) * 1e9
    signal_ghz = 4.8 if device == "jpa" else 6.6
    omega_s = 2.0 * math.pi * signal_ghz * 1e9
    delta = omega_p - omega_s
    pump_modes = _pump_modes(setting.pump_policy, setting.pump_order)
    grid_nt = max(16, 2 * max(pump_modes) + 2)
    pump_problem = FullPumpProblem(
        C=circuit.C,
        G=circuit.G,
        K=circuit.K,
        Bphi=circuit.Bphi,
        branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
        grid=HarmonicGrid(np.asarray(pump_modes), grid_nt, omega_p),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=pump_current,
    )
    solver = HarmonicNewtonKrylovSolver(_settings())
    pump_state, pump_reports = solver.solve_continuation(
        pump_problem, continuation_steps=12
    )
    if not pump_reports[-1].converged:
        raise RuntimeError("pump solve did not converge")

    basis = _basis(setting, pump_modes, omega_p, delta)
    pump_basis = PumpBasis(pump_modes, setting.pump_policy, omega_p)
    initial = promote_pump_solution(pump_state, pump_basis, basis)
    pump_source = np.zeros_like(initial)
    pump_source[basis.index_of(basis.pump_tone), circuit.port_to_index[1]] = (
        0.5 * pump_current
    )
    signal_unit = MultiToneDrive(
        basis.signal_tone, circuit.port_to_index[1], 1.0
    ).to_coeffs(basis, circuit.node_count)
    currents = np.geomspace(
        signal_current_min_a, signal_current_max_a, n_signal_power
    )
    off_problem = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.signal_turn_on(
            np.zeros_like(pump_source), signal_unit * signal_current_min_a
        ),
    )
    off_state, off_report = solver.solve_one(
        off_problem, np.zeros_like(initial), 1.0
    )
    if not off_report.converged:
        raise RuntimeError("pump-off signal reference did not converge")
    signal_row = basis.index_of(basis.signal_tone)
    out_node = circuit.port_to_index[1]
    off_flux_per_amp = (
        off_state[signal_row, out_node] / signal_current_min_a
    )

    solved_currents: list[float] = []
    gains: list[float] = []
    state_by_current: dict[float, np.ndarray] = {}
    previous = initial
    previous_previous = None
    previous_current = 0.0
    previous_previous_current = 0.0
    for current in currents:
        problem = FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.signal_turn_on(
                pump_source, signal_unit * float(current)
            ),
        )
        solved = solve_signal_power_point(
            problem,
            previous,
            previous_previous,
            float(current),
            pump_source=pump_source,
            signal_source=signal_unit,
            solver=solver,
            signal_current_prev_a=previous_current,
            signal_current_prevprev_a=previous_previous_current,
            recovery="ladder",
            pump_seed=initial,
        )
        if solved.status != "VALID_SOLVED":
            break
        state = solved.state
        previous_previous = previous
        previous_previous_current = previous_current
        previous = state
        previous_current = float(current)
        state_by_current[float(current)] = state
        gain_ratio = abs(
            state[signal_row, out_node]
            / (off_flux_per_amp * float(current))
        )
        solved_currents.append(float(current))
        gains.append(20.0 * math.log10(gain_ratio))
        if gains[-1] <= gains[0] - 1.1:
            break
    bracket = None
    target = gains[0] - 1.0
    for left, right in zip(range(len(gains) - 1), range(1, len(gains))):
        if gains[left] > target >= gains[right]:
            bracket = (left, right)
            break
    if bracket is None:
        raise RuntimeError("signal sweep did not produce a P1dB bracket")

    def evaluate(power_dbm: float) -> float:
        current = math.sqrt(2.0 * 1e-3 * 10.0 ** (power_dbm / 10.0) / Z0_OHM)
        nearest = min(
            state_by_current,
            key=lambda value: abs(math.log(value / current)),
        )
        problem = FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.signal_turn_on(
                pump_source, signal_unit * float(current)
            ),
        )
        solved = solve_signal_power_point(
            problem,
            state_by_current[nearest],
            None,
            float(current),
            pump_source=pump_source,
            signal_source=signal_unit,
            solver=solver,
            signal_current_prev_a=nearest,
            recovery="ladder",
            pump_seed=initial,
        )
        if solved.status != "VALID_SOLVED":
            raise RuntimeError(f"refined P1dB solve failed at {power_dbm:.6f} dBm")
        state_by_current[float(current)] = solved.state
        gain_ratio = abs(solved.state[signal_row, out_node] / (off_flux_per_amp * current))
        gain = 20.0 * math.log10(gain_ratio)
        return target - gain

    p0 = _current_to_dbm(solved_currents[bracket[0]])
    p1 = _current_to_dbm(solved_currents[bracket[1]])
    refined_p1db = refine_p1db(
        evaluate,
        (p0, p1),
        tolerance_db=0.1,
        target_db=0.0,
    )
    pump_norm = float(np.linalg.norm(pump_state))
    even_rows = [
        row for row, mode in enumerate(pump_modes) if mode % 2 == 0
    ]
    even_fraction = (
        0.0
        if not even_rows
        else float(np.linalg.norm(pump_state[even_rows]) / pump_norm)
    )
    return ConvergenceResult(
        setting=setting,
        p1db_dbm=refined_p1db,
        small_signal_gain_db=gains[0],
        n_tones=basis.n_tones,
        n_p=int(basis.n_p),
        n_delta=int(basis.n_delta),
        even_pump_fraction=even_fraction,
        runtime_s=time.perf_counter() - started,
        p1db_method="refined",
    )


def convergence_settings() -> list[ConvergenceSetting]:
    """Return the Q, pump-order, torus, basis, and symmetry study matrix."""
    settings = [
        ConvergenceSetting("lattice", q, "odd", 3, 1)
        for q in (1, 2, 3)
    ]
    settings.extend(
        ConvergenceSetting("lattice", 3, "odd", order, 1)
        for order in (1, 2, 3)
    )
    settings.extend(
        ConvergenceSetting("lattice", 3, "odd", 3, scale)
        for scale in (1, 2)
    )
    settings.extend(
        [
            ConvergenceSetting("three_tone", 1, "odd", 1, 1),
            ConvergenceSetting("lattice", 1, "odd", 1, 1),
            ConvergenceSetting("lattice", 3, "dense", 5, 1),
            ConvergenceSetting("lattice", 3, "odd", 3, 1),
        ]
    )
    return list(dict.fromkeys(settings))


def _row(result: ConvergenceResult) -> dict[str, object]:
    setting = result.setting
    return {
        "basis_kind": setting.basis_kind,
        "signal_order_max": setting.signal_order_max,
        "pump_policy": setting.pump_policy,
        "pump_order": setting.pump_order,
        "torus_scale": setting.torus_scale,
        "n_tones": result.n_tones,
        "n_p": result.n_p,
        "n_delta": result.n_delta,
        "p1db_dbm": result.p1db_dbm,
        "small_signal_gain_db": result.small_signal_gain_db,
        "even_pump_fraction": result.even_pump_fraction,
        "runtime_s": result.runtime_s,
        "p1db_method": result.p1db_method,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-signal-power", type=int, default=41)
    parser.add_argument("--device", choices=("jtwpa", "fqjtwpa"), required=True)
    args = parser.parse_args(argv)
    results = [
        solve_jpa_p1db(
            setting, device=args.device, n_signal_power=args.n_signal_power
        )
        for setting in convergence_settings()
    ]
    rows = [_row(result) for result in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {
        "signal_order_max": [
            result
            for result in results
            if result.setting.basis_kind == "lattice"
            and result.setting.pump_policy == "odd"
            and result.setting.pump_order == 3
            and result.setting.torus_scale == 1
        ],
        "pump_order": [
            result
            for result in results
            if result.setting.basis_kind == "lattice"
            and result.setting.signal_order_max == 3
            and result.setting.pump_policy == "odd"
            and result.setting.torus_scale == 1
        ],
        "torus_scale": [
            result
            for result in results
            if result.setting.basis_kind == "lattice"
            and result.setting.signal_order_max == 3
            and result.setting.pump_policy == "odd"
            and result.setting.pump_order == 3
        ],
    }
    for knob, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda result: getattr(result.setting, knob),
        )
        delta = abs(ordered[-1].p1db_dbm - ordered[-2].p1db_dbm)
        if delta >= 0.2:
            raise RuntimeError(
                f"{knob} P1dB did not converge: delta={delta:.6f} dB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
