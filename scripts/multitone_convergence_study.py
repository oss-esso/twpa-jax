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
from twpa_solver.core.nonlinear import make_branch_law
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
    status: str = "OK"
    message: str = ""


class SettingBudgetExceeded(RuntimeError):
    """One setting ran past its wall budget; the rest of the matrix continues."""


def _settings(preconditioner: str = "real_coupled") -> NewtonKrylovSettings:
    return NewtonKrylovSettings(
        newton_tol=1e-10,
        max_newton=30,
        gmres_rtol=1e-8,
        gmres_atol=0.0,
        gmres_restart=40,
        gmres_maxiter=60,
        min_alpha=1.0 / 1024.0,
        preconditioner=preconditioner,
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
    n_signal_power: int = 15,
    signal_current_min_a: float = 1e-12,
    signal_current_max_a: float = 2e-7,
    budget_s: float = 0.0,
) -> ConvergenceResult:
    """Solve one real nonlinear P1dB point for a convergence setting.

    ``budget_s > 0`` bounds the wall time spent on this setting. The coarse
    sweep and the refinement bracket both check it between solves, so a
    setting that runs long is reported as ``TIMEOUT`` instead of stalling the
    whole matrix.
    """
    started = time.perf_counter()

    def check_budget(stage: str) -> None:
        if budget_s > 0.0 and time.perf_counter() - started > budget_s:
            raise SettingBudgetExceeded(
                f"exceeded {budget_s:.0f} s budget during {stage}"
            )

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
        branch=make_branch_law(circuit),
        grid=HarmonicGrid(np.asarray(pump_modes), grid_nt, omega_p),
        pump_node_index=circuit.port_to_index[1],
        pump_current_a=pump_current,
    )
    # The pump solve keeps the plain real_coupled preconditioner so its
    # iterate path matches production exactly. The multitone solves take
    # real_coupled_fast, which is the same matrix with the scatter map and
    # symbolic factorization cached -- the study was previously refactoring
    # from scratch every Newton step, which is what made Q=3 unaffordable.
    pump_solver = HarmonicNewtonKrylovSolver(_settings("real_coupled"))
    solver = HarmonicNewtonKrylovSolver(_settings("real_coupled_fast"))
    pump_state, pump_reports = pump_solver.solve_continuation(
        pump_problem, continuation_steps=12
    )
    if not pump_reports[-1].converged:
        raise RuntimeError("pump solve did not converge")
    check_budget("pump solve")

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
        check_budget("coarse signal sweep")
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
        check_budget("P1dB refinement")
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
        "status": result.status,
        "message": result.message,
    }


FIELDNAMES = (
    "basis_kind", "signal_order_max", "pump_policy", "pump_order",
    "torus_scale", "n_tones", "n_p", "n_delta", "p1db_dbm",
    "small_signal_gain_db", "even_pump_fraction", "runtime_s",
    "p1db_method", "status", "message",
)


def _failed(
    setting: ConvergenceSetting, status: str, message: str, runtime_s: float
) -> ConvergenceResult:
    return ConvergenceResult(
        setting=setting,
        p1db_dbm=float("nan"),
        small_signal_gain_db=float("nan"),
        n_tones=0,
        n_p=0,
        n_delta=0,
        even_pump_fraction=float("nan"),
        runtime_s=runtime_s,
        p1db_method="none",
        status=status,
        message=message,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--n-signal-power",
        type=int,
        default=15,
        help=(
            "Coarse sweep points. The bracket only has to contain the "
            "crossing -- the reported P1dB comes from nonlinear refinement "
            "inside it -- so a dense coarse grid buys nothing but wall time."
        ),
    )
    parser.add_argument("--device", choices=("jtwpa", "fqjtwpa"), required=True)
    parser.add_argument(
        "--per-setting-budget-s",
        type=float,
        default=0.0,
        help=(
            "Wall budget per setting; 0 disables. A setting over budget is "
            "written as TIMEOUT and the matrix continues, so one slow corner "
            "cannot cost the whole study."
        ),
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Written incrementally: a setting that overruns must not cost the rows
    # already paid for. The previous version buffered every result and wrote
    # once at the end, so an interrupted matrix produced no CSV at all.
    results: list[ConvergenceResult] = []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        handle.flush()
        for setting in convergence_settings():
            started = time.perf_counter()
            try:
                result = solve_jpa_p1db(
                    setting,
                    device=args.device,
                    n_signal_power=args.n_signal_power,
                    budget_s=args.per_setting_budget_s,
                )
            except SettingBudgetExceeded as exc:
                result = _failed(
                    setting, "TIMEOUT", str(exc), time.perf_counter() - started
                )
            except RuntimeError as exc:
                result = _failed(
                    setting, "FAILED", str(exc), time.perf_counter() - started
                )
            except MemoryError as exc:
                # Not a RuntimeError, so it would otherwise escape and kill the
                # matrix. The large-torus corners of this study genuinely do not
                # fit on a 7 GB machine; that is a result about the setting, not
                # a reason to lose the settings already solved.
                result = _failed(
                    setting, "OOM", str(exc) or "out of memory",
                    time.perf_counter() - started,
                )
            results.append(result)
            writer.writerow(_row(result))
            handle.flush()
            print(
                f"{setting} -> {result.status} "
                f"p1db={result.p1db_dbm:.6f} dBm "
                f"runtime={result.runtime_s:.1f} s",
                flush=True,
            )

    results = [result for result in results if result.status == "OK"]
    if not results:
        raise RuntimeError("no convergence setting completed; see the CSV")

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
    # Every knob's number is printed before anything raises. A study that
    # reports only the first failing knob makes the run worth repeating; one
    # that reports all of them makes it worth reading.
    failures: list[str] = []
    for knob, values in grouped.items():
        ordered = sorted(values, key=lambda result: getattr(result.setting, knob))
        if len(ordered) < 2:
            print(
                f"acceptance {knob}: SKIPPED, only {len(ordered)} completed "
                "setting(s) -- gate not evaluated",
                flush=True,
            )
            failures.append(f"{knob}: incomplete ({len(ordered)} settings)")
            continue
        delta = abs(ordered[-1].p1db_dbm - ordered[-2].p1db_dbm)
        verdict = "PASS" if delta < 0.2 else "FAIL"
        print(
            f"acceptance {knob}: |dP1dB|={delta:.6f} dB between "
            f"{getattr(ordered[-2].setting, knob)} and "
            f"{getattr(ordered[-1].setting, knob)} -> {verdict}",
            flush=True,
        )
        if delta >= 0.2:
            failures.append(f"{knob}: delta={delta:.6f} dB")
    if failures:
        raise RuntimeError(
            "P1dB basis convergence not accepted: " + "; ".join(failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
