"""Opt-in nonlinear correction and continuation for a period-doubled pump."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from twpa_solver.core.linear import default_loss_model_for
from twpa_solver.pump.floquet import build_period_doubled_seed
from twpa_solver.pump.diagnostics import branch_stress_metrics
from twpa_solver.pump.problem import FullPumpProblem, HarmonicGrid
from twpa_solver.pump.solver import HarmonicNewtonKrylovSolver, StepReport


@dataclass(frozen=True)
class PeriodDoubledCorrection:
    """Result of correcting one or more Floquet-derived nonlinear seeds."""

    state: np.ndarray
    report: StepReport
    perturbation_amplitude: float
    perturbation_sign: float
    attempted: int


@dataclass(frozen=True)
class PeriodDoubledContinuation:
    """Continuation result in physical current ratio ``mu=I/I_ref``."""

    state: np.ndarray
    mu: float
    info: dict[str, Any]


def build_period_doubled_problem(
    circuit: Any,
    branch: Any,
    basis: Any,
    *,
    pump_current_a: float,
    pump_port: int,
    dc_branch_flux: np.ndarray | None = None,
    nt: int | None = None,
    environment: Any | None = None,
    loss_model: str | None = None,
) -> FullPumpProblem:
    """Build the production HB problem for a half-pump basis.

    The physical pump source is placed at ``basis.source_mode`` (normally
    mode two), not at mode one.  No circuit parameter is changed.
    """
    max_mode = int(max(basis.modes))
    sample_nt = max(2 * max_mode + 2, 40) if nt is None else int(nt)
    if sample_nt % 2:
        sample_nt += 1
    grid = HarmonicGrid(
        modes=np.asarray(basis.modes, dtype=int),
        nt=sample_nt,
        omega=float(basis.omega_p),
    )
    return FullPumpProblem(
        circuit.C,
        circuit.G,
        circuit.K,
        circuit.Bphi,
        branch,
        grid,
        circuit.port_to_index[int(pump_port)],
        float(pump_current_a),
        dc_branch_flux=dc_branch_flux,
        environment=environment,
        source_mode=int(basis.source_mode),
        loss_model=loss_model or default_loss_model_for(circuit),
    )


def correct_period_doubled_seed(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    pump_state: np.ndarray,
    pump_basis: Any,
    floquet_vector: np.ndarray,
    floquet_sidebands: list[int] | np.ndarray,
    target_basis: Any,
    *,
    amplitudes: tuple[float, ...] = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5),
    signs: tuple[float, ...] = (1.0, -1.0),
) -> PeriodDoubledCorrection | None:
    """Try signed Floquet perturbations and correct them with production HB."""
    if not amplitudes or any(value <= 0.0 for value in amplitudes):
        raise ValueError("amplitudes must contain positive values")
    if not signs or any(value == 0.0 for value in signs):
        raise ValueError("signs must contain nonzero values")
    attempted = 0
    for amplitude in amplitudes:
        for sign in signs:
            attempted += 1
            seed = build_period_doubled_seed(
                pump_state,
                pump_basis,
                floquet_vector,
                floquet_sidebands,
                target_basis,
                perturbation_amplitude=float(amplitude),
                perturbation_sign=float(sign),
            )
            state, report = solver.solve_one(problem, seed, 1.0)
            if report.converged:
                return PeriodDoubledCorrection(
                    state=state,
                    report=report,
                    perturbation_amplitude=float(amplitude),
                    perturbation_sign=float(sign),
                    attempted=attempted,
                )
    return None


def continue_period_doubled_branch(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    correction: PeriodDoubledCorrection,
    *,
    i_ref_a: float,
    mu_start: float = 1.0,
    mu_target: float = 1.5,
    ds: float = 0.02,
    max_steps: int = 200,
    max_wall_s: float = 0.0,
) -> PeriodDoubledContinuation:
    """Continue a corrected period-doubled state in physical pump current."""
    if i_ref_a <= 0.0:
        raise ValueError("i_ref_a must be positive")
    if mu_target == mu_start:
        return PeriodDoubledContinuation(correction.state, mu_start, {"reached_target": True})
    state, mu, info = solver.solve_arclength_mu(
        problem,
        correction.state,
        float(mu_start),
        i_ref=float(i_ref_a),
        target_mu=float(mu_target),
        ds=float(ds),
        max_steps=int(max_steps),
        max_wall_s=float(max_wall_s),
    )
    return PeriodDoubledContinuation(state=state, mu=float(mu), info=info)


def continue_until_utilization(
    solver: HarmonicNewtonKrylovSolver,
    problem: FullPumpProblem,
    correction: PeriodDoubledCorrection,
    *,
    i_ref_a: float,
    mu_start: float = 1.0,
    mu_max: float = 2.0,
    mu_step: float = 0.02,
    break_threshold: float = 0.999999,
    ds: float = 0.01,
    max_steps_per_segment: int = 30,
    max_wall_s: float = 0.0,
) -> PeriodDoubledContinuation:
    """Advance in short PALC segments until utilization or solver failure.

    Short segments make the junction-current stopping gate observable without
    changing the circuit equations.  The returned ``info`` contains every
    accepted endpoint and the exact stress metrics used by the stopping rule.
    """
    if not 0.0 < mu_step:
        raise ValueError("mu_step must be positive")
    if break_threshold <= 0.0:
        raise ValueError("break_threshold must be positive")
    if mu_max < mu_start:
        raise ValueError("mu_max must be >= mu_start")
    state = np.array(correction.state, dtype=np.complex128, copy=True)
    mu = float(mu_start)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    terminal_reason = "target_mu_reached"
    while mu < mu_max - 1.0e-12:
        if max_wall_s > 0.0 and time.perf_counter() - started > max_wall_s:
            terminal_reason = "wall_time_limit"
            break
        target = min(mu_max, mu + mu_step)
        segment = PeriodDoubledCorrection(
            state=state,
            report=correction.report,
            perturbation_amplitude=correction.perturbation_amplitude,
            perturbation_sign=correction.perturbation_sign,
            attempted=correction.attempted,
        )
        result = continue_period_doubled_branch(
            solver,
            problem,
            segment,
            i_ref_a=i_ref_a,
            mu_start=mu,
            mu_target=target,
            ds=min(ds, max(target - mu, 1.0e-6)),
            max_steps=max_steps_per_segment,
            max_wall_s=max_wall_s,
        )
        state = result.state
        mu = result.mu
        stress = branch_stress_metrics(problem, state)
        history.append({"mu": mu, "stress": stress, "info": result.info})
        if stress["branch_current_max_over_ic_all"] is not None and stress[
            "branch_current_max_over_ic_all"
        ] >= break_threshold:
            terminal_reason = "junction_utilization_reached"
            break
        if not result.info.get("reached_target", False):
            terminal_reason = "continuation_failed_or_stopped"
            break
    info = {
        "reached_target": bool(mu >= mu_max - 1.0e-12),
        "terminal_reason": terminal_reason,
        "history": history,
        "max_utilization": max(
            (
                item["stress"]["branch_current_max_over_ic_all"]
                for item in history
                if item["stress"]["branch_current_max_over_ic_all"] is not None
            ),
            default=None,
        ),
    }
    return PeriodDoubledContinuation(state=state, mu=mu, info=info)
