"""Signal-power continuation primitives for finite-signal solves."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath
from twpa_solver.multitone.compression_curve import (
    CompressionCurve,
    build_compression_curve,
    depletion_only_model,
    refine_p1db,
)


# Fixed-step count for the continuation fallback, matching the cold-start path.
_FALLBACK_FIXED_STEPS = 20


@dataclass
class SignalPowerPoint:
    signal_current_a: float
    state: np.ndarray
    report: object
    status: str
    used_recovery: str = "previous"
    last_converged_signal_current_a: float = 0.0


def _problem_with_path(problem, path: AffineSourcePath):
    if isinstance(problem, FullMultiToneProblem):
        return replace(problem, source_path=path)
    full = replace(problem.full, source_path=path)
    # The partition carries the cached Schur complements and the cached fast
    # preconditioner scatter/symbolic factorization, so rebuilding the problem
    # per power point is cheap. ``preconditioner`` must be carried across or the
    # rebuilt problem silently reverts to the default.
    return type(problem)(
        full,
        problem.partition,
        linear_apply_mode=problem.linear_apply_mode,
        preconditioner=problem.preconditioner,
    )


def solve_signal_power_point(
    problem,
    X_prev: np.ndarray,
    X_prevprev: np.ndarray | None,
    signal_current_a: float,
    *,
    pump_source: np.ndarray,
    signal_source: np.ndarray,
    solver,
    signal_current_prev_a: float = 0.0,
    signal_current_prevprev_a: float = 0.0,
    recovery: str = "plain",
    pump_seed: np.ndarray | None = None,
    signal_substep_init_db: float = 0.5,
    signal_substep_min_db: float = 0.01,
    continuation_deadline_s: float = 0.0,
    arclength_recovery: bool = False,
) -> SignalPowerPoint:
    """Solve one signal-power point with the selected recovery ladder."""
    if recovery not in {"plain", "ladder"}:
        raise ValueError(f"unknown signal recovery mode {recovery!r}")
    path = AffineSourcePath.signal_turn_on(pump_source, signal_source * signal_current_a)
    candidate_problem = _problem_with_path(problem, path)
    if recovery == "ladder" and signal_current_prev_a <= 0.0:
        state, reports, trace = solver.solve_adaptive_continuation(
            candidate_problem,
            None,
            initial_step=0.25,
            min_step=0.01,
            growth=1.5,
            shrink=0.5,
            fallback_fixed_steps=20,
            max_wall_s=continuation_deadline_s,
        )
        report = reports[-1]
        status = _failure_status(state, report, trace.failure_reason)
        return SignalPowerPoint(
            signal_current_a,
            state,
            report,
            status,
            "adaptive_first_point",
            signal_current_a if status == "VALID_SOLVED" else 0.0,
        )

    state, report = solver.solve_one(
        candidate_problem,
        np.array(X_prev, copy=True),
        1.0,
    )
    if report.converged:
        return SignalPowerPoint(
            signal_current_a,
            state,
            report,
            "VALID_SOLVED",
            "previous",
            signal_current_a,
        )

    if recovery == "plain" and X_prevprev is not None and signal_current_prev_a != signal_current_prevprev_a:
        ratio = (signal_current_a - signal_current_prev_a) / (
            signal_current_prev_a - signal_current_prevprev_a
        )
        candidate = X_prev + ratio * (X_prev - X_prevprev)
        state, report = solver.solve_one(candidate_problem, candidate, 1.0)
        status = _failure_status(state, report, report.failure_reason)
        return SignalPowerPoint(
            signal_current_a,
            state,
            report,
            status,
            "secant",
            signal_current_a if status == "VALID_SOLVED" else signal_current_prev_a,
        )
    if recovery == "plain":
        status = _failure_status(state, report, report.failure_reason)
        return SignalPowerPoint(
            signal_current_a,
            state,
            report,
            status,
            "previous",
            signal_current_prev_a,
        )

    seed = np.asarray(pump_seed if pump_seed is not None else X_prev)
    if signal_current_prev_a > 0.0:
        seed = seed + (signal_current_a / signal_current_prev_a) * (X_prev - seed)
    state, report = solver.solve_one(candidate_problem, seed, 1.0)
    if report.converged:
        return SignalPowerPoint(
            signal_current_a, state, report, "VALID_SOLVED", "rescaled_floquet", signal_current_a
        )

    substep_path = AffineSourcePath.signal_substep(
        pump_source,
        signal_source * signal_current_prev_a,
        signal_source * signal_current_a,
    )
    substep_problem = _problem_with_path(problem, substep_path)
    span_db = 20.0 * math.log10(signal_current_a / signal_current_prev_a)
    initial_step = min(1.0, signal_substep_init_db / max(span_db, 1e-12))
    min_step = min(initial_step, signal_substep_min_db / max(span_db, 1e-12))
    state, reports, trace = solver.solve_adaptive_continuation(
        substep_problem,
        X_prev,
        initial_step=initial_step,
        min_step=min_step,
        growth=1.5,
        shrink=0.5,
        # Sized like the cold-start path, NOT as span_db / min_step: the latter
        # makes the fallback finest exactly when adaptive stepping has already
        # failed, scheduling thousands of micro-steps that only the wall-clock
        # deadline ever stops (measured: 6881 steps, ~29 h, for one power point).
        fallback_fixed_steps=_FALLBACK_FIXED_STEPS,
        max_wall_s=continuation_deadline_s,
    )
    report = reports[-1]
    if report.converged and trace.accepted_lambdas and trace.accepted_lambdas[-1] >= 1.0 - 1e-12:
        return SignalPowerPoint(
            signal_current_a, state, report, "VALID_SOLVED", "signal_substep", signal_current_a
        )
    last_lambda = trace.accepted_lambdas[-1] if trace.accepted_lambdas else 0.0
    last_current = signal_current_prev_a + last_lambda * (
        signal_current_a - signal_current_prev_a
    )
    if arclength_recovery:
        state_arc, _lambda_arc, info = solver.solve_arclength(
            substep_problem,
            X_prev,
            0.0,
            target_lam=1.0,
            max_wall_s=continuation_deadline_s,
        )
        if info["reached_target"] and np.all(np.isfinite(state_arc)):
            return SignalPowerPoint(
                signal_current_a,
                state_arc,
                report,
                "VALID_SOLVED",
                "arclength",
                signal_current_a,
            )
        if info["terminal_reason"] == "deadline":
            return SignalPowerPoint(
                signal_current_a, state_arc, report, "DEADLINE", "arclength", last_current
            )
    status = _failure_status(state, report, trace.failure_reason)
    if status == "SIGNAL_CONTINUATION_FAILED" and last_lambda > 0.0:
        status = "SIGNAL_SUBSTEP_STALL"
    return SignalPowerPoint(
        signal_current_a, state, report, status, "signal_substep", last_current
    )


def _failure_status(state: np.ndarray, report: object, reason: str) -> str:
    if not np.all(np.isfinite(state)):
        return "NONFINITE_STATE"
    text = f"{getattr(report, 'failure_reason', '')} {reason}".lower()
    if "deadline" in text or "exceeded" in text:
        return "DEADLINE"
    return "VALID_SOLVED" if getattr(report, "converged", False) else "SIGNAL_CONTINUATION_FAILED"


def run_compression_sweep(
    solve_point,
    signal_currents_a: list[float],
    *,
    small_signal_gain_db: float,
) -> CompressionCurve:
    """Run a sequential signal-current sweep through a caller-provided solver."""
    gains: list[float] = []
    previous = previous_previous = None
    previous_current = previous_previous_current = 0.0
    for current in signal_currents_a:
        point = solve_point(
            float(current), previous, previous_previous,
            previous_current, previous_previous_current,
        )
        gains.append(float(point.gain_db))
        previous_previous = previous
        previous_previous_current = previous_current
        previous = point.state
        previous_current = float(current)
    # The public curve representation uses dBm; callers can pass currents in
    # dBm-equivalent units when they need a pure numerical sweep.
    return build_compression_curve(
        [float(value) for value in signal_currents_a], gains, small_signal_gain_db
    )
