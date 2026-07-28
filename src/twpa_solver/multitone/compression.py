"""Signal-power continuation primitives for finite-signal solves."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.source import AffineSourcePath
from twpa_solver.multitone.compression_curve import (
    CompressionCurve,
    build_compression_curve,
    depletion_only_model,
    refine_p1db,
)


@dataclass
class SignalPowerPoint:
    signal_current_a: float
    state: np.ndarray
    report: object
    status: str
    used_recovery: str = "previous"


def _problem_with_path(problem, path: AffineSourcePath):
    if isinstance(problem, FullMultiToneProblem):
        return replace(problem, source_path=path)
    full = replace(problem.full, source_path=path)
    return type(problem)(full, problem.partition, linear_apply_mode=problem.linear_apply_mode)


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
) -> SignalPowerPoint:
    """Solve one signal-power point, using a secant predictor when available."""
    path = AffineSourcePath.signal_turn_on(pump_source, signal_source * signal_current_a)
    candidate_problem = _problem_with_path(problem, path)
    candidate = np.array(X_prev, copy=True)
    recovery = "previous"
    if X_prevprev is not None and signal_current_prev_a != signal_current_prevprev_a:
        ratio = (signal_current_a - signal_current_prev_a) / (
            signal_current_prev_a - signal_current_prevprev_a
        )
        candidate = X_prev + ratio * (X_prev - X_prevprev)
        recovery = "secant"
    state, reports = solver.solve_one(candidate_problem, candidate, 1.0)
    report = reports[-1] if isinstance(reports, list) else reports
    status = "VALID_SOLVED" if report.converged else "SIGNAL_CONTINUATION_FAILED"
    return SignalPowerPoint(signal_current_a, state, report, status, recovery)


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
