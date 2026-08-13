#!/usr/bin/env python3
"""Fixed-step diagnostics for the periodically driven normalized RCSJ.

The numerical core is deliberately small and dependency-light.  It is also
usable by the circuit-attractor route: the Lyapunov routine accepts an
arbitrary vector field and Jacobian, while the CLI supplies the normalized
Shukrinov driven pendulum.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - dependency is declared, fallback is explicit
    njit = None

Array = np.ndarray
VectorField = Callable[[float, Array], Array]
Jacobian = Callable[[float, Array], Array]


if njit is not None:
    @njit(cache=True)
    def _fast_rcsj_cvc(
        currents: Array, beta: float, omega: float, amplitude: float, h: float,
        transient_steps: int, averaging_steps: int,
    ) -> tuple[Array, Array, Array, Array]:
        """Numba implementation of the same fixed-step continuation protocol."""
        means = np.empty(currents.size, dtype=np.float64)
        final_phase = np.empty(currents.size, dtype=np.float64)
        final_velocity = np.empty(currents.size, dtype=np.float64)
        period_counts = np.zeros(currents.size, dtype=np.int64)
        phase = 0.0
        velocity = 0.0
        for index in range(currents.size):
            current = currents[index]
            total = transient_steps + averaging_steps
            velocity_sum = 0.0
            max_section = max(averaging_steps * h * omega / (2.0 * math.pi) + 4.0, 4.0)
            section = np.empty(int(max_section), dtype=np.float64)
            section_count = 0
            for step in range(total):
                t = step * h
                previous_velocity = velocity
                k1p = velocity
                k1v = current + amplitude * math.cos(omega * t) - beta * velocity - math.sin(phase)
                p2 = phase + 0.5 * h * k1p
                v2 = velocity + 0.5 * h * k1v
                k2p = v2
                k2v = current + amplitude * math.cos(omega * (t + 0.5 * h)) - beta * v2 - math.sin(p2)
                p3 = phase + 0.5 * h * k2p
                v3 = velocity + 0.5 * h * k2v
                k3p = v3
                k3v = current + amplitude * math.cos(omega * (t + 0.5 * h)) - beta * v3 - math.sin(p3)
                p4 = phase + h * k3p
                v4 = velocity + h * k3v
                k4p = v4
                k4v = current + amplitude * math.cos(omega * (t + h)) - beta * v4 - math.sin(p4)
                phase += h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
                velocity += h * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
                if step >= transient_steps:
                    velocity_sum += velocity
                    previous_cycle = math.floor(omega * t / (2.0 * math.pi))
                    current_cycle = math.floor(omega * (t + h) / (2.0 * math.pi))
                    for cycle in range(int(previous_cycle + 1), int(current_cycle + 1)):
                        if section_count < section.size:
                            target = 2.0 * math.pi * cycle / omega
                            fraction = (target - t) / h
                            section[section_count] = previous_velocity + fraction * (velocity - previous_velocity)
                            section_count += 1
            means[index] = velocity_sum / max(averaging_steps, 1)
            final_phase[index] = phase
            final_velocity[index] = velocity
            if section_count >= 4:
                span = np.max(section[:section_count]) - np.min(section[:section_count])
                scale = max(span, 1.0e-15)
                limit = min(64, section_count // 3)
                for period in range(1, limit + 1):
                    error = 0.0
                    count = section_count - period
                    for item in range(count):
                        error += abs(section[item + period] - section[item]) / scale
                    if error / count <= 0.05:
                        period_counts[index] = period
                        break
        return means, final_phase, final_velocity, period_counts
else:  # pragma: no cover
    _fast_rcsj_cvc = None


@dataclass(frozen=True)
class RCSJParameters:
    beta: float = 0.3
    omega: float = 0.5
    amplitude: float = 0.8
    h: float = 1.0 / 32.0

    def rhs(self, t: float, state: Array, dc_current: float = 0.0) -> Array:
        phase, velocity = np.asarray(state, dtype=float)
        return np.array((velocity, dc_current + self.amplitude * math.cos(self.omega * t)
                         - self.beta * velocity - math.sin(phase)), dtype=float)

    def jacobian(self, _t: float, state: Array) -> Array:
        phase = float(np.asarray(state)[0])
        return np.array(((0.0, 1.0), (-math.cos(phase), -self.beta)), dtype=float)


def rk4_step(rhs: VectorField, t: float, y: Array, h: float) -> Array:
    """One classical RK4 step, without modifying ``y``."""
    k1 = np.asarray(rhs(t, y), dtype=float)
    k2 = np.asarray(rhs(t + h / 2.0, y + h * k1 / 2.0), dtype=float)
    k3 = np.asarray(rhs(t + h / 2.0, y + h * k2 / 2.0), dtype=float)
    k4 = np.asarray(rhs(t + h, y + h * k3), dtype=float)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_rk4(
    rhs: VectorField, initial: Array, *, t0: float = 0.0, h: float = 1.0 / 32.0,
    steps: int, record_stride: int = 1,
) -> tuple[Array, Array]:
    """Integrate a vector field with the prescribed fixed RK4 step."""
    if steps < 0 or h <= 0.0 or record_stride <= 0:
        raise ValueError("steps >= 0, h > 0 and record_stride > 0 are required")
    y = np.asarray(initial, dtype=float).copy()
    count = steps // record_stride + 1
    times = np.empty(count, dtype=float)
    states = np.empty((count, y.size), dtype=float)
    times[0], states[0] = t0, y
    out = 1
    t = float(t0)
    for step in range(1, steps + 1):
        y = rk4_step(rhs, t, y, h)
        t += h
        if step % record_stride == 0:
            times[out], states[out] = t, y
            out += 1
    return times[:out], states[:out]


def lyapunov_exponents(
    rhs: VectorField, jacobian: Jacobian, initial: Array, *, h: float,
    steps: int, renormalize_every: int = 8, t0: float = 0.0,
    transient_steps: int = 0,
) -> Array:
    """Estimate all exponents with tangent RK4 and QR re-orthogonalization.

    The QR sum is accumulated in natural-log units and divided by elapsed
    physical time.  No exponent-specific fitting or convergence threshold is
    hidden in this routine, which keeps it suitable for linear validation.
    """
    y = np.asarray(initial, dtype=float).copy()
    n = y.size

    def tangent_rhs(t: float, z: Array) -> Array:
        yy = z[:n]
        tangent = z[n:].reshape(n, n)
        return np.concatenate((rhs(t, yy), (np.asarray(jacobian(t, yy)) @ tangent).ravel()))

    t = float(t0)
    for _ in range(int(transient_steps)):
        y = rk4_step(rhs, t, y, h)
        t += h
    q = np.eye(n, dtype=float)
    logs = np.zeros(n, dtype=float)
    used = 0
    for step in range(int(steps)):
        z = np.concatenate((y, q.ravel()))
        z = rk4_step(tangent_rhs, t, z, h)
        y, q = z[:n], z[n:].reshape(n, n)
        t += h
        if (step + 1) % renormalize_every == 0:
            q, r = np.linalg.qr(q)
            diagonal = np.abs(np.diag(r))
            if np.any(diagonal == 0.0):
                raise FloatingPointError("tangent basis collapsed during QR step")
            logs += np.log(diagonal)
            used += renormalize_every
    if used == 0:
        raise ValueError("steps must include at least one QR renormalization")
    return logs / (used * h)


def drive_phase_poincare(times: Array, states: Array, omega: float, *, phase: float = 0.0) -> Array:
    """Interpolate state rows at crossings of the chosen drive phase."""
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if times.size != states.shape[0] or times.size < 2:
        return np.empty((0, states.shape[1] if states.ndim == 2 else 0))
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    cycles = (omega * times - phase) / (2.0 * math.pi)
    k = np.floor(cycles).astype(np.int64)
    indices = np.flatnonzero(k[1:] > k[:-1])
    points = []
    for i in indices:
        target = phase + 2.0 * math.pi * (k[i] + 1) / omega
        fraction = (target - times[i]) / (times[i + 1] - times[i])
        points.append(states[i] + fraction * (states[i + 1] - states[i]))
    return np.asarray(points, dtype=float)


def poincare_period_count(points: Array, *, tolerance: float = 0.05) -> int:
    """Estimate the number of distinct locked points in a drive-phase section.

    The raw section contains one row per drive cycle.  For a Josephson phase
    that advances by a non-integer multiple of ``2*pi`` per cycle, clustering
    the unwrapped phase would incorrectly report every row as distinct.  This
    routine therefore clusters phase modulo ``2*pi`` together with velocity,
    using a dimensionless tolerance in each coordinate.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 2:
        return 0
    values = values[:, :2]
    phase = np.mod(values[:, 0], 2.0 * math.pi)
    velocity = values[:, 1]
    vspan = max(float(np.ptp(velocity)), 1.0e-15)
    # A repeated orbit is best identified from its sequence, not from a
    # global cluster count: the latter over-splits a noisy locked orbit and
    # cannot distinguish a drifting phase from a finite period.
    features = np.column_stack((
        np.sin(phase), np.cos(phase), (velocity - np.mean(velocity)) / vspan,
    ))
    max_period = min(64, values.shape[0] // 3)
    for period in range(1, max_period + 1):
        error = np.linalg.norm(features[period:] - features[:-period], axis=1)
        if error.size and float(np.median(error)) <= tolerance:
            return period
    return 0


def stroboscopic_period_count(values: Array, *, tolerance: float = 0.05) -> int:
    """Estimate the period of a scalar drive-phase section sequence."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return 0
    scale = max(float(np.ptp(values)), 1.0e-15)
    max_period = min(64, values.size // 3)
    for period in range(1, max_period + 1):
        error = np.abs(values[period:] - values[:-period]) / scale
        if error.size and float(np.median(error)) <= tolerance:
            return period
    return 0


def box_counting_dimension(points: Array, scales: Array | None = None) -> float:
    """Estimate a planar box-counting dimension by a log-log regression."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        return float("nan")
    values = values[:, :2] if values.shape[1] > 1 else np.column_stack((values[:, 0], np.zeros(len(values))))
    lo, hi = np.min(values, axis=0), np.max(values, axis=0)
    span = np.maximum(hi - lo, 1e-15)
    levels = np.asarray(scales if scales is not None else [4, 6, 8, 12, 16, 24, 32], dtype=int)
    counts = []
    used = []
    for boxes in levels:
        if boxes < 2:
            continue
        ij = np.floor((values - lo) / span * boxes).astype(int)
        ij = np.minimum(ij, boxes - 1)
        count = np.unique(ij, axis=0).shape[0]
        if count > 1:
            used.append(boxes); counts.append(count)
    if len(counts) < 2:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(counts), 1)[0])


def staircase_box_counting_dimension(
    currents: Array, voltages: Array, scales: Array | None = None,
) -> float:
    """Estimate the paper's dimension from the CVC staircase ``(I, <V>)``.

    This is deliberately separate from the Poincare-section dimension: the
    Shukrinov ``D`` gate is measured on the geometric CVC object after a
    high-resolution downward continuation.  Normalization is handled by the
    generic box counter, so current and voltage units do not affect the slope.
    """
    currents = np.asarray(currents, dtype=float).reshape(-1)
    voltages = np.asarray(voltages, dtype=float).reshape(-1)
    if currents.size != voltages.size:
        raise ValueError("currents and voltages must have equal length")
    finite = np.isfinite(currents) & np.isfinite(voltages)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    return box_counting_dimension(
        np.column_stack((currents[finite], voltages[finite])), scales=scales,
    )


def feigenbaum_ratios(bifurcation_parameters: Array) -> Array:
    """Return successive spacing ratios ``(d[n-1]/d[n])``."""
    values = np.asarray(bifurcation_parameters, dtype=float)
    spacings = np.abs(np.diff(values))
    if spacings.size < 2:
        return np.empty(0, dtype=float)
    return spacings[:-1] / np.maximum(spacings[1:], np.finfo(float).tiny)


def extract_period_doubling_sequence(
    control_values: Array, period_counts: Array,
) -> dict:
    """Extract resolvable period-doubling brackets from sampled orbit periods.

    The input is intentionally the measured stroboscopic period sequence, not
    a list of hand-entered bifurcation values.  Samples are ordered by control
    value before adjacent transitions are inspected, so the result is valid
    for either continuation direction.  A candidate is the midpoint of the
    first two samples that bracket an observed ``p -> 2p`` transition; it is a
    diagnostic estimate, not a root solve.
    """
    controls = np.asarray(control_values, dtype=float).reshape(-1)
    periods = np.asarray(period_counts, dtype=float).reshape(-1)
    if controls.size != periods.size:
        raise ValueError("control_values and period_counts must have equal length")
    finite = np.isfinite(controls) & np.isfinite(periods) & (periods >= 1.0)
    if np.count_nonzero(finite) < 2:
        return {
            "status": "insufficient_period_samples",
            "controls_sorted": [], "periods_sorted": [],
            "bifurcation_brackets": [], "bifurcation_parameters": [],
            "feigenbaum_ratios": [],
        }
    order = np.argsort(controls[finite], kind="stable")
    sampled_controls = controls[finite][order]
    sampled_periods = np.rint(periods[finite][order]).astype(int)
    brackets = []
    for left in range(sampled_controls.size - 1):
        p0, p1 = int(sampled_periods[left]), int(sampled_periods[left + 1])
        if p1 == 2 * p0:
            c0, c1 = float(sampled_controls[left]), float(sampled_controls[left + 1])
            brackets.append({
                "lower_control": c0,
                "upper_control": c1,
                "estimated_control": 0.5 * (c0 + c1),
                "from_period": p0,
                "to_period": p1,
            })
    candidates = np.asarray(
        [item["estimated_control"] for item in brackets], dtype=float,
    )
    return {
        "status": "sequence_detected" if brackets else "no_adjacent_doubling_detected",
        "controls_sorted": sampled_controls.tolist(),
        "periods_sorted": sampled_periods.tolist(),
        "bifurcation_brackets": brackets,
        "bifurcation_parameters": candidates.tolist(),
        "feigenbaum_ratios": feigenbaum_ratios(candidates).tolist(),
    }


def _continuation(args: argparse.Namespace) -> dict:
    params = RCSJParameters(args.beta, args.omega, args.amplitude, args.step)
    if args.currents:
        currents = np.asarray([float(item) for item in args.currents.split(",") if item.strip()], dtype=float)
        if currents.size == 0:
            raise ValueError("--currents must contain at least one comma-separated value")
    elif getattr(args, "current_step", None) is not None:
        step = abs(float(args.current_step))
        if step <= 0.0:
            raise ValueError("--current-step must be positive")
        span = float(args.stop_current) - float(args.start_current)
        count = int(round(abs(span) / step)) + 1
        if count < 2 or not np.isclose(abs(span), (count - 1) * step, rtol=0.0, atol=1e-12):
            raise ValueError("current range must be an integer multiple of --current-step")
        currents = np.linspace(args.start_current, args.stop_current, count)
    else:
        currents = np.linspace(args.start_current, args.stop_current, args.num)
    if getattr(args, "fast_cvc", False):
        if _fast_rcsj_cvc is None:
            raise RuntimeError("--fast-cvc requires numba")
        transient_steps = round(max(0.0, float(args.transient)) / params.h)
        averaging_steps = round(max(0.0, float(getattr(args, "averaging", 0.0))) / params.h)
        means, phases, velocities, period_counts = _fast_rcsj_cvc(
            np.asarray(currents, dtype=float), params.beta, params.omega,
            params.amplitude, params.h, transient_steps, averaging_steps,
        )
        rows = [{
            "I_dc": float(current), "mean_voltage": float(mean),
            "final_phase": float(phase), "final_voltage": float(velocity),
            "poincare_points": None,
            "poincare_period_count": int(period_count) if period_count > 0 else None,
            "box_dimension": None, "lyapunov_1": None, "lyapunov_2": None,
            "lyapunov_sum": None, "damping_sum_rule_error": None,
            } for current, mean, phase, velocity, period_count in zip(
                currents, means, phases, velocities, period_counts,
            )]
        return {
            "parameters": asdict(params), "direction": "downward", "rows": rows,
            "staircase_box_dimension": float(staircase_box_counting_dimension(currents, means)),
            "accelerated": True,
            "period_doubling": extract_period_doubling_sequence(
                currents, period_counts,
            ),
        }
    state = np.array((0.0, 0.0), dtype=float)
    rows = []
    for current in currents:
        rhs = lambda t, y, current=current: params.rhs(t, y, current)
        transient = max(0.0, float(args.transient))
        averaging = max(0.0, float(getattr(args, "averaging", 0.0)))
        times, states = integrate_rk4(rhs, state, h=params.h,
                                      steps=round((transient + averaging) / params.h),
                                      record_stride=max(1, round(args.record_stride / params.h)))
        state = states[-1]
        if averaging > 0.0:
            steady_mask = times >= (times[0] + transient)
            steady_times, steady = times[steady_mask], states[steady_mask]
        else:
            steady_times, steady = times[len(times) // 2:], states[len(states) // 2:]
        section = drive_phase_poincare(steady_times, steady, params.omega)
        row = {"I_dc": float(current), "mean_voltage": float(np.mean(steady[:, 1])),
               "final_phase": float(state[0]), "final_voltage": float(state[1]),
               "poincare_points": int(section.shape[0]),
               "poincare_period_count": int(stroboscopic_period_count(section[:, 1])),
               "box_dimension": box_counting_dimension(section)}
        if getattr(args, "lyapunov_steps", 0):
            exponents = lyapunov_exponents(
                rhs, lambda t, y: params.jacobian(t, y), state, h=params.h,
                steps=int(args.lyapunov_steps), renormalize_every=8,
            )
            row.update({"lyapunov_1": float(np.max(exponents)),
                        "lyapunov_2": float(np.min(exponents)),
                        "lyapunov_sum": float(np.sum(exponents)),
                        "damping_sum_rule_error": float(np.sum(exponents) + params.beta)})
        else:
            row.update({"lyapunov_1": None, "lyapunov_2": None,
                        "lyapunov_sum": None, "damping_sum_rule_error": None})
        rows.append(row)
    staircase_dimension = staircase_box_counting_dimension(
        [row["I_dc"] for row in rows], [row["mean_voltage"] for row in rows],
    )
    period_doubling = extract_period_doubling_sequence(
        [row["I_dc"] for row in rows],
        [row["poincare_period_count"] for row in rows],
    )
    return {
        "parameters": asdict(params), "direction": "downward", "rows": rows,
        "staircase_box_dimension": float(staircase_dimension),
        "period_doubling": period_doubling,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--step", type=float, default=1.0 / 32.0)
    parser.add_argument("--start-current", type=float, default=1.2)
    parser.add_argument("--stop-current", type=float, default=0.0)
    parser.add_argument("--num", type=int, default=121)
    parser.add_argument(
        "--currents", type=str, default=None,
        help="explicit comma-separated continuation currents; overrides start/stop/num",
    )
    parser.add_argument(
        "--current-step", type=float, default=None,
        help="uniform current step for high-resolution scans; overrides num",
    )
    parser.add_argument(
        "--fast-cvc", action="store_true",
        help="use numba for high-resolution CVC scans; records scalar section periods but no LEs",
    )
    parser.add_argument("--transient", type=float, default=1000.0)
    parser.add_argument("--averaging", type=float, default=10000.0,
                        help="steady-state averaging duration; use 0 for the legacy half-tail")
    parser.add_argument("--record-stride", type=float, default=1.0)
    parser.add_argument("--lyapunov-steps", type=int, default=0,
                        help="tangent steps per continuation point; 0 records no LEs")
    parser.add_argument("--output", type=Path, default=Path("outputs/chaos/phase1/rcsj.json"))
    args = parser.parse_args(argv)
    result = _continuation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "points": len(result["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
