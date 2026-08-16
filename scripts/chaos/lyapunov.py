"""Benettin largest-Lyapunov measurements for reference systems and Guarcello.

The module contains small, explicit RK4 and map propagators for validation.
The Guarcello path imports the existing paper TD kernel's coefficients and
device model read-only, then propagates the exact tangent of its known-time-
level recurrence alongside the state.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Array = np.ndarray
Flow = Callable[[float, Array], Array]
Jacobian = Callable[[float, Array], Array]
Map = Callable[[Array], Array]
MapJacobian = Callable[[Array], Array]


def rk4_tangent_step(
    rhs: Flow, jacobian: Jacobian, t: float, state: Array, tangent: Array, h: float,
) -> tuple[Array, Array]:
    """Advance state and one tangent vector with the same RK4 tableau."""
    n = state.size

    def combined(time: float, value: Array) -> Array:
        y = value[:n]
        delta = value[n:]
        return np.concatenate((rhs(time, y), jacobian(time, y) @ delta))

    z = np.concatenate((state, tangent))
    k1 = combined(t, z)
    k2 = combined(t + h / 2.0, z + h * k1 / 2.0)
    k3 = combined(t + h / 2.0, z + h * k2 / 2.0)
    k4 = combined(t + h, z + h * k3)
    result = z + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return result[:n], result[n:]


def benettin_flow(
    rhs: Flow,
    jacobian: Jacobian,
    initial: Array,
    *,
    h: float,
    steps: int,
    renormalize_every: int = 8,
    transient_steps: int = 0,
    checkpoint_steps: int | None = None,
    seed: int = 0,
) -> dict[str, Array | float | int]:
    """Estimate the largest flow exponent and return its convergence curve."""
    if h <= 0.0 or steps <= 0 or renormalize_every <= 0:
        raise ValueError("h, steps, and renormalize_every must be positive")
    state = np.asarray(initial, dtype=float).copy()
    t = 0.0
    for _ in range(transient_steps):
        state, _ = rk4_tangent_step(rhs, jacobian, t, state, np.zeros_like(state), h)
        t += h
    tangent = np.random.default_rng(seed).normal(size=state.size)
    tangent /= np.linalg.norm(tangent)
    checkpoint = checkpoint_steps or max(renormalize_every, steps // 20)
    logs = 0.0
    used = 0
    curve_t: list[float] = []
    curve_lambda: list[float] = []
    for step in range(1, steps + 1):
        state, tangent = rk4_tangent_step(rhs, jacobian, t, state, tangent, h)
        t += h
        if step % renormalize_every == 0:
            norm = float(np.linalg.norm(tangent))
            if not math.isfinite(norm) or norm <= 0.0:
                raise FloatingPointError("tangent norm collapsed or diverged")
            logs += math.log(norm)
            tangent /= norm
            used += renormalize_every
        if step % checkpoint == 0 or step == steps:
            if used:
                curve_t.append(used * h)
                curve_lambda.append(logs / (used * h))
    return {
        "lambda_1": float(curve_lambda[-1]),
        "integration_time": float(used * h),
        "renormalize_every": renormalize_every,
        "h": h,
        "curve_time": np.asarray(curve_t, dtype=float),
        "curve_lambda": np.asarray(curve_lambda, dtype=float),
        "final_state": state,
    }


def benettin_map(
    mapping: Map,
    jacobian: MapJacobian,
    initial: Array,
    *,
    steps: int,
    transient_steps: int = 0,
    renormalize_every: int = 1,
    checkpoint_steps: int | None = None,
    seed: int = 0,
) -> dict[str, Array | float | int]:
    """Estimate the largest exponent of a discrete map per map iteration."""
    if steps <= 0 or renormalize_every <= 0:
        raise ValueError("steps and renormalize_every must be positive")
    state = np.asarray(initial, dtype=float).copy()
    for _ in range(transient_steps):
        state = np.asarray(mapping(state), dtype=float)
    tangent = np.random.default_rng(seed).normal(size=state.size)
    tangent /= np.linalg.norm(tangent)
    logs = 0.0
    used = 0
    checkpoint = checkpoint_steps or max(renormalize_every, steps // 20)
    curve_t: list[float] = []
    curve_lambda: list[float] = []
    for step in range(1, steps + 1):
        tangent = np.asarray(jacobian(state), dtype=float) @ tangent
        state = np.asarray(mapping(state), dtype=float)
        if step % renormalize_every == 0:
            norm = float(np.linalg.norm(tangent))
            if not math.isfinite(norm) or norm <= 0.0:
                raise FloatingPointError("map tangent norm collapsed or diverged")
            logs += math.log(norm)
            tangent /= norm
            used += renormalize_every
        if step % checkpoint == 0 or step == steps:
            if used:
                curve_t.append(float(used))
                curve_lambda.append(logs / used)
    return {
        "lambda_1": float(curve_lambda[-1]),
        "integration_time": float(used),
        "renormalize_every": renormalize_every,
        "curve_time": np.asarray(curve_t, dtype=float),
        "curve_lambda": np.asarray(curve_lambda, dtype=float),
        "final_state": state,
    }


def _reference_cases() -> list[tuple[str, float, Callable[[], dict[str, Array | float | int]], float]]:
    """Return the four mandatory reference cases and literature values."""
    def lorenz() -> dict[str, Array | float | int]:
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        rhs = lambda _t, y: np.array((sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]))
        jac = lambda _t, y: np.array(((-sigma, sigma, 0.0), (rho - y[2], -1.0, -y[0]), (y[1], y[0], -beta)))
        return benettin_flow(rhs, jac, np.array((1.0, 1.0, 1.0)), h=0.01, steps=300_000, transient_steps=10_000, checkpoint_steps=15_000, renormalize_every=8)

    def henon() -> dict[str, Array | float | int]:
        a, b = 1.4, 0.3
        mapping = lambda y: np.array((1.0 - a * y[0] ** 2 + y[1], b * y[0]))
        jac = lambda y: np.array(((-2.0 * a * y[0], 1.0), (b, 0.0)))
        return benettin_map(mapping, jac, np.array((0.1, 0.1)), steps=300_000, transient_steps=1_000, checkpoint_steps=15_000)

    def rossler() -> dict[str, Array | float | int]:
        a, b, c = 0.2, 0.2, 5.7
        rhs = lambda _t, y: np.array((-y[1] - y[2], y[0] + a * y[1], b + y[2] * (y[0] - c)))
        jac = lambda _t, y: np.array(((0.0, -1.0, -1.0), (1.0, a, 0.0), (y[2], 0.0, y[0] - c)))
        return benettin_flow(rhs, jac, np.array((1.0, 1.0, 1.0)), h=0.02, steps=500_000, transient_steps=10_000, checkpoint_steps=25_000, renormalize_every=8)

    def oscillator() -> dict[str, Array | float | int]:
        omega, damping, drive = 1.0, 0.1, 0.7
        rhs = lambda t, y: np.array((y[1], -omega**2 * y[0] - 2.0 * damping * omega * y[1] + drive * math.cos(t)))
        jac = lambda _t, _y: np.array(((0.0, 1.0), (-omega**2, -2.0 * damping * omega)))
        return benettin_flow(rhs, jac, np.array((0.0, 0.0)), h=0.01, steps=100_000, transient_steps=1_000, checkpoint_steps=5_000, renormalize_every=8)

    return [
        ("Lorenz", 0.906, lorenz, 0.05),
        ("Henon", 0.419, henon, 0.05),
        ("Rossler", 0.0714, rossler, 0.05),
        ("damped_driven_linear_oscillator", -0.1, oscillator, 0.05),
    ]


def validate_references() -> dict[str, object]:
    """Run and gate all mandatory reference systems."""
    rows: list[dict[str, object]] = []
    for name, literature, runner, tolerance in _reference_cases():
        result = runner()
        measured = float(result["lambda_1"])
        relative_error = abs(measured - literature) / max(abs(literature), 1e-300)
        rows.append({
            "system": name,
            "literature_lambda_1": literature,
            "measured_lambda_1": measured,
            "relative_error": relative_error,
            "pass": relative_error <= tolerance,
            "renormalize_every": int(result["renormalize_every"]),
            "curve_time": np.asarray(result["curve_time"]).tolist(),
            "curve_lambda": np.asarray(result["curve_lambda"]).tolist(),
        })
    return {"status": "PASS" if all(bool(row["pass"]) for row in rows) else "NOT_ESTABLISHED", "rows": rows}


def _load_paper_module() -> object:
    path = Path(__file__).resolve().parents[2] / "docs/development/chaos_papers/guarcello_jtwpa_fdtd.py"
    spec = importlib.util.spec_from_file_location("guarcello_jtwpa_fdtd_for_lyapunov", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load existing Guarcello kernel: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class GuarcelloTangentConfig:
    pump_dbm: float
    pump_ghz: float = 7.0
    dt_norm: float = 0.01
    signal_dbm: float = -300.0
    transient_steps: int = 40_000
    steps: int = 200_000
    renormalize_every: int = 20
    port_update: str = "stable"


def _guarcello_step_factory(config: GuarcelloTangentConfig) -> tuple[Callable[[float, Array, Array], tuple[Array, Array]], float]:
    """Build a state/tangent step from the existing Guarcello recurrence."""
    paper = _load_paper_module()
    device = paper.Device()
    dt_s = config.dt_norm / device.omega_plasma
    coeff = paper.build_coefficients(device, dt_s)
    n = device.n_cells
    cminus = np.asarray(coeff["cminus"])
    alpha_p, alpha_m = np.asarray(coeff["alpha_p"]), np.asarray(coeff["alpha_m"])
    at_p, at_m = np.asarray(coeff["at_p"]), np.asarray(coeff["at_m"])
    fdiag, ft_diag = float(coeff["fdiag"]), np.asarray(coeff["ft_diag"])
    upper = np.asarray(coeff["upper"])
    lu_diag, lu_mult = np.asarray(coeff["lu_diag"]), np.asarray(coeff["lu_mult"])
    pump_w = 2.0 * math.pi * config.pump_ghz * 1e9
    signal_w = 2.0 * math.pi * config.pump_ghz * 1e9
    pump_vpk = paper.dbm_to_vpk(config.pump_dbm, "50ohm", device.ri_ohm)
    signal_vpk = paper.dbm_to_vpk(config.signal_dbm, "50ohm", device.ri_ohm)
    bias_a = 0.0
    tau = -1.0
    stable = config.port_update == "stable"
    omega_i = 1.0 / (device.ri_ohm * device.ci_f)
    ei = math.exp(-omega_i * dt_s)
    a_l = (1.0 + device.cg_f / device.cl_f) / (device.cg_f * device.rl_ohm)
    el = math.exp(-a_l * dt_s)
    ic = device.ic_a

    def cpr(phi: Array) -> tuple[Array, Array]:
        if tau > 0.0:
            raise ValueError("the tangent path currently supports the sinusoidal CPR only")
        return ic * np.sin(phi), ic * np.cos(phi)

    def solve(rhs: Array) -> Array:
        y = np.empty(n, dtype=float)
        result = np.empty(n, dtype=float)
        y[0] = rhs[0]
        for index in range(1, n):
            y[index] = rhs[index] - lu_mult[index - 1] * y[index - 1]
        result[-1] = y[-1] / lu_diag[-1]
        for index in range(n - 2, -1, -1):
            result[index] = (y[index] - upper[index] * result[index + 1]) / lu_diag[index]
        return result

    def step(time: float, state: Array, tangent: Array) -> tuple[Array, Array]:
        phi_prev, phi_cur = state[:n], state[n:2 * n]
        ii_cur, il_cur = float(state[-2]), float(state[-1])
        dprev, dcur = tangent[:n], tangent[n:2 * n]
        dii_cur, dil_cur = float(tangent[-2]), float(tangent[-1])
        current, current_derivative = cpr(phi_cur)
        f = fdiag * phi_cur + current
        ft = ft_diag * phi_cur + current
        df = fdiag * dcur + current_derivative * dcur
        dft = ft_diag * dcur + current_derivative * dcur
        rhs = np.empty(n, dtype=float)
        drhs = np.empty(n, dtype=float)
        if n == 1:
            rhs[0] = cminus[0] * ii_cur - (1 + cminus[0]) * ft[0]
            rhs[0] += -(1 + cminus[0]) * at_m[0] * phi_prev[0] + il_cur + 2 * bias_a
            drhs[0] = cminus[0] * dii_cur - (1 + cminus[0]) * dft[0]
            drhs[0] += -(1 + cminus[0]) * at_m[0] * dprev[0] + dil_cur
        else:
            rhs[0] = cminus[0] * ii_cur - (1 + cminus[0]) * ft[0] + f[1]
            rhs[0] += -(1 + cminus[0]) * at_m[0] * phi_prev[0] + alpha_m[1] * phi_prev[1]
            drhs[0] = cminus[0] * dii_cur - (1 + cminus[0]) * dft[0] + df[1]
            drhs[0] += -(1 + cminus[0]) * at_m[0] * dprev[0] + alpha_m[1] * dprev[1]
            for index in range(1, n - 1):
                cm = cminus[index]
                rhs[index] = cm * f[index - 1] - (1 + cm) * ft[index] + f[index + 1]
                rhs[index] += cm * alpha_m[index - 1] * phi_prev[index - 1] - (1 + cm) * at_m[index] * phi_prev[index] + alpha_m[index + 1] * phi_prev[index + 1]
                drhs[index] = cm * df[index - 1] - (1 + cm) * dft[index] + df[index + 1]
                drhs[index] += cm * alpha_m[index - 1] * dprev[index - 1] - (1 + cm) * at_m[index] * dprev[index] + alpha_m[index + 1] * dprev[index + 1]
            index = n - 1
            cm = cminus[index]
            rhs[index] = cm * f[index - 1] - (1 + cm) * ft[index] + il_cur
            rhs[index] += cm * alpha_m[index - 1] * phi_prev[index - 1] - (1 + cm) * at_m[index] * phi_prev[index] + bias_a
            drhs[index] = cm * df[index - 1] - (1 + cm) * dft[index] + dil_cur
            drhs[index] += cm * alpha_m[index - 1] * dprev[index - 1] - (1 + cm) * at_m[index] * dprev[index]
        phi_next = solve(rhs)
        dnext = solve(drhs)
        i1 = alpha_p[0] * phi_next[0] + f[0] + alpha_m[0] * phi_prev[0]
        inow = alpha_p[-1] * phi_next[-1] + f[-1] + alpha_m[-1] * phi_prev[-1]
        di1 = alpha_p[0] * dnext[0] + df[0] + alpha_m[0] * dprev[0]
        dinow = alpha_p[-1] * dnext[-1] + df[-1] + alpha_m[-1] * dprev[-1]
        vdot = pump_vpk * pump_w * math.cos(pump_w * time)
        vdot += signal_vpk * signal_w * math.cos(signal_w * time)
        if stable:
            ii_next = ei * ii_cur + (1 - ei) * (i1 + device.ci_f * vdot - bias_a)
            il_next = el * il_cur + (1 - el) * (inow - bias_a) / (1 + device.cg_f / device.cl_f)
            dii_next = ei * dii_cur + (1 - ei) * di1
            dil_next = el * dil_cur + (1 - el) * dinow / (1 + device.cg_f / device.cl_f)
        else:
            raise ValueError("paper-centered port tangent is intentionally unsupported")
        return (
            np.concatenate((phi_cur, phi_next, np.array((ii_next, il_next)))),
            np.concatenate((dcur, dnext, np.array((dii_next, dil_next)))),
        )
    return step, dt_s


def run_guarcello(config: GuarcelloTangentConfig) -> dict[str, object]:
    """Run one Guarcello point and return its length convergence curve."""
    step, dt_s = _guarcello_step_factory(config)
    paper = _load_paper_module()
    n = paper.Device().n_cells
    state = np.zeros(2 * n + 2, dtype=float)
    tangent = np.random.default_rng(0).normal(size=state.size)
    tangent /= np.linalg.norm(tangent)
    for transient_index in range(config.transient_steps):
        state, _ = step(transient_index * dt_s, state, np.zeros_like(state))
    logs = 0.0
    curve_t: list[float] = []
    curve_lambda: list[float] = []
    used = 0
    checkpoint = max(config.renormalize_every, config.steps // 20)
    for index in range(1, config.steps + 1):
        time = (config.transient_steps + index) * dt_s
        state, tangent = step(time, state, tangent)
        if index % config.renormalize_every == 0:
            norm = float(np.linalg.norm(tangent))
            if not math.isfinite(norm) or norm <= 0.0:
                raise FloatingPointError("Guarcello tangent norm collapsed or diverged")
            logs += math.log(norm)
            tangent /= norm
            used += config.renormalize_every
        if index % checkpoint == 0 or index == config.steps:
            curve_t.append(used * dt_s)
            curve_lambda.append(logs / max(used * dt_s, 1e-300))
    return {
        "pump_power_dbm": config.pump_dbm,
        "lambda_1": curve_lambda[-1],
        "integration_time_s": curve_t[-1],
        "integration_time_pump_periods": curve_t[-1] * config.pump_ghz * 1e9,
        "renormalize_every": config.renormalize_every,
        "dt_norm": config.dt_norm,
        "curve_time_s": curve_t,
        "curve_lambda_1": curve_lambda,
        "status": "MEASURED",
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/chaos/lyapunov/lyapunov.json"))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--transient-steps", type=int, default=40_000)
    parser.add_argument("--renormalize-every", type=int, default=20)
    args = parser.parse_args(argv)
    validation = validate_references()
    payload: dict[str, object] = {"validation": validation}
    if validation["status"] != "PASS" or args.validate_only:
        payload["device_runs"] = []
        _atomic_json(args.output, payload)
        print(json.dumps({"output": str(args.output), "status": validation["status"]}))
        return 0 if validation["status"] == "PASS" else 2
    powers = (-55.0, -54.0461, -53.8816)
    payload["device_runs"] = [
        run_guarcello(GuarcelloTangentConfig(
            pump_dbm=power, steps=args.steps, transient_steps=args.transient_steps,
            renormalize_every=args.renormalize_every,
        )) for power in powers
    ]
    payload["renormalization_sensitivity"] = [
        {
            "renormalize_every": interval,
            "points": [
                run_guarcello(GuarcelloTangentConfig(
                    pump_dbm=power, steps=args.steps, transient_steps=args.transient_steps,
                    renormalize_every=interval,
                )) for power in powers
            ],
        }
        for interval in (10, 40)
    ]
    payload["device_verdict"] = "NOT_ESTABLISHED"
    payload["device_verdict_reason"] = (
        "The three runs share a large positive tangent exponent at all tested "
        "powers and renormalization intervals; this is a numerical-mode result, "
        "not evidence of the expected physical transition."
    )
    _atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": validation["status"], "device_points": len(payload["device_runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
