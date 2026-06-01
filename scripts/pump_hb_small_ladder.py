"""
Run a small-ladder pump harmonic-balance simulation.

This script is the production smoke-test for the nonlinear distributed HB stack.
It intentionally defaults to a small ladder so that failures are easy to inspect
before moving to the 100 mm / 20,000-cell industrial simulator.

Examples
--------
Quick small-ladder pump solve:

    python scripts/pump_hb_small_ladder.py --quick --output-dir outputs/pump_hb_small

More demanding local run:

    python scripts/pump_hb_small_ladder.py ^
      --n-cells 128 ^
      --pump-frequency-ghz 10.5 ^
      --pump-current-ratio 0.10 ^
      --harmonic-orders -3 -1 1 3 ^
      --output-dir outputs/pump_hb_128

Use a component CSV layout:

    python scripts/pump_hb_small_ladder.py ^
      --layout-csv outputs/linear_100mm_baseline/linear_100mm_layout_components.csv ^
      --n-cells-limit 256 ^
      --output-dir outputs/pump_hb_from_csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class PumpSolverMode(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    FALLBACK_LINEAR_PUMP = "fallback_linear_pump"


class NumericalBackend(str, Enum):
    DENSE = "dense"
    NEWTON_KRYLOV = "newton_krylov"


@dataclass(frozen=True)
class PumpHBSmallLadderConfig:
    n_cells: int
    n_cells_limit: int | None
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float
    layout_csv: str | None

    pump_frequency_ghz: float
    pump_current_ratio: float
    pump_phase_rad: float
    i_star_a: float
    l0_scale: float
    nonlinear_beta: float

    harmonic_orders: tuple[int, ...]
    n_time: int
    max_iter: int
    tolerance: float
    damping: float
    continuation_steps: int
    solver_mode: PumpSolverMode
    numerical_backend: NumericalBackend

    layout_kind: str
    include_resonators: bool
    disorder_std: float
    seed: int

    output_dir: str
    name: str
    quick: bool
    make_plots: bool
    save_checkpoint: bool
    export_netlist: bool

    @property
    def length_m(self) -> float:
        return self.length_mm * 1e-3

    @property
    def pump_frequency_hz(self) -> float:
        return self.pump_frequency_ghz * 1e9

    @property
    def omega_p_rad_s(self) -> float:
        return 2.0 * math.pi * self.pump_frequency_hz

    @property
    def pump_current_a(self) -> float:
        return self.pump_current_ratio * self.i_star_a

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["solver_mode"] = self.solver_mode.value
        d["numerical_backend"] = self.numerical_backend.value
        d["harmonic_orders"] = list(self.harmonic_orders)
        return d


@dataclass(frozen=True)
class StageResult:
    name: str
    status: RunStatus
    elapsed_s: float
    summary: Mapping[str, Any]
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "summary": jsonify(self.summary),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class PumpHBSmallLadderResult:
    config: PumpHBSmallLadderConfig
    status: RunStatus
    elapsed_s: float
    stages: tuple[StageResult, ...]
    artifact_paths: Mapping[str, str]
    metadata: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "config": self.config.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "artifact_paths": dict(self.artifact_paths),
            "metadata": jsonify(self.metadata),
        }


def jsonify(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, complex):
        return {
            "real": float(np.real(obj)),
            "imag": float(np.imag(obj)),
            "abs": float(abs(obj)),
        }
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return jsonify(obj.to_dict())
        except TypeError:
            try:
                return jsonify(obj.to_dict(include_arrays=False))
            except TypeError:
                return repr(obj)
    if isinstance(obj, Mapping):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return jsonify(arr.item())
        if np.iscomplexobj(arr):
            return {
                "array_shape": tuple(int(v) for v in arr.shape),
                "array_dtype": str(arr.dtype),
                "min_abs": float(np.nanmin(np.abs(arr))) if arr.size else None,
                "max_abs": float(np.nanmax(np.abs(arr))) if arr.size else None,
                "mean_abs": float(np.nanmean(np.abs(arr))) if arr.size else None,
            }
        return {
            "array_shape": tuple(int(v) for v in arr.shape),
            "array_dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "mean": float(np.nanmean(arr)) if arr.size else None,
        }
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def run_stage(name: str, fn: Callable[[], Mapping[str, Any]]) -> StageResult:
    start = time.perf_counter()
    try:
        summary = dict(fn())
        status = RunStatus(summary.pop("status", RunStatus.PASS.value))
        messages = tuple(str(m) for m in summary.pop("messages", ()))
        return StageResult(
            name=name,
            status=status,
            elapsed_s=time.perf_counter() - start,
            summary=summary,
            messages=messages,
        )
    except Exception as exc:
        return StageResult(
            name=name,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            summary={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
            messages=(f"ERROR: {type(exc).__name__}: {exc}",),
        )


def call_with_supported_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**dict(kwargs))
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def construct_with_supported_kwargs(cls_or_fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    return call_with_supported_kwargs(cls_or_fn, kwargs)


def get_attr_any(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            continue
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
            if value is not None:
                return value
    return default


def array_summary(x: Any) -> dict[str, Any]:
    arr = np.asarray(x)
    out = {
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size == 0:
        return out
    finite = np.isfinite(arr)
    out["finite"] = bool(np.all(finite))
    if not np.any(finite):
        return out
    if np.iscomplexobj(arr):
        abs_arr = np.abs(arr[finite])
        out.update(
            {
                "min_abs": float(np.min(abs_arr)),
                "max_abs": float(np.max(abs_arr)),
                "mean_abs": float(np.mean(abs_arr)),
            }
        )
    else:
        finite_arr = arr[finite]
        out.update(
            {
                "min": float(np.min(finite_arr)),
                "max": float(np.max(finite_arr)),
                "mean": float(np.mean(finite_arr)),
            }
        )
    return out


def layout_summary(layout: Any) -> dict[str, Any]:
    if hasattr(layout, "summary"):
        return jsonify(layout.summary())

    n_cells = int(get_attr_any(layout, "n_cells", default=-1))
    length_m = get_attr_any(layout, "length_m", default=None)
    total_length_m = None
    if length_m is not None:
        total_length_m = float(jnp.sum(jnp.asarray(length_m)))

    return {
        "name": get_attr_any(layout, "name", default="layout"),
        "n_cells": n_cells,
        "total_length_m": total_length_m,
        "z0_ohm": float(get_attr_any(layout, "z0_ohm", default=50.0)),
    }


def maybe_truncate_layout(layout: Any, config: PumpHBSmallLadderConfig) -> Any:
    """
    Truncate a loaded long layout to a small prefix for local pump-HB smoke tests.
    """
    if config.n_cells_limit is None:
        return layout

    n_total = int(get_attr_any(layout, "n_cells", default=config.n_cells))
    n = min(int(config.n_cells_limit), n_total)

    if n <= 0 or n >= n_total:
        return layout

    from twpa.core.layout import make_layout_from_arrays

    def arr(name: str, default: float = 0.0) -> jax.Array:
        value = get_attr_any(layout, name, default=None)
        if value is None:
            return jnp.full((n,), default, dtype=jnp.float64)
        return jnp.asarray(value)[:n]

    kwargs = {
        "length_m": arr("length_m"),
        "L_series_H": arr("L_series_H"),
        "C_shunt_F": arr("C_shunt_F"),
        "R_series_ohm": arr("R_series_ohm"),
        "G_shunt_S": arr("G_shunt_S"),
        "C_stub_F": arr("C_stub_F"),
        "L_res_H": arr("L_res_H"),
        "C_res_F": arr("C_res_F"),
        "C_couple_F": arr("C_couple_F"),
        "z0_ohm": float(get_attr_any(layout, "z0_ohm", default=config.z0_ohm)),
        "name": f"{get_attr_any(layout, 'name', default=config.name)}_first_{n}",
        "metadata": {
            "source": "scripts.pump_hb_small_ladder.maybe_truncate_layout",
            "original_layout": layout_summary(layout),
            "n_cells_limit": n,
        },
    }
    return call_with_supported_kwargs(make_layout_from_arrays, kwargs)


def load_layout_from_csv(config: PumpHBSmallLadderConfig) -> Any:
    from twpa.io.netlist import load_layout_component_csv

    if config.layout_csv is None:
        raise ValueError("layout_csv is None")

    layout = load_layout_component_csv(
        config.layout_csv,
        z0_ohm=config.z0_ohm,
        name=config.name,
        metadata={
            "source": "scripts.pump_hb_small_ladder",
            "layout_csv": config.layout_csv,
        },
    )
    return maybe_truncate_layout(layout, config)


def build_uniform_layout_fallback(config: PumpHBSmallLadderConfig) -> Any:
    """
    Build a simple uniform small LC ladder.

    Uses:
        Z0 = sqrt(L/C)
        vp = 1 / sqrt(L' C')
        L_cell = Z0 * dx / vp
        C_cell = dx / (Z0 * vp)
    """
    from twpa.core.layout import make_layout_from_arrays

    n = int(config.n_cells)
    dx = config.length_m / n
    z0 = float(config.z0_ohm)
    vp = float(config.phase_velocity_m_per_s)

    L_cell = z0 * dx / vp
    C_cell = dx / (z0 * vp)

    rng = np.random.default_rng(config.seed)
    if config.disorder_std > 0.0:
        L_scale = rng.lognormal(mean=0.0, sigma=config.disorder_std, size=n)
        C_scale = rng.lognormal(mean=0.0, sigma=config.disorder_std, size=n)
    else:
        L_scale = np.ones(n)
        C_scale = np.ones(n)

    C_stub = np.zeros(n)
    L_res = np.zeros(n)
    C_res = np.zeros(n)
    C_couple = np.zeros(n)

    if config.include_resonators:
        period = max(1, n // 16)
        idx = np.arange(n)
        mask = idx % period == 0
        C_stub[mask] = 2e-15
        L_res[mask] = 1e-9
        C_res[mask] = 5e-15
        C_couple[mask] = 1e-15

    kwargs = {
        "length_m": jnp.full((n,), dx, dtype=jnp.float64),
        "L_series_H": jnp.asarray(config.l0_scale * L_cell * L_scale, dtype=jnp.float64),
        "C_shunt_F": jnp.asarray(C_cell * C_scale, dtype=jnp.float64),
        "R_series_ohm": jnp.zeros((n,), dtype=jnp.float64),
        "G_shunt_S": jnp.zeros((n,), dtype=jnp.float64),
        "C_stub_F": jnp.asarray(C_stub, dtype=jnp.float64),
        "L_res_H": jnp.asarray(L_res, dtype=jnp.float64),
        "C_res_F": jnp.asarray(C_res, dtype=jnp.float64),
        "C_couple_F": jnp.asarray(C_couple, dtype=jnp.float64),
        "z0_ohm": z0,
        "name": config.name,
        "metadata": {
            "source": "scripts.pump_hb_small_ladder.build_uniform_layout_fallback",
            "layout_kind": config.layout_kind,
            "n_cells": n,
            "length_m": config.length_m,
            "dx_m": dx,
            "phase_velocity_m_per_s": vp,
            "z0_ohm": z0,
            "L_cell_nominal_H": L_cell,
            "C_cell_nominal_F": C_cell,
            "l0_scale": config.l0_scale,
            "disorder_std": config.disorder_std,
            "seed": config.seed,
            "include_resonators": config.include_resonators,
        },
    }

    return call_with_supported_kwargs(make_layout_from_arrays, kwargs)


def build_or_load_layout(config: PumpHBSmallLadderConfig) -> tuple[Any, dict[str, Any]]:
    if config.layout_csv is not None:
        layout = load_layout_from_csv(config)
        return layout, {
            "builder": "load_layout_component_csv",
            "layout_csv": config.layout_csv,
            "layout": layout_summary(layout),
        }

    try:
        from twpa.workflows.synthetic_benchmarks import (
            SyntheticLayoutKind,
            SyntheticLayoutSpec,
            build_synthetic_layout,
        )

        try:
            kind = SyntheticLayoutKind(config.layout_kind)
        except Exception:
            kind = getattr(SyntheticLayoutKind, config.layout_kind.upper(), SyntheticLayoutKind.UNIFORM)

        spec_kwargs = {
            "kind": kind,
            "n_cells": config.n_cells,
            "length_m": config.length_m,
            "z0_ohm": config.z0_ohm,
            "phase_velocity_m_per_s": config.phase_velocity_m_per_s,
            "include_resonators": config.include_resonators,
            "disorder_std": config.disorder_std,
            "seed": config.seed,
            "name": config.name,
        }
        spec = call_with_supported_kwargs(SyntheticLayoutSpec, spec_kwargs)
        layout = build_synthetic_layout(spec)
        return layout, {
            "builder": "twpa.workflows.synthetic_benchmarks.build_synthetic_layout",
            "spec": jsonify(spec),
            "layout": layout_summary(layout),
        }

    except Exception as exc:
        layout = build_uniform_layout_fallback(config)
        return layout, {
            "builder": "fallback_uniform_lc_layout",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "layout": layout_summary(layout),
        }


def build_frequency_plan(config: PumpHBSmallLadderConfig) -> Any:
    """
    Build a FrequencyPlan if the package exposes one; otherwise return a dict.
    """
    frequencies_hz = jnp.asarray(
        [order * config.pump_frequency_hz for order in config.harmonic_orders],
        dtype=jnp.float64,
    )
    labels = tuple(
        "pump" if order == 1 else f"{order:+d}pump"
        for order in config.harmonic_orders
    )

    try:
        from twpa.core.frequency_plan import make_pump_only_plan

        return make_pump_only_plan(
            config.pump_frequency_hz,
            n_harmonics=max(abs(order) for order in config.harmonic_orders),
            include_negative=any(order < 0 for order in config.harmonic_orders),
            include_dc=0 in config.harmonic_orders,
            sort="frequency",
        )

    except Exception:
        return {
            "frequencies_hz": frequencies_hz,
            "labels": labels,
            "harmonic_orders": config.harmonic_orders,
            "pump_label": "pump",
            "metadata": {
                "source": "fallback_frequency_plan_dict",
                "pump_frequency_hz": config.pump_frequency_hz,
            },
        }


def build_nonlinear_params(config: PumpHBSmallLadderConfig) -> Any:
    """
    Build NonlinearParams if available; otherwise return a dict.
    """
    try:
        from twpa.core.params import NonlinearParams

        kwargs = {
            "I_star_A": config.i_star_a,
            "I_star": config.i_star_a,
            "nonlinear_beta": config.nonlinear_beta,
            "beta_nl": config.nonlinear_beta,
            "kinetic_inductance_beta": config.nonlinear_beta,
            "name": "pump_hb_small_ladder_nonlinear_params",
            "metadata": {
                "source": "scripts.pump_hb_small_ladder",
                "pump_current_ratio": config.pump_current_ratio,
            },
        }
        return call_with_supported_kwargs(NonlinearParams, kwargs)

    except Exception:
        return {
            "I_star_A": config.i_star_a,
            "nonlinear_beta": config.nonlinear_beta,
            "metadata": {
                "source": "fallback_nonlinear_params_dict",
            },
        }


def build_pump_drive(config: PumpHBSmallLadderConfig) -> Any:
    """
    Build PumpDriveConfig if available; otherwise return a dict.
    """
    try:
        from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig

        return PumpDriveConfig.from_current_rms(
            pump_frequency_hz=config.pump_frequency_hz,
            current_rms_A=config.pump_current_a,
            source_impedance_ohm=config.z0_ohm,
            pump_label="pump",
            phase_rad=config.pump_phase_rad,
        )

    except Exception:
        return {
            "pump_frequency_hz": config.pump_frequency_hz,
            "pump_current_A": config.pump_current_a,
            "pump_current_ratio": config.pump_current_ratio,
            "pump_phase_rad": config.pump_phase_rad,
            "pump_label": "pump",
            "metadata": {
                "source": "fallback_pump_drive_dict",
                "I_star_A": config.i_star_a,
            },
        }


def build_pump_solver_config(config: PumpHBSmallLadderConfig) -> Any:
    """
    Build PumpHBLadderConfig if available; otherwise return a dict.
    """
    try:
        from twpa.core.hb_fft import HBProjectionConfig
        from twpa.core.params import SolverBackend, SolverConfig
        from twpa.nonlinear.distributed_hb import DistributedHBConfig
        from twpa.nonlinear.pump_hb_ladder import PumpHBLadderConfig
        from twpa.solvers.hb_solver import DenseNewtonConfig

        if config.numerical_backend == NumericalBackend.NEWTON_KRYLOV:
            solver = SolverConfig(
                backend=SolverBackend.NEWTON_KRYLOV,
                max_iter=config.max_iter,
                abs_tol=config.tolerance,
                rel_tol=config.tolerance,
                damping_initial=config.damping,
                verbose=True,
            )
        else:
            solver = DenseNewtonConfig(
                max_iter=config.max_iter,
                abs_tol=config.tolerance,
                rel_tol=config.tolerance,
                damping_initial=config.damping,
                verbose=True,
            )

        return PumpHBLadderConfig(
            n_pump_harmonics=max(abs(order) for order in config.harmonic_orders),
            include_negative_frequencies=any(order < 0 for order in config.harmonic_orders),
            include_dc=0 in config.harmonic_orders,
            distributed=DistributedHBConfig(),
            projection=HBProjectionConfig(n_time_samples=config.n_time),
            solver=solver,
            name="pump_hb_small_ladder_solver_config",
        )

    except Exception:
        return {
            "harmonic_orders": config.harmonic_orders,
            "n_time": config.n_time,
            "max_iter": config.max_iter,
            "tolerance": config.tolerance,
            "damping": config.damping,
            "continuation_steps": config.continuation_steps,
            "metadata": {
                "source": "fallback_pump_solver_config_dict",
            },
        }


def _plan_frequencies(plan: Any, config: PumpHBSmallLadderConfig) -> jax.Array:
    value = get_attr_any(plan, "frequencies_hz", default=None)
    if value is not None:
        return jnp.asarray(value, dtype=jnp.float64)
    return jnp.asarray(
        [order * config.pump_frequency_hz for order in config.harmonic_orders],
        dtype=jnp.float64,
    )


def _pump_index(plan: Any, config: PumpHBSmallLadderConfig) -> int:
    labels = get_attr_any(plan, "labels", default=None)
    if labels is not None:
        labels = tuple(str(x) for x in labels)
        if "pump" in labels:
            return labels.index("pump")
    if 1 in config.harmonic_orders:
        return list(config.harmonic_orders).index(1)
    return int(np.argmin(np.abs(np.asarray(_plan_frequencies(plan, config)) - config.pump_frequency_hz)))


def run_package_pump_solver(
    *,
    layout: Any,
    nonlinear_params: Any,
    frequency_plan: Any,
    pump_drive: Any,
    pump_config: Any,
    config: PumpHBSmallLadderConfig,
) -> tuple[Any, dict[str, Any]]:
    """
    Try package-native pump HB solvers with multiple possible API names.
    """
    import twpa.nonlinear.pump_hb_ladder as phb

    candidate_names = [
        "solve_pump_hb_ladder",
        "run_pump_hb_ladder",
        "solve_distributed_pump_hb",
        "solve_pump_hb",
        "solve_pump_solution",
        "run_pump_solution",
    ]

    last_errors: list[str] = []

    for name in candidate_names:
        fn = getattr(phb, name, None)
        if fn is None:
            continue

        kwargs = {
            "layout": layout,
            "nonlinear_params": nonlinear_params,
            "params": nonlinear_params,
            "frequency_plan": frequency_plan,
            "plan": frequency_plan,
            "pump_drive": pump_drive,
            "drive": pump_drive,
            "config": pump_config,
            "solver_config": pump_config,
            "pump_config": pump_config,
            "harmonic_orders": config.harmonic_orders,
            "n_time": config.n_time,
            "max_iter": config.max_iter,
            "tolerance": config.tolerance,
        }

        try:
            result = call_with_supported_kwargs(fn, kwargs)
            summary = summarize_pump_result(result, config=config)
            passed = bool(summary["finite"] and summary["converged"] and summary["finite_residual"])
            return result, {
                "status": RunStatus.PASS.value if passed else RunStatus.PARTIAL.value,
                "solver_function": name,
                "result": jsonify(result),
                **summary,
                "messages": (
                    f"PASS: package-native pump solver `{name}` converged with a finite residual."
                    if passed
                    else f"PARTIAL: package-native pump solver `{name}` ran but did not converge with a finite residual."
                ),
            }
        except Exception as exc:
            last_errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No compatible package-native pump solver succeeded. Errors:\n"
        + "\n".join(last_errors[-10:])
    )


def fallback_linear_pump_solution(
    *,
    layout: Any,
    frequency_plan: Any,
    nonlinear_params: Any,
    pump_drive: Any,
    config: PumpHBSmallLadderConfig,
) -> dict[str, Any]:
    """
    Fallback pump profile when full HB solver is unavailable.

    This is not a nonlinear HB solve. It is a deterministic linear traveling-pump
    initialization with weak cubic harmonic estimates. It is useful for debugging
    downstream artifacts and plotting, but the returned status is PARTIAL.
    """
    n_cells = int(get_attr_any(layout, "n_cells", default=config.n_cells))
    length_m = get_attr_any(layout, "length_m", default=None)

    if length_m is None:
        dz = config.length_m / n_cells
        lengths = jnp.full((n_cells,), dz, dtype=jnp.float64)
    else:
        lengths = jnp.asarray(length_m, dtype=jnp.float64)
        n_cells = int(lengths.shape[0])

    z_nodes = jnp.concatenate([jnp.asarray([0.0], dtype=jnp.float64), jnp.cumsum(lengths)])
    z_branches = 0.5 * (z_nodes[:-1] + z_nodes[1:])

    freqs = _plan_frequencies(frequency_plan, config)
    n_freq = int(freqs.shape[0])
    pump_idx = _pump_index(frequency_plan, config)

    beta_p = 2.0 * jnp.pi * config.pump_frequency_hz / config.phase_velocity_m_per_s
    alpha_p = 0.0

    branch_current = jnp.zeros((n_freq, n_cells), dtype=jnp.complex128)
    node_voltage = jnp.zeros((n_freq, n_cells + 1), dtype=jnp.complex128)

    phase_branch = jnp.exp(-1j * beta_p * z_branches - alpha_p * z_branches)
    phase_node = jnp.exp(-1j * beta_p * z_nodes - alpha_p * z_nodes)

    I_p = config.pump_current_a * jnp.exp(1j * config.pump_phase_rad)
    V_p = config.z0_ohm * I_p

    branch_current = branch_current.at[pump_idx, :].set(I_p * phase_branch)
    node_voltage = node_voltage.at[pump_idx, :].set(V_p * phase_node)

    # Very small cubic harmonic diagnostic estimate from L(I) ~ L0(1 + I^2/I*^2).
    if 3 in config.harmonic_orders:
        h3_idx = list(config.harmonic_orders).index(3)
        cubic_scale = config.nonlinear_beta * (config.pump_current_ratio**2) / 4.0
        branch_current = branch_current.at[h3_idx, :].set(cubic_scale * I_p * jnp.exp(-1j * 3.0 * beta_p * z_branches))
        node_voltage = node_voltage.at[h3_idx, :].set(config.z0_ohm * branch_current[h3_idx, :].mean() * jnp.exp(-1j * 3.0 * beta_p * z_nodes))

    residual_norm = jnp.asarray(np.nan, dtype=jnp.float64)

    return {
        "kind": "fallback_linear_pump_solution",
        "status": RunStatus.PARTIAL.value,
        "frequency_plan": frequency_plan,
        "layout": layout_summary(layout),
        "nonlinear_params": jsonify(nonlinear_params),
        "pump_drive": jsonify(pump_drive),
        "node_voltage_coeffs_V": node_voltage,
        "branch_current_coeffs_A": branch_current,
        "z_nodes_m": z_nodes,
        "z_branches_m": z_branches,
        "residual_norm": residual_norm,
        "converged": False,
        "finite": bool(jnp.all(jnp.isfinite(branch_current)) and jnp.all(jnp.isfinite(node_voltage))),
        "messages": (
            "PARTIAL: used fallback linear traveling-pump profile, not a nonlinear HB solve.",
        ),
    }


def run_pump_solver(
    *,
    layout: Any,
    nonlinear_params: Any,
    frequency_plan: Any,
    pump_drive: Any,
    pump_config: Any,
    config: PumpHBSmallLadderConfig,
) -> tuple[Any, dict[str, Any]]:
    if config.solver_mode in {PumpSolverMode.AUTO, PumpSolverMode.PACKAGE}:
        try:
            return run_package_pump_solver(
                layout=layout,
                nonlinear_params=nonlinear_params,
                frequency_plan=frequency_plan,
                pump_drive=pump_drive,
                pump_config=pump_config,
                config=config,
            )
        except Exception as exc:
            if config.solver_mode == PumpSolverMode.PACKAGE:
                raise
            package_error = f"{type(exc).__name__}: {exc}"
    else:
        package_error = "package solver disabled by --solver-mode fallback_linear_pump"

    fallback = fallback_linear_pump_solution(
        layout=layout,
        frequency_plan=frequency_plan,
        nonlinear_params=nonlinear_params,
        pump_drive=pump_drive,
        config=config,
    )
    return fallback, {
        "status": RunStatus.PARTIAL.value,
        "solver_function": "fallback_linear_pump_solution",
        "package_error": package_error,
        "result": jsonify(fallback),
        **summarize_pump_result(fallback, config=config),
        "messages": (
            "PARTIAL: package-native pump HB solver unavailable; used fallback linear pump profile.",
            package_error,
        ),
    }


def extract_pump_arrays(result: Any, config: PumpHBSmallLadderConfig) -> dict[str, Any]:
    """
    Extract common pump-HB arrays from result objects/dicts.
    """
    state = get_attr_any(result, "state", default=None)
    if state is None:
        state = result

    node_voltage = get_attr_any(
        state,
        "node_voltage_coeffs_V",
        "node_voltage_coefficients_V",
        "V_coeffs",
        "V",
        default=None,
    )
    branch_current = get_attr_any(
        state,
        "branch_current_coeffs_A",
        "branch_current_coefficients_A",
        "I_L_coeffs",
        "I_coeffs",
        "I",
        default=None,
    )

    residual = get_attr_any(result, "residual", default=None)
    residual_norm = get_attr_any(
        result,
        "residual_norm",
        "final_residual_norm",
        default=get_attr_any(residual, "norm", "residual_norm", default=None),
    )

    frequency_plan = get_attr_any(result, "frequency_plan", "plan", default=None)
    if frequency_plan is None:
        frequency_plan = build_frequency_plan(config)

    freqs = _plan_frequencies(frequency_plan, config)

    arrays = {
        "frequencies_hz": freqs,
        "node_voltage_coeffs_V": None if node_voltage is None else jnp.asarray(node_voltage),
        "branch_current_coeffs_A": None if branch_current is None else jnp.asarray(branch_current),
        "residual_norm": None if residual_norm is None else jnp.asarray(residual_norm),
    }

    z_nodes = get_attr_any(result, "z_nodes_m", "node_positions_m", default=None)
    z_branches = get_attr_any(result, "z_branches_m", "branch_positions_m", default=None)

    if z_nodes is not None:
        arrays["z_nodes_m"] = jnp.asarray(z_nodes, dtype=jnp.float64)
    if z_branches is not None:
        arrays["z_branches_m"] = jnp.asarray(z_branches, dtype=jnp.float64)

    return arrays


def summarize_pump_result(result: Any, *, config: PumpHBSmallLadderConfig) -> dict[str, Any]:
    arrays = extract_pump_arrays(result, config)

    node_voltage = arrays.get("node_voltage_coeffs_V")
    branch_current = arrays.get("branch_current_coeffs_A")
    residual_norm = arrays.get("residual_norm")
    finite_residual = False

    finite_parts = []
    if node_voltage is not None:
        finite_parts.append(bool(jnp.all(jnp.isfinite(node_voltage))))
    if branch_current is not None:
        finite_parts.append(bool(jnp.all(jnp.isfinite(branch_current))))
    if residual_norm is not None:
        rn = np.asarray(residual_norm)
        if rn.size and np.all(np.isfinite(rn)):
            finite_parts.append(True)
            finite_residual = True

    finite = all(finite_parts) if finite_parts else False

    pump_idx = None
    max_pump_current = None
    max_current_ratio = None
    pump_output_ratio = None

    if branch_current is not None:
        plan = get_attr_any(result, "frequency_plan", "plan", default=build_frequency_plan(config))
        pump_idx = _pump_index(plan, config)
        I = jnp.asarray(branch_current)
        if I.ndim >= 2 and pump_idx < I.shape[0]:
            pump_profile = jnp.abs(I[pump_idx])
            max_pump_current = float(jnp.nanmax(pump_profile))
            max_current_ratio = max_pump_current / max(config.i_star_a, 1e-300)
            if pump_profile.size >= 2:
                pump_output_ratio = float(pump_profile[-1] / jnp.maximum(pump_profile[0], 1e-300))

    converged = bool(get_attr_any(result, "converged", "success", default=False))
    status_value = get_attr_any(result, "status", default=None)
    if isinstance(status_value, Enum):
        status_value = status_value.value

    summary = {
        "finite": finite,
        "finite_residual": finite_residual,
        "converged": converged,
        "status_value": status_value,
        "pump_index": pump_idx,
        "max_pump_current_A": max_pump_current,
        "max_current_ratio": max_current_ratio,
        "pump_output_input_current_ratio": pump_output_ratio,
        "arrays": {
            key: None if value is None else array_summary(value)
            for key, value in arrays.items()
        },
    }

    return summary


def write_component_csv(layout: Any, path: Path) -> Path:
    try:
        from twpa.io.netlist import write_layout_component_csv

        return write_layout_component_csv(layout, path)
    except Exception:
        n_cells = int(get_attr_any(layout, "n_cells", default=0))
        if n_cells <= 0:
            raise

        keys = [
            "length_m",
            "L_series_H",
            "C_shunt_F",
            "R_series_ohm",
            "G_shunt_S",
            "C_stub_F",
            "L_res_H",
            "C_res_F",
            "C_couple_F",
        ]

        arrays = {}
        for key in keys:
            value = get_attr_any(layout, key, default=None)
            if value is None:
                arrays[key] = np.zeros(n_cells)
            else:
                arrays[key] = np.asarray(value, dtype=float)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cell_index", *keys])
            writer.writeheader()
            for i in range(n_cells):
                row = {"cell_index": i}
                row.update({key: float(arrays[key][i]) for key in keys})
                writer.writerow(row)

        return path


def write_pump_profile_csv(
    path: Path,
    *,
    arrays: Mapping[str, Any],
    config: PumpHBSmallLadderConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    branch_current = arrays.get("branch_current_coeffs_A")
    node_voltage = arrays.get("node_voltage_coeffs_V")

    if branch_current is None:
        raise ValueError("branch_current_coeffs_A unavailable")

    I = np.asarray(branch_current)
    pump_idx = list(config.harmonic_orders).index(1) if 1 in config.harmonic_orders else 0

    n_cells = I.shape[-1]
    z_branches = arrays.get("z_branches_m")
    if z_branches is None:
        z = np.arange(n_cells, dtype=float)
    else:
        z = np.asarray(z_branches, dtype=float)

    fields = [
        "cell_index",
        "z_m",
        "pump_current_abs_A",
        "pump_current_ratio",
        "pump_current_phase_rad",
    ]

    include_voltage = node_voltage is not None
    if include_voltage:
        fields.extend(["pump_voltage_abs_V", "pump_voltage_phase_rad"])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(n_cells):
            row = {
                "cell_index": i,
                "z_m": float(z[i]) if i < len(z) else float(i),
                "pump_current_abs_A": float(np.abs(I[pump_idx, i])),
                "pump_current_ratio": float(np.abs(I[pump_idx, i]) / max(config.i_star_a, 1e-300)),
                "pump_current_phase_rad": float(np.angle(I[pump_idx, i])),
            }

            if include_voltage:
                V = np.asarray(node_voltage)
                node_i = min(i, V.shape[-1] - 1)
                row.update(
                    {
                        "pump_voltage_abs_V": float(np.abs(V[pump_idx, node_i])),
                        "pump_voltage_phase_rad": float(np.angle(V[pump_idx, node_i])),
                    }
                )

            writer.writerow(row)

    return path


def export_artifacts(
    *,
    layout: Any,
    nonlinear_params: Any,
    frequency_plan: Any,
    pump_drive: Any,
    pump_config: Any,
    pump_result: Any,
    config: PumpHBSmallLadderConfig,
    stages: list[StageResult],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    arrays = extract_pump_arrays(pump_result, config)

    arrays_payload: dict[str, Any] = {
        "metadata_json": json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "layout": layout_summary(layout),
                    "nonlinear_params": nonlinear_params,
                    "frequency_plan": frequency_plan,
                    "pump_drive": pump_drive,
                    "pump_config": pump_config,
                    "pump_result_summary": summarize_pump_result(pump_result, config=config),
                }
            )
        ),
    }

    for key, value in arrays.items():
        if value is not None:
            arrays_payload[key] = np.asarray(value)

    arrays_npz = output_dir / "pump_hb_small_ladder_arrays.npz"
    np.savez_compressed(arrays_npz, **arrays_payload)
    paths["arrays_npz"] = str(arrays_npz)

    component_csv = write_component_csv(layout, output_dir / "pump_hb_small_ladder_components.csv")
    paths["layout_components_csv"] = str(component_csv)

    if arrays.get("branch_current_coeffs_A") is not None:
        profile_csv = write_pump_profile_csv(
            output_dir / "pump_hb_small_ladder_profile.csv",
            arrays=arrays,
            config=config,
        )
        paths["pump_profile_csv"] = str(profile_csv)

    if config.export_netlist:
        try:
            from twpa.io.netlist import write_netlist_bundle

            netlist_paths = write_netlist_bundle(
                layout,
                output_dir / "netlist",
                prefix="pump_hb_small_ladder",
            )
            paths.update({f"netlist_{k}": v for k, v in netlist_paths.items()})
        except Exception as exc:
            err_path = output_dir / "netlist_export_error.txt"
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            paths["netlist_error_txt"] = str(err_path)
            stages.append(
                StageResult(
                    name="netlist_export",
                    status=RunStatus.PARTIAL,
                    elapsed_s=0.0,
                    summary={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                    messages=(f"PARTIAL: netlist export failed: {exc}",),
                )
            )

    if config.save_checkpoint:
        try:
            from twpa.io.checkpoints import (
                CheckpointKind,
                CheckpointMetadata,
                save_checkpoint,
            )

            checkpoint_path = output_dir / "pump_hb_small_ladder_checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.PUMP_HB,
                    name="pump_hb_small_ladder",
                    source="scripts.pump_hb_small_ladder",
                    extra={
                        "config": config.to_dict(),
                        "layout": layout_summary(layout),
                    },
                ),
                arrays={k: v for k, v in arrays.items() if v is not None},
                payload={
                    "pump_result": jsonify(pump_result),
                    "pump_result_summary": summarize_pump_result(pump_result, config=config),
                },
            )
            paths["checkpoint_npz"] = str(checkpoint_path)
        except Exception as exc:
            err_path = output_dir / "checkpoint_error.txt"
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            paths["checkpoint_error_txt"] = str(err_path)
            stages.append(
                StageResult(
                    name="checkpoint",
                    status=RunStatus.PARTIAL,
                    elapsed_s=0.0,
                    summary={
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                    messages=(f"PARTIAL: checkpoint export failed: {exc}",),
                )
            )

    if config.make_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from twpa.plotting.diagnostics import PlotConfig, plot_pump_profile, save_figure

            fig, _ = plot_pump_profile(
                pump_result,
                config=PlotConfig(title="Small-ladder pump current profile"),
                quantity="current_ratio",
            )
            p = save_figure(fig, output_dir / "pump_hb_small_ladder_profile.png")
            paths["pump_profile_png"] = str(p)
            plt.close(fig)

        except Exception:
            try:
                # Local fallback plotting directly from arrays.
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                branch_current = arrays.get("branch_current_coeffs_A")
                if branch_current is not None:
                    I = np.asarray(branch_current)
                    pump_idx = list(config.harmonic_orders).index(1) if 1 in config.harmonic_orders else 0
                    y = np.abs(I[pump_idx]) / max(config.i_star_a, 1e-300)

                    z = arrays.get("z_branches_m")
                    if z is None:
                        x = np.arange(y.shape[0])
                        xlabel = "Cell index"
                    else:
                        x = np.asarray(z) * 1e3
                        xlabel = "Position (mm)"

                    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
                    ax.plot(x, y)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("|I_pump| / I*")
                    ax.set_title("Small-ladder pump current profile")
                    ax.grid(True)
                    fig.tight_layout()
                    p = output_dir / "pump_hb_small_ladder_profile.png"
                    fig.savefig(p, bbox_inches="tight")
                    plt.close(fig)
                    paths["pump_profile_png"] = str(p)
                else:
                    raise RuntimeError("branch_current_coeffs_A unavailable for fallback plot")

            except Exception as exc:
                err_path = output_dir / "plotting_error.txt"
                err_path.write_text(traceback.format_exc(), encoding="utf-8")
                paths["plotting_error_txt"] = str(err_path)
                stages.append(
                    StageResult(
                        name="plotting",
                        status=RunStatus.PARTIAL,
                        elapsed_s=0.0,
                        summary={
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                        },
                        messages=(f"PARTIAL: plotting failed: {exc}",),
                    )
                )

    return paths


def result_markdown(result: PumpHBSmallLadderResult) -> str:
    cfg = result.config

    lines = [
        "# Pump HB small-ladder run",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- cells: `{cfg.n_cells}`",
        f"- length: `{cfg.length_mm:.6g} mm`",
        f"- pump frequency: `{cfg.pump_frequency_ghz:.6g} GHz`",
        f"- pump current ratio: `{cfg.pump_current_ratio:.6g}`",
        f"- harmonic orders: `{list(cfg.harmonic_orders)}`",
        f"- solver mode: `{cfg.solver_mode.value}`",
        "",
        "## Stage summary",
        "",
        "| stage | status | elapsed s | messages |",
        "|---|---|---:|---|",
    ]

    for stage in result.stages:
        msg = "<br>".join(stage.messages[:3])
        lines.append(
            f"| `{stage.name}` | `{stage.status.value}` | "
            f"{stage.elapsed_s:.6g} | {msg} |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "| key | path |",
        "|---|---|",
    ]

    for key, path in result.artifact_paths.items():
        lines.append(f"| `{key}` | `{path}` |")

    lines += [
        "",
        "## Key summaries",
        "",
    ]

    for stage in result.stages:
        if stage.name in {
            "build_layout",
            "build_problem",
            "pump_solve",
            "artifact_export",
        }:
            lines += [
                f"### {stage.name}",
                "",
                "```json",
                json.dumps(jsonify(stage.summary), indent=2)[:7000],
                "```",
                "",
            ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small-ladder pump harmonic-balance simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument(
        "--n-cells-limit",
        type=int,
        default=None,
        help="When loading a CSV layout, truncate to this many first cells.",
    )
    parser.add_argument("--length-mm", type=float, default=1.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)
    parser.add_argument("--layout-csv", type=str, default=None)

    parser.add_argument("--pump-frequency-ghz", type=float, default=10.0)
    parser.add_argument("--pump-current-ratio", type=float, default=0.08)
    parser.add_argument("--pump-phase-rad", type=float, default=0.0)
    parser.add_argument("--i-star-a", type=float, default=5e-3)
    parser.add_argument("--l0-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-beta", type=float, default=1.0)

    parser.add_argument(
        "--harmonic-orders",
        type=int,
        nargs="+",
        default=[-3, -1, 1, 3],
        help="Integer pump harmonic orders included in the HB basis.",
    )
    parser.add_argument("--n-time", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--continuation-steps", type=int, default=8)
    parser.add_argument(
        "--solver-mode",
        choices=[m.value for m in PumpSolverMode],
        default=PumpSolverMode.AUTO.value,
    )
    parser.add_argument(
        "--numerical-backend",
        choices=[m.value for m in NumericalBackend],
        default=NumericalBackend.DENSE.value,
        help="Dense is the tiny reference backend; newton_krylov is matrix-free.",
    )

    parser.add_argument("--layout-kind", type=str, default="uniform")
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument("--disorder-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pump_hb_small_ladder"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="pump_hb_small_ladder")

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--export-netlist", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> PumpHBSmallLadderConfig:
    n_cells = int(args.n_cells)
    n_time = int(args.n_time)
    max_iter = int(args.max_iter)
    continuation_steps = int(args.continuation_steps)
    harmonic_orders = tuple(int(h) for h in args.harmonic_orders)

    if args.quick:
        n_cells = min(n_cells, 8)
        n_time = min(n_time, 32)
        max_iter = min(max_iter, 20)
        continuation_steps = min(continuation_steps, 4)

    if n_cells <= 0:
        raise ValueError("--n-cells must be positive")
    if args.n_cells_limit is not None and int(args.n_cells_limit) <= 0:
        raise ValueError("--n-cells-limit must be positive when provided")
    if args.length_mm <= 0.0:
        raise ValueError("--length-mm must be positive")
    if args.z0_ohm <= 0.0:
        raise ValueError("--z0-ohm must be positive")
    if args.phase_velocity_m_per_s <= 0.0:
        raise ValueError("--phase-velocity-m-per-s must be positive")
    if args.pump_frequency_ghz <= 0.0:
        raise ValueError("--pump-frequency-ghz must be positive")
    if args.i_star_a <= 0.0:
        raise ValueError("--i-star-a must be positive")
    if args.pump_current_ratio < 0.0:
        raise ValueError("--pump-current-ratio must be non-negative")
    if args.l0_scale <= 0.0:
        raise ValueError("--l0-scale must be positive")
    if not harmonic_orders:
        raise ValueError("--harmonic-orders may not be empty")
    if 1 not in harmonic_orders:
        raise ValueError("--harmonic-orders must include 1 for the pump")
    if len(set(harmonic_orders)) != len(harmonic_orders):
        raise ValueError("--harmonic-orders may not contain duplicates")
    if any(h == 0 for h in harmonic_orders):
        raise ValueError("--harmonic-orders should not include 0 for this pump-only run")
    if n_time < 2 * len(harmonic_orders):
        raise ValueError("--n-time should be at least 2 * number of harmonic orders")
    if max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")
    if args.damping <= 0.0:
        raise ValueError("--damping must be positive")
    if continuation_steps <= 0:
        raise ValueError("--continuation-steps must be positive")
    if args.disorder_std < 0.0:
        raise ValueError("--disorder-std must be non-negative")

    return PumpHBSmallLadderConfig(
        n_cells=n_cells,
        n_cells_limit=None if args.n_cells_limit is None else int(args.n_cells_limit),
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        layout_csv=args.layout_csv,
        pump_frequency_ghz=float(args.pump_frequency_ghz),
        pump_current_ratio=float(args.pump_current_ratio),
        pump_phase_rad=float(args.pump_phase_rad),
        i_star_a=float(args.i_star_a),
        l0_scale=float(args.l0_scale),
        nonlinear_beta=float(args.nonlinear_beta),
        harmonic_orders=harmonic_orders,
        n_time=n_time,
        max_iter=max_iter,
        tolerance=float(args.tolerance),
        damping=float(args.damping),
        continuation_steps=continuation_steps,
        solver_mode=PumpSolverMode(args.solver_mode),
        numerical_backend=NumericalBackend(args.numerical_backend),
        layout_kind=str(args.layout_kind),
        include_resonators=bool(args.include_resonators),
        disorder_std=float(args.disorder_std),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        name=str(args.name),
        quick=bool(args.quick),
        make_plots=not bool(args.no_plots),
        save_checkpoint=not bool(args.no_checkpoint),
        export_netlist=bool(args.export_netlist),
    )


def _stage_build_layout(holders: dict[str, Any], config: PumpHBSmallLadderConfig) -> dict[str, Any]:
    layout, summary = build_or_load_layout(config)
    holders["layout"] = layout
    return {
        "status": RunStatus.PASS.value,
        **summary,
        "messages": ("PASS: layout built/loaded.",),
    }


def _stage_build_problem(holders: dict[str, Any], config: PumpHBSmallLadderConfig) -> dict[str, Any]:
    frequency_plan = build_frequency_plan(config)
    nonlinear_params = build_nonlinear_params(config)
    pump_drive = build_pump_drive(config)
    pump_config = build_pump_solver_config(config)

    holders["frequency_plan"] = frequency_plan
    holders["nonlinear_params"] = nonlinear_params
    holders["pump_drive"] = pump_drive
    holders["pump_config"] = pump_config

    return {
        "status": RunStatus.PASS.value,
        "frequency_plan": jsonify(frequency_plan),
        "nonlinear_params": jsonify(nonlinear_params),
        "pump_drive": jsonify(pump_drive),
        "pump_config": jsonify(pump_config),
        "pump_current_A": config.pump_current_a,
        "messages": ("PASS: pump HB problem objects built.",),
    }


def _stage_pump_solve(holders: dict[str, Any], config: PumpHBSmallLadderConfig) -> dict[str, Any]:
    pump_result, summary = run_pump_solver(
        layout=holders["layout"],
        nonlinear_params=holders["nonlinear_params"],
        frequency_plan=holders["frequency_plan"],
        pump_drive=holders["pump_drive"],
        pump_config=holders["pump_config"],
        config=config,
    )

    holders["pump_result"] = pump_result

    return summary


def finalize_result(result: PumpHBSmallLadderResult, output_dir: Path) -> int:
    summary_json = output_dir / "pump_hb_small_ladder_summary.json"
    summary_md = output_dir / "pump_hb_small_ladder_summary.md"

    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }

    result = PumpHBSmallLadderResult(
        config=result.config,
        status=result.status,
        elapsed_s=result.elapsed_s,
        stages=result.stages,
        artifact_paths=artifact_paths,
        metadata=result.metadata,
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    summary_md.write_text(result_markdown(result), encoding="utf-8")

    print()
    print(f"[pump-hb-small] status: {result.status.value}")
    print(f"[pump-hb-small] summary JSON: {summary_json}")
    print(f"[pump-hb-small] summary MD:   {summary_md}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[pump-hb-small] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "python": sys.version,
        "jax": {
            "version": getattr(jax, "__version__", None),
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "devices": [str(d) for d in jax.devices()],
        },
        "script": "scripts/pump_hb_small_ladder.py",
    }

    holders: dict[str, Any] = {}
    stages: list[StageResult] = []
    artifacts: dict[str, str] = {}

    print("[pump-hb-small] building/loading layout...")
    stage = run_stage("build_layout", lambda: _stage_build_layout(holders, config))
    stages.append(stage)
    print(f"[pump-hb-small] build_layout: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = PumpHBSmallLadderResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[pump-hb-small] building pump HB problem...")
    stage = run_stage("build_problem", lambda: _stage_build_problem(holders, config))
    stages.append(stage)
    print(f"[pump-hb-small] build_problem: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = PumpHBSmallLadderResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[pump-hb-small] solving pump HB...")
    stage = run_stage("pump_solve", lambda: _stage_pump_solve(holders, config))
    stages.append(stage)
    print(f"[pump-hb-small] pump_solve: {stage.status.value}")

    if stage.status == RunStatus.ERROR:
        result = PumpHBSmallLadderResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[pump-hb-small] exporting artifacts...")
    try:
        artifacts.update(
            export_artifacts(
                layout=holders["layout"],
                nonlinear_params=holders["nonlinear_params"],
                frequency_plan=holders["frequency_plan"],
                pump_drive=holders["pump_drive"],
                pump_config=holders["pump_config"],
                pump_result=holders["pump_result"],
                config=config,
                stages=stages,
                output_dir=output_dir,
            )
        )
        stages.append(
            StageResult(
                name="artifact_export",
                status=RunStatus.PASS,
                elapsed_s=0.0,
                summary={"n_artifacts": len(artifacts)},
                messages=("PASS: artifacts exported.",),
            )
        )
    except Exception as exc:
        stages.append(
            StageResult(
                name="artifact_export",
                status=RunStatus.ERROR,
                elapsed_s=0.0,
                summary={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                messages=(f"ERROR: artifact export failed: {exc}",),
            )
        )

    hard_fail = any(s.status in {RunStatus.FAIL, RunStatus.ERROR} for s in stages)
    partial = any(s.status == RunStatus.PARTIAL for s in stages)

    if hard_fail:
        status = RunStatus.ERROR
    elif partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.PASS

    result = PumpHBSmallLadderResult(
        config=config,
        status=status,
        elapsed_s=time.perf_counter() - start,
        stages=tuple(stages),
        artifact_paths=artifacts,
        metadata=metadata,
    )

    return finalize_result(result, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
