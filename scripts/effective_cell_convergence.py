"""
Study effective-cell convergence for a TWPA linear model.

This script answers:

    How many effective cells are needed to reproduce the pump-off response of a
    fine 100 mm / 20,000-cell TWPA model?

It builds or loads a fine layout, constructs coarsened layouts with fewer
effective cells, runs pump-off linear scans, compares each coarsened response to
a reference response, and exports convergence metrics.

Examples
--------
Quick smoke test:

    python scripts/effective_cell_convergence.py --quick --output-dir outputs/effective_cell_quick

100 mm convergence against a 20,000-cell reference:

    python scripts/effective_cell_convergence.py ^
      --reference-cells 20000 ^
      --effective-cells 250 500 1000 2000 5000 10000 ^
      --length-mm 100 ^
      --n-frequency 501 ^
      --output-dir outputs/effective_cell_convergence

Use an existing component CSV as the fine layout:

    python scripts/effective_cell_convergence.py ^
      --layout-csv outputs/linear_100mm_baseline/linear_100mm_layout_components.csv ^
      --effective-cells 500 1000 2000 5000 ^
      --output-dir outputs/effective_cell_from_csv
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


class CoarseningMethod(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    CONSERVATIVE_GROUPING = "conservative_grouping"


@dataclass(frozen=True)
class EffectiveCellConvergenceConfig:
    reference_cells: int
    effective_cells: tuple[int, ...]
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float

    f_min_ghz: float
    f_max_ghz: float
    n_frequency: int

    layout_csv: str | None
    layout_kind: str
    include_resonators: bool
    disorder_std: float
    seed: int

    coarsening_method: CoarseningMethod
    pass_rms_db: float
    pass_max_db: float
    pass_group_delay_rms_ps: float

    output_dir: str
    name: str
    quick: bool
    make_plots: bool
    save_checkpoint: bool
    export_csv: bool
    export_netlists: bool

    @property
    def length_m(self) -> float:
        return self.length_mm * 1e-3

    @property
    def frequency_hz(self) -> jax.Array:
        return jnp.linspace(
            self.f_min_ghz * 1e9,
            self.f_max_ghz * 1e9,
            self.n_frequency,
            dtype=jnp.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coarsening_method"] = self.coarsening_method.value
        d["effective_cells"] = list(self.effective_cells)
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
class CoarsenedRunResult:
    effective_cells: int
    status: RunStatus
    elapsed_s: float
    coarsening_summary: Mapping[str, Any]
    scan_summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_cells": self.effective_cells,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "coarsening_summary": jsonify(self.coarsening_summary),
            "scan_summary": jsonify(self.scan_summary),
            "metrics": jsonify(self.metrics),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class EffectiveCellConvergenceResult:
    config: EffectiveCellConvergenceConfig
    status: RunStatus
    elapsed_s: float
    stages: tuple[StageResult, ...]
    runs: tuple[CoarsenedRunResult, ...]
    artifact_paths: Mapping[str, str]
    metadata: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.runs if r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "n_runs": self.n_runs,
            "n_passed": self.n_passed,
            "config": self.config.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "runs": [r.to_dict() for r in self.runs],
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
    if np.iscomplexobj(arr):
        out.update(
            {
                "min_abs": float(np.nanmin(np.abs(arr))),
                "max_abs": float(np.nanmax(np.abs(arr))),
                "mean_abs": float(np.nanmean(np.abs(arr))),
            }
        )
    else:
        out.update(
            {
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
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


def layout_array(layout: Any, name: str, *, default: float = 0.0) -> jax.Array:
    n = int(get_attr_any(layout, "n_cells", default=0))
    value = get_attr_any(layout, name, default=None)
    if value is None:
        return jnp.full((n,), default, dtype=jnp.float64)
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.shape != (n,):
        raise ValueError(f"layout.{name} must have shape {(n,)}, got {arr.shape}")
    return arr


def build_uniform_layout_fallback(config: EffectiveCellConvergenceConfig) -> Any:
    from twpa.core.layout import make_layout_from_arrays

    n = int(config.reference_cells)
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
        period = max(1, n // 128)
        idx = np.arange(n)
        mask = idx % period == 0
        C_stub[mask] = 2e-15
        L_res[mask] = 1e-9
        C_res[mask] = 5e-15
        C_couple[mask] = 1e-15

    return call_with_supported_kwargs(
        make_layout_from_arrays,
        {
            "length_m": jnp.full((n,), dx, dtype=jnp.float64),
            "L_series_H": jnp.asarray(L_cell * L_scale, dtype=jnp.float64),
            "C_shunt_F": jnp.asarray(C_cell * C_scale, dtype=jnp.float64),
            "R_series_ohm": jnp.zeros((n,), dtype=jnp.float64),
            "G_shunt_S": jnp.zeros((n,), dtype=jnp.float64),
            "C_stub_F": jnp.asarray(C_stub, dtype=jnp.float64),
            "L_res_H": jnp.asarray(L_res, dtype=jnp.float64),
            "C_res_F": jnp.asarray(C_res, dtype=jnp.float64),
            "C_couple_F": jnp.asarray(C_couple, dtype=jnp.float64),
            "z0_ohm": z0,
            "name": f"{config.name}_reference_{n}",
            "metadata": {
                "source": "scripts.effective_cell_convergence.build_uniform_layout_fallback",
                "n_cells": n,
                "length_m": config.length_m,
                "dx_m": dx,
                "z0_ohm": z0,
                "phase_velocity_m_per_s": vp,
                "L_cell_nominal_H": L_cell,
                "C_cell_nominal_F": C_cell,
                "include_resonators": config.include_resonators,
                "disorder_std": config.disorder_std,
                "seed": config.seed,
            },
        },
    )


def build_or_load_reference_layout(config: EffectiveCellConvergenceConfig) -> tuple[Any, dict[str, Any]]:
    if config.layout_csv is not None:
        from twpa.io.netlist import load_layout_component_csv

        layout = load_layout_component_csv(
            config.layout_csv,
            z0_ohm=config.z0_ohm,
            name=f"{config.name}_reference_from_csv",
            metadata={
                "source": "scripts.effective_cell_convergence",
                "layout_csv": config.layout_csv,
            },
        )
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

        spec = call_with_supported_kwargs(
            SyntheticLayoutSpec,
            {
                "kind": kind,
                "n_cells": config.reference_cells,
                "length_m": config.length_m,
                "z0_ohm": config.z0_ohm,
                "phase_velocity_m_per_s": config.phase_velocity_m_per_s,
                "include_resonators": config.include_resonators,
                "disorder_std": config.disorder_std,
                "seed": config.seed,
                "name": f"{config.name}_reference_{config.reference_cells}",
            },
        )
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


def conservative_group_layout(
    layout: Any,
    n_effective: int,
    *,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    """
    Coarsen a layout by grouping adjacent fine cells.

    Conservative grouping preserves total series impedance and total shunt
    admittance at low frequency:

        length_group = sum length_i
        L_group      = sum L_i
        R_group      = sum R_i
        C_group      = sum C_i
        G_group      = sum G_i

    Optional stub/resonator capacitive loads are summed. Resonator L values are
    averaged over nonzero cells because an exact equivalent for distributed
    resonator loading is topology-dependent.
    """
    from twpa.core.layout import make_layout_from_arrays

    n_fine = int(get_attr_any(layout, "n_cells"))
    if n_effective <= 0:
        raise ValueError("n_effective must be positive")
    if n_effective > n_fine:
        raise ValueError(
            f"n_effective={n_effective} exceeds fine-cell count n_fine={n_fine}"
        )

    boundaries = np.linspace(0, n_fine, n_effective + 1)
    starts = np.floor(boundaries[:-1]).astype(int)
    ends = np.floor(boundaries[1:]).astype(int)
    ends[-1] = n_fine

    # Ensure no empty groups when n_effective <= n_fine.
    for i in range(n_effective):
        if ends[i] <= starts[i]:
            ends[i] = starts[i] + 1
        if i + 1 < n_effective and starts[i + 1] < ends[i]:
            starts[i + 1] = ends[i]

    arrays = {
        "length_m": layout_array(layout, "length_m"),
        "L_series_H": layout_array(layout, "L_series_H"),
        "C_shunt_F": layout_array(layout, "C_shunt_F"),
        "R_series_ohm": layout_array(layout, "R_series_ohm"),
        "G_shunt_S": layout_array(layout, "G_shunt_S"),
        "C_stub_F": layout_array(layout, "C_stub_F"),
        "L_res_H": layout_array(layout, "L_res_H"),
        "C_res_F": layout_array(layout, "C_res_F"),
        "C_couple_F": layout_array(layout, "C_couple_F"),
    }

    grouped: dict[str, list[float]] = {key: [] for key in arrays}

    for start, end in zip(starts, ends):
        sl = slice(int(start), int(end))

        for key in [
            "length_m",
            "L_series_H",
            "C_shunt_F",
            "R_series_ohm",
            "G_shunt_S",
            "C_stub_F",
            "C_res_F",
            "C_couple_F",
        ]:
            grouped[key].append(float(jnp.sum(arrays[key][sl])))

        L_res_slice = np.asarray(arrays["L_res_H"][sl])
        nonzero = L_res_slice[np.abs(L_res_slice) > 0.0]
        grouped["L_res_H"].append(float(np.mean(nonzero)) if nonzero.size else 0.0)

    coarsened = call_with_supported_kwargs(
        make_layout_from_arrays,
        {
            "length_m": jnp.asarray(grouped["length_m"], dtype=jnp.float64),
            "L_series_H": jnp.asarray(grouped["L_series_H"], dtype=jnp.float64),
            "C_shunt_F": jnp.asarray(grouped["C_shunt_F"], dtype=jnp.float64),
            "R_series_ohm": jnp.asarray(grouped["R_series_ohm"], dtype=jnp.float64),
            "G_shunt_S": jnp.asarray(grouped["G_shunt_S"], dtype=jnp.float64),
            "C_stub_F": jnp.asarray(grouped["C_stub_F"], dtype=jnp.float64),
            "L_res_H": jnp.asarray(grouped["L_res_H"], dtype=jnp.float64),
            "C_res_F": jnp.asarray(grouped["C_res_F"], dtype=jnp.float64),
            "C_couple_F": jnp.asarray(grouped["C_couple_F"], dtype=jnp.float64),
            "z0_ohm": float(get_attr_any(layout, "z0_ohm", default=50.0)),
            "name": name,
            "metadata": {
                "source": "scripts.effective_cell_convergence.conservative_group_layout",
                "fine_layout": layout_summary(layout),
                "n_fine": n_fine,
                "n_effective": n_effective,
                "group_size_min": int(np.min(ends - starts)),
                "group_size_max": int(np.max(ends - starts)),
            },
        },
    )

    fine_total_length = float(jnp.sum(arrays["length_m"]))
    coarse_total_length = float(jnp.sum(layout_array(coarsened, "length_m")))

    summary = {
        "method": CoarseningMethod.CONSERVATIVE_GROUPING.value,
        "n_fine": n_fine,
        "n_effective": n_effective,
        "group_size_min": int(np.min(ends - starts)),
        "group_size_max": int(np.max(ends - starts)),
        "fine_total_length_m": fine_total_length,
        "coarse_total_length_m": coarse_total_length,
        "length_error_m": coarse_total_length - fine_total_length,
        "layout": layout_summary(coarsened),
    }

    return coarsened, summary


def try_package_coarsening(
    layout: Any,
    n_effective: int,
    *,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    import twpa.linear.coarsening as coarsening_module

    candidate_names = [
        "coarsen_layout",
        "make_effective_layout",
        "effective_cell_layout",
        "coarsen_line_layout",
    ]

    errors: list[str] = []

    for fn_name in candidate_names:
        fn = getattr(coarsening_module, fn_name, None)
        if fn is None:
            continue

        kwargs = {
            "layout": layout,
            "n_effective": n_effective,
            "n_cells": n_effective,
            "target_n_cells": n_effective,
            "name": name,
        }

        try:
            coarsened = call_with_supported_kwargs(fn, kwargs)
            return coarsened, {
                "method": f"package_{fn_name}",
                "n_fine": int(get_attr_any(layout, "n_cells")),
                "n_effective": int(get_attr_any(coarsened, "n_cells", default=n_effective)),
                "layout": layout_summary(coarsened),
                "messages": (f"PASS: package coarsening `{fn_name}` succeeded.",),
            }
        except Exception as exc:
            errors.append(f"{fn_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No compatible package coarsening function succeeded. Errors:\n"
        + "\n".join(errors[-10:])
    )


def coarsen_layout(
    layout: Any,
    n_effective: int,
    *,
    config: EffectiveCellConvergenceConfig,
) -> tuple[Any, dict[str, Any]]:
    name = f"{config.name}_effective_{n_effective}"

    if config.coarsening_method in {CoarseningMethod.AUTO, CoarseningMethod.PACKAGE}:
        try:
            return try_package_coarsening(layout, n_effective, name=name)
        except Exception as exc:
            if config.coarsening_method == CoarseningMethod.PACKAGE:
                raise
            package_error = f"{type(exc).__name__}: {exc}"
    else:
        package_error = "package coarsening disabled"

    coarsened, summary = conservative_group_layout(layout, n_effective, name=name)
    summary["package_error"] = package_error
    return coarsened, summary


def run_linear_scan(layout: Any, frequency_hz: jax.Array) -> tuple[Any, dict[str, Any]]:
    from twpa.linear.cascade import run_linear_scan as package_run_linear_scan

    scan = package_run_linear_scan(frequency_hz, layout)

    s = get_attr_any(scan, "s", default=None)
    s21 = get_attr_any(scan, "s21", default=None)
    s21_db = get_attr_any(scan, "s21_db", default=None)

    if s21 is None and s is not None:
        s21 = jnp.asarray(s)[:, 1, 0]

    if s21_db is None and s21 is not None:
        s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(s21), 1e-300))

    if s21_db is None:
        raise RuntimeError("Linear scan does not expose s21_db, s21, or s.")

    finite = bool(jnp.all(jnp.isfinite(jnp.asarray(s21_db))))

    summary = {
        "finite": finite,
        "frequency_hz": array_summary(frequency_hz),
        "s21_db": array_summary(s21_db),
        "s21": None if s21 is None else array_summary(s21),
        "s": None if s is None else array_summary(s),
        "scan": jsonify(scan),
    }

    return scan, summary


def extract_scan_arrays(scan: Any) -> dict[str, jax.Array]:
    frequency_hz = get_attr_any(scan, "frequency_hz", default=None)
    s = get_attr_any(scan, "s", default=None)
    s21 = get_attr_any(scan, "s21", default=None)
    s21_db = get_attr_any(scan, "s21_db", default=None)

    if frequency_hz is None:
        raise RuntimeError("Scan has no frequency_hz.")

    if s21 is None and s is not None:
        s21 = jnp.asarray(s)[:, 1, 0]

    if s21_db is None and s21 is not None:
        s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(s21), 1e-300))

    if s21_db is None:
        raise RuntimeError("Scan has no S21 observable.")

    out = {
        "frequency_hz": jnp.asarray(frequency_hz, dtype=jnp.float64),
        "s21_db": jnp.asarray(s21_db, dtype=jnp.float64),
    }

    if s21 is not None:
        out["s21"] = jnp.asarray(s21, dtype=jnp.complex128)
    if s is not None:
        out["s"] = jnp.asarray(s, dtype=jnp.complex128)

    return out


def phase_group_delay_ps(frequency_hz: Any, s21: Any) -> np.ndarray:
    f = np.asarray(frequency_hz, dtype=float)
    s = np.asarray(s21, dtype=np.complex128)
    omega = 2.0 * np.pi * f
    phase = np.unwrap(np.angle(s))
    gd = -np.gradient(phase, omega)
    return gd * 1e12


def compare_to_reference(
    reference_scan: Any,
    candidate_scan: Any,
    *,
    config: EffectiveCellConvergenceConfig,
) -> dict[str, Any]:
    ref = extract_scan_arrays(reference_scan)
    cand = extract_scan_arrays(candidate_scan)

    f_ref = np.asarray(ref["frequency_hz"], dtype=float)
    f_cand = np.asarray(cand["frequency_hz"], dtype=float)

    if f_ref.shape != f_cand.shape or np.nanmax(np.abs(f_ref - f_cand)) > 1e-6:
        raise ValueError("Reference and candidate frequency grids do not match.")

    ref_s21_db = np.asarray(ref["s21_db"], dtype=float)
    cand_s21_db = np.asarray(cand["s21_db"], dtype=float)
    err_db = cand_s21_db - ref_s21_db

    rms_db = float(np.sqrt(np.nanmean(err_db**2)))
    max_abs_db = float(np.nanmax(np.abs(err_db)))
    mean_abs_db = float(np.nanmean(np.abs(err_db)))

    metrics: dict[str, Any] = {
        "rms_s21_error_db": rms_db,
        "max_abs_s21_error_db": max_abs_db,
        "mean_abs_s21_error_db": mean_abs_db,
        "candidate_s21_db_min": float(np.nanmin(cand_s21_db)),
        "candidate_s21_db_max": float(np.nanmax(cand_s21_db)),
        "reference_s21_db_min": float(np.nanmin(ref_s21_db)),
        "reference_s21_db_max": float(np.nanmax(ref_s21_db)),
    }

    gd_rms_ps = None
    gd_max_ps = None

    if "s21" in ref and "s21" in cand:
        ref_gd = phase_group_delay_ps(ref["frequency_hz"], ref["s21"])
        cand_gd = phase_group_delay_ps(cand["frequency_hz"], cand["s21"])
        gd_err = cand_gd - ref_gd

        gd_rms_ps = float(np.sqrt(np.nanmean(gd_err**2)))
        gd_max_ps = float(np.nanmax(np.abs(gd_err)))

        metrics.update(
            {
                "rms_group_delay_error_ps": gd_rms_ps,
                "max_abs_group_delay_error_ps": gd_max_ps,
            }
        )

    passed = (
        rms_db <= config.pass_rms_db
        and max_abs_db <= config.pass_max_db
        and (
            gd_rms_ps is None
            or gd_rms_ps <= config.pass_group_delay_rms_ps
        )
    )

    metrics.update(
        {
            "passed_thresholds": passed,
            "pass_rms_db": config.pass_rms_db,
            "pass_max_db": config.pass_max_db,
            "pass_group_delay_rms_ps": config.pass_group_delay_rms_ps,
        }
    )

    return metrics


def run_coarsened_case(
    *,
    reference_layout: Any,
    reference_scan: Any,
    n_effective: int,
    config: EffectiveCellConvergenceConfig,
) -> tuple[CoarsenedRunResult, Any | None]:
    start = time.perf_counter()
    messages: list[str] = []

    try:
        coarsened_layout, coarsening_summary = coarsen_layout(
            reference_layout,
            n_effective,
            config=config,
        )
        messages.extend(coarsening_summary.get("messages", ()))

        scan, scan_summary = run_linear_scan(coarsened_layout, config.frequency_hz)

        if not scan_summary.get("finite", False):
            status = RunStatus.FAIL
            messages.append("FAIL: coarsened linear scan contains non-finite S21.")
            metrics = {}
        else:
            metrics = compare_to_reference(reference_scan, scan, config=config)
            if metrics["passed_thresholds"]:
                status = RunStatus.PASS
                messages.append("PASS: coarsened response satisfies convergence thresholds.")
            else:
                status = RunStatus.PARTIAL
                messages.append("PARTIAL: coarsened response ran but did not satisfy convergence thresholds.")

        return (
            CoarsenedRunResult(
                effective_cells=int(n_effective),
                status=status,
                elapsed_s=time.perf_counter() - start,
                coarsening_summary=coarsening_summary,
                scan_summary=scan_summary,
                metrics=metrics,
                messages=tuple(messages),
            ),
            scan,
        )

    except Exception as exc:
        return (
            CoarsenedRunResult(
                effective_cells=int(n_effective),
                status=RunStatus.ERROR,
                elapsed_s=time.perf_counter() - start,
                coarsening_summary={},
                scan_summary={},
                metrics={},
                messages=(f"ERROR: {type(exc).__name__}: {exc}", traceback.format_exc()),
            ),
            None,
        )


def write_layout_component_csv(layout: Any, path: Path) -> Path:
    try:
        from twpa.io.netlist import write_layout_component_csv as package_write

        return package_write(layout, path)
    except Exception:
        n = int(get_attr_any(layout, "n_cells", default=0))
        if n <= 0:
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
        arrays = {key: np.asarray(layout_array(layout, key), dtype=float) for key in keys}

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cell_index", *keys])
            writer.writeheader()
            for i in range(n):
                row = {"cell_index": i}
                row.update({key: float(arrays[key][i]) for key in keys})
                writer.writerow(row)

        return path


def write_convergence_csv(path: Path, runs: Sequence[CoarsenedRunResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "effective_cells",
        "status",
        "elapsed_s",
        "rms_s21_error_db",
        "max_abs_s21_error_db",
        "mean_abs_s21_error_db",
        "rms_group_delay_error_ps",
        "max_abs_group_delay_error_ps",
        "passed_thresholds",
        "method",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for run in runs:
            row = {
                "effective_cells": run.effective_cells,
                "status": run.status.value,
                "elapsed_s": run.elapsed_s,
                "method": run.coarsening_summary.get("method"),
            }
            row.update({key: run.metrics.get(key) for key in fields if key in run.metrics})
            writer.writerow(row)

    return path


def write_scan_npz_bundle(
    path: Path,
    *,
    reference_scan: Any,
    candidate_scans: Mapping[int, Any],
    config: EffectiveCellConvergenceConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    ref = extract_scan_arrays(reference_scan)
    payload: dict[str, Any] = {
        "frequency_hz": np.asarray(ref["frequency_hz"]),
        "reference_s21_db": np.asarray(ref["s21_db"]),
        "metadata_json": json.dumps(jsonify({"config": config.to_dict()})),
    }
    if "s21" in ref:
        payload["reference_s21"] = np.asarray(ref["s21"])

    for n_eff, scan in candidate_scans.items():
        arrs = extract_scan_arrays(scan)
        payload[f"effective_{n_eff}_s21_db"] = np.asarray(arrs["s21_db"])
        if "s21" in arrs:
            payload[f"effective_{n_eff}_s21"] = np.asarray(arrs["s21"])

    np.savez_compressed(path, **payload)
    return path


def write_plots(
    output_dir: Path,
    *,
    reference_scan: Any,
    candidate_scans: Mapping[int, Any],
    runs: Sequence[CoarsenedRunResult],
    config: EffectiveCellConvergenceConfig,
) -> dict[str, str]:
    paths: dict[str, str] = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        err = output_dir / "plotting_unavailable.txt"
        err.write_text(str(exc), encoding="utf-8")
        return {"plotting_unavailable_txt": str(err)}

    ref = extract_scan_arrays(reference_scan)
    f_ghz = np.asarray(ref["frequency_hz"], dtype=float) / 1e9
    ref_s21 = np.asarray(ref["s21_db"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(f_ghz, ref_s21, label=f"reference {config.reference_cells}")
    for n_eff in sorted(candidate_scans):
        arrs = extract_scan_arrays(candidate_scans[n_eff])
        ax.plot(f_ghz, np.asarray(arrs["s21_db"], dtype=float), label=f"{n_eff}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.set_title("Effective-cell S21 convergence")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    p = output_dir / "effective_cell_s21_overlay.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths["s21_overlay_png"] = str(p)

    good_runs = [r for r in runs if r.metrics]
    if good_runs:
        x = np.asarray([r.effective_cells for r in good_runs], dtype=float)
        rms = np.asarray([r.metrics.get("rms_s21_error_db", np.nan) for r in good_runs], dtype=float)
        maxerr = np.asarray([r.metrics.get("max_abs_s21_error_db", np.nan) for r in good_runs], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        ax.loglog(x, rms, marker="o", label="RMS |ΔS21|")
        ax.loglog(x, maxerr, marker="s", label="max |ΔS21|")
        ax.axhline(config.pass_rms_db, linestyle="--", linewidth=1.0, label="RMS threshold")
        ax.axhline(config.pass_max_db, linestyle=":", linewidth=1.0, label="max threshold")
        ax.set_xlabel("Effective cells")
        ax.set_ylabel("Error (dB)")
        ax.set_title("S21 convergence versus effective-cell count")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        p = output_dir / "effective_cell_error_convergence.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths["error_convergence_png"] = str(p)

        gd_values = [
            r.metrics.get("rms_group_delay_error_ps")
            for r in good_runs
            if r.metrics.get("rms_group_delay_error_ps") is not None
        ]
        if gd_values:
            gd = np.asarray(
                [
                    r.metrics.get("rms_group_delay_error_ps", np.nan)
                    for r in good_runs
                ],
                dtype=float,
            )
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
            ax.loglog(x, gd, marker="o")
            ax.axhline(config.pass_group_delay_rms_ps, linestyle="--", linewidth=1.0)
            ax.set_xlabel("Effective cells")
            ax.set_ylabel("RMS group-delay error (ps)")
            ax.set_title("Group-delay convergence")
            ax.grid(True)
            fig.tight_layout()
            p = output_dir / "effective_cell_group_delay_convergence.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            paths["group_delay_convergence_png"] = str(p)

    return paths


def export_artifacts(
    *,
    config: EffectiveCellConvergenceConfig,
    stages: list[StageResult],
    runs: Sequence[CoarsenedRunResult],
    reference_layout: Any,
    reference_scan: Any,
    candidate_scans: Mapping[int, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    ref_csv = write_layout_component_csv(
        reference_layout,
        output_dir / "reference_layout_components.csv",
    )
    paths["reference_layout_components_csv"] = str(ref_csv)

    convergence_csv = write_convergence_csv(
        output_dir / "effective_cell_convergence.csv",
        runs,
    )
    paths["convergence_csv"] = str(convergence_csv)

    scans_npz = write_scan_npz_bundle(
        output_dir / "effective_cell_scans.npz",
        reference_scan=reference_scan,
        candidate_scans=candidate_scans,
        config=config,
    )
    paths["scans_npz"] = str(scans_npz)

    runs_json = output_dir / "effective_cell_runs.json"
    runs_json.write_text(
        json.dumps(jsonify([r.to_dict() for r in runs]), indent=2),
        encoding="utf-8",
    )
    paths["runs_json"] = str(runs_json)

    if config.export_netlists:
        try:
            from twpa.io.netlist import write_netlist_bundle

            ref_paths = write_netlist_bundle(
                reference_layout,
                output_dir / "netlists" / "reference",
                prefix="reference_layout",
            )
            paths.update({f"reference_netlist_{k}": v for k, v in ref_paths.items()})
        except Exception as exc:
            err = output_dir / "reference_netlist_error.txt"
            err.write_text(traceback.format_exc(), encoding="utf-8")
            paths["reference_netlist_error_txt"] = str(err)
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

            ref_arrays = extract_scan_arrays(reference_scan)
            arrays = {
                "frequency_hz": ref_arrays["frequency_hz"],
                "reference_s21_db": ref_arrays["s21_db"],
            }
            if "s21" in ref_arrays:
                arrays["reference_s21"] = ref_arrays["s21"]

            for n_eff, scan in candidate_scans.items():
                scan_arrays = extract_scan_arrays(scan)
                arrays[f"effective_{n_eff}_s21_db"] = scan_arrays["s21_db"]
                if "s21" in scan_arrays:
                    arrays[f"effective_{n_eff}_s21"] = scan_arrays["s21"]

            checkpoint_path = output_dir / "effective_cell_convergence_checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.LINEAR_SCAN,
                    name="effective_cell_convergence",
                    source="scripts.effective_cell_convergence",
                    extra={
                        "config": config.to_dict(),
                        "reference_layout": layout_summary(reference_layout),
                    },
                ),
                arrays=arrays,
                payload={
                    "runs": [r.to_dict() for r in runs],
                    "reference_layout": layout_summary(reference_layout),
                },
            )
            paths["checkpoint_npz"] = str(checkpoint_path)

        except Exception as exc:
            err = output_dir / "checkpoint_error.txt"
            err.write_text(traceback.format_exc(), encoding="utf-8")
            paths["checkpoint_error_txt"] = str(err)
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
        plot_paths = write_plots(
            output_dir,
            reference_scan=reference_scan,
            candidate_scans=candidate_scans,
            runs=runs,
            config=config,
        )
        paths.update(plot_paths)

    return paths


def result_markdown(result: EffectiveCellConvergenceResult) -> str:
    cfg = result.config

    lines = [
        "# Effective-cell convergence",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- reference cells: `{cfg.reference_cells}`",
        f"- effective-cell candidates: `{list(cfg.effective_cells)}`",
        f"- frequency range: `{cfg.f_min_ghz:.6g}`–`{cfg.f_max_ghz:.6g} GHz`",
        f"- frequency points: `{cfg.n_frequency}`",
        f"- coarsening method: `{cfg.coarsening_method.value}`",
        f"- passed runs: `{result.n_passed}/{result.n_runs}`",
        "",
        "## Coarsened runs",
        "",
        "| effective cells | status | elapsed s | RMS S21 error dB | max S21 error dB | RMS group-delay error ps | method |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]

    for run in result.runs:
        lines.append(
            f"| {run.effective_cells} | `{run.status.value}` | "
            f"{run.elapsed_s:.6g} | "
            f"{run.metrics.get('rms_s21_error_db', '')} | "
            f"{run.metrics.get('max_abs_s21_error_db', '')} | "
            f"{run.metrics.get('rms_group_delay_error_ps', '')} | "
            f"`{run.coarsening_summary.get('method', '')}` |"
        )

    lines += [
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
        "## Reference and configuration summary",
        "",
    ]

    for stage in result.stages:
        if stage.name in {"build_reference_layout", "reference_scan"}:
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
        description="Study TWPA effective-cell convergence against a fine reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--reference-cells", type=int, default=20000)
    parser.add_argument(
        "--effective-cells",
        type=int,
        nargs="+",
        default=[250, 500, 1000, 2000, 5000, 10000],
    )
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)

    parser.add_argument("--f-min-ghz", type=float, default=1.0)
    parser.add_argument("--f-max-ghz", type=float, default=14.0)
    parser.add_argument("--n-frequency", type=int, default=501)

    parser.add_argument("--layout-csv", type=str, default=None)
    parser.add_argument("--layout-kind", type=str, default="uniform")
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument("--disorder-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--coarsening-method",
        choices=[m.value for m in CoarseningMethod],
        default=CoarseningMethod.AUTO.value,
    )
    parser.add_argument("--pass-rms-db", type=float, default=0.10)
    parser.add_argument("--pass-max-db", type=float, default=0.50)
    parser.add_argument("--pass-group-delay-rms-ps", type=float, default=5.0)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/effective_cell_convergence"))
    parser.add_argument("--name", type=str, default="effective_cell_convergence")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--export-netlists", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> EffectiveCellConvergenceConfig:
    reference_cells = int(args.reference_cells)
    effective_cells = tuple(int(x) for x in args.effective_cells)
    n_frequency = int(args.n_frequency)

    if args.quick:
        if reference_cells == 20000:
            reference_cells = 2000
        effective_cells = tuple(x for x in effective_cells if x <= reference_cells)
        if not effective_cells:
            effective_cells = (100, 250, 500, 1000)
        effective_cells = tuple(min(x, reference_cells) for x in effective_cells)
        if n_frequency == 501:
            n_frequency = 201

    effective_cells = tuple(sorted(set(effective_cells)))

    if reference_cells <= 0:
        raise ValueError("--reference-cells must be positive")
    if not effective_cells:
        raise ValueError("--effective-cells may not be empty")
    if any(x <= 0 for x in effective_cells):
        raise ValueError("All --effective-cells values must be positive")
    if any(x > reference_cells for x in effective_cells):
        raise ValueError("Effective-cell counts may not exceed --reference-cells")
    if args.length_mm <= 0.0:
        raise ValueError("--length-mm must be positive")
    if args.z0_ohm <= 0.0:
        raise ValueError("--z0-ohm must be positive")
    if args.phase_velocity_m_per_s <= 0.0:
        raise ValueError("--phase-velocity-m-per-s must be positive")
    if args.f_min_ghz <= 0.0 or args.f_max_ghz <= args.f_min_ghz:
        raise ValueError("Require 0 < --f-min-ghz < --f-max-ghz")
    if n_frequency < 2:
        raise ValueError("--n-frequency must be at least 2")
    if args.disorder_std < 0.0:
        raise ValueError("--disorder-std must be non-negative")
    if args.pass_rms_db < 0.0 or args.pass_max_db < 0.0:
        raise ValueError("Pass thresholds must be non-negative")
    if args.pass_group_delay_rms_ps < 0.0:
        raise ValueError("--pass-group-delay-rms-ps must be non-negative")

    if args.layout_csv is not None and not Path(args.layout_csv).exists():
        raise FileNotFoundError(args.layout_csv)

    return EffectiveCellConvergenceConfig(
        reference_cells=reference_cells,
        effective_cells=effective_cells,
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        f_min_ghz=float(args.f_min_ghz),
        f_max_ghz=float(args.f_max_ghz),
        n_frequency=n_frequency,
        layout_csv=args.layout_csv,
        layout_kind=str(args.layout_kind),
        include_resonators=bool(args.include_resonators),
        disorder_std=float(args.disorder_std),
        seed=int(args.seed),
        coarsening_method=CoarseningMethod(args.coarsening_method),
        pass_rms_db=float(args.pass_rms_db),
        pass_max_db=float(args.pass_max_db),
        pass_group_delay_rms_ps=float(args.pass_group_delay_rms_ps),
        output_dir=str(args.output_dir),
        name=str(args.name),
        quick=bool(args.quick),
        make_plots=not bool(args.no_plots),
        save_checkpoint=not bool(args.no_checkpoint),
        export_csv=not bool(args.no_csv),
        export_netlists=bool(args.export_netlists),
    )


def _stage_build_reference_layout(
    holders: dict[str, Any],
    config: EffectiveCellConvergenceConfig,
) -> dict[str, Any]:
    layout, summary = build_or_load_reference_layout(config)
    holders["reference_layout"] = layout
    return {
        "status": RunStatus.PASS.value,
        **summary,
        "messages": ("PASS: reference layout built/loaded.",),
    }


def _stage_reference_scan(
    holders: dict[str, Any],
    config: EffectiveCellConvergenceConfig,
) -> dict[str, Any]:
    scan, summary = run_linear_scan(
        holders["reference_layout"],
        config.frequency_hz,
    )
    holders["reference_scan"] = scan

    finite = bool(summary.get("finite", False))
    return {
        "status": RunStatus.PASS.value if finite else RunStatus.FAIL.value,
        **summary,
        "messages": (
            "PASS: reference linear scan completed."
            if finite
            else "FAIL: reference linear scan returned non-finite S21."
        ),
    }


def finalize_result(result: EffectiveCellConvergenceResult, output_dir: Path) -> int:
    summary_json = output_dir / "effective_cell_convergence_summary.json"
    summary_md = output_dir / "effective_cell_convergence_summary.md"

    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }

    result = EffectiveCellConvergenceResult(
        config=result.config,
        status=result.status,
        elapsed_s=result.elapsed_s,
        stages=result.stages,
        runs=result.runs,
        artifact_paths=artifact_paths,
        metadata=result.metadata,
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    summary_md.write_text(result_markdown(result), encoding="utf-8")

    print()
    print(f"[effective-cell] status: {result.status.value}")
    print(f"[effective-cell] passed runs: {result.n_passed}/{result.n_runs}")
    print(f"[effective-cell] summary JSON: {summary_json}")
    print(f"[effective-cell] summary MD:   {summary_md}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[effective-cell] invalid arguments: {exc}", file=sys.stderr)
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
        "script": "scripts/effective_cell_convergence.py",
    }

    holders: dict[str, Any] = {}
    stages: list[StageResult] = []
    runs: list[CoarsenedRunResult] = []
    candidate_scans: dict[int, Any] = {}
    artifacts: dict[str, str] = {}

    print("[effective-cell] building/loading reference layout...")
    stage = run_stage("build_reference_layout", lambda: _stage_build_reference_layout(holders, config))
    stages.append(stage)
    print(f"[effective-cell] build_reference_layout: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = EffectiveCellConvergenceResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            runs=tuple(runs),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[effective-cell] running reference scan...")
    stage = run_stage("reference_scan", lambda: _stage_reference_scan(holders, config))
    stages.append(stage)
    print(f"[effective-cell] reference_scan: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = EffectiveCellConvergenceResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            runs=tuple(runs),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    reference_layout = holders["reference_layout"]
    reference_scan = holders["reference_scan"]

    for n_eff in config.effective_cells:
        print(f"[effective-cell] coarsened run n_effective={n_eff}...")
        run, scan = run_coarsened_case(
            reference_layout=reference_layout,
            reference_scan=reference_scan,
            n_effective=n_eff,
            config=config,
        )
        runs.append(run)
        if scan is not None:
            candidate_scans[n_eff] = scan
        print(
            f"[effective-cell] n={n_eff}: {run.status.value}, "
            f"rms={run.metrics.get('rms_s21_error_db', 'NA')}"
        )

    stages.append(
        StageResult(
            name="coarsened_runs",
            status=RunStatus.PASS if all(r.status != RunStatus.ERROR for r in runs) else RunStatus.PARTIAL,
            elapsed_s=sum(r.elapsed_s for r in runs),
            summary={
                "n_runs": len(runs),
                "n_pass": sum(1 for r in runs if r.status == RunStatus.PASS),
                "n_partial": sum(1 for r in runs if r.status == RunStatus.PARTIAL),
                "n_error": sum(1 for r in runs if r.status == RunStatus.ERROR),
            },
            messages=("PASS: coarsened cases completed." if all(r.status != RunStatus.ERROR for r in runs) else "PARTIAL: at least one coarsened case errored.",),
        )
    )

    print("[effective-cell] exporting artifacts...")
    try:
        artifacts.update(
            export_artifacts(
                config=config,
                stages=stages,
                runs=runs,
                reference_layout=reference_layout,
                reference_scan=reference_scan,
                candidate_scans=candidate_scans,
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
    any_run_error = any(r.status == RunStatus.ERROR for r in runs)
    all_runs_pass = bool(runs) and all(r.status == RunStatus.PASS for r in runs)

    if hard_fail:
        status = RunStatus.ERROR
    elif any_run_error:
        status = RunStatus.PARTIAL
    elif all_runs_pass:
        status = RunStatus.PASS
    else:
        status = RunStatus.PARTIAL

    result = EffectiveCellConvergenceResult(
        config=config,
        status=status,
        elapsed_s=time.perf_counter() - start,
        stages=tuple(stages),
        runs=tuple(runs),
        artifact_paths=artifacts,
        metadata=metadata,
    )

    return finalize_result(result, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
