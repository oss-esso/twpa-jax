"""
Extract dispersion diagnostics from a TWPA linear response.

This script can operate in two modes:

1. Read an existing linear-scan NPZ containing frequency_hz plus either:
       s
       s21
       s21_real/s21_imag
       s21_db

2. Build/load a layout, run a pump-off linear scan, and then extract dispersion.

Examples
--------
From a previous linear baseline NPZ:

    python scripts/extract_dispersion.py ^
      --input-npz outputs/linear_100mm_baseline/linear_100mm_baseline_arrays.npz ^
      --length-mm 100 ^
      --output-dir outputs/dispersion_from_npz

From a component CSV layout:

    python scripts/extract_dispersion.py ^
      --layout-csv outputs/linear_100mm_baseline/linear_100mm_layout_components.csv ^
      --f-min-ghz 1 ^
      --f-max-ghz 14 ^
      --n-frequency 1001 ^
      --output-dir outputs/dispersion_from_layout

Quick fallback synthetic layout:

    python scripts/extract_dispersion.py --quick --output-dir outputs/dispersion_quick
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
from typing import Any, Callable, Mapping

import numpy as np

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


C0_M_PER_S = 299_792_458.0


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class DispersionMethod(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    S21_PHASE = "s21_phase"


class PhaseSign(str, Enum):
    AUTO = "auto"
    NEGATIVE_PHASE = "negative_phase"
    POSITIVE_PHASE = "positive_phase"


@dataclass(frozen=True)
class ExtractDispersionConfig:
    input_npz: str | None
    layout_csv: str | None
    output_dir: str

    n_cells: int
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float

    f_min_ghz: float
    f_max_ghz: float
    n_frequency: int

    method: DispersionMethod
    phase_sign: PhaseSign
    unwrap_phase: bool
    remove_linear_phase_offset: bool
    stopband_threshold_db: float
    stopband_margin_db: float

    layout_kind: str
    include_resonators: bool
    disorder_std: float
    seed: int
    quick: bool

    make_plots: bool
    save_checkpoint: bool
    export_csv: bool
    name: str

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
        d["method"] = self.method.value
        d["phase_sign"] = self.phase_sign.value
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
class DispersionExtractionResult:
    config: ExtractDispersionConfig
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


def infer_total_length_m(layout: Any | None, config: ExtractDispersionConfig) -> float:
    if layout is not None:
        length_m = get_attr_any(layout, "length_m", default=None)
        if length_m is not None:
            total = float(jnp.sum(jnp.asarray(length_m)))
            if total > 0.0:
                return total

        total = get_attr_any(layout, "total_length_m", default=None)
        if total is not None and float(total) > 0.0:
            return float(total)

    if config.length_m <= 0.0:
        raise ValueError("Could not infer physical length; pass --length-mm.")
    return config.length_m


def load_npz_linear_response(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    npz = np.load(p, allow_pickle=True)

    if "frequency_hz" not in npz:
        raise ValueError(f"{p}: expected frequency_hz in NPZ.")

    frequency_hz = jnp.asarray(npz["frequency_hz"], dtype=jnp.float64)

    s = None
    s21 = None
    s21_db = None

    if "s" in npz:
        s = jnp.asarray(npz["s"], dtype=jnp.complex128)
        if s.ndim != 3 or s.shape[1:] != (2, 2):
            raise ValueError(f"{p}: s must have shape (F, 2, 2), got {s.shape}")
        s21 = s[:, 1, 0]

    if "s21" in npz:
        s21 = jnp.asarray(npz["s21"], dtype=jnp.complex128)

    if "s21_real" in npz and "s21_imag" in npz:
        s21 = jnp.asarray(npz["s21_real"], dtype=jnp.float64) + 1j * jnp.asarray(
            npz["s21_imag"],
            dtype=jnp.float64,
        )

    if "s21_db" in npz:
        s21_db = jnp.asarray(npz["s21_db"], dtype=jnp.float64)

    if s21_db is None and s21 is not None:
        s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(s21), 1e-300))

    metadata = {}
    if "metadata_json" in npz:
        try:
            raw = npz["metadata_json"]
            if hasattr(raw, "item"):
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            metadata = json.loads(str(raw))
        except Exception as exc:
            metadata = {"metadata_parse_error": str(exc)}

    if s21 is None and s21_db is None:
        raise ValueError(f"{p}: expected s, s21, s21_real/s21_imag, or s21_db.")

    return {
        "frequency_hz": frequency_hz,
        "s": s,
        "s21": s21,
        "s21_db": s21_db,
        "metadata": {
            "source": "load_npz_linear_response",
            "path": str(p),
            **metadata,
        },
    }


def load_layout_from_csv(config: ExtractDispersionConfig) -> Any:
    from twpa.io.netlist import load_layout_component_csv

    if config.layout_csv is None:
        raise ValueError("layout_csv is None")

    return load_layout_component_csv(
        config.layout_csv,
        z0_ohm=config.z0_ohm,
        name=config.name,
        metadata={
            "source": "scripts.extract_dispersion",
            "layout_csv": config.layout_csv,
        },
    )


def build_uniform_layout_fallback(config: ExtractDispersionConfig) -> Any:
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
        period = max(1, n // 64)
        idx = np.arange(n)
        mask = idx % period == 0
        C_stub[mask] = 2e-15
        L_res[mask] = 1e-9
        C_res[mask] = 5e-15
        C_couple[mask] = 1e-15

    kwargs = {
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
        "name": config.name,
        "metadata": {
            "source": "scripts.extract_dispersion.build_uniform_layout_fallback",
            "n_cells": n,
            "length_m": config.length_m,
            "dx_m": dx,
            "phase_velocity_m_per_s": vp,
            "z0_ohm": z0,
            "L_cell_nominal_H": L_cell,
            "C_cell_nominal_F": C_cell,
            "disorder_std": config.disorder_std,
            "seed": config.seed,
            "include_resonators": config.include_resonators,
        },
    }

    return call_with_supported_kwargs(make_layout_from_arrays, kwargs)


def build_or_load_layout(config: ExtractDispersionConfig) -> tuple[Any, dict[str, Any]]:
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


def run_linear_scan_for_layout(layout: Any, config: ExtractDispersionConfig) -> dict[str, Any]:
    from twpa.linear.cascade import run_linear_scan

    scan = run_linear_scan(config.frequency_hz, layout)

    s = get_attr_any(scan, "s", default=None)
    s21 = get_attr_any(scan, "s21", default=None)
    s21_db = get_attr_any(scan, "s21_db", default=None)

    if s21 is None and s is not None:
        s21 = jnp.asarray(s)[:, 1, 0]

    if s21_db is None and s21 is not None:
        s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(s21), 1e-300))

    if s21 is None and s21_db is None:
        raise RuntimeError("Linear scan did not expose s, s21, or s21_db.")

    return {
        "frequency_hz": config.frequency_hz,
        "s": s,
        "s21": s21,
        "s21_db": s21_db,
        "scan": scan,
        "metadata": {
            "source": "run_linear_scan_for_layout",
            "scan": jsonify(scan),
        },
    }


def phase_unwrap_np(phase: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return phase
    return np.unwrap(phase)


def choose_phase_sign(
    frequency_hz: np.ndarray,
    phase_rad: np.ndarray,
    config: ExtractDispersionConfig,
) -> float:
    if config.phase_sign == PhaseSign.NEGATIVE_PHASE:
        return -1.0
    if config.phase_sign == PhaseSign.POSITIVE_PHASE:
        return 1.0

    # For passive forward propagation with the convention S21 ~ exp(-i beta L),
    # unwrapped phase usually decreases with frequency, so -phase/length gives
    # positive beta. Choose the sign that makes median slope-positive beta.
    slope = np.nanmedian(np.gradient(phase_rad, frequency_hz))
    return -1.0 if slope < 0.0 else 1.0


def extract_dispersion_from_s21_phase(
    *,
    frequency_hz: Any,
    s21: Any | None,
    s21_db: Any | None,
    length_m: float,
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    f = np.asarray(frequency_hz, dtype=float)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("frequency_hz must be a 1D array with at least two points.")

    omega = 2.0 * np.pi * f

    if s21 is None:
        if s21_db is None:
            raise ValueError("Need s21 or s21_db for fallback extraction.")
        # Magnitude-only data cannot yield phase/beta. Return alpha-only partial.
        mag = 10.0 ** (np.asarray(s21_db, dtype=float) / 20.0)
        alpha = -np.log(np.maximum(mag, 1e-300)) / length_m
        return {
            "status": RunStatus.PARTIAL.value,
            "method": "magnitude_only_alpha",
            "frequency_hz": jnp.asarray(f, dtype=jnp.float64),
            "omega_rad_s": jnp.asarray(omega, dtype=jnp.float64),
            "phase_rad": None,
            "beta_rad_per_m": None,
            "alpha_np_per_m": jnp.asarray(alpha, dtype=jnp.float64),
            "group_delay_s": None,
            "phase_velocity_m_per_s": None,
            "effective_index": None,
            "messages": (
                "PARTIAL: only S21 magnitude was available; extracted attenuation alpha only.",
            ),
        }

    s21_arr = np.asarray(s21, dtype=np.complex128)
    if s21_arr.shape != f.shape:
        raise ValueError(f"s21 shape {s21_arr.shape} does not match frequency shape {f.shape}")

    mag = np.abs(s21_arr)
    phase = phase_unwrap_np(np.angle(s21_arr), config.unwrap_phase)

    sign = choose_phase_sign(f, phase, config)

    beta = sign * phase / length_m

    if config.remove_linear_phase_offset:
        # Remove an arbitrary constant beta offset by anchoring beta(f_min) >= 0.
        # This avoids negative beta offsets from port-reference phase while
        # preserving dispersion slope.
        beta = beta - beta[0]
        if np.nanmedian(np.gradient(beta, f)) < 0:
            beta = -beta
        beta = beta + max(0.0, 2.0 * np.pi * f[0] / config.phase_velocity_m_per_s)

    alpha = -np.log(np.maximum(mag, 1e-300)) / length_m

    group_delay = -np.gradient(phase, omega)
    beta_slope = np.gradient(beta, omega)
    group_velocity = np.where(np.abs(beta_slope) > 1e-300, 1.0 / beta_slope, np.nan)
    phase_velocity = np.where(np.abs(beta) > 1e-300, omega / beta, np.nan)
    effective_index = beta * C0_M_PER_S / np.maximum(omega, 1e-300)

    return {
        "status": RunStatus.PASS.value,
        "method": "s21_phase",
        "frequency_hz": jnp.asarray(f, dtype=jnp.float64),
        "omega_rad_s": jnp.asarray(omega, dtype=jnp.float64),
        "s21": jnp.asarray(s21_arr, dtype=jnp.complex128),
        "s21_db": jnp.asarray(20.0 * np.log10(np.maximum(mag, 1e-300)), dtype=jnp.float64),
        "phase_rad": jnp.asarray(phase, dtype=jnp.float64),
        "beta_rad_per_m": jnp.asarray(beta, dtype=jnp.float64),
        "alpha_np_per_m": jnp.asarray(alpha, dtype=jnp.float64),
        "group_delay_s": jnp.asarray(group_delay, dtype=jnp.float64),
        "group_velocity_m_per_s": jnp.asarray(group_velocity, dtype=jnp.float64),
        "phase_velocity_m_per_s": jnp.asarray(phase_velocity, dtype=jnp.float64),
        "effective_index": jnp.asarray(effective_index, dtype=jnp.float64),
        "phase_sign_multiplier": sign,
        "length_m": length_m,
        "messages": (
            "PASS: fallback dispersion extraction from unwrapped S21 phase completed.",
        ),
    }


def try_package_dispersion(
    *,
    layout: Any | None,
    linear_response: Mapping[str, Any],
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    """
    Try package-native dispersion extraction before falling back to S21 phase.
    """
    from twpa.linear import dispersion as dispersion_module

    if layout is not None and hasattr(dispersion_module, "extract_layout_dispersion"):
        result = dispersion_module.extract_layout_dispersion(
            linear_response["frequency_hz"],
            layout,
        )
        beta = get_attr_any(
            result,
            "beta_preferred_rad_per_m",
            "beta_eff_rad_per_m",
            "beta_rad_per_m",
            default=None,
        )
        alpha = get_attr_any(
            result,
            "alpha_preferred_np_per_m",
            "alpha_np_per_m",
            default=None,
        )
        return {
            "status": RunStatus.PASS.value,
            "method": "package_extract_layout_dispersion",
            "package_result": result,
            "frequency_hz": linear_response["frequency_hz"],
            "beta_rad_per_m": beta,
            "alpha_np_per_m": alpha,
            "messages": (
                "PASS: package-native extract_layout_dispersion completed.",
            ),
        }

    candidate_names = [
        "extract_dispersion_from_s21",
        "extract_propagation_from_s21",
        "dispersion_from_s21",
    ]
    for name in candidate_names:
        fn = getattr(dispersion_module, name, None)
        if fn is None:
            continue

        kwargs = {
            "frequency_hz": linear_response["frequency_hz"],
            "s21": linear_response.get("s21"),
            "s21_db": linear_response.get("s21_db"),
            "length_m": config.length_m,
            "unwrap_phase": config.unwrap_phase,
        }
        result = call_with_supported_kwargs(fn, kwargs)
        beta = get_attr_any(
            result,
            "beta_preferred_rad_per_m",
            "beta_eff_rad_per_m",
            "beta_rad_per_m",
            default=None,
        )
        alpha = get_attr_any(
            result,
            "alpha_preferred_np_per_m",
            "alpha_np_per_m",
            default=None,
        )
        return {
            "status": RunStatus.PASS.value,
            "method": f"package_{name}",
            "package_result": result,
            "frequency_hz": linear_response["frequency_hz"],
            "beta_rad_per_m": beta,
            "alpha_np_per_m": alpha,
            "messages": (
                f"PASS: package-native {name} completed.",
            ),
        }

    raise RuntimeError("No compatible package-native dispersion extractor found.")


def normalize_dispersion_result(
    raw: Mapping[str, Any],
    *,
    linear_response: Mapping[str, Any],
    layout: Any | None,
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    frequency_hz = get_attr_any(raw, "frequency_hz", default=linear_response["frequency_hz"])
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)
    omega = 2.0 * jnp.pi * f

    s21 = get_attr_any(raw, "s21", default=linear_response.get("s21"))
    s21_db = get_attr_any(raw, "s21_db", default=linear_response.get("s21_db"))

    beta = get_attr_any(
        raw,
        "beta_preferred_rad_per_m",
        "beta_eff_rad_per_m",
        "beta_rad_per_m",
        default=raw.get("beta_rad_per_m"),
    )
    alpha = get_attr_any(
        raw,
        "alpha_preferred_np_per_m",
        "alpha_np_per_m",
        default=raw.get("alpha_np_per_m"),
    )
    phase = get_attr_any(raw, "phase_rad", default=raw.get("phase_rad"))
    group_delay = get_attr_any(raw, "group_delay_s", default=raw.get("group_delay_s"))
    phase_velocity = get_attr_any(
        raw,
        "phase_velocity_m_per_s",
        default=raw.get("phase_velocity_m_per_s"),
    )
    group_velocity = get_attr_any(
        raw,
        "group_velocity_m_per_s",
        default=raw.get("group_velocity_m_per_s"),
    )
    effective_index = get_attr_any(raw, "effective_index", default=raw.get("effective_index"))

    if beta is not None:
        beta = jnp.asarray(beta, dtype=jnp.float64)
        if phase_velocity is None:
            phase_velocity = omega / jnp.where(jnp.abs(beta) > 1e-300, beta, jnp.nan)
        if effective_index is None:
            effective_index = beta * C0_M_PER_S / jnp.maximum(omega, 1e-300)

        if group_velocity is None:
            beta_np = np.asarray(beta, dtype=float)
            omega_np = np.asarray(omega, dtype=float)
            d_beta_d_omega = np.gradient(beta_np, omega_np)
            group_velocity = jnp.asarray(
                np.where(np.abs(d_beta_d_omega) > 1e-300, 1.0 / d_beta_d_omega, np.nan),
                dtype=jnp.float64,
            )

    if s21_db is None and s21 is not None:
        s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(s21)), 1e-300))

    out = {
        "status": raw.get("status", RunStatus.PASS.value),
        "method": raw.get("method", "unknown"),
        "frequency_hz": f,
        "omega_rad_s": omega,
        "s21": None if s21 is None else jnp.asarray(s21, dtype=jnp.complex128),
        "s21_db": None if s21_db is None else jnp.asarray(s21_db, dtype=jnp.float64),
        "phase_rad": None if phase is None else jnp.asarray(phase, dtype=jnp.float64),
        "beta_rad_per_m": None if beta is None else jnp.asarray(beta, dtype=jnp.float64),
        "alpha_np_per_m": None if alpha is None else jnp.asarray(alpha, dtype=jnp.float64),
        "group_delay_s": None if group_delay is None else jnp.asarray(group_delay, dtype=jnp.float64),
        "group_velocity_m_per_s": None if group_velocity is None else jnp.asarray(group_velocity, dtype=jnp.float64),
        "phase_velocity_m_per_s": None if phase_velocity is None else jnp.asarray(phase_velocity, dtype=jnp.float64),
        "effective_index": None if effective_index is None else jnp.asarray(effective_index, dtype=jnp.float64),
        "layout": None if layout is None else layout_summary(layout),
        "raw": jsonify(raw),
        "messages": tuple(raw.get("messages", ())),
    }

    return out


def extract_dispersion(
    *,
    linear_response: Mapping[str, Any],
    layout: Any | None,
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    method = config.method
    length_m = infer_total_length_m(layout, config)

    if method in {DispersionMethod.AUTO, DispersionMethod.PACKAGE}:
        try:
            raw = try_package_dispersion(
                layout=layout,
                linear_response=linear_response,
                config=config,
            )
            return normalize_dispersion_result(
                raw,
                linear_response=linear_response,
                layout=layout,
                config=config,
            )
        except Exception as exc:
            if method == DispersionMethod.PACKAGE:
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
        else:
            fallback_reason = ""

    else:
        fallback_reason = "method explicitly set to s21_phase"

    raw = extract_dispersion_from_s21_phase(
        frequency_hz=linear_response["frequency_hz"],
        s21=linear_response.get("s21"),
        s21_db=linear_response.get("s21_db"),
        length_m=length_m,
        config=config,
    )
    raw["fallback_reason"] = fallback_reason

    return normalize_dispersion_result(
        raw,
        linear_response=linear_response,
        layout=layout,
        config=config,
    )


def detect_stopbands_fallback(
    frequency_hz: Any,
    s21_db: Any,
    *,
    threshold_db: float,
    margin_db: float,
) -> list[dict[str, Any]]:
    f = np.asarray(frequency_hz, dtype=float)
    y = np.asarray(s21_db, dtype=float)

    if f.shape != y.shape:
        raise ValueError("frequency_hz and s21_db shapes must match.")

    finite = np.isfinite(y)
    if not np.any(finite):
        return []

    baseline = float(np.nanmedian(y[finite]))
    cutoff = min(float(threshold_db), baseline - float(margin_db))

    mask = finite & (y <= cutoff)

    bands: list[dict[str, Any]] = []
    if not np.any(mask):
        return bands

    idx = np.where(mask)[0]
    start = int(idx[0])
    prev = int(idx[0])

    for current in idx[1:]:
        current = int(current)
        if current == prev + 1:
            prev = current
            continue

        bands.append(_band_from_indices(f, y, start, prev, cutoff))
        start = current
        prev = current

    bands.append(_band_from_indices(f, y, start, prev, cutoff))
    return bands


def _band_from_indices(
    frequency_hz: np.ndarray,
    s21_db: np.ndarray,
    start_idx: int,
    end_idx: int,
    cutoff_db: float,
) -> dict[str, Any]:
    local = s21_db[start_idx : end_idx + 1]
    min_local_idx = int(np.nanargmin(local)) + start_idx
    return {
        "start_index": start_idx,
        "end_index": end_idx,
        "start_frequency_hz": float(frequency_hz[start_idx]),
        "end_frequency_hz": float(frequency_hz[end_idx]),
        "center_frequency_hz": float(0.5 * (frequency_hz[start_idx] + frequency_hz[end_idx])),
        "width_hz": float(frequency_hz[end_idx] - frequency_hz[start_idx]),
        "min_s21_db": float(s21_db[min_local_idx]),
        "min_frequency_hz": float(frequency_hz[min_local_idx]),
        "cutoff_db": float(cutoff_db),
    }


def detect_stopbands(
    dispersion: Mapping[str, Any],
    config: ExtractDispersionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frequency_hz = dispersion["frequency_hz"]
    s21_db = dispersion.get("s21_db")

    if s21_db is None:
        return [], {
            "status": RunStatus.PARTIAL.value,
            "n_stopbands": 0,
            "messages": (
                "PARTIAL: no s21_db available; stopband detection skipped.",
            ),
        }

    try:
        from twpa.linear import dispersion as dispersion_module

        fn = getattr(dispersion_module, "detect_stopbands", None)
        if fn is not None:
            raw = call_with_supported_kwargs(
                fn,
                {
                    "frequency_hz": frequency_hz,
                    "s21_db": s21_db,
                    "threshold_db": config.stopband_threshold_db,
                    "margin_db": config.stopband_margin_db,
                },
            )
            if isinstance(raw, list):
                bands = [jsonify(b) for b in raw]
            else:
                bands = jsonify(raw)
                if isinstance(bands, dict) and "stopbands" in bands:
                    bands = bands["stopbands"]
                if not isinstance(bands, list):
                    bands = []
            return bands, {
                "status": RunStatus.PASS.value,
                "method": "package_detect_stopbands",
                "n_stopbands": len(bands),
                "stopbands": bands,
                "messages": (
                    "PASS: package-native stopband detection completed.",
                ),
            }

    except Exception as exc:
        package_error = f"{type(exc).__name__}: {exc}"
    else:
        package_error = "package detector unavailable"

    bands = detect_stopbands_fallback(
        frequency_hz,
        s21_db,
        threshold_db=config.stopband_threshold_db,
        margin_db=config.stopband_margin_db,
    )

    return bands, {
        "status": RunStatus.PASS.value,
        "method": "fallback_s21_threshold",
        "package_error": package_error,
        "n_stopbands": len(bands),
        "stopbands": bands,
        "messages": (
            "PASS: fallback S21-threshold stopband detection completed.",
        ),
    }


def write_dispersion_csv(
    path: Path,
    dispersion: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    frequency_hz = np.asarray(dispersion["frequency_hz"], dtype=float)
    n = frequency_hz.shape[0]

    columns: dict[str, np.ndarray] = {
        "frequency_hz": frequency_hz,
        "frequency_ghz": frequency_hz / 1e9,
    }

    for key in [
        "s21_db",
        "phase_rad",
        "beta_rad_per_m",
        "alpha_np_per_m",
        "group_delay_s",
        "group_velocity_m_per_s",
        "phase_velocity_m_per_s",
        "effective_index",
    ]:
        value = dispersion.get(key)
        if value is not None:
            arr = np.asarray(value)
            if arr.shape == (n,):
                columns[key] = arr.astype(float)

    s21 = dispersion.get("s21")
    if s21 is not None:
        s21_arr = np.asarray(s21, dtype=np.complex128)
        if s21_arr.shape == (n,):
            columns["s21_real"] = np.real(s21_arr)
            columns["s21_imag"] = np.imag(s21_arr)
            columns["s21_abs"] = np.abs(s21_arr)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns.keys()))
        writer.writeheader()
        for i in range(n):
            writer.writerow({key: float(value[i]) for key, value in columns.items()})

    return path


def write_stopbands_csv(path: Path, stopbands: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "start_index",
        "end_index",
        "start_frequency_hz",
        "end_frequency_hz",
        "center_frequency_hz",
        "width_hz",
        "min_s21_db",
        "min_frequency_hz",
        "cutoff_db",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for band in stopbands:
            writer.writerow({key: band.get(key, "") for key in fields})

    return path


def export_artifacts(
    *,
    config: ExtractDispersionConfig,
    linear_response: Mapping[str, Any],
    layout: Any | None,
    dispersion: Mapping[str, Any],
    stopbands: list[dict[str, Any]],
    stages: list[StageResult],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    arrays_path = output_dir / "dispersion_arrays.npz"

    npz_payload: dict[str, Any] = {
        "frequency_hz": np.asarray(dispersion["frequency_hz"]),
        "metadata_json": json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "layout": None if layout is None else layout_summary(layout),
                    "method": dispersion.get("method"),
                    "stopbands": stopbands,
                }
            )
        ),
    }

    for key in [
        "s21",
        "s21_db",
        "phase_rad",
        "beta_rad_per_m",
        "alpha_np_per_m",
        "group_delay_s",
        "group_velocity_m_per_s",
        "phase_velocity_m_per_s",
        "effective_index",
    ]:
        value = dispersion.get(key)
        if value is not None:
            npz_payload[key] = np.asarray(value)

    np.savez_compressed(arrays_path, **npz_payload)
    paths["arrays_npz"] = str(arrays_path)

    stopbands_json = output_dir / "stopbands.json"
    stopbands_json.write_text(json.dumps(jsonify(stopbands), indent=2), encoding="utf-8")
    paths["stopbands_json"] = str(stopbands_json)

    if config.export_csv:
        dispersion_csv = write_dispersion_csv(output_dir / "dispersion_table.csv", dispersion)
        stopbands_csv = write_stopbands_csv(output_dir / "stopbands.csv", stopbands)
        paths["dispersion_csv"] = str(dispersion_csv)
        paths["stopbands_csv"] = str(stopbands_csv)

    if config.save_checkpoint:
        try:
            from twpa.io.checkpoints import (
                CheckpointKind,
                CheckpointMetadata,
                save_checkpoint,
            )

            checkpoint_path = output_dir / "dispersion_checkpoint.npz"
            arrays = {
                key: value
                for key, value in npz_payload.items()
                if key != "metadata_json"
            }
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.LINEAR_SCAN,
                    name="dispersion_extraction",
                    source="scripts.extract_dispersion",
                    extra={
                        "config": config.to_dict(),
                        "layout": None if layout is None else layout_summary(layout),
                    },
                ),
                arrays=arrays,
                payload={
                    "dispersion": {
                        key: value
                        for key, value in dispersion.items()
                        if key not in {
                            "frequency_hz",
                            "omega_rad_s",
                            "s21",
                            "s21_db",
                            "phase_rad",
                            "beta_rad_per_m",
                            "alpha_np_per_m",
                            "group_delay_s",
                            "group_velocity_m_per_s",
                            "phase_velocity_m_per_s",
                            "effective_index",
                        }
                    },
                    "stopbands": stopbands,
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

            from twpa.plotting.diagnostics import (
                PlotConfig,
                plot_dispersion,
                plot_s21,
                plot_stopbands,
                save_figure,
            )

            scan_like = {
                "frequency_hz": dispersion["frequency_hz"],
                "s21_db": dispersion.get("s21_db"),
                "s": linear_response.get("s"),
            }

            if dispersion.get("s21_db") is not None:
                fig, _ = plot_s21(
                    scan_like,
                    config=PlotConfig(title="S21 used for dispersion extraction"),
                )
                p = save_figure(fig, output_dir / "dispersion_s21.png")
                paths["s21_png"] = str(p)

                import matplotlib.pyplot as plt

                plt.close(fig)

                fig, _ = plot_stopbands(
                    scan_like,
                    stopbands=stopbands,
                    config=PlotConfig(title="Detected stopbands"),
                )
                p = save_figure(fig, output_dir / "dispersion_stopbands.png")
                paths["stopbands_png"] = str(p)
                plt.close(fig)

            if dispersion.get("beta_rad_per_m") is not None:
                fig, _ = plot_dispersion(
                    dispersion,
                    config=PlotConfig(title="Extracted propagation constant"),
                    quantity="beta",
                )
                p = save_figure(fig, output_dir / "dispersion_beta.png")
                paths["beta_png"] = str(p)

                import matplotlib.pyplot as plt

                plt.close(fig)

            if dispersion.get("alpha_np_per_m") is not None:
                fig, _ = plot_dispersion(
                    dispersion,
                    config=PlotConfig(title="Extracted attenuation"),
                    quantity="alpha",
                )
                p = save_figure(fig, output_dir / "dispersion_alpha.png")
                paths["alpha_png"] = str(p)

                import matplotlib.pyplot as plt

                plt.close(fig)

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


def result_markdown(result: DispersionExtractionResult) -> str:
    cfg = result.config

    lines = [
        "# Dispersion extraction",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- method: `{cfg.method.value}`",
        f"- physical length: `{cfg.length_mm:.6g} mm`",
        f"- output directory: `{cfg.output_dir}`",
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
        "## Key numerical summaries",
        "",
    ]

    for stage in result.stages:
        if stage.name in {"load_or_build_input", "linear_scan", "dispersion", "stopbands"}:
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
        description="Extract TWPA dispersion from a linear response or layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-npz", type=str, default=None)
    parser.add_argument("--layout-csv", type=str, default=None)

    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)

    parser.add_argument("--f-min-ghz", type=float, default=1.0)
    parser.add_argument("--f-max-ghz", type=float, default=14.0)
    parser.add_argument("--n-frequency", type=int, default=501)

    parser.add_argument(
        "--method",
        choices=[m.value for m in DispersionMethod],
        default=DispersionMethod.AUTO.value,
    )
    parser.add_argument(
        "--phase-sign",
        choices=[s.value for s in PhaseSign],
        default=PhaseSign.AUTO.value,
    )
    parser.add_argument("--no-unwrap-phase", action="store_true")
    parser.add_argument(
        "--remove-linear-phase-offset",
        action="store_true",
        help="Anchor extracted beta to a positive physical baseline. Useful when port-reference phase introduces an offset.",
    )

    parser.add_argument(
        "--stopband-threshold-db",
        type=float,
        default=-20.0,
        help="Absolute S21 threshold for fallback stopband detection.",
    )
    parser.add_argument(
        "--stopband-margin-db",
        type=float,
        default=10.0,
        help="Fallback stopband threshold also requires S21 below median minus this margin.",
    )

    parser.add_argument("--layout-kind", type=str, default="uniform")
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument("--disorder-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dispersion"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="dispersion_extraction")

    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--quick", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> ExtractDispersionConfig:
    n_cells = int(args.n_cells)
    n_frequency = int(args.n_frequency)

    if args.quick:
        if n_cells == 20000:
            n_cells = 2000
        if n_frequency == 501:
            n_frequency = 201

    if args.input_npz is not None and args.layout_csv is not None:
        # This is allowed: NPZ gives response, CSV gives physical length/layout metadata.
        pass

    if args.input_npz is None and args.layout_csv is None:
        # Allowed: build fallback/synthetic layout.
        pass

    if n_cells <= 0:
        raise ValueError("--n-cells must be positive")
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
    if args.stopband_margin_db < 0.0:
        raise ValueError("--stopband-margin-db must be non-negative")

    return ExtractDispersionConfig(
        input_npz=args.input_npz,
        layout_csv=args.layout_csv,
        output_dir=str(args.output_dir),
        n_cells=n_cells,
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        f_min_ghz=float(args.f_min_ghz),
        f_max_ghz=float(args.f_max_ghz),
        n_frequency=n_frequency,
        method=DispersionMethod(args.method),
        phase_sign=PhaseSign(args.phase_sign),
        unwrap_phase=not bool(args.no_unwrap_phase),
        remove_linear_phase_offset=bool(args.remove_linear_phase_offset),
        stopband_threshold_db=float(args.stopband_threshold_db),
        stopband_margin_db=float(args.stopband_margin_db),
        layout_kind=str(args.layout_kind),
        include_resonators=bool(args.include_resonators),
        disorder_std=float(args.disorder_std),
        seed=int(args.seed),
        quick=bool(args.quick),
        make_plots=not bool(args.no_plots),
        save_checkpoint=not bool(args.no_checkpoint),
        export_csv=not bool(args.no_csv),
        name=str(args.name),
    )


def _stage_load_or_build_input(
    holders: dict[str, Any],
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    layout = None
    layout_info = None

    if config.layout_csv is not None or config.input_npz is None:
        layout, layout_info = build_or_load_layout(config)
        holders["layout"] = layout

    if config.input_npz is not None:
        response = load_npz_linear_response(config.input_npz)
        holders["linear_response"] = response
        if layout is not None:
            holders["layout"] = layout

        return {
            "status": RunStatus.PASS.value,
            "mode": "input_npz",
            "input_npz": config.input_npz,
            "linear_response": {
                "frequency_hz": array_summary(response["frequency_hz"]),
                "s": None if response.get("s") is None else array_summary(response["s"]),
                "s21": None if response.get("s21") is None else array_summary(response["s21"]),
                "s21_db": None if response.get("s21_db") is None else array_summary(response["s21_db"]),
                "metadata": response.get("metadata", {}),
            },
            "layout_info": layout_info,
            "messages": ("PASS: loaded linear response from NPZ.",),
        }

    holders["layout"] = layout
    return {
        "status": RunStatus.PASS.value,
        "mode": "layout_scan_required",
        "layout_info": layout_info,
        "messages": ("PASS: layout built/loaded; linear scan required.",),
    }


def _stage_linear_scan(
    holders: dict[str, Any],
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    if "linear_response" in holders:
        return {
            "status": RunStatus.PASS.value,
            "skipped": True,
            "reason": "input_npz already supplied linear response",
            "messages": ("PASS: skipped linear scan because input NPZ was provided.",),
        }

    layout = holders.get("layout")
    if layout is None:
        raise RuntimeError("No layout available for linear scan.")

    response = run_linear_scan_for_layout(layout, config)
    holders["linear_response"] = response

    s21_db = response.get("s21_db")
    finite = bool(s21_db is not None and jnp.all(jnp.isfinite(jnp.asarray(s21_db))))

    return {
        "status": RunStatus.PASS.value if finite else RunStatus.FAIL.value,
        "linear_response": {
            "frequency_hz": array_summary(response["frequency_hz"]),
            "s": None if response.get("s") is None else array_summary(response["s"]),
            "s21": None if response.get("s21") is None else array_summary(response["s21"]),
            "s21_db": None if response.get("s21_db") is None else array_summary(response["s21_db"]),
            "metadata": response.get("metadata", {}),
        },
        "finite_s21_db": finite,
        "messages": (
            "PASS: linear scan completed."
            if finite
            else "FAIL: linear scan completed but S21 dB is missing or non-finite."
        ),
    }


def _stage_dispersion(
    holders: dict[str, Any],
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    response = holders["linear_response"]
    layout = holders.get("layout")

    dispersion = extract_dispersion(
        linear_response=response,
        layout=layout,
        config=config,
    )
    holders["dispersion"] = dispersion

    has_beta = dispersion.get("beta_rad_per_m") is not None
    has_alpha = dispersion.get("alpha_np_per_m") is not None

    if not has_beta and not has_alpha:
        status = RunStatus.FAIL.value
        message = "FAIL: no beta or alpha extracted."
    elif not has_beta:
        status = RunStatus.PARTIAL.value
        message = "PARTIAL: alpha extracted but beta unavailable."
    else:
        status = RunStatus.PASS.value
        message = "PASS: dispersion extraction completed."

    return {
        "status": status,
        "method": dispersion.get("method"),
        "has_beta": has_beta,
        "has_alpha": has_alpha,
        "frequency_hz": array_summary(dispersion["frequency_hz"]),
        "beta_rad_per_m": None if dispersion.get("beta_rad_per_m") is None else array_summary(dispersion["beta_rad_per_m"]),
        "alpha_np_per_m": None if dispersion.get("alpha_np_per_m") is None else array_summary(dispersion["alpha_np_per_m"]),
        "group_delay_s": None if dispersion.get("group_delay_s") is None else array_summary(dispersion["group_delay_s"]),
        "effective_index": None if dispersion.get("effective_index") is None else array_summary(dispersion["effective_index"]),
        "messages": (message, *tuple(dispersion.get("messages", ()))),
    }


def _stage_stopbands(
    holders: dict[str, Any],
    config: ExtractDispersionConfig,
) -> dict[str, Any]:
    dispersion = holders["dispersion"]
    stopbands, summary = detect_stopbands(dispersion, config)
    holders["stopbands"] = stopbands
    return summary


def finalize_result(result: DispersionExtractionResult, output_dir: Path) -> int:
    summary_json = output_dir / "dispersion_summary.json"
    summary_md = output_dir / "dispersion_summary.md"

    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }

    result = DispersionExtractionResult(
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
    print(f"[dispersion] status: {result.status.value}")
    print(f"[dispersion] summary JSON: {summary_json}")
    print(f"[dispersion] summary MD:   {summary_md}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[dispersion] invalid arguments: {exc}", file=sys.stderr)
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
        "script": "scripts/extract_dispersion.py",
    }

    holders: dict[str, Any] = {}
    stages: list[StageResult] = []
    artifacts: dict[str, str] = {}

    print("[dispersion] loading/building input...")
    stage = run_stage("load_or_build_input", lambda: _stage_load_or_build_input(holders, config))
    stages.append(stage)
    print(f"[dispersion] load_or_build_input: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = DispersionExtractionResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[dispersion] running/loading linear scan...")
    stage = run_stage("linear_scan", lambda: _stage_linear_scan(holders, config))
    stages.append(stage)
    print(f"[dispersion] linear_scan: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = DispersionExtractionResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[dispersion] extracting dispersion...")
    stage = run_stage("dispersion", lambda: _stage_dispersion(holders, config))
    stages.append(stage)
    print(f"[dispersion] dispersion: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = DispersionExtractionResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[dispersion] detecting stopbands...")
    stage = run_stage("stopbands", lambda: _stage_stopbands(holders, config))
    stages.append(stage)
    print(f"[dispersion] stopbands: {stage.status.value}")

    print("[dispersion] exporting artifacts...")
    try:
        artifacts.update(
            export_artifacts(
                config=config,
                linear_response=holders["linear_response"],
                layout=holders.get("layout"),
                dispersion=holders["dispersion"],
                stopbands=holders.get("stopbands", []),
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

    result = DispersionExtractionResult(
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
