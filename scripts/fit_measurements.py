"""
Fit TWPA model parameters to measured gain / S-parameter data.

This script is the production calibration entry point for real measurements.
It accepts CSV or NPZ measurement files, normalizes them into a common dataset,
fits selected physical scale parameters, and exports fitted curves, residuals,
parameter tables, plots, checkpoints, and a markdown report.

It tries the package-native inference stack first when available:

    twpa.io.measurement
    twpa.inference.fitting
    twpa.inference.recovery

If those APIs are unavailable, it falls back to a transparent analytic surrogate
fit. The fallback is marked PARTIAL because it is not a full HB-backed fit.

Examples
--------
Fit a CSV containing frequency_ghz and gain_db:

    python scripts/fit_measurements.py ^
      --measurement-csv data/measured_gain.csv ^
      --output-dir outputs/fit_measurements

Fit an NPZ with frequency_hz and s21 or signal_gain_db:

    python scripts/fit_measurements.py ^
      --measurement-npz data/measured_sparams.npz ^
      --pump-frequency-ghz 10.0 ^
      --pump-current-ratio 0.08 ^
      --output-dir outputs/fit_measurements_npz

Force fallback least-squares mode:

    python scripts/fit_measurements.py ^
      --measurement-csv data/measured_gain.csv ^
      --fit-mode fallback_scipy ^
      --output-dir outputs/fit_measurements_fallback

Require package-native fitting:

    python scripts/fit_measurements.py ^
      --measurement-csv data/measured_gain.csv ^
      --fit-mode package ^
      --fail-on-package-fallback ^
      --output-dir outputs/fit_measurements_package
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
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


try:
    from scipy.optimize import least_squares

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    least_squares = None


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class FitMode(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    FALLBACK_SCIPY = "fallback_scipy"
    FALLBACK_RANDOM = "fallback_random"


@dataclass(frozen=True)
class MeasurementFitConfig:
    measurement_csv: str | None
    measurement_npz: str | None
    output_dir: str
    name: str
    quick: bool

    fit_mode: FitMode
    fail_on_package_fallback: bool

    pump_frequency_ghz: float
    pump_current_ratio: float
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float

    fit_l_scale: bool
    fit_c_scale: bool
    fit_i_star_scale: bool
    fit_loss_scale: bool
    fit_pump_scale: bool
    fit_gain_offset_db: bool
    fit_phase_offset_rad: bool

    lower_l_scale: float
    upper_l_scale: float
    lower_c_scale: float
    upper_c_scale: float
    lower_i_star_scale: float
    upper_i_star_scale: float
    lower_loss_scale: float
    upper_loss_scale: float
    lower_pump_scale: float
    upper_pump_scale: float
    lower_gain_offset_db: float
    upper_gain_offset_db: float
    lower_phase_offset_rad: float
    upper_phase_offset_rad: float

    initial_l_scale: float
    initial_c_scale: float
    initial_i_star_scale: float
    initial_loss_scale: float
    initial_pump_scale: float
    initial_gain_offset_db: float
    initial_phase_offset_rad: float

    gain_weight: float
    idler_weight: float
    phase_weight: float
    group_delay_weight: float
    gain_sigma_db: float
    idler_sigma_db: float
    phase_sigma_rad: float
    group_delay_sigma_ps: float

    fit_max_evals: int
    fit_random_samples: int
    fit_robust_loss: str
    fit_f_scale: float

    make_plots: bool
    save_checkpoint: bool
    export_csv: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fit_mode"] = self.fit_mode.value
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
class MeasurementFitResult:
    config: MeasurementFitConfig
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


def active_parameter_names(config: MeasurementFitConfig) -> tuple[str, ...]:
    names = []
    if config.fit_l_scale:
        names.append("l_scale")
    if config.fit_c_scale:
        names.append("c_scale")
    if config.fit_i_star_scale:
        names.append("i_star_scale")
    if config.fit_loss_scale:
        names.append("loss_scale")
    if config.fit_pump_scale:
        names.append("pump_scale")
    if config.fit_gain_offset_db:
        names.append("gain_offset_db")
    if config.fit_phase_offset_rad:
        names.append("phase_offset_rad")
    return tuple(names)


def initial_parameters(config: MeasurementFitConfig) -> dict[str, float]:
    return {
        "l_scale": float(config.initial_l_scale),
        "c_scale": float(config.initial_c_scale),
        "i_star_scale": float(config.initial_i_star_scale),
        "loss_scale": float(config.initial_loss_scale),
        "pump_scale": float(config.initial_pump_scale),
        "gain_offset_db": float(config.initial_gain_offset_db),
        "phase_offset_rad": float(config.initial_phase_offset_rad),
    }


def parameter_bounds(config: MeasurementFitConfig) -> dict[str, tuple[float, float]]:
    return {
        "l_scale": (float(config.lower_l_scale), float(config.upper_l_scale)),
        "c_scale": (float(config.lower_c_scale), float(config.upper_c_scale)),
        "i_star_scale": (float(config.lower_i_star_scale), float(config.upper_i_star_scale)),
        "loss_scale": (float(config.lower_loss_scale), float(config.upper_loss_scale)),
        "pump_scale": (float(config.lower_pump_scale), float(config.upper_pump_scale)),
        "gain_offset_db": (float(config.lower_gain_offset_db), float(config.upper_gain_offset_db)),
        "phase_offset_rad": (float(config.lower_phase_offset_rad), float(config.upper_phase_offset_rad)),
    }


def encode_params(params: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(params[name]) for name in names], dtype=float)


def decode_params(
    vector: Sequence[float],
    names: Sequence[str],
    base: Mapping[str, float],
) -> dict[str, float]:
    out = dict(base)
    for name, value in zip(names, vector):
        out[name] = float(value)
    return out


def bounds_arrays(
    bounds: Mapping[str, tuple[float, float]],
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([bounds[name][0] for name in names], dtype=float)
    upper = np.asarray([bounds[name][1] for name in names], dtype=float)
    return lower, upper


def read_csv_columns(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns: dict[str, list[float]] = {}

        for row in reader:
            for key, value in row.items():
                if key is None:
                    continue
                key_norm = key.strip()
                if key_norm == "":
                    continue
                columns.setdefault(key_norm, [])
                try:
                    columns[key_norm].append(float(str(value).strip()))
                except Exception:
                    columns[key_norm].append(np.nan)

    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def pick_column(
    columns: Mapping[str, np.ndarray],
    candidates: Sequence[str],
) -> np.ndarray | None:
    lower_map = {key.lower(): key for key in columns}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            return np.asarray(columns[key])
    return None


def normalize_measurement_columns(
    *,
    columns: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    frequency_hz = pick_column(
        columns,
        [
            "frequency_hz",
            "freq_hz",
            "f_hz",
            "frequency",
            "freq",
        ],
    )
    if frequency_hz is None:
        frequency_ghz = pick_column(
            columns,
            [
                "frequency_ghz",
                "freq_ghz",
                "f_ghz",
            ],
        )
        if frequency_ghz is not None:
            frequency_hz = np.asarray(frequency_ghz, dtype=float) * 1e9

    if frequency_hz is None:
        frequency_mhz = pick_column(
            columns,
            [
                "frequency_mhz",
                "freq_mhz",
                "f_mhz",
            ],
        )
        if frequency_mhz is not None:
            frequency_hz = np.asarray(frequency_mhz, dtype=float) * 1e6

    if frequency_hz is None:
        raise ValueError("Measurement data must include frequency_hz or frequency_ghz.")

    signal_gain_db = pick_column(
        columns,
        [
            "signal_gain_db",
            "gain_db",
            "s21_db",
            "S21_db",
            "gain",
        ],
    )

    idler_conversion_db = pick_column(
        columns,
        [
            "idler_conversion_db",
            "conversion_db",
            "idler_gain_db",
        ],
    )

    phase_rad = pick_column(
        columns,
        [
            "s21_phase_rad",
            "phase_rad",
            "s21_phase",
        ],
    )

    group_delay_ps = pick_column(
        columns,
        [
            "group_delay_ps",
            "gd_ps",
        ],
    )

    s21_real = pick_column(columns, ["s21_real", "S21_real", "real_s21"])
    s21_imag = pick_column(columns, ["s21_imag", "S21_imag", "imag_s21"])
    s21_abs = pick_column(columns, ["s21_abs", "S21_abs", "abs_s21", "mag_s21"])

    s21 = None
    if s21_real is not None and s21_imag is not None:
        s21 = np.asarray(s21_real, dtype=float) + 1j * np.asarray(s21_imag, dtype=float)
    elif s21_abs is not None and phase_rad is not None:
        s21 = np.asarray(s21_abs, dtype=float) * np.exp(1j * np.asarray(phase_rad, dtype=float))

    if signal_gain_db is None and s21 is not None:
        signal_gain_db = 20.0 * np.log10(np.maximum(np.abs(s21), 1e-300))

    if signal_gain_db is None:
        raise ValueError("Measurement data must include gain_db/s21_db or complex S21.")

    order = np.argsort(np.asarray(frequency_hz, dtype=float))

    dataset = {
        "source": source,
        "frequency_hz": jnp.asarray(np.asarray(frequency_hz, dtype=float)[order], dtype=jnp.float64),
        "signal_gain_db": jnp.asarray(np.asarray(signal_gain_db, dtype=float)[order], dtype=jnp.float64),
        "idler_conversion_db": None
        if idler_conversion_db is None
        else jnp.asarray(np.asarray(idler_conversion_db, dtype=float)[order], dtype=jnp.float64),
        "s21_phase_rad": None
        if phase_rad is None
        else jnp.asarray(np.asarray(phase_rad, dtype=float)[order], dtype=jnp.float64),
        "group_delay_ps": None
        if group_delay_ps is None
        else jnp.asarray(np.asarray(group_delay_ps, dtype=float)[order], dtype=jnp.float64),
        "s21": None if s21 is None else jnp.asarray(np.asarray(s21)[order], dtype=jnp.complex128),
        "raw_column_names": list(columns.keys()),
    }

    finite_mask = np.isfinite(np.asarray(dataset["frequency_hz"])) & np.isfinite(
        np.asarray(dataset["signal_gain_db"])
    )

    if not np.any(finite_mask):
        raise ValueError("No finite frequency/gain samples found.")

    # Apply the finite mask consistently.
    for key in ["frequency_hz", "signal_gain_db", "idler_conversion_db", "s21_phase_rad", "group_delay_ps", "s21"]:
        value = dataset.get(key)
        if value is not None:
            dataset[key] = jnp.asarray(value)[finite_mask]

    return dataset


def load_npz_measurement(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    npz = np.load(p, allow_pickle=True)
    columns: dict[str, Any] = {key: npz[key] for key in npz.files if key != "metadata_json"}

    dataset = normalize_measurement_columns(columns=columns, source=str(p))

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
            metadata = {"metadata_parse_error": f"{type(exc).__name__}: {exc}"}

    dataset["metadata"] = metadata
    return dataset


def load_csv_measurement(path: str | Path) -> dict[str, Any]:
    columns = read_csv_columns(path)
    dataset = normalize_measurement_columns(columns=columns, source=str(path))
    dataset["metadata"] = {"source_format": "csv"}
    return dataset


def try_package_measurement_load(config: MeasurementFitConfig) -> dict[str, Any]:
    import twpa.io.measurement as measurement_module

    candidate_names = [
        "load_measurement_dataset",
        "load_measurements",
        "read_measurement_dataset",
        "read_measurements",
    ]

    input_path = config.measurement_csv or config.measurement_npz
    if input_path is None:
        raise ValueError("No measurement path supplied.")

    errors: list[str] = []

    for name in candidate_names:
        fn = getattr(measurement_module, name, None)
        if fn is None:
            continue

        kwargs = {
            "path": input_path,
            "filename": input_path,
            "measurement_path": input_path,
            "pump_frequency_hz": config.pump_frequency_ghz * 1e9,
            "pump_current_ratio": config.pump_current_ratio,
            "config": config,
        }

        try:
            result = call_with_supported_kwargs(fn, kwargs)
            dataset = normalize_package_dataset(result, fallback_source=input_path)
            dataset["package_loader"] = name
            dataset["package_raw"] = jsonify(result)
            return dataset
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No compatible package measurement loader succeeded. Errors:\n"
        + "\n".join(errors[-10:])
    )


def normalize_package_dataset(dataset: Any, *, fallback_source: str | None) -> dict[str, Any]:
    frequency_hz = get_attr_any(dataset, "frequency_hz", "freq_hz", "f_hz", default=None)
    if frequency_hz is None:
        frequency_ghz = get_attr_any(dataset, "frequency_ghz", "freq_ghz", default=None)
        if frequency_ghz is not None:
            frequency_hz = jnp.asarray(frequency_ghz, dtype=jnp.float64) * 1e9

    gain_db = get_attr_any(dataset, "signal_gain_db", "gain_db", "s21_db", default=None)
    idler_db = get_attr_any(dataset, "idler_conversion_db", "conversion_db", default=None)
    s21 = get_attr_any(dataset, "s21", "S21", default=None)
    phase = get_attr_any(dataset, "s21_phase_rad", "phase_rad", default=None)
    group_delay_ps = get_attr_any(dataset, "group_delay_ps", "gd_ps", default=None)

    if frequency_hz is None:
        raise ValueError("Package measurement dataset does not expose frequency_hz.")

    if gain_db is None and s21 is not None:
        gain_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(s21)), 1e-300))

    if gain_db is None:
        raise ValueError("Package measurement dataset does not expose gain or S21.")

    if phase is None and s21 is not None:
        phase = jnp.unwrap(jnp.angle(jnp.asarray(s21)))

    return {
        "source": fallback_source or "package_dataset",
        "frequency_hz": jnp.asarray(frequency_hz, dtype=jnp.float64),
        "signal_gain_db": jnp.asarray(gain_db, dtype=jnp.float64),
        "idler_conversion_db": None if idler_db is None else jnp.asarray(idler_db, dtype=jnp.float64),
        "s21_phase_rad": None if phase is None else jnp.asarray(phase, dtype=jnp.float64),
        "group_delay_ps": None if group_delay_ps is None else jnp.asarray(group_delay_ps, dtype=jnp.float64),
        "s21": None if s21 is None else jnp.asarray(s21, dtype=jnp.complex128),
        "metadata": get_attr_any(dataset, "metadata", default={}),
    }


def load_measurement_dataset(config: MeasurementFitConfig) -> dict[str, Any]:
    if config.fit_mode in {FitMode.AUTO, FitMode.PACKAGE}:
        try:
            dataset = try_package_measurement_load(config)
            dataset["load_mode"] = "package"
            dataset["messages"] = (
                f"PASS: package measurement loader `{dataset.get('package_loader')}` completed.",
            )
            return dataset
        except Exception as exc:
            if config.fit_mode == FitMode.PACKAGE and config.fail_on_package_fallback:
                raise
            package_error = f"{type(exc).__name__}: {exc}"
    else:
        package_error = "package measurement loader disabled"

    if config.measurement_csv is not None:
        dataset = load_csv_measurement(config.measurement_csv)
    elif config.measurement_npz is not None:
        dataset = load_npz_measurement(config.measurement_npz)
    else:
        raise ValueError("Provide --measurement-csv or --measurement-npz.")

    dataset["load_mode"] = "fallback_local_parser"
    dataset["package_error"] = package_error
    dataset["messages"] = (
        "PARTIAL: used local fallback measurement parser.",
        package_error,
    )
    return dataset


def fallback_surrogate_model(
    params: Mapping[str, float],
    *,
    frequency_hz: Any,
    config: MeasurementFitConfig,
) -> dict[str, jax.Array]:
    f = np.asarray(frequency_hz, dtype=float)

    l_scale = float(params["l_scale"])
    c_scale = float(params["c_scale"])
    i_star_scale = float(params["i_star_scale"])
    loss_scale = float(params["loss_scale"])
    pump_scale = float(params["pump_scale"])
    gain_offset_db = float(params["gain_offset_db"])
    phase_offset_rad = float(params["phase_offset_rad"])

    vp = config.phase_velocity_m_per_s / math.sqrt(max(l_scale * c_scale, 1e-300))
    length_m = config.length_mm * 1e-3

    pump_frequency_hz = config.pump_frequency_ghz * 1e9
    center_hz = 0.5 * pump_frequency_hz * (l_scale / c_scale) ** 0.12

    bandwidth_hz = max(
        0.10 * pump_frequency_hz,
        0.18 * pump_frequency_hz / math.sqrt(max(l_scale * c_scale, 1e-300)),
    )
    detuning = (f - center_hz) / bandwidth_hz

    pump_strength = config.pump_current_ratio * pump_scale / max(i_star_scale, 1e-300)
    nonlinear_gain_np = 5.5 * pump_strength**2 * length_m / 0.1

    phase_mismatch = 1.8 * detuning + 0.35 * (l_scale - c_scale)
    phase_matching = np.sinc(phase_mismatch / np.pi) ** 2

    gain_power = 1.0 + np.sinh(nonlinear_gain_np * np.sqrt(np.maximum(phase_matching, 0.0))) ** 2

    attenuation_np = 0.020 * loss_scale * (length_m / 0.1) * (f / max(center_hz, 1.0)) ** 0.7
    ripple_db = 0.10 * np.sin(2.0 * np.pi * (f - f[0]) / max(f[-1] - f[0], 1.0) * 3.0)

    signal_gain_db = (
        10.0 * np.log10(np.maximum(gain_power, 1e-300))
        - 8.686 * attenuation_np
        + ripple_db
        + gain_offset_db
    )

    beta = 2.0 * np.pi * f / vp
    phase_rad = -beta * length_m + phase_offset_rad
    group_delay_s = np.gradient(-phase_rad, 2.0 * np.pi * f)
    group_delay_ps = group_delay_s * 1e12

    s21_mag = 10.0 ** (signal_gain_db / 20.0)
    s21 = s21_mag * np.exp(1j * phase_rad)

    idler_conversion_db = signal_gain_db - 3.0 - 10.0 * np.log10(1.0 + detuning**2)

    return {
        "frequency_hz": jnp.asarray(f, dtype=jnp.float64),
        "signal_gain_db": jnp.asarray(signal_gain_db, dtype=jnp.float64),
        "idler_conversion_db": jnp.asarray(idler_conversion_db, dtype=jnp.float64),
        "s21_phase_rad": jnp.asarray(phase_rad, dtype=jnp.float64),
        "group_delay_ps": jnp.asarray(group_delay_ps, dtype=jnp.float64),
        "s21": jnp.asarray(s21, dtype=jnp.complex128),
        "center_frequency_hz": jnp.asarray(center_hz, dtype=jnp.float64),
        "bandwidth_hz": jnp.asarray(bandwidth_hz, dtype=jnp.float64),
    }


def residual_vector(
    vector: np.ndarray,
    *,
    names: Sequence[str],
    base_params: Mapping[str, float],
    dataset: Mapping[str, Any],
    config: MeasurementFitConfig,
) -> np.ndarray:
    params = decode_params(vector, names, base=base_params)
    pred = fallback_surrogate_model(
        params,
        frequency_hz=dataset["frequency_hz"],
        config=config,
    )

    residuals: list[np.ndarray] = []

    gain_meas = np.asarray(dataset["signal_gain_db"], dtype=float)
    gain_pred = np.asarray(pred["signal_gain_db"], dtype=float)
    gain_mask = np.isfinite(gain_meas) & np.isfinite(gain_pred)
    if np.any(gain_mask) and config.gain_weight != 0.0:
        residuals.append(
            config.gain_weight
            * (gain_pred[gain_mask] - gain_meas[gain_mask])
            / max(config.gain_sigma_db, 1e-9)
        )

    idler_meas = dataset.get("idler_conversion_db")
    if idler_meas is not None and config.idler_weight != 0.0:
        idler_meas_arr = np.asarray(idler_meas, dtype=float)
        idler_pred = np.asarray(pred["idler_conversion_db"], dtype=float)
        mask = np.isfinite(idler_meas_arr) & np.isfinite(idler_pred)
        if np.any(mask):
            residuals.append(
                config.idler_weight
                * (idler_pred[mask] - idler_meas_arr[mask])
                / max(config.idler_sigma_db, 1e-9)
            )

    phase_meas = dataset.get("s21_phase_rad")
    if phase_meas is not None and config.phase_weight != 0.0:
        phase_meas_arr = np.unwrap(np.asarray(phase_meas, dtype=float))
        phase_pred = np.unwrap(np.asarray(pred["s21_phase_rad"], dtype=float))
        mask = np.isfinite(phase_meas_arr) & np.isfinite(phase_pred)
        if np.any(mask):
            phase_res = phase_pred[mask] - phase_meas_arr[mask]
            phase_res = phase_res - np.nanmean(phase_res)
            residuals.append(
                config.phase_weight
                * phase_res
                / max(config.phase_sigma_rad, 1e-12)
            )

    gd_meas = dataset.get("group_delay_ps")
    if gd_meas is not None and config.group_delay_weight != 0.0:
        gd_meas_arr = np.asarray(gd_meas, dtype=float)
        gd_pred = np.asarray(pred["group_delay_ps"], dtype=float)
        mask = np.isfinite(gd_meas_arr) & np.isfinite(gd_pred)
        if np.any(mask):
            residuals.append(
                config.group_delay_weight
                * (gd_pred[mask] - gd_meas_arr[mask])
                / max(config.group_delay_sigma_ps, 1e-9)
            )

    if not residuals:
        raise ValueError("No usable residual terms were constructed from the measurement dataset.")

    return np.concatenate([r.ravel() for r in residuals])


def try_package_fit(
    *,
    dataset: Mapping[str, Any],
    config: MeasurementFitConfig,
) -> dict[str, Any]:
    import twpa.inference.fitting as fitting_module

    candidate_names = [
        "fit_measurement_dataset",
        "fit_measurements",
        "fit_twpa_parameters",
        "run_parameter_fit",
        "fit_parameters",
    ]

    errors: list[str] = []

    for name in candidate_names:
        fn = getattr(fitting_module, name, None)
        if fn is None:
            continue

        kwargs = {
            "dataset": dataset,
            "measurement": dataset,
            "measurements": dataset,
            "initial_params": initial_parameters(config),
            "bounds": parameter_bounds(config),
            "active_names": active_parameter_names(config),
            "pump_frequency_hz": config.pump_frequency_ghz * 1e9,
            "pump_current_ratio": config.pump_current_ratio,
            "length_m": config.length_mm * 1e-3,
            "max_evals": config.fit_max_evals,
            "max_nfev": config.fit_max_evals,
            "config": config,
        }

        try:
            result = call_with_supported_kwargs(fn, kwargs)

            fit_params = get_attr_any(
                result,
                "fit_params",
                "params",
                "parameters",
                default=None,
            )
            if fit_params is None and isinstance(result, Mapping):
                fit_params = (
                    result.get("fit_params")
                    or result.get("params")
                    or result.get("parameters")
                )

            if fit_params is None:
                raise RuntimeError("Package fit result does not expose fitted parameters.")

            prediction = get_attr_any(
                result,
                "prediction",
                "fit_curve",
                "model",
                default=None,
            )
            if prediction is None and isinstance(result, Mapping):
                prediction = result.get("prediction") or result.get("fit_curve") or result.get("model")

            fit_params = {k: float(v) for k, v in dict(fit_params).items()}
            if prediction is None:
                prediction = fallback_surrogate_model(
                    {**initial_parameters(config), **fit_params},
                    frequency_hz=dataset["frequency_hz"],
                    config=config,
                )
            else:
                prediction = normalize_prediction(prediction, dataset=dataset, config=config)

            return {
                "status": RunStatus.PASS.value,
                "mode": f"package_{name}",
                "success": bool(get_attr_any(result, "success", "converged", default=True)),
                "fit_params": fit_params,
                "prediction": prediction,
                "loss": get_attr_any(result, "loss", "cost", default=None),
                "residual_norm": get_attr_any(result, "residual_norm", default=None),
                "n_evaluations": get_attr_any(result, "n_evaluations", "nfev", default=None),
                "package_result": jsonify(result),
                "messages": (f"PASS: package-native fitter `{name}` completed.",),
            }
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No compatible package fitter succeeded. Errors:\n"
        + "\n".join(errors[-10:])
    )


def normalize_prediction(
    prediction: Any,
    *,
    dataset: Mapping[str, Any],
    config: MeasurementFitConfig,
) -> dict[str, Any]:
    frequency_hz = get_attr_any(
        prediction,
        "frequency_hz",
        "signal_frequency_hz",
        default=dataset["frequency_hz"],
    )
    signal_gain_db = get_attr_any(
        prediction,
        "signal_gain_db",
        "gain_db",
        "s21_db",
        default=None,
    )
    idler_conversion_db = get_attr_any(
        prediction,
        "idler_conversion_db",
        "conversion_db",
        default=None,
    )
    phase = get_attr_any(
        prediction,
        "s21_phase_rad",
        "phase_rad",
        default=None,
    )
    group_delay_ps = get_attr_any(
        prediction,
        "group_delay_ps",
        "gd_ps",
        default=None,
    )
    s21 = get_attr_any(prediction, "s21", "S21", default=None)

    if signal_gain_db is None and s21 is not None:
        signal_gain_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(s21)), 1e-300))
    if signal_gain_db is None:
        raise ValueError("Prediction does not expose gain or S21.")
    if phase is None and s21 is not None:
        phase = jnp.unwrap(jnp.angle(jnp.asarray(s21)))
    if idler_conversion_db is None:
        idler_conversion_db = jnp.asarray(signal_gain_db) - 3.0

    return {
        "frequency_hz": jnp.asarray(frequency_hz, dtype=jnp.float64),
        "signal_gain_db": jnp.asarray(signal_gain_db, dtype=jnp.float64),
        "idler_conversion_db": jnp.asarray(idler_conversion_db, dtype=jnp.float64),
        "s21_phase_rad": None if phase is None else jnp.asarray(phase, dtype=jnp.float64),
        "group_delay_ps": None if group_delay_ps is None else jnp.asarray(group_delay_ps, dtype=jnp.float64),
        "s21": None if s21 is None else jnp.asarray(s21, dtype=jnp.complex128),
        "raw": jsonify(prediction),
    }


def fallback_random_fit(
    *,
    dataset: Mapping[str, Any],
    config: MeasurementFitConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(10101)
    names = active_parameter_names(config)
    base = initial_parameters(config)
    bounds = parameter_bounds(config)
    lower, upper = bounds_arrays(bounds, names)

    best_x = encode_params(base, names)
    best_res = residual_vector(
        best_x,
        names=names,
        base_params=base,
        dataset=dataset,
        config=config,
    )
    best_loss = float(np.mean(best_res**2))

    records = [{"evaluation_index": 0, "loss": best_loss, "x": best_x.tolist()}]

    for i in range(int(config.fit_random_samples)):
        x = rng.uniform(lower, upper)
        res = residual_vector(
            x,
            names=names,
            base_params=base,
            dataset=dataset,
            config=config,
        )
        loss = float(np.mean(res**2))
        records.append({"evaluation_index": i + 1, "loss": loss, "x": x.tolist()})

        if loss < best_loss:
            best_x = x
            best_res = res
            best_loss = loss

    fit_params = decode_params(best_x, names, base=base)
    prediction = fallback_surrogate_model(
        fit_params,
        frequency_hz=dataset["frequency_hz"],
        config=config,
    )

    return {
        "status": RunStatus.PARTIAL.value,
        "mode": "fallback_random",
        "success": True,
        "fit_params": fit_params,
        "prediction": prediction,
        "x": best_x,
        "loss": best_loss,
        "residual_norm": float(np.linalg.norm(best_res)),
        "n_evaluations": int(config.fit_random_samples),
        "records": records,
        "messages": ("PARTIAL: used fallback random-search fit.",),
    }


def fallback_scipy_fit(
    *,
    dataset: Mapping[str, Any],
    config: MeasurementFitConfig,
) -> dict[str, Any]:
    if not SCIPY_AVAILABLE:
        return fallback_random_fit(dataset=dataset, config=config)

    names = active_parameter_names(config)
    base = initial_parameters(config)
    bounds = parameter_bounds(config)
    lower, upper = bounds_arrays(bounds, names)
    x0 = np.clip(encode_params(base, names), lower, upper)

    records: list[dict[str, Any]] = []

    def fun(x: np.ndarray) -> np.ndarray:
        res = residual_vector(
            x,
            names=names,
            base_params=base,
            dataset=dataset,
            config=config,
        )
        records.append(
            {
                "evaluation_index": len(records) + 1,
                "loss": float(np.mean(res**2)),
                "x": np.asarray(x, dtype=float).tolist(),
            }
        )
        return res

    result = least_squares(
        fun,
        x0,
        bounds=(lower, upper),
        max_nfev=config.fit_max_evals,
        loss=config.fit_robust_loss,
        f_scale=config.fit_f_scale,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        verbose=0,
    )

    final_res = residual_vector(
        np.asarray(result.x, dtype=float),
        names=names,
        base_params=base,
        dataset=dataset,
        config=config,
    )
    fit_params = decode_params(result.x, names, base=base)
    prediction = fallback_surrogate_model(
        fit_params,
        frequency_hz=dataset["frequency_hz"],
        config=config,
    )

    return {
        "status": RunStatus.PARTIAL.value,
        "mode": "fallback_scipy",
        "success": bool(result.success),
        "fit_params": fit_params,
        "prediction": prediction,
        "x": np.asarray(result.x, dtype=float),
        "loss": float(np.mean(final_res**2)),
        "cost": float(result.cost),
        "residual_norm": float(np.linalg.norm(final_res)),
        "n_evaluations": int(result.nfev),
        "message": str(result.message),
        "records": records,
        "messages": ("PARTIAL: used fallback SciPy least-squares fit.",),
    }


def fit_measurement_dataset(
    *,
    dataset: Mapping[str, Any],
    config: MeasurementFitConfig,
) -> dict[str, Any]:
    if not active_parameter_names(config):
        raise ValueError("At least one fit parameter must be enabled.")

    if config.fit_mode in {FitMode.AUTO, FitMode.PACKAGE}:
        try:
            return try_package_fit(dataset=dataset, config=config)
        except Exception as exc:
            if config.fit_mode == FitMode.PACKAGE or config.fail_on_package_fallback:
                raise
            package_error = f"{type(exc).__name__}: {exc}"
    else:
        package_error = "package fitter disabled"

    if config.fit_mode == FitMode.FALLBACK_RANDOM:
        result = fallback_random_fit(dataset=dataset, config=config)
    else:
        result = fallback_scipy_fit(dataset=dataset, config=config)

    result["package_error"] = package_error
    result["messages"] = (
        *tuple(result.get("messages", ())),
        package_error,
    )
    return result


def compute_fit_metrics(
    *,
    dataset: Mapping[str, Any],
    fit_result: Mapping[str, Any],
    config: MeasurementFitConfig,
) -> dict[str, Any]:
    prediction = fit_result["prediction"]

    gain_meas = np.asarray(dataset["signal_gain_db"], dtype=float)
    gain_pred = np.asarray(prediction["signal_gain_db"], dtype=float)
    gain_res = gain_pred - gain_meas
    gain_mask = np.isfinite(gain_res)

    metrics: dict[str, Any] = {
        "mode": fit_result.get("mode"),
        "success": fit_result.get("success"),
        "loss": fit_result.get("loss"),
        "residual_norm": fit_result.get("residual_norm"),
        "n_evaluations": fit_result.get("n_evaluations"),
        "gain_rms_error_db": float(np.sqrt(np.nanmean(gain_res[gain_mask] ** 2))) if np.any(gain_mask) else None,
        "gain_mean_error_db": float(np.nanmean(gain_res[gain_mask])) if np.any(gain_mask) else None,
        "gain_max_abs_error_db": float(np.nanmax(np.abs(gain_res[gain_mask]))) if np.any(gain_mask) else None,
    }

    idler_meas = dataset.get("idler_conversion_db")
    idler_pred = prediction.get("idler_conversion_db")
    if idler_meas is not None and idler_pred is not None:
        res = np.asarray(idler_pred, dtype=float) - np.asarray(idler_meas, dtype=float)
        mask = np.isfinite(res)
        metrics.update(
            {
                "idler_rms_error_db": float(np.sqrt(np.nanmean(res[mask] ** 2))) if np.any(mask) else None,
                "idler_max_abs_error_db": float(np.nanmax(np.abs(res[mask]))) if np.any(mask) else None,
            }
        )

    phase_meas = dataset.get("s21_phase_rad")
    phase_pred = prediction.get("s21_phase_rad")
    if phase_meas is not None and phase_pred is not None:
        res = np.unwrap(np.asarray(phase_pred, dtype=float)) - np.unwrap(np.asarray(phase_meas, dtype=float))
        res = res - np.nanmean(res)
        mask = np.isfinite(res)
        metrics.update(
            {
                "phase_rms_error_rad": float(np.sqrt(np.nanmean(res[mask] ** 2))) if np.any(mask) else None,
                "phase_max_abs_error_rad": float(np.nanmax(np.abs(res[mask]))) if np.any(mask) else None,
            }
        )

    gd_meas = dataset.get("group_delay_ps")
    gd_pred = prediction.get("group_delay_ps")
    if gd_meas is not None and gd_pred is not None:
        res = np.asarray(gd_pred, dtype=float) - np.asarray(gd_meas, dtype=float)
        mask = np.isfinite(res)
        metrics.update(
            {
                "group_delay_rms_error_ps": float(np.sqrt(np.nanmean(res[mask] ** 2))) if np.any(mask) else None,
                "group_delay_max_abs_error_ps": float(np.nanmax(np.abs(res[mask]))) if np.any(mask) else None,
            }
        )

    return metrics


def write_parameters_csv(path: Path, fit_result: Mapping[str, Any], config: MeasurementFitConfig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fit_params = fit_result["fit_params"]
    initial = initial_parameters(config)
    bounds = parameter_bounds(config)
    active = set(active_parameter_names(config))

    fields = [
        "parameter",
        "fit",
        "initial",
        "lower",
        "upper",
        "active",
        "delta_from_initial",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for name in [
            "l_scale",
            "c_scale",
            "i_star_scale",
            "loss_scale",
            "pump_scale",
            "gain_offset_db",
            "phase_offset_rad",
        ]:
            fit_value = fit_params.get(name, initial[name])
            lo, hi = bounds[name]
            writer.writerow(
                {
                    "parameter": name,
                    "fit": fit_value,
                    "initial": initial[name],
                    "lower": lo,
                    "upper": hi,
                    "active": name in active,
                    "delta_from_initial": fit_value - initial[name],
                }
            )

    return path


def write_fit_curve_csv(
    path: Path,
    *,
    dataset: Mapping[str, Any],
    fit_result: Mapping[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    pred = fit_result["prediction"]

    f = np.asarray(dataset["frequency_hz"], dtype=float)
    meas_gain = np.asarray(dataset["signal_gain_db"], dtype=float)
    pred_gain = np.asarray(pred["signal_gain_db"], dtype=float)

    fields = [
        "frequency_hz",
        "frequency_ghz",
        "measured_gain_db",
        "fitted_gain_db",
        "gain_residual_db",
    ]

    idler_meas = dataset.get("idler_conversion_db")
    idler_pred = pred.get("idler_conversion_db")
    if idler_meas is not None and idler_pred is not None:
        fields.extend(
            [
                "measured_idler_conversion_db",
                "fitted_idler_conversion_db",
                "idler_residual_db",
            ]
        )

    phase_meas = dataset.get("s21_phase_rad")
    phase_pred = pred.get("s21_phase_rad")
    if phase_meas is not None and phase_pred is not None:
        fields.extend(
            [
                "measured_phase_rad",
                "fitted_phase_rad",
                "phase_residual_rad",
            ]
        )

    gd_meas = dataset.get("group_delay_ps")
    gd_pred = pred.get("group_delay_ps")
    if gd_meas is not None and gd_pred is not None:
        fields.extend(
            [
                "measured_group_delay_ps",
                "fitted_group_delay_ps",
                "group_delay_residual_ps",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for i in range(f.shape[0]):
            row = {
                "frequency_hz": float(f[i]),
                "frequency_ghz": float(f[i] / 1e9),
                "measured_gain_db": float(meas_gain[i]),
                "fitted_gain_db": float(pred_gain[i]),
                "gain_residual_db": float(pred_gain[i] - meas_gain[i]),
            }

            if idler_meas is not None and idler_pred is not None:
                row.update(
                    {
                        "measured_idler_conversion_db": float(np.asarray(idler_meas)[i]),
                        "fitted_idler_conversion_db": float(np.asarray(idler_pred)[i]),
                        "idler_residual_db": float(np.asarray(idler_pred)[i] - np.asarray(idler_meas)[i]),
                    }
                )

            if phase_meas is not None and phase_pred is not None:
                row.update(
                    {
                        "measured_phase_rad": float(np.asarray(phase_meas)[i]),
                        "fitted_phase_rad": float(np.asarray(phase_pred)[i]),
                        "phase_residual_rad": float(np.asarray(phase_pred)[i] - np.asarray(phase_meas)[i]),
                    }
                )

            if gd_meas is not None and gd_pred is not None:
                row.update(
                    {
                        "measured_group_delay_ps": float(np.asarray(gd_meas)[i]),
                        "fitted_group_delay_ps": float(np.asarray(gd_pred)[i]),
                        "group_delay_residual_ps": float(np.asarray(gd_pred)[i] - np.asarray(gd_meas)[i]),
                    }
                )

            writer.writerow(row)

    return path


def write_loss_history_csv(path: Path, fit_result: Mapping[str, Any]) -> Path | None:
    records = fit_result.get("records")
    if not records:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["evaluation_index", "loss", "x_json"]

    with path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            writer.writerow(
                {
                    "evaluation_index": rec.get("evaluation_index"),
                    "loss": rec.get("loss"),
                    "x_json": json.dumps(jsonify(rec.get("x"))),
                }
            )

    return path


def export_artifacts(
    *,
    config: MeasurementFitConfig,
    dataset: Mapping[str, Any],
    fit_result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stages: list[StageResult],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    arrays_npz = output_dir / "fit_measurements_arrays.npz"
    pred = fit_result["prediction"]

    payload: dict[str, Any] = {
        "frequency_hz": np.asarray(dataset["frequency_hz"]),
        "measured_signal_gain_db": np.asarray(dataset["signal_gain_db"]),
        "fitted_signal_gain_db": np.asarray(pred["signal_gain_db"]),
        "metadata_json": json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "dataset": {
                        "source": dataset.get("source"),
                        "load_mode": dataset.get("load_mode"),
                        "metadata": dataset.get("metadata"),
                    },
                    "fit_params": fit_result.get("fit_params"),
                    "metrics": metrics,
                    "fit_mode": fit_result.get("mode"),
                }
            )
        ),
    }

    optional_pairs = {
        "measured_idler_conversion_db": dataset.get("idler_conversion_db"),
        "fitted_idler_conversion_db": pred.get("idler_conversion_db"),
        "measured_s21_phase_rad": dataset.get("s21_phase_rad"),
        "fitted_s21_phase_rad": pred.get("s21_phase_rad"),
        "measured_group_delay_ps": dataset.get("group_delay_ps"),
        "fitted_group_delay_ps": pred.get("group_delay_ps"),
        "measured_s21": dataset.get("s21"),
        "fitted_s21": pred.get("s21"),
    }
    for key, value in optional_pairs.items():
        if value is not None:
            payload[key] = np.asarray(value)

    np.savez_compressed(arrays_npz, **payload)
    artifacts["arrays_npz"] = str(arrays_npz)

    metrics_json = output_dir / "fit_measurements_metrics.json"
    metrics_json.write_text(json.dumps(jsonify(metrics), indent=2), encoding="utf-8")
    artifacts["metrics_json"] = str(metrics_json)

    fit_params_json = output_dir / "fit_measurements_parameters.json"
    fit_params_json.write_text(
        json.dumps(jsonify(fit_result.get("fit_params", {})), indent=2),
        encoding="utf-8",
    )
    artifacts["parameters_json"] = str(fit_params_json)

    if config.export_csv:
        params_csv = write_parameters_csv(output_dir / "fit_measurements_parameters.csv", fit_result, config)
        curve_csv = write_fit_curve_csv(output_dir / "fit_measurements_curve.csv", dataset=dataset, fit_result=fit_result)
        artifacts["parameters_csv"] = str(params_csv)
        artifacts["fit_curve_csv"] = str(curve_csv)

        history_csv = write_loss_history_csv(output_dir / "fit_measurements_loss_history.csv", fit_result)
        if history_csv is not None:
            artifacts["loss_history_csv"] = str(history_csv)

    if config.save_checkpoint:
        try:
            from twpa.io.checkpoints import (
                CheckpointKind,
                CheckpointMetadata,
                save_checkpoint,
            )

            checkpoint_path = output_dir / "fit_measurements_checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.CALIBRATION,
                    name="fit_measurements",
                    source="scripts.fit_measurements",
                    extra={
                        "config": config.to_dict(),
                        "fit_params": fit_result.get("fit_params"),
                        "metrics": metrics,
                    },
                ),
                arrays={k: v for k, v in payload.items() if k != "metadata_json"},
                payload={
                    "fit_result": {
                        k: v
                        for k, v in fit_result.items()
                        if k not in {"prediction", "records"}
                    },
                    "metrics": metrics,
                },
            )
            artifacts["checkpoint_npz"] = str(checkpoint_path)

        except Exception as exc:
            error_path = output_dir / "checkpoint_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            artifacts["checkpoint_error_txt"] = str(error_path)
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

            f_ghz = np.asarray(dataset["frequency_hz"], dtype=float) / 1e9
            measured = np.asarray(dataset["signal_gain_db"], dtype=float)
            fitted = np.asarray(pred["signal_gain_db"], dtype=float)
            residual = fitted - measured

            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
            ax.plot(f_ghz, measured, marker="o", linestyle="", label="measured")
            ax.plot(f_ghz, fitted, label="fit")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Gain / S21 (dB)")
            ax.set_title("Measurement fit")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            p = output_dir / "fit_measurements_gain_fit.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            artifacts["gain_fit_png"] = str(p)

            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
            ax.axhline(0.0, linewidth=1.0)
            ax.plot(f_ghz, residual, marker="o")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Residual (dB)")
            ax.set_title("Gain residual")
            ax.grid(True)
            fig.tight_layout()
            p = output_dir / "fit_measurements_gain_residual.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            artifacts["gain_residual_png"] = str(p)

            records = fit_result.get("records")
            if records:
                losses = [
                    r.get("loss", np.nan)
                    for r in records
                    if isinstance(r, Mapping)
                ]
                if losses:
                    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
                    ax.semilogy(np.arange(1, len(losses) + 1), losses)
                    ax.set_xlabel("Evaluation")
                    ax.set_ylabel("Loss")
                    ax.set_title("Fit objective history")
                    ax.grid(True)
                    fig.tight_layout()
                    p = output_dir / "fit_measurements_loss_history.png"
                    fig.savefig(p, bbox_inches="tight")
                    plt.close(fig)
                    artifacts["loss_history_png"] = str(p)

            phase_meas = dataset.get("s21_phase_rad")
            phase_fit = pred.get("s21_phase_rad")
            if phase_meas is not None and phase_fit is not None:
                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
                ax.plot(f_ghz, np.unwrap(np.asarray(phase_meas, dtype=float)), marker="o", linestyle="", label="measured")
                ax.plot(f_ghz, np.unwrap(np.asarray(phase_fit, dtype=float)), label="fit")
                ax.set_xlabel("Frequency (GHz)")
                ax.set_ylabel("Unwrapped phase (rad)")
                ax.set_title("S21 phase fit")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                p = output_dir / "fit_measurements_phase_fit.png"
                fig.savefig(p, bbox_inches="tight")
                plt.close(fig)
                artifacts["phase_fit_png"] = str(p)

        except Exception as exc:
            error_path = output_dir / "plotting_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            artifacts["plotting_error_txt"] = str(error_path)
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

    return artifacts


def result_markdown(result: MeasurementFitResult) -> str:
    cfg = result.config
    metrics = result.metadata.get("fit_metrics", {})
    fit_params = result.metadata.get("fit_params", {})

    lines = [
        "# Measurement fit",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- measurement CSV: `{cfg.measurement_csv}`",
        f"- measurement NPZ: `{cfg.measurement_npz}`",
        f"- fit mode: `{cfg.fit_mode.value}`",
        f"- pump frequency: `{cfg.pump_frequency_ghz:.6g} GHz`",
        f"- pump current ratio: `{cfg.pump_current_ratio:.6g}`",
        f"- active parameters: `{list(active_parameter_names(cfg))}`",
        "",
        "## Fit metrics",
        "",
        f"- gain RMS error: `{metrics.get('gain_rms_error_db')}` dB",
        f"- gain max absolute error: `{metrics.get('gain_max_abs_error_db')}` dB",
        f"- phase RMS error: `{metrics.get('phase_rms_error_rad')}` rad",
        f"- group-delay RMS error: `{metrics.get('group_delay_rms_error_ps')}` ps",
        f"- loss: `{metrics.get('loss')}`",
        f"- residual norm: `{metrics.get('residual_norm')}`",
        "",
        "## Fitted parameters",
        "",
        "| parameter | value |",
        "|---|---:|",
    ]

    for key, value in fit_params.items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Stages",
        "",
        "| stage | status | elapsed s | messages |",
        "|---|---|---:|---|",
    ]

    for stage in result.stages:
        msg = "<br>".join(stage.messages[:3])
        lines.append(
            f"| `{stage.name}` | `{stage.status.value}` | {stage.elapsed_s:.6g} | {msg} |"
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
        "## Detailed summaries",
        "",
    ]

    for stage in result.stages:
        if stage.name in {"load_measurement", "fit", "metrics"}:
            lines += [
                f"### {stage.name}",
                "",
                "```json",
                json.dumps(jsonify(stage.summary), indent=2)[:8000],
                "```",
                "",
            ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit TWPA parameters to measurement data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--measurement-csv", type=str, default=None)
    parser.add_argument("--measurement-npz", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fit_measurements"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="fit_measurements")
    parser.add_argument("--quick", action="store_true")

    parser.add_argument(
        "--fit-mode",
        choices=[m.value for m in FitMode],
        default=FitMode.AUTO.value,
    )
    parser.add_argument("--fail-on-package-fallback", action="store_true")

    parser.add_argument("--pump-frequency-ghz", type=float, default=10.0)
    parser.add_argument("--pump-current-ratio", type=float, default=0.08)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)

    parser.add_argument("--fit-l-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-c-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-i-star-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-loss-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-pump-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-gain-offset-db", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-phase-offset-rad", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lower-l-scale", type=float, default=0.85)
    parser.add_argument("--upper-l-scale", type=float, default=1.15)
    parser.add_argument("--lower-c-scale", type=float, default=0.85)
    parser.add_argument("--upper-c-scale", type=float, default=1.15)
    parser.add_argument("--lower-i-star-scale", type=float, default=0.60)
    parser.add_argument("--upper-i-star-scale", type=float, default=1.60)
    parser.add_argument("--lower-loss-scale", type=float, default=0.20)
    parser.add_argument("--upper-loss-scale", type=float, default=5.00)
    parser.add_argument("--lower-pump-scale", type=float, default=0.50)
    parser.add_argument("--upper-pump-scale", type=float, default=1.80)
    parser.add_argument("--lower-gain-offset-db", type=float, default=-20.0)
    parser.add_argument("--upper-gain-offset-db", type=float, default=20.0)
    parser.add_argument("--lower-phase-offset-rad", type=float, default=-12.566370614359172)
    parser.add_argument("--upper-phase-offset-rad", type=float, default=12.566370614359172)

    parser.add_argument("--initial-l-scale", type=float, default=1.0)
    parser.add_argument("--initial-c-scale", type=float, default=1.0)
    parser.add_argument("--initial-i-star-scale", type=float, default=1.0)
    parser.add_argument("--initial-loss-scale", type=float, default=1.0)
    parser.add_argument("--initial-pump-scale", type=float, default=1.0)
    parser.add_argument("--initial-gain-offset-db", type=float, default=0.0)
    parser.add_argument("--initial-phase-offset-rad", type=float, default=0.0)

    parser.add_argument("--gain-weight", type=float, default=1.0)
    parser.add_argument("--idler-weight", type=float, default=0.5)
    parser.add_argument("--phase-weight", type=float, default=0.2)
    parser.add_argument("--group-delay-weight", type=float, default=0.2)
    parser.add_argument("--gain-sigma-db", type=float, default=0.25)
    parser.add_argument("--idler-sigma-db", type=float, default=0.50)
    parser.add_argument("--phase-sigma-rad", type=float, default=0.05)
    parser.add_argument("--group-delay-sigma-ps", type=float, default=5.0)

    parser.add_argument("--fit-max-evals", type=int, default=800)
    parser.add_argument("--fit-random-samples", type=int, default=3000)
    parser.add_argument(
        "--fit-robust-loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="soft_l1",
    )
    parser.add_argument("--fit-f-scale", type=float, default=1.0)

    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-csv", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> MeasurementFitConfig:
    if args.measurement_csv is None and args.measurement_npz is None:
        raise ValueError("Provide --measurement-csv or --measurement-npz.")
    if args.measurement_csv is not None and args.measurement_npz is not None:
        raise ValueError("Provide only one of --measurement-csv or --measurement-npz.")
    if args.measurement_csv is not None and not Path(args.measurement_csv).exists():
        raise FileNotFoundError(args.measurement_csv)
    if args.measurement_npz is not None and not Path(args.measurement_npz).exists():
        raise FileNotFoundError(args.measurement_npz)

    fit_max_evals = int(args.fit_max_evals)
    fit_random_samples = int(args.fit_random_samples)

    if args.quick:
        fit_max_evals = min(fit_max_evals, 200)
        fit_random_samples = min(fit_random_samples, 500)

    if args.pump_frequency_ghz <= 0.0:
        raise ValueError("--pump-frequency-ghz must be positive")
    if args.pump_current_ratio < 0.0:
        raise ValueError("--pump-current-ratio must be non-negative")
    if args.length_mm <= 0.0:
        raise ValueError("--length-mm must be positive")
    if args.z0_ohm <= 0.0:
        raise ValueError("--z0-ohm must be positive")
    if args.phase_velocity_m_per_s <= 0.0:
        raise ValueError("--phase-velocity-m-per-s must be positive")
    if fit_max_evals <= 0:
        raise ValueError("--fit-max-evals must be positive")
    if fit_random_samples <= 0:
        raise ValueError("--fit-random-samples must be positive")
    if args.fit_f_scale <= 0.0:
        raise ValueError("--fit-f-scale must be positive")

    if not any(
        [
            args.fit_l_scale,
            args.fit_c_scale,
            args.fit_i_star_scale,
            args.fit_loss_scale,
            args.fit_pump_scale,
            args.fit_gain_offset_db,
            args.fit_phase_offset_rad,
        ]
    ):
        raise ValueError("At least one fit parameter must be enabled.")

    checks = {
        "l_scale": (args.lower_l_scale, args.upper_l_scale, args.initial_l_scale),
        "c_scale": (args.lower_c_scale, args.upper_c_scale, args.initial_c_scale),
        "i_star_scale": (args.lower_i_star_scale, args.upper_i_star_scale, args.initial_i_star_scale),
        "loss_scale": (args.lower_loss_scale, args.upper_loss_scale, args.initial_loss_scale),
        "pump_scale": (args.lower_pump_scale, args.upper_pump_scale, args.initial_pump_scale),
        "gain_offset_db": (args.lower_gain_offset_db, args.upper_gain_offset_db, args.initial_gain_offset_db),
        "phase_offset_rad": (args.lower_phase_offset_rad, args.upper_phase_offset_rad, args.initial_phase_offset_rad),
    }
    for name, (lo, hi, init) in checks.items():
        if lo >= hi:
            raise ValueError(f"{name}: lower bound must be smaller than upper bound")
        if not (lo <= init <= hi):
            raise ValueError(f"{name}: initial value must lie inside bounds")

    for name in [
        "gain_weight",
        "idler_weight",
        "phase_weight",
        "group_delay_weight",
    ]:
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    for name in [
        "gain_sigma_db",
        "idler_sigma_db",
        "phase_sigma_rad",
        "group_delay_sigma_ps",
    ]:
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    return MeasurementFitConfig(
        measurement_csv=args.measurement_csv,
        measurement_npz=args.measurement_npz,
        output_dir=str(args.output_dir),
        name=str(args.name),
        quick=bool(args.quick),
        fit_mode=FitMode(args.fit_mode),
        fail_on_package_fallback=bool(args.fail_on_package_fallback),
        pump_frequency_ghz=float(args.pump_frequency_ghz),
        pump_current_ratio=float(args.pump_current_ratio),
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        fit_l_scale=bool(args.fit_l_scale),
        fit_c_scale=bool(args.fit_c_scale),
        fit_i_star_scale=bool(args.fit_i_star_scale),
        fit_loss_scale=bool(args.fit_loss_scale),
        fit_pump_scale=bool(args.fit_pump_scale),
        fit_gain_offset_db=bool(args.fit_gain_offset_db),
        fit_phase_offset_rad=bool(args.fit_phase_offset_rad),
        lower_l_scale=float(args.lower_l_scale),
        upper_l_scale=float(args.upper_l_scale),
        lower_c_scale=float(args.lower_c_scale),
        upper_c_scale=float(args.upper_c_scale),
        lower_i_star_scale=float(args.lower_i_star_scale),
        upper_i_star_scale=float(args.upper_i_star_scale),
        lower_loss_scale=float(args.lower_loss_scale),
        upper_loss_scale=float(args.upper_loss_scale),
        lower_pump_scale=float(args.lower_pump_scale),
        upper_pump_scale=float(args.upper_pump_scale),
        lower_gain_offset_db=float(args.lower_gain_offset_db),
        upper_gain_offset_db=float(args.upper_gain_offset_db),
        lower_phase_offset_rad=float(args.lower_phase_offset_rad),
        upper_phase_offset_rad=float(args.upper_phase_offset_rad),
        initial_l_scale=float(args.initial_l_scale),
        initial_c_scale=float(args.initial_c_scale),
        initial_i_star_scale=float(args.initial_i_star_scale),
        initial_loss_scale=float(args.initial_loss_scale),
        initial_pump_scale=float(args.initial_pump_scale),
        initial_gain_offset_db=float(args.initial_gain_offset_db),
        initial_phase_offset_rad=float(args.initial_phase_offset_rad),
        gain_weight=float(args.gain_weight),
        idler_weight=float(args.idler_weight),
        phase_weight=float(args.phase_weight),
        group_delay_weight=float(args.group_delay_weight),
        gain_sigma_db=float(args.gain_sigma_db),
        idler_sigma_db=float(args.idler_sigma_db),
        phase_sigma_rad=float(args.phase_sigma_rad),
        group_delay_sigma_ps=float(args.group_delay_sigma_ps),
        fit_max_evals=fit_max_evals,
        fit_random_samples=fit_random_samples,
        fit_robust_loss=str(args.fit_robust_loss),
        fit_f_scale=float(args.fit_f_scale),
        make_plots=not bool(args.no_plots),
        save_checkpoint=not bool(args.no_checkpoint),
        export_csv=not bool(args.no_csv),
    )


def _stage_load_measurement(holders: dict[str, Any], config: MeasurementFitConfig) -> dict[str, Any]:
    dataset = load_measurement_dataset(config)
    holders["dataset"] = dataset

    return {
        "status": RunStatus.PASS.value if dataset.get("load_mode") == "package" else RunStatus.PARTIAL.value,
        "source": dataset.get("source"),
        "load_mode": dataset.get("load_mode"),
        "frequency_hz": array_summary(dataset["frequency_hz"]),
        "signal_gain_db": array_summary(dataset["signal_gain_db"]),
        "idler_conversion_db": None if dataset.get("idler_conversion_db") is None else array_summary(dataset["idler_conversion_db"]),
        "s21_phase_rad": None if dataset.get("s21_phase_rad") is None else array_summary(dataset["s21_phase_rad"]),
        "group_delay_ps": None if dataset.get("group_delay_ps") is None else array_summary(dataset["group_delay_ps"]),
        "s21": None if dataset.get("s21") is None else array_summary(dataset["s21"]),
        "raw_column_names": dataset.get("raw_column_names"),
        "metadata": dataset.get("metadata"),
        "messages": tuple(dataset.get("messages", ("PASS: measurement dataset loaded.",))),
    }


def _stage_fit(holders: dict[str, Any], config: MeasurementFitConfig) -> dict[str, Any]:
    fit_result = fit_measurement_dataset(
        dataset=holders["dataset"],
        config=config,
    )
    holders["fit_result"] = fit_result

    status = RunStatus(fit_result.get("status", RunStatus.PASS.value))
    if not bool(fit_result.get("success", True)) and status == RunStatus.PASS:
        status = RunStatus.FAIL

    return {
        "status": status.value,
        "mode": fit_result.get("mode"),
        "success": fit_result.get("success"),
        "fit_params": fit_result.get("fit_params"),
        "loss": fit_result.get("loss"),
        "residual_norm": fit_result.get("residual_norm"),
        "n_evaluations": fit_result.get("n_evaluations"),
        "message": fit_result.get("message"),
        "package_error": fit_result.get("package_error"),
        "prediction": {
            "frequency_hz": array_summary(fit_result["prediction"]["frequency_hz"]),
            "signal_gain_db": array_summary(fit_result["prediction"]["signal_gain_db"]),
        },
        "messages": tuple(fit_result.get("messages", ("PASS: fit completed.",))),
    }


def _stage_metrics(holders: dict[str, Any], config: MeasurementFitConfig) -> dict[str, Any]:
    metrics = compute_fit_metrics(
        dataset=holders["dataset"],
        fit_result=holders["fit_result"],
        config=config,
    )
    holders["metrics"] = metrics

    gain_rms = metrics.get("gain_rms_error_db")
    status = RunStatus.PASS
    if gain_rms is None or not np.isfinite(gain_rms):
        status = RunStatus.FAIL
    elif gain_rms > 3.0:
        status = RunStatus.PARTIAL

    return {
        "status": status.value,
        **metrics,
        "messages": (
            "PASS: fit metrics computed."
            if status == RunStatus.PASS
            else "PARTIAL: fit metrics computed but residuals are large."
            if status == RunStatus.PARTIAL
            else "FAIL: fit metrics are not finite."
        ),
    }


def finalize_result(result: MeasurementFitResult, output_dir: Path) -> int:
    summary_json = output_dir / "fit_measurements_summary.json"
    summary_md = output_dir / "fit_measurements_summary.md"

    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }

    result = MeasurementFitResult(
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
    print(f"[fit-measurements] status: {result.status.value}")
    print(f"[fit-measurements] summary JSON: {summary_json}")
    print(f"[fit-measurements] summary MD:   {summary_md}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[fit-measurements] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "python": sys.version,
        "jax": {
            "version": getattr(jax, "__version__", None),
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "devices": [str(d) for d in jax.devices()],
        },
        "scipy_available": SCIPY_AVAILABLE,
        "script": "scripts/fit_measurements.py",
    }

    holders: dict[str, Any] = {}
    stages: list[StageResult] = []
    artifacts: dict[str, str] = {}

    print("[fit-measurements] loading measurement...")
    stage = run_stage("load_measurement", lambda: _stage_load_measurement(holders, config))
    stages.append(stage)
    print(f"[fit-measurements] load_measurement: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = MeasurementFitResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[fit-measurements] fitting parameters...")
    stage = run_stage("fit", lambda: _stage_fit(holders, config))
    stages.append(stage)
    print(f"[fit-measurements] fit: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = MeasurementFitResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[fit-measurements] computing metrics...")
    stage = run_stage("metrics", lambda: _stage_metrics(holders, config))
    stages.append(stage)
    print(f"[fit-measurements] metrics: {stage.status.value}")

    print("[fit-measurements] exporting artifacts...")
    try:
        artifacts.update(
            export_artifacts(
                config=config,
                dataset=holders["dataset"],
                fit_result=holders["fit_result"],
                metrics=holders["metrics"],
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

    metadata["fit_params"] = holders.get("fit_result", {}).get("fit_params", {})
    metadata["fit_metrics"] = holders.get("metrics", {})

    hard_fail = any(s.status in {RunStatus.FAIL, RunStatus.ERROR} for s in stages)
    partial = any(s.status == RunStatus.PARTIAL for s in stages)

    if hard_fail:
        status = RunStatus.ERROR
    elif partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.PASS

    result = MeasurementFitResult(
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
