"""
Run synthetic parameter-recovery experiments for the TWPA stack.

This script creates synthetic measurement data from a known parameter set, adds
controlled noise, fits the same parameter family back from the synthetic data,
and reports recovery accuracy.

It is designed for two use cases:

    1. Production package mode:
       Use ``twpa.inference.synthetic``, ``twpa.inference.fitting``, and
       ``twpa.inference.recovery`` when their APIs are available.

    2. Robust fallback mode:
       Use a transparent analytic gain/dispersion surrogate so the recovery
       pipeline can still be smoke-tested before the full package-native
       inference stack is wired.

Examples
--------
Quick smoke test:

    python scripts/synthetic_recovery.py --quick --output-dir outputs/synthetic_recovery_quick

Run 10 noisy recovery trials:

    python scripts/synthetic_recovery.py ^
      --n-trials 10 ^
      --noise-std-db 0.15 ^
      --fit-max-evals 600 ^
      --output-dir outputs/synthetic_recovery

Force fallback surrogate mode:

    python scripts/synthetic_recovery.py ^
      --synthetic-mode fallback_surrogate ^
      --fit-mode fallback_scipy ^
      --output-dir outputs/synthetic_recovery_fallback

Try package-native inference only:

    python scripts/synthetic_recovery.py ^
      --synthetic-mode package ^
      --fit-mode package ^
      --fail-on-package-fallback ^
      --output-dir outputs/synthetic_recovery_package
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


try:
    from scipy.optimize import least_squares, differential_evolution

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    least_squares = None
    differential_evolution = None


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class SyntheticMode(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    FALLBACK_SURROGATE = "fallback_surrogate"


class FitMode(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    FALLBACK_SCIPY = "fallback_scipy"
    FALLBACK_RANDOM = "fallback_random"


@dataclass(frozen=True)
class SyntheticRecoveryConfig:
    output_dir: str
    name: str
    seed: int
    quick: bool

    synthetic_mode: SyntheticMode
    fit_mode: FitMode
    fail_on_package_fallback: bool

    n_trials: int
    noise_std_db: float
    noise_std_phase_rad: float

    signal_f_min_ghz: float
    signal_f_max_ghz: float
    n_signal: int
    pump_frequency_ghz: float
    pump_current_ratio: float
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float

    true_l_scale: float
    true_c_scale: float
    true_i_star_scale: float
    true_loss_scale: float
    true_pump_scale: float

    fit_l_scale: bool
    fit_c_scale: bool
    fit_i_star_scale: bool
    fit_loss_scale: bool
    fit_pump_scale: bool

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

    initial_l_scale: float
    initial_c_scale: float
    initial_i_star_scale: float
    initial_loss_scale: float
    initial_pump_scale: float

    fit_max_evals: int
    fit_random_samples: int
    fit_robust_loss: str
    fit_f_scale: float

    make_plots: bool
    save_checkpoint: bool
    export_csv: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["synthetic_mode"] = self.synthetic_mode.value
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
class RecoveryTrialResult:
    trial_index: int
    status: RunStatus
    elapsed_s: float
    seed: int
    synthetic_summary: Mapping[str, Any]
    fit_summary: Mapping[str, Any]
    truth_comparison: Mapping[str, Any]
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "seed": self.seed,
            "synthetic_summary": jsonify(self.synthetic_summary),
            "fit_summary": jsonify(self.fit_summary),
            "truth_comparison": jsonify(self.truth_comparison),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class SyntheticRecoveryResult:
    config: SyntheticRecoveryConfig
    status: RunStatus
    elapsed_s: float
    stages: tuple[StageResult, ...]
    trials: tuple[RecoveryTrialResult, ...]
    artifact_paths: Mapping[str, str]
    metadata: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_pass(self) -> int:
        return sum(1 for t in self.trials if t.status == RunStatus.PASS)

    @property
    def n_partial(self) -> int:
        return sum(1 for t in self.trials if t.status == RunStatus.PARTIAL)

    @property
    def n_error(self) -> int:
        return sum(1 for t in self.trials if t.status in {RunStatus.FAIL, RunStatus.ERROR})

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "n_trials": self.n_trials,
            "n_pass": self.n_pass,
            "n_partial": self.n_partial,
            "n_error": self.n_error,
            "config": self.config.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "trials": [t.to_dict() for t in self.trials],
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
    finite = np.isfinite(arr.real) & np.isfinite(arr.imag) if np.iscomplexobj(arr) else np.isfinite(arr)
    out["finite_count"] = int(np.count_nonzero(finite))
    if not np.any(finite):
        out["available"] = False
        return out
    out["available"] = True
    if np.iscomplexobj(arr):
        values = np.abs(arr[finite])
        out.update(
            {
                "min_abs": float(np.min(values)),
                "max_abs": float(np.max(values)),
                "mean_abs": float(np.mean(values)),
            }
        )
    else:
        values = arr[finite]
        out.update(
            {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
        )
    return out


def active_parameter_names(config: SyntheticRecoveryConfig) -> tuple[str, ...]:
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
    return tuple(names)


def truth_parameters(config: SyntheticRecoveryConfig) -> dict[str, float]:
    return {
        "l_scale": float(config.true_l_scale),
        "c_scale": float(config.true_c_scale),
        "i_star_scale": float(config.true_i_star_scale),
        "loss_scale": float(config.true_loss_scale),
        "pump_scale": float(config.true_pump_scale),
    }


def initial_parameters(config: SyntheticRecoveryConfig) -> dict[str, float]:
    return {
        "l_scale": float(config.initial_l_scale),
        "c_scale": float(config.initial_c_scale),
        "i_star_scale": float(config.initial_i_star_scale),
        "loss_scale": float(config.initial_loss_scale),
        "pump_scale": float(config.initial_pump_scale),
    }


def parameter_bounds(config: SyntheticRecoveryConfig) -> dict[str, tuple[float, float]]:
    return {
        "l_scale": (float(config.lower_l_scale), float(config.upper_l_scale)),
        "c_scale": (float(config.lower_c_scale), float(config.upper_c_scale)),
        "i_star_scale": (float(config.lower_i_star_scale), float(config.upper_i_star_scale)),
        "loss_scale": (float(config.lower_loss_scale), float(config.upper_loss_scale)),
        "pump_scale": (float(config.lower_pump_scale), float(config.upper_pump_scale)),
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


def signal_grid_hz(config: SyntheticRecoveryConfig) -> jax.Array:
    return jnp.linspace(
        config.signal_f_min_ghz * 1e9,
        config.signal_f_max_ghz * 1e9,
        config.n_signal,
        dtype=jnp.float64,
    )


def fallback_surrogate_model(
    params: Mapping[str, float],
    config: SyntheticRecoveryConfig,
) -> dict[str, jax.Array]:
    """
    Transparent analytic surrogate for gain and phase.

    This is not a replacement for full HB. It is a deterministic nonlinear
    response model with enough structure to validate recovery infrastructure:

        - L and C scales shift phase velocity and band center.
        - I* and pump scales change parametric gain strength.
        - loss scale controls distributed attenuation.
    """
    f = signal_grid_hz(config)
    f_np = np.asarray(f, dtype=float)

    l_scale = float(params["l_scale"])
    c_scale = float(params["c_scale"])
    i_star_scale = float(params["i_star_scale"])
    loss_scale = float(params["loss_scale"])
    pump_scale = float(params["pump_scale"])

    vp = config.phase_velocity_m_per_s / math.sqrt(max(l_scale * c_scale, 1e-300))
    length_m = config.length_mm * 1e-3

    pump_frequency_hz = config.pump_frequency_ghz * 1e9
    center_hz = 0.5 * pump_frequency_hz * (l_scale / c_scale) ** 0.12

    bandwidth_hz = max(0.10 * pump_frequency_hz, 0.18 * pump_frequency_hz / math.sqrt(max(l_scale * c_scale, 1e-300)))
    detuning = (f_np - center_hz) / bandwidth_hz

    pump_strength = config.pump_current_ratio * pump_scale / max(i_star_scale, 1e-300)
    nonlinear_gain_np = 5.5 * pump_strength**2 * length_m / 0.1

    phase_mismatch = 1.8 * detuning + 0.35 * (l_scale - c_scale)
    phase_matching = np.sinc(phase_mismatch / np.pi) ** 2

    gain_power = 1.0 + np.sinh(nonlinear_gain_np * np.sqrt(np.maximum(phase_matching, 0.0))) ** 2

    attenuation_np = 0.020 * loss_scale * (length_m / 0.1) * (f_np / max(center_hz, 1.0)) ** 0.7
    ripple_db = 0.10 * np.sin(2.0 * np.pi * (f_np - f_np[0]) / max(f_np[-1] - f_np[0], 1.0) * 3.0)

    gain_db = 10.0 * np.log10(np.maximum(gain_power, 1e-300)) - 8.686 * attenuation_np + ripple_db

    beta = 2.0 * np.pi * f_np / vp
    phase_rad = -beta * length_m
    group_delay_s = np.gradient(-phase_rad, 2.0 * np.pi * f_np)

    s21_mag = 10.0 ** (gain_db / 20.0)
    s21 = s21_mag * np.exp(1j * phase_rad)

    idler_conversion_db = gain_db - 3.0 - 10.0 * np.log10(1.0 + detuning**2)

    return {
        "signal_frequency_hz": jnp.asarray(f_np, dtype=jnp.float64),
        "signal_gain_db": jnp.asarray(gain_db, dtype=jnp.float64),
        "idler_conversion_db": jnp.asarray(idler_conversion_db, dtype=jnp.float64),
        "s21": jnp.asarray(s21, dtype=jnp.complex128),
        "s21_phase_rad": jnp.asarray(phase_rad, dtype=jnp.float64),
        "group_delay_s": jnp.asarray(group_delay_s, dtype=jnp.float64),
        "center_frequency_hz": jnp.asarray(center_hz, dtype=jnp.float64),
        "bandwidth_hz": jnp.asarray(bandwidth_hz, dtype=jnp.float64),
    }


def add_measurement_noise(
    clean: Mapping[str, Any],
    *,
    config: SyntheticRecoveryConfig,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    gain_clean = np.asarray(clean["signal_gain_db"], dtype=float)
    idler_clean = np.asarray(clean["idler_conversion_db"], dtype=float)
    phase_clean = np.asarray(clean["s21_phase_rad"], dtype=float)
    s21_clean = np.asarray(clean["s21"], dtype=np.complex128)

    gain_noise = rng.normal(0.0, config.noise_std_db, size=gain_clean.shape)
    idler_noise = rng.normal(0.0, config.noise_std_db, size=idler_clean.shape)
    phase_noise = rng.normal(0.0, config.noise_std_phase_rad, size=phase_clean.shape)

    gain_meas = gain_clean + gain_noise
    idler_meas = idler_clean + idler_noise
    phase_meas = phase_clean + phase_noise

    s21_mag_meas = 10.0 ** (gain_meas / 20.0)
    s21_meas = s21_mag_meas * np.exp(1j * phase_meas)

    return {
        **dict(clean),
        "signal_gain_db_clean": jnp.asarray(gain_clean, dtype=jnp.float64),
        "idler_conversion_db_clean": jnp.asarray(idler_clean, dtype=jnp.float64),
        "s21_phase_rad_clean": jnp.asarray(phase_clean, dtype=jnp.float64),
        "s21_clean": jnp.asarray(s21_clean, dtype=jnp.complex128),
        "signal_gain_db": jnp.asarray(gain_meas, dtype=jnp.float64),
        "idler_conversion_db": jnp.asarray(idler_meas, dtype=jnp.float64),
        "s21_phase_rad": jnp.asarray(phase_meas, dtype=jnp.float64),
        "s21": jnp.asarray(s21_meas, dtype=jnp.complex128),
        "gain_noise_db": jnp.asarray(gain_noise, dtype=jnp.float64),
        "idler_noise_db": jnp.asarray(idler_noise, dtype=jnp.float64),
        "phase_noise_rad": jnp.asarray(phase_noise, dtype=jnp.float64),
        "noise": {
            "noise_std_db": float(config.noise_std_db),
            "noise_std_phase_rad": float(config.noise_std_phase_rad),
            "seed": int(seed),
        },
    }


def try_package_synthetic_dataset(
    *,
    truth: Mapping[str, float],
    config: SyntheticRecoveryConfig,
    seed: int,
) -> dict[str, Any]:
    import twpa.inference.synthetic as synthetic_module

    candidate_names = [
        "generate_synthetic_recovery_dataset",
        "make_synthetic_recovery_dataset",
        "generate_synthetic_measurements",
        "make_synthetic_measurements",
        "simulate_synthetic_measurement",
    ]

    errors: list[str] = []

    for name in candidate_names:
        fn = getattr(synthetic_module, name, None)
        if fn is None:
            continue

        kwargs = {
            "truth": dict(truth),
            "true_params": dict(truth),
            "parameters": dict(truth),
            "signal_frequency_hz": signal_grid_hz(config),
            "pump_frequency_hz": config.pump_frequency_ghz * 1e9,
            "pump_current_ratio": config.pump_current_ratio,
            "length_m": config.length_mm * 1e-3,
            "z0_ohm": config.z0_ohm,
            "phase_velocity_m_per_s": config.phase_velocity_m_per_s,
            "noise_std_db": config.noise_std_db,
            "noise_std_phase_rad": config.noise_std_phase_rad,
            "seed": seed,
            "config": config,
        }

        try:
            result = call_with_supported_kwargs(fn, kwargs)
            dataset = normalize_dataset(result, config=config)
            dataset["mode"] = f"package_{name}"
            dataset["package_result"] = jsonify(result)
            return dataset
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    from twpa.core.hb_fft import HBProjectionConfig
    from twpa.core.layout import make_uniform_layout
    from twpa.core.params import LineParams, NonlinearParams, SolverBackend, SolverConfig
    from twpa.inference.synthetic import (
        SyntheticNoiseConfig,
        generate_combined_synthetic_dataset,
    )
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig, PumpHBLadderConfig

    n_cells = 4 if config.quick else 8
    base_layout = make_uniform_layout(
        LineParams.from_z0_vp(
            length_m=config.length_mm * 1e-3,
            n_cells=n_cells,
            z0_ohm=config.z0_ohm,
            phase_velocity_m_per_s=config.phase_velocity_m_per_s,
            R_per_m_ohm=0.05,
            name="synthetic_recovery_linear_base",
        )
    )
    package_truth = {
        "L_scale": float(truth["l_scale"]),
        "C_scale": float(truth["c_scale"]),
        "R_scale": float(truth["loss_scale"]),
        "I_star_scale": float(truth["i_star_scale"]),
        "pump_current_scale": float(truth["pump_scale"]),
    }
    nonlinear = NonlinearParams(I_star_A=5e-3)
    pump_drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=config.pump_frequency_ghz * 1e9,
        current_rms_A=config.pump_current_ratio * nonlinear.I_star_A,
        source_impedance_ohm=config.z0_ohm,
    )
    pump_config = PumpHBLadderConfig(
        n_pump_harmonics=1,
        projection=HBProjectionConfig(n_time_samples=16),
        solver=SolverConfig(
            backend=SolverBackend.NEWTON_KRYLOV,
            max_iter=20,
            abs_tol=1e-9,
            rel_tol=1e-9,
            verbose=False,
        ),
    )
    package_dataset = generate_combined_synthetic_dataset(
        base_layout,
        nonlinear_params=nonlinear,
        sparameter_frequency_hz=signal_grid_hz(config),
        signal_frequency_hz=signal_grid_hz(config),
        pump_drive=pump_drive,
        pump_config=pump_config,
        noise=SyntheticNoiseConfig(
            s_db_std=config.noise_std_db,
            gain_db_std=config.noise_std_db,
            seed=seed,
        ),
        true_parameters=package_truth,
        metadata={"adapter": "scripts.synthetic_recovery.package_nonlinear_hb"},
    )
    assert package_dataset.gain is not None
    assert package_dataset.sparameters is not None
    return {
        "signal_frequency_hz": package_dataset.gain.signal_frequency_hz,
        "signal_gain_db": package_dataset.gain.signal_gain_db_noisy,
        "idler_conversion_db": package_dataset.gain.idler_conversion_db_noisy,
        "s21": package_dataset.sparameters.s_noisy[:, 1, 0],
        "s21_phase_rad": jnp.unwrap(jnp.angle(package_dataset.sparameters.s_noisy[:, 1, 0])),
        "mode": "package_nonlinear_hb_combined",
        "status": RunStatus.PASS.value,
        "noise": package_dataset.gain.noise.to_dict(),
        "package_dataset": package_dataset,
        "package_base_layout": base_layout,
        "package_nonlinear_params": nonlinear,
        "package_pump_drive": pump_drive,
        "package_pump_config": pump_config,
        "adapter_errors": errors[-10:],
        "messages": (
            "PASS: package-native nonlinear HB combined synthetic dataset generated.",
        ),
    }


def normalize_dataset(dataset: Any, *, config: SyntheticRecoveryConfig) -> dict[str, Any]:
    signal_frequency_hz = get_attr_any(
        dataset,
        "signal_frequency_hz",
        "frequency_hz",
        default=signal_grid_hz(config),
    )
    gain_db = get_attr_any(
        dataset,
        "signal_gain_db",
        "gain_db",
        default=None,
    )
    idler_db = get_attr_any(
        dataset,
        "idler_conversion_db",
        "conversion_db",
        default=None,
    )
    s21 = get_attr_any(dataset, "s21", "S21", default=None)
    phase = get_attr_any(dataset, "s21_phase_rad", "phase_rad", default=None)

    if gain_db is None and s21 is not None:
        gain_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(s21)), 1e-300))

    if s21 is None and gain_db is not None and phase is not None:
        s21 = 10.0 ** (jnp.asarray(gain_db) / 20.0) * jnp.exp(1j * jnp.asarray(phase))

    if idler_db is None and gain_db is not None:
        idler_db = jnp.asarray(gain_db) - 3.0

    if phase is None and s21 is not None:
        phase = jnp.unwrap(jnp.angle(jnp.asarray(s21)))

    if gain_db is None:
        raise ValueError("Dataset does not expose signal gain or S21.")

    return {
        "signal_frequency_hz": jnp.asarray(signal_frequency_hz, dtype=jnp.float64),
        "signal_gain_db": jnp.asarray(gain_db, dtype=jnp.float64),
        "idler_conversion_db": jnp.asarray(idler_db, dtype=jnp.float64),
        "s21": None if s21 is None else jnp.asarray(s21, dtype=jnp.complex128),
        "s21_phase_rad": None if phase is None else jnp.asarray(phase, dtype=jnp.float64),
        "raw": jsonify(dataset),
    }


def generate_synthetic_dataset(
    *,
    truth: Mapping[str, float],
    config: SyntheticRecoveryConfig,
    seed: int,
) -> dict[str, Any]:
    if config.synthetic_mode in {SyntheticMode.AUTO, SyntheticMode.PACKAGE}:
        try:
            dataset = try_package_synthetic_dataset(
                truth=truth,
                config=config,
                seed=seed,
            )
            dataset.setdefault("status", RunStatus.PASS.value)
            dataset.setdefault(
                "messages",
                (f"PASS: package synthetic generator `{dataset['mode']}` completed.",),
            )
            return dataset
        except Exception as exc:
            raise
    else:
        package_error = "package synthetic generator disabled"

    clean = fallback_surrogate_model(truth, config)
    noisy = add_measurement_noise(clean, config=config, seed=seed)
    noisy["status"] = RunStatus.PARTIAL.value
    noisy["mode"] = "fallback_surrogate"
    noisy["package_error"] = package_error
    noisy["messages"] = (
        "PARTIAL: used fallback analytic surrogate to generate synthetic data.",
        package_error,
    )
    return noisy


def residual_vector(
    vector: np.ndarray,
    *,
    names: Sequence[str],
    base_params: Mapping[str, float],
    dataset: Mapping[str, Any],
    config: SyntheticRecoveryConfig,
) -> np.ndarray:
    params = decode_params(vector, names, base=base_params)
    pred = fallback_surrogate_model(params, config)

    gain_pred = np.asarray(pred["signal_gain_db"], dtype=float)
    gain_meas = np.asarray(dataset["signal_gain_db"], dtype=float)

    residuals = [(gain_pred - gain_meas) / max(config.noise_std_db, 1e-6)]

    idler_pred = np.asarray(pred["idler_conversion_db"], dtype=float)
    idler_meas = np.asarray(dataset["idler_conversion_db"], dtype=float)
    residuals.append(0.5 * (idler_pred - idler_meas) / max(config.noise_std_db, 1e-6))

    phase_meas = dataset.get("s21_phase_rad")
    if phase_meas is not None and config.noise_std_phase_rad > 0.0:
        phase_pred = np.asarray(pred["s21_phase_rad"], dtype=float)
        phase_meas_arr = np.asarray(phase_meas, dtype=float)
        phase_res = np.unwrap(phase_pred) - np.unwrap(phase_meas_arr)
        phase_res = phase_res - np.nanmean(phase_res)
        residuals.append(0.2 * phase_res / max(config.noise_std_phase_rad, 1e-9))

    return np.concatenate([np.asarray(r, dtype=float).ravel() for r in residuals])


def fallback_random_fit(
    *,
    dataset: Mapping[str, Any],
    config: SyntheticRecoveryConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + 971)
    names = active_parameter_names(config)
    base = initial_parameters(config)
    bounds = parameter_bounds(config)
    lower, upper = bounds_arrays(bounds, names)

    best_x = encode_params(base, names)
    best_res = residual_vector(best_x, names=names, base_params=base, dataset=dataset, config=config)
    best_loss = float(np.mean(best_res**2))

    n_samples = int(config.fit_random_samples)
    records = []

    for i in range(n_samples):
        x = rng.uniform(lower, upper)
        res = residual_vector(x, names=names, base_params=base, dataset=dataset, config=config)
        loss = float(np.mean(res**2))
        records.append({"evaluation_index": i + 1, "loss": loss})

        if loss < best_loss:
            best_loss = loss
            best_x = x
            best_res = res

    fit_params = decode_params(best_x, names, base=base)

    return {
        "status": RunStatus.PARTIAL.value,
        "mode": "fallback_random",
        "success": True,
        "fit_params": fit_params,
        "x": best_x,
        "loss": best_loss,
        "residual_norm": float(np.linalg.norm(best_res)),
        "n_evaluations": n_samples,
        "records": records,
        "messages": (
            "PARTIAL: used fallback random-search fit.",
        ),
    }


def fallback_scipy_fit(
    *,
    dataset: Mapping[str, Any],
    config: SyntheticRecoveryConfig,
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
        res = residual_vector(x, names=names, base_params=base, dataset=dataset, config=config)
        loss = float(np.mean(res**2))
        records.append(
            {
                "evaluation_index": len(records) + 1,
                "loss": loss,
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

    fit_params = decode_params(result.x, names, base=base)
    final_res = residual_vector(result.x, names=names, base_params=base, dataset=dataset, config=config)

    return {
        "status": RunStatus.PARTIAL.value,
        "mode": "fallback_scipy",
        "success": bool(result.success),
        "fit_params": fit_params,
        "x": np.asarray(result.x, dtype=float),
        "loss": float(np.mean(final_res**2)),
        "cost": float(result.cost),
        "residual_norm": float(np.linalg.norm(final_res)),
        "n_evaluations": int(result.nfev),
        "message": str(result.message),
        "records": records,
        "messages": (
            "PARTIAL: used fallback SciPy least-squares fit.",
        ),
    }


def try_package_fit(
    *,
    dataset: Mapping[str, Any],
    config: SyntheticRecoveryConfig,
) -> dict[str, Any]:
    import twpa.inference.fitting as fitting_module

    package_dataset = dataset.get("package_dataset")
    base_layout = dataset.get("package_base_layout")
    nonlinear = dataset.get("package_nonlinear_params")
    pump_drive = dataset.get("package_pump_drive")
    pump_config = dataset.get("package_pump_config")
    if (
        package_dataset is not None
        and base_layout is not None
        and nonlinear is not None
        and pump_drive is not None
        and pump_config is not None
        and getattr(package_dataset, "gain", None) is not None
    ):
        from twpa.inference.synthetic import (
            make_gain_frequency_plan,
            make_gain_sweep_config_for_frequencies,
        )
        from twpa.workflows.calibration import (
            CalibrationOptimizerConfig,
            CalibrationParameterSpec,
            CalibrationTarget,
            CalibrationVectorSpec,
            ParameterTransform,
            calibrate,
            finite_difference_residual_jacobian,
        )

        active = set(active_parameter_names(config))
        scale_map = {
            "l_scale": ("L_scale", config.initial_l_scale, config.lower_l_scale, config.upper_l_scale),
            "c_scale": ("C_scale", config.initial_c_scale, config.lower_c_scale, config.upper_c_scale),
            "loss_scale": ("R_scale", config.initial_loss_scale, config.lower_loss_scale, config.upper_loss_scale),
            "i_star_scale": ("I_star_scale", config.initial_i_star_scale, config.lower_i_star_scale, config.upper_i_star_scale),
            "pump_scale": ("pump_current_scale", config.initial_pump_scale, config.lower_pump_scale, config.upper_pump_scale),
        }
        specs = [
            CalibrationParameterSpec(
                package_name,
                initial=initial,
                lower=lower,
                upper=upper,
                transform=ParameterTransform.LOG,
            )
            for script_name, (package_name, initial, lower, upper) in scale_map.items()
            if script_name in active
        ]
        gain_dataset = package_dataset.gain
        assert gain_dataset is not None

        def target_plan_factory(pump_result: Any) -> Any:
            return make_gain_frequency_plan(
                pump_frequency_hz=pump_result.drive.pump_frequency_hz,
                signal_frequency_hz=gain_dataset.signal_frequency_hz,
                idler_frequency_hz=gain_dataset.idler_frequency_hz,
                pump_label=pump_result.drive.pump_label,
                signal_labels=gain_dataset.signal_labels,
                idler_labels=gain_dataset.idler_labels,
                n_pump_harmonics=pump_config.n_pump_harmonics,
                include_negative=pump_config.include_negative_frequencies,
                include_dc=pump_config.include_dc,
            )

        output_impedance = (
            1.0 / pump_config.distributed.load_conductance_S
            if pump_config.distributed.load_conductance_S > 0.0
            else pump_drive.source_impedance_ohm
        )

        def sweep_config_factory(_plan: Any) -> Any:
            return make_gain_sweep_config_for_frequencies(
                signal_labels=gain_dataset.signal_labels,
                idler_labels=gain_dataset.idler_labels,
                input_node=pump_config.distributed.input_node,
                output_node=None,
                input_impedance_ohm=pump_drive.source_impedance_ohm,
                output_impedance_ohm=output_impedance,
            )

        target = CalibrationTarget(
            base_layout=base_layout,
            base_nonlinear_params=nonlinear,
            pump_drive=pump_drive,
            pump_config=pump_config,
            target_plan_factory=target_plan_factory,
            sweep_config_factory=sweep_config_factory,
        )
        vector_spec = CalibrationVectorSpec(tuple(specs))
        calibration = calibrate(
            target,
            vector_spec,
            sparameter_data=package_dataset.sparameters.to_calibration_data(),
            gain_data=gain_dataset.to_calibration_data(),
            optimizer_config=CalibrationOptimizerConfig(
                max_evaluations=config.fit_max_evals,
                verbose=False,
            ),
        )
        diagnostics = finite_difference_residual_jacobian(
            target,
            vector_spec,
            calibration.best_encoded_vector,
            sparameter_data=package_dataset.sparameters.to_calibration_data(),
            gain_data=gain_dataset.to_calibration_data(),
        )
        reverse = {package_name: script_name for script_name, (package_name, *_rest) in scale_map.items()}
        fit_params = initial_parameters(config)
        fit_params.update(
            {
                reverse[name]: float(value)
                for name, value in calibration.best_parameters.items()
                if name in reverse
            }
        )
        return {
            "status": RunStatus.PASS.value if calibration.success else RunStatus.PARTIAL.value,
            "mode": "package_nonlinear_hb_calibration",
            "success": bool(calibration.success),
            "fit_params": fit_params,
            "loss": calibration.loss,
            "residual_norm": float(jnp.linalg.norm(calibration.best_evaluation.residual)),
            "n_evaluations": calibration.metadata.get("nfev"),
            "message": calibration.message,
            "identifiability": diagnostics,
            "messages": (
                "PASS: package-native nonlinear HB calibration completed."
                if calibration.success
                else "PARTIAL: package-native nonlinear HB calibration did not fully converge.",
            ),
        }

    if package_dataset is not None and base_layout is not None:
        from twpa.workflows.calibration import (
            CalibrationOptimizerConfig,
            CalibrationParameterSpec,
            CalibrationVectorSpec,
            ParameterTransform,
            SParameterCalibrationData,
            calibrate,
            make_linear_calibration_target,
        )

        active = set(active_parameter_names(config))
        specs = []
        scale_map = {
            "l_scale": ("L_scale", config.initial_l_scale, config.lower_l_scale, config.upper_l_scale),
            "c_scale": ("C_scale", config.initial_c_scale, config.lower_c_scale, config.upper_c_scale),
            "loss_scale": ("R_scale", config.initial_loss_scale, config.lower_loss_scale, config.upper_loss_scale),
        }
        for script_name, (package_name, initial, lower, upper) in scale_map.items():
            if script_name in active:
                specs.append(
                    CalibrationParameterSpec(
                        package_name,
                        initial=initial,
                        lower=lower,
                        upper=upper,
                        transform=ParameterTransform.LOG,
                    )
                )

        if not specs:
            raise RuntimeError("Package linear recovery has no identifiable active parameters.")

        calibration = calibrate(
            make_linear_calibration_target(base_layout),
            CalibrationVectorSpec(tuple(specs)),
            sparameter_data=SParameterCalibrationData(
                frequency_hz=package_dataset.frequency_hz,
                s=package_dataset.s_noisy,
                s21_db=package_dataset.s21_db_noisy,
            ),
            optimizer_config=CalibrationOptimizerConfig(
                max_evaluations=config.fit_max_evals,
                verbose=False,
            ),
        )
        reverse = {"L_scale": "l_scale", "C_scale": "c_scale", "R_scale": "loss_scale"}
        fit_params = initial_parameters(config)
        fit_params.update(
            {
                reverse[name]: float(value)
                for name, value in calibration.best_parameters.items()
                if name in reverse
            }
        )
        unidentifiable = sorted(active - set(scale_map))
        status = RunStatus.PARTIAL if unidentifiable else RunStatus.PASS
        return {
            "status": status.value,
            "mode": "package_linear_sparameter_calibration",
            "success": bool(calibration.success),
            "fit_params": fit_params,
            "loss": calibration.loss,
            "residual_norm": float(jnp.linalg.norm(calibration.best_evaluation.residual)),
            "n_evaluations": calibration.metadata.get("nfev"),
            "message": calibration.message,
            "identifiability": {
                "identifiable": sorted(active & set(scale_map)),
                "unidentifiable": unidentifiable,
            },
            "messages": (
                "PARTIAL: package-native linear S-parameter calibration completed."
                if unidentifiable
                else "PASS: package-native linear S-parameter calibration completed.",
                *(
                    (f"PARTIAL: unidentifiable parameters retained at initial values: {unidentifiable}",)
                    if unidentifiable
                    else ()
                ),
            ),
        }

    candidate_names = [
        "fit_synthetic_dataset",
        "fit_measurement_dataset",
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
            "max_evals": config.fit_max_evals,
            "max_nfev": config.fit_max_evals,
            "config": config,
        }

        try:
            result = call_with_supported_kwargs(fn, kwargs)
            fit_params = get_attr_any(result, "fit_params", "params", "parameters", default=None)
            if fit_params is None and isinstance(result, Mapping):
                fit_params = result.get("fit_params") or result.get("params") or result.get("parameters")
            if fit_params is None:
                raise RuntimeError("Package fit result does not expose fitted parameters.")

            return {
                "status": RunStatus.PASS.value,
                "mode": f"package_{name}",
                "success": bool(get_attr_any(result, "success", "converged", default=True)),
                "fit_params": {k: float(v) for k, v in dict(fit_params).items()},
                "package_result": jsonify(result),
                "messages": (
                    f"PASS: package-native fitter `{name}` completed.",
                ),
            }
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No compatible package fitter succeeded. Errors:\n"
        + "\n".join(errors[-10:])
    )


def fit_dataset(
    *,
    dataset: Mapping[str, Any],
    config: SyntheticRecoveryConfig,
) -> dict[str, Any]:
    if not active_parameter_names(config):
        raise ValueError("At least one fit_* parameter flag must be enabled.")

    if config.fit_mode in {FitMode.AUTO, FitMode.PACKAGE}:
        try:
            return try_package_fit(dataset=dataset, config=config)
        except Exception as exc:
            raise
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


def compare_fit_to_truth(
    *,
    truth: Mapping[str, float],
    fit_params: Mapping[str, float],
    config: SyntheticRecoveryConfig,
) -> dict[str, Any]:
    rows = []
    abs_rel_errors = []

    for name in active_parameter_names(config):
        t = float(truth[name])
        f = float(fit_params.get(name, np.nan))
        abs_error = f - t
        rel_error = abs_error / max(abs(t), 1e-300)
        abs_rel_errors.append(abs(rel_error))

        rows.append(
            {
                "name": name,
                "true": t,
                "fit": f,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "rel_error_percent": 100.0 * rel_error,
            }
        )

    max_abs_rel_error = float(np.nanmax(abs_rel_errors)) if abs_rel_errors else np.nan
    rms_rel_error = float(np.sqrt(np.nanmean(np.asarray(abs_rel_errors) ** 2))) if abs_rel_errors else np.nan

    passed = bool(max_abs_rel_error < 0.10)

    return {
        "passed": passed,
        "max_abs_rel_error": max_abs_rel_error,
        "rms_rel_error": rms_rel_error,
        "rows": rows,
    }


def run_trial(
    *,
    trial_index: int,
    config: SyntheticRecoveryConfig,
) -> tuple[RecoveryTrialResult, Mapping[str, Any]]:
    start = time.perf_counter()
    seed = config.seed + 1009 * trial_index
    truth = truth_parameters(config)

    try:
        dataset = generate_synthetic_dataset(
            truth=truth,
            config=config,
            seed=seed,
        )

        fit = fit_dataset(
            dataset=dataset,
            config=config,
        )

        comparison = compare_fit_to_truth(
            truth=truth,
            fit_params=fit["fit_params"],
            config=config,
        )

        synthetic_status = RunStatus(dataset.get("status", RunStatus.PASS.value))
        fit_status = RunStatus(fit.get("status", RunStatus.PASS.value))

        if comparison["passed"] and synthetic_status == RunStatus.PASS and fit_status == RunStatus.PASS:
            status = RunStatus.PASS
        elif comparison["passed"]:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.PARTIAL

        messages = (
            *tuple(dataset.get("messages", ())),
            *tuple(fit.get("messages", ())),
            (
                "PASS: fitted parameters recovered truth within tolerance."
                if comparison["passed"]
                else "PARTIAL: fit completed but recovery error exceeded tolerance."
            ),
        )

        trial = RecoveryTrialResult(
            trial_index=trial_index,
            status=status,
            elapsed_s=time.perf_counter() - start,
            seed=seed,
            synthetic_summary={
                "mode": dataset.get("mode"),
                "signal_frequency_hz": array_summary(dataset["signal_frequency_hz"]),
                "signal_gain_db": array_summary(dataset["signal_gain_db"]),
                "idler_conversion_db": array_summary(dataset["idler_conversion_db"]),
                "noise": dataset.get("noise"),
            },
            fit_summary={
                "mode": fit.get("mode"),
                "success": fit.get("success"),
                "fit_params": fit.get("fit_params"),
                "loss": fit.get("loss"),
                "residual_norm": fit.get("residual_norm"),
                "n_evaluations": fit.get("n_evaluations"),
                "message": fit.get("message"),
                "package_error": fit.get("package_error"),
                "identifiability": fit.get("identifiability", dataset.get("identifiability")),
            },
            truth_comparison=comparison,
            messages=tuple(str(m) for m in messages),
        )

        return trial, {"dataset": dataset, "fit": fit, "truth": truth}

    except Exception as exc:
        trial = RecoveryTrialResult(
            trial_index=trial_index,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            seed=seed,
            synthetic_summary={},
            fit_summary={},
            truth_comparison={},
            messages=(f"ERROR: {type(exc).__name__}: {exc}", traceback.format_exc()),
        )
        return trial, {}


def write_trial_parameters_csv(
    path: Path,
    trials: Sequence[RecoveryTrialResult],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "trial_index",
        "status",
        "elapsed_s",
        "parameter",
        "true",
        "fit",
        "abs_error",
        "rel_error",
        "rel_error_percent",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            rows = trial.truth_comparison.get("rows", []) if trial.truth_comparison else []
            for row in rows:
                writer.writerow(
                    {
                        "trial_index": trial.trial_index,
                        "status": trial.status.value,
                        "elapsed_s": trial.elapsed_s,
                        "parameter": row.get("name"),
                        "true": row.get("true"),
                        "fit": row.get("fit"),
                        "abs_error": row.get("abs_error"),
                        "rel_error": row.get("rel_error"),
                        "rel_error_percent": row.get("rel_error_percent"),
                    }
                )

    return path


def write_trial_summary_csv(
    path: Path,
    trials: Sequence[RecoveryTrialResult],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "trial_index",
        "status",
        "elapsed_s",
        "seed",
        "synthetic_mode",
        "fit_mode",
        "fit_success",
        "loss",
        "residual_norm",
        "n_evaluations",
        "max_abs_rel_error",
        "rms_rel_error",
        "comparison_passed",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for trial in trials:
            writer.writerow(
                {
                    "trial_index": trial.trial_index,
                    "status": trial.status.value,
                    "elapsed_s": trial.elapsed_s,
                    "seed": trial.seed,
                    "synthetic_mode": trial.synthetic_summary.get("mode"),
                    "fit_mode": trial.fit_summary.get("mode"),
                    "fit_success": trial.fit_summary.get("success"),
                    "loss": trial.fit_summary.get("loss"),
                    "residual_norm": trial.fit_summary.get("residual_norm"),
                    "n_evaluations": trial.fit_summary.get("n_evaluations"),
                    "max_abs_rel_error": trial.truth_comparison.get("max_abs_rel_error"),
                    "rms_rel_error": trial.truth_comparison.get("rms_rel_error"),
                    "comparison_passed": trial.truth_comparison.get("passed"),
                }
            )

    return path


def write_dataset_npz(
    path: Path,
    trial_payloads: Sequence[Mapping[str, Any]],
    config: SyntheticRecoveryConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metadata_json": json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "n_trials": len(trial_payloads),
                }
            )
        )
    }

    for item in trial_payloads:
        if not item:
            continue
        trial_index = int(item.get("trial_index", -1))
        dataset = item.get("dataset", {})
        fit = item.get("fit", {})

        prefix = f"trial_{trial_index}"

        for key in [
            "signal_frequency_hz",
            "signal_gain_db",
            "signal_gain_db_clean",
            "idler_conversion_db",
            "idler_conversion_db_clean",
            "s21",
            "s21_clean",
            "s21_phase_rad",
            "s21_phase_rad_clean",
        ]:
            if key in dataset and dataset[key] is not None:
                payload[f"{prefix}_{key}"] = np.asarray(dataset[key])

        if "records" in fit and fit["records"]:
            losses = [
                rec.get("loss", np.nan)
                for rec in fit["records"]
                if isinstance(rec, Mapping)
            ]
            payload[f"{prefix}_fit_loss_history"] = np.asarray(losses, dtype=float)

    np.savez_compressed(path, **payload)
    return path


def aggregate_trials(trials: Sequence[RecoveryTrialResult]) -> dict[str, Any]:
    comparisons = [t.truth_comparison for t in trials if t.truth_comparison]
    rows = []
    for t in trials:
        for row in t.truth_comparison.get("rows", []) if t.truth_comparison else []:
            rows.append({"trial_index": t.trial_index, **row})

    by_parameter: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_parameter.setdefault(str(row["name"]), []).append(row)

    parameter_summary = {}
    for name, group in by_parameter.items():
        rel = np.asarray([g.get("rel_error", np.nan) for g in group], dtype=float)
        fit = np.asarray([g.get("fit", np.nan) for g in group], dtype=float)
        truth = np.asarray([g.get("true", np.nan) for g in group], dtype=float)

        parameter_summary[name] = {
            "n": len(group),
            "true_mean": float(np.nanmean(truth)),
            "fit_mean": float(np.nanmean(fit)),
            "fit_std": float(np.nanstd(fit)),
            "rel_error_mean": float(np.nanmean(rel)),
            "rel_error_std": float(np.nanstd(rel)),
            "max_abs_rel_error": float(np.nanmax(np.abs(rel))),
        }

    max_errors = np.asarray(
        [c.get("max_abs_rel_error", np.nan) for c in comparisons],
        dtype=float,
    )
    rms_errors = np.asarray(
        [c.get("rms_rel_error", np.nan) for c in comparisons],
        dtype=float,
    )

    return {
        "n_trials": len(trials),
        "n_pass": sum(1 for t in trials if t.status == RunStatus.PASS),
        "n_partial": sum(1 for t in trials if t.status == RunStatus.PARTIAL),
        "n_error": sum(1 for t in trials if t.status in {RunStatus.FAIL, RunStatus.ERROR}),
        "comparison_pass_count": sum(1 for c in comparisons if c.get("passed")),
        "max_abs_rel_error_mean": float(np.nanmean(max_errors)) if max_errors.size else None,
        "max_abs_rel_error_max": float(np.nanmax(max_errors)) if max_errors.size else None,
        "rms_rel_error_mean": float(np.nanmean(rms_errors)) if rms_errors.size else None,
        "parameter_summary": parameter_summary,
    }


def write_plots(
    output_dir: Path,
    *,
    trials: Sequence[RecoveryTrialResult],
    trial_payloads: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
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

    # Plot first available synthetic dataset.
    first_payload = next((p for p in trial_payloads if p and "dataset" in p), None)
    if first_payload is not None:
        dataset = first_payload["dataset"]
        f_ghz = np.asarray(dataset["signal_frequency_hz"], dtype=float) / 1e9
        gain = np.asarray(dataset["signal_gain_db"], dtype=float)
        clean = dataset.get("signal_gain_db_clean")
        clean_arr = None if clean is None else np.asarray(clean, dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        if clean_arr is not None:
            ax.plot(f_ghz, clean_arr, label="clean")
        ax.scatter(f_ghz, gain, s=12, label="noisy")
        ax.set_xlabel("Signal frequency (GHz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title("Synthetic recovery dataset preview")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        p = output_dir / "synthetic_dataset_preview.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths["dataset_preview_png"] = str(p)

        fit = first_payload.get("fit", {})
        records = fit.get("records", [])
        if records:
            losses = np.asarray(
                [r.get("loss", np.nan) for r in records if isinstance(r, Mapping)],
                dtype=float,
            )
            if losses.size:
                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
                ax.semilogy(np.arange(1, losses.size + 1), losses)
                ax.set_xlabel("Evaluation")
                ax.set_ylabel("Loss")
                ax.set_title("Fit objective history")
                ax.grid(True)
                fig.tight_layout()
                p = output_dir / "synthetic_fit_history.png"
                fig.savefig(p, bbox_inches="tight")
                plt.close(fig)
                paths["fit_history_png"] = str(p)

    rows = []
    for trial in trials:
        for row in trial.truth_comparison.get("rows", []) if trial.truth_comparison else []:
            rows.append(row)

    if rows:
        names = sorted({str(row["name"]) for row in rows})
        means = []
        stds = []
        for name in names:
            values = [
                float(row["rel_error_percent"])
                for row in rows
                if str(row["name"]) == name and row.get("rel_error_percent") is not None
            ]
            means.append(float(np.nanmean(values)) if values else np.nan)
            stds.append(float(np.nanstd(values)) if values else np.nan)

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Relative error (%)")
        ax.set_title("Synthetic recovery parameter errors")
        ax.grid(True, axis="y")
        fig.tight_layout()
        p = output_dir / "synthetic_recovery_parameter_errors.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths["parameter_errors_png"] = str(p)

    return paths


def export_artifacts(
    *,
    config: SyntheticRecoveryConfig,
    stages: Sequence[StageResult],
    trials: Sequence[RecoveryTrialResult],
    trial_payloads: Sequence[Mapping[str, Any]],
    output_dir: Path,
    elapsed_s: float,
    metadata: Mapping[str, Any],
) -> SyntheticRecoveryResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    aggregate = aggregate_trials(trials)

    aggregate_json = output_dir / "synthetic_recovery_aggregate.json"
    aggregate_json.write_text(json.dumps(jsonify(aggregate), indent=2), encoding="utf-8")
    artifacts["aggregate_json"] = str(aggregate_json)

    trials_json = output_dir / "synthetic_recovery_trials.json"
    trials_json.write_text(
        json.dumps(jsonify([t.to_dict() for t in trials]), indent=2),
        encoding="utf-8",
    )
    artifacts["trials_json"] = str(trials_json)

    if config.export_csv:
        trial_summary_csv = write_trial_summary_csv(
            output_dir / "synthetic_recovery_trial_summary.csv",
            trials,
        )
        trial_params_csv = write_trial_parameters_csv(
            output_dir / "synthetic_recovery_parameters.csv",
            trials,
        )
        artifacts["trial_summary_csv"] = str(trial_summary_csv)
        artifacts["parameters_csv"] = str(trial_params_csv)

    datasets_npz = write_dataset_npz(
        output_dir / "synthetic_recovery_datasets.npz",
        trial_payloads,
        config,
    )
    artifacts["datasets_npz"] = str(datasets_npz)

    if config.make_plots:
        artifacts.update(
            write_plots(
                output_dir,
                trials=trials,
                trial_payloads=trial_payloads,
                aggregate=aggregate,
            )
        )

    if config.save_checkpoint:
        try:
            from twpa.io.checkpoints import (
                CheckpointKind,
                CheckpointMetadata,
                save_checkpoint,
            )

            checkpoint_arrays: dict[str, Any] = {}
            first_payload = next((p for p in trial_payloads if p and "dataset" in p), None)
            if first_payload is not None:
                dataset = first_payload["dataset"]
                for key in [
                    "signal_frequency_hz",
                    "signal_gain_db",
                    "signal_gain_db_clean",
                    "idler_conversion_db",
                    "s21",
                ]:
                    if key in dataset and dataset[key] is not None:
                        checkpoint_arrays[key] = dataset[key]

            checkpoint_path = output_dir / "synthetic_recovery_checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.CALIBRATION,
                    name="synthetic_recovery",
                    source="scripts.synthetic_recovery",
                    extra={
                        "config": config.to_dict(),
                        "aggregate": aggregate,
                    },
                ),
                arrays=checkpoint_arrays,
                payload={
                    "trials": [t.to_dict() for t in trials],
                    "aggregate": aggregate,
                },
            )
            artifacts["checkpoint_npz"] = str(checkpoint_path)

        except Exception as exc:
            error_path = output_dir / "checkpoint_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            artifacts["checkpoint_error_txt"] = str(error_path)

    hard_fail = any(t.status in {RunStatus.FAIL, RunStatus.ERROR} for t in trials)
    partial = any(t.status == RunStatus.PARTIAL for t in trials)

    if hard_fail:
        status = RunStatus.ERROR
    elif partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.PASS

    summary_json = output_dir / "synthetic_recovery_summary.json"
    summary_md = output_dir / "synthetic_recovery_summary.md"

    artifacts["summary_json"] = str(summary_json)
    artifacts["summary_md"] = str(summary_md)

    result = SyntheticRecoveryResult(
        config=config,
        status=status,
        elapsed_s=elapsed_s,
        stages=tuple(stages),
        trials=tuple(trials),
        artifact_paths=artifacts,
        metadata={**dict(metadata), "aggregate": aggregate},
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    summary_md.write_text(result_markdown(result, aggregate), encoding="utf-8")

    return result


def result_markdown(
    result: SyntheticRecoveryResult,
    aggregate: Mapping[str, Any],
) -> str:
    cfg = result.config

    lines = [
        "# Synthetic recovery",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- trials: `{result.n_trials}`",
        f"- pass/partial/error: `{result.n_pass}/{result.n_partial}/{result.n_error}`",
        f"- synthetic mode: `{cfg.synthetic_mode.value}`",
        f"- fit mode: `{cfg.fit_mode.value}`",
        f"- noise std: `{cfg.noise_std_db}` dB",
        f"- active parameters: `{list(active_parameter_names(cfg))}`",
        f"- comparison pass count: `{aggregate.get('comparison_pass_count')}`",
        f"- mean max relative error: `{aggregate.get('max_abs_rel_error_mean')}`",
        f"- worst max relative error: `{aggregate.get('max_abs_rel_error_max')}`",
        "",
        "## Truth parameters",
        "",
        "| parameter | true | initial | lower | upper | fit? |",
        "|---|---:|---:|---:|---:|---|",
    ]

    truth = truth_parameters(cfg)
    init = initial_parameters(cfg)
    bounds = parameter_bounds(cfg)
    active = set(active_parameter_names(cfg))

    for name in ["l_scale", "c_scale", "i_star_scale", "loss_scale", "pump_scale"]:
        lo, hi = bounds[name]
        lines.append(
            f"| `{name}` | {truth[name]} | {init[name]} | {lo} | {hi} | `{name in active}` |"
        )

    lines += [
        "",
        "## Trial summary",
        "",
        "| trial | status | synthetic mode | fit mode | loss | max rel error | RMS rel error | elapsed s |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]

    for trial in result.trials:
        lines.append(
            f"| {trial.trial_index} | `{trial.status.value}` | "
            f"`{trial.synthetic_summary.get('mode', '')}` | "
            f"`{trial.fit_summary.get('mode', '')}` | "
            f"{trial.fit_summary.get('loss', '')} | "
            f"{trial.truth_comparison.get('max_abs_rel_error', '')} | "
            f"{trial.truth_comparison.get('rms_rel_error', '')} | "
            f"{trial.elapsed_s:.6g} |"
        )

    lines += [
        "",
        "## Parameter aggregate",
        "",
        "| parameter | fit mean | fit std | mean rel error | std rel error | max abs rel error |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, row in aggregate.get("parameter_summary", {}).items():
        lines.append(
            f"| `{name}` | {row.get('fit_mean')} | {row.get('fit_std')} | "
            f"{row.get('rel_error_mean')} | {row.get('rel_error_std')} | "
            f"{row.get('max_abs_rel_error')} |"
        )

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

    failed = [t for t in result.trials if t.status in {RunStatus.FAIL, RunStatus.ERROR}]
    if failed:
        lines += [
            "",
            "## Failed/error trials",
            "",
        ]
        for trial in failed:
            lines += [
                f"### Trial {trial.trial_index}",
                "",
                *[f"- {m}" for m in trial.messages[:5]],
                "",
            ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TWPA synthetic parameter-recovery experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_recovery"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="synthetic_recovery")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument(
        "--synthetic-mode",
        choices=[m.value for m in SyntheticMode],
        default=SyntheticMode.PACKAGE.value,
    )
    parser.add_argument(
        "--fit-mode",
        choices=[m.value for m in FitMode],
        default=FitMode.PACKAGE.value,
    )
    parser.add_argument("--fail-on-package-fallback", action="store_true")

    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--noise-std-db", type=float, default=0.10)
    parser.add_argument("--noise-std-phase-rad", type=float, default=0.01)

    parser.add_argument("--signal-f-min-ghz", type=float, default=4.0)
    parser.add_argument("--signal-f-max-ghz", type=float, default=8.0)
    parser.add_argument("--n-signal", type=int, default=81)
    parser.add_argument("--pump-frequency-ghz", type=float, default=10.0)
    parser.add_argument("--pump-current-ratio", type=float, default=0.08)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)

    parser.add_argument("--true-l-scale", type=float, default=1.025)
    parser.add_argument("--true-c-scale", type=float, default=0.985)
    parser.add_argument("--true-i-star-scale", type=float, default=0.92)
    parser.add_argument("--true-loss-scale", type=float, default=1.15)
    parser.add_argument("--true-pump-scale", type=float, default=1.04)

    parser.add_argument("--fit-l-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-c-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-i-star-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-loss-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-pump-scale", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lower-l-scale", type=float, default=0.90)
    parser.add_argument("--upper-l-scale", type=float, default=1.10)
    parser.add_argument("--lower-c-scale", type=float, default=0.90)
    parser.add_argument("--upper-c-scale", type=float, default=1.10)
    parser.add_argument("--lower-i-star-scale", type=float, default=0.70)
    parser.add_argument("--upper-i-star-scale", type=float, default=1.30)
    parser.add_argument("--lower-loss-scale", type=float, default=0.50)
    parser.add_argument("--upper-loss-scale", type=float, default=2.00)
    parser.add_argument("--lower-pump-scale", type=float, default=0.70)
    parser.add_argument("--upper-pump-scale", type=float, default=1.30)

    parser.add_argument("--initial-l-scale", type=float, default=1.0)
    parser.add_argument("--initial-c-scale", type=float, default=1.0)
    parser.add_argument("--initial-i-star-scale", type=float, default=1.0)
    parser.add_argument("--initial-loss-scale", type=float, default=1.0)
    parser.add_argument("--initial-pump-scale", type=float, default=1.0)

    parser.add_argument("--fit-max-evals", type=int, default=500)
    parser.add_argument("--fit-random-samples", type=int, default=2000)
    parser.add_argument(
        "--fit-robust-loss",
        type=str,
        default="soft_l1",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
    )
    parser.add_argument("--fit-f-scale", type=float, default=1.0)

    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-csv", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> SyntheticRecoveryConfig:
    n_trials = int(args.n_trials)
    n_signal = int(args.n_signal)
    fit_max_evals = int(args.fit_max_evals)
    fit_random_samples = int(args.fit_random_samples)

    if args.quick:
        n_trials = min(n_trials, 2)
        n_signal = min(n_signal, 31)
        fit_max_evals = min(fit_max_evals, 150)
        fit_random_samples = min(fit_random_samples, 300)

    if n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.noise_std_db < 0.0:
        raise ValueError("--noise-std-db must be non-negative")
    if args.noise_std_phase_rad < 0.0:
        raise ValueError("--noise-std-phase-rad must be non-negative")
    if args.signal_f_min_ghz <= 0.0:
        raise ValueError("--signal-f-min-ghz must be positive")
    if args.signal_f_max_ghz <= args.signal_f_min_ghz:
        raise ValueError("--signal-f-max-ghz must exceed --signal-f-min-ghz")
    if n_signal < 2:
        raise ValueError("--n-signal must be at least 2")
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

    bounds = {
        "l_scale": (args.lower_l_scale, args.upper_l_scale, args.initial_l_scale, args.true_l_scale),
        "c_scale": (args.lower_c_scale, args.upper_c_scale, args.initial_c_scale, args.true_c_scale),
        "i_star_scale": (args.lower_i_star_scale, args.upper_i_star_scale, args.initial_i_star_scale, args.true_i_star_scale),
        "loss_scale": (args.lower_loss_scale, args.upper_loss_scale, args.initial_loss_scale, args.true_loss_scale),
        "pump_scale": (args.lower_pump_scale, args.upper_pump_scale, args.initial_pump_scale, args.true_pump_scale),
    }

    for name, (lo, hi, init, truth) in bounds.items():
        if lo >= hi:
            raise ValueError(f"{name}: lower bound must be smaller than upper bound")
        if not (lo <= init <= hi):
            raise ValueError(f"{name}: initial value must lie inside bounds")
        if not (lo <= truth <= hi):
            raise ValueError(f"{name}: true value must lie inside bounds")

    if not any(
        [
            args.fit_l_scale,
            args.fit_c_scale,
            args.fit_i_star_scale,
            args.fit_loss_scale,
            args.fit_pump_scale,
        ]
    ):
        raise ValueError("At least one --fit-* flag must be enabled")

    return SyntheticRecoveryConfig(
        output_dir=str(args.output_dir),
        name=str(args.name),
        seed=int(args.seed),
        quick=bool(args.quick),
        synthetic_mode=SyntheticMode(args.synthetic_mode),
        fit_mode=FitMode(args.fit_mode),
        fail_on_package_fallback=bool(args.fail_on_package_fallback),
        n_trials=n_trials,
        noise_std_db=float(args.noise_std_db),
        noise_std_phase_rad=float(args.noise_std_phase_rad),
        signal_f_min_ghz=float(args.signal_f_min_ghz),
        signal_f_max_ghz=float(args.signal_f_max_ghz),
        n_signal=n_signal,
        pump_frequency_ghz=float(args.pump_frequency_ghz),
        pump_current_ratio=float(args.pump_current_ratio),
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        true_l_scale=float(args.true_l_scale),
        true_c_scale=float(args.true_c_scale),
        true_i_star_scale=float(args.true_i_star_scale),
        true_loss_scale=float(args.true_loss_scale),
        true_pump_scale=float(args.true_pump_scale),
        fit_l_scale=bool(args.fit_l_scale),
        fit_c_scale=bool(args.fit_c_scale),
        fit_i_star_scale=bool(args.fit_i_star_scale),
        fit_loss_scale=bool(args.fit_loss_scale),
        fit_pump_scale=bool(args.fit_pump_scale),
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
        initial_l_scale=float(args.initial_l_scale),
        initial_c_scale=float(args.initial_c_scale),
        initial_i_star_scale=float(args.initial_i_star_scale),
        initial_loss_scale=float(args.initial_loss_scale),
        initial_pump_scale=float(args.initial_pump_scale),
        fit_max_evals=fit_max_evals,
        fit_random_samples=fit_random_samples,
        fit_robust_loss=str(args.fit_robust_loss),
        fit_f_scale=float(args.fit_f_scale),
        make_plots=not bool(args.no_plots),
        save_checkpoint=not bool(args.no_checkpoint),
        export_csv=not bool(args.no_csv),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[synthetic-recovery] invalid arguments: {exc}", file=sys.stderr)
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
        "scipy_available": SCIPY_AVAILABLE,
        "script": "scripts/synthetic_recovery.py",
    }

    stages: list[StageResult] = []

    stages.append(
        run_stage(
            "configuration",
            lambda: {
                "status": RunStatus.PASS.value,
                "truth": truth_parameters(config),
                "initial": initial_parameters(config),
                "bounds": parameter_bounds(config),
                "active_parameter_names": active_parameter_names(config),
                "messages": ("PASS: synthetic recovery configuration validated.",),
            },
        )
    )

    trials: list[RecoveryTrialResult] = []
    trial_payloads: list[Mapping[str, Any]] = []

    print("[synthetic-recovery] running trials...")
    trial_start = time.perf_counter()

    for trial_index in range(config.n_trials):
        print(f"[synthetic-recovery] trial {trial_index}...")
        trial, payload = run_trial(trial_index=trial_index, config=config)
        trials.append(trial)

        if payload:
            trial_payloads.append({"trial_index": trial_index, **dict(payload)})
        else:
            trial_payloads.append({})

        print(
            f"[synthetic-recovery] trial {trial_index}: "
            f"{trial.status.value}, "
            f"max_rel_error={trial.truth_comparison.get('max_abs_rel_error', 'NA')}"
        )

    stages.append(
        StageResult(
            name="trials",
            status=RunStatus.PASS
            if all(t.status in {RunStatus.PASS, RunStatus.PARTIAL} for t in trials)
            else RunStatus.PARTIAL,
            elapsed_s=time.perf_counter() - trial_start,
            summary={
                "n_trials": len(trials),
                "n_pass": sum(1 for t in trials if t.status == RunStatus.PASS),
                "n_partial": sum(1 for t in trials if t.status == RunStatus.PARTIAL),
                "n_error": sum(1 for t in trials if t.status == RunStatus.ERROR),
            },
            messages=("PASS: trial loop completed.",),
        )
    )

    elapsed_s = time.perf_counter() - start

    result = export_artifacts(
        config=config,
        stages=stages,
        trials=trials,
        trial_payloads=trial_payloads,
        output_dir=output_dir,
        elapsed_s=elapsed_s,
        metadata=metadata,
    )

    print()
    print(f"[synthetic-recovery] status: {result.status.value}")
    print(f"[synthetic-recovery] trials: {result.n_trials}")
    print(f"[synthetic-recovery] pass/partial/error: {result.n_pass}/{result.n_partial}/{result.n_error}")
    print(f"[synthetic-recovery] summary JSON: {result.artifact_paths.get('summary_json')}")
    print(f"[synthetic-recovery] summary MD:   {result.artifact_paths.get('summary_md')}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())
