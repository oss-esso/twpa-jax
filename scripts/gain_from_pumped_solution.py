"""
Compute small-signal gain from a pumped TWPA solution.

This script consumes a pump-HB solution, typically produced by:

    scripts/pump_hb_small_ladder.py

and computes signal/idler gain over a signal-frequency sweep.

It tries the package-native gain solver first. If that solver is unavailable or
incompatible with the loaded checkpoint/NPZ representation, it falls back to a
transparent coupled-mode gain estimate. The fallback is marked PARTIAL because
it is not a full linearized HB solve.

Examples
--------
From a pump-HB NPZ:

    python scripts/gain_from_pumped_solution.py ^
      --pump-npz outputs/pump_hb_small_ladder/pump_hb_small_ladder_arrays.npz ^
      --signal-f-min-ghz 4 ^
      --signal-f-max-ghz 8 ^
      --n-signal 81 ^
      --output-dir outputs/gain_from_pump

With a layout component CSV:

    python scripts/gain_from_pumped_solution.py ^
      --pump-npz outputs/pump_hb_small_ladder/pump_hb_small_ladder_arrays.npz ^
      --layout-csv outputs/pump_hb_small_ladder/pump_hb_small_ladder_components.csv ^
      --pump-frequency-ghz 10 ^
      --i-star-a 5e-3 ^
      --output-dir outputs/gain_from_pump_csv
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
from types import SimpleNamespace
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


class GainSolverMode(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    FALLBACK_COUPLED_MODE = "fallback_coupled_mode"


@dataclass(frozen=True)
class GainFromPumpConfig:
    pump_npz: str
    layout_csv: str | None
    output_dir: str

    pump_frequency_ghz: float | None
    signal_f_min_ghz: float
    signal_f_max_ghz: float
    n_signal: int

    z0_ohm: float
    length_mm: float
    phase_velocity_m_per_s: float
    i_star_a: float
    nonlinear_beta: float
    signal_current_rms_a: float

    input_node: int
    output_node: int | None
    solver_mode: GainSolverMode
    require_package_solver: bool

    make_plots: bool
    save_checkpoint: bool
    export_csv: bool
    name: str

    @property
    def length_m(self) -> float:
        return self.length_mm * 1e-3

    @property
    def signal_frequency_hz(self) -> jax.Array:
        return jnp.linspace(
            self.signal_f_min_ghz * 1e9,
            self.signal_f_max_ghz * 1e9,
            self.n_signal,
            dtype=jnp.float64,
        )

    @property
    def pump_frequency_hz(self) -> float | None:
        if self.pump_frequency_ghz is None:
            return None
        return self.pump_frequency_ghz * 1e9

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["solver_mode"] = self.solver_mode.value
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
class GainFromPumpResult:
    config: GainFromPumpConfig
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


def load_metadata_json_from_npz(npz: Any) -> dict[str, Any]:
    if "metadata_json" not in npz:
        return {}
    try:
        raw = npz["metadata_json"]
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))
    except Exception as exc:
        return {"metadata_parse_error": f"{type(exc).__name__}: {exc}"}


def nested_get(mapping: Mapping[str, Any], path: Sequence[Any], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if isinstance(cur, Mapping):
            cur = cur.get(key, default)
        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)) and isinstance(key, int):
            if 0 <= key < len(cur):
                cur = cur[key]
            else:
                return default
        else:
            return default
    return cur


def infer_pump_frequency_hz(metadata: Mapping[str, Any], config: GainFromPumpConfig) -> float:
    if config.pump_frequency_hz is not None:
        return float(config.pump_frequency_hz)

    candidates = [
        nested_get(metadata, ["config", "pump_frequency_ghz"]),
        nested_get(metadata, ["config", "pump_frequency_hz"]),
        nested_get(metadata, ["pump_drive", "pump_frequency_hz"]),
        nested_get(metadata, ["pump_drive", "frequency_hz"]),
        nested_get(metadata, ["frequency_plan", "metadata", "pump_frequency_hz"]),
    ]

    for value in candidates:
        if value is None:
            continue
        value = float(value)
        if value < 1e7:
            return value * 1e9
        return value

    raise ValueError(
        "Could not infer pump frequency. Pass --pump-frequency-ghz explicitly."
    )


def infer_i_star_a(metadata: Mapping[str, Any], config: GainFromPumpConfig) -> float:
    candidates = [
        config.i_star_a,
        nested_get(metadata, ["config", "i_star_a"]),
        nested_get(metadata, ["nonlinear_params", "I_star_A"]),
        nested_get(metadata, ["pump_drive", "metadata", "I_star_A"]),
    ]
    for value in candidates:
        if value is not None and float(value) > 0.0:
            return float(value)
    return float(config.i_star_a)


def load_pump_solution_npz(path: str | Path, config: GainFromPumpConfig) -> dict[str, Any]:
    p = Path(path)
    npz = np.load(p, allow_pickle=True)
    metadata = load_metadata_json_from_npz(npz)

    required_any = ["branch_current_coeffs_A", "node_voltage_coeffs_V"]
    if not any(key in npz for key in required_any):
        raise ValueError(
            f"{p}: expected at least one of {required_any} in pump solution NPZ."
        )

    arrays: dict[str, Any] = {}

    for key in [
        "frequencies_hz",
        "node_voltage_coeffs_V",
        "branch_current_coeffs_A",
        "residual_norm",
        "z_nodes_m",
        "z_branches_m",
    ]:
        if key in npz:
            arrays[key] = jnp.asarray(npz[key])

    pump_frequency_hz = infer_pump_frequency_hz(metadata, config)
    i_star_a = infer_i_star_a(metadata, config)

    if "frequencies_hz" not in arrays:
        # Minimal assumption: the pump solution contains only the fundamental.
        arrays["frequencies_hz"] = jnp.asarray([pump_frequency_hz], dtype=jnp.float64)

    branch_current = arrays.get("branch_current_coeffs_A")
    node_voltage = arrays.get("node_voltage_coeffs_V")

    if branch_current is not None:
        branch_current = jnp.asarray(branch_current, dtype=jnp.complex128)
        arrays["branch_current_coeffs_A"] = branch_current
    if node_voltage is not None:
        node_voltage = jnp.asarray(node_voltage, dtype=jnp.complex128)
        arrays["node_voltage_coeffs_V"] = node_voltage

    return {
        "path": str(p),
        "arrays": arrays,
        "metadata": metadata,
        "pump_frequency_hz": pump_frequency_hz,
        "i_star_a": i_star_a,
    }


def layout_summary(layout: Any) -> dict[str, Any]:
    if layout is None:
        return {}
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


def load_layout_from_csv(config: GainFromPumpConfig) -> Any | None:
    if config.layout_csv is None:
        return None

    from twpa.io.netlist import load_layout_component_csv

    return load_layout_component_csv(
        config.layout_csv,
        z0_ohm=config.z0_ohm,
        name=config.name,
        metadata={
            "source": "scripts.gain_from_pumped_solution",
            "layout_csv": config.layout_csv,
        },
    )


def infer_length_m(
    *,
    layout: Any | None,
    pump_solution: Mapping[str, Any],
    config: GainFromPumpConfig,
) -> float:
    if layout is not None:
        length = get_attr_any(layout, "length_m", default=None)
        if length is not None:
            total = float(jnp.sum(jnp.asarray(length)))
            if total > 0.0:
                return total

    arrays = pump_solution["arrays"]
    z_nodes = arrays.get("z_nodes_m")
    z_branches = arrays.get("z_branches_m")

    if z_nodes is not None:
        z = np.asarray(z_nodes, dtype=float)
        if z.size >= 2:
            return float(np.nanmax(z) - np.nanmin(z))

    if z_branches is not None:
        z = np.asarray(z_branches, dtype=float)
        if z.size >= 2:
            dz = float(np.nanmedian(np.diff(np.sort(z))))
            return float(np.nanmax(z) - np.nanmin(z) + dz)

    if config.length_m <= 0.0:
        raise ValueError("Could not infer length; pass --length-mm.")
    return config.length_m


def infer_n_cells(
    *,
    layout: Any | None,
    pump_solution: Mapping[str, Any],
) -> int:
    if layout is not None:
        n = get_attr_any(layout, "n_cells", default=None)
        if n is not None:
            return int(n)

    arrays = pump_solution["arrays"]
    branch = arrays.get("branch_current_coeffs_A")
    if branch is not None:
        return int(jnp.asarray(branch).shape[-1])

    node = arrays.get("node_voltage_coeffs_V")
    if node is not None:
        return int(jnp.asarray(node).shape[-1] - 1)

    raise ValueError("Could not infer n_cells.")


def make_minimal_layout_if_missing(
    *,
    layout: Any | None,
    pump_solution: Mapping[str, Any],
    config: GainFromPumpConfig,
) -> Any | None:
    if layout is not None:
        return layout

    try:
        from twpa.core.layout import make_layout_from_arrays

        n = infer_n_cells(layout=None, pump_solution=pump_solution)
        length_m = infer_length_m(layout=None, pump_solution=pump_solution, config=config)
        dx = length_m / n
        z0 = config.z0_ohm
        vp = config.phase_velocity_m_per_s

        L_cell = z0 * dx / vp
        C_cell = dx / (z0 * vp)

        return call_with_supported_kwargs(
            make_layout_from_arrays,
            {
                "length_m": jnp.full((n,), dx, dtype=jnp.float64),
                "L_series_H": jnp.full((n,), L_cell, dtype=jnp.float64),
                "C_shunt_F": jnp.full((n,), C_cell, dtype=jnp.float64),
                "R_series_ohm": jnp.zeros((n,), dtype=jnp.float64),
                "G_shunt_S": jnp.zeros((n,), dtype=jnp.float64),
                "C_stub_F": jnp.zeros((n,), dtype=jnp.float64),
                "L_res_H": jnp.zeros((n,), dtype=jnp.float64),
                "C_res_F": jnp.zeros((n,), dtype=jnp.float64),
                "C_couple_F": jnp.zeros((n,), dtype=jnp.float64),
                "z0_ohm": z0,
                "name": f"{config.name}_minimal_layout",
                "metadata": {
                    "source": "scripts.gain_from_pumped_solution.make_minimal_layout_if_missing",
                    "reason": "layout_csv not provided",
                    "length_m": length_m,
                    "n_cells": n,
                },
            },
        )
    except Exception:
        return None


def frequency_plan_like(
    *,
    pump_solution: Mapping[str, Any],
    pump_frequency_hz: float,
) -> Any:
    freqs = jnp.asarray(pump_solution["arrays"].get("frequencies_hz"), dtype=jnp.float64)

    try:
        from twpa.core.frequency_plan import make_pump_only_plan

        ratios = np.asarray(freqs, dtype=float) / pump_frequency_hz
        return make_pump_only_plan(
            pump_frequency_hz,
            n_harmonics=max(abs(int(round(value))) for value in ratios),
            include_negative=bool(np.any(ratios < 0.0)),
            include_dc=bool(np.any(np.isclose(ratios, 0.0))),
            sort="frequency",
        )
    except Exception:
        labels = []
        for f in np.asarray(freqs, dtype=float):
            ratio = f / pump_frequency_hz
            nearest = int(round(ratio))
            labels.append("pump" if nearest == 1 else f"{nearest:+d}pump")
        return SimpleNamespace(
            frequencies_hz=freqs,
            labels=tuple(labels),
            n_tones=int(freqs.shape[0]),
            reference_frequency_hz=pump_frequency_hz,
            reference_pump_hz=pump_frequency_hz,
            name="loaded_pump_frequency_plan",
            metadata={
                "source": "fallback_frequency_plan_like",
            },
            position_of_label=lambda label: tuple(labels).index(label),
            to_dict=lambda: {
                "frequencies_hz": np.asarray(freqs).tolist(),
                "labels": list(labels),
                "reference_frequency_hz": pump_frequency_hz,
            },
        )


def build_pump_result_like(
    *,
    pump_solution: Mapping[str, Any],
    layout: Any | None,
    config: GainFromPumpConfig,
) -> Any:
    pump_frequency_hz = float(pump_solution["pump_frequency_hz"])
    plan = frequency_plan_like(
        pump_solution=pump_solution,
        pump_frequency_hz=pump_frequency_hz,
    )

    arrays = pump_solution["arrays"]
    if layout is None:
        raise ValueError("A layout is required to reconstruct a pump result for native gain.")

    from twpa.core.params import NonlinearParams
    from twpa.nonlinear.distributed_hb import (
        DistributedHBConfig,
        DistributedHBState,
        make_kinetic_model_from_layout,
    )
    from twpa.nonlinear.pump_hb_ladder import (
        PumpDriveConfig,
        make_pump_injection,
    )

    state = DistributedHBState(
        node_voltage_coeffs_V=arrays.get("node_voltage_coeffs_V"),
        branch_current_coeffs_A=arrays.get("branch_current_coeffs_A"),
    )

    metadata = pump_solution.get("metadata", {})
    drive_current_rms_a = float(
        nested_get(metadata, ["pump_drive", "current_rms_A"], config.i_star_a * 0.08)
    )
    drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=pump_frequency_hz,
        current_rms_A=drive_current_rms_a,
        source_impedance_ohm=config.z0_ohm,
        pump_label="pump",
        input_node=config.input_node,
        phase_rad=0.0,
    )

    nonlinear_params = NonlinearParams(
        I_star_A=float(pump_solution["i_star_a"]),
        beta_nl=float(config.nonlinear_beta),
    )
    hb_config = DistributedHBConfig(
        **dict(nested_get(metadata, ["pump_config", "distributed"], {}))
    )
    ki_model = make_kinetic_model_from_layout(layout, nonlinear_params)
    injected_current_coeffs_A = make_pump_injection(plan, layout, drive, hb_config)
    residual_norm = np.asarray(arrays.get("residual_norm", np.nan))
    converged = bool(residual_norm.size and np.all(np.isfinite(residual_norm)))

    return SimpleNamespace(
        state=state,
        frequency_plan=plan,
        plan=plan,
        layout=layout,
        hb_config=hb_config,
        ki_model=ki_model,
        injected_current_coeffs_A=injected_current_coeffs_A,
        converged=converged,
        drive=drive,
        pump_drive=drive,
        nonlinear_params=nonlinear_params,
        metadata={
            "source": "scripts.gain_from_pumped_solution.build_pump_result_like",
            "loaded_from": pump_solution["path"],
        },
        to_dict=lambda: {
            "kind": "pump_result_like",
            "frequency_plan": jsonify(plan),
            "layout": layout_summary(layout),
            "drive": jsonify(drive),
            "nonlinear_params": jsonify(nonlinear_params),
            "hb_config": jsonify(hb_config),
            "converged": converged,
            "state": {
                "node_voltage_coeffs_V": array_summary(state.node_voltage_coeffs_V)
                if state.node_voltage_coeffs_V is not None
                else None,
                "branch_current_coeffs_A": array_summary(state.branch_current_coeffs_A)
                if state.branch_current_coeffs_A is not None
                else None,
            },
        },
    )


def build_signal_labels(n: int) -> tuple[str, ...]:
    return tuple(f"signal_{i}" for i in range(n))


def build_idler_labels(n: int) -> tuple[str, ...]:
    return tuple(f"idler_{i}" for i in range(n))


def try_package_gain_solver(
    *,
    pump_result_like: Any,
    signal_frequency_hz: jax.Array,
    config: GainFromPumpConfig,
) -> dict[str, Any]:
    pump_frequency_hz = float(get_attr_any(pump_result_like.drive, "pump_frequency_hz"))
    idler_frequency_hz = 2.0 * pump_frequency_hz - signal_frequency_hz

    signal_labels = build_signal_labels(int(signal_frequency_hz.shape[0]))
    idler_labels = build_idler_labels(int(signal_frequency_hz.shape[0]))

    from twpa.inference.synthetic import (
        make_gain_frequency_plan,
        make_gain_sweep_config_for_frequencies,
    )

    target_plan = make_gain_frequency_plan(
        pump_frequency_hz=pump_frequency_hz,
        signal_frequency_hz=signal_frequency_hz,
        idler_frequency_hz=idler_frequency_hz,
        pump_label="pump",
        signal_labels=signal_labels,
        idler_labels=idler_labels,
        n_pump_harmonics=3,
        include_negative=True,
        include_dc=False,
    )

    sweep_config = make_gain_sweep_config_for_frequencies(
        signal_labels=signal_labels,
        idler_labels=idler_labels,
        input_node=config.input_node,
        output_node=config.output_node,
        signal_current_rms_A=config.signal_current_rms_a + 0j,
        input_impedance_ohm=config.z0_ohm,
        output_impedance_ohm=config.z0_ohm,
        name="gain_from_loaded_pump_sweep",
    )

    import twpa.nonlinear.gain as gain_module

    candidate_names = [
        "solve_gain_sweep_from_pump",
        "run_gain_sweep_from_pump",
        "solve_small_signal_gain_from_pump",
        "gain_sweep_from_pump",
    ]

    last_errors: list[str] = []

    for name in candidate_names:
        fn = getattr(gain_module, name, None)
        if fn is None:
            continue

        kwargs = {
            "pump_result": pump_result_like,
            "pumped_result": pump_result_like,
            "pump_solution": pump_result_like,
            "target_plan": target_plan,
            "frequency_plan": target_plan,
            "plan": target_plan,
            "sweep_config": sweep_config,
            "config": sweep_config,
        }

        try:
            result = call_with_supported_kwargs(fn, kwargs)
            parsed = parse_package_gain_result(
                result,
                signal_frequency_hz=signal_frequency_hz,
                idler_frequency_hz=idler_frequency_hz,
            )
            parsed.update(
                {
                    "status": RunStatus.PASS.value,
                    "solver_function": name,
                    "target_plan": jsonify(target_plan),
                    "sweep_config": jsonify(sweep_config),
                    "package_result": jsonify(result),
                    "messages": (
                        f"PASS: package-native gain solver `{name}` completed.",
                    ),
                }
            )
            return parsed
        except Exception as exc:
            last_errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No compatible package-native gain solver succeeded. Errors:\n"
        + "\n".join(last_errors[-10:])
    )


def parse_package_gain_result(
    result: Any,
    *,
    signal_frequency_hz: jax.Array,
    idler_frequency_hz: jax.Array,
) -> dict[str, Any]:
    points = get_attr_any(result, "points", default=None)

    if points is not None:
        signal_gain_db = []
        idler_conversion_db = []
        converged = []

        for p in list(points):
            signal_gain_db.append(float(get_attr_any(p, "signal_gain_db", "gain_db")))
            idler = get_attr_any(p, "idler_conversion_db", "conversion_db", default=np.nan)
            idler_conversion_db.append(np.nan if idler is None else float(idler))
            converged.append(bool(get_attr_any(p, "converged", "success", default=True)))

        return {
            "signal_frequency_hz": signal_frequency_hz,
            "idler_frequency_hz": idler_frequency_hz,
            "signal_gain_db": jnp.asarray(signal_gain_db, dtype=jnp.float64),
            "idler_conversion_db": jnp.asarray(idler_conversion_db, dtype=jnp.float64),
            "converged": jnp.asarray(converged, dtype=bool),
            "finite": bool(np.all(np.isfinite(signal_gain_db))),
        }

    signal_gain = get_attr_any(result, "signal_gain_db", "gain_db", default=None)
    if signal_gain is None:
        raise RuntimeError("Package gain result does not expose points or signal_gain_db.")

    idler = get_attr_any(result, "idler_conversion_db", "conversion_db", default=None)
    if idler is None:
        idler = jnp.full_like(signal_frequency_hz, jnp.nan)

    return {
        "signal_frequency_hz": signal_frequency_hz,
        "idler_frequency_hz": idler_frequency_hz,
        "signal_gain_db": jnp.asarray(signal_gain, dtype=jnp.float64),
        "idler_conversion_db": jnp.asarray(idler, dtype=jnp.float64),
        "converged": jnp.ones(signal_frequency_hz.shape, dtype=bool),
        "finite": bool(jnp.all(jnp.isfinite(jnp.asarray(signal_gain)))),
    }


def pump_current_profile(
    *,
    pump_solution: Mapping[str, Any],
    pump_frequency_hz: float,
) -> jax.Array:
    arrays = pump_solution["arrays"]
    branch = arrays.get("branch_current_coeffs_A")

    if branch is None:
        raise ValueError("branch_current_coeffs_A is required for gain fallback.")

    branch = jnp.asarray(branch, dtype=jnp.complex128)
    freqs = jnp.asarray(arrays.get("frequencies_hz"), dtype=jnp.float64)
    idx = int(jnp.argmin(jnp.abs(freqs - pump_frequency_hz)))

    if branch.ndim != 2:
        raise ValueError(f"Expected branch_current_coeffs_A shape (H, N), got {branch.shape}")

    return branch[idx]


def fallback_coupled_mode_gain(
    *,
    pump_solution: Mapping[str, Any],
    layout: Any | None,
    config: GainFromPumpConfig,
) -> dict[str, Any]:
    pump_frequency_hz = float(pump_solution["pump_frequency_hz"])
    signal_frequency_hz = config.signal_frequency_hz
    idler_frequency_hz = 2.0 * pump_frequency_hz - signal_frequency_hz

    length_m = infer_length_m(layout=layout, pump_solution=pump_solution, config=config)
    i_star_a = float(pump_solution["i_star_a"])

    pump_profile = pump_current_profile(
        pump_solution=pump_solution,
        pump_frequency_hz=pump_frequency_hz,
    )
    pump_rms = float(jnp.sqrt(jnp.mean(jnp.abs(pump_profile) ** 2)))
    pump_ratio = pump_rms / max(i_star_a, 1e-300)

    fs = np.asarray(signal_frequency_hz, dtype=float)
    fi = np.asarray(idler_frequency_hz, dtype=float)
    fp = pump_frequency_hz

    vp = float(config.phase_velocity_m_per_s)

    beta_s = 2.0 * np.pi * fs / vp
    beta_i = 2.0 * np.pi * fi / vp
    beta_p = 2.0 * np.pi * fp / vp

    delta_beta = beta_s + beta_i - 2.0 * beta_p

    gamma = float(config.nonlinear_beta) * beta_p / max(i_star_a**2, 1e-300)
    kappa = 0.25 * gamma * pump_rms**2

    g2 = kappa**2 - (0.5 * delta_beta) ** 2

    signal_gain_power = np.empty_like(fs)
    idler_conversion_power = np.empty_like(fs)

    for idx, value in enumerate(g2):
        if value >= 0.0:
            g = math.sqrt(value)
            gl = g * length_m
            gain_amp = math.cosh(gl)
            conv_amp = 0.0 if g == 0.0 else abs(kappa / g) * math.sinh(gl)
        else:
            q = math.sqrt(-value)
            ql = q * length_m
            gain_amp = math.sqrt(max(1.0 + (kappa / q) ** 2 * math.sin(ql) ** 2, 0.0))
            conv_amp = abs(kappa / q) * abs(math.sin(ql))

        signal_gain_power[idx] = max(gain_amp**2, 1e-300)
        idler_conversion_power[idx] = max(conv_amp**2, 1e-300)

    signal_gain_db = 10.0 * np.log10(signal_gain_power)
    idler_conversion_db = 10.0 * np.log10(idler_conversion_power)

    phase_match_db = -20.0 * np.log10(np.maximum(np.abs(delta_beta * length_m), 1e-12))

    return {
        "status": RunStatus.PARTIAL.value,
        "solver_function": "fallback_coupled_mode_gain",
        "signal_frequency_hz": jnp.asarray(signal_frequency_hz, dtype=jnp.float64),
        "idler_frequency_hz": jnp.asarray(idler_frequency_hz, dtype=jnp.float64),
        "signal_gain_db": jnp.asarray(signal_gain_db, dtype=jnp.float64),
        "idler_conversion_db": jnp.asarray(idler_conversion_db, dtype=jnp.float64),
        "delta_beta_rad_per_m": jnp.asarray(delta_beta, dtype=jnp.float64),
        "phase_match_db": jnp.asarray(phase_match_db, dtype=jnp.float64),
        "coupling_kappa_rad_per_m": float(kappa),
        "pump_rms_A": float(pump_rms),
        "pump_current_ratio_rms": float(pump_ratio),
        "length_m": float(length_m),
        "finite": bool(np.all(np.isfinite(signal_gain_db))),
        "converged": jnp.zeros(signal_frequency_hz.shape, dtype=bool),
        "messages": (
            "PARTIAL: used fallback coupled-mode estimate, not full linearized HB gain.",
        ),
    }


def run_gain_solver(
    *,
    pump_solution: Mapping[str, Any],
    layout: Any | None,
    config: GainFromPumpConfig,
) -> dict[str, Any]:
    pump_result_like = build_pump_result_like(
        pump_solution=pump_solution,
        layout=layout,
        config=config,
    )

    if config.solver_mode in {GainSolverMode.AUTO, GainSolverMode.PACKAGE}:
        try:
            return try_package_gain_solver(
                pump_result_like=pump_result_like,
                signal_frequency_hz=config.signal_frequency_hz,
                config=config,
            )
        except Exception as exc:
            if config.solver_mode == GainSolverMode.PACKAGE or config.require_package_solver:
                raise
            package_error = f"{type(exc).__name__}: {exc}"
    else:
        package_error = "package solver disabled by --solver-mode fallback_coupled_mode"

    result = fallback_coupled_mode_gain(
        pump_solution=pump_solution,
        layout=layout,
        config=config,
    )
    result["package_error"] = package_error
    result["messages"] = (
        *tuple(result.get("messages", ())),
        package_error,
    )
    return result


def gain_summary(gain_result: Mapping[str, Any]) -> dict[str, Any]:
    signal_gain = jnp.asarray(gain_result["signal_gain_db"], dtype=jnp.float64)
    idler = gain_result.get("idler_conversion_db")
    finite = bool(jnp.all(jnp.isfinite(signal_gain)))

    max_idx = int(jnp.nanargmax(signal_gain)) if signal_gain.size else -1

    summary = {
        "finite": finite,
        "solver_function": gain_result.get("solver_function"),
        "signal_frequency_hz": array_summary(gain_result["signal_frequency_hz"]),
        "idler_frequency_hz": array_summary(gain_result["idler_frequency_hz"]),
        "signal_gain_db": array_summary(signal_gain),
        "max_gain_db": float(signal_gain[max_idx]) if max_idx >= 0 else None,
        "max_gain_frequency_hz": float(jnp.asarray(gain_result["signal_frequency_hz"])[max_idx])
        if max_idx >= 0
        else None,
        "n_points": int(signal_gain.size),
    }

    if idler is not None:
        summary["idler_conversion_db"] = array_summary(idler)

    for key in [
        "coupling_kappa_rad_per_m",
        "pump_rms_A",
        "pump_current_ratio_rms",
        "length_m",
        "package_error",
    ]:
        if key in gain_result:
            summary[key] = jsonify(gain_result[key])

    return summary


def write_gain_csv(path: Path, gain_result: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = np.asarray(gain_result["signal_frequency_hz"], dtype=float)
    fi = np.asarray(gain_result["idler_frequency_hz"], dtype=float)
    gain = np.asarray(gain_result["signal_gain_db"], dtype=float)

    idler = gain_result.get("idler_conversion_db")
    idler_arr = None if idler is None else np.asarray(idler, dtype=float)

    delta_beta = gain_result.get("delta_beta_rad_per_m")
    delta_beta_arr = None if delta_beta is None else np.asarray(delta_beta, dtype=float)

    phase_match = gain_result.get("phase_match_db")
    phase_match_arr = None if phase_match is None else np.asarray(phase_match, dtype=float)

    fields = [
        "signal_frequency_hz",
        "signal_frequency_ghz",
        "idler_frequency_hz",
        "idler_frequency_ghz",
        "signal_gain_db",
    ]
    if idler_arr is not None:
        fields.append("idler_conversion_db")
    if delta_beta_arr is not None:
        fields.append("delta_beta_rad_per_m")
    if phase_match_arr is not None:
        fields.append("phase_match_db")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(fs.shape[0]):
            row = {
                "signal_frequency_hz": float(fs[i]),
                "signal_frequency_ghz": float(fs[i] / 1e9),
                "idler_frequency_hz": float(fi[i]),
                "idler_frequency_ghz": float(fi[i] / 1e9),
                "signal_gain_db": float(gain[i]),
            }
            if idler_arr is not None:
                row["idler_conversion_db"] = float(idler_arr[i])
            if delta_beta_arr is not None:
                row["delta_beta_rad_per_m"] = float(delta_beta_arr[i])
            if phase_match_arr is not None:
                row["phase_match_db"] = float(phase_match_arr[i])
            writer.writerow(row)

    return path


def export_artifacts(
    *,
    pump_solution: Mapping[str, Any],
    layout: Any | None,
    gain_result: Mapping[str, Any],
    config: GainFromPumpConfig,
    stages: list[StageResult],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    arrays_path = output_dir / "gain_from_pumped_solution_arrays.npz"

    payload: dict[str, Any] = {
        "signal_frequency_hz": np.asarray(gain_result["signal_frequency_hz"]),
        "idler_frequency_hz": np.asarray(gain_result["idler_frequency_hz"]),
        "signal_gain_db": np.asarray(gain_result["signal_gain_db"]),
        "metadata_json": json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "pump_solution_path": pump_solution["path"],
                    "pump_solution_metadata": pump_solution["metadata"],
                    "layout": layout_summary(layout),
                    "gain_summary": gain_summary(gain_result),
                }
            )
        ),
    }

    for key in [
        "idler_conversion_db",
        "delta_beta_rad_per_m",
        "phase_match_db",
        "converged",
    ]:
        if key in gain_result and gain_result[key] is not None:
            payload[key] = np.asarray(gain_result[key])

    np.savez_compressed(arrays_path, **payload)
    paths["arrays_npz"] = str(arrays_path)

    if config.export_csv:
        csv_path = write_gain_csv(output_dir / "gain_from_pumped_solution.csv", gain_result)
        paths["gain_csv"] = str(csv_path)

    summary_json = output_dir / "gain_from_pumped_solution_gain_summary.json"
    summary_json.write_text(json.dumps(jsonify(gain_summary(gain_result)), indent=2), encoding="utf-8")
    paths["gain_summary_json"] = str(summary_json)

    if config.save_checkpoint:
        try:
            from twpa.io.checkpoints import (
                CheckpointKind,
                CheckpointMetadata,
                save_checkpoint,
            )

            checkpoint_path = output_dir / "gain_from_pumped_solution_checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.GAIN_SWEEP,
                    name="gain_from_pumped_solution",
                    source="scripts.gain_from_pumped_solution",
                    extra={
                        "config": config.to_dict(),
                        "layout": layout_summary(layout),
                        "pump_solution_path": pump_solution["path"],
                    },
                ),
                arrays={k: v for k, v in payload.items() if k != "metadata_json"},
                payload={
                    "gain_result": {
                        k: v
                        for k, v in gain_result.items()
                        if k
                        not in {
                            "signal_frequency_hz",
                            "idler_frequency_hz",
                            "signal_gain_db",
                            "idler_conversion_db",
                            "delta_beta_rad_per_m",
                            "phase_match_db",
                            "converged",
                        }
                    },
                    "gain_summary": gain_summary(gain_result),
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

            from twpa.plotting.gain_maps import (
                GainMapPlotConfig,
                plot_gain_sweep,
                save_gain_figure,
            )

            fig, _ = plot_gain_sweep(
                gain_result["signal_frequency_hz"],
                gain_result["signal_gain_db"],
                idler_conversion_db=gain_result.get("idler_conversion_db"),
                config=GainMapPlotConfig(title="Gain from pumped solution"),
                show_idler=True,
            )
            p = save_gain_figure(fig, output_dir / "gain_from_pumped_solution.png")
            paths["gain_png"] = str(p)

            import matplotlib.pyplot as plt

            plt.close(fig)

        except Exception as exc:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fs = np.asarray(gain_result["signal_frequency_hz"], dtype=float) / 1e9
                gain = np.asarray(gain_result["signal_gain_db"], dtype=float)

                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
                ax.plot(fs, gain, label="signal gain")

                idler = gain_result.get("idler_conversion_db")
                if idler is not None:
                    ax.plot(fs, np.asarray(idler, dtype=float), label="idler conversion")

                ax.set_xlabel("Signal frequency (GHz)")
                ax.set_ylabel("Gain / conversion (dB)")
                ax.set_title("Gain from pumped solution")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()

                p = output_dir / "gain_from_pumped_solution.png"
                fig.savefig(p, bbox_inches="tight")
                plt.close(fig)
                paths["gain_png"] = str(p)

            except Exception:
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


def result_markdown(result: GainFromPumpResult) -> str:
    cfg = result.config

    lines = [
        "# Gain from pumped solution",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- pump NPZ: `{cfg.pump_npz}`",
        f"- signal range: `{cfg.signal_f_min_ghz:.6g}`–`{cfg.signal_f_max_ghz:.6g} GHz`",
        f"- signal points: `{cfg.n_signal}`",
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
        if stage.name in {"load_inputs", "gain_solve", "artifact_export"}:
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
        description="Compute signal/idler gain from a pumped TWPA solution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pump-npz", type=str, required=True)
    parser.add_argument("--layout-csv", type=str, default=None)

    parser.add_argument(
        "--pump-frequency-ghz",
        type=float,
        default=None,
        help="Pump frequency. If omitted, the script tries to infer it from pump NPZ metadata.",
    )
    parser.add_argument("--signal-f-min-ghz", type=float, default=4.0)
    parser.add_argument("--signal-f-max-ghz", type=float, default=8.0)
    parser.add_argument("--n-signal", type=int, default=81)

    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--length-mm", type=float, default=1.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)
    parser.add_argument("--i-star-a", type=float, default=5e-3)
    parser.add_argument("--nonlinear-beta", type=float, default=1.0)
    parser.add_argument("--signal-current-rms-a", type=float, default=1e-12)

    parser.add_argument("--input-node", type=int, default=0)
    parser.add_argument("--output-node", type=int, default=None)
    parser.add_argument(
        "--solver-mode",
        choices=[m.value for m in GainSolverMode],
        default=GainSolverMode.AUTO.value,
    )
    parser.add_argument(
        "--require-package-solver",
        action="store_true",
        help="Fail instead of using the coupled-mode fallback if package gain solve fails.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gain_from_pumped_solution"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="gain_from_pumped_solution")

    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-csv", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> GainFromPumpConfig:
    pump_npz = Path(args.pump_npz)
    if not pump_npz.exists():
        raise FileNotFoundError(pump_npz)

    if args.layout_csv is not None and not Path(args.layout_csv).exists():
        raise FileNotFoundError(args.layout_csv)

    if args.pump_frequency_ghz is not None and args.pump_frequency_ghz <= 0.0:
        raise ValueError("--pump-frequency-ghz must be positive when provided.")
    if args.signal_f_min_ghz <= 0.0:
        raise ValueError("--signal-f-min-ghz must be positive.")
    if args.signal_f_max_ghz <= args.signal_f_min_ghz:
        raise ValueError("--signal-f-max-ghz must exceed --signal-f-min-ghz.")
    if int(args.n_signal) < 2:
        raise ValueError("--n-signal must be at least 2.")
    if args.z0_ohm <= 0.0:
        raise ValueError("--z0-ohm must be positive.")
    if args.length_mm <= 0.0:
        raise ValueError("--length-mm must be positive.")
    if args.phase_velocity_m_per_s <= 0.0:
        raise ValueError("--phase-velocity-m-per-s must be positive.")
    if args.i_star_a <= 0.0:
        raise ValueError("--i-star-a must be positive.")
    if args.signal_current_rms_a <= 0.0:
        raise ValueError("--signal-current-rms-a must be positive.")

    return GainFromPumpConfig(
        pump_npz=str(pump_npz),
        layout_csv=args.layout_csv,
        output_dir=str(args.output_dir),
        pump_frequency_ghz=None if args.pump_frequency_ghz is None else float(args.pump_frequency_ghz),
        signal_f_min_ghz=float(args.signal_f_min_ghz),
        signal_f_max_ghz=float(args.signal_f_max_ghz),
        n_signal=int(args.n_signal),
        z0_ohm=float(args.z0_ohm),
        length_mm=float(args.length_mm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        i_star_a=float(args.i_star_a),
        nonlinear_beta=float(args.nonlinear_beta),
        signal_current_rms_a=float(args.signal_current_rms_a),
        input_node=int(args.input_node),
        output_node=None if args.output_node is None else int(args.output_node),
        solver_mode=GainSolverMode(args.solver_mode),
        require_package_solver=bool(args.require_package_solver),
        make_plots=not bool(args.no_plots),
        save_checkpoint=not bool(args.no_checkpoint),
        export_csv=not bool(args.no_csv),
        name=str(args.name),
    )


def _stage_load_inputs(holders: dict[str, Any], config: GainFromPumpConfig) -> dict[str, Any]:
    pump_solution = load_pump_solution_npz(config.pump_npz, config)
    layout = load_layout_from_csv(config)
    layout = make_minimal_layout_if_missing(
        layout=layout,
        pump_solution=pump_solution,
        config=config,
    )

    holders["pump_solution"] = pump_solution
    holders["layout"] = layout

    arrays = pump_solution["arrays"]

    return {
        "status": RunStatus.PASS.value,
        "pump_solution": {
            "path": pump_solution["path"],
            "pump_frequency_hz": pump_solution["pump_frequency_hz"],
            "i_star_a": pump_solution["i_star_a"],
            "arrays": {k: array_summary(v) for k, v in arrays.items()},
            "metadata": pump_solution["metadata"],
        },
        "layout": layout_summary(layout),
        "messages": ("PASS: pump solution and layout inputs loaded.",),
    }


def _stage_gain_solve(holders: dict[str, Any], config: GainFromPumpConfig) -> dict[str, Any]:
    gain_result = run_gain_solver(
        pump_solution=holders["pump_solution"],
        layout=holders["layout"],
        config=config,
    )
    holders["gain_result"] = gain_result

    finite = bool(gain_result.get("finite", False))
    status = RunStatus(gain_result.get("status", RunStatus.PASS.value))

    if not finite and status == RunStatus.PASS:
        status = RunStatus.FAIL

    return {
        "status": status.value,
        **gain_summary(gain_result),
        "messages": tuple(gain_result.get("messages", ())),
    }


def finalize_result(result: GainFromPumpResult, output_dir: Path) -> int:
    summary_json = output_dir / "gain_from_pumped_solution_summary.json"
    summary_md = output_dir / "gain_from_pumped_solution_summary.md"

    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }

    result = GainFromPumpResult(
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
    print(f"[gain-from-pump] status: {result.status.value}")
    print(f"[gain-from-pump] summary JSON: {summary_json}")
    print(f"[gain-from-pump] summary MD:   {summary_md}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[gain-from-pump] invalid arguments: {exc}", file=sys.stderr)
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
        "script": "scripts/gain_from_pumped_solution.py",
    }

    holders: dict[str, Any] = {}
    stages: list[StageResult] = []
    artifacts: dict[str, str] = {}

    print("[gain-from-pump] loading inputs...")
    stage = run_stage("load_inputs", lambda: _stage_load_inputs(holders, config))
    stages.append(stage)
    print(f"[gain-from-pump] load_inputs: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = GainFromPumpResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[gain-from-pump] solving gain...")
    stage = run_stage("gain_solve", lambda: _stage_gain_solve(holders, config))
    stages.append(stage)
    print(f"[gain-from-pump] gain_solve: {stage.status.value}")

    if stage.status == RunStatus.ERROR:
        result = GainFromPumpResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[gain-from-pump] exporting artifacts...")
    try:
        artifacts.update(
            export_artifacts(
                pump_solution=holders["pump_solution"],
                layout=holders["layout"],
                gain_result=holders["gain_result"],
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

    result = GainFromPumpResult(
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
