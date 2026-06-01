"""
Run a pump-off 100 mm linear baseline simulation.

This script builds or loads a 100 mm TWPA layout, runs a pump-off linear
S-parameter scan, extracts dispersion/stopband diagnostics when available, and
writes reproducible artifacts.

Examples
--------
Fast 100 mm uniform baseline:

    python scripts/linear_100mm_baseline.py --output-dir outputs/linear_100mm

20,000-cell industrial baseline:

    python scripts/linear_100mm_baseline.py ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --f-min-ghz 1 ^
      --f-max-ghz 14 ^
      --n-frequency 1001 ^
      --output-dir outputs/linear_100mm_20k

Use a component CSV layout:

    python scripts/linear_100mm_baseline.py ^
      --layout-csv path/to/layout_components.csv ^
      --output-dir outputs/linear_from_csv
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


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(frozen=True)
class Linear100mmConfig:
    """
    CLI-resolved configuration for the 100 mm linear baseline.
    """

    n_cells: int
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float
    f_min_ghz: float
    f_max_ghz: float
    n_frequency: int
    layout_csv: str | None
    output_dir: str
    layout_kind: str
    include_resonators: bool
    disorder_std: float
    seed: int
    export_netlist: bool
    make_plots: bool
    save_checkpoint: bool
    quick: bool
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
        return asdict(self)


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
class Linear100mmResult:
    config: Linear100mmConfig
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
    """
    Call fn with only the keyword arguments accepted by its signature.
    """
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**dict(kwargs))
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


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


def build_uniform_layout_fallback(config: Linear100mmConfig) -> Any:
    """
    Build a simple uniform LC ladder using the core layout constructor.

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
        period = max(1, n // 64)
        idx = np.arange(n)
        resonator_mask = idx % period == 0
        C_stub[resonator_mask] = 2e-15
        L_res[resonator_mask] = 1e-9
        C_res[resonator_mask] = 5e-15
        C_couple[resonator_mask] = 1e-15

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
            "source": "scripts.linear_100mm_baseline.build_uniform_layout_fallback",
            "layout_kind": config.layout_kind,
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


def load_layout_from_csv(config: Linear100mmConfig) -> Any:
    from twpa.io.netlist import load_layout_component_csv

    if config.layout_csv is None:
        raise ValueError("layout_csv is None")

    return load_layout_component_csv(
        config.layout_csv,
        z0_ohm=config.z0_ohm,
        name=config.name,
        metadata={
            "source": "scripts.linear_100mm_baseline",
            "layout_csv": config.layout_csv,
        },
    )


def build_layout(config: Linear100mmConfig) -> tuple[Any, dict[str, Any]]:
    """
    Build/load the simulation layout.
    """
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

        kind_value = config.layout_kind
        try:
            kind = SyntheticLayoutKind(kind_value)
        except Exception:
            kind = getattr(SyntheticLayoutKind, kind_value.upper(), SyntheticLayoutKind.UNIFORM)

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


def run_linear_scan_stage(layout: Any, config: Linear100mmConfig) -> tuple[Any, dict[str, Any]]:
    from twpa.linear.cascade import run_linear_scan

    frequency_hz = config.frequency_hz

    scan = run_linear_scan(frequency_hz, layout)

    s = get_attr_any(scan, "s", default=None)
    s21_db = get_attr_any(scan, "s21_db", default=None)

    if s21_db is None and s is not None:
        s_arr = jnp.asarray(s)
        s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(s_arr[:, 1, 0]), 1e-300))

    if s21_db is None:
        raise RuntimeError("Linear scan did not expose s21_db or s matrix.")

    s21_db_arr = jnp.asarray(s21_db)
    finite = bool(jnp.all(jnp.isfinite(s21_db_arr)))

    summary = {
        "frequency_hz": array_summary(frequency_hz),
        "s21_db": array_summary(s21_db_arr),
        "finite": finite,
        "scan": jsonify(scan),
    }

    if s is not None:
        summary["s"] = array_summary(s)

    return scan, summary


def run_dispersion_stage(layout: Any, config: Linear100mmConfig) -> tuple[Any | None, dict[str, Any]]:
    try:
        from twpa.linear.dispersion import extract_layout_dispersion

        frequency_hz = config.frequency_hz
        dispersion = extract_layout_dispersion(frequency_hz, layout)

        beta = get_attr_any(
            dispersion,
            "beta_preferred_rad_per_m",
            "beta_eff_rad_per_m",
            "beta_rad_per_m",
            default=None,
        )
        alpha = get_attr_any(
            dispersion,
            "alpha_preferred_np_per_m",
            "alpha_np_per_m",
            default=None,
        )

        summary = {
            "status": RunStatus.PASS.value,
            "dispersion": jsonify(dispersion),
            "beta": None if beta is None else array_summary(beta),
            "alpha": None if alpha is None else array_summary(alpha),
            "messages": ("PASS: dispersion extraction completed.",),
        }
        return dispersion, summary

    except Exception as exc:
        return None, {
            "status": RunStatus.PARTIAL.value,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "messages": (
                "PARTIAL: dispersion extraction failed or is unavailable.",
                f"{type(exc).__name__}: {exc}",
            ),
        }


def detect_stopbands_stage(scan: Any, config: Linear100mmConfig) -> tuple[Any | None, dict[str, Any]]:
    try:
        from twpa.linear.dispersion import detect_stopbands

        frequency_hz = get_attr_any(scan, "frequency_hz", default=config.frequency_hz)
        s21_db = get_attr_any(scan, "s21_db", default=None)

        if s21_db is None:
            s = get_attr_any(scan, "s", default=None)
            if s is None:
                raise RuntimeError("No s21_db or s matrix available for stopband detection.")
            s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(s)[:, 1, 0]), 1e-300))

        stopbands = detect_stopbands(frequency_hz, s21_db)

        return stopbands, {
            "status": RunStatus.PASS.value,
            "n_stopbands": len(stopbands) if hasattr(stopbands, "__len__") else None,
            "stopbands": jsonify(stopbands),
            "messages": ("PASS: stopband detection completed.",),
        }

    except Exception as exc:
        return None, {
            "status": RunStatus.PARTIAL.value,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "messages": (
                "PARTIAL: stopband detection failed or is unavailable.",
                f"{type(exc).__name__}: {exc}",
            ),
        }


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


def export_artifacts(
    *,
    layout: Any,
    scan: Any,
    dispersion: Any | None,
    stopbands: Any | None,
    config: Linear100mmConfig,
    stages: list[StageResult],
    output_dir: Path,
) -> dict[str, str]:
    """
    Write all baseline artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    frequency_hz = np.asarray(config.frequency_hz)
    s = get_attr_any(scan, "s", default=None)
    s21_db = get_attr_any(scan, "s21_db", default=None)

    if s21_db is None and s is not None:
        s21_db = 20.0 * jnp.log10(jnp.maximum(jnp.abs(jnp.asarray(s)[:, 1, 0]), 1e-300))

    arrays_npz = output_dir / "linear_100mm_baseline_arrays.npz"
    payload = {
        "frequency_hz": frequency_hz,
        "s21_db": np.asarray(s21_db) if s21_db is not None else np.asarray([]),
        "metadata_json": json.dumps(
            {
                "config": config.to_dict(),
                "layout": layout_summary(layout),
            }
        ),
    }

    if s is not None:
        payload["s"] = np.asarray(s)

    if dispersion is not None:
        beta = get_attr_any(
            dispersion,
            "beta_preferred_rad_per_m",
            "beta_eff_rad_per_m",
            "beta_rad_per_m",
            default=None,
        )
        alpha = get_attr_any(
            dispersion,
            "alpha_preferred_np_per_m",
            "alpha_np_per_m",
            default=None,
        )
        if beta is not None:
            payload["beta_rad_per_m"] = np.asarray(beta)
        if alpha is not None:
            payload["alpha_np_per_m"] = np.asarray(alpha)

    np.savez_compressed(arrays_npz, **payload)
    paths["arrays_npz"] = str(arrays_npz)

    component_csv = write_component_csv(layout, output_dir / "linear_100mm_layout_components.csv")
    paths["layout_components_csv"] = str(component_csv)

    if config.export_netlist:
        try:
            from twpa.io.netlist import write_netlist_bundle

            netlist_paths = write_netlist_bundle(
                layout,
                output_dir / "netlist",
                prefix="linear_100mm",
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

            checkpoint_path = output_dir / "linear_100mm_baseline_checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.LINEAR_SCAN,
                    name="linear_100mm_baseline",
                    source="scripts.linear_100mm_baseline",
                    extra={
                        "config": config.to_dict(),
                        "layout": layout_summary(layout),
                    },
                ),
                arrays={k: v for k, v in payload.items() if k != "metadata_json"},
                payload={
                    "scan": jsonify(scan),
                    "dispersion": jsonify(dispersion),
                    "stopbands": jsonify(stopbands),
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

            fig, _ = plot_s21(
                scan,
                config=PlotConfig(title="100 mm pump-off S21 baseline"),
            )
            s21_png = save_figure(fig, output_dir / "linear_100mm_s21.png")
            paths["s21_png"] = str(s21_png)

            import matplotlib.pyplot as plt

            plt.close(fig)

            if dispersion is not None:
                fig, _ = plot_dispersion(
                    dispersion,
                    config=PlotConfig(title="100 mm extracted dispersion"),
                    quantity="beta",
                )
                disp_png = save_figure(fig, output_dir / "linear_100mm_dispersion_beta.png")
                paths["dispersion_beta_png"] = str(disp_png)
                plt.close(fig)

            if stopbands is not None:
                fig, _ = plot_stopbands(
                    scan,
                    stopbands=stopbands,
                    config=PlotConfig(title="100 mm stopband diagnostics"),
                )
                stop_png = save_figure(fig, output_dir / "linear_100mm_stopbands.png")
                paths["stopbands_png"] = str(stop_png)
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


def result_markdown(result: Linear100mmResult) -> str:
    cfg = result.config
    lines = [
        "# Linear 100 mm baseline",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- cells: `{cfg.n_cells}`",
        f"- physical length: `{cfg.length_mm:.6g} mm`",
        f"- frequency range: `{cfg.f_min_ghz:.6g}`–`{cfg.f_max_ghz:.6g} GHz`",
        f"- frequency points: `{cfg.n_frequency}`",
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
        if stage.name in {"build_layout", "linear_scan", "dispersion", "stopbands"}:
            lines += [
                f"### {stage.name}",
                "",
                "```json",
                json.dumps(jsonify(stage.summary), indent=2)[:6000],
                "```",
                "",
            ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a 100 mm pump-off linear TWPA baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)

    parser.add_argument("--f-min-ghz", type=float, default=1.0)
    parser.add_argument("--f-max-ghz", type=float, default=14.0)
    parser.add_argument("--n-frequency", type=int, default=501)

    parser.add_argument("--layout-csv", type=str, default=None)
    parser.add_argument(
        "--layout-kind",
        type=str,
        default="uniform",
        help="Synthetic layout kind if workflow synthetic layout builder is available.",
    )
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument(
        "--disorder-std",
        type=float,
        default=0.0,
        help="Lognormal disorder standard deviation for fallback/synthetic layouts.",
    )
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/linear_100mm_baseline"),
    )
    parser.add_argument("--name", type=str, default="linear_100mm_baseline")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )

    parser.add_argument(
        "--export-netlist",
        action="store_true",
        help="Export SPICE/component/netlist bundle.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable diagnostic plot generation.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint generation.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: reduce n_cells and n_frequency unless explicitly overridden.",
    )

    return parser


def resolve_config(args: argparse.Namespace) -> Linear100mmConfig:
    n_cells = int(args.n_cells)
    n_frequency = int(args.n_frequency)

    if args.quick:
        if n_cells == 20000:
            n_cells = 2000
        if n_frequency == 501:
            n_frequency = 201

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

    return Linear100mmConfig(
        n_cells=n_cells,
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        f_min_ghz=float(args.f_min_ghz),
        f_max_ghz=float(args.f_max_ghz),
        n_frequency=n_frequency,
        layout_csv=args.layout_csv,
        output_dir=str(args.output_dir),
        layout_kind=str(args.layout_kind),
        include_resonators=bool(args.include_resonators),
        disorder_std=float(args.disorder_std),
        seed=int(args.seed),
        export_netlist=bool(args.export_netlist),
        make_plots=not bool(args.no_plots),
        save_checkpoint=not bool(args.no_checkpoint),
        quick=bool(args.quick),
        name=str(args.name),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[linear-100mm] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: list[StageResult] = []
    artifacts: dict[str, str] = {}
    metadata: dict[str, Any] = {
        "python": sys.version,
        "jax": {
            "version": getattr(jax, "__version__", None),
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "devices": [str(d) for d in jax.devices()],
        },
        "script": "scripts/linear_100mm_baseline.py",
    }

    layout_holder: dict[str, Any] = {}
    scan_holder: dict[str, Any] = {}
    dispersion_holder: dict[str, Any] = {}
    stopband_holder: dict[str, Any] = {}

    print("[linear-100mm] building/loading layout...")
    stage = run_stage(
        "build_layout",
        lambda: _stage_build_layout(layout_holder, config),
    )
    stages.append(stage)
    print(f"[linear-100mm] build_layout: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = Linear100mmResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[linear-100mm] running linear scan...")
    stage = run_stage(
        "linear_scan",
        lambda: _stage_linear_scan(layout_holder, scan_holder, config),
    )
    stages.append(stage)
    print(f"[linear-100mm] linear_scan: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = Linear100mmResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[linear-100mm] extracting dispersion...")
    stage = run_stage(
        "dispersion",
        lambda: _stage_dispersion(layout_holder, dispersion_holder, config),
    )
    stages.append(stage)
    print(f"[linear-100mm] dispersion: {stage.status.value}")

    print("[linear-100mm] detecting stopbands...")
    stage = run_stage(
        "stopbands",
        lambda: _stage_stopbands(scan_holder, stopband_holder, config),
    )
    stages.append(stage)
    print(f"[linear-100mm] stopbands: {stage.status.value}")

    print("[linear-100mm] exporting artifacts...")
    try:
        artifacts.update(
            export_artifacts(
                layout=layout_holder["layout"],
                scan=scan_holder["scan"],
                dispersion=dispersion_holder.get("dispersion"),
                stopbands=stopband_holder.get("stopbands"),
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

    result = Linear100mmResult(
        config=config,
        status=status,
        elapsed_s=time.perf_counter() - start,
        stages=tuple(stages),
        artifact_paths=artifacts,
        metadata=metadata,
    )

    return finalize_result(result, output_dir)


def _stage_build_layout(holder: dict[str, Any], config: Linear100mmConfig) -> dict[str, Any]:
    layout, summary = build_layout(config)
    holder["layout"] = layout
    return {
        "status": RunStatus.PASS.value,
        **summary,
        "messages": ("PASS: layout built/loaded.",),
    }


def _stage_linear_scan(
    layout_holder: dict[str, Any],
    scan_holder: dict[str, Any],
    config: Linear100mmConfig,
) -> dict[str, Any]:
    scan, summary = run_linear_scan_stage(layout_holder["layout"], config)
    scan_holder["scan"] = scan

    finite = bool(summary.get("finite", False))
    return {
        "status": RunStatus.PASS.value if finite else RunStatus.FAIL.value,
        **summary,
        "messages": (
            "PASS: linear scan completed with finite S21."
            if finite
            else "FAIL: linear scan completed but S21 contains non-finite values."
        ),
    }


def _stage_dispersion(
    layout_holder: dict[str, Any],
    dispersion_holder: dict[str, Any],
    config: Linear100mmConfig,
) -> dict[str, Any]:
    dispersion, summary = run_dispersion_stage(layout_holder["layout"], config)
    if dispersion is not None:
        dispersion_holder["dispersion"] = dispersion
    return summary


def _stage_stopbands(
    scan_holder: dict[str, Any],
    stopband_holder: dict[str, Any],
    config: Linear100mmConfig,
) -> dict[str, Any]:
    stopbands, summary = detect_stopbands_stage(scan_holder["scan"], config)
    if stopbands is not None:
        stopband_holder["stopbands"] = stopbands
    return summary


def finalize_result(result: Linear100mmResult, output_dir: Path) -> int:
    summary_json = output_dir / "linear_100mm_baseline_summary.json"
    summary_md = output_dir / "linear_100mm_baseline_summary.md"

    # Include these two report files in the result before writing.
    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    result = Linear100mmResult(
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
    print(f"[linear-100mm] status: {result.status.value}")
    print(f"[linear-100mm] summary JSON: {summary_json}")
    print(f"[linear-100mm] summary MD:   {summary_md}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())
