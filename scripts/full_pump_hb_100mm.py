"""
Run the full 100 mm pump harmonic-balance workflow.

This is the production-oriented pump solve entry point for the industrial TWPA
target: long chip, many cells, JAX-backed backend, checkpointed artifacts, and
explicit PASS/PARTIAL/ERROR reporting.

It reuses the lower-level machinery from ``scripts/pump_hb_small_ladder.py`` but
adds industrial-scale orchestration:

    1. Build/load the 100 mm layout.
    2. Build pump HB problem objects.
    3. Optionally run warm-up effective-cell pump solves.
    4. Run the requested full pump solve.
    5. Export arrays, profile CSV, checkpoint, plots, and summary reports.

Examples
--------
Quick smoke run:

    python scripts/full_pump_hb_100mm.py --quick --output-dir outputs/full_pump_100mm_quick

Industrial 100 mm / 20,000-cell run:

    python scripts/full_pump_hb_100mm.py ^
      --n-cells 20000 ^
      --length-mm 100 ^
      --pump-frequency-ghz 10 ^
      --pump-current-ratio 0.08 ^
      --output-dir outputs/full_pump_100mm

Use an existing layout CSV:

    python scripts/full_pump_hb_100mm.py ^
      --layout-csv outputs/linear_100mm_baseline/linear_100mm_layout_components.csv ^
      --output-dir outputs/full_pump_100mm_from_csv

Run warm-up sizes before the full solve:

    python scripts/full_pump_hb_100mm.py ^
      --warmup-cells 500 1000 2000 ^
      --n-cells 20000 ^
      --output-dir outputs/full_pump_100mm_warmup
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class FullPumpMode(str, Enum):
    AUTO = "auto"
    PACKAGE = "package"
    FALLBACK_LINEAR_PUMP = "fallback_linear_pump"


@dataclass(frozen=True)
class FullPump100mmConfig:
    n_cells: int
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
    solver_mode: FullPumpMode
    numerical_backend: str

    warmup_cells: tuple[int, ...]
    run_warmup: bool
    allow_partial_fallback: bool

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
    export_profile_csv: bool

    @property
    def length_m(self) -> float:
        return self.length_mm * 1e-3

    @property
    def pump_frequency_hz(self) -> float:
        return self.pump_frequency_ghz * 1e9

    @property
    def pump_current_a(self) -> float:
        return self.pump_current_ratio * self.i_star_a

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["solver_mode"] = self.solver_mode.value
        d["harmonic_orders"] = list(self.harmonic_orders)
        d["warmup_cells"] = list(self.warmup_cells)
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
class WarmupResult:
    n_cells: int
    status: RunStatus
    elapsed_s: float
    summary: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "summary": jsonify(self.summary),
            "artifact_paths": dict(self.artifact_paths),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class FullPump100mmResult:
    config: FullPump100mmConfig
    status: RunStatus
    elapsed_s: float
    stages: tuple[StageResult, ...]
    warmups: tuple[WarmupResult, ...]
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
            "warmups": [w.to_dict() for w in self.warmups],
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


def to_small_config(
    config: FullPump100mmConfig,
    *,
    n_cells: int | None = None,
    n_cells_limit: int | None = None,
    length_mm: float | None = None,
    output_dir: str | Path | None = None,
    name: str | None = None,
):
    """
    Convert the full-run config into the reusable small-ladder config type.
    """
    from scripts.pump_hb_small_ladder import (
        NumericalBackend,
        PumpHBSmallLadderConfig,
        PumpSolverMode,
    )

    if n_cells is None:
        n_cells = config.n_cells

    if length_mm is None:
        length_mm = config.length_mm

    if output_dir is None:
        output_dir = config.output_dir

    if name is None:
        name = config.name

    if config.solver_mode == FullPumpMode.PACKAGE:
        solver_mode = PumpSolverMode.PACKAGE
    elif config.solver_mode == FullPumpMode.FALLBACK_LINEAR_PUMP:
        solver_mode = PumpSolverMode.FALLBACK_LINEAR_PUMP
    else:
        solver_mode = PumpSolverMode.AUTO

    return PumpHBSmallLadderConfig(
        n_cells=int(n_cells),
        n_cells_limit=n_cells_limit,
        length_mm=float(length_mm),
        z0_ohm=config.z0_ohm,
        phase_velocity_m_per_s=config.phase_velocity_m_per_s,
        layout_csv=config.layout_csv,
        pump_frequency_ghz=config.pump_frequency_ghz,
        pump_current_ratio=config.pump_current_ratio,
        pump_phase_rad=config.pump_phase_rad,
        i_star_a=config.i_star_a,
        l0_scale=config.l0_scale,
        nonlinear_beta=config.nonlinear_beta,
        harmonic_orders=config.harmonic_orders,
        n_time=config.n_time,
        max_iter=config.max_iter,
        tolerance=config.tolerance,
        damping=config.damping,
        continuation_steps=config.continuation_steps,
        solver_mode=solver_mode,
        numerical_backend=NumericalBackend(config.numerical_backend),
        layout_kind=config.layout_kind,
        include_resonators=config.include_resonators,
        disorder_std=config.disorder_std,
        seed=config.seed,
        output_dir=str(output_dir),
        name=str(name),
        quick=False,
        make_plots=config.make_plots,
        save_checkpoint=config.save_checkpoint,
        export_netlist=config.export_netlist,
    )


def build_full_layout(config: FullPump100mmConfig) -> tuple[Any, Mapping[str, Any]]:
    """
    Build/load the full layout through the lower-level small-ladder builder.
    """
    from scripts.pump_hb_small_ladder import build_or_load_layout

    small_config = to_small_config(config)
    return build_or_load_layout(small_config)


def build_problem_objects(config: FullPump100mmConfig) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    """
    Build frequency plan, nonlinear params, pump drive, and solver config.
    """
    from scripts.pump_hb_small_ladder import (
        build_frequency_plan,
        build_nonlinear_params,
        build_pump_drive,
        build_pump_solver_config,
    )

    small_config = to_small_config(config)

    frequency_plan = build_frequency_plan(small_config)
    nonlinear_params = build_nonlinear_params(small_config)
    pump_drive = build_pump_drive(small_config)
    pump_config = build_pump_solver_config(small_config)

    summary = {
        "frequency_plan": jsonify(frequency_plan),
        "nonlinear_params": jsonify(nonlinear_params),
        "pump_drive": jsonify(pump_drive),
        "pump_config": jsonify(pump_config),
        "pump_current_A": config.pump_current_a,
    }

    return frequency_plan, nonlinear_params, pump_drive, pump_config, summary


def run_full_pump_solve(
    *,
    layout: Any,
    nonlinear_params: Any,
    frequency_plan: Any,
    pump_drive: Any,
    pump_config: Any,
    config: FullPump100mmConfig,
) -> tuple[Any, Mapping[str, Any]]:
    """
    Run the full pump solve through the package/fallback backend.
    """
    from scripts.pump_hb_small_ladder import run_pump_solver

    small_config = to_small_config(config)

    pump_result, summary = run_pump_solver(
        layout=layout,
        nonlinear_params=nonlinear_params,
        frequency_plan=frequency_plan,
        pump_drive=pump_drive,
        pump_config=pump_config,
        config=small_config,
    )

    status = RunStatus(summary.get("status", RunStatus.PASS.value))
    if status == RunStatus.PARTIAL and not config.allow_partial_fallback:
        summary = {
            **dict(summary),
            "status": RunStatus.FAIL.value,
            "messages": (
                *tuple(summary.get("messages", ())),
                "FAIL: partial fallback was produced but --no-allow-partial-fallback is active.",
            ),
        }

    return pump_result, summary


def coarsen_layout_for_warmup(layout: Any, n_cells: int, *, name: str) -> Any:
    """
    Coarsen a full layout for a warm-up solve.
    """
    try:
        from scripts.effective_cell_convergence import conservative_group_layout

        coarsened, _ = conservative_group_layout(layout, int(n_cells), name=name)
        return coarsened
    except Exception:
        # Fallback: if coarsening is unavailable, build a fresh uniform layout
        # with the requested cell count and same total length.
        from scripts.pump_hb_small_ladder import build_uniform_layout_fallback

        full = layout_summary(layout)
        total_length_m = full.get("total_length_m") or 0.100

        dummy = FullPump100mmConfig(
            n_cells=int(n_cells),
            length_mm=float(total_length_m) * 1e3,
            z0_ohm=float(full.get("z0_ohm") or 50.0),
            phase_velocity_m_per_s=1.2e8,
            layout_csv=None,
            pump_frequency_ghz=10.0,
            pump_current_ratio=0.05,
            pump_phase_rad=0.0,
            i_star_a=5e-3,
            l0_scale=1.0,
            nonlinear_beta=1.0,
            harmonic_orders=(-3, -1, 1, 3),
            n_time=64,
            max_iter=20,
            tolerance=1e-10,
            damping=1.0,
            continuation_steps=4,
            solver_mode=FullPumpMode.FALLBACK_LINEAR_PUMP,
            warmup_cells=(),
            run_warmup=False,
            allow_partial_fallback=True,
            layout_kind="uniform",
            include_resonators=False,
            disorder_std=0.0,
            seed=1234,
            output_dir="outputs",
            name=name,
            quick=False,
            make_plots=False,
            save_checkpoint=False,
            export_netlist=False,
            export_profile_csv=False,
        )
        return build_uniform_layout_fallback(to_small_config(dummy))


def run_warmup_solve(
    *,
    full_layout: Any,
    n_cells: int,
    config: FullPump100mmConfig,
    output_dir: Path,
) -> WarmupResult:
    """
    Run one warm-up pump solve on a coarsened layout.
    """
    start = time.perf_counter()
    run_dir = output_dir / "warmups" / f"warmup_{n_cells}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.pump_hb_small_ladder import (
            build_frequency_plan,
            build_nonlinear_params,
            build_pump_drive,
            build_pump_solver_config,
            export_artifacts,
            run_pump_solver,
            summarize_pump_result,
        )

        warm_layout = coarsen_layout_for_warmup(
            full_layout,
            n_cells,
            name=f"{config.name}_warmup_{n_cells}",
        )

        warm_cfg = to_small_config(
            config,
            n_cells=n_cells,
            n_cells_limit=None,
            length_mm=float(layout_summary(warm_layout).get("total_length_m", config.length_m)) * 1e3,
            output_dir=run_dir,
            name=f"{config.name}_warmup_{n_cells}",
        )

        frequency_plan = build_frequency_plan(warm_cfg)
        nonlinear_params = build_nonlinear_params(warm_cfg)
        pump_drive = build_pump_drive(warm_cfg)
        pump_config = build_pump_solver_config(warm_cfg)

        pump_result, solve_summary = run_pump_solver(
            layout=warm_layout,
            nonlinear_params=nonlinear_params,
            frequency_plan=frequency_plan,
            pump_drive=pump_drive,
            pump_config=pump_config,
            config=warm_cfg,
        )

        stages: list[Any] = []
        artifact_paths = export_artifacts(
            layout=warm_layout,
            nonlinear_params=nonlinear_params,
            frequency_plan=frequency_plan,
            pump_drive=pump_drive,
            pump_config=pump_config,
            pump_result=pump_result,
            config=warm_cfg,
            stages=stages,
            output_dir=run_dir,
        )

        summary = {
            "layout": layout_summary(warm_layout),
            "solve": solve_summary,
            "pump_result_summary": summarize_pump_result(pump_result, config=warm_cfg),
        }

        status = RunStatus(solve_summary.get("status", RunStatus.PASS.value))
        messages = tuple(solve_summary.get("messages", ()))

        return WarmupResult(
            n_cells=n_cells,
            status=status,
            elapsed_s=time.perf_counter() - start,
            summary=summary,
            artifact_paths=artifact_paths,
            messages=messages,
        )

    except Exception as exc:
        error_path = run_dir / "warmup_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return WarmupResult(
            n_cells=n_cells,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            summary={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
            artifact_paths={"error_txt": str(error_path)},
            messages=(f"ERROR: warm-up solve failed: {type(exc).__name__}: {exc}",),
        )


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


def extract_pump_arrays(pump_result: Any, config: FullPump100mmConfig) -> dict[str, Any]:
    from scripts.pump_hb_small_ladder import extract_pump_arrays as extract_small_arrays

    return extract_small_arrays(pump_result, to_small_config(config))


def summarize_pump_result(pump_result: Any, config: FullPump100mmConfig) -> dict[str, Any]:
    from scripts.pump_hb_small_ladder import summarize_pump_result as summarize_small

    return summarize_small(pump_result, config=to_small_config(config))


def write_pump_profile_csv(
    path: Path,
    *,
    arrays: Mapping[str, Any],
    config: FullPump100mmConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    branch_current = arrays.get("branch_current_coeffs_A")
    node_voltage = arrays.get("node_voltage_coeffs_V")

    if branch_current is None:
        raise ValueError("branch_current_coeffs_A unavailable")

    I = np.asarray(branch_current)
    harmonic_orders = list(config.harmonic_orders)
    pump_idx = harmonic_orders.index(1) if 1 in harmonic_orders else 0

    n_cells = I.shape[-1]
    z_branches = arrays.get("z_branches_m")
    if z_branches is None:
        z = np.linspace(0.0, config.length_m, n_cells)
    else:
        z = np.asarray(z_branches, dtype=float)

    fields = [
        "cell_index",
        "z_m",
        "z_mm",
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
                "z_mm": float(z[i] * 1e3) if i < len(z) else float(i),
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


def export_full_artifacts(
    *,
    layout: Any,
    frequency_plan: Any,
    nonlinear_params: Any,
    pump_drive: Any,
    pump_config: Any,
    pump_result: Any,
    config: FullPump100mmConfig,
    stages: list[StageResult],
    warmups: Sequence[WarmupResult],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    arrays = extract_pump_arrays(pump_result, config)
    pump_summary = summarize_pump_result(pump_result, config)

    arrays_payload: dict[str, Any] = {
        "metadata_json": json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "layout": layout_summary(layout),
                    "frequency_plan": frequency_plan,
                    "nonlinear_params": nonlinear_params,
                    "pump_drive": pump_drive,
                    "pump_config": pump_config,
                    "pump_summary": pump_summary,
                    "warmups": [w.to_dict() for w in warmups],
                }
            )
        ),
    }

    for key, value in arrays.items():
        if value is not None:
            arrays_payload[key] = np.asarray(value)

    arrays_npz = output_dir / "full_pump_hb_100mm_arrays.npz"
    np.savez_compressed(arrays_npz, **arrays_payload)
    paths["arrays_npz"] = str(arrays_npz)

    layout_csv = write_component_csv(layout, output_dir / "full_pump_hb_100mm_layout_components.csv")
    paths["layout_components_csv"] = str(layout_csv)

    if config.export_profile_csv and arrays.get("branch_current_coeffs_A") is not None:
        profile_csv = write_pump_profile_csv(
            output_dir / "full_pump_hb_100mm_profile.csv",
            arrays=arrays,
            config=config,
        )
        paths["pump_profile_csv"] = str(profile_csv)

    pump_summary_json = output_dir / "full_pump_hb_100mm_pump_summary.json"
    pump_summary_json.write_text(json.dumps(jsonify(pump_summary), indent=2), encoding="utf-8")
    paths["pump_summary_json"] = str(pump_summary_json)

    if config.export_netlist:
        try:
            from twpa.io.netlist import write_netlist_bundle

            netlist_paths = write_netlist_bundle(
                layout,
                output_dir / "netlist",
                prefix="full_pump_hb_100mm",
            )
            paths.update({f"netlist_{k}": v for k, v in netlist_paths.items()})
        except Exception as exc:
            error_path = output_dir / "netlist_export_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            paths["netlist_error_txt"] = str(error_path)
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

            checkpoint_path = output_dir / "full_pump_hb_100mm_checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                metadata=CheckpointMetadata(
                    kind=CheckpointKind.PUMP_HB,
                    name="full_pump_hb_100mm",
                    source="scripts.full_pump_hb_100mm",
                    extra={
                        "config": config.to_dict(),
                        "layout": layout_summary(layout),
                    },
                ),
                arrays={k: v for k, v in arrays.items() if v is not None},
                payload={
                    "pump_result": jsonify(pump_result),
                    "pump_summary": pump_summary,
                    "warmups": [w.to_dict() for w in warmups],
                },
            )
            paths["checkpoint_npz"] = str(checkpoint_path)

        except Exception as exc:
            error_path = output_dir / "checkpoint_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            paths["checkpoint_error_txt"] = str(error_path)
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

            branch_current = arrays.get("branch_current_coeffs_A")
            if branch_current is None:
                raise RuntimeError("branch_current_coeffs_A unavailable for pump plot.")

            I = np.asarray(branch_current)
            harmonic_orders = list(config.harmonic_orders)
            pump_idx = harmonic_orders.index(1) if 1 in harmonic_orders else 0

            y = np.abs(I[pump_idx]) / max(config.i_star_a, 1e-300)
            z = arrays.get("z_branches_m")
            if z is None:
                x = np.linspace(0.0, config.length_mm, y.shape[0])
            else:
                x = np.asarray(z, dtype=float) * 1e3

            fig, ax = plt.subplots(figsize=(8, 4.8), dpi=140)
            ax.plot(x, y)
            ax.set_xlabel("Position (mm)")
            ax.set_ylabel("|I_pump| / I*")
            ax.set_title("Full 100 mm pump profile")
            ax.grid(True)
            fig.tight_layout()
            plot_path = output_dir / "full_pump_hb_100mm_profile.png"
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
            paths["pump_profile_png"] = str(plot_path)

        except Exception as exc:
            error_path = output_dir / "plotting_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            paths["plotting_error_txt"] = str(error_path)
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


def result_markdown(result: FullPump100mmResult) -> str:
    cfg = result.config

    lines = [
        "# Full pump HB 100 mm run",
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
        f"- warmups: `{list(cfg.warmup_cells) if cfg.run_warmup else []}`",
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

    if result.warmups:
        lines += [
            "",
            "## Warm-up solves",
            "",
            "| cells | status | elapsed s | messages |",
            "|---:|---|---:|---|",
        ]
        for warmup in result.warmups:
            msg = "<br>".join(warmup.messages[:3])
            lines.append(
                f"| {warmup.n_cells} | `{warmup.status.value}` | "
                f"{warmup.elapsed_s:.6g} | {msg} |"
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
            "warmups",
            "full_pump_solve",
            "artifact_export",
        }:
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
        description="Run the full 100 mm pump-HB workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)
    parser.add_argument("--layout-csv", type=str, default=None)

    parser.add_argument("--pump-frequency-ghz", type=float, default=10.0)
    parser.add_argument("--pump-current-ratio", type=float, default=0.08)
    parser.add_argument("--pump-phase-rad", type=float, default=0.0)
    parser.add_argument("--i-star-a", type=float, default=5e-3)
    parser.add_argument("--l0-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-beta", type=float, default=1.0)

    parser.add_argument("--harmonic-orders", type=int, nargs="+", default=[-3, -1, 1, 3])
    parser.add_argument("--n-time", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--continuation-steps", type=int, default=10)
    parser.add_argument(
        "--solver-mode",
        choices=[m.value for m in FullPumpMode],
        default=FullPumpMode.AUTO.value,
    )
    parser.add_argument(
        "--numerical-backend",
        choices=["dense", "newton_krylov"],
        default="dense",
        help="Dense is guarded and intended for tiny reference runs only.",
    )

    parser.add_argument(
        "--warmup-cells",
        type=int,
        nargs="*",
        default=[],
        help="Optional coarsened warm-up cell counts.",
    )
    parser.add_argument("--run-warmup", action="store_true")
    parser.add_argument(
        "--no-allow-partial-fallback",
        action="store_true",
        help="Treat fallback linear pump result as failure instead of PARTIAL.",
    )

    parser.add_argument("--layout-kind", type=str, default="uniform")
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument("--disorder-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_pump_hb_100mm"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="full_pump_hb_100mm")

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--export-netlist", action="store_true")
    parser.add_argument("--no-profile-csv", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> FullPump100mmConfig:
    n_cells = int(args.n_cells)
    n_time = int(args.n_time)
    max_iter = int(args.max_iter)
    continuation_steps = int(args.continuation_steps)
    harmonic_orders = tuple(int(h) for h in args.harmonic_orders)
    warmup_cells = tuple(int(x) for x in args.warmup_cells)

    if args.quick:
        if n_cells == 20000:
            n_cells = 8
        if not warmup_cells and args.run_warmup:
            warmup_cells = (100, 250, 500)
        n_time = min(n_time, 32)
        max_iter = min(max_iter, 25)
        continuation_steps = min(continuation_steps, 5)

    warmup_cells = tuple(sorted(set(x for x in warmup_cells if x < n_cells)))

    if n_cells <= 0:
        raise ValueError("--n-cells must be positive")
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
        raise ValueError("--harmonic-orders must include 1")
    if len(set(harmonic_orders)) != len(harmonic_orders):
        raise ValueError("--harmonic-orders may not contain duplicates")
    if any(h == 0 for h in harmonic_orders):
        raise ValueError("--harmonic-orders should not include 0")
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
    if args.layout_csv is not None and not Path(args.layout_csv).exists():
        raise FileNotFoundError(args.layout_csv)

    return FullPump100mmConfig(
        n_cells=n_cells,
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
        solver_mode=FullPumpMode(args.solver_mode),
        numerical_backend=str(args.numerical_backend),
        warmup_cells=warmup_cells,
        run_warmup=bool(args.run_warmup or warmup_cells),
        allow_partial_fallback=not bool(args.no_allow_partial_fallback),
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
        export_profile_csv=not bool(args.no_profile_csv),
    )


def _stage_build_layout(holders: dict[str, Any], config: FullPump100mmConfig) -> dict[str, Any]:
    layout, summary = build_full_layout(config)
    holders["layout"] = layout
    return {
        "status": RunStatus.PASS.value,
        **dict(summary),
        "messages": ("PASS: full 100 mm layout built/loaded.",),
    }


def _stage_build_problem(holders: dict[str, Any], config: FullPump100mmConfig) -> dict[str, Any]:
    frequency_plan, nonlinear_params, pump_drive, pump_config, summary = build_problem_objects(config)

    holders["frequency_plan"] = frequency_plan
    holders["nonlinear_params"] = nonlinear_params
    holders["pump_drive"] = pump_drive
    holders["pump_config"] = pump_config

    return {
        "status": RunStatus.PASS.value,
        **dict(summary),
        "messages": ("PASS: full pump HB problem built.",),
    }


def _stage_full_pump_solve(holders: dict[str, Any], config: FullPump100mmConfig) -> dict[str, Any]:
    pump_result, summary = run_full_pump_solve(
        layout=holders["layout"],
        nonlinear_params=holders["nonlinear_params"],
        frequency_plan=holders["frequency_plan"],
        pump_drive=holders["pump_drive"],
        pump_config=holders["pump_config"],
        config=config,
    )

    holders["pump_result"] = pump_result

    return dict(summary)


def finalize_result(result: FullPump100mmResult, output_dir: Path) -> int:
    summary_json = output_dir / "full_pump_hb_100mm_summary.json"
    summary_md = output_dir / "full_pump_hb_100mm_summary.md"

    artifact_paths = {
        **dict(result.artifact_paths),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }

    result = FullPump100mmResult(
        config=result.config,
        status=result.status,
        elapsed_s=result.elapsed_s,
        stages=result.stages,
        warmups=result.warmups,
        artifact_paths=artifact_paths,
        metadata=result.metadata,
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    summary_md.write_text(result_markdown(result), encoding="utf-8")

    print()
    print(f"[full-pump-100mm] status: {result.status.value}")
    print(f"[full-pump-100mm] summary JSON: {summary_json}")
    print(f"[full-pump-100mm] summary MD:   {summary_md}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[full-pump-100mm] invalid arguments: {exc}", file=sys.stderr)
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
        "script": "scripts/full_pump_hb_100mm.py",
    }

    holders: dict[str, Any] = {}
    stages: list[StageResult] = []
    warmups: list[WarmupResult] = []
    artifacts: dict[str, str] = {}

    print("[full-pump-100mm] building/loading layout...")
    stage = run_stage("build_layout", lambda: _stage_build_layout(holders, config))
    stages.append(stage)
    print(f"[full-pump-100mm] build_layout: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = FullPump100mmResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            warmups=tuple(warmups),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[full-pump-100mm] building pump problem...")
    stage = run_stage("build_problem", lambda: _stage_build_problem(holders, config))
    stages.append(stage)
    print(f"[full-pump-100mm] build_problem: {stage.status.value}")

    if stage.status in {RunStatus.FAIL, RunStatus.ERROR}:
        result = FullPump100mmResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            warmups=tuple(warmups),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    if config.run_warmup and config.warmup_cells:
        print("[full-pump-100mm] running warm-up solves...")
        warmup_start = time.perf_counter()

        for n_cells in config.warmup_cells:
            print(f"[full-pump-100mm] warmup n={n_cells}...")
            warmup = run_warmup_solve(
                full_layout=holders["layout"],
                n_cells=n_cells,
                config=config,
                output_dir=output_dir,
            )
            warmups.append(warmup)
            print(f"[full-pump-100mm] warmup n={n_cells}: {warmup.status.value}")

        stages.append(
            StageResult(
                name="warmups",
                status=RunStatus.PASS
                if all(w.status in {RunStatus.PASS, RunStatus.PARTIAL} for w in warmups)
                else RunStatus.PARTIAL,
                elapsed_s=time.perf_counter() - warmup_start,
                summary={
                    "n_warmups": len(warmups),
                    "n_pass": sum(1 for w in warmups if w.status == RunStatus.PASS),
                    "n_partial": sum(1 for w in warmups if w.status == RunStatus.PARTIAL),
                    "n_error": sum(1 for w in warmups if w.status == RunStatus.ERROR),
                    "warmups": [w.to_dict() for w in warmups],
                },
                messages=("PASS: warm-up stage completed.",),
            )
        )

    print("[full-pump-100mm] solving full pump HB...")
    stage = run_stage("full_pump_solve", lambda: _stage_full_pump_solve(holders, config))
    stages.append(stage)
    print(f"[full-pump-100mm] full_pump_solve: {stage.status.value}")

    if stage.status == RunStatus.ERROR:
        result = FullPump100mmResult(
            config=config,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            stages=tuple(stages),
            warmups=tuple(warmups),
            artifact_paths=artifacts,
            metadata=metadata,
        )
        return finalize_result(result, output_dir)

    print("[full-pump-100mm] exporting artifacts...")
    try:
        artifacts.update(
            export_full_artifacts(
                layout=holders["layout"],
                frequency_plan=holders["frequency_plan"],
                nonlinear_params=holders["nonlinear_params"],
                pump_drive=holders["pump_drive"],
                pump_config=holders["pump_config"],
                pump_result=holders["pump_result"],
                config=config,
                stages=stages,
                warmups=warmups,
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

    result = FullPump100mmResult(
        config=config,
        status=status,
        elapsed_s=time.perf_counter() - start,
        stages=tuple(stages),
        warmups=tuple(warmups),
        artifact_paths=artifacts,
        metadata=metadata,
    )

    return finalize_result(result, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
