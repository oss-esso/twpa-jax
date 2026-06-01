"""
Run a pump-HB scaling study over ladder size and pump drive.

This script orchestrates repeated calls to ``scripts/pump_hb_small_ladder.py``
and aggregates runtime, convergence, residual, and pump-profile metrics.

Examples
--------
Quick scaling smoke test:

    python scripts/pump_hb_scaling_study.py --quick --output-dir outputs/pump_hb_scaling_quick

Explicit sizes and pump ratios:

    python scripts/pump_hb_scaling_study.py ^
      --sizes 16 32 64 128 ^
      --pump-current-ratios 0.02 0.05 0.08 ^
      --output-dir outputs/pump_hb_scaling

Use package solver only and fail if fallback is needed:

    python scripts/pump_hb_scaling_study.py ^
      --sizes 32 64 128 ^
      --solver-mode package ^
      --fail-fast ^
      --output-dir outputs/pump_hb_scaling_package
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class RunStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass(frozen=True)
class ScalingStudyConfig:
    sizes: tuple[int, ...]
    pump_current_ratios: tuple[float, ...]
    harmonic_orders: tuple[int, ...]
    n_time: int
    max_iter: int
    tolerance: float
    damping: float
    continuation_steps: int
    solver_mode: str

    length_mm: float
    scale_length_with_cells: bool
    cell_length_um: float
    z0_ohm: float
    phase_velocity_m_per_s: float
    pump_frequency_ghz: float
    pump_phase_rad: float
    i_star_a: float
    l0_scale: float
    nonlinear_beta: float

    layout_csv: str | None
    layout_kind: str
    include_resonators: bool
    disorder_std: float
    seed: int

    script_path: str
    output_dir: str
    name: str
    quick: bool
    fail_fast: bool
    per_run_plots: bool
    per_run_checkpoint: bool
    export_netlist: bool
    python_executable: str

    def length_for_size_mm(self, n_cells: int) -> float:
        if self.scale_length_with_cells:
            return n_cells * self.cell_length_um * 1e-3
        return self.length_mm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScalingRunResult:
    run_index: int
    n_cells: int
    pump_current_ratio: float
    status: RunStatus
    returncode: int
    elapsed_s: float
    run_dir: str
    command: tuple[str, ...]
    summary_path: str | None
    stdout_path: str
    stderr_path: str
    parsed_summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "n_cells": self.n_cells,
            "pump_current_ratio": self.pump_current_ratio,
            "status": self.status.value,
            "passed": self.passed,
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "run_dir": self.run_dir,
            "command": list(self.command),
            "summary_path": self.summary_path,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "parsed_summary": jsonify(self.parsed_summary),
            "metrics": jsonify(self.metrics),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class ScalingStudyResult:
    config: ScalingStudyConfig
    status: RunStatus
    elapsed_s: float
    runs: tuple[ScalingRunResult, ...]
    artifact_paths: Mapping[str, str]
    metadata: Mapping[str, Any]

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.runs if r.status == RunStatus.PASS)

    @property
    def n_partial(self) -> int:
        return sum(1 for r in self.runs if r.status == RunStatus.PARTIAL)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.runs if r.status in {RunStatus.FAIL, RunStatus.ERROR})

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "n_runs": self.n_runs,
            "n_pass": self.n_pass,
            "n_partial": self.n_partial,
            "n_fail": self.n_fail,
            "config": self.config.to_dict(),
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
            return repr(obj)
    if isinstance(obj, Mapping):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return jsonify(arr.item())
        return {
            "array_shape": tuple(int(v) for v in arr.shape),
            "array_dtype": str(arr.dtype),
        }
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


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


def find_stage(summary: Mapping[str, Any], stage_name: str) -> Mapping[str, Any] | None:
    for stage in summary.get("stages", []) or []:
        if stage.get("name") == stage_name:
            return stage
    return None


def extract_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    pump_stage = find_stage(summary, "pump_solve") or {}
    build_stage = find_stage(summary, "build_layout") or {}
    artifact_stage = find_stage(summary, "artifact_export") or {}

    pump_summary = pump_stage.get("summary", {}) if isinstance(pump_stage, Mapping) else {}
    build_summary = build_stage.get("summary", {}) if isinstance(build_stage, Mapping) else {}

    layout_n_cells = nested_get(build_summary, ["layout", "n_cells"])
    if layout_n_cells is None:
        layout_n_cells = nested_get(build_summary, ["layout", "summary", "n_cells"])

    arrays = pump_summary.get("arrays", {}) if isinstance(pump_summary, Mapping) else {}

    return {
        "overall_status": summary.get("status"),
        "overall_elapsed_s": summary.get("elapsed_s"),
        "layout_n_cells": layout_n_cells,
        "pump_stage_status": pump_stage.get("status"),
        "pump_stage_elapsed_s": pump_stage.get("elapsed_s"),
        "artifact_stage_status": artifact_stage.get("status"),
        "finite": pump_summary.get("finite"),
        "converged": pump_summary.get("converged"),
        "status_value": pump_summary.get("status_value"),
        "solver_function": pump_summary.get("solver_function"),
        "max_pump_current_A": pump_summary.get("max_pump_current_A"),
        "max_current_ratio": pump_summary.get("max_current_ratio"),
        "pump_output_input_current_ratio": pump_summary.get("pump_output_input_current_ratio"),
        "node_voltage_shape": nested_get(arrays, ["node_voltage_coeffs_V", "shape"]),
        "branch_current_shape": nested_get(arrays, ["branch_current_coeffs_A", "shape"]),
        "residual_norm_shape": nested_get(arrays, ["residual_norm", "shape"]),
    }


def command_for_run(
    *,
    config: ScalingStudyConfig,
    run_dir: Path,
    n_cells: int,
    pump_current_ratio: float,
) -> list[str]:
    cmd = [
        config.python_executable,
        config.script_path,
        "--n-cells",
        str(n_cells),
        "--length-mm",
        f"{config.length_for_size_mm(n_cells):.16g}",
        "--z0-ohm",
        f"{config.z0_ohm:.16g}",
        "--phase-velocity-m-per-s",
        f"{config.phase_velocity_m_per_s:.16g}",
        "--pump-frequency-ghz",
        f"{config.pump_frequency_ghz:.16g}",
        "--pump-current-ratio",
        f"{pump_current_ratio:.16g}",
        "--pump-phase-rad",
        f"{config.pump_phase_rad:.16g}",
        "--i-star-a",
        f"{config.i_star_a:.16g}",
        "--l0-scale",
        f"{config.l0_scale:.16g}",
        "--nonlinear-beta",
        f"{config.nonlinear_beta:.16g}",
        "--n-time",
        str(config.n_time),
        "--max-iter",
        str(config.max_iter),
        "--tolerance",
        f"{config.tolerance:.16g}",
        "--damping",
        f"{config.damping:.16g}",
        "--continuation-steps",
        str(config.continuation_steps),
        "--solver-mode",
        config.solver_mode,
        "--layout-kind",
        config.layout_kind,
        "--disorder-std",
        f"{config.disorder_std:.16g}",
        "--seed",
        str(config.seed),
        "--output-dir",
        str(run_dir),
        "--name",
        f"{config.name}_n{n_cells}_r{pump_current_ratio:.6g}".replace(".", "p"),
    ]

    cmd.extend(["--harmonic-orders", *[str(h) for h in config.harmonic_orders]])

    if config.layout_csv is not None:
        cmd.extend(["--layout-csv", config.layout_csv, "--n-cells-limit", str(n_cells)])

    if config.include_resonators:
        cmd.append("--include-resonators")

    if config.quick:
        cmd.append("--quick")

    if not config.per_run_plots:
        cmd.append("--no-plots")

    if not config.per_run_checkpoint:
        cmd.append("--no-checkpoint")

    if config.export_netlist:
        cmd.append("--export-netlist")

    return cmd


def run_one(
    *,
    config: ScalingStudyConfig,
    run_index: int,
    n_cells: int,
    pump_current_ratio: float,
    output_dir: Path,
) -> ScalingRunResult:
    run_name = f"run_{run_index:04d}_n{n_cells}_r{pump_current_ratio:.6g}".replace(".", "p")
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    cmd = command_for_run(
        config=config,
        run_dir=run_dir,
        n_cells=n_cells,
        pump_current_ratio=pump_current_ratio,
    )

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    summary_path = run_dir / "pump_hb_small_ladder_summary.json"
    parsed_summary: dict[str, Any] = {}
    messages: list[str] = []

    if summary_path.exists():
        try:
            parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            messages.append(f"Could not parse summary JSON: {type(exc).__name__}: {exc}")
    else:
        messages.append("Summary JSON not found.")

    metrics = extract_metrics(parsed_summary) if parsed_summary else {}

    summary_status = parsed_summary.get("status")
    if proc.returncode == 0 and summary_status == RunStatus.PASS.value:
        status = RunStatus.PASS
    elif proc.returncode == 0 and summary_status == RunStatus.PARTIAL.value:
        status = RunStatus.PARTIAL
    elif proc.returncode == 0:
        status = RunStatus.PARTIAL
        messages.append(f"Process returned 0 but summary status is {summary_status!r}.")
    elif parsed_summary:
        status = RunStatus.ERROR
        messages.append(f"Process returned {proc.returncode}; parsed summary status is {summary_status!r}.")
    else:
        status = RunStatus.ERROR
        messages.append(f"Process returned {proc.returncode} and no summary was parsed.")

    if not messages:
        messages.append(f"{status.value.upper()}: run completed with return code {proc.returncode}.")

    return ScalingRunResult(
        run_index=run_index,
        n_cells=n_cells,
        pump_current_ratio=pump_current_ratio,
        status=status,
        returncode=int(proc.returncode),
        elapsed_s=elapsed,
        run_dir=str(run_dir),
        command=tuple(cmd),
        summary_path=str(summary_path) if summary_path.exists() else None,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        parsed_summary=parsed_summary,
        metrics=metrics,
        messages=tuple(messages),
    )


def write_runs_csv(path: Path, runs: Sequence[ScalingRunResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys: list[str] = []
    for run in runs:
        for key in run.metrics.keys():
            if key not in metric_keys:
                metric_keys.append(key)

    fields = [
        "run_index",
        "n_cells",
        "pump_current_ratio",
        "status",
        "returncode",
        "elapsed_s",
        "run_dir",
        *metric_keys,
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for run in runs:
            row = {
                "run_index": run.run_index,
                "n_cells": run.n_cells,
                "pump_current_ratio": run.pump_current_ratio,
                "status": run.status.value,
                "returncode": run.returncode,
                "elapsed_s": run.elapsed_s,
                "run_dir": run.run_dir,
            }

            for key in metric_keys:
                value = run.metrics.get(key)
                if isinstance(value, (list, tuple, dict)):
                    row[key] = json.dumps(jsonify(value))
                else:
                    row[key] = value

            writer.writerow(row)

    return path


def aggregate_scaling(runs: Sequence[ScalingRunResult]) -> dict[str, Any]:
    rows = []
    for run in runs:
        row = {
            "n_cells": run.n_cells,
            "pump_current_ratio": run.pump_current_ratio,
            "status": run.status.value,
            "elapsed_s": run.elapsed_s,
            **dict(run.metrics),
        }
        rows.append(row)

    by_size: dict[int, list[ScalingRunResult]] = {}
    by_ratio: dict[float, list[ScalingRunResult]] = {}

    for run in runs:
        by_size.setdefault(run.n_cells, []).append(run)
        by_ratio.setdefault(run.pump_current_ratio, []).append(run)

    def summarize_group(group: Sequence[ScalingRunResult]) -> dict[str, Any]:
        elapsed = np.asarray([g.elapsed_s for g in group], dtype=float)
        passed = [g for g in group if g.status == RunStatus.PASS]
        partial = [g for g in group if g.status == RunStatus.PARTIAL]
        failed = [g for g in group if g.status in {RunStatus.FAIL, RunStatus.ERROR}]

        pump_elapsed = []
        max_ratio = []
        for g in group:
            pe = g.metrics.get("pump_stage_elapsed_s")
            mr = g.metrics.get("max_current_ratio")
            if pe is not None:
                try:
                    pump_elapsed.append(float(pe))
                except Exception:
                    pass
            if mr is not None:
                try:
                    max_ratio.append(float(mr))
                except Exception:
                    pass

        return {
            "n": len(group),
            "n_pass": len(passed),
            "n_partial": len(partial),
            "n_fail": len(failed),
            "elapsed_s_mean": float(np.nanmean(elapsed)) if elapsed.size else None,
            "elapsed_s_min": float(np.nanmin(elapsed)) if elapsed.size else None,
            "elapsed_s_max": float(np.nanmax(elapsed)) if elapsed.size else None,
            "pump_stage_elapsed_s_mean": float(np.nanmean(pump_elapsed)) if pump_elapsed else None,
            "max_current_ratio_mean": float(np.nanmean(max_ratio)) if max_ratio else None,
        }

    return {
        "n_runs": len(runs),
        "n_pass": sum(1 for r in runs if r.status == RunStatus.PASS),
        "n_partial": sum(1 for r in runs if r.status == RunStatus.PARTIAL),
        "n_fail": sum(1 for r in runs if r.status in {RunStatus.FAIL, RunStatus.ERROR}),
        "by_size": {str(k): summarize_group(v) for k, v in sorted(by_size.items())},
        "by_pump_current_ratio": {
            f"{k:.12g}": summarize_group(v)
            for k, v in sorted(by_ratio.items(), key=lambda kv: kv[0])
        },
        "rows": rows,
    }


def write_plots(output_dir: Path, runs: Sequence[ScalingRunResult]) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        err = output_dir / "plotting_unavailable.txt"
        err.write_text(str(exc), encoding="utf-8")
        return {"plotting_unavailable_txt": str(err)}

    paths: dict[str, str] = {}

    data = [
        r for r in runs
        if r.status in {RunStatus.PASS, RunStatus.PARTIAL}
    ]

    if not data:
        return paths

    ratios = sorted({r.pump_current_ratio for r in data})
    for ratio in ratios:
        group = sorted([r for r in data if r.pump_current_ratio == ratio], key=lambda r: r.n_cells)
        x = np.asarray([r.n_cells for r in group], dtype=float)
        y = np.asarray([r.elapsed_s for r in group], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        ax.loglog(x, y, marker="o")
        ax.set_xlabel("Number of cells")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"Pump HB scaling, pump ratio = {ratio:g}")
        ax.grid(True)
        fig.tight_layout()

        path = output_dir / f"scaling_walltime_ratio_{ratio:.6g}.png".replace(".", "p")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths[f"walltime_ratio_{ratio:.6g}_png"] = str(path)

    for ratio in ratios:
        group = sorted([r for r in data if r.pump_current_ratio == ratio], key=lambda r: r.n_cells)
        x = np.asarray([r.n_cells for r in group], dtype=float)
        y = []
        for r in group:
            value = r.metrics.get("max_current_ratio")
            y.append(np.nan if value is None else float(value))
        y_arr = np.asarray(y, dtype=float)

        if np.all(np.isnan(y_arr)):
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        ax.plot(x, y_arr, marker="o")
        ax.set_xlabel("Number of cells")
        ax.set_ylabel("max |I_pump| / I*")
        ax.set_title(f"Pump-current ratio diagnostic, input ratio = {ratio:g}")
        ax.grid(True)
        fig.tight_layout()

        path = output_dir / f"scaling_current_ratio_{ratio:.6g}.png".replace(".", "p")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths[f"current_ratio_{ratio:.6g}_png"] = str(path)

    return paths


def result_markdown(result: ScalingStudyResult, aggregate: Mapping[str, Any]) -> str:
    lines = [
        "# Pump-HB scaling study",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- runs: `{result.n_runs}`",
        f"- pass/partial/fail: `{result.n_pass}/{result.n_partial}/{result.n_fail}`",
        f"- sizes: `{list(result.config.sizes)}`",
        f"- pump-current ratios: `{list(result.config.pump_current_ratios)}`",
        f"- solver mode: `{result.config.solver_mode}`",
        "",
        "## Runs",
        "",
        "| run | cells | pump ratio | status | return code | elapsed s | solver | max current ratio | run dir |",
        "|---:|---:|---:|---|---:|---:|---|---:|---|",
    ]

    for run in result.runs:
        solver = run.metrics.get("solver_function", "")
        max_ratio = run.metrics.get("max_current_ratio", "")
        lines.append(
            f"| {run.run_index} | {run.n_cells} | {run.pump_current_ratio:.6g} | "
            f"`{run.status.value}` | {run.returncode} | {run.elapsed_s:.6g} | "
            f"`{solver}` | {max_ratio} | `{run.run_dir}` |"
        )

    lines += [
        "",
        "## Aggregate by size",
        "",
        "| cells | runs | pass | partial | fail | mean elapsed s | mean pump-stage s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for size, row in aggregate.get("by_size", {}).items():
        lines.append(
            f"| {size} | {row.get('n')} | {row.get('n_pass')} | "
            f"{row.get('n_partial')} | {row.get('n_fail')} | "
            f"{row.get('elapsed_s_mean')} | {row.get('pump_stage_elapsed_s_mean')} |"
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

    failed = [r for r in result.runs if r.status in {RunStatus.FAIL, RunStatus.ERROR}]
    if failed:
        lines += [
            "",
            "## Failed/error runs",
            "",
        ]
        for run in failed:
            lines += [
                f"### Run {run.run_index}: n={run.n_cells}, ratio={run.pump_current_ratio:g}",
                "",
                *[f"- {m}" for m in run.messages],
                "",
                f"- stdout: `{run.stdout_path}`",
                f"- stderr: `{run.stderr_path}`",
                "",
            ]

    return "\n".join(lines)


def export_artifacts(
    *,
    config: ScalingStudyConfig,
    runs: Sequence[ScalingRunResult],
    output_dir: Path,
    elapsed_s: float,
    metadata: Mapping[str, Any],
) -> ScalingStudyResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_scaling(runs)
    artifacts: dict[str, str] = {}

    runs_csv = write_runs_csv(output_dir / "pump_hb_scaling_runs.csv", runs)
    artifacts["runs_csv"] = str(runs_csv)

    aggregate_json = output_dir / "pump_hb_scaling_aggregate.json"
    aggregate_json.write_text(json.dumps(jsonify(aggregate), indent=2), encoding="utf-8")
    artifacts["aggregate_json"] = str(aggregate_json)

    plot_paths = write_plots(output_dir, runs)
    artifacts.update(plot_paths)

    hard_fail = any(r.status in {RunStatus.FAIL, RunStatus.ERROR} for r in runs)
    partial = any(r.status == RunStatus.PARTIAL for r in runs)

    if hard_fail:
        status = RunStatus.ERROR
    elif partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.PASS

    summary_json = output_dir / "pump_hb_scaling_summary.json"
    summary_md = output_dir / "pump_hb_scaling_summary.md"

    artifacts["summary_json"] = str(summary_json)
    artifacts["summary_md"] = str(summary_md)

    result = ScalingStudyResult(
        config=config,
        status=status,
        elapsed_s=elapsed_s,
        runs=tuple(runs),
        artifact_paths=artifacts,
        metadata=metadata,
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    summary_md.write_text(result_markdown(result, aggregate), encoding="utf-8")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a pump-HB scaling study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--pump-current-ratios", type=float, nargs="+", default=[0.02, 0.05, 0.08])
    parser.add_argument("--harmonic-orders", type=int, nargs="+", default=[-3, -1, 1, 3])

    parser.add_argument("--n-time", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--continuation-steps", type=int, default=8)
    parser.add_argument(
        "--solver-mode",
        choices=["auto", "package", "fallback_linear_pump"],
        default="auto",
    )

    parser.add_argument("--length-mm", type=float, default=1.0)
    parser.add_argument(
        "--scale-length-with-cells",
        action="store_true",
        help="Use length_mm = n_cells * cell_length_um / 1000 for each run.",
    )
    parser.add_argument("--cell-length-um", type=float, default=10.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)
    parser.add_argument("--pump-frequency-ghz", type=float, default=10.0)
    parser.add_argument("--pump-phase-rad", type=float, default=0.0)
    parser.add_argument("--i-star-a", type=float, default=5e-3)
    parser.add_argument("--l0-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-beta", type=float, default=1.0)

    parser.add_argument("--layout-csv", type=str, default=None)
    parser.add_argument("--layout-kind", type=str, default="uniform")
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument("--disorder-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--script-path",
        type=str,
        default=str(Path("scripts") / "pump_hb_small_ladder.py"),
        help="Path to pump_hb_small_ladder.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pump_hb_scaling"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="pump_hb_scaling")
    parser.add_argument("--python-executable", type=str, default=sys.executable)

    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--per-run-plots", action="store_true")
    parser.add_argument("--per-run-checkpoint", action="store_true")
    parser.add_argument("--export-netlist", action="store_true")

    return parser


def resolve_config(args: argparse.Namespace) -> ScalingStudyConfig:
    sizes = tuple(int(x) for x in args.sizes)
    ratios = tuple(float(x) for x in args.pump_current_ratios)
    harmonic_orders = tuple(int(x) for x in args.harmonic_orders)

    if args.quick:
        sizes = tuple(x for x in sizes if x <= 64) or (16, 32)
        ratios = ratios[:2] if len(ratios) > 2 else ratios
        n_time = min(int(args.n_time), 32)
        max_iter = min(int(args.max_iter), 20)
        continuation_steps = min(int(args.continuation_steps), 4)
    else:
        n_time = int(args.n_time)
        max_iter = int(args.max_iter)
        continuation_steps = int(args.continuation_steps)

    if not sizes:
        raise ValueError("--sizes may not be empty")
    if any(s <= 0 for s in sizes):
        raise ValueError("All --sizes must be positive")
    if not ratios:
        raise ValueError("--pump-current-ratios may not be empty")
    if any(r < 0.0 for r in ratios):
        raise ValueError("All --pump-current-ratios must be non-negative")
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
    if args.length_mm <= 0.0:
        raise ValueError("--length-mm must be positive")
    if args.cell_length_um <= 0.0:
        raise ValueError("--cell-length-um must be positive")
    if args.z0_ohm <= 0.0:
        raise ValueError("--z0-ohm must be positive")
    if args.phase_velocity_m_per_s <= 0.0:
        raise ValueError("--phase-velocity-m-per-s must be positive")
    if args.pump_frequency_ghz <= 0.0:
        raise ValueError("--pump-frequency-ghz must be positive")
    if args.i_star_a <= 0.0:
        raise ValueError("--i-star-a must be positive")
    if args.l0_scale <= 0.0:
        raise ValueError("--l0-scale must be positive")
    if args.disorder_std < 0.0:
        raise ValueError("--disorder-std must be non-negative")

    script_path = Path(args.script_path)
    if not script_path.exists:
        # Keep this as a runtime warning rather than a ValueError because the
        # subprocess error is more informative if the path is relative to a
        # different working directory.
        pass

    return ScalingStudyConfig(
        sizes=sizes,
        pump_current_ratios=ratios,
        harmonic_orders=harmonic_orders,
        n_time=n_time,
        max_iter=max_iter,
        tolerance=float(args.tolerance),
        damping=float(args.damping),
        continuation_steps=continuation_steps,
        solver_mode=str(args.solver_mode),
        length_mm=float(args.length_mm),
        scale_length_with_cells=bool(args.scale_length_with_cells),
        cell_length_um=float(args.cell_length_um),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        pump_frequency_ghz=float(args.pump_frequency_ghz),
        pump_phase_rad=float(args.pump_phase_rad),
        i_star_a=float(args.i_star_a),
        l0_scale=float(args.l0_scale),
        nonlinear_beta=float(args.nonlinear_beta),
        layout_csv=args.layout_csv,
        layout_kind=str(args.layout_kind),
        include_resonators=bool(args.include_resonators),
        disorder_std=float(args.disorder_std),
        seed=int(args.seed),
        script_path=str(args.script_path),
        output_dir=str(args.output_dir),
        name=str(args.name),
        quick=bool(args.quick),
        fail_fast=bool(args.fail_fast),
        per_run_plots=bool(args.per_run_plots),
        per_run_checkpoint=bool(args.per_run_checkpoint),
        export_netlist=bool(args.export_netlist),
        python_executable=str(args.python_executable),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[pump-hb-scaling] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "python": sys.version,
        "script": "scripts/pump_hb_scaling_study.py",
        "cwd": str(Path.cwd()),
    }

    runs: list[ScalingRunResult] = []
    run_index = 0

    try:
        for n_cells in config.sizes:
            for ratio in config.pump_current_ratios:
                print(f"[pump-hb-scaling] run {run_index}: n={n_cells}, ratio={ratio:g}")

                run = run_one(
                    config=config,
                    run_index=run_index,
                    n_cells=n_cells,
                    pump_current_ratio=ratio,
                    output_dir=output_dir,
                )
                runs.append(run)

                print(
                    f"[pump-hb-scaling] run {run_index}: "
                    f"{run.status.value}, returncode={run.returncode}, elapsed={run.elapsed_s:.3f}s"
                )

                if config.fail_fast and run.status in {RunStatus.FAIL, RunStatus.ERROR}:
                    print("[pump-hb-scaling] fail-fast triggered.")
                    raise RuntimeError(
                        f"Run {run_index} failed with status={run.status.value}, returncode={run.returncode}"
                    )

                run_index += 1

    except Exception:
        error_path = output_dir / "pump_hb_scaling_orchestrator_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        metadata["orchestrator_error_txt"] = str(error_path)

    elapsed_s = time.perf_counter() - start

    result = export_artifacts(
        config=config,
        runs=runs,
        output_dir=output_dir,
        elapsed_s=elapsed_s,
        metadata=metadata,
    )

    print()
    print(f"[pump-hb-scaling] status: {result.status.value}")
    print(f"[pump-hb-scaling] runs: {result.n_runs}")
    print(f"[pump-hb-scaling] summary JSON: {result.artifact_paths.get('summary_json')}")
    print(f"[pump-hb-scaling] summary MD:   {result.artifact_paths.get('summary_md')}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())
