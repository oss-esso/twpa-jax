"""
Run a full 100 mm TWPA gain-map workflow.

This script maps small-signal gain over pump operating points:

    pump frequency × pump current ratio × signal frequency.

It uses two stages for each pump point:

    1. Solve or load the pumped steady state using the full 100 mm pump-HB stack.
    2. Compute the signal/idler gain sweep from that pumped solution.

The script is intentionally orchestration-heavy. It delegates the actual pump solve
to ``scripts/full_pump_hb_100mm.py`` and the small-signal gain calculation to
``scripts/gain_from_pumped_solution.py``. This keeps the production gain map
reproducible and inspectable one pump point at a time.

Examples
--------
Quick smoke test:

    python scripts/full_gain_map_100mm.py --quick --output-dir outputs/full_gain_map_quick

Small operating map:

    python scripts/full_gain_map_100mm.py ^
      --pump-frequency-ghz-values 9.8 10.0 10.2 ^
      --pump-current-ratios 0.05 0.08 0.10 ^
      --signal-f-min-ghz 4 ^
      --signal-f-max-ghz 8 ^
      --n-signal 81 ^
      --output-dir outputs/full_gain_map_100mm

Use an existing layout CSV:

    python scripts/full_gain_map_100mm.py ^
      --layout-csv outputs/linear_100mm_baseline/linear_100mm_layout_components.csv ^
      --output-dir outputs/full_gain_map_from_csv

Use fallback-only physics for a fast dry run:

    python scripts/full_gain_map_100mm.py ^
      --quick ^
      --pump-solver-mode fallback_linear_pump ^
      --gain-solver-mode fallback_coupled_mode ^
      --output-dir outputs/full_gain_map_fallback
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
class FullGainMap100mmConfig:
    n_cells: int
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float
    layout_csv: str | None

    pump_frequency_ghz_values: tuple[float, ...]
    pump_current_ratios: tuple[float, ...]
    pump_phase_rad: float
    i_star_a: float
    l0_scale: float
    nonlinear_beta: float

    signal_f_min_ghz: float
    signal_f_max_ghz: float
    n_signal: int
    signal_current_rms_a: float

    harmonic_orders: tuple[int, ...]
    n_time: int
    max_iter: int
    tolerance: float
    damping: float
    continuation_steps: int

    pump_solver_mode: str
    gain_solver_mode: str
    orchestration_mode: str
    require_package_gain_solver: bool
    allow_partial_pump_fallback: bool

    warmup_cells: tuple[int, ...]
    run_warmup: bool

    layout_kind: str
    include_resonators: bool
    disorder_std: float
    seed: int

    output_dir: str
    name: str
    quick: bool
    fail_fast: bool

    keep_per_point_plots: bool
    keep_per_point_checkpoints: bool
    export_per_point_netlist: bool
    export_profile_csv: bool
    make_summary_plots: bool

    full_pump_script: str
    gain_script: str
    python_executable: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GainMapPointResult:
    point_index: int
    pump_frequency_ghz: float
    pump_current_ratio: float
    status: RunStatus
    elapsed_s: float
    point_dir: str

    pump_status: RunStatus
    pump_returncode: int
    pump_summary_path: str | None
    pump_arrays_npz: str | None
    pump_layout_csv: str | None
    pump_stdout_path: str
    pump_stderr_path: str
    pump_summary: Mapping[str, Any]

    gain_status: RunStatus
    gain_returncode: int
    gain_summary_path: str | None
    gain_arrays_npz: str | None
    gain_csv: str | None
    gain_stdout_path: str
    gain_stderr_path: str
    gain_summary: Mapping[str, Any]

    metrics: Mapping[str, Any]
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_index": self.point_index,
            "pump_frequency_ghz": self.pump_frequency_ghz,
            "pump_current_ratio": self.pump_current_ratio,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "point_dir": self.point_dir,
            "pump_status": self.pump_status.value,
            "pump_returncode": self.pump_returncode,
            "pump_summary_path": self.pump_summary_path,
            "pump_arrays_npz": self.pump_arrays_npz,
            "pump_layout_csv": self.pump_layout_csv,
            "pump_stdout_path": self.pump_stdout_path,
            "pump_stderr_path": self.pump_stderr_path,
            "pump_summary": jsonify(self.pump_summary),
            "gain_status": self.gain_status.value,
            "gain_returncode": self.gain_returncode,
            "gain_summary_path": self.gain_summary_path,
            "gain_arrays_npz": self.gain_arrays_npz,
            "gain_csv": self.gain_csv,
            "gain_stdout_path": self.gain_stdout_path,
            "gain_stderr_path": self.gain_stderr_path,
            "gain_summary": jsonify(self.gain_summary),
            "metrics": jsonify(self.metrics),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class FullGainMap100mmResult:
    config: FullGainMap100mmConfig
    status: RunStatus
    elapsed_s: float
    points: tuple[GainMapPointResult, ...]
    artifact_paths: Mapping[str, str]
    metadata: Mapping[str, Any]

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_pass(self) -> int:
        return sum(1 for p in self.points if p.status == RunStatus.PASS)

    @property
    def n_partial(self) -> int:
        return sum(1 for p in self.points if p.status == RunStatus.PARTIAL)

    @property
    def n_error(self) -> int:
        return sum(1 for p in self.points if p.status in {RunStatus.FAIL, RunStatus.ERROR})

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "n_points": self.n_points,
            "n_pass": self.n_pass,
            "n_partial": self.n_partial,
            "n_error": self.n_error,
            "config": self.config.to_dict(),
            "points": [p.to_dict() for p in self.points],
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
        if isinstance(stage, Mapping) and stage.get("name") == stage_name:
            return stage
    return None


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "json_parse_error": f"{type(exc).__name__}: {exc}",
            "path": str(path),
        }


def status_from_summary_and_returncode(
    *,
    returncode: int,
    summary: Mapping[str, Any],
) -> RunStatus:
    summary_status = summary.get("status")
    if returncode == 0 and summary_status == RunStatus.PASS.value:
        return RunStatus.PASS
    if returncode == 0 and summary_status == RunStatus.PARTIAL.value:
        return RunStatus.PARTIAL
    if returncode == 0 and not summary:
        return RunStatus.ERROR
    if returncode == 0:
        return RunStatus.PARTIAL
    return RunStatus.ERROR


def run_subprocess(
    *,
    cmd: Sequence[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(cwd))

    start = time.perf_counter()
    proc = subprocess.run(
        list(cmd),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed_s = time.perf_counter() - start

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    return int(proc.returncode), elapsed_s


def point_name(index: int, pump_frequency_ghz: float, pump_current_ratio: float) -> str:
    return (
        f"point_{index:04d}_fp_{pump_frequency_ghz:.9g}_ir_{pump_current_ratio:.9g}"
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
    )


def pump_command_for_point(
    *,
    config: FullGainMap100mmConfig,
    point_dir: Path,
    pump_frequency_ghz: float,
    pump_current_ratio: float,
) -> list[str]:
    pump_dir = point_dir / "pump"

    cmd = [
        config.python_executable,
        config.full_pump_script,
        "--n-cells",
        str(config.n_cells),
        "--length-mm",
        f"{config.length_mm:.16g}",
        "--z0-ohm",
        f"{config.z0_ohm:.16g}",
        "--phase-velocity-m-per-s",
        f"{config.phase_velocity_m_per_s:.16g}",
        "--pump-frequency-ghz",
        f"{pump_frequency_ghz:.16g}",
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
        config.pump_solver_mode,
        "--layout-kind",
        config.layout_kind,
        "--disorder-std",
        f"{config.disorder_std:.16g}",
        "--seed",
        str(config.seed),
        "--output-dir",
        str(pump_dir),
        "--name",
        f"{config.name}_pump_{pump_frequency_ghz:.6g}_{pump_current_ratio:.6g}".replace(".", "p"),
        "--harmonic-orders",
        *[str(h) for h in config.harmonic_orders],
    ]

    if config.layout_csv is not None:
        cmd.extend(["--layout-csv", config.layout_csv])

    if config.include_resonators:
        cmd.append("--include-resonators")

    if config.run_warmup and config.warmup_cells:
        cmd.append("--run-warmup")
        cmd.extend(["--warmup-cells", *[str(x) for x in config.warmup_cells]])

    if not config.allow_partial_pump_fallback:
        cmd.append("--no-allow-partial-fallback")

    if not config.keep_per_point_plots:
        cmd.append("--no-plots")

    if not config.keep_per_point_checkpoints:
        cmd.append("--no-checkpoint")

    if config.export_per_point_netlist:
        cmd.append("--export-netlist")

    if not config.export_profile_csv:
        cmd.append("--no-profile-csv")

    return cmd


def gain_command_for_point(
    *,
    config: FullGainMap100mmConfig,
    point_dir: Path,
    pump_arrays_npz: Path,
    layout_csv: Path | None,
    pump_frequency_ghz: float,
) -> list[str]:
    gain_dir = point_dir / "gain"

    cmd = [
        config.python_executable,
        config.gain_script,
        "--pump-npz",
        str(pump_arrays_npz),
        "--pump-frequency-ghz",
        f"{pump_frequency_ghz:.16g}",
        "--signal-f-min-ghz",
        f"{config.signal_f_min_ghz:.16g}",
        "--signal-f-max-ghz",
        f"{config.signal_f_max_ghz:.16g}",
        "--n-signal",
        str(config.n_signal),
        "--z0-ohm",
        f"{config.z0_ohm:.16g}",
        "--length-mm",
        f"{config.length_mm:.16g}",
        "--phase-velocity-m-per-s",
        f"{config.phase_velocity_m_per_s:.16g}",
        "--i-star-a",
        f"{config.i_star_a:.16g}",
        "--nonlinear-beta",
        f"{config.nonlinear_beta:.16g}",
        "--signal-current-rms-a",
        f"{config.signal_current_rms_a:.16g}",
        "--solver-mode",
        config.gain_solver_mode,
        "--output-dir",
        str(gain_dir),
        "--name",
        f"{config.name}_gain_{pump_frequency_ghz:.6g}".replace(".", "p"),
    ]

    if layout_csv is not None:
        cmd.extend(["--layout-csv", str(layout_csv)])

    if config.require_package_gain_solver:
        cmd.append("--require-package-solver")

    if not config.keep_per_point_plots:
        cmd.append("--no-plots")

    if not config.keep_per_point_checkpoints:
        cmd.append("--no-checkpoint")

    return cmd


def extract_artifact_path(summary: Mapping[str, Any], key: str) -> str | None:
    path = nested_get(summary, ["artifact_paths", key])
    if path is None:
        return None
    return str(path)


def extract_gain_metrics(
    *,
    pump_summary: Mapping[str, Any],
    gain_summary: Mapping[str, Any],
    pump_frequency_ghz: float,
    pump_current_ratio: float,
) -> dict[str, Any]:
    gain_stage = find_stage(gain_summary, "gain_solve") or {}
    gain_stage_summary = gain_stage.get("summary", {}) if isinstance(gain_stage, Mapping) else {}

    pump_stage = find_stage(pump_summary, "full_pump_solve") or {}
    pump_stage_summary = pump_stage.get("summary", {}) if isinstance(pump_stage, Mapping) else {}

    pump_summary_json = extract_artifact_path(pump_summary, "pump_summary_json")
    gain_summary_json = extract_artifact_path(gain_summary, "gain_summary_json")

    external_gain_summary: dict[str, Any] = {}
    if gain_summary_json:
        external_gain_summary = read_json_if_exists(Path(gain_summary_json))

    max_gain_db = (
        gain_stage_summary.get("max_gain_db")
        if gain_stage_summary.get("max_gain_db") is not None
        else external_gain_summary.get("max_gain_db")
    )
    max_gain_frequency_hz = (
        gain_stage_summary.get("max_gain_frequency_hz")
        if gain_stage_summary.get("max_gain_frequency_hz") is not None
        else external_gain_summary.get("max_gain_frequency_hz")
    )

    return {
        "pump_frequency_ghz": pump_frequency_ghz,
        "pump_current_ratio": pump_current_ratio,
        "pump_status": pump_summary.get("status"),
        "gain_status": gain_summary.get("status"),
        "pump_solver_function": pump_stage_summary.get("solver_function"),
        "gain_solver_function": gain_stage_summary.get("solver_function")
        if gain_stage_summary.get("solver_function") is not None
        else external_gain_summary.get("solver_function"),
        "pump_elapsed_s": pump_summary.get("elapsed_s"),
        "gain_elapsed_s": gain_summary.get("elapsed_s"),
        "pump_stage_elapsed_s": pump_stage.get("elapsed_s"),
        "gain_stage_elapsed_s": gain_stage.get("elapsed_s"),
        "pump_max_current_ratio": pump_stage_summary.get("max_current_ratio"),
        "pump_output_input_current_ratio": pump_stage_summary.get("pump_output_input_current_ratio"),
        "max_gain_db": max_gain_db,
        "max_gain_frequency_hz": max_gain_frequency_hz,
        "max_gain_frequency_ghz": None if max_gain_frequency_hz is None else float(max_gain_frequency_hz) / 1e9,
        "gain_finite": gain_stage_summary.get("finite")
        if gain_stage_summary.get("finite") is not None
        else external_gain_summary.get("finite"),
        "gain_n_points": gain_stage_summary.get("n_points")
        if gain_stage_summary.get("n_points") is not None
        else external_gain_summary.get("n_points"),
        "pump_arrays_npz": extract_artifact_path(pump_summary, "arrays_npz"),
        "gain_arrays_npz": extract_artifact_path(gain_summary, "arrays_npz"),
        "gain_csv": extract_artifact_path(gain_summary, "gain_csv"),
    }


def run_gain_map_point(
    *,
    config: FullGainMap100mmConfig,
    point_index: int,
    pump_frequency_ghz: float,
    pump_current_ratio: float,
    output_dir: Path,
) -> GainMapPointResult:
    point_start = time.perf_counter()
    name = point_name(point_index, pump_frequency_ghz, pump_current_ratio)
    point_dir = output_dir / "points" / name
    point_dir.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []

    pump_stdout = point_dir / "pump_stdout.txt"
    pump_stderr = point_dir / "pump_stderr.txt"
    pump_cmd = pump_command_for_point(
        config=config,
        point_dir=point_dir,
        pump_frequency_ghz=pump_frequency_ghz,
        pump_current_ratio=pump_current_ratio,
    )

    pump_returncode, pump_elapsed_s = run_subprocess(
        cmd=pump_cmd,
        cwd=Path.cwd(),
        stdout_path=pump_stdout,
        stderr_path=pump_stderr,
    )

    pump_summary_path = point_dir / "pump" / "full_pump_hb_100mm_summary.json"
    pump_summary = read_json_if_exists(pump_summary_path)
    pump_status = status_from_summary_and_returncode(
        returncode=pump_returncode,
        summary=pump_summary,
    )

    pump_arrays = extract_artifact_path(pump_summary, "arrays_npz")
    pump_layout = extract_artifact_path(pump_summary, "layout_components_csv")

    if pump_status in {RunStatus.FAIL, RunStatus.ERROR} or pump_arrays is None:
        if pump_arrays is None:
            messages.append("ERROR: pump arrays NPZ was not produced.")
        messages.append(
            f"ERROR: pump stage failed with status={pump_status.value}, returncode={pump_returncode}."
        )

        elapsed_s = time.perf_counter() - point_start
        return GainMapPointResult(
            point_index=point_index,
            pump_frequency_ghz=pump_frequency_ghz,
            pump_current_ratio=pump_current_ratio,
            status=RunStatus.ERROR,
            elapsed_s=elapsed_s,
            point_dir=str(point_dir),
            pump_status=pump_status,
            pump_returncode=pump_returncode,
            pump_summary_path=str(pump_summary_path) if pump_summary_path.exists() else None,
            pump_arrays_npz=pump_arrays,
            pump_layout_csv=pump_layout,
            pump_stdout_path=str(pump_stdout),
            pump_stderr_path=str(pump_stderr),
            pump_summary=pump_summary,
            gain_status=RunStatus.ERROR,
            gain_returncode=-1,
            gain_summary_path=None,
            gain_arrays_npz=None,
            gain_csv=None,
            gain_stdout_path=str(point_dir / "gain_stdout.txt"),
            gain_stderr_path=str(point_dir / "gain_stderr.txt"),
            gain_summary={},
            metrics={},
            messages=tuple(messages),
        )

    gain_stdout = point_dir / "gain_stdout.txt"
    gain_stderr = point_dir / "gain_stderr.txt"

    gain_cmd = gain_command_for_point(
        config=config,
        point_dir=point_dir,
        pump_arrays_npz=Path(pump_arrays),
        layout_csv=None if pump_layout is None else Path(pump_layout),
        pump_frequency_ghz=pump_frequency_ghz,
    )

    gain_returncode, gain_elapsed_s = run_subprocess(
        cmd=gain_cmd,
        cwd=Path.cwd(),
        stdout_path=gain_stdout,
        stderr_path=gain_stderr,
    )

    gain_summary_path = point_dir / "gain" / "gain_from_pumped_solution_summary.json"
    gain_summary = read_json_if_exists(gain_summary_path)
    gain_status = status_from_summary_and_returncode(
        returncode=gain_returncode,
        summary=gain_summary,
    )

    gain_arrays = extract_artifact_path(gain_summary, "arrays_npz")
    gain_csv = extract_artifact_path(gain_summary, "gain_csv")

    metrics = extract_gain_metrics(
        pump_summary=pump_summary,
        gain_summary=gain_summary,
        pump_frequency_ghz=pump_frequency_ghz,
        pump_current_ratio=pump_current_ratio,
    )

    if gain_status == RunStatus.PASS and pump_status == RunStatus.PASS:
        status = RunStatus.PASS
        messages.append("PASS: pump and gain stages completed.")
    elif gain_status in {RunStatus.PASS, RunStatus.PARTIAL} and pump_status in {RunStatus.PASS, RunStatus.PARTIAL}:
        status = RunStatus.PARTIAL
        messages.append("PARTIAL: pump and gain completed, but at least one stage was partial.")
    else:
        status = RunStatus.ERROR
        messages.append(
            f"ERROR: gain stage failed with status={gain_status.value}, returncode={gain_returncode}."
        )

    elapsed_s = time.perf_counter() - point_start

    return GainMapPointResult(
        point_index=point_index,
        pump_frequency_ghz=pump_frequency_ghz,
        pump_current_ratio=pump_current_ratio,
        status=status,
        elapsed_s=elapsed_s,
        point_dir=str(point_dir),
        pump_status=pump_status,
        pump_returncode=pump_returncode,
        pump_summary_path=str(pump_summary_path) if pump_summary_path.exists() else None,
        pump_arrays_npz=pump_arrays,
        pump_layout_csv=pump_layout,
        pump_stdout_path=str(pump_stdout),
        pump_stderr_path=str(pump_stderr),
        pump_summary=pump_summary,
        gain_status=gain_status,
        gain_returncode=gain_returncode,
        gain_summary_path=str(gain_summary_path) if gain_summary_path.exists() else None,
        gain_arrays_npz=gain_arrays,
        gain_csv=gain_csv,
        gain_stdout_path=str(gain_stdout),
        gain_stderr_path=str(gain_stderr),
        gain_summary=gain_summary,
        metrics=metrics,
        messages=tuple(messages),
    )


def write_points_csv(path: Path, points: Sequence[GainMapPointResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys: list[str] = []
    for point in points:
        for key in point.metrics.keys():
            if key not in metric_keys:
                metric_keys.append(key)

    fields = [
        "point_index",
        "pump_frequency_ghz",
        "pump_current_ratio",
        "status",
        "elapsed_s",
        "pump_status",
        "pump_returncode",
        "gain_status",
        "gain_returncode",
        "point_dir",
        *metric_keys,
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for point in points:
            row = {
                "point_index": point.point_index,
                "pump_frequency_ghz": point.pump_frequency_ghz,
                "pump_current_ratio": point.pump_current_ratio,
                "status": point.status.value,
                "elapsed_s": point.elapsed_s,
                "pump_status": point.pump_status.value,
                "pump_returncode": point.pump_returncode,
                "gain_status": point.gain_status.value,
                "gain_returncode": point.gain_returncode,
                "point_dir": point.point_dir,
            }

            for key in metric_keys:
                value = point.metrics.get(key)
                if isinstance(value, (list, tuple, dict)):
                    row[key] = json.dumps(jsonify(value))
                else:
                    row[key] = value

            writer.writerow(row)

    return path


def load_gain_curve_from_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    out = {
        "signal_frequency_hz": np.asarray(npz["signal_frequency_hz"], dtype=float),
        "signal_gain_db": np.asarray(npz["signal_gain_db"], dtype=float),
    }
    if "idler_conversion_db" in npz:
        out["idler_conversion_db"] = np.asarray(npz["idler_conversion_db"], dtype=float)
    return out


def build_gain_cube(
    *,
    config: FullGainMap100mmConfig,
    points: Sequence[GainMapPointResult],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pump_f = np.asarray(config.pump_frequency_ghz_values, dtype=float)
    pump_r = np.asarray(config.pump_current_ratios, dtype=float)

    gain_cube = np.full(
        (pump_f.size, pump_r.size, config.n_signal),
        np.nan,
        dtype=float,
    )
    idler_cube = np.full_like(gain_cube, np.nan)
    point_status = np.full((pump_f.size, pump_r.size), "", dtype=object)
    point_elapsed_s = np.full((pump_f.size, pump_r.size), np.nan, dtype=float)
    signal_frequency_hz = None

    f_to_i = {float(v): i for i, v in enumerate(pump_f)}
    r_to_j = {float(v): j for j, v in enumerate(pump_r)}

    for point in points:
        point_status[
            f_to_i[float(point.pump_frequency_ghz)],
            r_to_j[float(point.pump_current_ratio)],
        ] = point.status.value
        point_elapsed_s[
            f_to_i[float(point.pump_frequency_ghz)],
            r_to_j[float(point.pump_current_ratio)],
        ] = point.elapsed_s

        if point.gain_arrays_npz is None or point.status in {RunStatus.FAIL, RunStatus.ERROR}:
            continue

        try:
            curve = load_gain_curve_from_npz(point.gain_arrays_npz)
        except Exception:
            continue

        fs = curve["signal_frequency_hz"]
        gain = curve["signal_gain_db"]

        if signal_frequency_hz is None:
            signal_frequency_hz = fs

        if gain.shape[0] != config.n_signal:
            continue

        i = f_to_i[float(point.pump_frequency_ghz)]
        j = r_to_j[float(point.pump_current_ratio)]

        gain_cube[i, j, :] = gain

        if "idler_conversion_db" in curve and curve["idler_conversion_db"].shape[0] == config.n_signal:
            idler_cube[i, j, :] = curve["idler_conversion_db"]

    if signal_frequency_hz is None:
        signal_frequency_hz = np.linspace(
            config.signal_f_min_ghz * 1e9,
            config.signal_f_max_ghz * 1e9,
            config.n_signal,
        )

    best_idx_flat = int(np.nanargmax(gain_cube)) if np.any(np.isfinite(gain_cube)) else -1
    if best_idx_flat >= 0:
        best_i, best_j, best_k = np.unravel_index(best_idx_flat, gain_cube.shape)
        best = {
            "max_gain_db": float(gain_cube[best_i, best_j, best_k]),
            "pump_frequency_ghz": float(pump_f[best_i]),
            "pump_current_ratio": float(pump_r[best_j]),
            "signal_frequency_hz": float(signal_frequency_hz[best_k]),
            "signal_frequency_ghz": float(signal_frequency_hz[best_k] / 1e9),
        }
    else:
        best = {
            "max_gain_db": None,
            "pump_frequency_ghz": None,
            "pump_current_ratio": None,
            "signal_frequency_hz": None,
            "signal_frequency_ghz": None,
        }

    aggregate = {
        "n_points": len(points),
        "n_pass": sum(1 for p in points if p.status == RunStatus.PASS),
        "n_partial": sum(1 for p in points if p.status == RunStatus.PARTIAL),
        "n_error": sum(1 for p in points if p.status in {RunStatus.FAIL, RunStatus.ERROR}),
        "best": best,
        "pump_frequency_ghz_values": pump_f.tolist(),
        "pump_current_ratios": pump_r.tolist(),
        "signal_frequency_hz": signal_frequency_hz.tolist(),
        "finite_gain_fraction": float(np.mean(np.isfinite(gain_cube))),
    }

    arrays = {
        "pump_frequency_ghz": pump_f,
        "pump_current_ratio": pump_r,
        "signal_frequency_hz": signal_frequency_hz,
        "gain_db": gain_cube,
        "idler_conversion_db": idler_cube,
        "point_status": point_status,
        "point_elapsed_s": point_elapsed_s,
    }

    return aggregate, arrays


def write_summary_plots(
    *,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
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

    pump_f = np.asarray(arrays["pump_frequency_ghz"], dtype=float)
    pump_r = np.asarray(arrays["pump_current_ratio"], dtype=float)
    signal_f = np.asarray(arrays["signal_frequency_hz"], dtype=float) / 1e9
    gain = np.asarray(arrays["gain_db"], dtype=float)

    if not np.any(np.isfinite(gain)):
        return paths

    max_gain_over_signal = np.nanmax(gain, axis=2)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    im = ax.imshow(
        max_gain_over_signal.T,
        origin="lower",
        aspect="auto",
        extent=[
            float(np.min(pump_f)),
            float(np.max(pump_f)),
            float(np.min(pump_r)),
            float(np.max(pump_r)),
        ],
    )
    ax.set_xlabel("Pump frequency (GHz)")
    ax.set_ylabel("Pump current ratio")
    ax.set_title("Max gain over signal band")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Max gain (dB)")
    fig.tight_layout()
    p = output_dir / "full_gain_map_max_gain_heatmap.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths["max_gain_heatmap_png"] = str(p)

    best = aggregate.get("best", {})
    best_fp = best.get("pump_frequency_ghz")
    best_ir = best.get("pump_current_ratio")
    if best_fp is not None and best_ir is not None:
        i = int(np.argmin(np.abs(pump_f - float(best_fp))))
        j = int(np.argmin(np.abs(pump_r - float(best_ir))))

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        ax.plot(signal_f, gain[i, j, :], label="signal gain")
        ax.set_xlabel("Signal frequency (GHz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title(f"Best gain curve: fp={best_fp:g} GHz, Ip/I*={best_ir:g}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        p = output_dir / "full_gain_map_best_curve.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths["best_curve_png"] = str(p)

    for i, fp in enumerate(pump_f):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        for j, ratio in enumerate(pump_r):
            y = gain[i, j, :]
            if np.any(np.isfinite(y)):
                ax.plot(signal_f, y, label=f"Ip/I*={ratio:g}")
        ax.set_xlabel("Signal frequency (GHz)")
        ax.set_ylabel("Gain (dB)")
        ax.set_title(f"Gain curves at pump {fp:g} GHz")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        p = output_dir / f"full_gain_map_curves_fp_{fp:.6g}.png".replace(".", "p")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths[f"curves_fp_{fp:.6g}_png"] = str(p)

    return paths


def export_artifacts(
    *,
    config: FullGainMap100mmConfig,
    points: Sequence[GainMapPointResult],
    output_dir: Path,
    elapsed_s: float,
    metadata: Mapping[str, Any],
) -> FullGainMap100mmResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    points_csv = write_points_csv(output_dir / "full_gain_map_100mm_points.csv", points)
    artifacts["points_csv"] = str(points_csv)

    points_json = output_dir / "full_gain_map_100mm_points.json"
    points_json.write_text(
        json.dumps(jsonify([p.to_dict() for p in points]), indent=2),
        encoding="utf-8",
    )
    artifacts["points_json"] = str(points_json)

    aggregate, arrays = build_gain_cube(config=config, points=points)

    aggregate_json = output_dir / "full_gain_map_100mm_aggregate.json"
    aggregate_json.write_text(json.dumps(jsonify(aggregate), indent=2), encoding="utf-8")
    artifacts["aggregate_json"] = str(aggregate_json)

    cube_npz = output_dir / "full_gain_map_100mm_cube.npz"
    np.savez_compressed(
        cube_npz,
        pump_frequency_ghz=arrays["pump_frequency_ghz"],
        pump_current_ratio=arrays["pump_current_ratio"],
        signal_frequency_hz=arrays["signal_frequency_hz"],
        gain_db=arrays["gain_db"],
        idler_conversion_db=arrays["idler_conversion_db"],
        point_status=arrays["point_status"],
        point_elapsed_s=arrays["point_elapsed_s"],
        metadata_json=json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "aggregate": aggregate,
                }
            )
        ),
    )
    artifacts["gain_cube_npz"] = str(cube_npz)

    if config.make_summary_plots:
        artifacts.update(
            write_summary_plots(
                output_dir=output_dir,
                arrays=arrays,
                aggregate=aggregate,
            )
        )

    hard_fail = any(p.status in {RunStatus.FAIL, RunStatus.ERROR} for p in points)
    partial = any(p.status == RunStatus.PARTIAL for p in points)

    if hard_fail:
        status = RunStatus.ERROR
    elif partial:
        status = RunStatus.PARTIAL
    else:
        status = RunStatus.PASS

    summary_json = output_dir / "full_gain_map_100mm_summary.json"
    summary_md = output_dir / "full_gain_map_100mm_summary.md"

    artifacts["summary_json"] = str(summary_json)
    artifacts["summary_md"] = str(summary_md)

    result = FullGainMap100mmResult(
        config=config,
        status=status,
        elapsed_s=elapsed_s,
        points=tuple(points),
        artifact_paths=artifacts,
        metadata={**dict(metadata), "aggregate": aggregate},
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    summary_md.write_text(result_markdown(result, aggregate), encoding="utf-8")

    return result


def result_markdown(
    result: FullGainMap100mmResult,
    aggregate: Mapping[str, Any],
) -> str:
    cfg = result.config
    best = aggregate.get("best", {})

    lines = [
        "# Full gain map 100 mm",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- points: `{result.n_points}`",
        f"- pass/partial/error: `{result.n_pass}/{result.n_partial}/{result.n_error}`",
        f"- pump frequencies: `{list(cfg.pump_frequency_ghz_values)}` GHz",
        f"- pump current ratios: `{list(cfg.pump_current_ratios)}`",
        f"- signal range: `{cfg.signal_f_min_ghz:.6g}`–`{cfg.signal_f_max_ghz:.6g} GHz`",
        f"- signal points: `{cfg.n_signal}`",
        f"- pump solver mode: `{cfg.pump_solver_mode}`",
        f"- gain solver mode: `{cfg.gain_solver_mode}`",
        "",
        "## Best operating point",
        "",
        f"- max gain: `{best.get('max_gain_db')}` dB",
        f"- pump frequency: `{best.get('pump_frequency_ghz')}` GHz",
        f"- pump current ratio: `{best.get('pump_current_ratio')}`",
        f"- signal frequency: `{best.get('signal_frequency_ghz')}` GHz",
        "",
        "## Points",
        "",
        "| point | pump GHz | Ip/I* | status | max gain dB | best signal GHz | pump solver | gain solver | elapsed s |",
        "|---:|---:|---:|---|---:|---:|---|---|---:|",
    ]

    for point in result.points:
        m = point.metrics
        lines.append(
            f"| {point.point_index} | "
            f"{point.pump_frequency_ghz:.6g} | "
            f"{point.pump_current_ratio:.6g} | "
            f"`{point.status.value}` | "
            f"{m.get('max_gain_db', '')} | "
            f"{m.get('max_gain_frequency_ghz', '')} | "
            f"`{m.get('pump_solver_function', '')}` | "
            f"`{m.get('gain_solver_function', '')}` | "
            f"{point.elapsed_s:.6g} |"
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

    failed = [p for p in result.points if p.status in {RunStatus.FAIL, RunStatus.ERROR}]
    if failed:
        lines += [
            "",
            "## Failed/error points",
            "",
        ]
        for point in failed:
            lines += [
                f"### Point {point.point_index}: fp={point.pump_frequency_ghz:g} GHz, Ip/I*={point.pump_current_ratio:g}",
                "",
                *[f"- {m}" for m in point.messages],
                f"- pump stdout: `{point.pump_stdout_path}`",
                f"- pump stderr: `{point.pump_stderr_path}`",
                f"- gain stdout: `{point.gain_stdout_path}`",
                f"- gain stderr: `{point.gain_stderr_path}`",
                "",
            ]

    return "\n".join(lines)


def parse_float_list(values: Sequence[str | float]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a full 100 mm pump-frequency/pump-power gain map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)
    parser.add_argument("--layout-csv", type=str, default=None)

    parser.add_argument(
        "--pump-frequency-ghz-values",
        type=float,
        nargs="+",
        default=[9.8, 10.0, 10.2],
    )
    parser.add_argument(
        "--pump-current-ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.08, 0.10],
    )
    parser.add_argument("--pump-phase-rad", type=float, default=0.0)
    parser.add_argument("--i-star-a", type=float, default=5e-3)
    parser.add_argument("--l0-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-beta", type=float, default=1.0)

    parser.add_argument("--signal-f-min-ghz", type=float, default=4.0)
    parser.add_argument("--signal-f-max-ghz", type=float, default=8.0)
    parser.add_argument("--n-signal", type=int, default=81)
    parser.add_argument("--signal-current-rms-a", type=float, default=1e-12)

    parser.add_argument("--harmonic-orders", type=int, nargs="+", default=[-3, -1, 1, 3])
    parser.add_argument("--n-time", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--continuation-steps", type=int, default=10)

    parser.add_argument(
        "--pump-solver-mode",
        choices=["auto", "package", "fallback_linear_pump"],
        default="auto",
    )
    parser.add_argument(
        "--gain-solver-mode",
        choices=["auto", "package", "fallback_coupled_mode"],
        default="auto",
    )
    parser.add_argument(
        "--orchestration-mode",
        choices=["native", "legacy_gain_script"],
        default="native",
        help="Native package orchestration is the default. legacy_gain_script is retained as a PARTIAL compatibility route.",
    )
    parser.add_argument("--require-package-gain-solver", action="store_true")
    parser.add_argument("--no-allow-partial-pump-fallback", action="store_true")

    parser.add_argument("--warmup-cells", type=int, nargs="*", default=[])
    parser.add_argument("--run-warmup", action="store_true")

    parser.add_argument("--layout-kind", type=str, default="uniform")
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument("--disorder-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_gain_map_100mm"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="full_gain_map_100mm")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--keep-per-point-plots", action="store_true")
    parser.add_argument("--keep-per-point-checkpoints", action="store_true")
    parser.add_argument("--export-per-point-netlist", action="store_true")
    parser.add_argument("--no-profile-csv", action="store_true")
    parser.add_argument("--no-summary-plots", action="store_true")

    parser.add_argument(
        "--full-pump-script",
        type=str,
        default=str(Path("scripts") / "full_pump_hb_100mm.py"),
    )
    parser.add_argument(
        "--gain-script",
        type=str,
        default=str(Path("scripts") / "gain_from_pumped_solution.py"),
    )
    parser.add_argument("--python-executable", type=str, default=sys.executable)

    return parser


def resolve_config(args: argparse.Namespace) -> FullGainMap100mmConfig:
    n_cells = int(args.n_cells)
    n_signal = int(args.n_signal)
    n_time = int(args.n_time)
    max_iter = int(args.max_iter)
    continuation_steps = int(args.continuation_steps)

    pump_frequency_values = tuple(float(v) for v in args.pump_frequency_ghz_values)
    pump_current_ratios = tuple(float(v) for v in args.pump_current_ratios)
    harmonic_orders = tuple(int(v) for v in args.harmonic_orders)
    warmup_cells = tuple(int(v) for v in args.warmup_cells)

    if args.quick:
        if n_cells == 20000:
            n_cells = 2000
        pump_frequency_values = pump_frequency_values[:2] if len(pump_frequency_values) > 2 else pump_frequency_values
        pump_current_ratios = pump_current_ratios[:2] if len(pump_current_ratios) > 2 else pump_current_ratios
        n_signal = min(n_signal, 31)
        n_time = min(n_time, 32)
        max_iter = min(max_iter, 25)
        continuation_steps = min(continuation_steps, 5)
        if args.run_warmup and not warmup_cells:
            warmup_cells = (100, 250)

    warmup_cells = tuple(sorted(set(x for x in warmup_cells if x > 0 and x < n_cells)))

    if n_cells <= 0:
        raise ValueError("--n-cells must be positive")
    if args.length_mm <= 0.0:
        raise ValueError("--length-mm must be positive")
    if args.z0_ohm <= 0.0:
        raise ValueError("--z0-ohm must be positive")
    if args.phase_velocity_m_per_s <= 0.0:
        raise ValueError("--phase-velocity-m-per-s must be positive")
    if not pump_frequency_values or any(v <= 0.0 for v in pump_frequency_values):
        raise ValueError("All pump frequencies must be positive")
    if not pump_current_ratios or any(v < 0.0 for v in pump_current_ratios):
        raise ValueError("All pump current ratios must be non-negative")
    if args.i_star_a <= 0.0:
        raise ValueError("--i-star-a must be positive")
    if args.l0_scale <= 0.0:
        raise ValueError("--l0-scale must be positive")
    if args.signal_f_min_ghz <= 0.0:
        raise ValueError("--signal-f-min-ghz must be positive")
    if args.signal_f_max_ghz <= args.signal_f_min_ghz:
        raise ValueError("--signal-f-max-ghz must exceed --signal-f-min-ghz")
    if n_signal < 2:
        raise ValueError("--n-signal must be at least 2")
    if args.signal_current_rms_a <= 0.0:
        raise ValueError("--signal-current-rms-a must be positive")
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

    return FullGainMap100mmConfig(
        n_cells=n_cells,
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        layout_csv=args.layout_csv,
        pump_frequency_ghz_values=pump_frequency_values,
        pump_current_ratios=pump_current_ratios,
        pump_phase_rad=float(args.pump_phase_rad),
        i_star_a=float(args.i_star_a),
        l0_scale=float(args.l0_scale),
        nonlinear_beta=float(args.nonlinear_beta),
        signal_f_min_ghz=float(args.signal_f_min_ghz),
        signal_f_max_ghz=float(args.signal_f_max_ghz),
        n_signal=n_signal,
        signal_current_rms_a=float(args.signal_current_rms_a),
        harmonic_orders=harmonic_orders,
        n_time=n_time,
        max_iter=max_iter,
        tolerance=float(args.tolerance),
        damping=float(args.damping),
        continuation_steps=continuation_steps,
        pump_solver_mode=str(args.pump_solver_mode),
        gain_solver_mode=str(args.gain_solver_mode),
        orchestration_mode=str(args.orchestration_mode),
        require_package_gain_solver=bool(args.require_package_gain_solver),
        allow_partial_pump_fallback=not bool(args.no_allow_partial_pump_fallback),
        warmup_cells=warmup_cells,
        run_warmup=bool(args.run_warmup or warmup_cells),
        layout_kind=str(args.layout_kind),
        include_resonators=bool(args.include_resonators),
        disorder_std=float(args.disorder_std),
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        name=str(args.name),
        quick=bool(args.quick),
        fail_fast=bool(args.fail_fast),
        keep_per_point_plots=bool(args.keep_per_point_plots),
        keep_per_point_checkpoints=bool(args.keep_per_point_checkpoints),
        export_per_point_netlist=bool(args.export_per_point_netlist),
        export_profile_csv=not bool(args.no_profile_csv),
        make_summary_plots=not bool(args.no_summary_plots),
        full_pump_script=str(args.full_pump_script),
        gain_script=str(args.gain_script),
        python_executable=str(args.python_executable),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.perf_counter()

    try:
        config = resolve_config(args)
    except Exception as exc:
        print(f"[full-gain-map-100mm] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.orchestration_mode == "native":
        try:
            try:
                from scripts.compression_sweep import build_or_load_native_layout
            except ModuleNotFoundError:
                from compression_sweep import build_or_load_native_layout
            from twpa.core.hb_fft import HBProjectionConfig
            from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
            from twpa.nonlinear.pump_hb_ladder import PumpHBLadderConfig
            from twpa.workflows.gain_map import (
                export_native_gain_map_artifacts,
                solve_native_gain_map,
            )

            layout, layout_info = build_or_load_native_layout(config)
            positive_orders = [abs(h) for h in config.harmonic_orders if h > 0]
            pump_config = PumpHBLadderConfig(
                n_pump_harmonics=max(positive_orders) if positive_orders else 1,
                include_negative_frequencies=any(h < 0 for h in config.harmonic_orders),
                include_dc=any(h == 0 for h in config.harmonic_orders),
                projection=HBProjectionConfig(n_time_samples=config.n_time),
                solver=SolverConfig(
                    backend=SolverBackend.NEWTON_KRYLOV,
                    max_iter=config.max_iter,
                    abs_tol=config.tolerance,
                    rel_tol=config.tolerance,
                    damping_initial=config.damping,
                    verbose=False,
                ),
            )
            native = solve_native_gain_map(
                layout,
                NonlinearParams(I_star_A=config.i_star_a, beta_nl=config.nonlinear_beta),
                pump_frequencies_hz=[value * 1e9 for value in config.pump_frequency_ghz_values],
                pump_current_ratios=config.pump_current_ratios,
                signal_frequencies_hz=np.linspace(
                    config.signal_f_min_ghz * 1e9,
                    config.signal_f_max_ghz * 1e9,
                    config.n_signal,
                ),
                pump_config=pump_config,
                source_impedance_ohm=config.z0_ohm,
                pump_phase_rad=config.pump_phase_rad,
                signal_current_rms_A=complex(config.signal_current_rms_a),
                metadata={"layout": layout_info, "script": "scripts/full_gain_map_100mm.py"},
            )
            artifacts = export_native_gain_map_artifacts(native, output_dir)
            print(f"[full-gain-map-100mm] status: {native.status}")
            print(f"[full-gain-map-100mm] summary JSON: {artifacts['summary_json']}")
            return 0
        except Exception:
            error_path = output_dir / "full_gain_map_100mm_orchestrator_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[full-gain-map-100mm] native error: {error_path}", file=sys.stderr)
            return 1

    metadata: dict[str, Any] = {
        "python": sys.version,
        "script": "scripts/full_gain_map_100mm.py",
        "cwd": str(Path.cwd()),
        "legacy_orchestration": True,
    }

    points: list[GainMapPointResult] = []
    point_index = 0

    try:
        for pump_frequency_ghz in config.pump_frequency_ghz_values:
            for pump_current_ratio in config.pump_current_ratios:
                print(
                    "[full-gain-map-100mm] point "
                    f"{point_index}: fp={pump_frequency_ghz:g} GHz, "
                    f"Ip/I*={pump_current_ratio:g}"
                )

                point = run_gain_map_point(
                    config=config,
                    point_index=point_index,
                    pump_frequency_ghz=pump_frequency_ghz,
                    pump_current_ratio=pump_current_ratio,
                    output_dir=output_dir,
                )
                points.append(point)

                print(
                    "[full-gain-map-100mm] point "
                    f"{point_index}: {point.status.value}, "
                    f"max_gain={point.metrics.get('max_gain_db', 'NA')}"
                )

                if config.fail_fast and point.status in {RunStatus.FAIL, RunStatus.ERROR}:
                    raise RuntimeError(
                        f"fail-fast: point {point_index} failed with status {point.status.value}"
                    )

                point_index += 1

    except Exception:
        error_path = output_dir / "full_gain_map_100mm_orchestrator_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        metadata["orchestrator_error_txt"] = str(error_path)

    elapsed_s = time.perf_counter() - start

    result = export_artifacts(
        config=config,
        points=points,
        output_dir=output_dir,
        elapsed_s=elapsed_s,
        metadata=metadata,
    )
    if result.status == RunStatus.PASS:
        result = FullGainMap100mmResult(
            config=result.config,
            status=RunStatus.PARTIAL,
            elapsed_s=result.elapsed_s,
            points=result.points,
            artifact_paths=result.artifact_paths,
            metadata={**dict(result.metadata), "legacy_orchestration": True},
        )
        Path(result.artifact_paths["summary_json"]).write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        Path(result.artifact_paths["summary_md"]).write_text(
            result_markdown(result, result.metadata.get("aggregate", {})),
            encoding="utf-8",
        )

    print()
    print(f"[full-gain-map-100mm] status: {result.status.value}")
    print(f"[full-gain-map-100mm] points: {result.n_points}")
    print(f"[full-gain-map-100mm] pass/partial/error: {result.n_pass}/{result.n_partial}/{result.n_error}")
    print(f"[full-gain-map-100mm] summary JSON: {result.artifact_paths.get('summary_json')}")
    print(f"[full-gain-map-100mm] summary MD:   {result.artifact_paths.get('summary_md')}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())
