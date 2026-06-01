"""
Run a TWPA gain-compression sweep.

This script estimates the input level where gain compresses by sweeping the
signal drive amplitude while keeping the pump operating point fixed.

Important
---------
The script delegates pump solving to:

    scripts/full_pump_hb_100mm.py

and gain evaluation to:

    scripts/gain_from_pumped_solution.py

If the gain backend is still a small-signal backend, the gain will not depend on
signal amplitude and this script will correctly report PARTIAL / no compression
found. Once a finite-signal gain backend is wired into gain_from_pumped_solution.py,
the same orchestration becomes a real compression sweep.

Examples
--------
Quick dry run:

    python scripts/compression_sweep.py --quick --output-dir outputs/compression_quick

Use signal power values in dBm:

    python scripts/compression_sweep.py ^
      --pump-frequency-ghz 10.0 ^
      --pump-current-ratio 0.08 ^
      --signal-power-dbm-values -150 -140 -130 -120 -110 -100 -90 -80 ^
      --target-signal-frequency-ghz 6.0 ^
      --output-dir outputs/compression_sweep

Use an already computed pump solution:

    python scripts/compression_sweep.py ^
      --pump-npz outputs/full_pump_hb_100mm/full_pump_hb_100mm_arrays.npz ^
      --layout-csv outputs/full_pump_hb_100mm/full_pump_hb_100mm_layout_components.csv ^
      --signal-power-dbm-values -150 -140 -130 -120 -110 -100 ^
      --output-dir outputs/compression_from_existing_pump

Fallback-only dry run:

    python scripts/compression_sweep.py ^
      --quick ^
      --pump-solver-mode fallback_linear_pump ^
      --gain-solver-mode fallback_coupled_mode ^
      --output-dir outputs/compression_fallback
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


class PointSolverMode(str, Enum):
    AUTO = "auto"
    NATIVE_FINITE_SIGNAL = "native_finite_signal"
    GAIN_SCRIPT = "gain_script"


@dataclass(frozen=True)
class CompressionSweepConfig:
    n_cells: int
    length_mm: float
    z0_ohm: float
    phase_velocity_m_per_s: float
    layout_csv: str | None
    pump_npz: str | None

    pump_frequency_ghz: float
    pump_current_ratio: float
    pump_phase_rad: float
    i_star_a: float
    l0_scale: float
    nonlinear_beta: float

    signal_f_min_ghz: float
    signal_f_max_ghz: float
    n_signal: int
    target_signal_frequency_ghz: float | None
    signal_current_rms_a_values: tuple[float, ...]
    signal_power_dbm_values: tuple[float, ...]

    compression_db: float
    reference_mode: str

    harmonic_orders: tuple[int, ...]
    n_time: int
    max_iter: int
    tolerance: float
    damping: float
    continuation_steps: int

    pump_solver_mode: str
    pump_numerical_backend: str
    gain_solver_mode: str
    point_solver_mode: str
    require_package_gain_solver: bool
    allow_partial_pump_fallback: bool

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
    export_profile_csv: bool
    make_summary_plots: bool

    full_pump_script: str
    gain_script: str
    python_executable: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompressionPointResult:
    point_index: int
    signal_current_rms_a: float
    signal_power_dbm: float
    status: RunStatus
    elapsed_s: float
    point_dir: str

    gain_status: RunStatus
    gain_returncode: int
    gain_summary_path: str | None
    gain_arrays_npz: str | None
    gain_csv: str | None
    gain_stdout_path: str
    gain_stderr_path: str
    gain_summary: Mapping[str, Any]

    target_gain_db: float | None
    target_signal_frequency_ghz: float | None
    max_gain_db: float | None
    max_gain_frequency_ghz: float | None
    gain_drop_db: float | None

    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status == RunStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_index": self.point_index,
            "signal_current_rms_a": self.signal_current_rms_a,
            "signal_power_dbm": self.signal_power_dbm,
            "status": self.status.value,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "point_dir": self.point_dir,
            "gain_status": self.gain_status.value,
            "gain_returncode": self.gain_returncode,
            "gain_summary_path": self.gain_summary_path,
            "gain_arrays_npz": self.gain_arrays_npz,
            "gain_csv": self.gain_csv,
            "gain_stdout_path": self.gain_stdout_path,
            "gain_stderr_path": self.gain_stderr_path,
            "gain_summary": jsonify(self.gain_summary),
            "target_gain_db": self.target_gain_db,
            "target_signal_frequency_ghz": self.target_signal_frequency_ghz,
            "max_gain_db": self.max_gain_db,
            "max_gain_frequency_ghz": self.max_gain_frequency_ghz,
            "gain_drop_db": self.gain_drop_db,
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class PumpPreparationResult:
    status: RunStatus
    elapsed_s: float
    pump_npz: str | None
    layout_csv: str | None
    summary_path: str | None
    stdout_path: str | None
    stderr_path: str | None
    summary: Mapping[str, Any]
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "elapsed_s": self.elapsed_s,
            "pump_npz": self.pump_npz,
            "layout_csv": self.layout_csv,
            "summary_path": self.summary_path,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "summary": jsonify(self.summary),
            "messages": list(self.messages),
        }


@dataclass(frozen=True)
class CompressionSweepResult:
    config: CompressionSweepConfig
    status: RunStatus
    elapsed_s: float
    pump: PumpPreparationResult
    points: tuple[CompressionPointResult, ...]
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
            "pump": self.pump.to_dict(),
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


def dbm_to_watt(dbm: float) -> float:
    return 1e-3 * 10.0 ** (dbm / 10.0)


def watt_to_dbm(watt: float) -> float:
    return 10.0 * np.log10(max(float(watt), 1e-300) / 1e-3)


def signal_current_rms_to_power_dbm(current_rms_a: float, z0_ohm: float) -> float:
    power_w = float(current_rms_a) ** 2 * float(z0_ohm)
    return watt_to_dbm(power_w)


def signal_power_dbm_to_current_rms(power_dbm: float, z0_ohm: float) -> float:
    return float(np.sqrt(dbm_to_watt(float(power_dbm)) / float(z0_ohm)))


def status_from_summary_and_returncode(returncode: int, summary: Mapping[str, Any]) -> RunStatus:
    summary_status = summary.get("status")
    if returncode == 0 and summary_status == RunStatus.PASS.value:
        return RunStatus.PASS
    if returncode == 0 and summary_status == RunStatus.PARTIAL.value:
        return RunStatus.PARTIAL
    if returncode == 0 and summary:
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


def extract_artifact_path(summary: Mapping[str, Any], key: str) -> str | None:
    value = nested_get(summary, ["artifact_paths", key])
    if value is None:
        return None
    return str(value)


def _native_point_solver_allowed(config: CompressionSweepConfig) -> tuple[bool, str]:
    mode = PointSolverMode(config.point_solver_mode)
    if mode == PointSolverMode.GAIN_SCRIPT:
        return False, "point solver mode forced to gain_script"
    if mode == PointSolverMode.NATIVE_FINITE_SIGNAL:
        return True, "forced native finite-signal point solver"
    return True, "auto native finite-signal point solver"


def should_prepare_external_pump(config: CompressionSweepConfig) -> bool:
    allowed, _ = _native_point_solver_allowed(config)
    return not allowed


def build_pump_command(config: CompressionSweepConfig, pump_dir: Path) -> list[str]:
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
        f"{config.pump_frequency_ghz:.16g}",
        "--pump-current-ratio",
        f"{config.pump_current_ratio:.16g}",
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
        "--numerical-backend",
        config.pump_numerical_backend,
        "--layout-kind",
        config.layout_kind,
        "--disorder-std",
        f"{config.disorder_std:.16g}",
        "--seed",
        str(config.seed),
        "--output-dir",
        str(pump_dir),
        "--name",
        f"{config.name}_pump",
        "--harmonic-orders",
        *[str(h) for h in config.harmonic_orders],
    ]

    if config.layout_csv is not None:
        cmd.extend(["--layout-csv", config.layout_csv])

    if config.include_resonators:
        cmd.append("--include-resonators")

    if not config.allow_partial_pump_fallback:
        cmd.append("--no-allow-partial-fallback")

    if not config.keep_per_point_plots:
        cmd.append("--no-plots")

    if not config.keep_per_point_checkpoints:
        cmd.append("--no-checkpoint")

    if not config.export_profile_csv:
        cmd.append("--no-profile-csv")

    return cmd


def prepare_pump_solution(config: CompressionSweepConfig, output_dir: Path) -> PumpPreparationResult:
    start = time.perf_counter()

    native_allowed, native_reason = _native_point_solver_allowed(config)
    if native_allowed:
        summary = {
            "source": "native_finite_signal_direct",
            "reason": native_reason,
            "target_signal_frequency_ghz": config.target_signal_frequency_ghz,
        }
        if config.pump_npz is not None:
            summary["ignored_pump_npz"] = config.pump_npz
        if config.layout_csv is not None:
            summary["layout_csv"] = config.layout_csv
        return PumpPreparationResult(
            status=RunStatus.PASS,
            elapsed_s=time.perf_counter() - start,
            pump_npz=config.pump_npz,
            layout_csv=config.layout_csv,
            summary_path=None,
            stdout_path=None,
            stderr_path=None,
            summary=summary,
            messages=("PASS: native finite-signal compression does not require external pump preparation.",),
        )

    if config.pump_npz is not None:
        pump_npz = Path(config.pump_npz)
        if not pump_npz.exists():
            return PumpPreparationResult(
                status=RunStatus.ERROR,
                elapsed_s=time.perf_counter() - start,
                pump_npz=str(pump_npz),
                layout_csv=config.layout_csv,
                summary_path=None,
                stdout_path=None,
                stderr_path=None,
                summary={},
                messages=(f"ERROR: provided pump NPZ does not exist: {pump_npz}",),
            )

        if config.layout_csv is not None and not Path(config.layout_csv).exists():
            return PumpPreparationResult(
                status=RunStatus.ERROR,
                elapsed_s=time.perf_counter() - start,
                pump_npz=str(pump_npz),
                layout_csv=config.layout_csv,
                summary_path=None,
                stdout_path=None,
                stderr_path=None,
                summary={},
                messages=(f"ERROR: provided layout CSV does not exist: {config.layout_csv}",),
            )

        return PumpPreparationResult(
            status=RunStatus.PASS,
            elapsed_s=time.perf_counter() - start,
            pump_npz=str(pump_npz),
            layout_csv=config.layout_csv,
            summary_path=None,
            stdout_path=None,
            stderr_path=None,
            summary={
                "source": "provided_pump_npz",
                "pump_npz": str(pump_npz),
                "layout_csv": config.layout_csv,
            },
            messages=("PASS: using provided pump NPZ.",),
        )

    pump_dir = output_dir / "pump"
    pump_stdout = output_dir / "pump_stdout.txt"
    pump_stderr = output_dir / "pump_stderr.txt"

    cmd = build_pump_command(config, pump_dir)
    returncode, elapsed_sub = run_subprocess(
        cmd=cmd,
        cwd=Path.cwd(),
        stdout_path=pump_stdout,
        stderr_path=pump_stderr,
    )

    summary_path = pump_dir / "full_pump_hb_100mm_summary.json"
    summary = read_json_if_exists(summary_path)
    status = status_from_summary_and_returncode(returncode, summary)

    pump_npz = extract_artifact_path(summary, "arrays_npz")
    layout_csv = extract_artifact_path(summary, "layout_components_csv")

    messages: list[str] = []
    if status in {RunStatus.PASS, RunStatus.PARTIAL} and pump_npz is not None:
        messages.append(f"{status.value.upper()}: pump solution prepared.")
    else:
        status = RunStatus.ERROR
        messages.append(f"ERROR: pump preparation failed with return code {returncode}.")
        if pump_npz is None:
            messages.append("ERROR: pump arrays NPZ was not found in the pump summary.")

    return PumpPreparationResult(
        status=status,
        elapsed_s=time.perf_counter() - start,
        pump_npz=pump_npz,
        layout_csv=layout_csv,
        summary_path=str(summary_path) if summary_path.exists() else None,
        stdout_path=str(pump_stdout),
        stderr_path=str(pump_stderr),
        summary=summary,
        messages=tuple(messages),
    )


def compression_point_name(index: int, signal_power_dbm: float) -> str:
    return (
        f"signal_{index:04d}_pin_{signal_power_dbm:.9g}dbm"
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
    )


def build_gain_command(
    *,
    config: CompressionSweepConfig,
    point_dir: Path,
    pump_npz: Path,
    layout_csv: Path | None,
    signal_current_rms_a: float,
) -> list[str]:
    gain_dir = point_dir / "gain"

    cmd = [
        config.python_executable,
        config.gain_script,
        "--pump-npz",
        str(pump_npz),
        "--pump-frequency-ghz",
        f"{config.pump_frequency_ghz:.16g}",
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
        f"{signal_current_rms_a:.16g}",
        "--solver-mode",
        config.gain_solver_mode,
        "--output-dir",
        str(gain_dir),
        "--name",
        f"{config.name}_signal_{signal_current_rms_a:.6g}".replace(".", "p"),
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


def load_gain_curve(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    out = {
        "signal_frequency_hz": np.asarray(npz["signal_frequency_hz"], dtype=float),
        "signal_gain_db": np.asarray(npz["signal_gain_db"], dtype=float),
    }
    if "idler_conversion_db" in npz:
        out["idler_conversion_db"] = np.asarray(npz["idler_conversion_db"], dtype=float)
    return out


def build_or_load_native_layout(config: CompressionSweepConfig) -> tuple[Any, dict[str, Any]]:
    if config.layout_csv is not None:
        from twpa.io.netlist import load_layout_component_csv

        layout = load_layout_component_csv(
            config.layout_csv,
            z0_ohm=config.z0_ohm,
            name=config.name,
            metadata={
                "source": "scripts.compression_sweep",
                "layout_csv": config.layout_csv,
            },
        )
        return layout, {
            "builder": "load_layout_component_csv",
            "layout_csv": config.layout_csv,
            "n_cells": int(layout.n_cells),
        }

    from twpa.workflows.synthetic_benchmarks import (
        SyntheticLayoutKind,
        SyntheticLayoutSpec,
        build_synthetic_layout,
    )

    try:
        kind = SyntheticLayoutKind(config.layout_kind)
    except Exception:
        kind = getattr(SyntheticLayoutKind, config.layout_kind.upper(), SyntheticLayoutKind.UNIFORM)

    if config.include_resonators and str(kind.value) == "uniform":
        kind = getattr(SyntheticLayoutKind, "STUB_PERIODIC", kind)

    spec = SyntheticLayoutSpec(
        kind=kind,
        n_cells=config.n_cells,
        length_m=config.length_mm * 1e-3,
        z0_ohm=config.z0_ohm,
        phase_velocity_m_per_s=config.phase_velocity_m_per_s,
        disorder_std_fraction=config.disorder_std,
        disorder_seed=config.seed,
        name=config.name,
    )
    layout = build_synthetic_layout(spec)
    return layout, {
        "builder": "twpa.workflows.synthetic_benchmarks.build_synthetic_layout",
        "spec": jsonify(spec),
        "n_cells": int(layout.n_cells),
    }


def run_native_finite_signal_point(
    *,
    config: CompressionSweepConfig,
    point_index: int,
    signal_current_rms_a: float,
    signal_power_dbm: float,
    output_dir: Path,
) -> CompressionPointResult:
    from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
    from twpa.core.hb_fft import HBProjectionConfig
    from twpa.nonlinear.finite_signal_hb import (
        FiniteSignalHBConfig,
        SignalDriveConfig,
        solve_finite_signal_hb,
    )
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig

    start = time.perf_counter()
    point_dir = output_dir / "points" / compression_point_name(point_index, signal_power_dbm)
    gain_dir = point_dir / "gain"
    gain_dir.mkdir(parents=True, exist_ok=True)
    gain_stdout = point_dir / "gain_stdout.txt"
    gain_stderr = point_dir / "gain_stderr.txt"

    try:
        layout, layout_info = build_or_load_native_layout(config)
        solver_backend = (
            SolverBackend.NEWTON_KRYLOV
            if config.pump_numerical_backend == "newton_krylov"
            else SolverBackend.DENSE
        )
        solver = SolverConfig(
            backend=solver_backend,
            max_iter=config.max_iter,
            abs_tol=config.tolerance,
            rel_tol=config.tolerance,
            damping_initial=config.damping,
            verbose=False,
        )
        positive_orders = [abs(h) for h in config.harmonic_orders if h > 0]
        n_pump_harmonics = max(positive_orders) if positive_orders else 1
        finite_cfg = FiniteSignalHBConfig(
            n_pump_harmonics=n_pump_harmonics,
            include_negative_frequencies=any(h < 0 for h in config.harmonic_orders),
            include_dc=any(h == 0 for h in config.harmonic_orders),
            include_second_order_sidebands=n_pump_harmonics >= 2,
            projection=HBProjectionConfig(
                n_time_samples=config.n_time,
                force_real_time_signal=True,
                enforce_conjugate_symmetry=True,
            ),
            solver=solver,
            name=f"{config.name}_native_finite_signal",
        )
        nonlinear = NonlinearParams(
            I_star_A=config.i_star_a,
            beta_nl=config.nonlinear_beta,
        )
        pump_drive = PumpDriveConfig.from_current_rms(
            pump_frequency_hz=config.pump_frequency_ghz * 1e9,
            current_rms_A=config.pump_current_ratio * config.i_star_a,
            source_impedance_ohm=config.z0_ohm,
            phase_rad=config.pump_phase_rad,
        )
        signal_drive = SignalDriveConfig(
            signal_frequency_hz=float(config.target_signal_frequency_ghz) * 1e9,
            current_rms_A=signal_current_rms_a,
            source_impedance_ohm=config.z0_ohm,
        )
        result = solve_finite_signal_hb(
            layout,
            nonlinear,
            pump_drive=pump_drive,
            signal_drive=signal_drive,
            finite_config=finite_cfg,
            metadata={
                "driver": "scripts.compression_sweep.run_native_finite_signal_point",
                "point_index": int(point_index),
                "signal_power_dbm": float(signal_power_dbm),
            },
        )
        residual_norm = float(result.distributed_result.residual.norm)
        finite_residual = bool(np.isfinite(residual_norm))
        gain_value = float(result.observables.signal_gain_db)
        point_status = RunStatus.PASS if result.converged and finite_residual else RunStatus.PARTIAL

        arrays_npz = gain_dir / "gain_from_pumped_solution_arrays.npz"
        np.savez(
            arrays_npz,
            signal_frequency_hz=np.asarray([signal_drive.signal_frequency_hz], dtype=float),
            signal_gain_db=np.asarray([result.observables.signal_gain_db], dtype=float),
            matched_power_gain_db=np.asarray([result.observables.matched_power_gain_db], dtype=float),
            idler_conversion_db=np.asarray(
                [np.nan if result.observables.idler_conversion_db is None else result.observables.idler_conversion_db],
                dtype=float,
            ),
            residual_norm=np.asarray(residual_norm, dtype=float),
        )
        gain_csv = gain_dir / "gain_from_pumped_solution_gain.csv"
        gain_csv.write_text(
            "\n".join(
                [
                    "signal_frequency_hz,signal_gain_db,matched_power_gain_db,idler_conversion_db,residual_norm",
                    (
                        f"{signal_drive.signal_frequency_hz:.16g},"
                        f"{result.observables.signal_gain_db:.16g},"
                        f"{result.observables.matched_power_gain_db:.16g},"
                        f"{float('nan') if result.observables.idler_conversion_db is None else result.observables.idler_conversion_db:.16g},"
                        f"{residual_norm:.16g}"
                    ),
                ]
            ),
            encoding="utf-8",
        )

        gain_summary = {
            "status": point_status.value,
            "driver": "native_finite_signal_hb",
            "converged": bool(result.converged),
            "finite_residual": finite_residual,
            "residual_norm": residual_norm,
            "signal_frequency_hz": signal_drive.signal_frequency_hz,
            "signal_power_dbm": signal_power_dbm,
            "signal_current_rms_a": signal_current_rms_a,
            "signal_gain_db": result.observables.signal_gain_db,
            "matched_power_gain_db": result.observables.matched_power_gain_db,
            "idler_conversion_db": result.observables.idler_conversion_db,
            "layout": layout_info,
            "artifact_paths": {
                "arrays_npz": str(arrays_npz),
                "gain_csv": str(gain_csv),
            },
            "metadata": {
                "selected_preconditioner": nested_get(
                    result.to_dict(),
                    ["distributed_result", "solver", "metadata", "selected_preconditioner"],
                ),
            },
        }
        gain_summary_path = gain_dir / "gain_from_pumped_solution_summary.json"
        gain_summary_path.write_text(json.dumps(jsonify(gain_summary), indent=2), encoding="utf-8")
        gain_stdout.write_text(
            f"native finite-signal compression point\nstatus={point_status.value}\nresidual_norm={residual_norm:.16g}\n",
            encoding="utf-8",
        )
        gain_stderr.write_text("", encoding="utf-8")

        messages = [
            (
                "PASS: native finite-signal HB point converged with finite residual."
                if point_status == RunStatus.PASS
                else "PARTIAL: native finite-signal HB point did not fully converge."
            )
        ]
        return CompressionPointResult(
            point_index=point_index,
            signal_current_rms_a=signal_current_rms_a,
            signal_power_dbm=signal_power_dbm,
            status=point_status,
            elapsed_s=time.perf_counter() - start,
            point_dir=str(point_dir),
            gain_status=point_status,
            gain_returncode=0,
            gain_summary_path=str(gain_summary_path),
            gain_arrays_npz=str(arrays_npz),
            gain_csv=str(gain_csv),
            gain_stdout_path=str(gain_stdout),
            gain_stderr_path=str(gain_stderr),
            gain_summary=gain_summary,
            target_gain_db=gain_value,
            target_signal_frequency_ghz=float(config.target_signal_frequency_ghz),
            max_gain_db=gain_value,
            max_gain_frequency_ghz=float(config.target_signal_frequency_ghz),
            gain_drop_db=None,
            messages=tuple(messages),
        )
    except Exception as exc:
        gain_stdout.write_text("", encoding="utf-8")
        gain_stderr.write_text(traceback.format_exc(), encoding="utf-8")
        return CompressionPointResult(
            point_index=point_index,
            signal_current_rms_a=signal_current_rms_a,
            signal_power_dbm=signal_power_dbm,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            point_dir=str(point_dir),
            gain_status=RunStatus.ERROR,
            gain_returncode=-1,
            gain_summary_path=None,
            gain_arrays_npz=None,
            gain_csv=None,
            gain_stdout_path=str(gain_stdout),
            gain_stderr_path=str(gain_stderr),
            gain_summary={
                "status": RunStatus.ERROR.value,
                "driver": "native_finite_signal_hb",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            target_gain_db=None,
            target_signal_frequency_ghz=config.target_signal_frequency_ghz,
            max_gain_db=None,
            max_gain_frequency_ghz=None,
            gain_drop_db=None,
            messages=(f"ERROR: native finite-signal HB point failed: {type(exc).__name__}: {exc}",),
        )


def _interpolate_gain_curve_p1db(
    signal_power_dbm: Sequence[float],
    gain_db: Sequence[float],
    compression_db: float,
) -> float | None:
    powers = np.asarray(signal_power_dbm, dtype=float)
    gains = np.asarray(gain_db, dtype=float)
    valid = np.isfinite(powers) & np.isfinite(gains)
    powers = powers[valid]
    gains = gains[valid]
    if powers.size == 0:
        return None
    drops = gains[0] - gains
    crossings = np.where(drops >= float(compression_db))[0]
    if crossings.size == 0:
        return None
    idx = int(crossings[0])
    if idx == 0:
        return float(powers[0])
    d0, d1 = drops[idx - 1], drops[idx]
    if abs(d1 - d0) <= 1e-300:
        return float(powers[idx])
    alpha = (float(compression_db) - d0) / (d1 - d0)
    return float(powers[idx - 1] + alpha * (powers[idx] - powers[idx - 1]))


def run_native_wideband_compression(
    *,
    config: CompressionSweepConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Run native finite-signal compression over the configured signal band."""
    from twpa.core.hb_fft import HBProjectionConfig
    from twpa.core.params import NonlinearParams, SolverBackend, SolverConfig
    from twpa.nonlinear.finite_signal_hb import (
        FiniteSignalHBConfig,
        SignalDriveConfig,
        sweep_signal_current_compression,
    )
    from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig

    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    layout, layout_info = build_or_load_native_layout(config)
    solver = SolverConfig(
        backend=(
            SolverBackend.NEWTON_KRYLOV
            if config.pump_numerical_backend == "newton_krylov"
            else SolverBackend.DENSE
        ),
        max_iter=config.max_iter,
        abs_tol=config.tolerance,
        rel_tol=config.tolerance,
        damping_initial=config.damping,
        verbose=False,
    )
    positive_orders = [abs(h) for h in config.harmonic_orders if h > 0]
    n_pump_harmonics = max(positive_orders) if positive_orders else 1
    finite_cfg = FiniteSignalHBConfig(
        n_pump_harmonics=n_pump_harmonics,
        include_negative_frequencies=any(h < 0 for h in config.harmonic_orders),
        include_dc=any(h == 0 for h in config.harmonic_orders),
        include_second_order_sidebands=n_pump_harmonics >= 2,
        projection=HBProjectionConfig(
            n_time_samples=config.n_time,
            force_real_time_signal=True,
            enforce_conjugate_symmetry=True,
        ),
        solver=solver,
        name=f"{config.name}_native_wideband_finite_signal",
    )
    nonlinear = NonlinearParams(I_star_A=config.i_star_a, beta_nl=config.nonlinear_beta)
    pump_drive = PumpDriveConfig.from_current_rms(
        pump_frequency_hz=config.pump_frequency_ghz * 1e9,
        current_rms_A=config.pump_current_ratio * config.i_star_a,
        source_impedance_ohm=config.z0_ohm,
        phase_rad=config.pump_phase_rad,
    )
    frequencies_ghz = np.linspace(
        config.signal_f_min_ghz,
        config.signal_f_max_ghz,
        config.n_signal,
    )
    gain_matrix = np.full((frequencies_ghz.size, len(config.signal_power_dbm_values)), np.nan)
    matched_gain_matrix = np.full_like(gain_matrix, np.nan)
    residual_matrix = np.full_like(gain_matrix, np.nan)
    converged_matrix = np.zeros_like(gain_matrix, dtype=bool)
    p1db = np.full(frequencies_ghz.shape, np.nan)

    for frequency_index, frequency_ghz in enumerate(frequencies_ghz):
        base_signal = SignalDriveConfig(
            signal_frequency_hz=float(frequency_ghz) * 1e9,
            current_rms_A=float(config.signal_current_rms_a_values[0]),
            source_impedance_ohm=config.z0_ohm,
        )
        sweep = sweep_signal_current_compression(
            layout,
            nonlinear,
            pump_drive=pump_drive,
            base_signal_drive=base_signal,
            signal_current_rms_values=config.signal_current_rms_a_values,
            finite_config=finite_cfg,
            reuse_previous_solution=True,
        )
        for power_index, point in enumerate(sweep.points):
            gain_matrix[frequency_index, power_index] = point.signal_gain_db
            matched_gain_matrix[frequency_index, power_index] = point.matched_power_gain_db
            residual_matrix[frequency_index, power_index] = point.result.distributed_result.residual.norm
            converged_matrix[frequency_index, power_index] = point.converged
        estimate = _interpolate_gain_curve_p1db(
            config.signal_power_dbm_values,
            gain_matrix[frequency_index],
            config.compression_db,
        )
        if estimate is not None:
            p1db[frequency_index] = estimate

    finite_residuals = bool(np.all(np.isfinite(residual_matrix)))
    all_converged = bool(np.all(converged_matrix))
    found_any_p1db = bool(np.any(np.isfinite(p1db)))
    status = RunStatus.PASS if all_converged and finite_residuals and found_any_p1db else RunStatus.PARTIAL

    arrays_path = output_dir / "compression_sweep_arrays.npz"
    np.savez_compressed(
        arrays_path,
        signal_frequency_ghz=frequencies_ghz,
        signal_power_dbm=np.asarray(config.signal_power_dbm_values, dtype=float),
        signal_current_rms_a=np.asarray(config.signal_current_rms_a_values, dtype=float),
        signal_gain_db=gain_matrix,
        matched_power_gain_db=matched_gain_matrix,
        residual_norm=residual_matrix,
        converged=converged_matrix,
        p1db_input_dbm=p1db,
    )
    csv_path = output_dir / "compression_sweep_wideband.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal_frequency_ghz", "p1db_input_dbm"])
        writer.writerows(zip(frequencies_ghz.tolist(), p1db.tolist()))

    artifacts = {
        "arrays_npz": str(arrays_path),
        "wideband_csv": str(csv_path),
        "summary_json": str(output_dir / "compression_sweep_summary.json"),
    }
    summary = {
        "status": status.value,
        "passed": status == RunStatus.PASS,
        "driver": "native_finite_signal_hb_wideband",
        "elapsed_s": time.perf_counter() - start,
        "config": config.to_dict(),
        "layout": layout_info,
        "n_signal_frequencies": int(frequencies_ghz.size),
        "n_power_points": len(config.signal_power_dbm_values),
        "all_converged": all_converged,
        "finite_residuals": finite_residuals,
        "n_p1db_found": int(np.sum(np.isfinite(p1db))),
        "artifact_paths": artifacts,
    }
    Path(artifacts["summary_json"]).write_text(
        json.dumps(jsonify(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def extract_gain_observables(
    gain_arrays_npz: str | None,
    *,
    config: CompressionSweepConfig,
) -> tuple[float | None, float | None, float | None, float | None]:
    if gain_arrays_npz is None:
        return None, None, None, None

    curve = load_gain_curve(gain_arrays_npz)
    f_ghz = curve["signal_frequency_hz"] / 1e9
    gain_db = curve["signal_gain_db"]

    if gain_db.size == 0 or not np.any(np.isfinite(gain_db)):
        return None, None, None, None

    max_idx = int(np.nanargmax(gain_db))
    max_gain_db = float(gain_db[max_idx])
    max_gain_frequency_ghz = float(f_ghz[max_idx])

    if config.target_signal_frequency_ghz is None:
        target_gain_db = max_gain_db
        target_frequency_ghz = max_gain_frequency_ghz
    else:
        idx = int(np.argmin(np.abs(f_ghz - config.target_signal_frequency_ghz)))
        target_gain_db = float(gain_db[idx])
        target_frequency_ghz = float(f_ghz[idx])

    return target_gain_db, target_frequency_ghz, max_gain_db, max_gain_frequency_ghz


def run_compression_point(
    *,
    config: CompressionSweepConfig,
    pump: PumpPreparationResult,
    point_index: int,
    signal_current_rms_a: float,
    signal_power_dbm: float,
    output_dir: Path,
) -> CompressionPointResult:
    native_allowed, _ = _native_point_solver_allowed(config)
    if native_allowed:
        return run_native_finite_signal_point(
            config=config,
            point_index=point_index,
            signal_current_rms_a=signal_current_rms_a,
            signal_power_dbm=signal_power_dbm,
            output_dir=output_dir,
        )

    start = time.perf_counter()
    point_dir = output_dir / "points" / compression_point_name(point_index, signal_power_dbm)
    point_dir.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []

    if pump.pump_npz is None:
        return CompressionPointResult(
            point_index=point_index,
            signal_current_rms_a=signal_current_rms_a,
            signal_power_dbm=signal_power_dbm,
            status=RunStatus.ERROR,
            elapsed_s=time.perf_counter() - start,
            point_dir=str(point_dir),
            gain_status=RunStatus.ERROR,
            gain_returncode=-1,
            gain_summary_path=None,
            gain_arrays_npz=None,
            gain_csv=None,
            gain_stdout_path=str(point_dir / "gain_stdout.txt"),
            gain_stderr_path=str(point_dir / "gain_stderr.txt"),
            gain_summary={},
            target_gain_db=None,
            target_signal_frequency_ghz=None,
            max_gain_db=None,
            max_gain_frequency_ghz=None,
            gain_drop_db=None,
            messages=("ERROR: pump NPZ is unavailable.",),
        )

    gain_stdout = point_dir / "gain_stdout.txt"
    gain_stderr = point_dir / "gain_stderr.txt"

    cmd = build_gain_command(
        config=config,
        point_dir=point_dir,
        pump_npz=Path(pump.pump_npz),
        layout_csv=None if pump.layout_csv is None else Path(pump.layout_csv),
        signal_current_rms_a=signal_current_rms_a,
    )

    returncode, _ = run_subprocess(
        cmd=cmd,
        cwd=Path.cwd(),
        stdout_path=gain_stdout,
        stderr_path=gain_stderr,
    )

    summary_path = point_dir / "gain" / "gain_from_pumped_solution_summary.json"
    summary = read_json_if_exists(summary_path)
    gain_status = status_from_summary_and_returncode(returncode, summary)

    gain_arrays_npz = extract_artifact_path(summary, "arrays_npz")
    gain_csv = extract_artifact_path(summary, "gain_csv")

    try:
        target_gain_db, target_freq_ghz, max_gain_db, max_gain_freq_ghz = extract_gain_observables(
            gain_arrays_npz,
            config=config,
        )
    except Exception as exc:
        target_gain_db = None
        target_freq_ghz = None
        max_gain_db = None
        max_gain_freq_ghz = None
        messages.append(f"ERROR: could not extract gain observables: {type(exc).__name__}: {exc}")
        gain_status = RunStatus.ERROR

    if gain_status in {RunStatus.PASS, RunStatus.PARTIAL}:
        status = (
            RunStatus.PARTIAL
            if PointSolverMode(config.point_solver_mode) == PointSolverMode.GAIN_SCRIPT
            else gain_status
        )
        messages.append(f"{status.value.upper()}: gain point completed.")
        if PointSolverMode(config.point_solver_mode) == PointSolverMode.GAIN_SCRIPT:
            messages.append("PARTIAL: explicit legacy gain-script orchestration selected.")
    else:
        status = RunStatus.ERROR
        messages.append(f"ERROR: gain point failed with return code {returncode}.")

    return CompressionPointResult(
        point_index=point_index,
        signal_current_rms_a=signal_current_rms_a,
        signal_power_dbm=signal_power_dbm,
        status=status,
        elapsed_s=time.perf_counter() - start,
        point_dir=str(point_dir),
        gain_status=gain_status,
        gain_returncode=returncode,
        gain_summary_path=str(summary_path) if summary_path.exists() else None,
        gain_arrays_npz=gain_arrays_npz,
        gain_csv=gain_csv,
        gain_stdout_path=str(gain_stdout),
        gain_stderr_path=str(gain_stderr),
        gain_summary=summary,
        target_gain_db=target_gain_db,
        target_signal_frequency_ghz=target_freq_ghz,
        max_gain_db=max_gain_db,
        max_gain_frequency_ghz=max_gain_freq_ghz,
        gain_drop_db=None,
        messages=tuple(messages),
    )


def apply_gain_drop(points: Sequence[CompressionPointResult], config: CompressionSweepConfig) -> list[CompressionPointResult]:
    valid = [p for p in points if p.target_gain_db is not None and np.isfinite(p.target_gain_db)]
    if not valid:
        return list(points)

    ordered = sorted(valid, key=lambda p: p.signal_power_dbm)

    if config.reference_mode == "first":
        reference_gain = float(ordered[0].target_gain_db)
    elif config.reference_mode == "max_low_power":
        n = max(1, min(3, len(ordered)))
        reference_gain = float(np.nanmax([p.target_gain_db for p in ordered[:n]]))
    else:
        reference_gain = float(ordered[0].target_gain_db)

    drop_by_index = {
        p.point_index: reference_gain - float(p.target_gain_db)
        for p in valid
    }

    updated: list[CompressionPointResult] = []
    for p in points:
        updated.append(
            CompressionPointResult(
                point_index=p.point_index,
                signal_current_rms_a=p.signal_current_rms_a,
                signal_power_dbm=p.signal_power_dbm,
                status=p.status,
                elapsed_s=p.elapsed_s,
                point_dir=p.point_dir,
                gain_status=p.gain_status,
                gain_returncode=p.gain_returncode,
                gain_summary_path=p.gain_summary_path,
                gain_arrays_npz=p.gain_arrays_npz,
                gain_csv=p.gain_csv,
                gain_stdout_path=p.gain_stdout_path,
                gain_stderr_path=p.gain_stderr_path,
                gain_summary=p.gain_summary,
                target_gain_db=p.target_gain_db,
                target_signal_frequency_ghz=p.target_signal_frequency_ghz,
                max_gain_db=p.max_gain_db,
                max_gain_frequency_ghz=p.max_gain_frequency_ghz,
                gain_drop_db=drop_by_index.get(p.point_index),
                messages=p.messages,
            )
        )

    return updated


def interpolate_compression_point(
    points: Sequence[CompressionPointResult],
    compression_db: float,
) -> dict[str, Any]:
    valid = [
        p for p in points
        if p.gain_drop_db is not None
        and p.target_gain_db is not None
        and np.isfinite(p.gain_drop_db)
        and np.isfinite(p.target_gain_db)
    ]
    valid = sorted(valid, key=lambda p: p.signal_power_dbm)

    if not valid:
        return {
            "found": False,
            "reason": "no valid gain-drop samples",
            "compression_db": compression_db,
        }

    drops = np.asarray([p.gain_drop_db for p in valid], dtype=float)
    powers = np.asarray([p.signal_power_dbm for p in valid], dtype=float)
    gains = np.asarray([p.target_gain_db for p in valid], dtype=float)

    crossing_indices = np.where(drops >= compression_db)[0]
    if crossing_indices.size == 0:
        return {
            "found": False,
            "reason": "compression threshold not reached",
            "compression_db": compression_db,
            "max_gain_drop_db": float(np.nanmax(drops)),
            "highest_signal_power_dbm": float(np.nanmax(powers)),
        }

    k = int(crossing_indices[0])
    if k == 0:
        p_comp = float(powers[0])
        gain_at_comp = float(gains[0])
    else:
        x0, x1 = drops[k - 1], drops[k]
        p0, p1 = powers[k - 1], powers[k]
        g0, g1 = gains[k - 1], gains[k]

        if abs(x1 - x0) < 1e-12:
            alpha = 1.0
        else:
            alpha = float((compression_db - x0) / (x1 - x0))

        p_comp = float(p0 + alpha * (p1 - p0))
        gain_at_comp = float(g0 + alpha * (g1 - g0))

    return {
        "found": True,
        "compression_db": compression_db,
        "input_power_dbm": p_comp,
        "gain_at_compression_db": gain_at_comp,
        "output_power_dbm": p_comp + gain_at_comp,
        "crossing_index": k,
        "reference_gain_db": float(gains[0] + drops[0]),
    }


def write_points_csv(path: Path, points: Sequence[CompressionPointResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "point_index",
        "signal_current_rms_a",
        "signal_power_dbm",
        "status",
        "elapsed_s",
        "target_gain_db",
        "target_signal_frequency_ghz",
        "max_gain_db",
        "max_gain_frequency_ghz",
        "gain_drop_db",
        "gain_status",
        "gain_returncode",
        "point_dir",
        "gain_arrays_npz",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for p in points:
            writer.writerow(
                {
                    "point_index": p.point_index,
                    "signal_current_rms_a": p.signal_current_rms_a,
                    "signal_power_dbm": p.signal_power_dbm,
                    "status": p.status.value,
                    "elapsed_s": p.elapsed_s,
                    "target_gain_db": p.target_gain_db,
                    "target_signal_frequency_ghz": p.target_signal_frequency_ghz,
                    "max_gain_db": p.max_gain_db,
                    "max_gain_frequency_ghz": p.max_gain_frequency_ghz,
                    "gain_drop_db": p.gain_drop_db,
                    "gain_status": p.gain_status.value,
                    "gain_returncode": p.gain_returncode,
                    "point_dir": p.point_dir,
                    "gain_arrays_npz": p.gain_arrays_npz,
                }
            )

    return path


def build_arrays(points: Sequence[CompressionPointResult]) -> dict[str, np.ndarray]:
    return {
        "signal_current_rms_a": np.asarray([p.signal_current_rms_a for p in points], dtype=float),
        "signal_power_dbm": np.asarray([p.signal_power_dbm for p in points], dtype=float),
        "target_gain_db": np.asarray(
            [np.nan if p.target_gain_db is None else p.target_gain_db for p in points],
            dtype=float,
        ),
        "max_gain_db": np.asarray(
            [np.nan if p.max_gain_db is None else p.max_gain_db for p in points],
            dtype=float,
        ),
        "gain_drop_db": np.asarray(
            [np.nan if p.gain_drop_db is None else p.gain_drop_db for p in points],
            dtype=float,
        ),
        "status": np.asarray([p.status.value for p in points], dtype=object),
    }


def write_summary_plots(
    *,
    output_dir: Path,
    points: Sequence[CompressionPointResult],
    compression: Mapping[str, Any],
    config: CompressionSweepConfig,
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

    valid = [
        p for p in sorted(points, key=lambda q: q.signal_power_dbm)
        if p.target_gain_db is not None and np.isfinite(p.target_gain_db)
    ]

    if not valid:
        return paths

    pin = np.asarray([p.signal_power_dbm for p in valid], dtype=float)
    gain = np.asarray([p.target_gain_db for p in valid], dtype=float)
    drop = np.asarray([np.nan if p.gain_drop_db is None else p.gain_drop_db for p in valid], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax.plot(pin, gain, marker="o")
    ax.set_xlabel("Input signal power (dBm)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("Gain compression sweep")
    ax.grid(True)
    fig.tight_layout()
    p = output_dir / "compression_gain_vs_input_power.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths["gain_vs_input_power_png"] = str(p)

    if np.any(np.isfinite(drop)):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
        ax.plot(pin, drop, marker="o")
        ax.axhline(config.compression_db, linestyle="--", linewidth=1.0)
        if compression.get("found"):
            ax.axvline(float(compression["input_power_dbm"]), linestyle=":", linewidth=1.0)
        ax.set_xlabel("Input signal power (dBm)")
        ax.set_ylabel("Gain drop (dB)")
        ax.set_title(f"{config.compression_db:g} dB compression estimate")
        ax.grid(True)
        fig.tight_layout()
        p = output_dir / "compression_gain_drop.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths["gain_drop_png"] = str(p)

    return paths


def export_artifacts(
    *,
    config: CompressionSweepConfig,
    pump: PumpPreparationResult,
    points: Sequence[CompressionPointResult],
    output_dir: Path,
    elapsed_s: float,
    metadata: Mapping[str, Any],
) -> CompressionSweepResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    points_csv = write_points_csv(output_dir / "compression_sweep_points.csv", points)
    artifacts["points_csv"] = str(points_csv)

    points_json = output_dir / "compression_sweep_points.json"
    points_json.write_text(
        json.dumps(jsonify([p.to_dict() for p in points]), indent=2),
        encoding="utf-8",
    )
    artifacts["points_json"] = str(points_json)

    arrays = build_arrays(points)
    compression = interpolate_compression_point(points, config.compression_db)

    compression_json = output_dir / "compression_sweep_compression_point.json"
    compression_json.write_text(json.dumps(jsonify(compression), indent=2), encoding="utf-8")
    artifacts["compression_json"] = str(compression_json)

    arrays_npz = output_dir / "compression_sweep_arrays.npz"
    np.savez_compressed(
        arrays_npz,
        **arrays,
        metadata_json=json.dumps(
            jsonify(
                {
                    "config": config.to_dict(),
                    "pump": pump.to_dict(),
                    "compression": compression,
                }
            )
        ),
    )
    artifacts["arrays_npz"] = str(arrays_npz)

    if config.make_summary_plots:
        artifacts.update(
            write_summary_plots(
                output_dir=output_dir,
                points=points,
                compression=compression,
                config=config,
            )
        )

    hard_fail = pump.status in {RunStatus.FAIL, RunStatus.ERROR} or any(
        p.status in {RunStatus.FAIL, RunStatus.ERROR} for p in points
    )
    partial = pump.status == RunStatus.PARTIAL or any(p.status == RunStatus.PARTIAL for p in points)

    if hard_fail:
        status = RunStatus.ERROR
    elif compression.get("found") and not partial:
        status = RunStatus.PASS
    else:
        status = RunStatus.PARTIAL

    summary_json = output_dir / "compression_sweep_summary.json"
    summary_md = output_dir / "compression_sweep_summary.md"

    artifacts["summary_json"] = str(summary_json)
    artifacts["summary_md"] = str(summary_md)

    result = CompressionSweepResult(
        config=config,
        status=status,
        elapsed_s=elapsed_s,
        pump=pump,
        points=tuple(points),
        artifact_paths=artifacts,
        metadata={**dict(metadata), "compression": compression},
    )

    summary_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    summary_md.write_text(result_markdown(result, compression), encoding="utf-8")

    return result


def result_markdown(
    result: CompressionSweepResult,
    compression: Mapping[str, Any],
) -> str:
    cfg = result.config

    lines = [
        "# Compression sweep",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- pump frequency: `{cfg.pump_frequency_ghz:.6g} GHz`",
        f"- pump current ratio: `{cfg.pump_current_ratio:.6g}`",
        f"- target signal frequency: `{cfg.target_signal_frequency_ghz}` GHz",
        f"- compression threshold: `{cfg.compression_db}` dB",
        f"- points: `{result.n_points}`",
        f"- pass/partial/error: `{result.n_pass}/{result.n_partial}/{result.n_error}`",
        "",
        "## Compression estimate",
        "",
        f"- found: `{compression.get('found')}`",
        f"- reason: `{compression.get('reason')}`",
        f"- input compression power: `{compression.get('input_power_dbm')}` dBm",
        f"- output compression power: `{compression.get('output_power_dbm')}` dBm",
        f"- gain at compression: `{compression.get('gain_at_compression_db')}` dB",
        f"- max observed gain drop: `{compression.get('max_gain_drop_db')}` dB",
        "",
        "## Pump preparation",
        "",
        f"- status: `{result.pump.status.value}`",
        f"- pump NPZ: `{result.pump.pump_npz}`",
        f"- layout CSV: `{result.pump.layout_csv}`",
        "",
        "## Sweep points",
        "",
        "| point | Pin dBm | Irms A | status | target gain dB | drop dB | max gain dB | target GHz | elapsed s |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    for p in result.points:
        lines.append(
            f"| {p.point_index} | "
            f"{p.signal_power_dbm:.6g} | "
            f"{p.signal_current_rms_a:.6g} | "
            f"`{p.status.value}` | "
            f"{p.target_gain_db} | "
            f"{p.gain_drop_db} | "
            f"{p.max_gain_db} | "
            f"{p.target_signal_frequency_ghz} | "
            f"{p.elapsed_s:.6g} |"
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
        for p in failed:
            lines += [
                f"### Point {p.point_index}: Pin={p.signal_power_dbm:g} dBm",
                "",
                *[f"- {m}" for m in p.messages],
                f"- gain stdout: `{p.gain_stdout_path}`",
                f"- gain stderr: `{p.gain_stderr_path}`",
                "",
            ]

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TWPA gain-compression sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--n-cells", type=int, default=20000)
    parser.add_argument("--length-mm", type=float, default=100.0)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--phase-velocity-m-per-s", type=float, default=1.2e8)
    parser.add_argument("--layout-csv", type=str, default=None)
    parser.add_argument("--pump-npz", type=str, default=None)

    parser.add_argument("--pump-frequency-ghz", type=float, default=10.0)
    parser.add_argument("--pump-current-ratio", type=float, default=0.08)
    parser.add_argument("--pump-phase-rad", type=float, default=0.0)
    parser.add_argument("--i-star-a", type=float, default=5e-3)
    parser.add_argument("--l0-scale", type=float, default=1.0)
    parser.add_argument("--nonlinear-beta", type=float, default=1.0)

    parser.add_argument("--signal-f-min-ghz", type=float, default=4.0)
    parser.add_argument("--signal-f-max-ghz", type=float, default=8.0)
    parser.add_argument("--n-signal", type=int, default=81)
    parser.add_argument("--target-signal-frequency-ghz", type=float, default=None)
    parser.add_argument("--signal-current-rms-a-values", type=float, nargs="*", default=None)
    parser.add_argument(
        "--signal-power-dbm-values",
        type=float,
        nargs="*",
        default=[-150, -145, -140, -135, -130, -125, -120, -115, -110, -105, -100, -95, -90],
    )

    parser.add_argument("--compression-db", type=float, default=1.0)
    parser.add_argument(
        "--reference-mode",
        choices=["first", "max_low_power"],
        default="first",
    )

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
        "--pump-numerical-backend",
        choices=["dense", "newton_krylov"],
        default="dense",
        help="Dense is guarded and intended for tiny reference runs only.",
    )
    parser.add_argument(
        "--gain-solver-mode",
        choices=["auto", "package", "fallback_coupled_mode"],
        default="auto",
    )
    parser.add_argument(
        "--point-solver-mode",
        choices=[m.value for m in PointSolverMode],
        default=PointSolverMode.AUTO.value,
        help="Compression point backend. gain_script is an explicit PARTIAL compatibility route.",
    )
    parser.add_argument("--require-package-gain-solver", action="store_true")
    parser.add_argument("--no-allow-partial-pump-fallback", action="store_true")

    parser.add_argument("--layout-kind", type=str, default="uniform")
    parser.add_argument("--include-resonators", action="store_true")
    parser.add_argument("--disorder-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compression_sweep"))
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument("--name", type=str, default="compression_sweep")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--keep-per-point-plots", action="store_true")
    parser.add_argument("--keep-per-point-checkpoints", action="store_true")
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


def resolve_config(args: argparse.Namespace) -> CompressionSweepConfig:
    n_cells = int(args.n_cells)
    n_signal = int(args.n_signal)
    n_time = int(args.n_time)
    max_iter = int(args.max_iter)
    continuation_steps = int(args.continuation_steps)
    harmonic_orders = tuple(int(v) for v in args.harmonic_orders)

    if args.quick:
        if n_cells == 20000:
            n_cells = 8
        n_signal = min(n_signal, 2)
        n_time = min(n_time, 32)
        max_iter = min(max_iter, 25)
        continuation_steps = min(continuation_steps, 5)

    if args.signal_current_rms_a_values:
        currents = tuple(float(v) for v in args.signal_current_rms_a_values)
        powers = tuple(signal_current_rms_to_power_dbm(v, args.z0_ohm) for v in currents)
    else:
        powers = tuple(float(v) for v in args.signal_power_dbm_values)
        currents = tuple(signal_power_dbm_to_current_rms(v, args.z0_ohm) for v in powers)

    if args.quick and len(currents) > 5:
        currents = currents[:5]
        powers = powers[:5]

    paired = sorted(zip(currents, powers), key=lambda pair: pair[1])
    currents = tuple(float(c) for c, _ in paired)
    powers = tuple(float(p) for _, p in paired)

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
    if args.pump_current_ratio < 0.0:
        raise ValueError("--pump-current-ratio must be non-negative")
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
    if args.target_signal_frequency_ghz is not None:
        if not (args.signal_f_min_ghz <= args.target_signal_frequency_ghz <= args.signal_f_max_ghz):
            raise ValueError("--target-signal-frequency-ghz must lie inside the signal sweep band")
    if not currents:
        raise ValueError("Provide at least one signal-current or signal-power value")
    if any(c <= 0.0 for c in currents):
        raise ValueError("All signal RMS currents must be positive")
    if args.compression_db <= 0.0:
        raise ValueError("--compression-db must be positive")
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
    if args.pump_npz is not None and not Path(args.pump_npz).exists():
        raise FileNotFoundError(args.pump_npz)

    return CompressionSweepConfig(
        n_cells=n_cells,
        length_mm=float(args.length_mm),
        z0_ohm=float(args.z0_ohm),
        phase_velocity_m_per_s=float(args.phase_velocity_m_per_s),
        layout_csv=args.layout_csv,
        pump_npz=args.pump_npz,
        pump_frequency_ghz=float(args.pump_frequency_ghz),
        pump_current_ratio=float(args.pump_current_ratio),
        pump_phase_rad=float(args.pump_phase_rad),
        i_star_a=float(args.i_star_a),
        l0_scale=float(args.l0_scale),
        nonlinear_beta=float(args.nonlinear_beta),
        signal_f_min_ghz=float(args.signal_f_min_ghz),
        signal_f_max_ghz=float(args.signal_f_max_ghz),
        n_signal=n_signal,
        target_signal_frequency_ghz=None
        if args.target_signal_frequency_ghz is None
        else float(args.target_signal_frequency_ghz),
        signal_current_rms_a_values=currents,
        signal_power_dbm_values=powers,
        compression_db=float(args.compression_db),
        reference_mode=str(args.reference_mode),
        harmonic_orders=harmonic_orders,
        n_time=n_time,
        max_iter=max_iter,
        tolerance=float(args.tolerance),
        damping=float(args.damping),
        continuation_steps=continuation_steps,
        pump_solver_mode=str(args.pump_solver_mode),
        pump_numerical_backend=str(args.pump_numerical_backend),
        gain_solver_mode=str(args.gain_solver_mode),
        point_solver_mode=str(args.point_solver_mode),
        require_package_gain_solver=bool(args.require_package_gain_solver),
        allow_partial_pump_fallback=not bool(args.no_allow_partial_pump_fallback),
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
        print(f"[compression] invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    native_allowed, _ = _native_point_solver_allowed(config)
    if native_allowed and config.target_signal_frequency_ghz is None:
        try:
            summary = run_native_wideband_compression(config=config, output_dir=output_dir)
        except Exception:
            error_path = output_dir / "compression_sweep_orchestrator_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[compression] native wideband error: {error_path}", file=sys.stderr)
            return 1
        print(f"[compression] status: {summary['status']}")
        print(f"[compression] summary JSON: {summary['artifact_paths']['summary_json']}")
        return 0 if summary["status"] in {RunStatus.PASS.value, RunStatus.PARTIAL.value} else 1

    metadata: dict[str, Any] = {
        "python": sys.version,
        "script": "scripts/compression_sweep.py",
        "cwd": str(Path.cwd()),
    }

    print("[compression] preparing pump solution...")
    pump = prepare_pump_solution(config, output_dir)
    print(f"[compression] pump preparation: {pump.status.value}")

    points: list[CompressionPointResult] = []

    if pump.status not in {RunStatus.PASS, RunStatus.PARTIAL}:
        elapsed_s = time.perf_counter() - start
        result = export_artifacts(
            config=config,
            pump=pump,
            points=points,
            output_dir=output_dir,
            elapsed_s=elapsed_s,
            metadata=metadata,
        )
        print(f"[compression] status: {result.status.value}")
        return 1

    try:
        for idx, (current, power_dbm) in enumerate(
            zip(config.signal_current_rms_a_values, config.signal_power_dbm_values)
        ):
            print(
                f"[compression] point {idx}: "
                f"Pin={power_dbm:g} dBm, Irms={current:.6g} A"
            )

            point = run_compression_point(
                config=config,
                pump=pump,
                point_index=idx,
                signal_current_rms_a=current,
                signal_power_dbm=power_dbm,
                output_dir=output_dir,
            )
            points.append(point)

            print(
                f"[compression] point {idx}: {point.status.value}, "
                f"gain={point.target_gain_db}"
            )

            if config.fail_fast and point.status in {RunStatus.FAIL, RunStatus.ERROR}:
                raise RuntimeError(
                    f"fail-fast: point {idx} failed with status {point.status.value}"
                )

    except Exception:
        error_path = output_dir / "compression_sweep_orchestrator_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        metadata["orchestrator_error_txt"] = str(error_path)

    points = apply_gain_drop(points, config)

    elapsed_s = time.perf_counter() - start

    result = export_artifacts(
        config=config,
        pump=pump,
        points=points,
        output_dir=output_dir,
        elapsed_s=elapsed_s,
        metadata=metadata,
    )

    print()
    print(f"[compression] status: {result.status.value}")
    print(f"[compression] points: {result.n_points}")
    print(f"[compression] pass/partial/error: {result.n_pass}/{result.n_partial}/{result.n_error}")
    print(f"[compression] summary JSON: {result.artifact_paths.get('summary_json')}")
    print(f"[compression] summary MD:   {result.artifact_paths.get('summary_md')}")

    return 0 if result.status in {RunStatus.PASS, RunStatus.PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())
