"""
Python launcher for Harmonia/JosephsonCircuits Julia simulations.

This module owns the Python side of the production boundary:

    Python launches Julia.
    Julia performs circuit generation / simulation.
    Julia writes status.json + simulation.h5.
    Python reads and validates the result.

It does not implement HB physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import shutil
import subprocess

from twpa.io.julia_bridge import JuliaSimulationStatus, read_status_json


@dataclass(frozen=True)
class JuliaEnginePaths:
    harmonia_jl_root: Path
    julia_executable: str = "julia"
    runner_relative_path: Path = Path("scripts/run_simulation.jl")

    @property
    def runner_path(self) -> Path:
        return self.harmonia_jl_root / self.runner_relative_path

    def validate(self) -> None:
        if not self.harmonia_jl_root.exists():
            raise FileNotFoundError(f"Harmonia.jl root does not exist: {self.harmonia_jl_root}")
        if not self.harmonia_jl_root.is_dir():
            raise NotADirectoryError(f"Harmonia.jl root is not a directory: {self.harmonia_jl_root}")
        if not self.runner_path.exists():
            raise FileNotFoundError(f"Julia runner script does not exist: {self.runner_path}")


@dataclass(frozen=True)
class JuliaRunResult:
    command: tuple[str, ...]
    returncode: int
    output_dir: Path
    status: JuliaSimulationStatus | None
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.status is not None and self.status.status == "PASS"


def build_julia_command(
    *,
    engine: JuliaEnginePaths,
    config_path: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    engine.validate()

    if not config_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    return (
        engine.julia_executable,
        "--project=.",
        str(engine.runner_path),
        "--config",
        str(config_path.resolve()),
        "--output",
        str(output_dir.resolve()),
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run_harmonia_simulation(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    harmonia_jl_root: str | Path,
    julia_executable: str = "julia",
    timeout_s: float | None = 300.0,
    force: bool = False,
    use_cache: bool = True,
) -> JuliaRunResult:
    """
    Launch Harmonia.jl/scripts/run_simulation.jl and return a typed result.

    Cache behavior
    --------------
    If output_dir/status.json already exists and use_cache=True and force=False,
    the existing status is returned without rerunning Julia.

    If force=True, output_dir is deleted first.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    engine = JuliaEnginePaths(
        harmonia_jl_root=Path(harmonia_jl_root),
        julia_executable=julia_executable,
    )

    status_path = output_dir / "status.json"

    if status_path.exists() and use_cache and not force:
        status = read_status_json(status_path)
        return JuliaRunResult(
            command=(),
            returncode=0,
            output_dir=output_dir,
            status=status,
            stdout="CACHE_HIT",
            stderr="",
        )

    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_julia_command(
        engine=engine,
        config_path=config_path,
        output_dir=output_dir,
    )

    try:
        proc = subprocess.run(
            command,
            cwd=engine.harmonia_jl_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""

        _write_text(output_dir / "python_runner_stdout.log", stdout)
        _write_text(output_dir / "python_runner_stderr.log", stderr + f"\nTIMEOUT after {timeout_s} s\n")

        return JuliaRunResult(
            command=command,
            returncode=124,
            output_dir=output_dir,
            status=read_status_json(status_path) if status_path.exists() else None,
            stdout=stdout,
            stderr=stderr,
        )

    _write_text(output_dir / "python_runner_stdout.log", proc.stdout)
    _write_text(output_dir / "python_runner_stderr.log", proc.stderr)

    status = read_status_json(status_path) if status_path.exists() else None

    return JuliaRunResult(
        command=command,
        returncode=proc.returncode,
        output_dir=output_dir,
        status=status,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )