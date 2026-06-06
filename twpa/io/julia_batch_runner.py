"""Batch launcher for Harmonia/JosephsonCircuits Julia simulations.

This module is intentionally separate from twpa.io.julia_runner.

The existing one-shot path remains:
    Python -> subprocess.run(julia scripts/run_simulation.jl ...)

This batch path is opt-in:
    Python -> one subprocess.run(julia scripts/run_simulation_batch.jl ...)
           -> many run_simulation.jl::main calls inside one Julia process
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from twpa.io.julia_bridge import JuliaSimulationStatus, read_status_json
from twpa.io.julia_runner import JuliaEnginePaths


@dataclass(frozen=True)
class JuliaBatchRunItem:
    config_path: Path
    output_dir: Path


@dataclass(frozen=True)
class JuliaBatchRunRecord:
    index: int
    config_path: Path
    output_dir: Path
    returncode: int
    status: str | None
    ok: bool
    runtime_s: float | None
    failure_reason: str | None


@dataclass(frozen=True)
class JuliaBatchRunResult:
    command: tuple[str, ...]
    returncode: int
    summary_path: Path
    manifest_path: Path
    records: tuple[JuliaBatchRunRecord, ...]
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and all(record.ok for record in self.records)

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_pass(self) -> int:
        return sum(1 for record in self.records if record.ok)

    @property
    def n_fail(self) -> int:
        return self.n_total - self.n_pass


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _batch_runner_path(engine: JuliaEnginePaths) -> Path:
    return engine.harmonia_jl_root / "scripts" / "run_simulation_batch.jl"


def build_julia_batch_command(
    *,
    engine: JuliaEnginePaths,
    manifest_path: Path,
    summary_path: Path,
) -> tuple[str, ...]:
    engine.validate()

    batch_runner = _batch_runner_path(engine)
    if not batch_runner.exists():
        raise FileNotFoundError(f"Julia batch runner script does not exist: {batch_runner}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {manifest_path}")

    return (
        engine.julia_executable,
        "--project=.",
        str(batch_runner),
        "--manifest",
        str(manifest_path.resolve()),
        "--summary",
        str(summary_path.resolve()),
    )


def _normalize_items(items: Iterable[tuple[str | Path, str | Path] | JuliaBatchRunItem]) -> list[JuliaBatchRunItem]:
    out: list[JuliaBatchRunItem] = []

    for item in items:
        if isinstance(item, JuliaBatchRunItem):
            config_path = Path(item.config_path)
            output_dir = Path(item.output_dir)
        else:
            config_path = Path(item[0])
            output_dir = Path(item[1])

        if not config_path.exists():
            raise FileNotFoundError(f"Config path does not exist: {config_path}")

        out.append(JuliaBatchRunItem(config_path=config_path, output_dir=output_dir))

    if not out:
        raise ValueError("Batch run requires at least one item")

    return out


def _record_from_status(index: int, config_path: Path, output_dir: Path, status: JuliaSimulationStatus) -> JuliaBatchRunRecord:
    return JuliaBatchRunRecord(
        index=index,
        config_path=config_path,
        output_dir=output_dir,
        returncode=0,
        status=status.status,
        ok=status.status == "PASS",
        runtime_s=status.runtime_s,
        failure_reason=status.failure_reason,
    )


def _record_from_summary(row: dict[str, Any]) -> JuliaBatchRunRecord:
    return JuliaBatchRunRecord(
        index=int(row["index"]),
        config_path=Path(str(row["config"])),
        output_dir=Path(str(row["output"])),
        returncode=int(row["returncode"]),
        status=None if row.get("status") is None else str(row.get("status")),
        ok=bool(row.get("ok")),
        runtime_s=None if row.get("runtime_s") is None else float(row.get("runtime_s")),
        failure_reason=None if row.get("failure_reason") is None else str(row.get("failure_reason")),
    )


def run_harmonia_simulation_batch(
    *,
    items: Iterable[tuple[str | Path, str | Path] | JuliaBatchRunItem],
    harmonia_jl_root: str | Path,
    julia_executable: str = "julia",
    timeout_s: float | None = 600.0,
    force: bool = False,
    use_cache: bool = True,
    batch_work_dir: str | Path | None = None,
) -> JuliaBatchRunResult:
    """Run multiple Harmonia simulations through one Julia batch process.

    Existing output_dir/status.json files are reused when use_cache=True and force=False.
    Only uncached items are sent to the Julia batch runner.
    """
    normalized = _normalize_items(items)
    engine = JuliaEnginePaths(
        harmonia_jl_root=Path(harmonia_jl_root),
        julia_executable=julia_executable,
        runner_relative_path=Path("scripts/run_simulation.jl"),
    )
    engine.validate()

    if batch_work_dir is None:
        common_parent = normalized[0].output_dir.parent
        batch_work_dir = common_parent / "_julia_batch_runner"

    batch_dir = Path(batch_work_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    cached_records: list[JuliaBatchRunRecord] = []
    to_run: list[JuliaBatchRunItem] = []

    for index, item in enumerate(normalized, start=1):
        status_path = item.output_dir / "status.json"

        if force and item.output_dir.exists():
            shutil.rmtree(item.output_dir)

        if status_path.exists() and use_cache and not force:
            cached_records.append(_record_from_status(index, item.config_path, item.output_dir, read_status_json(status_path)))
        else:
            item.output_dir.mkdir(parents=True, exist_ok=True)
            to_run.append(item)

    manifest_path = batch_dir / "batch_manifest.json"
    summary_path = batch_dir / "batch_summary.json"

    launched_records: list[JuliaBatchRunRecord] = []
    stdout = ""
    stderr = ""
    command: tuple[str, ...] = ()
    returncode = 0

    if to_run:
        manifest_rows = [
            {
                "config": str(item.config_path.resolve()),
                "output": str(item.output_dir.resolve()),
            }
            for item in to_run
        ]
        _write_json(manifest_path, {"runs": manifest_rows})

        command = build_julia_batch_command(
            engine=engine,
            manifest_path=manifest_path,
            summary_path=summary_path,
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
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            returncode = 124

        _write_text(batch_dir / "python_batch_runner_stdout.log", stdout)
        _write_text(batch_dir / "python_batch_runner_stderr.log", stderr)

        if summary_path.exists():
            summary = _read_json(summary_path)
            launched_records = [_record_from_summary(row) for row in summary.get("runs", [])]
        else:
            launched_records = [
                JuliaBatchRunRecord(
                    index=i,
                    config_path=item.config_path,
                    output_dir=item.output_dir,
                    returncode=returncode,
                    status=None,
                    ok=False,
                    runtime_s=None,
                    failure_reason="missing batch_summary.json",
                )
                for i, item in enumerate(to_run, start=1)
            ]
    else:
        _write_json(manifest_path, {"runs": []})
        _write_json(
            summary_path,
            {
                "status": "PASS",
                "batch_runner": "python_cache_only",
                "n_total": len(cached_records),
                "n_pass": sum(1 for record in cached_records if record.ok),
                "n_fail": sum(1 for record in cached_records if not record.ok),
                "runs": [],
            },
        )
        stdout = "CACHE_HIT"
        stderr = ""
        returncode = 0

    records = tuple(sorted(cached_records + launched_records, key=lambda r: r.index))

    if returncode == 0 and any(not record.ok for record in records):
        returncode = 1

    return JuliaBatchRunResult(
        command=command,
        returncode=returncode,
        summary_path=summary_path,
        manifest_path=manifest_path,
        records=records,
        stdout=stdout,
        stderr=stderr,
    )
