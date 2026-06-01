"""
Run registry for Julia/Harmonia simulation outputs.

The registry is the first campaign-level bookkeeping layer. It turns many
individual Julia run folders into one searchable table.

A run folder is expected to contain:

    status.json
    run_manifest.json       optional but recommended
    simulation.h5           required for PASS runs

The registry writes:

    runs.csv
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import json
import math

from twpa.io.julia_bridge import read_status_json


REGISTRY_COLUMNS = [
    "registered_at_utc",
    "run_id",
    "status",
    "simulation_type",
    "circuit_template",
    "solver_success",
    "residual_norm",
    "relative_residual_norm",
    "failure_reason",
    "runtime_s",
    "random_seed",
    "output_dir",
    "h5_path",
    "config_hash_sha1",
    "config_path",
    "schema_version",
    "julia_version",
    "josephsoncircuits_version",
    "harmonia_commit",
    "python_commit",
]


@dataclass(frozen=True)
class RegisteredRun:
    registered_at_utc: str
    run_id: str
    status: str
    simulation_type: str
    circuit_template: str
    solver_success: bool
    residual_norm: float | None
    relative_residual_norm: float | None
    failure_reason: str | None
    runtime_s: float | None
    random_seed: int | None
    output_dir: str
    h5_path: str | None
    config_hash_sha1: str | None
    config_path: str | None
    schema_version: str
    julia_version: str | None
    josephsoncircuits_version: str | None
    harmonia_commit: str | None
    python_commit: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)!r}")
    return obj


def _clean_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite registry float: {value}")
    return value


def registered_run_from_dir(run_dir: str | Path) -> RegisteredRun:
    run_path = Path(run_dir)
    status = read_status_json(run_path)
    manifest = _read_json_if_exists(run_path / "run_manifest.json")

    return RegisteredRun(
        registered_at_utc=utc_now_iso(),
        run_id=status.run_id,
        status=status.status,
        simulation_type=status.simulation_type,
        circuit_template=status.circuit_template,
        solver_success=status.solver_success,
        residual_norm=_clean_optional_float(status.residual_norm),
        relative_residual_norm=_clean_optional_float(status.relative_residual_norm),
        failure_reason=status.failure_reason,
        runtime_s=_clean_optional_float(status.runtime_s),
        random_seed=status.random_seed,
        output_dir=str(status.output_dir or run_path.resolve()),
        h5_path=str(status.h5_path) if status.h5_path is not None else None,
        config_hash_sha1=manifest.get("config_hash_sha1"),
        config_path=manifest.get("config_path") or status.raw.get("config_path"),
        schema_version=status.schema_version,
        julia_version=status.julia_version,
        josephsoncircuits_version=status.josephsoncircuits_version,
        harmonia_commit=status.harmonia_commit,
        python_commit=status.python_commit,
    )


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def append_registered_run(registry_csv: str | Path, run: RegisteredRun) -> None:
    registry_path = Path(registry_csv)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = registry_path.exists()
    row = asdict(run)

    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)

        if not file_exists:
            writer.writeheader()

        writer.writerow({key: _csv_value(row.get(key)) for key in REGISTRY_COLUMNS})


def register_run_dir(registry_csv: str | Path, run_dir: str | Path) -> RegisteredRun:
    run = registered_run_from_dir(run_dir)
    append_registered_run(registry_csv, run)
    return run


def read_registry(registry_csv: str | Path) -> list[dict[str, str]]:
    path = Path(registry_csv)
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def registry_summary(registry_csv: str | Path) -> dict[str, Any]:
    rows = read_registry(registry_csv)

    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}

    for row in rows:
        by_status[row.get("status", "UNKNOWN")] = by_status.get(row.get("status", "UNKNOWN"), 0) + 1
        by_type[row.get("simulation_type", "UNKNOWN")] = by_type.get(row.get("simulation_type", "UNKNOWN"), 0) + 1

    return {
        "registry_csv": str(Path(registry_csv)),
        "n_runs": len(rows),
        "by_status": by_status,
        "by_simulation_type": by_type,
    }