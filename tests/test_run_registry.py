from __future__ import annotations

import csv
import json
from pathlib import Path

from twpa.io.run_registry import (
    read_registry,
    register_run_dir,
    registry_summary,
)


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")


def _make_run_dir(tmp_path: Path, status: str = "PASS") -> Path:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_json(
        run_dir / "status.json",
        {
            "schema_version": "0.1.0",
            "run_id": "abc-123",
            "status": status,
            "simulation_type": "schema_smoke",
            "circuit_template": "matched_through_2port",
            "solver_success": status == "PASS",
            "residual_norm": None,
            "relative_residual_norm": None,
            "failure_reason": None if status == "PASS" else "intentional failure",
            "runtime_s": 1.25,
            "random_seed": 1234,
            "julia_version": "1.12.6",
            "josephsoncircuits_version": "NOT_USED",
            "harmonia_commit": "UNKNOWN",
            "python_commit": "UNKNOWN",
            "output_dir": str(run_dir),
            "h5_path": str(run_dir / "simulation.h5"),
        },
    )

    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "0.1.0",
            "run_id": "abc-123",
            "config_hash_sha1": "deadbeef",
            "config_path": "config.json",
        },
    )

    return run_dir


def test_register_run_dir_writes_csv(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, "PASS")
    registry = tmp_path / "runs.csv"

    registered = register_run_dir(registry, run_dir)

    assert registered.run_id == "abc-123"
    assert registered.status == "PASS"
    assert registry.exists()

    rows = read_registry(registry)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "abc-123"
    assert rows[0]["status"] == "PASS"
    assert rows[0]["simulation_type"] == "schema_smoke"
    assert rows[0]["config_hash_sha1"] == "deadbeef"


def test_registry_summary_counts_status_and_type(tmp_path: Path) -> None:
    registry = tmp_path / "runs.csv"

    run1 = _make_run_dir(tmp_path / "a", "PASS")
    run2 = _make_run_dir(tmp_path / "b", "FAIL")

    register_run_dir(registry, run1)
    register_run_dir(registry, run2)

    summary = registry_summary(registry)

    assert summary["n_runs"] == 2
    assert summary["by_status"] == {"PASS": 1, "FAIL": 1}
    assert summary["by_simulation_type"] == {"schema_smoke": 2}


def test_registry_csv_has_expected_columns(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, "PASS")
    registry = tmp_path / "runs.csv"

    register_run_dir(registry, run_dir)

    with registry.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

    assert columns is not None
    assert "run_id" in columns
    assert "status" in columns
    assert "simulation_type" in columns
    assert "output_dir" in columns
    assert "config_hash_sha1" in columns