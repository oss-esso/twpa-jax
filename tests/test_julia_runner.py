from __future__ import annotations

import json
from pathlib import Path

import pytest

from twpa.io.julia_runner import (
    JuliaEnginePaths,
    build_julia_command,
    run_harmonia_simulation,
)


def _write_status(run_dir: Path, status: str = "PASS") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "run_id": "cached",
                "status": status,
                "simulation_type": "schema_smoke",
                "circuit_template": "matched_through_2port",
                "solver_success": status == "PASS",
                "residual_norm": None,
                "relative_residual_norm": None,
                "failure_reason": None if status == "PASS" else "test failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_julia_command(tmp_path: Path) -> None:
    harmonia = tmp_path / "Harmonia.jl"
    scripts = harmonia / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_simulation.jl"
    runner.write_text("# runner\n", encoding="utf-8")

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    output = tmp_path / "out"

    cmd = build_julia_command(
        engine=JuliaEnginePaths(harmonia_jl_root=harmonia),
        config_path=config,
        output_dir=output,
    )

    assert cmd[0] == "julia"
    assert "--project=." in cmd
    assert "--config" in cmd
    assert "--output" in cmd
    assert str(config.resolve()) in cmd
    assert str(output.resolve()) in cmd


def test_runner_returns_cached_status_without_julia(tmp_path: Path) -> None:
    run_dir = tmp_path / "cached_run"
    _write_status(run_dir, "PASS")

    result = run_harmonia_simulation(
        config_path=tmp_path / "does_not_matter.json",
        output_dir=run_dir,
        harmonia_jl_root=tmp_path / "does_not_matter",
        use_cache=True,
        force=False,
    )

    assert result.ok
    assert result.status is not None
    assert result.status.status == "PASS"
    assert result.stdout == "CACHE_HIT"


def test_actual_schema_smoke_launch_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    config_path = harmonia_root / "examples" / "configs" / "schema_smoke.json"

    if not harmonia_root.exists() or not config_path.exists():
        pytest.skip("Local Harmonia.jl schema-smoke setup not available.")

    output_dir = tmp_path / "schema_smoke"

    result = run_harmonia_simulation(
        config_path=config_path,
        output_dir=output_dir,
        harmonia_jl_root=harmonia_root,
        force=True,
        timeout_s=120.0,
    )

    assert result.returncode == 0
    assert result.status is not None
    assert result.status.status == "PASS"
    assert (output_dir / "simulation.h5").exists()