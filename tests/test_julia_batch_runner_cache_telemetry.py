from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from twpa.io.julia_batch_runner import run_harmonia_simulation_batch


def _write_linear_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "simulation_type": "linear_sparams",
                "circuit_template": "analytic_transmission_line",
                "parameters": {
                    "z_ref_ohm": 50.0,
                    "z_line_ohm": 50.0,
                    "length_m": 0.1,
                    "phase_velocity_m_per_s": 1.2e8,
                    "attenuation_np_per_m": 0.0,
                },
                "axes": {
                    "frequency_hz": {
                        "start": 4.0e9,
                        "stop": 4.1e9,
                        "points": 2,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_julia_batch_runner_exposes_cache_telemetry() -> None:
    harmonia_root = Path(os.environ.get("HARMONIA_JL_ROOT", r"D:\Projects\Thesis\Harmonia.jl"))

    if not harmonia_root.exists():
        pytest.skip(f"Harmonia.jl root not found: {harmonia_root}")
    if not (harmonia_root / "scripts" / "run_simulation_batch.jl").exists():
        pytest.skip("run_simulation_batch.jl not available")
    if shutil.which("julia") is None:
        pytest.skip("julia executable not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="twpa_batch_cache_telemetry_"))

    try:
        config = tmp_root / "configs" / "linear.json"
        _write_linear_config(config)

        result = run_harmonia_simulation_batch(
            items=[
                (config, tmp_root / "runs" / "run_001"),
                (config, tmp_root / "runs" / "run_002"),
            ],
            harmonia_jl_root=harmonia_root,
            timeout_s=180.0,
            force=True,
            batch_work_dir=tmp_root / "_batch",
        )

        assert result.ok
        assert result.cache_telemetry is not None
        assert result.cache_telemetry["julia_process_reused"] is True
        assert result.cache_telemetry["setup_cache_integration"] == "not_wired"
        assert result.cache_telemetry["hbcompiled_circuit_base_enabled"] is False
        assert result.cache_telemetry["hbnumeric_matrix_cache_enabled"] is False
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
