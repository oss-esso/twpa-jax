from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from twpa.io.julia_batch_runner import run_harmonia_simulation_batch


def _write_linear_config(path: Path, *, length_m: float) -> None:
    payload = {
        "schema_version": "0.1.0",
        "simulation_type": "linear_sparams",
        "circuit_template": "analytic_transmission_line",
        "parameters": {
            "z_ref_ohm": 50.0,
            "z_line_ohm": 50.0,
            "length_m": length_m,
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
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_harmonia_simulation_batch_linear_smoke() -> None:
    harmonia_root = Path(os.environ.get("HARMONIA_JL_ROOT", r"D:\Projects\Thesis\Harmonia.jl"))

    if not harmonia_root.exists():
        pytest.skip(f"Harmonia.jl root not found: {harmonia_root}")
    if not (harmonia_root / "scripts" / "run_simulation_batch.jl").exists():
        pytest.skip("run_simulation_batch.jl not available")
    if shutil.which("julia") is None:
        pytest.skip("julia executable not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="twpa_julia_batch_runner_"))

    try:
        config1 = tmp_root / "configs" / "linear_001.json"
        config2 = tmp_root / "configs" / "linear_002.json"
        out1 = tmp_root / "runs" / "linear_001"
        out2 = tmp_root / "runs" / "linear_002"

        _write_linear_config(config1, length_m=0.10)
        _write_linear_config(config2, length_m=0.11)

        result = run_harmonia_simulation_batch(
            items=[(config1, out1), (config2, out2)],
            harmonia_jl_root=harmonia_root,
            timeout_s=180.0,
            force=True,
            batch_work_dir=tmp_root / "_batch",
        )

        assert result.ok
        assert result.returncode == 0
        assert result.n_total == 2
        assert result.n_pass == 2
        assert result.n_fail == 0
        assert result.summary_path.exists()
        assert result.manifest_path.exists()
        assert out1.joinpath("status.json").exists()
        assert out2.joinpath("status.json").exists()
        assert len(result.command) > 0

        cached = run_harmonia_simulation_batch(
            items=[(config1, out1), (config2, out2)],
            harmonia_jl_root=harmonia_root,
            timeout_s=180.0,
            force=False,
            use_cache=True,
            batch_work_dir=tmp_root / "_batch_cached",
        )

        assert cached.ok
        assert cached.returncode == 0
        assert cached.n_total == 2
        assert cached.n_pass == 2
        assert cached.stdout == "CACHE_HIT"
        assert cached.command == ()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
