from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from twpa.io.campaigns import run_parameter_campaign


def _make_linear_config(index: int, value: float) -> dict:
    return {
        "schema_version": "0.1.0",
        "simulation_type": "linear_sparams",
        "circuit_template": "analytic_transmission_line",
        "parameters": {
            "z_ref_ohm": 50.0,
            "z_line_ohm": 50.0,
            "length_m": float(value),
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


def test_run_parameter_campaign_batch_runner_linear_smoke() -> None:
    harmonia_root = Path(os.environ.get("HARMONIA_JL_ROOT", r"D:\Projects\Thesis\Harmonia.jl"))

    if not harmonia_root.exists():
        pytest.skip(f"Harmonia.jl root not found: {harmonia_root}")
    if not (harmonia_root / "scripts" / "run_simulation_batch.jl").exists():
        pytest.skip("run_simulation_batch.jl not available")
    if shutil.which("julia") is None:
        pytest.skip("julia executable not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="twpa_campaign_batch_runner_"))

    try:
        summary = run_parameter_campaign(
            values=[0.10, 0.11],
            parameter_name="length_m",
            campaign_type="linear_sparams_batch_smoke",
            harmonia_root=harmonia_root,
            campaign_dir=tmp_root / "campaign",
            make_config=_make_linear_config,
            run_name=lambda value: f"length_{int(round(float(value) * 1000)):03d}",
            timeout_s=180.0,
            force=True,
            use_batch_runner=True,
        )

        assert summary["use_batch_runner"] is True
        assert summary["n_requested"] == 2
        assert summary["n_launched"] == 2
        assert all(run["ok"] for run in summary["runs"])
        assert all(run["batch_runner"] is True for run in summary["runs"])

        batch_summary = tmp_root / "campaign" / "runs" / "_julia_batch_runner" / "batch_summary.json"
        assert batch_summary.exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
