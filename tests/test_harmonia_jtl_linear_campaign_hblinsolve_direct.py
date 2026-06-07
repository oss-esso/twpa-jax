from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_jtl_linear_campaign_cli_hblinsolve_direct_smoke() -> None:
    harmonia_root = Path(os.environ.get("HARMONIA_JL_ROOT", r"D:\Projects\Thesis\Harmonia.jl"))

    if not harmonia_root.exists():
        pytest.skip(f"Harmonia.jl root not found: {harmonia_root}")
    if not (harmonia_root / "scripts" / "run_simulation_batch.jl").exists():
        pytest.skip("run_simulation_batch.jl not available")
    if shutil.which("julia") is None:
        pytest.skip("julia executable not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="jtl_hblinsolve_direct_cli_"))

    try:
        campaign_dir = tmp_root / "campaign"
        cmd = [
            sys.executable,
            "scripts/run_harmonia_jtl_linear_campaign.py",
            "--harmonia-root",
            str(harmonia_root),
            "--campaign-dir",
            str(campaign_dir),
            "--lj-values-h",
            "1e-9",
            "1.1e-9",
            "--use-batch-runner",
            "--jtl-linear-backend",
            "hblinsolve_direct",
            "--force",
        ]

        subprocess.run(cmd, check=True)

        summary = json.loads((campaign_dir / "campaign_summary.json").read_text(encoding="utf-8"))

        assert summary["use_batch_runner"] is True
        assert summary["jtl_linear_backend"] == "hblinsolve_direct"
        assert summary["n_requested"] == 2
        assert summary["n_launched"] == 2
        assert all(run["ok"] for run in summary["runs"])

        run_dirs = [p for p in (campaign_dir / "runs").iterdir() if p.is_dir() and not p.name.startswith("_")]
        assert run_dirs

        for run_dir in run_dirs:
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            assert status["status"] == "PASS"
            assert status["jc_backend"] == "Harmonia.CircuitIR + JosephsonCircuits.hblinsolve"
            assert status["cache_telemetry"]["setup_cache_integration"] == "jtl_hblinsolve_direct"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
