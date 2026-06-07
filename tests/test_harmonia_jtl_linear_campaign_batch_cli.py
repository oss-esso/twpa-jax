from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_jtl_linear_campaign_cli_batch_runner_smoke() -> None:
    harmonia_root = Path(os.environ.get("HARMONIA_JL_ROOT", r"D:\Projects\Thesis\Harmonia.jl"))

    if not harmonia_root.exists():
        pytest.skip(f"Harmonia.jl root not found: {harmonia_root}")
    if not (harmonia_root / "scripts" / "run_simulation_batch.jl").exists():
        pytest.skip("run_simulation_batch.jl not available")
    if shutil.which("julia") is None:
        pytest.skip("julia executable not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="jtl_linear_campaign_batch_cli_"))

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
            "--force",
        ]

        subprocess.run(cmd, check=True)

        summary_path = campaign_dir / "campaign_summary.json"
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["use_batch_runner"] is True
        assert summary["n_requested"] == 2
        assert summary["n_launched"] == 2
        assert all(run["ok"] for run in summary["runs"])
        assert all(run["batch_runner"] is True for run in summary["runs"])

        batch_summary = campaign_dir / "runs" / "_julia_batch_runner" / "batch_summary.json"
        assert batch_summary.exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
