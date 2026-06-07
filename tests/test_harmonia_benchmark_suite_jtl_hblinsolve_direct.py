from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_benchmark_suite_jtl_hblinsolve_direct_smoke() -> None:
    harmonia_root = Path(os.environ.get("HARMONIA_JL_ROOT", r"D:\Projects\Thesis\Harmonia.jl"))

    if not harmonia_root.exists():
        pytest.skip(f"Harmonia.jl root not found: {harmonia_root}")
    if not (harmonia_root / "scripts" / "run_simulation_batch.jl").exists():
        pytest.skip("run_simulation_batch.jl not available")
    if shutil.which("julia") is None:
        pytest.skip("julia executable not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="benchmark_jtl_hblinsolve_direct_"))

    try:
        benchmark_dir = tmp_root / "benchmark"
        cmd = [
            sys.executable,
            "scripts/run_harmonia_benchmark_suite.py",
            "--harmonia-root",
            str(harmonia_root),
            "--benchmark-dir",
            str(benchmark_dir),
            "--suite",
            "minimal",
            "--repetitions",
            "1",
            "--use-batch-runner",
            "--jtl-linear-backend",
            "hblinsolve_direct",
            "--force",
        ]

        subprocess.run(cmd, check=True)

        with (benchmark_dir / "benchmark_runs.csv").open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows
        assert all(row["status"] == "PASS" for row in rows)

        summary = json.loads((benchmark_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
        assert summary["use_batch_runner"] is True
        assert summary["jtl_linear_backend"] == "hblinsolve_direct"

        jtl_status_files = list((benchmark_dir / "runs").glob("jtl_linear*/status.json"))
        assert jtl_status_files

        for status_path in jtl_status_files:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            assert status["status"] == "PASS"
            assert status["jc_backend"] == "Harmonia.CircuitIR + JosephsonCircuits.hblinsolve"
            assert status["cache_telemetry"]["setup_cache_integration"] == "jtl_hblinsolve_direct"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
