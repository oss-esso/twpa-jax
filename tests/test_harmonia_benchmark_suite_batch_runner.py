from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_harmonia_benchmark_suite_batch_runner_minimal_smoke() -> None:
    harmonia_root = Path(os.environ.get("HARMONIA_JL_ROOT", r"D:\Projects\Thesis\Harmonia.jl"))

    if not harmonia_root.exists():
        pytest.skip(f"Harmonia.jl root not found: {harmonia_root}")
    if not (harmonia_root / "scripts" / "run_simulation_batch.jl").exists():
        pytest.skip("run_simulation_batch.jl not available")
    if shutil.which("julia") is None:
        pytest.skip("julia executable not found")

    tmp_root = Path(tempfile.mkdtemp(prefix="benchmark_suite_batch_runner_"))

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
            "--force",
        ]

        subprocess.run(cmd, check=True)

        results_csv = benchmark_dir / "benchmark_results.csv"
        runs_csv = benchmark_dir / "benchmark_runs.csv"
        assert results_csv.exists()
        assert runs_csv.exists()

        with results_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows
        assert all(row["status"] == "PASS" for row in rows)
        assert all(row["batch_runner"] == "True" for row in rows)

        batch_summary = benchmark_dir / "runs" / "_julia_batch_runner" / "batch_summary.json"
        assert batch_summary.exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
