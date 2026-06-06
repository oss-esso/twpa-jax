from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def test_run_jc_cached_setup_workload_smoke() -> None:
    jc_repo = Path(os.environ.get("JOSEPHSONCIRCUITS_REPO", r"D:\Projects\Thesis\JosephsonCircuits.jl"))
    script = Path("scripts/run_jc_cached_setup_workload.py")

    if not jc_repo.exists():
        return
    if shutil.which("julia") is None:
        return

    tmp_root = Path(tempfile.mkdtemp(prefix="jc_cached_setup_workload_smoke_"))
    outdir = tmp_root / "jc_cached_setup_workload_smoke"

    cmd = [
        sys.executable,
        str(script),
        "--jc-repo",
        str(jc_repo),
        "--outdir",
        str(outdir),
        "--scenario",
        "mixed",
        "--cells",
        "1",
        "--nmodes",
        "1",
        "--requests-per-case",
        "2",
        "--repetitions",
        "1",
        "--force",
    ]

    try:
        subprocess.run(cmd, check=True)

        aggregate_csv = outdir / "jc_cached_setup_workload_results.csv"
        assert aggregate_csv.exists()

        with aggregate_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        row = rows[0]
        assert row["scenario"] == "mixed"
        assert int(row["equivalence_true"]) > 0
        assert int(row["equivalence_false"]) == 0
        assert float(row["speedup"]) > 0.0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
