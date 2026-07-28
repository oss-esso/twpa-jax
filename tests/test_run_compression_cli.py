from __future__ import annotations

import json

from scripts.run_compression import main


def test_run_compression_smoke_writes_artifacts(tmp_path) -> None:
    assert main(["--output-dir", str(tmp_path), "--n-signal-power", "5"]) == 0
    assert (tmp_path / "compression_points.csv").exists()
    assert (tmp_path / "compression_arrays.npz").exists()
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    assert summary["stability_status"] == "NOT_CHECKED"
