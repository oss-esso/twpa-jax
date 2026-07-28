from __future__ import annotations

import json

import pytest

from scripts.run_compression import build_parser, main


def test_multitone_preconditioner_defaults_exact_and_accepts_sector() -> None:
    parser = build_parser()
    default = parser.parse_args(["--output-dir", "unused"])
    sector = parser.parse_args(
        [
            "--output-dir",
            "unused",
            "--multitone-preconditioner",
            "floquet_sector",
        ]
    )
    assert default.multitone_preconditioner == "real_coupled_fast"
    assert sector.multitone_preconditioner == "floquet_sector"


def test_multitone_preconditioner_rejects_unknown_name() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--output-dir",
                "unused",
                "--multitone-preconditioner",
                "unknown",
            ]
        )


def test_run_compression_uses_sector_preconditioner(tmp_path) -> None:
    assert main(
        [
            "--output-dir",
            str(tmp_path),
            "--n-signal-power",
            "2",
            "--multitone-preconditioner",
            "floquet_sector",
        ]
    ) == 0
    summary = json.loads(
        (tmp_path / "compression_summary.json").read_text()
    )
    assert summary["multitone_preconditioner"] == "floquet_sector"


def test_run_compression_smoke_writes_artifacts(tmp_path) -> None:
    assert main(["--output-dir", str(tmp_path), "--n-signal-power", "5"]) == 0
    assert (tmp_path / "compression_points.csv").exists()
    assert (tmp_path / "compression_arrays.npz").exists()
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    assert summary["stability_status"] == "NOT_CHECKED"
    assert summary["multitone_preconditioner"] == "real_coupled_fast"
