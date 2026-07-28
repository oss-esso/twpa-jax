from __future__ import annotations

import json

import pytest

from scripts.run_compression import _resolve_attenuation, build_parser, main


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
    assert default.signal_ghz is None


def test_signal_frequency_defaults_to_pump_frequency(tmp_path) -> None:
    assert main(["--output-dir", str(tmp_path), "--n-signal-power", "1"]) == 0
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    assert summary["signal_ghz"] == summary["pump_freq_ghz"]


def test_no_gain_operating_point_suppresses_compression(tmp_path) -> None:
    assert main(
        [
            "--output-dir",
            str(tmp_path),
            "--signal-ghz",
            "4.5",
            "--n-signal-power",
            "2",
        ]
    ) == 0
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    points = (tmp_path / "compression_points.csv").read_text()
    assert summary["status"] == "NO_GAIN_AT_OPERATING_POINT"
    assert summary["p1db"] is None
    assert "nan" in points


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


def test_fixture_and_circuit_attenuation_defaults_are_distinct() -> None:
    parser = build_parser()
    fixture = parser.parse_args(["--output-dir", "unused", "--fixture", "jpa"])
    circuit = parser.parse_args(
        ["--output-dir", "unused", "--circuit-dir", "design"]
    )
    explicit = parser.parse_args(
        ["--output-dir", "unused", "--fixture", "jpa", "--attenuation-db", "7"]
    )
    assert _resolve_attenuation(fixture) == (0.0, "fixture_default_zero")
    assert _resolve_attenuation(circuit)[1] == "themis_default_loss_model"
    assert _resolve_attenuation(explicit) == (7.0, "explicit")


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
