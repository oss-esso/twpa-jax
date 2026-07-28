from __future__ import annotations

import json

import pytest

from scripts.run_compression import (
    _build_multitone_basis,
    _interpolate_p1db_current,
    _resolve_attenuation,
    build_parser,
    main,
)


def test_multitone_preconditioner_defaults_exact_and_accepts_sector() -> None:
    parser = build_parser()
    default = parser.parse_args(["--output-dir", "unused", "--signal-ghz", "4.5"])
    sector = parser.parse_args(
        [
            "--output-dir",
            "unused",
            "--signal-ghz",
            "4.5",
            "--multitone-preconditioner",
            "floquet_sector",
        ]
    )
    assert default.multitone_preconditioner == "real_coupled_fast"
    assert default.multitone_basis == "matched"
    assert default.multitone_sidebands == 2
    assert sector.multitone_preconditioner == "floquet_sector"
    assert default.signal_ghz == 4.5


def test_signal_frequency_is_required() -> None:
    with pytest.raises(SystemExit):
        main(["--output-dir", "unused"])


def test_p1db_interpolation_is_logarithmic_in_current() -> None:
    points = [
        {"signal_current_a": 1.0e-9, "compression_db": 0.5, "status": "VALID_SOLVED"},
        {"signal_current_a": 1.0e-8, "compression_db": 1.5, "status": "VALID_SOLVED"},
    ]
    assert _interpolate_p1db_current(points) == pytest.approx(10.0 ** -8.5)


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
                "--signal-ghz",
                "4.5",
                "--multitone-preconditioner",
                "unknown",
            ]
        )


def test_three_tone_guard_names_missing_pump_modes() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "unused",
            "--signal-ghz",
            "4.5",
            "--multitone-basis",
            "three_tone",
        ]
    )
    with pytest.raises(ValueError, match=r"pump_modes=\[1, 3\].*missing=\[3\]"):
        _build_multitone_basis(args, [1, 3], 10.0, 1.0)


def test_fixture_and_circuit_attenuation_defaults_are_distinct() -> None:
    parser = build_parser()
    fixture = parser.parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--fixture", "jpa"]
    )
    circuit = parser.parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--circuit-dir", "design"]
    )
    explicit = parser.parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--fixture", "jpa", "--attenuation-db", "7"]
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
            "--signal-ghz",
            "4.5",
            "--multitone-preconditioner",
            "floquet_sector",
        ]
    ) == 0
    summary = json.loads(
        (tmp_path / "compression_summary.json").read_text()
    )
    assert summary["multitone_preconditioner"] == "floquet_sector"


def test_run_compression_smoke_writes_artifacts(tmp_path) -> None:
    assert main(
        [
            "--output-dir",
            str(tmp_path),
            "--signal-ghz",
            "4.5",
            "--n-signal-power",
            "5",
        ]
    ) == 0
    assert (tmp_path / "compression_points.csv").exists()
    assert (tmp_path / "compression_arrays.npz").exists()
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    points = (tmp_path / "compression_points.csv").read_text()
    assert summary["stability_status"] == "NOT_CHECKED"
    assert summary["multitone_preconditioner"] == "real_coupled_fast"
    assert "pump_depletion_db" in points
    assert "signal_s21_real" in points
    assert "pump_s21_real" in points
    assert "idler_s21_real" in points


def test_spatial_profile_flag_is_explicit() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5"]
    )
    assert args.spatial_profiles is False
