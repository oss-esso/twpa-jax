from __future__ import annotations

import json
from pathlib import Path

from workflows import build_design_and_passive


def test_build_workflow_sequences_multiple_yaml_designs(
    tmp_path: Path, monkeypatch,
) -> None:
    design_a = tmp_path / "design_a.yaml"
    design_b = tmp_path / "design_b.yaml"
    output_root = tmp_path / "passive"
    events: list[tuple[str, str, str]] = []

    def fake_compile(argv: list[str]) -> None:
        design = argv[argv.index("--design") + 1]
        outdir = Path(argv[argv.index("--outdir") + 1])
        events.append(("build", design, str(outdir)))
        outdir.mkdir(parents=True, exist_ok=True)

    def fake_passive(design_dir: Path, _args: object) -> None:
        events.append(("passive", str(design_dir), str(design_dir)))

    def fake_coupler(design_dir: Path, _args: object, mode_override=None) -> None:
        events.append(("coupler", str(design_dir), str(mode_override)))

    monkeypatch.setattr(build_design_and_passive, "compile_design_main", fake_compile)
    monkeypatch.setattr(build_design_and_passive, "_write_passive", fake_passive)
    monkeypatch.setattr(build_design_and_passive, "_write_coupler_passive", fake_coupler)

    result = build_design_and_passive.main([
        "--design", str(design_a), str(design_b),
        "--design-dir", str(output_root),
    ])

    assert result == 0
    assert events == [
        ("build", str(design_a), str(output_root / "design_a")),
        ("passive", str(output_root / "design_a"), str(output_root / "design_a")),
        ("coupler", str(output_root / "design_a"), "None"),
        ("build", str(design_b), str(output_root / "design_b")),
        ("passive", str(output_root / "design_b"), str(output_root / "design_b")),
        ("coupler", str(output_root / "design_b"), "None"),
    ]


def test_build_workflow_skips_isolated_coupler_for_no_coupler_design(
    tmp_path: Path, monkeypatch,
) -> None:
    design = tmp_path / "design.yaml"
    output_dir = tmp_path / "compiled"
    events: list[str] = []

    def fake_compile(argv: list[str]) -> None:
        outdir = Path(argv[argv.index("--outdir") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "design_resolved.json").write_text(
            json.dumps({"blocks": [{"type": "port"}]}),
            encoding="utf-8",
        )

    monkeypatch.setattr(build_design_and_passive, "compile_design_main", fake_compile)
    monkeypatch.setattr(
        build_design_and_passive,
        "_write_passive",
        lambda _design_dir, _args: events.append("passive"),
    )
    monkeypatch.setattr(
        build_design_and_passive,
        "_write_coupler_passive",
        lambda _design_dir, _args, mode_override=None: events.append("coupler"),
    )

    result = build_design_and_passive.main([
        "--design", str(design), "--design-dir", str(output_dir),
    ])

    assert result == 0
    assert events == ["passive"]


def test_coupler_passive_settings_use_effective_block_override(
    tmp_path: Path,
) -> None:
    resolved = tmp_path / "design_resolved.json"
    resolved.write_text(
        json.dumps({
            "parameters": {
                "coupling_dB": -14.0,
                "coupler_freq_hz": 8.0e9,
                "Z0": 50.0,
                "cell_length_um": 10.0,
            },
            "coupler_settings": {
                "coupling_dB": -16.0,
                "coupler_freq_hz": 10.0e9,
                "Z0": 55.0,
                "cell_length_um": 12.0,
            },
            "coupler_mode": "ideal",
            "coupler_geometry": {"model": "two_line"},
        }),
        encoding="utf-8",
    )

    settings = build_design_and_passive._coupler_settings(tmp_path)

    assert settings == {
        "coupling_db": -16.0,
        "frequency_hz": 10.0e9,
        "z0_ohm": 55.0,
        "cell_length_um": 12.0,
        "mode": "ideal",
        "model": "two_line",
    }
