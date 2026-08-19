from __future__ import annotations

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
