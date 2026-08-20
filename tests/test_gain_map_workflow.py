from __future__ import annotations

from pathlib import Path

from workflows import run_gain_map_and_plots


def test_gain_map_output_dir_alias_creates_one_run_dir_per_design(
    tmp_path: Path, monkeypatch,
) -> None:
    design_a = tmp_path / "a"
    design_b = tmp_path / "b"
    design_a.mkdir()
    design_b.mkdir()
    output_root = tmp_path / "runs"
    jobs: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        run_gain_map_and_plots,
        "_fast_parallel_defaults",
        lambda run_args, _design: list(run_args),
    )

    def fake_run_one(
        design: Path, run_dir: Path, _args, _base_run_args, _slow,
    ) -> int:
        jobs.append((design, run_dir))
        return 0

    monkeypatch.setattr(run_gain_map_and_plots, "_run_one", fake_run_one)

    result = run_gain_map_and_plots.main([
        "--fast",
        "--design-dir", str(design_a), str(design_b),
        "--output-dir", str(output_root),
    ])

    assert result == 0
    assert jobs == [
        (design_a, output_root / "a"),
        (design_b, output_root / "b"),
    ]
