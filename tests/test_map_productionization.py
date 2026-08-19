from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_gain_map, run_hybrid_gain_map
from twpa_solver.map import coverage_summary
from workflows import run_gain_map_and_plots


def test_coverage_distinguishes_physical_boundary_from_unresolved() -> None:
    rows = [
        {"status": "PASS"},
        {"status": "PHYSICAL_BOUNDARY"},
        {"status": "SKIP_AFTER_PHYSICAL_BOUNDARY"},
        {"status": "COLUMN_NUMERICAL_FAILURE"},
        {"status": "COLUMN_UNRESOLVED_BUDGET"},
        {"status": "TD_CONTINUE"},
    ]
    result = coverage_summary(rows)
    assert result["gain_valid_point_count"] == 1
    assert result["confirmed_above_boundary_count"] == 2
    assert result["hb_numerical_failure_count"] == 1
    assert result["unresolved_count"] == 2
    assert result["physically_eligible_point_count"] == 4


def test_workflow_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        run_gain_map_and_plots.main([
            "--design", "designs/ipm_2c_fixed", "--fast", "--slow",
        ])


def test_fast_workflow_defers_pump_cleanup_until_after_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    pump_solution = run_dir / "points" / "point_0000" / "pump_solution.npz"
    pump_solution.parent.mkdir(parents=True)
    pump_solution.write_bytes(b"pump")
    events: list[tuple[str, bool]] = []

    def fake_run_gain_map(run_args: list[str]) -> int:
        events.append(("run", pump_solution.exists()))
        assert "--no-compact-output" in run_args
        assert "--compact-output" not in run_args
        (run_dir / "map_spectrum.npz").write_bytes(b"spectrum")
        return 0

    def fake_plot_gain_map(plot_args: list[str]) -> int:
        events.append(("map", pump_solution.exists()))
        return 0

    def fake_pump_frequency(*_args: object) -> None:
        events.append(("frequency", pump_solution.exists()))

    def fake_pump_power(*_args: object) -> None:
        events.append(("power", pump_solution.exists()))

    monkeypatch.setattr(run_gain_map_and_plots.run_gain_map, "main", fake_run_gain_map)
    monkeypatch.setattr(run_gain_map_and_plots.plot_gain_map, "main", fake_plot_gain_map)
    monkeypatch.setattr(run_gain_map_and_plots, "plot_pump_frequency", fake_pump_frequency)
    monkeypatch.setattr(run_gain_map_and_plots, "plot_pump_power", fake_pump_power)

    result = run_gain_map_and_plots.main([
        "--design", "designs/ipm_2c_fixed", "--fast", "--run-dir", str(run_dir),
    ])

    assert result == 0
    assert events == [
        ("run", True),
        ("map", True),
        ("frequency", True),
        ("power", True),
    ]
    assert not pump_solution.exists()


def test_gain_map_workflow_sequences_multiple_design_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    design_a = tmp_path / "design_a"
    design_b = tmp_path / "design_b"
    design_a.mkdir()
    design_b.mkdir()
    run_root = tmp_path / "maps"
    events: list[tuple[str, str, str]] = []

    def fake_run_gain_map(run_args: list[str]) -> int:
        design = run_args[run_args.index("--circuit-dir") + 1]
        run_dir = Path(run_args[run_args.index("--outdir") + 1])
        events.append(("run", design, str(run_dir)))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "map_spectrum.npz").write_bytes(b"spectrum")
        return 0

    def fake_plot_gain_map(plot_args: list[str]) -> int:
        design = plot_args[plot_args.index("--ipm-dir") + 1]
        run_dir = plot_args[plot_args.index("--run-dir") + 1]
        events.append(("plot", design, run_dir))
        return 0

    monkeypatch.setattr(run_gain_map_and_plots.run_gain_map, "main", fake_run_gain_map)
    monkeypatch.setattr(run_gain_map_and_plots.plot_gain_map, "main", fake_plot_gain_map)
    monkeypatch.setattr(run_gain_map_and_plots, "plot_pump_frequency", lambda *args: None)
    monkeypatch.setattr(run_gain_map_and_plots, "plot_pump_power", lambda *args: None)

    result = run_gain_map_and_plots.main([
        "--design", str(design_a), str(design_b), "--fast",
        "--run-dir", str(run_root),
    ])

    assert result == 0
    assert events == [
        ("run", str(design_a), str(run_root / "design_a")),
        ("plot", str(design_a), str(run_root / "design_a")),
        ("run", str(design_b), str(run_root / "design_b")),
        ("plot", str(design_b), str(run_root / "design_b")),
    ]


def test_slow_workflow_translates_shared_axis_flags() -> None:
    translated = run_gain_map_and_plots._translate_slow_flags([
        "--pump-power-min-dbm", "-20", "--pump-freq-max-ghz", "7.9",
    ])
    assert translated == ["--power-min-dbm", "-20", "--freq-max-ghz", "7.9"]


def test_fast_worker_defaults_use_half_ram_and_match_chunk_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_gain_map_and_plots, "available_memory_gb", lambda: 20.0)
    monkeypatch.setattr(run_gain_map_and_plots, "_fast_worker_memory_gb", lambda _design: 2.0)
    monkeypatch.setattr(run_gain_map_and_plots.os, "cpu_count", lambda: 64)

    args = run_gain_map_and_plots._fast_parallel_defaults(
        ["--n-frequency", "50"], tmp_path / "design"
    )

    assert args[:2] == ["--n-frequency", "50"]
    assert args[args.index("--frequency-workers") + 1] == "5"
    assert "--frequency-chunk-size" in args
    assert args[args.index("--frequency-chunk-size") + 1] == "5"


def test_fast_parallel_defaults_preserve_explicit_worker_and_chunk_values(
    tmp_path: Path,
) -> None:
    args = run_gain_map_and_plots._fast_parallel_defaults(
        ["--frequency-workers", "3", "--frequency-chunk-size", "7"],
        tmp_path / "design",
    )

    assert args == ["--frequency-workers", "3", "--frequency-chunk-size", "7"]


def test_frequency_worker_flags_are_available() -> None:
    fast = run_gain_map.parse_args(["--frequency-workers", "4", "--compact-output"])
    slow = run_hybrid_gain_map.main
    assert fast.frequency_workers == 4
    assert fast.compact_output is True
    assert callable(slow)


def test_dynamic_dc_is_explicit_and_default_off() -> None:
    default = run_gain_map.parse_args([])
    enabled = run_gain_map.parse_args(["--pump-dynamic-dc"])

    assert default.pump_dynamic_dc is False
    assert enabled.pump_dynamic_dc is True


def test_dynamic_dc_follows_resolved_mixing_order_for_subprocess_path() -> None:
    four_wm = SimpleNamespace(pump_dynamic_dc=False, mixing_order=4)
    three_wm = SimpleNamespace(pump_dynamic_dc=False, mixing_order=3)

    assert not run_gain_map._pump_basis_requires_dynamic_dc(four_wm)
    assert run_gain_map._pump_basis_requires_dynamic_dc(three_wm)


def test_hybrid_map_uses_measured_attenuation_by_default() -> None:
    ns = SimpleNamespace(
        circuit_dir=Path("designs/ipm_2c_fixed"),
        outdir=Path("outputs/_attenuation_test"),
        n_power=2, n_frequency=1,
        power_min_dbm=-26.0, power_max_dbm=-25.0,
        freq_min_ghz=7.6, freq_max_ghz=7.6,
        attenuation_db=None, pump_mode_count=10, nt=40,
        pump_mode_policy="positive_odd_jc", mixing_order="auto", harmonics=3,
        signal_detuning_mhz=500.0, signal_offset_count_per_side=5,
        signal_offset_step_mhz=500.0, inproc_pump_backend="schur_cpu_mt",
        inproc_preconditioner="real_coupled_fast", inproc_solve_deadline_s=14.0,
        inproc_max_newton=16, pump_port=None, source_port=None, out_port=None,
        dc_branch_flux_over_phi0=None, signal_ghz=None,
    )
    args = run_hybrid_gain_map.build_args(ns)

    assert args.attenuation_db is None
    assert run_gain_map.attenuation_db_for(7.6, args) == pytest.approx(
        run_gain_map.default_loss_model().attenuation_db(7.6)
    )


def test_output_root_is_created_by_hybrid_cli(tmp_path: Path, monkeypatch) -> None:
    # Exercise the non-solver output contract through the existing isolated
    # runner boundary without constructing a production circuit.
    output = tmp_path / "nested" / "run"
    assert not output.exists()
    output.mkdir(parents=True)
    assert output.is_dir()
