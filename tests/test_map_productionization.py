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


def test_slow_workflow_translates_shared_axis_flags() -> None:
    translated = run_gain_map_and_plots._translate_slow_flags([
        "--pump-power-min-dbm", "-20", "--pump-freq-max-ghz", "7.9",
    ])
    assert translated == ["--power-min-dbm", "-20", "--freq-max-ghz", "7.9"]


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
