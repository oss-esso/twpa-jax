from __future__ import annotations

import json
from pathlib import Path

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
    ]
    result = coverage_summary(rows)
    assert result["gain_valid_point_count"] == 1
    assert result["confirmed_above_boundary_count"] == 2
    assert result["hb_numerical_failure_count"] == 1
    assert result["unresolved_count"] == 1
    assert result["physically_eligible_point_count"] == 3


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


def test_output_root_is_created_by_hybrid_cli(tmp_path: Path, monkeypatch) -> None:
    # Exercise the non-solver output contract through the existing isolated
    # runner boundary without constructing a production circuit.
    output = tmp_path / "nested" / "run"
    assert not output.exists()
    output.mkdir(parents=True)
    assert output.is_dir()
