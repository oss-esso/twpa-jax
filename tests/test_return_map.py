"""Focused tests for the scalar return-map diagnostics."""

import math

import numpy as np

from scripts.chaos.return_map import (
    ClassificationThresholds,
    _circle_map,
    describe_section,
    run_validation,
)


def test_validation_gate_passes_all_references() -> None:
    result = run_validation()
    assert result["status"] == "PASS"
    assert all(case["passed"] for case in result["cases"].values())


def test_fixed_point_gates_rotation_and_finds_period_one() -> None:
    thresholds = ClassificationThresholds(1.0, "test")
    result = describe_section(np.zeros((100, 2)), 1.0, thresholds=thresholds)
    assert result["classification"] == "period-1"
    assert result["q_min"] == 1
    assert result["rotation_gate"] == "OFF"
    assert result["rho"] is None


def test_exact_period_five_has_q_min_five() -> None:
    thresholds = ClassificationThresholds(1.0, "test")
    cycle = np.tile(_circle_map(1.0 / 5.0, 5), (30, 1))
    result = describe_section(cycle, 1.0, thresholds=thresholds)
    assert result["q_min"] == 5
    assert result["period_q"] == 5


def test_golden_mean_rotation_is_recovered() -> None:
    thresholds = ClassificationThresholds(1.0, "test")
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    result = describe_section(_circle_map(golden), 1.0, thresholds=thresholds)
    assert result["rho"] is not None
    assert abs(float(result["rho"]) - golden) < 0.01
    assert result["locking_verdict"] == "UNLOCKED"
