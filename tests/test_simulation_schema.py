from __future__ import annotations

import json

import numpy as np
import pytest

from twpa.io.simulation_schema import (
    SCHEMA_VERSION,
    SimulationSchemaError,
    SimulationStatus,
    assert_json_serializable,
    classify_status,
    compute_two_port_metrics,
    normalize_s_parameter_shape,
    validate_status_payload,
)


def test_validate_status_payload_accepts_pass_with_null_residual() -> None:
    payload = validate_status_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "solver_success": True,
            "failure_reason": None,
            "residual_norm": None,
            "relative_residual_norm": None,
        }
    )

    assert payload["status"] == "PASS"
    assert payload["solver_success"] is True


def test_validate_status_payload_rejects_pass_with_failure_reason() -> None:
    with pytest.raises(SimulationSchemaError, match="failure_reason"):
        validate_status_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "solver_success": True,
                "failure_reason": "bad",
            }
        )


def test_validate_status_payload_rejects_pass_without_solver_success() -> None:
    with pytest.raises(SimulationSchemaError, match="solver_success"):
        validate_status_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "solver_success": False,
                "failure_reason": None,
            }
        )


def test_validate_status_payload_rejects_nonfinite_residual() -> None:
    with pytest.raises(SimulationSchemaError, match="residual_norm"):
        validate_status_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "FAIL",
                "solver_success": False,
                "residual_norm": float("nan"),
            }
        )


def test_assert_json_serializable_catches_set() -> None:
    with pytest.raises(SimulationSchemaError, match="not JSON serializable"):
        assert_json_serializable({"bad": {"this is a set"}})


def test_classify_status() -> None:
    assert classify_status(
        solver_success=True,
        arrays_finite=True,
        residual_norm=1e-9,
        residual_tolerance=1e-6,
    ) == SimulationStatus.PASS

    assert classify_status(
        solver_success=True,
        arrays_finite=True,
        is_fallback=True,
    ) == SimulationStatus.PARTIAL

    assert classify_status(
        solver_success=False,
        arrays_finite=True,
    ) == SimulationStatus.FAIL

    assert classify_status(
        solver_success=True,
        arrays_finite=False,
    ) == SimulationStatus.FAIL


def test_normalize_s_parameter_shape_frequency_first() -> None:
    s = np.zeros((5, 2, 2), dtype=np.complex128)
    out = normalize_s_parameter_shape(s, n_frequency=5, n_ports=2)

    assert out.shape == (5, 2, 2)


def test_normalize_s_parameter_shape_julia_h5py_order() -> None:
    s = np.zeros((2, 2, 5), dtype=np.complex128)
    out = normalize_s_parameter_shape(s, n_frequency=5, n_ports=2)

    assert out.shape == (5, 2, 2)


def test_compute_two_port_metrics_matched_through() -> None:
    frequency_hz = np.linspace(4e9, 8e9, 5)
    s = np.zeros((5, 2, 2), dtype=np.complex128)
    s[:, 1, 0] = 1.0
    s[:, 0, 1] = 1.0
    gain_db = np.zeros(5)

    metrics = compute_two_port_metrics(
        frequency_hz=frequency_hz,
        s_parameters=s,
        gain_db=gain_db,
    )

    assert metrics.frequency_points == 5
    assert metrics.s_shape == (5, 2, 2)
    assert metrics.max_abs_s11 == 0.0
    assert metrics.max_abs_s22 == 0.0
    assert metrics.reciprocal_error_max_abs == 0.0
    assert metrics.passivity_max_singular_value == pytest.approx(1.0)
    assert metrics.all_arrays_finite