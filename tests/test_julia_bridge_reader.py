from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from twpa.io.julia_bridge import load_julia_simulation, read_status_json


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj) + "\n", encoding="utf-8")


def test_read_status_json_pass_schema_smoke(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    _write_json(
        status_path,
        {
            "schema_version": "0.1.0",
            "run_id": "abc",
            "status": "PASS",
            "simulation_type": "schema_smoke",
            "circuit_template": "matched_through_2port",
            "solver_success": True,
            "residual_norm": None,
            "relative_residual_norm": None,
            "failure_reason": None,
            "runtime_s": 1.0,
            "random_seed": 1234,
            "julia_version": "1.12.6",
            "josephsoncircuits_version": "NOT_USED_IN_SCHEMA_SMOKE",
            "harmonia_commit": "UNKNOWN",
            "python_commit": "UNKNOWN",
            "h5_path": None,
            "output_dir": str(tmp_path),
        },
    )

    status = read_status_json(status_path)

    assert status.status == "PASS"
    assert status.solver_success is True
    assert status.simulation_type == "schema_smoke"
    assert status.residual_norm is None


def test_pass_status_rejects_nonfinite_residual(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    _write_json(
        status_path,
        {
            "schema_version": "0.1.0",
            "run_id": "abc",
            "status": "PASS",
            "simulation_type": "bad",
            "circuit_template": "bad",
            "solver_success": True,
            "residual_norm": float("nan"),
            "relative_residual_norm": None,
            "failure_reason": None,
        },
    )

    with pytest.raises(ValueError, match="residual_norm"):
        read_status_json(status_path)


def test_pass_status_rejects_failure_reason(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    _write_json(
        status_path,
        {
            "schema_version": "0.1.0",
            "run_id": "abc",
            "status": "PASS",
            "simulation_type": "bad",
            "circuit_template": "bad",
            "solver_success": True,
            "residual_norm": None,
            "relative_residual_norm": None,
            "failure_reason": "should not be here",
        },
    )

    with pytest.raises(ValueError, match="failure_reason"):
        read_status_json(status_path)


def test_load_actual_schema_smoke_if_available() -> None:
    run_dir = Path(r"D:\Projects\Thesis\outputs\julia_engine_smoke\schema_smoke_001")

    if not run_dir.exists():
        pytest.skip("Local Julia schema-smoke run directory not found.")

    data = load_julia_simulation(run_dir)

    assert data.status.status == "PASS"
    assert data.status.simulation_type == "schema_smoke"
    assert data.frequency_hz is not None
    assert data.s_parameters is not None
    assert data.gain_db is not None

    assert data.frequency_hz.shape == (5,)
    assert data.s_parameters.shape == (5, 2, 2)
    assert data.gain_db.shape == (5,)

    np.testing.assert_allclose(data.s_parameters[:, 0, 0], 0.0)
    np.testing.assert_allclose(data.s_parameters[:, 1, 1], 0.0)
    np.testing.assert_allclose(data.s_parameters[:, 1, 0], 1.0)
    np.testing.assert_allclose(data.s_parameters[:, 0, 1], 1.0)
    np.testing.assert_allclose(data.gain_db, 0.0)