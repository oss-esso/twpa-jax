from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.topology_artifacts import load_topology_artifact


def test_actual_harmonia_lumped_jpa_linear_jc_smoke_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    config_path = harmonia_root / "examples" / "configs" / "harmonia_lumped_jpa_linear_jc_smoke.json"

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists() or not config_path.exists():
        pytest.skip("Local Harmonia.jl lumped JPA linear JC smoke setup not available.")

    output_dir = tmp_path / "harmonia_lumped_jpa_linear_jc_smoke"

    result = run_harmonia_simulation(
        config_path=config_path,
        output_dir=output_dir,
        harmonia_jl_root=harmonia_root,
        force=True,
        timeout_s=180.0,
    )

    assert result.returncode == 0
    assert result.ok
    assert result.status is not None
    assert result.status.status == "PASS"
    assert result.status.simulation_type == "harmonia_lumped_jpa_linear_jc_smoke"

    data = load_julia_simulation(output_dir)

    assert data.status.status == "PASS"
    assert data.status.simulation_type == "harmonia_lumped_jpa_linear_jc_smoke"
    assert data.status.circuit_template == "circuit_ir_lumped_jpa_reflection_linear_jc"

    assert data.frequency_hz is not None
    assert data.s_parameters is not None
    assert data.gain_db is not None

    assert data.frequency_hz.shape == (11,)
    assert data.s_parameters.shape == (11, 1, 1)
    assert data.gain_db.shape == (11,)

    assert np.all(np.isfinite(data.frequency_hz))
    assert np.all(np.isfinite(data.s_parameters.real))
    assert np.all(np.isfinite(data.s_parameters.imag))
    assert np.all(np.isfinite(data.gain_db))
    assert np.max(np.abs(data.s_parameters)) > 0.0

    artifact = load_topology_artifact(output_dir)

    assert artifact.backend == "Harmonia.CircuitIR + JosephsonCircuits.hbsolve"
    assert artifact.topology_only is False
    assert artifact.n_ports == 1

    assert artifact.topology["expected_ir_elements"] == 4
    assert artifact.topology["expected_jc_tuples"] == 5
    assert artifact.topology["ir_element_count_match"] is True
    assert artifact.topology["jc_tuple_count_match"] is True

    names = set(artifact.topology["solver_export_names"])

    assert "P1" in names
    assert "R_P1" in names
    assert "C_jpa_Cc" in names
    assert "Lj_jpa_Lj" in names
    assert "C_jpa_Lj" in names