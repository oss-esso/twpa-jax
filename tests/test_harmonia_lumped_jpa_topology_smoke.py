from __future__ import annotations

from pathlib import Path

import pytest

from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.topology_artifacts import (
    load_topology_artifact,
    require_topology_counts,
)


def test_actual_harmonia_lumped_jpa_topology_smoke_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    config_path = harmonia_root / "examples" / "configs" / "harmonia_lumped_jpa_topology_smoke.json"

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists() or not config_path.exists():
        pytest.skip("Local Harmonia.jl lumped JPA topology smoke setup not available.")

    output_dir = tmp_path / "harmonia_lumped_jpa_topology_smoke"

    result = run_harmonia_simulation(
        config_path=config_path,
        output_dir=output_dir,
        harmonia_jl_root=harmonia_root,
        force=True,
        timeout_s=120.0,
    )

    assert result.returncode == 0
    assert result.ok
    assert result.status is not None
    assert result.status.status == "PASS"
    assert result.status.simulation_type == "harmonia_lumped_jpa_topology_smoke"

    artifact = load_topology_artifact(output_dir)

    assert artifact.status == "PASS"
    assert artifact.simulation_type == "harmonia_lumped_jpa_topology_smoke"
    assert artifact.circuit_template == "circuit_ir_lumped_jpa_reflection_topology"
    assert artifact.backend == "Harmonia.CircuitIR + Harmonia.add_lumped_jpa_reflection!"
    assert artifact.topology_only is True
    assert artifact.n_ports == 1

    assert artifact.topology["expected_ir_elements"] == 4
    assert artifact.topology["expected_jc_tuples"] == 5
    assert artifact.topology["ir_element_count_match"] is True
    assert artifact.topology["jc_tuple_count_match"] is True

    template_metadata = artifact.topology["template_metadata"]

    assert template_metadata["template"] == "harmonia_lumped_jpa_topology_smoke"
    assert template_metadata["old_source_file"] == "Harmonia/JPA_standard.jl"
    assert template_metadata["port_node"] == "n1"
    assert template_metadata["resonator_node"] == "n2"
    assert template_metadata["expected_ir_elements"] == 2
    assert template_metadata["expected_jc_tuples"] == 3

    require_topology_counts(
        artifact,
        n_elements=4,
        element_kind_counts={
            "P": 1,
            "R": 1,
            "C": 1,
            "Lj": 1,
        },
        required_names={
            "P1",
            "R_P1",
            "jpa_Cc",
            "jpa_Lj",
        },
        required_roles={
            "reflection_port",
            "port_impedance",
            "input_coupling_capacitance",
            "shunt_josephson_inductance",
        },
    )

    solver_names = set(artifact.topology["solver_export_names"])

    assert "P1" in solver_names
    assert "R_P1" in solver_names
    assert "C_jpa_Cc" in solver_names
    assert "Lj_jpa_Lj" in solver_names
    assert "C_jpa_Lj" in solver_names