from __future__ import annotations

from pathlib import Path

import pytest

from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.topology_artifacts import (
    load_topology_artifact,
    require_topology_counts,
)


def test_actual_harmonia_ethz_jtl_topology_smoke_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    config_path = harmonia_root / "examples" / "configs" / "harmonia_ethz_jtl_topology_smoke.json"

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists() or not config_path.exists():
        pytest.skip("Local Harmonia.jl ETHZ JTL topology smoke setup not available.")

    output_dir = tmp_path / "harmonia_ethz_jtl_topology_smoke"

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
    assert result.status.simulation_type == "harmonia_ethz_jtl_topology_smoke"

    artifact = load_topology_artifact(output_dir)

    assert artifact.status == "PASS"
    assert artifact.simulation_type == "harmonia_ethz_jtl_topology_smoke"
    assert artifact.circuit_template == "circuit_ir_ethz_jtl_chain_topology"
    assert artifact.backend == "Harmonia.CircuitIR + Harmonia.add_ethz_jtl_chain!"
    assert artifact.topology_only is True
    assert artifact.n_ports == 2

    assert artifact.topology["expected_ir_elements"] == 31
    assert artifact.topology["expected_jc_tuples"] == 40
    assert artifact.topology["ir_element_count_match"] is True
    assert artifact.topology["jc_tuple_count_match"] is True

    template_metadata = artifact.topology["template_metadata"]

    assert template_metadata["n_cells"] == 10
    assert template_metadata["junction_count"] == 9
    assert template_metadata["normal_cell_count"] == 6
    assert template_metadata["section_cell_count"] == 2
    assert template_metadata["short_section_count"] == 1
    assert template_metadata["long_section_count"] == 1
    assert template_metadata["distributed_segment_count"] == 5
    assert template_metadata["expected_ir_elements"] == 29
    assert template_metadata["expected_jc_tuples"] == 38

    require_topology_counts(
        artifact,
        n_elements=31,
        element_kind_counts={
            "P": 2,
            "C": 15,
            "Lj": 9,
            "L": 5,
        },
        required_names={
            "P1",
            "P2",
            "ethz_Cg_half_in",
            "ethz_Lj_1",
            "ethz_Cg_section_half_4",
            "ethz_Ll_4_1",
            "ethz_Cl_4_2",
            "ethz_Lj_4",
            "ethz_Cg_section_half_8",
            "ethz_Ll_8_3",
            "ethz_Cl_8_3",
            "ethz_Lj_8",
            "ethz_Cg_half_out",
        },
        required_roles={
            "input_half_ground_capacitance",
            "output_half_ground_capacitance",
            "ground_capacitance",
            "distributed_line_inductance",
            "distributed_line_capacitance",
            "distributed_line_terminal_capacitance",
            "series_josephson_junction_after_distributed_section",
        },
    )