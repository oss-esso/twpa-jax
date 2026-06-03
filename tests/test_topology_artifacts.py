from __future__ import annotations

from pathlib import Path

import pytest

from twpa.io.julia_runner import run_harmonia_simulation
from twpa.io.topology_artifacts import (
    load_topology_artifact,
    require_topology_counts,
)


def test_load_actual_jtl_topology_artifact_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    config_path = harmonia_root / "examples" / "configs" / "harmonia_jtl_topology_smoke.json"

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists() or not config_path.exists():
        pytest.skip("Local Harmonia.jl JTL topology smoke is not available.")

    output_dir = tmp_path / "jtl_topology"

    result = run_harmonia_simulation(
        config_path=config_path,
        output_dir=output_dir,
        harmonia_jl_root=harmonia_root,
        force=True,
        timeout_s=120.0,
    )

    assert result.ok

    artifact = load_topology_artifact(output_dir)

    assert artifact.status == "PASS"
    assert artifact.simulation_type == "harmonia_jtl_topology_smoke"
    assert artifact.topology_only is True
    assert artifact.n_ports == 2

    require_topology_counts(
        artifact,
        n_elements=10,
        element_kind_counts={"P": 2, "C": 4, "Lj": 4},
        required_names={"P1", "P2", "jtl_Cg_1", "jtl_Lj_1"},
        required_roles={"ground_capacitance", "josephson_series_element"},
    )