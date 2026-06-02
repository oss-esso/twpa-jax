from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from twpa.io.julia_bridge import load_julia_simulation
from twpa.io.julia_runner import run_harmonia_simulation


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if hasattr(value, "decode"):
        return value.decode("utf-8")

    # h5py may return NumPy scalar bytes, e.g. np.bytes_(b"...")
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)

    return str(value)


def test_actual_harmonia_jtl_topology_smoke_if_available(tmp_path: Path) -> None:
    harmonia_root = Path(r"D:\Projects\Thesis\Harmonia.jl")
    config_path = harmonia_root / "examples" / "configs" / "harmonia_jtl_topology_smoke.json"

    if not (harmonia_root / "scripts" / "run_simulation.jl").exists() or not config_path.exists():
        pytest.skip("Local Harmonia.jl JTL topology smoke setup not available.")

    output_dir = tmp_path / "harmonia_jtl_topology_smoke"

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
    assert result.status.simulation_type == "harmonia_jtl_topology_smoke"

    data = load_julia_simulation(output_dir)

    assert data.status.status == "PASS"
    assert data.status.simulation_type == "harmonia_jtl_topology_smoke"
    assert data.status.circuit_template == "circuit_ir_jtl_chain_topology"

    h5_path = output_dir / "simulation.h5"
    assert h5_path.exists()

    with h5py.File(h5_path, "r") as h5:
        assert _decode_h5_string(h5.attrs["simulation_type"]) == "harmonia_jtl_topology_smoke"
        assert _decode_h5_string(h5.attrs["backend"]) == "Harmonia.CircuitIR + Harmonia.add_jtl_chain!"
        assert bool(h5.attrs["topology_only"]) is True
        assert int(h5.attrs["n_ports"]) == 2

        topology = json.loads(_decode_h5_string(h5["topology"]["topology_json"][()]))
        summary = json.loads(_decode_h5_string(h5["topology"]["summary_json"][()]))
        circuit = json.loads(_decode_h5_string(h5["topology"]["circuit_json"][()]))

    assert topology["expected_ir_elements"] == 10
    assert topology["expected_jc_tuples"] == 14
    assert topology["ir_element_count_match"] is True
    assert topology["jc_tuple_count_match"] is True

    assert summary["n_elements"] == 10
    assert summary["element_kind_counts"]["P"] == 2
    assert summary["element_kind_counts"]["C"] == 4
    assert summary["element_kind_counts"]["Lj"] == 4

    assert len(circuit) == 10

    names = {row["name"] for row in circuit}
    assert "P1" in names
    assert "P2" in names
    assert "jtl_Cg_1" in names
    assert "jtl_Lj_1" in names