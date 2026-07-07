import json

import numpy as np

from twpa_solver_old.importers.julia_circuit_json import import_julia_circuit_json


def write_tiny_export(path):
    payload = {
        "schema": "harmonia_old_ipm_circuit_json_v1",
        "circuit": [
            {"name": "P1_0", "kind": "P", "node1": "1", "node2": "0", "raw_value": "1", "value": 1},
            {"name": "P2_0", "kind": "P", "node1": "3", "node2": "0", "raw_value": "2", "value": 2},
            {"name": "R1_0", "kind": "R", "node1": "1", "node2": "0", "raw_value": "50", "value": 50.0},
            {"name": "L1_2", "kind": "L", "node1": "1", "node2": "2", "raw_value": "1e-9", "value": 1e-9},
            {"name": "L3_0", "kind": "L", "node1": "3", "node2": "0", "raw_value": "2e-9", "value": 2e-9},
            {"name": "K1", "kind": "K", "node1": "L1_2", "node2": "L3_0", "raw_value": "0.2", "value": 0.2},
            {"name": "Lj2_3", "kind": "Lj", "node1": "2", "node2": "3", "raw_value": "3e-9", "value": 3e-9},
            {"name": "C2_3", "kind": "C", "node1": "2", "node2": "3", "raw_value": "1e-13", "value": 1e-13},
        ],
        "circuitdefs": {},
        "metadata": {"Nj": 1, "n_final": 3},
        "ports": [],
        "source_convention": {"power_offset_db": 32.0, "pump_port": 4},
        "harmonics": {"Npumpharmonics": [10], "Nmodulationharmonics": [5]},
        "map_axes": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_import_tiny_fixture_node_labels_and_counts(tmp_path):
    imported = import_julia_circuit_json(write_tiny_export(tmp_path / "tiny.json"))
    assert imported.node_labels == ("1", "2", "3")
    assert imported.model.num_nodes == 3
    assert imported.model.metadata["josephson_junction_count"] == 1
    assert imported.model.metadata["mutual_coupling_count"] == 1
    assert [p.name for p in imported.model.ports] == ["P1", "P2"]
    assert np.count_nonzero(imported.model.capacitance_f) > 0
