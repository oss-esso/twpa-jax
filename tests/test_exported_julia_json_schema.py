import json


def test_exported_julia_json_schema_tiny_fixture(tmp_path):
    path = tmp_path / "tiny.json"
    payload = {
        "schema": "harmonia_old_ipm_circuit_json_v1",
        "circuit": [{"name": "P1_0", "kind": "P", "node1": "1", "node2": "0", "value": 1}],
        "circuitdefs": {},
        "metadata": {"Nj": 1},
        "ports": [{"name": "P1_0", "node": "1", "ground": "0", "index": 1}],
        "source_convention": {"input_port": 1, "output_port": 2, "pump_port": 4, "power_offset_db": 32.0},
        "harmonics": {"Npumpharmonics": [10], "Nmodulationharmonics": [5]},
        "map_axes": {"pump_frequency_ghz_min": 6.0, "pump_frequency_ghz_max": 8.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == "harmonia_old_ipm_circuit_json_v1"
    assert loaded["source_convention"]["pump_port"] == 4
    assert loaded["harmonics"]["Npumpharmonics"] == [10]
