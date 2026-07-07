from pathlib import Path

import pytest

from twpa_solver_old.importers.julia_circuit_json import import_julia_circuit_json


def test_full_old_ipm_export_import_assembly():
    path = Path(r"D:\Projects\Thesis\outputs\new_twpa_solver\old_ipm_export\old_ipm_circuit.json")
    if not path.exists():
        pytest.skip("full old-IPM export artifact has not been generated")
    imported = import_julia_circuit_json(path)
    assert imported.raw["metadata"]["Nj"] == 2508
    assert imported.model.num_nodes == 3134
    assert imported.model.metadata["element_count"] == 8788
    assert imported.model.metadata["josephson_junction_count"] == 2507
    assert imported.model.metadata["mutual_coupling_count"] == 4
    assert [p.name for p in imported.model.ports] == ["P1", "P2", "P3", "P4"]
