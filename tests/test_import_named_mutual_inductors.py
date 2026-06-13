import numpy as np

from twpa_solver.importers.julia_circuit_json import import_julia_circuit_json

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_named_mutual_inductors_are_assembled(tmp_path):
    imported = import_julia_circuit_json(write_tiny_export(tmp_path / "tiny.json"))
    assert imported.mutual_couplings[0]["branch_1"] == "L1_2"
    assert imported.mutual_couplings[0]["branch_2"] == "L3_0"
    k = imported.model.linear_stiffness_h_inv
    assert np.allclose(k, k.T)
    assert abs(k[0, 2]) > 0.0
