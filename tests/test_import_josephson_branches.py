import numpy as np

from twpa_solver.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver.model.units import CONSTANTS

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_josephson_branch_import_sets_incidence_and_ic(tmp_path):
    imported = import_julia_circuit_json(write_tiny_export(tmp_path / "tiny.json"))
    assert imported.josephson_branch_names == ("Lj2_3",)
    assert imported.model.josephson_incidence.shape == (3, 1)
    assert imported.model.josephson is not None
    assert np.isclose(imported.model.josephson.critical_current_a[0], CONSTANTS.reduced_phi0 / 3e-9)
