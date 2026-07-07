from twpa_solver_old.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_imported_old_ipm_pump_residual_shape_tiny_fixture(tmp_path):
    imported = import_julia_circuit_json(write_tiny_export(tmp_path / "tiny.json"))
    residual = PumpAFTResidual(
        imported.model,
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1, source_current_peak_a=1e-6),
    )
    x0 = residual.initial_guess()
    r = residual(x0)
    assert x0.shape == (2 * imported.model.num_nodes,)
    assert r.shape == x0.shape
