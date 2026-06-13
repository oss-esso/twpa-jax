import numpy as np

from twpa_solver.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_imported_old_ipm_zero_source_zero_flux_residual(tmp_path):
    imported = import_julia_circuit_json(write_tiny_export(tmp_path / "tiny.json"))
    residual = PumpAFTResidual(
        imported.model,
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1, source_current_peak_a=0.0),
    )
    assert np.allclose(residual(residual.initial_guess()), 0.0)
