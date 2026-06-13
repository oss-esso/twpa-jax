from twpa_solver.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual
from twpa_solver.residuals.conversion import build_conversion_sparameters

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_imported_old_ipm_conversion_minimal_tiny_fixture(tmp_path):
    imported = import_julia_circuit_json(write_tiny_export(tmp_path / "tiny.json"))
    residual = PumpAFTResidual(
        imported.model,
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1, source_current_peak_a=0.0),
    )
    result = build_conversion_sparameters(
        imported.model,
        residual,
        residual.initial_guess(),
        6e9,
        0,
        pump_success=True,
        pump_status="VALID_CONVERGED",
    )
    assert result.s_conversion.shape == (2, 2)
    assert result.signal_gain_db == result.signal_gain_db
