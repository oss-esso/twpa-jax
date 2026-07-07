import numpy as np

from twpa_solver_old.importers.julia_circuit_json import import_julia_circuit_json
from twpa_solver_old.residuals.aft_hb import PumpAFTConfig, PumpAFTResidual

from test_import_old_ipm_json_tiny_fixture import write_tiny_export


def test_imported_old_ipm_pump_source_uses_port4_when_present(tmp_path):
    path = write_tiny_export(tmp_path / "tiny.json")
    # Append a P4 pump port to the fixture.
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["circuit"].append(
        {"name": "P4_0", "kind": "P", "node1": "2", "node2": "0", "raw_value": "4", "value": 4}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    imported = import_julia_circuit_json(path)
    residual = PumpAFTResidual(
        imported.model,
        PumpAFTConfig(pump_frequency_hz=6e9, harmonics=1, source_current_peak_a=2e-6),
    )
    source = residual.source_time()
    assert imported.model.pump_nodes == (imported.node_index["2"],)
    nonzero_nodes = np.flatnonzero(np.max(np.abs(source), axis=0) > 0)
    assert list(nonzero_nodes) == [imported.node_index["2"]]
