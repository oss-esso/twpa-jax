"""Import external circuit netlists into twpa_solver models."""

from twpa_solver_old.importers.julia_circuit_json import (
    ImportedJuliaCircuit,
    import_julia_circuit_json,
)

__all__ = ["ImportedJuliaCircuit", "import_julia_circuit_json"]
