from __future__ import annotations

from twpa_solver.model.ipm import (
    OLD_JULIA_TARGET_JUNCTIONS,
    IPMConfig,
    build_ipm_topology,
)


def test_old_constants_surrogate_topology_records_warning_metadata() -> None:
    model = build_ipm_topology("ipm_jtwpa_old_constants_compact_surrogate", IPMConfig(cells_per_line=4))
    assert model.metadata["old_julia_parity_mode"] is False
    assert model.metadata["surrogate_topology"] is True
    assert model.metadata["geometry_profile"] == "old_constants_compact_surrogate"
    assert model.metadata["historical_target_cells_or_junctions"] == OLD_JULIA_TARGET_JUNCTIONS
    assert model.metadata["old_julia_port_convention"]["pump_port_equivalent"] == 4


def test_deprecated_old_julia_parity_name_is_marked_as_surrogate() -> None:
    model = build_ipm_topology("ipm_jtwpa_old_julia_parity", IPMConfig(cells_per_line=4))
    assert model.metadata["requested_deprecated_topology"] == "ipm_jtwpa_old_julia_parity"
    assert "not old-Julia circuit parity" in model.metadata["warning"]
