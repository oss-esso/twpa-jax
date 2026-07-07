from __future__ import annotations

from twpa_solver_old.model.blocks import COUPLER_MODEL_TAXONOMY
from twpa_solver_old.model.ipm import IPMConfig, build_ipm_topology


def test_coupler_model_taxonomy_contains_required_labels() -> None:
    assert "marker" in COUPLER_MODEL_TAXONOMY
    assert "compact_coupled_inductor" in COUPLER_MODEL_TAXONOMY
    assert "distributed_coupled_cell" in COUPLER_MODEL_TAXONOMY
    assert "old_harmonia_cpw_approx" in COUPLER_MODEL_TAXONOMY


def test_topology_coupler_models_use_taxonomy() -> None:
    for name in (
        "ipm_jtwpa_reduced_marker",
        "ipm_jtwpa_physical_coupler",
        "ipm_jtwpa_old_constants_compact_surrogate",
    ):
        model = build_ipm_topology(name, IPMConfig(cells_per_line=1))
        assert model.metadata["coupler_model"] in COUPLER_MODEL_TAXONOMY
