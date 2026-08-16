from __future__ import annotations

from scripts.loss_sensitivity_floquet import TAN_DELTAS, variant_name


def test_phase3_loss_grid_is_reproducible() -> None:
    assert TAN_DELTAS == (0.0, 1e-5, 1e-4, 1e-3)
    assert variant_name(0.0) == "2c_base"
    assert variant_name(1e-4) == "2c_td1e-04"
