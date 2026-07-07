from __future__ import annotations

import numpy as np

from twpa_solver_old.model.blocks import DirectionalCouplerBlock
from twpa_solver_old.model.topology import (
    CircuitBuilder,
    coupled_inductor_branch_current,
)


def test_directional_coupler_current_matches_manual_calculation() -> None:
    builder = CircuitBuilder(4)
    block = DirectionalCouplerBlock(
        top_start=0,
        top_end=1,
        bottom_start=2,
        bottom_end=3,
        inductance_top_h=2.0,
        inductance_bottom_h=8.0,
        coupling_k=0.25,
        pump_source_node=3,
    )
    block.apply(builder)
    model = builder.build()
    phi = np.asarray([1.0, 0.5, -0.25, -0.75])
    branch_flux = np.asarray([phi[0] - phi[1], phi[2] - phi[3]])
    branch_current = coupled_inductor_branch_current(branch_flux, 2.0, 8.0, 0.25)
    manual = np.asarray(
        [branch_current[0], -branch_current[0], branch_current[1], -branch_current[1]]
    )
    np.testing.assert_allclose(model.linear_stiffness_h_inv @ phi, manual)
    assert model.pump_nodes == (3,)
    assert model.metadata["coupler_blocks"][0]["model"] == "compact_coupled_inductor"
