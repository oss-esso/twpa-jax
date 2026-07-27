from __future__ import annotations

import numpy as np

from twpa_solver.multitone.basis import ToneIndex, build_three_tone_basis
from twpa_solver.multitone.grid import TorusGrid


def test_torus_phase_rows_are_normalized_and_flat_ordered() -> None:
    grid = TorusGrid(build_three_tone_basis(10.0, 1.0))
    rows = grid.phase_rows([ToneIndex(0, 0), ToneIndex(1, 0)])

    assert rows.shape == (2, grid.nt)
    np.testing.assert_allclose(rows[0], 1.0 / grid.nt)
    np.testing.assert_allclose(
        rows[1], np.exp(-1j * grid.theta_flat[:, 0]) / grid.nt
    )
