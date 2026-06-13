from __future__ import annotations

import numpy as np

from twpa_solver.model.graph import Branch, branch_fluxes, incidence_matrix


def test_incidence_matrix_branch_flux_consistency() -> None:
    d = incidence_matrix(3, [Branch(0, 1), Branch(1, 2)])
    phi = np.asarray([1.0, 0.25, -0.5])
    np.testing.assert_allclose(branch_fluxes(phi, d), [0.75, 0.75])
