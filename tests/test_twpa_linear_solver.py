from __future__ import annotations

import numpy as np

from twpa_solver.model.ipm import IPMConfig, build_ipm_jtwpa
from twpa_solver.residuals.linear import solve_linear_sparameters


def test_passive_linearized_ladder_returns_finite_reciprocal_sparameters() -> None:
    model = build_ipm_jtwpa(IPMConfig(cells_per_line=2))
    result = solve_linear_sparameters(model, 6e9)
    assert result.s.shape == (2, 2)
    assert np.all(np.isfinite(result.s))
    np.testing.assert_allclose(result.s[1, 0], result.s[0, 1], rtol=1e-8, atol=1e-8)
