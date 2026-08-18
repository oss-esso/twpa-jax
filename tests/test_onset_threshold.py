from __future__ import annotations

import numpy as np

from scripts.chaos.onset_threshold import fit_squared_radius_threshold


def test_squared_radius_fit_recovers_supercritical_threshold() -> None:
    control = np.arange(0.0, 0.8, 0.1)
    radius = np.sqrt(np.maximum(control - 0.2, 0.0))
    result = fit_squared_radius_threshold(
        control, radius, period1_mask=control <= 0.2
    )

    assert result.status == "RESOLVED_CONTINUOUS"
    assert result.mu_c is not None
    np.testing.assert_allclose(result.mu_c, 0.2, atol=1e-12)


def test_squared_radius_fit_rejects_a_hard_finite_jump() -> None:
    control = np.arange(0.0, 0.8, 0.1)
    radius = np.where(control < 0.2, 0.0, 0.5)
    result = fit_squared_radius_threshold(
        control, radius, period1_mask=control < 0.2
    )

    assert result.status == "UNRESOLVED"
    assert result.mu_c is None
