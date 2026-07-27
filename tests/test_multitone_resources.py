from __future__ import annotations

import pytest

from twpa_solver.multitone.basis import build_three_tone_basis
from twpa_solver.multitone.resources import (
    ResourceLimitExceeded,
    estimate,
    guard,
)


class _Grid:
    n = 20


def test_resource_guard_rejects_before_allocation() -> None:
    resource = estimate(build_three_tone_basis(10.0, 1.0), _Grid(), 100, 4, "real_coupled_fast")

    with pytest.raises(ResourceLimitExceeded):
        guard(resource, 1e-12)


def test_resource_estimate_reports_dimensions() -> None:
    resource = estimate(build_three_tone_basis(10.0, 1.0), _Grid(), 5, 2, "floquet_sector")

    assert resource.n_tones == 3
    assert resource.n_torus == 36
    assert resource.matrix_dimension == 2 * 2 * 5
    assert resource.total_bytes > resource.coefficient_state_bytes
