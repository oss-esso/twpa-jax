from __future__ import annotations

import numpy as np
import pytest

import twpa_solver.pump.neimark_sacker as ns


def test_hill_vector_reshape_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="size"):
        ns._reshape_hill_vector(np.ones(3), (2, 2))


def test_first_lyapunov_coefficient_uses_supercritical_negative_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ns,
        "d2n_coeffs",
        lambda problem, state, vector: np.zeros_like(state),
    )
    monkeypatch.setattr(
        ns,
        "d3n_coeffs",
        lambda problem, state, vector: -np.ones_like(state),
    )

    coefficient = ns.first_lyapunov_coefficient(
        object(), np.zeros((2, 2), dtype=complex), np.ones((2, 2)), 10.0
    )

    assert coefficient < 0.0
