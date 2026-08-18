from __future__ import annotations

import numpy as np
import pytest

from twpa_solver.signal.branch_tracking import (
    order_multiplier_sets,
    stability_verdict,
    track_floquet_branch,
)
from twpa_solver.signal.stability import ComplexResonance, FloquetClassification


def _classification(magnitude: float, kind: str) -> FloquetClassification:
    return FloquetClassification(
        multiplier=magnitude * np.exp(0.4j),
        phase_rad=0.4,
        magnitude=magnitude,
        zone_frequency_ghz=1.0,
        kind=kind,
        near_unit_circle=True,
    )


def test_multiplier_ordering_preserves_near_crossing_identity() -> None:
    previous = np.array([0.95 + 0.05j, 0.25 + 0.80j])
    current = np.array([0.24 + 0.79j, 0.96 + 0.04j])

    ordered = order_multiplier_sets(previous, current)

    np.testing.assert_allclose(ordered, [current[1], current[0]])


def test_stability_verdict_does_not_call_unconverged_stable() -> None:
    assert stability_verdict(
        _classification(0.9, "NEIMARK_SACKER_CANDIDATE"), True
    ) == "STABLE"
    assert stability_verdict(
        _classification(1.1, "NEIMARK_SACKER_CANDIDATE"), True
    ) == "UNSTABLE_NS"
    assert stability_verdict(
        _classification(1.1, "FOLD_CANDIDATE"), True
    ) == "UNSTABLE_FOLD"
    assert stability_verdict(
        _classification(0.9, "NEIMARK_SACKER_CANDIDATE"), False
    ) == "UNDECIDED"


def test_ordering_rejects_a_shrinking_multiplier_set() -> None:
    with pytest.raises(ValueError, match="smaller"):
        order_multiplier_sets([1.0 + 0.0j, 0.5 + 0.0j], [1.0 + 0.0j])


def test_tracker_forwards_mode_vector_and_flags_low_overlap(monkeypatch) -> None:
    import twpa_solver.signal.branch_tracking as tracking

    vectors = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
    ]
    received: list[np.ndarray | None] = []

    def fake_refine(**kwargs: object) -> ComplexResonance:
        index = len(received)
        received.append(kwargs["v0"])
        signal = 1.0 + 0.01 * index
        omega = 2.0 * np.pi * signal * 1e9
        return ComplexResonance(
            omega=omega,
            signal_ghz=signal,
            eig_min=0.0j,
            growth_rate_per_s=0.0,
            converged=True,
            iterations=2,
            residual=1.0e-12,
            mode_vector=vectors[index],
        )

    monkeypatch.setattr(tracking, "refine_complex_resonance", fake_refine)
    branch = track_floquet_branch(
        circuit=None,
        khat_by_parameter=[{}, {}, {}],
        parameters=[0.0, 1.0, 2.0],
        omega_p=2.0 * np.pi * 1.0e9,
        ms=[0],
        seed_signal_ghz=1.0,
        loss_model="current_complex_c",
    )

    assert received[0] is None
    np.testing.assert_array_equal(received[1], vectors[0])
    np.testing.assert_array_equal(received[2], vectors[1])
    assert branch.points[0].mode_overlap is None
    assert branch.points[1].mode_overlap == pytest.approx(1.0)
    assert branch.points[2].mode_overlap == pytest.approx(0.0)
    assert branch.points[2].discontinuity
