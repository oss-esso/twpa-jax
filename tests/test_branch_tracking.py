from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.chaos.track_critical_root import (
    _crossing,
    _write_branch_json,
    parse_float_list,
    parse_signal_seed,
)
from twpa_solver.signal.branch_tracking import (
    FloquetBranchPoint,
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


def test_critical_root_json_uses_drive_dbm_gate_parameter(tmp_path: Path) -> None:
    resonance = ComplexResonance(
        omega=2.0 * np.pi * (0.72 - 1.0e-4j) * 1.0e9,
        signal_ghz=0.72 - 1.0e-4j,
        eig_min=0.0j,
        growth_rate_per_s=1.0e6,
        converged=True,
        iterations=4,
        residual=1.0e-12,
        mode_vector=np.array([1.0 + 0.0j]),
    )
    point = FloquetBranchPoint(
        parameter=-24.05,
        resonance=resonance,
        classification=_classification(1.01, "UNSTABLE_NS"),
        branch_index=0,
        discontinuity=False,
        stability_verdict="UNSTABLE_NS",
        mode_overlap=None,
    )
    path = tmp_path / "hb_floquet_branch.json"

    _write_branch_json(path, [point], 0.7228542 + 0.0j, 0.25, 0.8)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["points"][0]["parameter"] == pytest.approx(-24.05)
    assert payload["points"][0]["stability_verdict"] == "UNSTABLE_NS"


def test_critical_root_accepts_complex_initial_signal_seed() -> None:
    seed = parse_signal_seed("0.7370030224-0.0038445933j")

    assert seed.real == pytest.approx(0.7370030224)
    assert seed.imag == pytest.approx(-0.0038445933)


def test_critical_root_crossing_reports_real_generator_frequency() -> None:
    left = ComplexResonance(
        omega=0.0j,
        signal_ghz=0.72 - 1.0e-3j,
        eig_min=0.0j,
        growth_rate_per_s=-1.0,
        converged=True,
        iterations=1,
        residual=0.0,
    )
    right = ComplexResonance(
        omega=0.0j,
        signal_ghz=0.74 + 1.0e-3j,
        eig_min=0.0j,
        growth_rate_per_s=1.0,
        converged=True,
        iterations=1,
        residual=0.0,
    )
    points = [
        FloquetBranchPoint(
            parameter=-24.20,
            resonance=left,
            classification=_classification(0.99, "NEIMARK_SACKER_CANDIDATE"),
            branch_index=0,
            discontinuity=False,
            stability_verdict="STABLE",
            mode_overlap=None,
        ),
        FloquetBranchPoint(
            parameter=-24.25,
            resonance=right,
            classification=_classification(1.01, "NEIMARK_SACKER_CANDIDATE"),
            branch_index=0,
            discontinuity=False,
            stability_verdict="UNSTABLE_NS",
            mode_overlap=1.0,
        ),
    ]

    crossing = _crossing([(-24.20, points[0]), (-24.25, points[1])], 7.9)

    assert crossing is not None
    assert crossing["generator_frequency_ghz"] == pytest.approx(0.73)


def test_critical_root_accepts_descending_drive_ladder() -> None:
    assert parse_float_list("-24.05,-24.10,-24.15", name="drive") == [
        -24.05,
        -24.10,
        -24.15,
    ]
