from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from twpa_solver.chaos.routing import (
    Regime,
    classify_from_multiplier,
    classify_from_spectrum,
    probe_multiplier,
    route,
)


@pytest.mark.parametrize(
    ("magnitude", "expected"),
    [
        (0.997, Regime.PERIOD_1),
        (0.999, Regime.UNDECIDED),
        (1.000, Regime.UNDECIDED),
        (1.001, Regime.UNDECIDED),
        (1.003, Regime.TORUS),
    ],
)
def test_multiplier_classifier_preserves_the_crossing_band(
    magnitude: float,
    expected: Regime,
) -> None:
    verdict = classify_from_multiplier(magnitude)

    assert verdict.regime is expected
    assert verdict.reason


@pytest.mark.parametrize(
    ("on_lattice", "generator_share", "expected"),
    [
        (0.99999997, 0.3897, Regime.PERIOD_1),
        (0.99999997, 0.7698, Regime.TORUS),
        (0.8827, 0.10, Regime.UNDECIDED),
        (0.4620, 0.10, Regime.UNDECIDED),
        (0.0560, 0.26, Regime.BROADBAND),
    ],
)
def test_spectrum_classifier_uses_the_measured_gap(
    on_lattice: float,
    generator_share: float,
    expected: Regime,
) -> None:
    verdict = classify_from_spectrum(on_lattice, generator_share)

    assert verdict.regime is expected
    assert verdict.reason


def test_route_returns_method_name_only() -> None:
    assert route(classify_from_multiplier(0.9)) == "single_tone_hb"
    assert route(classify_from_multiplier(1.1)) == "two_frequency_hb"
    assert route(classify_from_spectrum(0.05, 0.1)) == "fdtd"
    assert route(classify_from_multiplier(1.0)) == "undecided"


@dataclass
class _Candidate:
    multiplier: complex
    mode_vector: np.ndarray
    signal_ghz: complex
    converged: bool = True


def test_probe_searches_both_half_planes_and_tracks_overlap() -> None:
    seeds: list[complex] = []

    def refine(seed: complex) -> _Candidate:
        seeds.append(seed)
        if seed.imag < 0.0:
            return _Candidate(1.2 + 0.0j, np.array([0.0, 1.0]), seed)
        return _Candidate(0.8 + 0.0j, np.array([1.0, 0.01]), seed)

    verdict = probe_multiplier(
        seed_signal_ghz=7.9,
        previous_mode_vector=np.array([1.0, 0.0]),
        refine=refine,
    )

    assert [seed.imag for seed in seeds] == [-0.001, 0.001]
    assert verdict.regime is Regime.PERIOD_1
    assert verdict.mode_overlap == pytest.approx(1.0, abs=1.0e-4)
    assert verdict.evidence == pytest.approx(0.8)
    assert verdict.searched_imaginary_half_planes == (-0.001, 0.001)


def test_probe_rejects_a_low_overlap_branch() -> None:
    def refine(seed: complex) -> _Candidate:
        return _Candidate(1.1 + 0.0j, np.array([0.0, 1.0]), seed)

    verdict = probe_multiplier(
        seed_signal_ghz=7.9,
        previous_mode_vector=np.array([1.0, 0.0]),
        refine=refine,
    )

    assert verdict.regime is Regime.UNDECIDED
    assert verdict.mode_overlap == pytest.approx(0.0)
    assert "overlap" in verdict.reason
