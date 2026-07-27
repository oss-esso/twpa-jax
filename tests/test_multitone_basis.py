from __future__ import annotations

import numpy as np
import pytest

from twpa_solver.multitone.basis import (
    MultiToneBasis,
    ToneIndex,
    build_lattice_basis,
    build_three_tone_basis,
    canonicalize,
)


def test_three_tone_transform_round_trip() -> None:
    basis = build_three_tone_basis(10.0, 1.0)
    rng = np.random.default_rng(4)
    coefficients = rng.normal(size=(basis.n_tones, 3)) + 1j * rng.normal(
        size=(basis.n_tones, 3)
    )

    waveform = basis.synthesize(coefficients)

    assert np.isrealobj(waveform)
    np.testing.assert_allclose(basis.project(waveform), coefficients, atol=1e-14)


def test_lattice_basis_has_required_tones_and_alias_guard() -> None:
    basis = build_lattice_basis([1, 3], 2, 10.0, 1.0, 40.0)

    assert basis.pump_tone in basis.tones
    assert basis.signal_tone in basis.tones
    assert basis.idler_tone in basis.tones
    assert basis.n_p >= 2 * 6 + 1
    assert basis.n_p % 2 == 0


def test_canonicalize_rejects_zero_frequency() -> None:
    with pytest.raises(ValueError, match="DC"):
        canonicalize(ToneIndex(0, 0), 10.0, 1.0)

    assert canonicalize(ToneIndex(-1, 0), 10.0, 1.0) == (ToneIndex(1, 0), True)


def test_basis_rejects_conjugate_pair_and_missing_required_tone() -> None:
    with pytest.raises(ValueError, match="conjugate"):
        MultiToneBasis(
            [ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1), ToneIndex(-1, 0)],
            10.0,
            1.0,
        )
    with pytest.raises(ValueError, match="missing"):
        MultiToneBasis([ToneIndex(1, 0)], 10.0, 1.0)


def test_cubic_mixing_projects_expected_coefficients() -> None:
    basis = MultiToneBasis(
        [ToneIndex(1, 0), ToneIndex(3, 0), ToneIndex(1, -1), ToneIndex(1, 1)],
        10.0,
        1.0,
    )
    waveform = basis.synthesize(
        np.array([[0.5], [0.0], [0.0], [0.0]], dtype=np.complex128)
    )
    cubic = waveform**3
    projected = basis.project(cubic)

    # Positive-phasor coefficients are half the usual cosine amplitudes.
    assert projected[basis.index_of(ToneIndex(1, 0)), 0] == pytest.approx(0.375)
    assert projected[basis.index_of(ToneIndex(3, 0)), 0] == pytest.approx(0.125)
