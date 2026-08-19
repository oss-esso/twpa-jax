"""Tests for the independent-frequency torus signal response lattice."""

from __future__ import annotations

import numpy as np
import pytest

from twpa_solver.multitone.basis import ToneIndex
from twpa_solver.multitone.torus_signal import response_tones


def test_response_lattice_retains_signal_and_signed_idler_partner() -> None:
    """The response lattice contains the signal and signed pump sidebands."""
    tones = response_tones(
        2.0 * np.pi * 7.9e9,
        2.0 * np.pi * 0.729e9,
        2.0 * np.pi * 7.4e9,
    )
    assert ToneIndex(0, 0) in tones
    assert ToneIndex(-2, 0) in tones
    assert len(tones) == 15


def test_response_lattice_rejects_near_dc_alias() -> None:
    """A response lattice must not silently retain a near-DC tone."""
    with pytest.raises(ValueError, match="near-DC"):
        response_tones(
            2.0 * np.pi * 7.9e9,
            2.0 * np.pi * 0.729e9,
            2.0 * np.pi * 15.8e9,
            h_min=-2,
            h_max=0,
            q_min=0,
            q_max=0,
        )
