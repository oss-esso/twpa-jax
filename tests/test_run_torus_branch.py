from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.run_torus_branch import (
    _effective_drive_dbm,
    _remap_state_basis,
    _torus_radius_squared,
)
from twpa_solver.multitone.basis import ToneIndex


def _basis(tones: list[ToneIndex]) -> SimpleNamespace:
    return SimpleNamespace(tones=tones, n_tones=len(tones))


def test_torus_radius_squared_uses_q_plus_minus_one_relative_to_q_zero() -> None:
    basis = _basis([ToneIndex(0, 0), ToneIndex(0, 1), ToneIndex(0, -1)])
    state = np.asarray([[2.0], [0.5], [0.5]], dtype=np.complex128)

    assert _torus_radius_squared(state, basis) == 0.125


def test_remap_state_basis_preserves_common_tones_and_zero_fills() -> None:
    source_basis = _basis([ToneIndex(0, 0), ToneIndex(1, 0)])
    target_basis = _basis(
        [ToneIndex(-1, 0), ToneIndex(0, 0), ToneIndex(1, 0)]
    )
    source = np.asarray([[2.0], [3.0]], dtype=np.complex128)

    promoted = _remap_state_basis(source, source_basis, target_basis)

    np.testing.assert_array_equal(promoted[:, 0], [0.0, 2.0, 3.0])


def test_effective_drive_dbm_uses_source_amplitude_scaling() -> None:
    assert _effective_drive_dbm(-24.0, 1.0) == -24.0
    np.testing.assert_allclose(_effective_drive_dbm(-24.0, 2.0), -17.97940009)
