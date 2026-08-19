from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_torus_branch import (
    _effective_drive_dbm,
    _remap_state_basis,
    _torus_radius_squared,
)
from scripts.chaos.run_branch_locked_torus import (
    _drive_ratio,
    _fit_normal_form,
    _normal_form_predictor,
)
from scripts.chaos.run_torus_signal_probe import gain_vs_off_db
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


def test_signal_probe_gain_is_normalized_to_pump_off_response() -> None:
    assert gain_vs_off_db(2.0 + 0.0j, 1.0 + 0.0j) == 20.0 * np.log10(2.0)


def test_normal_form_fit_recovers_two_known_torus_points() -> None:
    fit = _fit_normal_form(
        [(-24.05, 0.17808751132565295), (-23.90, 0.31612030798990987)]
    )

    np.testing.assert_allclose(fit["normal_form_slope"], 0.92021864, rtol=1e-6)
    np.testing.assert_allclose(fit["normal_form_pc_dbm"], -24.2435274, rtol=1e-6)
    np.testing.assert_allclose(fit["normal_form_r2_fit_r2"], 1.0)


def test_normal_form_predictor_scales_only_nonzero_generator_sector() -> None:
    basis = _basis(
        [ToneIndex(0, 0), ToneIndex(0, 1), ToneIndex(0, -1)]
    )
    previous = np.asarray(
        [[2.0], [np.sqrt(2.0 * 0.31612030798990987)],
         [np.sqrt(2.0 * 0.31612030798990987)]],
        dtype=np.complex128,
    )
    prior = np.asarray(
        [[1.0], [np.sqrt(2.0 * 0.17808751132565295)],
         [np.sqrt(2.0 * 0.17808751132565295)]],
        dtype=np.complex128,
    )
    predictor, fit = _normal_form_predictor(
        previous,
        prior,
        -23.90,
        -24.05,
        -23.80,
        basis,
        [(-24.05, 0.17808751132565295), (-23.90, 0.31612030798990987)],
    )

    np.testing.assert_allclose(predictor[0], 2.6666666666666665)
    expected_ratio = np.sqrt(0.4081421724565755 / 0.31612030798990987)
    np.testing.assert_allclose(predictor[1:], previous[1:] * expected_ratio)
    np.testing.assert_allclose(
        fit["normal_form_predicted_radius_ratio"], expected_ratio
    )


def test_drive_ratio_rejects_duplicate_accepted_drives() -> None:
    with pytest.raises(ValueError, match="accepted drive spacing is zero"):
        _drive_ratio(-23.9, -23.9, -23.8)
