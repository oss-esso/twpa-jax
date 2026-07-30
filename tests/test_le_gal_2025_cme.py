from __future__ import annotations

import numpy as np
import pytest

from references.le_gal_2025_gain_compression.cme import (
    CMEParameters,
    depletion_only_gain,
    envelopes_from_powers,
    integrate_cme,
    photon_flux,
    published_cme_parameters,
)


def test_cme_zero_coupling_is_passive() -> None:
    z, envelopes = integrate_cme(
        (1.0 + 2.0j, 0.3 - 0.1j, 0.2j),
        CMEParameters(length=2.0, coupling=0.0),
    )
    assert z[-1] == 2.0
    np.testing.assert_allclose(envelopes[:, -1], envelopes[:, 0])


def test_cme_lossless_photon_flux_is_conserved() -> None:
    _, envelopes = integrate_cme(
        (1.0, 0.2, 0.1j),
        CMEParameters(length=2.0, coupling=0.7, phase_mismatch=0.0),
    )
    flux = photon_flux(envelopes)
    assert np.max(np.abs(flux - flux[0])) < 1e-7


def test_cme_tighter_integration_converges() -> None:
    _, coarse = integrate_cme(
        (1.0, 0.1, 0.0), CMEParameters(coupling=0.8), rtol=1e-7, max_step=0.05
    )
    _, refined = integrate_cme(
        (1.0, 0.1, 0.0), CMEParameters(coupling=0.8), rtol=1e-10, max_step=0.01
    )
    np.testing.assert_allclose(coarse[:, -1], refined[:, -1], rtol=2e-5, atol=2e-7)


def test_simple_depletion_model_gives_paper_scale_p1db() -> None:
    gain = 10.0**2
    pump = 10.0 ** (-78.4 / 10.0)
    target = (10.0 ** 0.1 - 1.0) * pump / (2.0 * gain)
    signal_dbm = 10.0 * np.log10(target)
    assert signal_dbm == pytest.approx(-107.3, abs=0.2)


def test_published_cme_coefficients_are_derived_and_nonzero() -> None:
    parameters = published_cme_parameters(6.0e9)
    assert parameters.coupling != 0.0
    assert parameters.self_phase_p != 0.0
    assert parameters.phase_mismatch != 0.0
    initial = envelopes_from_powers(1.0e-11, 1.0e-14, parameters)
    assert initial[0] > initial[1] > 0.0


def test_published_four_wave_cme_conserves_photon_flux() -> None:
    parameters = published_cme_parameters(6.4e9)
    initial = envelopes_from_powers(
        10.0 ** ((-78.4 - 30.0) / 10.0),
        10.0 ** ((-115.0 - 30.0) / 10.0),
        parameters,
    )
    _, envelopes = integrate_cme(initial, parameters, points=401)
    flux = photon_flux(envelopes)
    assert np.max(np.abs(flux - flux[0])) / flux[0] < 1e-7
