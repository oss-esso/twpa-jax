import numpy as np
import pytest

from twpa_solver.core.kinetic import (
    KineticInductorAltBranchLaw,
    KineticInductorBranchLaw,
    kinetic_validity,
    resolve_ki_model,
)
from twpa_solver.core.nonlinear import CompositeBranchLaw, JosephsonBranchLaw


def _law(quartic=True):
    ic = np.array([1.15e-3, 1.15e-3])
    i2, i4 = resolve_ki_model("hung_2025", ic)
    return KineticInductorBranchLaw(
        np.array([1e-9, 1.2e-9]), ic, i2, i4 if quartic else None
    )


@pytest.mark.parametrize("quartic", [False, True])
def test_flux_current_round_trip(quartic):
    law = _law(quartic)
    currents = np.linspace(-2 * 1.15e-3, 2 * 1.15e-3, 17)[:, None]
    currents = np.repeat(currents, 2, axis=1)
    flux = law.flux(currents)
    np.testing.assert_allclose(law.current(flux), currents, rtol=1e-14, atol=1e-25)


def test_tangent_matches_finite_difference():
    law = _law()
    flux = np.array([[-2e-15, 0.0], [1e-15, 3e-15]])
    step = 1e-21
    numerical = (law.current(flux + step) - law.current(flux - step)) / (2 * step)
    np.testing.assert_allclose(law.tangent(flux), numerical, rtol=1e-7)


def test_paper_dc_tuning_and_threshold_status():
    law = _law()
    current = np.repeat(np.array([[0.0, 300e-6, 530e-6, 600e-6]]).T, 2, axis=1)
    expected = 1 + (current / 3.25e-3) ** 2 + (current / 1.70e-3) ** 4
    np.testing.assert_allclose(
        law.differential_inductance(current), expected * np.array([1e-9, 1.2e-9])
    )
    report = kinetic_validity(
        law, law.flux(np.array([[0.0, 0.0], [1.0e-3, 1.2e-3]])), np.zeros(2)
    )
    assert report["status"].tolist() == ["SUPERCONDUCTING", "THRESHOLD_CROSSED"]


def test_explicit_hung_scales_are_not_derived_from_ic():
    law = KineticInductorBranchLaw(
        np.array([835e-12]), np.array([1.15e-3]),
        np.array([3.25e-3]), np.array([1.70e-3]),
    )
    current = np.array([[0.0], [0.20e-3], [-0.20e-3], [0.60e-3], [-0.60e-3], [0.90e-3]])
    ratio = law.differential_inductance(current)[:, 0] / 835e-12
    expected = 1.0 + (current[:, 0] / 3.25e-3) ** 2 + (current[:, 0] / 1.70e-3) ** 4
    np.testing.assert_allclose(ratio, expected, rtol=1e-14)
    assert ratio[0] == pytest.approx(1.0)
    assert ratio[3] == pytest.approx(1.0496, rel=2e-4)
    np.testing.assert_allclose(
        law.differential_inductance(-current)[:, 0],
        law.differential_inductance(current)[:, 0],
        rtol=1e-14,
    )
    incorrect = 1.0 + (0.60e-3 / 1.15e-3) ** 2
    assert incorrect == pytest.approx(1.2722, rel=2e-3)
    assert ratio[3] < incorrect


def test_explicit_hung_tangent_matches_centered_flux_difference():
    law = KineticInductorBranchLaw(
        np.array([835e-12]), np.array([1.15e-3]),
        np.array([3.25e-3]), np.array([1.70e-3]),
    )
    currents = np.array([[0.0], [0.20e-3], [-0.20e-3], [0.60e-3], [-0.60e-3], [0.90e-3]])
    flux = law.flux(currents)
    step = 1e-6 * 835e-12 * 3.25e-3
    numerical = (law.current(flux + step) - law.current(flux - step)) / (2.0 * step)
    np.testing.assert_allclose(law.tangent(flux), numerical, rtol=2e-8, atol=2e-2)


@pytest.mark.parametrize("current, expected", [(0.0, "SUPERCONDUCTING"), (1.15e-3, "THRESHOLD_CROSSED"), (1.20e-3, "THRESHOLD_CROSSED")])
def test_ic_boundary_is_diagnostic_only(current, expected):
    law = KineticInductorBranchLaw(
        np.array([835e-12]), np.array([1.15e-3]),
        np.array([3.25e-3]), np.array([1.70e-3]),
    )
    report = kinetic_validity(law, law.flux(np.array([[current]])))
    assert report["status"][0] == expected
    assert law.istar2_a[0] == 3.25e-3
    assert law.istar4_a[0] == 1.70e-3


def test_composite_law_dispatches_and_validates_columns():
    jj = JosephsonBranchLaw(np.array([1e-6, 2e-6]), 1e-15)
    ki = _law()
    composite = CompositeBranchLaw((jj, ki), (np.array([0, 2]), np.array([1, 3])))
    flux = np.zeros((4, 4))
    flux[:, 1] = 2e-15
    expected = np.empty_like(flux)
    expected[:, [0, 2]] = jj.current(flux[:, [0, 2]])
    expected[:, [1, 3]] = ki.current(flux[:, [1, 3]])
    np.testing.assert_allclose(composite.current(flux), expected)
    with pytest.raises(ValueError):
        CompositeBranchLaw((jj, ki), (np.array([0, 1]), np.array([1, 2])))


def test_unknown_preset_lists_valid_names():
    with pytest.raises(ValueError, match="hung_2025"):
        resolve_ki_model("missing", 1e-3)


def test_alt_law_round_trip_and_finite_flux_boundary():
    law = KineticInductorAltBranchLaw(
        np.array([1e-9]), np.array([1.15e-3]), np.array([1.65e-3])
    )
    current = np.linspace(-0.99 * 1.65e-3, 0.99 * 1.65e-3, 11)[:, None]
    np.testing.assert_allclose(law.current(law.flux(current)), current, rtol=1e-12, atol=1e-18)
    expected_max = 1e-9 * 1.65e-3 * (np.pi / np.sin(np.pi / 2.21)) / 2.21
    assert law._phi_max()[0] == pytest.approx(expected_max, rel=1e-12)


def test_alt_law_out_of_domain_is_counted_and_finite():
    law = KineticInductorAltBranchLaw(
        np.array([1e-9]), np.array([1.15e-3]), np.array([1.65e-3])
    )
    value = law.current(np.array([[2.0 * law._phi_max()[0]]]))
    assert np.isfinite(value[0, 0]) and law.out_of_domain_samples == 1
