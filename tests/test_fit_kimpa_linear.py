import numpy as np
import pytest

from scripts.fit_kimpa_linear import (
    KimpaFitParameters,
    fit_parameters,
    model_dc_resonance_hz,
    model_s11,
    synthetic_dataset,
    validate_line_scales,
)


def _data():
    parameters = KimpaFitParameters(0.82e-9, 330e-15, 1.025, 2.0e5)
    f0 = model_dc_resonance_hz(np.array([0.0]), parameters)[0]
    frequencies = np.linspace(f0 * 0.96, f0 * 1.04, 121)
    currents = np.linspace(0.0, 600e-6, 13)
    return parameters, synthetic_dataset(parameters, frequencies, currents, noise_std_s11=2e-4, noise_std_dc_hz=5e5, seed=5)


def test_fit_recovers_synthetic_parameters():
    truth, data = _data()
    result = fit_parameters(data)
    fitted = result["parameters"]
    for name in ("Lk_h", "C_NR_f", "line_scale", "Qi"):
        assert fitted[name] == pytest.approx(getattr(truth, name), rel=0.02)


def test_s11_only_has_a_product_valley():
    truth, data = _data()
    alternate = KimpaFitParameters(0.70e-9, truth.Lk_h * truth.C_NR_f / 0.70e-9, truth.line_scale, truth.Qi)
    np.testing.assert_allclose(
        model_s11(data["frequency_hz"], truth),
        model_s11(data["frequency_hz"], alternate),
        rtol=0.0,
        atol=1e-12,
    )


def test_joint_objective_is_more_selective_than_s11_only():
    truth, data = _data()
    s11 = fit_parameters(data, include_dc=False)
    joint = fit_parameters(data, include_dc=True)
    s11_dc_error = np.mean((model_dc_resonance_hz(data["dc_current_a"], np.asarray(list(s11["parameters"].values()))) - data["dc_resonance_hz"]) ** 2)
    joint_dc_error = np.mean((model_dc_resonance_hz(data["dc_current_a"], np.asarray(list(joint["parameters"].values()))) - data["dc_resonance_hz"]) ** 2)
    assert joint_dc_error < s11_dc_error


def test_three_independent_line_lengths_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        validate_line_scales([0.95, 1.0, 1.05])


def test_model_outputs_have_expected_shapes():
    truth = KimpaFitParameters(0.8e-9, 330e-15, 1.0, 1e5)
    frequencies = np.linspace(7e9, 10e9, 10)
    assert model_s11(frequencies, truth).shape == frequencies.shape
    assert model_dc_resonance_hz(np.array([0.0, 500e-6]), truth).shape == (2,)
