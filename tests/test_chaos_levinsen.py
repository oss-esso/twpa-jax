from __future__ import annotations

import numpy as np

from scripts.chaos.levinsen_paramp import (
    LevinsenParameters,
    gamma_from_phasors,
    integrate_levinsen,
    levinsen_rhs,
    run_gain_point,
)


def test_four_state_integrator_reduces_to_free_rotation() -> None:
    parameters = LevinsenParameters(beta_c=1.0e20, josephson_current=0.0,
                                    dc_bias=0.0, pump_amplitude=0.0,
                                    tuned_circuit_frequency_hz=0.0)
    state = np.array([0.0, 1.25, 0.0, 0.0])
    derivative = levinsen_rhs(0.0, state, parameters, signal_current=0.0)
    assert np.allclose(derivative, [1.25, 0.0, 0.0, 0.0])
    times, states = integrate_levinsen(parameters, state, duration=0.02, step=0.001)
    assert times.size == states.shape[0]
    assert np.allclose(states[-1, 0], state[0] + state[1] * 0.02, atol=1e-10)


def test_gamma_readout_uses_load_current_sum() -> None:
    gamma = gamma_from_phasors(1.0 + 0.0j, 2.0 + 0.0j)
    assert np.isclose(gamma, 9.0)


def test_tuned_current_loads_the_junction_and_uses_resistor_q_coupling() -> None:
    parameters = LevinsenParameters(tuned_circuit_frequency_hz=240.0)
    state = np.array([0.0, 1.0, 0.0, 0.0])
    unloaded = levinsen_rhs(0.0, state, parameters)
    loaded = levinsen_rhs(0.0, state + np.array([0.0, 0.0, 0.0, 0.2]), parameters)
    assert loaded[1] < unloaded[1]
    expected_drive = parameters.tuned_omega / (
        parameters.load_resistance_ratio * parameters.tuned_q
    )
    driven = levinsen_rhs(0.0, state, parameters)
    assert np.isclose(driven[3], expected_drive)


def test_noise_control_is_not_evaluable_without_gain() -> None:
    result = run_gain_point(
        LevinsenParameters(tuned_circuit_frequency_hz=240.0, pump_amplitude=0.0),
        noise_amplitude=0.02, duration=2.0, step=0.002,
    )
    assert result["noise_result"] == "not_evaluable_no_gain"
    assert np.isnan(float(result["noise_temperature_ratio"]))
