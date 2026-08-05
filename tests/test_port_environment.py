import numpy as np
import pytest

from twpa_solver.core.environment import PortEnvironment


def test_null_environment_is_exactly_zero():
    environment = PortEnvironment(z1_ohm=0.0, z2_ohm=0.0)
    np.testing.assert_array_equal(environment.admittance(np.array([0.0, 1.0e10])), 0.0)


def test_environment_is_frequency_dependent():
    environment = PortEnvironment()
    assert environment.admittance(2.0 * np.pi * 8.0e9) != environment.admittance(2.0 * np.pi * 9.0e9)


def test_active_environment_is_rejected():
    with pytest.raises(ValueError, match="passive"):
        PortEnvironment(z1_ohm=60.0, z2_ohm=0.0)
