from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.core import (
    CircuitMatrices,
    default_loss_model_for,
    rcsj_parameters,
    stamp_rcsj_shunt,
)


def _circuit() -> CircuitMatrices:
    # Two branch columns share a node, matching the incidence structure used
    # by the distributed junction circuits while keeping the test tiny.
    bphi = sp.csr_matrix([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]])
    return CircuitMatrices(
        C=sp.eye(3, format="csr") * 55e-15,
        G=sp.diags([1.0 / 50.0, 0.0, 1.0 / 50.0], format="csr"),
        K=sp.eye(3, format="csr"),
        Bphi=bphi,
        Ic=np.array([3.4e-6, 3.4e-6]),
        metadata={"junction_capacitance_f": 55e-15},
    )


def test_infinite_resistance_is_exact_noop_and_keeps_analytic_loss_route() -> None:
    circuit = _circuit()
    control, params = stamp_rcsj_shunt(circuit, math.inf)

    assert control is circuit
    assert np.array_equal(control.G.data, circuit.G.data)
    assert not control.has_loss
    assert default_loss_model_for(control) == "current_complex_c"
    assert np.all(np.isinf(params.resistance_ohm))


def test_finite_rcsj_stamp_is_symmetric_positive_semidefinite() -> None:
    circuit = _circuit()
    damped, params = stamp_rcsj_shunt(circuit, 1.0e4)
    delta_g = (damped.G - circuit.G).toarray()

    assert np.array_equal(delta_g, delta_g.T)
    # The exact factorization is PSD; the zero mode acquires ordinary
    # double-precision eigensolver round-off at this scale.
    assert np.linalg.eigvalsh(delta_g).min() >= -1.0e-15
    assert np.all(params.resistance_ohm > 0.0)
    assert not damped.has_loss
    assert default_loss_model_for(damped) == "current_complex_c"


def test_ambegaokar_barattoff_and_damping_scalings() -> None:
    params = rcsj_parameters(
        np.array([3.4e-6]), 55e-15, 1.0,
        delta_ev=180e-6, pump_frequency_hz=7.12e9,
    )
    assert params.rn_ohm[0] == pytest.approx(math.pi * 180e-6 / (2.0 * 3.4e-6))
    assert params.beta_c[0] == pytest.approx(params.quality_factor[0] ** 2)
    weak = rcsj_parameters(np.array([3.4e-6]), 55e-15, 1.0e6)
    assert weak.damping_per_pump_period[0] == pytest.approx(
        params.damping_per_pump_period[0] / 1.0e6
    )


def test_rcsj_rejects_nonpositive_ratio() -> None:
    with pytest.raises(ValueError):
        stamp_rcsj_shunt(_circuit(), 0.0)
