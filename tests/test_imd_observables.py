from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.imd import ImProduct
from twpa_solver.multitone.observables import imd_products_dbc


def _circuit() -> CircuitMatrices:
    return CircuitMatrices(
        C=sp.eye(2, format="csr"), G=sp.eye(2, format="csr"),
        K=sp.eye(2, format="csr"), Bphi=sp.csr_matrix([[1.0], [-1.0]]),
        Ic=np.array([1.0]), port_to_index={1: 0, 2: 1},
    )


def _product(tone: ToneIndex) -> ImProduct:
    return ImProduct(3, 1, 2, tone, tone, False)


def test_signal_tone_is_zero_dbc() -> None:
    basis = MultiToneBasis(
        [ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1)], 10.0, 1.0
    )
    state = np.ones((basis.n_tones, 2), dtype=np.complex128) * 1e-3
    result = imd_products_dbc(state, basis, _circuit(), [_product(basis.signal_tone)], out_port=1)
    assert result["imd_o3_m1n2_dbc"] == 0.0


def test_absent_and_zero_tones_are_nan() -> None:
    basis = MultiToneBasis(
        [ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1)], 10.0, 1.0
    )
    absent = _product(ToneIndex(2, -2))
    result = imd_products_dbc(np.zeros((basis.n_tones, 2), complex), basis, _circuit(), [absent], out_port=1)
    assert np.isnan(result["imd_o3_m1n2_dbc"])


def test_imd_readout_is_selective_to_product_tone() -> None:
    basis = MultiToneBasis(
        [ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1), ToneIndex(2, -2)],
        10.0,
        1.0,
    )
    state = np.zeros((basis.n_tones, 2), dtype=np.complex128)
    state[basis.index_of(basis.pump_tone), 0] = 1.0
    state[basis.index_of(basis.signal_tone), 0] = 1.0
    state[basis.index_of(ToneIndex(2, -2)), 0] = 1e-3
    linear_circuit = CircuitMatrices(
        C=_circuit().C,
        G=_circuit().G,
        K=_circuit().K,
        Bphi=_circuit().Bphi,
        Ic=np.zeros(1),
        port_to_index=_circuit().port_to_index,
    )
    products = [_product(ToneIndex(2, -2))]
    result = imd_products_dbc(state, basis, linear_circuit, products, out_port=1)
    state[basis.index_of(basis.pump_tone), 0] = 1e6
    pump_changed = imd_products_dbc(
        state, basis, linear_circuit, products, out_port=1
    )
    assert np.isfinite(result["imd_o3_m1n2_dbc"])
    assert np.isclose(
        result["imd_o3_m1n2_dbc"], pump_changed["imd_o3_m1n2_dbc"], atol=1e-12
    )


def test_readout_is_independent_of_torus_grid_resolution() -> None:
    tones = [ToneIndex(1, 0), ToneIndex(1, -1), ToneIndex(1, 1), ToneIndex(2, -2)]
    coarse = MultiToneBasis(tones, 10.0, 1.0)
    fine = MultiToneBasis(tones, 10.0, 1.0, n_p=2 * coarse.n_p, n_delta=2 * coarse.n_delta)
    state = np.arange(coarse.n_tones * 2, dtype=float).reshape(coarse.n_tones, 2).astype(complex) * 1e-4
    products = [_product(ToneIndex(2, -2))]
    circuit = _circuit()
    circuit = CircuitMatrices(
        C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
        Ic=np.zeros_like(circuit.Ic), port_to_index=circuit.port_to_index,
    )
    coarse_result = imd_products_dbc(state, coarse, circuit, products, out_port=1)
    fine_result = imd_products_dbc(state, fine, circuit, products, out_port=1)
    assert coarse_result["imd_o3_m1n2_dbc"] == fine_result["imd_o3_m1n2_dbc"]
