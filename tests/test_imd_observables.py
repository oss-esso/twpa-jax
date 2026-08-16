from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.imd import (
    ImProduct,
    enumerate_im_products,
    enumerate_two_tone_im_products,
)
from twpa_solver.multitone.observables import imd_products_dbc
from twpa_solver.multitone.source import AffineSourcePath


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


def test_two_tone_rejects_degenerate_four_wave_mixing_placement() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        enumerate_two_tone_im_products(5, 1, -1)


def test_two_tone_rejects_pump_collision() -> None:
    with pytest.raises(ValueError, match="q=0"):
        enumerate_two_tone_im_products(5, 1, 2)


def test_two_tone_rejects_fundamental_collision() -> None:
    with pytest.raises(ValueError, match="fundamental"):
        enumerate_two_tone_im_products(3, 1, 3)


def test_two_tone_products_round_trip_to_physical_frequencies() -> None:
    omega_p = 2.0 * np.pi * 5.0e9
    delta = 2.0 * np.pi * 50.0e6
    products = enumerate_two_tone_im_products(5, 5, 7)
    for product in products:
        if product.ordering == "w1_minus_w2":
            raw_frequency = product.m * (omega_p + 5.0 * delta) - product.n * (
                omega_p + 7.0 * delta
            )
        else:
            raw_frequency = product.m * (omega_p + 7.0 * delta) - product.n * (
                omega_p + 5.0 * delta
            )
        reconstructed = product.tone.omega(omega_p, delta)
        assert reconstructed == pytest.approx(abs(raw_frequency), rel=1.0e-12)


def test_two_tone_source_keeps_independent_amplitudes() -> None:
    pump = np.zeros((2, 2), dtype=complex)
    tone_1 = np.ones((2, 2), dtype=complex)
    tone_2 = 2.0 * np.ones((2, 2), dtype=complex)
    path = AffineSourcePath.two_tone_signal_turn_on(
        pump, tone_1, tone_2, amplitude_1=3.0, amplitude_2=0.5
    )
    assert np.all(path.source(1.0) == 4.0)


def test_single_tone_imd_coordinates_match_legacy_fixture() -> None:
    omega_p = 2.0 * np.pi * 7.0e9
    delta = 2.0 * np.pi * 0.12e9
    actual = enumerate_im_products(5, omega_p, delta)
    expected = []
    for order in (3, 5):
        for m in range(1, order):
            n = order - m
            expected.append((order, m, n, m - n, -m))
    assert [
        (product.order, product.m, product.n, product.raw.h, product.raw.q)
        for product in actual
    ] == expected
