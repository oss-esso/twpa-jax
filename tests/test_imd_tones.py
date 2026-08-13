from __future__ import annotations

import math

import pytest

from twpa_solver.multitone.basis import build_sideband_matched_basis
from twpa_solver.multitone.imd import (
    enumerate_im_products,
    extend_basis_with_im_tones,
    required_new_tones,
)


OMEGA_P = 2.0 * math.pi * 7.0e9
DELTA = 2.0 * math.pi * 1.0e9


def _basis():
    return build_sideband_matched_basis(
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        10,
        OMEGA_P,
        DELTA,
        OMEGA_P * 25,
    )


def test_enumeration_maps_all_order_nine_products() -> None:
    products = enumerate_im_products(9, OMEGA_P, DELTA)
    assert len(products) == 20
    for product in products:
        assert product.raw.h == product.m - product.n
        assert product.raw.q == -product.m
        assert product.tone.omega(OMEGA_P, DELTA) > 0.0
        expected = product.m * (OMEGA_P - DELTA) - product.n * OMEGA_P
        assert product.raw.omega(OMEGA_P, DELTA) == pytest.approx(expected, rel=1e-9)
        assert product.conjugated == (product.raw.omega(OMEGA_P, DELTA) < 0.0)


def test_required_tones_and_extension_are_targeted() -> None:
    basis = _basis()
    products3 = enumerate_im_products(3, OMEGA_P, DELTA)
    products9 = enumerate_im_products(9, OMEGA_P, DELTA)
    assert len(required_new_tones(products3, basis)) == 1
    assert len(required_new_tones(products9, basis)) == 16
    extended = extend_basis_with_im_tones(basis, products9)
    for product in products9:
        extended.index_of(product.tone)
    assert extended.n_delta >= basis.n_delta
    assert extended.n_delta == 34
    assert extend_basis_with_im_tones(extended, products9) is extended


@pytest.mark.parametrize("order", [0, 2, 4, 10])
def test_invalid_orders_are_rejected(order: int) -> None:
    with pytest.raises(ValueError, match="odd integer"):
        enumerate_im_products(order, OMEGA_P, DELTA)
