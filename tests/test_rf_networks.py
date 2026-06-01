"""
Tests for twpa.linear.rf_networks.

These tests define the expected RF-network algebra layer for the TWPA stack:

    - ABCD identity, series impedance, and shunt admittance blocks
    - ABCD cascading
    - ABCD <-> S-parameter conversion
    - transmission-line ABCD blocks
    - shape/broadcast behavior over frequency grids
    - basic reciprocity/passivity sanity checks

The tests are API-tolerant in naming, but they require the module to expose
clear public helpers for the core two-port operations.
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import numpy as np
import pytest

import twpa.linear.rf_networks as rf


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def _get_any(*names: str) -> Any:
    for name in names:
        if hasattr(rf, name):
            return getattr(rf, name)
    raise AttributeError(f"twpa.linear.rf_networks is missing all of: {names}")


def _as_abcd(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128)
    assert arr.shape[-2:] == (2, 2), f"Expected ABCD shape (..., 2, 2), got {arr.shape}"
    return arr


def _as_s(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128)
    assert arr.shape[-2:] == (2, 2), f"Expected S shape (..., 2, 2), got {arr.shape}"
    return arr


def _identity_abcd() -> np.ndarray:
    for name in ["abcd_identity", "identity_abcd", "identity_twoport", "twoport_identity"]:
        if hasattr(rf, name):
            return _as_abcd(getattr(rf, name)())

    raise AttributeError(
        "twpa.linear.rf_networks must expose an ABCD identity helper such as "
        "abcd_identity() or identity_abcd()."
    )


def _series_abcd(Z: Any) -> np.ndarray:
    for name in [
        "series_impedance_abcd",
        "abcd_series_impedance",
        "series_abcd",
        "series_impedance",
    ]:
        if hasattr(rf, name):
            return _as_abcd(
                _call_with_supported_kwargs(
                    getattr(rf, name),
                    Z=Z,
                    z=Z,
                    impedance=Z,
                    impedance_ohm=Z,
                    Z_ohm=Z,
                )
            )

    raise AttributeError(
        "twpa.linear.rf_networks must expose a series impedance ABCD helper such "
        "as series_impedance_abcd(Z)."
    )


def _shunt_abcd(Y: Any) -> np.ndarray:
    for name in [
        "shunt_admittance_abcd",
        "abcd_shunt_admittance",
        "shunt_abcd",
        "shunt_admittance",
    ]:
        if hasattr(rf, name):
            return _as_abcd(
                _call_with_supported_kwargs(
                    getattr(rf, name),
                    Y=Y,
                    y=Y,
                    admittance=Y,
                    admittance_S=Y,
                    Y_S=Y,
                )
            )

    raise AttributeError(
        "twpa.linear.rf_networks must expose a shunt admittance ABCD helper such "
        "as shunt_admittance_abcd(Y)."
    )


def _cascade_abcd(*matrices: Any) -> np.ndarray:
    for name in ["cascade_abcd", "abcd_cascade", "cascade_twoports", "cascade"]:
        if hasattr(rf, name):
            fn = getattr(rf, name)

            try:
                return _as_abcd(fn(*matrices))
            except TypeError:
                pass

            return _as_abcd(
                _call_with_supported_kwargs(
                    fn,
                    matrices=list(matrices),
                    abcds=list(matrices),
                    networks=list(matrices),
                    twoports=list(matrices),
                )
            )

    raise AttributeError(
        "twpa.linear.rf_networks must expose an ABCD cascade helper such as "
        "cascade_abcd(A, B, ...)."
    )


def _abcd_to_s(abcd: Any, *, z0_ohm: float = 50.0) -> np.ndarray:
    for name in ["abcd_to_s", "abcd_to_sparams", "abcd_to_s_parameters", "s_from_abcd"]:
        if hasattr(rf, name):
            return _as_s(
                _call_with_supported_kwargs(
                    getattr(rf, name),
                    abcd=abcd,
                    ABCD=abcd,
                    matrix=abcd,
                    z0_ohm=z0_ohm,
                    Z0_ohm=z0_ohm,
                    z0=z0_ohm,
                    reference_impedance_ohm=z0_ohm,
                )
            )

    raise AttributeError(
        "twpa.linear.rf_networks must expose ABCD-to-S conversion such as "
        "abcd_to_s(abcd, z0_ohm=50)."
    )


def _s_to_abcd(s: Any, *, z0_ohm: float = 50.0) -> np.ndarray:
    for name in ["s_to_abcd", "sparams_to_abcd", "s_parameters_to_abcd", "abcd_from_s"]:
        if hasattr(rf, name):
            return _as_abcd(
                _call_with_supported_kwargs(
                    getattr(rf, name),
                    s=s,
                    S=s,
                    sparams=s,
                    matrix=s,
                    z0_ohm=z0_ohm,
                    Z0_ohm=z0_ohm,
                    z0=z0_ohm,
                    reference_impedance_ohm=z0_ohm,
                )
            )

    raise AttributeError(
        "twpa.linear.rf_networks must expose S-to-ABCD conversion such as "
        "s_to_abcd(s, z0_ohm=50)."
    )


def _transmission_line_abcd(
    *,
    gamma: Any,
    length_m: float,
    z0_ohm: float,
) -> np.ndarray:
    for name in [
        "transmission_line_abcd",
        "abcd_transmission_line",
        "line_abcd",
        "uniform_line_abcd",
    ]:
        if hasattr(rf, name):
            return _as_abcd(
                _call_with_supported_kwargs(
                    getattr(rf, name),
                    gamma=gamma,
                    propagation_constant=gamma,
                    gamma_per_m=gamma,
                    length_m=length_m,
                    length=length_m,
                    z0_ohm=z0_ohm,
                    Z0_ohm=z0_ohm,
                    characteristic_impedance_ohm=z0_ohm,
                )
            )

    raise AttributeError(
        "twpa.linear.rf_networks must expose a transmission-line ABCD helper such "
        "as transmission_line_abcd(gamma, length_m, z0_ohm)."
    )


def _det2(m: np.ndarray) -> np.ndarray:
    return m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]


def test_identity_abcd_is_2x2_identity() -> None:
    I = _identity_abcd()

    assert I.shape == (2, 2)
    np.testing.assert_allclose(I, np.eye(2), rtol=0.0, atol=0.0)


def test_series_impedance_abcd_matches_definition() -> None:
    Z = 12.5 + 3.0j
    A = _series_abcd(Z)

    expected = np.array([[1.0, Z], [0.0, 1.0]], dtype=np.complex128)
    np.testing.assert_allclose(A, expected, rtol=1e-14, atol=1e-14)


def test_shunt_admittance_abcd_matches_definition() -> None:
    Y = 2.0e-3 + 1.5e-3j
    A = _shunt_abcd(Y)

    expected = np.array([[1.0, 0.0], [Y, 1.0]], dtype=np.complex128)
    np.testing.assert_allclose(A, expected, rtol=1e-14, atol=1e-14)


def test_cascade_matches_matrix_multiplication() -> None:
    Z = 7.0 + 0.5j
    Y = 1.3e-3 - 0.2e-3j

    A = _series_abcd(Z)
    B = _shunt_abcd(Y)
    C = _series_abcd(2.0 * Z)

    cascaded = _cascade_abcd(A, B, C)
    expected = A @ B @ C

    np.testing.assert_allclose(cascaded, expected, rtol=1e-14, atol=1e-14)


def test_cascade_identity_is_noop() -> None:
    I = _identity_abcd()
    A = _series_abcd(25.0 + 4.0j)

    left = _cascade_abcd(I, A)
    right = _cascade_abcd(A, I)

    np.testing.assert_allclose(left, A, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(right, A, rtol=1e-14, atol=1e-14)


def test_series_and_shunt_blocks_are_reciprocal() -> None:
    series = _series_abcd(15.0 + 1.0j)
    shunt = _shunt_abcd(2e-3 + 0.5e-3j)

    assert _det2(series) == pytest.approx(1.0 + 0.0j)
    assert _det2(shunt) == pytest.approx(1.0 + 0.0j)


def test_cascade_of_reciprocal_blocks_is_reciprocal() -> None:
    blocks = [
        _series_abcd(2.0 + 1.0j),
        _shunt_abcd(1e-3 + 0.2e-3j),
        _series_abcd(4.0 - 0.5j),
        _shunt_abcd(0.5e-3 - 0.1e-3j),
    ]

    total = _cascade_abcd(*blocks)

    assert _det2(total) == pytest.approx(1.0 + 0.0j, rel=1e-12, abs=1e-12)


def test_abcd_to_s_identity_gives_matched_through() -> None:
    S = _abcd_to_s(_identity_abcd(), z0_ohm=50.0)

    expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    np.testing.assert_allclose(S, expected, rtol=1e-13, atol=1e-13)


def test_abcd_to_s_series_impedance_known_result() -> None:
    z0 = 50.0
    Z = 25.0
    S = _abcd_to_s(_series_abcd(Z), z0_ohm=z0)

    denom = 2.0 * z0 + Z
    expected = np.array(
        [
            [Z / denom, 2.0 * z0 / denom],
            [2.0 * z0 / denom, Z / denom],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(S, expected, rtol=1e-13, atol=1e-13)


def test_abcd_to_s_shunt_admittance_known_result() -> None:
    z0 = 50.0
    Y = 0.01
    S = _abcd_to_s(_shunt_abcd(Y), z0_ohm=z0)

    denom = 2.0 + Y * z0
    expected = np.array(
        [
            [-Y * z0 / denom, 2.0 / denom],
            [2.0 / denom, -Y * z0 / denom],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(S, expected, rtol=1e-13, atol=1e-13)


def test_abcd_s_round_trip_for_reciprocal_network() -> None:
    z0 = 50.0
    A = _cascade_abcd(
        _series_abcd(5.0 + 2.0j),
        _shunt_abcd(2e-3 + 0.5e-3j),
        _series_abcd(3.0 - 1.0j),
    )

    S = _abcd_to_s(A, z0_ohm=z0)
    A_roundtrip = _s_to_abcd(S, z0_ohm=z0)

    np.testing.assert_allclose(A_roundtrip, A, rtol=1e-12, atol=1e-12)


def test_s_parameters_of_reciprocal_abcd_are_symmetric_in_transmission() -> None:
    A = _cascade_abcd(
        _series_abcd(10.0 + 3.0j),
        _shunt_abcd(1e-3 + 2e-4j),
    )

    S = _abcd_to_s(A, z0_ohm=50.0)

    assert S[0, 1] == pytest.approx(S[1, 0], rel=1e-12, abs=1e-12)


def test_lossless_series_reactance_is_passive() -> None:
    X = 10.0
    S = _abcd_to_s(_series_abcd(1j * X), z0_ohm=50.0)

    # For a lossless reciprocal two-port with matched real reference impedance,
    # each column of S has unit power norm.
    column_power = np.sum(np.abs(S) ** 2, axis=0)

    np.testing.assert_allclose(column_power, np.ones(2), rtol=1e-13, atol=1e-13)


def test_lossless_shunt_susceptance_is_passive() -> None:
    B = 3e-3
    S = _abcd_to_s(_shunt_abcd(1j * B), z0_ohm=50.0)

    column_power = np.sum(np.abs(S) ** 2, axis=0)

    np.testing.assert_allclose(column_power, np.ones(2), rtol=1e-13, atol=1e-13)


def test_resistive_series_network_is_passive() -> None:
    S = _abcd_to_s(_series_abcd(25.0), z0_ohm=50.0)

    column_power = np.sum(np.abs(S) ** 2, axis=0)

    assert np.all(column_power <= 1.0 + 1e-13)
    assert np.all(column_power >= 0.0)


def test_transmission_line_zero_length_is_identity() -> None:
    gamma = 0.01 + 1j * 3.0
    A = _transmission_line_abcd(gamma=gamma, length_m=0.0, z0_ohm=50.0)

    np.testing.assert_allclose(A, np.eye(2), rtol=1e-14, atol=1e-14)


def test_transmission_line_abcd_matches_closed_form() -> None:
    gamma = 0.02 + 1j * 4.0
    length_m = 0.125
    z0 = 50.0

    A = _transmission_line_abcd(gamma=gamma, length_m=length_m, z0_ohm=z0)

    gl = gamma * length_m
    expected = np.array(
        [
            [np.cosh(gl), z0 * np.sinh(gl)],
            [np.sinh(gl) / z0, np.cosh(gl)],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(A, expected, rtol=1e-13, atol=1e-13)


def test_transmission_line_is_reciprocal() -> None:
    gamma = 0.01 + 1j * 2.0
    A = _transmission_line_abcd(gamma=gamma, length_m=0.2, z0_ohm=50.0)

    assert _det2(A) == pytest.approx(1.0 + 0.0j, rel=1e-12, abs=1e-12)


def test_transmission_line_matched_lossless_has_zero_reflection() -> None:
    beta = 12.3
    length_m = 0.05
    z0 = 50.0

    A = _transmission_line_abcd(gamma=1j * beta, length_m=length_m, z0_ohm=z0)
    S = _abcd_to_s(A, z0_ohm=z0)

    assert S[0, 0] == pytest.approx(0.0 + 0.0j, abs=1e-12)
    assert S[1, 1] == pytest.approx(0.0 + 0.0j, abs=1e-12)
    assert abs(S[1, 0]) == pytest.approx(1.0, rel=1e-12)
    assert abs(S[0, 1]) == pytest.approx(1.0, rel=1e-12)


def test_series_impedance_supports_frequency_arrays() -> None:
    Z = np.array([1.0, 2.0, 3.0]) + 1j * np.array([0.1, 0.2, 0.3])

    A = _series_abcd(Z)

    assert A.shape == (3, 2, 2)
    np.testing.assert_allclose(A[:, 0, 0], np.ones(3))
    np.testing.assert_allclose(A[:, 0, 1], Z)
    np.testing.assert_allclose(A[:, 1, 0], np.zeros(3))
    np.testing.assert_allclose(A[:, 1, 1], np.ones(3))


def test_shunt_admittance_supports_frequency_arrays() -> None:
    Y = np.array([1e-3, 2e-3, 3e-3]) + 1j * np.array([1e-4, 2e-4, 3e-4])

    A = _shunt_abcd(Y)

    assert A.shape == (3, 2, 2)
    np.testing.assert_allclose(A[:, 0, 0], np.ones(3))
    np.testing.assert_allclose(A[:, 0, 1], np.zeros(3))
    np.testing.assert_allclose(A[:, 1, 0], Y)
    np.testing.assert_allclose(A[:, 1, 1], np.ones(3))


def test_abcd_to_s_supports_frequency_arrays() -> None:
    Z = np.array([1.0, 5.0, 10.0]) + 1j * np.array([0.0, 2.0, 4.0])
    A = _series_abcd(Z)

    S = _abcd_to_s(A, z0_ohm=50.0)

    assert S.shape == (3, 2, 2)
    assert np.all(np.isfinite(S.real))
    assert np.all(np.isfinite(S.imag))


def test_cascade_supports_frequency_arrays() -> None:
    Z = np.array([1.0, 2.0, 3.0]) + 1j * np.array([0.1, 0.2, 0.3])
    Y = np.array([1e-3, 2e-3, 3e-3]) + 1j * np.array([1e-4, 2e-4, 3e-4])

    A = _series_abcd(Z)
    B = _shunt_abcd(Y)
    total = _cascade_abcd(A, B)

    expected = np.einsum("...ij,...jk->...ik", A, B)

    assert total.shape == (3, 2, 2)
    np.testing.assert_allclose(total, expected, rtol=1e-13, atol=1e-13)


def test_invalid_abcd_shape_is_rejected_by_conversion() -> None:
    bad = np.ones((3, 3), dtype=np.complex128)

    with pytest.raises((ValueError, AssertionError, IndexError)):
        _abcd_to_s(bad, z0_ohm=50.0)


def test_invalid_reference_impedance_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _abcd_to_s(_identity_abcd(), z0_ohm=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _abcd_to_s(_identity_abcd(), z0_ohm=-50.0)


def test_nan_inputs_do_not_silently_become_finite() -> None:
    A = _series_abcd(np.nan)

    S = _abcd_to_s(A, z0_ohm=50.0)

    assert np.any(~np.isfinite(S))