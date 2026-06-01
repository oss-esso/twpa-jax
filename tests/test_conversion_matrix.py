"""
Tests for twpa.nonlinear.conversion.

These tests define the expected behavior of the harmonic-domain conversion /
mixing-matrix layer used by the small-signal gain calculation around a pumped
solution.

Core convention
---------------
Given harmonic coefficient vectors a_q and x_j,

    y_k = sum_j a_{k-j} x_j

so multiplication in time corresponds to convolution in harmonic space.

The conversion matrix M[a] is therefore

    M[k, j] = a_{k-j}

with rows indexed by output harmonic orders k and columns indexed by input
harmonic orders j.

The tests are API-tolerant in naming, but require the module to expose a public
way to build or apply this harmonic multiplication/conversion matrix.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import twpa.nonlinear.conversion as conv


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def _get_attr_or_key(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Cannot convert object to mapping: {type(obj)!r}")


def _validate_orders(orders: Sequence[int]) -> np.ndarray:
    arr = np.asarray(orders)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("orders must be a non-empty 1D sequence")
    if not np.all(np.equal(arr, arr.astype(int))):
        raise ValueError("orders must be integer-valued")
    arr = arr.astype(int)
    if len(set(arr.tolist())) != arr.size:
        raise ValueError("orders must be unique")
    return arr


def _coeff_mapping(coeffs: Any, coeff_orders: Sequence[int]) -> dict[int, complex]:
    orders = _validate_orders(coeff_orders)
    arr = np.asarray(coeffs, dtype=np.complex128)
    if arr.shape[0] != orders.size:
        raise ValueError("coefficient/order length mismatch")
    return {int(order): complex(value) for order, value in zip(orders, arr)}


def _manual_conversion_matrix(
    coeffs: Any,
    *,
    coeff_orders: Sequence[int],
    input_orders: Sequence[int],
    output_orders: Sequence[int],
) -> np.ndarray:
    coeff_map = _coeff_mapping(coeffs, coeff_orders)
    in_orders = _validate_orders(input_orders)
    out_orders = _validate_orders(output_orders)

    M = np.zeros((out_orders.size, in_orders.size), dtype=np.complex128)

    for row, k in enumerate(out_orders):
        for col, j in enumerate(in_orders):
            M[row, col] = coeff_map.get(int(k - j), 0.0 + 0.0j)

    return M


def _manual_apply(
    coeffs: Any,
    x: Any,
    *,
    coeff_orders: Sequence[int],
    input_orders: Sequence[int],
    output_orders: Sequence[int],
) -> np.ndarray:
    M = _manual_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )
    return M @ np.asarray(x, dtype=np.complex128)


def _build_conversion_matrix(
    coeffs: Any,
    *,
    coeff_orders: Sequence[int],
    input_orders: Sequence[int],
    output_orders: Sequence[int] | None = None,
) -> np.ndarray:
    if output_orders is None:
        output_orders = input_orders

    for name in [
        "conversion_matrix",
        "build_conversion_matrix",
        "harmonic_conversion_matrix",
        "mixing_matrix",
        "build_mixing_matrix",
        "convolution_matrix",
        "coefficient_convolution_matrix",
        "harmonic_multiplication_matrix",
        "multiplication_matrix",
        "toeplitz_harmonic_matrix",
    ]:
        if hasattr(conv, name):
            out = _call_with_supported_kwargs(
                getattr(conv, name),
                coeffs=coeffs,
                coefficients=coeffs,
                kernel_coeffs=coeffs,
                pump_coeffs=coeffs,
                a=coeffs,
                input_orders=np.asarray(input_orders),
                in_orders=np.asarray(input_orders),
                source_orders=np.asarray(input_orders),
                output_orders=np.asarray(output_orders),
                out_orders=np.asarray(output_orders),
                target_orders=np.asarray(output_orders),
                coeff_orders=np.asarray(coeff_orders),
                coefficient_orders=np.asarray(coeff_orders),
                kernel_orders=np.asarray(coeff_orders),
                pump_orders=np.asarray(coeff_orders),
                orders=np.asarray(input_orders),
                harmonic_orders=np.asarray(input_orders),
            )
            value = _get_attr_or_key(out, "matrix", "M", "conversion_matrix", "mixing_matrix", default=out)
            return np.asarray(value, dtype=np.complex128)

    raise AttributeError(
        "twpa.nonlinear.conversion must expose a conversion/multiplication "
        "matrix builder such as conversion_matrix, convolution_matrix, or "
        "harmonic_multiplication_matrix."
    )


def _apply_conversion(
    coeffs: Any,
    x: Any,
    *,
    coeff_orders: Sequence[int],
    input_orders: Sequence[int],
    output_orders: Sequence[int] | None = None,
) -> np.ndarray:
    if output_orders is None:
        output_orders = input_orders

    for name in [
        "apply_conversion_matrix",
        "apply_mixing_matrix",
        "apply_harmonic_multiplication",
        "multiply_harmonics",
        "convolve_coefficients",
        "harmonic_convolution",
        "apply_convolution_matrix",
    ]:
        if hasattr(conv, name):
            out = _call_with_supported_kwargs(
                getattr(conv, name),
                coeffs=coeffs,
                coefficients=coeffs,
                kernel_coeffs=coeffs,
                pump_coeffs=coeffs,
                x=x,
                vector=x,
                input_coeffs=x,
                signal_coeffs=x,
                coeff_orders=np.asarray(coeff_orders),
                coefficient_orders=np.asarray(coeff_orders),
                kernel_orders=np.asarray(coeff_orders),
                input_orders=np.asarray(input_orders),
                in_orders=np.asarray(input_orders),
                output_orders=np.asarray(output_orders),
                out_orders=np.asarray(output_orders),
                orders=np.asarray(input_orders),
                harmonic_orders=np.asarray(input_orders),
            )
            value = _get_attr_or_key(out, "coeffs", "coefficients", "y", "output", default=out)
            return np.asarray(value, dtype=np.complex128)

    M = _build_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )
    return M @ np.asarray(x, dtype=np.complex128)


def _make_conversion_problem(
    *,
    coeffs: Any,
    coeff_orders: Sequence[int],
    input_orders: Sequence[int],
    output_orders: Sequence[int] | None = None,
) -> Any:
    if output_orders is None:
        output_orders = input_orders

    for name in [
        "ConversionMatrix",
        "HarmonicConversionMatrix",
        "MixingMatrix",
        "HarmonicMultiplicationMatrix",
        "ConversionProblem",
    ]:
        if hasattr(conv, name):
            cls = getattr(conv, name)
            return _call_with_supported_kwargs(
                cls,
                coeffs=coeffs,
                coefficients=coeffs,
                kernel_coeffs=coeffs,
                input_orders=tuple(input_orders),
                output_orders=tuple(output_orders),
                coeff_orders=tuple(coeff_orders),
                coefficient_orders=tuple(coeff_orders),
                orders=tuple(input_orders),
                harmonic_orders=tuple(input_orders),
            )

    for name in [
        "make_conversion_matrix",
        "make_mixing_matrix",
        "make_harmonic_multiplication_matrix",
        "make_conversion_problem",
    ]:
        if hasattr(conv, name):
            fn = getattr(conv, name)
            return _call_with_supported_kwargs(
                fn,
                coeffs=coeffs,
                coefficients=coeffs,
                kernel_coeffs=coeffs,
                input_orders=tuple(input_orders),
                output_orders=tuple(output_orders),
                coeff_orders=tuple(coeff_orders),
                coefficient_orders=tuple(coeff_orders),
                orders=tuple(input_orders),
                harmonic_orders=tuple(input_orders),
            )

    pytest.skip("Object-oriented conversion matrix wrapper is optional.")


def test_conversion_matrix_matches_manual_convolution() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1 - 0.2j, 2.0 + 0.0j, 0.3 + 0.4j])
    orders = np.array([-3, -1, 1, 3])

    M = _build_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=orders,
        output_orders=orders,
    )
    expected = _manual_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=orders,
        output_orders=orders,
    )

    assert M.shape == (orders.size, orders.size)
    np.testing.assert_allclose(M, expected, rtol=1e-14, atol=1e-14)


def test_apply_conversion_matches_manual_matrix_product() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1 - 0.2j, 2.0 + 0.0j, 0.3 + 0.4j])
    orders = np.array([-3, -1, 1, 3])
    x = np.array([0.5 + 0.1j, -1.0 + 0.2j, 0.4 - 0.3j, 0.1 + 0.0j])

    y = _apply_conversion(
        coeffs,
        x,
        coeff_orders=coeff_orders,
        input_orders=orders,
        output_orders=orders,
    )
    expected = _manual_apply(
        coeffs,
        x,
        coeff_orders=coeff_orders,
        input_orders=orders,
        output_orders=orders,
    )

    np.testing.assert_allclose(y, expected, rtol=1e-14, atol=1e-14)


def test_dc_kernel_is_scalar_identity() -> None:
    coeff_orders = np.array([0])
    coeffs = np.array([3.0 - 0.5j])
    orders = np.array([-3, -1, 1, 3])

    M = _build_conversion_matrix(coeffs, coeff_orders=coeff_orders, input_orders=orders)

    expected = (3.0 - 0.5j) * np.eye(orders.size, dtype=np.complex128)

    np.testing.assert_allclose(M, expected, rtol=1e-14, atol=1e-14)


def test_zero_kernel_gives_zero_matrix() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.zeros(3, dtype=np.complex128)
    orders = np.array([-3, -1, 1, 3])

    M = _build_conversion_matrix(coeffs, coeff_orders=coeff_orders, input_orders=orders)

    np.testing.assert_allclose(M, np.zeros((4, 4), dtype=np.complex128), atol=0.0, rtol=0.0)


def test_asymmetric_input_output_orders_are_supported() -> None:
    coeff_orders = np.array([-4, -2, 0, 2, 4])
    coeffs = np.array([0.02, 0.1, 1.0, -0.2, 0.03], dtype=np.complex128)
    input_orders = np.array([-1, 1])
    output_orders = np.array([-5, -3, -1, 1, 3, 5])

    M = _build_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )
    expected = _manual_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )

    assert M.shape == (output_orders.size, input_orders.size)
    np.testing.assert_allclose(M, expected, rtol=1e-14, atol=1e-14)


def test_frequency_shift_kernel_moves_coefficients_up() -> None:
    coeff_orders = np.array([2])
    coeffs = np.array([1.0 + 0.0j])
    input_orders = np.array([-3, -1, 1, 3])
    output_orders = np.array([-1, 1, 3, 5])

    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.complex128)

    y = _apply_conversion(
        coeffs,
        x,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )

    np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)


def test_frequency_shift_kernel_moves_coefficients_down() -> None:
    coeff_orders = np.array([-2])
    coeffs = np.array([1.0 + 0.0j])
    input_orders = np.array([-3, -1, 1, 3])
    output_orders = np.array([-5, -3, -1, 1])

    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.complex128)

    y = _apply_conversion(
        coeffs,
        x,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )

    np.testing.assert_allclose(y, x, rtol=0.0, atol=0.0)


def test_missing_kernel_orders_contribute_zero() -> None:
    coeff_orders = np.array([0])
    coeffs = np.array([1.0 + 0.0j])
    input_orders = np.array([-3, -1, 1, 3])
    output_orders = np.array([-5, -3, -1, 1, 3, 5])

    M = _build_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )

    expected = _manual_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=input_orders,
        output_orders=output_orders,
    )

    np.testing.assert_allclose(M, expected, rtol=0.0, atol=0.0)
    assert np.all(M[0] == 0.0)
    assert np.all(M[-1] == 0.0)


def test_conversion_matrix_is_linear_in_kernel_coefficients() -> None:
    coeff_orders = np.array([-2, 0, 2])
    a = np.array([0.1, 1.0, 0.2], dtype=np.complex128)
    b = np.array([-0.3j, 0.5, 0.7j], dtype=np.complex128)
    orders = np.array([-3, -1, 1, 3])

    Ma = _build_conversion_matrix(a, coeff_orders=coeff_orders, input_orders=orders)
    Mb = _build_conversion_matrix(b, coeff_orders=coeff_orders, input_orders=orders)
    Mab = _build_conversion_matrix(a + b, coeff_orders=coeff_orders, input_orders=orders)

    np.testing.assert_allclose(Mab, Ma + Mb, rtol=1e-14, atol=1e-14)


def test_conversion_application_is_linear_in_input_vector() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1, 1.0, 0.2], dtype=np.complex128)
    orders = np.array([-3, -1, 1, 3])
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)
    z = np.array([-0.2j, 0.4, -0.1, 0.3j], dtype=np.complex128)

    y_x = _apply_conversion(coeffs, x, coeff_orders=coeff_orders, input_orders=orders)
    y_z = _apply_conversion(coeffs, z, coeff_orders=coeff_orders, input_orders=orders)
    y_sum = _apply_conversion(coeffs, x + z, coeff_orders=coeff_orders, input_orders=orders)

    np.testing.assert_allclose(y_sum, y_x + y_z, rtol=1e-14, atol=1e-14)


def test_real_time_kernel_has_hermitian_toeplitz_symmetry() -> None:
    coeff_orders = np.array([-2, 0, 2])
    a2 = 0.3 + 0.4j
    coeffs = np.array([np.conj(a2), 1.5 + 0.0j, a2])
    orders = np.array([-3, -1, 1, 3])

    M = _build_conversion_matrix(coeffs, coeff_orders=coeff_orders, input_orders=orders)

    np.testing.assert_allclose(M, M.conj().T, rtol=1e-14, atol=1e-14)


def test_conjugate_symmetric_input_and_real_kernel_give_conjugate_symmetric_output() -> None:
    coeff_orders = np.array([-2, 0, 2])
    a2 = 0.1 - 0.2j
    coeffs = np.array([np.conj(a2), 1.0 + 0.0j, a2])
    orders = np.array([-3, -1, 1, 3])

    x = np.array(
        [
            0.1 - 0.03j,
            0.8 + 0.2j,
            0.8 - 0.2j,
            0.1 + 0.03j,
        ],
        dtype=np.complex128,
    )

    y = _apply_conversion(coeffs, x, coeff_orders=coeff_orders, input_orders=orders)

    assert y[0] == pytest.approx(np.conj(y[3]), rel=1e-13, abs=1e-13)
    assert y[1] == pytest.approx(np.conj(y[2]), rel=1e-13, abs=1e-13)


def test_matrix_matches_direct_time_domain_projection() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.15 + 0.05j, 1.0, 0.15 - 0.05j], dtype=np.complex128)
    orders = np.array([-3, -1, 1, 3])
    x = np.array([0.1 - 0.2j, 0.7 + 0.3j, 0.7 - 0.3j, 0.1 + 0.2j]) * 1e-3

    y = _apply_conversion(coeffs, x, coeff_orders=coeff_orders, input_orders=orders)

    n_time = 512
    t = np.arange(n_time) / n_time
    a_t = np.exp(1j * 2.0 * np.pi * np.outer(t, coeff_orders)) @ coeffs
    x_t = np.exp(1j * 2.0 * np.pi * np.outer(t, orders)) @ x
    y_t = a_t * x_t
    expected = np.exp(-1j * 2.0 * np.pi * np.outer(t, orders)).T @ y_t / n_time

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-15)


def test_two_stage_multiplication_matches_kernel_convolution_when_basis_is_large_enough() -> None:
    orders = np.arange(-6, 7)
    a_orders = np.array([-2, 0, 2])
    b_orders = np.array([-2, 0, 2])
    a = np.array([0.1, 1.0, 0.2], dtype=np.complex128)
    b = np.array([-0.3j, 0.5, 0.4j], dtype=np.complex128)
    x = np.zeros(orders.size, dtype=np.complex128)
    x[orders == -1] = 1.0 + 0.2j
    x[orders == 1] = 0.5 - 0.1j

    y_two_stage = _apply_conversion(
        a,
        _apply_conversion(b, x, coeff_orders=b_orders, input_orders=orders),
        coeff_orders=a_orders,
        input_orders=orders,
    )

    ab_orders = np.arange(-4, 5, 2)
    ab = np.zeros(ab_orders.size, dtype=np.complex128)
    b_map = _coeff_mapping(b, b_orders)
    for idx, q in enumerate(ab_orders):
        total = 0.0 + 0.0j
        for p, a_p in _coeff_mapping(a, a_orders).items():
            total += a_p * b_map.get(int(q - p), 0.0 + 0.0j)
        ab[idx] = total

    y_single_stage = _apply_conversion(ab, x, coeff_orders=ab_orders, input_orders=orders)

    np.testing.assert_allclose(y_two_stage, y_single_stage, rtol=1e-12, atol=1e-12)


def test_block_diagonal_conversion_matrix_if_available() -> None:
    helper = None
    for name in [
        "block_diagonal_conversion_matrix",
        "make_block_conversion_matrix",
        "conversion_matrix_for_branches",
        "batched_conversion_matrix",
    ]:
        if hasattr(conv, name):
            helper = getattr(conv, name)
            break

    if helper is None:
        pytest.skip("Block/batched conversion matrix helper is optional.")

    coeff_orders = np.array([-2, 0, 2])
    orders = np.array([-3, -1, 1, 3])
    coeffs = np.array(
        [
            [0.1, 1.0, 0.2],
            [0.2, 0.8, 0.1],
        ],
        dtype=np.complex128,
    )

    B = np.asarray(
        _call_with_supported_kwargs(
            helper,
            coeffs=coeffs,
            coefficients=coeffs,
            coeff_orders=coeff_orders,
            coefficient_orders=coeff_orders,
            input_orders=orders,
            output_orders=orders,
            orders=orders,
            harmonic_orders=orders,
        ),
        dtype=np.complex128,
    )

    assert B.shape == (2 * orders.size, 2 * orders.size)

    expected0 = _manual_conversion_matrix(coeffs[0], coeff_orders=coeff_orders, input_orders=orders, output_orders=orders)
    expected1 = _manual_conversion_matrix(coeffs[1], coeff_orders=coeff_orders, input_orders=orders, output_orders=orders)

    np.testing.assert_allclose(B[:4, :4], expected0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(B[4:, 4:], expected1, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(B[:4, 4:], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(B[4:, :4], 0.0, rtol=0.0, atol=0.0)


def test_signal_idler_pair_matrix_if_available() -> None:
    helper = None
    for name in [
        "signal_idler_conversion_matrix",
        "make_signal_idler_matrix",
        "four_wave_mixing_matrix",
        "small_signal_conversion_matrix",
    ]:
        if hasattr(conv, name):
            helper = getattr(conv, name)
            break

    if helper is None:
        pytest.skip("Signal-idler reduced conversion matrix is optional.")

    coupling = 0.1 + 0.2j
    detuning = 0.03
    loss = 0.01

    M = np.asarray(
        _call_with_supported_kwargs(
            helper,
            coupling=coupling,
            kappa=coupling,
            g=coupling,
            detuning=detuning,
            delta=detuning,
            loss=loss,
            alpha=loss,
        ),
        dtype=np.complex128,
    )

    assert M.shape[-2:] == (2, 2)
    assert np.all(np.isfinite(M.real))
    assert np.all(np.isfinite(M.imag))


def test_conversion_problem_object_if_available_is_mapping_like() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1, 1.0, 0.2], dtype=np.complex128)
    orders = np.array([-3, -1, 1, 3])

    obj = _make_conversion_problem(coeffs=coeffs, coeff_orders=coeff_orders, input_orders=orders)
    mapping = _as_mapping(obj)

    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0


def test_conversion_problem_object_can_apply_if_available() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1, 1.0, 0.2], dtype=np.complex128)
    orders = np.array([-3, -1, 1, 3])
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)

    obj = _make_conversion_problem(coeffs=coeffs, coeff_orders=coeff_orders, input_orders=orders)

    method = None
    for method_name in ["apply", "matvec", "multiply", "convolve"]:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            break

    if method is None:
        pytest.skip("Conversion object has no apply/matvec method.")

    y = np.asarray(
        _call_with_supported_kwargs(
            method,
            x=x,
            vector=x,
            input_coeffs=x,
        ),
        dtype=np.complex128,
    )
    expected = _manual_apply(coeffs, x, coeff_orders=coeff_orders, input_orders=orders, output_orders=orders)

    np.testing.assert_allclose(y, expected, rtol=1e-14, atol=1e-14)


def test_conversion_matrix_supports_real_float_inputs() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1, 1.0, 0.2])
    orders = np.array([-3, -1, 1, 3])
    x = np.array([1.0, 2.0, 3.0, 4.0])

    y = _apply_conversion(coeffs, x, coeff_orders=coeff_orders, input_orders=orders)
    expected = _manual_apply(coeffs, x, coeff_orders=coeff_orders, input_orders=orders, output_orders=orders)

    np.testing.assert_allclose(y, expected, rtol=1e-14, atol=1e-14)


def test_batch_vector_application_if_supported() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1, 1.0, 0.2], dtype=np.complex128)
    orders = np.array([-3, -1, 1, 3])
    X = np.array(
        [
            [1.0, 0.2],
            [2.0, 0.3],
            [3.0, 0.4],
            [4.0, 0.5],
        ],
        dtype=np.complex128,
    )

    try:
        Y = _apply_conversion(coeffs, X, coeff_orders=coeff_orders, input_orders=orders)
    except Exception as exc:
        pytest.skip(f"Batch vector application is optional: {type(exc).__name__}: {exc}")

    expected = _manual_conversion_matrix(
        coeffs,
        coeff_orders=coeff_orders,
        input_orders=orders,
        output_orders=orders,
    ) @ X

    assert Y.shape == X.shape
    np.testing.assert_allclose(Y, expected, rtol=1e-14, atol=1e-14)


def test_invalid_duplicate_input_orders_are_rejected() -> None:
    coeff_orders = np.array([0])
    coeffs = np.array([1.0])

    with pytest.raises((ValueError, AssertionError)):
        _build_conversion_matrix(coeffs, coeff_orders=coeff_orders, input_orders=[-1, -1])


def test_invalid_duplicate_coeff_orders_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _build_conversion_matrix(
            np.array([1.0, 2.0]),
            coeff_orders=[0, 0],
            input_orders=[-1, 1],
        )


def test_invalid_noninteger_orders_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, TypeError)):
        _build_conversion_matrix(
            np.array([1.0]),
            coeff_orders=[0],
            input_orders=[-1.5, 1.0],
        )

    with pytest.raises((ValueError, AssertionError, TypeError)):
        _build_conversion_matrix(
            np.array([1.0]),
            coeff_orders=[0.5],
            input_orders=[-1, 1],
        )


def test_coeff_length_mismatch_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _build_conversion_matrix(
            np.array([1.0, 2.0, 3.0]),
            coeff_orders=[-2, 0],
            input_orders=[-1, 1],
        )


def test_input_vector_length_mismatch_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _apply_conversion(
            np.array([1.0]),
            np.array([1.0, 2.0, 3.0]),
            coeff_orders=[0],
            input_orders=[-1, 1],
        )


def test_empty_orders_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _build_conversion_matrix(np.array([1.0]), coeff_orders=[0], input_orders=[])

    with pytest.raises((ValueError, AssertionError, IndexError)):
        _build_conversion_matrix(np.array([]), coeff_orders=[], input_orders=[-1, 1])


def test_nan_kernel_propagates_or_is_rejected() -> None:
    try:
        M = _build_conversion_matrix(
            np.array([np.nan]),
            coeff_orders=[0],
            input_orders=[-1, 1],
        )
    except (ValueError, AssertionError, FloatingPointError):
        return

    assert np.any(~np.isfinite(M))


def test_nan_input_vector_propagates_or_is_rejected() -> None:
    try:
        y = _apply_conversion(
            np.array([1.0]),
            np.array([np.nan, 1.0]),
            coeff_orders=[0],
            input_orders=[-1, 1],
        )
    except (ValueError, AssertionError, FloatingPointError):
        return

    assert np.any(~np.isfinite(y))


def test_large_basis_matrix_shape_and_finiteness() -> None:
    coeff_orders = np.arange(-10, 11, 2)
    coeffs = np.exp(-0.1 * np.abs(coeff_orders)).astype(np.complex128)
    orders = np.arange(-51, 52, 2)

    M = _build_conversion_matrix(coeffs, coeff_orders=coeff_orders, input_orders=orders)

    assert M.shape == (orders.size, orders.size)
    assert np.all(np.isfinite(M.real))
    assert np.all(np.isfinite(M.imag))


def test_conversion_matrix_is_json_serializable_through_summary() -> None:
    coeff_orders = np.array([-2, 0, 2])
    coeffs = np.array([0.1 + 0.2j, 1.0, 0.3 - 0.1j])
    orders = np.array([-3, -1, 1, 3])

    M = _build_conversion_matrix(coeffs, coeff_orders=coeff_orders, input_orders=orders)

    summary = {
        "shape": M.shape,
        "real": np.real(M).tolist(),
        "imag": np.imag(M).tolist(),
        "input_orders": orders.tolist(),
        "coeff_orders": coeff_orders.tolist(),
    }

    import json

    json.dumps(summary)
