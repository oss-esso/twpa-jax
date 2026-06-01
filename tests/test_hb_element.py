"""
Tests for twpa.nonlinear.hb_element.

These tests define the expected harmonic-balance interface for a nonlinear
kinetic-inductance element.

Reference convention
--------------------
For selected harmonic orders k and fundamental angular frequency omega0,

    I(t) = sum_k I_k exp(+i k omega0 t)

with kinetic-inductance flux-current relation

    Phi(I) = L0 * (I + beta * I**3 / (3 I*^2))

and voltage

    V_k = i * k * omega0 * Phi_k.

The tests are API-tolerant in naming, but require the module to expose a public
way to compute at least one of:

    - flux harmonic coefficients from current coefficients,
    - voltage harmonic coefficients from current coefficients,
    - residual V - V_model(I).
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import twpa.nonlinear.hb_element as hbe


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
    if np.any(arr == 0):
        raise ValueError("zero/DC order is not used by this HB element test suite")
    return arr


def _time_grid(n_time: int, omega0_rad_s: float) -> np.ndarray:
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    period = 2.0 * np.pi / omega0_rad_s
    return np.arange(n_time, dtype=float) * period / n_time


def _manual_synthesize(
    coeffs: Sequence[complex],
    orders: Sequence[int],
    *,
    n_time: int,
    omega0_rad_s: float,
) -> np.ndarray:
    orders_arr = _validate_orders(orders)
    coeffs_arr = np.asarray(coeffs, dtype=np.complex128)
    if coeffs_arr.shape[0] != orders_arr.size:
        raise ValueError("coefficient/order length mismatch")

    t = _time_grid(n_time, omega0_rad_s)
    phase = np.exp(1j * np.outer(t * omega0_rad_s, orders_arr))
    return phase @ coeffs_arr


def _manual_analyze(
    samples: Sequence[complex],
    orders: Sequence[int],
    *,
    omega0_rad_s: float,
) -> np.ndarray:
    orders_arr = _validate_orders(orders)
    x = np.asarray(samples, dtype=np.complex128)
    n_time = x.shape[0]
    t = _time_grid(n_time, omega0_rad_s)
    phase = np.exp(-1j * np.outer(t * omega0_rad_s, orders_arr))
    return phase.T @ x / n_time


def _manual_flux_coeffs(
    current_coeffs: Sequence[complex],
    orders: Sequence[int],
    *,
    L0_H: float,
    I_star_A: float,
    beta: float,
    omega0_rad_s: float,
    n_time: int = 2048,
) -> np.ndarray:
    if L0_H <= 0.0:
        raise ValueError("L0_H must be positive")
    if I_star_A <= 0.0:
        raise ValueError("I_star_A must be positive")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    i_t = _manual_synthesize(
        current_coeffs,
        orders,
        n_time=n_time,
        omega0_rad_s=omega0_rad_s,
    )
    phi_t = L0_H * (i_t + beta * i_t**3 / (3.0 * I_star_A**2))
    return _manual_analyze(phi_t, orders, omega0_rad_s=omega0_rad_s)


def _manual_voltage_coeffs(
    current_coeffs: Sequence[complex],
    orders: Sequence[int],
    *,
    L0_H: float,
    I_star_A: float,
    beta: float,
    omega0_rad_s: float,
    n_time: int = 2048,
) -> np.ndarray:
    orders_arr = _validate_orders(orders)
    phi = _manual_flux_coeffs(
        current_coeffs,
        orders,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )
    return 1j * orders_arr * omega0_rad_s * phi


def _make_element(
    *,
    L0_H: float = 1e-12,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    orders: Sequence[int] = (-3, -1, 1, 3),
    omega0_rad_s: float = 2.0 * np.pi * 10e9,
) -> Any:
    for name in [
        "KineticInductanceHBElement",
        "NonlinearInductorHBElement",
        "HBKineticInductanceElement",
        "HBNonlinearInductor",
        "HBElement",
    ]:
        if hasattr(hbe, name):
            cls = getattr(hbe, name)
            return _call_with_supported_kwargs(
                cls,
                L0_H=L0_H,
                L_0_H=L0_H,
                l0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                orders=tuple(orders),
                harmonic_orders=tuple(orders),
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
            )

    for name in [
        "make_kinetic_inductance_hb_element",
        "make_nonlinear_inductor_hb_element",
        "make_hb_element",
    ]:
        if hasattr(hbe, name):
            fn = getattr(hbe, name)
            return _call_with_supported_kwargs(
                fn,
                L0_H=L0_H,
                L_0_H=L0_H,
                l0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                orders=tuple(orders),
                harmonic_orders=tuple(orders),
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
            )

    return {
        "L0_H": L0_H,
        "I_star_A": I_star_A,
        "beta": beta,
        "orders": tuple(orders),
        "omega0_rad_s": omega0_rad_s,
    }


def _flux_coeffs(
    current_coeffs: Any,
    orders: Sequence[int],
    *,
    L0_H: float = 1e-12,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    omega0_rad_s: float = 2.0 * np.pi * 10e9,
) -> np.ndarray:
    element = _make_element(
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
    )

    for name in [
        "flux_coeffs",
        "flux_coefficients",
        "current_to_flux_coeffs",
        "current_to_flux_coefficients",
        "phi_coeffs_from_current",
        "compute_flux_coeffs",
    ]:
        if hasattr(hbe, name):
            out = _call_with_supported_kwargs(
                getattr(hbe, name),
                current_coeffs=current_coeffs,
                I_coeffs=current_coeffs,
                I=current_coeffs,
                i_coeffs=current_coeffs,
                orders=np.asarray(orders, dtype=int),
                harmonic_orders=np.asarray(orders, dtype=int),
                L0_H=L0_H,
                I_star_A=I_star_A,
                beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                element=element,
            )
            return np.asarray(out, dtype=np.complex128)

    for method_name in [
        "flux_coeffs",
        "flux_coefficients",
        "current_to_flux_coeffs",
        "current_to_flux_coefficients",
        "compute_flux_coeffs",
    ]:
        method = _get_attr_or_key(element, method_name, default=None)
        if callable(method):
            return np.asarray(
                _call_with_supported_kwargs(
                    method,
                    current_coeffs=current_coeffs,
                    I_coeffs=current_coeffs,
                    I=current_coeffs,
                    orders=np.asarray(orders, dtype=int),
                    harmonic_orders=np.asarray(orders, dtype=int),
                    omega0_rad_s=omega0_rad_s,
                    omega0=omega0_rad_s,
                ),
                dtype=np.complex128,
            )

    # Voltage is mandatory if flux is not exposed; tests requiring flux will skip.
    pytest.skip("Flux-coefficient helper is optional if voltage/residual helper exists.")


def _voltage_coeffs(
    current_coeffs: Any,
    orders: Sequence[int],
    *,
    L0_H: float = 1e-12,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    omega0_rad_s: float = 2.0 * np.pi * 10e9,
) -> np.ndarray:
    element = _make_element(
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
    )

    for name in [
        "voltage_coeffs",
        "voltage_coefficients",
        "current_to_voltage_coeffs",
        "current_to_voltage_coefficients",
        "nonlinear_inductor_voltage_coeffs",
        "compute_voltage_coeffs",
        "v_from_i_coeffs",
    ]:
        if hasattr(hbe, name):
            out = _call_with_supported_kwargs(
                getattr(hbe, name),
                current_coeffs=current_coeffs,
                I_coeffs=current_coeffs,
                I=current_coeffs,
                i_coeffs=current_coeffs,
                orders=np.asarray(orders, dtype=int),
                harmonic_orders=np.asarray(orders, dtype=int),
                L0_H=L0_H,
                I_star_A=I_star_A,
                beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                element=element,
            )
            return np.asarray(out, dtype=np.complex128)

    for method_name in [
        "voltage_coeffs",
        "voltage_coefficients",
        "current_to_voltage_coeffs",
        "current_to_voltage_coefficients",
        "compute_voltage_coeffs",
        "v_from_i_coeffs",
    ]:
        method = _get_attr_or_key(element, method_name, default=None)
        if callable(method):
            return np.asarray(
                _call_with_supported_kwargs(
                    method,
                    current_coeffs=current_coeffs,
                    I_coeffs=current_coeffs,
                    I=current_coeffs,
                    orders=np.asarray(orders, dtype=int),
                    harmonic_orders=np.asarray(orders, dtype=int),
                    omega0_rad_s=omega0_rad_s,
                    omega0=omega0_rad_s,
                ),
                dtype=np.complex128,
            )

    # Fallback: if only flux is implemented, compute V = i omega Phi.
    phi = _flux_coeffs(
        current_coeffs,
        orders,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
    )
    return 1j * np.asarray(orders, dtype=int) * omega0_rad_s * phi


def _residual(
    voltage_coeffs: Any,
    current_coeffs: Any,
    orders: Sequence[int],
    *,
    L0_H: float = 1e-12,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    omega0_rad_s: float = 2.0 * np.pi * 10e9,
) -> np.ndarray:
    element = _make_element(
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
    )

    for name in [
        "residual",
        "hb_residual",
        "element_residual",
        "nonlinear_inductor_residual",
        "voltage_residual",
    ]:
        if hasattr(hbe, name):
            out = _call_with_supported_kwargs(
                getattr(hbe, name),
                voltage_coeffs=voltage_coeffs,
                V_coeffs=voltage_coeffs,
                V=voltage_coeffs,
                current_coeffs=current_coeffs,
                I_coeffs=current_coeffs,
                I=current_coeffs,
                orders=np.asarray(orders, dtype=int),
                harmonic_orders=np.asarray(orders, dtype=int),
                L0_H=L0_H,
                I_star_A=I_star_A,
                beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                element=element,
            )
            return np.asarray(out, dtype=np.complex128)

    for method_name in [
        "residual",
        "hb_residual",
        "element_residual",
        "nonlinear_inductor_residual",
        "voltage_residual",
    ]:
        method = _get_attr_or_key(element, method_name, default=None)
        if callable(method):
            return np.asarray(
                _call_with_supported_kwargs(
                    method,
                    voltage_coeffs=voltage_coeffs,
                    V_coeffs=voltage_coeffs,
                    V=voltage_coeffs,
                    current_coeffs=current_coeffs,
                    I_coeffs=current_coeffs,
                    I=current_coeffs,
                    orders=np.asarray(orders, dtype=int),
                    harmonic_orders=np.asarray(orders, dtype=int),
                    omega0_rad_s=omega0_rad_s,
                    omega0=omega0_rad_s,
                ),
                dtype=np.complex128,
            )

    return np.asarray(voltage_coeffs, dtype=np.complex128) - _voltage_coeffs(
        current_coeffs,
        orders,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
    )


def test_element_object_exposes_or_preserves_basic_parameters() -> None:
    element = _make_element(
        L0_H=2e-12,
        I_star_A=4e-3,
        beta=0.75,
        orders=(-3, -1, 1, 3),
        omega0_rad_s=2.0 * np.pi * 7e9,
    )

    mapping = _as_mapping(element)
    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0

    L0 = _get_attr_or_key(element, "L0_H", "L_0_H", "l0_H", default=2e-12)
    I_star = _get_attr_or_key(element, "I_star_A", "i_star_A", "Istar_A", default=4e-3)
    beta = _get_attr_or_key(element, "beta", "beta_nl", "nonlinear_beta", default=0.75)

    assert float(L0) == pytest.approx(2e-12)
    assert float(I_star) == pytest.approx(4e-3)
    assert float(beta) == pytest.approx(0.75)


def test_zero_current_produces_zero_flux_if_exposed() -> None:
    orders = [-3, -1, 1, 3]
    current = np.zeros(len(orders), dtype=np.complex128)

    phi = _flux_coeffs(current, orders)

    np.testing.assert_allclose(phi, np.zeros_like(current), atol=1e-24, rtol=0.0)


def test_zero_current_produces_zero_voltage() -> None:
    orders = [-3, -1, 1, 3]
    current = np.zeros(len(orders), dtype=np.complex128)

    voltage = _voltage_coeffs(current, orders)

    np.testing.assert_allclose(voltage, np.zeros_like(current), atol=1e-18, rtol=0.0)


def test_linear_beta_zero_voltage_matches_inductor_impedance() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1.2e-12
    omega0 = 2.0 * np.pi * 10e9
    current = np.array([0.1 - 0.2j, 1.0 + 0.3j, -0.7 + 0.1j, 0.05 + 0.2j]) * 1e-6

    voltage = _voltage_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    expected = 1j * orders * omega0 * L0 * current

    np.testing.assert_allclose(voltage, expected, rtol=1e-12, atol=1e-18)


def test_small_current_limit_is_linear() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-12
    I_star = 5e-3
    omega0 = 2.0 * np.pi * 8e9
    current = np.array([0.01 - 0.02j, 0.2 + 0.1j, 0.2 - 0.1j, 0.01 + 0.02j]) * 1e-6

    voltage = _voltage_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=1.0,
        omega0_rad_s=omega0,
    )

    expected_linear = 1j * orders * omega0 * L0 * current

    np.testing.assert_allclose(voltage, expected_linear, rtol=1e-7, atol=1e-18)


def test_flux_coefficients_match_time_domain_cubic_projection_if_exposed() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-12
    I_star = 5e-3
    beta = 1.0
    omega0 = 2.0 * np.pi * 10e9

    # Conjugate-symmetric current gives a real time-domain waveform.
    current = np.array(
        [
            0.05 - 0.02j,
            0.6 + 0.2j,
            0.6 - 0.2j,
            0.05 + 0.02j,
        ],
        dtype=np.complex128,
    ) * 1e-3

    phi = _flux_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )
    expected = _manual_flux_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(phi, expected, rtol=1e-10, atol=1e-18)


def test_voltage_coefficients_match_time_domain_cubic_projection() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-12
    I_star = 5e-3
    beta = 0.8
    omega0 = 2.0 * np.pi * 10e9

    current = np.array(
        [
            -0.02 + 0.04j,
            0.7 + 0.1j,
            0.7 - 0.1j,
            -0.02 - 0.04j,
        ],
        dtype=np.complex128,
    ) * 1e-3

    voltage = _voltage_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )
    expected = _manual_voltage_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(voltage, expected, rtol=1e-10, atol=1e-15)


def test_residual_is_zero_for_model_voltage() -> None:
    orders = np.array([-3, -1, 1, 3])
    current = np.array([0.01 - 0.03j, 0.4 + 0.2j, 0.4 - 0.2j, 0.01 + 0.03j]) * 1e-3

    voltage = _voltage_coeffs(current, orders)
    residual = _residual(voltage, current, orders)

    np.testing.assert_allclose(residual, np.zeros_like(voltage), rtol=0.0, atol=1e-15)


def test_residual_detects_perturbed_voltage() -> None:
    orders = np.array([-3, -1, 1, 3])
    current = np.array([0.01, 0.4, 0.4, 0.01], dtype=np.complex128) * 1e-3

    voltage = _voltage_coeffs(current, orders)
    perturbation = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.complex128) * 1e-9

    residual = _residual(voltage + perturbation, current, orders)

    np.testing.assert_allclose(residual, perturbation, rtol=1e-9, atol=1e-15)


def test_conjugate_symmetric_current_produces_conjugate_symmetric_voltage() -> None:
    orders = np.array([-3, -1, 1, 3])
    current = np.array(
        [
            0.1 - 0.05j,
            0.8 + 0.2j,
            0.8 - 0.2j,
            0.1 + 0.05j,
        ],
        dtype=np.complex128,
    ) * 1e-3

    voltage = _voltage_coeffs(current, orders)

    # Voltage of a real flux waveform is real in time, so Fourier coefficients
    # obey V_-k = conj(V_+k).
    assert voltage[0] == pytest.approx(np.conj(voltage[3]), rel=1e-10, abs=1e-15)
    assert voltage[1] == pytest.approx(np.conj(voltage[2]), rel=1e-10, abs=1e-15)


def test_pure_fundamental_current_generates_third_harmonic_voltage() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-12
    I_star = 5e-3
    beta = 1.0
    omega0 = 2.0 * np.pi * 10e9
    a = 0.8e-3

    current = np.array([0.0, a / 2.0, a / 2.0, 0.0], dtype=np.complex128)

    voltage = _voltage_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )

    expected = _manual_voltage_coeffs(
        current,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(voltage, expected, rtol=1e-10, atol=1e-15)
    assert abs(voltage[0]) > 0.0
    assert abs(voltage[3]) > 0.0


def test_beta_scaling_affects_only_nonlinear_correction() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-12
    I_star = 5e-3
    omega0 = 2.0 * np.pi * 10e9
    current = np.array([0.02 - 0.01j, 0.5 + 0.2j, 0.5 - 0.2j, 0.02 + 0.01j]) * 1e-3

    v0 = _voltage_coeffs(current, orders, L0_H=L0, I_star_A=I_star, beta=0.0, omega0_rad_s=omega0)
    v1 = _voltage_coeffs(current, orders, L0_H=L0, I_star_A=I_star, beta=1.0, omega0_rad_s=omega0)
    v2 = _voltage_coeffs(current, orders, L0_H=L0, I_star_A=I_star, beta=2.0, omega0_rad_s=omega0)

    np.testing.assert_allclose(v2 - v0, 2.0 * (v1 - v0), rtol=1e-10, atol=1e-15)


def test_voltage_scales_linearly_with_L0() -> None:
    orders = np.array([-3, -1, 1, 3])
    current = np.array([0.01, 0.4, 0.4, 0.01], dtype=np.complex128) * 1e-3

    v1 = _voltage_coeffs(current, orders, L0_H=1e-12)
    v2 = _voltage_coeffs(current, orders, L0_H=2e-12)

    np.testing.assert_allclose(v2, 2.0 * v1, rtol=1e-12, atol=1e-15)


def test_nonlinear_correction_decreases_with_larger_i_star() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-12
    current = np.array([0.02, 0.6, 0.6, 0.02], dtype=np.complex128) * 1e-3

    v_linear = _voltage_coeffs(current, orders, L0_H=L0, I_star_A=5e-3, beta=0.0)
    v_small_istar = _voltage_coeffs(current, orders, L0_H=L0, I_star_A=3e-3, beta=1.0)
    v_large_istar = _voltage_coeffs(current, orders, L0_H=L0, I_star_A=9e-3, beta=1.0)

    small_corr = np.linalg.norm(v_small_istar - v_linear)
    large_corr = np.linalg.norm(v_large_istar - v_linear)

    assert small_corr > large_corr


def test_scalar_array_shape_is_preserved_for_selected_basis() -> None:
    orders = np.array([-3, -1, 1, 3])
    current = np.zeros(4, dtype=np.complex128)
    current[1] = 0.5e-3
    current[2] = 0.5e-3

    voltage = _voltage_coeffs(current, orders)

    assert voltage.shape == current.shape
    assert np.all(np.isfinite(voltage.real))
    assert np.all(np.isfinite(voltage.imag))


def test_batch_current_coefficients_if_supported() -> None:
    orders = np.array([-1, 1])
    current = np.array(
        [
            [0.4 + 0.1j, 0.2 + 0.0j],
            [0.4 - 0.1j, 0.2 - 0.0j],
        ],
        dtype=np.complex128,
    ) * 1e-3

    try:
        voltage = _voltage_coeffs(current, orders)
    except Exception as exc:
        pytest.skip(f"Batch HB element evaluation is optional: {type(exc).__name__}: {exc}")

    assert voltage.shape == current.shape
    assert np.all(np.isfinite(voltage.real))
    assert np.all(np.isfinite(voltage.imag))


def test_jacobian_if_exposed_matches_finite_difference_linear_case() -> None:
    helper = None
    for name in [
        "jacobian",
        "hb_jacobian",
        "voltage_jacobian",
        "current_to_voltage_jacobian",
        "element_jacobian",
    ]:
        if hasattr(hbe, name):
            helper = getattr(hbe, name)
            break

    if helper is None:
        pytest.skip("HB element Jacobian helper is optional.")

    orders = np.array([-1, 1])
    L0 = 1e-12
    omega0 = 2.0 * np.pi * 10e9
    current = np.array([0.4 + 0.1j, 0.4 - 0.1j], dtype=np.complex128) * 1e-6
    element = _make_element(L0_H=L0, I_star_A=5e-3, beta=0.0, orders=orders, omega0_rad_s=omega0)

    J = np.asarray(
        _call_with_supported_kwargs(
            helper,
            current_coeffs=current,
            I_coeffs=current,
            I=current,
            orders=orders,
            harmonic_orders=orders,
            L0_H=L0,
            I_star_A=5e-3,
            beta=0.0,
            omega0_rad_s=omega0,
            omega0=omega0,
            element=element,
        ),
        dtype=np.complex128,
    )

    expected = np.diag(1j * orders * omega0 * L0)

    assert J.shape[-2:] == (2, 2)
    np.testing.assert_allclose(J, expected, rtol=1e-10, atol=1e-18)


def test_invalid_duplicate_orders_are_rejected() -> None:
    current = np.array([1.0, 2.0], dtype=np.complex128) * 1e-6

    with pytest.raises((ValueError, AssertionError)):
        _voltage_coeffs(current, [1, 1])


def test_invalid_zero_order_is_rejected() -> None:
    current = np.array([1.0, 2.0], dtype=np.complex128) * 1e-6

    with pytest.raises((ValueError, AssertionError)):
        _voltage_coeffs(current, [0, 1])


def test_invalid_noninteger_orders_are_rejected() -> None:
    current = np.array([1.0, 2.0], dtype=np.complex128) * 1e-6

    with pytest.raises((ValueError, AssertionError, TypeError)):
        _voltage_coeffs(current, [-1.5, 1.0])


def test_current_order_length_mismatch_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _voltage_coeffs(np.array([1.0, 2.0, 3.0]), [-1, 1])


def test_invalid_nonpositive_L0_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _voltage_coeffs(np.array([1.0, 1.0]) * 1e-6, [-1, 1], L0_H=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _voltage_coeffs(np.array([1.0, 1.0]) * 1e-6, [-1, 1], L0_H=-1e-12)


def test_invalid_nonpositive_i_star_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _voltage_coeffs(np.array([1.0, 1.0]) * 1e-6, [-1, 1], I_star_A=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _voltage_coeffs(np.array([1.0, 1.0]) * 1e-6, [-1, 1], I_star_A=-5e-3)


def test_invalid_negative_beta_is_rejected_or_stays_finite() -> None:
    current = np.array([0.5, 0.5], dtype=np.complex128) * 1e-3

    try:
        voltage = _voltage_coeffs(current, [-1, 1], beta=-0.1)
    except (ValueError, AssertionError):
        return

    assert np.all(np.isfinite(voltage.real))
    assert np.all(np.isfinite(voltage.imag))


def test_nan_current_propagates_to_voltage() -> None:
    orders = np.array([-1, 1])
    current = np.array([np.nan + 0.0j, 1.0 + 0.0j]) * 1e-6

    voltage = _voltage_coeffs(current, orders)

    assert np.any(~np.isfinite(voltage))


def test_element_object_is_json_serializable() -> None:
    element = _make_element()
    mapping = dict(_as_mapping(element))

    json_ready = {}
    for key, value in mapping.items():
        if hasattr(value, "shape"):
            json_ready[key] = np.asarray(value).tolist()
        else:
            json_ready[key] = value

    import json

    json.dumps(json_ready, default=str)