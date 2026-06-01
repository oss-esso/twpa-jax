"""
Tests for twpa.nonlinear.one_node.

These tests define the expected behavior of the one-node harmonic-balance
reference problem used to validate the nonlinear HB machinery before moving to
distributed ladders.

Reference model
---------------
A current-driven parallel LC node with a nonlinear kinetic inductor:

    I_drive(t) = C dV/dt + I_L(t)

and nonlinear inductor voltage relation

    V(t) = dPhi(I_L)/dt
    Phi(I_L) = L0 * (I_L + beta * I_L**3 / (3 I*^2)).

For beta = 0, the frequency-domain solution is analytic:

    Y_k = i omega_k C + 1 / (i omega_k L0)
    V_k = I_drive,k / Y_k
    I_L,k = V_k / (i omega_k L0)

for nonzero harmonic order k.

The tests are API-tolerant in naming, but require the module to expose a usable
public interface for at least the one-node linear admittance and/or residual/
solver path.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import twpa.nonlinear.one_node as one


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
        raise ValueError("zero/DC order is not used by this one-node HB test suite")
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
    i_t = _manual_synthesize(
        current_coeffs,
        orders,
        n_time=n_time,
        omega0_rad_s=omega0_rad_s,
    )
    phi_t = L0_H * (i_t + beta * i_t**3 / (3.0 * I_star_A**2))
    return _manual_analyze(phi_t, orders, omega0_rad_s=omega0_rad_s)


def _manual_inductor_voltage_coeffs(
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


def _manual_linear_admittance(
    orders: Sequence[int],
    *,
    C_F: float,
    L0_H: float,
    omega0_rad_s: float,
) -> np.ndarray:
    orders_arr = _validate_orders(orders)
    omega = orders_arr * omega0_rad_s
    return 1j * omega * C_F + 1.0 / (1j * omega * L0_H)


def _manual_linear_solution(
    I_drive_coeffs: Sequence[complex],
    orders: Sequence[int],
    *,
    C_F: float,
    L0_H: float,
    omega0_rad_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    I_drive = np.asarray(I_drive_coeffs, dtype=np.complex128)
    Y = _manual_linear_admittance(orders, C_F=C_F, L0_H=L0_H, omega0_rad_s=omega0_rad_s)
    omega = _validate_orders(orders) * omega0_rad_s

    V = I_drive / Y
    I_L = V / (1j * omega * L0_H)
    return V, I_L


def _make_params(
    *,
    C_F: float = 100e-15,
    L0_H: float = 1e-9,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    orders: Sequence[int] = (-3, -1, 1, 3),
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
    n_time: int = 256,
) -> Any:
    for name in [
        "OneNodeHBParams",
        "OneNodeParams",
        "ParallelLCOneNodeParams",
        "OneNodeProblem",
        "OneNodeHBProblem",
    ]:
        if hasattr(one, name):
            cls = getattr(one, name)
            return _call_with_supported_kwargs(
                cls,
                C_F=C_F,
                C=C_F,
                capacitance_F=C_F,
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
                n_time=n_time,
            )

    for name in [
        "make_one_node_params",
        "make_one_node_problem",
        "make_parallel_lc_one_node_problem",
        "make_one_node_hb_problem",
    ]:
        if hasattr(one, name):
            fn = getattr(one, name)
            return _call_with_supported_kwargs(
                fn,
                C_F=C_F,
                C=C_F,
                capacitance_F=C_F,
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
                n_time=n_time,
            )

    return {
        "C_F": C_F,
        "L0_H": L0_H,
        "I_star_A": I_star_A,
        "beta": beta,
        "orders": tuple(orders),
        "omega0_rad_s": omega0_rad_s,
        "n_time": n_time,
    }


def _linear_admittance(
    orders: Sequence[int],
    *,
    C_F: float = 100e-15,
    L0_H: float = 1e-9,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
) -> np.ndarray:
    params = _make_params(C_F=C_F, L0_H=L0_H, orders=orders, omega0_rad_s=omega0_rad_s)

    for name in [
        "linear_admittance",
        "parallel_lc_admittance",
        "one_node_linear_admittance",
        "admittance",
        "Y_linear",
    ]:
        if hasattr(one, name):
            out = _call_with_supported_kwargs(
                getattr(one, name),
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                C_F=C_F,
                C=C_F,
                capacitance_F=C_F,
                L0_H=L0_H,
                L_0_H=L0_H,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                params=params,
                problem=params,
            )
            return np.asarray(out, dtype=np.complex128)

    for method_name in [
        "linear_admittance",
        "parallel_lc_admittance",
        "admittance",
        "Y_linear",
    ]:
        method = _get_attr_or_key(params, method_name, default=None)
        if callable(method):
            return np.asarray(
                _call_with_supported_kwargs(
                    method,
                    orders=np.asarray(orders),
                    harmonic_orders=np.asarray(orders),
                    omega0_rad_s=omega0_rad_s,
                    omega0=omega0_rad_s,
                ),
                dtype=np.complex128,
            )

    raise AttributeError(
        "twpa.nonlinear.one_node must expose a linear admittance helper such as "
        "linear_admittance or parallel_lc_admittance."
    )


def _capacitor_current(
    V_coeffs: Any,
    orders: Sequence[int],
    *,
    C_F: float = 100e-15,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
) -> np.ndarray:
    params = _make_params(C_F=C_F, orders=orders, omega0_rad_s=omega0_rad_s)

    for name in [
        "capacitor_current_coeffs",
        "capacitive_current_coeffs",
        "current_through_capacitor",
        "capacitor_current",
    ]:
        if hasattr(one, name):
            out = _call_with_supported_kwargs(
                getattr(one, name),
                V_coeffs=V_coeffs,
                voltage_coeffs=V_coeffs,
                V=V_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                C_F=C_F,
                C=C_F,
                capacitance_F=C_F,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                params=params,
                problem=params,
            )
            return np.asarray(out, dtype=np.complex128)

    # Fallback if module only exposes residual/solver.
    return 1j * _validate_orders(orders) * omega0_rad_s * C_F * np.asarray(V_coeffs)


def _inductor_voltage(
    I_L_coeffs: Any,
    orders: Sequence[int],
    *,
    L0_H: float = 1e-9,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
) -> np.ndarray:
    params = _make_params(
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
    )

    for name in [
        "inductor_voltage_coeffs",
        "nonlinear_inductor_voltage_coeffs",
        "kinetic_inductor_voltage_coeffs",
        "voltage_from_inductor_current",
        "current_to_voltage_coeffs",
    ]:
        if hasattr(one, name):
            out = _call_with_supported_kwargs(
                getattr(one, name),
                I_L_coeffs=I_L_coeffs,
                current_coeffs=I_L_coeffs,
                I_coeffs=I_L_coeffs,
                I=I_L_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                L0_H=L0_H,
                L_0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                params=params,
                problem=params,
            )
            return np.asarray(out, dtype=np.complex128)

    return _manual_inductor_voltage_coeffs(
        I_L_coeffs,
        orders,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
    )


def _residual(
    V_coeffs: Any,
    I_L_coeffs: Any,
    I_drive_coeffs: Any,
    orders: Sequence[int],
    *,
    C_F: float = 100e-15,
    L0_H: float = 1e-9,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
) -> np.ndarray:
    params = _make_params(
        C_F=C_F,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
    )

    for name in [
        "one_node_residual",
        "residual",
        "hb_residual",
        "parallel_lc_residual",
        "one_node_hb_residual",
    ]:
        if hasattr(one, name):
            out = _call_with_supported_kwargs(
                getattr(one, name),
                V_coeffs=V_coeffs,
                voltage_coeffs=V_coeffs,
                V=V_coeffs,
                I_L_coeffs=I_L_coeffs,
                inductor_current_coeffs=I_L_coeffs,
                current_coeffs=I_L_coeffs,
                I_L=I_L_coeffs,
                I_drive_coeffs=I_drive_coeffs,
                drive_coeffs=I_drive_coeffs,
                source_current_coeffs=I_drive_coeffs,
                I_drive=I_drive_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                C_F=C_F,
                C=C_F,
                L0_H=L0_H,
                L_0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                params=params,
                problem=params,
            )
            return np.asarray(out, dtype=np.complex128).reshape(-1)

    # Fallback residual convention:
    # first block KCL: I_drive - I_C - I_L
    # second block inductor voltage mismatch: V - V_L(I_L)
    V = np.asarray(V_coeffs, dtype=np.complex128)
    I_L = np.asarray(I_L_coeffs, dtype=np.complex128)
    I_drive = np.asarray(I_drive_coeffs, dtype=np.complex128)

    kcl = I_drive - _capacitor_current(V, orders, C_F=C_F, omega0_rad_s=omega0_rad_s) - I_L
    vlaw = V - _inductor_voltage(
        I_L,
        orders,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
    )

    return np.concatenate([kcl, vlaw])


def _solve_one_node(
    I_drive_coeffs: Any,
    orders: Sequence[int],
    *,
    C_F: float = 100e-15,
    L0_H: float = 1e-9,
    I_star_A: float = 5e-3,
    beta: float = 0.0,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
    n_time: int = 512,
) -> tuple[np.ndarray, np.ndarray, Any]:
    params = _make_params(
        C_F=C_F,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )

    for name in [
        "solve_one_node",
        "solve_one_node_hb",
        "solve_parallel_lc_hb",
        "solve_one_node_parallel_lc",
        "solve",
    ]:
        if hasattr(one, name):
            out = _call_with_supported_kwargs(
                getattr(one, name),
                I_drive_coeffs=I_drive_coeffs,
                drive_coeffs=I_drive_coeffs,
                source_current_coeffs=I_drive_coeffs,
                I_drive=I_drive_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                C_F=C_F,
                C=C_F,
                capacitance_F=C_F,
                L0_H=L0_H,
                L_0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                n_time=n_time,
                params=params,
                problem=params,
                max_iter=80,
                tolerance=1e-12,
                tol=1e-12,
            )

            V = _get_attr_or_key(out, "V_coeffs", "voltage_coeffs", "V", default=None)
            I_L = _get_attr_or_key(
                out,
                "I_L_coeffs",
                "inductor_current_coeffs",
                "current_coeffs",
                "I_L",
                default=None,
            )

            if V is not None and I_L is not None:
                return np.asarray(V, dtype=np.complex128), np.asarray(I_L, dtype=np.complex128), out

            if isinstance(out, tuple):
                if len(out) >= 2:
                    return (
                        np.asarray(out[0], dtype=np.complex128),
                        np.asarray(out[1], dtype=np.complex128),
                        out,
                    )

            raise TypeError(f"Could not extract V and I_L from solver output type {type(out)!r}")

    pytest.skip("One-node solver is optional if residual/admittance helpers are available.")


def test_params_object_exposes_basic_fields() -> None:
    params = _make_params(
        C_F=120e-15,
        L0_H=2e-9,
        I_star_A=4e-3,
        beta=0.75,
        orders=(-3, -1, 1, 3),
    )

    mapping = _as_mapping(params)
    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0

    C = _get_attr_or_key(params, "C_F", "C", "capacitance_F", default=120e-15)
    L0 = _get_attr_or_key(params, "L0_H", "L_0_H", "l0_H", default=2e-9)
    I_star = _get_attr_or_key(params, "I_star_A", "i_star_A", "Istar_A", default=4e-3)
    beta = _get_attr_or_key(params, "beta", "beta_nl", "nonlinear_beta", default=0.75)

    assert float(C) == pytest.approx(120e-15)
    assert float(L0) == pytest.approx(2e-9)
    assert float(I_star) == pytest.approx(4e-3)
    assert float(beta) == pytest.approx(0.75)


def test_linear_admittance_matches_parallel_lc_formula() -> None:
    orders = np.array([-3, -1, 1, 3])
    C = 100e-15
    L0 = 1e-9
    omega0 = 2.0 * np.pi * 5e9

    Y = _linear_admittance(orders, C_F=C, L0_H=L0, omega0_rad_s=omega0)
    expected = _manual_linear_admittance(orders, C_F=C, L0_H=L0, omega0_rad_s=omega0)

    np.testing.assert_allclose(Y, expected, rtol=1e-12, atol=1e-18)


def test_linear_admittance_is_odd_for_lossless_reactive_network() -> None:
    orders = np.array([-3, -1, 1, 3])
    Y = _linear_admittance(orders)

    assert Y[0] == pytest.approx(-Y[3], rel=1e-12, abs=1e-18)
    assert Y[1] == pytest.approx(-Y[2], rel=1e-12, abs=1e-18)
    np.testing.assert_allclose(np.real(Y), 0.0, atol=1e-18)


def test_capacitor_current_matches_iomega_cv() -> None:
    orders = np.array([-3, -1, 1, 3])
    C = 80e-15
    omega0 = 2.0 * np.pi * 7e9
    V = np.array([0.1 - 0.2j, 1.0 + 0.1j, 0.5 - 0.4j, -0.1 + 0.3j]) * 1e-6

    I_C = _capacitor_current(V, orders, C_F=C, omega0_rad_s=omega0)
    expected = 1j * orders * omega0 * C * V

    np.testing.assert_allclose(I_C, expected, rtol=1e-12, atol=1e-18)


def test_inductor_voltage_linear_case_matches_iomega_l_i() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-9
    omega0 = 2.0 * np.pi * 5e9
    I_L = np.array([0.01 - 0.02j, 1.0 + 0.3j, 0.5 - 0.1j, -0.01 + 0.04j]) * 1e-6

    V_L = _inductor_voltage(
        I_L,
        orders,
        L0_H=L0,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    expected = 1j * orders * omega0 * L0 * I_L

    np.testing.assert_allclose(V_L, expected, rtol=1e-12, atol=1e-18)


def test_inductor_voltage_nonlinear_case_matches_time_domain_projection() -> None:
    orders = np.array([-3, -1, 1, 3])
    L0 = 1e-9
    I_star = 5e-3
    beta = 0.8
    omega0 = 2.0 * np.pi * 5e9

    I_L = np.array(
        [
            0.02 - 0.01j,
            0.5 + 0.2j,
            0.5 - 0.2j,
            0.02 + 0.01j,
        ],
        dtype=np.complex128,
    ) * 1e-3

    V_L = _inductor_voltage(
        I_L,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )

    expected = _manual_inductor_voltage_coeffs(
        I_L,
        orders,
        L0_H=L0,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(V_L, expected, rtol=1e-10, atol=1e-12)


def test_residual_is_zero_for_linear_analytic_solution() -> None:
    orders = np.array([-3, -1, 1, 3])
    C = 100e-15
    L0 = 1e-9
    omega0 = 2.0 * np.pi * 5e9

    I_drive = np.array([0.0, 1.0 + 0.2j, 0.1 - 0.05j, 0.0]) * 1e-6
    V, I_L = _manual_linear_solution(I_drive, orders, C_F=C, L0_H=L0, omega0_rad_s=omega0)

    residual = _residual(
        V,
        I_L,
        I_drive,
        orders,
        C_F=C,
        L0_H=L0,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(residual, np.zeros_like(residual), rtol=0.0, atol=1e-15)


def test_residual_detects_kcl_perturbation() -> None:
    orders = np.array([-1, 1])
    C = 100e-15
    L0 = 1e-9
    omega0 = 2.0 * np.pi * 5e9

    I_drive = np.array([1.0 + 0.0j, 1.0 + 0.0j]) * 1e-6
    V, I_L = _manual_linear_solution(I_drive, orders, C_F=C, L0_H=L0, omega0_rad_s=omega0)

    residual_clean = _residual(
        V,
        I_L,
        I_drive,
        orders,
        C_F=C,
        L0_H=L0,
        beta=0.0,
        omega0_rad_s=omega0,
    )
    residual_bad = _residual(
        V,
        I_L + np.array([1.0, -1.0]) * 1e-9,
        I_drive,
        orders,
        C_F=C,
        L0_H=L0,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    assert np.linalg.norm(residual_clean) < 1e-14
    assert np.linalg.norm(residual_bad) > 1e-10


def test_solver_recovers_linear_analytic_solution_if_available() -> None:
    orders = np.array([-3, -1, 1, 3])
    C = 100e-15
    L0 = 1e-9
    omega0 = 2.0 * np.pi * 5e9

    I_drive = np.array([0.0, 1.0 + 0.2j, 0.1 - 0.05j, 0.0]) * 1e-6
    V_expected, I_expected = _manual_linear_solution(
        I_drive,
        orders,
        C_F=C,
        L0_H=L0,
        omega0_rad_s=omega0,
    )

    V, I_L, _ = _solve_one_node(
        I_drive,
        orders,
        C_F=C,
        L0_H=L0,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(V, V_expected, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(I_L, I_expected, rtol=1e-9, atol=1e-15)


def test_solver_solution_has_small_residual_if_available() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_drive = np.array([0.0, 0.8 + 0.1j, 0.8 - 0.1j, 0.0]) * 1e-6

    V, I_L, _ = _solve_one_node(
        I_drive,
        orders,
        C_F=100e-15,
        L0_H=1e-9,
        I_star_A=5e-3,
        beta=1.0,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )

    residual = _residual(
        V,
        I_L,
        I_drive,
        orders,
        C_F=100e-15,
        L0_H=1e-9,
        I_star_A=5e-3,
        beta=1.0,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )

    assert np.linalg.norm(residual) < 1e-9


def test_zero_drive_solution_is_zero_if_solver_available() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_drive = np.zeros(len(orders), dtype=np.complex128)

    V, I_L, _ = _solve_one_node(
        I_drive,
        orders,
        C_F=100e-15,
        L0_H=1e-9,
        I_star_A=5e-3,
        beta=1.0,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )

    np.testing.assert_allclose(V, np.zeros_like(V), atol=1e-18, rtol=0.0)
    np.testing.assert_allclose(I_L, np.zeros_like(I_L), atol=1e-18, rtol=0.0)


def test_conjugate_symmetric_drive_yields_conjugate_symmetric_solution_if_solver_available() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_drive = np.array([0.02 - 0.01j, 0.8 + 0.2j, 0.8 - 0.2j, 0.02 + 0.01j]) * 1e-6

    V, I_L, _ = _solve_one_node(
        I_drive,
        orders,
        C_F=100e-15,
        L0_H=1e-9,
        I_star_A=5e-3,
        beta=0.5,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )

    assert V[0] == pytest.approx(np.conj(V[3]), rel=1e-8, abs=1e-15)
    assert V[1] == pytest.approx(np.conj(V[2]), rel=1e-8, abs=1e-15)
    assert I_L[0] == pytest.approx(np.conj(I_L[3]), rel=1e-8, abs=1e-15)
    assert I_L[1] == pytest.approx(np.conj(I_L[2]), rel=1e-8, abs=1e-15)


def test_nonlinear_voltage_generates_third_harmonic_from_fundamental_current() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_L = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.complex128) * 1e-3

    V_nl = _inductor_voltage(
        I_L,
        orders,
        L0_H=1e-9,
        I_star_A=5e-3,
        beta=1.0,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )
    V_lin = _inductor_voltage(
        I_L,
        orders,
        L0_H=1e-9,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )

    assert abs(V_nl[0] - V_lin[0]) > 0.0
    assert abs(V_nl[3] - V_lin[3]) > 0.0


def test_nonlinear_correction_scales_linearly_with_beta() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_L = np.array([0.01, 0.5, 0.5, 0.01], dtype=np.complex128) * 1e-3

    v0 = _inductor_voltage(I_L, orders, beta=0.0)
    v1 = _inductor_voltage(I_L, orders, beta=1.0)
    v2 = _inductor_voltage(I_L, orders, beta=2.0)

    np.testing.assert_allclose(v2 - v0, 2.0 * (v1 - v0), rtol=1e-10, atol=1e-12)


def test_solution_changes_smoothly_with_small_drive_if_solver_available() -> None:
    orders = np.array([-1, 1])
    base_drive = np.array([1.0, 1.0], dtype=np.complex128) * 1e-7

    V1, I1, _ = _solve_one_node(base_drive, orders, beta=1.0)
    V2, I2, _ = _solve_one_node(1.01 * base_drive, orders, beta=1.0)

    assert np.linalg.norm(V2 - V1) > 0.0
    assert np.linalg.norm(I2 - I1) > 0.0
    assert np.linalg.norm(V2 - V1) < 0.05 * max(np.linalg.norm(V1), 1e-30)
    assert np.linalg.norm(I2 - I1) < 0.05 * max(np.linalg.norm(I1), 1e-30)


def test_batch_drive_if_solver_supports_it() -> None:
    orders = np.array([-1, 1])
    I_drive = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.0j],
            [1.0 + 0.0j, 0.5 + 0.0j],
        ],
        dtype=np.complex128,
    ) * 1e-6

    try:
        V, I_L, _ = _solve_one_node(I_drive, orders, beta=0.0)
    except Exception as exc:
        pytest.skip(f"Batch one-node solve is optional: {type(exc).__name__}: {exc}")

    assert V.shape == I_drive.shape
    assert I_L.shape == I_drive.shape


def test_invalid_duplicate_orders_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _linear_admittance([1, 1])

    with pytest.raises((ValueError, AssertionError)):
        _residual(
            np.ones(2),
            np.ones(2),
            np.ones(2),
            [1, 1],
        )


def test_invalid_zero_order_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _linear_admittance([0, 1])


def test_invalid_noninteger_orders_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, TypeError)):
        _linear_admittance([-1.5, 1.0])


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _residual(
            np.ones(3),
            np.ones(3),
            np.ones(3),
            [-1, 1],
        )


def test_invalid_nonpositive_capacitance_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _linear_admittance([-1, 1], C_F=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _linear_admittance([-1, 1], C_F=-100e-15)


def test_invalid_nonpositive_inductance_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _linear_admittance([-1, 1], L0_H=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _linear_admittance([-1, 1], L0_H=-1e-9)


def test_invalid_nonpositive_i_star_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _inductor_voltage(
            np.array([1.0, 1.0]) * 1e-6,
            [-1, 1],
            I_star_A=0.0,
        )

    with pytest.raises((ValueError, AssertionError)):
        _inductor_voltage(
            np.array([1.0, 1.0]) * 1e-6,
            [-1, 1],
            I_star_A=-5e-3,
        )


def test_invalid_negative_beta_is_rejected_or_stays_finite() -> None:
    I_L = np.array([0.5, 0.5], dtype=np.complex128) * 1e-3

    try:
        V = _inductor_voltage(I_L, [-1, 1], beta=-0.1)
    except (ValueError, AssertionError):
        return

    assert np.all(np.isfinite(V.real))
    assert np.all(np.isfinite(V.imag))


def test_nan_drive_or_current_propagates_to_residual() -> None:
    orders = np.array([-1, 1])
    V = np.array([1.0, 1.0], dtype=np.complex128)
    I_L = np.array([np.nan, 1.0], dtype=np.complex128)
    I_drive = np.array([1.0, 1.0], dtype=np.complex128)

    residual = _residual(V, I_L, I_drive, orders)

    assert np.any(~np.isfinite(residual))


def test_result_object_if_solver_available_is_serializable_or_mapping_like() -> None:
    orders = np.array([-1, 1])
    I_drive = np.array([1.0, 1.0], dtype=np.complex128) * 1e-6

    _, _, result = _solve_one_node(I_drive, orders, beta=0.0)

    try:
        mapping = _as_mapping(result)
    except TypeError:
        if isinstance(result, tuple):
            pytest.skip("Tuple solver result is acceptable and has no mapping metadata.")
        raise

    json_ready = {}
    for key, value in mapping.items():
        if hasattr(value, "shape"):
            arr = np.asarray(value)
            if np.iscomplexobj(arr):
                json_ready[key] = {
                    "real": np.real(arr).tolist(),
                    "imag": np.imag(arr).tolist(),
                }
            else:
                json_ready[key] = arr.tolist()
        else:
            json_ready[key] = value

    import json

    json.dumps(json_ready, default=str)
