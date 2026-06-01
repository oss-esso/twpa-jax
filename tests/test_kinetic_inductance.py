"""
Tests for twpa.nonlinear.kinetic_inductance.

These tests define the expected public behavior of the kinetic-inductance
nonlinearity layer used by the HB solver.

Reference model expected by the production stack
------------------------------------------------
For a current-dependent kinetic inductance,

    L(I) = L0 * (1 + beta * (I / I*)**2)

where:
    L0      is the small-signal inductance,
    I*      is the characteristic nonlinear current scale,
    beta    is a dimensionless nonlinear coefficient, normally 1.

The tests are API-tolerant: they accept several reasonable function/class names,
but they require the module to expose a usable public interface for:

    - current-dependent inductance,
    - derivative dL/dI,
    - inverse inductance or its equivalent,
    - flux-current relation or energy if implemented,
    - parameter object serialization if implemented.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping

import numpy as np
import pytest

import twpa.nonlinear.kinetic_inductance as ki


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


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


def _make_params(
    *,
    L0_H: float = 1e-12,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
) -> Any:
    for name in [
        "KineticInductanceParams",
        "NonlinearInductanceParams",
        "KineticInductanceModel",
    ]:
        if hasattr(ki, name):
            cls = getattr(ki, name)
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
            )

    for name in [
        "make_kinetic_inductance_params",
        "make_nonlinear_inductance_params",
        "make_kinetic_inductance_model",
    ]:
        if hasattr(ki, name):
            fn = getattr(ki, name)
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
            )

    return {
        "L0_H": L0_H,
        "I_star_A": I_star_A,
        "beta": beta,
    }


def _extract_L0(params: Any, default: float = 1e-12) -> float:
    return float(
        _get_attr_or_key(
            params,
            "L0_H",
            "L_0_H",
            "l0_H",
            "linear_inductance_H",
            "small_signal_inductance_H",
            default=default,
        )
    )


def _extract_I_star(params: Any, default: float = 5e-3) -> float:
    return float(
        _get_attr_or_key(
            params,
            "I_star_A",
            "i_star_A",
            "Istar_A",
            "current_scale_A",
            "nonlinear_current_A",
            default=default,
        )
    )


def _extract_beta(params: Any, default: float = 1.0) -> float:
    return float(
        _get_attr_or_key(
            params,
            "beta",
            "beta_nl",
            "nonlinear_beta",
            default=default,
        )
    )


def _kinetic_inductance(I_A: Any, *, L0_H: float = 1e-12, I_star_A: float = 5e-3, beta: float = 1.0) -> Any:
    params_obj = _make_params(L0_H=L0_H, I_star_A=I_star_A, beta=beta)

    for name in [
        "kinetic_inductance",
        "current_dependent_inductance",
        "nonlinear_inductance",
        "inductance",
        "L_of_I",
        "L_kinetic",
    ]:
        if hasattr(ki, name):
            fn = getattr(ki, name)
            return _call_with_supported_kwargs(
                fn,
                I_A=I_A,
                I=I_A,
                current_A=I_A,
                current=I_A,
                L0_H=L0_H,
                L_0_H=L0_H,
                l0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                params=params_obj,
                model=params_obj,
            )

    if hasattr(params_obj, "inductance") and callable(params_obj.inductance):
        return params_obj.inductance(I_A)

    if hasattr(params_obj, "L_of_I") and callable(params_obj.L_of_I):
        return params_obj.L_of_I(I_A)

    raise AttributeError(
        "twpa.nonlinear.kinetic_inductance must expose a current-dependent "
        "inductance function such as kinetic_inductance, L_of_I, or a model "
        "object with .inductance(I)."
    )


def _dL_dI(I_A: Any, *, L0_H: float = 1e-12, I_star_A: float = 5e-3, beta: float = 1.0) -> Any:
    params_obj = _make_params(L0_H=L0_H, I_star_A=I_star_A, beta=beta)

    for name in [
        "dL_dI",
        "kinetic_inductance_derivative",
        "current_dependent_inductance_derivative",
        "d_inductance_d_current",
        "dL_dcurrent",
    ]:
        if hasattr(ki, name):
            fn = getattr(ki, name)
            return _call_with_supported_kwargs(
                fn,
                I_A=I_A,
                I=I_A,
                current_A=I_A,
                current=I_A,
                L0_H=L0_H,
                L_0_H=L0_H,
                l0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                params=params_obj,
                model=params_obj,
            )

    if hasattr(params_obj, "dL_dI") and callable(params_obj.dL_dI):
        return params_obj.dL_dI(I_A)

    raise AttributeError(
        "twpa.nonlinear.kinetic_inductance must expose dL/dI via dL_dI or "
        "kinetic_inductance_derivative."
    )


def _inverse_inductance(I_A: Any, *, L0_H: float = 1e-12, I_star_A: float = 5e-3, beta: float = 1.0) -> Any:
    params_obj = _make_params(L0_H=L0_H, I_star_A=I_star_A, beta=beta)

    for name in [
        "inverse_kinetic_inductance",
        "inverse_inductance",
        "current_dependent_inverse_inductance",
        "inv_L_of_I",
        "L_inv_of_I",
    ]:
        if hasattr(ki, name):
            fn = getattr(ki, name)
            return _call_with_supported_kwargs(
                fn,
                I_A=I_A,
                I=I_A,
                current_A=I_A,
                current=I_A,
                L0_H=L0_H,
                L_0_H=L0_H,
                l0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                params=params_obj,
                model=params_obj,
            )

    if hasattr(params_obj, "inverse_inductance") and callable(params_obj.inverse_inductance):
        return params_obj.inverse_inductance(I_A)

    raise AttributeError(
        "twpa.nonlinear.kinetic_inductance must expose inverse inductance via "
        "inverse_kinetic_inductance or inverse_inductance."
    )


def _flux_from_current(I_A: Any, *, L0_H: float = 1e-12, I_star_A: float = 5e-3, beta: float = 1.0) -> Any:
    params_obj = _make_params(L0_H=L0_H, I_star_A=I_star_A, beta=beta)

    for name in [
        "flux_from_current",
        "current_to_flux",
        "phi_of_I",
        "flux_linkage",
    ]:
        if hasattr(ki, name):
            fn = getattr(ki, name)
            return _call_with_supported_kwargs(
                fn,
                I_A=I_A,
                I=I_A,
                current_A=I_A,
                current=I_A,
                L0_H=L0_H,
                L_0_H=L0_H,
                l0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                params=params_obj,
                model=params_obj,
            )

    if hasattr(params_obj, "flux_from_current") and callable(params_obj.flux_from_current):
        return params_obj.flux_from_current(I_A)

    pytest.skip("Flux-current relation is optional in kinetic_inductance.py.")


def _energy(I_A: Any, *, L0_H: float = 1e-12, I_star_A: float = 5e-3, beta: float = 1.0) -> Any:
    params_obj = _make_params(L0_H=L0_H, I_star_A=I_star_A, beta=beta)

    for name in [
        "magnetic_energy",
        "inductive_energy",
        "kinetic_inductance_energy",
        "energy",
        "energy_of_I",
    ]:
        if hasattr(ki, name):
            fn = getattr(ki, name)
            return _call_with_supported_kwargs(
                fn,
                I_A=I_A,
                I=I_A,
                current_A=I_A,
                current=I_A,
                L0_H=L0_H,
                L_0_H=L0_H,
                l0_H=L0_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                params=params_obj,
                model=params_obj,
            )

    if hasattr(params_obj, "energy") and callable(params_obj.energy):
        return params_obj.energy(I_A)

    pytest.skip("Inductive energy helper is optional in kinetic_inductance.py.")


def test_params_object_exposes_core_fields_when_available() -> None:
    p = _make_params(L0_H=2e-12, I_star_A=4e-3, beta=0.7)

    assert _extract_L0(p) == pytest.approx(2e-12)
    assert _extract_I_star(p) == pytest.approx(4e-3)
    assert _extract_beta(p) == pytest.approx(0.7)

    mapping = _as_mapping(p)
    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0


def test_kinetic_inductance_zero_current_equals_L0() -> None:
    L0 = 1.25e-12
    I_star = 5e-3

    L = _kinetic_inductance(0.0, L0_H=L0, I_star_A=I_star, beta=1.0)

    assert float(np.asarray(L)) == pytest.approx(L0, rel=1e-14)


def test_kinetic_inductance_matches_quadratic_model() -> None:
    L0 = 1e-12
    I_star = 5e-3
    beta = 0.8
    I = np.array([-0.5, -0.1, 0.0, 0.1, 0.5]) * I_star

    L = np.asarray(_kinetic_inductance(I, L0_H=L0, I_star_A=I_star, beta=beta), dtype=float)
    expected = L0 * (1.0 + beta * (I / I_star) ** 2)

    np.testing.assert_allclose(L, expected, rtol=1e-13, atol=0.0)


def test_kinetic_inductance_is_even_in_current() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.linspace(-0.8 * I_star, 0.8 * I_star, 101)

    L_pos = np.asarray(_kinetic_inductance(np.abs(I), L0_H=L0, I_star_A=I_star), dtype=float)
    L_neg = np.asarray(_kinetic_inductance(-np.abs(I), L0_H=L0, I_star_A=I_star), dtype=float)

    np.testing.assert_allclose(L_pos, L_neg, rtol=1e-14, atol=0.0)


def test_kinetic_inductance_increases_with_current_magnitude_for_positive_beta() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.linspace(0.0, I_star, 64)

    L = np.asarray(_kinetic_inductance(I, L0_H=L0, I_star_A=I_star, beta=1.0), dtype=float)

    assert np.all(np.diff(L) >= -1e-24)
    assert L[-1] > L[0]


def test_kinetic_inductance_supports_complex_current_by_magnitude_or_square_convention() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.array([0.0 + 0.0j, 0.1 + 0.2j, -0.2 + 0.1j]) * I_star

    L = np.asarray(_kinetic_inductance(I, L0_H=L0, I_star_A=I_star, beta=1.0))

    assert L.shape == I.shape
    assert np.all(np.isfinite(np.real(L)))
    assert np.all(np.real(L) > 0.0)


def test_dL_dI_matches_quadratic_model() -> None:
    L0 = 1e-12
    I_star = 5e-3
    beta = 0.75
    I = np.linspace(-0.5 * I_star, 0.5 * I_star, 17)

    dL = np.asarray(_dL_dI(I, L0_H=L0, I_star_A=I_star, beta=beta), dtype=float)
    expected = 2.0 * L0 * beta * I / (I_star**2)

    np.testing.assert_allclose(dL, expected, rtol=1e-12, atol=1e-24)


def test_dL_dI_is_odd_and_zero_at_origin() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.linspace(0.0, I_star, 21)

    dL_pos = np.asarray(_dL_dI(I, L0_H=L0, I_star_A=I_star), dtype=float)
    dL_neg = np.asarray(_dL_dI(-I, L0_H=L0, I_star_A=I_star), dtype=float)

    np.testing.assert_allclose(dL_neg, -dL_pos, rtol=1e-14, atol=1e-24)
    assert dL_pos[0] == pytest.approx(0.0, abs=1e-24)


def test_inverse_inductance_is_reciprocal_of_inductance() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.linspace(-0.8 * I_star, 0.8 * I_star, 23)

    L = np.asarray(_kinetic_inductance(I, L0_H=L0, I_star_A=I_star), dtype=float)
    inv_L = np.asarray(_inverse_inductance(I, L0_H=L0, I_star_A=I_star), dtype=float)

    np.testing.assert_allclose(inv_L, 1.0 / L, rtol=1e-12, atol=0.0)


def test_flux_from_current_matches_integral_of_L_of_I_if_exposed() -> None:
    L0 = 1e-12
    I_star = 5e-3
    beta = 1.0
    I = np.linspace(-0.5 * I_star, 0.5 * I_star, 31)

    phi = np.asarray(_flux_from_current(I, L0_H=L0, I_star_A=I_star, beta=beta), dtype=float)

    expected = L0 * (I + beta * I**3 / (3.0 * I_star**2))

    np.testing.assert_allclose(phi, expected, rtol=1e-12, atol=1e-24)


def test_flux_from_current_is_odd_if_exposed() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.linspace(0.0, 0.8 * I_star, 23)

    phi_pos = np.asarray(_flux_from_current(I, L0_H=L0, I_star_A=I_star), dtype=float)
    phi_neg = np.asarray(_flux_from_current(-I, L0_H=L0, I_star_A=I_star), dtype=float)

    np.testing.assert_allclose(phi_neg, -phi_pos, rtol=1e-13, atol=1e-24)


def test_energy_matches_integral_of_flux_if_exposed() -> None:
    L0 = 1e-12
    I_star = 5e-3
    beta = 1.0
    I = np.linspace(0.0, 0.7 * I_star, 19)

    U = np.asarray(_energy(I, L0_H=L0, I_star_A=I_star, beta=beta), dtype=float)

    expected = 0.5 * L0 * I**2 + L0 * beta * I**4 / (12.0 * I_star**2)

    np.testing.assert_allclose(U, expected, rtol=1e-12, atol=1e-24)
    assert U[0] == pytest.approx(0.0, abs=1e-30)
    assert np.all(np.diff(U) >= -1e-30)


def test_scalar_and_array_shapes_are_preserved() -> None:
    L0 = 1e-12
    I_star = 5e-3

    scalar = _kinetic_inductance(0.1 * I_star, L0_H=L0, I_star_A=I_star)
    assert np.asarray(scalar).shape == ()

    matrix = np.array([[0.0, 0.1], [0.2, 0.3]]) * I_star
    L = np.asarray(_kinetic_inductance(matrix, L0_H=L0, I_star_A=I_star))
    dL = np.asarray(_dL_dI(matrix, L0_H=L0, I_star_A=I_star))

    assert L.shape == matrix.shape
    assert dL.shape == matrix.shape


def test_invalid_nonpositive_L0_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _kinetic_inductance(0.0, L0_H=0.0, I_star_A=5e-3)

    with pytest.raises((ValueError, AssertionError)):
        _kinetic_inductance(0.0, L0_H=-1e-12, I_star_A=5e-3)


def test_invalid_nonpositive_I_star_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _kinetic_inductance(0.0, L0_H=1e-12, I_star_A=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _kinetic_inductance(0.0, L0_H=1e-12, I_star_A=-5e-3)


def test_invalid_negative_beta_is_rejected_or_remains_positive() -> None:
    L0 = 1e-12
    I_star = 5e-3

    try:
        L = np.asarray(_kinetic_inductance(I_star, L0_H=L0, I_star_A=I_star, beta=-0.1), dtype=float)
    except (ValueError, AssertionError):
        return

    assert np.all(L > 0.0)


def test_parameter_object_is_json_serializable() -> None:
    p = _make_params(L0_H=1e-12, I_star_A=5e-3, beta=1.0)
    mapping = dict(_as_mapping(p))

    json_ready = {}
    for key, value in mapping.items():
        if hasattr(value, "shape"):
            json_ready[key] = np.asarray(value).tolist()
        else:
            json_ready[key] = value

    import json

    json.dumps(json_ready, default=str)


def test_small_current_limit_is_linear_inductor() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.array([-1e-6, 0.0, 1e-6])

    L = np.asarray(_kinetic_inductance(I, L0_H=L0, I_star_A=I_star), dtype=float)

    np.testing.assert_allclose(L, L0, rtol=1e-7, atol=0.0)


def test_large_current_remains_finite_in_reasonable_range() -> None:
    L0 = 1e-12
    I_star = 5e-3
    I = np.linspace(-2.0 * I_star, 2.0 * I_star, 41)

    L = np.asarray(_kinetic_inductance(I, L0_H=L0, I_star_A=I_star), dtype=float)

    assert np.all(np.isfinite(L))
    assert np.all(L > 0.0)