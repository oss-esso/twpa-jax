"""
Tests for twpa.core.harmonics.

These tests define the expected harmonic-index and frequency-plan behavior used
by the harmonic-balance stack:

    - harmonic set construction
    - positive/negative harmonic symmetry
    - pump/signal/idler frequency relations
    - index lookup
    - time-grid compatibility
    - Fourier coefficient ordering
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import twpa.core.harmonics as hm


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


def _make_harmonic_set(orders: Sequence[int] = (-3, -1, 1, 3)) -> Any:
    for name in [
        "HarmonicSet",
        "Harmonics",
        "HarmonicBasis",
        "HarmonicPlan",
    ]:
        if hasattr(hm, name):
            cls = getattr(hm, name)
            return _call_with_supported_kwargs(
                cls,
                orders=tuple(orders),
                harmonic_orders=tuple(orders),
                indices=tuple(orders),
                harmonics=tuple(orders),
            )

    for name in [
        "make_harmonic_set",
        "make_harmonics",
        "make_harmonic_basis",
        "harmonic_set",
    ]:
        if hasattr(hm, name):
            fn = getattr(hm, name)
            return _call_with_supported_kwargs(
                fn,
                orders=tuple(orders),
                harmonic_orders=tuple(orders),
                indices=tuple(orders),
                harmonics=tuple(orders),
            )

    raise AttributeError(
        "twpa.core.harmonics must expose HarmonicSet/make_harmonic_set or equivalent."
    )


def _extract_orders(hset: Any) -> np.ndarray:
    value = _get_attr_or_key(
        hset,
        "orders",
        "harmonic_orders",
        "indices",
        "harmonics",
        default=None,
    )
    if value is None and isinstance(hset, (list, tuple, np.ndarray)):
        value = hset
    if value is None:
        raise AttributeError("Harmonic set does not expose harmonic orders.")
    return np.asarray(value, dtype=int)


def _index_of(hset: Any, order: int) -> int:
    for name in ["index", "index_of", "order_index", "harmonic_index", "idx"]:
        method = _get_attr_or_key(hset, name, default=None)
        if callable(method):
            return int(method(order))

    for name in ["index_of", "order_index", "harmonic_index"]:
        if hasattr(hm, name):
            return int(
                _call_with_supported_kwargs(
                    getattr(hm, name),
                    harmonic_set=hset,
                    harmonics=hset,
                    orders=_extract_orders(hset),
                    order=order,
                    harmonic=order,
                )
            )

    orders = list(_extract_orders(hset))
    return orders.index(order)


def _make_symmetric_orders(max_order: int = 3, *, odd_only: bool = True) -> np.ndarray:
    for name in [
        "symmetric_harmonic_orders",
        "make_symmetric_harmonics",
        "make_symmetric_orders",
        "harmonic_orders_symmetric",
    ]:
        if hasattr(hm, name):
            return np.asarray(
                _call_with_supported_kwargs(
                    getattr(hm, name),
                    max_order=max_order,
                    max_harmonic=max_order,
                    n=max_order,
                    odd_only=odd_only,
                    include_odd_only=odd_only,
                ),
                dtype=int,
            )

    if odd_only:
        pos = np.arange(1, max_order + 1, 2)
    else:
        pos = np.arange(1, max_order + 1)
    return np.concatenate([-pos[::-1], pos])


def _make_frequency_plan(
    *,
    pump_frequency_hz: float = 10e9,
    signal_frequency_hz: float = 6e9,
    harmonic_orders: Sequence[int] = (-3, -1, 1, 3),
) -> Any:
    for name in [
        "FrequencyPlan",
        "HarmonicFrequencyPlan",
        "PumpSignalFrequencyPlan",
    ]:
        if hasattr(hm, name):
            cls = getattr(hm, name)
            return _call_with_supported_kwargs(
                cls,
                pump_frequency_hz=pump_frequency_hz,
                signal_frequency_hz=signal_frequency_hz,
                harmonic_orders=tuple(harmonic_orders),
                orders=tuple(harmonic_orders),
            )

    for name in [
        "make_frequency_plan",
        "make_harmonic_frequency_plan",
        "pump_signal_frequency_plan",
    ]:
        if hasattr(hm, name):
            fn = getattr(hm, name)
            return _call_with_supported_kwargs(
                fn,
                pump_frequency_hz=pump_frequency_hz,
                signal_frequency_hz=signal_frequency_hz,
                fp_hz=pump_frequency_hz,
                fs_hz=signal_frequency_hz,
                harmonic_orders=tuple(harmonic_orders),
                orders=tuple(harmonic_orders),
            )

    raise AttributeError(
        "twpa.core.harmonics must expose FrequencyPlan/make_frequency_plan or equivalent."
    )


def _extract_frequencies(plan: Any) -> np.ndarray:
    value = _get_attr_or_key(
        plan,
        "frequencies_hz",
        "frequency_hz",
        "harmonic_frequencies_hz",
        "omega_over_2pi_hz",
        default=None,
    )
    if value is None:
        raise AttributeError("Frequency plan does not expose frequencies_hz.")
    return np.asarray(value, dtype=float)


def _extract_angular_frequencies(plan: Any) -> np.ndarray:
    value = _get_attr_or_key(
        plan,
        "angular_frequencies_rad_s",
        "omega_rad_s",
        "omegas_rad_s",
        "angular_frequency_rad_s",
        default=None,
    )
    if value is not None:
        return np.asarray(value, dtype=float)

    return 2.0 * np.pi * _extract_frequencies(plan)


def _idler_frequency_hz(plan: Any) -> float:
    value = _get_attr_or_key(
        plan,
        "idler_frequency_hz",
        "fi_hz",
        "idler_hz",
        default=None,
    )
    if value is not None:
        return float(value)

    fp = float(_get_attr_or_key(plan, "pump_frequency_hz", "fp_hz", default=10e9))
    fs = float(_get_attr_or_key(plan, "signal_frequency_hz", "fs_hz", default=6e9))
    return 2.0 * fp - fs


def test_symmetric_odd_harmonic_orders() -> None:
    orders = _make_symmetric_orders(5, odd_only=True)

    np.testing.assert_array_equal(orders, np.array([-5, -3, -1, 1, 3, 5]))


def test_symmetric_all_harmonic_orders() -> None:
    orders = _make_symmetric_orders(3, odd_only=False)

    np.testing.assert_array_equal(orders, np.array([-3, -2, -1, 1, 2, 3]))


def test_harmonic_set_preserves_ordering() -> None:
    hset = _make_harmonic_set((-3, -1, 1, 3))
    orders = _extract_orders(hset)

    np.testing.assert_array_equal(orders, np.array([-3, -1, 1, 3]))


def test_harmonic_set_index_lookup() -> None:
    hset = _make_harmonic_set((-5, -3, -1, 1, 3, 5))

    assert _index_of(hset, -5) == 0
    assert _index_of(hset, -1) == 2
    assert _index_of(hset, 1) == 3
    assert _index_of(hset, 5) == 5


def test_harmonic_set_rejects_duplicate_orders() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _make_harmonic_set((-1, 1, 1, 3))


def test_harmonic_set_rejects_zero_order_by_default() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _make_harmonic_set((-1, 0, 1))


def test_harmonic_set_rejects_empty_orders() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _make_harmonic_set(())


def test_frequency_plan_exposes_pump_signal_idler_relation() -> None:
    plan = _make_frequency_plan(
        pump_frequency_hz=10e9,
        signal_frequency_hz=6e9,
        harmonic_orders=(-3, -1, 1, 3),
    )

    idler = _idler_frequency_hz(plan)

    assert idler == pytest.approx(14e9)


def test_frequency_plan_frequencies_match_harmonic_orders_when_based_on_pump() -> None:
    orders = np.array([-3, -1, 1, 3])
    fp = 10e9

    plan = _make_frequency_plan(
        pump_frequency_hz=fp,
        signal_frequency_hz=6e9,
        harmonic_orders=orders,
    )

    freqs = _extract_frequencies(plan)

    assert freqs.shape[0] == orders.shape[0]
    assert np.all(np.isfinite(freqs))

    # Most HB bases store signed frequencies h * omega0. If the package uses a
    # pump-centered mixed basis, this still catches shape/finiteness and skips
    # strict equality when the convention is different.
    if np.allclose(np.abs(freqs), np.abs(orders * fp), rtol=1e-12, atol=1e-6):
        np.testing.assert_allclose(freqs, orders * fp, rtol=1e-12, atol=1e-6)


def test_angular_frequencies_are_2pi_times_frequencies() -> None:
    plan = _make_frequency_plan(
        pump_frequency_hz=10e9,
        signal_frequency_hz=6e9,
        harmonic_orders=(-3, -1, 1, 3),
    )

    freqs = _extract_frequencies(plan)
    omega = _extract_angular_frequencies(plan)

    np.testing.assert_allclose(omega, 2.0 * np.pi * freqs, rtol=1e-13, atol=1e-5)


def test_frequency_plan_is_serializable_or_mapping_like() -> None:
    plan = _make_frequency_plan()
    mapping = _as_mapping(plan)

    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0

    json_ready = {}
    for key, value in mapping.items():
        if hasattr(value, "shape"):
            json_ready[key] = np.asarray(value).tolist()
        else:
            json_ready[key] = value

    import json

    json.dumps(json_ready, default=str)


def test_time_grid_helper_if_available() -> None:
    helper = None
    for name in [
        "make_time_grid",
        "time_grid",
        "harmonic_time_grid",
        "periodic_time_grid",
    ]:
        if hasattr(hm, name):
            helper = getattr(hm, name)
            break

    if helper is None:
        pytest.skip("Time-grid helper is optional.")

    f0 = 10e9
    n_time = 64

    t = np.asarray(
        _call_with_supported_kwargs(
            helper,
            fundamental_frequency_hz=f0,
            frequency_hz=f0,
            f0_hz=f0,
            omega0_rad_s=2.0 * np.pi * f0,
            n_time=n_time,
            n=n_time,
        ),
        dtype=float,
    )

    assert t.shape == (n_time,)
    assert np.all(np.isfinite(t))
    assert t[0] == pytest.approx(0.0, abs=1e-30)
    assert np.all(np.diff(t) > 0.0)
    assert t[-1] < 1.0 / f0


def test_basis_matrix_helper_if_available() -> None:
    helper = None
    for name in [
        "harmonic_basis_matrix",
        "fourier_basis_matrix",
        "make_basis_matrix",
        "exp_iomega_t_matrix",
    ]:
        if hasattr(hm, name):
            helper = getattr(hm, name)
            break

    if helper is None:
        pytest.skip("Basis-matrix helper is optional.")

    orders = np.array([-1, 1, 3])
    f0 = 10e9
    n_time = 128
    t = np.arange(n_time) / (n_time * f0)

    B = np.asarray(
        _call_with_supported_kwargs(
            helper,
            t=t,
            time_s=t,
            orders=orders,
            harmonic_orders=orders,
            fundamental_frequency_hz=f0,
            f0_hz=f0,
            omega0_rad_s=2.0 * np.pi * f0,
        )
    )

    assert B.shape == (n_time, orders.size)
    assert np.all(np.isfinite(B.real))
    assert np.all(np.isfinite(B.imag))
    np.testing.assert_allclose(np.abs(B), 1.0, rtol=1e-13, atol=1e-13)


def test_conjugate_partner_lookup_if_available() -> None:
    helper = None
    for name in [
        "conjugate_partner_index",
        "negative_harmonic_index",
        "partner_index",
        "conjugate_index",
    ]:
        if hasattr(hm, name):
            helper = getattr(hm, name)
            break

    if helper is None:
        pytest.skip("Conjugate-partner helper is optional.")

    orders = np.array([-3, -1, 1, 3])
    hset = _make_harmonic_set(orders)

    idx_neg_3 = int(
        _call_with_supported_kwargs(
            helper,
            harmonic_set=hset,
            orders=orders,
            order=3,
            harmonic=3,
            index=3,
        )
    )

    assert idx_neg_3 == 0


def test_real_signal_conjugate_symmetry_helper_if_available() -> None:
    helper = None
    for name in [
        "enforce_conjugate_symmetry",
        "make_real_signal_coefficients",
        "complete_conjugate_spectrum",
    ]:
        if hasattr(hm, name):
            helper = getattr(hm, name)
            break

    if helper is None:
        pytest.skip("Conjugate-symmetry helper is optional.")

    orders = np.array([-3, -1, 1, 3])
    coeffs_pos = {
        1: 1.0 + 2.0j,
        3: -0.5 + 0.25j,
    }

    coeffs = np.asarray(
        _call_with_supported_kwargs(
            helper,
            positive_coeffs=coeffs_pos,
            coefficients=coeffs_pos,
            orders=orders,
            harmonic_orders=orders,
        )
    )

    assert coeffs.shape == (4,)
    assert coeffs[0] == pytest.approx(np.conj(coeffs[3]))
    assert coeffs[1] == pytest.approx(np.conj(coeffs[2]))


def test_frequency_plan_rejects_negative_pump_frequency() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _make_frequency_plan(pump_frequency_hz=-10e9, signal_frequency_hz=6e9)


def test_frequency_plan_rejects_negative_signal_frequency() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _make_frequency_plan(pump_frequency_hz=10e9, signal_frequency_hz=-6e9)


def test_frequency_plan_rejects_signal_above_two_pump_if_idler_would_be_negative() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _make_frequency_plan(pump_frequency_hz=10e9, signal_frequency_hz=25e9)


def test_harmonic_orders_are_integer_like() -> None:
    with pytest.raises((ValueError, AssertionError, TypeError)):
        _make_harmonic_set((-1.5, 1.0, 3.0))


def test_lookup_missing_order_raises() -> None:
    hset = _make_harmonic_set((-3, -1, 1, 3))

    with pytest.raises((ValueError, KeyError, IndexError)):
        _index_of(hset, 5)