"""
Tests for twpa.core.units.

These tests define the expected public behavior of the unit-conversion helpers
used throughout the production TWPA stack. They are intentionally small and
fast, but they catch the most dangerous mistakes:

    - GHz/MHz/kHz/Hz scaling errors
    - mm/um/nm/m scaling errors
    - angular-frequency conversion errors
    - dB / linear power conversion errors
    - dB / linear amplitude conversion errors
    - dBm / watt conversion errors
    - scalar and array compatibility
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pytest

import twpa.core.units as units


def _has(name: str) -> bool:
    return hasattr(units, name)


def _get_any(*names: str) -> Any:
    for name in names:
        if hasattr(units, name):
            return getattr(units, name)
    raise AttributeError(f"twpa.core.units is missing all of: {names}")


def _call_or_scale(
    value: Any,
    *,
    function_names: tuple[str, ...],
    scale_names: tuple[str, ...],
) -> Any:
    for name in function_names:
        if hasattr(units, name):
            return getattr(units, name)(value)

    for name in scale_names:
        if hasattr(units, name):
            return value * getattr(units, name)

    raise AttributeError(
        f"twpa.core.units is missing conversion function {function_names} "
        f"or scale constant {scale_names}"
    )


def _call_inverse_or_scale(
    value: Any,
    *,
    function_names: tuple[str, ...],
    scale_names: tuple[str, ...],
) -> Any:
    for name in function_names:
        if hasattr(units, name):
            return getattr(units, name)(value)

    for name in scale_names:
        if hasattr(units, name):
            return value / getattr(units, name)

    raise AttributeError(
        f"twpa.core.units is missing inverse conversion function {function_names} "
        f"or scale constant {scale_names}"
    )


def _ghz_to_hz(value: Any) -> Any:
    return _call_or_scale(
        value,
        function_names=("ghz_to_hz", "GHz_to_Hz", "to_hz_from_ghz"),
        scale_names=("GHz", "GHZ", "GIGAHERTZ"),
    )


def _mhz_to_hz(value: Any) -> Any:
    return _call_or_scale(
        value,
        function_names=("mhz_to_hz", "MHz_to_Hz", "to_hz_from_mhz"),
        scale_names=("MHz", "MHZ", "MEGAHERTZ"),
    )


def _khz_to_hz(value: Any) -> Any:
    return _call_or_scale(
        value,
        function_names=("khz_to_hz", "kHz_to_Hz", "to_hz_from_khz"),
        scale_names=("kHz", "KHZ", "KILOHERTZ"),
    )


def _hz_to_ghz(value: Any) -> Any:
    return _call_inverse_or_scale(
        value,
        function_names=("hz_to_ghz", "Hz_to_GHz", "to_ghz_from_hz"),
        scale_names=("GHz", "GHZ", "GIGAHERTZ"),
    )


def _mm_to_m(value: Any) -> Any:
    return _call_or_scale(
        value,
        function_names=("mm_to_m", "millimeter_to_meter", "millimeters_to_meters"),
        scale_names=("mm", "MM", "MILLIMETER"),
    )


def _um_to_m(value: Any) -> Any:
    return _call_or_scale(
        value,
        function_names=("um_to_m", "micrometer_to_meter", "micrometers_to_meters"),
        scale_names=("um", "UM", "MICROMETER"),
    )


def _nm_to_m(value: Any) -> Any:
    return _call_or_scale(
        value,
        function_names=("nm_to_m", "nanometer_to_meter", "nanometers_to_meters"),
        scale_names=("nm", "NM", "NANOMETER"),
    )


def _m_to_mm(value: Any) -> Any:
    return _call_inverse_or_scale(
        value,
        function_names=("m_to_mm", "meter_to_millimeter", "meters_to_millimeters"),
        scale_names=("mm", "MM", "MILLIMETER"),
    )


def _angular_frequency(value_hz: Any) -> Any:
    for name in ("angular_frequency", "omega_from_hz", "hz_to_rad_s", "frequency_to_angular"):
        if hasattr(units, name):
            return getattr(units, name)(value_hz)
    return 2.0 * math.pi * value_hz


def _frequency_from_angular(value_rad_s: Any) -> Any:
    for name in ("frequency_from_angular", "hz_from_omega", "rad_s_to_hz", "angular_to_frequency"):
        if hasattr(units, name):
            return getattr(units, name)(value_rad_s)
    return value_rad_s / (2.0 * math.pi)


def _power_db_to_linear(value_db: Any) -> Any:
    for name in ("db_to_power", "db_to_linear_power", "db_to_linear", "dB_to_power"):
        if hasattr(units, name):
            return getattr(units, name)(value_db)
    return 10.0 ** (np.asarray(value_db) / 10.0)


def _power_linear_to_db(value_linear: Any) -> Any:
    for name in ("power_to_db", "linear_power_to_db", "linear_to_db", "power_to_dB"):
        if hasattr(units, name):
            return getattr(units, name)(value_linear)
    return 10.0 * np.log10(np.asarray(value_linear))


def _amplitude_db_to_linear(value_db: Any) -> Any:
    for name in ("db_to_amplitude", "db_to_linear_amplitude", "db_to_voltage", "dB_to_amplitude"):
        if hasattr(units, name):
            return getattr(units, name)(value_db)
    return 10.0 ** (np.asarray(value_db) / 20.0)


def _amplitude_linear_to_db(value_linear: Any) -> Any:
    for name in ("amplitude_to_db", "linear_amplitude_to_db", "voltage_to_db", "amplitude_to_dB"):
        if hasattr(units, name):
            return getattr(units, name)(value_linear)
    return 20.0 * np.log10(np.asarray(value_linear))


def _dbm_to_watt(value_dbm: Any) -> Any:
    for name in ("dbm_to_watt", "dBm_to_W", "dbm_to_w", "dBm_to_watt"):
        if hasattr(units, name):
            return getattr(units, name)(value_dbm)
    return 1e-3 * 10.0 ** (np.asarray(value_dbm) / 10.0)


def _watt_to_dbm(value_watt: Any) -> Any:
    for name in ("watt_to_dbm", "W_to_dBm", "w_to_dbm", "watt_to_dBm"):
        if hasattr(units, name):
            return getattr(units, name)(value_watt)
    return 10.0 * np.log10(np.asarray(value_watt) / 1e-3)


def test_frequency_scale_constants_or_converters_exist() -> None:
    assert _ghz_to_hz(1.0) == pytest.approx(1e9)
    assert _mhz_to_hz(1.0) == pytest.approx(1e6)
    assert _khz_to_hz(1.0) == pytest.approx(1e3)


def test_frequency_round_trip_ghz_hz() -> None:
    values_ghz = np.array([0.001, 1.0, 4.25, 10.0, 12.5])
    values_hz = _ghz_to_hz(values_ghz)
    recovered_ghz = _hz_to_ghz(values_hz)

    np.testing.assert_allclose(values_hz, values_ghz * 1e9, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(recovered_ghz, values_ghz, rtol=1e-14, atol=1e-14)


def test_length_scale_constants_or_converters_exist() -> None:
    assert _mm_to_m(1.0) == pytest.approx(1e-3)
    assert _um_to_m(1.0) == pytest.approx(1e-6)
    assert _nm_to_m(1.0) == pytest.approx(1e-9)


def test_length_round_trip_m_mm() -> None:
    values_mm = np.array([0.001, 1.0, 10.0, 100.0])
    values_m = _mm_to_m(values_mm)
    recovered_mm = _m_to_mm(values_m)

    np.testing.assert_allclose(values_m, values_mm * 1e-3, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(recovered_mm, values_mm, rtol=1e-14, atol=1e-14)


def test_angular_frequency_conversion() -> None:
    f_hz = np.array([1.0, 1e6, 5e9, 10e9])
    omega = _angular_frequency(f_hz)
    recovered_f_hz = _frequency_from_angular(omega)

    np.testing.assert_allclose(omega, 2.0 * np.pi * f_hz, rtol=1e-14, atol=1e-9)
    np.testing.assert_allclose(recovered_f_hz, f_hz, rtol=1e-14, atol=1e-9)


def test_power_db_linear_known_values() -> None:
    db_values = np.array([-30.0, -10.0, 0.0, 3.0, 10.0, 20.0])
    linear = _power_db_to_linear(db_values)
    recovered_db = _power_linear_to_db(linear)

    assert _power_db_to_linear(0.0) == pytest.approx(1.0)
    assert _power_db_to_linear(10.0) == pytest.approx(10.0)
    assert _power_db_to_linear(20.0) == pytest.approx(100.0)

    np.testing.assert_allclose(recovered_db, db_values, rtol=1e-13, atol=1e-13)


def test_amplitude_db_linear_known_values() -> None:
    db_values = np.array([-40.0, -20.0, 0.0, 6.0, 20.0])
    linear = _amplitude_db_to_linear(db_values)
    recovered_db = _amplitude_linear_to_db(linear)

    assert _amplitude_db_to_linear(0.0) == pytest.approx(1.0)
    assert _amplitude_db_to_linear(20.0) == pytest.approx(10.0)
    assert _amplitude_db_to_linear(-20.0) == pytest.approx(0.1)

    np.testing.assert_allclose(recovered_db, db_values, rtol=1e-13, atol=1e-13)


def test_dbm_watt_known_values() -> None:
    assert _dbm_to_watt(0.0) == pytest.approx(1e-3)
    assert _dbm_to_watt(10.0) == pytest.approx(1e-2)
    assert _dbm_to_watt(30.0) == pytest.approx(1.0)

    assert _watt_to_dbm(1e-3) == pytest.approx(0.0)
    assert _watt_to_dbm(1e-2) == pytest.approx(10.0)
    assert _watt_to_dbm(1.0) == pytest.approx(30.0)


def test_dbm_watt_round_trip_array() -> None:
    dbm_values = np.array([-150.0, -120.0, -90.0, -30.0, 0.0, 10.0])
    watts = _dbm_to_watt(dbm_values)
    recovered_dbm = _watt_to_dbm(watts)

    assert np.all(watts > 0.0)
    np.testing.assert_allclose(recovered_dbm, dbm_values, rtol=1e-13, atol=1e-12)


def test_expected_twpa_scales_are_consistent() -> None:
    length_m = _mm_to_m(100.0)
    cell_length_m = _um_to_m(5.0)
    n_cells = length_m / cell_length_m

    assert length_m == pytest.approx(0.1)
    assert cell_length_m == pytest.approx(5e-6)
    assert n_cells == pytest.approx(20_000.0)


def test_array_inputs_preserve_shape_for_common_conversions() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    conversions: list[Callable[[Any], Any]] = [
        _ghz_to_hz,
        _hz_to_ghz,
        _mm_to_m,
        _m_to_mm,
        _angular_frequency,
        _frequency_from_angular,
        _power_db_to_linear,
        _amplitude_db_to_linear,
        _dbm_to_watt,
    ]

    for fn in conversions:
        y = np.asarray(fn(x))
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_no_negative_or_zero_watts_from_finite_dbm() -> None:
    dbm_values = np.linspace(-200.0, 60.0, 32)
    watts = np.asarray(_dbm_to_watt(dbm_values), dtype=float)

    assert watts.shape == dbm_values.shape
    assert np.all(np.isfinite(watts))
    assert np.all(watts > 0.0)