"""
Tests for twpa.core.params.

These tests define the expected behavior of the central TWPA parameter layer:
physical constants, transmission-line parameter conversion, cell-level derived
quantities, pump/signal operating parameters, and serializable config objects.

The tests are intentionally API-tolerant. They accept either small helper
functions or dataclass-style objects, but they require the module to expose a
clear public way to build physically consistent parameters.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

import numpy as np
import pytest

import twpa.core.params as params


def _has(name: str) -> bool:
    return hasattr(params, name)


def _get_any(*names: str) -> Any:
    for name in names:
        if hasattr(params, name):
            return getattr(params, name)
    raise AttributeError(f"twpa.core.params is missing all of: {names}")


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    import inspect

    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


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


def _get_attr_or_key(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _make_line_params(
    *,
    z0_ohm: float = 50.0,
    phase_velocity_m_per_s: float = 1.2e8,
) -> Any:
    candidate_functions = [
        "make_line_params",
        "make_transmission_line_params",
        "line_params_from_z0_vp",
        "transmission_line_from_z0_vp",
        "distributed_lc_from_z0_vp",
    ]

    for name in candidate_functions:
        if hasattr(params, name):
            return _call_with_supported_kwargs(
                getattr(params, name),
                z0_ohm=z0_ohm,
                Z0_ohm=z0_ohm,
                z0=z0_ohm,
                phase_velocity_m_per_s=phase_velocity_m_per_s,
                vp_m_per_s=phase_velocity_m_per_s,
                v_phase_m_per_s=phase_velocity_m_per_s,
            )

    candidate_classes = [
        "LineParams",
        "TransmissionLineParams",
        "DistributedLineParams",
    ]

    for name in candidate_classes:
        if hasattr(params, name):
            cls = getattr(params, name)
            return _call_with_supported_kwargs(
                cls,
                z0_ohm=z0_ohm,
                Z0_ohm=z0_ohm,
                z0=z0_ohm,
                phase_velocity_m_per_s=phase_velocity_m_per_s,
                vp_m_per_s=phase_velocity_m_per_s,
                v_phase_m_per_s=phase_velocity_m_per_s,
            )

    raise AttributeError(
        "twpa.core.params must expose a line-parameter builder/class such as "
        "make_line_params, line_params_from_z0_vp, or LineParams."
    )


def _make_cell_params(
    *,
    cell_length_m: float = 5e-6,
    z0_ohm: float = 50.0,
    phase_velocity_m_per_s: float = 1.2e8,
) -> Any:
    candidate_functions = [
        "make_cell_params",
        "make_unit_cell_params",
        "cell_params_from_z0_vp",
        "unit_cell_from_z0_vp",
        "lumped_cell_from_z0_vp",
    ]

    for name in candidate_functions:
        if hasattr(params, name):
            return _call_with_supported_kwargs(
                getattr(params, name),
                cell_length_m=cell_length_m,
                dx_m=cell_length_m,
                length_m=cell_length_m,
                z0_ohm=z0_ohm,
                Z0_ohm=z0_ohm,
                z0=z0_ohm,
                phase_velocity_m_per_s=phase_velocity_m_per_s,
                vp_m_per_s=phase_velocity_m_per_s,
                v_phase_m_per_s=phase_velocity_m_per_s,
            )

    candidate_classes = [
        "CellParams",
        "UnitCellParams",
        "LumpedCellParams",
    ]

    for name in candidate_classes:
        if hasattr(params, name):
            cls = getattr(params, name)
            return _call_with_supported_kwargs(
                cls,
                cell_length_m=cell_length_m,
                dx_m=cell_length_m,
                length_m=cell_length_m,
                z0_ohm=z0_ohm,
                Z0_ohm=z0_ohm,
                z0=z0_ohm,
                phase_velocity_m_per_s=phase_velocity_m_per_s,
                vp_m_per_s=phase_velocity_m_per_s,
                v_phase_m_per_s=phase_velocity_m_per_s,
            )

    line = _make_line_params(
        z0_ohm=z0_ohm,
        phase_velocity_m_per_s=phase_velocity_m_per_s,
    )

    if hasattr(params, "cell_from_line_params"):
        return _call_with_supported_kwargs(
            params.cell_from_line_params,
            line_params=line,
            line=line,
            cell_length_m=cell_length_m,
            dx_m=cell_length_m,
            length_m=cell_length_m,
        )

    raise AttributeError(
        "twpa.core.params must expose a cell-parameter builder/class such as "
        "make_cell_params, cell_params_from_z0_vp, or CellParams."
    )


def _make_chip_params(
    *,
    length_m: float = 0.1,
    n_cells: int = 20_000,
    z0_ohm: float = 50.0,
    phase_velocity_m_per_s: float = 1.2e8,
) -> Any:
    candidate_functions = [
        "make_chip_params",
        "make_twpa_params",
        "make_device_params",
        "twpa_params_from_design",
    ]

    for name in candidate_functions:
        if hasattr(params, name):
            return _call_with_supported_kwargs(
                getattr(params, name),
                length_m=length_m,
                total_length_m=length_m,
                n_cells=n_cells,
                z0_ohm=z0_ohm,
                Z0_ohm=z0_ohm,
                phase_velocity_m_per_s=phase_velocity_m_per_s,
                vp_m_per_s=phase_velocity_m_per_s,
            )

    candidate_classes = [
        "TWPAParams",
        "DeviceParams",
        "ChipParams",
        "TWPADeviceParams",
    ]

    for name in candidate_classes:
        if hasattr(params, name):
            cls = getattr(params, name)
            return _call_with_supported_kwargs(
                cls,
                length_m=length_m,
                total_length_m=length_m,
                n_cells=n_cells,
                z0_ohm=z0_ohm,
                Z0_ohm=z0_ohm,
                phase_velocity_m_per_s=phase_velocity_m_per_s,
                vp_m_per_s=phase_velocity_m_per_s,
            )

    raise AttributeError(
        "twpa.core.params must expose a chip/device parameter builder/class such "
        "as make_twpa_params, make_chip_params, or TWPAParams."
    )


def _make_operating_point(
    *,
    pump_frequency_hz: float = 10e9,
    signal_frequency_hz: float = 6e9,
    pump_current_a: float = 50e-6,
    i_star_a: float = 5e-3,
) -> Any:
    candidate_functions = [
        "make_operating_point",
        "make_pump_signal_params",
        "make_drive_params",
        "operating_point",
    ]

    for name in candidate_functions:
        if hasattr(params, name):
            return _call_with_supported_kwargs(
                getattr(params, name),
                pump_frequency_hz=pump_frequency_hz,
                signal_frequency_hz=signal_frequency_hz,
                pump_current_a=pump_current_a,
                i_star_a=i_star_a,
                I_star_A=i_star_a,
            )

    candidate_classes = [
        "OperatingPoint",
        "PumpSignalParams",
        "DriveParams",
    ]

    for name in candidate_classes:
        if hasattr(params, name):
            cls = getattr(params, name)
            return _call_with_supported_kwargs(
                cls,
                pump_frequency_hz=pump_frequency_hz,
                signal_frequency_hz=signal_frequency_hz,
                pump_current_a=pump_current_a,
                i_star_a=i_star_a,
                I_star_A=i_star_a,
            )

    raise AttributeError(
        "twpa.core.params must expose an operating-point builder/class such as "
        "make_operating_point or OperatingPoint."
    )


def _extract_l_per_m(line: Any) -> float:
    value = _get_attr_or_key(
        line,
        "L_per_m_H",
        "l_per_m_H",
        "L_prime_H_per_m",
        "L_series_per_m_H",
        "inductance_per_m_H",
        "L_per_m",
    )
    if value is None:
        pair = _get_attr_or_key(line, "distributed_lc", "lc_per_m")
        if pair is not None:
            value = pair[0]
    if value is None:
        raise AttributeError("Line params do not expose inductance per meter.")
    return float(value)


def _extract_c_per_m(line: Any) -> float:
    value = _get_attr_or_key(
        line,
        "C_per_m_F",
        "c_per_m_F",
        "C_prime_F_per_m",
        "C_shunt_per_m_F",
        "capacitance_per_m_F",
        "C_per_m",
    )
    if value is None:
        pair = _get_attr_or_key(line, "distributed_lc", "lc_per_m")
        if pair is not None:
            value = pair[1]
    if value is None:
        raise AttributeError("Line params do not expose capacitance per meter.")
    return float(value)


def _extract_cell_l(cell: Any) -> float:
    value = _get_attr_or_key(
        cell,
        "L_series_H",
        "L_cell_H",
        "series_inductance_H",
        "inductance_H",
        "L_H",
    )
    if value is None:
        raise AttributeError("Cell params do not expose series inductance.")
    return float(value)


def _extract_cell_c(cell: Any) -> float:
    value = _get_attr_or_key(
        cell,
        "C_shunt_F",
        "C_cell_F",
        "shunt_capacitance_F",
        "capacitance_F",
        "C_F",
    )
    if value is None:
        raise AttributeError("Cell params do not expose shunt capacitance.")
    return float(value)


def test_fundamental_flux_quantum_constant_exists_and_is_reasonable() -> None:
    phi0 = _get_any("PHI0_Wb", "PHI0", "FLUX_QUANTUM_WB", "Phi0_Wb", "phi0_Wb")
    assert float(phi0) == pytest.approx(2.067833848e-15, rel=5e-8)


def test_reduced_flux_quantum_constant_if_exposed() -> None:
    phi0 = float(_get_any("PHI0_Wb", "PHI0", "FLUX_QUANTUM_WB", "Phi0_Wb", "phi0_Wb"))

    reduced_candidates = [
        "PHI0_REDUCED_Wb",
        "PHI0_OVER_2PI_Wb",
        "REDUCED_FLUX_QUANTUM_WB",
        "phi0_reduced_Wb",
    ]

    for name in reduced_candidates:
        if hasattr(params, name):
            reduced = float(getattr(params, name))
            assert reduced == pytest.approx(phi0 / (2.0 * math.pi), rel=1e-12)
            return

    pytest.skip("Reduced flux quantum constant is optional.")


def test_josephson_inductance_helper_if_exposed() -> None:
    candidate_names = [
        "josephson_inductance",
        "josephson_inductance_H",
        "lj_from_ic",
        "LJ_from_Ic",
        "critical_current_to_lj",
    ]

    fn = None
    for name in candidate_names:
        if hasattr(params, name):
            fn = getattr(params, name)
            break

    if fn is None:
        pytest.skip("Josephson inductance helper is optional in params.py.")

    ic = np.array([1e-6, 2e-6, 10e-6])
    lj = np.asarray(fn(ic), dtype=float)

    phi0 = float(_get_any("PHI0_Wb", "PHI0", "FLUX_QUANTUM_WB", "Phi0_Wb", "phi0_Wb"))
    expected = phi0 / (2.0 * math.pi * ic)

    np.testing.assert_allclose(lj, expected, rtol=1e-12, atol=0.0)
    assert lj.shape == ic.shape


def test_line_params_from_z0_and_phase_velocity() -> None:
    z0 = 50.0
    vp = 1.2e8
    line = _make_line_params(z0_ohm=z0, phase_velocity_m_per_s=vp)

    l_per_m = _extract_l_per_m(line)
    c_per_m = _extract_c_per_m(line)

    assert l_per_m == pytest.approx(z0 / vp, rel=1e-12)
    assert c_per_m == pytest.approx(1.0 / (z0 * vp), rel=1e-12)

    recovered_z0 = math.sqrt(l_per_m / c_per_m)
    recovered_vp = 1.0 / math.sqrt(l_per_m * c_per_m)

    assert recovered_z0 == pytest.approx(z0, rel=1e-12)
    assert recovered_vp == pytest.approx(vp, rel=1e-12)


def test_cell_params_from_z0_phase_velocity_and_cell_length() -> None:
    z0 = 50.0
    vp = 1.2e8
    dx = 5e-6

    cell = _make_cell_params(
        cell_length_m=dx,
        z0_ohm=z0,
        phase_velocity_m_per_s=vp,
    )

    l_cell = _extract_cell_l(cell)
    c_cell = _extract_cell_c(cell)

    expected_l = z0 * dx / vp
    expected_c = dx / (z0 * vp)

    assert l_cell == pytest.approx(expected_l, rel=1e-12)
    assert c_cell == pytest.approx(expected_c, rel=1e-12)


def test_cell_params_recover_z0_and_phase_velocity() -> None:
    z0 = 50.0
    vp = 1.2e8
    dx = 10e-6

    cell = _make_cell_params(
        cell_length_m=dx,
        z0_ohm=z0,
        phase_velocity_m_per_s=vp,
    )

    l_cell = _extract_cell_l(cell)
    c_cell = _extract_cell_c(cell)

    recovered_z0 = math.sqrt(l_cell / c_cell)
    recovered_vp = dx / math.sqrt(l_cell * c_cell)

    assert recovered_z0 == pytest.approx(z0, rel=1e-12)
    assert recovered_vp == pytest.approx(vp, rel=1e-12)


def test_100mm_20000_cell_design_has_expected_cell_length() -> None:
    chip = _make_chip_params(length_m=0.1, n_cells=20_000)

    length_m = _get_attr_or_key(chip, "length_m", "total_length_m")
    n_cells = _get_attr_or_key(chip, "n_cells", "num_cells")
    cell_length_m = _get_attr_or_key(chip, "cell_length_m", "dx_m", "unit_cell_length_m")

    assert int(n_cells) == 20_000
    assert float(length_m) == pytest.approx(0.1)

    if cell_length_m is None:
        cell_length_m = float(length_m) / int(n_cells)

    assert float(cell_length_m) == pytest.approx(5e-6)


def test_chip_params_have_consistent_total_lumped_lc() -> None:
    z0 = 50.0
    vp = 1.2e8
    length_m = 0.1
    n_cells = 20_000

    chip = _make_chip_params(
        length_m=length_m,
        n_cells=n_cells,
        z0_ohm=z0,
        phase_velocity_m_per_s=vp,
    )

    mapping = _as_mapping(chip)

    total_l = _get_attr_or_key(
        chip,
        "total_L_H",
        "L_total_H",
        "total_series_inductance_H",
        default=None,
    )
    total_c = _get_attr_or_key(
        chip,
        "total_C_F",
        "C_total_F",
        "total_shunt_capacitance_F",
        default=None,
    )

    if total_l is None:
        cell = _make_cell_params(
            cell_length_m=length_m / n_cells,
            z0_ohm=z0,
            phase_velocity_m_per_s=vp,
        )
        total_l = n_cells * _extract_cell_l(cell)

    if total_c is None:
        cell = _make_cell_params(
            cell_length_m=length_m / n_cells,
            z0_ohm=z0,
            phase_velocity_m_per_s=vp,
        )
        total_c = n_cells * _extract_cell_c(cell)

    assert float(total_l) == pytest.approx(z0 * length_m / vp, rel=1e-12)
    assert float(total_c) == pytest.approx(length_m / (z0 * vp), rel=1e-12)

    assert mapping is not None


def test_operating_point_exposes_pump_signal_and_idler_frequencies() -> None:
    op = _make_operating_point(
        pump_frequency_hz=10e9,
        signal_frequency_hz=6e9,
        pump_current_a=50e-6,
        i_star_a=5e-3,
    )

    pump = _get_attr_or_key(op, "pump_frequency_hz", "pump_f_hz", "fp_hz")
    signal = _get_attr_or_key(op, "signal_frequency_hz", "signal_f_hz", "fs_hz")
    idler = _get_attr_or_key(op, "idler_frequency_hz", "idler_f_hz", "fi_hz", default=None)

    assert float(pump) == pytest.approx(10e9)
    assert float(signal) == pytest.approx(6e9)

    if idler is None:
        idler = 2.0 * float(pump) - float(signal)

    assert float(idler) == pytest.approx(14e9)


def test_operating_point_exposes_pump_current_ratio() -> None:
    op = _make_operating_point(
        pump_frequency_hz=10e9,
        signal_frequency_hz=6e9,
        pump_current_a=50e-6,
        i_star_a=5e-3,
    )

    ratio = _get_attr_or_key(
        op,
        "pump_current_ratio",
        "pump_ratio",
        "ip_over_i_star",
        default=None,
    )

    if ratio is None:
        pump_current = _get_attr_or_key(op, "pump_current_a", "pump_current_A", "I_pump_A")
        i_star = _get_attr_or_key(op, "i_star_a", "I_star_A", "Istar_A")
        ratio = float(pump_current) / float(i_star)

    assert float(ratio) == pytest.approx(0.01)


def test_parameter_objects_are_serializable_or_dataclass_like() -> None:
    objects = [
        _make_line_params(),
        _make_cell_params(),
        _make_chip_params(),
        _make_operating_point(),
    ]

    for obj in objects:
        mapping = _as_mapping(obj)
        assert isinstance(mapping, Mapping)
        assert len(mapping) > 0

        json_ready = {}
        for key, value in mapping.items():
            if isinstance(value, np.ndarray):
                json_ready[key] = value.tolist()
            elif hasattr(value, "shape"):
                json_ready[key] = np.asarray(value).tolist()
            else:
                json_ready[key] = value

        import json

        json.dumps(json_ready, default=str)


def test_invalid_negative_lengths_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _make_cell_params(cell_length_m=-5e-6)


def test_invalid_zero_impedance_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _make_line_params(z0_ohm=0.0)


def test_invalid_zero_phase_velocity_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _make_line_params(phase_velocity_m_per_s=0.0)


def test_vectorized_line_conversion_if_exposed() -> None:
    fn_names = [
        "line_params_from_z0_vp",
        "distributed_lc_from_z0_vp",
        "z0_vp_to_lc",
    ]

    fn = None
    for name in fn_names:
        if hasattr(params, name):
            fn = getattr(params, name)
            break

    if fn is None:
        pytest.skip("Vectorized line conversion helper is optional.")

    z0 = np.array([25.0, 50.0, 75.0])
    vp = np.array([1.0e8, 1.2e8, 1.5e8])

    out = _call_with_supported_kwargs(
        fn,
        z0_ohm=z0,
        Z0_ohm=z0,
        z0=z0,
        phase_velocity_m_per_s=vp,
        vp_m_per_s=vp,
    )

    if isinstance(out, tuple) and len(out) >= 2:
        l_per_m, c_per_m = out[0], out[1]
    else:
        l_per_m = _get_attr_or_key(out, "L_per_m_H", "L_prime_H_per_m", "L_per_m")
        c_per_m = _get_attr_or_key(out, "C_per_m_F", "C_prime_F_per_m", "C_per_m")

    np.testing.assert_allclose(l_per_m, z0 / vp, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(c_per_m, 1.0 / (z0 * vp), rtol=1e-12, atol=0.0)