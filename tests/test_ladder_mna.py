"""
Tests for twpa.linear.ladder_mna.

These tests define the expected linear modified-nodal-analysis layer for
distributed TWPA ladders.

Reference ladder convention
---------------------------
The canonical test ladder has:

    nodes: 0, 1, ..., N
    series inductors L_k between node k and node k+1, for k = 0, ..., N-1
    shunt capacitors C_n from node n to ground, for n = 0, ..., N

At angular frequency omega,

    y_L,k = 1 / (i omega L_k)
    y_C,n = i omega C_n

and the nodal admittance matrix is tridiagonal:

    Y[n,n] += y_C,n
    Y[k,k] += y_L,k
    Y[k+1,k+1] += y_L,k
    Y[k,k+1] -= y_L,k
    Y[k+1,k] -= y_L,k

The tests are API-tolerant in naming, but require the module to expose a public
way to build the ladder admittance matrix. Solver, ABCD, and S-parameter helpers
are tested when present.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import twpa.linear.ladder_mna as lmna


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


def _make_ladder_params(
    *,
    n_cells: int = 4,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
) -> Any:
    for name in [
        "LadderMNAParams",
        "LadderParams",
        "LinearLadderParams",
        "DistributedLadderParams",
        "LCMNAParams",
    ]:
        if hasattr(lmna, name):
            cls = getattr(lmna, name)
            return _call_with_supported_kwargs(
                cls,
                n_cells=n_cells,
                num_cells=n_cells,
                N=n_cells,
                L_series_H=L_series_H,
                L_H=L_series_H,
                series_inductance_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                shunt_capacitance_F=C_shunt_F,
            )

    for name in [
        "make_ladder_params",
        "make_linear_ladder_params",
        "make_lc_ladder_params",
        "make_distributed_ladder_params",
    ]:
        if hasattr(lmna, name):
            fn = getattr(lmna, name)
            return _call_with_supported_kwargs(
                fn,
                n_cells=n_cells,
                num_cells=n_cells,
                N=n_cells,
                L_series_H=L_series_H,
                L_H=L_series_H,
                series_inductance_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                shunt_capacitance_F=C_shunt_F,
            )

    return {
        "n_cells": n_cells,
        "L_series_H": L_series_H,
        "C_shunt_F": C_shunt_F,
    }


def _broadcast_series_l(L_series_H: float | Sequence[float], n_cells: int) -> np.ndarray:
    arr = np.asarray(L_series_H, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_cells, float(arr), dtype=float)
    if arr.shape != (n_cells,):
        raise ValueError(f"L_series_H must have shape ({n_cells},), got {arr.shape}")
    return arr


def _broadcast_shunt_c(C_shunt_F: float | Sequence[float], n_cells: int) -> np.ndarray:
    arr = np.asarray(C_shunt_F, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_cells + 1, float(arr), dtype=float)
    if arr.shape == (n_cells,):
        # Some implementations store one shunt C per cell. For the canonical
        # nodal matrix we split this into end half-caps plus interior sums.
        caps = np.zeros(n_cells + 1, dtype=float)
        caps[:-1] += 0.5 * arr
        caps[1:] += 0.5 * arr
        arr = caps
    if arr.shape != (n_cells + 1,):
        raise ValueError(f"C_shunt_F must have shape ({n_cells + 1},), got {arr.shape}")
    return arr


def _manual_y_matrix(
    *,
    omega_rad_s: float,
    n_cells: int,
    L_series_H: float | Sequence[float],
    C_shunt_F: float | Sequence[float],
) -> np.ndarray:
    if omega_rad_s <= 0.0:
        raise ValueError("omega_rad_s must be positive")
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    L = _broadcast_series_l(L_series_H, n_cells)
    C = _broadcast_shunt_c(C_shunt_F, n_cells)

    if np.any(L <= 0.0):
        raise ValueError("all series inductances must be positive")
    if np.any(C < 0.0):
        raise ValueError("all shunt capacitances must be non-negative")

    n_nodes = n_cells + 1
    Y = np.zeros((n_nodes, n_nodes), dtype=np.complex128)

    Y[np.arange(n_nodes), np.arange(n_nodes)] += 1j * omega_rad_s * C

    for k in range(n_cells):
        y_l = 1.0 / (1j * omega_rad_s * L[k])
        Y[k, k] += y_l
        Y[k + 1, k + 1] += y_l
        Y[k, k + 1] -= y_l
        Y[k + 1, k] -= y_l

    return Y


def _build_y_matrix(
    *,
    omega_rad_s: float,
    n_cells: int = 4,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
) -> np.ndarray:
    params = _make_ladder_params(
        n_cells=n_cells,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
    )

    for name in [
        "build_admittance_matrix",
        "admittance_matrix",
        "ladder_admittance_matrix",
        "build_ladder_y_matrix",
        "build_Y_matrix",
        "mna_admittance_matrix",
        "linear_ladder_admittance",
    ]:
        if hasattr(lmna, name):
            out = _call_with_supported_kwargs(
                getattr(lmna, name),
                omega_rad_s=omega_rad_s,
                omega=omega_rad_s,
                angular_frequency_rad_s=omega_rad_s,
                frequency_hz=omega_rad_s / (2.0 * np.pi),
                n_cells=n_cells,
                num_cells=n_cells,
                N=n_cells,
                L_series_H=L_series_H,
                L_H=L_series_H,
                series_inductance_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                shunt_capacitance_F=C_shunt_F,
                params=params,
                ladder=params,
            )
            value = _get_attr_or_key(out, "Y", "Y_matrix", "admittance_matrix", default=out)
            return np.asarray(value, dtype=np.complex128)

    for method_name in [
        "admittance_matrix",
        "build_admittance_matrix",
        "Y_matrix",
        "build_Y_matrix",
    ]:
        method = _get_attr_or_key(params, method_name, default=None)
        if callable(method):
            out = _call_with_supported_kwargs(
                method,
                omega_rad_s=omega_rad_s,
                omega=omega_rad_s,
                frequency_hz=omega_rad_s / (2.0 * np.pi),
            )
            return np.asarray(out, dtype=np.complex128)

    raise AttributeError(
        "twpa.linear.ladder_mna must expose an admittance-matrix builder such as "
        "build_admittance_matrix(), ladder_admittance_matrix(), or mna_admittance_matrix()."
    )


def _solve_ladder(
    current_injection_A: Any,
    *,
    omega_rad_s: float,
    n_cells: int = 4,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
) -> tuple[np.ndarray, Any]:
    params = _make_ladder_params(
        n_cells=n_cells,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
    )

    for name in [
        "solve_ladder",
        "solve_ladder_mna",
        "solve_mna",
        "solve_node_voltages",
        "solve_linear_ladder",
    ]:
        if hasattr(lmna, name):
            out = _call_with_supported_kwargs(
                getattr(lmna, name),
                current_injection_A=current_injection_A,
                I_inj_A=current_injection_A,
                source_current_A=current_injection_A,
                b=current_injection_A,
                rhs=current_injection_A,
                omega_rad_s=omega_rad_s,
                omega=omega_rad_s,
                angular_frequency_rad_s=omega_rad_s,
                frequency_hz=omega_rad_s / (2.0 * np.pi),
                n_cells=n_cells,
                num_cells=n_cells,
                N=n_cells,
                L_series_H=L_series_H,
                L_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                params=params,
                ladder=params,
            )
            V = _get_attr_or_key(out, "V", "node_voltages", "voltages", "x", default=None)
            if V is not None:
                return np.asarray(V, dtype=np.complex128), out
            return np.asarray(out, dtype=np.complex128), out

    pytest.skip("Ladder MNA solve helper is optional if admittance matrix is available.")


def _abcd_from_ladder(
    *,
    omega_rad_s: float,
    n_cells: int = 4,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
) -> np.ndarray:
    params = _make_ladder_params(
        n_cells=n_cells,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
    )

    for name in [
        "ladder_abcd",
        "abcd_from_ladder",
        "mna_to_abcd",
        "linear_ladder_abcd",
        "abcd_matrix",
    ]:
        if hasattr(lmna, name):
            out = _call_with_supported_kwargs(
                getattr(lmna, name),
                omega_rad_s=omega_rad_s,
                omega=omega_rad_s,
                angular_frequency_rad_s=omega_rad_s,
                frequency_hz=omega_rad_s / (2.0 * np.pi),
                n_cells=n_cells,
                num_cells=n_cells,
                N=n_cells,
                L_series_H=L_series_H,
                L_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                params=params,
                ladder=params,
            )
            value = _get_attr_or_key(out, "ABCD", "abcd", "matrix", default=out)
            return np.asarray(value, dtype=np.complex128)

    pytest.skip("ABCD export from ladder MNA is optional.")


def _s_from_ladder(
    *,
    omega_rad_s: float,
    n_cells: int = 4,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
    z0_ohm: float = 50.0,
) -> np.ndarray:
    params = _make_ladder_params(
        n_cells=n_cells,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
    )

    for name in [
        "ladder_sparameters",
        "sparameters_from_ladder",
        "ladder_sparams",
        "linear_ladder_sparameters",
        "mna_to_sparameters",
    ]:
        if hasattr(lmna, name):
            out = _call_with_supported_kwargs(
                getattr(lmna, name),
                omega_rad_s=omega_rad_s,
                omega=omega_rad_s,
                angular_frequency_rad_s=omega_rad_s,
                frequency_hz=omega_rad_s / (2.0 * np.pi),
                n_cells=n_cells,
                num_cells=n_cells,
                N=n_cells,
                L_series_H=L_series_H,
                L_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                z0_ohm=z0_ohm,
                Z0_ohm=z0_ohm,
                reference_impedance_ohm=z0_ohm,
                params=params,
                ladder=params,
            )
            value = _get_attr_or_key(out, "S", "s", "sparameters", "sparams", default=out)
            return np.asarray(value, dtype=np.complex128)

    pytest.skip("S-parameter export from ladder MNA is optional.")


def _manual_unit_cell_abcd(omega_rad_s: float, L_H: float, C_F: float) -> np.ndarray:
    Z = 1j * omega_rad_s * L_H
    Y = 1j * omega_rad_s * C_F

    series = np.array([[1.0, Z], [0.0, 1.0]], dtype=np.complex128)
    shunt = np.array([[1.0, 0.0], [Y, 1.0]], dtype=np.complex128)

    return series @ shunt


def _manual_cascade_abcd(matrices: Sequence[np.ndarray]) -> np.ndarray:
    out = np.eye(2, dtype=np.complex128)
    for matrix in matrices:
        out = out @ np.asarray(matrix, dtype=np.complex128)
    return out


def _abcd_to_s(abcd: np.ndarray, z0_ohm: float = 50.0) -> np.ndarray:
    A, B, C, D = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
    denom = A + B / z0_ohm + C * z0_ohm + D

    return np.array(
        [
            [(A + B / z0_ohm - C * z0_ohm - D) / denom, 2.0 * (A * D - B * C) / denom],
            [2.0 / denom, (-A + B / z0_ohm - C * z0_ohm + D) / denom],
        ],
        dtype=np.complex128,
    )


def test_ladder_params_object_is_mapping_or_dataclass_like() -> None:
    params = _make_ladder_params(n_cells=4, L_series_H=1e-12, C_shunt_F=2e-15)

    mapping = _as_mapping(params)
    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0

    n_cells = _get_attr_or_key(params, "n_cells", "num_cells", "N", default=4)
    assert int(n_cells) == 4


def test_admittance_matrix_matches_manual_uniform_ladder() -> None:
    omega = 2.0 * np.pi * 6e9
    n_cells = 4
    L = 1.2e-12
    C = 1.8e-15

    Y = _build_y_matrix(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=C,
    )
    expected = _manual_y_matrix(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=C,
    )

    assert Y.shape == (n_cells + 1, n_cells + 1)
    np.testing.assert_allclose(Y, expected, rtol=1e-12, atol=1e-18)


def test_admittance_matrix_matches_manual_nonuniform_ladder() -> None:
    omega = 2.0 * np.pi * 7e9
    n_cells = 3
    L = np.array([1.0, 1.2, 1.4]) * 1e-12
    C = np.array([0.7, 1.0, 1.1, 0.8]) * 1e-15

    Y = _build_y_matrix(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=C,
    )
    expected = _manual_y_matrix(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=C,
    )

    np.testing.assert_allclose(Y, expected, rtol=1e-12, atol=1e-18)


def test_admittance_matrix_is_symmetric_for_reciprocal_ladder() -> None:
    Y = _build_y_matrix(
        omega_rad_s=2.0 * np.pi * 6e9,
        n_cells=5,
        L_series_H=np.linspace(1.0, 1.4, 5) * 1e-12,
        C_shunt_F=np.linspace(0.8, 1.2, 6) * 1e-15,
    )

    np.testing.assert_allclose(Y, Y.T, rtol=1e-14, atol=1e-18)


def test_admittance_matrix_is_tridiagonal() -> None:
    n_cells = 6
    Y = _build_y_matrix(
        omega_rad_s=2.0 * np.pi * 5e9,
        n_cells=n_cells,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
    )

    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            if abs(i - j) > 1:
                assert Y[i, j] == pytest.approx(0.0 + 0.0j, abs=1e-24)


def test_admittance_matrix_has_expected_row_sums_from_shunt_caps_only() -> None:
    omega = 2.0 * np.pi * 5e9
    n_cells = 5
    C = np.linspace(1.0, 2.0, n_cells + 1) * 1e-15

    Y = _build_y_matrix(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=1e-12,
        C_shunt_F=C,
    )

    row_sums = np.sum(Y, axis=1)
    expected = 1j * omega * C

    np.testing.assert_allclose(row_sums, expected, rtol=1e-10, atol=1e-15)


def test_admittance_matrix_without_shunt_caps_has_zero_row_sums_if_supported() -> None:
    omega = 2.0 * np.pi * 5e9

    try:
        Y = _build_y_matrix(
            omega_rad_s=omega,
            n_cells=4,
            L_series_H=1e-12,
            C_shunt_F=0.0,
        )
    except (ValueError, AssertionError):
        pytest.skip("Zero shunt capacitance is intentionally unsupported.")

    np.testing.assert_allclose(np.sum(Y, axis=1), 0.0, rtol=0.0, atol=1e-18)


def test_admittance_matrix_scales_with_frequency_in_expected_way() -> None:
    n_cells = 3
    L = 1e-12
    C = 2e-15
    omega1 = 2.0 * np.pi * 4e9
    omega2 = 2.0 * np.pi * 8e9

    Y1 = _build_y_matrix(omega_rad_s=omega1, n_cells=n_cells, L_series_H=L, C_shunt_F=C)
    Y2 = _build_y_matrix(omega_rad_s=omega2, n_cells=n_cells, L_series_H=L, C_shunt_F=C)

    expected1 = _manual_y_matrix(omega_rad_s=omega1, n_cells=n_cells, L_series_H=L, C_shunt_F=C)
    expected2 = _manual_y_matrix(omega_rad_s=omega2, n_cells=n_cells, L_series_H=L, C_shunt_F=C)

    np.testing.assert_allclose(Y1, expected1, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(Y2, expected2, rtol=1e-12, atol=1e-18)


def test_solver_matches_numpy_solve_if_available() -> None:
    omega = 2.0 * np.pi * 6e9
    n_cells = 4
    L = 1e-12
    C = 2e-15

    # Add a small conductance to ground by using nonzero C and nonzero omega;
    # this matrix is nonsingular for this finite-frequency grounded ladder.
    I = np.zeros(n_cells + 1, dtype=np.complex128)
    I[0] = 1e-6
    I[-1] = -1e-6

    V, _ = _solve_ladder(
        I,
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=C,
    )

    Y = _manual_y_matrix(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=C,
    )
    expected = np.linalg.solve(Y, I)

    np.testing.assert_allclose(V, expected, rtol=1e-9, atol=1e-15)


def test_solver_residual_is_small_if_available() -> None:
    omega = 2.0 * np.pi * 5e9
    n_cells = 5
    I = np.zeros(n_cells + 1, dtype=np.complex128)
    I[0] = 1e-6
    I[2] = -0.25e-6
    I[-1] = -0.75e-6

    V, _ = _solve_ladder(
        I,
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=np.linspace(1.0, 1.5, n_cells) * 1e-12,
        C_shunt_F=np.linspace(1.0, 2.0, n_cells + 1) * 1e-15,
    )

    Y = _build_y_matrix(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=np.linspace(1.0, 1.5, n_cells) * 1e-12,
        C_shunt_F=np.linspace(1.0, 2.0, n_cells + 1) * 1e-15,
    )

    residual = Y @ V - I

    assert np.linalg.norm(residual) < 1e-12


def test_zero_current_injection_gives_zero_voltage_if_solver_available() -> None:
    omega = 2.0 * np.pi * 5e9
    n_cells = 4
    I = np.zeros(n_cells + 1, dtype=np.complex128)

    V, _ = _solve_ladder(
        I,
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
    )

    np.testing.assert_allclose(V, np.zeros_like(V), rtol=0.0, atol=1e-18)


def test_solver_output_object_is_serializable_or_tuple_if_available() -> None:
    omega = 2.0 * np.pi * 5e9
    n_cells = 3
    I = np.zeros(n_cells + 1, dtype=np.complex128)
    I[0] = 1e-6
    I[-1] = -1e-6

    _, result = _solve_ladder(
        I,
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
    )

    if isinstance(result, np.ndarray):
        pytest.skip("Raw ndarray solver result is acceptable and has no metadata.")

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


def test_abcd_single_cell_matches_series_then_shunt_if_available() -> None:
    omega = 2.0 * np.pi * 6e9
    L = 1e-12
    C = 2e-15

    A = _abcd_from_ladder(
        omega_rad_s=omega,
        n_cells=1,
        L_series_H=L,
        C_shunt_F=np.array([0.0, C]),
    )

    expected = _manual_unit_cell_abcd(omega, L, C)

    assert A.shape[-2:] == (2, 2)
    np.testing.assert_allclose(A, expected, rtol=1e-10, atol=1e-15)


def test_abcd_uniform_ladder_matches_cascaded_unit_cells_if_available() -> None:
    omega = 2.0 * np.pi * 5e9
    n_cells = 4
    L = 1e-12
    C = 2e-15

    A = _abcd_from_ladder(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=np.array([0.0, *([C] * n_cells)]),
    )

    expected = _manual_cascade_abcd(
        [_manual_unit_cell_abcd(omega, L, C) for _ in range(n_cells)]
    )

    np.testing.assert_allclose(A, expected, rtol=1e-9, atol=1e-14)


def test_abcd_of_reciprocal_ladder_has_unit_determinant_if_available() -> None:
    A = _abcd_from_ladder(
        omega_rad_s=2.0 * np.pi * 6e9,
        n_cells=5,
        L_series_H=np.linspace(1.0, 1.3, 5) * 1e-12,
        C_shunt_F=np.array([0.0, *list(np.linspace(1.5, 2.0, 5) * 1e-15)]),
    )

    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]

    assert det == pytest.approx(1.0 + 0.0j, rel=1e-9, abs=1e-12)


def test_sparameters_match_abcd_conversion_if_both_available() -> None:
    omega = 2.0 * np.pi * 6e9
    z0 = 50.0
    n_cells = 3
    L = 1e-12
    C = 2e-15

    A = _abcd_from_ladder(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=np.array([0.0, C, C, C]),
    )
    S = _s_from_ladder(
        omega_rad_s=omega,
        n_cells=n_cells,
        L_series_H=L,
        C_shunt_F=np.array([0.0, C, C, C]),
        z0_ohm=z0,
    )

    expected = _abcd_to_s(A, z0_ohm=z0)

    assert S.shape[-2:] == (2, 2)
    np.testing.assert_allclose(S, expected, rtol=1e-8, atol=1e-12)


def test_lossless_ladder_sparameters_are_reciprocal_if_available() -> None:
    S = _s_from_ladder(
        omega_rad_s=2.0 * np.pi * 6e9,
        n_cells=4,
        L_series_H=1e-12,
        C_shunt_F=np.array([0.0, 2e-15, 2e-15, 2e-15, 2e-15]),
        z0_ohm=50.0,
    )

    assert S[0, 1] == pytest.approx(S[1, 0], rel=1e-8, abs=1e-12)


def test_lossless_ladder_sparameters_are_passive_if_available() -> None:
    S = _s_from_ladder(
        omega_rad_s=2.0 * np.pi * 6e9,
        n_cells=4,
        L_series_H=1e-12,
        C_shunt_F=np.array([0.0, 2e-15, 2e-15, 2e-15, 2e-15]),
        z0_ohm=50.0,
    )

    column_power = np.sum(np.abs(S) ** 2, axis=0)

    assert np.all(column_power <= 1.0 + 1e-8)
    assert np.all(column_power >= -1e-12)


def test_frequency_vectorization_if_supported() -> None:
    omegas = 2.0 * np.pi * np.array([4e9, 5e9, 6e9])
    n_cells = 3

    try:
        Y = _build_y_matrix(
            omega_rad_s=omegas,
            n_cells=n_cells,
            L_series_H=1e-12,
            C_shunt_F=2e-15,
        )
    except Exception as exc:
        pytest.skip(f"Vectorized frequency admittance is optional: {type(exc).__name__}: {exc}")

    assert Y.shape == (omegas.size, n_cells + 1, n_cells + 1)

    for idx, omega in enumerate(omegas):
        expected = _manual_y_matrix(
            omega_rad_s=float(omega),
            n_cells=n_cells,
            L_series_H=1e-12,
            C_shunt_F=2e-15,
        )
        np.testing.assert_allclose(Y[idx], expected, rtol=1e-12, atol=1e-18)


def test_large_ladder_matrix_has_expected_shape_and_finite_entries() -> None:
    n_cells = 200
    Y = _build_y_matrix(
        omega_rad_s=2.0 * np.pi * 6e9,
        n_cells=n_cells,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
    )

    assert Y.shape == (n_cells + 1, n_cells + 1)
    assert np.all(np.isfinite(Y.real))
    assert np.all(np.isfinite(Y.imag))


def test_invalid_nonpositive_n_cells_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _build_y_matrix(
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=0,
            L_series_H=1e-12,
            C_shunt_F=2e-15,
        )


def test_invalid_nonpositive_frequency_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _build_y_matrix(
            omega_rad_s=0.0,
            n_cells=4,
            L_series_H=1e-12,
            C_shunt_F=2e-15,
        )

    with pytest.raises((ValueError, AssertionError)):
        _build_y_matrix(
            omega_rad_s=-2.0 * np.pi * 5e9,
            n_cells=4,
            L_series_H=1e-12,
            C_shunt_F=2e-15,
        )


def test_invalid_nonpositive_series_inductance_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _build_y_matrix(
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=4,
            L_series_H=0.0,
            C_shunt_F=2e-15,
        )

    with pytest.raises((ValueError, AssertionError)):
        _build_y_matrix(
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=4,
            L_series_H=-1e-12,
            C_shunt_F=2e-15,
        )


def test_invalid_negative_shunt_capacitance_is_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _build_y_matrix(
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=4,
            L_series_H=1e-12,
            C_shunt_F=-2e-15,
        )


def test_invalid_array_lengths_are_rejected() -> None:
    with pytest.raises((ValueError, AssertionError)):
        _build_y_matrix(
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=4,
            L_series_H=np.ones(3) * 1e-12,
            C_shunt_F=2e-15,
        )

    with pytest.raises((ValueError, AssertionError)):
        _build_y_matrix(
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=4,
            L_series_H=1e-12,
            C_shunt_F=np.ones(7) * 2e-15,
        )


def test_solver_rejects_rhs_length_mismatch_if_available() -> None:
    with pytest.raises((ValueError, AssertionError, IndexError, np.linalg.LinAlgError)):
        _solve_ladder(
            np.ones(3, dtype=np.complex128),
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=4,
            L_series_H=1e-12,
            C_shunt_F=2e-15,
        )


def test_nan_component_values_propagate_or_are_rejected() -> None:
    try:
        Y = _build_y_matrix(
            omega_rad_s=2.0 * np.pi * 5e9,
            n_cells=3,
            L_series_H=np.array([1e-12, np.nan, 1e-12]),
            C_shunt_F=2e-15,
        )
    except (ValueError, AssertionError):
        return

    assert np.any(~np.isfinite(Y))
