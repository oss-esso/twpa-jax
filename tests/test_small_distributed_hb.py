"""
Tests for twpa.nonlinear.distributed_hb.

These tests define the expected behavior of a small distributed harmonic-balance
ladder before scaling to the full 100 mm TWPA.

Reference model
---------------
For a ladder with N nonlinear series inductors and N+1 shunt capacitors:

    node voltages:      V_h[n], n = 0, ..., N
    branch currents:    I_h[k], k = 0, ..., N-1
    harmonic orders:    h in H

For each harmonic h:

    KCL at node n:
        I_drive,h[n] - i omega_h C_n V_h[n]
        - sum(branch currents leaving node n) = 0

    branch voltage law:
        V_h[k] - V_h[k+1] - V_L,h(I_k) = 0

where V_L,h is the harmonic projection of

    V_L(t) = d/dt Phi(I(t)),
    Phi(I) = L0 * (I + beta * I**3 / (3 I*^2)).

For beta = 0, the system must reduce to the linear ladder MNA solution.

The tests are API-tolerant in naming, but they require the module to expose a
usable residual function for the small distributed HB problem. Solver helpers
are tested when present.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import twpa.nonlinear.distributed_hb as dhb


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
        raise ValueError("zero/DC order is not used by this distributed HB test suite")
    return arr


def _broadcast_series_l(L_series_H: float | Sequence[float], n_cells: int) -> np.ndarray:
    arr = np.asarray(L_series_H, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_cells, float(arr), dtype=float)
    if arr.shape != (n_cells,):
        raise ValueError(f"L_series_H must have shape ({n_cells},), got {arr.shape}")
    if np.any(arr <= 0.0):
        raise ValueError("all series inductances must be positive")
    return arr


def _broadcast_shunt_c(C_shunt_F: float | Sequence[float], n_cells: int) -> np.ndarray:
    arr = np.asarray(C_shunt_F, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_cells + 1, float(arr), dtype=float)
    if arr.shape == (n_cells,):
        caps = np.zeros(n_cells + 1, dtype=float)
        caps[:-1] += 0.5 * arr
        caps[1:] += 0.5 * arr
        arr = caps
    if arr.shape != (n_cells + 1,):
        raise ValueError(f"C_shunt_F must have shape ({n_cells + 1},), got {arr.shape}")
    if np.any(arr < 0.0):
        raise ValueError("all shunt capacitances must be non-negative")
    return arr


def _time_grid(n_time: int, omega0_rad_s: float) -> np.ndarray:
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    period = 2.0 * np.pi / omega0_rad_s
    return np.arange(n_time, dtype=float) * period / n_time


def _manual_synthesize(
    coeffs: Any,
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
    return np.einsum("th,h...->t...", phase, coeffs_arr)


def _manual_analyze(
    samples: Any,
    orders: Sequence[int],
    *,
    omega0_rad_s: float,
) -> np.ndarray:
    orders_arr = _validate_orders(orders)
    x = np.asarray(samples, dtype=np.complex128)
    n_time = x.shape[0]
    t = _time_grid(n_time, omega0_rad_s)
    phase = np.exp(-1j * np.outer(t * omega0_rad_s, orders_arr))
    return np.einsum("th,t...->h...", phase, x) / n_time


def _manual_branch_voltage_coeffs(
    branch_current_coeffs: Any,
    orders: Sequence[int],
    *,
    L_series_H: float | Sequence[float],
    I_star_A: float,
    beta: float,
    omega0_rad_s: float,
    n_time: int = 2048,
) -> np.ndarray:
    orders_arr = _validate_orders(orders)
    I = np.asarray(branch_current_coeffs, dtype=np.complex128)

    if I.ndim != 2:
        raise ValueError("branch_current_coeffs must have shape (n_harmonics, n_cells)")
    if I.shape[0] != orders_arr.size:
        raise ValueError("branch_current_coeffs/order length mismatch")

    n_cells = I.shape[1]
    L = _broadcast_series_l(L_series_H, n_cells)

    if I_star_A <= 0.0:
        raise ValueError("I_star_A must be positive")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    i_t = _manual_synthesize(
        I,
        orders_arr,
        n_time=n_time,
        omega0_rad_s=omega0_rad_s,
    )
    phi_t = L[None, :] * (i_t + beta * i_t**3 / (3.0 * I_star_A**2))
    phi_coeffs = _manual_analyze(phi_t, orders_arr, omega0_rad_s=omega0_rad_s)

    return 1j * orders_arr[:, None] * omega0_rad_s * phi_coeffs


def _manual_y_matrix(
    *,
    omega_rad_s: float,
    n_cells: int,
    L_series_H: float | Sequence[float],
    C_shunt_F: float | Sequence[float],
) -> np.ndarray:
    if omega_rad_s == 0.0:
        raise ValueError("omega_rad_s must be nonzero")
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    L = _broadcast_series_l(L_series_H, n_cells)
    C = _broadcast_shunt_c(C_shunt_F, n_cells)

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


def _manual_linear_solution(
    drive_current_coeffs: Any,
    orders: Sequence[int],
    *,
    L_series_H: float | Sequence[float],
    C_shunt_F: float | Sequence[float],
    omega0_rad_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    orders_arr = _validate_orders(orders)
    I_drive = np.asarray(drive_current_coeffs, dtype=np.complex128)

    if I_drive.ndim != 2:
        raise ValueError("drive_current_coeffs must have shape (n_harmonics, n_nodes)")
    if I_drive.shape[0] != orders_arr.size:
        raise ValueError("drive_current_coeffs/order length mismatch")

    n_nodes = I_drive.shape[1]
    n_cells = n_nodes - 1
    L = _broadcast_series_l(L_series_H, n_cells)
    _broadcast_shunt_c(C_shunt_F, n_cells)

    V = np.zeros_like(I_drive, dtype=np.complex128)
    I_branch = np.zeros((orders_arr.size, n_cells), dtype=np.complex128)

    for h_idx, order in enumerate(orders_arr):
        omega = order * omega0_rad_s
        Y = _manual_y_matrix(
            omega_rad_s=omega,
            n_cells=n_cells,
            L_series_H=L,
            C_shunt_F=C_shunt_F,
        )
        V[h_idx] = np.linalg.solve(Y, I_drive[h_idx])

        for k in range(n_cells):
            I_branch[h_idx, k] = (
                V[h_idx, k] - V[h_idx, k + 1]
            ) / (1j * omega * L[k])

    return V, I_branch


def _manual_residual(
    node_voltage_coeffs: Any,
    branch_current_coeffs: Any,
    drive_current_coeffs: Any,
    orders: Sequence[int],
    *,
    L_series_H: float | Sequence[float],
    C_shunt_F: float | Sequence[float],
    I_star_A: float,
    beta: float,
    omega0_rad_s: float,
) -> np.ndarray:
    orders_arr = _validate_orders(orders)

    V = np.asarray(node_voltage_coeffs, dtype=np.complex128)
    I_b = np.asarray(branch_current_coeffs, dtype=np.complex128)
    I_drive = np.asarray(drive_current_coeffs, dtype=np.complex128)

    if V.ndim != 2:
        raise ValueError("node_voltage_coeffs must have shape (n_harmonics, n_nodes)")
    if I_b.ndim != 2:
        raise ValueError("branch_current_coeffs must have shape (n_harmonics, n_cells)")
    if I_drive.ndim != 2:
        raise ValueError("drive_current_coeffs must have shape (n_harmonics, n_nodes)")

    n_h = orders_arr.size
    n_nodes = V.shape[1]
    n_cells = n_nodes - 1

    if V.shape != (n_h, n_nodes):
        raise ValueError("node voltage shape mismatch")
    if I_b.shape != (n_h, n_cells):
        raise ValueError("branch current shape mismatch")
    if I_drive.shape != (n_h, n_nodes):
        raise ValueError("drive current shape mismatch")

    C = _broadcast_shunt_c(C_shunt_F, n_cells)
    _broadcast_series_l(L_series_H, n_cells)

    kcl = np.zeros_like(V, dtype=np.complex128)

    for h_idx, order in enumerate(orders_arr):
        omega = order * omega0_rad_s
        cap_current = 1j * omega * C * V[h_idx]

        branch_leaving = np.zeros(n_nodes, dtype=np.complex128)
        branch_leaving[0] += I_b[h_idx, 0]
        branch_leaving[-1] += -I_b[h_idx, -1]
        for n in range(1, n_nodes - 1):
            branch_leaving[n] += I_b[h_idx, n] - I_b[h_idx, n - 1]

        kcl[h_idx] = I_drive[h_idx] - cap_current - branch_leaving

    branch_voltage_model = _manual_branch_voltage_coeffs(
        I_b,
        orders_arr,
        L_series_H=L_series_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
    )

    branch_voltage_actual = V[:, :-1] - V[:, 1:]
    branch_law = branch_voltage_actual - branch_voltage_model

    return np.concatenate([kcl.reshape(-1), branch_law.reshape(-1)])


def _make_problem(
    *,
    n_cells: int = 3,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    orders: Sequence[int] = (-3, -1, 1, 3),
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
    n_time: int = 512,
) -> Any:
    for name in [
        "DistributedHBProblem",
        "DistributedHBParams",
        "DistributedLadderHBProblem",
        "HBDistributedLadder",
        "DistributedHBLadderParams",
    ]:
        if hasattr(dhb, name):
            cls = getattr(dhb, name)
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
        "make_distributed_hb_problem",
        "make_distributed_hb_params",
        "make_distributed_ladder_hb_problem",
        "make_hb_ladder_problem",
    ]:
        if hasattr(dhb, name):
            fn = getattr(dhb, name)
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
        "n_cells": n_cells,
        "L_series_H": L_series_H,
        "C_shunt_F": C_shunt_F,
        "I_star_A": I_star_A,
        "beta": beta,
        "orders": tuple(orders),
        "omega0_rad_s": omega0_rad_s,
        "n_time": n_time,
    }


def _branch_voltage_coeffs(
    branch_current_coeffs: Any,
    orders: Sequence[int],
    *,
    L_series_H: float | Sequence[float] = 1e-12,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
    n_time: int = 512,
) -> np.ndarray:
    I = np.asarray(branch_current_coeffs, dtype=np.complex128)
    n_cells = I.shape[1] if I.ndim == 2 else 1
    problem = _make_problem(
        n_cells=n_cells,
        L_series_H=L_series_H,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )

    for name in [
        "branch_voltage_coeffs",
        "nonlinear_branch_voltage_coeffs",
        "inductor_voltage_coeffs",
        "series_inductor_voltage_coeffs",
        "current_to_branch_voltage_coeffs",
        "compute_branch_voltage_coeffs",
    ]:
        if hasattr(dhb, name):
            out = _call_with_supported_kwargs(
                getattr(dhb, name),
                branch_current_coeffs=branch_current_coeffs,
                I_branch_coeffs=branch_current_coeffs,
                I_series_coeffs=branch_current_coeffs,
                I_coeffs=branch_current_coeffs,
                currents=branch_current_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                L_series_H=L_series_H,
                L_H=L_series_H,
                series_inductance_H=L_series_H,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                n_time=n_time,
                problem=problem,
                params=problem,
            )
            return np.asarray(out, dtype=np.complex128)

    for method_name in [
        "branch_voltage_coeffs",
        "nonlinear_branch_voltage_coeffs",
        "inductor_voltage_coeffs",
        "series_inductor_voltage_coeffs",
        "current_to_branch_voltage_coeffs",
    ]:
        method = _get_attr_or_key(problem, method_name, default=None)
        if callable(method):
            out = _call_with_supported_kwargs(
                method,
                branch_current_coeffs=branch_current_coeffs,
                I_branch_coeffs=branch_current_coeffs,
                I_coeffs=branch_current_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                n_time=n_time,
            )
            return np.asarray(out, dtype=np.complex128)

    return _manual_branch_voltage_coeffs(
        branch_current_coeffs,
        orders,
        L_series_H=L_series_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
        n_time=max(n_time, 2048),
    )


def _residual(
    node_voltage_coeffs: Any,
    branch_current_coeffs: Any,
    drive_current_coeffs: Any,
    orders: Sequence[int],
    *,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
    I_star_A: float = 5e-3,
    beta: float = 1.0,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
    n_time: int = 512,
) -> np.ndarray:
    V = np.asarray(node_voltage_coeffs, dtype=np.complex128)
    I_b = np.asarray(branch_current_coeffs, dtype=np.complex128)

    n_cells = I_b.shape[1] if I_b.ndim == 2 else V.shape[1] - 1
    problem = _make_problem(
        n_cells=n_cells,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )

    for name in [
        "distributed_hb_residual",
        "distributed_ladder_residual",
        "ladder_hb_residual",
        "hb_residual",
        "residual",
        "pump_hb_residual",
    ]:
        if hasattr(dhb, name):
            out = _call_with_supported_kwargs(
                getattr(dhb, name),
                node_voltage_coeffs=node_voltage_coeffs,
                V_node_coeffs=node_voltage_coeffs,
                V_coeffs=node_voltage_coeffs,
                V=node_voltage_coeffs,
                branch_current_coeffs=branch_current_coeffs,
                I_branch_coeffs=branch_current_coeffs,
                I_series_coeffs=branch_current_coeffs,
                I_coeffs=branch_current_coeffs,
                drive_current_coeffs=drive_current_coeffs,
                I_drive_coeffs=drive_current_coeffs,
                source_current_coeffs=drive_current_coeffs,
                I_drive=drive_current_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                L_series_H=L_series_H,
                L_H=L_series_H,
                series_inductance_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                shunt_capacitance_F=C_shunt_F,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                Istar_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                n_time=n_time,
                problem=problem,
                params=problem,
            )
            return np.asarray(out, dtype=np.complex128).reshape(-1)

    for method_name in [
        "residual",
        "hb_residual",
        "distributed_hb_residual",
        "distributed_ladder_residual",
        "ladder_hb_residual",
    ]:
        method = _get_attr_or_key(problem, method_name, default=None)
        if callable(method):
            out = _call_with_supported_kwargs(
                method,
                node_voltage_coeffs=node_voltage_coeffs,
                V_node_coeffs=node_voltage_coeffs,
                V_coeffs=node_voltage_coeffs,
                branch_current_coeffs=branch_current_coeffs,
                I_branch_coeffs=branch_current_coeffs,
                I_coeffs=branch_current_coeffs,
                drive_current_coeffs=drive_current_coeffs,
                I_drive_coeffs=drive_current_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                n_time=n_time,
            )
            return np.asarray(out, dtype=np.complex128).reshape(-1)

    raise AttributeError(
        "twpa.nonlinear.distributed_hb must expose a residual helper such as "
        "distributed_hb_residual, ladder_hb_residual, or residual."
    )


def _solve(
    drive_current_coeffs: Any,
    orders: Sequence[int],
    *,
    L_series_H: float | Sequence[float] = 1e-12,
    C_shunt_F: float | Sequence[float] = 2e-15,
    I_star_A: float = 5e-3,
    beta: float = 0.0,
    omega0_rad_s: float = 2.0 * np.pi * 5e9,
    n_time: int = 512,
) -> tuple[np.ndarray, np.ndarray, Any]:
    drive = np.asarray(drive_current_coeffs, dtype=np.complex128)
    n_cells = drive.shape[1] - 1

    problem = _make_problem(
        n_cells=n_cells,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
        I_star_A=I_star_A,
        beta=beta,
        orders=orders,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )

    for name in [
        "solve_distributed_hb",
        "solve_distributed_ladder_hb",
        "solve_ladder_hb",
        "solve_pump_hb_ladder",
        "solve",
    ]:
        if hasattr(dhb, name):
            out = _call_with_supported_kwargs(
                getattr(dhb, name),
                drive_current_coeffs=drive_current_coeffs,
                I_drive_coeffs=drive_current_coeffs,
                source_current_coeffs=drive_current_coeffs,
                I_drive=drive_current_coeffs,
                orders=np.asarray(orders),
                harmonic_orders=np.asarray(orders),
                L_series_H=L_series_H,
                L_H=L_series_H,
                C_shunt_F=C_shunt_F,
                C_F=C_shunt_F,
                I_star_A=I_star_A,
                i_star_A=I_star_A,
                beta=beta,
                beta_nl=beta,
                nonlinear_beta=beta,
                omega0_rad_s=omega0_rad_s,
                omega0=omega0_rad_s,
                n_time=n_time,
                max_iter=80,
                tolerance=1e-11,
                tol=1e-11,
                problem=problem,
                params=problem,
            )

            V = _get_attr_or_key(
                out,
                "node_voltage_coeffs",
                "V_node_coeffs",
                "V_coeffs",
                "V",
                default=None,
            )
            I_b = _get_attr_or_key(
                out,
                "branch_current_coeffs",
                "I_branch_coeffs",
                "I_series_coeffs",
                "I_coeffs",
                default=None,
            )

            if V is not None and I_b is not None:
                return np.asarray(V, dtype=np.complex128), np.asarray(I_b, dtype=np.complex128), out

            if isinstance(out, tuple) and len(out) >= 2:
                return (
                    np.asarray(out[0], dtype=np.complex128),
                    np.asarray(out[1], dtype=np.complex128),
                    out,
                )

            raise TypeError(f"Could not extract node voltages and branch currents from {type(out)!r}")

    pytest.skip("Distributed HB solver is optional if residual helper exists.")


def test_problem_object_exposes_basic_fields() -> None:
    problem = _make_problem(
        n_cells=3,
        L_series_H=1.2e-12,
        C_shunt_F=2.5e-15,
        I_star_A=4e-3,
        beta=0.75,
        orders=(-3, -1, 1, 3),
    )

    mapping = _as_mapping(problem)
    assert isinstance(mapping, Mapping)
    assert len(mapping) > 0

    n_cells = _get_attr_or_key(problem, "n_cells", "num_cells", "N", default=3)
    beta = _get_attr_or_key(problem, "beta", "beta_nl", "nonlinear_beta", default=0.75)

    assert int(n_cells) == 3
    assert float(beta) == pytest.approx(0.75)


def test_zero_state_has_zero_residual() -> None:
    orders = np.array([-3, -1, 1, 3])
    n_cells = 3

    V = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)
    I_b = np.zeros((orders.size, n_cells), dtype=np.complex128)
    I_drive = np.zeros_like(V)

    residual = _residual(V, I_b, I_drive, orders)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-18, rtol=0.0)


def test_branch_voltage_linear_case_matches_iomega_l_i() -> None:
    orders = np.array([-3, -1, 1, 3])
    L = np.array([1.0, 1.2, 1.4]) * 1e-12
    omega0 = 2.0 * np.pi * 5e9

    I_b = np.array(
        [
            [0.1 - 0.2j, 0.2 + 0.1j, -0.1 + 0.0j],
            [1.0 + 0.3j, 0.8 - 0.2j, 0.2 + 0.1j],
            [0.5 - 0.1j, -0.4 + 0.2j, 0.6 - 0.3j],
            [-0.1 + 0.4j, 0.1 + 0.2j, 0.3 + 0.0j],
        ],
        dtype=np.complex128,
    ) * 1e-6

    V_L = _branch_voltage_coeffs(
        I_b,
        orders,
        L_series_H=L,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    expected = 1j * orders[:, None] * omega0 * L[None, :] * I_b

    np.testing.assert_allclose(V_L, expected, rtol=1e-12, atol=1e-18)


def test_branch_voltage_nonlinear_case_matches_time_projection() -> None:
    orders = np.array([-3, -1, 1, 3])
    L = np.array([1.0, 1.2]) * 1e-12
    I_star = 5e-3
    beta = 0.8
    omega0 = 2.0 * np.pi * 6e9

    I_b = np.array(
        [
            [0.02 - 0.01j, -0.01 + 0.02j],
            [0.6 + 0.2j, 0.4 - 0.1j],
            [0.6 - 0.2j, 0.4 + 0.1j],
            [0.02 + 0.01j, -0.01 - 0.02j],
        ],
        dtype=np.complex128,
    ) * 1e-3

    V_L = _branch_voltage_coeffs(
        I_b,
        orders,
        L_series_H=L,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
        n_time=2048,
    )

    expected = _manual_branch_voltage_coeffs(
        I_b,
        orders,
        L_series_H=L,
        I_star_A=I_star,
        beta=beta,
        omega0_rad_s=omega0,
        n_time=2048,
    )

    np.testing.assert_allclose(V_L, expected, rtol=1e-10, atol=1e-15)


def test_residual_is_zero_for_manual_linear_solution() -> None:
    orders = np.array([-3, -1, 1, 3])
    n_cells = 3
    L = np.array([1.0, 1.2, 1.4]) * 1e-12
    C = np.array([1.0, 1.1, 1.2, 1.3]) * 2e-15
    omega0 = 2.0 * np.pi * 5e9

    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)
    I_drive[1, 0] = 1.0e-6
    I_drive[1, -1] = -1.0e-6
    I_drive[2, 0] = 0.4e-6
    I_drive[2, -1] = -0.4e-6

    V, I_b = _manual_linear_solution(
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        omega0_rad_s=omega0,
    )

    residual = _residual(
        V,
        I_b,
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(residual, np.zeros_like(residual), rtol=0.0, atol=1e-12)


def test_residual_detects_kcl_perturbation() -> None:
    orders = np.array([-1, 1])
    n_cells = 3
    L = 1e-12
    C = 2e-15
    omega0 = 2.0 * np.pi * 5e9

    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)
    I_drive[0, 0] = 1.0e-6
    I_drive[0, -1] = -1.0e-6
    I_drive[1, 0] = 1.0e-6
    I_drive[1, -1] = -1.0e-6

    V, I_b = _manual_linear_solution(
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        omega0_rad_s=omega0,
    )

    clean = _residual(
        V,
        I_b,
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        beta=0.0,
        omega0_rad_s=omega0,
    )
    perturbed = _residual(
        V,
        I_b + 1e-9,
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    assert np.linalg.norm(clean) < 1e-12
    assert np.linalg.norm(perturbed) > 1e-10


def test_residual_detects_branch_voltage_perturbation() -> None:
    orders = np.array([-1, 1])
    n_cells = 2
    L = 1e-12
    C = 2e-15
    omega0 = 2.0 * np.pi * 5e9

    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)
    I_drive[:, 0] = 1e-6
    I_drive[:, -1] = -1e-6

    V, I_b = _manual_linear_solution(
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        omega0_rad_s=omega0,
    )

    clean = _residual(V, I_b, I_drive, orders, L_series_H=L, C_shunt_F=C, beta=0.0, omega0_rad_s=omega0)

    V_bad = V.copy()
    V_bad[:, 1] += 1e-9

    bad = _residual(V_bad, I_b, I_drive, orders, L_series_H=L, C_shunt_F=C, beta=0.0, omega0_rad_s=omega0)

    assert np.linalg.norm(clean) < 1e-12
    assert np.linalg.norm(bad) > 1e-10


def test_conjugate_symmetric_branch_current_gives_conjugate_symmetric_voltage() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_b = np.array(
        [
            [0.05 - 0.02j, -0.01 + 0.03j],
            [0.7 + 0.2j, 0.4 - 0.1j],
            [0.7 - 0.2j, 0.4 + 0.1j],
            [0.05 + 0.02j, -0.01 - 0.03j],
        ],
        dtype=np.complex128,
    ) * 1e-3

    V_L = _branch_voltage_coeffs(I_b, orders, L_series_H=[1e-12, 1.2e-12], beta=1.0)

    assert V_L[0, 0] == pytest.approx(np.conj(V_L[3, 0]), rel=1e-10, abs=1e-15)
    assert V_L[1, 1] == pytest.approx(np.conj(V_L[2, 1]), rel=1e-10, abs=1e-15)


def test_pure_fundamental_branch_current_generates_third_harmonic_voltage() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_b = np.zeros((orders.size, 2), dtype=np.complex128)
    I_b[1, :] = 0.5e-3
    I_b[2, :] = 0.5e-3

    V_nl = _branch_voltage_coeffs(I_b, orders, L_series_H=[1e-12, 1.1e-12], beta=1.0)
    V_lin = _branch_voltage_coeffs(I_b, orders, L_series_H=[1e-12, 1.1e-12], beta=0.0)

    assert np.linalg.norm(V_nl[0] - V_lin[0]) > 0.0
    assert np.linalg.norm(V_nl[3] - V_lin[3]) > 0.0


def test_nonlinear_correction_scales_linearly_with_beta() -> None:
    orders = np.array([-3, -1, 1, 3])
    I_b = np.array(
        [
            [0.02, 0.01],
            [0.5, 0.4],
            [0.5, 0.4],
            [0.02, 0.01],
        ],
        dtype=np.complex128,
    ) * 1e-3

    v0 = _branch_voltage_coeffs(I_b, orders, beta=0.0)
    v1 = _branch_voltage_coeffs(I_b, orders, beta=1.0)
    v2 = _branch_voltage_coeffs(I_b, orders, beta=2.0)

    np.testing.assert_allclose(v2 - v0, 2.0 * (v1 - v0), rtol=1e-10, atol=1e-15)


def test_solver_recovers_manual_linear_solution_if_available() -> None:
    orders = np.array([-1, 1])
    n_cells = 4
    L = np.linspace(1.0, 1.4, n_cells) * 1e-12
    C = np.linspace(1.0, 1.5, n_cells + 1) * 2e-15
    omega0 = 2.0 * np.pi * 5e9

    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)
    I_drive[:, 0] = 1e-6
    I_drive[:, -1] = -1e-6

    V_expected, I_expected = _manual_linear_solution(
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        omega0_rad_s=omega0,
    )

    V, I_b, _ = _solve(
        I_drive,
        orders,
        L_series_H=L,
        C_shunt_F=C,
        I_star_A=5e-3,
        beta=0.0,
        omega0_rad_s=omega0,
    )

    np.testing.assert_allclose(V, V_expected, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(I_b, I_expected, rtol=1e-8, atol=1e-12)


def test_solver_solution_has_small_residual_if_available() -> None:
    orders = np.array([-3, -1, 1, 3])
    n_cells = 3
    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)
    I_drive[1, 0] = 0.8e-6
    I_drive[1, -1] = -0.8e-6
    I_drive[2, 0] = 0.8e-6
    I_drive[2, -1] = -0.8e-6

    V, I_b, _ = _solve(
        I_drive,
        orders,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
        I_star_A=5e-3,
        beta=1.0,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )

    residual = _residual(
        V,
        I_b,
        I_drive,
        orders,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
        I_star_A=5e-3,
        beta=1.0,
        omega0_rad_s=2.0 * np.pi * 5e9,
    )

    assert np.linalg.norm(residual) < 1e-8


def test_zero_drive_solver_solution_is_zero_if_available() -> None:
    orders = np.array([-1, 1])
    n_cells = 3
    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)

    V, I_b, _ = _solve(
        I_drive,
        orders,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
        beta=1.0,
    )

    np.testing.assert_allclose(V, np.zeros_like(V), rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(I_b, np.zeros_like(I_b), rtol=0.0, atol=1e-18)


def test_solver_preserves_conjugate_symmetry_if_available() -> None:
    orders = np.array([-3, -1, 1, 3])
    n_cells = 3
    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)

    I_drive[0, 0] = 0.02e-6 - 0.01e-6j
    I_drive[0, -1] = -I_drive[0, 0]
    I_drive[1, 0] = 0.8e-6 + 0.2e-6j
    I_drive[1, -1] = -I_drive[1, 0]
    I_drive[2, 0] = np.conj(I_drive[1, 0])
    I_drive[2, -1] = -I_drive[2, 0]
    I_drive[3, 0] = np.conj(I_drive[0, 0])
    I_drive[3, -1] = -I_drive[3, 0]

    V, I_b, _ = _solve(
        I_drive,
        orders,
        L_series_H=1e-12,
        C_shunt_F=2e-15,
        beta=0.5,
    )

    np.testing.assert_allclose(V[0], np.conj(V[3]), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(V[1], np.conj(V[2]), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(I_b[0], np.conj(I_b[3]), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(I_b[1], np.conj(I_b[2]), rtol=1e-7, atol=1e-12)


def test_result_object_if_solver_available_is_serializable_or_tuple() -> None:
    orders = np.array([-1, 1])
    n_cells = 2
    I_drive = np.zeros((orders.size, n_cells + 1), dtype=np.complex128)
    I_drive[:, 0] = 1e-6
    I_drive[:, -1] = -1e-6

    _, _, result = _solve(I_drive, orders, beta=0.0)

    if isinstance(result, tuple):
        pytest.skip("Tuple solver result is acceptable and has no mapping metadata.")

    mapping = _as_mapping(result)

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


def test_invalid_duplicate_orders_are_rejected() -> None:
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.zeros((2, n_cells), dtype=np.complex128)
    I_drive = np.zeros_like(V)

    with pytest.raises((ValueError, AssertionError)):
        _residual(V, I_b, I_drive, [1, 1])


def test_invalid_zero_order_is_rejected() -> None:
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.zeros((2, n_cells), dtype=np.complex128)
    I_drive = np.zeros_like(V)

    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _residual(V, I_b, I_drive, [0, 1])


def test_invalid_noninteger_orders_are_rejected() -> None:
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.zeros((2, n_cells), dtype=np.complex128)
    I_drive = np.zeros_like(V)

    with pytest.raises((ValueError, AssertionError, TypeError)):
        _residual(V, I_b, I_drive, [-1.5, 1.0])


def test_shape_mismatch_is_rejected() -> None:
    orders = np.array([-1, 1])
    V = np.zeros((2, 4), dtype=np.complex128)
    I_b = np.zeros((2, 2), dtype=np.complex128)
    I_drive = np.zeros((2, 4), dtype=np.complex128)

    with pytest.raises((ValueError, AssertionError, IndexError)):
        _residual(V, I_b, I_drive, orders)


def test_invalid_nonpositive_series_inductance_is_rejected() -> None:
    orders = np.array([-1, 1])
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.zeros((2, n_cells), dtype=np.complex128)
    I_drive = np.zeros_like(V)

    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _residual(V, I_b, I_drive, orders, L_series_H=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _residual(V, I_b, I_drive, orders, L_series_H=-1e-12)


def test_invalid_negative_shunt_capacitance_is_rejected() -> None:
    orders = np.array([-1, 1])
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.zeros((2, n_cells), dtype=np.complex128)
    I_drive = np.zeros_like(V)

    with pytest.raises((ValueError, AssertionError)):
        _residual(V, I_b, I_drive, orders, C_shunt_F=-2e-15)


def test_invalid_nonpositive_i_star_is_rejected() -> None:
    orders = np.array([-1, 1])
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.ones((2, n_cells), dtype=np.complex128) * 1e-6
    I_drive = np.zeros_like(V)

    with pytest.raises((ValueError, AssertionError, ZeroDivisionError)):
        _residual(V, I_b, I_drive, orders, I_star_A=0.0)

    with pytest.raises((ValueError, AssertionError)):
        _residual(V, I_b, I_drive, orders, I_star_A=-5e-3)


def test_invalid_negative_beta_is_rejected_or_stays_finite() -> None:
    orders = np.array([-1, 1])
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.ones((2, n_cells), dtype=np.complex128) * 1e-6
    I_drive = np.zeros_like(V)

    try:
        residual = _residual(V, I_b, I_drive, orders, beta=-0.1)
    except (ValueError, AssertionError):
        return

    assert np.all(np.isfinite(residual.real))
    assert np.all(np.isfinite(residual.imag))


def test_nan_inputs_propagate_to_residual_or_are_rejected() -> None:
    orders = np.array([-1, 1])
    n_cells = 2
    V = np.zeros((2, n_cells + 1), dtype=np.complex128)
    I_b = np.ones((2, n_cells), dtype=np.complex128) * 1e-6
    I_b[0, 0] = np.nan
    I_drive = np.zeros_like(V)

    try:
        residual = _residual(V, I_b, I_drive, orders)
    except (ValueError, AssertionError):
        return

    assert np.any(~np.isfinite(residual))
