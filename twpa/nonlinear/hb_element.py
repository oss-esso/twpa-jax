"""
twpa.nonlinear.hb_element
=========================

Element-level harmonic-balance residuals.

This module defines local HB element equations that are independent of a
specific circuit topology. Topology-level modules combine these into KCL/KVL
systems:

    twpa.nonlinear.one_node
    twpa.nonlinear.distributed_hb
    twpa.nonlinear.pump_hb_ladder

Frequency-domain convention
---------------------------
All coefficient arrays use shape:

    (K, ...)

where K is the number of tones/frequencies. The first dimension is always the
frequency/tone axis.

For a branch from node a to node b:

    V_drop,k = V_a,k - V_b,k

Series branch residual convention:

    r_branch,k = V_drop,k - Z_linear,k I_k - V_nonlinear,k

Shunt branch KCL convention:

    I_shunt,k = Y_k V_node,k

For nodal KCL, currents leaving the node through passive shunts/branches are
positive unless a topology module explicitly chooses the opposite convention.

Scope
-----
This file provides:
- linear capacitor/resistor/inductor HB equations,
- nonlinear kinetic-inductor HB branch equations,
- current/voltage source residual helpers,
- small local Jacobian/JVP utilities,
- validation diagnostics.

It does not assemble a distributed line. That happens in later files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from twpa.core.hb_fft import (
    HBProjectionConfig,
    HBProjectionGrid,
    linear_inductor_voltage_coefficients,
    nonlinear_inductor_branch_residual,
    nonlinear_inductor_voltage_coefficients,
)
from twpa.core.harmonics import coefficient_power_summary
from twpa.core.units import angular_frequency
from twpa.nonlinear.kinetic_inductance import KineticInductanceModel


ArrayLike = Any


@dataclass(frozen=True)
class KineticInductanceHBElement:
    """Selected-harmonic KI element configuration."""

    L0_H: float
    I_star_A: float
    beta: float = 1.0
    orders: tuple[int, ...] = (-3, -1, 1, 3)
    omega0_rad_s: float = 1.0

    def __post_init__(self) -> None:
        if self.L0_H <= 0.0:
            raise ValueError("L0_H must be positive")
        if self.I_star_A <= 0.0:
            raise ValueError("I_star_A must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        validated = _selected_orders(self.orders)
        object.__setattr__(self, "orders", tuple(int(v) for v in validated.tolist()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "L0_H": self.L0_H,
            "I_star_A": self.I_star_A,
            "beta": self.beta,
            "orders": list(self.orders),
            "omega0_rad_s": self.omega0_rad_s,
        }


def _selected_orders(orders: ArrayLike) -> jax.Array:
    arr = jnp.asarray(orders)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("orders must be a non-empty 1D array")
    if not bool(jnp.all(arr == jnp.round(arr))):
        raise ValueError("orders must be integer-valued")
    out = arr.astype(jnp.int64)
    if bool(jnp.any(out == 0)):
        raise ValueError("zero/DC order is not supported")
    if len(set(int(v) for v in out.tolist())) != int(out.size):
        raise ValueError("orders must be unique")
    return out


def flux_coeffs(
    current_coeffs: ArrayLike,
    orders: ArrayLike,
    *,
    L0_H: float,
    I_star_A: float,
    beta: float = 1.0,
    n_time: int = 2048,
    **_: Any,
) -> jax.Array:
    """Selected-harmonic integral flux projection for a KI branch."""
    if L0_H <= 0.0:
        raise ValueError("L0_H must be positive")
    if I_star_A <= 0.0:
        raise ValueError("I_star_A must be positive")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if int(n_time) <= 0:
        raise ValueError("n_time must be positive")
    k = _selected_orders(orders)
    current = jnp.asarray(current_coeffs, dtype=jnp.complex128)
    if current.shape[0] != k.shape[0]:
        raise ValueError("current_coeffs first dimension must match orders")
    sample = jnp.arange(int(n_time), dtype=jnp.float64)
    basis = jnp.exp(1j * 2.0 * jnp.pi * sample[:, None] * k[None, :] / int(n_time))
    current_t = jnp.einsum("th,h...->t...", basis, current)
    flux_t = L0_H * (current_t + beta * current_t**3 / (3.0 * I_star_A**2))
    return jnp.einsum("th,t...->h...", jnp.conj(basis), flux_t) / int(n_time)


def voltage_coeffs(
    current_coeffs: ArrayLike,
    orders: ArrayLike,
    *,
    omega0_rad_s: float,
    **kwargs: Any,
) -> jax.Array:
    """Selected-harmonic KI voltage coefficients, V_k = i*k*omega0*Phi_k."""
    k = _selected_orders(orders)
    phi = flux_coeffs(current_coeffs, k, **kwargs)
    omega = k.reshape((k.shape[0],) + (1,) * (phi.ndim - 1))
    return 1j * omega * float(omega0_rad_s) * phi


def residual(
    voltage_coeffs: ArrayLike,
    current_coeffs: ArrayLike,
    orders: ArrayLike,
    *,
    omega0_rad_s: float,
    **kwargs: Any,
) -> jax.Array:
    """Selected-harmonic branch residual V_drop - V_KI(I)."""
    return jnp.asarray(voltage_coeffs) - globals()["voltage_coeffs"](
        current_coeffs,
        orders,
        omega0_rad_s=omega0_rad_s,
        **kwargs,
    )


def voltage_jacobian(
    current_coeffs: ArrayLike,
    orders: ArrayLike,
    *,
    omega0_rad_s: float,
    **kwargs: Any,
) -> jax.Array:
    """Complex Jacobian dV/dI for selected-harmonic diagnostics."""
    current = jnp.asarray(current_coeffs, dtype=jnp.complex128)
    k = _selected_orders(orders)
    if float(kwargs.get("beta", 1.0)) == 0.0:
        return jnp.diag(1j * k * float(omega0_rad_s) * float(kwargs["L0_H"]))
    return jax.jacfwd(
        lambda value: voltage_coeffs(
            value,
            orders,
            omega0_rad_s=omega0_rad_s,
            **kwargs,
        ),
        holomorphic=True,
    )(current)


# ---------------------------------------------------------------------------
# Enums / helpers
# ---------------------------------------------------------------------------

class HBElementKind(str, Enum):
    """Supported HB element types."""

    SERIES_RESISTOR = "series_resistor"
    SERIES_INDUCTOR = "series_inductor"
    SERIES_RL = "series_rl"
    SERIES_KINETIC_INDUCTOR = "series_kinetic_inductor"
    SHUNT_CAPACITOR = "shunt_capacitor"
    SHUNT_CONDUCTANCE = "shunt_conductance"
    SHUNT_ADMITTANCE = "shunt_admittance"
    CURRENT_SOURCE = "current_source"
    VOLTAGE_SOURCE = "voltage_source"


def _as_complex_coeffs(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    if arr.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension, got {arr.shape}")
    return arr


def _as_frequency_array(frequencies_hz: ArrayLike) -> jax.Array:
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)
    if f.ndim != 1:
        raise ValueError(f"frequencies_hz must be 1D, got {f.shape}")
    return f


def _check_frequency_axis(coeffs: jax.Array, frequencies_hz: jax.Array, name: str) -> None:
    if coeffs.shape[0] != frequencies_hz.shape[0]:
        raise ValueError(
            f"{name} frequency axis length {coeffs.shape[0]} does not match "
            f"frequencies_hz length {frequencies_hz.shape[0]}"
        )


def _broadcast_element_parameter(
    value: ArrayLike,
    coeffs: jax.Array,
    *,
    name: str,
) -> jax.Array:
    """
    Broadcast scalar/array parameter to coefficient trailing dimensions.

    coeffs shape is (K, ...). Parameter may be scalar or broadcastable to (...).
    """
    p = jnp.asarray(value)
    trailing_shape = coeffs.shape[1:]
    if p.ndim == 0:
        return p
    try:
        return jnp.broadcast_to(p, trailing_shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {p.shape} cannot broadcast to coefficient "
            f"trailing shape {trailing_shape}"
        ) from exc


def _omega_broadcast(frequencies_hz: jax.Array, coeffs_ndim: int) -> jax.Array:
    omega = angular_frequency(frequencies_hz)
    return omega.reshape((omega.shape[0],) + (1,) * (coeffs_ndim - 1))


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {
            "array_shape": tuple(int(s) for s in obj.shape),
            "array_dtype": str(obj.dtype),
        }
    return obj


# ---------------------------------------------------------------------------
# Linear element coefficient laws
# ---------------------------------------------------------------------------

def series_resistor_voltage_coefficients(
    current_coeffs_A: ArrayLike,
    *,
    R_ohm: ArrayLike,
) -> jax.Array:
    """
    Series resistor voltage coefficients:

        V_k = R I_k
    """
    i = _as_complex_coeffs("current_coeffs_A", current_coeffs_A)
    R = _broadcast_element_parameter(R_ohm, i, name="R_ohm")
    return R * i


def series_inductor_voltage_coefficients(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L_H: ArrayLike,
) -> jax.Array:
    """
    Series linear inductor voltage coefficients:

        V_k = j omega_k L I_k
    """
    i = _as_complex_coeffs("current_coeffs_A", current_coeffs_A)
    f = _as_frequency_array(frequencies_hz)
    _check_frequency_axis(i, f, "current_coeffs_A")
    L = _broadcast_element_parameter(L_H, i, name="L_H")
    omega = _omega_broadcast(f, i.ndim)
    return 1j * omega * L * i


def series_rl_voltage_coefficients(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    R_ohm: ArrayLike = 0.0,
    L_H: ArrayLike = 0.0,
) -> jax.Array:
    """
    Series R-L branch voltage coefficients:

        V_k = (R + j omega_k L) I_k
    """
    i = _as_complex_coeffs("current_coeffs_A", current_coeffs_A)
    f = _as_frequency_array(frequencies_hz)
    _check_frequency_axis(i, f, "current_coeffs_A")

    R = _broadcast_element_parameter(R_ohm, i, name="R_ohm")
    L = _broadcast_element_parameter(L_H, i, name="L_H")
    omega = _omega_broadcast(f, i.ndim)

    return (R + 1j * omega * L) * i


def shunt_capacitor_current_coefficients(
    voltage_coeffs_V: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    C_F: ArrayLike,
) -> jax.Array:
    """
    Shunt capacitor current coefficients:

        I_k = j omega_k C V_k
    """
    v = _as_complex_coeffs("voltage_coeffs_V", voltage_coeffs_V)
    f = _as_frequency_array(frequencies_hz)
    _check_frequency_axis(v, f, "voltage_coeffs_V")

    C = _broadcast_element_parameter(C_F, v, name="C_F")
    omega = _omega_broadcast(f, v.ndim)

    return 1j * omega * C * v


def shunt_conductance_current_coefficients(
    voltage_coeffs_V: ArrayLike,
    *,
    G_S: ArrayLike,
) -> jax.Array:
    """
    Shunt conductance current coefficients:

        I_k = G V_k
    """
    v = _as_complex_coeffs("voltage_coeffs_V", voltage_coeffs_V)
    G = _broadcast_element_parameter(G_S, v, name="G_S")
    return G * v


def shunt_admittance_current_coefficients(
    voltage_coeffs_V: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    C_F: ArrayLike = 0.0,
    G_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    Shunt admittance current coefficients:

        I_k = (G + j omega_k C) V_k
    """
    v = _as_complex_coeffs("voltage_coeffs_V", voltage_coeffs_V)
    f = _as_frequency_array(frequencies_hz)
    _check_frequency_axis(v, f, "voltage_coeffs_V")

    G = _broadcast_element_parameter(G_S, v, name="G_S")
    C = _broadcast_element_parameter(C_F, v, name="C_F")
    omega = _omega_broadcast(f, v.ndim)

    return (G + 1j * omega * C) * v


def branch_voltage_drop(
    voltage_left_coeffs_V: ArrayLike,
    voltage_right_coeffs_V: ArrayLike | float = 0.0,
) -> jax.Array:
    """
    Branch voltage drop from left node to right node:

        V_drop = V_left - V_right
    """
    vl = _as_complex_coeffs("voltage_left_coeffs_V", voltage_left_coeffs_V)
    vr = jnp.asarray(voltage_right_coeffs_V)
    return vl - vr


# ---------------------------------------------------------------------------
# Element residuals
# ---------------------------------------------------------------------------

def series_linear_branch_residual(
    voltage_drop_coeffs_V: ArrayLike,
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    R_ohm: ArrayLike = 0.0,
    L_H: ArrayLike = 0.0,
) -> jax.Array:
    """
    Series linear R-L branch residual:

        r_k = V_drop,k - (R + j omega_k L) I_k
    """
    v = _as_complex_coeffs("voltage_drop_coeffs_V", voltage_drop_coeffs_V)
    i = _as_complex_coeffs("current_coeffs_A", current_coeffs_A)
    if v.shape != i.shape:
        raise ValueError(f"voltage/current shapes must match, got {v.shape} and {i.shape}")
    f = _as_frequency_array(frequencies_hz)
    _check_frequency_axis(v, f, "voltage_drop_coeffs_V")

    return v - series_rl_voltage_coefficients(
        i,
        f,
        R_ohm=R_ohm,
        L_H=L_H,
    )


def shunt_linear_branch_residual(
    current_coeffs_A: ArrayLike,
    voltage_coeffs_V: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    C_F: ArrayLike = 0.0,
    G_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    Shunt linear branch residual:

        r_k = I_k - (G + j omega_k C) V_k
    """
    i = _as_complex_coeffs("current_coeffs_A", current_coeffs_A)
    v = _as_complex_coeffs("voltage_coeffs_V", voltage_coeffs_V)
    if i.shape != v.shape:
        raise ValueError(f"current/voltage shapes must match, got {i.shape} and {v.shape}")
    f = _as_frequency_array(frequencies_hz)
    _check_frequency_axis(i, f, "current_coeffs_A")

    return i - shunt_admittance_current_coefficients(
        v,
        f,
        C_F=C_F,
        G_S=G_S,
    )


def current_source_residual(
    current_unknown_coeffs_A: ArrayLike,
    source_current_coeffs_A: ArrayLike,
    *,
    sign: int = +1,
) -> jax.Array:
    """
    Current-source residual.

    Convention:
        sign = +1 -> residual = I_unknown - I_source
        sign = -1 -> residual = I_unknown + I_source

    This is useful when a topology module introduces explicit source branch
    currents.
    """
    i = _as_complex_coeffs("current_unknown_coeffs_A", current_unknown_coeffs_A)
    src = jnp.asarray(source_current_coeffs_A, dtype=i.dtype)
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1")
    return i - sign * src


def voltage_source_residual(
    voltage_unknown_coeffs_V: ArrayLike,
    source_voltage_coeffs_V: ArrayLike,
) -> jax.Array:
    """
    Voltage-source residual:

        r_k = V_unknown,k - V_source,k
    """
    v = _as_complex_coeffs("voltage_unknown_coeffs_V", voltage_unknown_coeffs_V)
    src = jnp.asarray(source_voltage_coeffs_V, dtype=v.dtype)
    return v - src


def kcl_residual(
    currents_leaving_node_A: ArrayLike,
    injected_current_coeffs_A: ArrayLike | float = 0.0,
) -> jax.Array:
    """
    Generic KCL residual.

    Convention:
        passive currents leaving node are positive.
        injected source current entering node is positive.

    Residual:
        r_k = sum(I_leaving,k) - I_injected,k

    currents_leaving_node_A may have shape:
        (K, n_terms, ...)
    or already-summed shape:
        (K, ...)
    """
    currents = jnp.asarray(currents_leaving_node_A)
    if not jnp.issubdtype(currents.dtype, jnp.complexfloating):
        currents = currents.astype(jnp.complex128)

    if currents.ndim >= 2:
        # Interpret axis 1 as term axis only if that is how caller stacked terms.
        total = jnp.sum(currents, axis=1)
    else:
        total = currents

    return total - jnp.asarray(injected_current_coeffs_A, dtype=total.dtype)


# ---------------------------------------------------------------------------
# HB element classes
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HBSeriesKineticInductor:
    """
    Series nonlinear kinetic-inductor HB element.

    Residual:
        r_k = V_drop,k - R I_k - V_KI,k(I)

    Parameters
    ----------
    model:
        KineticInductanceModel defining L0, I_star, beta, quartic correction.
    R_series_ohm:
        Optional series resistance.
    name:
        Element name.
    metadata:
        Static metadata.
    """

    model: KineticInductanceModel
    R_series_ohm: Any = 0.0
    name: str = "series_ki"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        R = jnp.asarray(self.R_series_ohm)
        if bool(jnp.any(R < 0.0)):
            raise ValueError("R_series_ohm must be non-negative")
        object.__setattr__(self, "R_series_ohm", R)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def tree_flatten(self) -> tuple[tuple[Any, jax.Array], dict[str, Any]]:
        children = (self.model, self.R_series_ohm)
        aux = {"name": self.name, "metadata": dict(self.metadata or {})}
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[Any, jax.Array],
    ) -> "HBSeriesKineticInductor":
        return cls(
            model=children[0],
            R_series_ohm=children[1],
            name=aux["name"],
            metadata=aux["metadata"],
        )

    @property
    def kind(self) -> HBElementKind:
        return HBElementKind.SERIES_KINETIC_INDUCTOR

    def voltage_coefficients(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
        include_resistance: bool = True,
    ) -> jax.Array:
        """
        Element voltage coefficients V = R I + V_KI(I).
        """
        i = _as_complex_coeffs("current_coeffs_A", current_coeffs_A)

        v = self.model.voltage_coefficients(
            i,
            frequencies_hz,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )
        if include_resistance:
            v = v + self.R_series_ohm * i
        return v

    def residual(
        self,
        voltage_drop_coeffs_V: ArrayLike,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> jax.Array:
        """
        Series KI branch residual.
        """
        return self.model.branch_residual(
            voltage_drop_coeffs_V,
            current_coeffs_A,
            frequencies_hz,
            R_series_ohm=self.R_series_ohm,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    def jvp_voltage(
        self,
        current_coeffs_A: ArrayLike,
        tangent_current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Return V(I) and dV[I]·dI including series resistance.
        """
        i = _as_complex_coeffs("current_coeffs_A", current_coeffs_A)
        di = _as_complex_coeffs("tangent_current_coeffs_A", tangent_current_coeffs_A)

        v_nl, dv_nl = self.model.voltage_jvp(
            i,
            di,
            frequencies_hz,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )
        return v_nl + self.R_series_ohm * i, dv_nl + self.R_series_ohm * di

    def with_updates(self, **kwargs: Any) -> "HBSeriesKineticInductor":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "model": self.model.to_dict(),
            "R_series_ohm": _jsonify(self.R_series_ohm),
            "metadata": _jsonify(dict(self.metadata or {})),
        }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HBSeriesLinearElement:
    """
    Series linear R-L HB element.

    Residual:
        r_k = V_drop,k - (R + j omega_k L) I_k
    """

    R_ohm: Any = 0.0
    L_H: Any = 0.0
    name: str = "series_linear"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        R = jnp.asarray(self.R_ohm)
        L = jnp.asarray(self.L_H)
        if bool(jnp.any(R < 0.0)):
            raise ValueError("R_ohm must be non-negative")
        if bool(jnp.any(L < 0.0)):
            raise ValueError("L_H must be non-negative")
        object.__setattr__(self, "R_ohm", R)
        object.__setattr__(self, "L_H", L)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        return (self.R_ohm, self.L_H), {
            "name": self.name,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array],
    ) -> "HBSeriesLinearElement":
        return cls(
            R_ohm=children[0],
            L_H=children[1],
            name=aux["name"],
            metadata=aux["metadata"],
        )

    @property
    def kind(self) -> HBElementKind:
        if bool(jnp.all(self.L_H == 0.0)):
            return HBElementKind.SERIES_RESISTOR
        if bool(jnp.all(self.R_ohm == 0.0)):
            return HBElementKind.SERIES_INDUCTOR
        return HBElementKind.SERIES_RL

    def voltage_coefficients(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
    ) -> jax.Array:
        return series_rl_voltage_coefficients(
            current_coeffs_A,
            frequencies_hz,
            R_ohm=self.R_ohm,
            L_H=self.L_H,
        )

    def residual(
        self,
        voltage_drop_coeffs_V: ArrayLike,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
    ) -> jax.Array:
        return series_linear_branch_residual(
            voltage_drop_coeffs_V,
            current_coeffs_A,
            frequencies_hz,
            R_ohm=self.R_ohm,
            L_H=self.L_H,
        )

    def with_updates(self, **kwargs: Any) -> "HBSeriesLinearElement":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "R_ohm": _jsonify(self.R_ohm),
            "L_H": _jsonify(self.L_H),
            "metadata": _jsonify(dict(self.metadata or {})),
        }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HBShuntLinearElement:
    """
    Shunt linear G-C HB element.

    Current leaving node:
        I_k = (G + j omega_k C) V_k

    Residual if current is explicit:
        r_k = I_k - (G + j omega_k C) V_k
    """

    G_S: Any = 0.0
    C_F: Any = 0.0
    name: str = "shunt_linear"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        G = jnp.asarray(self.G_S)
        C = jnp.asarray(self.C_F)
        if bool(jnp.any(G < 0.0)):
            raise ValueError("G_S must be non-negative")
        if bool(jnp.any(C < 0.0)):
            raise ValueError("C_F must be non-negative")
        object.__setattr__(self, "G_S", G)
        object.__setattr__(self, "C_F", C)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        return (self.G_S, self.C_F), {
            "name": self.name,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array],
    ) -> "HBShuntLinearElement":
        return cls(
            G_S=children[0],
            C_F=children[1],
            name=aux["name"],
            metadata=aux["metadata"],
        )

    @property
    def kind(self) -> HBElementKind:
        if bool(jnp.all(self.C_F == 0.0)):
            return HBElementKind.SHUNT_CONDUCTANCE
        if bool(jnp.all(self.G_S == 0.0)):
            return HBElementKind.SHUNT_CAPACITOR
        return HBElementKind.SHUNT_ADMITTANCE

    def current_coefficients(
        self,
        voltage_coeffs_V: ArrayLike,
        frequencies_hz: ArrayLike,
    ) -> jax.Array:
        return shunt_admittance_current_coefficients(
            voltage_coeffs_V,
            frequencies_hz,
            C_F=self.C_F,
            G_S=self.G_S,
        )

    def residual(
        self,
        current_coeffs_A: ArrayLike,
        voltage_coeffs_V: ArrayLike,
        frequencies_hz: ArrayLike,
    ) -> jax.Array:
        return shunt_linear_branch_residual(
            current_coeffs_A,
            voltage_coeffs_V,
            frequencies_hz,
            C_F=self.C_F,
            G_S=self.G_S,
        )

    def with_updates(self, **kwargs: Any) -> "HBShuntLinearElement":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "G_S": _jsonify(self.G_S),
            "C_F": _jsonify(self.C_F),
            "metadata": _jsonify(dict(self.metadata or {})),
        }


# ---------------------------------------------------------------------------
# Local branch state/result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HBBranchState:
    """
    Generic branch state.

    Attributes
    ----------
    voltage_drop_coeffs_V:
        Branch voltage coefficients.
    current_coeffs_A:
        Branch current coefficients.
    """

    voltage_drop_coeffs_V: jax.Array
    current_coeffs_A: jax.Array

    def __post_init__(self) -> None:
        v = _as_complex_coeffs("voltage_drop_coeffs_V", self.voltage_drop_coeffs_V)
        i = _as_complex_coeffs("current_coeffs_A", self.current_coeffs_A)
        if v.shape != i.shape:
            raise ValueError(f"branch voltage/current shapes must match, got {v.shape}, {i.shape}")
        object.__setattr__(self, "voltage_drop_coeffs_V", v)
        object.__setattr__(self, "current_coeffs_A", i)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.voltage_drop_coeffs_V.shape

    def with_updates(self, **kwargs: Any) -> "HBBranchState":
        return replace(self, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "voltage_drop": coefficient_power_summary(self.voltage_drop_coeffs_V),
            "current": coefficient_power_summary(self.current_coeffs_A),
        }


@dataclass(frozen=True)
class HBElementEvaluation:
    """
    Evaluation result for one HB element.
    """

    residual: jax.Array
    voltage_coeffs_V: jax.Array | None = None
    current_coeffs_A: jax.Array | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def residual_norm(self) -> float:
        return float(jnp.linalg.norm(jnp.ravel(self.residual)))

    @property
    def residual_max_abs(self) -> float:
        return float(jnp.max(jnp.abs(self.residual)))

    def summary(self) -> dict[str, Any]:
        return {
            "residual": coefficient_power_summary(self.residual),
            "voltage": (
                None
                if self.voltage_coeffs_V is None
                else coefficient_power_summary(self.voltage_coeffs_V)
            ),
            "current": (
                None
                if self.current_coeffs_A is None
                else coefficient_power_summary(self.current_coeffs_A)
            ),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Local linearization helpers
# ---------------------------------------------------------------------------

def element_residual_jvp(
    residual_fn: Any,
    state: PyTree,
    tangent_state: PyTree,
) -> tuple[PyTree, PyTree]:
    """
    Generic JVP helper for an element residual function.

    residual_fn must accept one state PyTree.
    """
    return jax.jvp(residual_fn, (state,), (tangent_state,))


def linearize_element_residual(
    residual_fn: Any,
    state: PyTree,
) -> tuple[PyTree, Any]:
    """
    Return residual(state) and a linear map dstate -> dresidual.
    """
    return jax.linearize(residual_fn, state)


def local_jacobian_real(
    residual_fn: Any,
    x_complex: ArrayLike,
) -> jax.Array:
    """
    Dense real Jacobian of a local complex residual function.

    residual_fn accepts and returns complex arrays with the same flattened
    convention. This helper is for diagnostics only.
    """
    x = _as_complex_coeffs("x_complex", x_complex)
    shape = x.shape

    def pack(z: jax.Array) -> jax.Array:
        return jnp.concatenate([jnp.real(z).ravel(), jnp.imag(z).ravel()])

    def unpack(v: jax.Array) -> jax.Array:
        n = int(jnp.prod(jnp.asarray(shape)))
        return v[:n].reshape(shape) + 1j * v[n:].reshape(shape)

    def real_fn(v: jax.Array) -> jax.Array:
        return pack(residual_fn(unpack(v)))

    return jax.jacfwd(real_fn)(pack(x))


# ---------------------------------------------------------------------------
# Validation diagnostics
# ---------------------------------------------------------------------------

def validate_linear_series_element(
    frequencies_hz: ArrayLike,
    *,
    R_ohm: float = 3.0,
    L_H: float = 1e-9,
    current_scale_A: float = 1e-6,
) -> dict[str, Any]:
    """
    Sanity check for linear series R-L element.
    """
    f = _as_frequency_array(frequencies_hz)
    current = jnp.ones((f.shape[0],), dtype=jnp.complex128) * current_scale_A
    elem = HBSeriesLinearElement(R_ohm=R_ohm, L_H=L_H)
    voltage = elem.voltage_coefficients(current, f)
    residual = elem.residual(voltage, current, f)

    max_res = float(jnp.max(jnp.abs(residual)))
    return {
        "passed": bool(max_res < 1e-18),
        "max_residual_abs": max_res,
        "element": elem.to_dict(),
        "voltage_summary": coefficient_power_summary(voltage),
    }


def validate_linear_shunt_element(
    frequencies_hz: ArrayLike,
    *,
    G_S: float = 1e-6,
    C_F: float = 1e-12,
    voltage_scale_V: float = 1e-3,
) -> dict[str, Any]:
    """
    Sanity check for linear shunt G-C element.
    """
    f = _as_frequency_array(frequencies_hz)
    voltage = jnp.ones((f.shape[0],), dtype=jnp.complex128) * voltage_scale_V
    elem = HBShuntLinearElement(G_S=G_S, C_F=C_F)
    current = elem.current_coefficients(voltage, f)
    residual = elem.residual(current, voltage, f)

    max_res = float(jnp.max(jnp.abs(residual)))
    return {
        "passed": bool(max_res < 1e-18),
        "max_residual_abs": max_res,
        "element": elem.to_dict(),
        "current_summary": coefficient_power_summary(current),
    }


def validate_ki_element_linear_limit(
    frequencies_hz: ArrayLike,
    *,
    L0_H: float = 1e-9,
    I_star_A: float = 1e-3,
    current_scale_A: float = 1e-12,
    R_series_ohm: float = 0.0,
    projection_config: HBProjectionConfig | None = None,
) -> dict[str, Any]:
    """
    Check that the KI element approaches a linear inductor at tiny current.
    """
    f = _as_frequency_array(frequencies_hz)
    current = jnp.ones((f.shape[0],), dtype=jnp.complex128) * current_scale_A

    model = KineticInductanceModel.kinetic(L0_H=L0_H, I_star_A=I_star_A)
    elem = HBSeriesKineticInductor(model=model, R_series_ohm=R_series_ohm)

    v_ki = elem.voltage_coefficients(
        current,
        f,
        config=projection_config,
        fundamental_frequency_hz=float(jnp.min(jnp.abs(f[f != 0.0]))) if bool(jnp.any(f != 0.0)) else None,
    )
    v_lin = series_rl_voltage_coefficients(
        current,
        f,
        R_ohm=R_series_ohm,
        L_H=L0_H,
    )

    denom = jnp.maximum(jnp.linalg.norm(v_lin), 1e-300)
    rel_err = jnp.linalg.norm(v_ki - v_lin) / denom

    return {
        "passed": bool(float(rel_err) < 1e-6),
        "relative_error": float(rel_err),
        "element": elem.to_dict(),
        "v_ki_summary": coefficient_power_summary(v_ki),
        "v_linear_summary": coefficient_power_summary(v_lin),
    }


def run_hb_element_self_checks() -> dict[str, Any]:
    """
    Run a compact self-check suite for this module.
    """
    f = jnp.asarray([1.0e9, 2.0e9, 3.0e9], dtype=jnp.float64)

    series = validate_linear_series_element(f)
    shunt = validate_linear_shunt_element(f)
    ki = validate_ki_element_linear_limit(
        jnp.asarray([-3.0e9, -1.0e9, 1.0e9, 3.0e9], dtype=jnp.float64),
        projection_config=HBProjectionConfig(
            n_time_samples=256,
            force_real_time_signal=True,
            enforce_conjugate_symmetry=True,
        ),
    )

    return {
        "passed": bool(series["passed"] and shunt["passed"] and ki["passed"]),
        "linear_series": series,
        "linear_shunt": shunt,
        "ki_linear_limit": ki,
    }


__all__ = [
    "HBElementKind",
    "KineticInductanceHBElement",
    "flux_coeffs",
    "voltage_coeffs",
    "residual",
    "voltage_jacobian",
    "series_resistor_voltage_coefficients",
    "series_inductor_voltage_coefficients",
    "series_rl_voltage_coefficients",
    "shunt_capacitor_current_coefficients",
    "shunt_conductance_current_coefficients",
    "shunt_admittance_current_coefficients",
    "branch_voltage_drop",
    "series_linear_branch_residual",
    "shunt_linear_branch_residual",
    "current_source_residual",
    "voltage_source_residual",
    "kcl_residual",
    "HBSeriesKineticInductor",
    "HBSeriesLinearElement",
    "HBShuntLinearElement",
    "HBBranchState",
    "HBElementEvaluation",
    "element_residual_jvp",
    "linearize_element_residual",
    "local_jacobian_real",
    "validate_linear_series_element",
    "validate_linear_shunt_element",
    "validate_ki_element_linear_limit",
    "run_hb_element_self_checks",
]
