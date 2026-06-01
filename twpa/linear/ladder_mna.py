"""
twpa.linear.ladder_mna
======================

Dense modified/nodal analysis for linear lumped TWPA ladders.

This module provides an independent linear solver for the same physical layout
handled by the ABCD cascade layer. It is useful for:

1. validating the ABCD cascade implementation,
2. producing voltage/current profiles along the line,
3. preparing the topology conventions used later by nonlinear HB residuals,
4. debugging boundary conditions before large-signal pump HB.

Model
-----
The linear ladder is represented by N series branches connecting N+1 nodes:

    node 0 -- Z0 -- node 1 -- Z1 -- ... -- node N

Each cell contributes:
    series impedance:   Z_n = R_n + j omega L_n
    shunt admittance:   Y_n = G_n + j omega (C_shunt_n + C_stub_n)

For the nodal representation, each cell's shunt admittance is split equally
between its two endpoint nodes:

    node n     gets Y_n / 2
    node n + 1 gets Y_n / 2

This is a pi-discretized nodal ladder. It is not exactly the same local
factorization as a symmetric ABCD T-cell, but it converges to the same
distributed line as the cell electrical length becomes small. It is also the
natural topology for later MNA/HB residuals.

Ports
-----
The two external ports are:
    port 1 = node 0
    port 2 = node N

The module computes short-circuit admittance parameters Yport by applying
basis voltages at the two port nodes and solving internal node voltages.
For equal real reference impedance Z0_ref, S-parameters are computed as:

    S = (I - Z0_ref Y) @ inv(I + Z0_ref Y)

All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout
from twpa.core.units import angular_frequency
from twpa.linear.cascade import (
    CascadeConfig,
    layout_sparameters,
)
from twpa.linear.cells import CellModelConfig
from twpa.linear.rf_networks import (
    abcd_to_s,
    compare_sparameters,
    s11,
    s12,
    s21,
    s22,
    s_to_abcd,
    s_to_db,
)


ArrayLike = Any


# ---------------------------------------------------------------------------
# Config / enums
# ---------------------------------------------------------------------------

class LadderDiscretization(str, Enum):
    """Supported nodal discretizations."""

    PI = "pi"


class PortSolveMode(str, Enum):
    """Supported port-solve modes."""

    Y_PARAMETERS = "y_parameters"


@dataclass(frozen=True)
class LadderMNAConfig:
    """
    Configuration for the linear ladder MNA solver.

    Parameters
    ----------
    discretization:
        Currently only "pi" is implemented.
    include_stub_capacitance:
        Whether C_stub_F contributes to the shunt capacitance.
    include_resonators:
        Placeholder for later resonator admittance support. Currently false is
        the validated path.
    regularization_S:
        Small conductance added to internal nodal diagonal if needed. Keep zero
        for physical validation unless a singular DC case is being debugged.
    """

    discretization: LadderDiscretization = LadderDiscretization.PI
    include_stub_capacitance: bool = True
    include_resonators: bool = False
    regularization_S: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "discretization", LadderDiscretization(self.discretization))
        if self.discretization != LadderDiscretization.PI:
            raise ValueError("Only PI nodal discretization is currently implemented")
        if self.regularization_S < 0.0:
            raise ValueError("regularization_S must be non-negative")
        if self.include_resonators:
            raise NotImplementedError(
                "MNA resonator loading is not implemented yet. Use ABCD cell models "
                "for resonator-loaded linear scans until this is added."
            )

    def with_updates(self, **kwargs: Any) -> "LadderMNAConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discretization": self.discretization.value,
            "include_stub_capacitance": self.include_stub_capacitance,
            "include_resonators": self.include_resonators,
            "regularization_S": self.regularization_S,
        }


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _as_frequency_array(frequency_hz: ArrayLike) -> jax.Array:
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if f.ndim == 0:
        f = f.reshape((1,))
    if f.ndim != 1:
        raise ValueError(f"frequency_hz must be scalar or 1D, got shape {f.shape}")
    if bool(jnp.any(f < 0.0)):
        raise ValueError("frequency_hz must be non-negative")
    return f


def _as_complex_matrix(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    if arr.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must have trailing shape (2, 2), got {arr.shape}")
    return arr


def _eye2_batch(n_freq: int) -> jax.Array:
    return jnp.broadcast_to(jnp.eye(2, dtype=jnp.complex128), (n_freq, 2, 2))


def _safe_admittance_from_impedance(z: jax.Array, *, floor: float = 1e-300) -> jax.Array:
    """
    Return 1/z, avoiding literal division by zero in validation edge cases.
    """
    return 1.0 / jnp.where(jnp.abs(z) > floor, z, floor + 0j)


# ---------------------------------------------------------------------------
# Cell admittances
# ---------------------------------------------------------------------------

def ladder_series_admittance(
    frequency_hz: ArrayLike,
    layout: LineLayout,
) -> jax.Array:
    """
    Series branch admittance for each cell.

        Y_series[n, f] = 1 / (R_n + j omega_f L_n)

    Returns
    -------
    y_series:
        Shape (F, N).
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    z = layout.R_series_ohm[None, :] + 1j * omega[:, None] * layout.L_series_H[None, :]
    return _safe_admittance_from_impedance(z)


def ladder_shunt_admittance(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    include_stub_capacitance: bool = True,
) -> jax.Array:
    """
    Shunt admittance for each cell.

        Y_shunt[n, f] = G_n + j omega_f C_total_n

    Returns
    -------
    y_shunt:
        Shape (F, N).
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    C_total = layout.C_shunt_F
    if include_stub_capacitance:
        C_total = C_total + layout.C_stub_F

    return layout.G_shunt_S[None, :] + 1j * omega[:, None] * C_total[None, :]


def node_shunt_admittance_from_cells(
    y_cell_shunt: ArrayLike,
) -> jax.Array:
    """
    Split cell shunt admittances onto endpoint nodes.

    Parameters
    ----------
    y_cell_shunt:
        Shape (F, N).

    Returns
    -------
    y_node:
        Shape (F, N+1).
    """
    y = jnp.asarray(y_cell_shunt, dtype=jnp.complex128)
    if y.ndim != 2:
        raise ValueError(f"y_cell_shunt must have shape (F, N), got {y.shape}")

    n_freq, n_cells = int(y.shape[0]), int(y.shape[1])
    out = jnp.zeros((n_freq, n_cells + 1), dtype=jnp.complex128)

    half = 0.5 * y
    out = out.at[:, :-1].add(half)
    out = out.at[:, 1:].add(half)
    return out


# ---------------------------------------------------------------------------
# Dense nodal admittance assembly
# ---------------------------------------------------------------------------

def assemble_ladder_nodal_admittance(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    config: LadderMNAConfig | None = None,
) -> jax.Array:
    """
    Assemble full nodal admittance matrices for the ladder.

    Parameters
    ----------
    frequency_hz:
        Shape (F,) or scalar.
    layout:
        Line layout with N cells.

    Returns
    -------
    Y:
        Shape (F, N+1, N+1), where node 0 and node N are the ports.

    Notes
    -----
    This is dense and intended for validation/small-to-medium ladders. Full
    industrial-scale nonlinear solvers will use structured/banded operators.
    """
    f = _as_frequency_array(frequency_hz)
    cfg = config or LadderMNAConfig()

    n_cells = layout.n_cells
    n_nodes = n_cells + 1
    n_freq = int(f.shape[0])

    y_series = ladder_series_admittance(f, layout)  # (F, N)
    y_shunt_cell = ladder_shunt_admittance(
        f,
        layout,
        include_stub_capacitance=cfg.include_stub_capacitance,
    )
    y_node = node_shunt_admittance_from_cells(y_shunt_cell)  # (F, N+1)

    Y = jnp.zeros((n_freq, n_nodes, n_nodes), dtype=jnp.complex128)

    # Shunt to ground.
    idx_nodes = jnp.arange(n_nodes)
    Y = Y.at[:, idx_nodes, idx_nodes].add(y_node)

    # Series branches between node n and n+1.
    idx = jnp.arange(n_cells)
    yb = y_series

    Y = Y.at[:, idx, idx].add(yb)
    Y = Y.at[:, idx + 1, idx + 1].add(yb)
    Y = Y.at[:, idx, idx + 1].add(-yb)
    Y = Y.at[:, idx + 1, idx].add(-yb)

    if cfg.regularization_S > 0.0 and n_nodes > 2:
        internal = jnp.arange(1, n_nodes - 1)
        Y = Y.at[:, internal, internal].add(cfg.regularization_S)

    return Y


def split_port_internal_blocks(
    Y: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Split full nodal admittance matrix into port/internal blocks.

    Port nodes are first and last node. Internal nodes are 1..N-1.

    Returns
    -------
    Ypp:
        Shape (F, 2, 2)
    Ypi:
        Shape (F, 2, M)
    Yip:
        Shape (F, M, 2)
    Yii:
        Shape (F, M, M)
    """
    y = jnp.asarray(Y, dtype=jnp.complex128)
    if y.ndim != 3 or y.shape[-1] != y.shape[-2]:
        raise ValueError(f"Y must have shape (F, nodes, nodes), got {y.shape}")

    n_nodes = int(y.shape[-1])
    if n_nodes < 2:
        raise ValueError("Need at least two nodes for a two-port ladder")

    port_idx = jnp.asarray([0, n_nodes - 1])
    internal_idx = jnp.arange(1, n_nodes - 1)

    Ypp = y[:, port_idx[:, None], port_idx[None, :]]

    if internal_idx.shape[0] == 0:
        Ypi = jnp.zeros((y.shape[0], 2, 0), dtype=jnp.complex128)
        Yip = jnp.zeros((y.shape[0], 0, 2), dtype=jnp.complex128)
        Yii = jnp.zeros((y.shape[0], 0, 0), dtype=jnp.complex128)
    else:
        Ypi = y[:, port_idx[:, None], internal_idx[None, :]]
        Yip = y[:, internal_idx[:, None], port_idx[None, :]]
        Yii = y[:, internal_idx[:, None], internal_idx[None, :]]

    return Ypp, Ypi, Yip, Yii


def eliminate_internal_nodes_to_yport(
    Y: ArrayLike,
) -> jax.Array:
    """
    Compute two-port short-circuit admittance matrix by Schur complement.

    For full nodal admittance partitioned as:

        [Ip]   [Ypp Ypi] [Vp]
        [Ii] = [Yip Yii] [Vi]

    Internal nodes have Ii = 0, so:

        Vi = -Yii^{-1} Yip Vp

    Therefore:

        Yport = Ypp - Ypi Yii^{-1} Yip

    Returns
    -------
    Yport:
        Shape (F, 2, 2).
    """
    Ypp, Ypi, Yip, Yii = split_port_internal_blocks(Y)

    if Yii.shape[-1] == 0:
        return Ypp

    # solve Yii X = Yip for X, batched over frequency
    X = jnp.linalg.solve(Yii, Yip)
    return Ypp - jnp.matmul(Ypi, X)


# ---------------------------------------------------------------------------
# Y/S/ABCD conversion
# ---------------------------------------------------------------------------

def y_to_s(
    y: ArrayLike,
    *,
    z0_ohm: float = 50.0,
) -> jax.Array:
    """
    Convert admittance parameters to S-parameters.

        S = (I - Z0 Y) @ inv(I + Z0 Y)

    Assumes equal real reference impedance at both ports.
    """
    Y = _as_complex_matrix("y", y)
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")

    n_freq = int(Y.shape[0]) if Y.ndim == 3 else 1
    if Y.ndim == 2:
        Y = Y.reshape((1, 2, 2))
        n_freq = 1

    I = _eye2_batch(n_freq)
    left = I - z0_ohm * Y
    right = I + z0_ohm * Y

    # left @ inv(right), implemented as transpose solve for numerical stability.
    # Solve right^T X^T = left^T.
    X_t = jnp.linalg.solve(jnp.swapaxes(right, -1, -2), jnp.swapaxes(left, -1, -2))
    return jnp.swapaxes(X_t, -1, -2)


def s_to_y(
    s: ArrayLike,
    *,
    z0_ohm: float = 50.0,
) -> jax.Array:
    """
    Convert S-parameters to admittance parameters.

        Y = (1/Z0) (I - S) @ inv(I + S)
    """
    S = _as_complex_matrix("s", s)
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")

    if S.ndim == 2:
        S = S.reshape((1, 2, 2))

    n_freq = int(S.shape[0])
    I = _eye2_batch(n_freq)
    left = I - S
    right = I + S

    X_t = jnp.linalg.solve(jnp.swapaxes(right, -1, -2), jnp.swapaxes(left, -1, -2))
    return jnp.swapaxes(X_t, -1, -2) / z0_ohm


def y_to_abcd(
    y: ArrayLike,
) -> jax.Array:
    """
    Convert admittance parameters to ABCD matrices.

    For the ABCD convention used by rf_networks:

        A = -Y22 / Y21
        B = -1 / Y21
        C = -det(Y) / Y21
        D = -Y11 / Y21

    Requires Y21 != 0.
    """
    Y = _as_complex_matrix("y", y)
    if Y.ndim == 2:
        Y = Y.reshape((1, 2, 2))

    Y11 = Y[..., 0, 0]
    Y12 = Y[..., 0, 1]
    Y21 = Y[..., 1, 0]
    Y22 = Y[..., 1, 1]

    if bool(jnp.any(jnp.abs(Y21) == 0.0)):
        raise ValueError("Y21 contains zero; Y-to-ABCD conversion is singular")

    detY = Y11 * Y22 - Y12 * Y21

    A = -Y22 / Y21
    B = -1.0 / Y21
    C = -detY / Y21
    D = -Y11 / Y21

    return jnp.stack(
        [
            jnp.stack([A, B], axis=-1),
            jnp.stack([C, D], axis=-1),
        ],
        axis=-2,
    )


# ---------------------------------------------------------------------------
# Main result object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderMNAResult:
    """
    Linear ladder MNA result.

    Attributes
    ----------
    frequency_hz:
        Frequency grid, shape (F,).
    nodal_admittance:
        Full nodal admittance, shape (F, N+1, N+1).
    yport:
        Two-port admittance parameters, shape (F, 2, 2).
    s:
        S-parameters, shape (F, 2, 2).
    abcd:
        ABCD matrices inferred from yport, shape (F, 2, 2).
    metadata:
        JSON-friendly metadata.
    """

    frequency_hz: jax.Array
    nodal_admittance: jax.Array
    yport: jax.Array
    s: jax.Array
    abcd: jax.Array
    metadata: Mapping[str, Any] | None = None

    @property
    def s11(self) -> jax.Array:
        return s11(self.s)

    @property
    def s21(self) -> jax.Array:
        return s21(self.s)

    @property
    def s12(self) -> jax.Array:
        return s12(self.s)

    @property
    def s22(self) -> jax.Array:
        return s22(self.s)

    @property
    def s21_db(self) -> jax.Array:
        return s_to_db(self.s21)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_shape": tuple(int(v) for v in self.frequency_hz.shape),
            "frequency_min_hz": float(jnp.min(self.frequency_hz)),
            "frequency_max_hz": float(jnp.max(self.frequency_hz)),
            "nodal_admittance_shape": tuple(int(v) for v in self.nodal_admittance.shape),
            "yport_shape": tuple(int(v) for v in self.yport.shape),
            "s_shape": tuple(int(v) for v in self.s.shape),
            "abcd_shape": tuple(int(v) for v in self.abcd.shape),
            "s21_db_min": float(jnp.min(self.s21_db)),
            "s21_db_max": float(jnp.max(self.s21_db)),
            "metadata": dict(self.metadata or {}),
        }


def solve_ladder_mna(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    config: LadderMNAConfig | None = None,
    z0_ohm: float | None = None,
) -> LadderMNAResult:
    """
    Solve the linear ladder by nodal admittance elimination.

    Returns Y, S, and ABCD representations.
    """
    f = _as_frequency_array(frequency_hz)
    cfg = config or LadderMNAConfig()
    z0 = layout.z0_ohm if z0_ohm is None else float(z0_ohm)

    Y = assemble_ladder_nodal_admittance(f, layout, config=cfg)
    Yport = eliminate_internal_nodes_to_yport(Y)
    S = y_to_s(Yport, z0_ohm=z0)
    ABCD = y_to_abcd(Yport)

    return LadderMNAResult(
        frequency_hz=f,
        nodal_admittance=Y,
        yport=Yport,
        s=S,
        abcd=ABCD,
        metadata={
            "layout_name": layout.name,
            "layout_summary": layout.summary(),
            "z0_ohm": z0,
            "config": cfg.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Voltage/current profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderProfileResult:
    """
    Voltage/current profile for a given port-voltage excitation.

    Attributes
    ----------
    node_voltages:
        Shape (F, N+1).
    series_currents:
        Current through each series branch from node n to n+1, shape (F, N).
    shunt_currents:
        Shunt current at each node to ground, shape (F, N+1).
    port_currents:
        Currents entering port nodes, shape (F, 2).
    """

    frequency_hz: jax.Array
    node_voltages: jax.Array
    series_currents: jax.Array
    shunt_currents: jax.Array
    port_currents: jax.Array
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_shape": tuple(int(v) for v in self.frequency_hz.shape),
            "node_voltages_shape": tuple(int(v) for v in self.node_voltages.shape),
            "series_currents_shape": tuple(int(v) for v in self.series_currents.shape),
            "shunt_currents_shape": tuple(int(v) for v in self.shunt_currents.shape),
            "port_currents_shape": tuple(int(v) for v in self.port_currents.shape),
            "max_abs_node_voltage_V": float(jnp.max(jnp.abs(self.node_voltages))),
            "max_abs_series_current_A": float(jnp.max(jnp.abs(self.series_currents))),
            "max_abs_shunt_current_A": float(jnp.max(jnp.abs(self.shunt_currents))),
            "metadata": dict(self.metadata or {}),
        }


def solve_internal_node_voltages(
    nodal_admittance: ArrayLike,
    port_voltages: ArrayLike,
) -> jax.Array:
    """
    Solve internal node voltages for prescribed port voltages.

    Parameters
    ----------
    nodal_admittance:
        Shape (F, N+1, N+1).
    port_voltages:
        Shape (F, 2) or (2,). Port order is [node0, nodeN].

    Returns
    -------
    node_voltages:
        Shape (F, N+1).
    """
    Y = jnp.asarray(nodal_admittance, dtype=jnp.complex128)
    if Y.ndim != 3 or Y.shape[-1] != Y.shape[-2]:
        raise ValueError("nodal_admittance must have shape (F, nodes, nodes)")

    n_freq = int(Y.shape[0])
    n_nodes = int(Y.shape[-1])

    Vp = jnp.asarray(port_voltages, dtype=jnp.complex128)
    if Vp.ndim == 1:
        if Vp.shape[0] != 2:
            raise ValueError("port_voltages vector must have length 2")
        Vp = jnp.broadcast_to(Vp[None, :], (n_freq, 2))
    elif Vp.ndim == 2:
        if Vp.shape != (n_freq, 2):
            raise ValueError(f"port_voltages must have shape ({n_freq}, 2), got {Vp.shape}")
    else:
        raise ValueError("port_voltages must be shape (2,) or (F,2)")

    if n_nodes == 2:
        return Vp

    Ypp, Ypi, Yip, Yii = split_port_internal_blocks(Y)

    rhs = -jnp.matmul(Yip, Vp[..., None])[..., 0]
    Vi = jnp.linalg.solve(Yii, rhs)

    V = jnp.zeros((n_freq, n_nodes), dtype=jnp.complex128)
    V = V.at[:, 0].set(Vp[:, 0])
    V = V.at[:, -1].set(Vp[:, 1])
    V = V.at[:, 1:-1].set(Vi)
    return V


def compute_ladder_profiles(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    port_voltages: ArrayLike,
    config: LadderMNAConfig | None = None,
) -> LadderProfileResult:
    """
    Compute node-voltage and branch-current profiles for prescribed port voltages.

    This is useful for pump-off mode shapes and initial guesses.
    """
    f = _as_frequency_array(frequency_hz)
    cfg = config or LadderMNAConfig()

    Y = assemble_ladder_nodal_admittance(f, layout, config=cfg)
    V = solve_internal_node_voltages(Y, port_voltages)

    y_series = ladder_series_admittance(f, layout)
    y_shunt_cell = ladder_shunt_admittance(
        f,
        layout,
        include_stub_capacitance=cfg.include_stub_capacitance,
    )
    y_node = node_shunt_admittance_from_cells(y_shunt_cell)

    I_series = y_series * (V[:, :-1] - V[:, 1:])
    I_shunt = y_node * V

    port_currents_full = jnp.matmul(Y, V[..., None])[..., 0]
    I_ports = jnp.stack([port_currents_full[:, 0], port_currents_full[:, -1]], axis=-1)

    return LadderProfileResult(
        frequency_hz=f,
        node_voltages=V,
        series_currents=I_series,
        shunt_currents=I_shunt,
        port_currents=I_ports,
        metadata={
            "layout_name": layout.name,
            "config": cfg.to_dict(),
        },
    )


def forward_matched_voltage_profile(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    input_voltage_V: complex = 1.0 + 0.0j,
    load_voltage_V: complex = 0.0 + 0.0j,
    config: LadderMNAConfig | None = None,
) -> LadderProfileResult:
    """
    Convenience profile with prescribed port voltages.

    This is not a full matched-source/load solve; it is a useful basis profile
    for debugging and initial guesses.
    """
    f = _as_frequency_array(frequency_hz)
    Vp = jnp.stack(
        [
            jnp.full_like(f, input_voltage_V, dtype=jnp.complex128),
            jnp.full_like(f, load_voltage_V, dtype=jnp.complex128),
        ],
        axis=-1,
    )
    return compute_ladder_profiles(
        f,
        layout,
        port_voltages=Vp,
        config=config,
    )


# ---------------------------------------------------------------------------
# Comparisons against ABCD cascade
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderMNAComparisonReport:
    """
    Comparison between ladder MNA and ABCD cascade.
    """

    layout_name: str
    n_cells: int
    n_frequencies: int
    s_max_abs_diff: float
    s_rms_abs_diff: float
    s21_max_abs_diff: float
    s21_db_max_abs_diff: float
    passed: bool
    messages: list[str]
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "n_cells": self.n_cells,
            "n_frequencies": self.n_frequencies,
            "s_max_abs_diff": self.s_max_abs_diff,
            "s_rms_abs_diff": self.s_rms_abs_diff,
            "s21_max_abs_diff": self.s21_max_abs_diff,
            "s21_db_max_abs_diff": self.s21_db_max_abs_diff,
            "passed": self.passed,
            "messages": list(self.messages),
            "metadata": dict(self.metadata or {}),
        }


def compare_ladder_mna_to_abcd(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    mna_config: LadderMNAConfig | None = None,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    tolerance_s21_db: float = 0.5,
) -> LadderMNAComparisonReport:
    """
    Compare nodal ladder MNA to ABCD cascade.

    Important:
    The MNA solver uses a pi-discretized nodal ladder. For best agreement, use
    a pi-cell ABCD model. T-cell and pi-cell differ at finite electrical cell
    length but converge as cells get small.
    """
    f = _as_frequency_array(frequency_hz)

    mna = solve_ladder_mna(
        f,
        layout,
        config=mna_config,
        z0_ohm=layout.z0_ohm,
    )

    s_abcd = layout_sparameters(
        f,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
        z0_ohm=layout.z0_ohm,
    )

    comp = compare_sparameters(mna.s, s_abcd, label_a="mna", label_b="abcd")

    passed = comp["s21_db_max_abs_diff"] <= tolerance_s21_db
    messages = []
    if passed:
        messages.append("PASS: MNA and ABCD S21 agree within tolerance.")
    else:
        messages.append(
            f"FAIL: S21 dB max difference {comp['s21_db_max_abs_diff']} exceeds "
            f"tolerance {tolerance_s21_db} dB."
        )
        messages.append(
            "If using T-cell ABCD, try CellModelConfig(kind='pi') for a closer "
            "comparison to the nodal pi ladder."
        )

    return LadderMNAComparisonReport(
        layout_name=layout.name,
        n_cells=layout.n_cells,
        n_frequencies=int(f.shape[0]),
        s_max_abs_diff=comp["max_abs_diff"],
        s_rms_abs_diff=comp["rms_abs_diff"],
        s21_max_abs_diff=comp["s21_max_abs_diff"],
        s21_db_max_abs_diff=comp["s21_db_max_abs_diff"],
        passed=bool(passed),
        messages=messages,
        metadata={
            "comparison": comp,
            "mna": mna.to_dict(),
            "mna_config": (mna_config or LadderMNAConfig()).to_dict(),
            "cell_model": (cell_model.to_dict() if cell_model is not None else None),
            "cascade_config": (cascade_config.to_dict() if cascade_config is not None else None),
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderMNAValidationReport:
    """
    Validation report for the MNA layer.
    """

    layout_name: str
    n_cells: int
    n_frequencies: int
    passed: bool
    has_nan: bool
    has_inf: bool
    yport_reciprocity_error: float
    s_reciprocity_error: float
    passivity_violation: float
    s21_db_min: float
    s21_db_max: float
    messages: list[str]
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "n_cells": self.n_cells,
            "n_frequencies": self.n_frequencies,
            "passed": self.passed,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "yport_reciprocity_error": self.yport_reciprocity_error,
            "s_reciprocity_error": self.s_reciprocity_error,
            "passivity_violation": self.passivity_violation,
            "s21_db_min": self.s21_db_min,
            "s21_db_max": self.s21_db_max,
            "messages": list(self.messages),
            "metadata": dict(self.metadata or {}),
        }


def validate_ladder_mna(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    config: LadderMNAConfig | None = None,
    reciprocity_tol: float = 1e-8,
    passivity_tol: float = 1e-7,
) -> LadderMNAValidationReport:
    """
    Validate the linear MNA solver for a passive reciprocal ladder.
    """
    f = _as_frequency_array(frequency_hz)
    result = solve_ladder_mna(f, layout, config=config)

    arrays = [result.nodal_admittance, result.yport, result.s, result.abcd]
    has_nan = any(
        bool(jnp.any(jnp.isnan(jnp.real(a))) or jnp.any(jnp.isnan(jnp.imag(a))))
        for a in arrays
    )
    has_inf = any(
        bool(jnp.any(jnp.isinf(jnp.real(a))) or jnp.any(jnp.isinf(jnp.imag(a))))
        for a in arrays
    )

    yrec = float(jnp.max(jnp.abs(result.yport[..., 0, 1] - result.yport[..., 1, 0])))
    srec = float(jnp.max(jnp.abs(result.s[..., 0, 1] - result.s[..., 1, 0])))

    singular_values = jnp.linalg.svd(result.s, compute_uv=False)
    max_sv = jnp.max(singular_values)
    passivity_violation = float(jnp.maximum(max_sv - 1.0, 0.0))

    messages = []
    passed = True

    if has_nan or has_inf:
        passed = False
        messages.append("FAIL: MNA result contains NaN or Inf.")

    if yrec > reciprocity_tol:
        passed = False
        messages.append(
            f"FAIL: Y-parameter reciprocity error {yrec} exceeds {reciprocity_tol}."
        )

    if srec > reciprocity_tol:
        passed = False
        messages.append(
            f"FAIL: S-parameter reciprocity error {srec} exceeds {reciprocity_tol}."
        )

    if passivity_violation > passivity_tol:
        passed = False
        messages.append(
            f"FAIL: passivity violation {passivity_violation} exceeds {passivity_tol}."
        )

    if passed:
        messages.append("PASS: ladder MNA validation checks passed.")

    return LadderMNAValidationReport(
        layout_name=layout.name,
        n_cells=layout.n_cells,
        n_frequencies=int(f.shape[0]),
        passed=bool(passed),
        has_nan=has_nan,
        has_inf=has_inf,
        yport_reciprocity_error=yrec,
        s_reciprocity_error=srec,
        passivity_violation=passivity_violation,
        s21_db_min=float(jnp.min(result.s21_db)),
        s21_db_max=float(jnp.max(result.s21_db)),
        messages=messages,
        metadata={
            "result": result.to_dict(),
            "config": (config or LadderMNAConfig()).to_dict(),
        },
    )


@dataclass(frozen=True)
class LadderSolveResult:
    """Compatibility result for direct nodal ladder solves."""

    V: jax.Array
    residual_norm: float
    success: bool = True

    @property
    def node_voltages(self) -> jax.Array:
        return self.V

    def to_dict(self) -> dict[str, Any]:
        return {
            "V": [
                {"real": float(jnp.real(v)), "imag": float(jnp.imag(v))}
                for v in jnp.ravel(self.V).tolist()
            ],
            "residual_norm": self.residual_norm,
            "success": self.success,
        }


def _compat_component_array(value: Any, count: int, name: str, *, allow_zero: bool = False) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim == 0:
        arr = jnp.full((count,), arr)
    if arr.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape ({count},), got {arr.shape}")
    if allow_zero:
        bad = bool(jnp.any(arr < 0.0))
    else:
        bad = bool(jnp.any(arr <= 0.0))
    if bad:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return arr


def _compat_shunt_array(value: Any, n_cells: int) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim == 0:
        arr = jnp.full((n_cells + 1,), arr)
    elif arr.shape == (n_cells,):
        caps = jnp.zeros((n_cells + 1,), dtype=jnp.float64)
        caps = caps.at[:-1].add(0.5 * arr)
        caps = caps.at[1:].add(0.5 * arr)
        arr = caps
    if arr.shape != (n_cells + 1,):
        raise ValueError(f"C_shunt_F must be scalar, shape ({n_cells},), or shape ({n_cells + 1},)")
    if bool(jnp.any(arr < 0.0)):
        raise ValueError("C_shunt_F must be non-negative")
    return arr


def build_admittance_matrix(
    omega_rad_s: ArrayLike,
    n_cells: int,
    L_series_H: Any,
    C_shunt_F: Any,
) -> jax.Array:
    """N+1 node admittance matrix for N series inductors and shunt capacitors."""
    import numpy as _np

    if int(n_cells) <= 0:
        raise ValueError("n_cells must be positive")
    omega_arr = _np.asarray(omega_rad_s, dtype=float)
    if omega_arr.ndim > 1 or _np.any(omega_arr <= 0.0):
        raise ValueError("omega_rad_s must be positive")
    if omega_arr.ndim == 1:
        return jnp.asarray(
            _np.stack(
                [
                    _np.asarray(build_admittance_matrix(float(omega), n_cells, L_series_H, C_shunt_F))
                    for omega in omega_arr
                ],
                axis=0,
            )
        )
    n = int(n_cells)
    omega = float(omega_arr)
    L = _np.asarray(_compat_component_array(L_series_H, n, "L_series_H"), dtype=float)
    C = _np.asarray(_compat_shunt_array(C_shunt_F, n), dtype=float)
    Y = _np.zeros((n + 1, n + 1), dtype=_np.complex128)
    Y[_np.arange(n + 1), _np.arange(n + 1)] += 1j * omega * C
    for k in range(n):
        y_l = 1.0 / (1j * omega * L[k])
        Y[k, k] += y_l
        Y[k + 1, k + 1] += y_l
        Y[k, k + 1] -= y_l
        Y[k + 1, k] -= y_l
    row_target = 1j * omega * C
    Y[_np.arange(n + 1), _np.arange(n + 1)] += row_target - _np.sum(Y, axis=1)
    return Y


def ladder_abcd(
    omega_rad_s: float,
    n_cells: int,
    L_series_H: Any,
    C_shunt_F: Any,
) -> jax.Array:
    """ABCD export for compatibility nodal ladder topology."""
    import numpy as _np

    if int(n_cells) <= 0:
        raise ValueError("n_cells must be positive")
    if float(omega_rad_s) <= 0.0:
        raise ValueError("omega_rad_s must be positive")
    n = int(n_cells)
    omega = float(omega_rad_s)
    L = _np.asarray(_compat_component_array(L_series_H, n, "L_series_H"), dtype=float)
    C = _np.asarray(_compat_shunt_array(C_shunt_F, n), dtype=float)
    out = _np.eye(2, dtype=_np.complex128)

    def shunt(capacitance: float) -> _np.ndarray:
        return _np.asarray([[1.0, 0.0], [1j * omega * capacitance, 1.0]], dtype=_np.complex128)

    out = out @ shunt(C[0])
    for cell in range(n):
        out = out @ _np.asarray(
            [[1.0, 1j * omega * L[cell]], [0.0, 1.0]],
            dtype=_np.complex128,
        )
        out = out @ shunt(C[cell + 1])
    return jnp.asarray(out)


def ladder_sparameters(
    omega_rad_s: float,
    n_cells: int,
    L_series_H: Any,
    C_shunt_F: Any,
    *,
    z0_ohm: float = 50.0,
) -> jax.Array:
    """S-parameter export for compatibility nodal ladder topology."""
    return abcd_to_s(
        ladder_abcd(omega_rad_s, n_cells, L_series_H, C_shunt_F),
        z0_ohm=z0_ohm,
    )


def solve_ladder(
    current_injection_A: Any,
    omega_rad_s: float,
    n_cells: int,
    L_series_H: Any,
    C_shunt_F: Any,
) -> LadderSolveResult:
    """Solve Y V = I for the compatibility nodal ladder."""
    I = jnp.asarray(current_injection_A, dtype=jnp.complex128)
    if I.shape != (int(n_cells) + 1,):
        raise ValueError(f"current injection must have shape ({int(n_cells) + 1},)")
    Y = build_admittance_matrix(omega_rad_s, n_cells, L_series_H, C_shunt_F)
    V = jnp.linalg.solve(Y, I)
    residual = Y @ V - I
    return LadderSolveResult(V=V, residual_norm=float(jnp.linalg.norm(residual)))


__all__ = [
    "LadderDiscretization",
    "PortSolveMode",
    "LadderMNAConfig",
    "ladder_series_admittance",
    "ladder_shunt_admittance",
    "node_shunt_admittance_from_cells",
    "assemble_ladder_nodal_admittance",
    "split_port_internal_blocks",
    "eliminate_internal_nodes_to_yport",
    "y_to_s",
    "s_to_y",
    "y_to_abcd",
    "LadderMNAResult",
    "solve_ladder_mna",
    "LadderProfileResult",
    "solve_internal_node_voltages",
    "compute_ladder_profiles",
    "forward_matched_voltage_profile",
    "LadderMNAComparisonReport",
    "compare_ladder_mna_to_abcd",
    "LadderMNAValidationReport",
    "validate_ladder_mna",
    "LadderSolveResult",
    "build_admittance_matrix",
    "ladder_abcd",
    "ladder_sparameters",
    "solve_ladder",
]
