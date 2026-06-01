"""
twpa.nonlinear.distributed_hb
=============================

Dense distributed harmonic-balance residual for nonlinear TWPA ladders.

This is the first topology-level distributed HB model. It is intended as the
small/medium reference implementation before the industrial 20,000-cell solver.

Topology
--------
A layout with N cells has N+1 voltage nodes and N nonlinear series branches:

    node 0 -- branch 0 -- node 1 -- ... -- branch N-1 -- node N

Each branch is oriented left-to-right:

    I_branch[n] flows from node n to node n+1.

Each cell contributes:
    - nonlinear series kinetic inductance L_n(I),
    - optional series resistance R_n,
    - shunt capacitance/conductance split half to each endpoint node.

Unknowns
--------
For K HB tones:

    V[k, m]      node voltage coefficients, shape (K, N+1)
    I[k, n]      series branch current coefficients, shape (K, N)

Residuals
---------
KCL at every node:

    r_kcl[k, m] =
        I_shunt[k, m]
      + I_leaving_right[k, m]
      + I_leaving_left[k, m]
      + I_termination[k, m]
      - I_injected[k, m]

where:

    I_leaving_right at node m = I_branch[m]       if m < N
    I_leaving_left  at node m = -I_branch[m - 1]  if m > 0

Branch KVL for each nonlinear series branch:

    r_branch[k, n] =
        V[k, n] - V[k, n+1]
      - R_n I[k, n]
      - V_KI,k(I[:, n])

This module is dense and reference-oriented. It is suitable for:
    - N ~ 1 to a few hundred, depending on K and machine,
    - validating nonlinear laws,
    - validating continuation,
    - debugging boundary/source conventions,
    - generating reference answers for later block-banded/Newton-Krylov solvers.

The full 100 mm / 20,000-cell simulator should use this module only as the
truth model on reduced/coarsened layouts, not as the final numerical backend.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from twpa.core.frequency_plan import FrequencyPlan
from twpa.core.harmonics import (
    coefficient_power_summary,
    complex_to_real_vector,
    complex_tree_to_real_vector,
    real_vector_to_complex,
    set_single_rms_phasor_by_label,
    zeros_for_plan,
)
from twpa.core.hb_fft import HBProjectionConfig, HBProjectionGrid, make_projection_grid_from_plan
from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams, SolverConfig
from twpa.nonlinear.kinetic_inductance import KineticInductanceModel
from twpa.solvers.continuation import (
    ContinuationResult,
    ContinuationSolverConfig,
    solve_pump_power_continuation,
)
from twpa.solvers.hb_solver import (
    DenseNewtonConfig,
    HBSolverResult,
    check_residual_jacobian_consistency,
    solve_hb,
)
from twpa.solvers.preconditioners import (
    Preconditioner,
    PreconditionerKind,
    PreconditionerStatus,
)


ArrayLike = Any
PyTree = Any


# ---------------------------------------------------------------------------
# Enums / configuration
# ---------------------------------------------------------------------------

class DistributedHBSourceKind(str, Enum):
    """Supported distributed-HB source conventions."""

    CURRENT_INJECTION = "current_injection"


class DistributedHBTerminationKind(str, Enum):
    """Supported port termination models."""

    SHUNT_CONDUCTANCE = "shunt_conductance"
    NONE = "none"


@dataclass(frozen=True)
class DistributedHBConfig:
    """
    Configuration for the distributed HB ladder.

    Parameters
    ----------
    source_kind:
        Source convention. Current injection is the validated first path.
    input_node:
        Node index where input current is injected.
    output_node:
        Node index used as the output/load node. Defaults to the last node in
        helper functions if set to -1.
    termination_kind:
        Port termination model.
    source_conductance_S:
        Optional shunt source conductance at input_node.
        For a 50 ohm environment, use 1/50 S.
    load_conductance_S:
        Optional shunt load conductance at output_node.
    include_stub_capacitance:
        Whether layout.C_stub_F contributes to shunt capacitance.
    include_series_resistance:
        Whether layout.R_series_ohm contributes to branch KVL.
    use_layout_shunt_conductance:
        Whether layout.G_shunt_S contributes to shunt current.
    name:
        Human-readable simulation name.
    """

    source_kind: DistributedHBSourceKind = DistributedHBSourceKind.CURRENT_INJECTION
    input_node: int = 0
    output_node: int = -1
    termination_kind: DistributedHBTerminationKind = DistributedHBTerminationKind.SHUNT_CONDUCTANCE
    source_conductance_S: float = 1.0 / 50.0
    load_conductance_S: float = 1.0 / 50.0
    include_stub_capacitance: bool = True
    include_series_resistance: bool = True
    use_layout_shunt_conductance: bool = True
    name: str = "distributed_hb"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", DistributedHBSourceKind(self.source_kind))
        object.__setattr__(self, "termination_kind", DistributedHBTerminationKind(self.termination_kind))
        object.__setattr__(self, "input_node", int(self.input_node))
        object.__setattr__(self, "output_node", int(self.output_node))

        if self.input_node < 0:
            raise ValueError("input_node must be non-negative")
        if self.source_conductance_S < 0.0:
            raise ValueError("source_conductance_S must be non-negative")
        if self.load_conductance_S < 0.0:
            raise ValueError("load_conductance_S must be non-negative")

    def resolve_output_node(self, layout: LineLayout) -> int:
        if self.output_node < 0:
            return layout.n_cells
        if self.output_node > layout.n_cells:
            raise ValueError(
                f"output_node={self.output_node} exceeds last node {layout.n_cells}"
            )
        return self.output_node

    def with_updates(self, **kwargs: Any) -> "DistributedHBConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind.value,
            "input_node": self.input_node,
            "output_node": self.output_node,
            "termination_kind": self.termination_kind.value,
            "source_conductance_S": self.source_conductance_S,
            "load_conductance_S": self.load_conductance_S,
            "include_stub_capacitance": self.include_stub_capacitance,
            "include_series_resistance": self.include_series_resistance,
            "use_layout_shunt_conductance": self.use_layout_shunt_conductance,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# State / residual objects
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DistributedHBState:
    """
    Unknown state for a distributed nonlinear HB ladder.

    Attributes
    ----------
    node_voltage_coeffs_V:
        Shape (K, N+1).
    branch_current_coeffs_A:
        Shape (K, N).
    """

    node_voltage_coeffs_V: jax.Array
    branch_current_coeffs_A: jax.Array

    def __post_init__(self) -> None:
        v = _as_complex_2d("node_voltage_coeffs_V", self.node_voltage_coeffs_V)
        i = _as_complex_2d("branch_current_coeffs_A", self.branch_current_coeffs_A)

        if v.shape[0] != i.shape[0]:
            raise ValueError(
                f"tone axes must match, got V {v.shape[0]} and I {i.shape[0]}"
            )
        if v.shape[1] != i.shape[1] + 1:
            raise ValueError(
                f"node count must be branch count + 1, got V {v.shape}, I {i.shape}"
            )

        object.__setattr__(self, "node_voltage_coeffs_V", v)
        object.__setattr__(self, "branch_current_coeffs_A", i)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        return (self.node_voltage_coeffs_V, self.branch_current_coeffs_A), {}

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array],
    ) -> "DistributedHBState":
        return cls(
            node_voltage_coeffs_V=children[0],
            branch_current_coeffs_A=children[1],
        )

    @property
    def n_tones(self) -> int:
        return int(self.node_voltage_coeffs_V.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.node_voltage_coeffs_V.shape[1])

    @property
    def n_branches(self) -> int:
        return int(self.branch_current_coeffs_A.shape[1])

    def with_updates(self, **kwargs: Any) -> "DistributedHBState":
        return replace(self, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "n_tones": self.n_tones,
            "n_nodes": self.n_nodes,
            "n_branches": self.n_branches,
            "node_voltage_coeffs_V": coefficient_power_summary(self.node_voltage_coeffs_V),
            "branch_current_coeffs_A": coefficient_power_summary(self.branch_current_coeffs_A),
        }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DistributedHBResidual:
    """
    Residual for distributed HB ladder.

    Attributes
    ----------
    kcl_A:
        KCL residuals at nodes, shape (K, N+1).
    branch_kvl_V:
        Branch KVL residuals, shape (K, N).
    """

    kcl_A: jax.Array
    branch_kvl_V: jax.Array

    def __post_init__(self) -> None:
        kcl = _as_complex_2d("kcl_A", self.kcl_A)
        kvl = _as_complex_2d("branch_kvl_V", self.branch_kvl_V)

        if kcl.shape[0] != kvl.shape[0]:
            raise ValueError(
                f"tone axes must match, got KCL {kcl.shape[0]} and KVL {kvl.shape[0]}"
            )
        if kcl.shape[1] != kvl.shape[1] + 1:
            raise ValueError(
                f"KCL node count must be KVL branch count + 1, got {kcl.shape}, {kvl.shape}"
            )

        object.__setattr__(self, "kcl_A", kcl)
        object.__setattr__(self, "branch_kvl_V", kvl)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        return (self.kcl_A, self.branch_kvl_V), {}

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array],
    ) -> "DistributedHBResidual":
        return cls(kcl_A=children[0], branch_kvl_V=children[1])

    @property
    def n_tones(self) -> int:
        return int(self.kcl_A.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.kcl_A.shape[1])

    @property
    def n_branches(self) -> int:
        return int(self.branch_kvl_V.shape[1])

    @property
    def norm(self) -> float:
        return float(
            jnp.sqrt(
                jnp.sum(jnp.abs(self.kcl_A) ** 2)
                + jnp.sum(jnp.abs(self.branch_kvl_V) ** 2)
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_tones": self.n_tones,
            "n_nodes": self.n_nodes,
            "n_branches": self.n_branches,
            "kcl_A": coefficient_power_summary(self.kcl_A),
            "branch_kvl_V": coefficient_power_summary(self.branch_kvl_V),
            "combined_norm": self.norm,
        }


@dataclass(frozen=True)
class DistributedHBSolveResult:
    """
    Full result for distributed HB solve.
    """

    state: DistributedHBState
    residual: DistributedHBResidual
    solver_result: HBSolverResult
    frequency_plan: FrequencyPlan
    layout: LineLayout
    hb_config: DistributedHBConfig
    ki_model: KineticInductanceModel
    injected_current_coeffs_A: jax.Array
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.solver_result.converged

    @property
    def input_node(self) -> int:
        return self.hb_config.input_node

    @property
    def output_node(self) -> int:
        return self.hb_config.resolve_output_node(self.layout)

    @property
    def input_voltage_coeffs_V(self) -> jax.Array:
        return self.state.node_voltage_coeffs_V[:, self.input_node]

    @property
    def output_voltage_coeffs_V(self) -> jax.Array:
        return self.state.node_voltage_coeffs_V[:, self.output_node]

    def voltage_gain_by_label(self, output_label: str, input_label: str | None = None) -> complex:
        """
        Return complex voltage ratio Vout[label] / Vin[input_label].

        If input_label is None, uses the same label as output_label.
        """
        out_pos = self.frequency_plan.position_of_label(output_label)
        in_pos = out_pos if input_label is None else self.frequency_plan.position_of_label(input_label)
        vin = self.input_voltage_coeffs_V[in_pos]
        vout = self.output_voltage_coeffs_V[out_pos]
        return complex(vout / vin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "state": self.state.summary(),
            "residual": self.residual.summary(),
            "solver": self.solver_result.report.to_dict(),
            "frequency_plan": self.frequency_plan.to_dict(),
            "layout": self.layout.summary(),
            "hb_config": self.hb_config.to_dict(),
            "ki_model": self.ki_model.to_dict(),
            "injected_current_coeffs_A": coefficient_power_summary(self.injected_current_coeffs_A),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _as_complex_2d(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}")
    return arr


def _as_plan_node_coeffs(name: str, value: ArrayLike, plan: FrequencyPlan, layout: LineLayout) -> jax.Array:
    arr = _as_complex_2d(name, value)
    expected = (plan.n_tones, layout.n_cells + 1)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return arr


def _as_plan_branch_coeffs(name: str, value: ArrayLike, plan: FrequencyPlan, layout: LineLayout) -> jax.Array:
    arr = _as_complex_2d(name, value)
    expected = (plan.n_tones, layout.n_cells)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return arr


def _omega_column(plan: FrequencyPlan) -> jax.Array:
    return plan.angular_frequencies_rad_s[:, None]


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = jnp.asarray(obj)
        if arr.ndim == 0:
            try:
                return float(arr)
            except Exception:
                return str(arr)
        return {
            "array_shape": tuple(int(s) for s in arr.shape),
            "array_dtype": str(arr.dtype),
        }
    return obj


# ---------------------------------------------------------------------------
# Layout-to-HB parameter maps
# ---------------------------------------------------------------------------

def node_shunt_capacitance_from_layout(
    layout: LineLayout,
    *,
    include_stub_capacitance: bool = True,
) -> jax.Array:
    """
    Split cell shunt capacitances onto endpoint nodes.

    Returns shape (N+1,).
    """
    C_cell = layout.C_shunt_F
    if include_stub_capacitance:
        C_cell = C_cell + layout.C_stub_F

    out = jnp.zeros((layout.n_cells + 1,), dtype=C_cell.dtype)
    half = 0.5 * C_cell
    out = out.at[:-1].add(half)
    out = out.at[1:].add(half)
    return out


def node_shunt_conductance_from_layout(
    layout: LineLayout,
    *,
    use_layout_shunt_conductance: bool = True,
) -> jax.Array:
    """
    Split cell shunt conductances onto endpoint nodes.

    Returns shape (N+1,).
    """
    if not use_layout_shunt_conductance:
        return jnp.zeros((layout.n_cells + 1,), dtype=jnp.float64)

    G_cell = layout.G_shunt_S
    out = jnp.zeros((layout.n_cells + 1,), dtype=G_cell.dtype)
    half = 0.5 * G_cell
    out = out.at[:-1].add(half)
    out = out.at[1:].add(half)
    return out


def termination_conductance_nodes(
    layout: LineLayout,
    config: DistributedHBConfig,
) -> jax.Array:
    """
    Node-wise termination conductance array, shape (N+1,).
    """
    out = jnp.zeros((layout.n_cells + 1,), dtype=jnp.float64)

    if config.termination_kind == DistributedHBTerminationKind.NONE:
        return out

    if config.termination_kind != DistributedHBTerminationKind.SHUNT_CONDUCTANCE:
        raise ValueError(f"Unsupported termination kind {config.termination_kind}")

    output_node = config.resolve_output_node(layout)
    out = out.at[config.input_node].add(config.source_conductance_S)
    out = out.at[output_node].add(config.load_conductance_S)
    return out


def total_node_admittance(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
) -> jax.Array:
    """
    Total linear node admittance to ground.

        Y_node[k, m] = G_node[m] + G_term[m] + j omega_k C_node[m]

    Returns shape (K, N+1).
    """
    C_node = node_shunt_capacitance_from_layout(
        layout,
        include_stub_capacitance=config.include_stub_capacitance,
    )
    G_node = node_shunt_conductance_from_layout(
        layout,
        use_layout_shunt_conductance=config.use_layout_shunt_conductance,
    )
    G_term = termination_conductance_nodes(layout, config)

    return (G_node + G_term)[None, :] + 1j * _omega_column(plan) * C_node[None, :]


def make_kinetic_model_from_layout(
    layout: LineLayout,
    nonlinear: NonlinearParams,
    *,
    name: str = "layout_ki_model",
) -> KineticInductanceModel:
    """
    Build a branch-wise KI model from layout series inductances.
    """
    return KineticInductanceModel.from_params(
        L0_H=layout.L_series_H,
        params=nonlinear,
        name=name,
    )


# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------

def zeros_node_injection(plan: FrequencyPlan, layout: LineLayout) -> jax.Array:
    """
    Zero current-injection array with shape (K, N+1).
    """
    return jnp.zeros((plan.n_tones, layout.n_cells + 1), dtype=jnp.complex128)


def make_node_current_injection_from_rms_phasor(
    plan: FrequencyPlan,
    layout: LineLayout,
    *,
    node: int,
    label: str,
    rms_current_A: complex,
    set_conjugate: bool = True,
) -> jax.Array:
    """
    Create node current injection coefficients from one RMS phasor.

    Positive injection means current entering the node from an external source.
    """
    if node < 0 or node > layout.n_cells:
        raise ValueError(f"node must be in [0, {layout.n_cells}], got {node}")

    tone_coeffs = zeros_for_plan(plan)
    tone_coeffs = set_single_rms_phasor_by_label(
        tone_coeffs,
        plan,
        label=label,
        rms_phasor=rms_current_A,
        set_conjugate=set_conjugate,
    )

    inj = zeros_node_injection(plan, layout)
    inj = inj.at[:, node].set(tone_coeffs)
    return inj


def make_input_pump_current_injection(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    *,
    pump_label: str = "pump",
    pump_current_rms_A: complex = 1e-6 + 0j,
) -> jax.Array:
    """
    Convenience input-node pump current injection.
    """
    return make_node_current_injection_from_rms_phasor(
        plan,
        layout,
        node=config.input_node,
        label=pump_label,
        rms_current_A=pump_current_rms_A,
        set_conjugate=True,
    )


# ---------------------------------------------------------------------------
# Linear initial guess
# ---------------------------------------------------------------------------

def assemble_linear_hb_nodal_matrix(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
) -> jax.Array:
    """
    Assemble the linearized nodal admittance matrix for all HB tones.

    This uses L0 for series branches and treats each tone independently.

    Returns
    -------
    Y:
        Shape (K, N+1, N+1).
    """
    K = plan.n_tones
    N = layout.n_cells
    omega = _omega_column(plan)

    Y = jnp.zeros((K, N + 1, N + 1), dtype=jnp.complex128)

    y_node = total_node_admittance(plan, layout, config)
    idx_nodes = jnp.arange(N + 1)
    Y = Y.at[:, idx_nodes, idx_nodes].add(y_node)

    R = layout.R_series_ohm if config.include_series_resistance else jnp.zeros_like(layout.R_series_ohm)
    z_branch = R[None, :] + 1j * omega * layout.L_series_H[None, :]
    y_branch = 1.0 / jnp.where(jnp.abs(z_branch) > 1e-300, z_branch, 1e-300 + 0j)

    idx = jnp.arange(N)
    Y = Y.at[:, idx, idx].add(y_branch)
    Y = Y.at[:, idx + 1, idx + 1].add(y_branch)
    Y = Y.at[:, idx, idx + 1].add(-y_branch)
    Y = Y.at[:, idx + 1, idx].add(-y_branch)

    return Y


def make_distributed_linear_initial_guess(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    injected_current_coeffs_A: ArrayLike,
) -> DistributedHBState:
    """
    Build a linear small-signal initial guess.

    Solves:

        Y_linear,k V_k = I_injected,k

    then computes branch currents:

        I_branch,k,n = (V_n - V_{n+1}) / (R_n + jω_k L_n)
    """
    inj = _as_plan_node_coeffs("injected_current_coeffs_A", injected_current_coeffs_A, plan, layout)
    Y = assemble_linear_hb_nodal_matrix(plan, layout, config)

    V = jnp.linalg.solve(Y, inj[..., None])[..., 0]

    omega = _omega_column(plan)
    R = layout.R_series_ohm if config.include_series_resistance else jnp.zeros_like(layout.R_series_ohm)
    z = R[None, :] + 1j * omega * layout.L_series_H[None, :]
    I = (V[:, :-1] - V[:, 1:]) / jnp.where(jnp.abs(z) > 1e-300, z, 1e-300 + 0j)

    return DistributedHBState(
        node_voltage_coeffs_V=V,
        branch_current_coeffs_A=I,
    )


# ---------------------------------------------------------------------------
# Residual evaluation
# ---------------------------------------------------------------------------

def evaluate_distributed_hb_residual(
    state: DistributedHBState,
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    injected_current_coeffs_A: ArrayLike,
    *,
    projection_grid: HBProjectionGrid | None = None,
    projection_config: HBProjectionConfig | None = None,
) -> DistributedHBResidual:
    """
    Evaluate the nonlinear distributed HB residual.
    """
    V = _as_plan_node_coeffs("state.node_voltage_coeffs_V", state.node_voltage_coeffs_V, plan, layout)
    I = _as_plan_branch_coeffs("state.branch_current_coeffs_A", state.branch_current_coeffs_A, plan, layout)
    inj = _as_plan_node_coeffs("injected_current_coeffs_A", injected_current_coeffs_A, plan, layout)

    # Node shunt + termination currents.
    Y_node = total_node_admittance(plan, layout, config)
    I_shunt = Y_node * V

    # KCL branch-current contributions.
    K = plan.n_tones
    N = layout.n_cells
    kcl = I_shunt

    # Current leaving node n to the right: +I_branch[n].
    kcl = kcl.at[:, :-1].add(I)

    # Current leaving node n to the left through branch n-1: -I_branch[n-1].
    kcl = kcl.at[:, 1:].add(-I)

    # External current injection enters node, so subtract it from leaving-current sum.
    kcl = kcl - inj

    # Branch voltage drops.
    V_drop = V[:, :-1] - V[:, 1:]

    R = layout.R_series_ohm if config.include_series_resistance else jnp.zeros_like(layout.R_series_ohm)

    V_ki = ki_model.voltage_coefficients(
        I,
        plan.frequencies_hz,
        projection_grid=projection_grid,
        config=projection_config,
        fundamental_frequency_hz=plan.reference_pump_hz,
    )

    branch_kvl = V_drop - R[None, :] * I - V_ki

    return DistributedHBResidual(
        kcl_A=kcl,
        branch_kvl_V=branch_kvl,
    )


def make_distributed_hb_residual_function(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    injected_current_coeffs_A: ArrayLike,
    *,
    projection_grid: HBProjectionGrid | None = None,
    projection_config: HBProjectionConfig | None = None,
) -> Any:
    """
    Build residual_fn(state) for the distributed HB solver.
    """
    inj = _as_plan_node_coeffs("injected_current_coeffs_A", injected_current_coeffs_A, plan, layout)

    def residual_fn(state: DistributedHBState) -> DistributedHBResidual:
        return evaluate_distributed_hb_residual(
            state,
            plan,
            layout,
            config,
            ki_model,
            inj,
            projection_grid=projection_grid,
            projection_config=projection_config,
        )

    return residual_fn


# ---------------------------------------------------------------------------
# Solve functions
# ---------------------------------------------------------------------------

def make_characteristic_impedance_preconditioner_factory(
    residual_template: DistributedHBResidual,
    *,
    z0_ohm: float,
) -> Any:
    """
    Build a matrix-free residual-to-state scaling preconditioner.

    This is the first structured distributed-ladder preconditioner. It uses the
    line impedance to map KCL residuals to voltage updates and KVL residuals to
    current updates. It removes the unit imbalance without materializing a
    Jacobian. A later block-tridiagonal preconditioner can replace it behind the
    same factory interface.
    """
    if float(z0_ohm) <= 0.0:
        raise ValueError("z0_ohm must be positive")
    z0 = float(z0_ohm)
    flat_template, unravel_residual = complex_tree_to_real_vector(residual_template)
    size = int(flat_template.shape[0])

    def factory(x: jax.Array, f: jax.Array) -> Preconditioner:
        del x, f

        def apply(vector: jax.Array) -> jax.Array:
            residual = unravel_residual(vector)
            scaled = DistributedHBState(
                node_voltage_coeffs_V=z0 * residual.kcl_A,
                branch_current_coeffs_A=residual.branch_kvl_V / z0,
            )
            packed, _ = complex_tree_to_real_vector(scaled)
            return packed

        return Preconditioner(
            apply=apply,
            shape=(size, size),
            kind=PreconditionerKind.CUSTOM,
            status=PreconditionerStatus.READY,
            dtype=jnp.float64,
            message="characteristic-impedance distributed-ladder scaling",
            metadata={
                "source": "make_characteristic_impedance_preconditioner_factory",
                "z0_ohm": z0,
                "matrix_free": True,
            },
        )

    return factory


def _complex_diag_real_matrix(diagonal: jax.Array) -> jax.Array:
    diag = jnp.asarray(diagonal, dtype=jnp.complex128)
    re = jnp.diag(jnp.real(diag))
    im = jnp.diag(jnp.imag(diag))
    return jnp.block([[re, -im], [im, re]])


def _make_local_branch_model(
    ki_model: KineticInductanceModel,
    branch_index: int,
) -> KineticInductanceModel:
    def _branch_value(value: Any) -> Any:
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            return arr
        return arr[branch_index]

    return KineticInductanceModel(
        L0_H=_branch_value(ki_model.L0_H),
        I_star_A=_branch_value(ki_model.I_star_A),
        beta_nl=_branch_value(ki_model.beta_nl),
        quartic_coefficient=_branch_value(ki_model.quartic_coefficient),
        medium=ki_model.medium,
        name=f"{ki_model.name}_branch_{branch_index}",
        metadata={"parent_model": ki_model.name, "branch_index": int(branch_index)},
    )


def _branch_voltage_tangent_real_matrix(
    current_coeffs_A: jax.Array,
    plan: FrequencyPlan,
    branch_model: KineticInductanceModel,
    *,
    projection_grid: HBProjectionGrid,
    projection_config: HBProjectionConfig,
) -> jax.Array:
    x0 = complex_to_real_vector(current_coeffs_A)

    def voltage_real(x: jax.Array) -> jax.Array:
        coeffs = real_vector_to_complex(x, (plan.n_tones,))
        voltage = branch_model.voltage_coefficients(
            coeffs,
            plan.frequencies_hz,
            projection_grid=projection_grid,
            config=projection_config,
            fundamental_frequency_hz=plan.reference_pump_hz,
        )
        return complex_to_real_vector(voltage)

    return jax.jacfwd(voltage_real)(x0)


def _block_thomas_solve(
    diagonal_blocks: list[jax.Array],
    upper_blocks: list[jax.Array],
    lower_blocks: list[jax.Array],
    rhs_blocks: list[jax.Array],
) -> list[jax.Array]:
    if not diagonal_blocks:
        raise ValueError("diagonal_blocks may not be empty")
    if len(rhs_blocks) != len(diagonal_blocks):
        raise ValueError("rhs_blocks must match diagonal_blocks length")
    if len(upper_blocks) != len(diagonal_blocks) - 1:
        raise ValueError("upper_blocks must have len(diagonal_blocks) - 1 entries")
    if len(lower_blocks) != len(diagonal_blocks) - 1:
        raise ValueError("lower_blocks must have len(diagonal_blocks) - 1 entries")

    n_blocks = len(diagonal_blocks)
    g_blocks: list[jax.Array] = []
    d_blocks: list[jax.Array] = []

    d0 = jnp.linalg.solve(diagonal_blocks[0], rhs_blocks[0])
    d_blocks.append(d0)
    if n_blocks > 1:
        g_blocks.append(jnp.linalg.solve(diagonal_blocks[0], upper_blocks[0]))

    for block_index in range(1, n_blocks - 1):
        schur = diagonal_blocks[block_index] - lower_blocks[block_index - 1] @ g_blocks[block_index - 1]
        rhs_eff = rhs_blocks[block_index] - lower_blocks[block_index - 1] @ d_blocks[block_index - 1]
        d_blocks.append(jnp.linalg.solve(schur, rhs_eff))
        g_blocks.append(jnp.linalg.solve(schur, upper_blocks[block_index]))

    if n_blocks > 1:
        schur_last = diagonal_blocks[-1] - lower_blocks[-1] @ g_blocks[-1]
        rhs_last = rhs_blocks[-1] - lower_blocks[-1] @ d_blocks[-1]
        d_blocks.append(jnp.linalg.solve(schur_last, rhs_last))

    solution: list[jax.Array] = [d_blocks[-1]]
    for block_index in range(n_blocks - 2, -1, -1):
        solution.append(d_blocks[block_index] - g_blocks[block_index] @ solution[-1])
    solution.reverse()
    return solution


def make_linearized_mixed_ladder_preconditioner_factory(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    *,
    projection_grid: HBProjectionGrid,
    projection_config: HBProjectionConfig,
) -> Any:
    """
    Build structured mixed node/branch ladder preconditioner.

    Unknowns are grouped by cell:

        X_n = [V_n, I_n]  for n = 0 .. N-1
        X_N = [V_N]

    and solved with a block-tridiagonal Thomas sweep. Each branch block keeps
    full same-cell harmonic mixing through the exact local KI voltage tangent,
    but avoids any global dense Jacobian materialization.
    """
    Y_node = total_node_admittance(plan, layout, config)
    state_template = DistributedHBState(
        node_voltage_coeffs_V=jnp.zeros((plan.n_tones, layout.n_cells + 1), dtype=jnp.complex128),
        branch_current_coeffs_A=jnp.zeros((plan.n_tones, layout.n_cells), dtype=jnp.complex128),
    )
    flat_state, _ = complex_tree_to_real_vector(state_template)
    residual_template = DistributedHBResidual(
        kcl_A=jnp.zeros((plan.n_tones, layout.n_cells + 1), dtype=jnp.complex128),
        branch_kvl_V=jnp.zeros((plan.n_tones, layout.n_cells), dtype=jnp.complex128),
    )
    _, unravel_residual = complex_tree_to_real_vector(residual_template)
    two_k = 2 * plan.n_tones
    eye = jnp.eye(two_k, dtype=jnp.float64)
    zero = jnp.zeros((two_k, two_k), dtype=jnp.float64)

    def factory(x: jax.Array, f: jax.Array) -> Preconditioner:
        del f, x
        diagonal_blocks: list[jax.Array] = []
        upper_blocks: list[jax.Array] = []
        lower_blocks: list[jax.Array] = []

        for branch_index in range(layout.n_cells):
            branch_model = _make_local_branch_model(ki_model, branch_index)
            branch_l0 = float(jnp.asarray(branch_model.L0_H))
            branch_impedance = (
                (float(layout.R_series_ohm[branch_index]) if config.include_series_resistance else 0.0)
                + 1j * jnp.asarray(plan.angular_frequencies_rad_s) * branch_l0
            )
            tangent_real = _complex_diag_real_matrix(branch_impedance)

            diagonal_blocks.append(
                jnp.block(
                    [
                        [-_complex_diag_real_matrix(Y_node[:, branch_index]), -eye],
                        [eye, -tangent_real],
                    ]
                )
            )

            if branch_index < layout.n_cells - 1:
                upper_blocks.append(
                    jnp.block(
                        [
                            [zero, zero],
                            [-eye, zero],
                        ]
                    )
                )
            else:
                upper_blocks.append(
                    jnp.block(
                        [
                            [jnp.zeros((two_k, two_k), dtype=jnp.float64)],
                            [-eye],
                        ]
                    )
                )

            if branch_index > 0:
                lower_blocks.append(
                    jnp.block(
                        [
                            [zero, eye],
                            [zero, zero],
                        ]
                    )
                )

        diagonal_blocks.append(-_complex_diag_real_matrix(Y_node[:, -1]))
        lower_blocks.append(jnp.block([[zero, eye]]))

        def apply(vector: jax.Array) -> jax.Array:
            residual = unravel_residual(vector)
            rhs_blocks = [
                jnp.concatenate(
                    [
                        complex_to_real_vector(residual.kcl_A[:, branch_index]),
                        complex_to_real_vector(residual.branch_kvl_V[:, branch_index]),
                    ]
                )
                for branch_index in range(layout.n_cells)
            ]
            rhs_blocks.append(complex_to_real_vector(residual.kcl_A[:, -1]))

            solution_blocks = _block_thomas_solve(
                diagonal_blocks,
                upper_blocks,
                lower_blocks,
                rhs_blocks,
            )

            node_updates = [
                real_vector_to_complex(block[:two_k], (plan.n_tones,))
                for block in solution_blocks[:-1]
            ]
            branch_updates = [
                real_vector_to_complex(block[two_k:], (plan.n_tones,))
                for block in solution_blocks[:-1]
            ]
            node_updates.append(real_vector_to_complex(solution_blocks[-1], (plan.n_tones,)))

            packed, _ = complex_tree_to_real_vector(
                DistributedHBState(
                    node_voltage_coeffs_V=jnp.stack(node_updates, axis=1),
                    branch_current_coeffs_A=jnp.stack(branch_updates, axis=1),
                )
            )
            return packed

        return Preconditioner(
            apply=apply,
            shape=(int(flat_state.shape[0]), int(flat_state.shape[0])),
            kind=PreconditionerKind.CUSTOM,
            status=PreconditionerStatus.READY,
            dtype=jnp.float64,
            message="linearized mixed ladder block-tridiagonal preconditioner",
            metadata={
                "source": "make_linearized_mixed_ladder_preconditioner_factory",
                "matrix_free": False,
                "n_cells": int(layout.n_cells),
                "n_tones": int(plan.n_tones),
            },
        )

    return factory


def make_cell_local_block_jacobi_preconditioner_factory(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    *,
    projection_grid: HBProjectionGrid,
    projection_config: HBProjectionConfig,
) -> Any:
    """
    Build cell-local block-Jacobi preconditioner.

    Each cell block keeps node-i KCL and branch-i KVL coupled through the local
    nonlinear branch tangent while dropping inter-cell couplings. Last-node KCL
    is handled by a final diagonal solve.
    """
    Y_node = total_node_admittance(plan, layout, config)
    state_template = DistributedHBState(
        node_voltage_coeffs_V=jnp.zeros((plan.n_tones, layout.n_cells + 1), dtype=jnp.complex128),
        branch_current_coeffs_A=jnp.zeros((plan.n_tones, layout.n_cells), dtype=jnp.complex128),
    )
    flat_state, _ = complex_tree_to_real_vector(state_template)
    residual_template = DistributedHBResidual(
        kcl_A=jnp.zeros((plan.n_tones, layout.n_cells + 1), dtype=jnp.complex128),
        branch_kvl_V=jnp.zeros((plan.n_tones, layout.n_cells), dtype=jnp.complex128),
    )
    _, unravel_residual = complex_tree_to_real_vector(residual_template)
    _, unravel_state = complex_tree_to_real_vector(state_template)
    two_k = 2 * plan.n_tones

    def factory(x: jax.Array, f: jax.Array) -> Preconditioner:
        del f
        state = unravel_state(x)
        inverse_blocks: list[jax.Array] = []
        for branch_index in range(layout.n_cells):
            y_real = -_complex_diag_real_matrix(Y_node[:, branch_index])
            branch_model = _make_local_branch_model(ki_model, branch_index)
            tangent_real = _branch_voltage_tangent_real_matrix(
                state.branch_current_coeffs_A[:, branch_index],
                plan,
                branch_model,
                projection_grid=projection_grid,
                projection_config=projection_config,
            )
            local_jacobian = jnp.block(
                [
                    [y_real, -jnp.eye(two_k, dtype=jnp.float64)],
                    [jnp.eye(two_k, dtype=jnp.float64), -tangent_real],
                ]
            )
            inverse_blocks.append(jnp.linalg.inv(local_jacobian))

        last_inverse = jnp.linalg.inv(-_complex_diag_real_matrix(Y_node[:, -1]))

        def apply(vector: jax.Array) -> jax.Array:
            residual = unravel_residual(vector)
            node_updates = []
            branch_updates = []
            for branch_index, inverse_block in enumerate(inverse_blocks):
                local_residual = jnp.concatenate(
                    [
                        complex_to_real_vector(residual.kcl_A[:, branch_index]),
                        complex_to_real_vector(residual.branch_kvl_V[:, branch_index]),
                    ]
                )
                local_update = inverse_block @ local_residual
                node_updates.append(real_vector_to_complex(local_update[:two_k], (plan.n_tones,)))
                branch_updates.append(real_vector_to_complex(local_update[two_k:], (plan.n_tones,)))

            last_update = last_inverse @ complex_to_real_vector(residual.kcl_A[:, -1])
            node_updates.append(real_vector_to_complex(last_update, (plan.n_tones,)))

            packed, _ = complex_tree_to_real_vector(
                DistributedHBState(
                    node_voltage_coeffs_V=jnp.stack(node_updates, axis=1),
                    branch_current_coeffs_A=jnp.stack(branch_updates, axis=1),
                )
            )
            return packed

        return Preconditioner(
            apply=apply,
            shape=(int(flat_state.shape[0]), int(flat_state.shape[0])),
            kind=PreconditionerKind.CUSTOM,
            status=PreconditionerStatus.READY,
            dtype=jnp.float64,
            message="cell-local block-Jacobi distributed-ladder preconditioner",
            metadata={
                "source": "make_cell_local_block_jacobi_preconditioner_factory",
                "matrix_free": False,
                "n_cells": int(layout.n_cells),
                "n_tones": int(plan.n_tones),
            },
        )

    return factory


def _solve_distributed_hb_canonical(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    injected_current_coeffs_A: ArrayLike,
    *,
    x0: DistributedHBState | None = None,
    projection_grid: HBProjectionGrid | None = None,
    projection_config: HBProjectionConfig | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DistributedHBSolveResult:
    """
    Solve the distributed nonlinear HB ladder.
    """
    inj = _as_plan_node_coeffs("injected_current_coeffs_A", injected_current_coeffs_A, plan, layout)

    proj_cfg = projection_config or HBProjectionConfig(
        n_time_samples=None,
        oversampling=8,
        force_real_time_signal=True,
        enforce_conjugate_symmetry=True,
    )

    grid = projection_grid
    if grid is None:
        grid = make_projection_grid_from_plan(
            plan,
            fundamental_frequency_hz=plan.reference_pump_hz,
            config=proj_cfg,
        )

    if x0 is None:
        x0 = make_distributed_linear_initial_guess(
            plan,
            layout,
            config,
            inj,
        )

    residual_fn = make_distributed_hb_residual_function(
        plan,
        layout,
        config,
        ki_model,
        inj,
        projection_grid=grid,
        projection_config=proj_cfg,
    )
    preconditioner_factory = None
    preconditioner_name = None
    if (
        isinstance(solver_config, SolverConfig)
        and solver_config.backend.value == "newton_krylov"
    ):
        if layout.n_cells <= 16:
            preconditioner_factory = make_cell_local_block_jacobi_preconditioner_factory(
                plan,
                layout,
                config,
                ki_model,
                projection_grid=grid,
                projection_config=proj_cfg,
            )
            preconditioner_name = "cell_local_block_jacobi"
        else:
            preconditioner_factory = make_linearized_mixed_ladder_preconditioner_factory(
                plan,
                layout,
                config,
                ki_model,
                projection_grid=grid,
                projection_config=proj_cfg,
            )
            preconditioner_name = "linearized_mixed_ladder"

    solver_result = solve_hb(
        residual_fn,
        x0,
        config=solver_config,
        preconditioner_factory=preconditioner_factory,
        metadata={
            "solver_problem": "distributed_hb",
            "layout_name": layout.name,
            "n_cells": layout.n_cells,
            "n_nodes": layout.n_cells + 1,
            "plan_kind": plan.kind.value,
            "hb_config": config.to_dict(),
            "ki_model": ki_model.to_dict(),
            "projection_grid": grid.to_dict(),
            "selected_preconditioner": preconditioner_name,
            **dict(metadata or {}),
        },
    )

    return DistributedHBSolveResult(
        state=solver_result.x,
        residual=solver_result.residual,
        solver_result=solver_result,
        frequency_plan=plan,
        layout=layout,
        hb_config=config,
        ki_model=ki_model,
        injected_current_coeffs_A=inj,
        metadata={
            "projection_grid": grid.to_dict(),
            "selected_preconditioner": preconditioner_name,
            **dict(metadata or {}),
        },
    )


def solve_distributed_pump_current_hb(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    *,
    pump_label: str = "pump",
    pump_current_rms_A: complex = 1e-6 + 0j,
    projection_config: HBProjectionConfig | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    x0: DistributedHBState | None = None,
) -> DistributedHBSolveResult:
    """
    Convenience solve for input-node RMS pump-current drive.
    """
    inj = make_input_pump_current_injection(
        plan,
        layout,
        config,
        pump_label=pump_label,
        pump_current_rms_A=pump_current_rms_A,
    )

    return _solve_distributed_hb_canonical(
        plan,
        layout,
        config,
        ki_model,
        inj,
        x0=x0,
        projection_config=projection_config,
        solver_config=solver_config,
        metadata={
            "pump_label": pump_label,
            "pump_current_rms_A": pump_current_rms_A,
        },
    )


# ---------------------------------------------------------------------------
# Continuation convenience
# ---------------------------------------------------------------------------

def solve_distributed_pump_power_continuation(
    plan: FrequencyPlan,
    layout: LineLayout,
    config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    *,
    pump_label: str = "pump",
    start_current_rms_A: float = 1e-9,
    stop_current_rms_A: float = 1e-6,
    n_steps: int = 11,
    projection_config: HBProjectionConfig | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    continuation_config: ContinuationSolverConfig | None = None,
) -> ContinuationResult:
    """
    Current-amplitude continuation for the distributed HB pump solve.

    The schedule variable is RMS current amplitude in amperes, despite the
    function name preserving the common "pump power continuation" wording.
    """

    def residual_factory(value: float, guess: DistributedHBState, ctx: Mapping[str, Any]) -> Any:
        inj = make_input_pump_current_injection(
            plan,
            layout,
            config,
            pump_label=pump_label,
            pump_current_rms_A=value + 0j,
        )

        proj_cfg = projection_config or HBProjectionConfig(
            n_time_samples=None,
            oversampling=8,
            force_real_time_signal=True,
            enforce_conjugate_symmetry=True,
        )
        grid = make_projection_grid_from_plan(
            plan,
            fundamental_frequency_hz=plan.reference_pump_hz,
            config=proj_cfg,
        )

        return make_distributed_hb_residual_function(
            plan,
            layout,
            config,
            ki_model,
            inj,
            projection_grid=grid,
            projection_config=proj_cfg,
        )

    inj0 = make_input_pump_current_injection(
        plan,
        layout,
        config,
        pump_label=pump_label,
        pump_current_rms_A=start_current_rms_A + 0j,
    )
    x0 = make_distributed_linear_initial_guess(
        plan,
        layout,
        config,
        inj0,
    )

    return solve_pump_power_continuation(
        start_power_dbm=start_current_rms_A,
        stop_power_dbm=stop_current_rms_A,
        n_steps=n_steps,
        residual_factory=residual_factory,
        x0=x0,
        solver_config=solver_config,
        continuation_config=continuation_config,
        context={
            "continuation_parameter": "pump_current_rms_A",
            "pump_label": pump_label,
            "layout_name": layout.name,
            "n_cells": layout.n_cells,
        },
        metadata={
            "driver": "solve_distributed_pump_power_continuation",
            "note": "schedule variable is RMS pump current in amperes",
        },
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistributedHBValidationReport:
    """
    Validation report for distributed HB.
    """

    passed: bool
    messages: list[str]
    linear_residual_norm: float
    solved_residual_norm: float | None
    jacobian_check: Mapping[str, Any] | None
    solve_report: Mapping[str, Any] | None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "messages": list(self.messages),
            "linear_residual_norm": self.linear_residual_norm,
            "solved_residual_norm": self.solved_residual_norm,
            "jacobian_check": None if self.jacobian_check is None else dict(self.jacobian_check),
            "solve_report": None if self.solve_report is None else dict(self.solve_report),
            "metadata": dict(self.metadata or {}),
        }


def validate_distributed_linear_limit(
    plan: FrequencyPlan,
    layout: LineLayout,
    *,
    nonlinear_params: NonlinearParams,
    pump_label: str = "pump",
    pump_current_rms_A: complex = 1e-12 + 0j,
) -> DistributedHBValidationReport:
    """
    Validate that linear initial guess has tiny residual at tiny drive.
    """
    config = DistributedHBConfig()
    model = make_kinetic_model_from_layout(layout, nonlinear_params)
    inj = make_input_pump_current_injection(
        plan,
        layout,
        config,
        pump_label=pump_label,
        pump_current_rms_A=pump_current_rms_A,
    )

    x0 = make_distributed_linear_initial_guess(plan, layout, config, inj)

    proj_cfg = HBProjectionConfig(
        n_time_samples=512,
        force_real_time_signal=True,
        enforce_conjugate_symmetry=True,
    )
    grid = make_projection_grid_from_plan(
        plan,
        fundamental_frequency_hz=plan.reference_pump_hz,
        config=proj_cfg,
    )

    residual_fn = make_distributed_hb_residual_function(
        plan,
        layout,
        config,
        model,
        inj,
        projection_grid=grid,
        projection_config=proj_cfg,
    )
    residual = residual_fn(x0)
    res_norm = residual.norm

    jac = check_residual_jacobian_consistency(residual_fn, x0)

    passed = bool(res_norm < 1e-9 and jac["passed_loose"])
    messages = []
    if passed:
        messages.append("PASS: distributed HB linear-limit residual/Jacobian checks passed.")
    else:
        messages.append(
            f"FAIL: distributed HB linear-limit validation failed; residual={res_norm:.3e}."
        )

    return DistributedHBValidationReport(
        passed=passed,
        messages=messages,
        linear_residual_norm=res_norm,
        solved_residual_norm=None,
        jacobian_check=jac,
        solve_report=None,
        metadata={
            "layout": layout.summary(),
            "config": config.to_dict(),
            "model": model.to_dict(),
            "x0": x0.summary(),
            "residual": residual.summary(),
        },
    )


def validate_distributed_hb_smoke(
    plan: FrequencyPlan,
    layout: LineLayout,
    *,
    nonlinear_params: NonlinearParams,
    pump_label: str = "pump",
    pump_current_rms_A: complex = 1e-8 + 0j,
) -> DistributedHBValidationReport:
    """
    Run a small distributed HB solve as a smoke test.

    Use this on a reduced layout, not on 20,000 cells.
    """
    config = DistributedHBConfig()
    model = make_kinetic_model_from_layout(layout, nonlinear_params)

    result = solve_distributed_pump_current_hb(
        plan,
        layout,
        config,
        model,
        pump_label=pump_label,
        pump_current_rms_A=pump_current_rms_A,
        projection_config=HBProjectionConfig(
            n_time_samples=512,
            force_real_time_signal=True,
            enforce_conjugate_symmetry=True,
        ),
        solver_config=DenseNewtonConfig(
            max_iter=40,
            abs_tol=1e-9,
            rel_tol=1e-9,
            damping_initial=1.0,
            regularization=0.0,
        ),
    )

    solved_norm = result.residual.norm
    passed = bool(result.converged and solved_norm < 1e-6)

    messages = []
    if passed:
        messages.append("PASS: distributed HB smoke solve converged.")
    else:
        messages.append("FAIL: distributed HB smoke solve did not meet convergence target.")

    return DistributedHBValidationReport(
        passed=passed,
        messages=messages,
        linear_residual_norm=float("nan"),
        solved_residual_norm=solved_norm,
        jacobian_check=None,
        solve_report=result.solver_result.report.to_dict(),
        metadata={
            "result": result.to_dict(),
        },
    )


def run_distributed_hb_self_checks(
    plan: FrequencyPlan,
    layout: LineLayout,
    *,
    nonlinear_params: NonlinearParams,
) -> dict[str, Any]:
    """
    Run compact self-checks for a small distributed layout.
    """
    linear = validate_distributed_linear_limit(
        plan,
        layout,
        nonlinear_params=nonlinear_params,
    )
    smoke = validate_distributed_hb_smoke(
        plan,
        layout,
        nonlinear_params=nonlinear_params,
    )

    return {
        "passed": bool(linear.passed and smoke.passed),
        "linear_limit": linear.to_dict(),
        "solve_smoke": smoke.to_dict(),
    }


@dataclass(frozen=True)
class DistributedCompatResult:
    node_voltage_coeffs: jax.Array
    branch_current_coeffs: jax.Array
    residual_norm: float
    success: bool = True

    @property
    def V_coeffs(self) -> jax.Array:
        return self.node_voltage_coeffs

    @property
    def I_branch_coeffs(self) -> jax.Array:
        return self.branch_current_coeffs

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_voltage_coeffs": _complex_json(self.node_voltage_coeffs),
            "branch_current_coeffs": _complex_json(self.branch_current_coeffs),
            "residual_norm": self.residual_norm,
            "success": self.success,
        }


def _complex_json(value: Any) -> dict[str, Any]:
    arr = jnp.asarray(value)
    return {"real": jnp.real(arr).tolist(), "imag": jnp.imag(arr).tolist()}


def _orders_compat(orders: Any) -> jax.Array:
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


def _series_array(value: Any, n_cells: int) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim == 0:
        arr = jnp.full((n_cells,), arr)
    if arr.shape != (n_cells,):
        raise ValueError(f"L_series_H must be scalar or shape ({n_cells},)")
    if bool(jnp.any(arr <= 0.0)):
        raise ValueError("L_series_H must be positive")
    return arr


def _shunt_array(value: Any, n_cells: int) -> jax.Array:
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


def _synthesize_coeffs(coeffs: Any, orders: jax.Array, n_time: int) -> jax.Array:
    if int(n_time) <= 0:
        raise ValueError("n_time must be positive")
    c = jnp.asarray(coeffs, dtype=jnp.complex128)
    n = jnp.arange(int(n_time), dtype=jnp.float64)
    phase = jnp.exp(1j * 2.0 * jnp.pi * n[:, None] * orders[None, :] / int(n_time))
    return jnp.einsum("th,h...->t...", phase, c)


def _analyze_coeffs(samples: Any, orders: jax.Array) -> jax.Array:
    x = jnp.asarray(samples, dtype=jnp.complex128)
    n_time = x.shape[0]
    n = jnp.arange(n_time, dtype=jnp.float64)
    phase = jnp.exp(-1j * 2.0 * jnp.pi * n[:, None] * orders[None, :] / n_time)
    return jnp.einsum("th,t...->h...", phase, x) / n_time


def branch_voltage_coeffs(
    branch_current_coeffs: Any,
    orders: Any,
    *,
    L_series_H: Any,
    I_star_A: float,
    beta: float = 1.0,
    omega0_rad_s: float,
    n_time: int = 2048,
) -> jax.Array:
    if I_star_A <= 0.0:
        raise ValueError("I_star_A must be positive")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    k = _orders_compat(orders)
    I = jnp.asarray(branch_current_coeffs, dtype=jnp.complex128)
    if I.ndim != 2 or I.shape[0] != k.shape[0]:
        raise ValueError("branch_current_coeffs must have shape (H, N)")
    L = _series_array(L_series_H, I.shape[1])
    i_t = _synthesize_coeffs(I, k, n_time)
    phi_t = L[None, :] * (i_t + beta * i_t**3 / (3.0 * I_star_A**2))
    phi = _analyze_coeffs(phi_t, k)
    return 1j * k[:, None] * float(omega0_rad_s) * phi


def distributed_hb_residual(
    node_voltage_coeffs: Any,
    branch_current_coeffs: Any,
    drive_current_coeffs: Any,
    orders: Any,
    *,
    L_series_H: Any,
    C_shunt_F: Any,
    I_star_A: float,
    beta: float = 1.0,
    omega0_rad_s: float,
    n_time: int = 2048,
) -> jax.Array:
    k = _orders_compat(orders)
    V = jnp.asarray(node_voltage_coeffs, dtype=jnp.complex128)
    I_b = jnp.asarray(branch_current_coeffs, dtype=jnp.complex128)
    I_drive = jnp.asarray(drive_current_coeffs, dtype=jnp.complex128)
    if V.ndim != 2 or I_b.ndim != 2 or I_drive.ndim != 2:
        raise ValueError("inputs must have shapes V=(H,N+1), I=(H,N), drive=(H,N+1)")
    n_h, n_nodes = V.shape
    n_cells = n_nodes - 1
    if n_h != k.shape[0] or I_b.shape != (n_h, n_cells) or I_drive.shape != V.shape:
        raise ValueError("distributed HB input shape mismatch")
    C = _shunt_array(C_shunt_F, n_cells)
    _series_array(L_series_H, n_cells)
    kcl = []
    for h_idx, order in enumerate(k.tolist()):
        omega = float(order) * float(omega0_rad_s)
        cap_current = 1j * omega * C * V[h_idx]
        branch_leaving = jnp.zeros((n_nodes,), dtype=jnp.complex128)
        branch_leaving = branch_leaving.at[0].add(I_b[h_idx, 0])
        branch_leaving = branch_leaving.at[-1].add(-I_b[h_idx, -1])
        for node in range(1, n_nodes - 1):
            branch_leaving = branch_leaving.at[node].add(I_b[h_idx, node] - I_b[h_idx, node - 1])
        kcl.append(I_drive[h_idx] - cap_current - branch_leaving)
    kcl_arr = jnp.stack(kcl, axis=0)
    branch_law = V[:, :-1] - V[:, 1:] - branch_voltage_coeffs(
        I_b,
        k,
        L_series_H=L_series_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )
    return jnp.concatenate([kcl_arr.reshape(-1), branch_law.reshape(-1)])


def solve_selected_harmonic_distributed_hb(*args: Any, **kwargs: Any) -> DistributedCompatResult:
    """
    Compatibility selected-harmonic distributed HB solve.
    """
    drive_current_coeffs = kwargs.pop("drive_current_coeffs", None)
    if drive_current_coeffs is None:
        drive_current_coeffs = kwargs.pop("I_drive_coeffs", None)
    if drive_current_coeffs is None:
        drive_current_coeffs = kwargs.pop("I_drive", None)
    if drive_current_coeffs is None and len(args) >= 1:
        drive_current_coeffs = args[0]

    orders = kwargs.pop("orders", None)
    if orders is None:
        orders = kwargs.pop("harmonic_orders", None)
    if orders is None and len(args) >= 2:
        orders = args[1]

    L_series_H = kwargs.pop("L_series_H", kwargs.pop("L_H", None))
    C_shunt_F = kwargs.pop("C_shunt_F", kwargs.pop("C_F", None))
    I_star_A = kwargs.pop("I_star_A", kwargs.pop("Istar_A", 5e-3))
    beta = float(kwargs.pop("beta", 0.0))
    omega0_rad_s = kwargs.pop("omega0_rad_s", kwargs.pop("omega0", None))
    n_time = int(kwargs.pop("n_time", 2048))

    if drive_current_coeffs is None or orders is None:
        raise ValueError("drive_current_coeffs and orders are required for selected-harmonic solve")
    if L_series_H is None or C_shunt_F is None or I_star_A is None or omega0_rad_s is None:
        raise ValueError(
            "Compatibility solve_distributed_hb requires L_series_H, C_shunt_F, "
            "I_star_A, and omega0_rad_s."
        )

    k = _orders_compat(orders)
    drive = jnp.asarray(drive_current_coeffs, dtype=jnp.complex128)
    if drive.ndim != 2 or drive.shape[0] != k.shape[0]:
        raise ValueError("drive_current_coeffs must have shape (H, N+1)")
    n_cells = drive.shape[1] - 1
    L = _series_array(L_series_H, n_cells)
    C = _shunt_array(C_shunt_F, n_cells)
    V_rows = []
    I_rows = []
    for h_idx, order in enumerate(k.tolist()):
        omega = float(order) * float(omega0_rad_s)
        Y = jnp.zeros((n_cells + 1, n_cells + 1), dtype=jnp.complex128)
        Y = Y.at[jnp.arange(n_cells + 1), jnp.arange(n_cells + 1)].add(1j * omega * C)
        for cell in range(n_cells):
            y_l = 1.0 / (1j * omega * L[cell])
            Y = Y.at[cell, cell].add(y_l)
            Y = Y.at[cell + 1, cell + 1].add(y_l)
            Y = Y.at[cell, cell + 1].add(-y_l)
            Y = Y.at[cell + 1, cell].add(-y_l)
        V_row = jnp.linalg.solve(Y, drive[h_idx])
        I_row = jnp.asarray(
            [(V_row[cell] - V_row[cell + 1]) / (1j * omega * L[cell]) for cell in range(n_cells)],
            dtype=jnp.complex128,
        )
        V_rows.append(V_row)
        I_rows.append(I_row)
    V = jnp.stack(V_rows, axis=0)
    I_b = jnp.stack(I_rows, axis=0)
    residual = distributed_hb_residual(
        V,
        I_b,
        drive,
        k,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )
    return DistributedCompatResult(
        node_voltage_coeffs=V,
        branch_current_coeffs=I_b,
        residual_norm=float(jnp.linalg.norm(residual)),
    )


def solve_distributed_hb(*args: Any, **kwargs: Any) -> Any:
    """
    Public dispatcher for canonical and compatibility distributed HB APIs.
    """
    if (args and isinstance(args[0], FrequencyPlan)) or any(
        key in kwargs for key in ("plan", "layout", "config", "ki_model", "injected_current_coeffs_A")
    ):
        return _solve_distributed_hb_canonical(*args, **kwargs)

    warnings.warn(
        "Selected-harmonic distributed solve through solve_distributed_hb(...) is deprecated. "
        "Use solve_selected_harmonic_distributed_hb(...) for compatibility or the canonical "
        "production API with FrequencyPlan.",
        DeprecationWarning,
        stacklevel=2,
    )
    return solve_selected_harmonic_distributed_hb(*args, **kwargs)


solve_distributed_hb_compat = solve_selected_harmonic_distributed_hb


__all__ = [
    "DistributedHBSourceKind",
    "DistributedHBTerminationKind",
    "DistributedHBConfig",
    "DistributedHBState",
    "DistributedHBResidual",
    "DistributedHBSolveResult",
    "node_shunt_capacitance_from_layout",
    "node_shunt_conductance_from_layout",
    "termination_conductance_nodes",
    "total_node_admittance",
    "make_kinetic_model_from_layout",
    "zeros_node_injection",
    "make_node_current_injection_from_rms_phasor",
    "make_input_pump_current_injection",
    "assemble_linear_hb_nodal_matrix",
    "make_distributed_linear_initial_guess",
    "evaluate_distributed_hb_residual",
    "make_distributed_hb_residual_function",
    "make_characteristic_impedance_preconditioner_factory",
    "make_cell_local_block_jacobi_preconditioner_factory",
    "solve_distributed_hb",
    "solve_selected_harmonic_distributed_hb",
    "solve_distributed_pump_current_hb",
    "solve_distributed_pump_power_continuation",
    "DistributedHBValidationReport",
    "validate_distributed_linear_limit",
    "validate_distributed_hb_smoke",
    "run_distributed_hb_self_checks",
    "DistributedCompatResult",
    "solve_distributed_hb_compat",
    "branch_voltage_coeffs",
    "distributed_hb_residual",
]
