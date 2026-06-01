"""
twpa.nonlinear.linearization
============================

Small-signal linearization around pumped nonlinear HB states.

This module is the bridge between:

    pump-only large-signal HB
        -> linearized signal/idler equations
        -> gain, conversion, and operating maps

The main object is a JAX linearization of the distributed HB residual:

    R(x; drive) = 0

around a pumped operating point x0. The linearized equation is:

    J(x0) dx + r_source = 0

where dx contains small perturbations in node voltages and branch currents.

Important scope
---------------
This file provides a dense/reference linearization. It is correct-first, not the
final industrial 20,000-cell backend.

The same conceptual API can later be backed by:
    - structured block-banded Jacobians,
    - matrix-free JVPs,
    - Krylov solvers,
    - frequency-domain conversion matrices.

Frequency-plan strategy
-----------------------
The pumped solution may have a pump-only plan, while the small-signal problem
usually needs a richer plan containing pump, signal, idler, and mixing tones.

This module supports that by embedding a solved pump state into a target plan:

    pump plan:      [-3fp, -fp, +fp, +3fp]
    target plan:    pump tones + signal/idler sidebands

The embedded target state is zero at tones not present in the pump solve. JAX
then linearizes the nonlinear projection on the target plan. This captures
frequency mixing induced by the pump coefficients present in the operating
point.

For production, the target plan should be chosen deliberately to include the
sidebands needed by the intended gain model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Literal, Mapping

import jax
import jax.numpy as jnp

from twpa.core.frequency_plan import FrequencyPlan
from twpa.core.harmonics import (
    coefficient_power_summary,
    complex_to_real_vector,
    real_vector_to_complex,
    set_single_rms_phasor_by_label,
    zeros_for_plan,
)
from twpa.core.hb_fft import HBProjectionConfig, HBProjectionGrid, make_projection_grid_from_plan
from twpa.core.layout import LineLayout
from twpa.nonlinear.distributed_hb import (
    DistributedHBConfig,
    DistributedHBResidual,
    DistributedHBSolveResult,
    DistributedHBState,
    evaluate_distributed_hb_residual,
    make_characteristic_impedance_preconditioner_factory,
    make_distributed_hb_residual_function,
    zeros_node_injection,
)
from twpa.nonlinear.kinetic_inductance import KineticInductanceModel
from twpa.solvers.hb_solver import (
    DenseLinearSolveResult,
    LinearSolveMethod,
    dense_linear_solve,
    pack_residual_tree,
    pack_unknown_tree,
)
from twpa.solvers.linear_solvers import (
    IterativeLinearSolveConfig,
    IterativeLinearSolveResult,
    LinearOperator,
    LinearSolverMethod as IterativeLinearSolverMethod,
    solve_linear_system,
)


ArrayLike = Any
PyTree = Any


# ---------------------------------------------------------------------------
# Enums / configuration
# ---------------------------------------------------------------------------

class LinearizationBackend(str, Enum):
    """Supported reference linearization backends."""

    JAX_LINEARIZE = "jax_linearize"
    DENSE_JACOBIAN = "dense_jacobian"


class SmallSignalSourceKind(str, Enum):
    """Supported small-signal source types."""

    CURRENT_INJECTION = "current_injection"


@dataclass(frozen=True)
class SmallSignalLinearizationConfig:
    """
    Configuration for small-signal linearization.

    Parameters
    ----------
    backend:
        Linearization backend.
    projection:
        HB projection configuration for the target frequency plan.
    frequency_match_atol_hz:
        Absolute tolerance for embedding pump tones into a target plan.
    frequency_match_rtol:
        Relative tolerance for embedding pump tones.
    linear_solve_method:
        Dense solve method for reference small-signal solves.
    regularization:
        Optional Tikhonov/diagonal regularization for dense solves.
    """

    backend: LinearizationBackend = LinearizationBackend.JAX_LINEARIZE
    projection: HBProjectionConfig = HBProjectionConfig(
        n_time_samples=None,
        oversampling=8,
        force_real_time_signal=True,
        enforce_conjugate_symmetry=True,
    )
    frequency_match_atol_hz: float = 1e-3
    frequency_match_rtol: float = 1e-12
    linear_solve_method: LinearSolveMethod = LinearSolveMethod.AUTO
    regularization: float = 0.0
    max_dense_real_unknowns: int = 1024
    iterative_max_iter: int = 500
    iterative_atol: float = 1e-12
    iterative_rtol: float = 1e-8
    iterative_restart: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", LinearizationBackend(self.backend))
        object.__setattr__(self, "linear_solve_method", LinearSolveMethod(self.linear_solve_method))
        if self.frequency_match_atol_hz < 0.0:
            raise ValueError("frequency_match_atol_hz must be non-negative")
        if self.frequency_match_rtol < 0.0:
            raise ValueError("frequency_match_rtol must be non-negative")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if int(self.max_dense_real_unknowns) <= 0:
            raise ValueError("max_dense_real_unknowns must be positive")
        object.__setattr__(self, "max_dense_real_unknowns", int(self.max_dense_real_unknowns))
        if int(self.iterative_max_iter) <= 0:
            raise ValueError("iterative_max_iter must be positive")
        object.__setattr__(self, "iterative_max_iter", int(self.iterative_max_iter))
        if self.iterative_atol < 0.0:
            raise ValueError("iterative_atol must be non-negative")
        if self.iterative_rtol < 0.0:
            raise ValueError("iterative_rtol must be non-negative")
        if int(self.iterative_restart) <= 0:
            raise ValueError("iterative_restart must be positive")
        object.__setattr__(self, "iterative_restart", int(self.iterative_restart))

    def with_updates(self, **kwargs: Any) -> "SmallSignalLinearizationConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "projection": self.projection.to_dict(),
            "frequency_match_atol_hz": self.frequency_match_atol_hz,
            "frequency_match_rtol": self.frequency_match_rtol,
            "linear_solve_method": self.linear_solve_method.value,
            "regularization": self.regularization,
            "max_dense_real_unknowns": self.max_dense_real_unknowns,
            "iterative_max_iter": self.iterative_max_iter,
            "iterative_atol": self.iterative_atol,
            "iterative_rtol": self.iterative_rtol,
            "iterative_restart": self.iterative_restart,
        }


# ---------------------------------------------------------------------------
# Small-signal state/source/result objects
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SmallSignalState:
    """
    Small-signal perturbation state.

    Attributes
    ----------
    node_voltage_coeffs_V:
        Perturbation node-voltage coefficients, shape (K, N+1).
    branch_current_coeffs_A:
        Perturbation branch-current coefficients, shape (K, N).
    """

    node_voltage_coeffs_V: jax.Array
    branch_current_coeffs_A: jax.Array

    def __post_init__(self) -> None:
        v = _as_complex_2d("node_voltage_coeffs_V", self.node_voltage_coeffs_V)
        i = _as_complex_2d("branch_current_coeffs_A", self.branch_current_coeffs_A)
        if v.shape[0] != i.shape[0]:
            raise ValueError("voltage/current tone axes must match")
        if v.shape[1] != i.shape[1] + 1:
            raise ValueError("node count must equal branch count + 1")
        object.__setattr__(self, "node_voltage_coeffs_V", v)
        object.__setattr__(self, "branch_current_coeffs_A", i)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        return (self.node_voltage_coeffs_V, self.branch_current_coeffs_A), {}

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array],
    ) -> "SmallSignalState":
        return cls(
            node_voltage_coeffs_V=children[0],
            branch_current_coeffs_A=children[1],
        )

    @classmethod
    def zeros(cls, *, plan: FrequencyPlan, layout: LineLayout) -> "SmallSignalState":
        return cls(
            node_voltage_coeffs_V=jnp.zeros(
                (plan.n_tones, layout.n_cells + 1),
                dtype=jnp.complex128,
            ),
            branch_current_coeffs_A=jnp.zeros(
                (plan.n_tones, layout.n_cells),
                dtype=jnp.complex128,
            ),
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

    def as_distributed_state(self) -> DistributedHBState:
        return DistributedHBState(
            node_voltage_coeffs_V=self.node_voltage_coeffs_V,
            branch_current_coeffs_A=self.branch_current_coeffs_A,
        )

    @classmethod
    def from_distributed_state(cls, state: DistributedHBState) -> "SmallSignalState":
        return cls(
            node_voltage_coeffs_V=state.node_voltage_coeffs_V,
            branch_current_coeffs_A=state.branch_current_coeffs_A,
        )

    def with_updates(self, **kwargs: Any) -> "SmallSignalState":
        return replace(self, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "n_tones": self.n_tones,
            "n_nodes": self.n_nodes,
            "n_branches": self.n_branches,
            "node_voltage_coeffs_V": coefficient_power_summary(self.node_voltage_coeffs_V),
            "branch_current_coeffs_A": coefficient_power_summary(self.branch_current_coeffs_A),
        }


@dataclass(frozen=True)
class SmallSignalSource:
    """
    Small-signal current-injection source.

    Attributes
    ----------
    injected_current_coeffs_A:
        Node current injection, shape (K, N+1). Positive current enters node.
    source_kind:
        Source type.
    metadata:
        Static metadata.
    """

    injected_current_coeffs_A: jax.Array
    source_kind: SmallSignalSourceKind = SmallSignalSourceKind.CURRENT_INJECTION
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        inj = _as_complex_2d("injected_current_coeffs_A", self.injected_current_coeffs_A)
        object.__setattr__(self, "injected_current_coeffs_A", inj)
        object.__setattr__(self, "source_kind", SmallSignalSourceKind(self.source_kind))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def zeros(cls, *, plan: FrequencyPlan, layout: LineLayout) -> "SmallSignalSource":
        return cls(
            injected_current_coeffs_A=zeros_node_injection(plan, layout),
            metadata={"source": "zeros"},
        )

    @classmethod
    def current_phasor_at_node(
        cls,
        *,
        plan: FrequencyPlan,
        layout: LineLayout,
        node: int,
        label: str,
        rms_current_A: complex,
        set_conjugate: bool = True,
    ) -> "SmallSignalSource":
        if node < 0 or node > layout.n_cells:
            raise ValueError(f"node must be in [0, {layout.n_cells}], got {node}")

        coeffs = zeros_for_plan(plan)
        coeffs = set_single_rms_phasor_by_label(
            coeffs,
            plan,
            label=label,
            rms_phasor=rms_current_A,
            set_conjugate=set_conjugate,
        )

        inj = zeros_node_injection(plan, layout)
        inj = inj.at[:, node].set(coeffs)

        return cls(
            injected_current_coeffs_A=inj,
            metadata={
                "source": "current_phasor_at_node",
                "node": node,
                "label": label,
                "rms_current_A": rms_current_A,
                "set_conjugate": set_conjugate,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind.value,
            "injected_current_coeffs_A": coefficient_power_summary(self.injected_current_coeffs_A),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SmallSignalSolveResult:
    """
    Result of solving the linearized small-signal problem.

    The solved perturbation satisfies approximately:

        J dx = -R_source

    where R_source is the residual contribution of the small injected current.
    """

    perturbation: SmallSignalState
    residual_perturbation: DistributedHBResidual
    source: SmallSignalSource
    linear_solve: DenseLinearSolveResult | IterativeLinearSolveResult
    linearization: "DistributedHBLinearization"
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return bool(self.linear_solve.success)

    @property
    def node_voltage_coeffs_V(self) -> jax.Array:
        return self.perturbation.node_voltage_coeffs_V

    @property
    def branch_current_coeffs_A(self) -> jax.Array:
        return self.perturbation.branch_current_coeffs_A

    def voltage_gain_by_label(
        self,
        *,
        output_label: str,
        input_label: str | None = None,
        input_node: int = 0,
        output_node: int | None = None,
    ) -> complex:
        """
        Return small-signal voltage ratio Vout(output_label) / Vin(input_label).

        If input_label is None, the same label is used for input and output.
        """
        plan = self.linearization.plan
        out_node = self.linearization.layout.n_cells if output_node is None else output_node
        out_idx = plan.position_of_label(output_label)
        in_idx = out_idx if input_label is None else plan.position_of_label(input_label)

        vin = self.node_voltage_coeffs_V[in_idx, input_node]
        vout = self.node_voltage_coeffs_V[out_idx, out_node]
        return complex(vout / jnp.where(jnp.abs(vin) > 1e-300, vin, 1e-300 + 0j))

    def voltage_gain_db_by_label(
        self,
        *,
        output_label: str,
        input_label: str | None = None,
        input_node: int = 0,
        output_node: int | None = None,
    ) -> float:
        gain = self.voltage_gain_by_label(
            output_label=output_label,
            input_label=input_label,
            input_node=input_node,
            output_node=output_node,
        )
        return float(20.0 * jnp.log10(jnp.maximum(abs(gain), 1e-300)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "perturbation": self.perturbation.summary(),
            "residual_perturbation": self.residual_perturbation.summary(),
            "source": self.source.to_dict(),
            "linear_solve": self.linear_solve.to_dict(),
            "linearization": self.linearization.to_dict(include_matrix=False),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Pump-state embedding
# ---------------------------------------------------------------------------

def embed_coefficients_between_plans(
    source_coeffs: ArrayLike,
    source_plan: FrequencyPlan,
    target_plan: FrequencyPlan,
    *,
    atol_hz: float = 1e-3,
    rtol: float = 1e-12,
    target_trailing_shape: tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Embed coefficients from source_plan into target_plan by frequency matching.

    Parameters
    ----------
    source_coeffs:
        Shape (K_source, ...).
    source_plan:
        Plan corresponding to source_coeffs.
    target_plan:
        Target plan.
    atol_hz, rtol:
        Frequency matching tolerances.
    target_trailing_shape:
        Optional trailing shape for the target coefficients. If None, uses
        source_coeffs trailing shape.

    Returns
    -------
    target_coeffs:
        Shape (K_target, ...), zero where no source frequency is matched.
    """
    src = jnp.asarray(source_coeffs)
    if src.shape[0] != source_plan.n_tones:
        raise ValueError("source_coeffs first dimension does not match source_plan")

    trailing = src.shape[1:] if target_trailing_shape is None else target_trailing_shape
    out = jnp.zeros((target_plan.n_tones, *trailing), dtype=src.dtype)

    for i_t, f_t in enumerate(target_plan.frequencies_hz.tolist()):
        matches = []
        for i_s, f_s in enumerate(source_plan.frequencies_hz.tolist()):
            tol = atol_hz + rtol * max(abs(float(f_t)), abs(float(f_s)))
            if abs(float(f_t) - float(f_s)) <= tol:
                matches.append(i_s)

        if matches:
            out = out.at[i_t].set(src[matches[0]])

    return out


def embed_distributed_state_between_plans(
    source_state: DistributedHBState,
    source_plan: FrequencyPlan,
    target_plan: FrequencyPlan,
    *,
    atol_hz: float = 1e-3,
    rtol: float = 1e-12,
) -> DistributedHBState:
    """
    Embed a distributed HB state from source_plan into target_plan.
    """
    v = embed_coefficients_between_plans(
        source_state.node_voltage_coeffs_V,
        source_plan,
        target_plan,
        atol_hz=atol_hz,
        rtol=rtol,
    )
    i = embed_coefficients_between_plans(
        source_state.branch_current_coeffs_A,
        source_plan,
        target_plan,
        atol_hz=atol_hz,
        rtol=rtol,
    )
    return DistributedHBState(
        node_voltage_coeffs_V=v,
        branch_current_coeffs_A=i,
    )


def embed_pump_solution_in_target_plan(
    pump_result: DistributedHBSolveResult,
    target_plan: FrequencyPlan,
    *,
    config: SmallSignalLinearizationConfig | None = None,
) -> DistributedHBState:
    """
    Embed a solved pump state into a richer target frequency plan.
    """
    cfg = config or SmallSignalLinearizationConfig()
    return embed_distributed_state_between_plans(
        pump_result.state,
        pump_result.frequency_plan,
        target_plan,
        atol_hz=cfg.frequency_match_atol_hz,
        rtol=cfg.frequency_match_rtol,
    )


# ---------------------------------------------------------------------------
# Linearization object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistributedHBLinearization:
    """
    Linearization of distributed HB residual around an operating point.

    Attributes
    ----------
    operating_state:
        Pumped operating state in the target plan.
    operating_residual:
        Residual at operating_state.
    plan:
        Target frequency plan.
    layout:
        Layout.
    hb_config:
        Distributed HB configuration.
    ki_model:
        Branch-wise KI model.
    operating_injection_A:
        Injection used for the operating state, shape (K, N+1).
    projection_grid:
        Projection grid for the target plan.
    config:
        Linearization config.
    linear_map:
        Callable mapping DistributedHBState perturbation -> residual
        perturbation.
    dense_jacobian:
        Optional dense real Jacobian matrix.
    """

    operating_state: DistributedHBState
    operating_residual: DistributedHBResidual
    plan: FrequencyPlan
    layout: LineLayout
    hb_config: DistributedHBConfig
    ki_model: KineticInductanceModel
    operating_injection_A: jax.Array
    projection_grid: HBProjectionGrid
    config: SmallSignalLinearizationConfig
    linear_map: Callable[[DistributedHBState], DistributedHBResidual]
    dense_jacobian: jax.Array | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def unknown_size(self) -> int:
        x_vec, _ = pack_unknown_tree(self.operating_state)
        return int(x_vec.shape[0])

    @property
    def residual_size(self) -> int:
        r_vec = pack_residual_tree(self.operating_residual)
        return int(r_vec.shape[0])

    def apply(self, perturbation: DistributedHBState | SmallSignalState) -> DistributedHBResidual:
        """
        Apply J dx.
        """
        if isinstance(perturbation, SmallSignalState):
            perturbation = perturbation.as_distributed_state()
        return self.linear_map(perturbation)

    def dense_matrix(self) -> jax.Array:
        """
        Return dense real Jacobian matrix, building it if absent.

        If dense_jacobian was not created at construction, this rebuilds it
        eagerly from the stored residual function.
        """
        if self.dense_jacobian is not None:
            return self.dense_jacobian
        return build_dense_real_jacobian_from_linearization(self)

    def to_dict(self, *, include_matrix: bool = False) -> dict[str, Any]:
        out = {
            "plan": self.plan.to_dict(),
            "layout": self.layout.summary(),
            "hb_config": self.hb_config.to_dict(),
            "ki_model": self.ki_model.to_dict(),
            "operating_state": self.operating_state.summary(),
            "operating_residual": self.operating_residual.summary(),
            "operating_injection_A": coefficient_power_summary(self.operating_injection_A),
            "projection_grid": self.projection_grid.to_dict(),
            "config": self.config.to_dict(),
            "unknown_size": self.unknown_size,
            "residual_size": self.residual_size,
            "metadata": dict(self.metadata or {}),
        }
        if include_matrix:
            mat = self.dense_matrix()
            out["dense_jacobian"] = {
                "shape": tuple(int(v) for v in mat.shape),
                "max_abs": float(jnp.max(jnp.abs(mat))),
                "fro_norm": float(jnp.linalg.norm(mat)),
            }
        return out


def build_distributed_hb_linearization(
    operating_state: DistributedHBState,
    plan: FrequencyPlan,
    layout: LineLayout,
    hb_config: DistributedHBConfig,
    ki_model: KineticInductanceModel,
    *,
    operating_injection_A: ArrayLike | None = None,
    projection_grid: HBProjectionGrid | None = None,
    config: SmallSignalLinearizationConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DistributedHBLinearization:
    """
    Build a linearization of the distributed HB residual around operating_state.

    The default operating injection is zero. For an embedded pump solve, pass the
    embedded pump injection used to obtain the operating state.
    """
    cfg = config or SmallSignalLinearizationConfig()
    n_real_unknowns = 2 * plan.n_tones * (2 * layout.n_cells + 1)
    if (
        cfg.backend == LinearizationBackend.DENSE_JACOBIAN
        and n_real_unknowns > cfg.max_dense_real_unknowns
    ):
        jacobian_gib = (n_real_unknowns**2 * 8) / (1024**3)
        raise RuntimeError(
            "Dense reference gain linearization refused unsafe problem size: "
            f"{n_real_unknowns} real unknowns exceed max_dense_real_unknowns="
            f"{cfg.max_dense_real_unknowns}. One float64 Jacobian would require "
            f"about {jacobian_gib:.3f} GiB before solver copies. "
            "Reduce the target plan/layout or implement a structured matrix-free backend."
        )

    if operating_injection_A is None:
        operating_injection = zeros_node_injection(plan, layout)
    else:
        operating_injection = _as_plan_node_coeffs(
            "operating_injection_A",
            operating_injection_A,
            plan,
            layout,
        )

    grid = projection_grid
    if grid is None:
        grid = make_projection_grid_from_plan(
            plan,
            fundamental_frequency_hz=plan.reference_pump_hz,
            config=cfg.projection,
        )

    residual_fn = make_distributed_hb_residual_function(
        plan,
        layout,
        hb_config,
        ki_model,
        operating_injection,
        projection_grid=grid,
        projection_config=cfg.projection,
    )

    operating_residual, linear_map = jax.linearize(residual_fn, operating_state)

    dense_jacobian = None
    if cfg.backend == LinearizationBackend.DENSE_JACOBIAN:
        dense_jacobian = build_dense_real_jacobian(
            residual_fn,
            operating_state,
        )

    return DistributedHBLinearization(
        operating_state=operating_state,
        operating_residual=operating_residual,
        plan=plan,
        layout=layout,
        hb_config=hb_config,
        ki_model=ki_model,
        operating_injection_A=operating_injection,
        projection_grid=grid,
        config=cfg,
        linear_map=linear_map,
        dense_jacobian=dense_jacobian,
        metadata={
            "source": "build_distributed_hb_linearization",
            **dict(metadata or {}),
        },
    )


def build_linearization_from_pump_result(
    pump_result: DistributedHBSolveResult,
    *,
    target_plan: FrequencyPlan | None = None,
    operating_injection_A: ArrayLike | None = None,
    config: SmallSignalLinearizationConfig | None = None,
) -> DistributedHBLinearization:
    """
    Build a small-signal linearization from an existing distributed pump result.

    If target_plan is None, linearizes on the pump result's own frequency plan.
    If target_plan is provided, embeds the pump state and injection into it.
    """
    cfg = config or SmallSignalLinearizationConfig()
    plan = pump_result.frequency_plan if target_plan is None else target_plan

    if target_plan is None:
        operating_state = pump_result.state
        injection = (
            pump_result.injected_current_coeffs_A
            if operating_injection_A is None
            else operating_injection_A
        )
    else:
        operating_state = embed_pump_solution_in_target_plan(
            pump_result,
            target_plan,
            config=cfg,
        )
        if operating_injection_A is None:
            injection = embed_coefficients_between_plans(
                pump_result.injected_current_coeffs_A,
                pump_result.frequency_plan,
                target_plan,
                atol_hz=cfg.frequency_match_atol_hz,
                rtol=cfg.frequency_match_rtol,
            )
        else:
            injection = operating_injection_A

    return build_distributed_hb_linearization(
        operating_state,
        plan,
        pump_result.layout,
        pump_result.hb_config,
        pump_result.ki_model,
        operating_injection_A=injection,
        config=cfg,
        metadata={
            "source": "build_linearization_from_pump_result",
            "pump_result_converged": pump_result.converged,
        },
    )


# ---------------------------------------------------------------------------
# Dense real Jacobian and solve helpers
# ---------------------------------------------------------------------------

def build_dense_real_jacobian(
    residual_fn: Callable[[DistributedHBState], DistributedHBResidual],
    x0: DistributedHBState,
) -> jax.Array:
    """
    Build dense real Jacobian dR/dx at x0.

    Uses the same real packing convention as the dense HB solver.
    """
    x_vec, unravel = pack_unknown_tree(x0)

    def real_fn(x_vec_local: jax.Array) -> jax.Array:
        x_state = unravel(x_vec_local)
        return pack_residual_tree(residual_fn(x_state))

    return jax.jacfwd(real_fn)(x_vec)


def build_dense_real_jacobian_from_linearization(
    lin: DistributedHBLinearization,
) -> jax.Array:
    """
    Build dense real Jacobian by applying the stored linear map to basis vectors.
    """
    x_vec, unravel = pack_unknown_tree(lin.operating_state)
    n = int(x_vec.shape[0])

    def apply_column(e: jax.Array) -> jax.Array:
        dx = unravel(e)
        dr = lin.apply(dx)
        return pack_residual_tree(dr)

    eye = jnp.eye(n, dtype=jnp.float64)
    columns = jax.vmap(apply_column)(eye)
    return columns.T


def source_to_residual_perturbation(
    source: SmallSignalSource,
    *,
    plan: FrequencyPlan,
    layout: LineLayout,
) -> DistributedHBResidual:
    """
    Convert a current-injection source into a residual perturbation.

    Distributed KCL residual is:

        r_kcl = currents_leaving - injected_current

    Therefore a small source injection contributes:

        dr_source.kcl = -I_injected
        dr_source.branch = 0
    """
    inj = _as_plan_node_coeffs("source.injected_current_coeffs_A", source.injected_current_coeffs_A, plan, layout)
    return DistributedHBResidual(
        kcl_A=-inj,
        branch_kvl_V=jnp.zeros(
            (plan.n_tones, layout.n_cells),
            dtype=jnp.complex128,
        ),
    )


def solve_linearized_small_signal(
    linearization: DistributedHBLinearization,
    source: SmallSignalSource,
    *,
    method: LinearSolveMethod | None = None,
    regularization: float | None = None,
) -> SmallSignalSolveResult:
    """
    Solve the dense linearized small-signal problem.

    Equation:
        J dx + r_source = 0

    so:
        J dx = -r_source
    """
    n_real_unknowns = 2 * linearization.plan.n_tones * (
        2 * linearization.layout.n_cells + 1
    )
    if (
        linearization.config.backend == LinearizationBackend.DENSE_JACOBIAN
        and n_real_unknowns > linearization.config.max_dense_real_unknowns
    ):
        jacobian_gib = (n_real_unknowns**2 * 8) / (1024**3)
        raise RuntimeError(
            "Dense reference gain linearization refused unsafe problem size: "
            f"{n_real_unknowns} real unknowns exceed max_dense_real_unknowns="
            f"{linearization.config.max_dense_real_unknowns}. One float64 Jacobian "
            f"would require about {jacobian_gib:.3f} GiB before solver copies. "
            "Reduce the target plan/layout or implement a structured matrix-free backend."
        )

    src_res = source_to_residual_perturbation(
        source,
        plan=linearization.plan,
        layout=linearization.layout,
    )

    rhs = -pack_residual_tree(src_res)

    if linearization.config.backend == LinearizationBackend.DENSE_JACOBIAN:
        J = linearization.dense_matrix()
        solve = dense_linear_solve(
            J,
            rhs,
            method=linearization.config.linear_solve_method if method is None else method,
            regularization=linearization.config.regularization if regularization is None else regularization,
        )
    else:
        _, unravel = pack_unknown_tree(linearization.operating_state)

        def matvec(vector: jax.Array) -> jax.Array:
            return pack_residual_tree(linearization.apply(unravel(vector)))

        operator = LinearOperator(
            shape=(n_real_unknowns, n_real_unknowns),
            matvec=matvec,
            dtype=jnp.float64,
            name="small_signal_jax_linearize",
            metadata={"matrix_free": True},
        )
        preconditioner = make_characteristic_impedance_preconditioner_factory(
            linearization.operating_residual,
            z0_ohm=linearization.layout.z0_ohm,
        )(jnp.zeros_like(rhs), rhs)
        solve = solve_linear_system(
            operator,
            rhs,
            config=IterativeLinearSolveConfig(
                method=IterativeLinearSolverMethod.GMRES,
                max_iter=linearization.config.iterative_max_iter,
                atol=linearization.config.iterative_atol,
                rtol=linearization.config.iterative_rtol,
                restart=linearization.config.iterative_restart,
                allow_dense_fallback=False,
                require_convergence=False,
            ),
            preconditioner=preconditioner.to_linear_operator(),
        )

    _, unravel = pack_unknown_tree(linearization.operating_state)
    dx_state = unravel(solve.step)

    residual_perturbation = linearization.apply(dx_state)

    return SmallSignalSolveResult(
        perturbation=SmallSignalState.from_distributed_state(dx_state),
        residual_perturbation=residual_perturbation,
        source=source,
        linear_solve=solve,
        linearization=linearization,
        metadata={
            "source_residual": src_res.summary(),
            "rhs_norm": float(jnp.linalg.norm(rhs)),
            "matrix_free": linearization.config.backend == LinearizationBackend.JAX_LINEARIZE,
        },
    )


# ---------------------------------------------------------------------------
# Gain helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmallSignalGainPoint:
    """
    One signal/idler gain point from a linearized solve.
    """

    signal_label: str
    idler_label: str | None
    input_node: int
    output_node: int
    signal_gain_complex: complex
    signal_gain_db: float
    idler_conversion_complex: complex | None = None
    idler_conversion_db: float | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_label": self.signal_label,
            "idler_label": self.idler_label,
            "input_node": self.input_node,
            "output_node": self.output_node,
            "signal_gain_complex": {
                "real": float(jnp.real(self.signal_gain_complex)),
                "imag": float(jnp.imag(self.signal_gain_complex)),
                "abs": float(abs(self.signal_gain_complex)),
            },
            "signal_gain_db": self.signal_gain_db,
            "idler_conversion_complex": (
                None
                if self.idler_conversion_complex is None
                else {
                    "real": float(jnp.real(self.idler_conversion_complex)),
                    "imag": float(jnp.imag(self.idler_conversion_complex)),
                    "abs": float(abs(self.idler_conversion_complex)),
                }
            ),
            "idler_conversion_db": self.idler_conversion_db,
            "metadata": dict(self.metadata or {}),
        }


def compute_gain_point_from_solution(
    solution: SmallSignalSolveResult,
    *,
    signal_label: str,
    idler_label: str | None = None,
    input_node: int = 0,
    output_node: int | None = None,
) -> SmallSignalGainPoint:
    """
    Compute signal gain and optional idler conversion from a small-signal solve.
    """
    out_node = solution.linearization.layout.n_cells if output_node is None else output_node

    g_sig = solution.voltage_gain_by_label(
        output_label=signal_label,
        input_label=signal_label,
        input_node=input_node,
        output_node=out_node,
    )
    g_sig_db = float(20.0 * jnp.log10(jnp.maximum(abs(g_sig), 1e-300)))

    g_idler = None
    g_idler_db = None
    if idler_label is not None:
        g_idler = solution.voltage_gain_by_label(
            output_label=idler_label,
            input_label=signal_label,
            input_node=input_node,
            output_node=out_node,
        )
        g_idler_db = float(20.0 * jnp.log10(jnp.maximum(abs(g_idler), 1e-300)))

    return SmallSignalGainPoint(
        signal_label=signal_label,
        idler_label=idler_label,
        input_node=input_node,
        output_node=out_node,
        signal_gain_complex=g_sig,
        signal_gain_db=g_sig_db,
        idler_conversion_complex=g_idler,
        idler_conversion_db=g_idler_db,
        metadata={
            "linear_solve": solution.linear_solve.to_dict(),
            "solution": solution.to_dict(),
        },
    )


def solve_signal_current_gain(
    linearization: DistributedHBLinearization,
    *,
    signal_label: str,
    signal_current_rms_A: complex = 1e-12 + 0j,
    input_node: int = 0,
    output_node: int | None = None,
    idler_label: str | None = None,
    set_conjugate: bool = True,
) -> tuple[SmallSignalSolveResult, SmallSignalGainPoint]:
    """
    Inject a small signal current and compute gain.

    This is the main reference helper for small-signal gain validation.
    """
    source = SmallSignalSource.current_phasor_at_node(
        plan=linearization.plan,
        layout=linearization.layout,
        node=input_node,
        label=signal_label,
        rms_current_A=signal_current_rms_A,
        set_conjugate=set_conjugate,
    )

    solution = solve_linearized_small_signal(
        linearization,
        source,
    )

    gain = compute_gain_point_from_solution(
        solution,
        signal_label=signal_label,
        idler_label=idler_label,
        input_node=input_node,
        output_node=output_node,
    )

    return solution, gain


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearizationValidationReport:
    """
    Validation report for the linearization layer.
    """

    passed: bool
    messages: list[str]
    operating_residual_norm: float
    finite_difference_errors: Mapping[str, Any]
    dense_matrix_shape: tuple[int, int] | None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "messages": list(self.messages),
            "operating_residual_norm": self.operating_residual_norm,
            "finite_difference_errors": dict(self.finite_difference_errors),
            "dense_matrix_shape": self.dense_matrix_shape,
            "metadata": dict(self.metadata or {}),
        }


def validate_linearization_finite_difference(
    linearization: DistributedHBLinearization,
    *,
    eps_values: ArrayLike = jnp.asarray([1e-4, 1e-5, 1e-6, 1e-7]),
) -> LinearizationValidationReport:
    """
    Compare stored JVP linearization to finite differences of the nonlinear residual.
    """
    lin = linearization

    residual_fn = make_distributed_hb_residual_function(
        lin.plan,
        lin.layout,
        lin.hb_config,
        lin.ki_model,
        lin.operating_injection_A,
        projection_grid=lin.projection_grid,
        projection_config=lin.config.projection,
    )

    x_vec, unravel = pack_unknown_tree(lin.operating_state)

    direction = jnp.sin(jnp.arange(x_vec.shape[0], dtype=jnp.float64) + 1.0)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-300)
    dx = unravel(direction)

    r0 = residual_fn(lin.operating_state)
    jdx = lin.apply(dx)
    jdx_vec = pack_residual_tree(jdx)
    r0_vec = pack_residual_tree(r0)

    eps = jnp.asarray(eps_values, dtype=jnp.float64)
    errors = []

    for e in eps.tolist():
        x_trial = unravel(x_vec + float(e) * direction)
        r_trial = residual_fn(x_trial)
        fd = (pack_residual_tree(r_trial) - r0_vec) / float(e)
        err = jnp.linalg.norm(fd - jdx_vec) / jnp.maximum(jnp.linalg.norm(jdx_vec), 1e-300)
        errors.append(float(err))

    min_err = float(jnp.min(jnp.asarray(errors)))
    passed = min_err < 1e-3

    messages = []
    if passed:
        messages.append("PASS: linearization finite-difference check passed.")
    else:
        messages.append(f"FAIL: best finite-difference relative error {min_err:.3e} is too large.")

    dense_shape = None
    if lin.dense_jacobian is not None:
        dense_shape = tuple(int(v) for v in lin.dense_jacobian.shape)

    return LinearizationValidationReport(
        passed=passed,
        messages=messages,
        operating_residual_norm=lin.operating_residual.norm,
        finite_difference_errors={
            "eps_values": [float(v) for v in eps.tolist()],
            "relative_errors": errors,
            "min_relative_error": min_err,
        },
        dense_matrix_shape=dense_shape,
        metadata={
            "linearization": lin.to_dict(include_matrix=False),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_complex_2d(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}")
    return arr


def _as_plan_node_coeffs(
    name: str,
    value: ArrayLike,
    plan: FrequencyPlan,
    layout: LineLayout,
) -> jax.Array:
    arr = _as_complex_2d(name, value)
    expected = (plan.n_tones, layout.n_cells + 1)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return arr


__all__ = [
    "LinearizationBackend",
    "SmallSignalSourceKind",
    "SmallSignalLinearizationConfig",
    "SmallSignalState",
    "SmallSignalSource",
    "SmallSignalSolveResult",
    "embed_coefficients_between_plans",
    "embed_distributed_state_between_plans",
    "embed_pump_solution_in_target_plan",
    "DistributedHBLinearization",
    "build_distributed_hb_linearization",
    "build_linearization_from_pump_result",
    "build_dense_real_jacobian",
    "build_dense_real_jacobian_from_linearization",
    "source_to_residual_perturbation",
    "solve_linearized_small_signal",
    "SmallSignalGainPoint",
    "compute_gain_point_from_solution",
    "solve_signal_current_gain",
    "LinearizationValidationReport",
    "validate_linearization_finite_difference",
]
