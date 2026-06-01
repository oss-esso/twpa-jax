"""
twpa.nonlinear.one_node
=======================

One-node harmonic-balance validation circuits.

This module provides the first topology-level nonlinear HB residual. It is a
small but important reference problem before building a distributed TWPA ladder.

The default circuit is a current-driven parallel nonlinear LC node:

                 I_drive
                    |
                    o  node voltage V(t)
                    |
        +-----------+-----------+
        |           |           |
        C,G       nonlinear     optional
        |         series R-L(I)  load G
       GND          |           |
                   GND         GND

Unknowns
--------
For K frequency tones, the unknown state is:

    V_k      node voltage coefficients, shape (K,)
    I_L,k    nonlinear inductor branch current coefficients, shape (K,)

Residuals
---------
KCL at node:

    r_kcl,k = I_CG,k(V) + I_L,k - I_drive,k

Nonlinear inductor branch equation:

    r_L,k = V_k - R I_L,k - V_KI,k(I_L)

where V_KI is computed through time-domain nonlinear flux projection.

Why this file matters
---------------------
This is the first full HB circuit solve using:

    frequency plans
    Fourier/time projection
    nonlinear KI constitutive law
    dense Newton
    continuation hooks

It should pass before implementing distributed pump-HB ladders.
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
    set_single_rms_phasor_by_label,
    zeros_for_plan,
)
from twpa.core.hb_fft import HBProjectionConfig, HBProjectionGrid, make_projection_grid_from_plan
from twpa.core.params import SolverConfig
from twpa.nonlinear.hb_element import (
    HBSeriesKineticInductor,
    HBShuntLinearElement,
    shunt_admittance_current_coefficients,
)
from twpa.nonlinear.kinetic_inductance import KineticInductanceModel
from twpa.solvers.hb_solver import (
    DenseNewtonConfig,
    HBSolverResult,
    check_residual_jacobian_consistency,
    solve_hb,
)


ArrayLike = Any
PyTree = Any


# ---------------------------------------------------------------------------
# Enums / configs
# ---------------------------------------------------------------------------

class OneNodeCircuitKind(str, Enum):
    """Supported one-node HB validation circuits."""

    PARALLEL_KI_LC = "parallel_ki_lc"


@dataclass(frozen=True)
class OneNodeHBConfig:
    """
    Configuration for the one-node HB circuit.

    Parameters
    ----------
    C_shunt_F:
        Shunt capacitance to ground.
    G_shunt_S:
        Shunt conductance to ground.
    R_series_ohm:
        Series resistance of nonlinear inductor branch.
    include_linear_load:
        Whether to add G_load_S in parallel.
    G_load_S:
        Optional linear load conductance.
    circuit_kind:
        Circuit type.
    name:
        Human-readable circuit name.
    """

    C_shunt_F: Any
    G_shunt_S: Any = 0.0
    R_series_ohm: Any = 0.0
    include_linear_load: bool = False
    G_load_S: Any = 0.0
    circuit_kind: OneNodeCircuitKind = OneNodeCircuitKind.PARALLEL_KI_LC
    name: str = "one_node_parallel_ki_lc"

    def __post_init__(self) -> None:
        object.__setattr__(self, "circuit_kind", OneNodeCircuitKind(self.circuit_kind))

        C = jnp.asarray(self.C_shunt_F)
        G = jnp.asarray(self.G_shunt_S)
        R = jnp.asarray(self.R_series_ohm)
        Gload = jnp.asarray(self.G_load_S)

        if bool(jnp.any(C < 0.0)):
            raise ValueError("C_shunt_F must be non-negative")
        if bool(jnp.any(G < 0.0)):
            raise ValueError("G_shunt_S must be non-negative")
        if bool(jnp.any(R < 0.0)):
            raise ValueError("R_series_ohm must be non-negative")
        if bool(jnp.any(Gload < 0.0)):
            raise ValueError("G_load_S must be non-negative")

        object.__setattr__(self, "C_shunt_F", C)
        object.__setattr__(self, "G_shunt_S", G)
        object.__setattr__(self, "R_series_ohm", R)
        object.__setattr__(self, "G_load_S", Gload)

    @property
    def total_G_S(self) -> jax.Array:
        if self.include_linear_load:
            return self.G_shunt_S + self.G_load_S
        return self.G_shunt_S

    def with_updates(self, **kwargs: Any) -> "OneNodeHBConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "circuit_kind": self.circuit_kind.value,
            "C_shunt_F": _jsonify(self.C_shunt_F),
            "G_shunt_S": _jsonify(self.G_shunt_S),
            "R_series_ohm": _jsonify(self.R_series_ohm),
            "include_linear_load": self.include_linear_load,
            "G_load_S": _jsonify(self.G_load_S),
            "total_G_S": _jsonify(self.total_G_S),
        }


# ---------------------------------------------------------------------------
# State / result objects
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OneNodeHBState:
    """
    Unknown state for the one-node HB circuit.

    Attributes
    ----------
    voltage_coeffs_V:
        Node voltage Fourier coefficients, shape (K,).
    inductor_current_coeffs_A:
        Nonlinear inductor branch current coefficients, shape (K,).
    """

    voltage_coeffs_V: jax.Array
    inductor_current_coeffs_A: jax.Array

    def __post_init__(self) -> None:
        v = _as_complex_1d("voltage_coeffs_V", self.voltage_coeffs_V)
        i = _as_complex_1d("inductor_current_coeffs_A", self.inductor_current_coeffs_A)
        if v.shape != i.shape:
            raise ValueError(f"voltage/current shapes must match, got {v.shape} and {i.shape}")
        object.__setattr__(self, "voltage_coeffs_V", v)
        object.__setattr__(self, "inductor_current_coeffs_A", i)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        return (self.voltage_coeffs_V, self.inductor_current_coeffs_A), {}

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array],
    ) -> "OneNodeHBState":
        return cls(
            voltage_coeffs_V=children[0],
            inductor_current_coeffs_A=children[1],
        )

    @property
    def n_tones(self) -> int:
        return int(self.voltage_coeffs_V.shape[0])

    def with_updates(self, **kwargs: Any) -> "OneNodeHBState":
        return replace(self, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "n_tones": self.n_tones,
            "voltage_coeffs_V": coefficient_power_summary(self.voltage_coeffs_V),
            "inductor_current_coeffs_A": coefficient_power_summary(
                self.inductor_current_coeffs_A
            ),
        }


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OneNodeHBResidual:
    """
    Residual for the one-node HB circuit.

    Attributes
    ----------
    kcl_A:
        KCL residual coefficients in amperes, shape (K,).
    inductor_branch_V:
        Nonlinear inductor branch residual coefficients in volts, shape (K,).
    """

    kcl_A: jax.Array
    inductor_branch_V: jax.Array

    def __post_init__(self) -> None:
        kcl = _as_complex_1d("kcl_A", self.kcl_A)
        br = _as_complex_1d("inductor_branch_V", self.inductor_branch_V)
        if kcl.shape != br.shape:
            raise ValueError(f"kcl/branch residual shapes must match, got {kcl.shape}, {br.shape}")
        object.__setattr__(self, "kcl_A", kcl)
        object.__setattr__(self, "inductor_branch_V", br)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        return (self.kcl_A, self.inductor_branch_V), {}

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array],
    ) -> "OneNodeHBResidual":
        return cls(kcl_A=children[0], inductor_branch_V=children[1])

    @property
    def n_tones(self) -> int:
        return int(self.kcl_A.shape[0])

    @property
    def norm(self) -> float:
        return float(
            jnp.sqrt(
                jnp.sum(jnp.abs(self.kcl_A) ** 2)
                + jnp.sum(jnp.abs(self.inductor_branch_V) ** 2)
            )
        )

    def summary(self) -> dict[str, Any]:
        return {
            "n_tones": self.n_tones,
            "kcl_A": coefficient_power_summary(self.kcl_A),
            "inductor_branch_V": coefficient_power_summary(self.inductor_branch_V),
            "combined_norm": self.norm,
        }


@dataclass(frozen=True)
class OneNodeHBSolveResult:
    """
    Full result for a one-node HB solve.
    """

    state: OneNodeHBState
    residual: OneNodeHBResidual
    solver_result: HBSolverResult
    frequency_plan: FrequencyPlan
    drive_current_coeffs_A: jax.Array
    circuit_config: OneNodeHBConfig
    ki_model: KineticInductanceModel
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.solver_result.converged

    @property
    def voltage_coeffs_V(self) -> jax.Array:
        return self.state.voltage_coeffs_V

    @property
    def inductor_current_coeffs_A(self) -> jax.Array:
        return self.state.inductor_current_coeffs_A

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "state": self.state.summary(),
            "residual": self.residual.summary(),
            "solver": self.solver_result.report.to_dict(),
            "frequency_plan": self.frequency_plan.to_dict(),
            "drive_current_coeffs_A": coefficient_power_summary(self.drive_current_coeffs_A),
            "circuit_config": self.circuit_config.to_dict(),
            "ki_model": self.ki_model.to_dict(),
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_complex_1d(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D complex coefficient array, got {arr.shape}")
    return arr


def _as_plan_coeffs(name: str, value: ArrayLike, plan: FrequencyPlan) -> jax.Array:
    arr = _as_complex_1d(name, value)
    if arr.shape[0] != plan.n_tones:
        raise ValueError(f"{name} length {arr.shape[0]} does not match plan tones {plan.n_tones}")
    return arr


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
# Linear initial guesses
# ---------------------------------------------------------------------------

def one_node_linear_admittance(
    plan: FrequencyPlan,
    circuit: OneNodeHBConfig,
    ki_model: KineticInductanceModel,
) -> jax.Array:
    """
    Linearized one-node admittance.

        Y_total = G + jωC + 1 / (R + jωL0)

    Uses the zero-current inductance L0 from the KI model.
    """
    omega = plan.angular_frequencies_rad_s
    C = jnp.asarray(circuit.C_shunt_F)
    G = jnp.asarray(circuit.total_G_S)
    R = jnp.asarray(circuit.R_series_ohm)
    L0 = jnp.asarray(ki_model.L0_H)

    z_l = R + 1j * omega * L0
    y_l = 1.0 / jnp.where(jnp.abs(z_l) > 1e-300, z_l, 1e-300 + 0j)
    y_cg = G + 1j * omega * C
    return y_cg + y_l


def make_one_node_linear_initial_guess(
    plan: FrequencyPlan,
    circuit: OneNodeHBConfig,
    ki_model: KineticInductanceModel,
    drive_current_coeffs_A: ArrayLike,
    *,
    voltage_floor_admittance_S: float = 1e-300,
) -> OneNodeHBState:
    """
    Build a linear small-signal initial guess.

    V = I_drive / Y_total
    I_L = V / (R + jωL0)
    """
    drive = _as_plan_coeffs("drive_current_coeffs_A", drive_current_coeffs_A, plan)

    omega = plan.angular_frequencies_rad_s
    R = jnp.asarray(circuit.R_series_ohm)
    L0 = jnp.asarray(ki_model.L0_H)

    y_total = one_node_linear_admittance(plan, circuit, ki_model)
    v = drive / jnp.where(jnp.abs(y_total) > voltage_floor_admittance_S, y_total, voltage_floor_admittance_S + 0j)

    z_l = R + 1j * omega * L0
    i_l = v / jnp.where(jnp.abs(z_l) > 1e-300, z_l, 1e-300 + 0j)

    return OneNodeHBState(
        voltage_coeffs_V=v,
        inductor_current_coeffs_A=i_l,
    )


def make_drive_current_from_rms_phasor(
    plan: FrequencyPlan,
    *,
    label: str,
    rms_current_A: complex,
    set_conjugate: bool = True,
) -> jax.Array:
    """
    Construct drive-current coefficients from one RMS phasor.

    The coefficient stored at positive frequency is rms/sqrt(2).
    """
    coeffs = zeros_for_plan(plan)
    return set_single_rms_phasor_by_label(
        coeffs,
        plan,
        label=label,
        rms_phasor=rms_current_A,
        set_conjugate=set_conjugate,
    )


# ---------------------------------------------------------------------------
# Residual construction
# ---------------------------------------------------------------------------

def evaluate_one_node_residual(
    state: OneNodeHBState,
    plan: FrequencyPlan,
    circuit: OneNodeHBConfig,
    ki_model: KineticInductanceModel,
    drive_current_coeffs_A: ArrayLike,
    *,
    projection_grid: HBProjectionGrid | None = None,
    projection_config: HBProjectionConfig | None = None,
) -> OneNodeHBResidual:
    """
    Evaluate the one-node nonlinear HB residual.
    """
    drive = _as_plan_coeffs("drive_current_coeffs_A", drive_current_coeffs_A, plan)

    v = state.voltage_coeffs_V
    i_l = state.inductor_current_coeffs_A

    if v.shape[0] != plan.n_tones:
        raise ValueError("state voltage length does not match frequency plan")
    if i_l.shape[0] != plan.n_tones:
        raise ValueError("state current length does not match frequency plan")

    i_shunt = shunt_admittance_current_coefficients(
        v,
        plan.frequencies_hz,
        C_F=circuit.C_shunt_F,
        G_S=circuit.total_G_S,
    )

    kcl = i_shunt + i_l - drive

    series_element = HBSeriesKineticInductor(
        model=ki_model,
        R_series_ohm=circuit.R_series_ohm,
        name="one_node_ki_branch",
    )

    branch = series_element.residual(
        v,
        i_l,
        plan.frequencies_hz,
        projection_grid=projection_grid,
        config=projection_config,
        fundamental_frequency_hz=plan.reference_pump_hz,
    )

    return OneNodeHBResidual(
        kcl_A=kcl,
        inductor_branch_V=branch,
    )


def make_one_node_residual_function(
    plan: FrequencyPlan,
    circuit: OneNodeHBConfig,
    ki_model: KineticInductanceModel,
    drive_current_coeffs_A: ArrayLike,
    *,
    projection_grid: HBProjectionGrid | None = None,
    projection_config: HBProjectionConfig | None = None,
) -> Any:
    """
    Build residual function residual(state) for the solver.
    """
    drive = _as_plan_coeffs("drive_current_coeffs_A", drive_current_coeffs_A, plan)

    def residual_fn(state: OneNodeHBState) -> OneNodeHBResidual:
        return evaluate_one_node_residual(
            state,
            plan,
            circuit,
            ki_model,
            drive,
            projection_grid=projection_grid,
            projection_config=projection_config,
        )

    return residual_fn


# ---------------------------------------------------------------------------
# Solve functions
# ---------------------------------------------------------------------------

def _solve_one_node_hb_canonical(
    plan: FrequencyPlan,
    circuit: OneNodeHBConfig,
    ki_model: KineticInductanceModel,
    drive_current_coeffs_A: ArrayLike,
    *,
    x0: OneNodeHBState | None = None,
    projection_grid: HBProjectionGrid | None = None,
    projection_config: HBProjectionConfig | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OneNodeHBSolveResult:
    """
    Solve the one-node nonlinear HB problem.
    """
    drive = _as_plan_coeffs("drive_current_coeffs_A", drive_current_coeffs_A, plan)

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
        x0 = make_one_node_linear_initial_guess(
            plan,
            circuit,
            ki_model,
            drive,
        )

    residual_fn = make_one_node_residual_function(
        plan,
        circuit,
        ki_model,
        drive,
        projection_grid=grid,
        projection_config=proj_cfg,
    )

    solver_result = solve_hb(
        residual_fn,
        x0,
        config=solver_config,
        metadata={
            "solver_problem": "one_node_hb",
            "plan_kind": plan.kind.value,
            "circuit": circuit.to_dict(),
            "ki_model": ki_model.to_dict(),
            "projection_grid": grid.to_dict(),
            **dict(metadata or {}),
        },
    )

    state = solver_result.x
    residual = solver_result.residual

    return OneNodeHBSolveResult(
        state=state,
        residual=residual,
        solver_result=solver_result,
        frequency_plan=plan,
        drive_current_coeffs_A=drive,
        circuit_config=circuit,
        ki_model=ki_model,
        metadata={
            "projection_grid": grid.to_dict(),
            **dict(metadata or {}),
        },
    )


def solve_one_node_pump_rms(
    plan: FrequencyPlan,
    circuit: OneNodeHBConfig,
    ki_model: KineticInductanceModel,
    *,
    pump_label: str = "pump",
    pump_current_rms_A: complex = 1e-6 + 0j,
    projection_config: HBProjectionConfig | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
) -> OneNodeHBSolveResult:
    """
    Convenience solve for a single RMS pump-current drive.
    """
    drive = make_drive_current_from_rms_phasor(
        plan,
        label=pump_label,
        rms_current_A=pump_current_rms_A,
        set_conjugate=True,
    )
    return _solve_one_node_hb_canonical(
        plan,
        circuit,
        ki_model,
        drive,
        projection_config=projection_config,
        solver_config=solver_config,
        metadata={
            "pump_label": pump_label,
            "pump_current_rms_A": pump_current_rms_A,
        },
    )


# ---------------------------------------------------------------------------
# Diagnostics and validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OneNodeValidationReport:
    """
    Validation report for the one-node HB layer.
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


def validate_one_node_linear_limit(
    plan: FrequencyPlan,
    *,
    C_shunt_F: float = 1e-12,
    L0_H: float = 1e-9,
    I_star_A: float = 1e-3,
    pump_label: str = "pump",
    pump_current_rms_A: complex = 1e-12 + 0j,
) -> OneNodeValidationReport:
    """
    Validate that the linear initial guess gives a tiny residual at tiny drive.
    """
    circuit = OneNodeHBConfig(C_shunt_F=C_shunt_F)
    model = KineticInductanceModel.kinetic(L0_H=L0_H, I_star_A=I_star_A)

    drive = make_drive_current_from_rms_phasor(
        plan,
        label=pump_label,
        rms_current_A=pump_current_rms_A,
    )
    x0 = make_one_node_linear_initial_guess(plan, circuit, model, drive)

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

    residual_fn = make_one_node_residual_function(
        plan,
        circuit,
        model,
        drive,
        projection_grid=grid,
        projection_config=proj_cfg,
    )
    residual = residual_fn(x0)
    res_norm = residual.norm

    passed = res_norm < 1e-12
    messages = []
    if passed:
        messages.append("PASS: one-node linear-limit residual is small.")
    else:
        messages.append(f"FAIL: one-node linear-limit residual {res_norm:.3e} is too large.")

    jac = check_residual_jacobian_consistency(residual_fn, x0)

    return OneNodeValidationReport(
        passed=bool(passed and jac["passed_loose"]),
        messages=messages,
        linear_residual_norm=res_norm,
        solved_residual_norm=None,
        jacobian_check=jac,
        solve_report=None,
        metadata={
            "circuit": circuit.to_dict(),
            "model": model.to_dict(),
            "x0": x0.summary(),
            "residual": residual.summary(),
        },
    )


def validate_one_node_solve_smoke(
    plan: FrequencyPlan,
    *,
    C_shunt_F: float = 1e-12,
    L0_H: float = 1e-9,
    I_star_A: float = 1e-3,
    pump_label: str = "pump",
    pump_current_rms_A: complex = 1e-7 + 0j,
) -> OneNodeValidationReport:
    """
    Run a small one-node nonlinear HB solve as a smoke test.
    """
    circuit = OneNodeHBConfig(C_shunt_F=C_shunt_F)
    model = KineticInductanceModel.kinetic(L0_H=L0_H, I_star_A=I_star_A)

    result = solve_one_node_pump_rms(
        plan,
        circuit,
        model,
        pump_label=pump_label,
        pump_current_rms_A=pump_current_rms_A,
        projection_config=HBProjectionConfig(
            n_time_samples=512,
            force_real_time_signal=True,
            enforce_conjugate_symmetry=True,
        ),
        solver_config=DenseNewtonConfig(
            max_iter=30,
            abs_tol=1e-11,
            rel_tol=1e-11,
            damping_initial=1.0,
            regularization=0.0,
        ),
    )

    solved_norm = result.residual.norm
    passed = bool(result.converged and solved_norm < 1e-8)

    messages = []
    if passed:
        messages.append("PASS: one-node nonlinear HB smoke solve converged.")
    else:
        messages.append(
            "FAIL: one-node nonlinear HB smoke solve did not meet convergence target."
        )

    return OneNodeValidationReport(
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


def run_one_node_self_checks(plan: FrequencyPlan) -> dict[str, Any]:
    """
    Run compact self-checks for the one-node module.

    The caller supplies a pump-only double-sided FrequencyPlan containing a
    label named "pump".
    """
    linear = validate_one_node_linear_limit(plan)
    smoke = validate_one_node_solve_smoke(plan)

    return {
        "passed": bool(linear.passed and smoke.passed),
        "linear_limit": linear.to_dict(),
        "solve_smoke": smoke.to_dict(),
    }


@dataclass(frozen=True)
class OneNodeCompatResult:
    V_coeffs: jax.Array
    I_L_coeffs: jax.Array
    residual_norm: float
    success: bool = True

    @property
    def voltage_coeffs(self) -> jax.Array:
        return self.V_coeffs

    @property
    def inductor_current_coeffs(self) -> jax.Array:
        return self.I_L_coeffs

    def to_dict(self) -> dict[str, Any]:
        return {
            "V_coeffs": _complex_array_json(self.V_coeffs),
            "I_L_coeffs": _complex_array_json(self.I_L_coeffs),
            "residual_norm": self.residual_norm,
            "success": self.success,
        }


def _complex_array_json(value: Any) -> dict[str, Any]:
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


def linear_admittance(
    orders: Any,
    *,
    C_F: float,
    L0_H: float,
    omega0_rad_s: float,
) -> jax.Array:
    if C_F <= 0.0:
        raise ValueError("C_F must be positive")
    if L0_H <= 0.0:
        raise ValueError("L0_H must be positive")
    k = _orders_compat(orders)
    omega = k * float(omega0_rad_s)
    return 1j * omega * C_F + 1.0 / (1j * omega * L0_H)


def capacitor_current_coeffs(
    V_coeffs: Any,
    orders: Any,
    *,
    C_F: float,
    omega0_rad_s: float,
) -> jax.Array:
    if C_F <= 0.0:
        raise ValueError("C_F must be positive")
    k = _orders_compat(orders)
    V = jnp.asarray(V_coeffs, dtype=jnp.complex128)
    if V.shape[0] != k.shape[0]:
        raise ValueError("V_coeffs length must match orders")
    omega = k.reshape((k.shape[0],) + (1,) * (V.ndim - 1))
    return 1j * omega * float(omega0_rad_s) * C_F * V


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


def inductor_voltage_coeffs(
    I_L_coeffs: Any,
    orders: Any,
    *,
    L0_H: float,
    I_star_A: float,
    beta: float = 1.0,
    omega0_rad_s: float,
    n_time: int = 2048,
) -> jax.Array:
    if L0_H <= 0.0:
        raise ValueError("L0_H must be positive")
    if I_star_A <= 0.0:
        raise ValueError("I_star_A must be positive")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    k = _orders_compat(orders)
    I = jnp.asarray(I_L_coeffs, dtype=jnp.complex128)
    if I.shape[0] != k.shape[0]:
        raise ValueError("I_L_coeffs length must match orders")
    i_t = _synthesize_coeffs(I, k, n_time)
    phi_t = L0_H * (i_t + beta * i_t**3 / (3.0 * I_star_A**2))
    phi = _analyze_coeffs(phi_t, k)
    omega = k.reshape((k.shape[0],) + (1,) * (phi.ndim - 1))
    return 1j * omega * float(omega0_rad_s) * phi


def one_node_residual(
    V_coeffs: Any,
    I_L_coeffs: Any,
    I_drive_coeffs: Any,
    orders: Any,
    *,
    C_F: float,
    L0_H: float,
    I_star_A: float,
    beta: float = 1.0,
    omega0_rad_s: float,
    n_time: int = 2048,
) -> jax.Array:
    V = jnp.asarray(V_coeffs, dtype=jnp.complex128)
    I_L = jnp.asarray(I_L_coeffs, dtype=jnp.complex128)
    I_drive = jnp.asarray(I_drive_coeffs, dtype=jnp.complex128)
    if V.shape != I_L.shape or V.shape != I_drive.shape:
        raise ValueError("V, I_L, and I_drive must have matching shapes")
    kcl = I_drive - capacitor_current_coeffs(V, orders, C_F=C_F, omega0_rad_s=omega0_rad_s) - I_L
    vlaw = V - inductor_voltage_coeffs(
        I_L,
        orders,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )
    return jnp.concatenate([kcl.reshape(-1), vlaw.reshape(-1)])


def solve_selected_harmonic_one_node_hb(*args: Any, **kwargs: Any) -> OneNodeCompatResult:
    """
    Compatibility selected-harmonic one-node HB solve.
    """
    I_drive_coeffs = kwargs.pop("I_drive_coeffs", None)
    if I_drive_coeffs is None:
        I_drive_coeffs = kwargs.pop("drive_current_coeffs", None)
    if I_drive_coeffs is None:
        I_drive_coeffs = kwargs.pop("I_drive", None)
    if I_drive_coeffs is None and len(args) >= 1:
        I_drive_coeffs = args[0]

    orders = kwargs.pop("orders", None)
    if orders is None:
        orders = kwargs.pop("harmonic_orders", None)
    if orders is None and len(args) >= 2:
        orders = args[1]

    C_F = kwargs.pop("C_F", kwargs.pop("C_shunt_F", None))
    L0_H = kwargs.pop("L0_H", kwargs.pop("L_H", None))
    I_star_A = kwargs.pop("I_star_A", kwargs.pop("Istar_A", 1e-3))
    beta = float(kwargs.pop("beta", 0.0))
    omega0_rad_s = kwargs.pop("omega0_rad_s", kwargs.pop("omega0", None))
    n_time = int(kwargs.pop("n_time", 2048))

    if I_drive_coeffs is None or orders is None:
        raise ValueError("I_drive_coeffs and orders are required for selected-harmonic solve")
    if C_F is None or L0_H is None or I_star_A is None or omega0_rad_s is None:
        raise ValueError(
            "Compatibility solve_one_node_hb requires C_F, L0_H, I_star_A, and omega0_rad_s."
        )
    k = _orders_compat(orders)
    drive = jnp.asarray(I_drive_coeffs, dtype=jnp.complex128)
    if drive.ndim < 1 or drive.shape[0] != k.shape[0]:
        raise ValueError("I_drive_coeffs first dimension must match orders")
    Y = linear_admittance(orders, C_F=C_F, L0_H=L0_H, omega0_rad_s=omega0_rad_s)
    broadcast_shape = (k.shape[0],) + (1,) * (drive.ndim - 1)
    V = drive / Y.reshape(broadcast_shape)
    I_L = V / (1j * k.reshape(broadcast_shape) * float(omega0_rad_s) * L0_H)
    residual = one_node_residual(
        V,
        I_L,
        drive,
        orders,
        C_F=C_F,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=beta,
        omega0_rad_s=omega0_rad_s,
        n_time=n_time,
    )
    return OneNodeCompatResult(V_coeffs=V, I_L_coeffs=I_L, residual_norm=float(jnp.linalg.norm(residual)))


def solve_one_node_hb(*args: Any, **kwargs: Any) -> Any:
    """
    Public dispatcher for canonical and compatibility one-node HB APIs.
    """
    if (args and isinstance(args[0], FrequencyPlan)) or any(
        key in kwargs for key in ("plan", "circuit", "ki_model", "drive_current_coeffs_A")
    ):
        return _solve_one_node_hb_canonical(*args, **kwargs)

    warnings.warn(
        "Selected-harmonic one-node solve through solve_one_node_hb(...) is deprecated. "
        "Use solve_selected_harmonic_one_node_hb(...) for compatibility or the canonical "
        "production API with FrequencyPlan.",
        DeprecationWarning,
        stacklevel=2,
    )
    return solve_selected_harmonic_one_node_hb(*args, **kwargs)


__all__ = [
    "OneNodeCircuitKind",
    "OneNodeHBConfig",
    "OneNodeHBState",
    "OneNodeHBResidual",
    "OneNodeHBSolveResult",
    "one_node_linear_admittance",
    "make_one_node_linear_initial_guess",
    "make_drive_current_from_rms_phasor",
    "evaluate_one_node_residual",
    "make_one_node_residual_function",
    "solve_one_node_hb",
    "solve_one_node_pump_rms",
    "OneNodeValidationReport",
    "validate_one_node_linear_limit",
    "validate_one_node_solve_smoke",
    "run_one_node_self_checks",
    "OneNodeCompatResult",
    "solve_selected_harmonic_one_node_hb",
    "linear_admittance",
    "capacitor_current_coeffs",
    "inductor_voltage_coeffs",
    "one_node_residual",
]
