"""
twpa.solvers.continuation
=========================

Continuation / homotopy drivers for nonlinear HB solves.

Why this exists
---------------
Large-signal TWPA harmonic-balance problems are rarely solved robustly by
jumping directly to the final pump power, final nonlinearity, or final line
length. The stable route is continuation:

    easy problem -> slightly harder problem -> ... -> target problem

Common continuation parameters are:
    - pump power,
    - pump amplitude,
    - pump frequency,
    - nonlinearity strength,
    - source amplitude,
    - number of cells / effective length,
    - loss or mismatch strength.

This module is circuit-agnostic. It does not know how to build a TWPA residual.
Instead, the caller provides a residual factory:

    residual_fn = residual_factory(value, previous_solution, context)

and an initial guess factory or initial state.

Design rules
------------
1. Every continuation step returns a report.
2. Failed steps are not hidden.
3. Adaptive step shrinking is explicit.
4. Successful steps can grow the step size.
5. Results are restartable/serializable through summaries.
6. This module works with dense Newton first, but can dispatch to other solvers
   later through solve_hb().
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp

from twpa.core.params import ContinuationConfig, SolverConfig
from twpa.solvers.hb_solver import (
    DenseNewtonConfig,
    HBSolverReport,
    HBSolverResult,
    ResidualFn,
    SolverStatus,
    solve_hb,
    unknown_summary,
)


ArrayLike = Any
PyTree = Any


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ResidualFactory = Callable[[float, PyTree, Mapping[str, Any]], ResidualFn]
InitialGuessFactory = Callable[[float, Mapping[str, Any]], PyTree]
PredictorFn = Callable[[float, float, PyTree, PyTree | None, Mapping[str, Any]], PyTree]
AcceptFn = Callable[[HBSolverResult, float, Mapping[str, Any]], bool]


# ---------------------------------------------------------------------------
# Enums / configs
# ---------------------------------------------------------------------------

class ContinuationStatus(str, Enum):
    """Overall continuation status."""

    CONVERGED = "converged"
    PARTIAL = "partial"
    FAILED = "failed"
    EMPTY = "empty"


class StepStatus(str, Enum):
    """Single continuation step status."""

    CONVERGED = "converged"
    FAILED = "failed"
    SKIPPED = "skipped"


class ContinuationScheduleKind(str, Enum):
    """Schedule construction kind."""

    LINEAR = "linear"
    GEOMETRIC = "geometric"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class ContinuationSolverConfig:
    """
    Full continuation solver configuration.

    This wraps the core ContinuationConfig and adds adaptive/retry controls.

    Parameters
    ----------
    schedule_kind:
        Linear, geometric, or explicit schedule.
    adaptive:
        Whether to adaptively shrink/grow steps.
    max_step_retries:
        Maximum retries for a failing step before giving up.
    shrink_factor:
        Factor used to shrink a failed step.
    growth_factor:
        Factor used to grow after success.
    min_step_abs:
        Absolute lower bound on continuation step size.
    min_step_fraction:
        Relative lower bound based on full span.
    stop_on_failure:
        If true, stop when a step cannot be solved.
        If false, record failure and continue only when explicit schedules make
        that meaningful.
    reuse_previous_solution:
        If true, use previous solution as the next initial guess.
    use_secant_predictor:
        If true, use x_pred = x_k + slope * step from last two successful
        solutions when available.
    """

    schedule_kind: ContinuationScheduleKind = ContinuationScheduleKind.LINEAR
    adaptive: bool = True
    max_step_retries: int = 8
    shrink_factor: float = 0.5
    growth_factor: float = 1.25
    min_step_abs: float = 0.0
    min_step_fraction: float = 1e-3
    stop_on_failure: bool = True
    reuse_previous_solution: bool = True
    use_secant_predictor: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule_kind", ContinuationScheduleKind(self.schedule_kind))
        if int(self.max_step_retries) < 0:
            raise ValueError("max_step_retries must be non-negative")
        object.__setattr__(self, "max_step_retries", int(self.max_step_retries))
        if not (0.0 < self.shrink_factor < 1.0):
            raise ValueError("shrink_factor must be in (0, 1)")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1")
        if self.min_step_abs < 0.0:
            raise ValueError("min_step_abs must be non-negative")
        if not (0.0 < self.min_step_fraction <= 1.0):
            raise ValueError("min_step_fraction must be in (0, 1]")

    @classmethod
    def from_core_config(cls, config: ContinuationConfig) -> "ContinuationSolverConfig":
        """
        Build from twpa.core.params.ContinuationConfig.
        """
        return cls(
            schedule_kind=ContinuationScheduleKind.LINEAR,
            adaptive=config.adaptive,
            max_step_retries=8,
            shrink_factor=config.shrink_factor,
            growth_factor=config.growth_factor,
            min_step_abs=0.0,
            min_step_fraction=config.min_step_fraction,
            stop_on_failure=True,
            reuse_previous_solution=True,
            use_secant_predictor=True,
        )

    def with_updates(self, **kwargs: Any) -> "ContinuationSolverConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_kind": self.schedule_kind.value,
            "adaptive": self.adaptive,
            "max_step_retries": self.max_step_retries,
            "shrink_factor": self.shrink_factor,
            "growth_factor": self.growth_factor,
            "min_step_abs": self.min_step_abs,
            "min_step_fraction": self.min_step_fraction,
            "stop_on_failure": self.stop_on_failure,
            "reuse_previous_solution": self.reuse_previous_solution,
            "use_secant_predictor": self.use_secant_predictor,
        }


# ---------------------------------------------------------------------------
# Reports / results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContinuationStepReport:
    """
    Report for one continuation step attempt.
    """

    step_index: int
    target_value: float
    previous_value: float | None
    attempted_step_size: float | None
    status: StepStatus
    accepted: bool
    retries: int
    solver_report: HBSolverReport | None
    message: str
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.status == StepStatus.CONVERGED

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "target_value": self.target_value,
            "previous_value": self.previous_value,
            "attempted_step_size": self.attempted_step_size,
            "status": self.status.value,
            "accepted": self.accepted,
            "retries": self.retries,
            "solver_report": None if self.solver_report is None else self.solver_report.to_dict(),
            "message": self.message,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ContinuationResult:
    """
    Full continuation result.

    Attributes
    ----------
    values:
        Accepted continuation parameter values.
    solutions:
        Accepted solutions, same order as values.
    residuals:
        Residuals at accepted solutions.
    step_reports:
        Reports for accepted and failed steps.
    status:
        Overall continuation status.
    metadata:
        Extra metadata.
    """

    values: tuple[float, ...]
    solutions: tuple[PyTree, ...]
    residuals: tuple[PyTree, ...]
    step_reports: tuple[ContinuationStepReport, ...]
    status: ContinuationStatus
    metadata: Mapping[str, Any] | None = None

    @property
    def converged(self) -> bool:
        return self.status == ContinuationStatus.CONVERGED

    @property
    def partial(self) -> bool:
        return self.status == ContinuationStatus.PARTIAL

    @property
    def failed(self) -> bool:
        return self.status == ContinuationStatus.FAILED

    @property
    def last_solution(self) -> PyTree | None:
        return None if not self.solutions else self.solutions[-1]

    @property
    def last_value(self) -> float | None:
        return None if not self.values else self.values[-1]

    @property
    def n_accepted(self) -> int:
        return len(self.values)

    @property
    def n_failed_steps(self) -> int:
        return sum(1 for r in self.step_reports if r.status == StepStatus.FAILED)

    def final_solver_report(self) -> HBSolverReport | None:
        for report in reversed(self.step_reports):
            if report.solver_report is not None:
                return report.solver_report
        return None

    def summary_line(self) -> str:
        last = self.last_value
        last_text = "None" if last is None else f"{last:.9g}"
        return (
            f"{self.status.value}: accepted={self.n_accepted}, "
            f"failed_steps={self.n_failed_steps}, last_value={last_text}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "converged": self.converged,
            "n_accepted": self.n_accepted,
            "n_failed_steps": self.n_failed_steps,
            "values": list(self.values),
            "step_reports": [r.to_dict() for r in self.step_reports],
            "metadata": dict(self.metadata or {}),
        }


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------

def linear_schedule(start: float, stop: float, n_steps: int) -> jax.Array:
    """
    Inclusive linear schedule from start to stop.
    """
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive")
    if int(n_steps) == 1:
        return jnp.asarray([float(stop)], dtype=jnp.float64)
    return jnp.linspace(float(start), float(stop), int(n_steps))


def geometric_schedule(start: float, stop: float, n_steps: int) -> jax.Array:
    """
    Inclusive geometric schedule.

    Requires start and stop to have the same nonzero sign.
    """
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive")
    if int(n_steps) == 1:
        return jnp.asarray([float(stop)], dtype=jnp.float64)
    if start == 0.0 or stop == 0.0:
        raise ValueError("geometric_schedule requires nonzero start and stop")
    if start * stop < 0.0:
        raise ValueError("geometric_schedule requires start and stop with same sign")

    sign = 1.0 if start > 0.0 else -1.0
    vals = jnp.geomspace(abs(float(start)), abs(float(stop)), int(n_steps))
    return sign * vals


def make_continuation_schedule(
    *,
    start: float,
    stop: float,
    n_steps: int,
    kind: ContinuationScheduleKind | str = ContinuationScheduleKind.LINEAR,
) -> tuple[float, ...]:
    """
    Build an inclusive continuation schedule.
    """
    kind = ContinuationScheduleKind(kind)
    if kind == ContinuationScheduleKind.LINEAR:
        arr = linear_schedule(start, stop, n_steps)
    elif kind == ContinuationScheduleKind.GEOMETRIC:
        arr = geometric_schedule(start, stop, n_steps)
    else:
        raise ValueError("Explicit schedule must be passed directly, not built here")
    return tuple(float(v) for v in arr.tolist())


def schedule_from_core_config(
    config: ContinuationConfig,
    *,
    default_start: float,
    default_stop: float,
    kind: ContinuationScheduleKind | str = ContinuationScheduleKind.LINEAR,
) -> tuple[float, ...]:
    """
    Build schedule from core ContinuationConfig.
    """
    start = default_start if config.start_value is None else float(config.start_value)
    stop = default_stop if config.stop_value is None else float(config.stop_value)
    return make_continuation_schedule(
        start=start,
        stop=stop,
        n_steps=config.n_steps,
        kind=kind,
    )


def validate_schedule(values: Sequence[float]) -> tuple[float, ...]:
    """
    Validate and normalize an explicit schedule.
    """
    if len(values) == 0:
        raise ValueError("Continuation schedule may not be empty")
    cleaned = tuple(float(v) for v in values)
    if any(not jnp.isfinite(v) for v in cleaned):
        raise ValueError("Continuation schedule contains nonfinite values")
    return cleaned


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def identity_predictor(
    target_value: float,
    previous_value: float,
    previous_solution: PyTree,
    penultimate_solution: PyTree | None,
    context: Mapping[str, Any],
) -> PyTree:
    """
    Default predictor: reuse previous solution.
    """
    return previous_solution


def secant_predictor(
    target_value: float,
    previous_value: float,
    previous_solution: PyTree,
    penultimate_solution: PyTree | None,
    context: Mapping[str, Any],
) -> PyTree:
    """
    Secant predictor using last two accepted solutions.

    If no penultimate solution is available, returns previous_solution.

    The context may include "penultimate_value".
    """
    if penultimate_solution is None:
        return previous_solution

    penultimate_value = context.get("penultimate_value", None)
    if penultimate_value is None:
        return previous_solution

    denom = previous_value - float(penultimate_value)
    if abs(denom) <= 1e-300:
        return previous_solution

    ratio = (target_value - previous_value) / denom

    def extrapolate(x_prev: Any, x_pen: Any) -> Any:
        return x_prev + ratio * (x_prev - x_pen)

    return jax.tree_util.tree_map(extrapolate, previous_solution, penultimate_solution)


def choose_predictor(config: ContinuationSolverConfig) -> PredictorFn:
    """
    Choose default predictor from config.
    """
    if config.use_secant_predictor:
        return secant_predictor
    return identity_predictor


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def default_accept(result: HBSolverResult, value: float, context: Mapping[str, Any]) -> bool:
    """
    Default step acceptance: solver must converge.
    """
    return bool(result.converged)


def residual_threshold_accept(
    *,
    max_final_residual_norm: float,
) -> AcceptFn:
    """
    Build an accept function requiring convergence and residual below threshold.
    """
    if max_final_residual_norm <= 0.0:
        raise ValueError("max_final_residual_norm must be positive")

    def accept(result: HBSolverResult, value: float, context: Mapping[str, Any]) -> bool:
        return (
            result.converged
            and result.report.final_residual_norm <= max_final_residual_norm
        )

    return accept


# ---------------------------------------------------------------------------
# Core continuation solve
# ---------------------------------------------------------------------------

def _make_failed_report(
    *,
    step_index: int,
    target_value: float,
    previous_value: float | None,
    attempted_step_size: float | None,
    retries: int,
    solver_report: HBSolverReport | None,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuationStepReport:
    return ContinuationStepReport(
        step_index=step_index,
        target_value=float(target_value),
        previous_value=None if previous_value is None else float(previous_value),
        attempted_step_size=None if attempted_step_size is None else float(attempted_step_size),
        status=StepStatus.FAILED,
        accepted=False,
        retries=int(retries),
        solver_report=solver_report,
        message=message,
        metadata=dict(metadata or {}),
    )


def _minimum_step_size(
    schedule_start: float,
    schedule_stop: float,
    config: ContinuationSolverConfig,
) -> float:
    span = abs(schedule_stop - schedule_start)
    return max(float(config.min_step_abs), float(config.min_step_fraction) * max(span, 1e-300))


def solve_continuation(
    *,
    schedule: Sequence[float],
    residual_factory: ResidualFactory,
    x0: PyTree | None = None,
    initial_guess_factory: InitialGuessFactory | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    continuation_config: ContinuationSolverConfig | ContinuationConfig | None = None,
    predictor: PredictorFn | None = None,
    accept: AcceptFn | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuationResult:
    """
    Solve a sequence of nonlinear problems by continuation.

    Parameters
    ----------
    schedule:
        Explicit target values.
    residual_factory:
        Callable producing residual_fn(value, guess, context).
    x0:
        Initial guess for the first schedule value.
    initial_guess_factory:
        Alternative way to build first guess from value/context.
    solver_config:
        Config passed to solve_hb().
    continuation_config:
        ContinuationSolverConfig or core ContinuationConfig.
    predictor:
        Optional predictor for subsequent guesses.
    accept:
        Optional step accept function.
    context:
        Context passed to residual_factory/predictor/accept.
    metadata:
        Extra metadata for the final result.
    """
    values_target = validate_schedule(schedule)
    ctx_base = dict(context or {})

    if continuation_config is None:
        cont_cfg = ContinuationSolverConfig()
    elif isinstance(continuation_config, ContinuationConfig):
        cont_cfg = ContinuationSolverConfig.from_core_config(continuation_config)
    else:
        cont_cfg = continuation_config

    pred = predictor or choose_predictor(cont_cfg)
    accept_fn = accept or default_accept

    if x0 is None and initial_guess_factory is None:
        raise ValueError("Either x0 or initial_guess_factory must be provided")

    accepted_values: list[float] = []
    accepted_solutions: list[PyTree] = []
    accepted_residuals: list[PyTree] = []
    reports: list[ContinuationStepReport] = []

    schedule_start = values_target[0]
    schedule_stop = values_target[-1]
    min_step = _minimum_step_size(schedule_start, schedule_stop, cont_cfg)

    # ------------------------------------------------------------------
    # First step
    # ------------------------------------------------------------------
    first_value = values_target[0]
    if x0 is None:
        assert initial_guess_factory is not None
        guess = initial_guess_factory(first_value, ctx_base)
    else:
        guess = x0

    residual_fn = residual_factory(first_value, guess, ctx_base)
    result = solve_hb(
        residual_fn,
        guess,
        config=solver_config,
        metadata={
            "continuation_value": first_value,
            "continuation_step_index": 0,
            "continuation_role": "initial",
        },
    )

    accepted = accept_fn(result, first_value, ctx_base)

    reports.append(
        ContinuationStepReport(
            step_index=0,
            target_value=first_value,
            previous_value=None,
            attempted_step_size=None,
            status=StepStatus.CONVERGED if accepted else StepStatus.FAILED,
            accepted=bool(accepted),
            retries=0,
            solver_report=result.report,
            message="initial step accepted" if accepted else "initial step failed",
            metadata={
                "unknown_summary_initial": unknown_summary(guess),
                "unknown_summary_solution": unknown_summary(result.x),
            },
        )
    )

    if not accepted:
        return ContinuationResult(
            values=tuple(),
            solutions=tuple(),
            residuals=tuple(),
            step_reports=tuple(reports),
            status=ContinuationStatus.FAILED,
            metadata={
                **dict(metadata or {}),
                "continuation_config": cont_cfg.to_dict(),
                "schedule": list(values_target),
                "message": "initial continuation step failed",
            },
        )

    accepted_values.append(first_value)
    accepted_solutions.append(result.x)
    accepted_residuals.append(result.residual)

    if len(values_target) == 1:
        return ContinuationResult(
            values=tuple(accepted_values),
            solutions=tuple(accepted_solutions),
            residuals=tuple(accepted_residuals),
            step_reports=tuple(reports),
            status=ContinuationStatus.CONVERGED,
            metadata={
                **dict(metadata or {}),
                "continuation_config": cont_cfg.to_dict(),
                "schedule": list(values_target),
            },
        )

    # ------------------------------------------------------------------
    # Remaining steps
    # ------------------------------------------------------------------
    target_index = 1
    current_value = first_value
    current_solution = result.x
    penultimate_solution: PyTree | None = None
    penultimate_value: float | None = None

    while target_index < len(values_target):
        nominal_target = values_target[target_index]
        direction = 1.0 if nominal_target >= current_value else -1.0
        target_value = nominal_target
        retries = 0
        step_accepted = False
        last_solver_report: HBSolverReport | None = None
        last_message = ""

        while retries <= cont_cfg.max_step_retries:
            attempted_step = target_value - current_value
            if abs(attempted_step) < min_step and target_value != nominal_target:
                reports.append(
                    _make_failed_report(
                        step_index=target_index,
                        target_value=target_value,
                        previous_value=current_value,
                        attempted_step_size=attempted_step,
                        retries=retries,
                        solver_report=last_solver_report,
                        message=(
                            f"adaptive step below minimum {min_step:.3e}; "
                            f"last_message={last_message}"
                        ),
                    )
                )
                break

            step_context = {
                **ctx_base,
                "previous_value": current_value,
                "target_value": target_value,
                "nominal_target_value": nominal_target,
                "penultimate_value": penultimate_value,
                "step_index": target_index,
                "retry_index": retries,
            }

            if cont_cfg.reuse_previous_solution:
                guess = pred(
                    target_value,
                    current_value,
                    current_solution,
                    penultimate_solution,
                    step_context,
                )
            elif initial_guess_factory is not None:
                guess = initial_guess_factory(target_value, step_context)
            else:
                guess = current_solution

            residual_fn = residual_factory(target_value, guess, step_context)
            step_result = solve_hb(
                residual_fn,
                guess,
                config=solver_config,
                metadata={
                    "continuation_value": target_value,
                    "continuation_step_index": target_index,
                    "continuation_retry_index": retries,
                    "nominal_target_value": nominal_target,
                },
            )
            last_solver_report = step_result.report

            step_accepted = accept_fn(step_result, target_value, step_context)

            if step_accepted:
                reports.append(
                    ContinuationStepReport(
                        step_index=target_index,
                        target_value=target_value,
                        previous_value=current_value,
                        attempted_step_size=attempted_step,
                        status=StepStatus.CONVERGED,
                        accepted=True,
                        retries=retries,
                        solver_report=step_result.report,
                        message="step accepted",
                        metadata={
                            "nominal_target_value": nominal_target,
                            "unknown_summary_initial": unknown_summary(guess),
                            "unknown_summary_solution": unknown_summary(step_result.x),
                        },
                    )
                )

                penultimate_solution = current_solution
                penultimate_value = current_value

                current_solution = step_result.x
                current_value = target_value

                accepted_values.append(float(current_value))
                accepted_solutions.append(current_solution)
                accepted_residuals.append(step_result.residual)

                # If this was an adaptive intermediate target, continue toward
                # the same nominal schedule value rather than advancing index.
                if abs(current_value - nominal_target) <= 1e-15 * max(abs(nominal_target), 1.0):
                    target_index += 1
                else:
                    # Successful intermediate step. Try a bigger step next,
                    # but do not overshoot nominal_target.
                    if cont_cfg.adaptive:
                        remaining = nominal_target - current_value
                        grown = cont_cfg.growth_factor * (current_value - (penultimate_value if penultimate_value is not None else first_value))
                        if abs(grown) <= 0.0:
                            target_value = nominal_target
                        else:
                            candidate = current_value + direction * min(abs(grown), abs(remaining))
                            target_value = candidate
                    # Do not advance target_index yet.
                break

            last_message = step_result.report.summary_line()

            if not cont_cfg.adaptive:
                reports.append(
                    _make_failed_report(
                        step_index=target_index,
                        target_value=target_value,
                        previous_value=current_value,
                        attempted_step_size=attempted_step,
                        retries=retries,
                        solver_report=step_result.report,
                        message="step failed and adaptive continuation is disabled",
                        metadata={"nominal_target_value": nominal_target},
                    )
                )
                break

            # Shrink step toward current_value.
            attempted = target_value - current_value
            target_value = current_value + cont_cfg.shrink_factor * attempted
            retries += 1

        if step_accepted:
            continue

        # Could not solve this target.
        if cont_cfg.stop_on_failure:
            status = ContinuationStatus.PARTIAL if accepted_values else ContinuationStatus.FAILED
            return ContinuationResult(
                values=tuple(accepted_values),
                solutions=tuple(accepted_solutions),
                residuals=tuple(accepted_residuals),
                step_reports=tuple(reports),
                status=status,
                metadata={
                    **dict(metadata or {}),
                    "continuation_config": cont_cfg.to_dict(),
                    "schedule": list(values_target),
                    "failed_target_index": target_index,
                    "failed_nominal_target": nominal_target,
                    "message": "continuation stopped after failed step",
                },
            )

        target_index += 1

    return ContinuationResult(
        values=tuple(accepted_values),
        solutions=tuple(accepted_solutions),
        residuals=tuple(accepted_residuals),
        step_reports=tuple(reports),
        status=ContinuationStatus.CONVERGED,
        metadata={
            **dict(metadata or {}),
            "continuation_config": cont_cfg.to_dict(),
            "schedule": list(values_target),
        },
    )


# ---------------------------------------------------------------------------
# Common convenience drivers
# ---------------------------------------------------------------------------

def solve_linear_schedule_continuation(
    *,
    start: float,
    stop: float,
    n_steps: int,
    residual_factory: ResidualFactory,
    x0: PyTree | None = None,
    initial_guess_factory: InitialGuessFactory | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    continuation_config: ContinuationSolverConfig | ContinuationConfig | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuationResult:
    """
    Build a linear schedule and run solve_continuation().
    """
    if isinstance(continuation_config, ContinuationSolverConfig):
        kind = continuation_config.schedule_kind
    else:
        kind = ContinuationScheduleKind.LINEAR

    schedule = make_continuation_schedule(
        start=start,
        stop=stop,
        n_steps=n_steps,
        kind=kind,
    )
    return solve_continuation(
        schedule=schedule,
        residual_factory=residual_factory,
        x0=x0,
        initial_guess_factory=initial_guess_factory,
        solver_config=solver_config,
        continuation_config=continuation_config,
        context=context,
        metadata=metadata,
    )


def solve_pump_power_continuation(
    *,
    start_power_dbm: float,
    stop_power_dbm: float,
    n_steps: int,
    residual_factory: ResidualFactory,
    x0: PyTree | None = None,
    initial_guess_factory: InitialGuessFactory | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    continuation_config: ContinuationSolverConfig | ContinuationConfig | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuationResult:
    """
    Pump-power continuation in dBm.

    The residual_factory receives the current pump power in dBm.
    """
    ctx = {
        **dict(context or {}),
        "continuation_parameter": "pump_power_dbm",
        "start_power_dbm": start_power_dbm,
        "stop_power_dbm": stop_power_dbm,
    }

    return solve_linear_schedule_continuation(
        start=start_power_dbm,
        stop=stop_power_dbm,
        n_steps=n_steps,
        residual_factory=residual_factory,
        x0=x0,
        initial_guess_factory=initial_guess_factory,
        solver_config=solver_config,
        continuation_config=continuation_config,
        context=ctx,
        metadata={
            **dict(metadata or {}),
            "continuation_parameter": "pump_power_dbm",
        },
    )


def solve_frequency_continuation(
    *,
    start_frequency_hz: float,
    stop_frequency_hz: float,
    n_steps: int,
    residual_factory: ResidualFactory,
    x0: PyTree | None = None,
    initial_guess_factory: InitialGuessFactory | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    continuation_config: ContinuationSolverConfig | ContinuationConfig | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuationResult:
    """
    Pump/frequency continuation in Hz.

    The residual_factory receives the current frequency in Hz.
    """
    ctx = {
        **dict(context or {}),
        "continuation_parameter": "frequency_hz",
        "start_frequency_hz": start_frequency_hz,
        "stop_frequency_hz": stop_frequency_hz,
    }

    return solve_linear_schedule_continuation(
        start=start_frequency_hz,
        stop=stop_frequency_hz,
        n_steps=n_steps,
        residual_factory=residual_factory,
        x0=x0,
        initial_guess_factory=initial_guess_factory,
        solver_config=solver_config,
        continuation_config=continuation_config,
        context=ctx,
        metadata={
            **dict(metadata or {}),
            "continuation_parameter": "frequency_hz",
        },
    )


def solve_nonlinearity_continuation(
    *,
    start_scale: float = 0.0,
    stop_scale: float = 1.0,
    n_steps: int = 11,
    residual_factory: ResidualFactory,
    x0: PyTree | None = None,
    initial_guess_factory: InitialGuessFactory | None = None,
    solver_config: DenseNewtonConfig | SolverConfig | None = None,
    continuation_config: ContinuationSolverConfig | ContinuationConfig | None = None,
    context: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContinuationResult:
    """
    Nonlinearity-strength continuation.

    The residual_factory receives scale in [start_scale, stop_scale].
    """
    ctx = {
        **dict(context or {}),
        "continuation_parameter": "nonlinearity_scale",
        "start_scale": start_scale,
        "stop_scale": stop_scale,
    }

    return solve_linear_schedule_continuation(
        start=start_scale,
        stop=stop_scale,
        n_steps=n_steps,
        residual_factory=residual_factory,
        x0=x0,
        initial_guess_factory=initial_guess_factory,
        solver_config=solver_config,
        continuation_config=continuation_config,
        context=ctx,
        metadata={
            **dict(metadata or {}),
            "continuation_parameter": "nonlinearity_scale",
        },
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def continuation_table(result: ContinuationResult) -> str:
    """
    Markdown table of continuation steps.
    """
    lines = [
        "| step | target | status | accepted | retries | final residual | message |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]

    for report in result.step_reports:
        final_res = (
            float("nan")
            if report.solver_report is None
            else report.solver_report.final_residual_norm
        )
        lines.append(
            f"| {report.step_index} | {report.target_value:.9g} | "
            f"{report.status.value} | {int(report.accepted)} | {report.retries} | "
            f"{final_res:.3e} | {report.message} |"
        )

    return "\n".join(lines)


def continuation_values_array(result: ContinuationResult) -> jax.Array:
    """
    Accepted values as a JAX array.
    """
    if not result.values:
        return jnp.asarray([], dtype=jnp.float64)
    return jnp.asarray(result.values, dtype=jnp.float64)


def final_solution_or_raise(result: ContinuationResult) -> PyTree:
    """
    Return final accepted solution or raise a helpful error.
    """
    if result.last_solution is None:
        raise RuntimeError(f"No accepted continuation solution. Status: {result.status.value}")
    return result.last_solution


def assert_continuation_converged(result: ContinuationResult) -> None:
    """
    Raise RuntimeError if continuation did not fully converge.
    """
    if not result.converged:
        raise RuntimeError(result.summary_line())


@dataclass(frozen=True)
class ContinuationStep:
    parameter: float
    x: Any
    success: bool = True
    residual_norm: float | None = None
    n_iter: int | None = None
    message: str = ""

    @property
    def state(self) -> Any:
        return self.x

    @property
    def solution(self) -> Any:
        return self.x

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "x": _compat_json(self.x),
            "success": self.success,
            "residual_norm": self.residual_norm,
            "n_iter": self.n_iter,
            "message": self.message,
        }


@dataclass(frozen=True)
class SimpleContinuationResult:
    x: Any
    success: bool
    history: tuple[ContinuationStep, ...]
    parameter_values: tuple[float, ...]
    message: str = ""

    @property
    def state(self) -> Any:
        return self.x

    @property
    def solution(self) -> Any:
        return self.x

    @property
    def final_state(self) -> Any:
        return self.x

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": _compat_json(self.x),
            "success": self.success,
            "history": [step.to_dict() for step in self.history],
            "parameter_values": list(self.parameter_values),
            "message": self.message,
        }


@dataclass(frozen=True)
class ContinuationRunner:
    """Object-oriented wrapper around compatibility continuation_solve()."""

    solve_at_parameter: Callable[..., Any]
    parameter_values: Sequence[float]
    x0: Any
    fail_fast: bool = True
    max_step_retries: int = 0

    def run(
        self,
        *,
        x0: Any | None = None,
        parameter_values: Sequence[float] | None = None,
        **_: Any,
    ) -> SimpleContinuationResult:
        return continuation_solve(
            self.solve_at_parameter,
            self.x0 if x0 is None else x0,
            self.parameter_values if parameter_values is None else parameter_values,
            fail_fast=self.fail_fast,
            max_step_retries=self.max_step_retries,
        )

    solve = run
    execute = run


def _compat_json(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(jnp.real(value)), "imag": float(jnp.imag(value))}
    arr = None
    try:
        arr = jnp.asarray(value)
    except Exception:
        pass
    if arr is not None and arr.ndim >= 0 and hasattr(value, "__array__"):
        if jnp.iscomplexobj(arr):
            return {
                "real": jnp.real(arr).tolist(),
                "imag": jnp.imag(arr).tolist(),
            }
        return arr.tolist()
    if isinstance(value, Mapping):
        return {str(k): _compat_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compat_json(v) for v in value]
    return value


def _solver_state(result: Any) -> Any:
    if isinstance(result, Mapping):
        for key in ("x", "state", "solution", "final_state"):
            if key in result:
                return result[key]
        return result
    for key in ("x", "state", "solution", "final_state"):
        if hasattr(result, key):
            return getattr(result, key)
    return result


def _solver_success(result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("success", result.get("converged", True)))
    for key in ("success", "converged"):
        if hasattr(result, key):
            return bool(getattr(result, key))
    return True


def continuation_solve(
    solve_at_parameter: Callable[..., Any],
    x0: Any,
    parameter_values: Sequence[float],
    *,
    fail_fast: bool = True,
    max_step_retries: int = 0,
) -> SimpleContinuationResult:
    """Compatibility continuation wrapper: visit parameters in caller order."""
    if solve_at_parameter is None or not callable(solve_at_parameter):
        raise TypeError("solve_at_parameter must be callable")
    values = jnp.asarray(parameter_values, dtype=jnp.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("parameter_values must be a non-empty 1D sequence")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("parameter_values must be finite")
    current = jnp.asarray(x0)
    if bool(jnp.any(~jnp.isfinite(current))):
        raise FloatingPointError("initial state contains nonfinite values")

    history: list[ContinuationStep] = []
    message = ""
    success = True
    for raw in values.tolist():
        parameter = float(raw)
        attempt = 0
        while True:
            try:
                out = solve_at_parameter(
                    current,
                    parameter=parameter,
                    parameter_value=parameter,
                    lambda_value=parameter,
                    alpha=parameter,
                )
                step_success = _solver_success(out)
                next_state = _solver_state(out)
                step = ContinuationStep(
                    parameter=parameter,
                    x=next_state,
                    success=step_success,
                    residual_norm=(
                        float(out["residual_norm"])
                        if isinstance(out, Mapping) and "residual_norm" in out
                        else None
                    ),
                    n_iter=(
                        int(out["n_iter"])
                        if isinstance(out, Mapping) and "n_iter" in out
                        else None
                    ),
                    message=(
                        str(out.get("message", ""))
                        if isinstance(out, Mapping)
                        else ""
                    ),
                )
                history.append(step)
                if not step_success:
                    success = False
                    message = step.message or f"step failed at parameter {parameter}"
                    if fail_fast:
                        return SimpleContinuationResult(
                            x=current,
                            success=False,
                            history=tuple(history),
                            parameter_values=tuple(float(v) for v in values.tolist()),
                            message=message,
                        )
                else:
                    current = jnp.asarray(next_state)
                break
            except Exception as exc:
                if attempt < max_step_retries:
                    attempt += 1
                    continue
                if fail_fast:
                    raise
                success = False
                message = str(exc)
                history.append(
                    ContinuationStep(parameter=parameter, x=current, success=False, message=message)
                )
                return SimpleContinuationResult(
                    x=current,
                    success=False,
                    history=tuple(history),
                    parameter_values=tuple(float(v) for v in values.tolist()),
                    message=message,
                )
    return SimpleContinuationResult(
        x=current,
        success=success,
        history=tuple(history),
        parameter_values=tuple(float(v) for v in values.tolist()),
        message=message,
    )


make_continuation_grid = linear_schedule
continuation_grid = linear_schedule
parameter_grid = linear_schedule
linear_parameter_grid = linear_schedule
make_parameter_schedule = linear_schedule


__all__ = [
    "ArrayLike",
    "PyTree",
    "ResidualFactory",
    "InitialGuessFactory",
    "PredictorFn",
    "AcceptFn",
    "ContinuationStatus",
    "StepStatus",
    "ContinuationScheduleKind",
    "ContinuationSolverConfig",
    "ContinuationStepReport",
    "ContinuationResult",
    "linear_schedule",
    "geometric_schedule",
    "make_continuation_schedule",
    "schedule_from_core_config",
    "validate_schedule",
    "identity_predictor",
    "secant_predictor",
    "choose_predictor",
    "default_accept",
    "residual_threshold_accept",
    "solve_continuation",
    "solve_linear_schedule_continuation",
    "solve_pump_power_continuation",
    "solve_frequency_continuation",
    "solve_nonlinearity_continuation",
    "continuation_table",
    "continuation_values_array",
    "final_solution_or_raise",
    "assert_continuation_converged",
    "ContinuationStep",
    "SimpleContinuationResult",
    "ContinuationRunner",
    "continuation_solve",
    "make_continuation_grid",
    "continuation_grid",
    "parameter_grid",
    "linear_parameter_grid",
    "make_parameter_schedule",
]
