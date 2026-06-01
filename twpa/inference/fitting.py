"""
twpa.inference.fitting
======================

Parameter-fitting orchestration for TWPA inference workflows.

This module provides a lightweight optimizer-facing layer around:

    - PriorSet / ParameterPrior objects,
    - synthetic datasets,
    - calibration objective functions,
    - custom residual functions.

It is intentionally higher-level and more experiment-oriented than
``twpa.workflows.calibration``. The calibration workflow is the production
deterministic engine; this module adds:

    - random restarts,
    - prior-aware initialization,
    - generic residual/objective wrapping,
    - simple derivative-free fallback optimizers,
    - synthetic recovery convenience helpers.

Typical use
-----------
Define a residual function from physical parameters to a residual vector:

    def residual_fn(params):
        ...
        return residual_vector

Then run:

    result = run_parameter_fit(
        residual_fn,
        prior_set,
        config=FitConfig(method=FitOptimizerMethod.AUTO),
    )

For calibration targets, use:

    residual_fn = make_calibration_residual_function(
        target,
        sparameter_data=sdata,
        gain_data=gdata,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import json
import math
import time
import numpy as np

import jax
import jax.numpy as jnp

from twpa.inference.priors import ParameterSample, PriorSet


try:
    import scipy.optimize as scipy_opt

    SCIPY_AVAILABLE = True
except Exception:
    scipy_opt = None
    SCIPY_AVAILABLE = False


ArrayLike = Any
ParameterDict = Mapping[str, float]
ResidualFunction = Callable[[dict[str, float]], ArrayLike]
ScalarObjectiveFunction = Callable[[dict[str, float]], float]


class FitStatus(str, Enum):
    """Status of a parameter fit."""

    SUCCESS = "success"
    FAILED = "failed"
    MAX_EVALUATIONS = "max_evaluations"
    NO_IMPROVEMENT = "no_improvement"
    INVALID_INITIALIZATION = "invalid_initialization"
    ERROR = "error"


class FitOptimizerMethod(str, Enum):
    """Supported optimizers."""

    AUTO = "auto"
    SCIPY_LEAST_SQUARES = "scipy_least_squares"
    SCIPY_MINIMIZE = "scipy_minimize"
    COORDINATE_SEARCH = "coordinate_search"
    RANDOM_SEARCH = "random_search"


class FitLossKind(str, Enum):
    """Loss function applied to residual vectors."""

    HALF_SQUARED_L2 = "half_squared_l2"
    MEAN_SQUARED = "mean_squared"
    ROOT_MEAN_SQUARED = "root_mean_squared"
    L1 = "l1"
    HUBER = "huber"


@dataclass(frozen=True)
class FitConfig:
    """
    Fitting configuration.

    Parameters
    ----------
    method:
        Optimizer method. AUTO selects scipy_least_squares if SciPy is present,
        otherwise coordinate_search.
    max_evaluations:
        Maximum objective/residual evaluations.
    n_restarts:
        Number of restarts including the initial/prior median start.
    random_seed:
        Random seed used for restarts and random search.
    loss_kind:
        Scalar loss used for derivative-free methods and reporting.
    huber_delta:
        Huber transition scale if loss_kind=HUBER.
    include_log_prior_penalty:
        Whether to add ``-log_prior_weight * log_prior`` to scalar loss.
    log_prior_weight:
        Weight for prior penalty.
    initial_values:
        Optional physical-space initial values overriding prior initials.
    use_prior_samples_for_restarts:
        If True, restarts after the first are drawn from the prior set.
    coordinate_initial_step_fraction:
        Initial coordinate step as fraction of encoded prior bounds or 1.0 for
        unbounded coordinates.
    coordinate_step_decay:
        Coordinate-search step shrink factor.
    coordinate_min_step_fraction:
        Stop coordinate search once step scales fall below this value.
    xtol, ftol, gtol:
        Tolerances passed to SciPy when available and used by fallbacks where
        applicable.
    verbose:
        Print optimizer diagnostics.
    name:
        Diagnostic name.
    """

    method: FitOptimizerMethod = FitOptimizerMethod.AUTO
    max_evaluations: int = 200
    n_restarts: int = 1
    random_seed: int = 1234
    loss_kind: FitLossKind = FitLossKind.HALF_SQUARED_L2
    huber_delta: float = 1.0
    include_log_prior_penalty: bool = False
    log_prior_weight: float = 1.0
    initial_values: Mapping[str, float] | None = None
    use_prior_samples_for_restarts: bool = True
    coordinate_initial_step_fraction: float = 0.10
    coordinate_step_decay: float = 0.5
    coordinate_min_step_fraction: float = 1e-4
    xtol: float = 1e-8
    ftol: float = 1e-8
    gtol: float = 1e-8
    verbose: bool = False
    name: str = "parameter_fit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", FitOptimizerMethod(self.method))
        object.__setattr__(self, "loss_kind", FitLossKind(self.loss_kind))

        if int(self.max_evaluations) <= 0:
            raise ValueError("max_evaluations must be positive")
        if int(self.n_restarts) <= 0:
            raise ValueError("n_restarts must be positive")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if self.log_prior_weight < 0.0:
            raise ValueError("log_prior_weight must be non-negative")
        if self.coordinate_initial_step_fraction <= 0.0:
            raise ValueError("coordinate_initial_step_fraction must be positive")
        if not (0.0 < self.coordinate_step_decay < 1.0):
            raise ValueError("coordinate_step_decay must be in (0, 1)")
        if self.coordinate_min_step_fraction <= 0.0:
            raise ValueError("coordinate_min_step_fraction must be positive")

        object.__setattr__(self, "max_evaluations", int(self.max_evaluations))
        object.__setattr__(self, "n_restarts", int(self.n_restarts))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(
            self,
            "initial_values",
            None if self.initial_values is None else dict(self.initial_values),
        )

    def selected_method(self) -> FitOptimizerMethod:
        if self.method != FitOptimizerMethod.AUTO:
            return self.method
        return (
            FitOptimizerMethod.SCIPY_LEAST_SQUARES
            if SCIPY_AVAILABLE
            else FitOptimizerMethod.COORDINATE_SEARCH
        )

    def with_updates(self, **kwargs: Any) -> "FitConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "selected_method": self.selected_method().value,
            "max_evaluations": self.max_evaluations,
            "n_restarts": self.n_restarts,
            "random_seed": self.random_seed,
            "loss_kind": self.loss_kind.value,
            "huber_delta": self.huber_delta,
            "include_log_prior_penalty": self.include_log_prior_penalty,
            "log_prior_weight": self.log_prior_weight,
            "initial_values": dict(self.initial_values or {}),
            "use_prior_samples_for_restarts": self.use_prior_samples_for_restarts,
            "coordinate_initial_step_fraction": self.coordinate_initial_step_fraction,
            "coordinate_step_decay": self.coordinate_step_decay,
            "coordinate_min_step_fraction": self.coordinate_min_step_fraction,
            "xtol": self.xtol,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "verbose": self.verbose,
            "scipy_available": SCIPY_AVAILABLE,
            "name": self.name,
        }


@dataclass(frozen=True)
class FitEvaluation:
    """
    One objective evaluation.
    """

    parameters: Mapping[str, float]
    encoded_vector: jax.Array
    residual: jax.Array
    loss: float
    log_prior: float
    penalized_loss: float
    metadata: Mapping[str, Any] | None = None

    @property
    def residual_norm(self) -> float:
        return float(jnp.linalg.norm(self.residual))

    @property
    def n_residuals(self) -> int:
        return int(self.residual.size)

    def to_dict(self, *, include_residual: bool = False) -> dict[str, Any]:
        out = {
            "parameters": {str(k): float(v) for k, v in self.parameters.items()},
            "encoded_vector": np.asarray(self.encoded_vector).tolist(),
            "loss": self.loss,
            "log_prior": self.log_prior,
            "penalized_loss": self.penalized_loss,
            "residual_norm": self.residual_norm,
            "n_residuals": self.n_residuals,
            "metadata": dict(self.metadata or {}),
        }
        if include_residual:
            out["residual"] = np.asarray(self.residual).tolist()
        return out


@dataclass(frozen=True)
class FitIterationRecord:
    """
    Iteration/evaluation record for fallback optimizers.
    """

    evaluation_index: int
    restart_index: int
    loss: float
    penalized_loss: float
    residual_norm: float
    accepted: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_index": self.evaluation_index,
            "restart_index": self.restart_index,
            "loss": self.loss,
            "penalized_loss": self.penalized_loss,
            "residual_norm": self.residual_norm,
            "accepted": self.accepted,
            "message": self.message,
        }


@dataclass(frozen=True)
class FitResult:
    """
    Result of a parameter fit.
    """

    status: FitStatus
    success: bool
    best_evaluation: FitEvaluation
    prior_set: PriorSet
    config: FitConfig
    n_evaluations: int
    elapsed_s: float
    records: tuple[FitIterationRecord, ...]
    optimizer_message: str = ""
    optimizer_metadata: Mapping[str, Any] | None = None

    @property
    def best_parameters(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.best_evaluation.parameters.items()}

    @property
    def best_encoded_vector(self) -> jax.Array:
        return self.best_evaluation.encoded_vector

    @property
    def loss(self) -> float:
        return self.best_evaluation.loss

    @property
    def residual_norm(self) -> float:
        return self.best_evaluation.residual_norm

    def summary_line(self) -> str:
        return (
            f"FitResult(status={self.status.value}, success={self.success}, "
            f"loss={self.loss:.6e}, residual_norm={self.residual_norm:.6e}, "
            f"n_evaluations={self.n_evaluations})"
        )

    def to_dict(self, *, include_residual: bool = False) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "success": self.success,
            "best_evaluation": self.best_evaluation.to_dict(include_residual=include_residual),
            "best_parameters": self.best_parameters,
            "prior_set": self.prior_set.to_dict(),
            "config": self.config.to_dict(),
            "n_evaluations": self.n_evaluations,
            "elapsed_s": self.elapsed_s,
            "records": [r.to_dict() for r in self.records],
            "optimizer_message": self.optimizer_message,
            "optimizer_metadata": dict(self.optimizer_metadata or {}),
        }


def residual_to_loss(
    residual: ArrayLike,
    *,
    loss_kind: FitLossKind = FitLossKind.HALF_SQUARED_L2,
    huber_delta: float = 1.0,
) -> float:
    """
    Convert a residual vector to scalar loss.
    """
    r = jnp.ravel(jnp.asarray(residual, dtype=jnp.float64))
    loss_kind = FitLossKind(loss_kind)

    if r.size == 0:
        return 0.0

    if loss_kind == FitLossKind.HALF_SQUARED_L2:
        return float(0.5 * jnp.sum(r * r))

    if loss_kind == FitLossKind.MEAN_SQUARED:
        return float(jnp.mean(r * r))

    if loss_kind == FitLossKind.ROOT_MEAN_SQUARED:
        return float(jnp.sqrt(jnp.mean(r * r)))

    if loss_kind == FitLossKind.L1:
        return float(jnp.sum(jnp.abs(r)))

    if loss_kind == FitLossKind.HUBER:
        delta = float(huber_delta)
        abs_r = jnp.abs(r)
        quad = jnp.minimum(abs_r, delta)
        lin = abs_r - quad
        return float(jnp.sum(0.5 * quad * quad + delta * lin))

    raise ValueError(f"Unsupported loss kind {loss_kind}")


class _EvaluationCounter:
    """
    Shared objective evaluation counter and best-state tracker.
    """

    def __init__(
        self,
        residual_fn: ResidualFunction,
        prior_set: PriorSet,
        config: FitConfig,
    ) -> None:
        self.residual_fn = residual_fn
        self.prior_set = prior_set
        self.config = config
        self.n_evaluations = 0
        self.best: FitEvaluation | None = None
        self.records: list[FitIterationRecord] = []
        self.current_restart = 0

    def evaluate_vector(self, vector: ArrayLike, *, message: str = "") -> FitEvaluation:
        z = jnp.asarray(vector, dtype=jnp.float64)
        params = self.prior_set.decode_vector(z)
        return self.evaluate_parameters(params, encoded_vector=z, message=message)

    def evaluate_parameters(
        self,
        parameters: Mapping[str, float],
        *,
        encoded_vector: ArrayLike | None = None,
        message: str = "",
    ) -> FitEvaluation:
        self.n_evaluations += 1

        params = {str(k): float(v) for k, v in parameters.items()}
        z = (
            self.prior_set.encode_values(params)
            if encoded_vector is None
            else jnp.asarray(encoded_vector, dtype=jnp.float64)
        )

        try:
            residual = jnp.ravel(jnp.asarray(self.residual_fn(params), dtype=jnp.float64))
            if not bool(jnp.all(jnp.isfinite(residual))):
                residual = jnp.full_like(residual, 1e30)
                eval_message = "non-finite residual replaced by large residual"
            else:
                eval_message = message
        except Exception as exc:
            residual = jnp.asarray([1e30], dtype=jnp.float64)
            eval_message = f"residual function raised: {exc}"

        loss = residual_to_loss(
            residual,
            loss_kind=self.config.loss_kind,
            huber_delta=self.config.huber_delta,
        )

        log_prior = self.prior_set.log_prob(params)
        if not math.isfinite(log_prior):
            log_prior = -1e300

        penalized = loss
        if self.config.include_log_prior_penalty:
            penalized = loss - self.config.log_prior_weight * log_prior

        evaluation = FitEvaluation(
            parameters=params,
            encoded_vector=z,
            residual=residual,
            loss=float(loss),
            log_prior=float(log_prior),
            penalized_loss=float(penalized),
            metadata={
                "evaluation_index": self.n_evaluations,
                "message": eval_message,
            },
        )

        accepted = False
        if self.best is None or evaluation.penalized_loss < self.best.penalized_loss:
            self.best = evaluation
            accepted = True

        self.records.append(
            FitIterationRecord(
                evaluation_index=self.n_evaluations,
                restart_index=self.current_restart,
                loss=evaluation.loss,
                penalized_loss=evaluation.penalized_loss,
                residual_norm=evaluation.residual_norm,
                accepted=accepted,
                message=eval_message,
            )
        )

        if self.config.verbose:
            print(
                f"[fit] eval={self.n_evaluations} "
                f"restart={self.current_restart} "
                f"loss={evaluation.loss:.6e} "
                f"penalized={evaluation.penalized_loss:.6e} "
                f"residual={evaluation.residual_norm:.6e}"
                + (" *" if accepted else "")
            )

        return evaluation


def _initial_vectors(prior_set: PriorSet, config: FitConfig) -> list[jax.Array]:
    vectors: list[jax.Array] = []

    if config.initial_values is not None:
        values = prior_set.initial_values(include_disabled=True)
        values.update({str(k): float(v) for k, v in config.initial_values.items()})
        vectors.append(prior_set.encode_values(values))
    else:
        vectors.append(prior_set.initial_vector())

    if config.n_restarts > 1:
        rng = np.random.default_rng(config.random_seed)
        for _ in range(config.n_restarts - 1):
            if config.use_prior_samples_for_restarts:
                sample = prior_set.sample(rng=rng)
                vectors.append(sample.encoded_vector)
            else:
                lo, hi = prior_set.bounds_encoded()
                lo_np = np.asarray(lo, dtype=float)
                hi_np = np.asarray(hi, dtype=float)
                z = []
                for a, b, init in zip(lo_np, hi_np, np.asarray(prior_set.initial_vector())):
                    if np.isfinite(a) and np.isfinite(b):
                        z.append(rng.uniform(a, b))
                    else:
                        z.append(init + rng.normal(0.0, 1.0))
                vectors.append(jnp.asarray(z, dtype=jnp.float64))

    return vectors


def _scipy_least_squares(
    counter: _EvaluationCounter,
    initial_vectors: Sequence[jax.Array],
) -> tuple[FitStatus, str, dict[str, Any]]:
    if not SCIPY_AVAILABLE or scipy_opt is None:
        return FitStatus.FAILED, "SciPy unavailable", {}

    best_result = None
    total_budget = counter.config.max_evaluations

    for restart_idx, z0 in enumerate(initial_vectors):
        counter.current_restart = restart_idx
        remaining = max(1, total_budget - counter.n_evaluations)

        def fun(z_np: np.ndarray) -> np.ndarray:
            ev = counter.evaluate_vector(z_np, message="scipy least_squares")
            return np.asarray(ev.residual, dtype=float)

        lo, hi = counter.prior_set.bounds_encoded()
        lo_np = np.asarray(lo, dtype=float)
        hi_np = np.asarray(hi, dtype=float)

        try:
            result = scipy_opt.least_squares(
                fun,
                np.asarray(z0, dtype=float),
                bounds=(lo_np, hi_np),
                max_nfev=remaining,
                xtol=counter.config.xtol,
                ftol=counter.config.ftol,
                gtol=counter.config.gtol,
            )
            best_result = result
        except Exception as exc:
            return FitStatus.ERROR, f"SciPy least_squares raised: {exc}", {
                "restart_index": restart_idx,
            }

        if counter.n_evaluations >= total_budget:
            break

    success = best_result is not None and bool(getattr(best_result, "success", False))
    status = FitStatus.SUCCESS if success else FitStatus.MAX_EVALUATIONS
    message = "SciPy least_squares completed" if success else "SciPy least_squares stopped without success"

    metadata = {}
    if best_result is not None:
        metadata = {
            "scipy_success": bool(best_result.success),
            "scipy_status": int(best_result.status),
            "scipy_message": str(best_result.message),
            "scipy_cost": float(best_result.cost),
            "scipy_optimality": float(best_result.optimality),
            "scipy_nfev": int(best_result.nfev),
        }

    return status, message, metadata


def _scipy_minimize(
    counter: _EvaluationCounter,
    initial_vectors: Sequence[jax.Array],
) -> tuple[FitStatus, str, dict[str, Any]]:
    if not SCIPY_AVAILABLE or scipy_opt is None:
        return FitStatus.FAILED, "SciPy unavailable", {}

    best_result = None
    total_budget = counter.config.max_evaluations

    lo, hi = counter.prior_set.bounds_encoded()
    bounds = list(zip(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)))

    for restart_idx, z0 in enumerate(initial_vectors):
        counter.current_restart = restart_idx
        remaining = max(1, total_budget - counter.n_evaluations)

        def fun(z_np: np.ndarray) -> float:
            ev = counter.evaluate_vector(z_np, message="scipy minimize")
            return ev.penalized_loss

        try:
            result = scipy_opt.minimize(
                fun,
                np.asarray(z0, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxfun": remaining,
                    "ftol": counter.config.ftol,
                    "gtol": counter.config.gtol,
                    "maxiter": remaining,
                },
            )
            best_result = result
        except Exception as exc:
            return FitStatus.ERROR, f"SciPy minimize raised: {exc}", {
                "restart_index": restart_idx,
            }

        if counter.n_evaluations >= total_budget:
            break

    success = best_result is not None and bool(getattr(best_result, "success", False))
    status = FitStatus.SUCCESS if success else FitStatus.MAX_EVALUATIONS
    message = "SciPy minimize completed" if success else "SciPy minimize stopped without success"

    metadata = {}
    if best_result is not None:
        metadata = {
            "scipy_success": bool(best_result.success),
            "scipy_status": int(best_result.status),
            "scipy_message": str(best_result.message),
            "scipy_fun": float(best_result.fun),
            "scipy_nfev": int(best_result.nfev),
        }

    return status, message, metadata


def _coordinate_step_scales(prior_set: PriorSet, config: FitConfig) -> jax.Array:
    lo, hi = prior_set.bounds_encoded()
    lo_np = np.asarray(lo, dtype=float)
    hi_np = np.asarray(hi, dtype=float)

    scales = []
    for a, b in zip(lo_np, hi_np):
        if np.isfinite(a) and np.isfinite(b):
            scales.append(config.coordinate_initial_step_fraction * max(b - a, 1e-12))
        else:
            scales.append(config.coordinate_initial_step_fraction)

    return jnp.asarray(scales, dtype=jnp.float64)


def _coordinate_search(
    counter: _EvaluationCounter,
    initial_vectors: Sequence[jax.Array],
) -> tuple[FitStatus, str, dict[str, Any]]:
    total_budget = counter.config.max_evaluations
    min_step_scale = counter.config.coordinate_min_step_fraction

    global_best = None

    for restart_idx, z0 in enumerate(initial_vectors):
        counter.current_restart = restart_idx

        z = jnp.asarray(z0, dtype=jnp.float64)
        current = counter.evaluate_vector(z, message="coordinate initial")
        if global_best is None or current.penalized_loss < global_best.penalized_loss:
            global_best = current

        step = _coordinate_step_scales(counter.prior_set, counter.config)

        while counter.n_evaluations < total_budget:
            improved = False

            for dim in range(counter.prior_set.ndim):
                if counter.n_evaluations >= total_budget:
                    break

                candidates = [
                    z.at[dim].add(step[dim]),
                    z.at[dim].add(-step[dim]),
                ]

                for cand in candidates:
                    if counter.n_evaluations >= total_budget:
                        break

                    ev = counter.evaluate_vector(cand, message=f"coordinate dim {dim}")
                    if ev.penalized_loss < current.penalized_loss:
                        current = ev
                        z = ev.encoded_vector
                        improved = True

            if current.penalized_loss < global_best.penalized_loss:
                global_best = current

            if not improved:
                step = step * counter.config.coordinate_step_decay

            if float(jnp.max(jnp.abs(step))) < min_step_scale:
                return FitStatus.SUCCESS, "coordinate search reached minimum step scale", {
                    "restart_index": restart_idx,
                    "final_max_step": float(jnp.max(jnp.abs(step))),
                }

            if counter.n_evaluations >= total_budget:
                break

    return FitStatus.MAX_EVALUATIONS, "coordinate search reached max evaluations", {}


def _random_search(
    counter: _EvaluationCounter,
    initial_vectors: Sequence[jax.Array],
) -> tuple[FitStatus, str, dict[str, Any]]:
    rng = np.random.default_rng(counter.config.random_seed)

    for restart_idx, z0 in enumerate(initial_vectors):
        counter.current_restart = restart_idx
        counter.evaluate_vector(z0, message="random search initial")

    while counter.n_evaluations < counter.config.max_evaluations:
        counter.current_restart = min(counter.current_restart + 1, counter.config.n_restarts - 1)
        sample = counter.prior_set.sample(rng=rng)
        counter.evaluate_parameters(sample.values, encoded_vector=sample.encoded_vector, message="random prior sample")

    return FitStatus.MAX_EVALUATIONS, "random search reached max evaluations", {}


def run_parameter_fit(
    residual_fn: ResidualFunction,
    prior_set: PriorSet,
    *,
    config: FitConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FitResult:
    """
    Fit parameters by minimizing a residual function.

    Parameters
    ----------
    residual_fn:
        Function mapping physical parameter dictionary to a residual vector.
    prior_set:
        Prior/bounds/transform collection.
    config:
        Fit configuration.
    metadata:
        Optional metadata stored in the result.
    """
    cfg = config or FitConfig()
    start = time.perf_counter()

    if prior_set.ndim <= 0:
        raise ValueError("prior_set has no enabled parameters")

    counter = _EvaluationCounter(residual_fn, prior_set, cfg)
    initial_vectors = _initial_vectors(prior_set, cfg)

    method = cfg.selected_method()

    try:
        if method == FitOptimizerMethod.SCIPY_LEAST_SQUARES:
            status, message, optimizer_metadata = _scipy_least_squares(counter, initial_vectors)

        elif method == FitOptimizerMethod.SCIPY_MINIMIZE:
            status, message, optimizer_metadata = _scipy_minimize(counter, initial_vectors)

        elif method == FitOptimizerMethod.COORDINATE_SEARCH:
            status, message, optimizer_metadata = _coordinate_search(counter, initial_vectors)

        elif method == FitOptimizerMethod.RANDOM_SEARCH:
            status, message, optimizer_metadata = _random_search(counter, initial_vectors)

        else:
            raise ValueError(f"Unsupported fit optimizer method {method}")

    except Exception as exc:
        status = FitStatus.ERROR
        message = f"fit failed: {exc}"
        optimizer_metadata = {"error": str(exc)}

    elapsed = time.perf_counter() - start

    if counter.best is None:
        fallback_params = prior_set.initial_values(include_disabled=True)
        fallback_z = prior_set.encode_values(fallback_params)
        fallback_residual = jnp.asarray([1e30], dtype=jnp.float64)
        best = FitEvaluation(
            parameters=fallback_params,
            encoded_vector=fallback_z,
            residual=fallback_residual,
            loss=float(0.5 * jnp.sum(fallback_residual * fallback_residual)),
            log_prior=prior_set.log_prob(fallback_params),
            penalized_loss=float(0.5 * jnp.sum(fallback_residual * fallback_residual)),
            metadata={"fallback": True},
        )
        status = FitStatus.INVALID_INITIALIZATION
        message = "no valid objective evaluations were produced"
    else:
        best = counter.best

    success = status == FitStatus.SUCCESS

    if cfg.verbose:
        print("[fit]", message)
        print("[fit]", f"best loss={best.loss:.6e}, residual={best.residual_norm:.6e}")

    return FitResult(
        status=status,
        success=success,
        best_evaluation=best,
        prior_set=prior_set,
        config=cfg,
        n_evaluations=counter.n_evaluations,
        elapsed_s=elapsed,
        records=tuple(counter.records),
        optimizer_message=message,
        optimizer_metadata={
            **dict(optimizer_metadata or {}),
            **dict(metadata or {}),
        },
    )


def make_calibration_residual_function(
    target: Any,
    *,
    sparameter_data: Any | None = None,
    gain_data: Any | None = None,
) -> ResidualFunction:
    """
    Create a residual function from ``twpa.workflows.calibration`` objects.

    The function expects physical parameters and returns the calibration
    residual vector. It tries the production calibration API first and falls
    back to an evaluation object if needed.
    """
    from twpa.workflows import calibration as cal

    def residual_fn(parameters: dict[str, float]) -> jax.Array:
        if hasattr(cal, "calibration_residual_vector"):
            return jnp.ravel(
                jnp.asarray(
                    cal.calibration_residual_vector(
                        target,
                        parameters,
                        sparameter_data=sparameter_data,
                        gain_data=gain_data,
                    ),
                    dtype=jnp.float64,
                )
            )

        if hasattr(cal, "evaluate_calibration_objective"):
            evaluation = cal.evaluate_calibration_objective(
                target,
                parameters,
                sparameter_data=sparameter_data,
                gain_data=gain_data,
            )

            for attr in ["residual", "residual_vector", "combined_residual"]:
                if hasattr(evaluation, attr):
                    return jnp.ravel(jnp.asarray(getattr(evaluation, attr), dtype=jnp.float64))

            if hasattr(evaluation, "to_dict"):
                d = evaluation.to_dict()
                for key in ["residual", "residual_vector", "combined_residual"]:
                    if key in d:
                        return jnp.ravel(jnp.asarray(d[key], dtype=jnp.float64))
                if "loss" in d:
                    return jnp.asarray([math.sqrt(max(2.0 * float(d["loss"]), 0.0))], dtype=jnp.float64)

            if hasattr(evaluation, "loss"):
                return jnp.asarray([math.sqrt(max(2.0 * float(evaluation.loss), 0.0))], dtype=jnp.float64)

        raise RuntimeError(
            "Could not build calibration residual. Expected "
            "twpa.workflows.calibration.calibration_residual_vector or "
            "evaluate_calibration_objective."
        )

    return residual_fn


def run_calibration_fit(
    target: Any,
    prior_set: PriorSet,
    *,
    sparameter_data: Any | None = None,
    gain_data: Any | None = None,
    config: FitConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FitResult:
    """
    Fit a calibration target using the inference optimizer layer.
    """
    residual_fn = make_calibration_residual_function(
        target,
        sparameter_data=sparameter_data,
        gain_data=gain_data,
    )
    return run_parameter_fit(
        residual_fn,
        prior_set,
        config=config,
        metadata={
            "source": "run_calibration_fit",
            **dict(metadata or {}),
        },
    )


def fit_result_markdown(result: FitResult) -> str:
    """
    Markdown summary for a FitResult.
    """
    lines = [
        "# Parameter fit summary",
        "",
        f"- status: `{result.status.value}`",
        f"- success: `{result.success}`",
        f"- method: `{result.config.selected_method().value}`",
        f"- evaluations: `{result.n_evaluations}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        f"- loss: `{result.loss:.9e}`",
        f"- residual norm: `{result.residual_norm:.9e}`",
        "",
        "## Best parameters",
        "",
        "| parameter | value |",
        "|---|---:|",
    ]

    for name, value in result.best_parameters.items():
        lines.append(f"| `{name}` | `{value:.12g}` |")

    lines += [
        "",
        "## Active priors",
        "",
        result.prior_set.active_table(),
        "",
        "## Optimizer message",
        "",
        result.optimizer_message,
    ]

    return "\n".join(lines)


def export_fit_artifacts(
    result: FitResult,
    output_dir: str | Path,
    *,
    prefix: str = "fit",
    include_residual: bool = True,
) -> dict[str, str]:
    """
    Export fit result JSON and Markdown artifacts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{prefix}_result.json"
    md_path = out / f"{prefix}_summary.md"
    npz_path = out / f"{prefix}_arrays.npz"

    json_path.write_text(
        json.dumps(result.to_dict(include_residual=include_residual), indent=2),
        encoding="utf-8",
    )
    md_path.write_text(fit_result_markdown(result), encoding="utf-8")

    np.savez_compressed(
        npz_path,
        best_encoded_vector=np.asarray(result.best_encoded_vector),
        best_residual=np.asarray(result.best_evaluation.residual),
        records_loss=np.asarray([r.loss for r in result.records], dtype=float),
        records_penalized_loss=np.asarray([r.penalized_loss for r in result.records], dtype=float),
        records_residual_norm=np.asarray([r.residual_norm for r in result.records], dtype=float),
        records_accepted=np.asarray([r.accepted for r in result.records], dtype=bool),
    )

    return {
        "result_json": str(json_path),
        "summary_md": str(md_path),
        "arrays_npz": str(npz_path),
    }


def compare_fit_to_truth(
    result: FitResult,
    true_parameters: Mapping[str, float],
) -> dict[str, Any]:
    """
    Compare fitted parameters to known truth.
    """
    truth = {str(k): float(v) for k, v in true_parameters.items()}
    fitted = result.best_parameters

    rows = []
    for name, true_value in truth.items():
        fit_value = fitted.get(name)
        if fit_value is None:
            rows.append(
                {
                    "name": name,
                    "true": true_value,
                    "fit": None,
                    "absolute_error": None,
                    "relative_error": None,
                }
            )
            continue

        abs_err = fit_value - true_value
        rel_err = abs_err / true_value if abs(true_value) > 1e-300 else None

        rows.append(
            {
                "name": name,
                "true": true_value,
                "fit": fit_value,
                "absolute_error": abs_err,
                "relative_error": rel_err,
            }
        )

    finite_rel = [
        abs(row["relative_error"])
        for row in rows
        if row["relative_error"] is not None and math.isfinite(row["relative_error"])
    ]

    return {
        "n_parameters": len(rows),
        "max_abs_relative_error": max(finite_rel) if finite_rel else None,
        "rows": rows,
        "fit_status": result.status.value,
        "fit_success": result.success,
        "fit_loss": result.loss,
    }


def truth_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """
    Markdown table for compare_fit_to_truth output.
    """
    lines = [
        "# Fit versus truth",
        "",
        f"- parameters: `{comparison.get('n_parameters')}`",
        f"- max abs relative error: `{comparison.get('max_abs_relative_error')}`",
        "",
        "| parameter | truth | fit | absolute error | relative error |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in comparison.get("rows", []):
        lines.append(
            f"| `{row['name']}` | "
            f"{row['true']} | "
            f"{'' if row['fit'] is None else row['fit']} | "
            f"{'' if row['absolute_error'] is None else row['absolute_error']} | "
            f"{'' if row['relative_error'] is None else row['relative_error']} |"
        )

    return "\n".join(lines)


__all__ = [
    "SCIPY_AVAILABLE",
    "ArrayLike",
    "ParameterDict",
    "ResidualFunction",
    "ScalarObjectiveFunction",
    "FitStatus",
    "FitOptimizerMethod",
    "FitLossKind",
    "FitConfig",
    "FitEvaluation",
    "FitIterationRecord",
    "FitResult",
    "residual_to_loss",
    "run_parameter_fit",
    "make_calibration_residual_function",
    "run_calibration_fit",
    "fit_result_markdown",
    "export_fit_artifacts",
    "compare_fit_to_truth",
    "truth_comparison_markdown",
]