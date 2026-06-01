"""
twpa.inference.recovery
=======================

End-to-end synthetic parameter-recovery experiments for TWPA simulations.

This module connects:

    - priors and truth parameter samples,
    - synthetic S-parameter / gain data generation,
    - calibration/inference fitting,
    - recovery diagnostics and artifact export.

The goal is to answer questions such as:

    "If the true device has L/C/I* deviations, can my calibration workflow
    recover them from pump-off S-parameters and pump-on gain?"

The implementation is intentionally workflow-oriented and notebook-friendly.
It is not a replacement for production calibration in
``twpa.workflows.calibration``; rather, it stress-tests and documents it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import json
import time
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout
from twpa.core.params import NonlinearParams
from twpa.linear.cascade import CascadeConfig
from twpa.linear.cells import CellModelConfig
from twpa.nonlinear.pump_hb_ladder import PumpDriveConfig, PumpHBLadderConfig
from twpa.inference.priors import (
    ParameterSample,
    PriorSet,
    make_default_twpa_scale_prior_set,
)
from twpa.inference.synthetic import (
    SyntheticCombinedDataset,
    SyntheticGainDataset,
    SyntheticNoiseConfig,
    SyntheticSParameterDataset,
    generate_combined_synthetic_dataset,
)
from twpa.inference.fitting import (
    FitConfig,
    FitResult,
    compare_fit_to_truth,
    export_fit_artifacts,
    fit_result_markdown,
    run_parameter_fit,
    truth_comparison_markdown,
)


ArrayLike = Any
ResidualFactory = Callable[[SyntheticCombinedDataset, PriorSet], Callable[[dict[str, float]], ArrayLike]]


class RecoveryStatus(str, Enum):
    """Synthetic recovery experiment status."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


class RecoveryDatasetMode(str, Enum):
    """Which synthetic measurements to generate."""

    SPARAMETER_ONLY = "sparameter_only"
    GAIN_ONLY = "gain_only"
    COMBINED = "combined"


@dataclass(frozen=True)
class RecoveryToleranceConfig:
    """
    Tolerances for declaring a recovery experiment successful.

    Parameters
    ----------
    max_abs_relative_error:
        Maximum allowed relative parameter error for parameters with nonzero
        truth values.
    max_absolute_error:
        Maximum allowed absolute error for parameters with near-zero truth.
    require_fit_success:
        If True, fit_result.success must be True.
    parameters_to_check:
        Optional subset of truth parameters to check. If None, all truth
        parameters present in the fit are checked.
    """

    max_abs_relative_error: float = 0.10
    max_absolute_error: float = 1e-9
    require_fit_success: bool = True
    parameters_to_check: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_abs_relative_error < 0.0:
            raise ValueError("max_abs_relative_error must be non-negative")
        if self.max_absolute_error < 0.0:
            raise ValueError("max_absolute_error must be non-negative")
        if self.parameters_to_check is not None:
            object.__setattr__(
                self,
                "parameters_to_check",
                tuple(str(x) for x in self.parameters_to_check),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_relative_error": self.max_abs_relative_error,
            "max_absolute_error": self.max_absolute_error,
            "require_fit_success": self.require_fit_success,
            "parameters_to_check": (
                None if self.parameters_to_check is None else list(self.parameters_to_check)
            ),
        }


@dataclass(frozen=True)
class RecoveryExperimentConfig:
    """
    End-to-end synthetic recovery experiment configuration.

    Parameters
    ----------
    dataset_mode:
        S-parameter only, gain only, or combined synthetic data.
    sparameter_frequency_hz:
        Pump-off frequency grid. Required for SPARAMETER_ONLY or COMBINED.
    signal_frequency_hz:
        Pump-on signal grid. Required for GAIN_ONLY or COMBINED.
    noise:
        Synthetic noise model.
    truth_prior_set:
        Prior set used to sample truth values if true_parameters is None.
    fit_prior_set:
        Prior set used by the fitter. If None, truth_prior_set is reused.
    true_parameters:
        Explicit truth parameter dictionary. If None, sampled from truth_prior_set.
    fit_config:
        Fit optimizer configuration.
    tolerance:
        Recovery pass/fail tolerance.
    n_trials:
        Number of independent synthetic recovery trials.
    base_seed:
        Base random seed. Trial i uses base_seed + i.
    save_intermediate_datasets:
        Whether exported artifacts include generated NPZ datasets.
    name:
        Diagnostic name.
    """

    dataset_mode: RecoveryDatasetMode = RecoveryDatasetMode.COMBINED
    sparameter_frequency_hz: ArrayLike | None = None
    signal_frequency_hz: ArrayLike | None = None
    noise: SyntheticNoiseConfig = SyntheticNoiseConfig()
    truth_prior_set: PriorSet | None = None
    fit_prior_set: PriorSet | None = None
    true_parameters: Mapping[str, float] | None = None
    fit_config: FitConfig = FitConfig()
    tolerance: RecoveryToleranceConfig = RecoveryToleranceConfig()
    n_trials: int = 1
    base_seed: int = 1234
    save_intermediate_datasets: bool = True
    name: str = "synthetic_recovery_experiment"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_mode", RecoveryDatasetMode(self.dataset_mode))

        if int(self.n_trials) <= 0:
            raise ValueError("n_trials must be positive")
        object.__setattr__(self, "n_trials", int(self.n_trials))
        object.__setattr__(self, "base_seed", int(self.base_seed))

        if self.dataset_mode in {
            RecoveryDatasetMode.SPARAMETER_ONLY,
            RecoveryDatasetMode.COMBINED,
        } and self.sparameter_frequency_hz is None:
            raise ValueError("sparameter_frequency_hz is required for this dataset mode")

        if self.dataset_mode in {
            RecoveryDatasetMode.GAIN_ONLY,
            RecoveryDatasetMode.COMBINED,
        } and self.signal_frequency_hz is None:
            raise ValueError("signal_frequency_hz is required for this dataset mode")

        object.__setattr__(
            self,
            "truth_prior_set",
            self.truth_prior_set or make_default_twpa_scale_prior_set(),
        )
        object.__setattr__(
            self,
            "fit_prior_set",
            self.fit_prior_set or self.truth_prior_set or make_default_twpa_scale_prior_set(),
        )
        object.__setattr__(
            self,
            "true_parameters",
            None if self.true_parameters is None else dict(self.true_parameters),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def trial_noise(self, trial_index: int) -> SyntheticNoiseConfig:
        """
        Return a trial-specific noise config with deterministic seed.
        """
        return self.noise.with_updates(seed=self.base_seed + int(trial_index))

    def with_updates(self, **kwargs: Any) -> "RecoveryExperimentConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_mode": self.dataset_mode.value,
            "sparameter_frequency_hz": _array_summary(self.sparameter_frequency_hz),
            "signal_frequency_hz": _array_summary(self.signal_frequency_hz),
            "noise": self.noise.to_dict(),
            "truth_prior_set": None if self.truth_prior_set is None else self.truth_prior_set.to_dict(),
            "fit_prior_set": None if self.fit_prior_set is None else self.fit_prior_set.to_dict(),
            "true_parameters": None if self.true_parameters is None else dict(self.true_parameters),
            "fit_config": self.fit_config.to_dict(),
            "tolerance": self.tolerance.to_dict(),
            "n_trials": self.n_trials,
            "base_seed": self.base_seed,
            "save_intermediate_datasets": self.save_intermediate_datasets,
            "name": self.name,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RecoveryTrialResult:
    """
    Result of one synthetic recovery trial.
    """

    trial_index: int
    true_parameters: Mapping[str, float]
    dataset: SyntheticCombinedDataset
    fit_result: FitResult
    truth_comparison: Mapping[str, Any]
    passed: bool
    status: RecoveryStatus
    messages: tuple[str, ...]
    elapsed_s: float
    artifact_paths: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "true_parameters",
            {str(k): float(v) for k, v in self.true_parameters.items()},
        )
        object.__setattr__(self, "messages", tuple(str(m) for m in self.messages))
        object.__setattr__(self, "status", RecoveryStatus(self.status))
        object.__setattr__(self, "artifact_paths", dict(self.artifact_paths or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self, *, include_fit_residual: bool = False) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "true_parameters": dict(self.true_parameters),
            "dataset": self.dataset.to_dict(),
            "fit_result": self.fit_result.to_dict(include_residual=include_fit_residual),
            "truth_comparison": _jsonify(self.truth_comparison),
            "passed": self.passed,
            "status": self.status.value,
            "messages": list(self.messages),
            "elapsed_s": self.elapsed_s,
            "artifact_paths": dict(self.artifact_paths or {}),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RecoveryExperimentResult:
    """
    Result of an end-to-end synthetic recovery experiment.
    """

    config: RecoveryExperimentConfig
    trials: tuple[RecoveryTrialResult, ...]
    status: RecoveryStatus
    passed: bool
    elapsed_s: float
    artifact_paths: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_passed(self) -> int:
        return sum(1 for t in self.trials if t.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(self.n_trials, 1)

    def to_dict(self, *, include_fit_residual: bool = False) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "passed": self.passed,
            "n_trials": self.n_trials,
            "n_passed": self.n_passed,
            "pass_rate": self.pass_rate,
            "elapsed_s": self.elapsed_s,
            "config": self.config.to_dict(),
            "trials": [
                t.to_dict(include_fit_residual=include_fit_residual)
                for t in self.trials
            ],
            "artifact_paths": dict(self.artifact_paths or {}),
            "metadata": dict(self.metadata or {}),
        }


def _array_summary(x: Any) -> dict[str, Any] | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0:
        return {"shape": tuple(int(v) for v in arr.shape), "size": 0}
    return {
        "shape": tuple(int(v) for v in arr.shape),
        "size": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _jsonify(obj.to_dict())
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return arr.item()
        return {
            "array_shape": tuple(int(v) for v in arr.shape),
            "array_dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }
    return obj


def choose_true_parameters(
    config: RecoveryExperimentConfig,
    *,
    trial_index: int,
) -> dict[str, float]:
    """
    Choose truth parameters for a trial.

    Explicit config.true_parameters takes precedence. Otherwise, a deterministic
    sample is drawn from config.truth_prior_set.
    """
    if config.true_parameters is not None:
        return {str(k): float(v) for k, v in config.true_parameters.items()}

    assert config.truth_prior_set is not None
    sample = config.truth_prior_set.sample(seed=config.base_seed + 10_000 + int(trial_index))
    return dict(sample.values)


def build_default_recovery_residual_factory(
    *,
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None = None,
    pump_drive: PumpDriveConfig | None = None,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    pump_config: PumpHBLadderConfig | None = None,
) -> ResidualFactory:
    """
    Build a simple residual factory using synthetic datasets directly.

    This factory is intentionally generic:

    - For S-parameters, it compares model S21 dB to synthetic S21 dB.
    - For gain, if nonlinear inputs are provided, it attempts to use the
      production calibration workflow via twpa.workflows.calibration.
    - If gain calibration infrastructure is unavailable, it raises a clear
      error rather than silently pretending to fit gain.

    For full production recovery, prefer passing a custom residual_factory that
    uses ``twpa.workflows.calibration.CalibrationTarget`` directly.
    """
    from twpa.inference.synthetic import (
        apply_parameter_scales_to_layout,
        make_gain_frequency_plan,
        make_gain_sweep_config_for_frequencies,
    )
    from twpa.linear.cascade import run_linear_scan

    def factory(dataset: SyntheticCombinedDataset, prior_set: PriorSet):
        def residual_fn(params: dict[str, float]) -> jax.Array:
            residual_parts = []

            if dataset.sparameters is not None:
                model_layout = apply_parameter_scales_to_layout(layout, params)
                scan = run_linear_scan(
                    dataset.sparameters.frequency_hz,
                    model_layout,
                    cell_model=cell_model or CellModelConfig(),
                    cascade_config=cascade_config or CascadeConfig(),
                )
                residual_parts.append(scan.s21_db - dataset.sparameters.s21_db_noisy)

            if dataset.gain is not None:
                if nonlinear_params is None or pump_drive is None:
                    raise RuntimeError(
                        "Gain residual requested but nonlinear_params or pump_drive is missing. "
                        "Pass a custom residual_factory or provide nonlinear inputs."
                    )

                # Prefer production calibration objective if available.
                try:
                    from twpa.workflows.calibration import (
                        CalibrationTarget,
                        GainCalibrationData,
                        evaluate_calibration_objective,
                    )

                    gain_dataset = dataset.gain
                    assert gain_dataset is not None
                    pump_cfg = pump_config or PumpHBLadderConfig()
                    output_impedance = (
                        1.0 / pump_cfg.distributed.load_conductance_S
                        if pump_cfg.distributed.load_conductance_S > 0.0
                        else pump_drive.source_impedance_ohm
                    )

                    def target_plan_factory(pump_result: Any) -> Any:
                        return make_gain_frequency_plan(
                            pump_frequency_hz=pump_result.drive.pump_frequency_hz,
                            signal_frequency_hz=gain_dataset.signal_frequency_hz,
                            idler_frequency_hz=gain_dataset.idler_frequency_hz,
                            pump_label=pump_result.drive.pump_label,
                            signal_labels=gain_dataset.signal_labels,
                            idler_labels=gain_dataset.idler_labels,
                            n_pump_harmonics=pump_cfg.n_pump_harmonics,
                            include_negative=pump_cfg.include_negative_frequencies,
                            include_dc=pump_cfg.include_dc,
                        )

                    def sweep_config_factory(_plan: Any) -> Any:
                        return make_gain_sweep_config_for_frequencies(
                            signal_labels=gain_dataset.signal_labels,
                            idler_labels=gain_dataset.idler_labels,
                            input_node=pump_cfg.distributed.input_node,
                            output_node=(
                                None
                                if pump_cfg.distributed.output_node < 0
                                else pump_cfg.distributed.output_node
                            ),
                            input_impedance_ohm=pump_drive.source_impedance_ohm,
                            output_impedance_ohm=output_impedance,
                        )

                    target = CalibrationTarget(
                        base_layout=layout,
                        base_nonlinear_params=nonlinear_params,
                        cell_model=cell_model or CellModelConfig(),
                        cascade=cascade_config or CascadeConfig(),
                        pump_drive=pump_drive,
                        pump_config=pump_cfg,
                        target_plan_factory=target_plan_factory,
                        sweep_config_factory=sweep_config_factory,
                        metadata={"source": "recovery_default_gain_residual"},
                    )

                    gain_data = dataset.gain.to_calibration_data()
                    evaluation = evaluate_calibration_objective(
                        target,
                        params,
                        gain_data=gain_data,
                    )

                    if hasattr(evaluation, "residual"):
                        residual_parts.append(jnp.ravel(jnp.asarray(evaluation.residual)))
                    elif hasattr(evaluation, "residual_vector"):
                        residual_parts.append(jnp.ravel(jnp.asarray(evaluation.residual_vector)))
                    elif hasattr(evaluation, "loss"):
                        residual_parts.append(
                            jnp.asarray([jnp.sqrt(jnp.maximum(2.0 * evaluation.loss, 0.0))])
                        )
                    else:
                        raise RuntimeError("Unsupported calibration evaluation object.")

                except Exception as exc:
                    raise RuntimeError(
                        "Could not evaluate gain recovery residual. "
                        "Pass a custom residual_factory for this device/workflow. "
                        f"Original error: {exc}"
                    ) from exc

            if not residual_parts:
                return jnp.zeros((0,), dtype=jnp.float64)

            return jnp.concatenate([jnp.ravel(jnp.asarray(r, dtype=jnp.float64)) for r in residual_parts])

        return residual_fn

    return factory


def evaluate_recovery_pass_fail(
    fit_result: FitResult,
    true_parameters: Mapping[str, float],
    tolerance: RecoveryToleranceConfig,
) -> tuple[bool, RecoveryStatus, tuple[str, ...], dict[str, Any]]:
    """
    Evaluate recovery pass/fail against truth.
    """
    comparison = compare_fit_to_truth(fit_result, true_parameters)

    messages: list[str] = []
    passed = True

    if tolerance.require_fit_success and not fit_result.success:
        passed = False
        messages.append("fit_result.success is False")

    parameters_to_check = (
        set(tolerance.parameters_to_check)
        if tolerance.parameters_to_check is not None
        else None
    )

    for row in comparison["rows"]:
        name = row["name"]
        if parameters_to_check is not None and name not in parameters_to_check:
            continue

        if row["fit"] is None:
            passed = False
            messages.append(f"{name}: fitted value missing")
            continue

        true_value = float(row["true"])
        fit_value = float(row["fit"])
        abs_error = abs(fit_value - true_value)

        if abs(true_value) <= 1e-300:
            if abs_error > tolerance.max_absolute_error:
                passed = False
                messages.append(
                    f"{name}: absolute error {abs_error:.6e} exceeds "
                    f"{tolerance.max_absolute_error:.6e}"
                )
        else:
            rel_error = abs_error / abs(true_value)
            if rel_error > tolerance.max_abs_relative_error:
                passed = False
                messages.append(
                    f"{name}: relative error {rel_error:.6e} exceeds "
                    f"{tolerance.max_abs_relative_error:.6e}"
                )

    if passed:
        messages.append("PASS: recovery tolerances satisfied.")
        status = RecoveryStatus.PASS
    elif fit_result.success:
        status = RecoveryStatus.FAIL
    else:
        status = RecoveryStatus.ERROR

    return passed, status, tuple(messages), comparison


def generate_recovery_dataset(
    layout: LineLayout,
    nonlinear_params: NonlinearParams | None,
    *,
    pump_drive: PumpDriveConfig | None,
    config: RecoveryExperimentConfig,
    true_parameters: Mapping[str, float],
    trial_index: int,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    pump_config: PumpHBLadderConfig | None = None,
) -> SyntheticCombinedDataset:
    """
    Generate the synthetic dataset for one recovery trial.
    """
    noise = config.trial_noise(trial_index)

    s_grid = None
    g_grid = None

    if config.dataset_mode in {
        RecoveryDatasetMode.SPARAMETER_ONLY,
        RecoveryDatasetMode.COMBINED,
    }:
        s_grid = config.sparameter_frequency_hz

    if config.dataset_mode in {
        RecoveryDatasetMode.GAIN_ONLY,
        RecoveryDatasetMode.COMBINED,
    }:
        g_grid = config.signal_frequency_hz
        if nonlinear_params is None:
            raise ValueError("nonlinear_params is required for gain recovery datasets")
        if pump_drive is None:
            raise ValueError("pump_drive is required for gain recovery datasets")

    return generate_combined_synthetic_dataset(
        layout,
        nonlinear_params=nonlinear_params,
        sparameter_frequency_hz=s_grid,
        signal_frequency_hz=g_grid,
        pump_drive=pump_drive,
        cell_model=cell_model,
        cascade_config=cascade_config,
        pump_config=pump_config,
        noise=noise,
        true_parameters=true_parameters,
        metadata={
            "trial_index": trial_index,
            "recovery_config_name": config.name,
        },
    )


def run_recovery_trial(
    layout: LineLayout,
    *,
    config: RecoveryExperimentConfig,
    trial_index: int,
    residual_factory: ResidualFactory,
    nonlinear_params: NonlinearParams | None = None,
    pump_drive: PumpDriveConfig | None = None,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    pump_config: PumpHBLadderConfig | None = None,
    output_dir: str | Path | None = None,
) -> RecoveryTrialResult:
    """
    Run one synthetic recovery trial.
    """
    start = time.perf_counter()

    true_parameters = choose_true_parameters(config, trial_index=trial_index)

    dataset = generate_recovery_dataset(
        layout,
        nonlinear_params,
        pump_drive=pump_drive,
        config=config,
        true_parameters=true_parameters,
        trial_index=trial_index,
        cell_model=cell_model,
        cascade_config=cascade_config,
        pump_config=pump_config,
    )

    assert config.fit_prior_set is not None
    residual_fn = residual_factory(dataset, config.fit_prior_set)

    fit_result = run_parameter_fit(
        residual_fn,
        config.fit_prior_set,
        config=config.fit_config.with_updates(
            random_seed=config.base_seed + 20_000 + trial_index
        ),
        metadata={
            "trial_index": trial_index,
            "recovery_experiment": config.name,
        },
    )

    passed, status, messages, comparison = evaluate_recovery_pass_fail(
        fit_result,
        true_parameters,
        config.tolerance,
    )

    artifact_paths: dict[str, str] = {}
    if output_dir is not None:
        trial_dir = Path(output_dir) / f"trial_{trial_index:03d}"
        artifact_paths = export_recovery_trial_artifacts(
            RecoveryTrialResult(
                trial_index=trial_index,
                true_parameters=true_parameters,
                dataset=dataset,
                fit_result=fit_result,
                truth_comparison=comparison,
                passed=passed,
                status=status,
                messages=messages,
                elapsed_s=time.perf_counter() - start,
                artifact_paths={},
                metadata={},
            ),
            trial_dir,
            save_intermediate_datasets=config.save_intermediate_datasets,
        )

    return RecoveryTrialResult(
        trial_index=trial_index,
        true_parameters=true_parameters,
        dataset=dataset,
        fit_result=fit_result,
        truth_comparison=comparison,
        passed=passed,
        status=status,
        messages=messages,
        elapsed_s=time.perf_counter() - start,
        artifact_paths=artifact_paths,
        metadata={
            "source": "run_recovery_trial",
        },
    )


def run_synthetic_recovery_experiment(
    layout: LineLayout,
    *,
    config: RecoveryExperimentConfig,
    residual_factory: ResidualFactory | None = None,
    nonlinear_params: NonlinearParams | None = None,
    pump_drive: PumpDriveConfig | None = None,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    pump_config: PumpHBLadderConfig | None = None,
    output_dir: str | Path | None = None,
) -> RecoveryExperimentResult:
    """
    Run an end-to-end synthetic recovery experiment.
    """
    start = time.perf_counter()

    factory = residual_factory or build_default_recovery_residual_factory(
        layout=layout,
        nonlinear_params=nonlinear_params,
        pump_drive=pump_drive,
        cell_model=cell_model,
        cascade_config=cascade_config,
        pump_config=pump_config,
    )

    trials: list[RecoveryTrialResult] = []

    for trial_index in range(config.n_trials):
        try:
            trial = run_recovery_trial(
                layout,
                config=config,
                trial_index=trial_index,
                residual_factory=factory,
                nonlinear_params=nonlinear_params,
                pump_drive=pump_drive,
                cell_model=cell_model,
                cascade_config=cascade_config,
                pump_config=pump_config,
                output_dir=output_dir,
            )
        except Exception as exc:
            dummy_truth = choose_true_parameters(config, trial_index=trial_index)
            # Create a minimal failure by re-raising if no safe dataset exists.
            # In practice this path is mainly for reporting batch failures.
            raise RuntimeError(f"Recovery trial {trial_index} failed: {exc}") from exc

        trials.append(trial)

    n_passed = sum(1 for t in trials if t.passed)
    if n_passed == len(trials):
        status = RecoveryStatus.PASS
        passed = True
    elif n_passed == 0:
        status = RecoveryStatus.FAIL
        passed = False
    else:
        status = RecoveryStatus.PARTIAL
        passed = False

    result = RecoveryExperimentResult(
        config=config,
        trials=tuple(trials),
        status=status,
        passed=passed,
        elapsed_s=time.perf_counter() - start,
        artifact_paths={},
        metadata={
            "source": "run_synthetic_recovery_experiment",
            "layout": layout.summary(),
        },
    )

    artifact_paths: dict[str, str] = {}
    if output_dir is not None:
        artifact_paths = export_recovery_experiment_artifacts(result, output_dir)
        result = replace(result, artifact_paths=artifact_paths)

    return result


def export_recovery_trial_artifacts(
    trial: RecoveryTrialResult,
    output_dir: str | Path,
    *,
    save_intermediate_datasets: bool = True,
) -> dict[str, str]:
    """
    Export one recovery trial.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    summary_path = out / "trial_summary.json"
    summary_path.write_text(
        json.dumps(_jsonify(trial.to_dict(include_fit_residual=False)), indent=2),
        encoding="utf-8",
    )
    paths["trial_summary_json"] = str(summary_path)

    md_path = out / "trial_summary.md"
    md_path.write_text(recovery_trial_markdown(trial), encoding="utf-8")
    paths["trial_summary_md"] = str(md_path)

    comparison_md = out / "truth_comparison.md"
    comparison_md.write_text(truth_comparison_markdown(trial.truth_comparison), encoding="utf-8")
    paths["truth_comparison_md"] = str(comparison_md)

    fit_paths = export_fit_artifacts(
        trial.fit_result,
        out / "fit",
        prefix="fit",
        include_residual=True,
    )
    paths.update({f"fit_{k}": v for k, v in fit_paths.items()})

    if save_intermediate_datasets:
        dataset_paths = trial.dataset.save_npz_bundle(out / "datasets", prefix="synthetic")
        paths.update({f"dataset_{k}": v for k, v in dataset_paths.items()})

    return paths


def export_recovery_experiment_artifacts(
    result: RecoveryExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Export full recovery experiment artifacts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    summary_path = out / "recovery_experiment_summary.json"
    summary_path.write_text(
        json.dumps(_jsonify(result.to_dict(include_fit_residual=False)), indent=2),
        encoding="utf-8",
    )
    paths["summary_json"] = str(summary_path)

    md_path = out / "recovery_experiment_summary.md"
    md_path.write_text(recovery_experiment_markdown(result), encoding="utf-8")
    paths["summary_md"] = str(md_path)

    arrays_path = out / "recovery_experiment_arrays.npz"
    np.savez_compressed(
        arrays_path,
        passed=np.asarray([t.passed for t in result.trials], dtype=bool),
        loss=np.asarray([t.fit_result.loss for t in result.trials], dtype=float),
        residual_norm=np.asarray([t.fit_result.residual_norm for t in result.trials], dtype=float),
        elapsed_s=np.asarray([t.elapsed_s for t in result.trials], dtype=float),
    )
    paths["arrays_npz"] = str(arrays_path)

    return paths


def recovery_trial_markdown(trial: RecoveryTrialResult) -> str:
    """
    Markdown summary for one recovery trial.
    """
    lines = [
        f"# Recovery trial {trial.trial_index}",
        "",
        f"- status: `{trial.status.value}`",
        f"- passed: `{trial.passed}`",
        f"- elapsed: `{trial.elapsed_s:.6g} s`",
        f"- fit status: `{trial.fit_result.status.value}`",
        f"- fit loss: `{trial.fit_result.loss:.9e}`",
        f"- residual norm: `{trial.fit_result.residual_norm:.9e}`",
        "",
        "## Messages",
        "",
        *[f"- {m}" for m in trial.messages],
        "",
        "## Fit summary",
        "",
        fit_result_markdown(trial.fit_result),
        "",
        "## Truth comparison",
        "",
        truth_comparison_markdown(trial.truth_comparison),
    ]

    return "\n".join(lines)


def recovery_experiment_markdown(result: RecoveryExperimentResult) -> str:
    """
    Markdown summary for a full recovery experiment.
    """
    lines = [
        "# Synthetic recovery experiment",
        "",
        f"- status: `{result.status.value}`",
        f"- passed: `{result.passed}`",
        f"- trials: `{result.n_passed}/{result.n_trials}` passed",
        f"- pass rate: `{result.pass_rate:.6g}`",
        f"- elapsed: `{result.elapsed_s:.6g} s`",
        "",
        "## Trial summary",
        "",
        "| trial | status | passed | fit status | loss | residual norm | elapsed s |",
        "|---:|---|---|---|---:|---:|---:|",
    ]

    for t in result.trials:
        lines.append(
            f"| {t.trial_index} | `{t.status.value}` | `{t.passed}` | "
            f"`{t.fit_result.status.value}` | {t.fit_result.loss:.6e} | "
            f"{t.fit_result.residual_norm:.6e} | {t.elapsed_s:.6g} |"
        )

    lines += [
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(_jsonify(result.config.to_dict()), indent=2),
        "```",
    ]

    return "\n".join(lines)


def aggregate_parameter_errors(
    result: RecoveryExperimentResult,
) -> dict[str, Any]:
    """
    Aggregate parameter recovery errors across trials.
    """
    rows_by_name: dict[str, list[dict[str, Any]]] = {}

    for trial in result.trials:
        for row in trial.truth_comparison.get("rows", []):
            rows_by_name.setdefault(row["name"], []).append(row)

    summary: dict[str, Any] = {
        "n_trials": result.n_trials,
        "parameters": {},
    }

    for name, rows in rows_by_name.items():
        abs_errors = [
            abs(float(r["absolute_error"]))
            for r in rows
            if r.get("absolute_error") is not None
        ]
        rel_errors = [
            abs(float(r["relative_error"]))
            for r in rows
            if r.get("relative_error") is not None
        ]

        summary["parameters"][name] = {
            "n": len(rows),
            "absolute_error_mean": float(np.mean(abs_errors)) if abs_errors else None,
            "absolute_error_max": float(np.max(abs_errors)) if abs_errors else None,
            "relative_error_mean": float(np.mean(rel_errors)) if rel_errors else None,
            "relative_error_max": float(np.max(rel_errors)) if rel_errors else None,
        }

    return summary


__all__ = [
    "ArrayLike",
    "ResidualFactory",
    "RecoveryStatus",
    "RecoveryDatasetMode",
    "RecoveryToleranceConfig",
    "RecoveryExperimentConfig",
    "RecoveryTrialResult",
    "RecoveryExperimentResult",
    "choose_true_parameters",
    "build_default_recovery_residual_factory",
    "evaluate_recovery_pass_fail",
    "generate_recovery_dataset",
    "run_recovery_trial",
    "run_synthetic_recovery_experiment",
    "export_recovery_trial_artifacts",
    "export_recovery_experiment_artifacts",
    "recovery_trial_markdown",
    "recovery_experiment_markdown",
    "aggregate_parameter_errors",
]
