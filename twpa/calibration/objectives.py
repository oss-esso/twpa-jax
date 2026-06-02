"""
Calibration objectives for simulation-vs-target comparison.

This is the first ML/calibration layer above the Julia/Harmonia simulation
bridge. It does not run simulations. It only evaluates how well a candidate
simulation matches a target response.

Initial target:
    linear_sparams

Future targets:
    JosephsonCircuits gain maps
    pump-dependent S-parameters
    noise/QE/CM metrics
    compression curves
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from twpa.io.simulation_schema import compute_two_port_metrics, normalize_s_parameter_shape


@dataclass(frozen=True)
class SParameterObjectiveWeights:
    s21_complex: float = 1.0
    s11_match: float = 0.25
    s22_match: float = 0.25
    reciprocity: float = 0.10
    passivity: float = 1.0
    gain_db: float = 0.25


@dataclass(frozen=True)
class SParameterObjectiveResult:
    total_loss: float
    s21_complex_loss: float
    s11_match_loss: float
    s22_match_loss: float
    reciprocity_loss: float
    passivity_loss: float
    gain_db_loss: float
    candidate_metrics: dict[str, Any]
    target_metrics: dict[str, Any]
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mse_abs(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2))


def _mse_real(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=float) ** 2))


def _maybe_gain_db(s: np.ndarray, gain_db: np.ndarray | None) -> np.ndarray:
    if gain_db is not None:
        return np.asarray(gain_db, dtype=float)

    s21 = s[:, 1, 0]
    return 20.0 * np.log10(np.maximum(np.abs(s21), 1e-300))


def evaluate_sparameter_objective(
    *,
    frequency_hz: np.ndarray,
    candidate_s: np.ndarray,
    target_s: np.ndarray,
    candidate_gain_db: np.ndarray | None = None,
    target_gain_db: np.ndarray | None = None,
    weights: SParameterObjectiveWeights = SParameterObjectiveWeights(),
) -> SParameterObjectiveResult:
    """
    Compare candidate and target 2-port S-parameters.

    Expected S shape:
        (frequency, port_out, port_in)

    Loss components:
        s21_complex_loss:
            complex MSE on forward transmission S21.

        s11_match_loss, s22_match_loss:
            MSE of reflection-coefficient mismatch versus target.

        reciprocity_loss:
            candidate-only penalty for S21 != S12.

        passivity_loss:
            candidate-only penalty for singular values above 1.

        gain_db_loss:
            MSE between gain curves in dB.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    candidate_s = normalize_s_parameter_shape(
        np.asarray(candidate_s, dtype=np.complex128),
        n_frequency=frequency_hz.shape[0],
        n_ports=2,
    )
    target_s = normalize_s_parameter_shape(
        np.asarray(target_s, dtype=np.complex128),
        n_frequency=frequency_hz.shape[0],
        n_ports=2,
    )

    if candidate_s.shape != target_s.shape:
        raise ValueError(f"S-shape mismatch: {candidate_s.shape} vs {target_s.shape}")

    candidate_gain_db = _maybe_gain_db(candidate_s, candidate_gain_db)
    target_gain_db = _maybe_gain_db(target_s, target_gain_db)

    if candidate_gain_db.shape != target_gain_db.shape:
        raise ValueError(
            f"gain_db shape mismatch: {candidate_gain_db.shape} vs {target_gain_db.shape}"
        )

    c_s11 = candidate_s[:, 0, 0]
    c_s12 = candidate_s[:, 0, 1]
    c_s21 = candidate_s[:, 1, 0]
    c_s22 = candidate_s[:, 1, 1]

    t_s11 = target_s[:, 0, 0]
    t_s21 = target_s[:, 1, 0]
    t_s22 = target_s[:, 1, 1]

    s21_complex_loss = _mse_abs(c_s21 - t_s21)
    s11_match_loss = _mse_abs(c_s11 - t_s11)
    s22_match_loss = _mse_abs(c_s22 - t_s22)
    reciprocity_loss = _mse_abs(c_s21 - c_s12)

    singular_values = np.linalg.svd(candidate_s, compute_uv=False)
    passivity_violation = np.maximum(singular_values - 1.0, 0.0)
    passivity_loss = _mse_real(passivity_violation)

    gain_db_loss = _mse_real(candidate_gain_db - target_gain_db)

    total = (
        weights.s21_complex * s21_complex_loss
        + weights.s11_match * s11_match_loss
        + weights.s22_match * s22_match_loss
        + weights.reciprocity * reciprocity_loss
        + weights.passivity * passivity_loss
        + weights.gain_db * gain_db_loss
    )

    candidate_metrics = compute_two_port_metrics(
        frequency_hz=frequency_hz,
        s_parameters=candidate_s,
        gain_db=candidate_gain_db,
    ).to_dict()

    target_metrics = compute_two_port_metrics(
        frequency_hz=frequency_hz,
        s_parameters=target_s,
        gain_db=target_gain_db,
    ).to_dict()

    return SParameterObjectiveResult(
        total_loss=float(total),
        s21_complex_loss=float(s21_complex_loss),
        s11_match_loss=float(s11_match_loss),
        s22_match_loss=float(s22_match_loss),
        reciprocity_loss=float(reciprocity_loss),
        passivity_loss=float(passivity_loss),
        gain_db_loss=float(gain_db_loss),
        candidate_metrics=candidate_metrics,
        target_metrics=target_metrics,
        weights=asdict(weights),
    )


def evaluate_dataset_against_target(
    *,
    parameters: np.ndarray,
    frequency_hz: np.ndarray,
    s_complex: np.ndarray,
    gain_db: np.ndarray,
    target_index: int,
    weights: SParameterObjectiveWeights = SParameterObjectiveWeights(),
) -> list[dict[str, Any]]:
    """
    Evaluate every dataset sample against one target sample.

    Returns one dict per sample, sorted by original sample index.
    """
    parameters = np.asarray(parameters, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    s_complex = np.asarray(s_complex, dtype=np.complex128)
    gain_db = np.asarray(gain_db, dtype=float)

    if s_complex.ndim != 4:
        raise ValueError(f"s_complex must have shape (samples, freq, 2, 2), got {s_complex.shape}")

    n_samples = s_complex.shape[0]

    if target_index < 0 or target_index >= n_samples:
        raise IndexError(f"target_index={target_index} outside [0, {n_samples})")

    target_s = s_complex[target_index]
    target_gain = gain_db[target_index]

    rows: list[dict[str, Any]] = []

    for idx in range(n_samples):
        result = evaluate_sparameter_objective(
            frequency_hz=frequency_hz,
            candidate_s=s_complex[idx],
            target_s=target_s,
            candidate_gain_db=gain_db[idx],
            target_gain_db=target_gain,
            weights=weights,
        )

        rows.append(
            {
                "sample_index": idx,
                "target_index": target_index,
                "total_loss": result.total_loss,
                "s21_complex_loss": result.s21_complex_loss,
                "s11_match_loss": result.s11_match_loss,
                "s22_match_loss": result.s22_match_loss,
                "reciprocity_loss": result.reciprocity_loss,
                "passivity_loss": result.passivity_loss,
                "gain_db_loss": result.gain_db_loss,
                "parameters": parameters[idx].tolist(),
                "candidate_metrics": result.candidate_metrics,
            }
        )

    return rows