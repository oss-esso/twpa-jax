"""Tracked Floquet branches for harmonic-balance stability diagnostics.

The dense Floquet scan is useful for finding candidate roots, but it does not
provide a branch identity.  This module keeps one complex root tied to the
previous operating point and records discontinuities instead of silently
switching to a nearby mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from twpa_solver.signal.stability import (
    ComplexResonance,
    FloquetClassification,
    classify_floquet_resonance,
    refine_complex_resonance,
)
from twpa_solver.stability.tracking import track_multiplier_branches


@dataclass(frozen=True)
class FloquetBranchPoint:
    """One tracked root at one value of the continuation parameter."""

    parameter: float
    resonance: ComplexResonance
    classification: FloquetClassification
    branch_index: int
    discontinuity: bool
    stability_verdict: str
    mode_overlap: float | None


@dataclass(frozen=True)
class FloquetBranch:
    """A complete tracked branch and its numerical provenance."""

    points: tuple[FloquetBranchPoint, ...]
    seed_signal_ghz: complex
    discontinuity_threshold: float
    mode_overlap_threshold: float


def stability_verdict(classification: FloquetClassification, converged: bool) -> str:
    """Classify a tracked multiplier without treating unknowns as stable."""
    if not converged:
        return "UNDECIDED"
    if classification.magnitude <= 1.0:
        return "STABLE"
    if classification.kind == "NEIMARK_SACKER_CANDIDATE":
        return "UNSTABLE_NS"
    if classification.kind == "FOLD_CANDIDATE":
        return "UNSTABLE_FOLD"
    return "UNDECIDED"


def _as_seed(value: float | complex) -> complex:
    seed = complex(value)
    if not np.isfinite(seed.real) or not np.isfinite(seed.imag):
        raise ValueError("Floquet seed must be finite")
    return seed


def _mode_overlap(
    previous: np.ndarray | None, current: np.ndarray | None
) -> float | None:
    if previous is None or current is None:
        return None
    previous_array = np.asarray(previous, dtype=np.complex128).reshape(-1)
    current_array = np.asarray(current, dtype=np.complex128).reshape(-1)
    if previous_array.shape != current_array.shape:
        return 0.0
    previous_norm = np.linalg.norm(previous_array)
    current_norm = np.linalg.norm(current_array)
    if previous_norm == 0.0 or current_norm == 0.0:
        return 0.0
    return float(
        abs(np.vdot(previous_array, current_array))
        / (previous_norm * current_norm)
    )


def track_floquet_point(
    circuit: Any,
    khat: dict[int, sp.spmatrix],
    parameter: float,
    *,
    omega_p: float,
    ms: list[int],
    seed_signal_ghz: float | complex,
    loss_model: str,
    seed_mode_vector: np.ndarray | None = None,
    previous_multiplier: complex | None = None,
    khat_base: sp.spmatrix | None = None,
    max_iters: int = 30,
    tol: float = 1e-9,
    discontinuity_threshold: float = 0.25,
    mode_overlap_threshold: float = 0.8,
) -> FloquetBranchPoint:
    """Refine one point and retain continuity diagnostics."""
    resonance = refine_complex_resonance(
        circuit=circuit,
        khat=khat,
        omega_p=omega_p,
        ms=ms,
        signal_ghz_guess=_as_seed(seed_signal_ghz),
        loss_model=loss_model,
        max_iters=max_iters,
        tol=tol,
        khat_base=khat_base,
        v0=seed_mode_vector,
    )
    classification = classify_floquet_resonance(resonance, omega_p)
    multiplier = classification.multiplier
    displacement = (
        0.0
        if previous_multiplier is None
        else abs(multiplier - previous_multiplier)
        / (1.0 + abs(multiplier) + abs(previous_multiplier))
    )
    mode_overlap = _mode_overlap(seed_mode_vector, resonance.mode_vector)
    discontinuity = displacement > discontinuity_threshold or (
        mode_overlap is not None and mode_overlap < mode_overlap_threshold
    )
    return FloquetBranchPoint(
        parameter=float(parameter),
        resonance=resonance,
        classification=classification,
        branch_index=0,
        discontinuity=discontinuity,
        stability_verdict=stability_verdict(classification, resonance.converged),
        mode_overlap=mode_overlap,
    )


def track_floquet_branch(
    circuit: Any,
    khat_by_parameter: Sequence[dict[int, sp.spmatrix]],
    parameters: Sequence[float],
    *,
    omega_p: float,
    ms: list[int],
    seed_signal_ghz: float | complex,
    loss_model: str,
    khat_base_by_parameter: Sequence[sp.spmatrix | None] | None = None,
    max_iters: int = 30,
    tol: float = 1e-9,
    discontinuity_threshold: float = 0.25,
    mode_overlap_threshold: float = 0.8,
) -> FloquetBranch:
    """Refine and track one Floquet root across a parameter sequence.

    ``khat_by_parameter`` must be ordered in the same direction as
    ``parameters``.  The previous converged complex frequency seeds the next
    refinement.  A large multiplier displacement is retained as a
    discontinuity flag; it is never repaired by selecting a different root.
    """
    if len(khat_by_parameter) != len(parameters):
        raise ValueError("khat and parameter sequences must have equal length")
    if not khat_by_parameter:
        raise ValueError("at least one parameter setting is required")
    if khat_base_by_parameter is not None and len(khat_base_by_parameter) != len(
        parameters
    ):
        raise ValueError("khat_base sequence must match parameters")
    if omega_p <= 0.0:
        raise ValueError("omega_p must be positive")

    seed = _as_seed(seed_signal_ghz)
    previous_multiplier: complex | None = None
    previous_mode_vector: np.ndarray | None = None
    points: list[FloquetBranchPoint] = []
    for index, (parameter, khat) in enumerate(zip(parameters, khat_by_parameter)):
        base = None if khat_base_by_parameter is None else khat_base_by_parameter[index]
        point = track_floquet_point(
            circuit=circuit,
            khat=khat,
            parameter=parameter,
            omega_p=omega_p,
            ms=ms,
            seed_signal_ghz=seed,
            loss_model=loss_model,
            seed_mode_vector=previous_mode_vector,
            previous_multiplier=previous_multiplier,
            khat_base=base,
            max_iters=max_iters,
            tol=tol,
            discontinuity_threshold=discontinuity_threshold,
            mode_overlap_threshold=mode_overlap_threshold,
        )
        points.append(point)
        if point.resonance.converged:
            seed = complex(point.resonance.signal_ghz)
            previous_multiplier = point.classification.multiplier
            previous_mode_vector = point.resonance.mode_vector

    return FloquetBranch(
        points=tuple(points),
        seed_signal_ghz=_as_seed(seed_signal_ghz),
        discontinuity_threshold=float(discontinuity_threshold),
        mode_overlap_threshold=float(mode_overlap_threshold),
    )


def order_multiplier_sets(
    previous: Sequence[complex], current: Sequence[complex]
) -> np.ndarray:
    """Expose the project-wide multiplier matcher for branch-aware callers."""
    return track_multiplier_branches(
        np.asarray(previous, dtype=np.complex128),
        np.asarray(current, dtype=np.complex128),
    )


def serialize_branch(branch: FloquetBranch) -> dict[str, object]:
    """Return JSON-compatible branch data."""
    return {
        "seed_signal_ghz": {
            "real": branch.seed_signal_ghz.real,
            "imag": branch.seed_signal_ghz.imag,
        },
        "discontinuity_threshold": branch.discontinuity_threshold,
        "mode_overlap_threshold": branch.mode_overlap_threshold,
        "points": [
            {
                "parameter": point.parameter,
                "branch_index": point.branch_index,
                "discontinuity": point.discontinuity,
                "mode_overlap": point.mode_overlap,
                "stability_verdict": point.stability_verdict,
                "signal_ghz": {
                    "real": point.resonance.signal_ghz.real,
                    "imag": point.resonance.signal_ghz.imag,
                },
                "converged": point.resonance.converged,
                "iterations": point.resonance.iterations,
                "residual": point.resonance.residual,
                "mode_vector": (
                    None
                    if point.resonance.mode_vector is None
                    else {
                        "real": np.asarray(point.resonance.mode_vector).real.tolist(),
                        "imag": np.asarray(point.resonance.mode_vector).imag.tolist(),
                    }
                ),
                "growth_rate_per_s": point.resonance.growth_rate_per_s,
                "multiplier": {
                    "real": point.classification.multiplier.real,
                    "imag": point.classification.multiplier.imag,
                    "magnitude": point.classification.magnitude,
                    "phase_rad": point.classification.phase_rad,
                    "kind": point.classification.kind,
                },
            }
            for point in branch.points
        ],
    }
