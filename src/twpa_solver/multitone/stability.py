"""Finite-signal Floquet diagnostics for a converged multitone torus."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.sparse as sp

from twpa_solver.signal.stability import (
    NON_ANALYTIC_LOSS_MODELS,
    refine_complex_resonance,
    sigma_min_at_signal_ghz,
)


@dataclass(frozen=True)
class MultitoneStabilityResult:
    status: str
    dominant_exponent_per_s: float | None
    sigma_min: float | None
    matrix_size: int
    torus_resolution: tuple[int, int]


_NUMERICAL_GROWTH_TOLERANCE_PER_S = 1.0e3


def _q0_linearization(problem, state: np.ndarray) -> tuple[dict[int, sp.csr_matrix], list[int]]:
    spectral = problem.spectral_tangent_state(problem.tangent_state(state))
    khat = {
        int(key.h): matrix
        for key, matrix in spectral.khat.items()
        if int(key.q) == 0
    }
    ms = sorted(khat)
    return khat, ms


def assess_multitone_stability(
    problem,
    state: np.ndarray,
    *,
    signal_ghz: float | None = None,
    refine: bool = True,
) -> MultitoneStabilityResult:
    """Measure the q=0 Floquet slice of a finite-signal torus.

    The full two-frequency torus is projected onto its pump-periodic q=0
    slice, matching the existing signal Floquet implementation. Results are
    therefore a measured slice diagnostic, not an automatic classifier for
    every incommensurate perturbation.
    """
    khat, ms = _q0_linearization(problem, state)
    if not ms:
        return MultitoneStabilityResult(
            "INCONCLUSIVE", None, None, 0,
            (int(problem.basis.n_p), int(problem.basis.n_delta)),
        )
    if signal_ghz is None:
        signal_ghz = float(problem.basis.signal_tone.omega(problem.basis.omega_p, problem.basis.delta) / (2.0 * math.pi * 1e9))
    estimate = sigma_min_at_signal_ghz(
        problem.circuit,
        khat,
        problem.basis.omega_p,
        signal_ghz,
        ms,
        loss_model=str(problem.loss_model),
        iters=8,
    )
    if not refine or str(problem.loss_model) in NON_ANALYTIC_LOSS_MODELS:
        return MultitoneStabilityResult(
            "STABLE_PROXY" if estimate.sigma_min > 0.0 else "INCONCLUSIVE",
            None,
            float(estimate.sigma_min),
            estimate.matrix_size,
            (int(problem.basis.n_p), int(problem.basis.n_delta)),
        )
    try:
        resonance = refine_complex_resonance(
            problem.circuit,
            khat,
            problem.basis.omega_p,
            ms,
            signal_ghz,
            loss_model=str(problem.loss_model),
            max_iters=12,
        )
    except (TypeError, ValueError, RuntimeError):
        return MultitoneStabilityResult(
            "INCONCLUSIVE", None, float(estimate.sigma_min),
            estimate.matrix_size,
            (int(problem.basis.n_p), int(problem.basis.n_delta)),
        )
    if not resonance.converged:
        return MultitoneStabilityResult(
            "INCONCLUSIVE",
            float(resonance.growth_rate_per_s),
            float(estimate.sigma_min),
            estimate.matrix_size,
            (int(problem.basis.n_p), int(problem.basis.n_delta)),
        )
    return MultitoneStabilityResult(
        "UNSTABLE"
        if resonance.growth_rate_per_s > _NUMERICAL_GROWTH_TOLERANCE_PER_S
        else "STABLE",
        float(resonance.growth_rate_per_s),
        float(estimate.sigma_min),
        estimate.matrix_size,
        (int(problem.basis.n_p), int(problem.basis.n_delta)),
    )
