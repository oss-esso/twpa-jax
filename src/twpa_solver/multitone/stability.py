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
    reason: str = ""


_NUMERICAL_GROWTH_TOLERANCE_PER_S = 1.0e3

# A sideband this close to DC makes the conversion matrix nearly singular for
# reasons that have nothing to do with parametric gain: the circuit has no DC
# path, so its dynamic block collapses as omega -> 0. Measured on the jpa
# fixture (pump 4.75001 GHz, signal 4.75 GHz), the m=-1 sideband lands at
# -10 kHz and owns the minimal singular direction with mass 1.000000, five
# orders of magnitude below the next singular value -- and the resulting
# sigma_min is bit-identical whether the pump is on or off, even at
# psi/phi0 = 57000. Reporting that as STABLE is reporting the linear circuit.
_NEAR_DC_SIDEBAND_FRACTION = 1.0e-3


def _q0_linearization(problem, state: np.ndarray) -> tuple[dict[int, sp.csr_matrix], list[int]]:
    """Project the torus tangent onto its pump-periodic slice.

    ``khat`` is keyed by the Fourier index ``ell``; the sideband ladder ``ms``
    is a separate axis, and ``assemble_conversion_matrix`` couples them as
    ``ell = m - q``. The two must not be conflated: passing the khat keys as
    the ladder yields a ragged, asymmetric sideband set whose smallest
    singular value is set by rows the parametric coupling never reaches, so
    the resulting sigma_min does not respond to the pump state at all.

    The ladder is therefore built symmetric and contiguous, matching
    ``signal/floquet.py::sideband_list``. ``S = max_ell // 2`` is the inverse
    of the reference relation ``max_ell = max|m - q|`` over the ladder, so
    every block the assembly asks for is inside khat's support.
    """
    spectral = problem.spectral_tangent_state(problem.tangent_state(state))
    khat = {
        int(key.h): matrix
        for key, matrix in spectral.khat.items()
        if int(key.q) == 0
    }
    if not khat:
        return khat, []
    max_ell = max(abs(ell) for ell in khat)
    sidebands = max(1, max_ell // 2)
    ms = list(range(-sidebands, sidebands + 1))
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
    omega_p = float(problem.basis.omega_p)
    omega_s = 2.0 * math.pi * float(signal_ghz) * 1e9
    near_dc = [
        m for m in ms
        if abs(omega_s + m * omega_p) < _NEAR_DC_SIDEBAND_FRACTION * omega_p
    ]
    if near_dc:
        return MultitoneStabilityResult(
            "INCONCLUSIVE", None, None, 0,
            (int(problem.basis.n_p), int(problem.basis.n_delta)),
            reason=(
                f"sideband m={near_dc[0]} sits at "
                f"{(omega_s + near_dc[0] * omega_p) / (2.0 * math.pi):.6g} Hz, "
                "within the near-DC band where the conversion matrix is "
                "singular independently of the pump; sigma_min there does not "
                "measure parametric stability"
            ),
        )
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
