"""Independent diagnostics for nonlinear pump states.

The diagnostics in this module do not change the circuit equations or decide
whether a branch is physically admissible.  They provide the observables
needed to separate a validated high-current state from a failed Newton
iterate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twpa_solver.pump.problem import FullPumpProblem


def _critical_current(problem: FullPumpProblem) -> np.ndarray | None:
    """Return branch critical currents when the selected law exposes them."""
    for name in ("critical_current", "Ic"):
        value = getattr(problem.branch, name, None)
        if value is not None:
            current = np.asarray(value, dtype=float).reshape(-1)
            if current.size == problem.nb:
                return current
    return None


def branch_stress_metrics(
    problem: FullPumpProblem,
    X: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure junction current and tangent margins on a pump waveform.

    ``max_abs_current_over_ic`` is an observable, not a solver stopping rule.
    An ideal Josephson current law is bounded by ``Ic`` and therefore does not
    model quasiparticle switching or junction destruction.  ``min_cos_phase``
    reports the smallest Josephson tangent factor and is useful for detecting
    the conditioning loss near a zero differential inductance.
    """
    psi_dynamic = problem.branch_flux_time(X)
    psi_total = psi_dynamic + problem.dc_branch_flux[None, :]
    current = np.asarray(problem.branch.current(psi_total), dtype=float)
    tangent_fn = getattr(problem.branch, "tangent", None)
    if tangent_fn is None:
        tangent_fn = problem.branch.gamma
    tangent = np.asarray(tangent_fn(psi_total), dtype=float)
    peak_by_branch = np.max(np.abs(current), axis=0)
    strongest = int(np.argmax(peak_by_branch))

    critical = _critical_current(problem)
    if critical is None:
        ratio = None
        ratio_max = None
    else:
        ratio_by_branch = peak_by_branch / np.maximum(np.abs(critical), 1e-300)
        # The FDTD diagnostic reports max(|sin(phi)|), not the ratio on the
        # branch carrying the largest absolute current.  These are identical
        # for uniform-Ic devices, but differ for flux-pumped devices whose Ic
        # varies along the branch set.
        ratio = float(np.max(ratio_by_branch))
        ratio_max = float(np.max(ratio_by_branch))

    phi0 = getattr(problem.branch, "phi0", None)
    if phi0 is None:
        phase_max = None
        min_cos = None
    else:
        phase_reduced = psi_total / float(phi0)
        phase_max = float(np.max(np.abs(phase_reduced[:, strongest])))
        min_cos = float(np.min(np.cos(phase_reduced)))
    return {
        "branch_current_max_abs": float(peak_by_branch[strongest]),
        "branch_current_max_over_ic": ratio,
        "branch_current_max_over_ic_all": ratio_max,
        "strongest_branch_index": strongest,
        "strongest_branch_phase_max_abs_rad": phase_max,
        "branch_min_cos_phase": min_cos,
        "branch_min_tangent_abs": float(np.min(np.abs(tangent))),
    }


def branch_current_profile(problem: FullPumpProblem, X: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-junction current statistics for an HB waveform.

    This is the expanded counterpart of :func:`branch_stress_metrics`, intended
    for circuit-location plots and independent inspection.  No branch is
    clipped or omitted.  The peak current is the maximum absolute instantaneous
    current over the reconstructed retained waveform.
    """
    X = np.asarray(X, dtype=np.complex128)
    psi_dynamic = problem.branch_flux_time(X)
    dc_flux = np.asarray(
        getattr(problem, "dc_branch_flux", np.zeros(psi_dynamic.shape[-1])),
        dtype=np.float64,
    ).reshape(-1)
    current = np.asarray(
        problem.branch.current(psi_dynamic + dc_flux[None, :]),
        dtype=np.float64,
    )
    critical = _critical_current(problem)
    if critical is None or critical.size != current.shape[-1]:
        critical = np.full(current.shape[-1], np.nan, dtype=np.float64)
    peak_abs = np.max(np.abs(current), axis=0)
    ratio = np.divide(
        peak_abs,
        critical,
        out=np.full_like(peak_abs, np.nan),
        where=np.isfinite(critical) & (critical != 0.0),
    )
    return {
        "peak_abs_current_a": peak_abs,
        "rms_current_a": np.sqrt(np.mean(current * current, axis=0)),
        "mean_current_a": np.mean(current, axis=0),
        "critical_current_a": critical,
        "peak_ratio_ic": ratio,
        "peak_time_index": np.argmax(np.abs(current), axis=0).astype(np.int64),
    }


def residual_spectrum_summary(
    problem: FullPumpProblem,
    X: np.ndarray,
    source_scale: float,
    *,
    max_modes: int = 8,
) -> dict[str, Any]:
    """Return compact residual-spectrum telemetry for persisted reports."""
    spectrum = problem.residual_spectrum(X, source_scale)
    modes = np.asarray(spectrum["modes"], dtype=int)
    mode_rel = np.asarray(spectrum["mode_rel"], dtype=float)
    retained = {int(round(mode)) for mode in problem.grid.k}
    omitted = [
        {"mode": int(mode), "rel": float(value)}
        for mode, value in zip(modes, mode_rel)
        if int(mode) not in retained and np.isfinite(value)
    ]
    omitted.sort(key=lambda item: item["rel"], reverse=True)
    return {
        "nt": int(round(float(spectrum["nt"]))),
        "retained_modes": sorted(retained),
        "dominant_omitted_modes": omitted[:max_modes],
        "max_omitted_mode_rel": (
            float(omitted[0]["rel"]) if omitted else 0.0
        ),
    }
