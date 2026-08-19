"""Pure regime decisions and a dependency-injected Hill multiplier probe.

The classifier intentionally does not import a circuit or solver module.  The
optional Hill path is assembled only inside :func:`probe_multiplier`, after a
driver has supplied the already-built conversion matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

import numpy as np


class Regime(str, Enum):
    """Regimes understood by the post-Neimark--Sacker route."""

    PERIOD_1 = "PERIOD_1"
    TORUS = "TORUS"
    BROADBAND = "BROADBAND"
    UNDECIDED = "UNDECIDED"


PERIOD_1 = Regime.PERIOD_1
TORUS = Regime.TORUS
BROADBAND = Regime.BROADBAND
UNDECIDED = Regime.UNDECIDED


@dataclass(frozen=True)
class RegimeVerdict:
    """One conservative regime decision and its numerical provenance."""

    regime: Regime
    evidence: float
    margin: float
    reason: str
    mode_overlap: float | None = None
    multiplier: complex | None = None
    searched_imaginary_half_planes: tuple[float, ...] = ()
    mode_vector: np.ndarray | None = None
    signal_ghz: complex | None = None


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _undecided(evidence: float, reason: str, margin: float = 0.0) -> RegimeVerdict:
    return RegimeVerdict(
        regime=Regime.UNDECIDED,
        evidence=float(evidence),
        margin=float(margin),
        reason=reason,
    )


def classify_from_multiplier(
    magnitude: float,
    *,
    tolerance: float = 2.0e-3,
) -> RegimeVerdict:
    """Classify a named Floquet branch by its multiplier magnitude.

    ``margin`` is positive for a decision outside the tolerance band and
    negative inside it.  Keeping the tolerance explicit prevents a caller
    from mistaking a numerical near-crossing for a physical regime change.
    """
    value = _finite_scalar(magnitude, "magnitude")
    if value < 0.0:
        raise ValueError("magnitude must be non-negative")
    band = _finite_scalar(tolerance, "tolerance")
    if band < 0.0 or band >= 1.0:
        raise ValueError("tolerance must lie in [0, 1)")

    lower, upper = 1.0 - band, 1.0 + band
    if value < lower:
        return RegimeVerdict(
            regime=Regime.PERIOD_1,
            evidence=value,
            margin=lower - value,
            reason=f"multiplier magnitude {value:.9g} is below {lower:.9g}",
        )
    if value > upper:
        return RegimeVerdict(
            regime=Regime.TORUS,
            evidence=value,
            margin=value - upper,
            reason=f"multiplier magnitude {value:.9g} is above {upper:.9g}",
        )
    return _undecided(
        value,
        f"multiplier magnitude {value:.9g} lies inside [{lower:.9g}, {upper:.9g}]",
        margin=abs(value - 1.0) - band,
    )


def classify_from_spectrum(
    on_lattice: float,
    generator_share: float,
) -> RegimeVerdict:
    """Classify an FDTD spectrum using the measured ansatz thresholds."""
    lattice = _finite_scalar(on_lattice, "on_lattice")
    generator = _finite_scalar(generator_share, "generator_share")
    if not 0.0 <= lattice <= 1.0:
        raise ValueError("on_lattice must lie in [0, 1]")
    if not 0.0 <= generator <= 1.0:
        raise ValueError("generator_share must lie in [0, 1]")

    if lattice < 0.30:
        return RegimeVerdict(
            regime=Regime.BROADBAND,
            evidence=lattice,
            margin=0.30 - lattice,
            reason=f"on-lattice power {lattice:.9g} is below 0.30",
        )
    if lattice >= 0.90 and generator >= 0.60:
        return RegimeVerdict(
            regime=Regime.TORUS,
            evidence=lattice,
            margin=min(lattice - 0.90, generator - 0.60),
            reason=(
                f"on-lattice power {lattice:.9g} and generator share "
                f"{generator:.9g} meet the torus thresholds"
            ),
        )
    if lattice >= 0.90:
        return RegimeVerdict(
            regime=Regime.PERIOD_1,
            evidence=lattice,
            margin=min(lattice - 0.90, 0.60 - generator),
            reason=(
                f"on-lattice power {lattice:.9g} is high and generator share "
                f"{generator:.9g} is below the torus threshold"
            ),
        )
    return _undecided(
        lattice,
        f"on-lattice power {lattice:.9g} is between the measured regime thresholds",
        margin=-min(lattice - 0.30, 0.90 - lattice),
    )


def route(verdict: RegimeVerdict) -> str:
    """Return the downstream method name without importing a solver."""
    methods = {
        Regime.PERIOD_1: "single_tone_hb",
        Regime.TORUS: "two_frequency_hb",
        Regime.BROADBAND: "fdtd",
        Regime.UNDECIDED: "undecided",
    }
    try:
        regime = Regime(verdict.regime)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown regime {verdict.regime!r}") from exc
    return methods[regime]


def _mode_overlap(previous: Any, current: Any) -> float:
    old = np.asarray(previous, dtype=np.complex128).reshape(-1)
    new = np.asarray(current, dtype=np.complex128).reshape(-1)
    if old.shape != new.shape:
        return 0.0
    old_norm = float(np.linalg.norm(old))
    new_norm = float(np.linalg.norm(new))
    if old_norm == 0.0 or new_norm == 0.0:
        return 0.0
    return float(abs(np.vdot(old, new)) / (old_norm * new_norm))


def _candidate_values(
    candidate: Any,
    *,
    omega_p: float | None,
    seed_signal_ghz: complex,
) -> tuple[complex, np.ndarray | None, bool, float]:
    """Extract a multiplier, mode vector, convergence, and frequency score."""
    classification = getattr(candidate, "classification", None)
    multiplier = getattr(classification, "multiplier", None)
    if multiplier is None:
        multiplier = getattr(candidate, "multiplier", None)

    omega = getattr(candidate, "omega", None)
    signal = getattr(candidate, "signal_ghz", None)
    if multiplier is None and omega is not None:
        if omega_p is None or omega_p <= 0.0:
            raise ValueError("omega_p is required when a candidate exposes omega")
        multiplier = complex(np.exp(1j * complex(omega) * 2.0 * math.pi / omega_p))
    if multiplier is None and isinstance(candidate, (complex, float, int)):
        multiplier = complex(candidate)
    if multiplier is None:
        raise ValueError("Hill candidate does not expose a multiplier or omega")

    if signal is None and omega is not None:
        signal = complex(omega) / (2.0 * math.pi * 1.0e9)
    frequency = (
        float(complex(signal).real)
        if signal is not None
        else float(seed_signal_ghz.real)
    )
    mode_vector = getattr(candidate, "mode_vector", None)
    converged = bool(getattr(candidate, "converged", True))
    return complex(multiplier), mode_vector, converged, frequency


def probe_multiplier(
    magnitude: float | None = None,
    *,
    tolerance: float = 2.0e-3,
    mode_overlap: float | None = None,
    mode_overlap_threshold: float = 0.99,
    circuit: Any | None = None,
    khat: Any | None = None,
    khat_base: Any | None = None,
    omega_p: float | None = None,
    ms: list[int] | None = None,
    seed_signal_ghz: float | complex | None = None,
    loss_model: str | None = None,
    previous_mode_vector: Any | None = None,
    previous_multiplier: complex | None = None,
    refine: Callable[[complex], Any] | None = None,
    imaginary_seed_ghz: float = 1.0e-3,
    both_imaginary_half_planes: bool = True,
    max_iters: int = 30,
    tol: float = 1.0e-9,
) -> RegimeVerdict:
    """Probe a named Hill branch and conservatively classify its multiplier.

    The scalar ``magnitude`` form is useful for unit tests and stored results.
    When a Hill problem is supplied, the probe refines seeds in both imaginary
    half-planes.  It selects by mode overlap with the previous named branch,
    then by previous multiplier or seed frequency.  It never selects the
    branch with the largest multiplier magnitude.

    ``refine`` is an optional one-argument callback used by tests and drivers.
    Without it, the existing ``refine_complex_resonance`` implementation is
    imported lazily and called with the supplied Hill matrices.
    """
    threshold = _finite_scalar(mode_overlap_threshold, "mode_overlap_threshold")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("mode_overlap_threshold must lie in (0, 1]")
    if magnitude is not None and refine is None and circuit is None:
        verdict = classify_from_multiplier(magnitude, tolerance=tolerance)
        if mode_overlap is not None:
            overlap = _finite_scalar(mode_overlap, "mode_overlap")
            verdict = replace(verdict, mode_overlap=overlap)
            if overlap < threshold:
                return replace(
                    _undecided(
                        verdict.evidence,
                        f"mode overlap {overlap:.9g} is below {threshold:.9g}",
                        margin=overlap - threshold,
                    ),
                    mode_overlap=overlap,
                )
        return verdict

    if seed_signal_ghz is None:
        raise ValueError("seed_signal_ghz is required for a Hill probe")
    seed = complex(seed_signal_ghz)
    if not math.isfinite(seed.real) or not math.isfinite(seed.imag):
        raise ValueError("seed_signal_ghz must be finite")
    imag = _finite_scalar(imaginary_seed_ghz, "imaginary_seed_ghz")
    if imag <= 0.0:
        raise ValueError("imaginary_seed_ghz must be positive")
    if both_imaginary_half_planes:
        seeds = (-abs(imag), abs(imag))
    else:
        seeds = (0.0,)

    if refine is None:
        if circuit is None or khat is None or omega_p is None or ms is None:
            raise ValueError(
                "circuit, khat, omega_p, and ms are required for the Hill probe"
            )
        if loss_model is None:
            raise ValueError("loss_model is required for the Hill probe")
        from twpa_solver.signal.stability import refine_complex_resonance

        def refine_candidate(candidate_seed: complex) -> Any:
            return refine_complex_resonance(
                circuit=circuit,
                khat=khat,
                khat_base=khat_base,
                omega_p=float(omega_p),
                ms=ms,
                signal_ghz_guess=candidate_seed,
                loss_model=loss_model,
                max_iters=max_iters,
                tol=tol,
                v0=previous_mode_vector,
            )

        refine = refine_candidate

    candidates: list[tuple[complex, np.ndarray | None, bool, float, float]] = []
    errors: list[str] = []
    for imaginary in seeds:
        try:
            candidate = refine(complex(seed.real, imaginary))
            multiplier, vector, converged, frequency = _candidate_values(
                candidate,
                omega_p=omega_p,
                seed_signal_ghz=seed,
            )
            overlap = (
                _mode_overlap(previous_mode_vector, vector)
                if previous_mode_vector is not None and vector is not None
                else float(getattr(candidate, "mode_overlap", 1.0))
            )
            candidates.append((multiplier, vector, converged, frequency, overlap))
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"imaginary seed {imaginary:+g}: {exc}")

    if not candidates:
        reason = "Hill refinement returned no usable candidate"
        if errors:
            reason += "; " + "; ".join(errors)
        return replace(
            _undecided(float("nan"), reason),
            searched_imaginary_half_planes=tuple(seeds),
        )

    def selection_key(item: tuple[complex, np.ndarray | None, bool, float, float]) -> tuple[float, float]:
        multiplier, vector, _converged, frequency, overlap = item
        if previous_mode_vector is not None and vector is not None:
            return (overlap, 0.0)
        if previous_multiplier is not None:
            return (
                -abs(multiplier - previous_multiplier),
                -abs(frequency - seed.real),
            )
        return (-abs(frequency - seed.real), -abs(complex(multiplier)))

    selected = max(candidates, key=selection_key)
    selected_multiplier, selected_vector, converged, selected_frequency, selected_overlap = selected
    evidence = float(abs(selected_multiplier))
    if not converged:
        verdict = _undecided(evidence, "selected Hill branch did not converge")
    elif selected_overlap < threshold:
        verdict = _undecided(
            evidence,
            f"mode overlap {selected_overlap:.9g} is below {threshold:.9g}",
            margin=selected_overlap - threshold,
        )
    else:
        verdict = classify_from_multiplier(evidence, tolerance=tolerance)
        verdict = replace(
            verdict,
            reason=f"{verdict.reason}; named Hill branch selected by mode overlap",
        )
    return replace(
        verdict,
        mode_overlap=float(selected_overlap),
        multiplier=selected_multiplier,
        searched_imaginary_half_planes=tuple(seeds),
        mode_vector=selected_vector,
        signal_ghz=complex(selected_frequency),
    )
