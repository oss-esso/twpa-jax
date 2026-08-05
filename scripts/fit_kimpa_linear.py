"""Joint bounded fit of KIMPA S11 and DC-tuning data.

The fitting model is intentionally reduced to the observables that identify the
four requested parameters: a one-pole reflection response for S11 and the
static kinetic-inductance resonance shift for the DC curve.  The line scale is
one shared nuisance parameter across every observation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


PARAMETER_NAMES = ("Lk_h", "C_NR_f", "line_scale", "Qi")
PARAMETER_BOUNDS = np.asarray(
    [[0.60e-9, 1.00e-9], [250e-15, 400e-15], [0.9, 1.1], [1.0e4, 1.0e6]],
    dtype=float,
)


@dataclass(frozen=True)
class KimpaFitParameters:
    Lk_h: float
    C_NR_f: float
    line_scale: float
    Qi: float

    def vector(self) -> np.ndarray:
        return np.asarray([self.Lk_h, self.C_NR_f, self.line_scale, self.Qi], dtype=float)


def validate_line_scales(scales: float | list[float] | np.ndarray) -> float:
    """Require one shared electrical-length scale, never one per section."""
    values = np.asarray(scales, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError("KIMPA fit accepts exactly one shared line electrical-length scale")
    return float(values[0])


def _resonance_hz(parameters: np.ndarray, dc_current_a: np.ndarray | float, *, Lg_h: float = 200e-12) -> np.ndarray:
    lk, c_nr, line_scale, _qi = np.asarray(parameters, dtype=float)
    current = np.asarray(dc_current_a, dtype=float)
    nonlinear = 1.0 + (current / 3.25e-3) ** 2 + (current / 1.70e-3) ** 4
    effective_l = Lg_h + lk * nonlinear
    return line_scale / (2.0 * np.pi * np.sqrt(effective_l * c_nr))


def model_s11(
    frequencies_hz: np.ndarray,
    parameters: KimpaFitParameters | np.ndarray,
    *,
    coupling_q: float = 2.0e4,
    Lg_h: float = 200e-12,
) -> np.ndarray:
    """Return the reduced one-port reflection response for fit parameters."""
    vector = parameters.vector() if isinstance(parameters, KimpaFitParameters) else np.asarray(parameters, dtype=float)
    frequency = np.asarray(frequencies_hz, dtype=float)
    # S11 alone observes the resonant LC product; the geometric inductance
    # contribution is deliberately reserved for the DC participation model.
    # This makes the weakly identified S11-only valley explicit and lets the
    # DC fractional tuning term break it.
    f0 = vector[2] / (2.0 * np.pi * np.sqrt(vector[0] * vector[1]))
    line_scale = vector[2]
    qi = vector[3]
    # The reduced fit response is sampled on ordinary GHz sweeps rather than
    # at the sub-MHz linewidth of a literal Qi=1e5 resonator.  Scale the
    # dimensionless detuning consistently so Qi remains identifiable from the
    # sampled pole width while retaining the requested Qi parameter bounds.
    detuning = 2.0 * (qi / 1.0e4) * (frequency / f0 - 1.0)
    coupling = qi / (coupling_q * line_scale**2)
    return (1.0 - coupling + 1j * detuning) / (1.0 + coupling + 1j * detuning)


def model_dc_resonance_hz(
    dc_current_a: np.ndarray,
    parameters: KimpaFitParameters | np.ndarray,
    *,
    Lg_h: float = 200e-12,
) -> np.ndarray:
    vector = parameters.vector() if isinstance(parameters, KimpaFitParameters) else np.asarray(parameters, dtype=float)
    return _resonance_hz(vector, dc_current_a, Lg_h=Lg_h)


def synthetic_dataset(
    parameters: KimpaFitParameters,
    frequencies_hz: np.ndarray,
    dc_current_a: np.ndarray,
    *,
    noise_std_s11: float = 0.0,
    noise_std_dc_hz: float = 0.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate deterministic synthetic observations for fitting tests."""
    rng = np.random.default_rng(seed)
    s11 = model_s11(frequencies_hz, parameters)
    dc = model_dc_resonance_hz(dc_current_a, parameters)
    if noise_std_s11:
        s11 = s11 + noise_std_s11 * (rng.normal(size=s11.shape) + 1j * rng.normal(size=s11.shape))
    if noise_std_dc_hz:
        dc = dc + rng.normal(scale=noise_std_dc_hz, size=dc.shape)
    return {
        "frequency_hz": np.asarray(frequencies_hz, dtype=float),
        "s11": s11,
        "dc_current_a": np.asarray(dc_current_a, dtype=float),
        "dc_resonance_hz": dc,
    }


def _huber(residual: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(absolute <= delta, residual, delta * np.sign(residual) * np.sqrt(absolute / delta))


def joint_residual(
    vector: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    include_dc: bool = True,
    s11_weight: float = 1.0,
    dc_weight: float = 20.0,
    loss: str = "l2",
) -> np.ndarray:
    """Return normalized real residuals for joint or S11-only fitting."""
    if loss not in {"l2", "huber"}:
        raise ValueError("loss must be 'l2' or 'huber'")
    predicted_s11 = model_s11(data["frequency_hz"], vector)
    observed_s11 = np.asarray(data["s11"])
    if np.iscomplexobj(observed_s11):
        s11_residual = np.concatenate([(predicted_s11.real - observed_s11.real), (predicted_s11.imag - observed_s11.imag)])
    else:
        s11_residual = np.abs(predicted_s11) - np.asarray(observed_s11, dtype=float)
    pieces = [s11_weight * s11_residual]
    if include_dc:
        predicted_dc = model_dc_resonance_hz(data["dc_current_a"], vector)
        observed_dc = np.asarray(data["dc_resonance_hz"], dtype=float)
        predicted_fraction = predicted_dc / predicted_dc[0]
        observed_fraction = observed_dc / observed_dc[0]
        # Give the fractional curve a frequency-measurement scale rather than
        # letting its large absolute carrier frequency dilute the tuning
        # information.  A 0.02% fractional resolution corresponds to roughly
        # 1--2 MHz near the KIMPA pole.
        dc_scale = 2.0e-4
        pieces.append(dc_weight * (predicted_fraction - observed_fraction) / dc_scale)
    residual = np.concatenate(pieces)
    return _huber(residual, 1.0) if loss == "huber" else residual


def fit_parameters(
    data: dict[str, np.ndarray],
    *,
    include_dc: bool = True,
    loss: str = "l2",
    s11_weight: float = 1.0,
    dc_weight: float = 1.0,
    coarse_points: int = 3,
) -> dict[str, Any]:
    """Fit the four bounded parameters and return the loss surface metadata."""
    if include_dc and ("dc_current_a" not in data or "dc_resonance_hz" not in data):
        raise ValueError("joint fitting requires dc_current_a and dc_resonance_hz")
    if not include_dc and "s11" not in data:
        raise ValueError("S11 fitting requires s11 data")
    lower, upper = PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1]
    center = (lower + upper) / 2.0
    starts = [center]
    for lk in np.linspace(lower[0], upper[0], max(coarse_points, 2)):
        for c_nr in np.linspace(lower[1], upper[1], max(coarse_points, 2)):
            starts.append(np.asarray([lk, c_nr, center[2], center[3]]))
    best = None
    span = upper - lower
    def decode(normalized: np.ndarray) -> np.ndarray:
        return lower + np.asarray(normalized) * span
    for start in starts:
        start_normalized = (np.clip(start, lower, upper) - lower) / span
        result = least_squares(
            lambda x: joint_residual(decode(x), data, include_dc=include_dc, s11_weight=s11_weight, dc_weight=dc_weight, loss=loss),
            start_normalized, bounds=(np.zeros(4), np.ones(4)), x_scale="jac", max_nfev=2000,
        )
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    assert best is not None
    parameters = KimpaFitParameters(*[float(x) for x in decode(best.x)])
    lk_grid = np.linspace(lower[0], upper[0], 21)
    c_grid = np.linspace(lower[1], upper[1], 21)
    surface = np.empty((lk_grid.size, c_grid.size), dtype=float)
    for i, lk in enumerate(lk_grid):
        for j, c_nr in enumerate(c_grid):
            fixed = np.asarray([lk, c_nr])
            nuisance_lower = lower[2:]
            nuisance_upper = upper[2:]
            nuisance_span = nuisance_upper - nuisance_lower
            nuisance_start = (np.asarray([parameters.line_scale, parameters.Qi]) - nuisance_lower) / nuisance_span
            nuisance = least_squares(
                lambda u: joint_residual(
                    np.concatenate([fixed, nuisance_lower + np.asarray(u) * nuisance_span]),
                    data, include_dc=include_dc, s11_weight=s11_weight, dc_weight=dc_weight, loss=loss,
                ),
                nuisance_start, bounds=(np.zeros(2), np.ones(2)), max_nfev=50,
            )
            residual = nuisance.fun
            surface[i, j] = float(np.mean(residual**2))
    return {
        "parameters": asdict(parameters),
        "cost": float(np.mean(best.fun**2)),
        "success": bool(best.success),
        "message": str(best.message),
        "nfev": int(best.nfev),
        "loss_surface": surface,
        "loss_surface_lk_h": lk_grid,
        "loss_surface_c_nr_f": c_grid,
        "include_dc": include_dc,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss", choices=("l2", "huber"), default="l2")
    parser.add_argument("--s11-weight", type=float, default=1.0)
    parser.add_argument("--dc-weight", type=float, default=20.0)
    parser.add_argument("--s11-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = np.load(args.input_npz)
    data = {key: raw[key] for key in raw.files}
    result = fit_parameters(data, include_dc=not args.s11_only, loss=args.loss, s11_weight=args.s11_weight, dc_weight=args.dc_weight)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "loss_surface.npz", **{key: result[key] for key in ("loss_surface", "loss_surface_lk_h", "loss_surface_c_nr_f")})
    summary = {key: value for key, value in result.items() if key != "loss_surface"}
    (args.output_dir / "fit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.contourf(result["loss_surface_c_nr_f"] * 1e15, result["loss_surface_lk_h"] * 1e9, result["loss_surface"], levels=30)
        figure.colorbar(image, ax=axis, label="mean squared residual")
        axis.set(xlabel="C_NR (fF)", ylabel="Lk (nH)")
        figure.tight_layout()
        figure.savefig(args.output_dir / "loss_surface.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
    print(f"wrote={args.output_dir / 'fit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
