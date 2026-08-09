"""Independent reproducible multiplicative component scatter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScatterSpec:
    sigma: float = 0.0
    distribution: str = "normal"
    clip_min: float = 0.5
    clip_max: float = 1.5
    mode: str = "independent"

    def __post_init__(self) -> None:
        if self.mode not in {"independent", "plasma_locked"}:
            raise ValueError(f"unknown scatter mode: {self.mode}")


def draw_factors(spec: ScatterSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 0 or spec.sigma < 0:
        raise ValueError("scatter count and sigma must be non-negative")
    if spec.clip_min <= 0 or spec.clip_max < spec.clip_min:
        raise ValueError("scatter clip bounds must be positive and ordered")
    if spec.sigma == 0:
        return np.ones(n, dtype=float)
    if spec.distribution == "normal":
        factors = rng.normal(1.0, spec.sigma, n)
    elif spec.distribution == "uniform":
        width = np.sqrt(3.0) * spec.sigma
        factors = rng.uniform(1.0 - width, 1.0 + width, n)
    else:
        raise ValueError(f"unknown scatter distribution: {spec.distribution}")
    return np.clip(factors, spec.clip_min, spec.clip_max)


def component_rng(master_seed: int, component: str) -> np.random.Generator:
    streams = {"Lj": 0, "Cj": 1, "Cg": 2}
    if component not in streams:
        raise ValueError(f"unknown scatter component: {component}")
    if component == "Lj":
        return np.random.default_rng(int(master_seed))
    return np.random.default_rng(np.random.SeedSequence([int(master_seed), streams[component]]))


def apply_scatter(
    nominal: np.ndarray, spec: ScatterSpec, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    if spec.mode != "independent":
        raise ValueError("plasma_locked is derived and cannot be applied directly")
    nominal = np.asarray(nominal, dtype=float)
    factors = draw_factors(spec, nominal.size, rng)
    values = nominal * factors
    digest = hashlib.blake2b(factors.tobytes(), digest_size=16).hexdigest()
    metadata: dict[str, Any] = {
        "sigma": float(spec.sigma), "distribution": spec.distribution,
        "factor_min": float(factors.min()) if factors.size else None,
        "factor_max": float(factors.max()) if factors.size else None,
        "factor_mean": float(factors.mean()) if factors.size else None,
        "factor_std": float(factors.std()) if factors.size else None,
        "value_min": float(values.min()) if values.size else None,
        "value_max": float(values.max()) if values.size else None,
        "clip_hits": int(np.count_nonzero((factors == spec.clip_min) |
                                           (factors == spec.clip_max))),
        "factor_digest": digest,
        "mode": spec.mode,
    }
    return values, metadata
