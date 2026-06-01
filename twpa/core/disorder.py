"""
twpa.core.disorder
==================

Fabrication-disorder and parameter-variation utilities.

This module applies physically interpretable perturbations to vectorized
LineLayout objects:

    L_series_H[n]
    C_shunt_F[n]
    R_series_ohm[n]
    G_shunt_S[n]
    C_stub_F[n]

It supports:
- global process shifts,
- uncorrelated cell-to-cell variation,
- spatially correlated variation,
- linear tapers,
- localized defects / hotspots,
- Monte Carlo sample generation.

Why this exists
---------------
A production-grade TWPA simulator should not only simulate an ideal designed
line. It must also represent as-fabricated variability. This matters for:
- fitting measured pump-off S-parameters,
- gain-map uncertainty,
- synthetic recovery tests,
- BRIDGE dataset generation,
- robustness studies.

The functions here are deterministic given a PRNG key and config. They do not
perform any circuit simulation; they only produce perturbed layouts and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from .layout import LineLayout, apply_multiplicative_cell_perturbations
from .params import RuntimeConfig


ArrayLike = Any


# ---------------------------------------------------------------------------
# Enums / checks
# ---------------------------------------------------------------------------

class DistributionKind(str, Enum):
    """Supported scalar/random-field distributions."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    NONE = "none"


class CorrelationKernel(str, Enum):
    """Supported spatial-correlation kernels."""

    GAUSSIAN = "gaussian"
    BOXCAR = "boxcar"
    NONE = "none"


def _check_nonnegative(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _check_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {
            "array_shape": tuple(int(s) for s in obj.shape),
            "array_dtype": str(obj.dtype),
        }
    return obj


# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RandomFieldConfig:
    """
    Configuration for a multiplicative random field.

    The generated field is a positive multiplicative factor with mean close to 1.

    Parameters
    ----------
    enabled:
        Whether this field is applied.
    distribution:
        Random distribution before optional spatial filtering.
    std_fraction:
        Relative standard deviation. For normal fields, the raw factor is
        approximately 1 + std_fraction * N(0, 1). For lognormal fields, this is
        used as the log-space sigma.
    correlation_length_cells:
        Approximate smoothing scale in cells. If <= 0 or kernel is none, the
        field is uncorrelated cell-by-cell.
    kernel:
        Smoothing kernel.
    clip_min, clip_max:
        Final multiplicative factor clipping bounds.
    """

    enabled: bool = False
    distribution: DistributionKind = DistributionKind.LOGNORMAL
    std_fraction: float = 0.0
    correlation_length_cells: float = 0.0
    kernel: CorrelationKernel = CorrelationKernel.GAUSSIAN
    clip_min: float = 0.01
    clip_max: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "distribution", DistributionKind(self.distribution))
        object.__setattr__(self, "kernel", CorrelationKernel(self.kernel))
        _check_nonnegative("std_fraction", self.std_fraction)
        _check_nonnegative("correlation_length_cells", self.correlation_length_cells)
        _check_positive("clip_min", self.clip_min)
        _check_positive("clip_max", self.clip_max)
        if self.clip_max <= self.clip_min:
            raise ValueError("clip_max must exceed clip_min")

    def with_updates(self, **kwargs: Any) -> "RandomFieldConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class GlobalShiftConfig:
    """
    Deterministic global multiplicative process shifts.
    """

    L_scale: float = 1.0
    C_scale: float = 1.0
    R_scale: float = 1.0
    G_scale: float = 1.0
    C_stub_scale: float = 1.0

    def __post_init__(self) -> None:
        _check_positive("L_scale", self.L_scale)
        _check_positive("C_scale", self.C_scale)
        _check_nonnegative("R_scale", self.R_scale)
        _check_nonnegative("G_scale", self.G_scale)
        _check_nonnegative("C_stub_scale", self.C_stub_scale)

    def with_updates(self, **kwargs: Any) -> "GlobalShiftConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class TaperConfig:
    """
    Deterministic linear or quadratic taper across the chip.

    The taper factor is

        factor(x) = 1 + amplitude * u

    for linear, where u ranges from -1 to +1, or

        factor(x) = 1 + amplitude * (u^2 - mean(u^2))

    for quadratic.

    Parameters
    ----------
    enabled:
        Whether taper is applied.
    target:
        Which layout field to taper.
    amplitude_fraction:
        Relative taper amplitude.
    profile:
        "linear" or "quadratic".
    """

    enabled: bool = False
    target: Literal["L", "C", "R", "G", "C_stub"] = "L"
    amplitude_fraction: float = 0.0
    profile: Literal["linear", "quadratic"] = "linear"

    def __post_init__(self) -> None:
        if self.target not in {"L", "C", "R", "G", "C_stub"}:
            raise ValueError(f"Unsupported taper target {self.target!r}")
        if self.profile not in {"linear", "quadratic"}:
            raise ValueError(f"Unsupported taper profile {self.profile!r}")
        if abs(self.amplitude_fraction) >= 1.0:
            raise ValueError("abs(amplitude_fraction) must be < 1")

    def with_updates(self, **kwargs: Any) -> "TaperConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class HotspotConfig:
    """
    Localized defect/hotspot perturbation.

    Parameters
    ----------
    enabled:
        Whether the hotspot is applied.
    target:
        Which parameter is affected.
    center_fraction:
        Position along the line in [0, 1].
    width_cells:
        Gaussian width in cells.
    amplitude_fraction:
        Relative perturbation. Positive increases the parameter.
    """

    enabled: bool = False
    target: Literal["L", "C", "R", "G", "C_stub"] = "R"
    center_fraction: float = 0.5
    width_cells: float = 10.0
    amplitude_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.target not in {"L", "C", "R", "G", "C_stub"}:
            raise ValueError(f"Unsupported hotspot target {self.target!r}")
        if not (0.0 <= float(self.center_fraction) <= 1.0):
            raise ValueError("center_fraction must be in [0, 1]")
        _check_positive("width_cells", self.width_cells)
        if self.amplitude_fraction <= -1.0:
            raise ValueError("amplitude_fraction must be > -1 to keep factors positive")

    def with_updates(self, **kwargs: Any) -> "HotspotConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class DisorderConfig:
    """
    Complete layout-disorder configuration.

    Each random field is multiplicative and independent unless the caller uses
    correlated PRNG keys externally. A typical synthetic study may use:

        L_global_std_fraction ~ 1-5%
        C_global_std_fraction ~ 1-5%
        L_cell.std_fraction   ~ 0.1-2%
        C_cell.std_fraction   ~ 0.1-2%

    The actual values should be adjusted based on material/process data.
    """

    enabled: bool = False
    global_shift: GlobalShiftConfig = GlobalShiftConfig()

    random_L: RandomFieldConfig = RandomFieldConfig()
    random_C: RandomFieldConfig = RandomFieldConfig()
    random_R: RandomFieldConfig = RandomFieldConfig()
    random_G: RandomFieldConfig = RandomFieldConfig()
    random_C_stub: RandomFieldConfig = RandomFieldConfig()

    taper: TaperConfig = TaperConfig()
    hotspot: HotspotConfig = HotspotConfig()

    sample_id: int = 0
    name: str = "disorder"

    def with_updates(self, **kwargs: Any) -> "DisorderConfig":
        return replace(self, **kwargs)

    @classmethod
    def none(cls) -> "DisorderConfig":
        """No disorder."""
        return cls(enabled=False, name="none")

    @classmethod
    def mild_lc_process(
        cls,
        *,
        L_std: float = 0.01,
        C_std: float = 0.01,
        corr_cells: float = 25.0,
        sample_id: int = 0,
    ) -> "DisorderConfig":
        """
        Mild correlated L/C fabrication variation.
        """
        return cls(
            enabled=True,
            random_L=RandomFieldConfig(
                enabled=True,
                distribution=DistributionKind.LOGNORMAL,
                std_fraction=L_std,
                correlation_length_cells=corr_cells,
                kernel=CorrelationKernel.GAUSSIAN,
                clip_min=0.5,
                clip_max=2.0,
            ),
            random_C=RandomFieldConfig(
                enabled=True,
                distribution=DistributionKind.LOGNORMAL,
                std_fraction=C_std,
                correlation_length_cells=corr_cells,
                kernel=CorrelationKernel.GAUSSIAN,
                clip_min=0.5,
                clip_max=2.0,
            ),
            sample_id=sample_id,
            name="mild_lc_process",
        )

    @classmethod
    def hotspot_resistance(
        cls,
        *,
        center_fraction: float = 0.5,
        width_cells: float = 20.0,
        amplitude_fraction: float = 10.0,
        sample_id: int = 0,
    ) -> "DisorderConfig":
        """
        Localized series-resistance hotspot.
        """
        return cls(
            enabled=True,
            hotspot=HotspotConfig(
                enabled=True,
                target="R",
                center_fraction=center_fraction,
                width_cells=width_cells,
                amplitude_fraction=amplitude_fraction,
            ),
            sample_id=sample_id,
            name="hotspot_resistance",
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(
            {
                "enabled": self.enabled,
                "global_shift": self.global_shift,
                "random_L": self.random_L,
                "random_C": self.random_C,
                "random_R": self.random_R,
                "random_G": self.random_G,
                "random_C_stub": self.random_C_stub,
                "taper": self.taper,
                "hotspot": self.hotspot,
                "sample_id": self.sample_id,
                "name": self.name,
            }
        )


# ---------------------------------------------------------------------------
# Random field generation
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(sigma_cells: float, *, truncate: float = 4.0) -> jax.Array:
    """
    Build a normalized 1D Gaussian kernel.

    sigma_cells <= 0 returns [1].
    """
    if sigma_cells <= 0.0:
        return jnp.asarray([1.0], dtype=jnp.float64)
    radius = max(1, int(jnp.ceil(truncate * sigma_cells)))
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float64)
    kernel = jnp.exp(-0.5 * (x / sigma_cells) ** 2)
    return kernel / jnp.sum(kernel)


def boxcar_kernel_1d(width_cells: float) -> jax.Array:
    """
    Build a normalized boxcar kernel.

    width_cells <= 1 returns [1].
    """
    if width_cells <= 1.0:
        return jnp.asarray([1.0], dtype=jnp.float64)
    width = max(1, int(jnp.round(width_cells)))
    if width % 2 == 0:
        width += 1
    kernel = jnp.ones((width,), dtype=jnp.float64)
    return kernel / jnp.sum(kernel)


def smooth_field_1d(
    field: ArrayLike,
    *,
    kernel: CorrelationKernel | str,
    correlation_length_cells: float,
) -> jax.Array:
    """
    Smooth a 1D field using a normalized kernel.

    Edge handling uses padding by edge values.
    """
    x = jnp.asarray(field, dtype=jnp.float64)
    if x.ndim != 1:
        raise ValueError(f"field must be 1D, got shape {x.shape}")

    kernel = CorrelationKernel(kernel)
    if kernel == CorrelationKernel.NONE or correlation_length_cells <= 0.0:
        return x
    if kernel == CorrelationKernel.GAUSSIAN:
        k = gaussian_kernel_1d(correlation_length_cells)
    elif kernel == CorrelationKernel.BOXCAR:
        k = boxcar_kernel_1d(correlation_length_cells)
    else:
        raise ValueError(f"Unsupported kernel {kernel}")

    radius = int(k.shape[0] // 2)
    padded = jnp.pad(x, (radius, radius), mode="edge")
    return jnp.convolve(padded, k, mode="valid")


def standardize_field(field: ArrayLike, *, eps: float = 1e-12) -> jax.Array:
    """
    Return a zero-mean, unit-std version of a field.

    If the standard deviation is too small, returns zeros.
    """
    x = jnp.asarray(field, dtype=jnp.float64)
    centered = x - jnp.mean(x)
    std = jnp.std(centered)
    return jnp.where(std > eps, centered / std, jnp.zeros_like(centered))


def random_multiplicative_field(
    key: jax.Array,
    n: int,
    config: RandomFieldConfig,
) -> jax.Array:
    """
    Generate a positive multiplicative random field of length n.

    If config.enabled is false or std_fraction is zero, returns ones.
    """
    if int(n) <= 0:
        raise ValueError("n must be positive")
    n = int(n)

    if (not config.enabled) or config.distribution == DistributionKind.NONE or config.std_fraction == 0.0:
        return jnp.ones((n,), dtype=jnp.float64)

    dist = DistributionKind(config.distribution)

    if dist == DistributionKind.NORMAL:
        raw = jax.random.normal(key, (n,), dtype=jnp.float64)
        raw = smooth_field_1d(
            raw,
            kernel=config.kernel,
            correlation_length_cells=config.correlation_length_cells,
        )
        raw = standardize_field(raw)
        factor = 1.0 + config.std_fraction * raw

    elif dist == DistributionKind.LOGNORMAL:
        raw = jax.random.normal(key, (n,), dtype=jnp.float64)
        raw = smooth_field_1d(
            raw,
            kernel=config.kernel,
            correlation_length_cells=config.correlation_length_cells,
        )
        raw = standardize_field(raw)
        # Shift by -sigma^2/2 so the mean is approximately one before clipping.
        sigma = config.std_fraction
        factor = jnp.exp(sigma * raw - 0.5 * sigma**2)

    elif dist == DistributionKind.UNIFORM:
        raw = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0, dtype=jnp.float64)
        raw = smooth_field_1d(
            raw,
            kernel=config.kernel,
            correlation_length_cells=config.correlation_length_cells,
        )
        raw = standardize_field(raw)
        factor = 1.0 + config.std_fraction * raw

    else:
        raise ValueError(f"Unsupported distribution {dist}")

    return jnp.clip(factor, config.clip_min, config.clip_max)


# ---------------------------------------------------------------------------
# Deterministic spatial perturbations
# ---------------------------------------------------------------------------

def taper_factor(
    n: int,
    config: TaperConfig,
) -> jax.Array:
    """
    Generate a deterministic taper factor of length n.
    """
    if int(n) <= 0:
        raise ValueError("n must be positive")
    n = int(n)

    if not config.enabled or config.amplitude_fraction == 0.0:
        return jnp.ones((n,), dtype=jnp.float64)

    u = jnp.linspace(-1.0, 1.0, n)

    if config.profile == "linear":
        profile = u
    elif config.profile == "quadratic":
        raw = u**2
        profile = raw - jnp.mean(raw)
        profile = profile / jnp.max(jnp.abs(profile))
    else:
        raise ValueError(f"Unsupported taper profile {config.profile!r}")

    factor = 1.0 + config.amplitude_fraction * profile
    return jnp.maximum(factor, 1e-12)


def hotspot_factor(
    n: int,
    config: HotspotConfig,
) -> jax.Array:
    """
    Generate a localized Gaussian hotspot factor of length n.
    """
    if int(n) <= 0:
        raise ValueError("n must be positive")
    n = int(n)

    if not config.enabled or config.amplitude_fraction == 0.0:
        return jnp.ones((n,), dtype=jnp.float64)

    center = config.center_fraction * (n - 1)
    idx = jnp.arange(n, dtype=jnp.float64)
    profile = jnp.exp(-0.5 * ((idx - center) / config.width_cells) ** 2)
    factor = 1.0 + config.amplitude_fraction * profile
    return jnp.maximum(factor, 1e-12)


def _target_factors_from_taper(
    n: int,
    config: TaperConfig,
) -> dict[str, jax.Array]:
    one = jnp.ones((n,), dtype=jnp.float64)
    if not config.enabled:
        return {"L": one, "C": one, "R": one, "G": one, "C_stub": one}
    fac = taper_factor(n, config)
    return {
        "L": fac if config.target == "L" else one,
        "C": fac if config.target == "C" else one,
        "R": fac if config.target == "R" else one,
        "G": fac if config.target == "G" else one,
        "C_stub": fac if config.target == "C_stub" else one,
    }


def _target_factors_from_hotspot(
    n: int,
    config: HotspotConfig,
) -> dict[str, jax.Array]:
    one = jnp.ones((n,), dtype=jnp.float64)
    if not config.enabled:
        return {"L": one, "C": one, "R": one, "G": one, "C_stub": one}
    fac = hotspot_factor(n, config)
    return {
        "L": fac if config.target == "L" else one,
        "C": fac if config.target == "C" else one,
        "R": fac if config.target == "R" else one,
        "G": fac if config.target == "G" else one,
        "C_stub": fac if config.target == "C_stub" else one,
    }


# ---------------------------------------------------------------------------
# Applying disorder to layouts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DisorderSample:
    """
    One generated disorder sample.

    Attributes
    ----------
    layout:
        Perturbed layout.
    factors:
        Dictionary of multiplicative factor arrays.
    config:
        Disorder configuration used to generate the sample.
    seed:
        Integer seed used when available.
    metadata:
        Additional JSON-friendly metadata.
    """

    layout: LineLayout
    factors: Mapping[str, jax.Array]
    config: DisorderConfig
    seed: int | None = None
    metadata: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        factor_summary = {}
        for name, value in self.factors.items():
            arr = jnp.asarray(value)
            factor_summary[name] = {
                "min": float(jnp.min(arr)),
                "max": float(jnp.max(arr)),
                "mean": float(jnp.mean(arr)),
                "std": float(jnp.std(arr)),
            }
        return {
            "layout": self.layout.summary(),
            "config": self.config.to_dict(),
            "seed": self.seed,
            "factors": factor_summary,
            "metadata": _jsonify(dict(self.metadata or {})),
        }


def compose_disorder_factors(
    key: jax.Array,
    n: int,
    config: DisorderConfig,
) -> dict[str, jax.Array]:
    """
    Generate multiplicative factors for L, C, R, G, and C_stub.

    Returns ones if disorder is disabled.
    """
    if int(n) <= 0:
        raise ValueError("n must be positive")
    n = int(n)

    one = jnp.ones((n,), dtype=jnp.float64)
    if not config.enabled:
        return {"L": one, "C": one, "R": one, "G": one, "C_stub": one}

    keys = jax.random.split(key, 5)

    random_L = random_multiplicative_field(keys[0], n, config.random_L)
    random_C = random_multiplicative_field(keys[1], n, config.random_C)
    random_R = random_multiplicative_field(keys[2], n, config.random_R)
    random_G = random_multiplicative_field(keys[3], n, config.random_G)
    random_C_stub = random_multiplicative_field(keys[4], n, config.random_C_stub)

    taper = _target_factors_from_taper(n, config.taper)
    hotspot = _target_factors_from_hotspot(n, config.hotspot)

    factors = {
        "L": (
            config.global_shift.L_scale
            * random_L
            * taper["L"]
            * hotspot["L"]
        ),
        "C": (
            config.global_shift.C_scale
            * random_C
            * taper["C"]
            * hotspot["C"]
        ),
        "R": (
            config.global_shift.R_scale
            * random_R
            * taper["R"]
            * hotspot["R"]
        ),
        "G": (
            config.global_shift.G_scale
            * random_G
            * taper["G"]
            * hotspot["G"]
        ),
        "C_stub": (
            config.global_shift.C_stub_scale
            * random_C_stub
            * taper["C_stub"]
            * hotspot["C_stub"]
        ),
    }

    return factors


def apply_disorder_to_layout(
    layout: LineLayout,
    config: DisorderConfig,
    *,
    key: jax.Array | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> DisorderSample:
    """
    Apply disorder to a layout and return a DisorderSample.

    Either pass a JAX key or an integer seed. If neither is passed, seed 0 is
    used for deterministic behavior.
    """
    if key is None:
        if seed is None:
            seed = 0
        key = jax.random.PRNGKey(int(seed))

    factors = compose_disorder_factors(key, layout.n_cells, config)

    perturbed = apply_multiplicative_cell_perturbations(
        layout,
        L_factor=factors["L"],
        C_factor=factors["C"],
        R_factor=factors["R"],
        G_factor=factors["G"],
        C_stub_factor=factors["C_stub"],
        name=name or f"{layout.name}_{config.name}_sample{config.sample_id}",
    ).with_metadata(
        disorder_enabled=config.enabled,
        disorder_name=config.name,
        disorder_sample_id=config.sample_id,
        disorder_seed=seed,
    )

    return DisorderSample(
        layout=perturbed,
        factors=factors,
        config=config,
        seed=seed,
        metadata={
            "source_layout": layout.name,
            "perturbed_layout": perturbed.name,
        },
    )


def apply_global_shifts_to_layout(
    layout: LineLayout,
    shifts: GlobalShiftConfig,
    *,
    name: str | None = None,
) -> LineLayout:
    """
    Apply deterministic global scale factors only.
    """
    return layout.scaled(
        L_scale=shifts.L_scale,
        C_scale=shifts.C_scale,
        R_scale=shifts.R_scale,
        G_scale=shifts.G_scale,
        C_stub_scale=shifts.C_stub_scale,
    ).with_updates(name=name or f"{layout.name}_global_shift")


# ---------------------------------------------------------------------------
# Monte Carlo sample generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonteCarloDisorderConfig:
    """
    Monte Carlo disorder-generation configuration.

    Parameters
    ----------
    n_samples:
        Number of samples to generate.
    base_seed:
        Base PRNG seed.
    disorder:
        DisorderConfig template. sample_id is replaced for each sample.
    keep_factor_arrays:
        If false, scripts may discard factor arrays after summarizing. This flag
        is metadata only; this module always returns factors in memory.
    """

    n_samples: int = 10
    base_seed: int = 1234
    disorder: DisorderConfig = DisorderConfig.mild_lc_process()
    keep_factor_arrays: bool = True

    def __post_init__(self) -> None:
        if int(self.n_samples) <= 0:
            raise ValueError("n_samples must be positive")
        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "base_seed", int(self.base_seed))

    def with_updates(self, **kwargs: Any) -> "MonteCarloDisorderConfig":
        return replace(self, **kwargs)

    @classmethod
    def from_runtime(
        cls,
        runtime: RuntimeConfig,
        *,
        n_samples: int,
        disorder: DisorderConfig,
    ) -> "MonteCarloDisorderConfig":
        return cls(
            n_samples=n_samples,
            base_seed=runtime.random_seed,
            disorder=disorder,
        )


def generate_disorder_samples(
    layout: LineLayout,
    config: MonteCarloDisorderConfig,
) -> list[DisorderSample]:
    """
    Generate a list of Monte Carlo disorder samples.
    """
    base_key = jax.random.PRNGKey(config.base_seed)
    keys = jax.random.split(base_key, config.n_samples)

    samples: list[DisorderSample] = []
    for i in range(config.n_samples):
        disorder_i = config.disorder.with_updates(sample_id=i)
        seed_i = config.base_seed + i
        sample = apply_disorder_to_layout(
            layout,
            disorder_i,
            key=keys[i],
            seed=seed_i,
            name=f"{layout.name}_{disorder_i.name}_mc{i:04d}",
        )
        samples.append(sample)
    return samples


def summarize_disorder_samples(samples: list[DisorderSample]) -> dict[str, Any]:
    """
    Summarize a list of disorder samples.
    """
    if len(samples) == 0:
        raise ValueError("samples may not be empty")

    names = list(samples[0].factors.keys())
    summary: dict[str, Any] = {
        "n_samples": len(samples),
        "layout_names": [s.layout.name for s in samples],
        "factor_statistics": {},
    }

    for name in names:
        means = jnp.asarray([jnp.mean(jnp.asarray(s.factors[name])) for s in samples])
        stds = jnp.asarray([jnp.std(jnp.asarray(s.factors[name])) for s in samples])
        mins = jnp.asarray([jnp.min(jnp.asarray(s.factors[name])) for s in samples])
        maxs = jnp.asarray([jnp.max(jnp.asarray(s.factors[name])) for s in samples])
        summary["factor_statistics"][name] = {
            "mean_of_means": float(jnp.mean(means)),
            "std_of_means": float(jnp.std(means)),
            "mean_of_stds": float(jnp.mean(stds)),
            "global_min": float(jnp.min(mins)),
            "global_max": float(jnp.max(maxs)),
        }

    return summary


# ---------------------------------------------------------------------------
# Material-prior convenience constructors
# ---------------------------------------------------------------------------

def make_lognormal_process_config_from_relative_std(
    *,
    std_fraction: float,
    correlation_length_cells: float,
    clip_sigma: float = 5.0,
) -> RandomFieldConfig:
    """
    Build a lognormal random-field config from a desired relative std.

    clip bounds are set approximately to exp(±clip_sigma * std_fraction).
    """
    _check_nonnegative("std_fraction", std_fraction)
    _check_nonnegative("correlation_length_cells", correlation_length_cells)

    if std_fraction == 0.0:
        return RandomFieldConfig(enabled=False)

    clip_min = float(jnp.exp(-clip_sigma * std_fraction))
    clip_max = float(jnp.exp(+clip_sigma * std_fraction))
    return RandomFieldConfig(
        enabled=True,
        distribution=DistributionKind.LOGNORMAL,
        std_fraction=std_fraction,
        correlation_length_cells=correlation_length_cells,
        kernel=CorrelationKernel.GAUSSIAN,
        clip_min=clip_min,
        clip_max=clip_max,
    )


def make_nbtiN_like_lc_disorder(
    *,
    L_std_fraction: float = 0.02,
    C_std_fraction: float = 0.005,
    correlation_length_cells: float = 50.0,
    sample_id: int = 0,
) -> DisorderConfig:
    """
    Convenience config for mild NbTiN-like line-parameter variability.

    This is a prior-like engineering placeholder, not a calibrated fabrication
    model. Calibrate it from actual film/process measurements when available.
    """
    return DisorderConfig(
        enabled=True,
        random_L=make_lognormal_process_config_from_relative_std(
            std_fraction=L_std_fraction,
            correlation_length_cells=correlation_length_cells,
        ),
        random_C=make_lognormal_process_config_from_relative_std(
            std_fraction=C_std_fraction,
            correlation_length_cells=correlation_length_cells,
        ),
        sample_id=sample_id,
        name="nbtin_like_lc",
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def factor_correlation(a: ArrayLike, b: ArrayLike) -> float:
    """
    Pearson correlation between two factor arrays.
    """
    x = jnp.asarray(a, dtype=jnp.float64).ravel()
    y = jnp.asarray(b, dtype=jnp.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"Arrays must have same shape, got {x.shape} and {y.shape}")
    x0 = x - jnp.mean(x)
    y0 = y - jnp.mean(y)
    denom = jnp.linalg.norm(x0) * jnp.linalg.norm(y0)
    return float(jnp.where(denom > 0.0, jnp.vdot(x0, y0) / denom, 0.0))


def disorder_factor_report(factors: Mapping[str, ArrayLike]) -> dict[str, Any]:
    """
    JSON-friendly summary of multiplicative disorder factors.
    """
    report: dict[str, Any] = {}
    for name, value in factors.items():
        arr = jnp.asarray(value, dtype=jnp.float64)
        report[name] = {
            "min": float(jnp.min(arr)),
            "max": float(jnp.max(arr)),
            "mean": float(jnp.mean(arr)),
            "std": float(jnp.std(arr)),
            "p01": float(jnp.percentile(arr, 1.0)),
            "p50": float(jnp.percentile(arr, 50.0)),
            "p99": float(jnp.percentile(arr, 99.0)),
        }
    return report


def compare_layout_parameter_totals(
    base: LineLayout,
    perturbed: LineLayout,
) -> dict[str, Any]:
    """
    Compare integrated parameter totals between base and perturbed layouts.
    """
    def rel(new: jax.Array, old: jax.Array) -> float:
        denom = jnp.maximum(jnp.abs(old), 1e-300)
        return float((new - old) / denom)

    base_L = jnp.sum(base.L_series_H)
    new_L = jnp.sum(perturbed.L_series_H)

    base_C = jnp.sum(base.total_shunt_C_F)
    new_C = jnp.sum(perturbed.total_shunt_C_F)

    base_R = jnp.sum(base.R_series_ohm)
    new_R = jnp.sum(perturbed.R_series_ohm)

    base_G = jnp.sum(base.G_shunt_S)
    new_G = jnp.sum(perturbed.G_shunt_S)

    return {
        "base_layout": base.name,
        "perturbed_layout": perturbed.name,
        "L_total_base_H": float(base_L),
        "L_total_perturbed_H": float(new_L),
        "L_total_relative_change": rel(new_L, base_L),
        "C_total_base_F": float(base_C),
        "C_total_perturbed_F": float(new_C),
        "C_total_relative_change": rel(new_C, base_C),
        "R_total_base_ohm": float(base_R),
        "R_total_perturbed_ohm": float(new_R),
        "R_total_relative_change": rel(new_R, base_R),
        "G_total_base_S": float(base_G),
        "G_total_perturbed_S": float(new_G),
        "G_total_relative_change": rel(new_G, base_G),
    }


__all__ = [
    "DistributionKind",
    "CorrelationKernel",
    "RandomFieldConfig",
    "GlobalShiftConfig",
    "TaperConfig",
    "HotspotConfig",
    "DisorderConfig",
    "DisorderSample",
    "MonteCarloDisorderConfig",
    "gaussian_kernel_1d",
    "boxcar_kernel_1d",
    "smooth_field_1d",
    "standardize_field",
    "random_multiplicative_field",
    "taper_factor",
    "hotspot_factor",
    "compose_disorder_factors",
    "apply_disorder_to_layout",
    "apply_global_shifts_to_layout",
    "generate_disorder_samples",
    "summarize_disorder_samples",
    "make_lognormal_process_config_from_relative_std",
    "make_nbtiN_like_lc_disorder",
    "factor_correlation",
    "disorder_factor_report",
    "compare_layout_parameter_totals",
]