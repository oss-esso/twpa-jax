"""
twpa.inference.priors
=====================

Prior distributions and parameter-vector helpers for TWPA inference.

This module is intentionally simulator-agnostic. It provides:

    ParameterPrior
        One bounded/scaled prior distribution.

    PriorSet
        Ordered collection of priors with encode/decode/sample helpers.

    ParameterSample
        Named parameter realization with log-prior diagnostics.

These tools are useful for:

    - synthetic recovery experiments,
    - random initialization of fits,
    - bounded optimization,
    - Bayesian or quasi-Bayesian parameter scans,
    - identifiability studies.

The production deterministic calibration workflow in twpa.workflows.calibration
has its own parameter-vector classes. This module is broader and more
experiment-oriented.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import math
import numpy as np

import jax
import jax.numpy as jnp


ArrayLike = Any


class PriorKind(str, Enum):
    """Supported prior families."""

    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    FIXED = "fixed"


class ParameterTransform(str, Enum):
    """Parameter encoding transform."""

    LINEAR = "linear"
    LOG = "log"
    LOGIT = "logit"


@dataclass(frozen=True)
class ParameterPrior:
    """
    Prior for one scalar parameter.

    Parameters
    ----------
    name:
        Parameter name.
    kind:
        Prior distribution.
    lower, upper:
        Hard bounds in physical parameter space. Bounds are required for
        uniform/log-uniform/logit encoding.
    mean, std:
        Distribution parameters for normal/log-normal priors. For LOG_NORMAL,
        mean/std are in log-space unless metadata specifies otherwise.
    initial:
        Optional preferred initial value. If omitted, the prior median is used.
    transform:
        Encoding transform used by optimization helpers.
    enabled:
        Disabled priors are preserved in decoded dictionaries but omitted from
        active vectors.
    description:
        Human-readable parameter description.
    metadata:
        Extra metadata.
    """

    name: str
    kind: PriorKind = PriorKind.UNIFORM
    lower: float | None = None
    upper: float | None = None
    mean: float | None = None
    std: float | None = None
    initial: float | None = None
    transform: ParameterTransform = ParameterTransform.LINEAR
    enabled: bool = True
    description: str = ""
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PriorKind(self.kind))
        object.__setattr__(self, "transform", ParameterTransform(self.transform))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

        if not self.name:
            raise ValueError("name may not be empty")

        if self.kind in {PriorKind.UNIFORM, PriorKind.LOG_UNIFORM}:
            if self.lower is None or self.upper is None:
                raise ValueError(f"{self.kind.value} prior requires lower and upper")
            if self.upper <= self.lower:
                raise ValueError("upper must exceed lower")

        if self.kind == PriorKind.LOG_UNIFORM:
            if self.lower is None or self.lower <= 0.0:
                raise ValueError("log-uniform lower bound must be positive")

        if self.kind in {PriorKind.NORMAL, PriorKind.LOG_NORMAL}:
            if self.mean is None or self.std is None:
                raise ValueError(f"{self.kind.value} prior requires mean and std")
            if self.std <= 0.0:
                raise ValueError("std must be positive")

        if self.kind == PriorKind.LOG_NORMAL:
            if self.lower is not None and self.lower <= 0.0:
                raise ValueError("log-normal lower bound must be positive if provided")

        if self.kind == PriorKind.FIXED:
            value = self.initial if self.initial is not None else self.mean
            if value is None:
                raise ValueError("fixed prior requires initial or mean")
            object.__setattr__(self, "initial", float(value))
            object.__setattr__(self, "mean", float(value))
            object.__setattr__(self, "enabled", False)

        if self.transform == ParameterTransform.LOG:
            val = self.initial_value()
            if val <= 0.0:
                raise ValueError("LOG transform requires positive initial value")
            if self.lower is not None and self.lower <= 0.0:
                raise ValueError("LOG transform requires positive lower bound")

        if self.transform == ParameterTransform.LOGIT:
            if self.lower is None or self.upper is None:
                raise ValueError("LOGIT transform requires finite lower and upper bounds")
            val = self.initial_value()
            if not (self.lower < val < self.upper):
                # Nudge initial into the open interval.
                eps = 1e-12 * max(abs(self.upper - self.lower), 1.0)
                object.__setattr__(
                    self,
                    "initial",
                    float(np.clip(val, self.lower + eps, self.upper - eps)),
                )

    def with_updates(self, **kwargs: Any) -> "ParameterPrior":
        return replace(self, **kwargs)

    def initial_value(self) -> float:
        """
        Preferred initial value in physical space.
        """
        if self.initial is not None:
            return float(self.initial)

        if self.kind == PriorKind.UNIFORM:
            assert self.lower is not None and self.upper is not None
            return 0.5 * (self.lower + self.upper)

        if self.kind == PriorKind.LOG_UNIFORM:
            assert self.lower is not None and self.upper is not None
            return float(math.sqrt(self.lower * self.upper))

        if self.kind == PriorKind.NORMAL:
            assert self.mean is not None
            return float(self.mean)

        if self.kind == PriorKind.LOG_NORMAL:
            assert self.mean is not None
            return float(math.exp(self.mean))

        if self.kind == PriorKind.FIXED:
            assert self.mean is not None
            return float(self.mean)

        raise ValueError(f"Unsupported prior kind {self.kind}")

    def clip(self, value: float) -> float:
        """
        Clip a physical value to hard bounds if bounds are present.
        """
        x = float(value)
        if self.lower is not None:
            x = max(x, float(self.lower))
        if self.upper is not None:
            x = min(x, float(self.upper))
        return x

    def contains(self, value: float) -> bool:
        """
        Check hard-bound support.
        """
        x = float(value)
        if self.lower is not None and x < self.lower:
            return False
        if self.upper is not None and x > self.upper:
            return False
        if self.kind in {PriorKind.LOG_UNIFORM, PriorKind.LOG_NORMAL} and x <= 0.0:
            return False
        return True

    def encode(self, value: float) -> float:
        """
        Encode a physical value into optimization space.
        """
        x = float(value)

        if self.transform == ParameterTransform.LINEAR:
            return x

        if self.transform == ParameterTransform.LOG:
            if x <= 0.0:
                raise ValueError(f"{self.name}: LOG transform received non-positive value {x}")
            return float(math.log(x))

        if self.transform == ParameterTransform.LOGIT:
            assert self.lower is not None and self.upper is not None
            lo, hi = float(self.lower), float(self.upper)
            eps = 1e-15 * max(hi - lo, 1.0)
            x = float(np.clip(x, lo + eps, hi - eps))
            y = (x - lo) / (hi - lo)
            return float(math.log(y / (1.0 - y)))

        raise ValueError(f"Unsupported transform {self.transform}")

    def decode(self, encoded: float) -> float:
        """
        Decode an optimization-space value into physical space.
        """
        z = float(encoded)

        if self.transform == ParameterTransform.LINEAR:
            return self.clip(z)

        if self.transform == ParameterTransform.LOG:
            return self.clip(float(math.exp(z)))

        if self.transform == ParameterTransform.LOGIT:
            assert self.lower is not None and self.upper is not None
            lo, hi = float(self.lower), float(self.upper)
            y = 1.0 / (1.0 + math.exp(-z))
            return self.clip(lo + (hi - lo) * y)

        raise ValueError(f"Unsupported transform {self.transform}")

    def initial_encoded(self) -> float:
        return self.encode(self.initial_value())

    def log_prob(self, value: float) -> float:
        """
        Log prior density up to standard normalization.

        Returns -inf outside hard bounds.
        """
        x = float(value)
        if not self.contains(x):
            return float("-inf")

        if self.kind == PriorKind.FIXED:
            return 0.0 if abs(x - self.initial_value()) <= 1e-15 else float("-inf")

        if self.kind == PriorKind.UNIFORM:
            assert self.lower is not None and self.upper is not None
            return -math.log(self.upper - self.lower)

        if self.kind == PriorKind.LOG_UNIFORM:
            assert self.lower is not None and self.upper is not None
            return -math.log(x) - math.log(math.log(self.upper) - math.log(self.lower))

        if self.kind == PriorKind.NORMAL:
            assert self.mean is not None and self.std is not None
            z = (x - self.mean) / self.std
            return -0.5 * z * z - math.log(self.std) - 0.5 * math.log(2.0 * math.pi)

        if self.kind == PriorKind.LOG_NORMAL:
            assert self.mean is not None and self.std is not None
            if x <= 0.0:
                return float("-inf")
            lx = math.log(x)
            z = (lx - self.mean) / self.std
            return -0.5 * z * z - math.log(self.std) - math.log(x) - 0.5 * math.log(2.0 * math.pi)

        raise ValueError(f"Unsupported prior kind {self.kind}")

    def sample(self, rng: np.random.Generator) -> float:
        """
        Draw one physical-space sample.
        """
        if self.kind == PriorKind.FIXED:
            return self.initial_value()

        if self.kind == PriorKind.UNIFORM:
            assert self.lower is not None and self.upper is not None
            return float(rng.uniform(self.lower, self.upper))

        if self.kind == PriorKind.LOG_UNIFORM:
            assert self.lower is not None and self.upper is not None
            return float(math.exp(rng.uniform(math.log(self.lower), math.log(self.upper))))

        if self.kind == PriorKind.NORMAL:
            assert self.mean is not None and self.std is not None
            for _ in range(10_000):
                x = float(rng.normal(self.mean, self.std))
                if self.contains(x):
                    return x
            return self.clip(float(rng.normal(self.mean, self.std)))

        if self.kind == PriorKind.LOG_NORMAL:
            assert self.mean is not None and self.std is not None
            for _ in range(10_000):
                x = float(math.exp(rng.normal(self.mean, self.std)))
                if self.contains(x):
                    return x
            return self.clip(float(math.exp(rng.normal(self.mean, self.std))))

        raise ValueError(f"Unsupported prior kind {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "lower": self.lower,
            "upper": self.upper,
            "mean": self.mean,
            "std": self.std,
            "initial": self.initial,
            "initial_value": self.initial_value(),
            "initial_encoded": self.initial_encoded() if self.enabled else None,
            "transform": self.transform.value,
            "enabled": self.enabled,
            "description": self.description,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ParameterSample:
    """
    One named parameter sample.
    """

    values: Mapping[str, float]
    log_prior: float | None = None
    encoded_vector: jax.Array | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", {str(k): float(v) for k, v in self.values.items()})
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "log_prior": self.log_prior,
            "encoded_vector": (
                None
                if self.encoded_vector is None
                else np.asarray(self.encoded_vector).tolist()
            ),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PriorSet:
    """
    Ordered collection of ParameterPrior objects.
    """

    priors: tuple[ParameterPrior, ...]
    name: str = "prior_set"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        priors = tuple(self.priors)
        if not priors:
            raise ValueError("PriorSet requires at least one prior")

        names = [p.name for p in priors]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate parameter prior names: {duplicates}")

        object.__setattr__(self, "priors", priors)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.priors)

    @property
    def enabled_priors(self) -> tuple[ParameterPrior, ...]:
        return tuple(p for p in self.priors if p.enabled)

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.enabled_priors)

    @property
    def disabled_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.priors if not p.enabled)

    @property
    def ndim(self) -> int:
        return len(self.enabled_priors)

    def prior_for(self, name: str) -> ParameterPrior:
        for p in self.priors:
            if p.name == name:
                return p
        raise KeyError(name)

    def initial_values(self, *, include_disabled: bool = True) -> dict[str, float]:
        priors = self.priors if include_disabled else self.enabled_priors
        return {p.name: p.initial_value() for p in priors}

    def initial_vector(self) -> jax.Array:
        return jnp.asarray([p.initial_encoded() for p in self.enabled_priors], dtype=jnp.float64)

    def bounds_encoded(self) -> tuple[jax.Array, jax.Array]:
        """
        Encoded lower/upper bounds for active priors.

        Infinite bounds are used when the transform/prior does not imply a
        finite encoded interval.
        """
        lows = []
        highs = []

        for p in self.enabled_priors:
            if p.transform == ParameterTransform.LOGIT:
                lows.append(float("-inf"))
                highs.append(float("inf"))

            elif p.transform == ParameterTransform.LOG:
                lows.append(float("-inf") if p.lower is None else math.log(p.lower))
                highs.append(float("inf") if p.upper is None else math.log(p.upper))

            else:
                lows.append(float("-inf") if p.lower is None else p.lower)
                highs.append(float("inf") if p.upper is None else p.upper)

        return (
            jnp.asarray(lows, dtype=jnp.float64),
            jnp.asarray(highs, dtype=jnp.float64),
        )

    def decode_vector(
        self,
        vector: ArrayLike,
        *,
        include_disabled: bool = True,
    ) -> dict[str, float]:
        """
        Decode active-vector values into a full parameter dictionary.
        """
        vec = np.asarray(vector, dtype=float)
        if vec.shape != (self.ndim,):
            raise ValueError(f"Expected vector shape {(self.ndim,)}, got {vec.shape}")

        out: dict[str, float] = {}
        idx = 0

        for p in self.priors:
            if p.enabled:
                out[p.name] = p.decode(float(vec[idx]))
                idx += 1
            elif include_disabled:
                out[p.name] = p.initial_value()

        return out

    def encode_values(
        self,
        values: Mapping[str, float],
        *,
        strict: bool = False,
    ) -> jax.Array:
        """
        Encode a parameter dictionary into active-vector form.
        """
        encoded = []

        for p in self.enabled_priors:
            if p.name not in values:
                if strict:
                    raise KeyError(f"Missing parameter value {p.name!r}")
                value = p.initial_value()
            else:
                value = float(values[p.name])
            encoded.append(p.encode(value))

        return jnp.asarray(encoded, dtype=jnp.float64)

    def log_prob(self, values: Mapping[str, float]) -> float:
        total = 0.0
        for p in self.priors:
            value = float(values.get(p.name, p.initial_value()))
            lp = p.log_prob(value)
            if not math.isfinite(lp):
                return float("-inf")
            total += lp
        return float(total)

    def contains(self, values: Mapping[str, float]) -> bool:
        for p in self.priors:
            value = float(values.get(p.name, p.initial_value()))
            if not p.contains(value):
                return False
        return True

    def sample(
        self,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        include_encoded: bool = True,
    ) -> ParameterSample:
        """
        Draw one sample from the prior set.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        values = {p.name: p.sample(rng) for p in self.priors}
        encoded = self.encode_values(values) if include_encoded else None
        lp = self.log_prob(values)

        return ParameterSample(
            values=values,
            log_prior=lp,
            encoded_vector=encoded,
            metadata={
                "prior_set": self.name,
                "seed": seed,
            },
        )

    def sample_many(
        self,
        n: int,
        *,
        seed: int | None = None,
        include_encoded: bool = True,
    ) -> tuple[ParameterSample, ...]:
        """
        Draw many samples.
        """
        if int(n) <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        return tuple(
            self.sample(rng=rng, include_encoded=include_encoded)
            for _ in range(int(n))
        )

    def update_initials(self, values: Mapping[str, float]) -> "PriorSet":
        """
        Return a new PriorSet with updated initial values.
        """
        new_priors = []
        for p in self.priors:
            if p.name in values:
                new_priors.append(p.with_updates(initial=float(values[p.name])))
            else:
                new_priors.append(p)
        return replace(self, priors=tuple(new_priors))

    def active_table(self) -> str:
        """
        Markdown table of active priors.
        """
        lines = [
            "| name | kind | initial | lower | upper | transform |",
            "|---|---|---:|---:|---:|---|",
        ]

        for p in self.enabled_priors:
            lines.append(
                f"| `{p.name}` | `{p.kind.value}` | "
                f"{p.initial_value():.6g} | "
                f"{'' if p.lower is None else f'{p.lower:.6g}'} | "
                f"{'' if p.upper is None else f'{p.upper:.6g}'} | "
                f"`{p.transform.value}` |"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ndim": self.ndim,
            "names": list(self.names),
            "enabled_names": list(self.enabled_names),
            "disabled_names": list(self.disabled_names),
            "priors": [p.to_dict() for p in self.priors],
            "metadata": dict(self.metadata or {}),
        }


def make_scale_prior(
    name: str,
    *,
    lower: float = 0.5,
    upper: float = 2.0,
    initial: float = 1.0,
    kind: PriorKind = PriorKind.LOG_UNIFORM,
    enabled: bool = True,
    description: str = "",
) -> ParameterPrior:
    """
    Convenience prior for positive multiplicative scale factors.
    """
    return ParameterPrior(
        name=name,
        kind=kind,
        lower=lower,
        upper=upper,
        initial=initial,
        transform=ParameterTransform.LOG,
        enabled=enabled,
        description=description or f"Multiplicative scale prior for {name}.",
    )


def make_positive_prior(
    name: str,
    *,
    lower: float,
    upper: float,
    initial: float | None = None,
    kind: PriorKind = PriorKind.LOG_UNIFORM,
    enabled: bool = True,
    description: str = "",
) -> ParameterPrior:
    """
    Convenience prior for positive physical parameters.
    """
    if lower <= 0.0:
        raise ValueError("lower must be positive")
    if upper <= lower:
        raise ValueError("upper must exceed lower")

    return ParameterPrior(
        name=name,
        kind=kind,
        lower=lower,
        upper=upper,
        initial=initial if initial is not None else math.sqrt(lower * upper),
        transform=ParameterTransform.LOG,
        enabled=enabled,
        description=description or f"Positive prior for {name}.",
    )


def make_linear_prior(
    name: str,
    *,
    lower: float,
    upper: float,
    initial: float | None = None,
    enabled: bool = True,
    description: str = "",
) -> ParameterPrior:
    """
    Convenience bounded linear-uniform prior.
    """
    return ParameterPrior(
        name=name,
        kind=PriorKind.UNIFORM,
        lower=lower,
        upper=upper,
        initial=initial if initial is not None else 0.5 * (lower + upper),
        transform=ParameterTransform.LINEAR,
        enabled=enabled,
        description=description or f"Linear uniform prior for {name}.",
    )


def make_fixed_prior(
    name: str,
    value: float,
    *,
    description: str = "",
) -> ParameterPrior:
    """
    Convenience fixed parameter.
    """
    return ParameterPrior(
        name=name,
        kind=PriorKind.FIXED,
        initial=float(value),
        mean=float(value),
        enabled=False,
        description=description or f"Fixed value for {name}.",
    )


def make_default_twpa_scale_prior_set(
    *,
    include_linear: bool = True,
    include_nonlinear: bool = True,
    include_pump: bool = True,
) -> PriorSet:
    """
    Default scale-prior set for common TWPA calibration/recovery studies.
    """
    priors: list[ParameterPrior] = []

    if include_linear:
        priors.extend(
            [
                make_scale_prior("L_scale", lower=0.5, upper=2.0),
                make_scale_prior("C_scale", lower=0.5, upper=2.0),
                make_scale_prior("C_stub_scale", lower=0.05, upper=20.0),
                make_scale_prior("R_scale", lower=0.01, upper=100.0, enabled=False),
                make_scale_prior("G_scale", lower=0.01, upper=100.0, enabled=False),
            ]
        )

    if include_nonlinear:
        priors.extend(
            [
                make_scale_prior("I_star_scale", lower=0.1, upper=10.0),
                make_scale_prior("beta_nl_scale", lower=0.1, upper=10.0, enabled=False),
            ]
        )

    if include_pump:
        priors.extend(
            [
                make_scale_prior("pump_current_scale", lower=0.1, upper=10.0),
                make_linear_prior("pump_power_offset_db", lower=-6.0, upper=6.0, initial=0.0, enabled=False),
            ]
        )

    return PriorSet(
        tuple(priors),
        name="default_twpa_scale_prior_set",
        metadata={
            "include_linear": include_linear,
            "include_nonlinear": include_nonlinear,
            "include_pump": include_pump,
        },
    )


def samples_to_array(
    samples: Sequence[ParameterSample],
    names: Sequence[str] | None = None,
) -> tuple[jax.Array, tuple[str, ...]]:
    """
    Convert ParameterSample objects to a dense array.

    Returns shape ``(n_samples, n_parameters)``.
    """
    if not samples:
        raise ValueError("samples may not be empty")

    if names is None:
        names = tuple(samples[0].values.keys())
    else:
        names = tuple(names)

    rows = []
    for sample in samples:
        rows.append([sample.values[name] for name in names])

    return jnp.asarray(rows, dtype=jnp.float64), tuple(names)


def summarize_samples(
    samples: Sequence[ParameterSample],
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Summary statistics for parameter samples.
    """
    arr, names_tuple = samples_to_array(samples, names)
    arr_np = np.asarray(arr)

    return {
        "n_samples": int(arr_np.shape[0]),
        "names": list(names_tuple),
        "mean": {name: float(arr_np[:, i].mean()) for i, name in enumerate(names_tuple)},
        "std": {name: float(arr_np[:, i].std(ddof=1)) if arr_np.shape[0] > 1 else 0.0 for i, name in enumerate(names_tuple)},
        "min": {name: float(arr_np[:, i].min()) for i, name in enumerate(names_tuple)},
        "max": {name: float(arr_np[:, i].max()) for i, name in enumerate(names_tuple)},
    }


__all__ = [
    "ArrayLike",
    "PriorKind",
    "ParameterTransform",
    "ParameterPrior",
    "ParameterSample",
    "PriorSet",
    "make_scale_prior",
    "make_positive_prior",
    "make_linear_prior",
    "make_fixed_prior",
    "make_default_twpa_scale_prior_set",
    "samples_to_array",
    "summarize_samples",
]