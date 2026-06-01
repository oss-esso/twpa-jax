"""
twpa.core.harmonics
===================

Harmonic-balance coefficient bookkeeping.

This module defines the Fourier-coefficient conventions used by the simulator
and provides low-level utilities for converting between:

    complex Fourier coefficients
    real flattened Newton vectors
    RMS RF phasors
    time-domain samples

Fourier convention
------------------
For a real-valued scalar waveform x(t), we use the double-sided expansion

    x(t) = sum_k X_k exp(i omega_k t)

where real-valuedness requires

    X(-omega) = conj(X(+omega)).

For a single sinusoid represented by an RMS phasor x_rms,

    x(t) = sqrt(2) Re[x_rms exp(i omega t)]

so the positive-frequency double-sided Fourier coefficient is

    X(+omega) = x_rms / sqrt(2)

and

    X(-omega) = conj(x_rms) / sqrt(2).

This distinction matters because RF power-wave variables normally use RMS
phasors, while FFT-based nonlinear constitutive laws naturally use peak/time
waveforms.

Scope
-----
This file contains pure bookkeeping. It does not implement nonlinear inductors
or circuit residuals. Those appear later in:

    twpa.core.hb_fft
    twpa.nonlinear.hb_element
    twpa.nonlinear.pump_hb_ladder
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping

import jax
import jax.numpy as jnp

from .frequency_plan import FrequencyPlan as CanonicalFrequencyPlan
from .units import angular_frequency


ArrayLike = Any


# ---------------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------------

def _as_1d_array(name: str, value: ArrayLike, dtype: Any | None = None) -> jax.Array:
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _as_complex_array(value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    return arr


def _check_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _check_odd(name: str, value: int) -> int:
    value = int(value)
    if value % 2 != 1:
        raise ValueError(f"{name} must be odd, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Complex <-> real vector packing
# ---------------------------------------------------------------------------

def complex_to_real_vector(z: ArrayLike) -> jax.Array:
    """
    Pack a complex array into a real vector.

    Convention
    ----------
    Given z with arbitrary shape, return

        [Re(z).ravel(), Im(z).ravel()]

    This convention is used by Newton solvers that expect real unknown vectors.
    """
    zz = jnp.asarray(z)
    return jnp.concatenate([jnp.real(zz).ravel(), jnp.imag(zz).ravel()])


def real_vector_to_complex(x: ArrayLike, shape: tuple[int, ...]) -> jax.Array:
    """
    Unpack a real vector into a complex array with the given shape.

    Inverse of complex_to_real_vector().
    """
    xx = jnp.asarray(x)
    n = int(jnp.prod(jnp.asarray(shape)))
    if xx.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {xx.shape}")
    if xx.shape[0] != 2 * n:
        raise ValueError(f"x length must be {2*n} for complex shape {shape}, got {xx.shape[0]}")
    re = xx[:n].reshape(shape)
    im = xx[n:].reshape(shape)
    return re + 1j * im


def complex_tree_to_real_vector(tree: Any) -> tuple[jax.Array, Any]:
    """
    Pack all leaves of a complex/real PyTree into one real vector.

    Returns
    -------
    vector:
        Flat real vector.
    unravel_fn:
        Callable mapping a real vector back to the original tree structure.

    Notes
    -----
    This is useful for solvers that need to flatten structured state objects.
    Complex leaves are represented by real and imaginary parts. Real leaves are
    included as-is.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    specs: list[dict[str, Any]] = []
    parts: list[jax.Array] = []

    for leaf in leaves:
        arr = jnp.asarray(leaf)
        is_complex = bool(jnp.issubdtype(arr.dtype, jnp.complexfloating))
        specs.append(
            {
                "shape": arr.shape,
                "dtype": arr.dtype,
                "is_complex": is_complex,
                "size": arr.size,
            }
        )
        if is_complex:
            parts.append(complex_to_real_vector(arr))
        else:
            parts.append(arr.ravel().astype(jnp.float64))

    if parts:
        vector = jnp.concatenate(parts)
    else:
        vector = jnp.asarray([], dtype=jnp.float64)

    def unravel_fn(x: ArrayLike) -> Any:
        xx = jnp.asarray(x)
        out_leaves = []
        offset = 0
        for spec in specs:
            size = int(spec["size"])
            shape = tuple(spec["shape"])
            if spec["is_complex"]:
                chunk = xx[offset : offset + 2 * size]
                out_leaves.append(real_vector_to_complex(chunk, shape))
                offset += 2 * size
            else:
                chunk = xx[offset : offset + size]
                out_leaves.append(chunk.reshape(shape).astype(spec["dtype"]))
                offset += size
        if offset != xx.shape[0]:
            raise ValueError(f"Unused entries while unraveling: offset={offset}, len={xx.shape[0]}")
        return jax.tree_util.tree_unflatten(treedef, out_leaves)

    return vector, unravel_fn


# ---------------------------------------------------------------------------
# RMS phasor <-> Fourier coefficient conversions
# ---------------------------------------------------------------------------

def rms_phasor_to_positive_fourier(x_rms: ArrayLike) -> jax.Array:
    """
    Convert RMS RF phasor to positive-frequency Fourier coefficient.

    If

        x(t) = sqrt(2) Re[x_rms exp(i omega t)]

    then

        X(+omega) = x_rms / sqrt(2).
    """
    return jnp.asarray(x_rms) / jnp.sqrt(2.0)


def positive_fourier_to_rms_phasor(x_pos: ArrayLike) -> jax.Array:
    """
    Convert positive-frequency Fourier coefficient to RMS RF phasor.
    """
    return jnp.sqrt(2.0) * jnp.asarray(x_pos)


def peak_phasor_to_positive_fourier(x_peak: ArrayLike) -> jax.Array:
    """
    Convert peak phasor to positive-frequency Fourier coefficient.

    If x(t) = Re[x_peak exp(i omega t)], then X(+omega) = x_peak / 2.
    """
    return jnp.asarray(x_peak) / 2.0


def positive_fourier_to_peak_phasor(x_pos: ArrayLike) -> jax.Array:
    """
    Convert positive-frequency Fourier coefficient to peak phasor.
    """
    return 2.0 * jnp.asarray(x_pos)


def make_real_sinusoid_coefficients_from_rms(
    x_rms: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """
    Return positive and negative Fourier coefficients for a real sinusoid.

    Returns
    -------
    x_pos, x_neg:
        x_pos = x_rms / sqrt(2)
        x_neg = conj(x_rms) / sqrt(2)
    """
    x_pos = rms_phasor_to_positive_fourier(x_rms)
    x_neg = jnp.conj(x_pos)
    return x_pos, x_neg


# ---------------------------------------------------------------------------
# Harmonic index basis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HarmonicIndexBasis:
    """
    Integer harmonic index basis for commensurate HB.

    Example
    -------
    n_harmonics = 3, include_negative = True, include_dc = True gives

        indices = [-3, -2, -1, 0, +1, +2, +3]

    This basis is useful for pump-only HB with harmonics of one fundamental
    frequency. For arbitrary sidebands, use FrequencyPlan directly.
    """

    indices: jax.Array
    fundamental_frequency_hz: float
    labels: tuple[str, ...]
    include_negative: bool = True
    include_dc: bool = True
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        idx = _as_1d_array("indices", self.indices, dtype=jnp.int32)
        if idx.shape[0] == 0:
            raise ValueError("indices may not be empty")
        if float(self.fundamental_frequency_hz) <= 0.0:
            raise ValueError("fundamental_frequency_hz must be positive")
        if len(self.labels) != int(idx.shape[0]):
            raise ValueError("labels length must match indices length")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        object.__setattr__(self, "indices", idx)
        object.__setattr__(self, "fundamental_frequency_hz", float(self.fundamental_frequency_hz))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_tones(self) -> int:
        return int(self.indices.shape[0])

    @property
    def frequencies_hz(self) -> jax.Array:
        return self.indices.astype(jnp.float64) * self.fundamental_frequency_hz

    @property
    def angular_frequencies_rad_s(self) -> jax.Array:
        return angular_frequency(self.frequencies_hz)

    @property
    def positive_mask(self) -> jax.Array:
        return self.indices > 0

    @property
    def negative_mask(self) -> jax.Array:
        return self.indices < 0

    @property
    def dc_mask(self) -> jax.Array:
        return self.indices == 0

    def position_of_index(self, harmonic_index: int) -> int:
        matches = jnp.where(self.indices == int(harmonic_index), size=self.n_tones, fill_value=-1)[0]
        valid = [int(m) for m in matches.tolist() if int(m) >= 0]
        if not valid:
            raise KeyError(f"Harmonic index {harmonic_index} not in basis")
        return valid[0]

    def position_of_label(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise KeyError(f"Unknown harmonic label {label!r}") from exc

    def conjugate_position_map(self, require_all: bool = False) -> dict[int, int | None]:
        mapping: dict[int, int | None] = {}
        idx_list = [int(v) for v in self.indices.tolist()]
        for pos, k in enumerate(idx_list):
            target = -k
            if target in idx_list:
                mapping[pos] = idx_list.index(target)
            else:
                if require_all:
                    raise ValueError(f"No conjugate index for harmonic {k}")
                mapping[pos] = None
        return mapping

    def to_frequency_plan(self) -> FrequencyPlan:
        """
        Convert to a FrequencyPlan with generic labels/roles.

        This avoids duplicating too much semantic logic here.
        """
        from .frequency_plan import FrequencyPlanKind, ToneRole

        roles = []
        for k in self.indices.tolist():
            if int(k) == 0:
                roles.append(ToneRole.DC)
            elif abs(int(k)) == 1:
                roles.append(ToneRole.PUMP)
            else:
                roles.append(ToneRole.PUMP_HARMONIC)

        return FrequencyPlan(
            frequencies_hz=self.frequencies_hz,
            labels=self.labels,
            roles=tuple(roles),
            indices=self.indices,
            kind=FrequencyPlanKind.PUMP_ONLY,
            reference_pump_hz=self.fundamental_frequency_hz,
            reference_signal_hz=None,
            metadata={
                **dict(self.metadata or {}),
                "source": "HarmonicIndexBasis.to_frequency_plan",
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": [int(k) for k in self.indices.tolist()],
            "labels": list(self.labels),
            "fundamental_frequency_hz": self.fundamental_frequency_hz,
            "n_tones": self.n_tones,
            "include_negative": self.include_negative,
            "include_dc": self.include_dc,
            "frequencies_hz": [float(f) for f in self.frequencies_hz.tolist()],
            "metadata": dict(self.metadata or {}),
        }


def make_harmonic_index_basis(
    fundamental_frequency_hz: float,
    *,
    n_harmonics: int,
    include_negative: bool = True,
    include_dc: bool = True,
) -> HarmonicIndexBasis:
    """
    Construct a harmonic index basis.

    Parameters
    ----------
    fundamental_frequency_hz:
        Base frequency f0.
    n_harmonics:
        Highest positive harmonic index.
    include_negative:
        Include negative harmonics.
    include_dc:
        Include index zero.
    """
    n_harmonics = _check_positive_int("n_harmonics", n_harmonics)
    if fundamental_frequency_hz <= 0.0:
        raise ValueError("fundamental_frequency_hz must be positive")

    indices: list[int] = []
    if include_negative:
        indices.extend(range(-n_harmonics, 0))
    if include_dc:
        indices.append(0)
    indices.extend(range(1, n_harmonics + 1))

    labels = []
    for k in indices:
        if k == 0:
            labels.append("dc")
        elif k > 0:
            labels.append("h1" if k == 1 else f"h{k}")
        else:
            labels.append("neg_h1" if k == -1 else f"neg_h{abs(k)}")

    return HarmonicIndexBasis(
        indices=jnp.asarray(indices, dtype=jnp.int32),
        fundamental_frequency_hz=fundamental_frequency_hz,
        labels=tuple(labels),
        include_negative=include_negative,
        include_dc=include_dc,
        metadata={"n_harmonics": n_harmonics},
    )


# ---------------------------------------------------------------------------
# Conjugate symmetry
# ---------------------------------------------------------------------------

def enforce_conjugate_symmetry_by_indices(
    coeffs: ArrayLike,
    indices: ArrayLike,
    *,
    average_pairs: bool = True,
) -> jax.Array:
    """
    Enforce X(-k) = conj(X(+k)) for coefficients indexed by integer harmonics.

    Parameters
    ----------
    coeffs:
        Complex coefficients with harmonic dimension first, shape (K, ...).
    indices:
        Integer harmonic indices, shape (K,).
    average_pairs:
        If true, replace each pair by the average implied symmetric pair.
        If false, positive-frequency coefficients dominate and negative ones
        are overwritten from them.

    Returns
    -------
    jax.Array
        Symmetrized coefficient array.
    """
    c = _as_complex_array(coeffs)
    idx = _as_1d_array("indices", indices, dtype=jnp.int32)
    if c.shape[0] != idx.shape[0]:
        raise ValueError("coeffs first dimension must match indices length")

    out = c
    idx_list = [int(v) for v in idx.tolist()]

    for pos, k in enumerate(idx_list):
        if k < 0:
            continue
        if k == 0:
            out = out.at[pos].set(jnp.real(out[pos]) + 0j)
            continue
        if -k not in idx_list:
            continue
        neg_pos = idx_list.index(-k)
        if average_pairs:
            pos_val = 0.5 * (out[pos] + jnp.conj(out[neg_pos]))
        else:
            pos_val = out[pos]
        out = out.at[pos].set(pos_val)
        out = out.at[neg_pos].set(jnp.conj(pos_val))

    return out


def conjugate_symmetry_error_by_indices(
    coeffs: ArrayLike,
    indices: ArrayLike,
) -> jax.Array:
    """
    Return max absolute error in X(-k) = conj(X(+k)).

    Missing pairs are ignored.
    """
    c = _as_complex_array(coeffs)
    idx = _as_1d_array("indices", indices, dtype=jnp.int32)
    if c.shape[0] != idx.shape[0]:
        raise ValueError("coeffs first dimension must match indices length")

    idx_list = [int(v) for v in idx.tolist()]
    errors = []

    for pos, k in enumerate(idx_list):
        if k <= 0:
            continue
        if -k in idx_list:
            neg_pos = idx_list.index(-k)
            errors.append(jnp.max(jnp.abs(c[neg_pos] - jnp.conj(c[pos]))))

    if not errors:
        return jnp.asarray(0.0)
    return jnp.max(jnp.asarray(errors))


def enforce_conjugate_symmetry_by_frequencies(
    coeffs: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    atol_hz: float = 1e-6,
    rtol: float = 1e-12,
    average_pairs: bool = True,
) -> jax.Array:
    """
    Enforce conjugate symmetry for arbitrary signed frequency arrays.

    This is useful for FrequencyPlan-based sideband sets.
    """
    c = _as_complex_array(coeffs)
    f = _as_1d_array("frequencies_hz", frequencies_hz, dtype=jnp.float64)
    if c.shape[0] != f.shape[0]:
        raise ValueError("coeffs first dimension must match frequency length")

    out = c
    used: set[int] = set()

    for i, fi_raw in enumerate(f.tolist()):
        if i in used:
            continue
        fi = float(fi_raw)

        if abs(fi) <= atol_hz:
            out = out.at[i].set(jnp.real(out[i]) + 0j)
            used.add(i)
            continue

        matches = [
            j
            for j, fj_raw in enumerate(f.tolist())
            if j != i
            and abs(float(fj_raw) + fi) <= (atol_hz + rtol * abs(fi))
        ]

        if not matches:
            used.add(i)
            continue

        j = matches[0]
        if average_pairs:
            if fi > 0:
                pos, neg = i, j
            else:
                pos, neg = j, i
            pos_val = 0.5 * (out[pos] + jnp.conj(out[neg]))
        else:
            if fi > 0:
                pos, neg = i, j
            else:
                pos, neg = j, i
            pos_val = out[pos]

        out = out.at[pos].set(pos_val)
        out = out.at[neg].set(jnp.conj(pos_val))
        used.add(i)
        used.add(j)

    return out


# ---------------------------------------------------------------------------
# Coefficient array construction
# ---------------------------------------------------------------------------

def zeros_for_plan(
    plan: CanonicalFrequencyPlan,
    *,
    trailing_shape: tuple[int, ...] = (),
    dtype: Any = jnp.complex128,
) -> jax.Array:
    """
    Complex zero coefficient array with first dimension = number of tones.
    """
    return jnp.zeros((plan.n_tones, *trailing_shape), dtype=dtype)


def zeros_for_basis(
    basis: HarmonicIndexBasis,
    *,
    trailing_shape: tuple[int, ...] = (),
    dtype: Any = jnp.complex128,
) -> jax.Array:
    """
    Complex zero coefficient array with first dimension = number of harmonics.
    """
    return jnp.zeros((basis.n_tones, *trailing_shape), dtype=dtype)


def set_single_rms_phasor_by_label(
    coeffs: ArrayLike,
    plan: CanonicalFrequencyPlan,
    *,
    label: str,
    rms_phasor: ArrayLike,
    set_conjugate: bool = True,
) -> jax.Array:
    """
    Insert one RMS phasor into a coefficient array by plan label.

    The positive-frequency coefficient is rms/sqrt(2). If set_conjugate is
    true and the signed-opposite frequency exists in the plan, that coefficient
    is set to conj(rms/sqrt(2)).
    """
    c = _as_complex_array(coeffs)
    pos = plan.position_of_label(label)
    f = float(plan.frequencies_hz[pos])
    x_pos = rms_phasor_to_positive_fourier(rms_phasor)
    c = c.at[pos].set(x_pos)

    if set_conjugate:
        matches = plan.find_frequency(-f)
        if matches:
            c = c.at[matches[0]].set(jnp.conj(x_pos))
    return c


def set_single_peak_phasor_by_label(
    coeffs: ArrayLike,
    plan: CanonicalFrequencyPlan,
    *,
    label: str,
    peak_phasor: ArrayLike,
    set_conjugate: bool = True,
) -> jax.Array:
    """
    Insert one peak phasor into a coefficient array by plan label.
    """
    c = _as_complex_array(coeffs)
    pos = plan.position_of_label(label)
    f = float(plan.frequencies_hz[pos])
    x_pos = peak_phasor_to_positive_fourier(peak_phasor)
    c = c.at[pos].set(x_pos)

    if set_conjugate:
        matches = plan.find_frequency(-f)
        if matches:
            c = c.at[matches[0]].set(jnp.conj(x_pos))
    return c


# ---------------------------------------------------------------------------
# Time grids and direct Fourier synthesis/projection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeGrid:
    """
    Uniform time grid for one period of a fundamental frequency.

    Parameters
    ----------
    t_s:
        Time samples in seconds, shape (Nt,).
    fundamental_frequency_hz:
        Fundamental frequency.
    endpoint:
        Whether the period endpoint is included. For FFT-style grids this
        should normally be false.
    """

    t_s: jax.Array
    fundamental_frequency_hz: float
    endpoint: bool = False

    def __post_init__(self) -> None:
        t = _as_1d_array("t_s", self.t_s, dtype=jnp.float64)
        if t.shape[0] <= 1:
            raise ValueError("TimeGrid requires at least two samples")
        if float(self.fundamental_frequency_hz) <= 0.0:
            raise ValueError("fundamental_frequency_hz must be positive")
        object.__setattr__(self, "t_s", t)
        object.__setattr__(self, "fundamental_frequency_hz", float(self.fundamental_frequency_hz))

    @property
    def n_samples(self) -> int:
        return int(self.t_s.shape[0])

    @property
    def period_s(self) -> float:
        return 1.0 / self.fundamental_frequency_hz

    @property
    def dt_s(self) -> float:
        return float(self.t_s[1] - self.t_s[0])

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.dt_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "fundamental_frequency_hz": self.fundamental_frequency_hz,
            "period_s": self.period_s,
            "dt_s": self.dt_s,
            "sample_rate_hz": self.sample_rate_hz,
            "endpoint": self.endpoint,
        }


def _canonical_make_time_grid(
    fundamental_frequency_hz: float,
    *,
    n_samples: int,
    endpoint: bool = False,
) -> TimeGrid:
    """
    Build a uniform time grid over one period.

    Use endpoint=False for Fourier/FFT projections to avoid duplicating t=0
    and t=T.
    """
    n_samples = _check_positive_int("n_samples", n_samples)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if fundamental_frequency_hz <= 0.0:
        raise ValueError("fundamental_frequency_hz must be positive")

    period = 1.0 / fundamental_frequency_hz
    t = jnp.linspace(0.0, period, n_samples, endpoint=endpoint)
    return TimeGrid(t_s=t, fundamental_frequency_hz=fundamental_frequency_hz, endpoint=endpoint)


def recommended_time_samples(
    *,
    n_tones: int,
    oversampling: int = 4,
    force_odd: bool = False,
    minimum: int = 32,
) -> int:
    """
    Recommend a time-grid size for direct nonlinear projection.

    For cubic nonlinearities, oversampling helps avoid aliasing in validation
    residuals. Industrial code can later use more tailored grids.
    """
    n_tones = _check_positive_int("n_tones", n_tones)
    oversampling = _check_positive_int("oversampling", oversampling)
    minimum = _check_positive_int("minimum", minimum)

    n = max(minimum, oversampling * (2 * n_tones + 1))
    if force_odd and n % 2 == 0:
        n += 1
    return int(n)


def synthesize_time_series(
    coeffs: ArrayLike,
    frequencies_hz: ArrayLike,
    t_s: ArrayLike,
) -> jax.Array:
    """
    Directly synthesize time samples from arbitrary signed frequencies.

        x(t_j) = sum_k X_k exp(i 2π f_k t_j)

    Parameters
    ----------
    coeffs:
        Fourier coefficients, shape (K, ...).
    frequencies_hz:
        Signed frequencies, shape (K,).
    t_s:
        Time grid, shape (Nt,).

    Returns
    -------
    samples:
        Shape (Nt, ...).
    """
    c = jnp.asarray(coeffs)
    f = _as_1d_array("frequencies_hz", frequencies_hz, dtype=jnp.float64)
    t = _as_1d_array("t_s", t_s, dtype=jnp.float64)
    if c.shape[0] != f.shape[0]:
        raise ValueError("coeffs first dimension must match frequencies length")

    phase = jnp.exp(1j * 2.0 * jnp.pi * t[:, None] * f[None, :])
    return jnp.einsum("tk,k...->t...", phase, c)


def project_time_series(
    samples: ArrayLike,
    frequencies_hz: ArrayLike,
    t_s: ArrayLike,
) -> jax.Array:
    """
    Directly project time samples onto arbitrary signed frequencies.

        X_k ≈ mean_j x(t_j) exp(-i 2π f_k t_j)

    This is exact for commensurate tones on a compatible uniform grid and
    approximate otherwise.
    """
    x = jnp.asarray(samples)
    f = _as_1d_array("frequencies_hz", frequencies_hz, dtype=jnp.float64)
    t = _as_1d_array("t_s", t_s, dtype=jnp.float64)

    if x.shape[0] != t.shape[0]:
        raise ValueError("samples first dimension must match time grid length")

    phase = jnp.exp(-1j * 2.0 * jnp.pi * t[:, None] * f[None, :])
    return jnp.einsum("tk,t...->k...", phase, x) / t.shape[0]


def synthesize_from_plan(
    coeffs: ArrayLike,
    plan: CanonicalFrequencyPlan,
    t_s: ArrayLike,
) -> jax.Array:
    """Synthesize time samples from a FrequencyPlan."""
    return synthesize_time_series(coeffs, plan.frequencies_hz, t_s)


def project_to_plan(
    samples: ArrayLike,
    plan: CanonicalFrequencyPlan,
    t_s: ArrayLike,
) -> jax.Array:
    """Project time samples onto a FrequencyPlan."""
    return project_time_series(samples, plan.frequencies_hz, t_s)


def synthesize_from_basis(
    coeffs: ArrayLike,
    basis: HarmonicIndexBasis,
    t_s: ArrayLike,
) -> jax.Array:
    """Synthesize time samples from a HarmonicIndexBasis."""
    return synthesize_time_series(coeffs, basis.frequencies_hz, t_s)


def project_to_basis(
    samples: ArrayLike,
    basis: HarmonicIndexBasis,
    t_s: ArrayLike,
) -> jax.Array:
    """Project time samples onto a HarmonicIndexBasis."""
    return project_time_series(samples, basis.frequencies_hz, t_s)


# ---------------------------------------------------------------------------
# Frequency resolution / commensurability helpers
# ---------------------------------------------------------------------------

def infer_fundamental_from_frequencies(
    frequencies_hz: ArrayLike,
    *,
    tolerance_hz: float = 1e-3,
    max_denominator: int = 10_000,
) -> float:
    """
    Best-effort fundamental frequency inference for nearly commensurate tones.

    This routine is intentionally conservative and intended for diagnostics,
    not for critical numerical logic. It uses a simple floating-point gcd-like
    iteration.

    For pump-only plans, prefer the explicit pump frequency.
    """
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)
    abs_f = jnp.sort(jnp.abs(f[f != 0.0]))
    if abs_f.size == 0:
        raise ValueError("Cannot infer fundamental from only DC/zero frequencies")

    # Floating point Euclidean algorithm.
    g = float(abs_f[0])
    for value in abs_f[1:].tolist():
        x = float(value)
        while x > tolerance_hz:
            g, x = x, g % x
        if g < tolerance_hz:
            g = tolerance_hz
            break
    return float(g)


def frequency_integer_indices(
    frequencies_hz: ArrayLike,
    fundamental_frequency_hz: float,
    *,
    atol_hz: float = 1e-6,
) -> jax.Array:
    """
    Express frequencies as integer multiples of a fundamental.

    Raises if any frequency is not close to an integer multiple.
    """
    if fundamental_frequency_hz <= 0.0:
        raise ValueError("fundamental_frequency_hz must be positive")
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)
    raw = f / fundamental_frequency_hz
    rounded = jnp.rint(raw).astype(jnp.int32)
    err = jnp.abs(f - rounded.astype(jnp.float64) * fundamental_frequency_hz)
    if bool(jnp.any(err > atol_hz)):
        max_err = float(jnp.max(err))
        raise ValueError(
            f"Frequencies are not integer multiples of {fundamental_frequency_hz} Hz; "
            f"max error={max_err} Hz"
        )
    return rounded


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def coefficient_power_summary(coeffs: ArrayLike) -> dict[str, float]:
    """
    Return basic magnitude summary for a coefficient array.
    """
    c = jnp.asarray(coeffs)
    mag = jnp.abs(c)
    return {
        "max_abs": float(jnp.max(mag)),
        "min_abs": float(jnp.min(mag)),
        "mean_abs": float(jnp.mean(mag)),
        "l2_norm": float(jnp.linalg.norm(c.ravel())),
        "size": float(c.size),
    }


def harmonic_table(
    frequencies_hz: ArrayLike,
    coeffs: ArrayLike | None = None,
    labels: Iterable[str] | None = None,
) -> str:
    """
    Build a human-readable table of frequencies and optional coefficient magnitudes.
    """
    f = _as_1d_array("frequencies_hz", frequencies_hz, dtype=jnp.float64)
    labels_list = list(labels) if labels is not None else [f"k{i}" for i in range(f.shape[0])]
    if len(labels_list) != f.shape[0]:
        raise ValueError("labels length must match frequency count")

    lines = ["idx | label | f_Hz | f_GHz | |coeff|", "---: | --- | ---: | ---: | ---:"]
    if coeffs is None:
        mags = [float("nan")] * f.shape[0]
    else:
        c = jnp.asarray(coeffs)
        if c.shape[0] != f.shape[0]:
            raise ValueError("coeffs first dimension must match frequency count")
        mags = [float(jnp.max(jnp.abs(c[i]))) for i in range(f.shape[0])]

    for i in range(f.shape[0]):
        fi = float(f[i])
        lines.append(f"{i} | {labels_list[i]} | {fi:.9e} | {fi/1e9:.9g} | {mags[i]:.9e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compatibility public helpers for selected-harmonic HB tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HarmonicSet:
    """Ordered, nonzero integer harmonic basis."""

    orders: tuple[int, ...]

    def __post_init__(self) -> None:
        raw = list(self.orders)
        if not raw:
            raise ValueError("orders must be non-empty")
        if any(int(k) != k for k in raw):
            raise ValueError("orders must be integer-valued")
        orders = tuple(int(k) for k in raw)
        if any(k == 0 for k in orders):
            raise ValueError("zero-order/DC harmonics are not in this basis")
        if len(set(orders)) != len(orders):
            raise ValueError("orders must be unique")
        object.__setattr__(self, "orders", orders)

    @property
    def harmonic_orders(self) -> tuple[int, ...]:
        return self.orders

    def index(self, order: int) -> int:
        try:
            return self.orders.index(int(order))
        except ValueError as exc:
            raise ValueError(f"harmonic order {order} is not present") from exc

    def to_dict(self) -> dict[str, list[int]]:
        return {"orders": list(self.orders)}


@dataclass(frozen=True)
class PumpSignalFrequencyPlan:
    """Minimal pump/signal/idler frequency plan for public API stability."""

    pump_frequency_hz: float
    signal_frequency_hz: float
    harmonic_orders: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.pump_frequency_hz <= 0.0:
            raise ValueError("pump_frequency_hz must be positive")
        if self.signal_frequency_hz <= 0.0:
            raise ValueError("signal_frequency_hz must be positive")
        if 2.0 * self.pump_frequency_hz - self.signal_frequency_hz <= 0.0:
            raise ValueError("idler frequency 2*fp - fs must be positive")
        hset = HarmonicSet(tuple(self.harmonic_orders))
        object.__setattr__(self, "harmonic_orders", hset.orders)

    @property
    def orders(self) -> tuple[int, ...]:
        return self.harmonic_orders

    @property
    def idler_frequency_hz(self) -> float:
        return 2.0 * self.pump_frequency_hz - self.signal_frequency_hz

    @property
    def frequencies_hz(self) -> jax.Array:
        return jnp.asarray(self.harmonic_orders, dtype=jnp.float64) * self.pump_frequency_hz

    @property
    def angular_frequencies_rad_s(self) -> jax.Array:
        return 2.0 * jnp.pi * self.frequencies_hz

    def to_dict(self) -> dict[str, Any]:
        return {
            "pump_frequency_hz": self.pump_frequency_hz,
            "signal_frequency_hz": self.signal_frequency_hz,
            "idler_frequency_hz": self.idler_frequency_hz,
            "harmonic_orders": list(self.harmonic_orders),
            "frequencies_hz": [float(v) for v in self.frequencies_hz.tolist()],
        }


SelectedHarmonicFrequencyPlan = PumpSignalFrequencyPlan


def symmetric_harmonic_orders(max_order: int, *, odd_only: bool = True) -> jax.Array:
    if max_order <= 0:
        raise ValueError("max_order must be positive")
    step = 2 if odd_only else 1
    start = 1 if odd_only else 1
    pos = jnp.arange(start, int(max_order) + 1, step, dtype=jnp.int64)
    return jnp.concatenate((-pos[::-1], pos))


def make_selected_harmonic_frequency_plan(
    pump_frequency_hz: float,
    signal_frequency_hz: float,
    harmonic_orders: Iterable[int],
) -> PumpSignalFrequencyPlan:
    return PumpSignalFrequencyPlan(
        pump_frequency_hz=float(pump_frequency_hz),
        signal_frequency_hz=float(signal_frequency_hz),
        harmonic_orders=tuple(int(k) if int(k) == k else k for k in harmonic_orders),
    )


_rich_make_time_grid = _canonical_make_time_grid


def make_time_grid(
    fundamental_frequency_hz: float | None = None,
    *,
    n_samples: int | None = None,
    n_time: int | None = None,
    n: int | None = None,
    omega0_rad_s: float | None = None,
    omega0: float | None = None,
    period_s: float | None = None,
    endpoint: bool = False,
    **_: Any,
) -> TimeGrid:
    if n_samples is None:
        n_samples = n_time if n_time is not None else n
    if n_samples is None:
        raise ValueError("n_samples/n_time is required")
    if fundamental_frequency_hz is None:
        omega = omega0_rad_s if omega0_rad_s is not None else omega0
        if omega is not None:
            fundamental_frequency_hz = float(omega) / (2.0 * jnp.pi)
        elif period_s is not None:
            fundamental_frequency_hz = 1.0 / float(period_s)
    if fundamental_frequency_hz is None:
        raise ValueError("fundamental_frequency_hz or omega0_rad_s is required")
    return _rich_make_time_grid(
        float(fundamental_frequency_hz),
        n_samples=int(n_samples),
        endpoint=endpoint,
    )


def make_selected_harmonic_time_grid(
    *,
    n_time: int | None = None,
    n: int | None = None,
    n_samples: int | None = None,
    omega0_rad_s: float | None = None,
    omega0: float | None = None,
    period_s: float | None = None,
    fundamental_frequency_hz: float | None = None,
    **_: Any,
) -> jax.Array:
    count = n_time if n_time is not None else (n if n is not None else n_samples)
    if count is None or int(count) <= 0:
        raise ValueError("n_time must be positive")
    omega = omega0_rad_s if omega0_rad_s is not None else omega0
    if period_s is None:
        if omega is not None:
            period_s = 2.0 * jnp.pi / float(omega)
        elif fundamental_frequency_hz is not None:
            period_s = 1.0 / float(fundamental_frequency_hz)
        else:
            period_s = 1.0
    return jnp.arange(int(count), dtype=jnp.float64) * (float(period_s) / int(count))


def _timegrid_array(self: TimeGrid, dtype: Any | None = None) -> Any:
    import numpy as _np

    return _np.asarray(self.t_s, dtype=dtype)


TimeGrid.__array__ = _timegrid_array  # type: ignore[attr-defined]


def harmonic_basis_matrix(
    *,
    t: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    orders: ArrayLike | None = None,
    harmonic_orders: ArrayLike | None = None,
    fundamental_frequency_hz: float | None = None,
    f0_hz: float | None = None,
    omega0_rad_s: float | None = None,
    omega0: float | None = None,
    **_: Any,
) -> jax.Array:
    t_arr = jnp.asarray(time_s if time_s is not None else t, dtype=jnp.float64)
    order_arr = _compat_orders(harmonic_orders if harmonic_orders is not None else orders, allow_zero=True)
    om = omega0_rad_s if omega0_rad_s is not None else omega0
    if om is None:
        f0 = fundamental_frequency_hz if fundamental_frequency_hz is not None else f0_hz
        if f0 is None:
            raise ValueError("fundamental frequency is required")
        om = 2.0 * jnp.pi * float(f0)
    return jnp.exp(1j * jnp.outer(t_arr * float(om), order_arr))


def _compat_orders(orders: ArrayLike, *, allow_zero: bool = False) -> jax.Array:
    arr = jnp.asarray(orders)
    if arr.ndim != 1:
        raise ValueError("orders must be one-dimensional")
    if arr.size == 0:
        raise ValueError("orders must be non-empty")
    if not bool(jnp.all(arr == jnp.round(arr))):
        raise ValueError("orders must be integer-valued")
    out = arr.astype(jnp.int64)
    if not allow_zero and bool(jnp.any(out == 0)):
        raise ValueError("zero-order/DC harmonics are not supported here")
    if len(set(int(v) for v in out.tolist())) != int(out.size):
        raise ValueError("orders must be unique")
    return out


def conjugate_partner_index(
    *,
    harmonic_set: HarmonicSet | None = None,
    orders: ArrayLike | None = None,
    order: int | None = None,
    harmonic: int | None = None,
    index: int | None = None,
    **_: Any,
) -> int:
    order_arr = list(harmonic_set.orders if harmonic_set is not None else _compat_orders(orders).tolist())
    target = order if order is not None else harmonic
    if target is None:
        if index is None:
            raise ValueError("order or index is required")
        target = order_arr[int(index)]
    return order_arr.index(-int(target))


def enforce_conjugate_symmetry(
    *,
    positive_coeffs: Mapping[int, complex] | None = None,
    coefficients: Mapping[int, complex] | None = None,
    orders: ArrayLike,
    harmonic_orders: ArrayLike | None = None,
    **_: Any,
) -> jax.Array:
    coeff_map = positive_coeffs if positive_coeffs is not None else coefficients
    if coeff_map is None:
        raise ValueError("positive coefficients are required")
    order_arr = _compat_orders(harmonic_orders if harmonic_orders is not None else orders)
    values = []
    for k in order_arr.tolist():
        if k > 0:
            values.append(complex(coeff_map.get(int(k), 0.0)))
        else:
            values.append(complex(jnp.conj(jnp.asarray(coeff_map.get(int(-k), 0.0)))))
    return jnp.asarray(values, dtype=jnp.complex128)


def make_frequency_plan(
    pump_frequency_hz: float,
    signal_frequency_hz: float,
    harmonic_orders: Iterable[int],
) -> PumpSignalFrequencyPlan:
    warnings.warn(
        "twpa.core.harmonics.make_frequency_plan is compatibility-only. "
        "Use make_selected_harmonic_frequency_plan(...) for selected-harmonic "
        "tests or twpa.core.frequency_plan helpers for production plans.",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_selected_harmonic_frequency_plan(
        pump_frequency_hz,
        signal_frequency_hz,
        harmonic_orders,
    )


__all__ = [
    "ArrayLike",
    "complex_to_real_vector",
    "real_vector_to_complex",
    "complex_tree_to_real_vector",
    "rms_phasor_to_positive_fourier",
    "positive_fourier_to_rms_phasor",
    "peak_phasor_to_positive_fourier",
    "positive_fourier_to_peak_phasor",
    "make_real_sinusoid_coefficients_from_rms",
    "HarmonicIndexBasis",
    "HarmonicSet",
    "PumpSignalFrequencyPlan",
    "SelectedHarmonicFrequencyPlan",
    "CanonicalFrequencyPlan",
    "symmetric_harmonic_orders",
    "make_frequency_plan",
    "make_selected_harmonic_frequency_plan",
    "make_harmonic_index_basis",
    "enforce_conjugate_symmetry_by_indices",
    "conjugate_symmetry_error_by_indices",
    "enforce_conjugate_symmetry_by_frequencies",
    "zeros_for_plan",
    "zeros_for_basis",
    "set_single_rms_phasor_by_label",
    "set_single_peak_phasor_by_label",
    "TimeGrid",
    "make_time_grid",
    "make_selected_harmonic_time_grid",
    "recommended_time_samples",
    "synthesize_time_series",
    "project_time_series",
    "synthesize_from_plan",
    "project_to_plan",
    "synthesize_from_basis",
    "project_to_basis",
    "infer_fundamental_from_frequencies",
    "frequency_integer_indices",
    "coefficient_power_summary",
    "harmonic_table",
    "harmonic_basis_matrix",
    "conjugate_partner_index",
    "enforce_conjugate_symmetry",
]
