"""
twpa.core.hb_fft
================

Harmonic-balance nonlinear projection utilities.

This module is the bridge between frequency-domain unknowns and nonlinear
time-domain constitutive laws.

For a nonlinear inductor, the cleanest implementation path is:

    I_k  ->  i(t_j)
    lambda(t_j) = L(i) i(t_j)
    Lambda_k <- projection(lambda(t_j))
    V_k = i omega_k Lambda_k

where lambda is the flux linkage.

For the default KI-TWPA model,

    L(I) = L0 [1 + beta_nl (I/I_star)^2]

so

    lambda(I) = L(I) I
              = L0 [I + beta_nl I^3/I_star^2]

This is exactly the kind of "first nonlinear residual" layer needed before
building a full distributed ladder HB solver.

Important limitations
---------------------
This module performs single-time-axis direct Fourier synthesis/projection.
That is exact for commensurate tone sets on a compatible time grid, e.g.
pump-only harmonics. For truly incommensurate multi-tone HB, a multi-dimensional
HB basis is needed. The industrial workflow avoids needing a full incommensurate
finite-signal nonlinear solve for every frequency by using:

    1. pump-only HB,
    2. linearization around the pumped solution,
    3. small-signal conversion matrix.

Finite-signal compression can later use a controlled commensurate approximation
or a proper multi-fundamental HB extension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import prod
from typing import Any, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp

from .frequency_plan import FrequencyPlan
from .harmonics import (
    TimeGrid,
    coefficient_power_summary,
    enforce_conjugate_symmetry_by_frequencies,
    frequency_integer_indices,
    infer_fundamental_from_frequencies,
    make_time_grid,
    project_time_series,
    recommended_time_samples,
    synthesize_time_series,
)


ArrayLike = Any


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _as_complex_array(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    if arr.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension, got shape {arr.shape}")
    return arr


def _as_float_array(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        arr = arr.astype(jnp.float64)
    return arr


def _check_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _broadcast_param_to_samples(param: ArrayLike, sample_shape: tuple[int, ...]) -> jax.Array:
    """
    Broadcast a scalar/array parameter to a time-sample trailing shape.

    If samples have shape (Nt, ...), sample_shape is (...).
    """
    p = jnp.asarray(param)
    if p.ndim == 0:
        return p
    try:
        return jnp.broadcast_to(p, sample_shape)
    except ValueError as exc:
        raise ValueError(
            f"Parameter with shape {p.shape} cannot broadcast to sample trailing "
            f"shape {sample_shape}"
        ) from exc


# ---------------------------------------------------------------------------
# Projection configuration
# ---------------------------------------------------------------------------

class ProjectionMode(str, Enum):
    """HB nonlinear projection basis."""

    AUTO = "auto"
    SINGLE_PERIOD = "single_period"
    MULTI_FUNDAMENTAL = "multi_fundamental"


@dataclass(frozen=True)
class HBProjectionConfig:
    """
    Configuration for direct HB synthesis/projection.

    Parameters
    ----------
    n_time_samples:
        Number of time samples. If None, a recommendation is computed from
        the number of tones and oversampling.
    oversampling:
        Oversampling factor used when n_time_samples is None.
    endpoint:
        Whether the final time sample equals one full period. Usually false.
    force_real_time_signal:
        If true, small imaginary numerical residue in synthesized time-domain
        signals is discarded by taking real().
    enforce_conjugate_symmetry:
        If true, symmetrize input coefficients before synthesis and projected
        output coefficients after projection.
    commensurability_atol_hz:
        Tolerance for checking whether frequencies are integer multiples of
        the fundamental.
    """

    n_time_samples: int | None = None
    oversampling: int = 8
    endpoint: bool = False
    force_real_time_signal: bool = True
    enforce_conjugate_symmetry: bool = True
    commensurability_atol_hz: float = 1e-3
    mode: ProjectionMode = ProjectionMode.AUTO
    multi_fundamental_samples_per_axis: tuple[int, ...] | None = None
    max_projection_samples: int = 65_536

    def __post_init__(self) -> None:
        if self.n_time_samples is not None and int(self.n_time_samples) <= 1:
            raise ValueError("n_time_samples must be > 1")
        if int(self.oversampling) <= 0:
            raise ValueError("oversampling must be positive")
        object.__setattr__(self, "oversampling", int(self.oversampling))
        _check_positive("commensurability_atol_hz", self.commensurability_atol_hz)
        object.__setattr__(self, "mode", ProjectionMode(self.mode))
        if self.multi_fundamental_samples_per_axis is not None:
            samples = tuple(int(v) for v in self.multi_fundamental_samples_per_axis)
            if not samples or any(v <= 1 for v in samples):
                raise ValueError("multi_fundamental_samples_per_axis values must be > 1")
            object.__setattr__(self, "multi_fundamental_samples_per_axis", samples)
        if int(self.max_projection_samples) <= 0:
            raise ValueError("max_projection_samples must be positive")
        object.__setattr__(self, "max_projection_samples", int(self.max_projection_samples))

    def with_updates(self, **kwargs: Any) -> "HBProjectionConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_time_samples": self.n_time_samples,
            "oversampling": self.oversampling,
            "endpoint": self.endpoint,
            "force_real_time_signal": self.force_real_time_signal,
            "enforce_conjugate_symmetry": self.enforce_conjugate_symmetry,
            "commensurability_atol_hz": self.commensurability_atol_hz,
            "mode": self.mode.value,
            "multi_fundamental_samples_per_axis": self.multi_fundamental_samples_per_axis,
            "max_projection_samples": self.max_projection_samples,
        }


@dataclass(frozen=True)
class HBProjectionGrid:
    """
    Time grid plus frequency metadata used for direct projection.
    """

    time_grid: TimeGrid
    frequencies_hz: jax.Array
    integer_indices: jax.Array | None
    is_commensurate: bool
    mode: ProjectionMode = ProjectionMode.SINGLE_PERIOD
    lattice_indices: jax.Array | None = None
    phase_points_rad: jax.Array | None = None
    samples_per_axis: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def t_s(self) -> jax.Array:
        return self.time_grid.t_s

    @property
    def n_time_samples(self) -> int:
        return self.time_grid.n_samples

    @property
    def fundamental_frequency_hz(self) -> float:
        return self.time_grid.fundamental_frequency_hz

    @property
    def angular_frequencies_rad_s(self) -> jax.Array:
        return 2.0 * jnp.pi * self.frequencies_hz

    def synthesize(self, coeffs: ArrayLike, *, force_real: bool = False) -> jax.Array:
        """Synthesize samples using this grid's single-period or torus basis."""
        if self.mode == ProjectionMode.MULTI_FUNDAMENTAL:
            if self.phase_points_rad is None or self.lattice_indices is None:
                raise ValueError("Multi-fundamental grid is missing torus basis metadata")
            basis = jnp.exp(1j * (self.phase_points_rad @ self.lattice_indices.T))
            samples = jnp.einsum("tk,k...->t...", basis, jnp.asarray(coeffs))
            return jnp.real(samples) if force_real else samples
        return synthesize_time_series(coeffs, self.frequencies_hz, self.t_s).real if force_real else synthesize_time_series(coeffs, self.frequencies_hz, self.t_s)

    def project(self, samples: ArrayLike) -> jax.Array:
        """Project samples using this grid's single-period or torus basis."""
        if self.mode == ProjectionMode.MULTI_FUNDAMENTAL:
            if self.phase_points_rad is None or self.lattice_indices is None:
                raise ValueError("Multi-fundamental grid is missing torus basis metadata")
            basis = jnp.exp(1j * (self.phase_points_rad @ self.lattice_indices.T))
            return jnp.einsum("tk,t...->k...", jnp.conj(basis), jnp.asarray(samples)) / basis.shape[0]
        return project_time_series(samples, self.frequencies_hz, self.t_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_grid": self.time_grid.to_dict(),
            "n_frequencies": int(self.frequencies_hz.shape[0]),
            "is_commensurate": bool(self.is_commensurate),
            "mode": self.mode.value,
            "integer_indices": (
                None
                if self.integer_indices is None
                else [int(v) for v in self.integer_indices.tolist()]
            ),
            "lattice_indices": (
                None
                if self.lattice_indices is None
                else [[int(v) for v in row] for row in self.lattice_indices.tolist()]
            ),
            "samples_per_axis": self.samples_per_axis,
            "metadata": dict(self.metadata or {}),
        }


def make_projection_grid(
    frequencies_hz: ArrayLike,
    *,
    fundamental_frequency_hz: float | None = None,
    config: HBProjectionConfig | None = None,
) -> HBProjectionGrid:
    """
    Create a time grid for direct Fourier synthesis/projection.

    If fundamental_frequency_hz is None, the function tries to infer a
    fundamental from the given frequencies. For pump-only plans, pass the pump
    frequency explicitly.
    """
    cfg = config or HBProjectionConfig()
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)
    if f.ndim != 1:
        raise ValueError(f"frequencies_hz must be 1D, got shape {f.shape}")
    if f.shape[0] == 0:
        raise ValueError("frequencies_hz cannot be empty")

    if fundamental_frequency_hz is None:
        fundamental = infer_fundamental_from_frequencies(
            f,
            tolerance_hz=cfg.commensurability_atol_hz,
        )
        inferred = True
    else:
        _check_positive("fundamental_frequency_hz", fundamental_frequency_hz)
        fundamental = float(fundamental_frequency_hz)
        inferred = False

    try:
        integer_indices = frequency_integer_indices(
            f,
            fundamental,
            atol_hz=cfg.commensurability_atol_hz,
        )
        is_commensurate = True
    except ValueError:
        integer_indices = None
        is_commensurate = False

    if cfg.n_time_samples is None:
        n_samples = recommended_time_samples(
            n_tones=int(f.shape[0]),
            oversampling=cfg.oversampling,
            force_odd=False,
            minimum=32,
        )
    else:
        n_samples = int(cfg.n_time_samples)

    time_grid = make_time_grid(
        fundamental,
        n_samples=n_samples,
        endpoint=cfg.endpoint,
    )

    return HBProjectionGrid(
        time_grid=time_grid,
        frequencies_hz=f,
        integer_indices=integer_indices,
        is_commensurate=is_commensurate,
        mode=ProjectionMode.SINGLE_PERIOD,
        metadata={
            "fundamental_inferred": inferred,
            "config": cfg.to_dict(),
        },
    )


def make_multi_fundamental_projection_grid(
    frequencies_hz: ArrayLike,
    *,
    fundamental_frequencies_hz: ArrayLike,
    lattice_indices: ArrayLike,
    config: HBProjectionConfig | None = None,
) -> HBProjectionGrid:
    """Create an N-dimensional torus grid for incommensurate HB projection."""
    cfg = config or HBProjectionConfig()
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)
    fundamentals = jnp.asarray(fundamental_frequencies_hz, dtype=jnp.float64)
    lattice = jnp.asarray(lattice_indices, dtype=jnp.int32)
    if f.ndim != 1 or f.shape[0] == 0:
        raise ValueError("frequencies_hz must be a non-empty 1D array")
    if fundamentals.ndim != 1 or fundamentals.shape[0] == 0:
        raise ValueError("fundamental_frequencies_hz must be a non-empty 1D array")
    if bool(jnp.any(fundamentals <= 0.0)):
        raise ValueError("fundamental_frequencies_hz must be positive")
    if lattice.shape != (f.shape[0], fundamentals.shape[0]):
        raise ValueError(
            "lattice_indices shape must be "
            f"{(int(f.shape[0]), int(fundamentals.shape[0]))}, got {lattice.shape}"
        )
    represented = lattice.astype(jnp.float64) @ fundamentals
    if not bool(jnp.allclose(represented, f, rtol=1e-12, atol=cfg.commensurability_atol_hz)):
        raise ValueError("lattice_indices do not represent frequencies_hz")

    n_axes = int(fundamentals.shape[0])
    if cfg.multi_fundamental_samples_per_axis is not None:
        samples_per_axis = cfg.multi_fundamental_samples_per_axis
        if len(samples_per_axis) != n_axes:
            raise ValueError(
                "multi_fundamental_samples_per_axis length must match the number of fundamentals"
            )
    else:
        extents = jnp.max(jnp.abs(lattice), axis=0).tolist()
        samples_per_axis = tuple(max(8, 2 * int(extent) * cfg.oversampling) for extent in extents)

    total_samples = prod(samples_per_axis)
    if total_samples > cfg.max_projection_samples:
        raise ValueError(
            "Multi-fundamental HB projection grid refused unsafe allocation: "
            f"{total_samples} samples exceed max_projection_samples={cfg.max_projection_samples}"
        )
    for axis, (count, extent) in enumerate(zip(samples_per_axis, jnp.max(jnp.abs(lattice), axis=0).tolist())):
        if count <= 2 * int(extent):
            raise ValueError(
                f"Multi-fundamental axis {axis} is under-resolved: "
                f"samples={count}, max lattice index={int(extent)}"
            )

    axes = [
        2.0 * jnp.pi * jnp.arange(count, dtype=jnp.float64) / count
        for count in samples_per_axis
    ]
    meshes = jnp.meshgrid(*axes, indexing="ij")
    phase_points = jnp.stack([mesh.reshape(-1) for mesh in meshes], axis=1)
    placeholder_time_grid = make_time_grid(
        float(fundamentals[0]),
        n_samples=total_samples,
        endpoint=False,
    )
    return HBProjectionGrid(
        time_grid=placeholder_time_grid,
        frequencies_hz=f,
        integer_indices=None,
        is_commensurate=False,
        mode=ProjectionMode.MULTI_FUNDAMENTAL,
        lattice_indices=lattice,
        phase_points_rad=phase_points,
        samples_per_axis=samples_per_axis,
        metadata={
            "fundamental_frequencies_hz": [float(v) for v in fundamentals.tolist()],
            "config": cfg.to_dict(),
        },
    )


def make_projection_grid_from_plan(
    plan: FrequencyPlan,
    *,
    fundamental_frequency_hz: float | None = None,
    config: HBProjectionConfig | None = None,
) -> HBProjectionGrid:
    """
    Create a projection grid from a FrequencyPlan.

    If fundamental_frequency_hz is omitted and the plan has a reference pump,
    the pump frequency is used.
    """
    if fundamental_frequency_hz is None and plan.reference_pump_hz is not None:
        fundamental_frequency_hz = plan.reference_pump_hz
    return make_projection_grid(
        plan.frequencies_hz,
        fundamental_frequency_hz=fundamental_frequency_hz,
        config=config,
    )


# ---------------------------------------------------------------------------
# Direct projection wrappers
# ---------------------------------------------------------------------------

def coefficients_to_time(
    coeffs: ArrayLike,
    frequencies_hz: ArrayLike,
    t_s: ArrayLike,
    *,
    force_real: bool = False,
) -> jax.Array:
    """
    Synthesize time-domain samples from Fourier coefficients.

    Parameters
    ----------
    coeffs:
        Shape (K, ...).
    frequencies_hz:
        Shape (K,).
    t_s:
        Shape (Nt,).
    force_real:
        If true, return real(samples). This should only be used when the
        coefficient set is meant to represent a real waveform.
    """
    samples = synthesize_time_series(coeffs, frequencies_hz, t_s)
    if force_real:
        return jnp.real(samples)
    return samples


def time_to_coefficients(
    samples: ArrayLike,
    frequencies_hz: ArrayLike,
    t_s: ArrayLike,
) -> jax.Array:
    """
    Project time-domain samples back to Fourier coefficients.

    Parameters
    ----------
    samples:
        Shape (Nt, ...).
    frequencies_hz:
        Shape (K,).
    t_s:
        Shape (Nt,).
    """
    return project_time_series(samples, frequencies_hz, t_s)


def project_nonlinear_time_function(
    coeffs: ArrayLike,
    frequencies_hz: ArrayLike,
    t_s: ArrayLike,
    time_function: Any,
    *,
    force_real_input: bool = True,
    enforce_symmetry: bool = False,
    function_kwargs: Mapping[str, Any] | None = None,
) -> jax.Array:
    """
    Generic nonlinear projection.

    Steps:
        coeffs -> x(t)
        y(t) = time_function(x(t), **function_kwargs)
        y(t) -> output coefficients

    This is the reusable engine behind nonlinear inductor projection.
    """
    c = _as_complex_array("coeffs", coeffs)
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)
    kwargs = dict(function_kwargs or {})

    if enforce_symmetry:
        c = enforce_conjugate_symmetry_by_frequencies(c, f)

    x_t = _production_coefficients_to_time(c, f, t_s, force_real=force_real_input)
    y_t = time_function(x_t, **kwargs)
    y_coeffs = _production_time_to_coefficients(y_t, f, t_s)

    if enforce_symmetry:
        y_coeffs = enforce_conjugate_symmetry_by_frequencies(y_coeffs, f)

    return y_coeffs


# ---------------------------------------------------------------------------
# Kinetic-inductance / generic chi(3) nonlinear inductor laws
# ---------------------------------------------------------------------------

def nonlinear_inductance_time(
    current_t_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
) -> jax.Array:
    """
    Current-dependent nonlinear inductance in the time domain.

        L(I) = L0 [1 + beta_nl (I/I*)^2 + q (I/I*)^4]

    Parameters can be scalars or arrays broadcastable to current_t_A trailing
    dimensions.
    """
    i_t = _as_float_array("current_t_A", current_t_A)
    sample_shape = i_t.shape[1:]

    L0 = _broadcast_param_to_samples(L0_H, sample_shape)
    I_star = _broadcast_param_to_samples(I_star_A, sample_shape)
    beta = _broadcast_param_to_samples(beta_nl, sample_shape)
    q = _broadcast_param_to_samples(quartic_coefficient, sample_shape)

    ratio = i_t / I_star
    return L0 * (1.0 + beta * ratio**2 + q * ratio**4)


def nonlinear_flux_linkage_time(
    current_t_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
) -> jax.Array:
    """
    Nonlinear flux linkage lambda(I) = L(I) I.

    For the default cubic KI model:

        lambda(I) = L0 [I + beta_nl I^3/I_star^2]

    With quartic correction in L(I):

        lambda(I) = L0 [I + beta_nl I^3/I_star^2 + q I^5/I_star^4]
    """
    i_t = _as_float_array("current_t_A", current_t_A)
    L_t = nonlinear_inductance_time(
        i_t,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
    )
    return L_t * i_t


def nonlinear_incremental_inductance_time(
    current_t_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
) -> jax.Array:
    """
    Incremental inductance d lambda / dI.

    If

        lambda(I) = L0 [I + beta I^3/I*^2 + q I^5/I*^4],

    then

        d lambda/dI = L0 [1 + 3 beta I^2/I*^2 + 5 q I^4/I*^4].

    This is useful for analytic Jacobians/preconditioners later.
    """
    i_t = _as_float_array("current_t_A", current_t_A)
    sample_shape = i_t.shape[1:]

    L0 = _broadcast_param_to_samples(L0_H, sample_shape)
    I_star = _broadcast_param_to_samples(I_star_A, sample_shape)
    beta = _broadcast_param_to_samples(beta_nl, sample_shape)
    q = _broadcast_param_to_samples(quartic_coefficient, sample_shape)

    ratio = i_t / I_star
    return L0 * (1.0 + 3.0 * beta * ratio**2 + 5.0 * q * ratio**4)


def voltage_coefficients_from_flux_coefficients(
    flux_coeffs_Wb: ArrayLike,
    frequencies_hz: ArrayLike,
) -> jax.Array:
    """
    Convert flux-linkage Fourier coefficients to voltage coefficients.

        V_k = d lambda/dt = i omega_k Lambda_k
    """
    lam = _as_complex_array("flux_coeffs_Wb", flux_coeffs_Wb)
    omega = 2.0 * jnp.pi * jnp.asarray(frequencies_hz, dtype=jnp.float64)

    if lam.shape[0] != omega.shape[0]:
        raise ValueError("flux_coeffs first dimension must match frequencies length")

    return 1j * omega.reshape((omega.shape[0],) + (1,) * (lam.ndim - 1)) * lam


def linear_inductor_voltage_coefficients(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
) -> jax.Array:
    """
    Linear inductor voltage coefficients.

        V_k = i omega_k L0 I_k
    """
    i_coeff = _as_complex_array("current_coeffs_A", current_coeffs_A)
    omega = 2.0 * jnp.pi * jnp.asarray(frequencies_hz, dtype=jnp.float64)
    L0 = jnp.asarray(L0_H)

    return (
        1j
        * omega.reshape((omega.shape[0],) + (1,) * (i_coeff.ndim - 1))
        * L0
        * i_coeff
    )


def nonlinear_inductor_voltage_coefficients(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
    projection_grid: HBProjectionGrid | None = None,
    config: HBProjectionConfig | None = None,
    fundamental_frequency_hz: float | None = None,
) -> jax.Array:
    """
    Project nonlinear inductor voltage coefficients.

    This computes:

        I_k -> i(t)
        lambda(t) = L(i) i(t)
        Lambda_k <- projection(lambda(t))
        V_k = i omega_k Lambda_k

    Parameters
    ----------
    current_coeffs_A:
        Fourier coefficients of current, shape (K, ...).
    frequencies_hz:
        Signed frequencies, shape (K,).
    L0_H:
        Linear inductance. Scalar or broadcastable to trailing dimensions.
    I_star_A:
        Nonlinearity current scale. Scalar or broadcastable.
    beta_nl:
        Nonlinear coefficient. KI default is 1.
    quartic_coefficient:
        Optional quartic correction in L(I).
    projection_grid:
        Optional precomputed grid.
    config:
        Projection config if grid is not provided.
    fundamental_frequency_hz:
        Optional fundamental if grid is not provided.

    Returns
    -------
    voltage_coeffs:
        Complex voltage coefficients, shape (K, ...).
    """
    i_coeff = _as_complex_array("current_coeffs_A", current_coeffs_A)
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)
    if i_coeff.shape[0] != f.shape[0]:
        raise ValueError("current_coeffs first dimension must match frequencies length")

    cfg = config or HBProjectionConfig()

    grid = projection_grid
    if grid is None:
        grid = make_projection_grid(
            f,
            fundamental_frequency_hz=fundamental_frequency_hz,
            config=cfg,
        )

    coeffs_for_synthesis = i_coeff
    if cfg.enforce_conjugate_symmetry:
        coeffs_for_synthesis = enforce_conjugate_symmetry_by_frequencies(
            coeffs_for_synthesis,
            f,
        )

    current_t = grid.synthesize(
        coeffs_for_synthesis,
        force_real=cfg.force_real_time_signal,
    )

    flux_t = nonlinear_flux_linkage_time(
        current_t,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
    )

    flux_coeffs = grid.project(flux_t)

    if cfg.enforce_conjugate_symmetry:
        flux_coeffs = enforce_conjugate_symmetry_by_frequencies(flux_coeffs, f)

    return voltage_coefficients_from_flux_coefficients(flux_coeffs, f)


def nonlinear_inductor_flux_coefficients(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
    projection_grid: HBProjectionGrid | None = None,
    config: HBProjectionConfig | None = None,
    fundamental_frequency_hz: float | None = None,
) -> jax.Array:
    """
    Project nonlinear flux-linkage coefficients only.

    This is useful for testing cubic scaling independently of the derivative.
    """
    i_coeff = _as_complex_array("current_coeffs_A", current_coeffs_A)
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)

    cfg = config or HBProjectionConfig()
    grid = projection_grid
    if grid is None:
        grid = make_projection_grid(
            f,
            fundamental_frequency_hz=fundamental_frequency_hz,
            config=cfg,
        )

    coeffs_for_synthesis = i_coeff
    if cfg.enforce_conjugate_symmetry:
        coeffs_for_synthesis = enforce_conjugate_symmetry_by_frequencies(
            coeffs_for_synthesis,
            f,
        )

    current_t = grid.synthesize(
        coeffs_for_synthesis,
        force_real=cfg.force_real_time_signal,
    )

    flux_t = nonlinear_flux_linkage_time(
        current_t,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
    )

    flux_coeffs = grid.project(flux_t)
    if cfg.enforce_conjugate_symmetry:
        flux_coeffs = enforce_conjugate_symmetry_by_frequencies(flux_coeffs, f)
    return flux_coeffs


# ---------------------------------------------------------------------------
# Residual helpers for a nonlinear branch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NonlinearInductorProjectionResult:
    """
    Result object for nonlinear inductor projection.
    """

    voltage_coeffs_V: jax.Array
    flux_coeffs_Wb: jax.Array
    current_time_A: jax.Array
    flux_time_Wb: jax.Array
    projection_grid: HBProjectionGrid

    def summary(self) -> dict[str, Any]:
        return {
            "voltage_coeffs": coefficient_power_summary(self.voltage_coeffs_V),
            "flux_coeffs": coefficient_power_summary(self.flux_coeffs_Wb),
            "current_time_abs_max_A": float(jnp.max(jnp.abs(self.current_time_A))),
            "flux_time_abs_max_Wb": float(jnp.max(jnp.abs(self.flux_time_Wb))),
            "projection_grid": self.projection_grid.to_dict(),
        }


def project_nonlinear_inductor_full(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
    projection_grid: HBProjectionGrid | None = None,
    config: HBProjectionConfig | None = None,
    fundamental_frequency_hz: float | None = None,
) -> NonlinearInductorProjectionResult:
    """
    Full nonlinear inductor projection with intermediate arrays returned.

    Use this for diagnostics and tests. Residual functions should generally use
    nonlinear_inductor_voltage_coefficients() to avoid carrying intermediates.
    """
    i_coeff = _as_complex_array("current_coeffs_A", current_coeffs_A)
    f = jnp.asarray(frequencies_hz, dtype=jnp.float64)

    cfg = config or HBProjectionConfig()
    grid = projection_grid
    if grid is None:
        grid = make_projection_grid(
            f,
            fundamental_frequency_hz=fundamental_frequency_hz,
            config=cfg,
        )

    coeffs_for_synthesis = i_coeff
    if cfg.enforce_conjugate_symmetry:
        coeffs_for_synthesis = enforce_conjugate_symmetry_by_frequencies(
            coeffs_for_synthesis,
            f,
        )

    current_t = grid.synthesize(
        coeffs_for_synthesis,
        force_real=cfg.force_real_time_signal,
    )
    flux_t = nonlinear_flux_linkage_time(
        current_t,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
    )
    flux_coeffs = grid.project(flux_t)
    if cfg.enforce_conjugate_symmetry:
        flux_coeffs = enforce_conjugate_symmetry_by_frequencies(flux_coeffs, f)
    voltage_coeffs = voltage_coefficients_from_flux_coefficients(flux_coeffs, f)

    return NonlinearInductorProjectionResult(
        voltage_coeffs_V=voltage_coeffs,
        flux_coeffs_Wb=flux_coeffs,
        current_time_A=current_t,
        flux_time_Wb=flux_t,
        projection_grid=grid,
    )


def nonlinear_inductor_branch_residual(
    voltage_drop_coeffs_V: ArrayLike,
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    R_series_ohm: ArrayLike = 0.0,
    quartic_coefficient: ArrayLike = 0.0,
    projection_grid: HBProjectionGrid | None = None,
    config: HBProjectionConfig | None = None,
    fundamental_frequency_hz: float | None = None,
) -> jax.Array:
    """
    Frequency-domain residual for one nonlinear series branch.

    Residual convention:

        r_k = V_drop,k - R I_k - V_L,k

    where

        V_L,k = i omega_k Lambda_k

    Parameters may have trailing dimensions matching current coefficients.
    """
    v_drop = _as_complex_array("voltage_drop_coeffs_V", voltage_drop_coeffs_V)
    i_coeff = _as_complex_array("current_coeffs_A", current_coeffs_A)

    if v_drop.shape != i_coeff.shape:
        raise ValueError(
            f"voltage_drop_coeffs and current_coeffs must have same shape, "
            f"got {v_drop.shape} and {i_coeff.shape}"
        )

    v_l = nonlinear_inductor_voltage_coefficients(
        i_coeff,
        frequencies_hz,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
        projection_grid=projection_grid,
        config=config,
        fundamental_frequency_hz=fundamental_frequency_hz,
    )

    R = jnp.asarray(R_series_ohm)
    v_r = R * i_coeff

    return v_drop - v_r - v_l


# ---------------------------------------------------------------------------
# Linearization helpers
# ---------------------------------------------------------------------------

def jvp_nonlinear_inductor_voltage(
    current_coeffs_A: ArrayLike,
    tangent_current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
    projection_grid: HBProjectionGrid | None = None,
    config: HBProjectionConfig | None = None,
    fundamental_frequency_hz: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Return nonlinear inductor voltage and its JVP with respect to current coeffs.

    This is the local ingredient needed later for matrix-free Newton/Krylov and
    small-signal linearization around a pumped solution.
    """

    def fn(i_coeff: jax.Array) -> jax.Array:
        return nonlinear_inductor_voltage_coefficients(
            i_coeff,
            frequencies_hz,
            L0_H=L0_H,
            I_star_A=I_star_A,
            beta_nl=beta_nl,
            quartic_coefficient=quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    primal, tangent = jax.jvp(
        fn,
        (jnp.asarray(current_coeffs_A),),
        (jnp.asarray(tangent_current_coeffs_A),),
    )
    return primal, tangent


def linearize_nonlinear_inductor_voltage(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
    projection_grid: HBProjectionGrid | None = None,
    config: HBProjectionConfig | None = None,
    fundamental_frequency_hz: float | None = None,
) -> tuple[jax.Array, Any]:
    """
    Return voltage coefficients and a linearized function.

    The returned linear function maps delta_current_coeffs to
    delta_voltage_coeffs using JAX's exact linearization of the projection code.
    """

    def fn(i_coeff: jax.Array) -> jax.Array:
        return nonlinear_inductor_voltage_coefficients(
            i_coeff,
            frequencies_hz,
            L0_H=L0_H,
            I_star_A=I_star_A,
            beta_nl=beta_nl,
            quartic_coefficient=quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    return jax.linearize(fn, jnp.asarray(current_coeffs_A))


# ---------------------------------------------------------------------------
# Validation diagnostics
# ---------------------------------------------------------------------------

def cubic_third_harmonic_test_coefficients(
    *,
    fundamental_frequency_hz: float,
    current_rms_A: float,
    n_harmonics: int = 3,
) -> tuple[jax.Array, jax.Array]:
    """
    Build a simple real sinusoidal current coefficient set for cubic tests.

    Returns
    -------
    frequencies_hz:
        [-n f0, ..., -f0, +f0, ..., +n f0], no DC.
    coeffs:
        Coefficients with only ±f0 populated.
    """
    if n_harmonics < 3:
        raise ValueError("n_harmonics must be at least 3 for third-harmonic test")
    _check_positive("fundamental_frequency_hz", fundamental_frequency_hz)
    _check_positive("current_rms_A", current_rms_A)

    indices = jnp.concatenate(
        [
            -jnp.arange(n_harmonics, 0, -1),
            jnp.arange(1, n_harmonics + 1),
        ]
    )
    frequencies = indices.astype(jnp.float64) * fundamental_frequency_hz
    coeffs = jnp.zeros((2 * n_harmonics,), dtype=jnp.complex128)

    pos_fund_idx = int(jnp.where(indices == 1, size=1)[0][0])
    neg_fund_idx = int(jnp.where(indices == -1, size=1)[0][0])

    x_pos = current_rms_A / jnp.sqrt(2.0)
    coeffs = coeffs.at[pos_fund_idx].set(x_pos)
    coeffs = coeffs.at[neg_fund_idx].set(jnp.conj(x_pos))

    return frequencies, coeffs


def estimate_third_harmonic_cubic_slope(
    *,
    fundamental_frequency_hz: float = 1.0e9,
    amplitudes_rms_A: ArrayLike = jnp.asarray([1e-8, 2e-8, 4e-8, 8e-8]),
    L0_H: float = 1e-9,
    I_star_A: float = 1e-3,
    beta_nl: float = 1.0,
    n_harmonics: int = 5,
    config: HBProjectionConfig | None = None,
) -> dict[str, Any]:
    """
    Estimate cubic scaling slope of the third-harmonic flux coefficient.

    For lambda = L0 [I + beta I^3/I_star^2], the third-harmonic component
    should scale as amplitude^3 at small amplitudes.
    """
    amps = jnp.asarray(amplitudes_rms_A, dtype=jnp.float64)
    if amps.ndim != 1 or amps.shape[0] < 2:
        raise ValueError("amplitudes_rms_A must be a 1D array with at least two entries")

    cfg = config or HBProjectionConfig(
        n_time_samples=512,
        force_real_time_signal=True,
        enforce_conjugate_symmetry=True,
    )

    mags = []

    for amp in amps.tolist():
        f, coeffs = cubic_third_harmonic_test_coefficients(
            fundamental_frequency_hz=fundamental_frequency_hz,
            current_rms_A=float(amp),
            n_harmonics=n_harmonics,
        )
        grid = make_projection_grid(
            f,
            fundamental_frequency_hz=fundamental_frequency_hz,
            config=cfg,
        )
        flux = nonlinear_inductor_flux_coefficients(
            coeffs,
            f,
            L0_H=L0_H,
            I_star_A=I_star_A,
            beta_nl=beta_nl,
            projection_grid=grid,
            config=cfg,
        )
        pos_3_matches = jnp.where(jnp.isclose(f, 3.0 * fundamental_frequency_hz), size=1)[0]
        pos_3 = int(pos_3_matches[0])
        mags.append(float(jnp.abs(flux[pos_3])))

    log_amp = jnp.log(amps)
    log_mag = jnp.log(jnp.asarray(mags))
    A = jnp.stack([log_amp, jnp.ones_like(log_amp)], axis=1)
    slope, intercept = jnp.linalg.lstsq(A, log_mag, rcond=None)[0]

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "amplitudes_rms_A": [float(v) for v in amps.tolist()],
        "third_harmonic_flux_abs_Wb": mags,
        "expected_slope": 3.0,
        "passed": bool(abs(float(slope) - 3.0) < 1e-2),
    }


def compare_linear_limit(
    current_coeffs_A: ArrayLike,
    frequencies_hz: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    scale_factor: float = 1e-6,
    projection_grid: HBProjectionGrid | None = None,
    config: HBProjectionConfig | None = None,
    fundamental_frequency_hz: float | None = None,
) -> dict[str, Any]:
    """
    Check that nonlinear inductor voltage approaches linear inductor voltage
    when current amplitude is scaled down.
    """
    i_coeff = jnp.asarray(current_coeffs_A) * scale_factor

    v_nl = nonlinear_inductor_voltage_coefficients(
        i_coeff,
        frequencies_hz,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        projection_grid=projection_grid,
        config=config,
        fundamental_frequency_hz=fundamental_frequency_hz,
    )
    v_lin = linear_inductor_voltage_coefficients(
        i_coeff,
        frequencies_hz,
        L0_H=L0_H,
    )

    denom = jnp.maximum(jnp.linalg.norm(v_lin.ravel()), 1e-300)
    rel_err = jnp.linalg.norm((v_nl - v_lin).ravel()) / denom

    return {
        "relative_error": float(rel_err),
        "scale_factor": float(scale_factor),
        "passed": bool(float(rel_err) < 1e-6),
        "v_nl_summary": coefficient_power_summary(v_nl),
        "v_lin_summary": coefficient_power_summary(v_lin),
    }


__all__ = [
    "ProjectionMode",
    "HBProjectionConfig",
    "HBProjectionGrid",
    "make_projection_grid",
    "make_multi_fundamental_projection_grid",
    "make_projection_grid_from_plan",
    "coefficients_to_time",
    "time_to_coefficients",
    "project_nonlinear_time_function",
    "nonlinear_inductance_time",
    "nonlinear_flux_linkage_time",
    "nonlinear_incremental_inductance_time",
    "voltage_coefficients_from_flux_coefficients",
    "linear_inductor_voltage_coefficients",
    "nonlinear_inductor_voltage_coefficients",
    "nonlinear_inductor_flux_coefficients",
    "NonlinearInductorProjectionResult",
    "project_nonlinear_inductor_full",
    "nonlinear_inductor_branch_residual",
    "jvp_nonlinear_inductor_voltage",
    "linearize_nonlinear_inductor_voltage",
    "cubic_third_harmonic_test_coefficients",
    "estimate_third_harmonic_cubic_slope",
    "compare_linear_limit",
]


# ---------------------------------------------------------------------------
# Compatibility selected-harmonic FFT API
# ---------------------------------------------------------------------------

_production_coefficients_to_time = coefficients_to_time
_production_time_to_coefficients = time_to_coefficients
_production_make_time_grid = make_time_grid


def _selected_orders(orders: ArrayLike | None, harmonic_orders: ArrayLike | None) -> jax.Array:
    source = harmonic_orders if harmonic_orders is not None else orders
    if source is None:
        raise ValueError("orders/harmonic_orders is required")
    arr = jnp.asarray(source)
    if arr.ndim != 1:
        raise ValueError("orders must be one-dimensional")
    if arr.size == 0:
        raise ValueError("orders must be non-empty")
    if not bool(jnp.all(arr == jnp.round(arr))):
        raise ValueError("orders must be integer-valued")
    out = arr.astype(jnp.int64)
    if len(set(int(v) for v in out.tolist())) != int(out.size):
        raise ValueError("orders must be unique")
    return out


def _compat_time_grid(
    *,
    n_time: int | None = None,
    n: int | None = None,
    n_samples: int | None = None,
    omega0_rad_s: float | None = None,
    omega0: float | None = None,
    fundamental_angular_frequency_rad_s: float | None = None,
    period_s: float | None = None,
    fundamental_frequency_hz: float | None = None,
    **_: Any,
) -> jax.Array:
    count = n_time if n_time is not None else (n if n is not None else n_samples)
    if count is None or int(count) <= 0:
        raise ValueError("n_time must be positive")
    omega = omega0_rad_s if omega0_rad_s is not None else omega0
    if omega is None:
        omega = fundamental_angular_frequency_rad_s
    if period_s is None:
        if omega is not None:
            period_s = 2.0 * jnp.pi / float(omega)
        elif fundamental_frequency_hz is not None:
            period_s = 1.0 / float(fundamental_frequency_hz)
        else:
            period_s = 1.0
    return jnp.arange(int(count), dtype=jnp.float64) * (float(period_s) / int(count))


def make_time_grid(
    fundamental_frequency_hz: float | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Compatibility dispatcher for time-grid creation.

    Production calls with `(fundamental_frequency_hz, n_samples=...)` are
    routed to the canonical TimeGrid constructor; selected-harmonic compatibility
    calls continue to return a plain time array.
    """
    if fundamental_frequency_hz is not None:
        if args:
            raise TypeError("make_time_grid accepts at most one positional argument")
        n_samples = kwargs.pop("n_samples", None)
        if n_samples is None:
            n_samples = kwargs.pop("n_time", None)
        if n_samples is None:
            n_samples = kwargs.pop("n", None)
        if n_samples is None:
            raise ValueError("n_samples/n_time is required when fundamental_frequency_hz is set")
        endpoint = bool(kwargs.pop("endpoint", False))
        return _production_make_time_grid(
            float(fundamental_frequency_hz),
            n_samples=int(n_samples),
            endpoint=endpoint,
        )
    return _compat_time_grid(**kwargs)


def harmonic_basis_matrix(
    *,
    orders: ArrayLike | None = None,
    harmonic_orders: ArrayLike | None = None,
    n_time: int | None = None,
    n: int | None = None,
    t: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    omega0_rad_s: float | None = None,
    omega0: float | None = None,
    **kwargs: Any,
) -> jax.Array:
    order_arr = _selected_orders(orders, harmonic_orders)
    omega = omega0_rad_s if omega0_rad_s is not None else omega0
    if omega is None:
        omega = 2.0 * jnp.pi
    if time_s is not None or t is not None:
        t_arr = jnp.asarray(time_s if time_s is not None else t, dtype=jnp.float64)
    else:
        t_arr = _compat_time_grid(n_time=n_time, n=n, omega0_rad_s=float(omega), **kwargs)
    return jnp.exp(1j * jnp.outer(t_arr * float(omega), order_arr))


def coefficients_to_time(
    coeffs: ArrayLike,
    frequencies_hz: ArrayLike | None = None,
    t_s: ArrayLike | None = None,
    *,
    orders: ArrayLike | None = None,
    harmonic_orders: ArrayLike | None = None,
    n_time: int | None = None,
    n: int | None = None,
    t: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    omega0_rad_s: float | None = None,
    omega0: float | None = None,
    force_real: bool = False,
    **kwargs: Any,
) -> jax.Array:
    """Synthesize x(t) = sum_k X_k exp(+i k omega0 t)."""
    if frequencies_hz is not None and t_s is not None and orders is None and harmonic_orders is None:
        return _production_coefficients_to_time(
            coeffs,
            frequencies_hz,
            t_s,
            force_real=force_real,
        )
    c = jnp.asarray(coeffs)
    order_arr = _selected_orders(orders, harmonic_orders)
    if c.shape[0] != order_arr.shape[0]:
        raise ValueError("coefficients first dimension must match orders length")
    count = n_time if n_time is not None else n
    if count is None:
        count = jnp.asarray(time_s if time_s is not None else t).shape[0]
    if int(count) <= 0:
        raise ValueError("n_time must be positive")
    if int(count) <= 2 * int(jnp.max(jnp.abs(order_arr))):
        raise ValueError("insufficient time samples for selected harmonics")
    B = harmonic_basis_matrix(
        orders=order_arr,
        n_time=int(count),
        t=time_s if time_s is not None else t,
        omega0_rad_s=omega0_rad_s,
        omega0=omega0,
        **kwargs,
    )
    out = jnp.einsum("tk,k...->t...", B, c)
    return jnp.real(out) if force_real else out


def time_to_coefficients(
    samples: ArrayLike,
    frequencies_hz: ArrayLike | None = None,
    t_s: ArrayLike | None = None,
    *,
    orders: ArrayLike | None = None,
    harmonic_orders: ArrayLike | None = None,
    t: ArrayLike | None = None,
    time_s: ArrayLike | None = None,
    omega0_rad_s: float | None = None,
    omega0: float | None = None,
    **kwargs: Any,
) -> jax.Array:
    """Project X_k = mean_n x(t_n) exp(-i k omega0 t_n)."""
    if frequencies_hz is not None and t_s is not None and orders is None and harmonic_orders is None:
        return _production_time_to_coefficients(samples, frequencies_hz, t_s)
    x = jnp.asarray(samples)
    order_arr = _selected_orders(orders, harmonic_orders)
    B = harmonic_basis_matrix(
        orders=order_arr,
        n_time=x.shape[0],
        t=time_s if time_s is not None else t,
        omega0_rad_s=omega0_rad_s,
        omega0=omega0,
        **kwargs,
    )
    return jnp.einsum("tk,t...->k...", jnp.conj(B), x) / x.shape[0]


def convolve_coefficients(
    a: ArrayLike | None = None,
    b: ArrayLike | None = None,
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    coeffs_a: ArrayLike | None = None,
    coeffs_b: ArrayLike | None = None,
    orders: ArrayLike | None = None,
    harmonic_orders: ArrayLike | None = None,
    output_orders: ArrayLike | None = None,
    **_: Any,
) -> jax.Array:
    """Selected-order convolution y_k = sum_p a_p b_{k-p}."""
    lhs = coeffs_a if coeffs_a is not None else (a if a is not None else x)
    rhs = coeffs_b if coeffs_b is not None else (b if b is not None else y)
    if lhs is None or rhs is None:
        raise ValueError("two coefficient vectors are required")
    in_orders = _selected_orders(orders, harmonic_orders)
    out_orders = _selected_orders(output_orders if output_orders is not None else in_orders, None)
    lhs_arr = jnp.asarray(lhs)
    rhs_arr = jnp.asarray(rhs)
    if lhs_arr.shape[0] != in_orders.shape[0] or rhs_arr.shape[0] != in_orders.shape[0]:
        raise ValueError("coefficient lengths must match orders")
    left = {int(k): lhs_arr[i] for i, k in enumerate(in_orders.tolist())}
    right = {int(k): rhs_arr[i] for i, k in enumerate(in_orders.tolist())}
    values = []
    for k in out_orders.tolist():
        total = 0.0 + 0.0j
        for p, a_p in left.items():
            total = total + a_p * right.get(int(k) - p, 0.0)
        values.append(total)
    return jnp.asarray(values, dtype=jnp.result_type(lhs_arr, rhs_arr, jnp.complex128))


for _name in [
    "make_time_grid",
    "harmonic_basis_matrix",
    "convolve_coefficients",
]:
    if _name not in __all__:
        __all__.append(_name)
