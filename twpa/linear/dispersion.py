"""
twpa.linear.dispersion
======================

Dispersion extraction and phase-matching diagnostics for TWPA layouts.

This module analyzes the pump-off linear response of a lumped line and extracts:

    beta(f)
    alpha(f)
    Bloch propagation constant
    group delay
    phase velocity
    group velocity
    stopband indicators
    DP4WM phase mismatch

The linear/pump-off layer must be validated before nonlinear HB is trusted.
For a KI-TWPA with periodic loading or capacitive stubs, dispersion engineering
is central: the pump is intentionally placed near an engineered dispersion
feature/bandgap to improve phase matching.

Two complementary extraction routes are provided:

1. From S21 phase:
       beta_eff ≈ -unwrap(angle(S21)) / length

   This is useful for the complete finite line.

2. From a unit-cell or supercell ABCD matrix:
       cosh(gamma d) = (A + D) / 2

   so
       gamma = arccosh((A + D)/2) / d

   This is useful for periodic/Bloch analysis.

All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout, extract_supercell
from twpa.core.units import angular_frequency
from twpa.linear.cascade import (
    CascadeConfig,
    CascadeStrategy,
    cascade_layout_abcd,
    run_linear_scan,
)
from twpa.linear.cells import CellModelConfig, layout_cell_abcd
from twpa.linear.rf_networks import (
    abcd_to_s,
    effective_beta_from_s21,
    group_delay_from_s21,
    s21,
    s_to_db,
    unwrap_phase,
)


ArrayLike = Any


# ---------------------------------------------------------------------------
# Enums / config
# ---------------------------------------------------------------------------

class DispersionExtractionMethod(str, Enum):
    """Supported dispersion extraction methods."""

    S21_PHASE = "s21_phase"
    BLOCH_ABCD = "bloch_abcd"
    BOTH = "both"


class StopbandMetric(str, Enum):
    """Stopband detection metric."""

    S21_DB = "s21_db"
    BLOCH_ALPHA = "bloch_alpha"
    BOTH = "both"


@dataclass(frozen=True)
class DispersionConfig:
    """
    Dispersion-extraction configuration.

    Parameters
    ----------
    method:
        Extraction method.
    cells_per_supercell:
        Number of cells per supercell for Bloch extraction.
    unwrap_s21_phase:
        Whether to unwrap S21 phase.
    beta_sign:
        Convention sign for S21-based beta extraction.
    stopband_s21_threshold_db:
        Frequencies below this S21 value are marked as stopband-like.
    stopband_alpha_threshold_np_per_m:
        Frequencies with Bloch alpha above this value are marked as stopband-like.
    smooth_group_delay:
        If true, apply a small moving average to group delay.
    group_delay_smoothing_points:
        Moving-average length for group delay.
    """

    method: DispersionExtractionMethod = DispersionExtractionMethod.BOTH
    cells_per_supercell: int = 1
    unwrap_s21_phase: bool = True
    beta_sign: Literal["positive", "negative"] = "positive"
    stopband_s21_threshold_db: float = -10.0
    stopband_alpha_threshold_np_per_m: float = 1.0
    smooth_group_delay: bool = False
    group_delay_smoothing_points: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", DispersionExtractionMethod(self.method))
        if int(self.cells_per_supercell) <= 0:
            raise ValueError("cells_per_supercell must be positive")
        object.__setattr__(self, "cells_per_supercell", int(self.cells_per_supercell))
        if self.beta_sign not in {"positive", "negative"}:
            raise ValueError("beta_sign must be 'positive' or 'negative'")
        if int(self.group_delay_smoothing_points) <= 0:
            raise ValueError("group_delay_smoothing_points must be positive")
        object.__setattr__(
            self,
            "group_delay_smoothing_points",
            int(self.group_delay_smoothing_points),
        )

    def with_updates(self, **kwargs: Any) -> "DispersionConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "cells_per_supercell": self.cells_per_supercell,
            "unwrap_s21_phase": self.unwrap_s21_phase,
            "beta_sign": self.beta_sign,
            "stopband_s21_threshold_db": self.stopband_s21_threshold_db,
            "stopband_alpha_threshold_np_per_m": self.stopband_alpha_threshold_np_per_m,
            "smooth_group_delay": self.smooth_group_delay,
            "group_delay_smoothing_points": self.group_delay_smoothing_points,
        }


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _as_frequency_array(frequency_hz: ArrayLike) -> jax.Array:
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if f.ndim == 0:
        f = f.reshape((1,))
    if f.ndim != 1:
        raise ValueError(f"frequency_hz must be scalar or 1D, got shape {f.shape}")
    if bool(jnp.any(f < 0.0)):
        raise ValueError("frequency_hz must be non-negative")
    return f


def _check_abcd_batch(abcd: ArrayLike) -> jax.Array:
    m = jnp.asarray(abcd)
    if not jnp.issubdtype(m.dtype, jnp.complexfloating):
        m = m.astype(jnp.complex128)
    if m.ndim != 3 or m.shape[-2:] != (2, 2):
        raise ValueError(f"ABCD must have shape (F, 2, 2), got {m.shape}")
    return m


def _moving_average_1d(x: jax.Array, points: int) -> jax.Array:
    if points <= 1:
        return x
    if points % 2 == 0:
        points += 1
    kernel = jnp.ones((points,), dtype=x.dtype) / points
    radius = points // 2
    padded = jnp.pad(x, (radius, radius), mode="edge")
    return jnp.convolve(padded, kernel, mode="valid")


def _gradient_safe(y: jax.Array, x: jax.Array) -> jax.Array:
    """
    Wrapper around jnp.gradient for 1D arrays.
    """
    if y.shape[0] < 2:
        return jnp.zeros_like(y)
    return jnp.gradient(y, x)


# ---------------------------------------------------------------------------
# Bloch extraction
# ---------------------------------------------------------------------------

def bloch_gamma_from_abcd(
    abcd: ArrayLike,
    *,
    period_m: float,
    branch: Literal["principal", "positive_beta"] = "positive_beta",
) -> jax.Array:
    """
    Extract Bloch propagation constant gamma from a unit/supercell ABCD matrix.

        cosh(gamma d) = (A + D) / 2

    Parameters
    ----------
    abcd:
        ABCD matrices, shape (F, 2, 2).
    period_m:
        Physical period length d.
    branch:
        Branch choice. "positive_beta" flips the sign if needed so beta >= 0
        for positive frequencies.

    Returns
    -------
    gamma:
        Complex propagation constant alpha + j beta, shape (F,).
    """
    if period_m <= 0.0:
        raise ValueError("period_m must be positive")

    m = _check_abcd_batch(abcd)
    A = m[..., 0, 0]
    D = m[..., 1, 1]
    trace_half = 0.5 * (A + D)

    gamma = jnp.arccosh(trace_half) / period_m

    if branch == "principal":
        return gamma
    if branch == "positive_beta":
        beta = jnp.imag(gamma)
        gamma = jnp.where(beta < 0.0, -gamma, gamma)
        return gamma

    raise ValueError(f"Unsupported branch {branch!r}")


def bloch_beta_alpha_from_abcd(
    abcd: ArrayLike,
    *,
    period_m: float,
    branch: Literal["principal", "positive_beta"] = "positive_beta",
) -> tuple[jax.Array, jax.Array]:
    """
    Return alpha and beta from Bloch gamma.

        gamma = alpha + j beta
    """
    gamma = bloch_gamma_from_abcd(abcd, period_m=period_m, branch=branch)
    alpha = jnp.real(gamma)
    beta = jnp.imag(gamma)
    return alpha, beta


def bloch_impedance_from_abcd(
    abcd: ArrayLike,
    *,
    kind: Literal["B_over_C", "voltage_current_ratio"] = "B_over_C",
) -> jax.Array:
    """
    Estimate Bloch impedance from ABCD matrix.

    For reciprocal periodic cells, a common expression is

        Z_B = ± sqrt(B / C)

    Branch choice is nontrivial near stopbands. This diagnostic returns the
    principal square root and should not be treated as a calibrated port
    impedance without validation.
    """
    m = _check_abcd_batch(abcd)
    B = m[..., 0, 1]
    C = m[..., 1, 0]

    if kind == "B_over_C":
        return jnp.sqrt(B / C)

    if kind == "voltage_current_ratio":
        A = m[..., 0, 0]
        D = m[..., 1, 1]
        gamma_d = jnp.arccosh(0.5 * (A + D))
        # One eigenvalue is exp(gamma d). For eigenvector [V, I],
        # (A-lambda)V + B I = 0 -> Z = V/I = -B/(A-lambda)
        lam = jnp.exp(gamma_d)
        return -B / (A - lam)

    raise ValueError(f"Unsupported Bloch impedance kind {kind!r}")


# ---------------------------------------------------------------------------
# S21-based extraction
# ---------------------------------------------------------------------------

def s21_phase_beta(
    frequency_hz: ArrayLike,
    s21_values: ArrayLike,
    *,
    length_m: float,
    sign: Literal["positive", "negative"] = "positive",
) -> jax.Array:
    """
    Extract effective beta from S21 phase.

        beta ≈ -unwrap(angle(S21)) / length

    This is a finite-line effective value, not a Bloch eigenvalue.
    """
    f = _as_frequency_array(frequency_hz)
    y = jnp.asarray(s21_values)
    if y.shape[0] != f.shape[0]:
        raise ValueError("s21_values first dimension must match frequency length")
    return effective_beta_from_s21(f, y, length_m=length_m, sign=sign)


def phase_velocity_from_beta(
    frequency_hz: ArrayLike,
    beta_rad_per_m: ArrayLike,
    *,
    beta_floor: float = 1e-300,
) -> jax.Array:
    """
    Phase velocity:

        v_p = omega / beta
    """
    f = _as_frequency_array(frequency_hz)
    beta = jnp.asarray(beta_rad_per_m, dtype=jnp.float64)
    omega = angular_frequency(f)
    denom = jnp.where(jnp.abs(beta) > beta_floor, beta, jnp.nan)
    return omega / denom


def group_velocity_from_beta(
    frequency_hz: ArrayLike,
    beta_rad_per_m: ArrayLike,
    *,
    beta_smoothing_points: int = 1,
) -> jax.Array:
    """
    Group velocity:

        v_g = d omega / d beta

    computed numerically from beta(f).
    """
    f = _as_frequency_array(frequency_hz)
    beta = jnp.asarray(beta_rad_per_m, dtype=jnp.float64)
    if beta.shape[0] != f.shape[0]:
        raise ValueError("beta first dimension must match frequency length")

    if beta_smoothing_points > 1:
        beta = _moving_average_1d(beta, beta_smoothing_points)

    omega = angular_frequency(f)
    d_beta_d_omega = _gradient_safe(beta, omega)
    return 1.0 / d_beta_d_omega


def beta_derivatives(
    frequency_hz: ArrayLike,
    beta_rad_per_m: ArrayLike,
) -> dict[str, jax.Array]:
    """
    Return numerical beta derivatives with respect to angular frequency.

    Outputs:
        beta1 = d beta / d omega
        beta2 = d² beta / d omega²
        beta3 = d³ beta / d omega³
    """
    f = _as_frequency_array(frequency_hz)
    beta = jnp.asarray(beta_rad_per_m, dtype=jnp.float64)
    if beta.shape[0] != f.shape[0]:
        raise ValueError("beta first dimension must match frequency length")

    omega = angular_frequency(f)
    beta1 = _gradient_safe(beta, omega)
    beta2 = _gradient_safe(beta1, omega)
    beta3 = _gradient_safe(beta2, omega)

    return {
        "beta1_s_per_m": beta1,
        "beta2_s2_per_m": beta2,
        "beta3_s3_per_m": beta3,
    }


# ---------------------------------------------------------------------------
# Main result object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispersionResult:
    """
    Dispersion extraction result.

    Attributes
    ----------
    frequency_hz:
        Frequency grid.
    s:
        S-parameters, shape (F, 2, 2), if available.
    s21_db:
        S21 magnitude in dB, if available.
    beta_s21_rad_per_m:
        Effective beta from S21 phase, if available.
    group_delay_s:
        Group delay from S21, if available.
    alpha_bloch_np_per_m:
        Bloch attenuation from supercell ABCD, if available.
    beta_bloch_rad_per_m:
        Bloch beta from supercell ABCD, if available.
    bloch_impedance_ohm:
        Diagnostic Bloch impedance, if available.
    metadata:
        Report metadata.
    """

    frequency_hz: jax.Array
    s: jax.Array | None = None
    s21_db: jax.Array | None = None
    beta_s21_rad_per_m: jax.Array | None = None
    group_delay_s: jax.Array | None = None
    alpha_bloch_np_per_m: jax.Array | None = None
    beta_bloch_rad_per_m: jax.Array | None = None
    bloch_impedance_ohm: jax.Array | None = None
    phase_velocity_m_per_s: jax.Array | None = None
    group_velocity_m_per_s: jax.Array | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def beta_preferred_rad_per_m(self) -> jax.Array:
        """
        Preferred beta: Bloch if available, otherwise S21 phase beta.
        """
        if self.beta_bloch_rad_per_m is not None:
            return self.beta_bloch_rad_per_m
        if self.beta_s21_rad_per_m is not None:
            return self.beta_s21_rad_per_m
        raise ValueError("No beta data available")

    @property
    def alpha_preferred_np_per_m(self) -> jax.Array:
        """
        Preferred alpha: Bloch if available, otherwise zeros.
        """
        if self.alpha_bloch_np_per_m is not None:
            return self.alpha_bloch_np_per_m
        return jnp.zeros_like(self.frequency_hz)

    def stopband_mask(
        self,
        *,
        metric: StopbandMetric | str = StopbandMetric.BOTH,
        s21_threshold_db: float | None = None,
        alpha_threshold_np_per_m: float | None = None,
    ) -> jax.Array:
        """
        Return boolean stopband mask.
        """
        metric = StopbandMetric(metric)
        masks = []

        if metric in {StopbandMetric.S21_DB, StopbandMetric.BOTH}:
            if self.s21_db is not None:
                threshold = -10.0 if s21_threshold_db is None else s21_threshold_db
                masks.append(self.s21_db < threshold)

        if metric in {StopbandMetric.BLOCH_ALPHA, StopbandMetric.BOTH}:
            if self.alpha_bloch_np_per_m is not None:
                threshold = 1.0 if alpha_threshold_np_per_m is None else alpha_threshold_np_per_m
                masks.append(self.alpha_bloch_np_per_m > threshold)

        if not masks:
            return jnp.zeros_like(self.frequency_hz, dtype=bool)

        out = masks[0]
        for mask in masks[1:]:
            out = out | mask
        return out

    def beta_at_frequency(
        self,
        frequency_hz: ArrayLike,
        *,
        source: Literal["preferred", "s21", "bloch"] = "preferred",
    ) -> jax.Array:
        """
        Interpolate beta at one or more frequencies.
        """
        f_query = jnp.asarray(frequency_hz, dtype=jnp.float64)
        if source == "preferred":
            beta = self.beta_preferred_rad_per_m
        elif source == "s21":
            if self.beta_s21_rad_per_m is None:
                raise ValueError("beta_s21_rad_per_m is unavailable")
            beta = self.beta_s21_rad_per_m
        elif source == "bloch":
            if self.beta_bloch_rad_per_m is None:
                raise ValueError("beta_bloch_rad_per_m is unavailable")
            beta = self.beta_bloch_rad_per_m
        else:
            raise ValueError(f"Unsupported beta source {source!r}")

        return jnp.interp(f_query, self.frequency_hz, beta)

    def alpha_at_frequency(
        self,
        frequency_hz: ArrayLike,
    ) -> jax.Array:
        """
        Interpolate preferred alpha at one or more frequencies.
        """
        return jnp.interp(
            jnp.asarray(frequency_hz, dtype=jnp.float64),
            self.frequency_hz,
            self.alpha_preferred_np_per_m,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "frequency_shape": tuple(int(v) for v in self.frequency_hz.shape),
            "frequency_min_hz": float(jnp.min(self.frequency_hz)),
            "frequency_max_hz": float(jnp.max(self.frequency_hz)),
            "metadata": dict(self.metadata or {}),
        }

        if self.s21_db is not None:
            out.update(
                {
                    "s21_db_min": float(jnp.min(self.s21_db)),
                    "s21_db_max": float(jnp.max(self.s21_db)),
                }
            )
        if self.beta_s21_rad_per_m is not None:
            out.update(
                {
                    "beta_s21_min_rad_per_m": float(jnp.min(self.beta_s21_rad_per_m)),
                    "beta_s21_max_rad_per_m": float(jnp.max(self.beta_s21_rad_per_m)),
                }
            )
        if self.beta_bloch_rad_per_m is not None:
            out.update(
                {
                    "beta_bloch_min_rad_per_m": float(jnp.min(self.beta_bloch_rad_per_m)),
                    "beta_bloch_max_rad_per_m": float(jnp.max(self.beta_bloch_rad_per_m)),
                }
            )
        if self.alpha_bloch_np_per_m is not None:
            out.update(
                {
                    "alpha_bloch_min_np_per_m": float(jnp.min(self.alpha_bloch_np_per_m)),
                    "alpha_bloch_max_np_per_m": float(jnp.max(self.alpha_bloch_np_per_m)),
                }
            )
        if self.phase_velocity_m_per_s is not None:
            out.update(
                {
                    "phase_velocity_min_m_per_s": float(jnp.nanmin(self.phase_velocity_m_per_s)),
                    "phase_velocity_max_m_per_s": float(jnp.nanmax(self.phase_velocity_m_per_s)),
                }
            )
        if self.group_velocity_m_per_s is not None:
            out.update(
                {
                    "group_velocity_min_m_per_s": float(jnp.nanmin(self.group_velocity_m_per_s)),
                    "group_velocity_max_m_per_s": float(jnp.nanmax(self.group_velocity_m_per_s)),
                }
            )

        return out


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_dispersion_from_s21(
    frequency_hz: ArrayLike,
    s_parameters: ArrayLike,
    *,
    length_m: float,
    config: DispersionConfig | None = None,
) -> DispersionResult:
    """
    Extract finite-line effective dispersion from S21.
    """
    cfg = config or DispersionConfig(method=DispersionExtractionMethod.S21_PHASE)
    f = _as_frequency_array(frequency_hz)
    s = jnp.asarray(s_parameters)
    if s.shape[-2:] != (2, 2):
        raise ValueError(f"s_parameters must have trailing shape (2,2), got {s.shape}")
    if s.shape[0] != f.shape[0]:
        raise ValueError("s_parameters first dimension must match frequency length")

    y21 = s21(s)
    s21_db = s_to_db(y21)
    beta = s21_phase_beta(
        f,
        y21,
        length_m=length_m,
        sign=cfg.beta_sign,
    )
    gd = group_delay_from_s21(f, y21)

    if cfg.smooth_group_delay:
        gd = _moving_average_1d(gd, cfg.group_delay_smoothing_points)

    vp = phase_velocity_from_beta(f, beta)
    vg = group_velocity_from_beta(f, beta)

    return DispersionResult(
        frequency_hz=f,
        s=s,
        s21_db=s21_db,
        beta_s21_rad_per_m=beta,
        group_delay_s=gd,
        phase_velocity_m_per_s=vp,
        group_velocity_m_per_s=vg,
        metadata={
            "source": "extract_dispersion_from_s21",
            "length_m": length_m,
            "config": cfg.to_dict(),
        },
    )


def extract_bloch_dispersion_from_abcd(
    frequency_hz: ArrayLike,
    cell_or_supercell_abcd: ArrayLike,
    *,
    period_m: float,
    config: DispersionConfig | None = None,
) -> DispersionResult:
    """
    Extract Bloch dispersion from a cell or supercell ABCD matrix.
    """
    cfg = config or DispersionConfig(method=DispersionExtractionMethod.BLOCH_ABCD)
    f = _as_frequency_array(frequency_hz)
    abcd = _check_abcd_batch(cell_or_supercell_abcd)

    if abcd.shape[0] != f.shape[0]:
        raise ValueError("ABCD first dimension must match frequency length")

    alpha, beta = bloch_beta_alpha_from_abcd(
        abcd,
        period_m=period_m,
        branch="positive_beta",
    )
    z_bloch = bloch_impedance_from_abcd(abcd)
    vp = phase_velocity_from_beta(f, beta)
    vg = group_velocity_from_beta(f, beta)

    return DispersionResult(
        frequency_hz=f,
        alpha_bloch_np_per_m=alpha,
        beta_bloch_rad_per_m=beta,
        bloch_impedance_ohm=z_bloch,
        phase_velocity_m_per_s=vp,
        group_velocity_m_per_s=vg,
        metadata={
            "source": "extract_bloch_dispersion_from_abcd",
            "period_m": period_m,
            "config": cfg.to_dict(),
        },
    )


def extract_layout_dispersion(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    dispersion_config: DispersionConfig | None = None,
) -> DispersionResult:
    """
    Extract dispersion for a complete layout.

    Depending on dispersion_config.method, this computes S21-based finite-line
    beta, Bloch beta/alpha from a cell or supercell, or both.
    """
    f = _as_frequency_array(frequency_hz)
    cell_cfg = cell_model or CellModelConfig()
    disp_cfg = dispersion_config or DispersionConfig()
    cas_cfg = cascade_config or CascadeConfig()

    s = None
    s21_db_values = None
    beta_s21 = None
    group_delay = None

    alpha_bloch = None
    beta_bloch = None
    z_bloch = None

    if disp_cfg.method in {
        DispersionExtractionMethod.S21_PHASE,
        DispersionExtractionMethod.BOTH,
    }:
        scan = run_linear_scan(
            f,
            layout,
            cell_model=cell_cfg,
            cascade_config=cas_cfg,
        )
        s = scan.s
        s21_db_values = scan.s21_db
        beta_s21 = scan.beta_eff_rad_per_m
        group_delay = scan.group_delay_s
        if disp_cfg.smooth_group_delay:
            group_delay = _moving_average_1d(
                group_delay,
                disp_cfg.group_delay_smoothing_points,
            )

    if disp_cfg.method in {
        DispersionExtractionMethod.BLOCH_ABCD,
        DispersionExtractionMethod.BOTH,
    }:
        if disp_cfg.cells_per_supercell <= 1:
            supercell = extract_supercell(
                layout,
                start_cell=0,
                cells_per_supercell=1,
                name=f"{layout.name}_bloch_cell",
            )
        else:
            supercell = extract_supercell(
                layout,
                start_cell=0,
                cells_per_supercell=disp_cfg.cells_per_supercell,
                name=f"{layout.name}_bloch_supercell",
            )

        cells = layout_cell_abcd(f, supercell, config=cell_cfg)

        # Cascade the cells in the supercell only.
        from twpa.linear.cascade import cascade_cell_abcd_direct

        supercell_abcd = cascade_cell_abcd_direct(cells)
        period_m = supercell.total_length_m
        bloch = extract_bloch_dispersion_from_abcd(
            f,
            supercell_abcd,
            period_m=period_m,
            config=disp_cfg,
        )
        alpha_bloch = bloch.alpha_bloch_np_per_m
        beta_bloch = bloch.beta_bloch_rad_per_m
        z_bloch = bloch.bloch_impedance_ohm

    preferred_beta = beta_bloch if beta_bloch is not None else beta_s21
    if preferred_beta is not None:
        vp = phase_velocity_from_beta(f, preferred_beta)
        vg = group_velocity_from_beta(f, preferred_beta)
    else:
        vp = None
        vg = None

    return DispersionResult(
        frequency_hz=f,
        s=s,
        s21_db=s21_db_values,
        beta_s21_rad_per_m=beta_s21,
        group_delay_s=group_delay,
        alpha_bloch_np_per_m=alpha_bloch,
        beta_bloch_rad_per_m=beta_bloch,
        bloch_impedance_ohm=z_bloch,
        phase_velocity_m_per_s=vp,
        group_velocity_m_per_s=vg,
        metadata={
            "source": "extract_layout_dispersion",
            "layout_name": layout.name,
            "layout_summary": layout.summary(),
            "cell_model": cell_cfg.to_dict(),
            "cascade_config": cas_cfg.to_dict(),
            "dispersion_config": disp_cfg.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Stopband detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StopbandInterval:
    """
    One detected stopband-like interval.
    """

    start_hz: float
    stop_hz: float
    center_hz: float
    width_hz: float
    min_s21_db: float | None = None
    max_alpha_np_per_m: float | None = None
    n_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "center_hz": self.center_hz,
            "width_hz": self.width_hz,
            "start_GHz": self.start_hz / 1e9,
            "stop_GHz": self.stop_hz / 1e9,
            "center_GHz": self.center_hz / 1e9,
            "width_GHz": self.width_hz / 1e9,
            "min_s21_db": self.min_s21_db,
            "max_alpha_np_per_m": self.max_alpha_np_per_m,
            "n_points": self.n_points,
        }


def contiguous_true_intervals(mask: ArrayLike) -> list[tuple[int, int]]:
    """
    Convert a boolean mask to inclusive index intervals [(start, stop), ...].
    """
    m = jnp.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError("mask must be 1D")

    intervals: list[tuple[int, int]] = []
    in_run = False
    start = 0

    for i, value in enumerate(m.tolist()):
        if bool(value) and not in_run:
            start = i
            in_run = True
        elif not bool(value) and in_run:
            intervals.append((start, i - 1))
            in_run = False

    if in_run:
        intervals.append((start, int(m.shape[0]) - 1))

    return intervals


def detect_stopbands(
    dispersion: DispersionResult,
    *,
    metric: StopbandMetric | str = StopbandMetric.BOTH,
    s21_threshold_db: float = -10.0,
    alpha_threshold_np_per_m: float = 1.0,
    min_points: int = 2,
) -> list[StopbandInterval]:
    """
    Detect stopband-like intervals.
    """
    f = dispersion.frequency_hz
    mask = dispersion.stopband_mask(
        metric=metric,
        s21_threshold_db=s21_threshold_db,
        alpha_threshold_np_per_m=alpha_threshold_np_per_m,
    )

    intervals = []
    for start, stop in contiguous_true_intervals(mask):
        n_points = stop - start + 1
        if n_points < min_points:
            continue

        f_start = float(f[start])
        f_stop = float(f[stop])
        center = 0.5 * (f_start + f_stop)
        width = f_stop - f_start

        min_s21 = None
        if dispersion.s21_db is not None:
            min_s21 = float(jnp.min(dispersion.s21_db[start : stop + 1]))

        max_alpha = None
        if dispersion.alpha_bloch_np_per_m is not None:
            max_alpha = float(jnp.max(dispersion.alpha_bloch_np_per_m[start : stop + 1]))

        intervals.append(
            StopbandInterval(
                start_hz=f_start,
                stop_hz=f_stop,
                center_hz=center,
                width_hz=width,
                min_s21_db=min_s21,
                max_alpha_np_per_m=max_alpha,
                n_points=n_points,
            )
        )

    return intervals


# ---------------------------------------------------------------------------
# DP4WM phase matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DP4WMPhaseMatchingResult:
    """
    Degenerate-pump 4WM phase-matching result.

    For each signal frequency fs:

        fi = 2 fp - fs
        delta_beta_linear = beta_s + beta_i - 2 beta_p

    Optional nonlinear correction:

        delta_beta_total = delta_beta_linear + delta_beta_nl

    Sign conventions vary in the literature. This object stores both the linear
    and user-supplied nonlinear correction explicitly.
    """

    signal_frequency_hz: jax.Array
    idler_frequency_hz: jax.Array
    pump_frequency_hz: float
    beta_signal_rad_per_m: jax.Array
    beta_idler_rad_per_m: jax.Array
    beta_pump_rad_per_m: float
    delta_beta_linear_rad_per_m: jax.Array
    delta_beta_nonlinear_rad_per_m: jax.Array
    delta_beta_total_rad_per_m: jax.Array
    valid_idler_mask: jax.Array
    metadata: Mapping[str, Any] | None = None

    @property
    def best_phase_matched_index(self) -> int:
        masked = jnp.where(
            self.valid_idler_mask,
            jnp.abs(self.delta_beta_total_rad_per_m),
            jnp.inf,
        )
        return int(jnp.argmin(masked))

    @property
    def best_signal_frequency_hz(self) -> float:
        return float(self.signal_frequency_hz[self.best_phase_matched_index])

    @property
    def best_delta_beta_rad_per_m(self) -> float:
        return float(self.delta_beta_total_rad_per_m[self.best_phase_matched_index])

    def to_dict(self) -> dict[str, Any]:
        return {
            "pump_frequency_hz": self.pump_frequency_hz,
            "pump_frequency_GHz": self.pump_frequency_hz / 1e9,
            "n_signal_points": int(self.signal_frequency_hz.shape[0]),
            "valid_idler_count": int(jnp.sum(self.valid_idler_mask)),
            "best_phase_matched_index": self.best_phase_matched_index,
            "best_signal_frequency_hz": self.best_signal_frequency_hz,
            "best_signal_frequency_GHz": self.best_signal_frequency_hz / 1e9,
            "best_delta_beta_rad_per_m": self.best_delta_beta_rad_per_m,
            "delta_beta_linear_min_rad_per_m": float(jnp.nanmin(self.delta_beta_linear_rad_per_m)),
            "delta_beta_linear_max_rad_per_m": float(jnp.nanmax(self.delta_beta_linear_rad_per_m)),
            "delta_beta_total_min_rad_per_m": float(jnp.nanmin(self.delta_beta_total_rad_per_m)),
            "delta_beta_total_max_rad_per_m": float(jnp.nanmax(self.delta_beta_total_rad_per_m)),
            "metadata": dict(self.metadata or {}),
        }


def nonlinear_delta_beta_dp4wm_simple(
    *,
    beta_pump_rad_per_m: ArrayLike,
    pump_current_peak_A: ArrayLike,
    I_star_A: ArrayLike,
    coefficient: float = 1.0 / 4.0,
    sign: Literal["positive", "negative"] = "negative",
) -> jax.Array:
    """
    Simple reduced-theory nonlinear phase correction scale for DP4WM.

    A common KI-TWPA phase-matching expression has a correction proportional to

        beta_p * I_p^2 / I_star^2

    with convention-dependent numerical factors and signs. This function is a
    diagnostic comparator only; HB remains the source of truth.

    Default:
        delta_beta_nl = - beta_p * I_p^2 / (4 I_star^2)
    """
    beta_p = jnp.asarray(beta_pump_rad_per_m)
    Ip = jnp.asarray(pump_current_peak_A)
    Istar = jnp.asarray(I_star_A)
    value = coefficient * beta_p * (Ip / Istar) ** 2
    if sign == "positive":
        return value
    if sign == "negative":
        return -value
    raise ValueError("sign must be 'positive' or 'negative'")


def compute_dp4wm_phase_matching(
    dispersion: DispersionResult,
    *,
    pump_frequency_hz: float,
    signal_frequency_hz: ArrayLike,
    nonlinear_delta_beta_rad_per_m: ArrayLike | float = 0.0,
    beta_source: Literal["preferred", "s21", "bloch"] = "preferred",
) -> DP4WMPhaseMatchingResult:
    """
    Compute degenerate-pump 4WM phase mismatch from extracted beta(f).

        fi = 2 fp - fs
        Δβ = βs + βi - 2βp

    Frequencies with fi <= 0 are marked invalid.
    """
    if pump_frequency_hz <= 0.0:
        raise ValueError("pump_frequency_hz must be positive")

    fs = jnp.asarray(signal_frequency_hz, dtype=jnp.float64)
    if fs.ndim == 0:
        fs = fs.reshape((1,))
    if fs.ndim != 1:
        raise ValueError("signal_frequency_hz must be scalar or 1D")
    if bool(jnp.any(fs <= 0.0)):
        raise ValueError("signal frequencies must be positive")

    fi = 2.0 * pump_frequency_hz - fs
    valid = fi > 0.0

    beta_s = dispersion.beta_at_frequency(fs, source=beta_source)
    beta_i = dispersion.beta_at_frequency(jnp.maximum(fi, 0.0), source=beta_source)
    beta_p = float(dispersion.beta_at_frequency(pump_frequency_hz, source=beta_source))

    delta_linear = beta_s + beta_i - 2.0 * beta_p
    delta_nl = jnp.asarray(nonlinear_delta_beta_rad_per_m, dtype=jnp.float64)
    if delta_nl.ndim == 0:
        delta_nl = jnp.full_like(delta_linear, delta_nl)
    elif delta_nl.shape != delta_linear.shape:
        raise ValueError(
            f"nonlinear_delta_beta shape must be scalar or {delta_linear.shape}, "
            f"got {delta_nl.shape}"
        )

    delta_total = delta_linear + delta_nl

    # Mark invalid idlers as nan in delta arrays but preserve raw beta values.
    delta_linear = jnp.where(valid, delta_linear, jnp.nan)
    delta_nl = jnp.where(valid, delta_nl, jnp.nan)
    delta_total = jnp.where(valid, delta_total, jnp.nan)

    return DP4WMPhaseMatchingResult(
        signal_frequency_hz=fs,
        idler_frequency_hz=fi,
        pump_frequency_hz=float(pump_frequency_hz),
        beta_signal_rad_per_m=beta_s,
        beta_idler_rad_per_m=beta_i,
        beta_pump_rad_per_m=beta_p,
        delta_beta_linear_rad_per_m=delta_linear,
        delta_beta_nonlinear_rad_per_m=delta_nl,
        delta_beta_total_rad_per_m=delta_total,
        valid_idler_mask=valid,
        metadata={
            "source": "compute_dp4wm_phase_matching",
            "beta_source": beta_source,
            "dispersion_metadata": dict(dispersion.metadata or {}),
        },
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispersionValidationReport:
    """
    High-level dispersion validation report.
    """

    layout_name: str
    passed: bool
    frequency_min_hz: float
    frequency_max_hz: float
    beta_available: bool
    stopband_count: int
    stopbands: list[Mapping[str, Any]]
    messages: list[str]
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "passed": self.passed,
            "frequency_min_hz": self.frequency_min_hz,
            "frequency_max_hz": self.frequency_max_hz,
            "beta_available": self.beta_available,
            "stopband_count": self.stopband_count,
            "stopbands": [dict(x) for x in self.stopbands],
            "messages": list(self.messages),
            "metadata": dict(self.metadata or {}),
        }


def validate_dispersion_result(
    dispersion: DispersionResult,
    *,
    layout_name: str = "layout",
    expected_stopband: bool | None = None,
    stopband_metric: StopbandMetric | str = StopbandMetric.BOTH,
    s21_threshold_db: float = -10.0,
    alpha_threshold_np_per_m: float = 1.0,
) -> DispersionValidationReport:
    """
    Validate that dispersion extraction produced usable data.
    """
    messages: list[str] = []
    passed = True

    beta_available = (
        dispersion.beta_s21_rad_per_m is not None
        or dispersion.beta_bloch_rad_per_m is not None
    )
    if not beta_available:
        passed = False
        messages.append("FAIL: no beta extraction available.")

    if beta_available:
        beta = dispersion.beta_preferred_rad_per_m
        if bool(jnp.any(jnp.isnan(beta))):
            passed = False
            messages.append("FAIL: beta contains NaN.")
        if bool(jnp.any(jnp.isinf(beta))):
            passed = False
            messages.append("FAIL: beta contains Inf.")

    stopbands = detect_stopbands(
        dispersion,
        metric=stopband_metric,
        s21_threshold_db=s21_threshold_db,
        alpha_threshold_np_per_m=alpha_threshold_np_per_m,
    )

    if expected_stopband is True and len(stopbands) == 0:
        passed = False
        messages.append("FAIL: expected stopband, but none detected.")
    elif expected_stopband is False and len(stopbands) > 0:
        passed = False
        messages.append("FAIL: stopband detected, but none expected.")

    if passed:
        messages.append("PASS: dispersion validation checks passed.")

    return DispersionValidationReport(
        layout_name=layout_name,
        passed=bool(passed),
        frequency_min_hz=float(jnp.min(dispersion.frequency_hz)),
        frequency_max_hz=float(jnp.max(dispersion.frequency_hz)),
        beta_available=bool(beta_available),
        stopband_count=len(stopbands),
        stopbands=[s.to_dict() for s in stopbands],
        messages=messages,
        metadata={
            "dispersion": dispersion.to_dict(),
            "stopband_metric": StopbandMetric(stopband_metric).value,
            "s21_threshold_db": s21_threshold_db,
            "alpha_threshold_np_per_m": alpha_threshold_np_per_m,
        },
    )


__all__ = [
    "DispersionExtractionMethod",
    "StopbandMetric",
    "DispersionConfig",
    "bloch_gamma_from_abcd",
    "bloch_beta_alpha_from_abcd",
    "bloch_impedance_from_abcd",
    "s21_phase_beta",
    "phase_velocity_from_beta",
    "group_velocity_from_beta",
    "beta_derivatives",
    "DispersionResult",
    "extract_dispersion_from_s21",
    "extract_bloch_dispersion_from_abcd",
    "extract_layout_dispersion",
    "StopbandInterval",
    "contiguous_true_intervals",
    "detect_stopbands",
    "DP4WMPhaseMatchingResult",
    "nonlinear_delta_beta_dp4wm_simple",
    "compute_dp4wm_phase_matching",
    "DispersionValidationReport",
    "validate_dispersion_result",
]