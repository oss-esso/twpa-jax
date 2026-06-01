"""
twpa.nonlinear.kinetic_inductance
=================================

Kinetic-inductance and generic weak-current chi(3) nonlinear-inductor models.

This module provides the local nonlinear constitutive laws used later by
harmonic-balance residuals.

Default KI-TWPA model
---------------------
For a kinetic-inductance line, the weak-current model is

    L(I) = L0 [1 + beta_nl (I / I_star)^2 + q (I / I_star)^4]

with

    beta_nl = 1

for kinetic inductance. The flux linkage is

    lambda(I) = L(I) I

and the branch voltage is

    V(t) = d lambda(I(t)) / dt.

The quartic correction q is disabled by default. It is included as a hook for
compression/saturation studies, but the main validated path should first use
the cubic model only.

Design rules
------------
1. All quantities are SI.
2. Functions are JAX-compatible.
3. No hidden global state.
4. Local constitutive laws are separate from circuit topology.
5. Frequency-domain projection is delegated to twpa.core.hb_fft.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from twpa.core.hb_fft import (
    HBProjectionConfig,
    HBProjectionGrid,
    compare_linear_limit,
    estimate_third_harmonic_cubic_slope,
    jvp_nonlinear_inductor_voltage,
    linear_inductor_voltage_coefficients,
    linearize_nonlinear_inductor_voltage,
    nonlinear_flux_linkage_time,
    nonlinear_incremental_inductance_time,
    nonlinear_inductance_time,
    nonlinear_inductor_branch_residual,
    nonlinear_inductor_flux_coefficients,
    nonlinear_inductor_voltage_coefficients,
    project_nonlinear_inductor_full,
)
from twpa.core.params import NonlinearMedium, NonlinearParams


ArrayLike = Any


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _check_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _check_nonnegative(name: str, value: float) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _as_float_array(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        arr = arr.astype(jnp.float64)
    if bool(jnp.any(jnp.isnan(arr))):
        raise ValueError(f"{name} contains NaN")
    return arr


def _as_complex_array(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value)
    if not jnp.issubdtype(arr.dtype, jnp.complexfloating):
        arr = arr.astype(jnp.complex128)
    return arr


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
# Local nonlinear model object
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KineticInductanceModel:
    """
    Local current-dependent nonlinear inductance model.

    Parameters
    ----------
    L0_H:
        Linear zero-current inductance. Can be scalar or array.
    I_star_A:
        Characteristic nonlinearity current. Can be scalar or array.
    beta_nl:
        Cubic nonlinear coefficient. KI default is 1.
        Weak-current JJ/SQUID approximation uses 0.5.
    quartic_coefficient:
        Optional q in L(I) = L0 [1 + beta (I/I*)^2 + q (I/I*)^4].
    medium:
        Semantic nonlinear-medium label.
    name:
        Human-readable model name.
    metadata:
        Static metadata.

    Notes
    -----
    This model is registered as a JAX PyTree. Numerical fields are dynamic
    leaves. Static fields are auxiliary data.
    """

    L0_H: Any
    I_star_A: Any
    beta_nl: Any = 1.0
    quartic_coefficient: Any = 0.0
    medium: NonlinearMedium = NonlinearMedium.KINETIC_INDUCTANCE
    name: str = "kinetic_inductance"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        L0 = _as_float_array("L0_H", self.L0_H)
        I_star = _as_float_array("I_star_A", self.I_star_A)
        beta = _as_float_array("beta_nl", self.beta_nl)
        q = _as_float_array("quartic_coefficient", self.quartic_coefficient)

        if bool(jnp.any(L0 <= 0.0)):
            raise ValueError("L0_H must be positive everywhere")
        if bool(jnp.any(I_star <= 0.0)):
            raise ValueError("I_star_A must be positive everywhere")
        if bool(jnp.any(beta < 0.0)):
            raise ValueError("beta_nl must be non-negative everywhere")

        object.__setattr__(self, "L0_H", L0)
        object.__setattr__(self, "I_star_A", I_star)
        object.__setattr__(self, "beta_nl", beta)
        object.__setattr__(self, "quartic_coefficient", q)
        object.__setattr__(self, "medium", NonlinearMedium(self.medium))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    # ------------------------------------------------------------------
    # JAX PyTree implementation
    # ------------------------------------------------------------------

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], dict[str, Any]]:
        children = (
            self.L0_H,
            self.I_star_A,
            self.beta_nl,
            self.quartic_coefficient,
        )
        aux = {
            "medium": self.medium,
            "name": self.name,
            "metadata": dict(self.metadata or {}),
        }
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: dict[str, Any],
        children: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> "KineticInductanceModel":
        return cls(
            L0_H=children[0],
            I_star_A=children[1],
            beta_nl=children[2],
            quartic_coefficient=children[3],
            medium=aux["medium"],
            name=aux["name"],
            metadata=aux["metadata"],
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        *,
        L0_H: ArrayLike,
        params: NonlinearParams,
        name: str | None = None,
    ) -> "KineticInductanceModel":
        """
        Build a model from NonlinearParams and a linear inductance.
        """
        return cls(
            L0_H=L0_H,
            I_star_A=params.I_star_A,
            beta_nl=params.beta_nl,
            quartic_coefficient=(
                params.quartic_coefficient
                if params.include_quartic_correction
                else 0.0
            ),
            medium=params.medium,
            name=name or params.medium.value,
            metadata={
                "source": "KineticInductanceModel.from_params",
                "dc_bias_A": params.dc_bias_A,
                "include_quartic_correction": params.include_quartic_correction,
            },
        )

    @classmethod
    def kinetic(
        cls,
        *,
        L0_H: ArrayLike,
        I_star_A: ArrayLike,
        name: str = "kinetic_inductance",
    ) -> "KineticInductanceModel":
        """
        Convenience constructor for KI.

            beta_nl = 1
        """
        return cls(
            L0_H=L0_H,
            I_star_A=I_star_A,
            beta_nl=1.0,
            medium=NonlinearMedium.KINETIC_INDUCTANCE,
            name=name,
        )

    @classmethod
    def josephson_weak_current(
        cls,
        *,
        L0_H: ArrayLike,
        I_c_A: ArrayLike,
        name: str = "josephson_weak_current",
    ) -> "KineticInductanceModel":
        """
        Weak-current Josephson-inductance approximation.

            L(I) ≈ L0 [1 + 0.5 (I/Ic)^2]
        """
        return cls(
            L0_H=L0_H,
            I_star_A=I_c_A,
            beta_nl=0.5,
            medium=NonlinearMedium.JOSEPHSON,
            name=name,
        )

    @classmethod
    def squid_weak_current(
        cls,
        *,
        L0_H: ArrayLike,
        I_c_eff_A: ArrayLike,
        name: str = "squid_weak_current",
    ) -> "KineticInductanceModel":
        """
        Weak-current symmetric SQUID approximation.

            L(I) ≈ L0 [1 + 0.5 (I/Ic_eff)^2]
        """
        return cls(
            L0_H=L0_H,
            I_star_A=I_c_eff_A,
            beta_nl=0.5,
            medium=NonlinearMedium.SQUID,
            name=name,
        )

    # ------------------------------------------------------------------
    # Immutable modifiers
    # ------------------------------------------------------------------

    def with_updates(self, **kwargs: Any) -> "KineticInductanceModel":
        return replace(self, **kwargs)

    def scaled(
        self,
        *,
        L0_scale: float = 1.0,
        I_star_scale: float = 1.0,
        beta_scale: float = 1.0,
    ) -> "KineticInductanceModel":
        """
        Return a copy with global multiplicative parameter scaling.
        """
        _check_positive("L0_scale", L0_scale)
        _check_positive("I_star_scale", I_star_scale)
        _check_nonnegative("beta_scale", beta_scale)
        return replace(
            self,
            L0_H=self.L0_H * L0_scale,
            I_star_A=self.I_star_A * I_star_scale,
            beta_nl=self.beta_nl * beta_scale,
            metadata={
                **dict(self.metadata or {}),
                "L0_scale": L0_scale,
                "I_star_scale": I_star_scale,
                "beta_scale": beta_scale,
            },
        )

    # ------------------------------------------------------------------
    # Time-domain constitutive laws
    # ------------------------------------------------------------------

    def inductance(self, current_A: ArrayLike) -> jax.Array:
        """
        L(I) in henry.
        """
        return nonlinear_inductance_time(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
        )

    def flux_linkage(self, current_A: ArrayLike) -> jax.Array:
        """
        lambda(I) = L(I) I in weber.
        """
        return nonlinear_flux_linkage_time(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
        )

    def incremental_inductance(self, current_A: ArrayLike) -> jax.Array:
        """
        d lambda / dI in henry.
        """
        return nonlinear_incremental_inductance_time(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
        )

    def nonlinear_fraction(self, current_A: ArrayLike) -> jax.Array:
        """
        Dimensionless nonlinear correction to L0:

            beta (I/I*)^2 + q (I/I*)^4
        """
        i = _as_float_array("current_A", current_A)
        ratio = i / self.I_star_A
        return self.beta_nl * ratio**2 + self.quartic_coefficient * ratio**4

    # ------------------------------------------------------------------
    # Frequency-domain HB laws
    # ------------------------------------------------------------------

    def linear_voltage_coefficients(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
    ) -> jax.Array:
        """
        Linear voltage coefficients:

            V_k = i omega_k L0 I_k
        """
        return linear_inductor_voltage_coefficients(
            current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
        )

    def flux_coefficients(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> jax.Array:
        """
        Nonlinear flux-linkage coefficients Lambda_k.
        """
        return nonlinear_inductor_flux_coefficients(
            current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    def voltage_coefficients(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> jax.Array:
        """
        Nonlinear inductor voltage coefficients.
        """
        return nonlinear_inductor_voltage_coefficients(
            current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    def branch_residual(
        self,
        voltage_drop_coeffs_V: ArrayLike,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        R_series_ohm: ArrayLike = 0.0,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> jax.Array:
        """
        Branch residual:

            r_k = V_drop,k - R I_k - V_L,k
        """
        return nonlinear_inductor_branch_residual(
            voltage_drop_coeffs_V,
            current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            R_series_ohm=R_series_ohm,
            quartic_coefficient=self.quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    def voltage_jvp(
        self,
        current_coeffs_A: ArrayLike,
        tangent_current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Return V(I) and JVP dV[I]·dI.
        """
        return jvp_nonlinear_inductor_voltage(
            current_coeffs_A,
            tangent_current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    def linearize_voltage(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> tuple[jax.Array, Any]:
        """
        Return V(I0) and a linear function dI -> dV.
        """
        return linearize_nonlinear_inductor_voltage(
            current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    def full_projection(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> Any:
        """
        Diagnostic full projection with intermediate time-domain arrays.
        """
        return project_nonlinear_inductor_full(
            current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            quartic_coefficient=self.quartic_coefficient,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    # ------------------------------------------------------------------
    # Diagnostics / validation
    # ------------------------------------------------------------------

    def small_signal_nonlinear_fraction(
        self,
        current_peak_A: ArrayLike,
    ) -> jax.Array:
        """
        Estimate nonlinear fraction at a peak current amplitude.

            beta (I_peak/I*)^2 + q (I_peak/I*)^4
        """
        return self.nonlinear_fraction(current_peak_A)

    def validate_zero_current(self) -> dict[str, Any]:
        """
        Check basic zero-current identities:

            L(0) = L0
            lambda(0) = 0
            d lambda/dI |0 = L0
        """
        zero = jnp.zeros_like(self.L0_H)
        L_zero = self.inductance(zero)
        lam_zero = self.flux_linkage(zero)
        L_inc_zero = self.incremental_inductance(zero)

        L_err = jnp.max(jnp.abs(L_zero - self.L0_H))
        lam_err = jnp.max(jnp.abs(lam_zero))
        inc_err = jnp.max(jnp.abs(L_inc_zero - self.L0_H))

        passed = bool(float(L_err) < 1e-24 and float(lam_err) < 1e-24 and float(inc_err) < 1e-24)
        return {
            "passed": passed,
            "L_zero_max_abs_error_H": float(L_err),
            "lambda_zero_max_abs_Wb": float(lam_err),
            "incremental_L_zero_max_abs_error_H": float(inc_err),
        }

    def validate_derivatives_at_zero(self) -> dict[str, Any]:
        """
        Validate derivatives of L(I) at zero for the cubic model.

        For

            L(I) = L0 [1 + beta (I/I*)^2 + q (I/I*)^4]

        expected:
            dL/dI |0 = 0
            d2L/dI2 |0 = 2 L0 beta / I*^2
        """
        def scalar_L(i_scalar: jax.Array, L0: jax.Array, Istar: jax.Array, beta: jax.Array, q: jax.Array) -> jax.Array:
            ratio = i_scalar / Istar
            return L0 * (1.0 + beta * ratio**2 + q * ratio**4)

        # Use representative scalar values: maximums are enough for model sanity.
        L0 = jnp.ravel(self.L0_H)[0]
        Istar = jnp.ravel(self.I_star_A)[0]
        beta = jnp.ravel(self.beta_nl)[0]
        q = jnp.ravel(self.quartic_coefficient)[0]
        zero = jnp.asarray(0.0, dtype=jnp.float64)

        d1 = jax.grad(lambda x: scalar_L(x, L0, Istar, beta, q))(zero)
        d2 = jax.grad(jax.grad(lambda x: scalar_L(x, L0, Istar, beta, q)))(zero)
        expected_d2 = 2.0 * L0 * beta / Istar**2

        return {
            "dL_dI_at_zero": float(d1),
            "expected_dL_dI_at_zero": 0.0,
            "d2L_dI2_at_zero": float(d2),
            "expected_d2L_dI2_at_zero": float(expected_d2),
            "d2_relative_error": float(
                jnp.abs(d2 - expected_d2) / jnp.maximum(jnp.abs(expected_d2), 1e-300)
            ),
            "passed": bool(
                abs(float(d1)) < 1e-18
                and float(jnp.abs(d2 - expected_d2) / jnp.maximum(jnp.abs(expected_d2), 1e-300)) < 1e-12
            ),
        }

    def validate_linear_limit(
        self,
        current_coeffs_A: ArrayLike,
        frequencies_hz: ArrayLike,
        *,
        scale_factor: float = 1e-6,
        projection_grid: HBProjectionGrid | None = None,
        config: HBProjectionConfig | None = None,
        fundamental_frequency_hz: float | None = None,
    ) -> dict[str, Any]:
        """
        Validate that nonlinear HB voltage tends to linear voltage for tiny current.
        """
        return compare_linear_limit(
            current_coeffs_A,
            frequencies_hz,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta_nl=self.beta_nl,
            scale_factor=scale_factor,
            projection_grid=projection_grid,
            config=config,
            fundamental_frequency_hz=fundamental_frequency_hz,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-friendly model description.
        """
        return {
            "name": self.name,
            "medium": self.medium.value,
            "L0_H": _jsonify(self.L0_H),
            "I_star_A": _jsonify(self.I_star_A),
            "beta_nl": _jsonify(self.beta_nl),
            "quartic_coefficient": _jsonify(self.quartic_coefficient),
            "metadata": _jsonify(dict(self.metadata or {})),
        }


# ---------------------------------------------------------------------------
# Standalone convenience functions
# ---------------------------------------------------------------------------

def kinetic_inductance(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta: ArrayLike | None = None,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
) -> jax.Array:
    """
    Compute L(I) for the generic weak-current chi(3) model.

    Complex inputs are interpreted as phasors and use ``abs(I / I_star)``.
    The time-domain HB path supplies real instantaneous currents.
    """
    if beta is not None:
        beta_nl = beta
    current = jnp.asarray(current_A)
    if jnp.issubdtype(current.dtype, jnp.complexfloating):
        L0 = jnp.asarray(L0_H)
        Istar = jnp.asarray(I_star_A)
        beta_value = jnp.asarray(beta_nl)
        quartic = jnp.asarray(quartic_coefficient)
        if float(jnp.min(L0)) <= 0.0:
            raise ValueError("L0_H must be positive")
        if float(jnp.min(Istar)) <= 0.0:
            raise ValueError("I_star_A must be positive")
        if float(jnp.min(beta_value)) < 0.0:
            raise ValueError("beta_nl must be non-negative")
        ratio = jnp.abs(current / Istar)
        return L0 * (1.0 + beta_value * ratio**2 + quartic * ratio**4)
    return nonlinear_inductance_time(
        current,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
    )


@dataclass(frozen=True)
class KineticInductanceParams:
    """Small public parameter object for the quadratic KI model."""

    L0_H: float
    I_star_A: float
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.L0_H <= 0.0:
            raise ValueError("L0_H must be positive")
        if self.I_star_A <= 0.0:
            raise ValueError("I_star_A must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")

    def inductance(self, current_A: ArrayLike) -> jax.Array:
        return kinetic_inductance(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta=self.beta,
        )

    def dL_dI(self, current_A: ArrayLike) -> jax.Array:
        return dL_dI(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta=self.beta,
        )

    def inverse_inductance(self, current_A: ArrayLike) -> jax.Array:
        return inverse_kinetic_inductance(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta=self.beta,
        )

    def flux_from_current(self, current_A: ArrayLike) -> jax.Array:
        return flux_from_current(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta=self.beta,
        )

    def energy(self, current_A: ArrayLike) -> jax.Array:
        return magnetic_energy(
            current_A,
            L0_H=self.L0_H,
            I_star_A=self.I_star_A,
            beta=self.beta,
        )

    def to_dict(self) -> dict[str, float]:
        return {"L0_H": self.L0_H, "I_star_A": self.I_star_A, "beta": self.beta}


def _beta_value(beta: ArrayLike | None, beta_nl: ArrayLike) -> ArrayLike:
    return beta_nl if beta is None else beta


def dL_dI(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta: ArrayLike | None = None,
    beta_nl: ArrayLike = 1.0,
) -> jax.Array:
    """Derivative of L(I) = L0 * (1 + beta * (I/I*)**2)."""
    i = _as_float_array("current_A", current_A)
    L0 = jnp.asarray(L0_H)
    Istar = jnp.asarray(I_star_A)
    b = jnp.asarray(_beta_value(beta, beta_nl))
    if float(jnp.min(L0)) <= 0.0:
        raise ValueError("L0_H must be positive")
    if float(jnp.min(Istar)) <= 0.0:
        raise ValueError("I_star_A must be positive")
    if float(jnp.min(b)) < 0.0:
        raise ValueError("beta must be non-negative")
    return 2.0 * L0 * b * i / (Istar**2)


def inverse_kinetic_inductance(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta: ArrayLike | None = None,
    beta_nl: ArrayLike = 1.0,
) -> jax.Array:
    """Reciprocal of the quadratic current-dependent inductance."""
    return 1.0 / kinetic_inductance(
        current_A,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta=_beta_value(beta, beta_nl),
    )


def flux_from_current(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta: ArrayLike | None = None,
    beta_nl: ArrayLike = 1.0,
) -> jax.Array:
    """Integral flux relation Phi(I) = integral L(I) dI."""
    i = _as_float_array("current_A", current_A)
    L0 = jnp.asarray(L0_H)
    Istar = jnp.asarray(I_star_A)
    b = jnp.asarray(_beta_value(beta, beta_nl))
    if float(jnp.min(L0)) <= 0.0:
        raise ValueError("L0_H must be positive")
    if float(jnp.min(Istar)) <= 0.0:
        raise ValueError("I_star_A must be positive")
    if float(jnp.min(b)) < 0.0:
        raise ValueError("beta must be non-negative")
    return L0 * (i + b * i**3 / (3.0 * Istar**2))


def magnetic_energy(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta: ArrayLike | None = None,
    beta_nl: ArrayLike = 1.0,
) -> jax.Array:
    """Energy primitive U(I) = integral Phi(I) dI for the integral-flux model."""
    i = _as_float_array("current_A", current_A)
    L0 = jnp.asarray(L0_H)
    Istar = jnp.asarray(I_star_A)
    b = jnp.asarray(_beta_value(beta, beta_nl))
    if float(jnp.min(L0)) <= 0.0:
        raise ValueError("L0_H must be positive")
    if float(jnp.min(Istar)) <= 0.0:
        raise ValueError("I_star_A must be positive")
    if float(jnp.min(b)) < 0.0:
        raise ValueError("beta must be non-negative")
    return 0.5 * L0 * i**2 + L0 * b * i**4 / (12.0 * Istar**2)


def kinetic_flux_linkage(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
) -> jax.Array:
    """
    Compute lambda(I) = L(I) I.
    """
    return nonlinear_flux_linkage_time(
        current_A,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
    )


def kinetic_incremental_inductance(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
) -> jax.Array:
    """
    Compute d lambda / dI.
    """
    return nonlinear_incremental_inductance_time(
        current_A,
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        quartic_coefficient=quartic_coefficient,
    )


def kinetic_energy_density_like(
    current_A: ArrayLike,
    *,
    L0_H: ArrayLike,
    I_star_A: ArrayLike,
    beta_nl: ArrayLike = 1.0,
    quartic_coefficient: ArrayLike = 0.0,
) -> jax.Array:
    """
    Magnetic/kinetic energy-like primitive U(I) = integral_0^I lambda(i) di.

    For

        lambda(I) = L0 [I + beta I^3/I*^2 + q I^5/I*^4],

    the primitive is

        U(I) = L0 [I^2/2 + beta I^4/(4 I*^2) + q I^6/(6 I*^4)].

    This is useful for diagnostics and later energy-flow checks.
    """
    i = _as_float_array("current_A", current_A)
    L0 = jnp.asarray(L0_H)
    Istar = jnp.asarray(I_star_A)
    beta = jnp.asarray(beta_nl)
    q = jnp.asarray(quartic_coefficient)

    return L0 * (
        0.5 * i**2
        + beta * i**4 / (4.0 * Istar**2)
        + q * i**6 / (6.0 * Istar**4)
    )


def estimate_I_star_from_pump_ratio(
    *,
    pump_current_peak_A: float,
    pump_ratio: float,
) -> float:
    """
    Estimate I_star from a known peak current ratio.

        pump_ratio = I_p / I_star

    so

        I_star = I_p / pump_ratio
    """
    _check_positive("pump_current_peak_A", pump_current_peak_A)
    _check_positive("pump_ratio", pump_ratio)
    return float(pump_current_peak_A / pump_ratio)


def estimate_pump_ratio(
    *,
    pump_current_peak_A: ArrayLike,
    I_star_A: ArrayLike,
) -> jax.Array:
    """
    Return I_p / I_star.
    """
    return jnp.asarray(pump_current_peak_A) / jnp.asarray(I_star_A)


def nonlinear_phase_shift_scale(
    *,
    beta_p_rad_per_m: ArrayLike,
    pump_current_peak_A: ArrayLike,
    I_star_A: ArrayLike,
    coefficient: float = 1.0 / 8.0,
) -> jax.Array:
    """
    Rough nonlinear phase-shift scale used for sanity checks.

    A common reduced-theory scale has the form

        Δphi_per_length ~ beta_p * |I_p|^2 / (8 I_star^2)

    up to convention-dependent factors. This function is a diagnostic only;
    the HB solver remains the source of truth.
    """
    return (
        coefficient
        * jnp.asarray(beta_p_rad_per_m)
        * (jnp.asarray(pump_current_peak_A) / jnp.asarray(I_star_A)) ** 2
    )


# ---------------------------------------------------------------------------
# Validation suite for this module
# ---------------------------------------------------------------------------

def run_kinetic_inductance_self_checks(
    *,
    L0_H: float = 1e-9,
    I_star_A: float = 1e-3,
    beta_nl: float = 1.0,
) -> dict[str, Any]:
    """
    Run local sanity checks for the nonlinear inductor model.

    This is not a replacement for pytest. It is useful inside notebooks and
    validation scripts.
    """
    model = KineticInductanceModel.kinetic(
        L0_H=L0_H,
        I_star_A=I_star_A,
    ).with_updates(beta_nl=jnp.asarray(beta_nl))

    zero = model.validate_zero_current()
    deriv = model.validate_derivatives_at_zero()

    cubic = estimate_third_harmonic_cubic_slope(
        fundamental_frequency_hz=1.0e9,
        amplitudes_rms_A=jnp.asarray([1e-8, 2e-8, 4e-8, 8e-8]),
        L0_H=L0_H,
        I_star_A=I_star_A,
        beta_nl=beta_nl,
        n_harmonics=5,
        config=HBProjectionConfig(
            n_time_samples=1024,
            force_real_time_signal=True,
            enforce_conjugate_symmetry=True,
        ),
    )

    passed = bool(zero["passed"] and deriv["passed"] and cubic["passed"])
    return {
        "passed": passed,
        "zero_current": zero,
        "derivatives_at_zero": deriv,
        "third_harmonic_cubic_scaling": cubic,
    }


__all__ = [
    "KineticInductanceModel",
    "KineticInductanceParams",
    "kinetic_inductance",
    "dL_dI",
    "inverse_kinetic_inductance",
    "flux_from_current",
    "magnetic_energy",
    "kinetic_flux_linkage",
    "kinetic_incremental_inductance",
    "kinetic_energy_density_like",
    "estimate_I_star_from_pump_ratio",
    "estimate_pump_ratio",
    "nonlinear_phase_shift_scale",
    "run_kinetic_inductance_self_checks",
]
