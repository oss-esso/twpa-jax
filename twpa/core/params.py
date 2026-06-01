"""
twpa.core.params
================

Immutable parameter and configuration containers for the TWPA simulator.

Design goals
------------
1. Keep all internal quantities in SI units.
2. Keep objects immutable, so accidental mutation cannot corrupt JAX traces.
3. Separate physical device parameters from numerical solver configuration.
4. Support small validation circuits and industrial 100 mm / 20,000-cell lines.
5. Keep the default path KI-TWPA / degenerate 4WM, while leaving hooks for
   JJ, SQUID, and more general chi(3) media later.
6. Avoid putting large per-cell arrays in plain Python dataclasses unless they
   are already vectorized JAX arrays.

Important convention
--------------------
The simulator uses *cell-level* parameters for circuit calculations:

    L_series_H_per_cell
    C_shunt_F_per_cell
    R_series_ohm_per_cell
    G_shunt_S_per_cell

and *per-length* parameters for physical reporting / layout generation:

    L_per_m_H
    C_per_m_F
    R_per_m_ohm
    G_per_m_S

For an industrial 100 mm line with 20,000 cells:

    dx = 0.1 / 20_000 = 5e-6 m

so

    L_cell = L_per_m * dx
    C_cell = C_per_m * dx

The nonlinear kinetic inductance model used by the default KI track is

    Lk(I) = L0 * (1 + beta_nl * (I / I_star)^2)

with beta_nl = 1 for kinetic inductance. This same parameterization can
represent the weak-current limit of Josephson or symmetric SQUID media with
beta_nl = 0.5, but the first production implementation targets KI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from .units import (
    CONSTANTS,
    GHz,
    MHz,
    fF,
    pH,
    dbm_to_watts,
    watts_to_ipeak,
)


ArrayLike = Any

PHI0_Wb: float = CONSTANTS.phi0
PHI0_OVER_2PI_Wb: float = CONSTANTS.reduced_phi0


@dataclass(frozen=True)
class BasicLineParams:
    z0_ohm: Any
    phase_velocity_m_per_s: Any
    L_per_m_H: Any
    C_per_m_F: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "z0_ohm": self.z0_ohm,
            "phase_velocity_m_per_s": self.phase_velocity_m_per_s,
            "L_per_m_H": self.L_per_m_H,
            "C_per_m_F": self.C_per_m_F,
        }


@dataclass(frozen=True)
class CellParams:
    cell_length_m: float
    L_series_H: float
    C_shunt_F: float
    z0_ohm: float = 50.0
    phase_velocity_m_per_s: float = 1.2e8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TWPAParams:
    length_m: float
    n_cells: int
    cell_length_m: float
    z0_ohm: float
    phase_velocity_m_per_s: float
    total_L_H: float
    total_C_F: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatingPoint:
    pump_frequency_hz: float
    signal_frequency_hz: float
    idler_frequency_hz: float
    pump_current_a: float
    i_star_a: float
    pump_current_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def josephson_inductance(Ic_A: ArrayLike) -> jax.Array:
    ic = jnp.asarray(Ic_A, dtype=jnp.float64)
    if bool(jnp.any(ic <= 0.0)):
        raise ValueError("Ic_A must be positive")
    return PHI0_OVER_2PI_Wb / ic


def line_params_from_z0_vp(
    z0_ohm: ArrayLike = 50.0,
    phase_velocity_m_per_s: ArrayLike = 1.2e8,
) -> BasicLineParams:
    z0 = jnp.asarray(z0_ohm, dtype=jnp.float64)
    vp = jnp.asarray(phase_velocity_m_per_s, dtype=jnp.float64)
    if bool(jnp.any(z0 <= 0.0)):
        raise ValueError("z0_ohm must be positive")
    if bool(jnp.any(vp <= 0.0)):
        raise ValueError("phase_velocity_m_per_s must be positive")
    return BasicLineParams(
        z0_ohm=z0,
        phase_velocity_m_per_s=vp,
        L_per_m_H=z0 / vp,
        C_per_m_F=1.0 / (z0 * vp),
    )


def make_line_params(
    z0_ohm: ArrayLike = 50.0,
    phase_velocity_m_per_s: ArrayLike = 1.2e8,
) -> BasicLineParams:
    return line_params_from_z0_vp(z0_ohm, phase_velocity_m_per_s)


def cell_params_from_z0_vp(
    cell_length_m: float = 5e-6,
    z0_ohm: float = 50.0,
    phase_velocity_m_per_s: float = 1.2e8,
) -> CellParams:
    _check_positive("cell_length_m", cell_length_m)
    line = line_params_from_z0_vp(z0_ohm, phase_velocity_m_per_s)
    return CellParams(
        cell_length_m=float(cell_length_m),
        L_series_H=float(line.L_per_m_H) * float(cell_length_m),
        C_shunt_F=float(line.C_per_m_F) * float(cell_length_m),
        z0_ohm=float(z0_ohm),
        phase_velocity_m_per_s=float(phase_velocity_m_per_s),
    )


def make_cell_params(
    cell_length_m: float = 5e-6,
    z0_ohm: float = 50.0,
    phase_velocity_m_per_s: float = 1.2e8,
) -> CellParams:
    return cell_params_from_z0_vp(cell_length_m, z0_ohm, phase_velocity_m_per_s)


def make_twpa_params(
    length_m: float = 0.1,
    n_cells: int = 20_000,
    z0_ohm: float = 50.0,
    phase_velocity_m_per_s: float = 1.2e8,
) -> TWPAParams:
    _check_positive("length_m", length_m)
    if int(n_cells) <= 0:
        raise ValueError("n_cells must be positive")
    dx = float(length_m) / int(n_cells)
    cell = cell_params_from_z0_vp(dx, z0_ohm, phase_velocity_m_per_s)
    return TWPAParams(
        length_m=float(length_m),
        n_cells=int(n_cells),
        cell_length_m=dx,
        z0_ohm=float(z0_ohm),
        phase_velocity_m_per_s=float(phase_velocity_m_per_s),
        total_L_H=float(cell.L_series_H) * int(n_cells),
        total_C_F=float(cell.C_shunt_F) * int(n_cells),
    )


def make_chip_params(**kwargs: Any) -> TWPAParams:
    if "total_length_m" in kwargs and "length_m" not in kwargs:
        kwargs["length_m"] = kwargs.pop("total_length_m")
    else:
        kwargs.pop("total_length_m", None)
    if "Z0_ohm" in kwargs and "z0_ohm" not in kwargs:
        kwargs["z0_ohm"] = kwargs.pop("Z0_ohm")
    else:
        kwargs.pop("Z0_ohm", None)
    if "vp_m_per_s" in kwargs and "phase_velocity_m_per_s" not in kwargs:
        kwargs["phase_velocity_m_per_s"] = kwargs.pop("vp_m_per_s")
    else:
        kwargs.pop("vp_m_per_s", None)
    return make_twpa_params(**kwargs)


def make_operating_point(
    pump_frequency_hz: float = 10e9,
    signal_frequency_hz: float = 6e9,
    pump_current_a: float = 50e-6,
    i_star_a: float = 5e-3,
    I_star_A: float | None = None,
) -> OperatingPoint:
    if I_star_A is not None:
        i_star_a = I_star_A
    _check_positive("pump_frequency_hz", pump_frequency_hz)
    _check_positive("signal_frequency_hz", signal_frequency_hz)
    _check_positive("i_star_a", i_star_a)
    idler = 2.0 * float(pump_frequency_hz) - float(signal_frequency_hz)
    if idler <= 0.0:
        raise ValueError("idler frequency must be positive")
    return OperatingPoint(
        pump_frequency_hz=float(pump_frequency_hz),
        signal_frequency_hz=float(signal_frequency_hz),
        idler_frequency_hz=idler,
        pump_current_a=float(pump_current_a),
        i_star_a=float(i_star_a),
        pump_current_ratio=float(pump_current_a) / float(i_star_a),
    )


# ---------------------------------------------------------------------------
# Enums / literal helpers
# ---------------------------------------------------------------------------

class NonlinearMedium(str, Enum):
    """Supported nonlinear-medium families."""

    KINETIC_INDUCTANCE = "kinetic_inductance"
    JOSEPHSON = "josephson"
    SQUID = "squid"
    GENERIC_CHI3 = "generic_chi3"


class MixingRegime(str, Enum):
    """Supported wave-mixing regimes."""

    DP4WM = "degenerate_pump_4wm"
    NP4WM = "nondegenerate_pump_4wm"
    DC3WM = "dc_biased_3wm"


class SolverBackend(str, Enum):
    """Linear/Newton solver backend choices."""

    DENSE = "dense"
    NEWTON_KRYLOV = "newton_krylov"
    BLOCK_BANDED = "block_banded"
    AUTO = "auto"


class RuntimePlatform(str, Enum):
    """Preferred runtime platform."""

    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Generic immutable helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenConfig:
    """
    Base class for immutable configuration-like dataclasses.

    This provides three conveniences:
    - .with_updates(...)
    - .to_dict()
    - .pretty()

    It intentionally avoids clever validation magic. Each subclass performs
    its own physical checks in __post_init__.
    """

    def with_updates(self, **kwargs: Any) -> Any:
        """Return a copy with selected fields changed."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary suitable for JSON post-processing."""
        raw = asdict(self)
        return _jsonify(raw)

    def pretty(self) -> str:
        """Human-readable multi-line representation."""
        items = self.to_dict()
        lines = [f"{self.__class__.__name__}:"]
        for key, value in items.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion of dataclass output to JSON-friendly objects."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {
            "array_shape": tuple(int(s) for s in obj.shape),
            "array_dtype": str(obj.dtype),
        }
    return obj


def _check_positive(name: str, value: float) -> None:
    if not float(value) > 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _check_nonnegative(name: str, value: float) -> None:
    if not float(value) >= 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _check_probability_like(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


# ---------------------------------------------------------------------------
# Physical/device-level parameter objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialParams(FrozenConfig):
    """
    Superconducting-film material parameters.

    These are not all used by the first HB residual, but they are important for
    reporting, priors, synthetic recovery, and later material-aware fitting.

    Parameters
    ----------
    name:
        Human-readable material name.
    sheet_kinetic_inductance_H_per_square:
        Sheet kinetic inductance Lk□.
    critical_temperature_K:
        Superconducting critical temperature.
    sheet_resistance_ohm_per_square:
        Normal-state sheet resistance. Optional if unknown.
    thickness_m:
        Film thickness. Optional if unknown.
    london_penetration_depth_m:
        Optional London penetration depth.
    gap_J:
        Optional superconducting gap. If not given, a BCS estimate can be
        computed from Tc using estimate_bcs_gap_J().
    """

    name: str = "NbTiN"
    sheet_kinetic_inductance_H_per_square: float = 8.5 * pH
    critical_temperature_K: float = 13.0
    sheet_resistance_ohm_per_square: float | None = None
    thickness_m: float | None = None
    london_penetration_depth_m: float | None = None
    gap_J: float | None = None

    def __post_init__(self) -> None:
        _check_positive(
            "sheet_kinetic_inductance_H_per_square",
            self.sheet_kinetic_inductance_H_per_square,
        )
        _check_positive("critical_temperature_K", self.critical_temperature_K)
        if self.sheet_resistance_ohm_per_square is not None:
            _check_positive(
                "sheet_resistance_ohm_per_square",
                self.sheet_resistance_ohm_per_square,
            )
        if self.thickness_m is not None:
            _check_positive("thickness_m", self.thickness_m)
        if self.london_penetration_depth_m is not None:
            _check_positive("london_penetration_depth_m", self.london_penetration_depth_m)
        if self.gap_J is not None:
            _check_positive("gap_J", self.gap_J)

    def estimate_bcs_gap_J(self) -> float:
        """
        Estimate the superconducting gap using weak-coupling BCS.

            Δ = 1.762 k_B Tc
        """
        return 1.762 * CONSTANTS.k_B * self.critical_temperature_K

    def effective_gap_J(self) -> float:
        """Return explicit gap if provided; otherwise BCS estimate."""
        if self.gap_J is not None:
            return self.gap_J
        return self.estimate_bcs_gap_J()

    def estimate_sheet_inductance_from_rs_tc(self) -> float | None:
        """
        Estimate sheet kinetic inductance from Rs and Tc using

            Lk□ = hbar Rs / (pi Δ)

        Returns None if sheet resistance is unavailable.
        """
        if self.sheet_resistance_ohm_per_square is None:
            return None
        delta = self.effective_gap_J()
        return CONSTANTS.hbar * self.sheet_resistance_ohm_per_square / (
            float(jnp.pi) * delta
        )


@dataclass(frozen=True)
class NonlinearParams(FrozenConfig):
    """
    Parameters of the nonlinear inductance law.

    Default KI relation:
        L(I) = L0 * (1 + beta_nl * (I / I_star)^2)

    For KI:
        beta_nl = 1

    For weak-current Josephson or symmetric SQUID approximation:
        beta_nl = 0.5
    """

    medium: NonlinearMedium = NonlinearMedium.KINETIC_INDUCTANCE
    beta_nl: float = 1.0
    I_star_A: float = 3.2e-3
    dc_bias_A: float = 0.0
    include_quartic_correction: bool = False
    quartic_coefficient: float = 0.0

    def __post_init__(self) -> None:
        medium = NonlinearMedium(self.medium)
        object.__setattr__(self, "medium", medium)
        _check_positive("I_star_A", self.I_star_A)
        _check_nonnegative("beta_nl", self.beta_nl)
        _check_nonnegative("quartic_coefficient", abs(self.quartic_coefficient))
        if medium == NonlinearMedium.KINETIC_INDUCTANCE and abs(self.beta_nl - 1.0) > 1e-12:
            # Allow advanced users to override beta_nl, but make accidental
            # mismatches obvious by not silently rewriting it.
            pass

    @classmethod
    def kinetic_inductance(cls, I_star_A: float = 3.2e-3) -> "NonlinearParams":
        """Convenience constructor for the default KI model."""
        return cls(
            medium=NonlinearMedium.KINETIC_INDUCTANCE,
            beta_nl=1.0,
            I_star_A=I_star_A,
        )

    @classmethod
    def josephson_weak_current(cls, I_c_A: float) -> "NonlinearParams":
        """Weak-current Josephson-inductance approximation."""
        return cls(
            medium=NonlinearMedium.JOSEPHSON,
            beta_nl=0.5,
            I_star_A=I_c_A,
        )

    @classmethod
    def squid_weak_current(cls, I_c_eff_A: float) -> "NonlinearParams":
        """Weak-current symmetric-SQUID approximation."""
        return cls(
            medium=NonlinearMedium.SQUID,
            beta_nl=0.5,
            I_star_A=I_c_eff_A,
        )


@dataclass(frozen=True)
class LineParams(FrozenConfig):
    """
    Per-unit-length transmission-line parameters.

    These are the physical continuous-line parameters used to generate
    per-cell circuit parameters.

    For a basic lossless line:
        Z0 ≈ sqrt(L_per_m_H / C_per_m_F)
        vp ≈ 1 / sqrt(L_per_m_H * C_per_m_F)

    Losses:
        R_per_m_ohm is series resistance per meter.
        G_per_m_S is shunt conductance per meter.
    """

    length_m: float = 100e-3
    n_cells: int = 20_000
    L_per_m_H: float = 16.64e-6
    C_per_m_F: float = 6.45e-9
    R_per_m_ohm: float = 0.0
    G_per_m_S: float = 0.0
    z0_ohm: float = 50.0
    name: str = "100mm_KI_TWPA"

    def __post_init__(self) -> None:
        _check_positive("length_m", self.length_m)
        if int(self.n_cells) <= 0:
            raise ValueError(f"n_cells must be positive, got {self.n_cells!r}")
        object.__setattr__(self, "n_cells", int(self.n_cells))
        _check_positive("L_per_m_H", self.L_per_m_H)
        _check_positive("C_per_m_F", self.C_per_m_F)
        _check_nonnegative("R_per_m_ohm", self.R_per_m_ohm)
        _check_nonnegative("G_per_m_S", self.G_per_m_S)
        _check_positive("z0_ohm", self.z0_ohm)

    @property
    def dx_m(self) -> float:
        """Cell length in meters."""
        return self.length_m / self.n_cells

    @property
    def L_cell_H(self) -> float:
        """Series inductance per cell."""
        return self.L_per_m_H * self.dx_m

    @property
    def C_cell_F(self) -> float:
        """Shunt capacitance per cell."""
        return self.C_per_m_F * self.dx_m

    @property
    def R_cell_ohm(self) -> float:
        """Series resistance per cell."""
        return self.R_per_m_ohm * self.dx_m

    @property
    def G_cell_S(self) -> float:
        """Shunt conductance per cell."""
        return self.G_per_m_S * self.dx_m

    @property
    def characteristic_impedance_ohm(self) -> float:
        """Lossless characteristic impedance inferred from L/C."""
        return float(jnp.sqrt(self.L_per_m_H / self.C_per_m_F))

    @property
    def phase_velocity_m_per_s(self) -> float:
        """Lossless phase velocity inferred from L and C."""
        return float(1.0 / jnp.sqrt(self.L_per_m_H * self.C_per_m_F))

    @property
    def phase_velocity_fraction_c(self) -> float:
        """Lossless phase velocity as a fraction of c."""
        return self.phase_velocity_m_per_s / CONSTANTS.c0

    @property
    def lc_cutoff_rad_s_cell(self) -> float:
        """
        Approximate artificial angular cutoff of one discrete LC cell.

        For a simple lumped LC ladder, the artificial cutoff scale is of order

            omega_c ≈ 2 / sqrt(L_cell C_cell)

        This is a guardrail, not a substitute for actual dispersion extraction.
        """
        return float(2.0 / jnp.sqrt(self.L_cell_H * self.C_cell_F))

    @property
    def lc_cutoff_hz_cell(self) -> float:
        """Approximate artificial cutoff in Hz."""
        return self.lc_cutoff_rad_s_cell / (2.0 * float(jnp.pi))

    def cell_arrays(self) -> dict[str, jax.Array]:
        """
        Return uniform per-cell arrays as JAX arrays.

        This is useful for quickly constructing a vectorized LineLayout before
        the more detailed layout module is introduced.
        """
        n = self.n_cells
        return {
            "length_m": jnp.full((n,), self.dx_m),
            "L_series_H": jnp.full((n,), self.L_cell_H),
            "C_shunt_F": jnp.full((n,), self.C_cell_F),
            "R_series_ohm": jnp.full((n,), self.R_cell_ohm),
            "G_shunt_S": jnp.full((n,), self.G_cell_S),
        }

    @classmethod
    def from_z0_vp(
        cls,
        *,
        length_m: float,
        n_cells: int,
        z0_ohm: float,
        phase_velocity_m_per_s: float,
        R_per_m_ohm: float = 0.0,
        G_per_m_S: float = 0.0,
        name: str = "line_from_z0_vp",
    ) -> "LineParams":
        """
        Construct L and C per unit length from Z0 and phase velocity.

            L = Z0 / vp
            C = 1 / (Z0 vp)
        """
        _check_positive("z0_ohm", z0_ohm)
        _check_positive("phase_velocity_m_per_s", phase_velocity_m_per_s)
        return cls(
            length_m=length_m,
            n_cells=n_cells,
            L_per_m_H=z0_ohm / phase_velocity_m_per_s,
            C_per_m_F=1.0 / (z0_ohm * phase_velocity_m_per_s),
            R_per_m_ohm=R_per_m_ohm,
            G_per_m_S=G_per_m_S,
            z0_ohm=z0_ohm,
            name=name,
        )


@dataclass(frozen=True)
class PeriodicLoadingParams(FrozenConfig):
    """
    Periodic loading / dispersion-engineering parameters.

    This is intentionally generic. The first linear-layout generator can use
    sinusoidal or block modulation of C_shunt and/or L_series. Later, this will
    be replaced or extended by layout-derived cell arrays.

    Parameters
    ----------
    enabled:
        If false, no modulation is applied.
    period_m:
        Spatial modulation period.
    capacitance_modulation_fraction:
        Relative modulation amplitude of C.
    inductance_modulation_fraction:
        Relative modulation amplitude of L.
    phase_rad:
        Spatial phase offset.
    waveform:
        "sinusoidal" or "square".
    """

    enabled: bool = False
    period_m: float = 122.7e-6
    capacitance_modulation_fraction: float = 0.0
    inductance_modulation_fraction: float = 0.0
    phase_rad: float = 0.0
    waveform: Literal["sinusoidal", "square"] = "sinusoidal"

    def __post_init__(self) -> None:
        _check_positive("period_m", self.period_m)
        if abs(self.capacitance_modulation_fraction) >= 1.0:
            raise ValueError("capacitance_modulation_fraction must have |value| < 1")
        if abs(self.inductance_modulation_fraction) >= 1.0:
            raise ValueError("inductance_modulation_fraction must have |value| < 1")
        if self.waveform not in {"sinusoidal", "square"}:
            raise ValueError(f"Unsupported waveform {self.waveform!r}")


@dataclass(frozen=True)
class ResonatorLoadingParams(FrozenConfig):
    """
    Optional periodic shunt resonator loading.

    The first implementation can ignore this, but the object is defined now so
    the architecture is stable when resonator-loaded supercells are added.
    """

    enabled: bool = False
    every_n_cells: int = 8
    C_couple_F: float = 0.0
    L_res_H: float = 0.0
    C_res_F: float = 0.0
    loss_res_ohm: float = 0.0

    def __post_init__(self) -> None:
        if int(self.every_n_cells) <= 0:
            raise ValueError("every_n_cells must be positive")
        object.__setattr__(self, "every_n_cells", int(self.every_n_cells))
        _check_nonnegative("C_couple_F", self.C_couple_F)
        _check_nonnegative("L_res_H", self.L_res_H)
        _check_nonnegative("C_res_F", self.C_res_F)
        _check_nonnegative("loss_res_ohm", self.loss_res_ohm)

    @property
    def resonance_hz(self) -> float | None:
        """Return LC resonance in Hz if L and C are positive."""
        if self.L_res_H <= 0.0 or self.C_res_F <= 0.0:
            return None
        omega = 1.0 / jnp.sqrt(self.L_res_H * self.C_res_F)
        return float(omega / (2.0 * jnp.pi))


@dataclass(frozen=True)
class DeviceParams(FrozenConfig):
    """
    Complete physical device parameter bundle.

    This combines material, line, nonlinearity, and loading parameters.
    Numerical solver parameters live elsewhere.
    """

    line: LineParams = LineParams()
    material: MaterialParams = MaterialParams()
    nonlinear: NonlinearParams = NonlinearParams()
    periodic_loading: PeriodicLoadingParams = PeriodicLoadingParams()
    resonator_loading: ResonatorLoadingParams = ResonatorLoadingParams()
    mixing_regime: MixingRegime = MixingRegime.DP4WM

    def __post_init__(self) -> None:
        object.__setattr__(self, "mixing_regime", MixingRegime(self.mixing_regime))

    @classmethod
    def default_100mm_kitwpa(cls) -> "DeviceParams":
        """
        Default 100 mm / 20,000-cell KI-TWPA-style device.

        The numerical values are reasonable baseline placeholders, inspired by
        100 mm NbTiN microstrip KIT designs:
            length = 100 mm
            Z0 ≈ 50 ohm
            vp ≈ 0.010 c
            L ≈ 16.64 uH/m
            C ≈ 6.45 nF/m

        They are not a substitute for fitting to the actual device layout or
        measured pump-off S21.
        """
        return cls(
            line=LineParams(
                length_m=100e-3,
                n_cells=20_000,
                L_per_m_H=16.64e-6,
                C_per_m_F=6.45e-9,
                R_per_m_ohm=0.0,
                G_per_m_S=0.0,
                z0_ohm=50.0,
                name="default_100mm_kitwpa",
            ),
            material=MaterialParams(
                name="NbTiN",
                sheet_kinetic_inductance_H_per_square=8.5 * pH,
                critical_temperature_K=13.0,
            ),
            nonlinear=NonlinearParams.kinetic_inductance(I_star_A=3.2e-3),
            periodic_loading=PeriodicLoadingParams(
                enabled=True,
                period_m=122.7e-6,
                capacitance_modulation_fraction=0.05,
                inductance_modulation_fraction=0.0,
                waveform="sinusoidal",
            ),
            resonator_loading=ResonatorLoadingParams(enabled=False),
            mixing_regime=MixingRegime.DP4WM,
        )


# ---------------------------------------------------------------------------
# Frequency/pump/signal configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PumpConfig(FrozenConfig):
    """
    Pump-tone configuration.

    Parameters
    ----------
    frequency_hz:
        Pump frequency.
    available_power_dbm:
        Available source power at the device reference plane unless explicitly
        corrected outside this object.
    source_impedance_ohm:
        Reference impedance for converting available power to current/voltage.
    harmonics:
        Number of pump harmonics to include in pump-only HB.
    phase_rad:
        Pump phase at the input reference plane.
    """

    frequency_hz: float = 10.6 * GHz
    available_power_dbm: float = -20.5
    source_impedance_ohm: float = 50.0
    harmonics: int = 3
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        _check_positive("frequency_hz", self.frequency_hz)
        _check_positive("source_impedance_ohm", self.source_impedance_ohm)
        if int(self.harmonics) <= 0:
            raise ValueError("harmonics must be positive")
        object.__setattr__(self, "harmonics", int(self.harmonics))

    @property
    def angular_frequency_rad_s(self) -> float:
        return 2.0 * float(jnp.pi) * self.frequency_hz

    @property
    def available_power_W(self) -> float:
        return float(dbm_to_watts(self.available_power_dbm))

    @property
    def source_peak_current_A(self) -> float:
        """
        Peak current corresponding to available power and source impedance.

        For actual line current, the solver boundary condition must still account
        for source/load matching. This property is only a convenient scale.
        """
        return float(watts_to_ipeak(self.available_power_W, self.source_impedance_ohm))

    def with_power_dbm(self, power_dbm: float) -> "PumpConfig":
        return self.with_updates(available_power_dbm=power_dbm)

    def with_frequency_hz(self, frequency_hz: float) -> "PumpConfig":
        return self.with_updates(frequency_hz=frequency_hz)


@dataclass(frozen=True)
class SignalConfig(FrozenConfig):
    """
    Signal/probe configuration for gain and conversion calculations.
    """

    frequency_start_hz: float = 4.0 * GHz
    frequency_stop_hz: float = 8.0 * GHz
    n_points: int = 401
    input_power_dbm: float = -130.0
    source_port: int = 1
    output_port: int = 2
    sideband_order: int = 3

    def __post_init__(self) -> None:
        _check_positive("frequency_start_hz", self.frequency_start_hz)
        _check_positive("frequency_stop_hz", self.frequency_stop_hz)
        if self.frequency_stop_hz <= self.frequency_start_hz:
            raise ValueError("frequency_stop_hz must exceed frequency_start_hz")
        if int(self.n_points) <= 1:
            raise ValueError("n_points must exceed 1")
        if int(self.source_port) <= 0 or int(self.output_port) <= 0:
            raise ValueError("ports are 1-indexed positive integers")
        if int(self.sideband_order) < 1:
            raise ValueError("sideband_order must be >= 1")
        object.__setattr__(self, "n_points", int(self.n_points))
        object.__setattr__(self, "source_port", int(self.source_port))
        object.__setattr__(self, "output_port", int(self.output_port))
        object.__setattr__(self, "sideband_order", int(self.sideband_order))

    def frequency_grid_hz(self) -> jax.Array:
        """Signal frequency grid in Hz."""
        return jnp.linspace(
            self.frequency_start_hz,
            self.frequency_stop_hz,
            self.n_points,
        )


@dataclass(frozen=True)
class FrequencyGridConfig(FrozenConfig):
    """
    Generic frequency grid for pump-off S-parameter/dispersion scans.
    """

    start_hz: float = 1.0 * GHz
    stop_hz: float = 20.0 * GHz
    n_points: int = 1901

    def __post_init__(self) -> None:
        _check_positive("start_hz", self.start_hz)
        _check_positive("stop_hz", self.stop_hz)
        if self.stop_hz <= self.start_hz:
            raise ValueError("stop_hz must exceed start_hz")
        if int(self.n_points) <= 1:
            raise ValueError("n_points must exceed 1")
        object.__setattr__(self, "n_points", int(self.n_points))

    def frequencies_hz(self) -> jax.Array:
        return jnp.linspace(self.start_hz, self.stop_hz, self.n_points)

    def angular_frequencies_rad_s(self) -> jax.Array:
        return 2.0 * jnp.pi * self.frequencies_hz()


# ---------------------------------------------------------------------------
# Numerical configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverConfig(FrozenConfig):
    """
    Nonlinear solver configuration.

    Dense Newton is fine for validation. Industrial 20,000-cell simulations
    should use matrix-free or block-structured methods.
    """

    backend: SolverBackend = SolverBackend.AUTO
    max_iter: int = 50
    abs_tol: float = 1e-10
    rel_tol: float = 1e-10
    step_tol: float = 1e-12
    damping_initial: float = 1.0
    damping_min: float = 1e-6
    damping_backtracking_factor: float = 0.5
    max_backtracking_steps: int = 20
    regularization: float = 0.0
    fail_on_nonconvergence: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", SolverBackend(self.backend))
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        object.__setattr__(self, "max_iter", int(self.max_iter))
        _check_positive("abs_tol", self.abs_tol)
        _check_positive("rel_tol", self.rel_tol)
        _check_positive("step_tol", self.step_tol)
        _check_positive("damping_initial", self.damping_initial)
        _check_positive("damping_min", self.damping_min)
        _check_probability_like(
            "damping_backtracking_factor",
            self.damping_backtracking_factor,
        )
        if self.damping_backtracking_factor in (0.0, 1.0):
            raise ValueError("damping_backtracking_factor must be strictly between 0 and 1")
        if int(self.max_backtracking_steps) < 0:
            raise ValueError("max_backtracking_steps must be non-negative")
        object.__setattr__(
            self,
            "max_backtracking_steps",
            int(self.max_backtracking_steps),
        )
        _check_nonnegative("regularization", self.regularization)


@dataclass(frozen=True)
class KrylovConfig(FrozenConfig):
    """
    Matrix-free Newton-Krylov linear-solve configuration.
    """

    gmres_tol: float = 1e-8
    gmres_atol: float = 1e-12
    gmres_restart: int = 50
    gmres_maxiter: int = 500
    use_preconditioner: bool = True
    preconditioner: Literal["none", "diagonal", "block_diagonal", "linear_ladder"] = (
        "diagonal"
    )

    def __post_init__(self) -> None:
        _check_positive("gmres_tol", self.gmres_tol)
        _check_nonnegative("gmres_atol", self.gmres_atol)
        if int(self.gmres_restart) <= 0:
            raise ValueError("gmres_restart must be positive")
        if int(self.gmres_maxiter) <= 0:
            raise ValueError("gmres_maxiter must be positive")
        object.__setattr__(self, "gmres_restart", int(self.gmres_restart))
        object.__setattr__(self, "gmres_maxiter", int(self.gmres_maxiter))
        if self.preconditioner not in {
            "none",
            "diagonal",
            "block_diagonal",
            "linear_ladder",
        }:
            raise ValueError(f"Unsupported preconditioner {self.preconditioner!r}")


@dataclass(frozen=True)
class ContinuationConfig(FrozenConfig):
    """
    Continuation/homotopy configuration.

    The most common production route is pump-power continuation:
        low pump power -> target pump power
    """

    enabled: bool = True
    parameter_name: str = "pump_power_dbm"
    n_steps: int = 21
    start_value: float | None = None
    stop_value: float | None = None
    adaptive: bool = True
    min_step_fraction: float = 1e-3
    growth_factor: float = 1.25
    shrink_factor: float = 0.5

    def __post_init__(self) -> None:
        if int(self.n_steps) <= 0:
            raise ValueError("n_steps must be positive")
        object.__setattr__(self, "n_steps", int(self.n_steps))
        _check_positive("min_step_fraction", self.min_step_fraction)
        if not (0.0 < self.min_step_fraction <= 1.0):
            raise ValueError("min_step_fraction must be in (0, 1]")
        _check_positive("growth_factor", self.growth_factor)
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1")
        _check_positive("shrink_factor", self.shrink_factor)
        if not (0.0 < self.shrink_factor < 1.0):
            raise ValueError("shrink_factor must be in (0, 1)")


@dataclass(frozen=True)
class ScalingConfig(FrozenConfig):
    """
    Residual and unknown scaling configuration.

    Branch equations are voltage residuals; KCL equations are current residuals.
    Scaling them separately is essential for stable Newton/Krylov methods.
    """

    enabled: bool = True
    voltage_scale_V: float = 1e-3
    current_scale_A: float = 1e-6
    min_voltage_scale_V: float = 1e-12
    min_current_scale_A: float = 1e-15

    def __post_init__(self) -> None:
        _check_positive("voltage_scale_V", self.voltage_scale_V)
        _check_positive("current_scale_A", self.current_scale_A)
        _check_positive("min_voltage_scale_V", self.min_voltage_scale_V)
        _check_positive("min_current_scale_A", self.min_current_scale_A)

    @classmethod
    def from_pump_power(
        cls,
        pump_dbm: float,
        z0_ohm: float = 50.0,
        voltage_floor: float = 1e-9,
        current_floor: float = 1e-12,
    ) -> "ScalingConfig":
        """
        Build rough voltage/current scales from available pump power.
        """
        p = dbm_to_watts(pump_dbm)
        v_peak = jnp.sqrt(2.0 * p * z0_ohm)
        i_peak = jnp.sqrt(2.0 * p / z0_ohm)
        return cls(
            enabled=True,
            voltage_scale_V=float(jnp.maximum(v_peak, voltage_floor)),
            current_scale_A=float(jnp.maximum(i_peak, current_floor)),
            min_voltage_scale_V=voltage_floor,
            min_current_scale_A=current_floor,
        )


@dataclass(frozen=True)
class RuntimeConfig(FrozenConfig):
    """
    JAX/runtime configuration.

    This object does not itself call jax.config.update because that should happen
    at process start before importing heavy numerical modules.
    """

    platform: RuntimePlatform = RuntimePlatform.AUTO
    enable_x64: bool = True
    jit: bool = True
    chunk_size: int = 256
    batch_size: int = 32
    checkpoint_dir: str = "outputs/checkpoints"
    output_dir: str = "outputs"
    random_seed: int = 1234

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", RuntimePlatform(self.platform))
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "random_seed", int(self.random_seed))

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


@dataclass(frozen=True)
class SimulationConfig(FrozenConfig):
    """
    Top-level simulation configuration for scripts.

    This combines device parameters, pump/signal/frequency grids, solvers, and
    runtime choices into one object.
    """

    device: DeviceParams = DeviceParams.default_100mm_kitwpa()
    pump: PumpConfig = PumpConfig()
    signal: SignalConfig = SignalConfig()
    linear_grid: FrequencyGridConfig = FrequencyGridConfig()
    solver: SolverConfig = SolverConfig()
    krylov: KrylovConfig = KrylovConfig()
    continuation: ContinuationConfig = ContinuationConfig()
    scaling: ScalingConfig = ScalingConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    run_name: str = "twpa_run"

    @classmethod
    def default_validation_small(cls) -> "SimulationConfig":
        """
        Tiny configuration for fast validation tests.
        """
        device = DeviceParams.default_100mm_kitwpa()
        small_line = device.line.with_updates(
            length_m=1e-3,
            n_cells=10,
            name="validation_10_cell_line",
        )
        device = device.with_updates(
            line=small_line,
            periodic_loading=device.periodic_loading.with_updates(enabled=False),
        )
        return cls(
            device=device,
            pump=PumpConfig(
                frequency_hz=6.0 * GHz,
                available_power_dbm=-80.0,
                harmonics=3,
            ),
            signal=SignalConfig(
                frequency_start_hz=4.0 * GHz,
                frequency_stop_hz=8.0 * GHz,
                n_points=21,
                input_power_dbm=-130.0,
                sideband_order=2,
            ),
            linear_grid=FrequencyGridConfig(
                start_hz=1.0 * GHz,
                stop_hz=12.0 * GHz,
                n_points=101,
            ),
            solver=SolverConfig(
                backend=SolverBackend.DENSE,
                max_iter=30,
                abs_tol=1e-10,
                rel_tol=1e-10,
            ),
            continuation=ContinuationConfig(
                enabled=True,
                n_steps=5,
            ),
            scaling=ScalingConfig.from_pump_power(-80.0),
            run_name="validation_small",
        )

    @classmethod
    def default_100mm_industrial(cls) -> "SimulationConfig":
        """
        Full 100 mm / 20,000-cell baseline configuration.
        """
        device = DeviceParams.default_100mm_kitwpa()
        pump = PumpConfig(
            frequency_hz=10.6 * GHz,
            available_power_dbm=-20.5,
            harmonics=3,
        )
        return cls(
            device=device,
            pump=pump,
            signal=SignalConfig(
                frequency_start_hz=4.0 * GHz,
                frequency_stop_hz=8.0 * GHz,
                n_points=401,
                input_power_dbm=-130.0,
                sideband_order=3,
            ),
            linear_grid=FrequencyGridConfig(
                start_hz=1.0 * GHz,
                stop_hz=20.0 * GHz,
                n_points=1901,
            ),
            solver=SolverConfig(
                backend=SolverBackend.AUTO,
                max_iter=60,
                abs_tol=1e-9,
                rel_tol=1e-9,
                step_tol=1e-11,
                fail_on_nonconvergence=False,
                verbose=True,
            ),
            krylov=KrylovConfig(
                gmres_tol=1e-8,
                gmres_atol=1e-12,
                gmres_restart=80,
                gmres_maxiter=1000,
                use_preconditioner=True,
                preconditioner="linear_ladder",
            ),
            continuation=ContinuationConfig(
                enabled=True,
                parameter_name="pump_power_dbm",
                n_steps=31,
                start_value=-90.0,
                stop_value=-20.5,
                adaptive=True,
            ),
            scaling=ScalingConfig.from_pump_power(-20.5),
            runtime=RuntimeConfig(
                platform=RuntimePlatform.AUTO,
                enable_x64=True,
                jit=True,
                chunk_size=512,
                batch_size=16,
                checkpoint_dir="outputs/checkpoints",
                output_dir="outputs",
            ),
            run_name="industrial_100mm_20000cell",
        )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def simulation_config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    """JSON-friendly dict for a SimulationConfig."""
    return config.to_dict()


def summarize_device(config: SimulationConfig | DeviceParams) -> dict[str, Any]:
    """
    Compact physical summary for logs and reports.
    """
    device = config.device if isinstance(config, SimulationConfig) else config
    line = device.line
    return {
        "name": line.name,
        "length_m": line.length_m,
        "n_cells": line.n_cells,
        "dx_m": line.dx_m,
        "L_per_m_H": line.L_per_m_H,
        "C_per_m_F": line.C_per_m_F,
        "L_cell_H": line.L_cell_H,
        "C_cell_F": line.C_cell_F,
        "z0_target_ohm": line.z0_ohm,
        "z0_lc_ohm": line.characteristic_impedance_ohm,
        "phase_velocity_m_per_s": line.phase_velocity_m_per_s,
        "phase_velocity_fraction_c": line.phase_velocity_fraction_c,
        "lc_cutoff_hz_cell": line.lc_cutoff_hz_cell,
        "medium": device.nonlinear.medium.value,
        "beta_nl": device.nonlinear.beta_nl,
        "I_star_A": device.nonlinear.I_star_A,
        "periodic_loading_enabled": device.periodic_loading.enabled,
        "period_m": device.periodic_loading.period_m,
    }


def make_prng_key(config: RuntimeConfig | SimulationConfig) -> jax.Array:
    """Create a JAX PRNG key from runtime config."""
    runtime = config.runtime if isinstance(config, SimulationConfig) else config
    return jax.random.PRNGKey(runtime.random_seed)


__all__ = [
    "ArrayLike",
    "NonlinearMedium",
    "MixingRegime",
    "SolverBackend",
    "RuntimePlatform",
    "FrozenConfig",
    "MaterialParams",
    "NonlinearParams",
    "LineParams",
    "PeriodicLoadingParams",
    "ResonatorLoadingParams",
    "DeviceParams",
    "PumpConfig",
    "SignalConfig",
    "FrequencyGridConfig",
    "SolverConfig",
    "KrylovConfig",
    "ContinuationConfig",
    "ScalingConfig",
    "RuntimeConfig",
    "SimulationConfig",
    "simulation_config_to_dict",
    "summarize_device",
    "make_prng_key",
]
