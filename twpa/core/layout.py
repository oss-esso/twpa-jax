"""
twpa.core.layout
================

Vectorized layout representation for long TWPA transmission lines.

This module converts physical line parameters into per-cell arrays:

    length_m[N]
    L_series_H[N]
    C_shunt_F[N]
    R_series_ohm[N]
    G_shunt_S[N]

and optional loading arrays:

    C_stub_F[N]
    L_res_H[N]
    C_res_F[N]
    C_couple_F[N]

The important design rule is that industrial-size lines must not be stored as
lists of Python cell objects inside numerical functions. A 100 mm / 20,000-cell
line must be represented by vectorized JAX arrays.

This file does not build ABCD matrices yet. That will be done in
twpa.linear.cells and twpa.linear.cascade. This file only defines the physical
layout/cell data that those modules consume.

References encoded in the design
--------------------------------
- The ADS/KI-TWPA modeling workflow represents a superconducting transmission
  line as many lumped unit cells with nonlinear inductors and shunt capacitors.
- The industrial simulator must support both uniform artificial lines and
  periodic loading / bandgap engineering.
- For scalable simulation, supercell grouping is explicit but still represented
  as arrays, not Python loops inside JIT paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import jax
import jax.numpy as jnp

from .params import (
    DeviceParams,
    LineParams,
    PeriodicLoadingParams,
    ResonatorLoadingParams,
)


ArrayLike = Any


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _as_1d_array(name: str, value: ArrayLike, *, dtype: Any | None = None) -> jax.Array:
    """Convert an input to a 1D JAX array with a helpful error message."""
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    return arr


def _broadcast_to_n(name: str, value: ArrayLike, n: int, *, dtype: Any | None = None) -> jax.Array:
    """
    Convert scalar or 1D array to a 1D array of length n.

    Scalars are broadcast. Arrays must already have length n.
    """
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return jnp.full((n,), arr, dtype=arr.dtype if dtype is None else dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
    return arr


def _require_same_length(named_arrays: Mapping[str, jax.Array]) -> int:
    """Validate that all arrays have the same first dimension."""
    lengths = {name: int(arr.shape[0]) for name, arr in named_arrays.items()}
    unique = set(lengths.values())
    if len(unique) != 1:
        raise ValueError(f"All layout arrays must have same length, got {lengths}")
    return next(iter(unique))


def _require_nonnegative_array(name: str, arr: jax.Array) -> None:
    """Eager validation helper. Do not use inside JIT."""
    min_value = float(jnp.min(arr))
    if min_value < 0.0:
        raise ValueError(f"{name} must be non-negative; minimum is {min_value}")


def _require_positive_array(name: str, arr: jax.Array) -> None:
    """Eager validation helper. Do not use inside JIT."""
    min_value = float(jnp.min(arr))
    if min_value <= 0.0:
        raise ValueError(f"{name} must be positive; minimum is {min_value}")


def _safe_json_value(value: Any) -> Any:
    """Best-effort JSON-friendly conversion for metadata/reporting."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "array_shape": tuple(int(s) for s in value.shape),
            "array_dtype": str(value.dtype),
        }
    return value


# ---------------------------------------------------------------------------
# Cell-array layout object
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LineLayout:
    """
    Vectorized transmission-line layout.

    Parameters
    ----------
    length_m:
        Cell lengths, shape (N,).
    L_series_H:
        Series inductance per cell, shape (N,).
    C_shunt_F:
        Shunt capacitance per cell, shape (N,).
    R_series_ohm:
        Series resistance per cell, shape (N,).
    G_shunt_S:
        Shunt conductance per cell, shape (N,).
    C_stub_F:
        Extra shunt stub capacitance per cell, shape (N,).
    L_res_H, C_res_F, C_couple_F:
        Optional resonator-loading arrays. Zero values mean disabled at that
        cell. These are placeholders for later resonator/supercell models.
    z0_ohm:
        Reference impedance.
    name:
        Human-readable layout name.
    metadata:
        Small static metadata dictionary. Do not place large arrays here.

    Notes
    -----
    This object is registered as a JAX PyTree. The arrays are dynamic leaves.
    The name and metadata are static auxiliary data. This means the layout can
    be passed into JAX-transformed functions, but changing metadata will trigger
    recompilation if used as a static arg.
    """

    length_m: jax.Array
    L_series_H: jax.Array
    C_shunt_F: jax.Array
    R_series_ohm: jax.Array
    G_shunt_S: jax.Array
    C_stub_F: jax.Array
    L_res_H: jax.Array
    C_res_F: jax.Array
    C_couple_F: jax.Array
    z0_ohm: float = 50.0
    name: str = "line_layout"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        arrays = {
            "length_m": _as_1d_array("length_m", self.length_m),
            "L_series_H": _as_1d_array("L_series_H", self.L_series_H),
            "C_shunt_F": _as_1d_array("C_shunt_F", self.C_shunt_F),
            "R_series_ohm": _as_1d_array("R_series_ohm", self.R_series_ohm),
            "G_shunt_S": _as_1d_array("G_shunt_S", self.G_shunt_S),
            "C_stub_F": _as_1d_array("C_stub_F", self.C_stub_F),
            "L_res_H": _as_1d_array("L_res_H", self.L_res_H),
            "C_res_F": _as_1d_array("C_res_F", self.C_res_F),
            "C_couple_F": _as_1d_array("C_couple_F", self.C_couple_F),
        }
        _require_same_length(arrays)

        _require_positive_array("length_m", arrays["length_m"])
        _require_positive_array("L_series_H", arrays["L_series_H"])
        _require_positive_array("C_shunt_F", arrays["C_shunt_F"])
        _require_nonnegative_array("R_series_ohm", arrays["R_series_ohm"])
        _require_nonnegative_array("G_shunt_S", arrays["G_shunt_S"])
        _require_nonnegative_array("C_stub_F", arrays["C_stub_F"])
        _require_nonnegative_array("L_res_H", arrays["L_res_H"])
        _require_nonnegative_array("C_res_F", arrays["C_res_F"])
        _require_nonnegative_array("C_couple_F", arrays["C_couple_F"])

        if float(self.z0_ohm) <= 0.0:
            raise ValueError(f"z0_ohm must be positive, got {self.z0_ohm!r}")

        for key, arr in arrays.items():
            object.__setattr__(self, key, arr)

        object.__setattr__(self, "z0_ohm", float(self.z0_ohm))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))

    # ------------------------------------------------------------------
    # JAX PyTree implementation
    # ------------------------------------------------------------------

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], dict[str, Any]]:
        children = (
            self.length_m,
            self.L_series_H,
            self.C_shunt_F,
            self.R_series_ohm,
            self.G_shunt_S,
            self.C_stub_F,
            self.L_res_H,
            self.C_res_F,
            self.C_couple_F,
        )
        aux = {
            "z0_ohm": self.z0_ohm,
            "name": self.name,
            "metadata": dict(self.metadata or {}),
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: dict[str, Any], children: tuple[jax.Array, ...]) -> "LineLayout":
        return cls(
            length_m=children[0],
            L_series_H=children[1],
            C_shunt_F=children[2],
            R_series_ohm=children[3],
            G_shunt_S=children[4],
            C_stub_F=children[5],
            L_res_H=children[6],
            C_res_F=children[7],
            C_couple_F=children[8],
            z0_ohm=aux["z0_ohm"],
            name=aux["name"],
            metadata=aux["metadata"],
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of lumped cells."""
        return int(self.length_m.shape[0])

    @property
    def total_length_m(self) -> float:
        """Total physical length in meters."""
        return float(jnp.sum(self.length_m))

    @property
    def mean_dx_m(self) -> float:
        """Mean cell length."""
        return float(jnp.mean(self.length_m))

    @property
    def total_shunt_C_F(self) -> jax.Array:
        """
        Total shunt capacitance per cell, including stub capacitance.

        This is the default capacitance used by linear-cell construction.
        """
        return self.C_shunt_F + self.C_stub_F

    @property
    def x_cell_start_m(self) -> jax.Array:
        """Cell start positions, shape (N,)."""
        return jnp.concatenate(
            [jnp.asarray([0.0], dtype=self.length_m.dtype), jnp.cumsum(self.length_m[:-1])]
        )

    @property
    def x_cell_center_m(self) -> jax.Array:
        """Cell center positions, shape (N,)."""
        return self.x_cell_start_m + 0.5 * self.length_m

    @property
    def x_cell_stop_m(self) -> jax.Array:
        """Cell stop positions, shape (N,)."""
        return jnp.cumsum(self.length_m)

    @property
    def has_resonators(self) -> bool:
        """Whether any resonator/coupling entries are nonzero."""
        return bool(
            float(jnp.max(self.L_res_H)) > 0.0
            or float(jnp.max(self.C_res_F)) > 0.0
            or float(jnp.max(self.C_couple_F)) > 0.0
        )

    @property
    def has_stub_loading(self) -> bool:
        """Whether any stub capacitance entries are nonzero."""
        return bool(float(jnp.max(self.C_stub_F)) > 0.0)

    @property
    def characteristic_impedance_cell_ohm(self) -> jax.Array:
        """
        Approximate per-cell lossless characteristic impedance.

        Z_cell ≈ sqrt(L_series / C_total)

        This is a diagnostic, not a rigorous Bloch impedance for a periodic line.
        """
        return jnp.sqrt(self.L_series_H / self.total_shunt_C_F)

    @property
    def phase_velocity_cell_m_per_s(self) -> jax.Array:
        """
        Approximate per-cell phase velocity.

        v_cell ≈ dx / sqrt(L_cell C_cell)
        """
        return self.length_m / jnp.sqrt(self.L_series_H * self.total_shunt_C_F)

    @property
    def artificial_cutoff_hz_cell(self) -> jax.Array:
        """
        Approximate LC-ladder artificial cutoff for each cell.

            f_c ≈ [2 / sqrt(L C)] / (2π)

        This is a guardrail used by validation reports.
        """
        omega_c = 2.0 / jnp.sqrt(self.L_series_H * self.total_shunt_C_F)
        return omega_c / (2.0 * jnp.pi)

    # ------------------------------------------------------------------
    # Immutable modifiers
    # ------------------------------------------------------------------

    def with_updates(self, **kwargs: Any) -> "LineLayout":
        """Return an immutable copy with selected fields replaced."""
        return replace(self, **kwargs)

    def with_metadata(self, **metadata_updates: Any) -> "LineLayout":
        """Return a copy with merged metadata."""
        current = dict(self.metadata or {})
        current.update(metadata_updates)
        return replace(self, metadata=current)

    def scaled(
        self,
        *,
        L_scale: float = 1.0,
        C_scale: float = 1.0,
        R_scale: float = 1.0,
        G_scale: float = 1.0,
        C_stub_scale: float = 1.0,
    ) -> "LineLayout":
        """
        Return a copy with global scale factors applied.

        Useful for synthetic fitting experiments and parameter sweeps.
        """
        return replace(
            self,
            L_series_H=self.L_series_H * L_scale,
            C_shunt_F=self.C_shunt_F * C_scale,
            R_series_ohm=self.R_series_ohm * R_scale,
            G_shunt_S=self.G_shunt_S * G_scale,
            C_stub_F=self.C_stub_F * C_stub_scale,
            metadata={
                **dict(self.metadata or {}),
                "L_scale": L_scale,
                "C_scale": C_scale,
                "R_scale": R_scale,
                "G_scale": G_scale,
                "C_stub_scale": C_stub_scale,
            },
        )

    def as_cell_dict(self) -> dict[str, jax.Array]:
        """Return the main arrays as a dictionary."""
        return {
            "length_m": self.length_m,
            "L_series_H": self.L_series_H,
            "C_shunt_F": self.C_shunt_F,
            "R_series_ohm": self.R_series_ohm,
            "G_shunt_S": self.G_shunt_S,
            "C_stub_F": self.C_stub_F,
            "L_res_H": self.L_res_H,
            "C_res_F": self.C_res_F,
            "C_couple_F": self.C_couple_F,
        }

    def summary(self) -> dict[str, Any]:
        """Compact JSON-friendly layout summary."""
        return {
            "name": self.name,
            "n_cells": self.n_cells,
            "total_length_m": self.total_length_m,
            "mean_dx_m": self.mean_dx_m,
            "z0_ohm": self.z0_ohm,
            "L_series_H_min": float(jnp.min(self.L_series_H)),
            "L_series_H_max": float(jnp.max(self.L_series_H)),
            "C_total_F_min": float(jnp.min(self.total_shunt_C_F)),
            "C_total_F_max": float(jnp.max(self.total_shunt_C_F)),
            "R_series_ohm_max": float(jnp.max(self.R_series_ohm)),
            "G_shunt_S_max": float(jnp.max(self.G_shunt_S)),
            "has_stub_loading": self.has_stub_loading,
            "has_resonators": self.has_resonators,
            "artificial_cutoff_hz_min": float(jnp.min(self.artificial_cutoff_hz_cell)),
            "artificial_cutoff_hz_max": float(jnp.max(self.artificial_cutoff_hz_cell)),
            "metadata": _safe_json_value(dict(self.metadata or {})),
        }


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def make_uniform_layout(
    line: LineParams,
    *,
    name: str | None = None,
    dtype: Any | None = None,
) -> LineLayout:
    """
    Build a uniform vectorized layout from LineParams.

    Parameters
    ----------
    line:
        Per-unit-length line parameters.
    name:
        Optional layout name.
    dtype:
        Optional dtype for arrays.
    """
    n = int(line.n_cells)
    length = jnp.full((n,), line.dx_m, dtype=dtype)
    L = jnp.full((n,), line.L_cell_H, dtype=dtype)
    C = jnp.full((n,), line.C_cell_F, dtype=dtype)
    R = jnp.full((n,), line.R_cell_ohm, dtype=dtype)
    G = jnp.full((n,), line.G_cell_S, dtype=dtype)
    zeros = jnp.zeros((n,), dtype=L.dtype)

    return LineLayout(
        length_m=length,
        L_series_H=L,
        C_shunt_F=C,
        R_series_ohm=R,
        G_shunt_S=G,
        C_stub_F=zeros,
        L_res_H=zeros,
        C_res_F=zeros,
        C_couple_F=zeros,
        z0_ohm=line.z0_ohm,
        name=name or line.name,
        metadata={
            "source": "make_uniform_layout",
            "line_name": line.name,
            "length_m": line.length_m,
            "n_cells": line.n_cells,
            "L_per_m_H": line.L_per_m_H,
            "C_per_m_F": line.C_per_m_F,
            "R_per_m_ohm": line.R_per_m_ohm,
            "G_per_m_S": line.G_per_m_S,
        },
    )


def make_layout_from_arrays(
    *,
    length_m: ArrayLike,
    L_series_H: ArrayLike,
    C_shunt_F: ArrayLike,
    R_series_ohm: ArrayLike | float = 0.0,
    G_shunt_S: ArrayLike | float = 0.0,
    C_stub_F: ArrayLike | float = 0.0,
    L_res_H: ArrayLike | float = 0.0,
    C_res_F: ArrayLike | float = 0.0,
    C_couple_F: ArrayLike | float = 0.0,
    z0_ohm: float = 50.0,
    name: str = "array_layout",
    metadata: Mapping[str, Any] | None = None,
    dtype: Any | None = None,
) -> LineLayout:
    """
    Build a layout from explicit scalar/array cell data.

    Scalar optional arrays are broadcast to length N.
    """
    length = _as_1d_array("length_m", length_m, dtype=dtype)
    n = int(length.shape[0])

    L = _broadcast_to_n("L_series_H", L_series_H, n, dtype=dtype)
    C = _broadcast_to_n("C_shunt_F", C_shunt_F, n, dtype=dtype)
    R = _broadcast_to_n("R_series_ohm", R_series_ohm, n, dtype=dtype)
    G = _broadcast_to_n("G_shunt_S", G_shunt_S, n, dtype=dtype)
    C_stub = _broadcast_to_n("C_stub_F", C_stub_F, n, dtype=dtype)
    L_res = _broadcast_to_n("L_res_H", L_res_H, n, dtype=dtype)
    C_res = _broadcast_to_n("C_res_F", C_res_F, n, dtype=dtype)
    C_couple = _broadcast_to_n("C_couple_F", C_couple_F, n, dtype=dtype)

    return LineLayout(
        length_m=length,
        L_series_H=L,
        C_shunt_F=C,
        R_series_ohm=R,
        G_shunt_S=G,
        C_stub_F=C_stub,
        L_res_H=L_res,
        C_res_F=C_res,
        C_couple_F=C_couple,
        z0_ohm=z0_ohm,
        name=name,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Periodic loading
# ---------------------------------------------------------------------------

def periodic_modulation_profile(
    x_m: ArrayLike,
    *,
    period_m: float,
    fraction: float,
    phase_rad: float = 0.0,
    waveform: Literal["sinusoidal", "square"] = "sinusoidal",
) -> jax.Array:
    """
    Return multiplicative profile 1 + fraction * waveform(x).

    The result is dimensionless. The returned profile is guaranteed positive
    if |fraction| < 1.
    """
    x = jnp.asarray(x_m)
    theta = 2.0 * jnp.pi * x / period_m + phase_rad

    if waveform == "sinusoidal":
        wave = jnp.sin(theta)
    elif waveform == "square":
        wave = jnp.where(jnp.sin(theta) >= 0.0, 1.0, -1.0)
    else:
        raise ValueError(f"Unsupported periodic waveform {waveform!r}")

    return 1.0 + fraction * wave


def apply_periodic_loading(
    layout: LineLayout,
    loading: PeriodicLoadingParams,
    *,
    apply_to_stub: bool = False,
) -> LineLayout:
    """
    Apply periodic L/C loading to an existing layout.

    Parameters
    ----------
    layout:
        Base line layout.
    loading:
        Periodic loading configuration.
    apply_to_stub:
        If false, capacitance modulation is applied directly to C_shunt_F.
        If true, it is applied as an additional C_stub_F contribution whose
        zero-mean baseline is separated from the original C_shunt_F.

    Returns
    -------
    LineLayout
        New layout with periodic loading applied.

    Notes
    -----
    For early validation, direct modulation of C_shunt_F is easiest. For later
    physical layouts with actual capacitive stubs, use explicit cell arrays or
    apply_to_stub=True.
    """
    if not loading.enabled:
        return layout.with_metadata(periodic_loading_enabled=False)

    x = layout.x_cell_center_m

    L_profile = periodic_modulation_profile(
        x,
        period_m=loading.period_m,
        fraction=loading.inductance_modulation_fraction,
        phase_rad=loading.phase_rad,
        waveform=loading.waveform,
    )
    C_profile = periodic_modulation_profile(
        x,
        period_m=loading.period_m,
        fraction=loading.capacitance_modulation_fraction,
        phase_rad=loading.phase_rad,
        waveform=loading.waveform,
    )

    if apply_to_stub:
        # Add only the variation around the base capacitance as a separate stub.
        # This keeps total C = base C + stub C. Since stub capacitance cannot be
        # negative, shift by the minimum if necessary.
        delta_C = layout.C_shunt_F * (C_profile - 1.0)
        min_delta = jnp.min(delta_C)
        shifted_delta_C = jnp.where(min_delta < 0.0, delta_C - min_delta, delta_C)
        new_C = layout.C_shunt_F
        new_stub = layout.C_stub_F + shifted_delta_C
        stub_shift = float(jnp.where(min_delta < 0.0, -min_delta, 0.0))
    else:
        new_C = layout.C_shunt_F * C_profile
        new_stub = layout.C_stub_F
        stub_shift = 0.0

    return layout.with_updates(
        L_series_H=layout.L_series_H * L_profile,
        C_shunt_F=new_C,
        C_stub_F=new_stub,
        metadata={
            **dict(layout.metadata or {}),
            "periodic_loading_enabled": True,
            "period_m": loading.period_m,
            "capacitance_modulation_fraction": loading.capacitance_modulation_fraction,
            "inductance_modulation_fraction": loading.inductance_modulation_fraction,
            "phase_rad": loading.phase_rad,
            "waveform": loading.waveform,
            "apply_to_stub": apply_to_stub,
            "stub_shift_F_if_any": stub_shift,
        },
    )


def add_sinusoidal_stub_loading(
    layout: LineLayout,
    *,
    average_stub_C_F: float,
    modulation_fraction: float,
    period_m: float,
    phase_rad: float = 0.0,
    waveform: Literal["sinusoidal", "square"] = "sinusoidal",
) -> LineLayout:
    """
    Add explicitly positive periodic stub capacitance.

    C_stub(x) = average_stub_C_F * [1 + modulation_fraction * waveform(x)]

    Requires |modulation_fraction| < 1.
    """
    if average_stub_C_F < 0.0:
        raise ValueError("average_stub_C_F must be non-negative")
    if abs(modulation_fraction) >= 1.0:
        raise ValueError("|modulation_fraction| must be < 1")

    profile = periodic_modulation_profile(
        layout.x_cell_center_m,
        period_m=period_m,
        fraction=modulation_fraction,
        phase_rad=phase_rad,
        waveform=waveform,
    )
    C_stub = layout.C_stub_F + average_stub_C_F * profile

    return layout.with_updates(
        C_stub_F=C_stub,
        metadata={
            **dict(layout.metadata or {}),
            "explicit_stub_loading": True,
            "average_stub_C_F": average_stub_C_F,
            "stub_modulation_fraction": modulation_fraction,
            "stub_period_m": period_m,
            "stub_phase_rad": phase_rad,
            "stub_waveform": waveform,
        },
    )


# ---------------------------------------------------------------------------
# Resonator loading
# ---------------------------------------------------------------------------

def apply_resonator_loading(
    layout: LineLayout,
    loading: ResonatorLoadingParams,
    *,
    offset: int = 0,
) -> LineLayout:
    """
    Mark cells with periodic shunt resonator loading.

    This does not yet define how resonators are converted into ABCD/MNA
    elements. It only places their parameters in vectorized arrays.

    Parameters
    ----------
    layout:
        Base layout.
    loading:
        Resonator loading configuration.
    offset:
        First loaded cell index modulo every_n_cells.
    """
    if not loading.enabled:
        return layout.with_metadata(resonator_loading_enabled=False)

    n = layout.n_cells
    idx = jnp.arange(n)
    mask = (idx - int(offset)) % int(loading.every_n_cells) == 0

    L_res = jnp.where(mask, loading.L_res_H, layout.L_res_H)
    C_res = jnp.where(mask, loading.C_res_F, layout.C_res_F)
    C_couple = jnp.where(mask, loading.C_couple_F, layout.C_couple_F)

    return layout.with_updates(
        L_res_H=L_res,
        C_res_F=C_res,
        C_couple_F=C_couple,
        metadata={
            **dict(layout.metadata or {}),
            "resonator_loading_enabled": True,
            "resonator_every_n_cells": loading.every_n_cells,
            "resonator_offset": int(offset),
            "C_couple_F": loading.C_couple_F,
            "L_res_H": loading.L_res_H,
            "C_res_F": loading.C_res_F,
            "loss_res_ohm": loading.loss_res_ohm,
            "resonance_hz": loading.resonance_hz,
        },
    )


# ---------------------------------------------------------------------------
# Complete device layout construction
# ---------------------------------------------------------------------------

def make_device_layout(
    device: DeviceParams,
    *,
    dtype: Any | None = None,
    apply_periodic_to_stub: bool = False,
) -> LineLayout:
    """
    Build a complete vectorized layout from DeviceParams.

    Order:
        1. uniform base line
        2. periodic L/C loading
        3. resonator loading markers
    """
    layout = make_uniform_layout(device.line, dtype=dtype)
    layout = apply_periodic_loading(
        layout,
        device.periodic_loading,
        apply_to_stub=apply_periodic_to_stub,
    )
    layout = apply_resonator_loading(layout, device.resonator_loading)

    return layout.with_metadata(
        source="make_device_layout",
        device_line_name=device.line.name,
        nonlinear_medium=device.nonlinear.medium.value,
        mixing_regime=device.mixing_regime.value,
        I_star_A=device.nonlinear.I_star_A,
        beta_nl=device.nonlinear.beta_nl,
    )


# ---------------------------------------------------------------------------
# Supercell utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupercellIndex:
    """
    Static description of a repeated supercell grouping.

    Parameters
    ----------
    cells_per_supercell:
        Number of cells in each full supercell.
    n_full_supercells:
        Number of complete supercells.
    remainder_cells:
        Number of trailing cells that do not form a complete supercell.
    """

    cells_per_supercell: int
    n_full_supercells: int
    remainder_cells: int

    @property
    def has_remainder(self) -> bool:
        return self.remainder_cells > 0

    @property
    def total_cells(self) -> int:
        return self.cells_per_supercell * self.n_full_supercells + self.remainder_cells

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "cells_per_supercell": self.cells_per_supercell,
            "n_full_supercells": self.n_full_supercells,
            "remainder_cells": self.remainder_cells,
            "has_remainder": self.has_remainder,
            "total_cells": self.total_cells,
        }


def make_supercell_index(n_cells: int, cells_per_supercell: int) -> SupercellIndex:
    """Return supercell grouping metadata."""
    if int(n_cells) <= 0:
        raise ValueError("n_cells must be positive")
    if int(cells_per_supercell) <= 0:
        raise ValueError("cells_per_supercell must be positive")
    n_cells = int(n_cells)
    cells_per_supercell = int(cells_per_supercell)
    return SupercellIndex(
        cells_per_supercell=cells_per_supercell,
        n_full_supercells=n_cells // cells_per_supercell,
        remainder_cells=n_cells % cells_per_supercell,
    )


def extract_supercell(
    layout: LineLayout,
    *,
    start_cell: int = 0,
    cells_per_supercell: int,
    name: str | None = None,
) -> LineLayout:
    """
    Extract one supercell as a smaller LineLayout.

    This is useful for validating periodic cell blocks before cascading them.
    """
    if start_cell < 0:
        raise ValueError("start_cell must be non-negative")
    if cells_per_supercell <= 0:
        raise ValueError("cells_per_supercell must be positive")
    stop = start_cell + cells_per_supercell
    if stop > layout.n_cells:
        raise ValueError(
            f"Requested cells [{start_cell}:{stop}] exceed layout length {layout.n_cells}"
        )

    sl = slice(start_cell, stop)
    return LineLayout(
        length_m=layout.length_m[sl],
        L_series_H=layout.L_series_H[sl],
        C_shunt_F=layout.C_shunt_F[sl],
        R_series_ohm=layout.R_series_ohm[sl],
        G_shunt_S=layout.G_shunt_S[sl],
        C_stub_F=layout.C_stub_F[sl],
        L_res_H=layout.L_res_H[sl],
        C_res_F=layout.C_res_F[sl],
        C_couple_F=layout.C_couple_F[sl],
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_supercell_{start_cell}_{stop}",
        metadata={
            **dict(layout.metadata or {}),
            "source": "extract_supercell",
            "parent_layout": layout.name,
            "start_cell": start_cell,
            "cells_per_supercell": cells_per_supercell,
        },
    )


def reshape_full_supercells(
    arr: ArrayLike,
    *,
    cells_per_supercell: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Reshape a 1D array into full supercells plus remainder.

    Returns
    -------
    full:
        Shape (n_full_supercells, cells_per_supercell).
    remainder:
        Shape (remainder_cells,).

    This helper is intentionally generic; higher-level cascade code can use it
    for chunked/supercell matrix products.
    """
    values = _as_1d_array("arr", arr)
    idx = make_supercell_index(values.shape[0], cells_per_supercell)
    n_full = idx.n_full_supercells * idx.cells_per_supercell
    full = values[:n_full].reshape((idx.n_full_supercells, idx.cells_per_supercell))
    remainder = values[n_full:]
    return full, remainder


# ---------------------------------------------------------------------------
# Coarsening / effective layouts
# ---------------------------------------------------------------------------

def coarsen_uniform_layout(
    layout: LineLayout,
    *,
    factor: int,
    method: Literal["sum_lc", "preserve_z0_vp"] = "sum_lc",
    name: str | None = None,
) -> LineLayout:
    """
    Coarsen a layout by grouping adjacent cells.

    This is for reduced/effective-cell studies, not a replacement for the full
    industrial layout.

    Parameters
    ----------
    factor:
        Number of fine cells per coarse cell.
    method:
        "sum_lc":
            Directly sum series/shunt element values in each group. This is
            the natural coarse representation for a cascade of small lumped
            sections when only a lower-resolution model is needed.
        "preserve_z0_vp":
            Sum length, then infer L and C from group-mean Z0 and vp. This
            is smoother for strongly modulated lines but can wash out bandgap
            physics. Use only for explicit reduced surrogates.

    Notes
    -----
    If the layout contains periodic loading, do not use arbitrary coarsening
    for production results. Coarsen by full supercell periods only.
    """
    if int(factor) <= 0:
        raise ValueError("factor must be positive")
    factor = int(factor)

    idx = make_supercell_index(layout.n_cells, factor)
    if idx.remainder_cells != 0:
        raise ValueError(
            "coarsen_uniform_layout currently requires n_cells divisible by factor; "
            f"got n_cells={layout.n_cells}, factor={factor}, remainder={idx.remainder_cells}"
        )

    def grouped_sum(x: jax.Array) -> jax.Array:
        return x.reshape((idx.n_full_supercells, factor)).sum(axis=1)

    def grouped_mean(x: jax.Array) -> jax.Array:
        return x.reshape((idx.n_full_supercells, factor)).mean(axis=1)

    length = grouped_sum(layout.length_m)
    R = grouped_sum(layout.R_series_ohm)
    G = grouped_sum(layout.G_shunt_S)
    C_stub = grouped_sum(layout.C_stub_F)

    if method == "sum_lc":
        L = grouped_sum(layout.L_series_H)
        C = grouped_sum(layout.C_shunt_F)
    elif method == "preserve_z0_vp":
        z = grouped_mean(layout.characteristic_impedance_cell_ohm)
        vp = grouped_mean(layout.phase_velocity_cell_m_per_s)
        L_per_m = z / vp
        C_per_m = 1.0 / (z * vp)
        L = L_per_m * length
        C = C_per_m * length
    else:
        raise ValueError(f"Unsupported coarsening method {method!r}")

    # Resonators are not safely coarsened by summing L values. Preserve only
    # explicit markers by summing capacitances and keeping harmonic-mean-ish
    # inductance impossible here. For now: disallow active resonators.
    if layout.has_resonators:
        raise ValueError(
            "Coarsening layouts with resonator loading is not implemented safely yet. "
            "Extract exact supercells instead."
        )

    zeros = jnp.zeros_like(length)

    return LineLayout(
        length_m=length,
        L_series_H=L,
        C_shunt_F=C,
        R_series_ohm=R,
        G_shunt_S=G,
        C_stub_F=C_stub,
        L_res_H=zeros,
        C_res_F=zeros,
        C_couple_F=zeros,
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_coarse_x{factor}",
        metadata={
            **dict(layout.metadata or {}),
            "source": "coarsen_uniform_layout",
            "parent_layout": layout.name,
            "coarsening_factor": factor,
            "coarsening_method": method,
            "warning": (
                "Reduced/effective layout. Validate dispersion and gain convergence "
                "against fine layout before using for conclusions."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Disorder / perturbation helpers
# ---------------------------------------------------------------------------

def apply_multiplicative_cell_perturbations(
    layout: LineLayout,
    *,
    L_factor: ArrayLike | float = 1.0,
    C_factor: ArrayLike | float = 1.0,
    R_factor: ArrayLike | float = 1.0,
    G_factor: ArrayLike | float = 1.0,
    C_stub_factor: ArrayLike | float = 1.0,
    name: str | None = None,
) -> LineLayout:
    """
    Apply explicit multiplicative perturbation factors to cell arrays.

    This is the deterministic hook used by later random-disorder code.
    """
    n = layout.n_cells
    Lf = _broadcast_to_n("L_factor", L_factor, n)
    Cf = _broadcast_to_n("C_factor", C_factor, n)
    Rf = _broadcast_to_n("R_factor", R_factor, n)
    Gf = _broadcast_to_n("G_factor", G_factor, n)
    Csf = _broadcast_to_n("C_stub_factor", C_stub_factor, n)

    if float(jnp.min(Lf)) <= 0.0:
        raise ValueError("L_factor must be positive everywhere")
    if float(jnp.min(Cf)) <= 0.0:
        raise ValueError("C_factor must be positive everywhere")
    if float(jnp.min(Rf)) < 0.0:
        raise ValueError("R_factor must be non-negative everywhere")
    if float(jnp.min(Gf)) < 0.0:
        raise ValueError("G_factor must be non-negative everywhere")
    if float(jnp.min(Csf)) < 0.0:
        raise ValueError("C_stub_factor must be non-negative everywhere")

    return layout.with_updates(
        L_series_H=layout.L_series_H * Lf,
        C_shunt_F=layout.C_shunt_F * Cf,
        R_series_ohm=layout.R_series_ohm * Rf,
        G_shunt_S=layout.G_shunt_S * Gf,
        C_stub_F=layout.C_stub_F * Csf,
        name=name or f"{layout.name}_perturbed",
        metadata={
            **dict(layout.metadata or {}),
            "source": "apply_multiplicative_cell_perturbations",
            "perturbed": True,
        },
    )


# ---------------------------------------------------------------------------
# Import/export helpers
# ---------------------------------------------------------------------------

def layout_to_npz_dict(layout: LineLayout) -> dict[str, Any]:
    """
    Convert layout to a dict suitable for numpy.savez.

    Metadata is deliberately omitted because npz is array-oriented. Save the
    summary separately as JSON in reporting code.
    """
    return {
        "length_m": layout.length_m,
        "L_series_H": layout.L_series_H,
        "C_shunt_F": layout.C_shunt_F,
        "R_series_ohm": layout.R_series_ohm,
        "G_shunt_S": layout.G_shunt_S,
        "C_stub_F": layout.C_stub_F,
        "L_res_H": layout.L_res_H,
        "C_res_F": layout.C_res_F,
        "C_couple_F": layout.C_couple_F,
        "z0_ohm": jnp.asarray(layout.z0_ohm),
    }


def layout_from_npz_arrays(
    arrays: Mapping[str, ArrayLike],
    *,
    name: str = "loaded_layout",
    metadata: Mapping[str, Any] | None = None,
) -> LineLayout:
    """
    Build a layout from arrays loaded from an NPZ-like mapping.
    """
    required = [
        "length_m",
        "L_series_H",
        "C_shunt_F",
        "R_series_ohm",
        "G_shunt_S",
        "C_stub_F",
        "L_res_H",
        "C_res_F",
        "C_couple_F",
    ]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Missing layout arrays: {missing}")

    z0 = float(jnp.asarray(arrays.get("z0_ohm", 50.0)))

    return LineLayout(
        length_m=jnp.asarray(arrays["length_m"]),
        L_series_H=jnp.asarray(arrays["L_series_H"]),
        C_shunt_F=jnp.asarray(arrays["C_shunt_F"]),
        R_series_ohm=jnp.asarray(arrays["R_series_ohm"]),
        G_shunt_S=jnp.asarray(arrays["G_shunt_S"]),
        C_stub_F=jnp.asarray(arrays["C_stub_F"]),
        L_res_H=jnp.asarray(arrays["L_res_H"]),
        C_res_F=jnp.asarray(arrays["C_res_F"]),
        C_couple_F=jnp.asarray(arrays["C_couple_F"]),
        z0_ohm=z0,
        name=name,
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_layout_for_frequency(
    layout: LineLayout,
    *,
    max_frequency_hz: float,
    cutoff_safety_factor: float = 3.0,
) -> dict[str, Any]:
    """
    Return a validation report checking the LC artificial cutoff guard.

    Parameters
    ----------
    layout:
        Layout to check.
    max_frequency_hz:
        Highest intended simulation frequency.
    cutoff_safety_factor:
        Require min artificial cutoff > cutoff_safety_factor * max_frequency_hz.

    Returns
    -------
    dict
        JSON-friendly report. This function does not raise on failure; scripts
        can decide whether to fail hard.
    """
    if max_frequency_hz <= 0.0:
        raise ValueError("max_frequency_hz must be positive")
    if cutoff_safety_factor <= 0.0:
        raise ValueError("cutoff_safety_factor must be positive")

    cutoff = layout.artificial_cutoff_hz_cell
    min_cutoff = float(jnp.min(cutoff))
    required = cutoff_safety_factor * max_frequency_hz
    passed = min_cutoff > required

    return {
        "layout_name": layout.name,
        "n_cells": layout.n_cells,
        "total_length_m": layout.total_length_m,
        "max_frequency_hz": float(max_frequency_hz),
        "cutoff_safety_factor": float(cutoff_safety_factor),
        "required_min_cutoff_hz": float(required),
        "actual_min_cutoff_hz": min_cutoff,
        "actual_max_cutoff_hz": float(jnp.max(cutoff)),
        "passed": bool(passed),
        "message": (
            "PASS: artificial LC cutoff guard satisfied."
            if passed
            else "FAIL: artificial LC cutoff too close to operating band."
        ),
    }


def compare_layouts_basic(a: LineLayout, b: LineLayout) -> dict[str, Any]:
    """
    Basic comparison report between two layouts.

    Useful for checking a reduced/effective layout against a fine layout.
    """
    return {
        "a_name": a.name,
        "b_name": b.name,
        "a_n_cells": a.n_cells,
        "b_n_cells": b.n_cells,
        "a_total_length_m": a.total_length_m,
        "b_total_length_m": b.total_length_m,
        "relative_length_error": float(
            abs(a.total_length_m - b.total_length_m) / max(abs(a.total_length_m), 1e-300)
        ),
        "a_mean_z_cell_ohm": float(jnp.mean(a.characteristic_impedance_cell_ohm)),
        "b_mean_z_cell_ohm": float(jnp.mean(b.characteristic_impedance_cell_ohm)),
        "a_mean_vp_m_per_s": float(jnp.mean(a.phase_velocity_cell_m_per_s)),
        "b_mean_vp_m_per_s": float(jnp.mean(b.phase_velocity_cell_m_per_s)),
        "a_total_L_H": float(jnp.sum(a.L_series_H)),
        "b_total_L_H": float(jnp.sum(b.L_series_H)),
        "a_total_C_F": float(jnp.sum(a.total_shunt_C_F)),
        "b_total_C_F": float(jnp.sum(b.total_shunt_C_F)),
    }


__all__ = [
    "LineLayout",
    "SupercellIndex",
    "make_uniform_layout",
    "make_layout_from_arrays",
    "make_device_layout",
    "periodic_modulation_profile",
    "apply_periodic_loading",
    "add_sinusoidal_stub_loading",
    "apply_resonator_loading",
    "make_supercell_index",
    "extract_supercell",
    "reshape_full_supercells",
    "coarsen_uniform_layout",
    "apply_multiplicative_cell_perturbations",
    "layout_to_npz_dict",
    "layout_from_npz_arrays",
    "validate_layout_for_frequency",
    "compare_layouts_basic",
]