"""
twpa.linear.cells
=================

Cell-level linear RF models for vectorized TWPA layouts.

This module converts LineLayout arrays into per-cell ABCD matrices.

The hierarchy is:

    twpa.core.layout.LineLayout
        -> twpa.linear.cells.cell_abcd(...)
        -> twpa.linear.cascade.cascade_layout_abcd(...)
        -> pump-off S-parameters / dispersion

Only local cell construction lives here. Long-line cascading, chunking, and
supercell repetition live in twpa.linear.cascade.

Supported first-pass cell models
--------------------------------
1. T-cell:
       series Z/2 -> shunt Y -> series Z/2

2. Pi-cell:
       shunt Y/2 -> series Z -> shunt Y/2

3. Loaded T-cell:
       same as T-cell, but C_total = C_shunt + C_stub

4. Resonator-loaded T-cell:
       T-cell with an additional shunt resonator admittance.
       This is a linear placeholder for later resonator/supercell models.

Shape conventions
-----------------
For one cell and F frequencies:

    abcd.shape = (F, 2, 2)

For N cells and F frequencies:

    abcd_cells.shape = (N, F, 2, 2)

This is intentionally cell-major because long-line cascade scans over cells.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout
from twpa.core.units import angular_frequency
from twpa.linear.rf_networks import (
    abcd_loaded_t_cell,
    abcd_pi_cell,
    abcd_shunt_admittance,
    abcd_shunt_parallel_lc,
    abcd_t_cell,
    cascade_abcd,
)


ArrayLike = Any


# ---------------------------------------------------------------------------
# Enums / config
# ---------------------------------------------------------------------------

class CellModelKind(str, Enum):
    """Supported cell-level linear circuit models."""

    T = "t"
    PI = "pi"
    LOADED_T = "loaded_t"
    RESONATOR_LOADED_T = "resonator_loaded_t"


@dataclass(frozen=True)
class CellModelConfig:
    """
    Configuration for converting layout cells to ABCD matrices.

    Parameters
    ----------
    kind:
        Cell model type.
    include_stub_capacitance:
        If true, C_total = C_shunt + C_stub.
    include_resonators:
        If true, cells with L_res/C_res/C_couple are given an additional
        shunt resonator admittance.
    resonator_mode:
        "parallel_lc":
            Use a simple shunt parallel LC admittance.
        "ignore_coupling_cap":
            Ignore C_couple_F and place L_res/C_res directly to ground.
        "series_coupled_parallel_lc":
            Approximate coupling capacitor in series with a parallel LC
            branch before shunting to ground. This is still a linear small
            model, not a full EM replacement.
    """

    kind: CellModelKind = CellModelKind.LOADED_T
    include_stub_capacitance: bool = True
    include_resonators: bool = False
    resonator_mode: Literal[
        "parallel_lc",
        "ignore_coupling_cap",
        "series_coupled_parallel_lc",
    ] = "parallel_lc"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CellModelKind(self.kind))
        if self.resonator_mode not in {
            "parallel_lc",
            "ignore_coupling_cap",
            "series_coupled_parallel_lc",
        }:
            raise ValueError(f"Unsupported resonator_mode {self.resonator_mode!r}")

    def with_updates(self, **kwargs: Any) -> "CellModelConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "include_stub_capacitance": self.include_stub_capacitance,
            "include_resonators": self.include_resonators,
            "resonator_mode": self.resonator_mode,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_frequency_array(frequency_hz: ArrayLike) -> jax.Array:
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if f.ndim == 0:
        f = f.reshape((1,))
    if f.ndim != 1:
        raise ValueError(f"frequency_hz must be scalar or 1D, got shape {f.shape}")
    if bool(jnp.any(f < 0.0)):
        raise ValueError("frequency_hz must be non-negative for linear RF cells")
    return f


def _as_cell_array(name: str, value: ArrayLike) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D cell array, got shape {arr.shape}")
    return arr


def _broadcast_cell_frequency(cell_values: jax.Array, frequency_hz: jax.Array) -> jax.Array:
    """
    Broadcast cell values shape (N,) and frequencies shape (F,) to (N, F).
    """
    return cell_values[:, None] * jnp.ones((1, frequency_hz.shape[0]), dtype=cell_values.dtype)


def _zeros_cell_frequency(n_cells: int, n_freq: int) -> jax.Array:
    return jnp.zeros((n_cells, n_freq), dtype=jnp.float64)


def _ones_cell_frequency(n_cells: int, n_freq: int) -> jax.Array:
    return jnp.ones((n_cells, n_freq), dtype=jnp.float64)


def _stack_abcd_from_entries(A: jax.Array, B: jax.Array, C: jax.Array, D: jax.Array) -> jax.Array:
    """
    Stack entries with shape (...,) into ABCD matrices (..., 2, 2).
    """
    return jnp.stack(
        [
            jnp.stack([A, B], axis=-1),
            jnp.stack([C, D], axis=-1),
        ],
        axis=-2,
    )


def _series_impedance_matrix_from_z(z: jax.Array) -> jax.Array:
    """
    ABCD series impedance for z with shape (...,).
    """
    zeros = jnp.zeros_like(z)
    ones = jnp.ones_like(z)
    return _stack_abcd_from_entries(ones, z, zeros, ones)


def _shunt_admittance_matrix_from_y(y: jax.Array) -> jax.Array:
    """
    ABCD shunt admittance for y with shape (...,).
    """
    zeros = jnp.zeros_like(y)
    ones = jnp.ones_like(y)
    return _stack_abcd_from_entries(ones, zeros, y, ones)


def _matmul_abcd(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.matmul(a, b)


# ---------------------------------------------------------------------------
# Cell admittance/impedance construction
# ---------------------------------------------------------------------------

def cell_series_impedance(
    frequency_hz: ArrayLike,
    *,
    L_series_H: ArrayLike,
    R_series_ohm: ArrayLike = 0.0,
) -> jax.Array:
    """
    Series impedance Z = R + j omega L for per-cell arrays.

    Parameters
    ----------
    frequency_hz:
        Shape (F,) or scalar.
    L_series_H:
        Shape (N,).
    R_series_ohm:
        Shape (N,) or scalar.

    Returns
    -------
    Z:
        Shape (N, F).
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    L = _as_cell_array("L_series_H", L_series_H)
    n = int(L.shape[0])

    R = jnp.asarray(R_series_ohm, dtype=jnp.float64)
    if R.ndim == 0:
        R = jnp.full((n,), R)
    elif R.ndim != 1 or R.shape[0] != n:
        raise ValueError(f"R_series_ohm must be scalar or shape ({n},), got {R.shape}")

    return R[:, None] + 1j * omega[None, :] * L[:, None]


def cell_shunt_admittance(
    frequency_hz: ArrayLike,
    *,
    C_shunt_F: ArrayLike,
    G_shunt_S: ArrayLike = 0.0,
    C_stub_F: ArrayLike = 0.0,
    include_stub_capacitance: bool = True,
) -> jax.Array:
    """
    Shunt admittance Y = G + j omega C_total for per-cell arrays.

    Returns shape (N, F).
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    C = _as_cell_array("C_shunt_F", C_shunt_F)
    n = int(C.shape[0])

    G = jnp.asarray(G_shunt_S, dtype=jnp.float64)
    if G.ndim == 0:
        G = jnp.full((n,), G)
    elif G.ndim != 1 or G.shape[0] != n:
        raise ValueError(f"G_shunt_S must be scalar or shape ({n},), got {G.shape}")

    C_stub = jnp.asarray(C_stub_F, dtype=jnp.float64)
    if C_stub.ndim == 0:
        C_stub = jnp.full((n,), C_stub)
    elif C_stub.ndim != 1 or C_stub.shape[0] != n:
        raise ValueError(f"C_stub_F must be scalar or shape ({n},), got {C_stub.shape}")

    C_total = C + C_stub if include_stub_capacitance else C
    return G[:, None] + 1j * omega[None, :] * C_total[:, None]


def cell_parallel_lc_resonator_admittance(
    frequency_hz: ArrayLike,
    *,
    L_res_H: ArrayLike,
    C_res_F: ArrayLike,
    G_res_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    Shunt parallel LC resonator admittance per cell.

        Y = G + j omega C + 1/(j omega L)

    Cells with L_res_H <= 0 or C_res_F <= 0 produce zero admittance.

    Returns shape (N, F).
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    if bool(jnp.any(f == 0.0)):
        raise ValueError("Parallel LC resonator admittance is singular at DC")

    L = _as_cell_array("L_res_H", L_res_H)
    C = _as_cell_array("C_res_F", C_res_F)
    if L.shape != C.shape:
        raise ValueError("L_res_H and C_res_F must have same shape")
    n = int(L.shape[0])

    G = jnp.asarray(G_res_S, dtype=jnp.float64)
    if G.ndim == 0:
        G = jnp.full((n,), G)
    elif G.ndim != 1 or G.shape[0] != n:
        raise ValueError(f"G_res_S must be scalar or shape ({n},), got {G.shape}")

    active = (L > 0.0) & (C > 0.0)
    y_raw = G[:, None] + 1j * omega[None, :] * C[:, None] + 1.0 / (
        1j * omega[None, :] * L[:, None]
    )
    return jnp.where(active[:, None], y_raw, 0.0 + 0.0j)


def cell_series_coupled_parallel_lc_admittance(
    frequency_hz: ArrayLike,
    *,
    C_couple_F: ArrayLike,
    L_res_H: ArrayLike,
    C_res_F: ArrayLike,
    G_res_S: ArrayLike = 0.0,
) -> jax.Array:
    """
    Approximate admittance of a coupling capacitor in series with a parallel LC.

    Circuit:
        node -> C_couple series -> parallel(L_res, C_res, G_res) -> ground

    Equivalent:
        Z_total = Z_couple + 1 / Y_parallel
        Y_total = 1 / Z_total

    Inactive cells where any of C_couple, L_res, C_res <= 0 produce zero.
    """
    f = _as_frequency_array(frequency_hz)
    omega = angular_frequency(f)

    if bool(jnp.any(f == 0.0)):
        raise ValueError("Series-coupled resonator is singular at DC")

    Cc = _as_cell_array("C_couple_F", C_couple_F)
    L = _as_cell_array("L_res_H", L_res_H)
    C = _as_cell_array("C_res_F", C_res_F)

    if not (Cc.shape == L.shape == C.shape):
        raise ValueError("C_couple_F, L_res_H, and C_res_F must have same shape")

    n = int(L.shape[0])

    G = jnp.asarray(G_res_S, dtype=jnp.float64)
    if G.ndim == 0:
        G = jnp.full((n,), G)
    elif G.ndim != 1 or G.shape[0] != n:
        raise ValueError(f"G_res_S must be scalar or shape ({n},), got {G.shape}")

    active = (Cc > 0.0) & (L > 0.0) & (C > 0.0)

    y_parallel = G[:, None] + 1j * omega[None, :] * C[:, None] + 1.0 / (
        1j * omega[None, :] * L[:, None]
    )
    z_parallel = 1.0 / y_parallel
    z_couple = 1.0 / (1j * omega[None, :] * Cc[:, None])
    y_total = 1.0 / (z_couple + z_parallel)

    return jnp.where(active[:, None], y_total, 0.0 + 0.0j)


def cell_resonator_admittance(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    mode: Literal[
        "parallel_lc",
        "ignore_coupling_cap",
        "series_coupled_parallel_lc",
    ] = "parallel_lc",
) -> jax.Array:
    """
    Resonator-loading admittance for a layout.

    Returns shape (N, F).
    """
    if mode == "parallel_lc" or mode == "ignore_coupling_cap":
        return cell_parallel_lc_resonator_admittance(
            frequency_hz,
            L_res_H=layout.L_res_H,
            C_res_F=layout.C_res_F,
            G_res_S=0.0,
        )
    if mode == "series_coupled_parallel_lc":
        return cell_series_coupled_parallel_lc_admittance(
            frequency_hz,
            C_couple_F=layout.C_couple_F,
            L_res_H=layout.L_res_H,
            C_res_F=layout.C_res_F,
            G_res_S=0.0,
        )
    raise ValueError(f"Unsupported resonator mode {mode!r}")


# ---------------------------------------------------------------------------
# Vectorized cell ABCD construction
# ---------------------------------------------------------------------------

def t_cell_abcd_from_zy(
    z_series: ArrayLike,
    y_shunt: ArrayLike,
) -> jax.Array:
    """
    Build T-cell ABCD matrices from Z and Y arrays.

    Parameters
    ----------
    z_series:
        Shape (...,).
    y_shunt:
        Shape (...,).

    Returns
    -------
    abcd:
        Shape (..., 2, 2).
    """
    z = jnp.asarray(z_series, dtype=jnp.complex128)
    y = jnp.asarray(y_shunt, dtype=jnp.complex128)
    if z.shape != y.shape:
        raise ValueError(f"z_series and y_shunt must have same shape, got {z.shape}, {y.shape}")

    left = _series_impedance_matrix_from_z(0.5 * z)
    shunt = _shunt_admittance_matrix_from_y(y)
    right = _series_impedance_matrix_from_z(0.5 * z)
    return _matmul_abcd(_matmul_abcd(left, shunt), right)


def pi_cell_abcd_from_zy(
    z_series: ArrayLike,
    y_shunt: ArrayLike,
) -> jax.Array:
    """
    Build pi-cell ABCD matrices from Z and Y arrays.

    Parameters
    ----------
    z_series:
        Shape (...,).
    y_shunt:
        Shape (...,).

    Returns
    -------
    abcd:
        Shape (..., 2, 2).
    """
    z = jnp.asarray(z_series, dtype=jnp.complex128)
    y = jnp.asarray(y_shunt, dtype=jnp.complex128)
    if z.shape != y.shape:
        raise ValueError(f"z_series and y_shunt must have same shape, got {z.shape}, {y.shape}")

    left = _shunt_admittance_matrix_from_y(0.5 * y)
    series = _series_impedance_matrix_from_z(z)
    right = _shunt_admittance_matrix_from_y(0.5 * y)
    return _matmul_abcd(_matmul_abcd(left, series), right)


def layout_cell_abcd(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    config: CellModelConfig | None = None,
) -> jax.Array:
    """
    Build per-cell ABCD matrices for a complete layout.

    Parameters
    ----------
    frequency_hz:
        Shape (F,) or scalar.
    layout:
        Vectorized LineLayout with N cells.
    config:
        Cell model configuration.

    Returns
    -------
    abcd_cells:
        Shape (N, F, 2, 2).
    """
    cfg = config or CellModelConfig()
    f = _as_frequency_array(frequency_hz)

    z = cell_series_impedance(
        f,
        L_series_H=layout.L_series_H,
        R_series_ohm=layout.R_series_ohm,
    )
    y = cell_shunt_admittance(
        f,
        C_shunt_F=layout.C_shunt_F,
        G_shunt_S=layout.G_shunt_S,
        C_stub_F=layout.C_stub_F,
        include_stub_capacitance=cfg.include_stub_capacitance,
    )

    if cfg.include_resonators:
        y_res = cell_resonator_admittance(
            f,
            layout,
            mode=cfg.resonator_mode,
        )
        y = y + y_res

    if cfg.kind == CellModelKind.T:
        return t_cell_abcd_from_zy(z, y)

    if cfg.kind == CellModelKind.PI:
        return pi_cell_abcd_from_zy(z, y)

    if cfg.kind == CellModelKind.LOADED_T:
        return t_cell_abcd_from_zy(z, y)

    if cfg.kind == CellModelKind.RESONATOR_LOADED_T:
        # Force resonators on for this named kind.
        if not cfg.include_resonators:
            cfg = cfg.with_updates(include_resonators=True)
            return layout_cell_abcd(f, layout, config=cfg)
        return t_cell_abcd_from_zy(z, y)

    raise ValueError(f"Unsupported cell model kind {cfg.kind}")


def single_cell_abcd(
    frequency_hz: ArrayLike,
    *,
    L_series_H: float,
    C_shunt_F: float,
    R_series_ohm: float = 0.0,
    G_shunt_S: float = 0.0,
    C_stub_F: float = 0.0,
    kind: CellModelKind | str = CellModelKind.LOADED_T,
) -> jax.Array:
    """
    Convenience function for one cell.

    Returns shape (F, 2, 2).
    """
    f = _as_frequency_array(frequency_hz)
    kind = CellModelKind(kind)

    C_total = C_shunt_F + C_stub_F

    if kind == CellModelKind.PI:
        return abcd_pi_cell(
            f,
            L_series_H=L_series_H,
            C_shunt_F=C_total,
            R_series_ohm=R_series_ohm,
            G_shunt_S=G_shunt_S,
        )

    return abcd_t_cell(
        f,
        L_series_H=L_series_H,
        C_shunt_F=C_total,
        R_series_ohm=R_series_ohm,
        G_shunt_S=G_shunt_S,
    )


# ---------------------------------------------------------------------------
# Cell-level derived quantities
# ---------------------------------------------------------------------------

def cell_lc_cutoff_hz(
    *,
    L_series_H: ArrayLike,
    C_shunt_F: ArrayLike,
) -> jax.Array:
    """
    Approximate artificial LC-ladder cutoff frequency.

        omega_c ≈ 2 / sqrt(L C)
        f_c = omega_c / (2 pi)

    Returns array broadcast from L and C.
    """
    L = jnp.asarray(L_series_H, dtype=jnp.float64)
    C = jnp.asarray(C_shunt_F, dtype=jnp.float64)
    return (2.0 / jnp.sqrt(L * C)) / (2.0 * jnp.pi)


def cell_characteristic_impedance_ohm(
    *,
    L_series_H: ArrayLike,
    C_shunt_F: ArrayLike,
) -> jax.Array:
    """
    Approximate cell impedance sqrt(L/C).
    """
    L = jnp.asarray(L_series_H, dtype=jnp.float64)
    C = jnp.asarray(C_shunt_F, dtype=jnp.float64)
    return jnp.sqrt(L / C)


def cell_phase_velocity_m_per_s(
    *,
    length_m: ArrayLike,
    L_series_H: ArrayLike,
    C_shunt_F: ArrayLike,
) -> jax.Array:
    """
    Approximate local phase velocity:

        v ≈ dx / sqrt(L_cell C_cell)
    """
    dx = jnp.asarray(length_m, dtype=jnp.float64)
    L = jnp.asarray(L_series_H, dtype=jnp.float64)
    C = jnp.asarray(C_shunt_F, dtype=jnp.float64)
    return dx / jnp.sqrt(L * C)


def layout_cell_parameter_summary(layout: LineLayout) -> dict[str, Any]:
    """
    Summary of local cell parameters.
    """
    C_total = layout.total_shunt_C_F
    z = cell_characteristic_impedance_ohm(
        L_series_H=layout.L_series_H,
        C_shunt_F=C_total,
    )
    vp = cell_phase_velocity_m_per_s(
        length_m=layout.length_m,
        L_series_H=layout.L_series_H,
        C_shunt_F=C_total,
    )
    fc = cell_lc_cutoff_hz(
        L_series_H=layout.L_series_H,
        C_shunt_F=C_total,
    )

    return {
        "layout_name": layout.name,
        "n_cells": layout.n_cells,
        "length_m_total": layout.total_length_m,
        "L_series_H_min": float(jnp.min(layout.L_series_H)),
        "L_series_H_max": float(jnp.max(layout.L_series_H)),
        "C_total_F_min": float(jnp.min(C_total)),
        "C_total_F_max": float(jnp.max(C_total)),
        "Z_cell_ohm_min": float(jnp.min(z)),
        "Z_cell_ohm_max": float(jnp.max(z)),
        "Z_cell_ohm_mean": float(jnp.mean(z)),
        "vp_cell_m_per_s_min": float(jnp.min(vp)),
        "vp_cell_m_per_s_max": float(jnp.max(vp)),
        "vp_cell_m_per_s_mean": float(jnp.mean(vp)),
        "lc_cutoff_hz_min": float(jnp.min(fc)),
        "lc_cutoff_hz_max": float(jnp.max(fc)),
        "has_stub_loading": layout.has_stub_loading,
        "has_resonators": layout.has_resonators,
    }


# ---------------------------------------------------------------------------
# Unit-cell validation helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellValidationReport:
    """
    Cell-level validation report.
    """

    layout_name: str
    n_cells: int
    n_frequencies: int
    cell_model: Mapping[str, Any]
    cutoff_guard_passed: bool
    min_cutoff_hz: float
    max_frequency_hz: float
    cutoff_safety_factor: float
    max_abs_abcd_entry: float
    has_nan: bool
    has_inf: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "n_cells": self.n_cells,
            "n_frequencies": self.n_frequencies,
            "cell_model": dict(self.cell_model),
            "cutoff_guard_passed": self.cutoff_guard_passed,
            "min_cutoff_hz": self.min_cutoff_hz,
            "max_frequency_hz": self.max_frequency_hz,
            "cutoff_safety_factor": self.cutoff_safety_factor,
            "max_abs_abcd_entry": self.max_abs_abcd_entry,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "message": self.message,
        }


def validate_layout_cells(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    config: CellModelConfig | None = None,
    cutoff_safety_factor: float = 3.0,
) -> CellValidationReport:
    """
    Build cell ABCD matrices and validate basic numerical/physical guards.

    This does not cascade the full line.
    """
    cfg = config or CellModelConfig()
    f = _as_frequency_array(frequency_hz)

    cells = layout_cell_abcd(f, layout, config=cfg)
    max_abs = jnp.max(jnp.abs(cells))
    has_nan = bool(jnp.any(jnp.isnan(jnp.real(cells))) or jnp.any(jnp.isnan(jnp.imag(cells))))
    has_inf = bool(jnp.any(jnp.isinf(jnp.real(cells))) or jnp.any(jnp.isinf(jnp.imag(cells))))

    cutoff = cell_lc_cutoff_hz(
        L_series_H=layout.L_series_H,
        C_shunt_F=layout.total_shunt_C_F,
    )
    min_cutoff = float(jnp.min(cutoff))
    max_freq = float(jnp.max(f))
    required = cutoff_safety_factor * max_freq
    cutoff_passed = bool(min_cutoff > required)

    if has_nan or has_inf:
        message = "FAIL: cell ABCD contains NaN or Inf."
    elif not cutoff_passed:
        message = "FAIL: artificial LC cutoff too close to requested frequency range."
    else:
        message = "PASS: cell ABCD construction checks passed."

    return CellValidationReport(
        layout_name=layout.name,
        n_cells=layout.n_cells,
        n_frequencies=int(f.shape[0]),
        cell_model=cfg.to_dict(),
        cutoff_guard_passed=cutoff_passed,
        min_cutoff_hz=min_cutoff,
        max_frequency_hz=max_freq,
        cutoff_safety_factor=float(cutoff_safety_factor),
        max_abs_abcd_entry=float(max_abs),
        has_nan=has_nan,
        has_inf=has_inf,
        message=message,
    )


# ---------------------------------------------------------------------------
# Small analytic tests
# ---------------------------------------------------------------------------

def compare_t_and_pi_cell_low_frequency(
    frequency_hz: ArrayLike,
    *,
    L_series_H: float,
    C_shunt_F: float,
    z0_ohm: float = 50.0,
) -> dict[str, Any]:
    """
    Compare T-cell and pi-cell ABCD matrices at low frequencies.

    They are not exactly identical, but should approach the same distributed
    line behavior as cell electrical length becomes small.
    """
    f = _as_frequency_array(frequency_hz)

    t = single_cell_abcd(
        f,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
        kind=CellModelKind.T,
    )
    p = single_cell_abcd(
        f,
        L_series_H=L_series_H,
        C_shunt_F=C_shunt_F,
        kind=CellModelKind.PI,
    )

    diff = t - p
    return {
        "max_abs_abcd_diff": float(jnp.max(jnp.abs(diff))),
        "rms_abs_abcd_diff": float(jnp.sqrt(jnp.mean(jnp.abs(diff) ** 2))),
        "max_frequency_hz": float(jnp.max(f)),
        "cell_cutoff_hz": float(
            cell_lc_cutoff_hz(L_series_H=L_series_H, C_shunt_F=C_shunt_F)
        ),
        "z0_ohm": float(z0_ohm),
    }


__all__ = [
    "CellModelKind",
    "CellModelConfig",
    "cell_series_impedance",
    "cell_shunt_admittance",
    "cell_parallel_lc_resonator_admittance",
    "cell_series_coupled_parallel_lc_admittance",
    "cell_resonator_admittance",
    "t_cell_abcd_from_zy",
    "pi_cell_abcd_from_zy",
    "layout_cell_abcd",
    "single_cell_abcd",
    "cell_lc_cutoff_hz",
    "cell_characteristic_impedance_ohm",
    "cell_phase_velocity_m_per_s",
    "layout_cell_parameter_summary",
    "CellValidationReport",
    "validate_layout_cells",
    "compare_t_and_pi_cell_low_frequency",
]