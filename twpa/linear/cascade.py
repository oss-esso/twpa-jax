"""
twpa.linear.cascade
===================

Long-line ABCD cascading utilities for vectorized TWPA layouts.

This module turns per-cell ABCD matrices into complete two-port networks:

    LineLayout
        -> layout_cell_abcd(...)
        -> cascade_layout_abcd(...)
        -> S-parameters, S21, diagnostics, dispersion inputs

Why this module exists
----------------------
A 100 mm / 20,000-cell line cannot be handled casually in notebooks. The
cascade layer must expose several strategies:

1. direct_scan
   Simple JAX lax.scan over cells. Good for correctness and moderate N.

2. chunked_scan
   Cascade cells in chunks, then cascade chunk results. Useful for memory and
   diagnostics.

3. repeated_cell_power
   Fast path for uniform cells.

4. repeated_supercell_power
   Fast path for periodic supercells.

The first industrial validation target is pump-off S21/dispersion. Nonlinear
HB should only be trusted once this layer is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from twpa.core.layout import (
    LineLayout,
    SupercellIndex,
    extract_supercell,
    make_supercell_index,
)
from twpa.linear.cells import CellModelConfig, layout_cell_abcd
from twpa.linear.rf_networks import (
    NetworkDiagnostics,
    abcd_determinant,
    abcd_identity,
    abcd_power,
    abcd_to_s,
    cascade_scan_abcd,
    compare_sparameters,
    diagnose_two_port,
    effective_beta_from_s21,
    group_delay_from_s21,
    ideal_lossless_line_abcd,
    lossy_line_abcd_rlgc,
    s21,
    s_to_db,
)


ArrayLike = Any


# ---------------------------------------------------------------------------
# Config / result objects
# ---------------------------------------------------------------------------

class CascadeStrategy(str, Enum):
    """Supported long-line cascade strategies."""

    DIRECT_SCAN = "direct_scan"
    CHUNKED_SCAN = "chunked_scan"
    REPEATED_CELL_POWER = "repeated_cell_power"
    REPEATED_SUPERCELL_POWER = "repeated_supercell_power"
    AUTO = "auto"


@dataclass(frozen=True)
class CascadeConfig:
    """
    Configuration for cascading a layout.

    Parameters
    ----------
    strategy:
        Cascade strategy.
    chunk_size:
        Number of cells per chunk for chunked_scan.
    cells_per_supercell:
        Supercell size for repeated_supercell_power.
    validate_uniform_atol:
        Absolute tolerance used when checking whether a layout is uniform.
    validate_periodic_atol:
        Absolute tolerance used when checking whether a layout is periodic.
    allow_remainder:
        Whether repeated_supercell_power may cascade a trailing non-full
        supercell remainder.
    """

    strategy: CascadeStrategy = CascadeStrategy.AUTO
    chunk_size: int = 512
    cells_per_supercell: int = 1
    validate_uniform_atol: float = 0.0
    validate_periodic_atol: float = 0.0
    allow_remainder: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", CascadeStrategy(self.strategy))
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if int(self.cells_per_supercell) <= 0:
            raise ValueError("cells_per_supercell must be positive")
        if self.validate_uniform_atol < 0.0:
            raise ValueError("validate_uniform_atol must be non-negative")
        if self.validate_periodic_atol < 0.0:
            raise ValueError("validate_periodic_atol must be non-negative")
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        object.__setattr__(self, "cells_per_supercell", int(self.cells_per_supercell))

    def with_updates(self, **kwargs: Any) -> "CascadeConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "chunk_size": self.chunk_size,
            "cells_per_supercell": self.cells_per_supercell,
            "validate_uniform_atol": self.validate_uniform_atol,
            "validate_periodic_atol": self.validate_periodic_atol,
            "allow_remainder": self.allow_remainder,
        }


@dataclass(frozen=True)
class CascadeResult:
    """
    Result of a layout cascade.

    Attributes
    ----------
    abcd:
        Full line ABCD matrix, shape (F, 2, 2).
    frequency_hz:
        Frequency grid, shape (F,).
    layout_name:
        Name of the layout.
    strategy:
        Strategy actually used.
    cell_model:
        Cell model config.
    metadata:
        Extra report metadata.
    """

    abcd: jax.Array
    frequency_hz: jax.Array
    layout_name: str
    strategy: CascadeStrategy
    cell_model: CellModelConfig
    metadata: Mapping[str, Any] | None = None

    @property
    def s_parameters(self) -> jax.Array:
        z0 = float(dict(self.metadata or {}).get("z0_ohm", 50.0))
        return abcd_to_s(self.abcd, z0_ohm=z0)

    @property
    def s21(self) -> jax.Array:
        return s21(self.s_parameters)

    @property
    def s21_db(self) -> jax.Array:
        return s_to_db(self.s21)

    def diagnostics(self, *, z0_ohm: float | None = None) -> NetworkDiagnostics:
        z0 = z0_ohm
        if z0 is None:
            z0 = float(dict(self.metadata or {}).get("z0_ohm", 50.0))
        return diagnose_two_port(
            self.abcd,
            z0_ohm=z0,
            metadata={
                "layout_name": self.layout_name,
                "strategy": self.strategy.value,
                "cell_model": self.cell_model.to_dict(),
                **dict(self.metadata or {}),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "strategy": self.strategy.value,
            "frequency_shape": tuple(int(v) for v in self.frequency_hz.shape),
            "abcd_shape": tuple(int(v) for v in self.abcd.shape),
            "cell_model": self.cell_model.to_dict(),
            "metadata": dict(self.metadata or {}),
            "diagnostics": self.diagnostics().to_dict(),
        }


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _as_frequency_array(frequency_hz: ArrayLike) -> jax.Array:
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if f.ndim == 0:
        f = f.reshape((1,))
    if f.ndim != 1:
        raise ValueError(f"frequency_hz must be scalar or 1D, got shape {f.shape}")
    if bool(jnp.any(f < 0.0)):
        raise ValueError("frequency_hz must be non-negative for linear cascades")
    return f


def _check_cell_abcd_shape(cells: ArrayLike) -> jax.Array:
    m = jnp.asarray(cells)
    if not jnp.issubdtype(m.dtype, jnp.complexfloating):
        m = m.astype(jnp.complex128)
    if m.ndim != 4 or m.shape[-2:] != (2, 2):
        raise ValueError(f"cell ABCD array must have shape (N, F, 2, 2), got {m.shape}")
    return m


def _check_line_abcd_shape(abcd: ArrayLike) -> jax.Array:
    m = jnp.asarray(abcd)
    if not jnp.issubdtype(m.dtype, jnp.complexfloating):
        m = m.astype(jnp.complex128)
    if m.ndim != 3 or m.shape[-2:] != (2, 2):
        raise ValueError(f"line ABCD array must have shape (F, 2, 2), got {m.shape}")
    return m


def _identity_for_frequency(frequency_hz: jax.Array, dtype: Any = jnp.complex128) -> jax.Array:
    return abcd_identity(frequency_hz, dtype=dtype)


# ---------------------------------------------------------------------------
# Uniform / periodic layout checks
# ---------------------------------------------------------------------------

def layout_is_uniform(layout: LineLayout, *, atol: float = 0.0) -> bool:
    """
    Check whether all cell arrays are uniform.

    This checks the arrays that affect the linear cell model.
    """
    arrays = [
        layout.length_m,
        layout.L_series_H,
        layout.C_shunt_F,
        layout.R_series_ohm,
        layout.G_shunt_S,
        layout.C_stub_F,
        layout.L_res_H,
        layout.C_res_F,
        layout.C_couple_F,
    ]

    for arr in arrays:
        if arr.shape[0] <= 1:
            continue
        if not bool(jnp.all(jnp.abs(arr - arr[0]) <= atol)):
            return False
    return True


def layout_is_periodic(
    layout: LineLayout,
    *,
    cells_per_supercell: int,
    atol: float = 0.0,
    ignore_remainder: bool = True,
) -> bool:
    """
    Check whether layout repeats every cells_per_supercell cells.

    If ignore_remainder is true, trailing incomplete supercells are ignored.
    """
    idx = make_supercell_index(layout.n_cells, cells_per_supercell)
    if idx.n_full_supercells <= 1:
        return True
    if idx.remainder_cells and not ignore_remainder:
        return False

    n_full_cells = idx.n_full_supercells * idx.cells_per_supercell

    arrays = [
        layout.length_m[:n_full_cells],
        layout.L_series_H[:n_full_cells],
        layout.C_shunt_F[:n_full_cells],
        layout.R_series_ohm[:n_full_cells],
        layout.G_shunt_S[:n_full_cells],
        layout.C_stub_F[:n_full_cells],
        layout.L_res_H[:n_full_cells],
        layout.C_res_F[:n_full_cells],
        layout.C_couple_F[:n_full_cells],
    ]

    for arr in arrays:
        blocks = arr.reshape((idx.n_full_supercells, idx.cells_per_supercell))
        ref = blocks[0]
        if not bool(jnp.all(jnp.abs(blocks - ref[None, :]) <= atol)):
            return False
    return True


def choose_cascade_strategy(layout: LineLayout, config: CascadeConfig) -> CascadeStrategy:
    """
    Choose a concrete strategy from CascadeStrategy.AUTO.
    """
    if config.strategy != CascadeStrategy.AUTO:
        return config.strategy

    if layout_is_uniform(layout, atol=config.validate_uniform_atol):
        return CascadeStrategy.REPEATED_CELL_POWER

    if (
        config.cells_per_supercell > 1
        and layout_is_periodic(
            layout,
            cells_per_supercell=config.cells_per_supercell,
            atol=config.validate_periodic_atol,
            ignore_remainder=config.allow_remainder,
        )
    ):
        return CascadeStrategy.REPEATED_SUPERCELL_POWER

    if layout.n_cells > config.chunk_size:
        return CascadeStrategy.CHUNKED_SCAN

    return CascadeStrategy.DIRECT_SCAN


# ---------------------------------------------------------------------------
# Core cascade routines
# ---------------------------------------------------------------------------

def cascade_cell_abcd_direct(cell_abcd: ArrayLike) -> jax.Array:
    """
    Direct scan cascade over cells.

    Parameters
    ----------
    cell_abcd:
        Shape (N, F, 2, 2).

    Returns
    -------
    abcd:
        Shape (F, 2, 2).
    """
    cells = _check_cell_abcd_shape(cell_abcd)
    return cascade_scan_abcd(cells)


def cascade_cell_abcd_chunked(
    cell_abcd: ArrayLike,
    *,
    chunk_size: int,
) -> jax.Array:
    """
    Chunked cascade over cells.

    The algorithm:
        1. Split cells into chunks.
        2. Cascade each chunk.
        3. Cascade chunk matrices.

    This keeps intermediate diagnostics manageable and is often a good default
    for thousands of cells.
    """
    cells = _check_cell_abcd_shape(cell_abcd)
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_size = int(chunk_size)

    n_cells, n_freq = int(cells.shape[0]), int(cells.shape[1])
    if n_cells == 0:
        raise ValueError("Cannot cascade zero cells")

    chunk_results = []
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        chunk_results.append(cascade_cell_abcd_direct(cells[start:stop]))

    chunks = jnp.stack(chunk_results, axis=0)
    return cascade_cell_abcd_direct(chunks)


def cascade_repeated_cell_power(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    validate_uniform: bool = True,
    atol: float = 0.0,
) -> jax.Array:
    """
    Fast cascade for a uniform layout:

        M_total = M_cell ** N
    """
    f = _as_frequency_array(frequency_hz)
    cfg = cell_model or CellModelConfig()

    if validate_uniform and not layout_is_uniform(layout, atol=atol):
        raise ValueError("layout is not uniform; repeated_cell_power is invalid")

    first = extract_first_cell_layout(layout)
    cell = layout_cell_abcd(f, first, config=cfg)
    cell = cell[0]  # (F, 2, 2)
    return abcd_power(cell, layout.n_cells)


def cascade_repeated_supercell_power(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cells_per_supercell: int,
    cell_model: CellModelConfig | None = None,
    validate_periodic: bool = True,
    atol: float = 0.0,
    allow_remainder: bool = True,
) -> jax.Array:
    """
    Fast cascade for repeated supercells.

    The full line is interpreted as:

        M_total = M_supercell ** n_full @ M_remainder

    if a remainder exists and allow_remainder is true.
    """
    f = _as_frequency_array(frequency_hz)
    cfg = cell_model or CellModelConfig()
    idx = make_supercell_index(layout.n_cells, cells_per_supercell)

    if validate_periodic and not layout_is_periodic(
        layout,
        cells_per_supercell=cells_per_supercell,
        atol=atol,
        ignore_remainder=allow_remainder,
    ):
        raise ValueError("layout is not periodic with requested supercell size")

    supercell = extract_supercell(
        layout,
        start_cell=0,
        cells_per_supercell=cells_per_supercell,
        name=f"{layout.name}_first_supercell",
    )
    supercell_cells = layout_cell_abcd(f, supercell, config=cfg)
    supercell_abcd = cascade_cell_abcd_direct(supercell_cells)
    total = abcd_power(supercell_abcd, idx.n_full_supercells)

    if idx.remainder_cells:
        if not allow_remainder:
            raise ValueError(
                f"Layout has {idx.remainder_cells} remainder cells but allow_remainder=False"
            )
        remainder = extract_supercell(
            layout,
            start_cell=idx.n_full_supercells * idx.cells_per_supercell,
            cells_per_supercell=idx.remainder_cells,
            name=f"{layout.name}_remainder",
        )
        remainder_cells = layout_cell_abcd(f, remainder, config=cfg)
        remainder_abcd = cascade_cell_abcd_direct(remainder_cells)
        total = jnp.matmul(total, remainder_abcd)

    return total


def extract_first_cell_layout(layout: LineLayout) -> LineLayout:
    """
    Return layout consisting only of the first cell.
    """
    return extract_supercell(layout, start_cell=0, cells_per_supercell=1, name=f"{layout.name}_first_cell")


def cascade_layout_abcd(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
) -> CascadeResult:
    """
    Cascade a complete layout into a full ABCD network.

    Returns a CascadeResult with metadata and diagnostics helpers.
    """
    f = _as_frequency_array(frequency_hz)
    cell_cfg = cell_model or CellModelConfig()
    cas_cfg = cascade_config or CascadeConfig()

    strategy = choose_cascade_strategy(layout, cas_cfg)

    if strategy == CascadeStrategy.REPEATED_CELL_POWER:
        abcd = cascade_repeated_cell_power(
            f,
            layout,
            cell_model=cell_cfg,
            validate_uniform=True,
            atol=cas_cfg.validate_uniform_atol,
        )

    elif strategy == CascadeStrategy.REPEATED_SUPERCELL_POWER:
        abcd = cascade_repeated_supercell_power(
            f,
            layout,
            cells_per_supercell=cas_cfg.cells_per_supercell,
            cell_model=cell_cfg,
            validate_periodic=True,
            atol=cas_cfg.validate_periodic_atol,
            allow_remainder=cas_cfg.allow_remainder,
        )

    elif strategy == CascadeStrategy.DIRECT_SCAN:
        cells = layout_cell_abcd(f, layout, config=cell_cfg)
        abcd = cascade_cell_abcd_direct(cells)

    elif strategy == CascadeStrategy.CHUNKED_SCAN:
        cells = layout_cell_abcd(f, layout, config=cell_cfg)
        abcd = cascade_cell_abcd_chunked(cells, chunk_size=cas_cfg.chunk_size)

    else:
        raise ValueError(f"Unsupported cascade strategy {strategy}")

    return CascadeResult(
        abcd=_check_line_abcd_shape(abcd),
        frequency_hz=f,
        layout_name=layout.name,
        strategy=strategy,
        cell_model=cell_cfg,
        metadata={
            "z0_ohm": layout.z0_ohm,
            "n_cells": layout.n_cells,
            "total_length_m": layout.total_length_m,
            "cascade_config": cas_cfg.to_dict(),
            "layout_summary": layout.summary(),
        },
    )


def layout_sparameters(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    z0_ohm: float | None = None,
) -> jax.Array:
    """
    Convenience function: layout -> S-parameters.

    Returns shape (F, 2, 2).
    """
    result = cascade_layout_abcd(
        frequency_hz,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
    )
    z0 = layout.z0_ohm if z0_ohm is None else z0_ohm
    return abcd_to_s(result.abcd, z0_ohm=z0)


def layout_s21(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    z0_ohm: float | None = None,
) -> jax.Array:
    """
    Convenience function: layout -> S21.
    """
    return s21(
        layout_sparameters(
            frequency_hz,
            layout,
            cell_model=cell_model,
            cascade_config=cascade_config,
            z0_ohm=z0_ohm,
        )
    )


# ---------------------------------------------------------------------------
# Chunk diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkCascadeReport:
    """
    Diagnostics for chunked cascades.
    """

    n_cells: int
    n_chunks: int
    chunk_size: int
    frequency_shape: tuple[int, ...]
    chunk_det_error_max: float
    final_det_error_max: float
    chunk_max_abs_entry_max: float
    final_max_abs_entry: float
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_chunks": self.n_chunks,
            "chunk_size": self.chunk_size,
            "frequency_shape": self.frequency_shape,
            "chunk_det_error_max": self.chunk_det_error_max,
            "final_det_error_max": self.final_det_error_max,
            "chunk_max_abs_entry_max": self.chunk_max_abs_entry_max,
            "final_max_abs_entry": self.final_max_abs_entry,
            "metadata": dict(self.metadata or {}),
        }


def diagnose_chunked_cascade(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    chunk_size: int = 512,
) -> ChunkCascadeReport:
    """
    Build per-chunk cascade diagnostics.

    This is helpful when very long ABCD products become ill-conditioned near
    stopbands.
    """
    f = _as_frequency_array(frequency_hz)
    cfg = cell_model or CellModelConfig()
    cells = layout_cell_abcd(f, layout, config=cfg)

    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    chunk_size = int(chunk_size)

    chunk_results = []
    for start in range(0, layout.n_cells, chunk_size):
        stop = min(start + chunk_size, layout.n_cells)
        chunk_results.append(cascade_cell_abcd_direct(cells[start:stop]))

    chunks = jnp.stack(chunk_results, axis=0)
    final = cascade_cell_abcd_direct(chunks)

    chunk_det_error = jnp.max(jnp.abs(abcd_determinant(chunks) - 1.0))
    final_det_error = jnp.max(jnp.abs(abcd_determinant(final) - 1.0))
    chunk_max_abs = jnp.max(jnp.abs(chunks))
    final_max_abs = jnp.max(jnp.abs(final))

    return ChunkCascadeReport(
        n_cells=layout.n_cells,
        n_chunks=int(chunks.shape[0]),
        chunk_size=chunk_size,
        frequency_shape=tuple(int(v) for v in f.shape),
        chunk_det_error_max=float(chunk_det_error),
        final_det_error_max=float(final_det_error),
        chunk_max_abs_entry_max=float(chunk_max_abs),
        final_max_abs_entry=float(final_max_abs),
        metadata={
            "layout_name": layout.name,
            "cell_model": cfg.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Linear baseline comparisons
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearBaselineComparison:
    """
    Comparison between cascaded lumped layout and analytic uniform line.
    """

    layout_name: str
    reference_kind: str
    max_abs_s_diff: float
    rms_abs_s_diff: float
    s21_max_abs_diff: float
    s21_db_max_abs_diff: float
    cascade_diagnostics: Mapping[str, Any]
    reference_diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "reference_kind": self.reference_kind,
            "max_abs_s_diff": self.max_abs_s_diff,
            "rms_abs_s_diff": self.rms_abs_s_diff,
            "s21_max_abs_diff": self.s21_max_abs_diff,
            "s21_db_max_abs_diff": self.s21_db_max_abs_diff,
            "cascade_diagnostics": dict(self.cascade_diagnostics),
            "reference_diagnostics": dict(self.reference_diagnostics),
            "metadata": dict(self.metadata or {}),
        }


def compare_layout_to_uniform_rlgc_line(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    use_lossy_reference: bool = True,
) -> LinearBaselineComparison:
    """
    Compare cascaded lumped layout to an analytic uniform RLGC line.

    This is meaningful mainly for uniform layouts or weakly perturbed layouts.
    Periodic/bandgap layouts should be compared through dispersion and full S21,
    not expected to match a uniform line.
    """
    f = _as_frequency_array(frequency_hz)

    result = cascade_layout_abcd(
        f,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
    )
    s_layout = abcd_to_s(result.abcd, z0_ohm=layout.z0_ohm)

    L_per_m = float(jnp.sum(layout.L_series_H) / layout.total_length_m)
    C_per_m = float(jnp.sum(layout.total_shunt_C_F) / layout.total_length_m)
    R_per_m = float(jnp.sum(layout.R_series_ohm) / layout.total_length_m)
    G_per_m = float(jnp.sum(layout.G_shunt_S) / layout.total_length_m)

    if use_lossy_reference:
        ref_abcd = lossy_line_abcd_rlgc(
            f,
            length_m=layout.total_length_m,
            R_per_m_ohm=R_per_m,
            L_per_m_H=L_per_m,
            G_per_m_S=G_per_m,
            C_per_m_F=C_per_m,
        )
        reference_kind = "lossy_rlgc"
    else:
        ref_abcd = ideal_lossless_line_abcd(
            f,
            length_m=layout.total_length_m,
            L_per_m_H=L_per_m,
            C_per_m_F=C_per_m,
        )
        reference_kind = "ideal_lossless"

    s_ref = abcd_to_s(ref_abcd, z0_ohm=layout.z0_ohm)
    comp = compare_sparameters(s_layout, s_ref, label_a="layout", label_b=reference_kind)

    return LinearBaselineComparison(
        layout_name=layout.name,
        reference_kind=reference_kind,
        max_abs_s_diff=comp["max_abs_diff"],
        rms_abs_s_diff=comp["rms_abs_diff"],
        s21_max_abs_diff=comp["s21_max_abs_diff"],
        s21_db_max_abs_diff=comp["s21_db_max_abs_diff"],
        cascade_diagnostics=result.diagnostics(z0_ohm=layout.z0_ohm).to_dict(),
        reference_diagnostics=diagnose_two_port(ref_abcd, z0_ohm=layout.z0_ohm).to_dict(),
        metadata={
            "L_per_m_H_effective": L_per_m,
            "C_per_m_F_effective": C_per_m,
            "R_per_m_ohm_effective": R_per_m,
            "G_per_m_S_effective": G_per_m,
            "layout_summary": layout.summary(),
            "cascade_strategy": result.strategy.value,
        },
    )


# ---------------------------------------------------------------------------
# Pump-off scan result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearScanResult:
    """
    Pump-off linear scan result.

    Attributes
    ----------
    frequency_hz:
        Frequency grid.
    s:
        S-parameters, shape (F, 2, 2).
    abcd:
        ABCD matrices, shape (F, 2, 2).
    beta_eff_rad_per_m:
        Effective beta extracted from S21 phase.
    group_delay_s:
        Group delay extracted from S21 phase.
    metadata:
        Report metadata.
    """

    frequency_hz: jax.Array
    s: jax.Array
    abcd: jax.Array
    beta_eff_rad_per_m: jax.Array
    group_delay_s: jax.Array
    metadata: Mapping[str, Any] | None = None

    @property
    def s21(self) -> jax.Array:
        return s21(self.s)

    @property
    def s21_db(self) -> jax.Array:
        return s_to_db(self.s21)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency_shape": tuple(int(v) for v in self.frequency_hz.shape),
            "s_shape": tuple(int(v) for v in self.s.shape),
            "abcd_shape": tuple(int(v) for v in self.abcd.shape),
            "s21_db_min": float(jnp.min(self.s21_db)),
            "s21_db_max": float(jnp.max(self.s21_db)),
            "beta_eff_min_rad_per_m": float(jnp.min(self.beta_eff_rad_per_m)),
            "beta_eff_max_rad_per_m": float(jnp.max(self.beta_eff_rad_per_m)),
            "group_delay_min_s": float(jnp.min(self.group_delay_s)),
            "group_delay_max_s": float(jnp.max(self.group_delay_s)),
            "metadata": dict(self.metadata or {}),
        }


def run_linear_scan(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
) -> LinearScanResult:
    """
    Run a pump-off linear S-parameter scan for a layout.
    """
    f = _as_frequency_array(frequency_hz)
    result = cascade_layout_abcd(
        f,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
    )
    sm = abcd_to_s(result.abcd, z0_ohm=layout.z0_ohm)
    s21_values = s21(sm)

    beta_eff = effective_beta_from_s21(
        f,
        s21_values,
        length_m=layout.total_length_m,
    )
    tau_g = group_delay_from_s21(f, s21_values)

    return LinearScanResult(
        frequency_hz=f,
        s=sm,
        abcd=result.abcd,
        beta_eff_rad_per_m=beta_eff,
        group_delay_s=tau_g,
        metadata={
            "layout_name": layout.name,
            "z0_ohm": layout.z0_ohm,
            "n_cells": layout.n_cells,
            "total_length_m": layout.total_length_m,
            "cell_model": (cell_model or CellModelConfig()).to_dict(),
            "cascade": result.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CascadeValidationReport:
    """
    High-level validation report for the cascade layer.
    """

    layout_name: str
    strategy: str
    n_cells: int
    n_frequencies: int
    passed: bool
    has_nan: bool
    has_inf: bool
    max_abs_abcd_entry: float
    det_error_max: float
    reciprocity_error_max: float
    passivity_violation: float
    s21_db_min: float
    s21_db_max: float
    messages: list[str]
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.layout_name,
            "strategy": self.strategy,
            "n_cells": self.n_cells,
            "n_frequencies": self.n_frequencies,
            "passed": self.passed,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "max_abs_abcd_entry": self.max_abs_abcd_entry,
            "det_error_max": self.det_error_max,
            "reciprocity_error_max": self.reciprocity_error_max,
            "passivity_violation": self.passivity_violation,
            "s21_db_min": self.s21_db_min,
            "s21_db_max": self.s21_db_max,
            "messages": list(self.messages),
            "metadata": dict(self.metadata or {}),
        }


def validate_cascade(
    frequency_hz: ArrayLike,
    layout: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    det_tolerance: float = 1e-6,
    reciprocity_tolerance: float = 1e-9,
    passivity_tolerance: float = 1e-7,
) -> CascadeValidationReport:
    """
    Validate a linear cascade.

    This is meant for lossless/weakly lossy passive layouts. Active gain is not
    expected in this linear pump-off layer.
    """
    f = _as_frequency_array(frequency_hz)
    result = cascade_layout_abcd(
        f,
        layout,
        cell_model=cell_model,
        cascade_config=cascade_config,
    )

    abcd = result.abcd
    has_nan = bool(jnp.any(jnp.isnan(jnp.real(abcd))) or jnp.any(jnp.isnan(jnp.imag(abcd))))
    has_inf = bool(jnp.any(jnp.isinf(jnp.real(abcd))) or jnp.any(jnp.isinf(jnp.imag(abcd))))
    max_abs = float(jnp.max(jnp.abs(abcd)))

    diag = result.diagnostics(z0_ohm=layout.z0_ohm).to_dict()

    messages: list[str] = []
    passed = True

    if has_nan or has_inf:
        passed = False
        messages.append("FAIL: ABCD contains NaN or Inf.")

    if diag["det_abcd_max_abs_error"] > det_tolerance:
        passed = False
        messages.append(
            f"FAIL: determinant error {diag['det_abcd_max_abs_error']} exceeds {det_tolerance}."
        )

    if diag["reciprocity_s_max_abs_error"] > reciprocity_tolerance:
        passed = False
        messages.append(
            f"FAIL: reciprocity error {diag['reciprocity_s_max_abs_error']} exceeds "
            f"{reciprocity_tolerance}."
        )

    if diag["passivity_violation"] > passivity_tolerance:
        passed = False
        messages.append(
            f"FAIL: passivity violation {diag['passivity_violation']} exceeds "
            f"{passivity_tolerance}."
        )

    if passed:
        messages.append("PASS: cascade validation checks passed.")

    return CascadeValidationReport(
        layout_name=layout.name,
        strategy=result.strategy.value,
        n_cells=layout.n_cells,
        n_frequencies=int(f.shape[0]),
        passed=bool(passed),
        has_nan=has_nan,
        has_inf=has_inf,
        max_abs_abcd_entry=max_abs,
        det_error_max=diag["det_abcd_max_abs_error"],
        reciprocity_error_max=diag["reciprocity_s_max_abs_error"],
        passivity_violation=diag["passivity_violation"],
        s21_db_min=diag["s21_db_min"],
        s21_db_max=diag["s21_db_max"],
        messages=messages,
        metadata={
            "cell_model": (cell_model or CellModelConfig()).to_dict(),
            "cascade_config": (cascade_config or CascadeConfig()).to_dict(),
            "diagnostics": diag,
        },
    )


__all__ = [
    "CascadeStrategy",
    "CascadeConfig",
    "CascadeResult",
    "layout_is_uniform",
    "layout_is_periodic",
    "choose_cascade_strategy",
    "cascade_cell_abcd_direct",
    "cascade_cell_abcd_chunked",
    "cascade_repeated_cell_power",
    "cascade_repeated_supercell_power",
    "extract_first_cell_layout",
    "cascade_layout_abcd",
    "layout_sparameters",
    "layout_s21",
    "ChunkCascadeReport",
    "diagnose_chunked_cascade",
    "LinearBaselineComparison",
    "compare_layout_to_uniform_rlgc_line",
    "LinearScanResult",
    "run_linear_scan",
    "CascadeValidationReport",
    "validate_cascade",
]