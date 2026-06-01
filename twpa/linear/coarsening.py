"""
twpa.linear.coarsening
======================

Effective-cell and supercell coarsening utilities.

A 100 mm / 20,000-cell TWPA simulator should not jump directly from toy HB
to the full microscopic line. It needs a controlled hierarchy:

    N_eff = 20, 50, 100, 200, 500, 1000, 5000, 20000

The purpose of this module is to create reduced/effective layouts while making
the approximation explicit and auditable.

Core rule
---------
Do not average away real periodic loading unless the result is explicitly marked
as a surrogate. For bandgap/dispersion-engineered lines, preserve exact
supercells whenever possible.

Coarsening modes
----------------
1. exact_group_sum
   Groups adjacent cells and sums series/shunt lumped values.

2. preserve_z0_vp
   Groups cells while preserving group-averaged impedance and phase velocity.

3. exact_supercell
   Preserves an integer number of complete supercells.

4. repeat_supercell
   Extracts one exact supercell and repeats it to a target length.

5. uniform_surrogate
   Builds a uniform line with the same total length and integrated L/C/R/G.

This module does not run the linear cascade itself. It provides layout objects
that can be sent to twpa.linear.cascade and twpa.linear.dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp

from twpa.core.layout import (
    LineLayout,
    compare_layouts_basic,
    coarsen_uniform_layout,
    extract_supercell,
    make_layout_from_arrays,
    make_supercell_index,
)
from twpa.linear.dispersion import (
    DispersionConfig,
    extract_layout_dispersion,
)
from twpa.linear.cells import CellModelConfig


ArrayLike = Any


# ---------------------------------------------------------------------------
# Enums / configs
# ---------------------------------------------------------------------------

class CoarseningMethod(str, Enum):
    """Supported layout reduction methods."""

    EXACT_GROUP_SUM = "exact_group_sum"
    PRESERVE_Z0_VP = "preserve_z0_vp"
    EXACT_SUPERCELL = "exact_supercell"
    REPEAT_SUPERCELL = "repeat_supercell"
    UNIFORM_SURROGATE = "uniform_surrogate"


@dataclass(frozen=True)
class CoarseningConfig:
    """
    Configuration for effective-cell layout generation.

    Parameters
    ----------
    method:
        Coarsening method.
    factor:
        Number of fine cells per coarse cell for group-based methods.
    cells_per_supercell:
        Supercell size for periodic-layout methods.
    target_n_cells:
        Target cell count for surrogate/repetition methods.
    preserve_total_length:
        Whether to enforce total length equality.
    allow_remainder:
        Whether group/supercell operations may keep a trailing remainder.
    mark_surrogate:
        Whether to add metadata warning that the layout is approximate.
    """

    method: CoarseningMethod = CoarseningMethod.EXACT_GROUP_SUM
    factor: int = 10
    cells_per_supercell: int = 1
    target_n_cells: int | None = None
    preserve_total_length: bool = True
    allow_remainder: bool = False
    mark_surrogate: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", CoarseningMethod(self.method))
        if int(self.factor) <= 0:
            raise ValueError("factor must be positive")
        if int(self.cells_per_supercell) <= 0:
            raise ValueError("cells_per_supercell must be positive")
        object.__setattr__(self, "factor", int(self.factor))
        object.__setattr__(self, "cells_per_supercell", int(self.cells_per_supercell))
        if self.target_n_cells is not None and int(self.target_n_cells) <= 0:
            raise ValueError("target_n_cells must be positive if provided")
        if self.target_n_cells is not None:
            object.__setattr__(self, "target_n_cells", int(self.target_n_cells))

    def with_updates(self, **kwargs: Any) -> "CoarseningConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "factor": self.factor,
            "cells_per_supercell": self.cells_per_supercell,
            "target_n_cells": self.target_n_cells,
            "preserve_total_length": self.preserve_total_length,
            "allow_remainder": self.allow_remainder,
            "mark_surrogate": self.mark_surrogate,
        }


@dataclass(frozen=True)
class CoarseningResult:
    """
    Result of a layout coarsening operation.
    """

    original: LineLayout
    reduced: LineLayout
    config: CoarseningConfig
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original.summary(),
            "reduced": self.reduced.summary(),
            "config": self.config.to_dict(),
            "report": dict(self.report),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_divisible(n: int, factor: int, *, allow_remainder: bool) -> None:
    if n % factor != 0 and not allow_remainder:
        raise ValueError(
            f"n={n} is not divisible by factor={factor}. "
            "Set allow_remainder=True or choose a divisor."
        )


def _group_reduce(
    arr: jax.Array,
    *,
    factor: int,
    reducer: Literal["sum", "mean"],
    allow_remainder: bool,
) -> jax.Array:
    """
    Reduce a 1D cell array by grouping adjacent cells.

    Remainder cells, if allowed, form one smaller final group.
    """
    if arr.ndim != 1:
        raise ValueError(f"arr must be 1D, got {arr.shape}")

    n = int(arr.shape[0])
    _require_divisible(n, factor, allow_remainder=allow_remainder)

    groups = []
    for start in range(0, n, factor):
        stop = min(start + factor, n)
        chunk = arr[start:stop]
        if reducer == "sum":
            groups.append(jnp.sum(chunk))
        elif reducer == "mean":
            groups.append(jnp.mean(chunk))
        else:
            raise ValueError(f"Unsupported reducer {reducer!r}")

    return jnp.asarray(groups, dtype=arr.dtype)


def _safe_ratio(num: jax.Array, den: jax.Array, floor: float = 1e-300) -> jax.Array:
    return num / jnp.maximum(jnp.abs(den), floor)


def _metadata_warning(method: CoarseningMethod) -> str:
    if method == CoarseningMethod.EXACT_SUPERCELL:
        return "Exact supercell extraction; no averaging inside the selected supercell."
    if method == CoarseningMethod.REPEAT_SUPERCELL:
        return (
            "Repeated exact supercell surrogate. Valid only if the full layout is "
            "periodic or this approximation has been explicitly validated."
        )
    if method == CoarseningMethod.UNIFORM_SURROGATE:
        return (
            "Uniform surrogate. This can destroy periodic bandgap physics. Use only "
            "for controlled convergence tests, not final conclusions."
        )
    return (
        "Reduced/effective layout. Validate pump-off S21, dispersion, and gain "
        "convergence against the fine layout."
    )


# ---------------------------------------------------------------------------
# Coarsening methods
# ---------------------------------------------------------------------------

def coarsen_exact_group_sum(
    layout: LineLayout,
    *,
    factor: int,
    allow_remainder: bool = False,
    name: str | None = None,
) -> LineLayout:
    """
    Group adjacent cells by summing lumped series/shunt values.

    This preserves total integrated L, C, R, G and total length, but it changes
    the discretization and can distort high-frequency/cutoff behavior.
    """
    if int(factor) <= 0:
        raise ValueError("factor must be positive")
    factor = int(factor)

    if layout.has_resonators:
        raise ValueError(
            "Exact group-sum coarsening with resonators is not implemented safely. "
            "Use exact supercell extraction or a topology-specific reduction."
        )

    return make_layout_from_arrays(
        length_m=_group_reduce(
            layout.length_m,
            factor=factor,
            reducer="sum",
            allow_remainder=allow_remainder,
        ),
        L_series_H=_group_reduce(
            layout.L_series_H,
            factor=factor,
            reducer="sum",
            allow_remainder=allow_remainder,
        ),
        C_shunt_F=_group_reduce(
            layout.C_shunt_F,
            factor=factor,
            reducer="sum",
            allow_remainder=allow_remainder,
        ),
        R_series_ohm=_group_reduce(
            layout.R_series_ohm,
            factor=factor,
            reducer="sum",
            allow_remainder=allow_remainder,
        ),
        G_shunt_S=_group_reduce(
            layout.G_shunt_S,
            factor=factor,
            reducer="sum",
            allow_remainder=allow_remainder,
        ),
        C_stub_F=_group_reduce(
            layout.C_stub_F,
            factor=factor,
            reducer="sum",
            allow_remainder=allow_remainder,
        ),
        L_res_H=0.0,
        C_res_F=0.0,
        C_couple_F=0.0,
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_coarse_sum_x{factor}",
        metadata={
            **dict(layout.metadata or {}),
            "source": "coarsen_exact_group_sum",
            "parent_layout": layout.name,
            "coarsening_factor": factor,
            "allow_remainder": allow_remainder,
            "warning": _metadata_warning(CoarseningMethod.EXACT_GROUP_SUM),
        },
    )


def coarsen_preserve_z0_vp(
    layout: LineLayout,
    *,
    factor: int,
    allow_remainder: bool = False,
    name: str | None = None,
) -> LineLayout:
    """
    Coarsen by preserving group-averaged characteristic impedance and velocity.

    For each group:
        length = sum(dx)
        Z_group = mean(sqrt(L/C))
        vp_group = mean(dx/sqrt(L C))
        L_group = Z_group / vp_group * length
        C_group = 1/(Z_group vp_group) * length

    This can be smoother than direct sums, but it is an explicit surrogate and
    may wash out periodic loading physics.
    """
    if int(factor) <= 0:
        raise ValueError("factor must be positive")
    factor = int(factor)

    if layout.has_resonators:
        raise ValueError(
            "Preserve-z0-vp coarsening with resonators is not implemented safely."
        )

    C_total = layout.total_shunt_C_F
    z_cell = jnp.sqrt(layout.L_series_H / C_total)
    vp_cell = layout.length_m / jnp.sqrt(layout.L_series_H * C_total)

    length = _group_reduce(
        layout.length_m,
        factor=factor,
        reducer="sum",
        allow_remainder=allow_remainder,
    )
    z_group = _group_reduce(
        z_cell,
        factor=factor,
        reducer="mean",
        allow_remainder=allow_remainder,
    )
    vp_group = _group_reduce(
        vp_cell,
        factor=factor,
        reducer="mean",
        allow_remainder=allow_remainder,
    )

    L_per_m = z_group / vp_group
    C_per_m = 1.0 / (z_group * vp_group)

    L = L_per_m * length
    C_total_group = C_per_m * length

    # Preserve the average split between base C and stub C in each group.
    C_base_sum = _group_reduce(
        layout.C_shunt_F,
        factor=factor,
        reducer="sum",
        allow_remainder=allow_remainder,
    )
    C_stub_sum = _group_reduce(
        layout.C_stub_F,
        factor=factor,
        reducer="sum",
        allow_remainder=allow_remainder,
    )
    C_sum = C_base_sum + C_stub_sum
    stub_fraction = _safe_ratio(C_stub_sum, C_sum)
    C_stub = C_total_group * stub_fraction
    C_base = C_total_group - C_stub

    R = _group_reduce(
        layout.R_series_ohm,
        factor=factor,
        reducer="sum",
        allow_remainder=allow_remainder,
    )
    G = _group_reduce(
        layout.G_shunt_S,
        factor=factor,
        reducer="sum",
        allow_remainder=allow_remainder,
    )

    return make_layout_from_arrays(
        length_m=length,
        L_series_H=L,
        C_shunt_F=C_base,
        R_series_ohm=R,
        G_shunt_S=G,
        C_stub_F=C_stub,
        L_res_H=0.0,
        C_res_F=0.0,
        C_couple_F=0.0,
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_coarse_z0vp_x{factor}",
        metadata={
            **dict(layout.metadata or {}),
            "source": "coarsen_preserve_z0_vp",
            "parent_layout": layout.name,
            "coarsening_factor": factor,
            "allow_remainder": allow_remainder,
            "warning": _metadata_warning(CoarseningMethod.PRESERVE_Z0_VP),
        },
    )


def make_uniform_surrogate_layout(
    layout: LineLayout,
    *,
    target_n_cells: int,
    name: str | None = None,
) -> LineLayout:
    """
    Build a uniform surrogate with the same total length and integrated totals.

    This preserves:
        total length
        total L
        total C_total
        total R
        total G
        total C_stub fraction

    It destroys periodic loading and stopband physics.
    """
    if int(target_n_cells) <= 0:
        raise ValueError("target_n_cells must be positive")
    n = int(target_n_cells)

    total_length = layout.total_length_m
    dx = total_length / n

    L_total = jnp.sum(layout.L_series_H)
    C_base_total = jnp.sum(layout.C_shunt_F)
    C_stub_total = jnp.sum(layout.C_stub_F)
    R_total = jnp.sum(layout.R_series_ohm)
    G_total = jnp.sum(layout.G_shunt_S)

    return make_layout_from_arrays(
        length_m=jnp.full((n,), dx),
        L_series_H=jnp.full((n,), L_total / n),
        C_shunt_F=jnp.full((n,), C_base_total / n),
        R_series_ohm=jnp.full((n,), R_total / n),
        G_shunt_S=jnp.full((n,), G_total / n),
        C_stub_F=jnp.full((n,), C_stub_total / n),
        L_res_H=0.0,
        C_res_F=0.0,
        C_couple_F=0.0,
        z0_ohm=layout.z0_ohm,
        name=name or f"{layout.name}_uniform_surrogate_{n}",
        metadata={
            **dict(layout.metadata or {}),
            "source": "make_uniform_surrogate_layout",
            "parent_layout": layout.name,
            "target_n_cells": n,
            "warning": _metadata_warning(CoarseningMethod.UNIFORM_SURROGATE),
        },
    )


def extract_exact_supercell_layout(
    layout: LineLayout,
    *,
    cells_per_supercell: int,
    start_cell: int = 0,
    name: str | None = None,
) -> LineLayout:
    """
    Extract one exact supercell from a layout.
    """
    supercell = extract_supercell(
        layout,
        start_cell=start_cell,
        cells_per_supercell=cells_per_supercell,
        name=name or f"{layout.name}_exact_supercell_{cells_per_supercell}",
    )
    return supercell.with_metadata(
        source="extract_exact_supercell_layout",
        parent_layout=layout.name,
        start_cell=start_cell,
        cells_per_supercell=cells_per_supercell,
        warning=_metadata_warning(CoarseningMethod.EXACT_SUPERCELL),
    )


def repeat_supercell_layout(
    supercell: LineLayout,
    *,
    n_repeats: int,
    name: str | None = None,
    preserve_parent_metadata: bool = True,
) -> LineLayout:
    """
    Repeat an exact supercell n_repeats times.

    This creates an explicit vectorized layout, not an implicit cascade power.
    Use cascade.repeated_supercell_power for faster linear-only cascading.
    """
    if int(n_repeats) <= 0:
        raise ValueError("n_repeats must be positive")
    n_repeats = int(n_repeats)

    def tile(x: jax.Array) -> jax.Array:
        return jnp.tile(x, (n_repeats,))

    metadata = dict(supercell.metadata or {}) if preserve_parent_metadata else {}
    metadata.update(
        {
            "source": "repeat_supercell_layout",
            "supercell_layout": supercell.name,
            "n_repeats": n_repeats,
            "cells_per_supercell": supercell.n_cells,
            "warning": _metadata_warning(CoarseningMethod.REPEAT_SUPERCELL),
        }
    )

    return make_layout_from_arrays(
        length_m=tile(supercell.length_m),
        L_series_H=tile(supercell.L_series_H),
        C_shunt_F=tile(supercell.C_shunt_F),
        R_series_ohm=tile(supercell.R_series_ohm),
        G_shunt_S=tile(supercell.G_shunt_S),
        C_stub_F=tile(supercell.C_stub_F),
        L_res_H=tile(supercell.L_res_H),
        C_res_F=tile(supercell.C_res_F),
        C_couple_F=tile(supercell.C_couple_F),
        z0_ohm=supercell.z0_ohm,
        name=name or f"{supercell.name}_repeat_{n_repeats}",
        metadata=metadata,
    )


def make_repeated_supercell_surrogate(
    layout: LineLayout,
    *,
    cells_per_supercell: int,
    target_n_cells: int | None = None,
    target_length_m: float | None = None,
    name: str | None = None,
) -> LineLayout:
    """
    Extract the first supercell and repeat it to a target cell count/length.

    If target_n_cells is given, it must be divisible by cells_per_supercell.
    If target_length_m is given instead, the number of repeats is rounded to
    the nearest integer.
    """
    if int(cells_per_supercell) <= 0:
        raise ValueError("cells_per_supercell must be positive")
    cells_per_supercell = int(cells_per_supercell)

    supercell = extract_exact_supercell_layout(
        layout,
        cells_per_supercell=cells_per_supercell,
    )

    if target_n_cells is not None:
        if int(target_n_cells) <= 0:
            raise ValueError("target_n_cells must be positive")
        if int(target_n_cells) % cells_per_supercell != 0:
            raise ValueError("target_n_cells must be divisible by cells_per_supercell")
        n_repeats = int(target_n_cells) // cells_per_supercell
    elif target_length_m is not None:
        if target_length_m <= 0.0:
            raise ValueError("target_length_m must be positive")
        n_repeats = max(1, int(round(target_length_m / supercell.total_length_m)))
    else:
        idx = make_supercell_index(layout.n_cells, cells_per_supercell)
        n_repeats = idx.n_full_supercells

    return repeat_supercell_layout(
        supercell,
        n_repeats=n_repeats,
        name=name or f"{layout.name}_supercell_surrogate_{cells_per_supercell}x{n_repeats}",
    ).with_metadata(
        parent_layout=layout.name,
        target_n_cells=target_n_cells,
        target_length_m=target_length_m,
    )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def coarsen_layout(
    layout: LineLayout,
    config: CoarseningConfig,
    *,
    name: str | None = None,
) -> CoarseningResult:
    """
    Dispatch layout coarsening according to CoarseningConfig.
    """
    method = config.method

    if method == CoarseningMethod.EXACT_GROUP_SUM:
        reduced = coarsen_exact_group_sum(
            layout,
            factor=config.factor,
            allow_remainder=config.allow_remainder,
            name=name,
        )

    elif method == CoarseningMethod.PRESERVE_Z0_VP:
        reduced = coarsen_preserve_z0_vp(
            layout,
            factor=config.factor,
            allow_remainder=config.allow_remainder,
            name=name,
        )

    elif method == CoarseningMethod.EXACT_SUPERCELL:
        reduced = extract_exact_supercell_layout(
            layout,
            cells_per_supercell=config.cells_per_supercell,
            name=name,
        )

    elif method == CoarseningMethod.REPEAT_SUPERCELL:
        reduced = make_repeated_supercell_surrogate(
            layout,
            cells_per_supercell=config.cells_per_supercell,
            target_n_cells=config.target_n_cells,
            name=name,
        )

    elif method == CoarseningMethod.UNIFORM_SURROGATE:
        if config.target_n_cells is None:
            raise ValueError("UNIFORM_SURROGATE requires target_n_cells")
        reduced = make_uniform_surrogate_layout(
            layout,
            target_n_cells=config.target_n_cells,
            name=name,
        )

    else:
        raise ValueError(f"Unsupported coarsening method {method}")

    report = make_coarsening_report(layout, reduced, config)

    return CoarseningResult(
        original=layout,
        reduced=reduced,
        config=config,
        report=report,
    )


# ---------------------------------------------------------------------------
# Hierarchy generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoarseningHierarchyConfig:
    """
    Configuration for generating an N_eff hierarchy.

    Parameters
    ----------
    target_cell_counts:
        Desired reduced cell counts.
    method:
        Coarsening method for non-full targets.
    preserve_supercells:
        If true, round target cell counts to multiples of cells_per_supercell.
    cells_per_supercell:
        Supercell size to preserve.
    include_original:
        Append the original fine layout at the end.
    """

    target_cell_counts: tuple[int, ...] = (20, 50, 100, 200, 500, 1000, 5000)
    method: CoarseningMethod = CoarseningMethod.EXACT_GROUP_SUM
    preserve_supercells: bool = True
    cells_per_supercell: int = 1
    include_original: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", CoarseningMethod(self.method))
        if not self.target_cell_counts:
            raise ValueError("target_cell_counts may not be empty")
        cleaned = tuple(int(n) for n in self.target_cell_counts)
        if any(n <= 0 for n in cleaned):
            raise ValueError("all target_cell_counts must be positive")
        object.__setattr__(self, "target_cell_counts", cleaned)
        if int(self.cells_per_supercell) <= 0:
            raise ValueError("cells_per_supercell must be positive")
        object.__setattr__(self, "cells_per_supercell", int(self.cells_per_supercell))

    def with_updates(self, **kwargs: Any) -> "CoarseningHierarchyConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class CoarseningHierarchy:
    """
    Set of reduced layouts for convergence studies.
    """

    layouts: tuple[LineLayout, ...]
    reports: tuple[Mapping[str, Any], ...]
    config: CoarseningHierarchyConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layouts": len(self.layouts),
            "layout_names": [layout.name for layout in self.layouts],
            "n_cells": [layout.n_cells for layout in self.layouts],
            "reports": [dict(r) for r in self.reports],
            "config": {
                "target_cell_counts": list(self.config.target_cell_counts),
                "method": self.config.method.value,
                "preserve_supercells": self.config.preserve_supercells,
                "cells_per_supercell": self.config.cells_per_supercell,
                "include_original": self.config.include_original,
            },
        }


def _round_target_to_supercell_multiple(target: int, cells_per_supercell: int) -> int:
    if cells_per_supercell <= 1:
        return target
    rounded = max(cells_per_supercell, int(round(target / cells_per_supercell)) * cells_per_supercell)
    return rounded


def generate_coarsening_hierarchy(
    layout: LineLayout,
    config: CoarseningHierarchyConfig | None = None,
) -> CoarseningHierarchy:
    """
    Generate a set of reduced layouts for convergence studies.
    """
    cfg = config or CoarseningHierarchyConfig()
    layouts: list[LineLayout] = []
    reports: list[Mapping[str, Any]] = []

    for target in cfg.target_cell_counts:
        if target >= layout.n_cells:
            continue

        target_eff = (
            _round_target_to_supercell_multiple(target, cfg.cells_per_supercell)
            if cfg.preserve_supercells
            else target
        )
        target_eff = min(target_eff, layout.n_cells)

        if cfg.method in {
            CoarseningMethod.EXACT_GROUP_SUM,
            CoarseningMethod.PRESERVE_Z0_VP,
        }:
            factor = max(1, int(round(layout.n_cells / target_eff)))
            if factor <= 1:
                continue
            local_cfg = CoarseningConfig(
                method=cfg.method,
                factor=factor,
                allow_remainder=True,
            )
        elif cfg.method == CoarseningMethod.UNIFORM_SURROGATE:
            local_cfg = CoarseningConfig(
                method=CoarseningMethod.UNIFORM_SURROGATE,
                target_n_cells=target_eff,
            )
        elif cfg.method == CoarseningMethod.REPEAT_SUPERCELL:
            local_cfg = CoarseningConfig(
                method=CoarseningMethod.REPEAT_SUPERCELL,
                cells_per_supercell=cfg.cells_per_supercell,
                target_n_cells=target_eff,
            )
        else:
            raise ValueError(f"Unsupported hierarchy method {cfg.method}")

        result = coarsen_layout(
            layout,
            local_cfg,
            name=f"{layout.name}_Neff{target_eff}",
        )
        layouts.append(result.reduced)
        reports.append(result.report)

    if cfg.include_original:
        layouts.append(layout.with_metadata(hierarchy_original=True))
        reports.append(
            {
                "method": "original",
                "n_cells": layout.n_cells,
                "layout_name": layout.name,
            }
        )

    return CoarseningHierarchy(
        layouts=tuple(layouts),
        reports=tuple(reports),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Reports and validation
# ---------------------------------------------------------------------------

def make_coarsening_report(
    original: LineLayout,
    reduced: LineLayout,
    config: CoarseningConfig,
) -> dict[str, Any]:
    """
    Create an auditable report for a coarsening operation.
    """
    base = compare_layouts_basic(original, reduced)

    total_L_original = jnp.sum(original.L_series_H)
    total_L_reduced = jnp.sum(reduced.L_series_H)
    total_C_original = jnp.sum(original.total_shunt_C_F)
    total_C_reduced = jnp.sum(reduced.total_shunt_C_F)
    total_R_original = jnp.sum(original.R_series_ohm)
    total_R_reduced = jnp.sum(reduced.R_series_ohm)
    total_G_original = jnp.sum(original.G_shunt_S)
    total_G_reduced = jnp.sum(reduced.G_shunt_S)

    def rel_err(new: jax.Array, old: jax.Array) -> float:
        return float(jnp.abs(new - old) / jnp.maximum(jnp.abs(old), 1e-300))

    return {
        "method": config.method.value,
        "original_name": original.name,
        "reduced_name": reduced.name,
        "original_n_cells": original.n_cells,
        "reduced_n_cells": reduced.n_cells,
        "cell_reduction_factor": original.n_cells / reduced.n_cells,
        "basic_comparison": base,
        "total_L_relative_error": rel_err(total_L_reduced, total_L_original),
        "total_C_relative_error": rel_err(total_C_reduced, total_C_original),
        "total_R_relative_error": rel_err(total_R_reduced, total_R_original),
        "total_G_relative_error": rel_err(total_G_reduced, total_G_original),
        "total_length_relative_error": abs(
            reduced.total_length_m - original.total_length_m
        )
        / max(abs(original.total_length_m), 1e-300),
        "warning": _metadata_warning(config.method),
        "config": config.to_dict(),
    }


@dataclass(frozen=True)
class CoarseningDispersionComparison:
    """
    Dispersion/S21 comparison between original and reduced layouts.
    """

    original_name: str
    reduced_name: str
    frequency_shape: tuple[int, ...]
    s21_db_rms_error: float
    s21_db_max_abs_error: float
    beta_rms_relative_error: float
    beta_max_relative_error: float
    stopband_mask_mismatch_fraction: float
    metadata: Mapping[str, Any] | None = None

    @property
    def passed_loose(self) -> bool:
        return (
            self.s21_db_rms_error < 3.0
            and self.beta_rms_relative_error < 0.1
            and self.stopband_mask_mismatch_fraction < 0.1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_name": self.original_name,
            "reduced_name": self.reduced_name,
            "frequency_shape": self.frequency_shape,
            "s21_db_rms_error": self.s21_db_rms_error,
            "s21_db_max_abs_error": self.s21_db_max_abs_error,
            "beta_rms_relative_error": self.beta_rms_relative_error,
            "beta_max_relative_error": self.beta_max_relative_error,
            "stopband_mask_mismatch_fraction": self.stopband_mask_mismatch_fraction,
            "passed_loose": self.passed_loose,
            "metadata": dict(self.metadata or {}),
        }


def compare_coarsened_dispersion(
    frequency_hz: ArrayLike,
    original: LineLayout,
    reduced: LineLayout,
    *,
    cell_model: CellModelConfig | None = None,
    dispersion_config: DispersionConfig | None = None,
    stopband_s21_threshold_db: float = -10.0,
) -> CoarseningDispersionComparison:
    """
    Compare pump-off S21/dispersion between fine and reduced layouts.

    This is the validation gate for using an effective layout downstream.
    """
    f = jnp.asarray(frequency_hz, dtype=jnp.float64)
    if f.ndim != 1:
        raise ValueError("frequency_hz must be 1D")

    disp_cfg = dispersion_config or DispersionConfig()

    d0 = extract_layout_dispersion(
        f,
        original,
        cell_model=cell_model,
        dispersion_config=disp_cfg,
    )
    d1 = extract_layout_dispersion(
        f,
        reduced,
        cell_model=cell_model,
        dispersion_config=disp_cfg,
    )

    if d0.s21_db is None or d1.s21_db is None:
        raise ValueError("Both dispersion results must include S21 data")

    s21_err = d1.s21_db - d0.s21_db
    s21_rms = jnp.sqrt(jnp.mean(s21_err**2))
    s21_max = jnp.max(jnp.abs(s21_err))

    beta0 = d0.beta_preferred_rad_per_m
    beta1 = d1.beta_preferred_rad_per_m
    beta_rel = jnp.abs(beta1 - beta0) / jnp.maximum(jnp.abs(beta0), 1e-300)

    mask0 = d0.s21_db < stopband_s21_threshold_db
    mask1 = d1.s21_db < stopband_s21_threshold_db
    mismatch = jnp.mean((mask0 != mask1).astype(jnp.float64))

    return CoarseningDispersionComparison(
        original_name=original.name,
        reduced_name=reduced.name,
        frequency_shape=tuple(int(v) for v in f.shape),
        s21_db_rms_error=float(s21_rms),
        s21_db_max_abs_error=float(s21_max),
        beta_rms_relative_error=float(jnp.sqrt(jnp.mean(beta_rel**2))),
        beta_max_relative_error=float(jnp.max(beta_rel)),
        stopband_mask_mismatch_fraction=float(mismatch),
        metadata={
            "stopband_s21_threshold_db": stopband_s21_threshold_db,
            "original_dispersion": d0.to_dict(),
            "reduced_dispersion": d1.to_dict(),
        },
    )


def compare_hierarchy_dispersion(
    frequency_hz: ArrayLike,
    hierarchy: CoarseningHierarchy,
    *,
    reference: LineLayout | None = None,
    cell_model: CellModelConfig | None = None,
    dispersion_config: DispersionConfig | None = None,
) -> list[CoarseningDispersionComparison]:
    """
    Compare each layout in a hierarchy to a reference layout.

    If reference is None, the last hierarchy layout is used.
    """
    if len(hierarchy.layouts) == 0:
        raise ValueError("hierarchy contains no layouts")

    ref = reference or hierarchy.layouts[-1]
    comparisons = []

    for layout in hierarchy.layouts:
        if layout is ref or layout.name == ref.name:
            continue
        comparisons.append(
            compare_coarsened_dispersion(
                frequency_hz,
                ref,
                layout,
                cell_model=cell_model,
                dispersion_config=dispersion_config,
            )
        )

    return comparisons


__all__ = [
    "CoarseningMethod",
    "CoarseningConfig",
    "CoarseningResult",
    "coarsen_exact_group_sum",
    "coarsen_preserve_z0_vp",
    "make_uniform_surrogate_layout",
    "extract_exact_supercell_layout",
    "repeat_supercell_layout",
    "make_repeated_supercell_surrogate",
    "coarsen_layout",
    "CoarseningHierarchyConfig",
    "CoarseningHierarchy",
    "generate_coarsening_hierarchy",
    "make_coarsening_report",
    "CoarseningDispersionComparison",
    "compare_coarsened_dispersion",
    "compare_hierarchy_dispersion",
]