"""
twpa.io.netlist
===============

Netlist import/export helpers for TWPA layouts.

The main production use is exporting simulator-native ``LineLayout`` objects to
SPICE-like subcircuits for external checking in tools such as ngspice, WRspice,
Keysight ADS-style netlist bridges, or custom circuit validators.

Supported exports
-----------------
SPICE subcircuit
    Lumped ladder with one series branch per cell and shunt loading at nodes.

CSV component table
    Human-readable one-row-per-cell component table.

JSON layout summary
    Simulator metadata and compact array summaries.

The exported netlists are intended as an interoperability/debugging layer, not
as the single source of truth. The simulator-native ``LineLayout`` remains the
authoritative representation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import math
import re
import csv
import numpy as np

import jax
import jax.numpy as jnp

from twpa.core.layout import LineLayout, make_layout_from_arrays


ArrayLike = Any


class NetlistFormat(str, Enum):
    """Supported netlist/export formats."""

    SPICE_SUBCKT = "spice_subckt"
    COMPONENT_CSV = "component_csv"
    JSON_SUMMARY = "json_summary"


class ShuntPlacement(str, Enum):
    """
    Shunt capacitance/conductance placement convention.

    PI_SPLIT:
        Half of each cell shunt is placed on the left node and half on the
        right node. This is closest to a symmetric pi-section model.

    RIGHT_NODE:
        Full cell shunt is placed at the right node of each cell.

    LEFT_NODE:
        Full cell shunt is placed at the left node of each cell.

    MID_NODE:
        Introduce an internal midpoint node per cell; series branch is split
        into two halves and the full shunt is attached at the midpoint.
    """

    PI_SPLIT = "pi_split"
    RIGHT_NODE = "right_node"
    LEFT_NODE = "left_node"
    MID_NODE = "mid_node"


class SeriesBranchModel(str, Enum):
    """
    Series branch export convention.
    """

    RL_SERIES = "rl_series"
    L_ONLY_WITH_R_OMITTED_IF_ZERO = "l_only_with_r_omitted_if_zero"


class ResonatorExportMode(str, Enum):
    """
    How optional resonator/stub arrays are exported.

    SHUNT_CAP_ONLY:
        Export C_stub_F as an additional shunt capacitor to ground.

    SERIES_LC_TO_GROUND:
        If L_res_H and C_res_F are present/nonzero, export a series LC branch
        to ground at the selected shunt node.

    COUPLED_RESONATOR:
        If C_couple_F, L_res_H, and C_res_F are present/nonzero, export a
        capacitively coupled resonator branch.

    OMIT:
        Do not export resonator/stub loading.
    """

    SHUNT_CAP_ONLY = "shunt_cap_only"
    SERIES_LC_TO_GROUND = "series_lc_to_ground"
    COUPLED_RESONATOR = "coupled_resonator"
    OMIT = "omit"


@dataclass(frozen=True)
class NetlistExportConfig:
    """
    Configuration for exporting a LineLayout to a netlist.

    Parameters
    ----------
    subckt_name:
        Name used in the SPICE ``.subckt`` line.
    input_port:
        External input port name.
    output_port:
        External output port name.
    ground:
        Ground/reference node name.
    internal_node_prefix:
        Prefix for generated internal nodes.
    element_prefix:
        Prefix added to all generated element names.
    shunt_placement:
        Shunt placement convention.
    series_branch_model:
        Series branch convention.
    resonator_mode:
        Export convention for optional C_stub/L_res/C_res/C_couple arrays.
    include_cell_comments:
        Include per-cell comments in SPICE output.
    include_metadata_header:
        Include metadata comments at the top of SPICE output.
    min_export_value:
        Values with absolute magnitude below this threshold are omitted for
        optional passive components.
    numeric_format:
        Python format string for component values.
    """

    subckt_name: str = "twpa_line"
    input_port: str = "in"
    output_port: str = "out"
    ground: str = "0"
    internal_node_prefix: str = "n"
    midpoint_node_prefix: str = "m"
    resonator_node_prefix: str = "r"
    element_prefix: str = "X"
    shunt_placement: ShuntPlacement = ShuntPlacement.PI_SPLIT
    series_branch_model: SeriesBranchModel = SeriesBranchModel.RL_SERIES
    resonator_mode: ResonatorExportMode = ResonatorExportMode.SHUNT_CAP_ONLY
    include_cell_comments: bool = False
    include_metadata_header: bool = True
    include_model_footer: bool = True
    min_export_value: float = 0.0
    numeric_format: str = ".12e"
    line_wrap_columns: int = 120
    name: str = "netlist_export"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shunt_placement", ShuntPlacement(self.shunt_placement))
        object.__setattr__(self, "series_branch_model", SeriesBranchModel(self.series_branch_model))
        object.__setattr__(self, "resonator_mode", ResonatorExportMode(self.resonator_mode))

        for attr in ["subckt_name", "input_port", "output_port", "ground"]:
            value = getattr(self, attr)
            if not value:
                raise ValueError(f"{attr} may not be empty")

        if self.min_export_value < 0.0:
            raise ValueError("min_export_value must be non-negative")

    def with_updates(self, **kwargs: Any) -> "NetlistExportConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subckt_name": self.subckt_name,
            "input_port": self.input_port,
            "output_port": self.output_port,
            "ground": self.ground,
            "internal_node_prefix": self.internal_node_prefix,
            "midpoint_node_prefix": self.midpoint_node_prefix,
            "resonator_node_prefix": self.resonator_node_prefix,
            "element_prefix": self.element_prefix,
            "shunt_placement": self.shunt_placement.value,
            "series_branch_model": self.series_branch_model.value,
            "resonator_mode": self.resonator_mode.value,
            "include_cell_comments": self.include_cell_comments,
            "include_metadata_header": self.include_metadata_header,
            "include_model_footer": self.include_model_footer,
            "min_export_value": self.min_export_value,
            "numeric_format": self.numeric_format,
            "line_wrap_columns": self.line_wrap_columns,
            "name": self.name,
        }


@dataclass(frozen=True)
class NetlistExportResult:
    """
    Result of a netlist export operation.
    """

    text: str
    format: NetlistFormat
    config: NetlistExportConfig
    n_cells: int
    n_elements: int
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", NetlistFormat(self.format))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text, encoding="utf-8")
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format.value,
            "config": self.config.to_dict(),
            "n_cells": self.n_cells,
            "n_elements": self.n_elements,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata or {}),
        }


def _safe_spice_name(name: str) -> str:
    """
    Sanitize a string for use as a SPICE identifier.
    """
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    if not s:
        s = "unnamed"
    if not re.match(r"^[A-Za-z]", s):
        s = "n_" + s
    return s


def _fmt(value: Any, config: NetlistExportConfig) -> str:
    """
    Format a numeric SPICE component value.
    """
    x = float(np.asarray(value))
    if not math.isfinite(x):
        raise ValueError(f"Non-finite component value {x}")
    return format(x, config.numeric_format)


def _present(value: Any, config: NetlistExportConfig) -> bool:
    return abs(float(np.asarray(value))) > config.min_export_value


def _array(layout: LineLayout, name: str, default: float = 0.0) -> jax.Array:
    """
    Fetch a layout array, falling back to zeros if missing.
    """
    value = getattr(layout, name, None)
    if value is None:
        return jnp.full((layout.n_cells,), default, dtype=jnp.float64)
    arr = jnp.asarray(value)
    if arr.shape != (layout.n_cells,):
        raise ValueError(f"layout.{name} must have shape {(layout.n_cells,)}, got {arr.shape}")
    return arr


def _node_name(index: int, layout: LineLayout, config: NetlistExportConfig) -> str:
    if index == 0:
        return config.input_port
    if index == layout.n_cells:
        return config.output_port
    return f"{config.internal_node_prefix}{index}"


def _mid_node_name(index: int, config: NetlistExportConfig) -> str:
    return f"{config.midpoint_node_prefix}{index}"


def _res_node_name(index: int, config: NetlistExportConfig) -> str:
    return f"{config.resonator_node_prefix}{index}"


def _cell_shunt_nodes(
    index: int,
    layout: LineLayout,
    config: NetlistExportConfig,
) -> tuple[tuple[str, float], ...]:
    """
    Return shunt node/fraction pairs for a cell.
    """
    left = _node_name(index, layout, config)
    right = _node_name(index + 1, layout, config)

    if config.shunt_placement == ShuntPlacement.LEFT_NODE:
        return ((left, 1.0),)

    if config.shunt_placement == ShuntPlacement.RIGHT_NODE:
        return ((right, 1.0),)

    if config.shunt_placement == ShuntPlacement.PI_SPLIT:
        return ((left, 0.5), (right, 0.5))

    if config.shunt_placement == ShuntPlacement.MID_NODE:
        return ((_mid_node_name(index, config), 1.0),)

    raise ValueError(f"Unsupported shunt placement {config.shunt_placement}")


def _append_series_branch(
    lines: list[str],
    *,
    cell_index: int,
    node_a: str,
    node_b: str,
    L_H: float,
    R_ohm: float,
    config: NetlistExportConfig,
) -> int:
    """
    Append series R/L elements. Returns number of elements added.
    """
    count = 0
    prefix = config.element_prefix

    if config.series_branch_model == SeriesBranchModel.L_ONLY_WITH_R_OMITTED_IF_ZERO:
        if _present(R_ohm, config):
            r_node = f"{node_a}_r{cell_index}"
            lines.append(f"R{prefix}{cell_index}_SER {node_a} {r_node} {_fmt(R_ohm, config)}")
            lines.append(f"L{prefix}{cell_index}_SER {r_node} {node_b} {_fmt(L_H, config)}")
            count += 2
        else:
            lines.append(f"L{prefix}{cell_index}_SER {node_a} {node_b} {_fmt(L_H, config)}")
            count += 1
        return count

    if config.series_branch_model == SeriesBranchModel.RL_SERIES:
        if _present(R_ohm, config):
            r_node = f"{node_a}_r{cell_index}"
            lines.append(f"R{prefix}{cell_index}_SER {node_a} {r_node} {_fmt(R_ohm, config)}")
            lines.append(f"L{prefix}{cell_index}_SER {r_node} {node_b} {_fmt(L_H, config)}")
            count += 2
        else:
            lines.append(f"L{prefix}{cell_index}_SER {node_a} {node_b} {_fmt(L_H, config)}")
            count += 1
        return count

    raise ValueError(f"Unsupported series branch model {config.series_branch_model}")


def _append_shunt_loading(
    lines: list[str],
    *,
    cell_index: int,
    node: str,
    fraction: float,
    C_F: float,
    G_S: float,
    C_stub_F: float,
    L_res_H: float,
    C_res_F: float,
    C_couple_F: float,
    config: NetlistExportConfig,
) -> int:
    """
    Append shunt C/G/stub/resonator loading. Returns number of elements added.
    """
    count = 0
    prefix = config.element_prefix

    C_eff = fraction * C_F
    G_eff = fraction * G_S
    C_stub_eff = fraction * C_stub_F

    if _present(C_eff, config):
        lines.append(f"C{prefix}{cell_index}_{node}_SH {node} {config.ground} {_fmt(C_eff, config)}")
        count += 1

    if _present(G_eff, config):
        R_equiv = 1.0 / G_eff
        lines.append(f"R{prefix}{cell_index}_{node}_GSH {node} {config.ground} {_fmt(R_equiv, config)}")
        count += 1

    if config.resonator_mode == ResonatorExportMode.OMIT:
        return count

    if config.resonator_mode == ResonatorExportMode.SHUNT_CAP_ONLY:
        if _present(C_stub_eff, config):
            lines.append(f"C{prefix}{cell_index}_{node}_STUB {node} {config.ground} {_fmt(C_stub_eff, config)}")
            count += 1
        return count

    if config.resonator_mode == ResonatorExportMode.SERIES_LC_TO_GROUND:
        if _present(L_res_H, config) and _present(C_res_F, config):
            rnode = f"{_res_node_name(cell_index, config)}_{node}"
            lines.append(f"L{prefix}{cell_index}_{node}_RES {node} {rnode} {_fmt(L_res_H, config)}")
            lines.append(f"C{prefix}{cell_index}_{node}_RES {rnode} {config.ground} {_fmt(C_res_F, config)}")
            count += 2
        elif _present(C_stub_eff, config):
            lines.append(f"C{prefix}{cell_index}_{node}_STUB {node} {config.ground} {_fmt(C_stub_eff, config)}")
            count += 1
        return count

    if config.resonator_mode == ResonatorExportMode.COUPLED_RESONATOR:
        if _present(C_couple_F, config) and _present(L_res_H, config) and _present(C_res_F, config):
            rnode = f"{_res_node_name(cell_index, config)}_{node}"
            lines.append(f"C{prefix}{cell_index}_{node}_COUP {node} {rnode} {_fmt(C_couple_F * fraction, config)}")
            lines.append(f"L{prefix}{cell_index}_{node}_RES {rnode} {config.ground} {_fmt(L_res_H, config)}")
            lines.append(f"C{prefix}{cell_index}_{node}_RES {rnode} {config.ground} {_fmt(C_res_F, config)}")
            count += 3
        elif _present(C_stub_eff, config):
            lines.append(f"C{prefix}{cell_index}_{node}_STUB {node} {config.ground} {_fmt(C_stub_eff, config)}")
            count += 1
        return count

    raise ValueError(f"Unsupported resonator export mode {config.resonator_mode}")


def export_layout_to_spice_subckt(
    layout: LineLayout,
    config: NetlistExportConfig | None = None,
) -> NetlistExportResult:
    """
    Export a LineLayout as a SPICE ``.subckt``.

    The generated subcircuit has external ports:

        .subckt <name> <input_port> <output_port> <ground>

    and internal nodes are generated deterministically.
    """
    cfg = config or NetlistExportConfig()
    subckt_name = _safe_spice_name(cfg.subckt_name)

    L = _array(layout, "L_series_H")
    C = _array(layout, "C_shunt_F")
    R = _array(layout, "R_series_ohm")
    G = _array(layout, "G_shunt_S")
    C_stub = _array(layout, "C_stub_F")
    L_res = _array(layout, "L_res_H")
    C_res = _array(layout, "C_res_F")
    C_couple = _array(layout, "C_couple_F")

    lines: list[str] = []
    warnings: list[str] = []
    n_elements = 0

    if cfg.include_metadata_header:
        lines.extend(
            [
                f"* TWPA layout export",
                f"* layout_name: {getattr(layout, 'name', 'unnamed')}",
                f"* n_cells: {layout.n_cells}",
                f"* total_length_m: {float(jnp.sum(_array(layout, 'length_m'))):.12e}",
                f"* z0_ohm: {getattr(layout, 'z0_ohm', None)}",
                f"* shunt_placement: {cfg.shunt_placement.value}",
                f"* resonator_mode: {cfg.resonator_mode.value}",
                f"* generated_by: twpa.io.netlist.export_layout_to_spice_subckt",
                "",
            ]
        )

    lines.append(f".subckt {subckt_name} {cfg.input_port} {cfg.output_port} {cfg.ground}")

    for i in range(layout.n_cells):
        if cfg.include_cell_comments:
            lines.append(f"* cell {i}")

        left = _node_name(i, layout, cfg)
        right = _node_name(i + 1, layout, cfg)

        if config and cfg.shunt_placement == ShuntPlacement.MID_NODE:
            mid = _mid_node_name(i, cfg)
            n_elements += _append_series_branch(
                lines,
                cell_index=i,
                node_a=left,
                node_b=mid,
                L_H=float(L[i]) * 0.5,
                R_ohm=float(R[i]) * 0.5,
                config=cfg,
            )
            n_elements += _append_series_branch(
                lines,
                cell_index=i,
                node_a=mid,
                node_b=right,
                L_H=float(L[i]) * 0.5,
                R_ohm=float(R[i]) * 0.5,
                config=cfg.with_updates(element_prefix=f"{cfg.element_prefix}{i}_B"),
            )
        else:
            n_elements += _append_series_branch(
                lines,
                cell_index=i,
                node_a=left,
                node_b=right,
                L_H=float(L[i]),
                R_ohm=float(R[i]),
                config=cfg,
            )

        for node, fraction in _cell_shunt_nodes(i, layout, cfg):
            n_elements += _append_shunt_loading(
                lines,
                cell_index=i,
                node=node,
                fraction=fraction,
                C_F=float(C[i]),
                G_S=float(G[i]),
                C_stub_F=float(C_stub[i]),
                L_res_H=float(L_res[i]),
                C_res_F=float(C_res[i]),
                C_couple_F=float(C_couple[i]),
                config=cfg,
            )

    lines.append(f".ends {subckt_name}")

    if cfg.include_model_footer:
        lines.extend(
            [
                "",
                "* End of TWPA subcircuit export",
            ]
        )

    text = "\n".join(lines) + "\n"

    return NetlistExportResult(
        text=text,
        format=NetlistFormat.SPICE_SUBCKT,
        config=cfg,
        n_cells=layout.n_cells,
        n_elements=n_elements,
        warnings=tuple(warnings),
        metadata={
            "layout": layout.summary() if hasattr(layout, "summary") else {},
            "subckt_name": subckt_name,
        },
    )


def write_spice_subckt(
    layout: LineLayout,
    path: str | Path,
    config: NetlistExportConfig | None = None,
) -> Path:
    """
    Export and write a SPICE subcircuit file.
    """
    result = export_layout_to_spice_subckt(layout, config)
    return result.write(path)


def layout_to_component_rows(layout: LineLayout) -> list[dict[str, Any]]:
    """
    Convert a LineLayout to one row per cell.
    """
    arrays = {
        "length_m": _array(layout, "length_m"),
        "L_series_H": _array(layout, "L_series_H"),
        "C_shunt_F": _array(layout, "C_shunt_F"),
        "R_series_ohm": _array(layout, "R_series_ohm"),
        "G_shunt_S": _array(layout, "G_shunt_S"),
        "C_stub_F": _array(layout, "C_stub_F"),
        "L_res_H": _array(layout, "L_res_H"),
        "C_res_F": _array(layout, "C_res_F"),
        "C_couple_F": _array(layout, "C_couple_F"),
    }

    rows: list[dict[str, Any]] = []
    for i in range(layout.n_cells):
        row = {"cell_index": i}
        for key, arr in arrays.items():
            row[key] = float(arr[i])
        rows.append(row)
    return rows


def write_layout_component_csv(
    layout: LineLayout,
    path: str | Path,
) -> Path:
    """
    Write a component CSV table with one row per cell.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = layout_to_component_rows(layout)
    if not rows:
        raise ValueError("layout has no cells")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return path


def load_layout_component_csv(
    path: str | Path,
    *,
    z0_ohm: float = 50.0,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LineLayout:
    """
    Load a LineLayout from a component CSV written by write_layout_component_csv.
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: empty component CSV")

    def col(name_: str, default: float = 0.0) -> jax.Array:
        return jnp.asarray([float(row.get(name_, default) or default) for row in rows], dtype=jnp.float64)

    return make_layout_from_arrays(
        length_m=col("length_m"),
        L_series_H=col("L_series_H"),
        C_shunt_F=col("C_shunt_F"),
        R_series_ohm=col("R_series_ohm"),
        G_shunt_S=col("G_shunt_S"),
        C_stub_F=col("C_stub_F"),
        L_res_H=col("L_res_H"),
        C_res_F=col("C_res_F"),
        C_couple_F=col("C_couple_F"),
        z0_ohm=z0_ohm,
        name=name or path.stem,
        metadata={
            "source": "load_layout_component_csv",
            "source_path": str(path),
            **dict(metadata or {}),
        },
    )


def write_layout_json_summary(
    layout: LineLayout,
    path: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Write a compact JSON layout summary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "length_m": _array(layout, "length_m"),
        "L_series_H": _array(layout, "L_series_H"),
        "C_shunt_F": _array(layout, "C_shunt_F"),
        "R_series_ohm": _array(layout, "R_series_ohm"),
        "G_shunt_S": _array(layout, "G_shunt_S"),
        "C_stub_F": _array(layout, "C_stub_F"),
        "L_res_H": _array(layout, "L_res_H"),
        "C_res_F": _array(layout, "C_res_F"),
        "C_couple_F": _array(layout, "C_couple_F"),
    }

    def arr_summary(arr: jax.Array) -> dict[str, Any]:
        a = np.asarray(arr)
        return {
            "shape": tuple(int(v) for v in a.shape),
            "min": float(np.nanmin(a)) if a.size else None,
            "max": float(np.nanmax(a)) if a.size else None,
            "mean": float(np.nanmean(a)) if a.size else None,
            "sum": float(np.nansum(a)) if a.size else None,
        }

    payload = {
        "layout": layout.summary() if hasattr(layout, "summary") else {},
        "array_summaries": {key: arr_summary(value) for key, value in arrays.items()},
        "extra": dict(extra or {}),
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_netlist_bundle(
    layout: LineLayout,
    output_dir: str | Path,
    *,
    config: NetlistExportConfig | None = None,
    prefix: str | None = None,
) -> dict[str, str]:
    """
    Write SPICE, CSV, and JSON layout export artifacts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = prefix or _safe_spice_name(getattr(layout, "name", "twpa_layout"))

    spice_path = write_spice_subckt(
        layout,
        out / f"{base}.cir",
        config=config,
    )
    csv_path = write_layout_component_csv(layout, out / f"{base}_components.csv")
    json_path = write_layout_json_summary(layout, out / f"{base}_layout_summary.json")

    index = {
        "spice_subckt": str(spice_path),
        "component_csv": str(csv_path),
        "layout_summary_json": str(json_path),
    }

    index_path = out / f"{base}_netlist_bundle_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    index["index_json"] = str(index_path)

    return index


def netlist_export_markdown(result: NetlistExportResult) -> str:
    """
    Markdown summary for a netlist export.
    """
    lines = [
        "# Netlist export",
        "",
        f"- format: `{result.format.value}`",
        f"- subcircuit: `{result.config.subckt_name}`",
        f"- cells: `{result.n_cells}`",
        f"- elements: `{result.n_elements}`",
        f"- shunt placement: `{result.config.shunt_placement.value}`",
        f"- resonator mode: `{result.config.resonator_mode.value}`",
        "",
    ]

    if result.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in result.warnings]
        lines.append("")

    lines += [
        "## Ports",
        "",
        f"- input: `{result.config.input_port}`",
        f"- output: `{result.config.output_port}`",
        f"- ground: `{result.config.ground}`",
    ]

    return "\n".join(lines)


__all__ = [
    "ArrayLike",
    "NetlistFormat",
    "ShuntPlacement",
    "SeriesBranchModel",
    "ResonatorExportMode",
    "NetlistExportConfig",
    "NetlistExportResult",
    "export_layout_to_spice_subckt",
    "write_spice_subckt",
    "layout_to_component_rows",
    "write_layout_component_csv",
    "load_layout_component_csv",
    "write_layout_json_summary",
    "write_netlist_bundle",
    "netlist_export_markdown",
]