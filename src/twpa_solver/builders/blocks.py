"""Generic topology blocks built from the solver's existing primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from twpa_solver.builders.ipm import (
    CouplerDiscrete,
    Element,
    add,
    add_edge_coupled_directional_coupler,
    add_jtl_element,
    add_tl,
)


@dataclass
class BlockRecord:
    path: str
    type: str
    cursors: list[str] = field(default_factory=list)
    start_nodes: dict[str, int] = field(default_factory=dict)
    end_nodes: dict[str, int] = field(default_factory=dict)
    count: int = 1


@dataclass
class BuildContext:
    circuit: list[Element]
    cursors: dict[str, int]
    cell_index: int
    ground: int
    coupler: CouplerDiscrete | None = None
    named_nodes: dict[str, int] = field(default_factory=dict)
    named_elements: dict[str, str] = field(default_factory=dict)
    blocks: list[BlockRecord] = field(default_factory=list)
    cursor_nodes: dict[str, set[int]] = field(default_factory=dict)
    mod_array: np.ndarray | None = None
    plan: Any = None

    def record(self, path: str, kind: str, names: list[str],
               starts: dict[str, int] | None = None) -> None:
        self.blocks.append(BlockRecord(path, kind, names, dict(starts or {})))

    def node(self, path: str, value: int) -> None:
        self.named_nodes[path] = int(value)


def _cursor(ctx: BuildContext, cfg: Mapping[str, Any], key: str = "cursor") -> str:
    name = str(cfg[key])
    if name not in ctx.cursors:
        raise ValueError(f"unknown cursor {name!r} in block")
    return name


def build_port(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None:
    cursor = _cursor(ctx, cfg)
    node = ctx.cursors[cursor]
    add(ctx.circuit, f"P{node}_{ctx.ground}", node, ctx.ground,
        int(cfg["port"]), "port")
    ctx.node(path, node)
    ctx.record(path, "port", [cursor], {cursor: node})


def build_resistor(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None:
    cursor = _cursor(ctx, cfg)
    node = ctx.cursors[cursor]
    add(ctx.circuit, f"R{node}_{ctx.ground}", node, ctx.ground,
        float(cfg["value"]), "resistor")
    ctx.node(path, node)
    ctx.record(path, "resistor", [cursor], {cursor: node})


def build_transmission_line(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None:
    cursor = _cursor(ctx, cfg)
    start = ctx.cursors[cursor]
    count = int(cfg["cells"])
    if count < 0:
        raise ValueError(f"{path}.cells must be non-negative")
    end = add_tl(ctx.circuit, start, ctx.ground, float(cfg["L"]),
                 float(cfg["C"]), count)
    ctx.cursors[cursor] = end
    ctx.node(path, start)
    ctx.node(f"{path}.end", end)
    for i in range(count):
        ctx.node(f"{path}.cell[{i}].left", start + i)
        ctx.node(f"{path}.cell[{i}].right", start + i + 1)
    record = BlockRecord(path, "transmission_line", [cursor], {cursor: start},
                         {cursor: end}, count)
    ctx.blocks.append(record)


def build_jj_line(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None:
    cursor = _cursor(ctx, cfg)
    start = ctx.cursors[cursor]
    count = int(cfg["cells"])
    if count <= 0:
        raise ValueError(f"{path}.cells must be positive")
    cg, lj, cj = float(cfg["Cg"]), float(cfg["Lj"]), float(cfg["Cj"])
    for i in range(count):
        node = start + i
        factor = 1.0 if ctx.mod_array is None else float(ctx.mod_array[ctx.cell_index])
        explicit = None
        if ctx.plan is not None:
            explicit = (float(ctx.plan.cg[ctx.cell_index]) / (2 if i == 0 else 1),
                        float(ctx.plan.lj[ctx.cell_index]),
                        float(ctx.plan.cj[ctx.cell_index]))
        before = len(ctx.circuit)
        add_jtl_element(ctx.circuit, node, ctx.ground, cg / 2 if i == 0 else cg,
                        lj, cj, mod_factor=factor, cell_index=ctx.cell_index,
                        explicit=explicit)
        generated = ctx.circuit[before:]
        ctx.named_elements[f"{path}.cell[{i}].Cg"] = generated[0].name
        ctx.named_elements[f"{path}.cell[{i}].Lj"] = generated[1].name
        ctx.named_elements[f"{path}.cell[{i}].Cj"] = generated[2].name
        ctx.named_elements[f"{path}.cell[{i}].right"] = generated[1].name
        ctx.node(f"{path}.cell[{i}].left", node)
        ctx.node(f"{path}.cell[{i}].right", node + 1)
        ctx.cell_index += 1
    end = start + count
    end_cg = cg / 2 if ctx.plan is None else float(ctx.plan.cg[ctx.cell_index - 1]) / 2
    add(ctx.circuit, f"C{end}_{ctx.ground}_JTL_end", end, ctx.ground, end_cg,
        "capacitor", role="jtl_cg", cell_index=ctx.cell_index - 1)
    ctx.cursors[cursor] = end
    ctx.node(path, start)
    ctx.node(f"{path}.end", end)
    ctx.blocks.append(BlockRecord(path, "jj_line", [cursor], {cursor: start},
                                  {cursor: end}, count))


def build_ipm_topology(params: Any, coupler: CouplerDiscrete,
                       mod_array: np.ndarray | None = None,
                       plan: Any = None) -> tuple[list[Element], dict[str, int]]:
    """Build the legacy IPM topology through the generic block dispatchers."""
    if mod_array is not None and plan is not None:
        raise ValueError("make_ipm accepts either mod_array or plan, not both")
    values = np.ones(params.num_rows * params.array_length, dtype=float) if mod_array is None else mod_array
    ctx = BuildContext([], {"signal": params.start_node_top, "pump": params.start_node_bot},
                       0, params.ground, coupler=coupler, mod_array=values, plan=plan)

    def emit(kind: str, cfg: dict[str, Any], path: str) -> None:
        before = len(ctx.circuit)
        from twpa_solver.builders.registry import BLOCK_BUILDERS
        BLOCK_BUILDERS[kind](ctx, cfg, path)
        # Keep block paths useful to callers of the generic builder.
        for index, element in enumerate(ctx.circuit[before:]):
            ctx.named_elements[f"{path}.element[{index}]"] = element.name

    p = params
    emit("port", {"cursor": "signal", "port": 1}, "input_signal")
    emit("resistor", {"cursor": "signal", "value": p.Rleft}, "input_signal_resistor")
    emit("transmission_line", {"cursor": "signal", "cells": p.len1,
                                "L": p.Ll, "C": p.Cl}, "input_signal_tl")
    emit("port", {"cursor": "pump", "port": 3}, "input_pump")
    emit("resistor", {"cursor": "pump", "value": p.Rm}, "input_pump_resistor")
    emit("transmission_line", {"cursor": "pump", "cells": p.len3,
                                "L": p.Ll, "C": p.Cl}, "input_pump_tl")
    emit("directional_coupler", {"cursors": ["signal", "pump"]}, "input_coupler")
    for row in range(1, p.num_rows):
        emit("jj_line", {"cursor": "signal", "cells": p.array_length,
                          "Lj": p.Lj, "Cj": p.Cj, "Cg": p.Cg}, f"row[{row}]")
        if row % p.arrays_per_dc == 0:
            emit("transmission_line", {"cursor": "signal", "cells": p.length_of_long_TL,
                                        "L": p.Ll, "C": p.Cl}, f"row[{row}].long")
            emit("transmission_line", {"cursor": "pump", "cells": p.coupler_section_length,
                                        "L": p.Ll, "C": p.Cl}, f"row[{row}].section")
            emit("directional_coupler", {"cursors": ["signal", "pump"]}, f"row[{row}].coupler")
        else:
            emit("transmission_line", {"cursor": "signal", "cells": p.length_of_short_TL,
                                        "L": p.Ll, "C": p.Cl}, f"row[{row}].short")
    emit("jj_line", {"cursor": "signal", "cells": p.array_length,
                      "Lj": p.Lj, "Cj": p.Cj, "Cg": p.Cg}, "final_array")
    emit("transmission_line", {"cursor": "signal", "cells": p.len2,
                                "L": p.Ll, "C": p.Cl}, "output_signal_tl")
    emit("resistor", {"cursor": "signal", "value": p.Rright}, "output_signal_resistor")
    emit("port", {"cursor": "signal", "port": 2}, "output_signal")
    emit("transmission_line", {"cursor": "pump", "cells": p.len4,
                                "L": p.Ll, "C": p.Cl}, "output_pump_tl")
    emit("resistor", {"cursor": "pump", "value": p.Rm}, "output_pump_resistor")
    emit("port", {"cursor": "pump", "port": 4}, "output_pump")
    return ctx.circuit, {"top_end_node": ctx.cursors["signal"],
                         "bottom_end_node": ctx.cursors["pump"],
                         "jj_mod_used": ctx.cell_index}


def build_directional_coupler(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None:
    names = [str(x) for x in cfg["cursors"]]
    if len(names) != 2 or any(x not in ctx.cursors for x in names):
        raise ValueError(f"{path}.cursors must name two existing cursors")
    if ctx.coupler is None:
        raise ValueError(f"{path} requires coupler parameters")
    starts = {names[0]: ctx.cursors[names[0]], names[1]: ctx.cursors[names[1]]}
    ends = add_edge_coupled_directional_coupler(ctx.circuit, starts[names[0]], starts[names[1]],
                                                ctx.ground, ctx.coupler)
    ctx.cursors[names[0]], ctx.cursors[names[1]] = ends
    ctx.node(path, starts[names[0]])
    ctx.blocks.append(BlockRecord(path, "directional_coupler", names, starts,
                                  {names[0]: ends[0], names[1]: ends[1]},
                                  ctx.coupler.N_coupled))


def build_raw_element(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None:
    nodes = cfg["nodes"]
    if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
        raise ValueError(f"{path}.nodes must contain two endpoints")
    add(ctx.circuit, str(cfg["name"]), nodes[0], nodes[1], cfg["value"], str(cfg["kind"]))
    ctx.record(path, "raw_element", [])
