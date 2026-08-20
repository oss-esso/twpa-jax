from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import zlib

import numpy as np

from twpa_solver.builders.blocks import BlockRecord
from twpa_solver.builders.ipm import Element, IPMParams
from twpa_solver.builders.profiles import Selection, Segment, parse_profile_shorthand
from twpa_solver.builders.scatter import ScatterSpec, component_rng, draw_factors
from twpa_solver.circuit import Circuit
from twpa_solver.circuit.technology import Technology, load_technology
from twpa_solver.circuit.elements import ElementRef
from twpa_solver.circuit.handles import CellHandle, CouplerHandle, LineHandle
from twpa_solver.circuit.nodes import Node
from twpa_solver.design.errors import (DesignCollisionError, DesignParameterError,
                                       DesignResolutionError, DesignSchemaError)
from twpa_solver.design.model import CompiledDesign, PortRecord
from twpa_solver.design.normalize import normalize_design
from twpa_solver.design.parameters import (
    resolve_parameter_definitions,
    resolve_parameters,
)
from twpa_solver.design.schema import validate_design


def _expanded(items: list[Any], prefix: str = "topology", depth: int = 0,
              ancestors: tuple[str, ...] = ()) -> list[tuple[dict[str, Any], str]]:
    if depth > 2:
        raise DesignSchemaError(f"{prefix}: repeat nesting deeper than 2 is unsupported")
    result: list[tuple[dict[str, Any], str]] = []
    for index, item in enumerate(items):
        path = f"{prefix}[{index}]"
        if not isinstance(item, Mapping):
            raise DesignSchemaError(f"{path}: expected a mapping")
        if "repeat" not in item:
            name = str(item.get("name", item.get("type", f"block_{index}")))
            result.append((dict(item), ".".join((*ancestors, name))))
            continue
        repeat = item["repeat"]
        if not isinstance(repeat, Mapping):
            raise DesignSchemaError(f"{path}.repeat: expected a mapping")
        count = repeat.get("count")
        if not isinstance(count, int) or count < 0:
            raise DesignSchemaError(f"{path}.repeat.count: expected non-negative integer")
        name = str(repeat.get("name", "repeat"))
        nested = repeat.get("topology")
        if not isinstance(nested, list):
            raise DesignSchemaError(f"{path}.repeat.topology: expected a sequence")
        for occurrence in range(count):
            nested_items = _expanded(nested, f"{path}.repeat.topology", depth + 1,
                                     (*ancestors, f"{name}[{occurrence}]"))
            for cfg, nested_path in nested_items:
                cfg = dict(cfg)
                local_name = str(cfg.get("name", cfg.get("type", "block")))
                cfg["name"] = local_name
                result.append((cfg, nested_path))
    return result


def _config_for_block(cfg: dict[str, Any]) -> dict[str, Any]:
    """Preserve block declarations for the public builder resolver."""

    return dict(cfg)


def _expand_composite_topology(
    items: list[Any], params: Mapping[str, Any]
) -> list[tuple[dict[str, Any], str]]:
    """Expand repeats and composite blocks while retaining hierarchical paths."""

    result: list[tuple[dict[str, Any], str]] = []
    for cfg, generated_path in _expanded(items):
        source_name = str(cfg.get("name", cfg.get("type", "block")))
        prefix = (
            generated_path[:-len(source_name)]
            if generated_path.endswith(source_name) else ""
        )
        for primitive in _expand_composites([cfg], params):
            primitive = dict(primitive)
            primitive_name = str(
                primitive.get("name", primitive.get("type", "block"))
            )
            primitive_path = f"{prefix}{primitive_name}" if prefix else primitive_name
            result.append((primitive, primitive_path))
    return result


def _technology(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = spec.get("technology")
    if not name:
        return dict(spec)
    source = Path(str(spec.get("_source", "design.yaml")))
    try:
        technology = load_technology(
            str(name),
            search_paths=(source.parent / "technology",
                          Path(__file__).resolve().parents[3] / "designs" / "technology"),
        )
    except (FileNotFoundError, ValueError) as error:
        raise DesignSchemaError(str(error)) from error
    technology_parameters = {
        **dict(technology.components),
        **dict(technology.architecture),
    }
    merged = dict(spec)
    current_parameters = merged.get("parameters", {})
    if not isinstance(current_parameters, Mapping):
        raise DesignSchemaError("parameters: expected a mapping")
    merged["parameters"] = {**technology_parameters, **dict(current_parameters)}
    if spec.get("_default_cursors"):
        merged["cursors"] = dict(technology.cursors)
    if spec.get("_default_ground"):
        merged["ground"] = technology.ground
    if "coupler_mode" not in merged:
        merged["coupler_mode"] = technology.coupler_mode
    merged["_technology"] = technology
    merged["_technology_parameters"] = technology_parameters
    return merged


def _profile_target_rows(spec: Mapping[str, Any], target: str) -> tuple[int, int] | None:
    row = 0
    for item in spec.get("topology", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "ipm_line":
            count = int(item.get("rows", item.get("arrays", 0)))
            name = str(item.get("name", ""))
            if target == name:
                return row, row + count - 1
            prefix = f"{name}.row["
            if target.startswith(prefix) and target.endswith("]"):
                try:
                    index = int(target[len(prefix):-1])
                except ValueError:
                    return None
                if 0 <= index < count:
                    return row + index, row + index
            row += count
        elif item.get("type") in {"jj_line", "ipm_row", "ipm_tail"}:
            if item.get("name") == target:
                count = int(item.get("rows", 1))
                return row, row + count - 1
            row += int(item.get("rows", 1))
    return None


def _profile_segments(spec: Mapping[str, Any], parameter: str) -> list[Segment]:
    profiles = spec.get("profiles", {})
    if not isinstance(profiles, Mapping) or parameter not in profiles:
        value = spec.get("parameters", {}).get(parameter)
        if isinstance(value, Mapping) and "profile" in value:
            profiles = {parameter: value["profile"]}
        else:
            return []
    raw = profiles[parameter]
    items = raw if isinstance(raw, list) else [raw]
    segments: list[Segment] = []
    for item in items:
        if isinstance(item, str):
            segments.append(parse_profile_shorthand(item))
            continue
        if not isinstance(item, Mapping):
            raise DesignSchemaError(f"profiles.{parameter}: expected a mapping or list")
        shape = str(item.get("type", item.get("shape", ""))).lower()
        if shape == "half_sine":
            shape, expression = "custom", "sin(pi*t/2)"
        elif shape in {"constant", "const"}:
            shape, expression = "const", None
        else:
            expression = item.get("expression")
        if shape not in {"const", "linear", "custom", "half_cosine", "power",
                         "parabola", "tanh", "sine", "cosine"}:
            raise DesignSchemaError(f"profiles.{parameter}: unknown profile type {shape!r}")
        domain = str(item.get("domain", "selection"))
        if domain not in {"selection", "per_row"}:
            raise DesignSchemaError(f"profiles.{parameter}: unknown domain {domain!r}")
        target = item.get("target", "all")
        selection = Selection()
        if target != "all":
            rows = _profile_target_rows(spec, str(target))
            if rows is None:
                raise DesignSchemaError(
                    f"profiles.{parameter}: target {target!r} was not found")
            selection = Selection(rows=rows)
        if "start" not in item or ("stop" not in item and "end" not in item):
            raise DesignSchemaError(
                f"profiles.{parameter}: start and stop are required")
        segments.append(Segment(shape=shape, start=float(item["start"]),
                                end=float(item.get("stop", item.get("end"))),
                                domain=domain, expression=expression,
                                select=selection))
    return segments


def _design_plan(
    spec: Mapping[str, Any], parameters: dict[str, Any],
    topology: list[dict[str, Any]],
) -> Any:
    scatter_keys = {"lj_scatter_sigma", "cj_scatter_sigma", "cg_scatter_sigma"}
    has_scatter = any(float(parameters.get(key, 0.0)) != 0.0 for key in scatter_keys)
    if not spec.get("profiles") and not any(
        isinstance(value, Mapping) and "profile" in value
        for value in parameters.values()
    ) and not has_scatter:
        return None
    plan_parameters = dict(parameters)
    plan_parameters["jtl_row_count"] = sum(
        int(item.get("rows", item.get("arrays", 1))) if isinstance(item, Mapping)
        and item.get("type") in {"ipm_line", "ipm_tail", "jtl", "jj_line"} else 1
        for item in topology
        if isinstance(item, Mapping)
        and item.get("type") in {"ipm_line", "ipm_tail", "ipm_row", "jtl", "jj_line"}
    )
    params = IPMParams(**{key: value for key, value in plan_parameters.items()
                          if key in IPMParams.__dataclass_fields__})
    from twpa_solver.builders.ipm import build_component_plan
    from twpa_solver.builders.scatter import ScatterSpec
    cj_mode = str(parameters.get("cj_scatter_mode", "independent"))
    distribution = str(parameters.get("scatter_distribution", "normal"))
    clip_min = float(parameters.get("scatter_clip_min", 0.5))
    clip_max = float(parameters.get("scatter_clip_max", 1.5))
    return build_component_plan(
        params,
        lj_segments=_profile_segments(spec, "Lj"),
        cg_segments=_profile_segments(spec, "Cg"),
        lj_scatter=ScatterSpec(float(parameters.get("lj_scatter_sigma", 0.0)),
                               distribution, clip_min, clip_max),
        cj_scatter=ScatterSpec(float(parameters.get("cj_scatter_sigma", 0.0)),
                               distribution, clip_min, clip_max, cj_mode),
        cg_scatter=ScatterSpec(float(parameters.get("cg_scatter_sigma", 0.0)),
                               distribution, clip_min, clip_max),
        seed=int(parameters.get("scatter_seed", 1)),
    )


def _expand_composites(items: list[Any], params: Mapping[str, Any]) -> list[Any]:
    expanded: list[Any] = []
    for item in items:
        if not isinstance(item, Mapping) or item.get("action") is not None:
            expanded.append(item)
            continue
        kind = item.get("type")
        if kind == "input_ports":
            if "cursor" in item:
                cursor = str(item["cursor"])
                resistance = item.get(
                    "resistance",
                    params.get("Rleft" if cursor == "signal" else "Rm", 50.0),
                )
                name = str(item["name"])
                expanded.extend([
                    {
                        "type": "port",
                        "name": f"{name}.port",
                        "cursor": cursor,
                        "port": item["port"],
                    },
                    {
                        "type": "resistor",
                        "name": f"{name}.termination",
                        "cursor": cursor,
                        "value": resistance,
                    },
                ])
                cells = int(item.get(
                    "cpw_cells", params.get(f"{cursor}_input_cpw_cells", 0)
                ))
                if cells > 0:
                    expanded.append({
                        "type": "transmission_line",
                        "name": f"{name}.cpw",
                        "cursor": cursor,
                        "cells": cells,
                        "L": params.get("Ll", 0.0),
                        "C": params.get("Cl", 0.0),
                    })
                continue
            expanded.extend([
                {"type": "port", "name": "input_signal", "cursor": "signal", "port": 1},
                {"type": "resistor", "name": "input_signal_resistor", "cursor": "signal",
                 "value": params.get("Rleft", 50.0)},
                {"type": "transmission_line", "name": "input_signal_tl", "cursor": "signal",
                 "cells": params.get("signal_input_cpw_cells", params.get("len1", 0)),
                 "L": params.get("Ll", 0.0),
                 "C": params.get("Cl", 0.0)},
                {"type": "port", "name": "input_pump", "cursor": "pump", "port": 3},
                {"type": "resistor", "name": "input_pump_resistor", "cursor": "pump",
                 "value": params.get("Rm", 50.0)},
                {"type": "transmission_line", "name": "input_pump_tl", "cursor": "pump",
                 "cells": params.get("pump_input_cpw_cells", params.get("len3", 0)),
                 "L": params.get("Ll", 0.0),
                 "C": params.get("Cl", 0.0)},
            ])
            continue
        if kind == "output_ports":
            if "cursor" in item:
                cursor = str(item["cursor"])
                resistance = item.get(
                    "resistance",
                    params.get("Rright" if cursor == "signal" else "Rm", 50.0),
                )
                name = str(item["name"])
                cells = int(item.get(
                    "cpw_cells", params.get(f"{cursor}_output_cpw_cells", 0)
                ))
                if cells > 0:
                    expanded.append({
                        "type": "transmission_line",
                        "name": f"{name}.cpw",
                        "cursor": cursor,
                        "cells": cells,
                        "L": params.get("Ll", 0.0),
                        "C": params.get("Cl", 0.0),
                    })
                expanded.extend([
                    {
                        "type": "resistor",
                        "name": f"{name}.termination",
                        "cursor": cursor,
                        "value": resistance,
                    },
                    {
                        "type": "port",
                        "name": f"{name}.port",
                        "cursor": cursor,
                        "port": item["port"],
                    },
                ])
                continue
            expanded.extend([
                {"type": "transmission_line", "name": "output_signal_tl", "cursor": "signal",
                 "cells": params.get("signal_output_cpw_cells", params.get("len2", 50)),
                 "L": params.get("Ll", 0.0),
                 "C": params.get("Cl", 0.0)},
                {"type": "resistor", "name": "output_signal_resistor", "cursor": "signal",
                 "value": params.get("Rright", 50.0)},
                {"type": "port", "name": "output_signal", "cursor": "signal", "port": 2},
                {"type": "transmission_line", "name": "output_pump_tl", "cursor": "pump",
                 "cells": params.get("pump_output_cpw_cells", params.get("len4", 0)),
                 "L": params.get("Ll", 0.0),
                 "C": params.get("Cl", 0.0)},
                {"type": "resistor", "name": "output_pump_resistor", "cursor": "pump",
                 "value": params.get("Rm", 50.0)},
                {"type": "port", "name": "output_pump", "cursor": "pump", "port": 4},
            ])
            continue
        if kind in {"signal_port", "pump_port"}:
            expanded.append({"type": "port", "name": item["name"],
                             "cursor": "signal" if kind == "signal_port" else "pump",
                             "port": item["port"]})
            continue
        if kind == "jtl":
            rows = int(item.get("rows", 1))
            cells = int(item.get(
                "cells", params.get("jtl_cells_per_array", params.get("array_length", 0))
            ))
            if rows < 1:
                raise DesignSchemaError(
                    f"{item.get('name', kind)}.rows must be positive"
                )
            if cells < 1:
                raise DesignSchemaError(
                    f"{item.get('name', kind)}.cells must be positive"
                )
            name = str(item["name"])
            cursor = str(item["cursor"])
            for index in range(rows):
                row = {
                    "type": "jj_line",
                    "name": f"{name}.row[{index}]",
                    "cursor": cursor,
                    "cells": cells,
                    "Lj": item.get("Lj", params.get("Lj")),
                    "Cj": item.get("Cj", params.get("Cj")),
                    "Cg": item.get("Cg", params.get("Cg")),
                }
                row.update({key: item[key] for key in (
                    "lj_scatter_sigma", "cj_scatter_sigma", "cg_scatter_sigma",
                    "cj_scatter_mode", "scatter_seed", "tan_delta",
                ) if key in item})
                if index == 0 and item.get("join_cursors"):
                    row["join_cursors"] = item["join_cursors"]
                expanded.append(row)
            continue
        if kind in {"ipm_line", "ipm_tail"} and (
            "rows" in item or kind == "ipm_tail"
            or (kind == "ipm_line" and "arrays" not in item)
        ):
            count = int(item.get("rows", params.get("jtl_rows_per_coupler", 3)))
            if count < 1:
                raise DesignSchemaError(f"{item.get('name', kind)}.rows must be positive")
            name = str(item["name"])
            cursor = item.get("cursor", "signal")
            for index in range(count):
                row = {"type": "jj_line",
                       "name": f"{name}.row[{index}].array[0]",
                       "cursor": cursor,
                       "cells": item.get("cells", params.get(
                           "jtl_cells_per_array", params.get("array_length", 0))),
                       "Lj": item.get("Lj", params.get("Lj")),
                       "Cj": item.get("Cj", params.get("Cj")),
                       "Cg": item.get("Cg", params.get("Cg"))}
                row.update({key: item[key] for key in (
                    "lj_scatter_sigma", "cj_scatter_sigma", "cg_scatter_sigma",
                    "cj_scatter_mode", "scatter_seed", "tan_delta",
                ) if key in item})
                expanded.append(row)
                if index + 1 < count:
                    expanded.append({"type": "transmission_line",
                                     "name": f"{name}.row[{index}].short_tl",
                                     "cursor": cursor,
                                     "cells": item.get("between", params.get(
                                         "inter_array_cpw_cells",
                                         params.get("length_of_short_TL", 0),
                                     )),
                                     "L": params.get("Ll", 0.0), "C": params.get("Cl", 0.0)})
            if kind == "ipm_line" and item.get("end_coupler", True):
                expanded.extend([
                    {"type": "transmission_line", "name": f"{name}.long_tl",
                     "cursor": cursor, "cells": item.get(
                         "trailing_signal_cpw_cells", params.get(
                             "signal_inter_coupler_cpw_cells",
                             params.get("length_of_long_TL", 0),
                         )
                     ),
                     "L": params.get("Ll", 0.0), "C": params.get("Cl", 0.0)},
                    {"type": "transmission_line", "name": f"{name}.pump_section",
                     "cursor": "pump", "cells": item.get(
                         "trailing_pump_cpw_cells", params.get(
                             "pump_inter_coupler_cpw_cells",
                             params.get("coupler_section_length", 0),
                         )
                     ),
                     "L": params.get("Ll", 0.0), "C": params.get("Cl", 0.0)},
                ])
            continue
        if kind == "ipm_line":
            count = int(item["arrays"])
            if count < 1:
                raise DesignSchemaError(f"{item.get('name', 'ipm_line')}.arrays must be positive")
            name = str(item["name"])
            between = item.get(
                "between", params.get(
                    "inter_array_cpw_cells",
                    params.get("length_of_short_TL", params.get("short_tl", 0)),
                )
            )
            for index in range(count):
                lower = {key: value for key, value in item.items()
                         if key not in {"type", "arrays", "between"}}
                lower.update({"type": "jj_line", "name": f"{name}.array[{index}]"})
                lower.setdefault("cells", item.get(
                    "cells", params.get("jtl_cells_per_array", params.get("array_length", 0))
                ))
                expanded.append(lower)
                if index + 1 < count and int(between) > 0:
                    expanded.append({"type": "transmission_line",
                                     "name": f"{name}.link[{index}]", "cursor": item["cursor"],
                                     "cells": int(between), "L": params.get("Ll", 0.0),
                                     "C": params.get("Cl", 0.0)})
            continue
        if kind == "ipm_row":
            expanded.append({**item, "type": "jj_line"})
            continue
        expanded.append(item)
    return expanded


@dataclass
class _BlockSpec:
    """Symbolic block record retained until legacy node numbers are assigned."""

    path: str
    kind: str
    cursors: list[str]
    starts: dict[str, Node]
    ends: dict[str, Node]
    count: int = 1


@dataclass
class _AdapterState:
    """State used while adapting YAML blocks to a symbolic Circuit."""

    circuit: Circuit
    paths: dict[str, Any]
    legacy_nodes: dict[int, Node]
    named_nodes: dict[str, Node]
    named_elements: dict[str, ElementRef]
    blocks: list[_BlockSpec]
    cell_index: int = 0
    coupler_geometry: dict[str, Any] | None = None
    coupler_settings: dict[str, Any] | None = None
    loss_by_role: dict[str, float] = field(default_factory=dict)

    def register_node(self, path: str, node: Node) -> None:
        self.named_nodes[path] = node

    def register_element(self, path: str, element: ElementRef) -> None:
        self.named_elements[path] = element

    def register_path(self, name: str) -> None:
        path = self.paths[name]
        base = self.circuit.graph.legacy_path_bases[name]
        for offset, node in enumerate(path.nodes):
            solver_node = base + offset
            prior = self.legacy_nodes.get(solver_node)
            if prior is not None and prior is not node:
                raise DesignCollisionError(
                    f"cursor collision at node {solver_node}: "
                    f"{prior.path!r} overlaps {node.path!r}"
                )
            self.legacy_nodes[solver_node] = node

    def refresh_path(self, name: str) -> None:
        self.register_path(name)

    def resolve_node(self, value: Any, path: str) -> Node:
        if isinstance(value, Node):
            return value
        if isinstance(value, str):
            if value == "ground":
                return self.circuit.ground
            try:
                return self.named_nodes[value]
            except KeyError as error:
                raise DesignResolutionError(
                    f"{path}: unknown node path {value!r}"
                ) from error
        try:
            solver_node = int(value)
        except (TypeError, ValueError) as error:
            raise DesignResolutionError(
                f"{path}: expected a node number or named node path"
            ) from error
        if solver_node == 0:
            return self.circuit.ground
        try:
            return self.legacy_nodes[solver_node]
        except KeyError as error:
            raise DesignResolutionError(
                f"{path}: unknown legacy node {solver_node}"
            ) from error

    def resolve_element(self, value: Any, path: str) -> ElementRef:
        if isinstance(value, ElementRef):
            return value
        if not isinstance(value, str):
            raise DesignResolutionError(f"{path}: expected an element path or name")
        element = self.named_elements.get(value)
        if element is not None:
            return element
        for candidate in self.circuit.graph.elements:
            if candidate.name == value:
                return candidate
        raise DesignResolutionError(f"{path}: unknown element {value!r}")

    def record_block(
        self,
        path: str,
        kind: str,
        cursors: list[str],
        starts: dict[str, Node],
        ends: dict[str, Node],
        count: int = 1,
    ) -> None:
        self.blocks.append(_BlockSpec(path, kind, cursors, starts, ends, count))


def _register_line(
    state: _AdapterState,
    path: str,
    cursor: str,
    line: LineHandle,
    kind: str,
) -> None:
    """Register public line nodes and aliases used by the YAML surface."""

    state.register_node(path, line.input)
    state.register_node(f"{path}.end", line.output)
    for index, cell in enumerate(line.cells):
        if cell.left is not None:
            state.register_node(f"{path}.cell[{index}].left", cell.left)
        if cell.right is not None:
            state.register_node(f"{path}.cell[{index}].right", cell.right)
        if cell.Cg is not None:
            state.register_element(f"{path}.cell[{index}].Cg", cell.Cg)
        if cell.Lj is not None:
            state.register_element(f"{path}.cell[{index}].Lj", cell.Lj)
            state.register_element(f"{path}.cell[{index}].right", cell.Lj)
        if cell.Cj is not None:
            state.register_element(f"{path}.cell[{index}].Cj", cell.Cj)
        for name, element in cell.extras.items():
            state.register_element(f"{path}.cell[{index}].{name}", element)
    state.record_block(
        path,
        kind,
        [cursor],
        {cursor: line.input},
        {cursor: line.output},
        len(line.cells),
    )
    state.refresh_path(cursor)


def _join_adapter_paths(
    state: _AdapterState,
    cursor_names: list[str],
    path: str,
) -> None:
    """Join cursor endpoints and retain the first cursor as the shared path."""

    if len(cursor_names) < 2:
        raise DesignSchemaError(f"{path}.join_cursors: expected at least two cursors")
    if len(set(cursor_names)) != len(cursor_names):
        raise DesignSchemaError(f"{path}.join_cursors: cursor names must be unique")
    if any(cursor not in state.paths for cursor in cursor_names):
        raise DesignSchemaError(f"{path}.join_cursors: unknown cursor name")
    primary = state.paths[cursor_names[0]]
    for cursor in cursor_names[1:]:
        secondary = state.paths[cursor]
        if secondary is primary:
            continue
        removed = secondary.end
        retained = state.circuit.join_path_ends(primary, secondary)
        for name, node in tuple(state.named_nodes.items()):
            if node is removed:
                state.named_nodes[name] = retained
        for number, node in tuple(state.legacy_nodes.items()):
            if node is removed:
                state.legacy_nodes[number] = retained
        for block in state.blocks:
            block.starts = {
                name: retained if node is removed else node
                for name, node in block.starts.items()
            }
            block.ends = {
                name: retained if node is removed else node
                for name, node in block.ends.items()
            }
        state.paths[cursor] = primary


def _apply_plan_values(
    state: _AdapterState,
    line: LineHandle,
    plan: Any,
    offset: int,
) -> None:
    """Apply the existing component plan through public element handles."""

    if plan is None:
        return
    end = offset + len(line.cells)
    if end > len(plan.lj) or end > len(plan.cj) or end > len(plan.cg):
        raise DesignSchemaError("component plan is shorter than the YAML JJ topology")
    for index, cell in enumerate(line.cells):
        if cell.Lj is not None:
            state.circuit.set_value(cell.Lj, float(plan.lj[offset + index]))
        if cell.Cj is not None:
            state.circuit.set_value(cell.Cj, float(plan.cj[offset + index]))
        if cell.Cg is not None:
            cg = float(plan.cg[offset + index])
            state.circuit.set_value(cell.Cg, cg / (2.0 if index == 0 else 1.0))
    end_ref = state.circuit.graph.named_elements.get(f"{line.path}.end.Cg")
    if end_ref is not None:
        state.circuit.set_value(end_ref, float(plan.cg[end - 1]) / 2.0)


def _apply_local_component_settings(
    state: _AdapterState,
    line: LineHandle,
    cfg: Mapping[str, Any],
    path: str,
    parameters: Mapping[str, Any],
) -> None:
    """Apply optional per-line scatter and dielectric loss settings."""

    local_seed = int(cfg.get("scatter_seed", parameters.get("scatter_seed", 1)))
    block_seed = local_seed ^ zlib.crc32(path.encode("utf-8"))
    distribution = str(parameters.get("scatter_distribution", "normal"))
    clip_min = float(parameters.get("scatter_clip_min", 0.5))
    clip_max = float(parameters.get("scatter_clip_max", 1.5))
    lj_sigma = float(cfg.get("lj_scatter_sigma", 0.0))
    cj_sigma = float(cfg.get("cj_scatter_sigma", 0.0))
    cg_sigma = float(cfg.get("cg_scatter_sigma", 0.0))
    cj_mode = str(cfg.get("cj_scatter_mode", parameters.get(
        "cj_scatter_mode", "independent"
    )))
    if lj_sigma or cj_sigma or cg_sigma:
        lj_factors = draw_factors(
            ScatterSpec(lj_sigma, distribution, clip_min, clip_max),
            len(line.cells), component_rng(block_seed, "Lj"),
        )
        if cj_mode == "plasma_locked":
            if cj_sigma:
                raise DesignSchemaError(
                    f"{path}.cj_scatter_sigma must be zero in plasma_locked mode"
                )
            cj_factors = 1.0 / lj_factors
        else:
            cj_factors = draw_factors(
                ScatterSpec(cj_sigma, distribution, clip_min, clip_max),
                len(line.cells), component_rng(block_seed, "Cj"),
            )
        cg_factors = draw_factors(
            ScatterSpec(cg_sigma, distribution, clip_min, clip_max),
            len(line.cells), component_rng(block_seed, "Cg"),
        )
        for index, cell in enumerate(line.cells):
            if cell.Lj is not None:
                state.circuit.set_value(cell.Lj, float(cell.Lj.value) * lj_factors[index])
            if cell.Cj is not None:
                state.circuit.set_value(cell.Cj, float(cell.Cj.value) * cj_factors[index])
            if cell.Cg is not None:
                state.circuit.set_value(cell.Cg, float(cell.Cg.value) * cg_factors[index])

    if "tan_delta" in cfg:
        role = f"jtl_cg@{path}"
        tangent = float(cfg["tan_delta"])
        if tangent < 0.0:
            raise DesignSchemaError(f"{path}.tan_delta must be non-negative")
        state.loss_by_role[role] = tangent
        for cell in line.cells:
            if cell.Cg is not None:
                cell.Cg.role = role
        end_ref = state.circuit.graph.named_elements.get(f"{line.path}.end.Cg")
        if end_ref is not None:
            end_ref.role = role


def _add_raw_element(
    state: _AdapterState,
    cfg: Mapping[str, Any],
    path: str,
) -> ElementRef:
    """Emit one YAML raw element through a public primitive builder."""

    nodes = cfg.get("nodes")
    if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
        raise DesignSchemaError(f"{path}.nodes: expected two endpoints")
    kind = str(cfg["kind"])
    name = str(cfg.get("name", path))
    first = state.resolve_node(nodes[0], f"{path}.nodes[0]")
    if kind == "mutual_inductor_k":
        first_element = state.resolve_element(nodes[0], f"{path}.nodes[0]")
        second_element = state.resolve_element(nodes[1], f"{path}.nodes[1]")
        if first_element.kind != "linear_inductor" or second_element.kind != "linear_inductor":
            raise DesignSchemaError(f"{path}.nodes: mutual endpoints must be linear inductors")
        result = state.circuit.add_mutual_inductor(
            first_element,
            second_element,
            float(cfg["value"]),
            name=name,
            path=path,
        )
        result.value = cfg["value"]
        return result
    second = state.resolve_node(nodes[1], f"{path}.nodes[1]")
    value = float(cfg["value"])
    if kind == "capacitor":
        result = state.circuit.add_capacitor(first, second, value, name=name, path=path)
        result.value = cfg["value"]
        return result
    if kind == "coupling_capacitor":
        result = state.circuit.add_coupling_capacitor(
            first, second, value, name=name, path=path
        )
        result.value = cfg["value"]
        return result
    if kind == "linear_inductor":
        result = state.circuit.add_inductor(first, second, value, name=name, path=path)
        result.value = cfg["value"]
        return result
    if kind == "josephson_inductor":
        result = state.circuit.add_josephson_inductor(
            first, second, value, name=name, path=path
        )
        result.value = cfg["value"]
        return result
    if kind == "resistor":
        result = state.circuit.add_resistor(first, second, value, name=name, path=path)
        result.value = cfg["value"]
        return result
    if kind == "port":
        state.circuit.add_port(first, number=int(value), name=name)
        return state.circuit.graph.elements[-1]
    raise DesignSchemaError(f"{path}.kind: unsupported element kind {kind!r}")


def _apply_adapter_action(
    state: _AdapterState,
    cfg: Mapping[str, Any],
    path: str,
) -> None:
    """Apply an inline or document patch through Circuit methods."""

    action = cfg.get("action")
    if action == "add":
        ref = _add_raw_element(state, cfg, path)
        state.register_element(path, ref)
        return
    target = cfg.get("target")
    if not isinstance(target, str):
        raise DesignResolutionError(f"{path}.target: expected an exact path")
    try:
        ref = state.resolve_element(target, path)
    except DesignResolutionError as error:
        raise DesignSchemaError(
            f"{path}.target {target!r}: expected one match, found 0"
        ) from error
    if action == "set":
        state.circuit.set_value(ref, cfg["value"])
    elif action == "remove":
        state.circuit.remove(ref)
    else:
        raise DesignSchemaError(f"{path}.action: unknown action {action!r}")


def _coupler_metadata(state: _AdapterState) -> dict[str, Any]:
    """Return the stable coupler metadata exposed by the old design result."""

    return dict(state.coupler_geometry or {})


def _add_first_array_aliases_adapter(state: _AdapterState) -> None:
    """Preserve compact and historical aliases from the YAML compiler."""

    for table in (state.named_nodes, state.named_elements):
        additions: dict[str, Any] = {}
        for key, value in table.items():
            marker = ".array[0]."
            if marker in key:
                additions.setdefault(key.replace(marker, ".", 1), value)
        table.update(additions)
        legacy: dict[str, Any] = {}
        for key, value in table.items():
            if key.startswith("line_1.row["):
                legacy[
                    key.replace("line_1.row[", "period[0].row[", 1)
                    .replace(".array[0].", ".array.", 1)
                ] = value
        table.update(legacy)


def _emit_adapter_block(
    state: _AdapterState,
    cfg: dict[str, Any],
    path: str,
    parameters: Mapping[str, Any],
    coupler_mode: str,
    plan: Any,
) -> None:
    """Emit one expanded YAML block with public Circuit builders."""

    kind = str(cfg.get("type"))
    if kind in {"signal_port", "pump_port"}:
        cfg = {**cfg, "type": "port", "cursor": (
            "signal" if kind == "signal_port" else "pump"
        )}
        kind = "port"
    if kind in {"capacitor", "inductor"}:
        cfg = {
            **cfg,
            "type": "raw_element",
            "value": cfg["C"] if kind == "capacitor" else cfg["L"],
            "kind": "capacitor" if kind == "capacitor" else "linear_inductor",
        }
        kind = "raw_element"
    if kind == "coupler":
        kind = "directional_coupler"
    if kind == "directional_coupler":
        cursor_names = [str(item) for item in cfg.get("cursors", ["signal", "pump"])]
        if len(cursor_names) != 2:
            raise DesignSchemaError(f"{path}.cursors: expected two cursor names")
        signal = state.paths[cursor_names[0]]
        pump = state.paths[cursor_names[1]]
        # A block may override the technology/design defaults locally.  The
        # line-scoped normalizer preserves these fields; references on the
        # other physical line contain only the shared block name.
        coupler_parameters = {
            key: cfg[key]
            for key in ("coupling_dB", "coupler_freq_hz", "Z0", "cell_length_um")
            if key in cfg
        }
        effective_coupler_parameters = {**parameters, **coupler_parameters}
        before = len(state.circuit.graph.elements)
        effective_coupler_settings = {
            "coupling_dB": float(effective_coupler_parameters.get("coupling_dB", -14.0)),
            "coupler_freq_hz": float(effective_coupler_parameters.get("coupler_freq_hz", 8.0e9)),
            "Z0": float(effective_coupler_parameters.get("Z0", 50.0)),
            "cell_length_um": float(effective_coupler_parameters.get("cell_length_um", 10.0)),
        }
        handle = state.circuit.add_directional_coupler(
            signal,
            pump,
            coupling_db=effective_coupler_settings["coupling_dB"],
            frequency=effective_coupler_settings["coupler_freq_hz"],
            z0=effective_coupler_settings["Z0"],
            mode=coupler_mode,
            cell_length_um=effective_coupler_settings["cell_length_um"],
            name=path,
        )
        geometry = handle.geometry
        state.coupler_settings = effective_coupler_settings
        state.coupler_geometry = {
            "width_um": geometry.width_um,
            "gap_between_lines_um": geometry.gap_between_lines_um,
            "gap_to_ground_um": geometry.gap_to_ground_um,
            "length_um": geometry.length_um,
            "coupling_db": geometry.k_db,
            "z_input_ohm": geometry.z_input_ohm,
            "model": geometry.model,
        }
        for index, element in enumerate(state.circuit.graph.elements[before:]):
            state.register_element(f"{path}.element[{index}]", element)
        state.record_block(
            path,
            "directional_coupler",
            cursor_names,
            {cursor_names[0]: handle.signal_in, cursor_names[1]: handle.pump_in},
            {cursor_names[0]: handle.signal_out, cursor_names[1]: handle.pump_out},
            len(handle.cells),
        )
        state.refresh_path(cursor_names[0])
        state.refresh_path(cursor_names[1])
        return

    cursor_name = str(cfg["cursor"]) if "cursor" in cfg else ""
    if cursor_name and cursor_name not in state.paths:
        raise DesignSchemaError(f"{path}.cursor: unknown cursor {cursor_name!r}")
    path_obj = state.paths.get(cursor_name)
    if kind == "port":
        if path_obj is None:
            raise DesignSchemaError(f"{path}.cursor: required")
        node = path_obj.end
        state.circuit.add_port(node, number=int(cfg["port"]))
        state.register_node(path, node)
        state.record_block(path, "port", [cursor_name], {cursor_name: node}, {cursor_name: node})
        return
    if kind == "resistor":
        if path_obj is None:
            raise DesignSchemaError(f"{path}.cursor: required")
        node = path_obj.end
        state.circuit.add_resistor(node, state.circuit.ground, float(cfg["value"]))
        state.register_node(path, node)
        state.record_block(
            path, "resistor", [cursor_name], {cursor_name: node}, {cursor_name: node}
        )
        return
    if kind == "transmission_line":
        line = state.circuit.add_transmission_line(
            path_obj,
            cells=int(cfg["cells"]),
            L=float(cfg["L"]) if cfg.get("L") is not None else None,
            C=float(cfg["C"]) if cfg.get("C") is not None else None,
            name=path,
        )
        _register_line(state, path, cursor_name, line, kind)
        return
    if kind == "jj_line":
        join_cursors = [str(value) for value in cfg.get("join_cursors", [])]
        if join_cursors:
            _join_adapter_paths(state, join_cursors, path)
            path_obj = state.paths[cursor_name]
        start_index = state.cell_index
        line = state.circuit.add_jj_line(
            path_obj,
            cells=int(cfg["cells"]),
            Lj=float(cfg["Lj"]) if cfg.get("Lj") is not None else None,
            Cj=float(cfg["Cj"]) if cfg.get("Cj") is not None else None,
            Cg=float(cfg["Cg"]) if cfg.get("Cg") is not None else None,
            cell_index_start=start_index,
            name=path,
        )
        _apply_plan_values(state, line, plan, start_index)
        _apply_local_component_settings(state, line, cfg, path, parameters)
        state.cell_index += len(line.cells)
        _register_line(state, path, cursor_name, line, kind)
        return
    if kind == "rf_squid_line":
        before = len(state.circuit.graph.elements)
        line = state.circuit.add_rf_squid_line(
            path_obj,
            cells=int(cfg["cells"]),
            Ic=float(cfg["Ic"]),
            Lj=float(cfg["Lj"]) if cfg.get("Lj") is not None else None,
            Lm=float(cfg["Lm"]),
            Lw=float(cfg["Lw"]),
            Lpar=float(cfg["Lpar"]),
            Cj=float(cfg["Cj"]),
            Cg=float(cfg["Cg"]) if cfg.get("Cg") is not None else None,
            Cg_pattern=cfg.get("Cg_pattern"),
            Cg_pattern_counts=cfg.get("Cg_pattern_counts"),
            name=path,
        )
        for cell in line.cells:
            refs = [cell.Lj, cell.Cj, cell.Cg, *cell.extras.values()]
            for ref in refs:
                if ref is not None:
                    ref.cell_index = state.cell_index
            state.cell_index += 1
        start_number = next(
            number for number, node in state.legacy_nodes.items()
            if node is line.input
        )
        exceptional_numbers: dict[Node, int] = {}
        for index, node in enumerate(line._nodes):
            exceptional_numbers[node] = start_number + 3 * index
            if index == len(line._nodes) - 1:
                continue
            cell_path = f"{path}.cell[{index}]"
            exceptional_numbers[state.circuit.graph.named_nodes[f"{cell_path}.wire"]] = (
                start_number + 3 * index + 1
            )
            exceptional_numbers[state.circuit.graph.named_nodes[f"{cell_path}.branch"]] = (
                start_number + 3 * index + 2
            )
        state.circuit.set_legacy_node_numbers(exceptional_numbers)
        _register_line(state, path, cursor_name, line, kind)
        for index, element in enumerate(state.circuit.graph.elements[before:]):
            state.register_element(f"{path}.element[{index}]", element)
        return
    if kind == "raw_element":
        ref = _add_raw_element(state, cfg, path)
        state.register_element(path, ref)
        state.record_block(path, "raw_element", [], {}, {})
        return
    raise DesignSchemaError(f"{path}.type: unknown block type {kind!r}")


def compile_design(spec: Mapping[str, Any], *, coupler_mode: str | None = None,
                   strict: bool = False, plan: Any = None) -> CompiledDesign:
    # Keep the distinction between parameters declared by the design author
    # and defaults injected by a technology preset.  Strict validation applies
    # only to the former; a preset is a shared parameter catalogue, not a
    # second set of declarations that every design must reference explicitly.
    spec = normalize_design(spec)
    declared_parameters = set(spec.get("parameters", {}))
    spec = _technology(spec)
    validate_design(spec)
    raw_parameters = dict(spec.get("parameters", {}))
    for key, value in list(raw_parameters.items()):
        if isinstance(value, Mapping) and "profile" in value:
            raw_parameters[key] = value.get("value", value["profile"].get("start"))
    base_parameters = dict(spec.get("_technology_parameters", {}))
    parameters = resolve_parameter_definitions(
        raw_parameters,
        base_parameters,
        "parameters",
    )
    resolved_source_topology = resolve_parameters(
        spec["topology"], parameters, "topology", base_parameters=base_parameters
    )
    expanded_topology = _expand_composite_topology(resolved_source_topology, parameters)
    topology = [cfg for cfg, _ in expanded_topology]
    design_plan = _design_plan(spec, parameters, topology)
    if strict:
        from twpa_solver.design.parameters import parameter_references
        used = parameter_references(spec["topology"])
        # Composite blocks consume parameters inside their compiler expansion,
        # so those references are not visible in the compact source topology.
        implicit_by_block = {
            "input_ports": {"Rleft", "Rm", "Ll", "Cl", "signal_input_cpw_cells",
                             "pump_input_cpw_cells"},
            "output_ports": {"Rright", "Rm", "Ll", "Cl", "signal_output_cpw_cells",
                              "pump_output_cpw_cells"},
            "directional_coupler": {"coupling_dB", "Z0", "coupler_freq_hz",
                                    "cell_length_um"},
            "coupler": {"coupling_dB", "Z0", "coupler_freq_hz",
                         "cell_length_um"},
            "ipm_line": {"jtl_cells_per_array", "Lj", "Cj", "Cg", "Ll", "Cl",
                         "inter_array_cpw_cells", "signal_inter_coupler_cpw_cells",
                         "pump_inter_coupler_cpw_cells"},
            "ipm_tail": {"jtl_cells_per_array", "Lj", "Cj", "Cg", "Ll", "Cl",
                         "inter_array_cpw_cells"},
                "jtl": {"Lj", "Cj", "Cg"},
                "jj_line": {"Lj", "Cj", "Cg"},
                "ipm_row": {"Lj", "Cj", "Cg"},
                "transmission_line": {"Ll", "Cl"},
        }
        for item in topology:
            if isinstance(item, Mapping):
                used.update(implicit_by_block.get(str(item.get("type")), set()))
        used.update({
            "tan_delta", "tan_delta_jtl_cg", "tan_delta_coupling_cap",
            "tan_delta_by_role", "lj_scatter_sigma", "cj_scatter_sigma",
            "cg_scatter_sigma", "cj_scatter_mode", "scatter_distribution",
            "scatter_clip_min", "scatter_clip_max", "scatter_seed",
        })
        unused = sorted(declared_parameters - used)
        if unused:
            raise DesignParameterError(f"parameters: declared but unused {unused}")
    cursors = {str(key): int(value) for key, value in spec["cursors"].items()}
    coupler_choice = str(coupler_mode or spec.get(
        "coupler_mode", parameters.get("coupler_mode", "auto")))
    if coupler_choice not in {"auto", "ideal", "optimize"}:
        raise DesignSchemaError(
            "coupler_mode: expected 'auto', 'ideal', or 'optimize'"
        )
    effective_plan = plan or design_plan
    technology = spec.get("_technology")
    if technology is not None and not isinstance(technology, Technology):
        raise DesignSchemaError("_technology: invalid loaded technology")
    circuit = Circuit(str(spec["name"]), technology=technology)
    circuit.set_design_parameters(parameters)
    paths = {name: circuit.path(name) for name in cursors}
    circuit.set_legacy_path_bases(cursors)
    state = _AdapterState(
        circuit=circuit,
        paths=paths,
        legacy_nodes={},
        named_nodes={"ground": circuit.ground},
        named_elements={},
        blocks=[],
    )
    for cursor in paths:
        state.register_node(f"{cursor}.start", paths[cursor].start)
        state.register_path(cursor)

    for cfg, generated_path in expanded_topology:
        if cfg.get("action") is not None:
            _apply_adapter_action(state, cfg, generated_path)
            continue
        block = _config_for_block(dict(cfg))
        _emit_adapter_block(
            state,
            block,
            generated_path,
            parameters,
            coupler_choice,
            effective_plan,
        )

    for index, patch in enumerate(spec.get("patches", [])):
        _apply_adapter_action(state, patch, f"patches[{index}]")

    _add_first_array_aliases_adapter(state)
    state.named_nodes.update({
        path: node for path, node in circuit.graph.named_nodes.items()
    })
    for path, element in circuit.graph.named_elements.items():
        state.named_elements.setdefault(path, element)
    compiled = circuit.compile(node_numbering="legacy")
    active_refs = [
        ref for ref in circuit.graph.elements if not ref.removed
    ]
    compiled_names = {
        id(ref): element.name
        for ref, element in zip(active_refs, compiled.elements)
    }
    compiled_node_numbers = {
        id(node): number for node, number in compiled.node_map.items()
    }
    named_nodes = {
        path: compiled_node_numbers[id(node)]
        for path, node in state.named_nodes.items()
        if id(node) in compiled_node_numbers
    }
    named_elements = {
        path: compiled_names.get(id(element), element.name)
        for path, element in state.named_elements.items()
    }
    blocks = [
        BlockRecord(
            item.path,
            item.kind,
            item.cursors,
            {cursor: compiled.node_map[node] for cursor, node in item.starts.items()},
            {cursor: compiled.node_map[node] for cursor, node in item.ends.items()},
            item.count,
        )
        for item in state.blocks
    ]
    ports = {
        number: PortRecord(port.number, port.node)
        for number, port in compiled.ports.items()
    }
    metadata = {
        "schema_version": 1,
        "parameters": parameters,
        "technology": spec.get("technology"),
        "technology_parameters": spec.get("_technology_parameters", {}),
        "coupler_mode": coupler_choice,
        "source": spec.get("_source"),
        "coupler_geometry": _coupler_metadata(state),
        "coupler_settings": dict(state.coupler_settings or {}),
        "loss": {
            "default": float(parameters.get("tan_delta", 0.0)),
            "by_role": {
                "jtl_cg": float(parameters.get("tan_delta_jtl_cg", 0.0)),
                "coupling_cap": float(parameters.get("tan_delta_coupling_cap", 0.0)),
                **dict(parameters.get("tan_delta_by_role", {})),
                **state.loss_by_role,
            },
        },
        "profiles": spec.get("profiles", {}),
        "component_plan": (
            effective_plan.metadata if effective_plan is not None else {}
        ),
    }
    return CompiledDesign(
        str(spec["name"]),
        compiled.elements,
        {
            cursor: compiled.node_map[paths[cursor].end]
            for cursor in paths
        },
        named_nodes,
        named_elements,
        ports,
        blocks,
        metadata,
        effective_plan,
    )
