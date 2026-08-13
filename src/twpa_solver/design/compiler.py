from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from twpa_solver.builders.blocks import BuildContext
from twpa_solver.builders.coupler import make_coupler_discrete
from twpa_solver.builders.ipm import Element, IPMParams
from twpa_solver.builders.profiles import Selection, Segment, parse_profile_shorthand
from twpa_solver.builders.scatter import ScatterSpec
from twpa_solver.builders.registry import BLOCK_BUILDERS
from twpa_solver.design.errors import (DesignCollisionError, DesignParameterError,
                                       DesignSchemaError)
from twpa_solver.design.model import CompiledDesign, PortRecord
from twpa_solver.design.parameters import resolve_parameters
from twpa_solver.design.patches import apply_patches
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


def _config_for_block(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    block = dict(cfg)
    kind = block.get("type")
    if kind == "transmission_line":
        block.setdefault("L", params.get("Ll", params.get("L", 0.0)))
        block.setdefault("C", params.get("Cl", params.get("C", 0.0)))
    if kind == "jj_line":
        for key in ("Lj", "Cj", "Cg"):
            block.setdefault(key, params.get(key))
    return block


def _technology(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = spec.get("technology")
    if not name:
        return dict(spec)
    source = Path(str(spec.get("_source", "design.yaml")))
    path = source.parent / "technology" / f"{name}.yaml"
    if not path.exists():
        path = Path(__file__).resolve().parents[3] / "designs" / "technology" / f"{name}.yaml"
    if not path.exists():
        raise DesignSchemaError(f"technology preset not found: {name!r} ({path})")
    try:
        preset = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DesignSchemaError(f"{path}: invalid technology YAML: {exc}") from exc
    if not isinstance(preset, Mapping):
        raise DesignSchemaError(f"{path}: technology preset must be a mapping")
    unknown = set(preset) - {"name", "parameters", "cursors", "ground",
                             "coupler_mode"}
    if unknown:
        raise DesignSchemaError(f"technology {name!r}: unknown keys {sorted(unknown)}")
    merged = dict(spec)
    for key in ("parameters", "cursors"):
        defaults = preset.get(key, {})
        current = merged.get(key, {})
        if isinstance(defaults, Mapping) and isinstance(current, Mapping):
            merged[key] = {**defaults, **current}
    if spec.get("_default_cursors") and "cursors" in preset:
        merged["cursors"] = dict(preset["cursors"])
    for key in ("ground",):
        if key not in merged and key in preset:
            merged[key] = preset[key]
    if spec.get("_default_ground") and "ground" in preset:
        merged["ground"] = preset["ground"]
    if "coupler_mode" not in merged and "coupler_mode" in preset:
        merged["coupler_mode"] = preset["coupler_mode"]
    merged["_technology_parameters"] = dict(preset.get("parameters", {}))
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


def _design_plan(spec: Mapping[str, Any], parameters: dict[str, Any]) -> Any:
    if not spec.get("profiles") and not any(
        isinstance(value, Mapping) and "profile" in value
        for value in parameters.values()
    ):
        return None
    plan_parameters = dict(parameters)
    if "num_rows" not in plan_parameters:
        plan_parameters["num_rows"] = sum(
            int(item.get("rows", item.get("arrays", 1))) if isinstance(item, Mapping)
            and item.get("type") in {"ipm_line", "ipm_tail"} else 1
            for item in spec.get("topology", [])
            if isinstance(item, Mapping)
            and item.get("type") in {"ipm_line", "ipm_tail", "ipm_row", "jj_line"}
        )
    params = IPMParams(**{key: value for key, value in plan_parameters.items()
                          if key in IPMParams.__dataclass_fields__})
    from twpa_solver.builders.ipm import build_component_plan
    return build_component_plan(
        params,
        lj_segments=_profile_segments(spec, "Lj"),
        cg_segments=_profile_segments(spec, "Cg"),
    )


def _expand_composites(items: list[Any], params: Mapping[str, Any]) -> list[Any]:
    expanded: list[Any] = []
    for item in items:
        if not isinstance(item, Mapping) or item.get("action") is not None:
            expanded.append(item)
            continue
        kind = item.get("type")
        if kind == "input_ports":
            expanded.extend([
                {"type": "port", "name": "input_signal", "cursor": "signal", "port": 1},
                {"type": "resistor", "name": "input_signal_resistor", "cursor": "signal",
                 "value": params.get("Rleft", 50.0)},
                {"type": "transmission_line", "name": "input_signal_tl", "cursor": "signal",
                 "cells": params.get("len1", 0), "L": params.get("Ll", 0.0),
                 "C": params.get("Cl", 0.0)},
                {"type": "port", "name": "input_pump", "cursor": "pump", "port": 3},
                {"type": "resistor", "name": "input_pump_resistor", "cursor": "pump",
                 "value": params.get("Rm", 50.0)},
                {"type": "transmission_line", "name": "input_pump_tl", "cursor": "pump",
                 "cells": params.get("len3", 0), "L": params.get("Ll", 0.0),
                 "C": params.get("Cl", 0.0)},
            ])
            continue
        if kind == "output_ports":
            expanded.extend([
                {"type": "transmission_line", "name": "output_signal_tl", "cursor": "signal",
                 "cells": params.get("len2", 50), "L": params.get("Ll", 0.0),
                 "C": params.get("Cl", 0.0)},
                {"type": "resistor", "name": "output_signal_resistor", "cursor": "signal",
                 "value": params.get("Rright", 50.0)},
                {"type": "port", "name": "output_signal", "cursor": "signal", "port": 2},
                {"type": "transmission_line", "name": "output_pump_tl", "cursor": "pump",
                 "cells": params.get("len4", 0), "L": params.get("Ll", 0.0),
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
        if kind in {"ipm_line", "ipm_tail"} and ("rows" in item or kind == "ipm_tail"):
            count = int(item.get("rows", 0))
            if count < 1:
                raise DesignSchemaError(f"{item.get('name', kind)}.rows must be positive")
            name = str(item["name"])
            cursor = item.get("cursor", "signal")
            for index in range(count):
                expanded.append({"type": "jj_line",
                                 "name": f"{name}.row[{index}].array[0]",
                                 "cursor": cursor,
                                 "cells": item.get("cells", params.get("array_length", 0)),
                                 "Lj": item.get("Lj", params.get("Lj")),
                                 "Cj": item.get("Cj", params.get("Cj")),
                                 "Cg": item.get("Cg", params.get("Cg"))})
                if index + 1 < count:
                    expanded.append({"type": "transmission_line",
                                     "name": f"{name}.row[{index}].short_tl",
                                     "cursor": cursor,
                                     "cells": params.get("length_of_short_TL", 0),
                                     "L": params.get("Ll", 0.0), "C": params.get("Cl", 0.0)})
            if kind == "ipm_line" and item.get("end_coupler", True):
                expanded.extend([
                    {"type": "transmission_line", "name": f"{name}.long_tl",
                     "cursor": cursor, "cells": params.get("length_of_long_TL", 0),
                     "L": params.get("Ll", 0.0), "C": params.get("Cl", 0.0)},
                    {"type": "transmission_line", "name": f"{name}.pump_section",
                     "cursor": "pump", "cells": params.get("coupler_section_length", 0),
                     "L": params.get("Ll", 0.0), "C": params.get("Cl", 0.0)},
                ])
            continue
        if kind == "ipm_line":
            count = int(item["arrays"])
            if count < 1:
                raise DesignSchemaError(f"{item.get('name', 'ipm_line')}.arrays must be positive")
            name = str(item["name"])
            between = item.get("between", params.get("length_of_short_TL",
                                                        params.get("short_tl", 0)))
            for index in range(count):
                lower = {key: value for key, value in item.items()
                         if key not in {"type", "arrays", "between"}}
                lower.update({"type": "jj_line", "name": f"{name}.array[{index}]"})
                lower.setdefault("cells", item.get("cells", params.get("array_length", 0)))
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


def _register_elements(ctx: BuildContext, path: str, before: int) -> None:
    new = ctx.circuit[before:]
    for index, element in enumerate(new):
        ctx.named_elements[f"{path}.element[{index}]"] = element.name
    # The series branch is the addressable right edge for the common blocks.
    for i, element in enumerate(new):
        if element.kind in {"linear_inductor", "josephson_inductor"}:
            ctx.named_elements[f"{path}.cell[{i // 2}].right"] = element.name


def compile_design(spec: Mapping[str, Any], *, coupler_mode: str | None = None,
                   strict: bool = False, plan: Any = None) -> CompiledDesign:
    # Keep the distinction between parameters declared by the design author
    # and defaults injected by a technology preset.  Strict validation applies
    # only to the former; a preset is a shared parameter catalogue, not a
    # second set of declarations that every design must reference explicitly.
    declared_parameters = set(spec.get("parameters", {}))
    spec = _technology(spec)
    validate_design(spec)
    raw_parameters = dict(spec.get("parameters", {}))
    for key, value in list(raw_parameters.items()):
        if isinstance(value, Mapping) and "profile" in value:
            raw_parameters[key] = value.get("value", value["profile"].get("start"))
    parameters = resolve_parameters(raw_parameters, raw_parameters, "parameters")
    topology = resolve_parameters(_expand_composites(spec["topology"], parameters),
                                  parameters, "topology")
    if strict:
        from twpa_solver.design.parameters import parameter_references
        used = parameter_references(spec["topology"])
        # Composite blocks consume parameters inside their compiler expansion,
        # so those references are not visible in the compact source topology.
        implicit_by_block = {
            "input_ports": {"Rleft", "Rm", "Ll", "Cl", "len1", "len3"},
            "output_ports": {"Rright", "Rm", "Ll", "Cl", "len2", "len4"},
            "directional_coupler": {"coupling_dB", "Z0", "coupler_freq_hz",
                                     "cached_coupler_width_um",
                                     "cached_coupler_gap_um",
                                     "cached_coupler_gap_to_ground_um",
                                     "cached_coupler_length_um"},
            "coupler": {"coupling_dB", "Z0", "coupler_freq_hz",
                        "cached_coupler_width_um", "cached_coupler_gap_um",
                        "cached_coupler_gap_to_ground_um",
                        "cached_coupler_length_um"},
            "ipm_line": {"array_length", "Lj", "Cj", "Cg", "Ll", "Cl",
                         "length_of_short_TL", "short_tl"},
            "ipm_tail": {"array_length", "Lj", "Cj", "Cg", "Ll", "Cl"},
        }
        for item in spec["topology"]:
            if isinstance(item, Mapping):
                used.update(implicit_by_block.get(str(item.get("type")), set()))
        unused = sorted(declared_parameters - used)
        if unused:
            raise DesignParameterError(f"parameters: declared but unused {unused}")
    cursors = {str(key): int(value) for key, value in spec["cursors"].items()}
    coupler_choice = str(coupler_mode or spec.get(
        "coupler_mode", parameters.get("coupler_mode", "auto")))
    effective_plan = plan or _design_plan(spec, parameters)
    ctx = BuildContext([], dict(cursors), 0, int(spec["ground"]),
                       coupler=make_coupler_discrete(
                           IPMParams(**{key: value for key, value in parameters.items()
                                        if key in IPMParams.__dataclass_fields__}),
                           coupler_choice), plan=effective_plan)
    ctx.named_nodes["ground"] = ctx.ground
    owners: dict[int, str] = {}
    for cfg, generated_path in _expanded(topology):
        if cfg.get("action") is not None:
            _apply_inline_action(ctx, cfg, generated_path)
            continue
        kind = cfg.get("type")
        if kind == "coupler":
            kind = "directional_coupler"
            cfg["type"] = kind
        if kind == "directional_coupler":
            cfg.setdefault("cursors", ["signal", "pump"])
        if kind == "capacitor":
            cfg = {**cfg, "type": "raw_element", "value": cfg["C"], "kind": "capacitor"}
            kind = "raw_element"
        elif kind == "inductor":
            cfg = {**cfg, "type": "raw_element", "value": cfg["L"], "kind": "linear_inductor"}
            kind = "raw_element"
        if kind not in BLOCK_BUILDERS:
            raise DesignSchemaError(f"{generated_path}.type: unknown block type {kind!r}")
        block = _config_for_block(cfg, parameters)
        if kind == "raw_element":
            nodes = block.get("nodes", [])
            if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
                raise DesignSchemaError(f"{generated_path}.nodes: expected two endpoints")
            if block.get("kind") == "mutual_inductor_k":
                endpoints = [ctx.named_elements.get(endpoint, endpoint) for endpoint in nodes]
                if any(not any(element.name == endpoint and element.kind == "linear_inductor"
                               for element in ctx.circuit) for endpoint in endpoints):
                    raise DesignSchemaError(
                        f"{generated_path}.nodes: mutual endpoints must name linear inductors")
                block["nodes"] = endpoints
            else:
                endpoints = [ctx.named_nodes.get(endpoint, endpoint) for endpoint in nodes]
                if any(not isinstance(endpoint, int) for endpoint in endpoints):
                    raise DesignSchemaError(
                        f"{generated_path}.nodes: unknown node path or non-integer endpoint")
                block["nodes"] = endpoints
        before = len(ctx.circuit)
        BLOCK_BUILDERS[kind](ctx, block, generated_path)
        _register_elements(ctx, generated_path, before)
        for cursor, start in ctx.blocks[-1].start_nodes.items():
            end = ctx.blocks[-1].end_nodes.get(cursor, start)
            for node in range(min(start, end), max(start, end) + 1):
                prior = owners.get(node)
                if prior is not None and prior != cursor:
                    raise DesignCollisionError(
                        f"cursor collision at node {node}: {prior!r} overlaps {cursor!r}")
                owners[node] = cursor
    apply_patches(ctx, list(spec.get("patches", [])))
    _add_first_array_aliases(ctx)
    named_elements = dict(ctx.named_elements)
    element_names = [element.name for element in ctx.circuit]
    duplicates = {name for name in element_names if element_names.count(name) > 1}
    if duplicates:
        raise DesignCollisionError(f"duplicate element names: {sorted(duplicates)[:3]}")
    ports = {int(element.value): PortRecord(int(element.value), int(element.n1))
             for element in ctx.circuit if element.kind == "port"}
    return CompiledDesign(str(spec["name"]), ctx.circuit, ctx.cursors,
                          dict(ctx.named_nodes), named_elements, ports, ctx.blocks,
                          {"schema_version": 1, "parameters": parameters,
                           "technology": spec.get("technology"),
                           "technology_parameters": spec.get("_technology_parameters", {}),
                           "coupler_mode": coupler_choice, "source": spec.get("_source"),
                           "coupler_geometry": ({
                               "width_um": ctx.coupler.geometry.width_um,
                               "gap_between_lines_um": ctx.coupler.geometry.gap_between_lines_um,
                               "gap_to_ground_um": ctx.coupler.geometry.gap_to_ground_um,
                               "length_um": ctx.coupler.geometry.length_um,
                               "coupling_db": ctx.coupler.geometry.k_db,
                               "z_input_ohm": ctx.coupler.geometry.z_input_ohm,
                               "model": ctx.coupler.geometry.model,
                           } if ctx.coupler is not None else {}),
                           "profiles": spec.get("profiles", {}),
                           "component_plan": (effective_plan.metadata
                                              if effective_plan is not None else {})},
                          effective_plan)


def _apply_inline_action(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None:
    target = cfg.get("target")
    if not isinstance(target, str):
        raise DesignResolutionError(f"{path}.target: expected an exact path")
    element_name = ctx.named_elements.get(target, target)
    matches = [element for element in ctx.circuit if element.name == element_name]
    if len(matches) != 1:
        raise DesignResolutionError(
            f"{path}.target {target!r}: expected one element, found {len(matches)}")
    if cfg["action"] == "set":
        matches[0].value = cfg["value"]
    elif cfg["action"] == "remove":
        ctx.circuit.remove(matches[0])
    else:
        raise DesignSchemaError(f"{path}.action: unknown action")


def _add_first_array_aliases(ctx: BuildContext) -> None:
    """Expose ``line.cell[...]`` as a shorthand for a line's first array."""
    for table in (ctx.named_nodes, ctx.named_elements):
        additions: dict[str, Any] = {}
        for key, value in table.items():
            marker = ".array[0]."
            if marker in key:
                alias = key.replace(marker, ".", 1)
                additions.setdefault(alias, value)
        table.update(additions)
        legacy: dict[str, Any] = {}
        for key, value in table.items():
            if key.startswith("line_1.row["):
                legacy[key.replace("line_1.row[", "period[0].row[", 1)
                     .replace(".array[0].", ".array.", 1)] = value
        table.update(legacy)
