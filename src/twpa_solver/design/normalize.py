"""Normalize the concise YAML topology surface to compiler blocks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from twpa_solver.design.errors import DesignSchemaError


@dataclass(frozen=True)
class _Line:
    """One ordered physical path from the concise YAML surface."""

    cursor: str
    port_in: int
    port_out: int
    blocks: list[Mapping[str, Any]]


def _line_cursor(item: Mapping[str, Any], path: str) -> str:
    value = item.get("line")
    if isinstance(value, str) and value:
        return value
    if value == 1:
        return "signal"
    if value == 2:
        return "pump"
    raise DesignSchemaError(
        f"{path}.line: expected a cursor name, 1 for signal, or 2 for pump"
    )


def _line(item: Mapping[str, Any], path: str) -> _Line:
    allowed = {"line", "port_in", "port_out", "port in", "port out", "blocks"}
    unknown = set(item) - allowed
    if unknown:
        raise DesignSchemaError(f"{path}: unknown line fields {sorted(unknown)}")
    cursor = _line_cursor(item, path)
    port_in = item.get("port_in", item.get("port in"))
    port_out = item.get("port_out", item.get("port out"))
    if not isinstance(port_in, int) or isinstance(port_in, bool):
        raise DesignSchemaError(f"{path}.port_in: expected an integer")
    if not isinstance(port_out, int) or isinstance(port_out, bool):
        raise DesignSchemaError(f"{path}.port_out: expected an integer")
    blocks = item.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise DesignSchemaError(f"{path}.blocks: expected a non-empty sequence")
    if any(not isinstance(block, Mapping) for block in blocks):
        raise DesignSchemaError(f"{path}.blocks: every block must be a mapping")
    return _Line(cursor, port_in, port_out, blocks)


def _port_value(block: Mapping[str, Any], spaced: str, underscored: str) -> Any:
    return block.get(underscored, block.get(spaced))


def _coupler_cursors(
    block: Mapping[str, Any],
    lines: Mapping[str, _Line],
    path: str,
) -> list[str]:
    explicit = block.get("cursors")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != 2:
            raise DesignSchemaError(f"{path}.cursors: expected two line names")
        return [str(value) for value in explicit]
    signal_in = _port_value(block, "port in signal", "port_in_signal")
    pump_in = _port_value(block, "port in pump", "port_in_pump")
    signal_out = _port_value(block, "port out signal", "port_out_signal")
    pump_out = _port_value(block, "port out pump", "port_out_pump")
    if all(value is None for value in (signal_in, pump_in, signal_out, pump_out)):
        if "signal" in lines and "pump" in lines:
            return ["signal", "pump"]
        raise DesignSchemaError(
            f"{path}: provide cursors or signal and pump endpoint ports"
        )
    matches: list[str] = []
    for role, port_in, port_out in (
        ("signal", signal_in, signal_out),
        ("pump", pump_in, pump_out),
    ):
        candidates = [
            line.cursor
            for line in lines.values()
            if line.port_in == port_in and line.port_out == port_out
        ]
        if len(candidates) != 1:
            raise DesignSchemaError(
                f"{path}: {role} ports ({port_in}, {port_out}) match "
                f"{len(candidates)} lines"
            )
        matches.append(candidates[0])
    return matches


def _normalize_block(
    block: Mapping[str, Any],
    line: _Line,
    lines: Mapping[str, _Line],
    path: str,
) -> dict[str, Any]:
    kind = block.get("type")
    name = block.get("name")
    if not isinstance(kind, str) or not kind:
        raise DesignSchemaError(f"{path}.type: expected a non-empty string")
    if not isinstance(name, str) or not name:
        raise DesignSchemaError(f"{path}.name: expected a non-empty string")
    if kind == "cpw":
        allowed = {"type", "name", "cells", "L", "C"}
        unknown = set(block) - allowed
        if unknown:
            raise DesignSchemaError(f"{path}: unknown CPW fields {sorted(unknown)}")
        return {**block, "type": "transmission_line", "cursor": line.cursor}
    if kind == "jtl":
        allowed = {
            "type", "name", "rows", "jj_number", "jj number", "cells",
            "Lj", "Cj", "Cg",
        }
        unknown = set(block) - allowed
        if unknown:
            raise DesignSchemaError(f"{path}: unknown JTL fields {sorted(unknown)}")
        cells = block.get("cells", block.get("jj_number", block.get("jj number")))
        result = {
            key: value
            for key, value in block.items()
            if key not in {"jj_number", "jj number"}
        }
        result.update({"type": "jtl", "cursor": line.cursor})
        if cells is not None:
            result["cells"] = cells
        return result
    if kind == "directional_coupler":
        allowed = {
            "type", "name", "cursors", "port_in_signal", "port_in_pump",
            "port_out_signal", "port_out_pump", "port in signal",
            "port in pump", "port out signal", "port out pump",
        }
        unknown = set(block) - allowed
        if unknown:
            raise DesignSchemaError(
                f"{path}: unknown directional-coupler fields {sorted(unknown)}"
            )
        return {
            "type": "directional_coupler",
            "name": name,
            "cursors": _coupler_cursors(block, lines, path),
        }
    if kind in {"input_ports", "output_ports"}:
        allowed = {"type", "name", "resistance"}
        unknown = set(block) - allowed
        if unknown:
            raise DesignSchemaError(f"{path}: unknown port fields {sorted(unknown)}")
        port = line.port_in if kind == "input_ports" else line.port_out
        return {
            **block,
            "cursor": line.cursor,
            "port": port,
        }
    result = dict(block)
    result.setdefault("cursor", line.cursor)
    return result


def _canonical_kind(kind: Any) -> str:
    aliases = {"cpw": "transmission_line", "coupler": "directional_coupler"}
    return aliases.get(str(kind), str(kind))


def _ordered_names(
    sequences: list[list[str]],
    first_seen: Mapping[str, int],
) -> list[str]:
    successors = {name: set() for name in first_seen}
    indegree = {name: 0 for name in first_seen}
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            if right in successors[left]:
                continue
            successors[left].add(right)
            indegree[right] += 1
    ready = sorted(
        (name for name, degree in indegree.items() if degree == 0),
        key=first_seen.__getitem__,
    )
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for successor in sorted(successors[name], key=first_seen.__getitem__):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort(key=first_seen.__getitem__)
    if len(ordered) != len(first_seen):
        cyclic = sorted(name for name, degree in indegree.items() if degree > 0)
        raise DesignSchemaError(
            f"topology: line ordering contains a cycle through {cyclic}"
        )
    return ordered


def _normalize_lines(items: list[Any]) -> list[dict[str, Any]]:
    lines: dict[str, _Line] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise DesignSchemaError(f"topology[{index}]: expected a mapping")
        line = _line(item, f"topology[{index}]")
        if line.cursor in lines:
            raise DesignSchemaError(
                f"topology[{index}]: duplicate line {line.cursor!r}"
            )
        lines[line.cursor] = line
    input_ports = [line.port_in for line in lines.values()]
    if len(input_ports) != len(set(input_ports)):
        raise DesignSchemaError("topology: input port numbers must be unique")
    output_ports = [line.port_out for line in lines.values()]
    if set(input_ports) & set(output_ports):
        raise DesignSchemaError(
            "topology: an input and output cannot use the same port number"
        )
    for output_port in set(output_ports):
        matching = [line for line in lines.values() if line.port_out == output_port]
        if len(matching) < 2:
            continue
        final_blocks = [line.blocks[-1] for line in matching]
        final_names = {block.get("name") for block in final_blocks}
        final_types = {_canonical_kind(block.get("type")) for block in final_blocks}
        if len(final_names) != 1 or final_types != {"output_ports"}:
            raise DesignSchemaError(
                f"topology: shared output port {output_port} requires one "
                "shared output_ports name"
            )

    definitions: dict[str, dict[str, Any]] = {}
    definition_paths: dict[str, str] = {}
    occurrences: dict[str, list[str]] = {}
    first_seen: dict[str, int] = {}
    sequences: list[list[str]] = []
    next_index = 0
    for line_index, line in enumerate(lines.values()):
        sequence: list[str] = []
        for block_index, block in enumerate(line.blocks):
            path = f"topology[{line_index}].blocks[{block_index}]"
            name = block.get("name")
            if not isinstance(name, str) or not name:
                raise DesignSchemaError(f"{path}.name: expected a non-empty string")
            sequence.append(name)
            occurrences.setdefault(name, []).append(line.cursor)
            if name not in first_seen:
                first_seen[name] = next_index
                next_index += 1
            kind = _canonical_kind(block.get("type"))
            minimal = set(block) <= {"type", "name"}
            is_reference = (
                minimal
                and (
                    kind in {"directional_coupler", "ipm_line", "ipm_tail", "jtl"}
                    or (kind == "output_ports" and name in definitions)
                )
            )
            prior = definitions.get(name)
            if prior is not None:
                if _canonical_kind(prior.get("type")) != kind:
                    raise DesignSchemaError(
                        f"{path}: reference {name!r} changes block type"
                    )
                if prior.get("_reference") and not is_reference:
                    definitions[name] = _normalize_block(block, line, lines, path)
                    definition_paths[name] = path
                    continue
                if not is_reference:
                    raise DesignSchemaError(
                        f"{path}: block {name!r} was already defined at "
                        f"{definition_paths[name]}"
                    )
                continue
            if is_reference:
                definitions[name] = {"type": kind, "name": name, "_reference": True}
                definition_paths[name] = path
                continue
            definitions[name] = _normalize_block(block, line, lines, path)
            definition_paths[name] = path
        sequences.append(sequence)

    unresolved = [
        name for name, block in definitions.items() if block.get("_reference")
    ]
    if unresolved:
        raise DesignSchemaError(
            f"topology: references have no detailed declaration {sorted(unresolved)}"
        )
    for name, cursors in occurrences.items():
        block = definitions[name]
        unique_cursors = list(dict.fromkeys(cursors))
        if block.get("type") == "jtl" and len(unique_cursors) > 1:
            block["join_cursors"] = unique_cursors
    for output_port in set(output_ports):
        matching = [line for line in lines.values() if line.port_out == output_port]
        if len(matching) < 2:
            continue
        required = {line.cursor for line in matching}
        joins = [
            set(block.get("join_cursors", []))
            for block in definitions.values()
            if block.get("type") == "jtl"
        ]
        if not any(required <= cursors for cursors in joins):
            raise DesignSchemaError(
                f"topology: shared output port {output_port} requires a shared JTL"
            )
    return [definitions[name] for name in _ordered_names(sequences, first_seen)]


def normalize_design(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize concise aliases while preserving the original flat schema."""

    result = dict(spec)
    parameters = result.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise DesignSchemaError("parameters: expected a mapping")
    parameters = dict(parameters)
    config = parameters.pop("config", None)
    if config is not None:
        if not isinstance(config, str) or not config:
            raise DesignSchemaError("parameters.config: expected a preset name")
        technology = result.get("technology")
        if technology is not None and technology != config:
            raise DesignSchemaError(
                "parameters.config and technology select different presets"
            )
        result["technology"] = config
        result["parameters"] = parameters
    topology = result.get("topology")
    if isinstance(topology, list) and topology:
        line_items = [
            isinstance(item, Mapping) and "line" in item
            for item in topology
        ]
        if any(line_items):
            if not all(line_items):
                raise DesignSchemaError(
                    "topology: line-scoped and flat blocks cannot be mixed"
                )
            if "ground" not in result:
                result["ground"] = 0
                result["_default_ground"] = True
            if "cursors" not in result:
                result["cursors"] = {"signal": 1, "pump": 100_000}
                result["_default_cursors"] = True
            result["topology"] = _normalize_lines(topology)
    return result


__all__ = ["normalize_design"]
