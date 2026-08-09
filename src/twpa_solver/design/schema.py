from __future__ import annotations

from typing import Any, Mapping

from twpa_solver.design.errors import DesignSchemaError


def validate_design(spec: Mapping[str, Any]) -> None:
    allowed = {"schema_version", "name", "ground", "parameters", "cursors",
               "topology", "patches", "technology", "extends", "profiles",
               "coupler_mode",
               "_source", "_technology_parameters", "_default_cursors",
               "_default_ground"}
    unknown = set(spec) - allowed
    if unknown:
        raise DesignSchemaError(f"design: unknown top-level keys {sorted(unknown)}")
    if spec.get("schema_version") != 1:
        raise DesignSchemaError("schema_version: expected integer 1")
    if not isinstance(spec.get("name"), str) or not spec["name"]:
        raise DesignSchemaError("name: expected a non-empty string")
    if not isinstance(spec.get("ground"), int):
        raise DesignSchemaError("ground: expected an integer")
    if not isinstance(spec.get("parameters", {}), Mapping):
        raise DesignSchemaError("parameters: expected a mapping")
    if "technology" in spec and not isinstance(spec["technology"], str):
        raise DesignSchemaError("technology: expected a preset name")
    if "extends" in spec and not isinstance(spec["extends"], str):
        raise DesignSchemaError("extends: expected a YAML path")
    if "profiles" in spec and not isinstance(spec["profiles"], Mapping):
        raise DesignSchemaError("profiles: expected a mapping")
    cursors = spec.get("cursors")
    if not isinstance(cursors, Mapping) or not cursors:
        raise DesignSchemaError("cursors: expected a non-empty mapping")
    if any(not isinstance(v, int) for v in cursors.values()):
        raise DesignSchemaError("cursors: values must be integers")
    if not isinstance(spec.get("topology"), list):
        raise DesignSchemaError("topology: expected a sequence")
    _validate_topology(spec["topology"], "topology")
    if "patches" in spec and not isinstance(spec["patches"], list):
        raise DesignSchemaError("patches: expected a sequence")
    for index, patch in enumerate(spec.get("patches", [])):
        if not isinstance(patch, Mapping):
            raise DesignSchemaError(f"patches[{index}]: expected a mapping")
        action = patch.get("action")
        if action == "add":
            required = {"action", "name", "nodes", "value", "kind"}
        elif action == "set":
            required = {"action", "target", "value"}
        elif action == "remove":
            required = {"action", "target"}
        else:
            raise DesignSchemaError(f"patches[{index}].action: unknown action {action!r}")
        missing = required - set(patch)
        if missing:
            raise DesignSchemaError(f"patches[{index}]: missing fields {sorted(missing)}")


_FIELDS = {
    "port": {"type", "name", "cursor", "port"},
    "resistor": {"type", "name", "cursor", "value"},
    "transmission_line": {"type", "name", "cursor", "cells", "L", "C"},
    "jj_line": {"type", "name", "cursor", "cells", "Lj", "Cj", "Cg"},
    "rf_squid_line": {"type", "name", "cursor", "cells", "Ic", "Lj", "Lm",
                      "Lw", "Lpar", "Cj", "Cg", "Cg_pattern",
                      "Cg_pattern_counts"},
    "directional_coupler": {"type", "name", "cursors"},
    "raw_element": {"type", "name", "nodes", "value", "kind"},
    "coupler": {"type", "name", "cursors"},
    "capacitor": {"type", "name", "nodes", "C"},
    "inductor": {"type", "name", "nodes", "L"},
    "ipm_line": {"type", "name", "cursor", "arrays", "rows", "cells", "Lj", "Cj", "Cg", "between", "end_coupler"},
    "ipm_row": {"type", "name", "cursor", "cells", "Lj", "Cj", "Cg"},
    "ipm_tail": {"type", "name", "cursor", "rows", "cells", "Lj", "Cj", "Cg", "final_array"},
    "signal_port": {"type", "name", "port"},
    "pump_port": {"type", "name", "port"},
    "input_ports": {"type", "name"},
    "output_ports": {"type", "name"},
}
_REQUIRED = {
    "port": {"cursor", "port"},
    "resistor": {"cursor", "value"},
    "transmission_line": {"cursor", "cells", "L", "C"},
    "jj_line": {"cursor", "cells", "Lj", "Cj", "Cg"},
    "rf_squid_line": {"cursor", "cells", "Ic", "Lm", "Lw", "Lpar", "Cj"},
    "directional_coupler": set(),
    "raw_element": {"nodes", "value", "kind"},
    "coupler": {"cursors"},
    "capacitor": {"nodes", "C"},
    "inductor": {"nodes", "L"},
    "ipm_line": set(),
    "ipm_row": {"cursor", "cells", "Lj", "Cj", "Cg"},
    "ipm_tail": {"rows"},
    "signal_port": {"port"},
    "pump_port": {"port"},
    "input_ports": set(),
    "output_ports": set(),
}
_RAW_KINDS = {"capacitor", "coupling_capacitor", "linear_inductor",
              "josephson_inductor", "mutual_inductor_k", "resistor", "port"}


def _validate_topology(items: list[Any], path: str) -> None:
    names: set[str] = set()
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            raise DesignSchemaError(f"{item_path}: expected a mapping")
        if "repeat" in item:
            repeat = item["repeat"]
            if not isinstance(repeat, Mapping):
                raise DesignSchemaError(f"{item_path}.repeat: expected a mapping")
            nested = repeat.get("topology")
            if not isinstance(nested, list):
                raise DesignSchemaError(f"{item_path}.repeat.topology: expected a sequence")
            _validate_topology(nested, f"{item_path}.repeat.topology")
            continue
        if item.get("action") is not None:
            _validate_action(item, item_path)
            continue
        kind = item.get("type")
        if kind not in _FIELDS:
            raise DesignSchemaError(f"{item_path}.type: unknown block type {kind!r}")
        unknown = set(item) - _FIELDS[kind]
        if unknown:
            raise DesignSchemaError(f"{item_path}: unknown fields {sorted(unknown)}")
        missing = _REQUIRED[kind] - set(item)
        if missing:
            raise DesignSchemaError(f"{item_path}: missing fields {sorted(missing)}")
        if kind == "raw_element" and item.get("kind") not in _RAW_KINDS:
            raise DesignSchemaError(f"{item_path}.kind: unsupported element kind")
        if kind in {"capacitor", "inductor"} and len(item["nodes"]) != 2:
            raise DesignSchemaError(f"{item_path}.nodes: expected two endpoints")
        if kind == "ipm_line" and "arrays" not in item and "rows" not in item:
            raise DesignSchemaError(f"{item_path}: requires 'rows' or 'arrays'")
        if kind == "ipm_tail" and item.get("final_array", True) is not True:
            raise DesignSchemaError(f"{item_path}.final_array: expected true")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise DesignSchemaError(f"{item_path}.name: required non-empty string")
        if name in names:
            raise DesignSchemaError(f"{item_path}.name: duplicate name {name!r}")
        names.add(name)


def _validate_action(item: Mapping[str, Any], path: str) -> None:
    action = item.get("action")
    allowed = {"action", "target", "value"}
    if action not in {"set", "remove"}:
        raise DesignSchemaError(f"{path}.action: expected 'set' or 'remove'")
    missing = {"target"} | ({"value"} if action == "set" else set())
    if missing - set(item):
        raise DesignSchemaError(f"{path}: missing fields {sorted(missing - set(item))}")
    unknown = set(item) - allowed
    if unknown:
        raise DesignSchemaError(f"{path}: unknown fields {sorted(unknown)}")
