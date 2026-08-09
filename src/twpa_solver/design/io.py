from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from twpa_solver.design.errors import DesignSchemaError


def _merge(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(source: Path) -> dict[str, Any]:
    try:
        text = source.read_text(encoding="utf-8")
        # Flow-style YAML mappings require brace-containing substitutions to
        # be quoted; the compiler resolves them back to typed scalar values.
        text = re.sub(r"(?<![\"'])\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                      r'"${\1}"', text)
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DesignSchemaError(f"{source}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise DesignSchemaError(f"{source}: expected a mapping at the document root")
    return data


def load_design(path: str | Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load one design, resolving its single-parent inheritance chain."""
    source = Path(path).resolve()
    if source in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, source))
        raise DesignSchemaError(f"design inheritance cycle: {chain}")
    data = _load_yaml(source)
    parent_name = data.get("extends")
    if parent_name is not None:
        parent = load_design(source.parent / str(parent_name), _stack=(*_stack, source))
        data = _merge(parent, data)
    if "ground" not in data:
        data["ground"] = 0
        data["_default_ground"] = True
    if "cursors" not in data:
        data["cursors"] = {"signal": 1, "pump": 100_000}
        data["_default_cursors"] = True
    data["_source"] = str(source)
    return data
