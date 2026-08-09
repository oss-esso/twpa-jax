"""Strict whole-value parameter substitution."""

from __future__ import annotations

import re
from typing import Any

from twpa_solver.design.errors import DesignParameterError

_TOKEN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parameter_references(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(parameter_references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(parameter_references(item) for item in value))
    if isinstance(value, str):
        match = _TOKEN.match(value)
        return {match.group(1)} if match else set()
    return set()


def resolve_parameters(value: Any, parameters: dict[str, Any], path: str = "design") -> Any:
    if isinstance(value, dict):
        return {key: resolve_parameters(item, parameters, f"{path}.{key}")
                for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_parameters(item, parameters, f"{path}[{i}]")
                for i, item in enumerate(value)]
    if isinstance(value, str):
        match = _TOKEN.match(value)
        if match:
            name = match.group(1)
            if name not in parameters:
                raise DesignParameterError(f"{path}: unknown parameter {name!r}")
            return parameters[name]
        if "${" in value or "}" in value:
            raise DesignParameterError(
                f"{path}: parameter references must be a whole scalar value")
        if _NUMBER.match(value.strip()):
            return float(value)
    return value
