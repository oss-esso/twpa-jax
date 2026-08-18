"""Safe scalar parameter substitution and arithmetic expressions."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable, Mapping
from typing import Any

from twpa_solver.design.errors import DesignParameterError

_TOKEN = re.compile(
    r"\$\{(?P<qualified>base\.)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}"
    r"|\$(?P<short>[A-Za-z_][A-Za-z0-9_]*)\$"
)
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _token_parts(match: re.Match[str]) -> tuple[str, bool]:
    short = match.group("short")
    if short is not None:
        return short, True
    return str(match.group("name")), match.group("qualified") is not None


def parameter_references(value: Any) -> set[str]:
    """Return non-technology parameter names referenced by a nested value."""

    if isinstance(value, Mapping):
        return set().union(*(parameter_references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(parameter_references(item) for item in value))
    if not isinstance(value, str):
        return set()
    references: set[str] = set()
    for match in _TOKEN.finditer(value):
        name, is_base = _token_parts(match)
        if not is_base:
            references.add(name)
    return references


def _numeric(value: Any, path: str, token: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignParameterError(
            f"{path}: {token} must resolve to a number inside an expression"
        )
    return float(value)


def _evaluate_ast(node: ast.AST, values: Mapping[str, float], path: str) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body, values, path)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise DesignParameterError(f"{path}: expression constants must be numeric")
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_ast(node.left, values, path)
        right = _evaluate_ast(node.right, values, path)
        try:
            return _BINARY_OPERATORS[type(node.op)](left, right)
        except ArithmeticError as error:
            raise DesignParameterError(
                f"{path}: invalid arithmetic: {error}"
            ) from error
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](
            _evaluate_ast(node.operand, values, path)
        )
    raise DesignParameterError(
        f"{path}: only numeric literals and +, -, *, / are allowed"
    )


def _resolve_string(
    value: str,
    lookup: Callable[[str, bool], Any],
    path: str,
) -> Any:
    matches = list(_TOKEN.finditer(value))
    if not matches:
        if "${" in value or "}" in value:
            raise DesignParameterError(f"{path}: invalid parameter reference")
        if _NUMBER.match(value.strip()):
            return float(value)
        return value
    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        name, is_base = _token_parts(matches[0])
        return lookup(name, is_base)
    for match in matches:
        before = value[match.start() - 1] if match.start() else ""
        after = value[match.end()] if match.end() < len(value) else ""
        if (
            (before and (before.isalnum() or before in "_."))
            or (after and (after.isalnum() or after in "_."))
        ):
            raise DesignParameterError(
                f"{path}: parameter references must be a whole scalar value "
                "or a numeric expression"
            )

    expression = value
    resolved: dict[str, float] = {}
    for index, match in reversed(list(enumerate(matches))):
        name, is_base = _token_parts(match)
        token = match.group(0)
        placeholder = f"_parameter_{index}"
        resolved[placeholder] = _numeric(lookup(name, is_base), path, token)
        expression = expression[:match.start()] + placeholder + expression[match.end():]
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise DesignParameterError(
            f"{path}: invalid parameter expression {value!r}"
        ) from error
    return _evaluate_ast(parsed, resolved, path)


def _resolve_value(
    value: Any,
    lookup: Callable[[str, bool], Any],
    path: str,
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _resolve_value(item, lookup, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_value(item, lookup, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        return _resolve_string(value, lookup, path)
    return value


def resolve_parameters(
    value: Any,
    parameters: Mapping[str, Any],
    path: str = "design",
    *,
    base_parameters: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve references in a nested value against final and base parameters."""

    base = base_parameters or {}

    def lookup(name: str, is_base: bool) -> Any:
        source = base if is_base else parameters
        if name not in source:
            namespace = "base parameter" if is_base else "parameter"
            raise DesignParameterError(f"{path}: unknown {namespace} {name!r}")
        return source[name]

    return _resolve_value(value, lookup, path)


def resolve_parameter_definitions(
    parameters: Mapping[str, Any],
    base_parameters: Mapping[str, Any] | None = None,
    path: str = "parameters",
) -> dict[str, Any]:
    """Resolve parameter definitions recursively with cycle detection."""

    raw = dict(parameters)
    base = dict(base_parameters or {})
    resolved: dict[str, Any] = {}
    resolving: list[str] = []

    def resolve_name(name: str) -> Any:
        if name in resolved:
            return resolved[name]
        if name not in raw:
            raise DesignParameterError(f"{path}: unknown parameter {name!r}")
        if name in resolving:
            start = resolving.index(name)
            cycle = " -> ".join((*resolving[start:], name))
            raise DesignParameterError(f"{path}: parameter cycle {cycle}")
        resolving.append(name)

        def lookup(reference: str, is_base: bool) -> Any:
            if is_base:
                if reference not in base:
                    raise DesignParameterError(
                        f"{path}.{name}: unknown base parameter {reference!r}"
                    )
                return base[reference]
            return resolve_name(reference)

        resolved[name] = _resolve_value(raw[name], lookup, f"{path}.{name}")
        resolving.pop()
        return resolved[name]

    for parameter_name in raw:
        resolve_name(parameter_name)
    return resolved


__all__ = [
    "parameter_references",
    "resolve_parameter_definitions",
    "resolve_parameters",
]
