"""Safe, deterministic per-cell component profiles."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Selection:
    rows: tuple[int, int] | None = None
    index: tuple[int, int] | None = None
    fraction: tuple[float, float] | None = None


@dataclass(frozen=True)
class Segment:
    shape: str
    start: float
    end: float | None = None
    select: Selection = field(default_factory=Selection)
    domain: str = "selection"
    params: Mapping[str, float] = field(default_factory=dict)
    expression: str | None = None


def _si_value(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    suffixes = {"p": 1e-12, "f": 1e-15, "n": 1e-9, "u": 1e-6}
    text = value.strip().lower()
    for suffix, scale in suffixes.items():
        if text.endswith(suffix):
            return float(text[:-1]) * scale
    return float(text)


def _selection_indices(
    selection: Selection, *, n_cells: int, cells_per_row: int
) -> np.ndarray:
    if n_cells <= 0 or cells_per_row <= 0 or n_cells % cells_per_row:
        raise ValueError("n_cells must be a positive multiple of cells_per_row")
    selectors = [selection.rows, selection.index, selection.fraction]
    if sum(item is not None for item in selectors) > 1:
        raise ValueError("a selection may specify only one selector")
    if selection.rows is not None:
        lo, hi = selection.rows
        if lo < 0 or hi < lo or hi >= n_cells // cells_per_row:
            raise ValueError("row selection is outside the device")
        return np.concatenate(
            [
                np.arange(row * cells_per_row, (row + 1) * cells_per_row)
                for row in range(lo, hi + 1)
            ]
        )
    if selection.index is not None:
        lo, hi = selection.index
        if lo < 0 or hi < lo or hi >= n_cells:
            raise ValueError("index selection is outside the device")
        return np.arange(lo, hi + 1)
    if selection.fraction is not None:
        lo, hi = selection.fraction
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError("fraction selection must be within [0, 1]")
        start = int(math.floor(lo * n_cells))
        stop = int(math.ceil(hi * n_cells))
        return np.arange(start, max(start, stop))
    return np.arange(n_cells)


_CUSTOM_FUNCTIONS: Mapping[str, object] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
    "sinh": np.sinh, "cosh": np.cosh, "arcsin": np.arcsin,
    "arccos": np.arccos, "arctan": np.arctan, "exp": np.exp, "log": np.log,
    "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil, "minimum": np.minimum,
    "maximum": np.maximum,
}
_CUSTOM_CONSTANTS: Mapping[str, float] = {"pi": math.pi, "e": math.e}


def _custom_value(expression: str, t: np.ndarray) -> np.ndarray:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid custom profile expression") from exc
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant,
               ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
               ast.USub, ast.UAdd, ast.Load)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError("custom profile expression contains a forbidden construct")
    for node in ast.walk(tree):
        # Only bare whitelisted names may be called; this rejects attribute
        # lookups and any indirection that could reach outside the namespace.
        if isinstance(node, ast.Call):
            if node.keywords or not isinstance(node.func, ast.Name):
                raise ValueError("custom profile calls must be plain functions")
            if node.func.id not in _CUSTOM_FUNCTIONS:
                raise ValueError(f"custom profile function not allowed: {node.func.id}")
    permitted = {"t"} | set(_CUSTOM_FUNCTIONS) | set(_CUSTOM_CONSTANTS)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if not names <= permitted:
        raise ValueError(
            f"custom profile expression references unknown names: "
            f"{sorted(names - permitted)}"
        )
    code = compile(tree, "<custom-profile>", "eval")
    namespace = {"t": t, **_CUSTOM_FUNCTIONS, **_CUSTOM_CONSTANTS}
    try:
        result = eval(code, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise ValueError("custom profile expression could not be evaluated") from exc
    return np.broadcast_to(np.asarray(result, dtype=float), t.shape).astype(float)


def _shape_values(segment: Segment, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=float)
    if count == 1:
        t = np.array([0.0])
    else:
        t = np.linspace(0.0, 1.0, count)
    shape = segment.shape.lower()
    params = {key: float(value) for key, value in segment.params.items()}
    if shape == "const":
        if segment.end is not None and segment.end != segment.start:
            raise ValueError("const profiles require end to equal start")
        envelope = np.ones(count)
    elif shape == "linear":
        envelope = t
    elif shape == "power":
        exponent = params.get("p", params.get("exponent", 1.0))
        if exponent <= 0:
            raise ValueError("power exponent must be positive")
        envelope = t**exponent
    elif shape == "parabola":
        # vertex=0 is the plain t**2 parabola, so it is the natural default;
        # 0.5 puts the turning point at the midpoint and cannot be normalized.
        vertex = params.get("v", params.get("vertex", 0.0))
        denominator = (1.0 - vertex) ** 2 - vertex**2
        if vertex < 0 or vertex > 1 or denominator == 0:
            raise ValueError("parabola vertex must be in [0, 1] and not 0.5")
        envelope = ((t - vertex) ** 2 - vertex**2) / denominator
    elif shape == "half_cosine":
        envelope = (1.0 - np.cos(np.pi * t)) / 2.0
    elif shape in {"tanh", "sine", "cosine"}:
        if shape == "tanh":
            sharpness = params.get("k", params.get("sharpness", 1.0))
            if sharpness <= 0:
                raise ValueError("tanh sharpness must be positive")
            raw = np.tanh(sharpness * (2.0 * t - 1.0))
            envelope = (raw - raw[0]) / (raw[-1] - raw[0]) if count > 1 else raw
        else:
            periods = params.get("periods", 1.0)
            phase = params.get("phase", 0.0)
            angle = 2.0 * np.pi * periods * t + phase
            raw = np.sin(angle) if shape == "sine" else np.cos(angle)
            envelope = (raw + 1.0) / 2.0
    elif shape == "custom":
        if not segment.expression:
            raise ValueError("custom profiles require expression")
        envelope = _custom_value(segment.expression, t)
    else:
        raise ValueError(f"unknown profile shape: {segment.shape}")
    end = segment.start if segment.end is None else segment.end
    return float(segment.start) + (float(end) - float(segment.start)) * envelope


def evaluate_profile(
    segments: Sequence[Segment], *, n_cells: int, cells_per_row: int, base_value: float
) -> np.ndarray:
    """Evaluate segments in order; later segments overwrite earlier ones."""
    values = np.full(n_cells, float(base_value), dtype=float)
    for segment in segments:
        indices = _selection_indices(
            segment.select, n_cells=n_cells, cells_per_row=cells_per_row
        )
        if segment.domain not in {"selection", "per_row"}:
            raise ValueError("profile domain must be selection or per_row")
        if segment.domain == "selection":
            values[indices] = _shape_values(segment, len(indices))
        else:
            for row in np.unique(indices // cells_per_row):
                row_indices = indices[indices // cells_per_row == row]
                values[row_indices] = _shape_values(segment, len(row_indices))
    invalid = np.flatnonzero(values <= 0)
    if invalid.size:
        raise ValueError(f"profile produced a non-positive value at cell {int(invalid[0])}")
    return values


def _selection_from_text(text: str) -> Selection:
    if text == "all":
        return Selection()
    key, raw = text.split("=", 1)
    parts = raw.split("-")
    if key == "rows":
        return Selection(rows=(int(parts[0]), int(parts[-1])))
    if key == "index":
        return Selection(index=(int(parts[0]), int(parts[-1])))
    if key == "frac":
        return Selection(fraction=(float(parts[0]), float(parts[-1])))
    raise ValueError(f"unknown profile selector: {key}")


def parse_profile_shorthand(text: str) -> Segment:
    """Parse ``select:shape:start[->end][:k=v,...]``.

    ``domain`` and ``expression`` are reserved keys in the trailing field, so
    ``per_row`` and ``custom`` are reachable without a JSON file.
    """
    parts = text.split(":")
    if len(parts) < 3 or len(parts) > 4:
        raise ValueError("profile shorthand is select:shape:start[->end][:k=v,...]")
    start_end = parts[2].split("->")
    domain = "selection"
    expression = None
    params = {}
    if len(parts) == 4 and parts[3]:
        raw = dict(item.split("=", 1) for item in parts[3].split(","))
        domain = raw.pop("domain", "selection")
        expression = raw.pop("expression", None)
        params = {key: _si_value(value) for key, value in raw.items()}
    return Segment(
        shape=parts[1],
        start=_si_value(start_end[0]),
        end=None if len(start_end) == 1 else _si_value(start_end[1]),
        select=_selection_from_text(parts[0]),
        domain=domain,
        params=params,
        expression=expression,
    )


def _segment_from_mapping(data: Mapping[str, object]) -> Segment:
    select_data = data.get("select", {})
    if isinstance(select_data, str):
        selection = _selection_from_text(select_data)
    else:
        raw = select_data if isinstance(select_data, Mapping) else {}
        selection = Selection(
            rows=tuple(raw["rows"]) if "rows" in raw else None,
            index=tuple(raw["index"]) if "index" in raw else None,
            fraction=tuple(raw["fraction"]) if "fraction" in raw else None,
        )
    return Segment(
        shape=str(data["shape"]), start=_si_value(data["start"]),
        end=_si_value(data["end"]) if data.get("end") is not None else None,
        select=selection, domain=str(data.get("domain", "selection")),
        params={key: _si_value(value) for key, value in
                (data.get("params", {}) or {}).items()},
        expression=data.get("expression"),
    )


def segment_to_dict(segment: Segment) -> dict[str, object]:
    """Serialize a segment so a design can be regenerated from its summary."""
    select: dict[str, list[float]] = {}
    if segment.select.rows is not None:
        select["rows"] = list(segment.select.rows)
    if segment.select.index is not None:
        select["index"] = list(segment.select.index)
    if segment.select.fraction is not None:
        select["fraction"] = list(segment.select.fraction)
    return {
        "shape": segment.shape,
        "start": float(segment.start),
        "end": None if segment.end is None else float(segment.end),
        "select": select,
        "domain": segment.domain,
        "params": {key: float(value) for key, value in segment.params.items()},
        "expression": segment.expression,
    }


def segments_to_json(segments: Sequence[Segment]) -> list[dict[str, object]]:
    return [segment_to_dict(segment) for segment in segments]


def parse_profile_json(path: Path) -> dict[str, list[Segment]]:
    """Read the JSON profile mapping for Lj and Cg."""
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[Segment]] = {"Lj": [], "Cg": []}
    for key in result:
        for item in data.get(key, []):
            result[key].append(parse_profile_shorthand(item) if isinstance(item, str)
                               else _segment_from_mapping(item))
    return result
