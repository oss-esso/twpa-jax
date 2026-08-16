"""SPICE-like export of compiled symbolic circuits."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from .compiler import CompiledCircuit


def _format_value(value: float | int | str) -> str:
    return str(value)


def export_netlist(
    compiled: CompiledCircuit,
    path: str | PathLike[str] | None = None,
) -> str:
    """Return and optionally write a flat, inspectable netlist.

    Each line contains ``name n1 n2 value kind role``. The first four fields
    are sufficient to identify the electrical connection and value; the final
    fields preserve the solver element classification.
    """

    lines = [
        " ".join(
            (
                element.name,
                str(element.n1),
                str(element.n2),
                _format_value(element.value),
                element.kind,
                element.role,
            )
        )
        for element in compiled.elements
    ]
    text = "\n".join(lines) + ("\n" if lines else "")
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
