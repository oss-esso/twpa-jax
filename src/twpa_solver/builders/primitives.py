"""Shared netlist primitives.

The canonical implementations remain in :mod:`ipm` for backwards
compatibility; these aliases provide the generic builder boundary.
"""

from twpa_solver.builders.ipm import (
    Element,
    LossSpec,
    add,
    add_coupling,
    add_jj,
    add_jtl,
    add_jtl_element,
    add_tl,
    add_tl_element,
)

__all__ = [
    "Element", "LossSpec", "add", "add_coupling", "add_jj", "add_jtl",
    "add_jtl_element", "add_tl", "add_tl_element",
]
