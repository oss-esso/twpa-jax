"""Worked example of adding a block with technology-backed defaults."""

from __future__ import annotations

from ..elements import ElementRef
from ..nodes import Node


class ParallelLCBuilders:
    """Demonstrate a block with no central default-dispatch edit."""

    TECHNOLOGY_DEFAULTS = {"L": "Lk"}

    def add_parallel_lc(
        self,
        n1: Node,
        n2: Node,
        *,
        C: float,
        L: float | None = None,
        name: str | None = None,
    ) -> tuple[ElementRef, ElementRef]:
        """Add parallel branches, resolving only the declared inductance default."""

        inductance = self._resolve_builder_parameter(
            "L", L, ParallelLCBuilders.TECHNOLOGY_DEFAULTS, name or "parallel_lc"
        )
        prefix = name or "parallel_lc"
        linear = self.add_inductor(n1, n2, float(inductance), name=f"{prefix}.L")
        capacitor = self.add_capacitor(n1, n2, float(C), name=f"{prefix}.C")
        return linear, capacitor
