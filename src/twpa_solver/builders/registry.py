"""Registry for declarative topology blocks."""

from collections.abc import Callable
from typing import Any

from twpa_solver.builders import blocks

BlockBuilder = Callable[[blocks.BuildContext, dict[str, Any], str], None]
BLOCK_BUILDERS: dict[str, BlockBuilder] = {}


def register_block(name: str) -> Callable[[BlockBuilder], BlockBuilder]:
    def decorator(builder: BlockBuilder) -> BlockBuilder:
        if name in BLOCK_BUILDERS:
            raise ValueError(f"duplicate block registration: {name}")
        BLOCK_BUILDERS[name] = builder
        return builder
    return decorator


for _name in ("port", "resistor", "transmission_line", "jj_line",
              "directional_coupler", "raw_element"):
    BLOCK_BUILDERS[_name] = getattr(blocks, f"build_{_name}")

