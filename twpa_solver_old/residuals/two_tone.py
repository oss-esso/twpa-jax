"""Two-tone HB frequency-grid scaffold for compression/intermodulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoToneIndex:
    pump_order: int
    signal_order: int
    frequency_hz: float


def make_two_tone_grid(
    pump_frequency_hz: float,
    signal_frequency_hz: float,
    max_order: int,
) -> list[TwoToneIndex]:
    """Generate multi-index frequencies m fp + n fs."""
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    grid: list[TwoToneIndex] = []
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            grid.append(TwoToneIndex(m, n, m * pump_frequency_hz + n * signal_frequency_hz))
    return grid
