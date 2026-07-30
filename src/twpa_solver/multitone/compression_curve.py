"""Compression-curve data processing and P1dB refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CompressionPoint:
    signal_power_dbm: float
    gain_db: float
    compression_db: float
    status: str = "VALID_SOLVED"


@dataclass
class CompressionCurve:
    points: list[CompressionPoint]
    p1db_dbm: float | None
    first_1db_crossing_dbm: float | None
    number_of_crossings: int
    nonmonotonic_compression: bool


def depletion_only_model(gain_linear: float, signal_power: float, pump_power: float) -> float:
    """Return the trend-only depletion model from the plan."""
    if pump_power <= 0.0:
        raise ValueError("pump_power must be positive")
    return gain_linear / (1.0 + 2.0 * gain_linear * signal_power / pump_power)


def depletion_only_gain_db(
    gain_linear: float, signal_power: float, pump_power: float
) -> float:
    """Return the depletion model's power gain in dB."""
    model_gain = depletion_only_model(gain_linear, signal_power, pump_power)
    return float(10.0 * np.log10(model_gain))


def refine_p1db(
    evaluator: Callable[[float], float],
    bracket_dbm: tuple[float, float],
    *,
    tolerance_db: float = 0.01,
    target_db: float = 1.0,
) -> float:
    """Refine the first compression crossing by bracketed bisection."""
    low, high = map(float, bracket_dbm)
    f_low = evaluator(low) - target_db
    f_high = evaluator(high) - target_db
    if f_low >= 0.0 or f_high < 0.0:
        raise ValueError("bracket must start below and end at/above the target compression")
    while high - low > tolerance_db:
        middle = 0.5 * (low + high)
        if evaluator(middle) >= target_db:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def build_compression_curve(
    signal_power_dbm: list[float],
    gain_db: list[float],
    small_signal_gain_db: float,
) -> CompressionCurve:
    """Build curve metadata, including first crossing and nonmonotonicity."""
    points = [
        CompressionPoint(float(power), float(gain), float(small_signal_gain_db - gain))
        for power, gain in zip(signal_power_dbm, gain_db)
    ]
    compression = np.asarray([point.compression_db for point in points])
    crossings = np.flatnonzero(compression >= 1.0)
    first = float(points[int(crossings[0])].signal_power_dbm) if crossings.size else None
    return CompressionCurve(
        points=points,
        p1db_dbm=first,
        first_1db_crossing_dbm=first,
        number_of_crossings=int(np.count_nonzero((compression[:-1] < 1.0) & (compression[1:] >= 1.0))),
        nonmonotonic_compression=bool(np.any(np.diff(compression) < 0.0)),
    )
