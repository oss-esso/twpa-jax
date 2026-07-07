from __future__ import annotations

import math

PHI0 = 2.067833848e-15
PHI0_REDUCED = PHI0 / (2.0 * math.pi)


def lj_to_ic(lj_h: float, phi0_reduced: float = PHI0_REDUCED) -> float:
    """Convert Josephson inductance to critical current."""
    if lj_h <= 0.0:
        raise ValueError("lj_h must be positive")
    return phi0_reduced / float(lj_h)


def ic_to_lj(ic_a: float, phi0_reduced: float = PHI0_REDUCED) -> float:
    """Convert critical current to Josephson inductance."""
    if ic_a <= 0.0:
        raise ValueError("ic_a must be positive")
    return phi0_reduced / float(ic_a)
