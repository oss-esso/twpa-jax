"""Available power at a matched port.

Every drive port in the production netlists (``designs/ipm_2c_fixed`` and the
JC-parity fixtures) is an ideal current source in parallel with a fixed
conductance ``G0 = 1/Z0`` -- the ``G`` matrix has exactly one nonzero per port,
equal to ``1/Z0``, and nothing else in the circuit is conductive. This is the
standard matched wave-port construction: ``I`` is the incident wave's own
current amplitude (the conductance only absorbs reflections correctly), so
the available power is the traveling-wave form

    P_avail = 0.5 * I^2 * Z0 = I^2 * Z0 / 2

The Norton-generator reading (``I`` as the source's own short-circuit
current, splitting in half across the matched load, giving ``I^2 * Z0 / 8``)
does not apply here since ``I`` is defined as the wave amplitude, not a
generator's short-circuit current. ``norton`` is kept only as a selectable
convention for comparison.
"""

from __future__ import annotations

import math

PORT_POWER_CONVENTIONS = ("norton", "legacy_traveling_wave")
LEGACY_TW_OFFSET_DB = 10.0 * math.log10(4.0)  # 6.020599913279624


def _validate_convention(convention: str) -> None:
    if convention not in PORT_POWER_CONVENTIONS:
        raise ValueError(
            f"convention must be one of {PORT_POWER_CONVENTIONS}, got {convention!r}"
        )


def port_available_power_w(
    current_a: float, z0_ohm: float, convention: str = "legacy_traveling_wave"
) -> float:
    """Peak port drive current (A) -> available power (W).

    ``legacy_traveling_wave`` (default): ``I^2 * Z0 / 2``, correct for a
    matched wave port where ``I`` is the incident wave's own current
    amplitude.
    ``norton``: ``I^2 * Z0 / 8``, treats ``I`` as a Norton generator's
    short-circuit current split across a matched load; kept for comparison.
    """
    _validate_convention(convention)
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    divisor = 8.0 if convention == "norton" else 2.0
    return float(current_a) ** 2 * float(z0_ohm) / divisor


def port_current_from_power_a(
    power_w: float, z0_ohm: float, convention: str = "legacy_traveling_wave"
) -> float:
    """Available power (W) -> peak port drive current (A). Inverse of
    :func:`port_available_power_w`."""
    _validate_convention(convention)
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    if power_w < 0.0:
        raise ValueError("power_w must be non-negative")
    multiplier = 8.0 if convention == "norton" else 2.0
    return math.sqrt(multiplier * float(power_w) / float(z0_ohm))
