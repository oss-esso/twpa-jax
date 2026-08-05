"""Available power at a Norton-terminated port.

Every drive port in the production netlists (``designs/ipm_2c_fixed`` and the
JC-parity fixtures) is an ideal current source in parallel with a fixed
conductance ``G0 = 1/Z0`` -- the ``G`` matrix has exactly one nonzero per port,
equal to ``1/Z0``, and nothing else in the circuit is conductive. A Norton
source of peak current ``I`` splits evenly between its own ``Z0`` and the
``Z0`` load it drives, so the load sees peak current ``I/2`` and

    P_avail = 0.5 * (I/2)^2 * Z0 = I^2 * Z0 / 8

The traveling-wave form ``I^2 * Z0 / 2`` treats ``I`` as a wave amplitude
incident on a *matched* port with no parallel source resistance; applied to a
Norton drive current it overstates available power by exactly
``10*log10(4) = 6.0206 dB``. ``legacy_traveling_wave`` is kept only so
already-published numbers stay reproducible behind an explicit flag.
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
    current_a: float, z0_ohm: float, convention: str = "norton"
) -> float:
    """Peak port drive current (A) -> available power (W).

    ``norton`` (default): ``I^2 * Z0 / 8``, correct for an ideal current
    source in parallel with ``Z0`` driving a matched ``Z0`` load.
    ``legacy_traveling_wave``: ``I^2 * Z0 / 2``, the historical convention
    that treats ``I`` as an incident wave amplitude.
    """
    _validate_convention(convention)
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    divisor = 8.0 if convention == "norton" else 2.0
    return float(current_a) ** 2 * float(z0_ohm) / divisor


def port_current_from_power_a(
    power_w: float, z0_ohm: float, convention: str = "norton"
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
