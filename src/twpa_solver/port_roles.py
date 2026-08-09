"""Runtime port-role and mixing-order resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def available_ports(circuit_or_ports: Any) -> tuple[int, ...]:
    """Return the sorted numeric ports exposed by a circuit or port mapping."""
    mapping = getattr(circuit_or_ports, "port_to_index", circuit_or_ports)
    if not isinstance(mapping, Mapping):
        raise TypeError("expected a circuit with port_to_index or a port mapping")
    ports = tuple(sorted(int(port) for port in mapping))
    if not ports:
        raise ValueError("the circuit does not expose any ports")
    return ports


def resolve_port_roles(
    circuit_or_ports: Any,
    *,
    pump_port: int | None = None,
    source_port: int | None = None,
    out_port: int | None = None,
) -> dict[str, int]:
    """Resolve pump/source/output roles for one-, two-, and four-port devices.

    The conventional four-port IPM assignment is preserved (pump 4, signal
    input 1, signal output 2).  Smaller devices use the ports they actually
    expose: a two-port circuit uses port 1 as the drive and port 2 as the
    output, while a one-port reflection device uses port 1 for all roles.
    Explicit values always win and are validated against the circuit.
    """
    ports = available_ports(circuit_or_ports)
    port_set = set(ports)

    pump = int(pump_port) if pump_port is not None else (4 if 4 in port_set else ports[0])
    source = int(source_port) if source_port is not None else (1 if 1 in port_set else ports[0])
    if out_port is not None:
        output = int(out_port)
    elif 2 in port_set:
        output = 2
    else:
        output = source

    resolved = {"pump_port": pump, "source_port": source, "out_port": output}
    for role, port in resolved.items():
        if port not in port_set:
            raise ValueError(
                f"{role}={port} is not present; available ports are {list(ports)}"
            )
    return resolved


def external_bias_present(
    *,
    dc_current_a: float | None = None,
    dc_branch_flux_over_phi0: float | None = None,
    dc_solution: object | None = None,
    design_meta: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a runtime configuration contains an external bias."""
    if dc_solution is not None:
        return True
    if dc_current_a is not None and abs(float(dc_current_a)) > 0.0:
        return True
    if dc_branch_flux_over_phi0 is not None and abs(float(dc_branch_flux_over_phi0)) > 0.0:
        return True
    features = (design_meta or {}).get("features", {})
    if isinstance(features, Mapping) and bool(features.get("dc_bias")):
        return True
    return False


def resolve_mixing_order(
    requested: int | str | None,
    *,
    dc_current_a: float | None = None,
    dc_branch_flux_over_phi0: float | None = None,
    dc_solution: object | None = None,
    design_meta: Mapping[str, Any] | None = None,
) -> int:
    """Resolve ``auto`` to 3WM when external bias is present, otherwise 4WM."""
    if requested is None or str(requested).lower() == "auto":
        return 3 if external_bias_present(
            dc_current_a=dc_current_a,
            dc_branch_flux_over_phi0=dc_branch_flux_over_phi0,
            dc_solution=dc_solution,
            design_meta=design_meta,
        ) else 4
    value = int(requested)
    if value not in (3, 4):
        raise ValueError(f"mixing order must be 'auto', 3, or 4; got {requested!r}")
    return value
