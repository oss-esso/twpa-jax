"""The current two-conductor IPM design expressed through ``Circuit``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twpa_solver.circuit import Circuit, Technology, load_technology


_DESIGN_DEFAULTS: dict[str, float | int] = {
    "array_length": 418,
}


def _technology_value(technology: Technology, key: str) -> Any:
    """Read one component or architecture value from the loaded preset."""

    if key in technology.components:
        return technology.components[key]
    if key in technology.architecture:
        return technology.architecture[key]
    raise KeyError(f"technology {technology.name!r} has no value for {key!r}")


def _override_value(overrides: Any | None, key: str, default: Any) -> Any:
    """Read an optional legacy parameter object without importing its class."""

    if overrides is None:
        return default
    if isinstance(overrides, Mapping) and key in overrides:
        return overrides[key]
    return getattr(overrides, key, default)


def build_ipm_2c(
    overrides: Any | None = None,
    *,
    technology: Technology | str | None = None,
) -> Circuit:
    """Build the IPM 2c topology using shared technology defaults.

    ``overrides`` is retained as a compatibility boundary for callers that
    still pass an ``IPMParams`` instance.  The design itself does not import
    or construct that legacy parameter object.
    """

    selected_technology: Technology | str = technology or "ipm_default"
    if isinstance(selected_technology, str):
        loaded_technology = load_technology(selected_technology)
    else:
        loaded_technology = selected_technology
    circuit = Circuit("ipm_2c", technology=loaded_technology)
    technology = circuit.technology
    if technology is None:
        raise RuntimeError("ipm_default technology did not load")

    design_values = {
        key: _override_value(
            overrides,
            key,
            _technology_value(technology, key) if key != "array_length" else value,
        )
        for key, value in _DESIGN_DEFAULTS.items()
    }
    for key in ("Lj", "Cj", "Cg"):
        design_values[key] = _override_value(
            overrides, key, _technology_value(technology, key)
        )
    technology_keys = (
        "Ll", "Cl", "coupling_dB", "coupler_freq_hz", "Z0",
        "length_of_short_TL", "length_of_long_TL", "coupler_section_length",
        "len1", "len2", "len3", "len4", "Rleft", "Rright", "Rm",
        "cell_length_um", "num_rows", "arrays_per_dc",
    )
    values = {
        key: _override_value(overrides, key, _technology_value(technology, key))
        for key in technology_keys
    }
    start_node_top = _override_value(overrides, "start_node_top", technology.cursors["signal"])
    start_node_bot = _override_value(overrides, "start_node_bot", technology.cursors["pump"])
    if overrides is not None:
        circuit.set_design_parameters({"Ll": values["Ll"], "Cl": values["Cl"]})
    if int(values["num_rows"]) <= int(values["arrays_per_dc"]) + 1:
        raise ValueError("IPM 2c num_rows must leave a non-empty tail section")

    signal = circuit.path("signal")
    pump = circuit.path("pump")
    circuit.set_legacy_path_bases({
        "signal": int(start_node_top),
        "pump": int(start_node_bot),
    })

    circuit.add_port(signal.start, number=1, impedance=float(values["Rleft"]))
    circuit.add_resistor(signal.start, circuit.ground, float(values["Rleft"]))
    circuit.add_transmission_line(
        signal,
        cells=int(values["len1"]),
        name="input.signal_tl",
    )

    circuit.add_port(pump.start, number=3, impedance=float(values["Rm"]))
    circuit.add_resistor(pump.start, circuit.ground, float(values["Rm"]))
    circuit.add_transmission_line(
        pump,
        cells=int(values["len3"]),
        name="input.pump_tl",
    )

    coupler = {
        "coupling_db": float(values["coupling_dB"]),
        "frequency": float(values["coupler_freq_hz"]),
        "z0": float(values["Z0"]),
        "mode": _override_value(overrides, "coupler_mode", "cached"),
        "cell_length_um": float(values["cell_length_um"]),
    }
    circuit.add_directional_coupler(signal, pump, **coupler, name="input_coupler")

    circuit.add_ipm_section(
        signal,
        pump,
        rows=int(values["arrays_per_dc"]),
        array_length=int(design_values["array_length"]),
        Lj=float(design_values["Lj"]),
        Cj=float(design_values["Cj"]),
        Cg=float(design_values["Cg"]),
        short_tl_cells=int(values["length_of_short_TL"]),
        long_tl_cells=int(values["length_of_long_TL"]),
        coupler_section_cells=int(values["coupler_section_length"]),
        coupler=coupler,
        tl_L=float(values["Ll"]),
        tl_C=float(values["Cl"]),
        cell_index_start=0,
        name="section[0]",
    )

    tail_rows = int(values["num_rows"]) - int(values["arrays_per_dc"]) - 1
    circuit.add_ipm_section(
        signal,
        pump,
        rows=tail_rows,
        array_length=int(design_values["array_length"]),
        Lj=float(design_values["Lj"]),
        Cj=float(design_values["Cj"]),
        Cg=float(design_values["Cg"]),
        short_tl_cells=int(values["length_of_short_TL"]),
        long_tl_cells=0,
        coupler_section_cells=0,
        coupler=None,
        tl_L=float(values["Ll"]),
        tl_C=float(values["Cl"]),
        cell_index_start=int(values["arrays_per_dc"]) * int(design_values["array_length"]),
        name="section[1]",
    )
    circuit.add_jj_line(
        signal,
        cells=int(design_values["array_length"]),
        Lj=float(design_values["Lj"]),
        Cj=float(design_values["Cj"]),
        Cg=float(design_values["Cg"]),
        boundary_caps=True,
        cell_index_start=(int(values["num_rows"]) - 1) * int(design_values["array_length"]),
        name="section[1].final_array",
    )

    circuit.add_transmission_line(signal, cells=int(values["len2"]), name="output.signal_tl")
    circuit.add_resistor(signal.end, circuit.ground, float(values["Rright"]))
    circuit.add_port(signal.end, number=2, impedance=float(values["Rright"]))
    circuit.add_transmission_line(pump, cells=int(values["len4"]), name="output.pump_tl")
    circuit.add_resistor(pump.end, circuit.ground, float(values["Rm"]))
    circuit.add_port(pump.end, number=4, impedance=float(values["Rm"]))
    return circuit


__all__ = ["build_ipm_2c"]
