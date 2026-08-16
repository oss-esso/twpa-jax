"""The current two-conductor IPM design expressed through ``Circuit``."""

from __future__ import annotations

from twpa_solver.builders.ipm import IPMParams
from twpa_solver.circuit import Circuit


def build_ipm_2c(params: IPMParams | None = None) -> Circuit:
    """Build the current IPM 2c topology with the public circuit API."""

    configuration = params or IPMParams(start_node_bot=10_000)
    if configuration.num_rows <= configuration.arrays_per_dc + 1:
        raise ValueError("IPMParams.num_rows must leave a non-empty tail section")
    circuit = Circuit("ipm_2c")
    signal = circuit.path("signal")
    pump = circuit.path("pump")
    circuit.set_legacy_path_bases(
        {
            "signal": configuration.start_node_top,
            "pump": configuration.start_node_bot,
        }
    )

    circuit.add_port(signal.start, number=1, impedance=configuration.Rleft)
    circuit.add_resistor(signal.start, circuit.ground, configuration.Rleft)
    circuit.add_transmission_line(
        signal,
        cells=configuration.len1,
        L=configuration.Ll,
        C=configuration.Cl,
        name="input.signal_tl",
    )

    circuit.add_port(pump.start, number=3, impedance=configuration.Rm)
    circuit.add_resistor(pump.start, circuit.ground, configuration.Rm)
    circuit.add_transmission_line(
        pump,
        cells=configuration.len3,
        L=configuration.Ll,
        C=configuration.Cl,
        name="input.pump_tl",
    )

    coupler = {
        "coupling_db": configuration.coupling_dB,
        "frequency": configuration.coupler_freq_hz,
        "z0": configuration.Z0,
        "mode": "cached",
        "cell_length_um": configuration.cell_length_um,
    }
    circuit.add_directional_coupler(
        signal,
        pump,
        **coupler,
        name="input_coupler",
    )

    circuit.add_ipm_section(
        signal,
        pump,
        rows=configuration.arrays_per_dc,
        array_length=configuration.array_length,
        Lj=configuration.Lj,
        Cj=configuration.Cj,
        Cg=configuration.Cg,
        short_tl_cells=configuration.length_of_short_TL,
        long_tl_cells=configuration.length_of_long_TL,
        coupler_section_cells=configuration.coupler_section_length,
        coupler=coupler,
        tl_L=configuration.Ll,
        tl_C=configuration.Cl,
        cell_index_start=0,
        name="section[0]",
    )

    tail_rows = configuration.num_rows - configuration.arrays_per_dc - 1
    circuit.add_ipm_section(
        signal,
        pump,
        rows=tail_rows,
        array_length=configuration.array_length,
        Lj=configuration.Lj,
        Cj=configuration.Cj,
        Cg=configuration.Cg,
        short_tl_cells=configuration.length_of_short_TL,
        long_tl_cells=0,
        coupler_section_cells=0,
        coupler=None,
        tl_L=configuration.Ll,
        tl_C=configuration.Cl,
        cell_index_start=configuration.arrays_per_dc * configuration.array_length,
        name="section[1]",
    )
    circuit.add_jj_line(
        signal,
        cells=configuration.array_length,
        Lj=configuration.Lj,
        Cj=configuration.Cj,
        Cg=configuration.Cg,
        boundary_caps=True,
        cell_index_start=(configuration.num_rows - 1) * configuration.array_length,
        name="section[1].final_array",
    )

    circuit.add_transmission_line(
        signal,
        cells=configuration.len2,
        L=configuration.Ll,
        C=configuration.Cl,
        name="output.signal_tl",
    )
    circuit.add_resistor(signal.end, circuit.ground, configuration.Rright)
    circuit.add_port(signal.end, number=2, impedance=configuration.Rright)
    circuit.add_transmission_line(
        pump,
        cells=configuration.len4,
        L=configuration.Ll,
        C=configuration.Cl,
        name="output.pump_tl",
    )
    circuit.add_resistor(pump.end, circuit.ground, configuration.Rm)
    circuit.add_port(pump.end, number=4, impedance=configuration.Rm)
    return circuit


__all__ = ["build_ipm_2c"]
