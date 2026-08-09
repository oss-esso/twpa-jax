"""Lumped KIMPA reflection-amplifier fixtures and transmission-line ladders."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from twpa_solver.builders.jc_doc import CircuitBuilder
from twpa_solver.core import CircuitMatrices


def add_transmission_line_ladder(
    builder: CircuitBuilder,
    prefix: str,
    node_from: str,
    node_to: str,
    z0_ohm: float,
    electrical_length: str | float,
    reference_frequency_hz: float,
    cells: int,
) -> dict[str, float]:
    """Add a symmetric Pi-discretized transmission-line section."""
    if z0_ohm <= 0 or reference_frequency_hz <= 0 or cells < 1:
        raise ValueError("z0_ohm, reference_frequency_hz and cells must be positive")
    if isinstance(electrical_length, str):
        if electrical_length == "quarter":
            tau = 1.0 / (4.0 * reference_frequency_hz)
        elif electrical_length == "half":
            tau = 1.0 / (2.0 * reference_frequency_hz)
        else:
            raise ValueError("electrical_length must be 'quarter', 'half', or a positive delay")
    else:
        tau = float(electrical_length)
        if tau <= 0:
            raise ValueError("electrical_length delay must be positive")
    total_l = z0_ohm * tau
    total_c = tau / z0_ohm
    l_cell = total_l / cells
    c_cell = total_c / cells
    nodes = [node_from] + [f"{prefix}_{i}" for i in range(1, cells)] + [node_to]
    for i in range(cells):
        builder.linear_inductor(f"{prefix}_L_{i}", nodes[i], nodes[i + 1], l_cell)
    for i, node in enumerate(nodes):
        weight = 0.5 if i in (0, len(nodes) - 1) else 1.0
        builder.capacitor(f"{prefix}_C_{i}", node, "0", weight * c_cell)
    return {"tau_s": tau, "L_line_h": total_l, "C_line_f": total_c}


KIMPA_FIXTURES = {
    "kimpa_ideal_synthesis": {"Lk": 0.999e-9, "Lg": 200e-12, "lines": (67.6, 33.9, 180.0)},
    "kimpa_fabricated_nominal": {"Lk": 56.0**2 * 330e-15 - 200e-12, "Lg": 200e-12, "lines": (80.0, 30.0, 180.0)},
    "kimpa_measured_seed": {"Lk": 0.633e-9, "Lg": 200e-12, "lines": (80.0, 30.0, 180.0)},
    "kimpa_hung_2025": {
        "Lk": 835e-12, "Lg": 200e-12, "lines": (80.0, 30.0, 180.0),
        "Istar2": 3.25e-3, "Istar4": 1.70e-3,
    },
}


def build_kimpa(
    fixture: str = "kimpa_fabricated_nominal",
    *,
    reference_frequency_hz: float = 8.0e9,
    cells: tuple[int, int, int] = (20, 40, 20),
    c_nr_f: float = 330e-15,
    ic_a: float = 1.15e-3,
    qi: float = 1.0e5,
) -> CircuitMatrices:
    """Build one of the three documented KIMPA lumped fixtures."""
    if fixture not in KIMPA_FIXTURES:
        raise ValueError(f"unknown KIMPA fixture {fixture!r}; valid names: {', '.join(KIMPA_FIXTURES)}")
    if len(cells) != 3:
        raise ValueError("cells must contain input, resonator, and output cell counts")
    params = KIMPA_FIXTURES[fixture]
    builder = CircuitBuilder(fixture)
    builder.port("P1", "port", "0", 1)
    builder.resistor("R_port", "port", "0", 50.0)
    z_in, z_res, z_out = params["lines"]
    add_transmission_line_ladder(builder, "input", "port", "n1", z_in, "quarter", reference_frequency_hz, cells[0])
    add_transmission_line_ladder(builder, "resonator", "n1", "n2", z_res, "half", reference_frequency_hz, cells[1])
    add_transmission_line_ladder(builder, "output", "n2", "nr", z_out, "quarter", reference_frequency_hz, cells[2])
    builder.capacitor("C_NR", "nr", "0", c_nr_f)
    builder.linear_inductor("L_g", "nr", "nki", params["Lg"])
    builder.kinetic_inductor(
        "L_k", "nki", "0", params["Lk"], ic_a, "hung_2025",
        params.get("Istar2"), params.get("Istar4"),
    )
    g_nr = 2.0 * np.pi * reference_frequency_hz * c_nr_f / qi
    builder.resistor("R_Qi", "nr", "0", 1.0 / g_nr)
    assembled = builder.assemble()
    metadata = {
        "fixture": fixture,
        "reference_frequency_hz": reference_frequency_hz,
        "C_NR_f": c_nr_f,
        "Lk_h": params["Lk"],
        "Istar2_a": float(assembled["branch_law"].istar2_a[0]),
        "Istar4_a": float(assembled["branch_law"].istar4_a[0]),
        "Lg_h": params["Lg"],
        "Qi": qi,
        "branch_law": assembled["branch_law"].metadata,
        "loss_model": "real_shunt_G_at_reference_frequency",
    }
    return CircuitMatrices(
        C=assembled["C"], G=assembled["G"], K=assembled["K"], Bphi=assembled["Bphi"],
        Ic=assembled["Ic"], Lj=assembled["Lj"], port_to_index=assembled["ports"],
        metadata=metadata, branch_law=assembled["branch_law"],
    )
