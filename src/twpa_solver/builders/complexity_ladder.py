"""Small controlled circuits for high-drive Josephson-dynamics ablations.

The ladder deliberately uses the same netlist/stamping path as the production
IPM builder.  Rung 0 is linear and contains an explicit ``Lj``.  Nonlinear
rungs stamp the junction capacitance but leave the Josephson inductance out of
``K``; the loaded ``JosephsonBranchLaw`` supplies ``Ic*sin(phi)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver.builders.ipm import (
    Element,
    IPMParams,
    LossSpec,
    add,
    add_jj,
    add_jtl_element,
    build_matrices,
)
from twpa_solver.core import CircuitMatrices, save_circuit
from twpa_solver.core.constants import PHI0_REDUCED


@dataclass(frozen=True)
class LadderParameters:
    """Production-derived parameters shared by all lossless ladder rungs."""

    lj_h: float = IPMParams.Lj
    cj_f: float = IPMParams.Cj
    cg_f: float = IPMParams.Cg
    source_resistance_ohm: float = IPMParams.Rleft
    load_resistance_ohm: float = IPMParams.Rright
    cell_length_um: float = IPMParams.cell_length_um
    frequency_hz: float = 7.9e9

    @property
    def ic_a(self) -> float:
        return PHI0_REDUCED / self.lj_h

    @property
    def ll_f(self) -> float:
        return IPMParams.Ll_per_um * self.cell_length_um

    @property
    def cl_f(self) -> float:
        return IPMParams.Cl_per_um * self.cell_length_um

    def provenance(self) -> dict[str, Any]:
        return {
            "lj_h": {"value": self.lj_h, "source": "production IPMParams.Lj"},
            "cj_f": {"value": self.cj_f, "source": "production IPMParams.Cj"},
            "cg_f": {"value": self.cg_f, "source": "production IPMParams.Cg"},
            "ic_a": {"value": self.ic_a, "source": "derived phi0/Lj"},
            "source_resistance_ohm": {"value": self.source_resistance_ohm, "source": "production IPMParams.Rleft"},
            "load_resistance_ohm": {"value": self.load_resistance_ohm, "source": "production IPMParams.Rright"},
            "cell_length_um": {"value": self.cell_length_um, "source": "production IPMParams.cell_length_um"},
            "frequency_hz": {"value": self.frequency_hz, "source": "experiment setting"},
        }


def _finish_netlist(
    elements: list[Element], *, ports: dict[int, tuple[int, int]], metadata: dict[str, Any],
) -> CircuitMatrices:
    mats = build_matrices(elements, LossSpec())
    return CircuitMatrices(
        C=mats["C"], G=mats["G"], K=mats["K"], Bphi=mats["Bphi"],
        Ic=mats["Ic"], Lj=mats["Lj"], phi0=PHI0_REDUCED,
        nodes=mats["nodes"], port_to_index=mats["port_vectors"],
        metadata={**metadata, "ports": ports, "matrices": {
            "nodes": int(mats["C"].shape[0]), "jj_count": int(mats["Bphi"].shape[1]),
            "C_nnz": int(mats["C"].nnz), "G_nnz": int(mats["G"].nnz),
            "K_nnz": int(mats["K"].nnz), "Bphi_nnz": int(mats["Bphi"].nnz),
        }},
    )


def build_linear_fixture(params: LadderParameters | None = None) -> CircuitMatrices:
    """Rung 0: explicit linear Lj-Cj-Cg two-node consistency fixture."""
    p = params or LadderParameters()
    elements: list[Element] = []
    add(elements, "L_fixture", 1, 2, p.lj_h, "linear_inductor", role="fixture_lj")
    add(elements, "Cj_fixture", 1, 2, p.cj_f, "capacitor", role="jj_cj")
    add(elements, "Cg_input", 1, 0, p.cg_f, "capacitor", role="jtl_cg")
    add(elements, "Cg_output", 2, 0, p.cg_f, "capacitor", role="jtl_cg")
    add(elements, "R_source", 1, 0, p.source_resistance_ohm, "resistor", role="source_termination")
    add(elements, "R_load", 2, 0, p.load_resistance_ohm, "resistor", role="load_termination")
    add(elements, "P_source", 1, 0, 1, "port")
    add(elements, "P_load", 2, 0, 2, "port")
    return _finish_netlist(elements, ports={1: (1, 0), 2: (2, 0)}, metadata={
        "ladder_rung": 0, "topology": "linear_Lj_Cj_Cg_fixture",
        "nonlinear": False, "Rj": "infinity", "parameter_provenance": p.provenance(),
    })


def build_single_jj(params: LadderParameters | None = None) -> CircuitMatrices:
    """Rung 1: one nonlinear JJ cell with production-like terminations."""
    p = params or LadderParameters()
    elements: list[Element] = []
    add_jj(elements, 1, 2, p.lj_h, p.cj_f, cell_index=0)
    add(elements, "Cg_input", 1, 0, p.cg_f, "capacitor", role="jtl_cg")
    add(elements, "Cg_output", 2, 0, p.cg_f, "capacitor", role="jtl_cg")
    add(elements, "R_source", 1, 0, p.source_resistance_ohm, "resistor", role="source_termination")
    add(elements, "R_load", 2, 0, p.load_resistance_ohm, "resistor", role="load_termination")
    add(elements, "P_source", 1, 0, 1, "port")
    add(elements, "P_load", 2, 0, 2, "port")
    return _finish_netlist(elements, ports={1: (1, 0), 2: (2, 0)}, metadata={
        "ladder_rung": 1, "topology": "single_nonlinear_jj_cell",
        "nonlinear": True, "Rj": "infinity", "parameter_provenance": p.provenance(),
    })


def build_uniform_jtl(n_cells: int, params: LadderParameters | None = None) -> CircuitMatrices:
    """Build a uniform lossless nonlinear JJ transmission line."""
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    p = params or LadderParameters()
    elements: list[Element] = []
    for cell in range(n_cells):
        add_jtl_element(elements, cell + 1, 0, p.cg_f, p.lj_h, p.cj_f, cell_index=cell)
    add(elements, "R_source", 1, 0, p.source_resistance_ohm, "resistor", role="source_termination")
    add(elements, "R_load", n_cells + 1, 0, p.load_resistance_ohm, "resistor", role="load_termination")
    add(elements, "P_source", 1, 0, 1, "port")
    add(elements, "P_load", n_cells + 1, 0, 2, "port")
    topology = "UNIFORM_JTWPA_2508" if n_cells == 2508 else "uniform_nonlinear_jtl"
    return _finish_netlist(elements, ports={1: (1, 0), 2: (n_cells + 1, 0)}, metadata={
        "ladder_rung": 2, "topology": topology,
        "n_cells": n_cells, "nonlinear": True, "Rj": "infinity",
        "parameter_provenance": p.provenance(),
    })


def build_ipm_single_nonlinear_section(
    n_cells: int = IPMParams.array_length,
    params: LadderParameters | None = None,
) -> CircuitMatrices:
    """Build one actual IPM nonlinear section without couplers or side paths."""
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    p = params or LadderParameters()
    elements: list[Element] = []
    add(elements, "P_source", 1, 0, 1, "port")
    add(elements, "R_source", 1, 0, p.source_resistance_ohm, "resistor", role="source_termination")
    add_jtl_element(elements, 1, 0, p.cg_f / 2.0, p.lj_h, p.cj_f, cell_index=0)
    for cell in range(1, n_cells):
        add_jtl_element(elements, cell + 1, 0, p.cg_f, p.lj_h, p.cj_f, cell_index=cell)
    add(elements, f"C{n_cells + 1}_0_JTL_end", n_cells + 1, 0, p.cg_f / 2.0,
        "capacitor", role="jtl_cg")
    add(elements, "R_load", n_cells + 1, 0, p.load_resistance_ohm, "resistor", role="load_termination")
    add(elements, "P_load", n_cells + 1, 0, 2, "port")
    return _finish_netlist(elements, ports={1: (1, 0), 2: (n_cells + 1, 0)}, metadata={
        "ladder_rung": "ipm_single_section",
        "topology": "IPM_SINGLE_NONLINEAR_SECTION",
        "n_cells": n_cells,
        "source_embedding": "production_50_ohm",
        "nonlinear": True, "Rj": "infinity",
        "parameter_provenance": p.provenance(),
    })


def save_ladder_circuit(circuit: CircuitMatrices, outdir: str | Path) -> None:
    """Persist a ladder circuit using the production-compatible matrix format."""
    save_circuit(circuit, outdir)


def build_and_save_ladder(outdir: str | Path, rung: str, *, n_cells: int | None = None) -> CircuitMatrices:
    builders = {
        "linear": build_linear_fixture,
        "single_jj": build_single_jj,
        "ipm_section": build_ipm_single_nonlinear_section,
    }
    if rung == "jtl":
        if n_cells is None:
            raise ValueError("n_cells is required for rung=jtl")
        circuit = build_uniform_jtl(n_cells)
    elif rung in builders:
        circuit = builders[rung](n_cells=n_cells) if rung == "ipm_section" and n_cells is not None else builders[rung]()
    else:
        raise ValueError(f"unknown ladder rung {rung!r}")
    save_ladder_circuit(circuit, outdir)
    return circuit
