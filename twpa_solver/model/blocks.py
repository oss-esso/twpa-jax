"""Reusable topology blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from twpa_solver.model.topology import CircuitBuilder

COUPLER_MODEL_MARKER = "marker"
COUPLER_MODEL_COMPACT = "compact_coupled_inductor"
COUPLER_MODEL_DISTRIBUTED = "distributed_coupled_cell"
COUPLER_MODEL_OLD_HARMONIA = "old_harmonia_cpw_approx"
COUPLER_MODEL_TAXONOMY = (
    COUPLER_MODEL_MARKER,
    COUPLER_MODEL_COMPACT,
    COUPLER_MODEL_DISTRIBUTED,
    COUPLER_MODEL_OLD_HARMONIA,
)


class CircuitBlock(Protocol):
    """Block interface for modular circuit assembly."""

    def apply(self, builder: CircuitBuilder) -> None:
        """Stamp the block into a circuit builder."""


@dataclass(frozen=True)
class PortBlock:
    name: str
    node: int
    z0_ohm: float = 50.0

    def apply(self, builder: CircuitBuilder) -> None:
        builder.add_port(self.name, self.node, self.z0_ohm)


@dataclass(frozen=True)
class DirectionalCouplerBlock:
    """Physical two-line coupled-inductor directional coupler cell."""

    top_start: int
    top_end: int
    bottom_start: int
    bottom_end: int
    inductance_top_h: float
    inductance_bottom_h: float
    coupling_k: float
    shunt_capacitance_top_f: float = 0.0
    shunt_capacitance_bottom_f: float = 0.0
    mutual_capacitance_f: float = 0.0
    pump_source_node: int | None = None
    label: str = "directional_coupler"

    def apply(self, builder: CircuitBuilder) -> None:
        builder.add_coupled_inductors(
            (self.top_start, self.top_end),
            (self.bottom_start, self.bottom_end),
            self.inductance_top_h,
            self.inductance_bottom_h,
            self.coupling_k,
            self.label,
        )
        for node in (self.top_start, self.top_end):
            if self.shunt_capacitance_top_f:
                builder.add_shunt_capacitor(node, 0.5 * self.shunt_capacitance_top_f)
        for node in (self.bottom_start, self.bottom_end):
            if self.shunt_capacitance_bottom_f:
                builder.add_shunt_capacitor(node, 0.5 * self.shunt_capacitance_bottom_f)
        if self.mutual_capacitance_f:
            builder.add_branch_capacitor(
                self.top_start,
                self.bottom_start,
                0.5 * self.mutual_capacitance_f,
            )
            builder.add_branch_capacitor(
                self.top_end,
                self.bottom_end,
                0.5 * self.mutual_capacitance_f,
            )
        if self.pump_source_node is not None:
            builder.add_pump_node(self.pump_source_node)
        builder.metadata.setdefault("coupler_blocks", []).append(
            {
                "label": self.label,
                "model": COUPLER_MODEL_COMPACT,
                "top_branch": [self.top_start, self.top_end],
                "bottom_branch": [self.bottom_start, self.bottom_end],
                "L1_H": self.inductance_top_h,
                "L2_H": self.inductance_bottom_h,
                "k": self.coupling_k,
                "mutual_capacitance_F": self.mutual_capacitance_f,
                "pump_source_node": self.pump_source_node,
            }
        )


@dataclass(frozen=True)
class DirectionalCouplerMarkerBlock:
    """Reduced pump injection marker retained for regression comparisons."""

    node: int
    coupling_conductance_s: float = 0.0

    def apply(self, builder: CircuitBuilder) -> None:
        builder.add_pump_node(self.node)
        if self.coupling_conductance_s:
            builder.add_shunt_conductance(self.node, self.coupling_conductance_s)
        builder.metadata.setdefault("coupler_blocks", []).append(
            {
                "model": "idealized_marker",
                "taxonomy_model": COUPLER_MODEL_MARKER,
                "node": self.node,
                "coupling_conductance_S": self.coupling_conductance_s,
            }
        )


@dataclass(frozen=True)
class LinearCapacitorBlock:
    node: int
    capacitance_f: float

    def apply(self, builder: CircuitBuilder) -> None:
        builder.add_shunt_capacitor(self.node, self.capacitance_f)


@dataclass(frozen=True)
class LinearInductorBlock:
    start_node: int
    end_node: int
    inductance_h: float

    def apply(self, builder: CircuitBuilder) -> None:
        builder.add_series_inductor(self.start_node, self.end_node, self.inductance_h)


@dataclass(frozen=True)
class JosephsonTransmissionLineBlock:
    start_node: int
    num_cells: int
    critical_current_a: float
    shunt_capacitance_f: float
    label_prefix: str = "jtl"

    def apply(self, builder: CircuitBuilder) -> None:
        for idx in range(self.num_cells):
            a = self.start_node + idx
            b = a + 1
            builder.add_josephson(a, b, self.critical_current_a, f"{self.label_prefix}_{idx}")
            builder.add_shunt_capacitor(a, 0.5 * self.shunt_capacitance_f)
            builder.add_shunt_capacitor(b, 0.5 * self.shunt_capacitance_f)


@dataclass(frozen=True)
class RFSQUIDLineBlock:
    """Topology placeholder for future RF-SQUID branch implementation."""

    start_node: int
    num_cells: int

    def apply(self, builder: CircuitBuilder) -> None:
        builder.metadata.setdefault("scaffold_blocks", []).append("RFSQUIDLineBlock")


@dataclass(frozen=True)
class KineticInductanceLineBlock:
    """Topology placeholder for future kinetic inductance branch implementation."""

    start_node: int
    num_cells: int

    def apply(self, builder: CircuitBuilder) -> None:
        builder.metadata.setdefault("scaffold_blocks", []).append("KineticInductanceLineBlock")
