"""Public Circuit facade for symbolic graph construction."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

from .blocks import (
    CouplerBuilders,
    JJLineBuilders,
    ParallelLCBuilders,
    RFSquidLineBuilders,
    TransmissionLineBuilders,
)
from .cells import JJCellBuilders, RFSquidCellBuilders, TLCellBuilders
from .compiler import CompiledCircuit, NodeNumbering, compile_graph
from .architectures import IPMBuilders
from .graph import CircuitGraph
from .nodes import Node
from .paths import Path
from .ports import Port
from .primitives import PrimitiveBuilders
from .technology import Technology, load_technology, resolve_builder_parameter


class Circuit(
    PrimitiveBuilders,
    TLCellBuilders,
    JJCellBuilders,
    RFSquidCellBuilders,
    TransmissionLineBuilders,
    JJLineBuilders,
    ParallelLCBuilders,
    RFSquidLineBuilders,
    CouplerBuilders,
    IPMBuilders,
):
    """Own a symbolic circuit graph and compile it to the legacy IR."""

    def __init__(self, name: str, technology: str | Technology | None = None) -> None:
        if not name:
            raise ValueError("circuit name must not be empty")
        self.name = name
        self.technology = (
            load_technology(technology) if isinstance(technology, str) else technology
        )
        self._design_parameters: dict[str, object] = {}
        self.graph = CircuitGraph(owner_id=id(self))
        self._name_counters: dict[str, int] = {}
        self._paths: dict[str, Path] = {}

    def set_design_parameters(self, parameters: Mapping[str, object]) -> None:
        """Set design-level values used when a builder argument is omitted."""

        self._design_parameters = dict(parameters)

    def _resolve_builder_parameter(
        self,
        parameter: str,
        explicit: object,
        technology_defaults: Mapping[str, str],
        path: str,
    ) -> object:
        """Resolve a missing builder argument through the shared technology layer."""

        return resolve_builder_parameter(
            parameter,
            explicit,
            design_parameters=self._design_parameters,
            technology=self.technology,
            technology_defaults=technology_defaults,
            builder_defaults=getattr(self, "BUILDER_DEFAULTS", None),
            path=path,
        )

    def path(self, name: str) -> Path:
        """Create and register a named path with a fresh start node."""

        if not name:
            raise ValueError("path name must not be empty")
        if name in self._paths:
            raise ValueError(f"{name}: duplicate symbolic path name")
        start = self.node(f"{name}.start")
        path = Path(name=name, owner_id=self.graph.owner_id, _nodes=[start])
        self._paths[name] = path
        self.graph.path_nodes[name] = path._nodes
        return path

    def join_path_ends(self, primary: Path, secondary: Path) -> Node:
        """Join two path endpoints into one electrical node.

        The primary endpoint is retained. Existing elements, ports, named
        nodes, and path views that reference the secondary endpoint are
        redirected to it.
        """

        for path in (primary, secondary):
            if path.owner_id != self.graph.owner_id:
                raise ValueError(f"{path.name}: path belongs to another Circuit")
            if self._paths.get(path.name) is not path:
                raise ValueError(f"{path.name}: path is not registered by this Circuit")
        keep = primary.end
        remove = secondary.end
        if keep is remove:
            return keep
        if keep is self.graph.ground or remove is self.graph.ground:
            raise ValueError("path endpoints cannot be joined to ground")

        primary_base = self.graph.legacy_path_bases.get(primary.name)
        if primary_base is not None:
            primary_number = self.graph.legacy_node_numbers.get(
                keep,
                primary_base + len(primary.nodes) - 1,
            )
            self.graph.legacy_node_numbers[keep] = primary_number
        self.graph.legacy_node_numbers.pop(remove, None)

        for element in self.graph.elements:
            if element.n1 is remove:
                element.n1 = keep
            if element.n2 is remove:
                element.n2 = keep
        for number, port in tuple(self.graph.ports.items()):
            if port.node is remove:
                self.graph.ports[number] = Port(number, keep, port.impedance)
        for nodes in self.graph.path_nodes.values():
            for index, node in enumerate(nodes):
                if node is remove:
                    nodes[index] = keep
        for name, node in tuple(self.graph.named_nodes.items()):
            if node is remove:
                self.graph.named_nodes[name] = keep
        self.graph.nodes.remove(remove)
        return keep

    def set_legacy_path_bases(self, bases: Mapping[str, int]) -> None:
        """Register historical solver-number bases for compatibility mode."""

        for path_name, base in bases.items():
            if path_name not in self._paths:
                raise ValueError(f"{path_name}: legacy base requires a registered path")
            if not isinstance(base, int) or isinstance(base, bool) or base <= 0:
                raise ValueError(f"{path_name}: legacy base must be a positive integer")
        self.graph.legacy_path_bases = dict(bases)

    def set_legacy_node_numbers(self, numbers: Mapping[Node, int]) -> None:
        """Register exceptional legacy numbers for internal compatibility nodes."""

        graph_node_ids = {id(node) for node in self.graph.nodes}
        for node, number in numbers.items():
            if node.owner_id != self.graph.owner_id or id(node) not in graph_node_ids:
                raise ValueError(f"{node.path}: node does not belong to this Circuit")
            if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                raise ValueError(f"{node.path}: legacy node number must be positive")
        self.graph.legacy_node_numbers = dict(numbers)

    def compile(
        self,
        node_numbering: NodeNumbering = "creation",
    ) -> CompiledCircuit:
        """Compile the graph with deterministic solver-node allocation."""

        return compile_graph(self.graph, self.name, node_numbering)

    def export_netlist(
        self,
        path: str | PathLike[str] | None = None,
        *,
        node_numbering: NodeNumbering = "creation",
    ) -> str:
        """Compile and export the circuit as a SPICE-like netlist."""

        from .netlist_export import export_netlist

        return export_netlist(self.compile(node_numbering), path)
