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
from .paths import Path
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
