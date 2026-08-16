"""Level-1 symbolic circuit element builders."""

from __future__ import annotations

import math
from typing import Any

from .elements import ElementRef
from .nodes import Node
from .ports import Port
from .validation import validate_element_ref, validate_node, validate_positive


class PrimitiveBuilders:
    """Mixin implementing primitive graph construction operations."""

    def node(self, name: str | None = None) -> Node:
        """Create and register a symbolic node."""

        if name is not None and not name:
            raise ValueError("node name must not be empty")
        path = name if name is not None else f"node[{len(self.graph.nodes)}]"
        if path in self.graph.named_nodes:
            raise ValueError(f"{path}: duplicate symbolic node name")
        node = Node(
            uid=len(self.graph.nodes),
            owner_id=self.graph.owner_id,
            name=name,
            path=path,
        )
        self.graph.nodes.append(node)
        if name is not None:
            self.graph.named_nodes[path] = node
        return node

    @property
    def ground(self) -> Node:
        """Return the explicit ground node, which compiles to solver node 0."""

        return self.graph.ground

    def _element_name(self, prefix: str, explicit: str | None) -> str:
        if explicit is not None:
            if not explicit:
                raise ValueError("element name must not be empty")
            name = explicit
            if name in self.graph.named_elements:
                raise ValueError(f"{name}: duplicate explicit element name")
        else:
            count = self._name_counters.get(prefix, 0) + 1
            self._name_counters[prefix] = count
            name = f"{prefix}{count}"
        return name

    def _add_element(
        self,
        n1: Node | ElementRef,
        n2: Node | ElementRef,
        value: float | int | str,
        kind: str,
        role: str,
        prefix: str,
        name: str | None = None,
        cell_index: int | None = None,
        element_path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        endpoint_path = name or prefix
        if isinstance(n1, Node):
            validate_node(n1, self.graph.owner_id, endpoint_path)
        else:
            validate_element_ref(n1, self.graph.owner_id, endpoint_path)
        if isinstance(n2, Node):
            validate_node(n2, self.graph.owner_id, endpoint_path)
        else:
            validate_element_ref(n2, self.graph.owner_id, endpoint_path)

        element_name = self._element_name(prefix, name)
        ref = ElementRef(
            n1=n1,
            n2=n2,
            value=value,
            kind=kind,
            role=role,
            name=element_name,
            path=element_path or element_name,
            cell_index=cell_index,
            owner_id=self.graph.owner_id,
            auto_name=auto_name or name is None,
        )
        self.graph.elements.append(ref)
        self.graph.named_elements[ref.path] = ref
        return ref

    def add_resistor(
        self,
        n1: Node,
        n2: Node,
        R: float,
        *,
        name: str | None = None,
        path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add a resistor between two symbolic nodes."""

        resistance = validate_positive(R, "R", name or "resistor")
        return self._add_element(
            n1, n2, resistance, "resistor", "resistor", "R", name,
            element_path=path,
            auto_name=auto_name,
        )

    def add_capacitor(
        self,
        n1: Node,
        n2: Node,
        C: float,
        *,
        name: str | None = None,
        role: str = "capacitor",
        cell_index: int | None = None,
        path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add a capacitor between two symbolic nodes."""

        capacitance = validate_positive(C, "C", name or "capacitor")
        return self._add_element(
            n1,
            n2,
            capacitance,
            "capacitor",
            role,
            "C",
            name,
            cell_index,
            path,
            auto_name=auto_name,
        )

    def add_coupling_capacitor(
        self,
        n1: Node,
        n2: Node,
        C: float,
        *,
        name: str | None = None,
        cell_index: int | None = None,
        path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add a coupling capacitor with the legacy coupling kind and role."""

        capacitance = float(C)
        if not math.isfinite(capacitance):
            raise ValueError(f"{name or 'coupling_capacitor'}: C must be finite")
        return self._add_element(
            n1,
            n2,
            capacitance,
            "coupling_capacitor",
            "coupling_cap",
            "Cc",
            name,
            cell_index,
            path,
            auto_name=auto_name,
        )

    def add_inductor(
        self,
        n1: Node,
        n2: Node,
        L: float,
        *,
        name: str | None = None,
        role: str = "tl_l",
        cell_index: int | None = None,
        path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add a linear inductor between two symbolic nodes."""

        inductance = validate_positive(L, "L", name or "inductor")
        return self._add_element(
            n1,
            n2,
            inductance,
            "linear_inductor",
            role,
            "L",
            name,
            cell_index,
            path,
            auto_name=auto_name,
        )

    def add_jj(
        self,
        n1: Node,
        n2: Node,
        Lj: float,
        Cj: float,
        *,
        name: str | None = None,
        cell_index: int | None = None,
        path: str | None = None,
        lj_name: str | None = None,
        cj_name: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add the Josephson inductance and junction capacitance pair.

        The returned handle is the Josephson-inductor handle. Its
        ``companion`` attribute references the junction-capacitor handle.
        """

        lj = validate_positive(Lj, "Lj", name or "jj")
        cj = validate_positive(Cj, "Cj", name or "jj")
        base = name or self._element_name("JJ", None)
        lj_ref = self._add_element(
            n1,
            n2,
            lj,
            "josephson_inductor",
            "jj_lj",
            "Lj",
            lj_name or f"{base}.Lj",
            cell_index,
            f"{path or base}.Lj",
            auto_name=auto_name,
        )
        cj_ref = self._add_element(
            n1,
            n2,
            cj,
            "capacitor",
            "jj_cj",
            "Cj",
            cj_name or f"{base}.Cj",
            cell_index,
            f"{path or base}.Cj",
            auto_name=auto_name,
        )
        lj_ref.companion = cj_ref
        cj_ref.companion = lj_ref
        return lj_ref

    def add_jj_array(
        self,
        n1: Node,
        n2: Node,
        *,
        Lj: float,
        Cj: float,
        count: int,
        name: str | None = None,
        cell_index: int | None = None,
        path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add one lumped equivalent for a series Josephson-junction array.

        The approximation replaces ``count`` identical series junctions by
        one branch with ``Lj_eff = count * Lj`` and
        ``Cj_eff = Cj / count``.  It adds no symbolic nodes.  The physical
        critical current is the per-junction value supplied by the caller;
        this approximation is intended only when each junction operates well
        below that critical current.  The public API accepts physical kinetic
        inductance values directly and does not expose a squared form.
        """

        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count must be a positive integer")
        return self.add_jj(
            n1,
            n2,
            Lj=float(Lj) * count,
            Cj=float(Cj) / count,
            name=name,
            cell_index=cell_index,
            path=path,
            auto_name=auto_name,
        )

    def add_josephson_inductor(
        self,
        n1: Node,
        n2: Node,
        Lj: float,
        *,
        name: str | None = None,
        cell_index: int | None = None,
        path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add one Josephson branch without an accompanying junction capacitor."""

        inductance = validate_positive(Lj, "Lj", name or "josephson_inductor")
        return self._add_element(
            n1,
            n2,
            inductance,
            "josephson_inductor",
            "jj_lj",
            "Lj",
            name,
            cell_index,
            path,
            auto_name=auto_name,
        )

    def add_mutual_inductor(
        self,
        first: ElementRef,
        second: ElementRef,
        K: float,
        *,
        name: str | None = None,
        path: str | None = None,
        auto_name: bool = False,
    ) -> ElementRef:
        """Add a mutual-inductor coupling coefficient for two linear inductors."""

        validate_element_ref(first, self.graph.owner_id, name or "mutual")
        validate_element_ref(second, self.graph.owner_id, name or "mutual")
        if first.kind != "linear_inductor" or second.kind != "linear_inductor":
            raise ValueError(f"{name or 'mutual'}: endpoints must be linear inductors")
        if first is second:
            raise ValueError(f"{name or 'mutual'}: endpoints must be distinct")
        coupling = float(K)
        if coupling <= -1.0 or coupling >= 1.0:
            raise ValueError(f"{name or 'mutual'}: K must be strictly between -1 and 1")
        return self._add_element(
            first,
            second,
            coupling,
            "mutual_inductor_k",
            "mutual_k",
            "K",
            name,
            element_path=path,
            auto_name=auto_name,
        )

    def add_port(
        self,
        node: Node,
        *,
        number: int,
        impedance: float = 50.0,
        name: str | None = None,
    ) -> Port:
        """Attach a numbered external port to a symbolic node."""

        validate_node(node, self.graph.owner_id, f"port[{number}]")
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise ValueError(f"port[{number}]: number must be a positive integer")
        if number in self.graph.ports:
            raise ValueError(f"port[{number}]: duplicate port number")
        z0 = validate_positive(impedance, "impedance", f"port[{number}]")
        port = Port(number=number, node=node, impedance=z0)
        self.graph.ports[number] = port
        self._add_element(
            node,
            self.graph.ground,
            number,
            "port",
            "port",
            "P",
            name or f"P{number}",
            auto_name=name is None,
        )
        return port

    def set_value(self, ref: ElementRef, value: float | int | str) -> ElementRef:
        """Update one active element handle and return it."""

        validate_element_ref(ref, self.graph.owner_id, ref.path)
        if ref.removed:
            raise ValueError(f"{ref.path}: element has already been removed")
        ref.value = value
        return ref

    def remove(self, ref: ElementRef) -> None:
        """Remove an element from future compilations."""

        validate_element_ref(ref, self.graph.owner_id, ref.path)
        if ref.removed:
            raise ValueError(f"{ref.path}: element has already been removed")
        ref.removed = True
