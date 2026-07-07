"""Topology assembly for modular nonlinear node-flux circuits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from twpa_solver_old.model.graph import Branch, incidence_matrix
from twpa_solver_old.model.nonlinearities import JosephsonNonlinearity
from twpa_solver_old.model.ports import Port


@dataclass(frozen=True)
class CircuitModel:
    """Reduced node-flux model C phiddot + G phidot + i(phi) = i_src."""

    num_nodes: int
    capacitance_f: np.ndarray
    conductance_s: np.ndarray
    linear_stiffness_h_inv: np.ndarray
    josephson_incidence: np.ndarray
    josephson: JosephsonNonlinearity | None
    ports: tuple[Port, ...]
    pump_nodes: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def nonlinear_current(self, phi: np.ndarray) -> np.ndarray:
        if self.josephson is None or self.josephson_incidence.size == 0:
            return np.zeros_like(np.asarray(phi, dtype=float))
        branch_flux = np.asarray(phi, dtype=float) @ self.josephson_incidence
        current = self.josephson.current(branch_flux)
        return current @ self.josephson_incidence.T

    def nonlinear_derivative_matrices(self, phi_time: np.ndarray) -> np.ndarray:
        """Return K_nl(t) = D diag(dI/dpsi) D^T for each time sample."""
        phi = np.asarray(phi_time, dtype=float)
        if phi.ndim == 1:
            phi = phi[None, :]
        if self.josephson is None or self.josephson_incidence.size == 0:
            return np.broadcast_to(
                np.zeros_like(self.linear_stiffness_h_inv),
                (phi.shape[0], self.num_nodes, self.num_nodes),
            )
        branch_flux = phi @ self.josephson_incidence
        deriv = self.josephson.derivative(branch_flux)
        d = self.josephson_incidence
        return np.einsum("tb,nb,mb->tnm", deriv, d, d)

    def zero_signal_stiffness(self) -> np.ndarray:
        zero = np.zeros((1, self.num_nodes), dtype=float)
        return self.linear_stiffness_h_inv + self.nonlinear_derivative_matrices(zero)[0]

    def port_indices(self) -> list[int]:
        return [port.node for port in self.ports]


@dataclass
class CircuitBuilder:
    """Mutable assembler used by topology blocks."""

    num_nodes: int
    capacitance_f: np.ndarray = field(init=False)
    conductance_s: np.ndarray = field(init=False)
    linear_stiffness_h_inv: np.ndarray = field(init=False)
    josephson_branches: list[Branch] = field(default_factory=list)
    josephson_ics: list[float] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)
    pump_nodes: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capacitance_f = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        self.conductance_s = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        self.linear_stiffness_h_inv = np.zeros((self.num_nodes, self.num_nodes), dtype=float)

    def add_shunt_capacitor(self, node: int, capacitance_f: float) -> None:
        self.capacitance_f[node, node] += float(capacitance_f)

    def add_branch_capacitor(self, a: int, b: int, capacitance_f: float) -> None:
        self._stamp_branch_matrix(self.capacitance_f, a, b, float(capacitance_f))

    def add_shunt_conductance(self, node: int, conductance_s: float) -> None:
        self.conductance_s[node, node] += float(conductance_s)

    def add_series_inductor(self, a: int, b: int, inductance_h: float) -> None:
        self._stamp_branch_matrix(self.linear_stiffness_h_inv, a, b, 1.0 / inductance_h)

    def add_coupled_inductors(
        self,
        branch_1: tuple[int, int],
        branch_2: tuple[int, int],
        inductance_1_h: float,
        inductance_2_h: float,
        coupling_k: float,
        label: str,
    ) -> None:
        stiffness = coupled_inductor_stiffness(
            self.num_nodes,
            branch_1,
            branch_2,
            inductance_1_h,
            inductance_2_h,
            coupling_k,
        )
        self.linear_stiffness_h_inv += stiffness
        self.metadata.setdefault("coupled_inductors", []).append(
            {
                "label": label,
                "branch_1": list(branch_1),
                "branch_2": list(branch_2),
                "L1_H": float(inductance_1_h),
                "L2_H": float(inductance_2_h),
                "k": float(coupling_k),
            }
        )

    def add_josephson(self, a: int, b: int, critical_current_a: float, label: str) -> None:
        self.josephson_branches.append(Branch(a, b, label))
        self.josephson_ics.append(float(critical_current_a))

    def add_port(self, name: str, node: int, z0_ohm: float = 50.0) -> None:
        self.ports.append(Port(name, node, z0_ohm))
        self.add_shunt_conductance(node, 1.0 / z0_ohm)

    def add_pump_node(self, node: int) -> None:
        if node not in self.pump_nodes:
            self.pump_nodes.append(node)

    def build(self) -> CircuitModel:
        d = incidence_matrix(self.num_nodes, self.josephson_branches)
        josephson = None
        if self.josephson_ics:
            josephson = JosephsonNonlinearity(np.asarray(self.josephson_ics, dtype=float))
        return CircuitModel(
            num_nodes=self.num_nodes,
            capacitance_f=self.capacitance_f.copy(),
            conductance_s=self.conductance_s.copy(),
            linear_stiffness_h_inv=self.linear_stiffness_h_inv.copy(),
            josephson_incidence=d,
            josephson=josephson,
            ports=tuple(self.ports),
            pump_nodes=tuple(self.pump_nodes),
            metadata=dict(self.metadata),
        )

    def _stamp_branch_matrix(self, matrix: np.ndarray, a: int, b: int, value: float) -> None:
        matrix[a, a] += value
        matrix[b, b] += value
        matrix[a, b] -= value
        matrix[b, a] -= value


def coupled_inductor_stiffness(
    num_nodes: int,
    branch_1: tuple[int, int],
    branch_2: tuple[int, int],
    inductance_1_h: float,
    inductance_2_h: float,
    coupling_k: float,
) -> np.ndarray:
    """Return B L^-1 B^T for a two-branch coupled inductor."""
    if inductance_1_h <= 0.0 or inductance_2_h <= 0.0:
        raise ValueError("coupled-inductor inductances must be positive")
    if abs(coupling_k) >= 1.0:
        raise ValueError("|coupling_k| must be less than 1")
    b = np.zeros((num_nodes, 2), dtype=float)
    for col, branch in enumerate((branch_1, branch_2)):
        start, end = branch
        for node in branch:
            if not 0 <= node < num_nodes:
                raise ValueError(f"branch node {node} outside 0..{num_nodes - 1}")
        b[start, col] += 1.0
        b[end, col] -= 1.0
    mutual_h = coupling_k * np.sqrt(inductance_1_h * inductance_2_h)
    inductance = np.asarray(
        [[inductance_1_h, mutual_h], [mutual_h, inductance_2_h]],
        dtype=float,
    )
    return b @ np.linalg.inv(inductance) @ b.T


def coupled_inductor_branch_current(
    branch_flux: np.ndarray,
    inductance_1_h: float,
    inductance_2_h: float,
    coupling_k: float,
) -> np.ndarray:
    """Return L^-1 psi for a two-branch coupled inductor."""
    mutual_h = coupling_k * np.sqrt(inductance_1_h * inductance_2_h)
    inductance = np.asarray(
        [[inductance_1_h, mutual_h], [mutual_h, inductance_2_h]],
        dtype=float,
    )
    return np.linalg.solve(inductance, np.asarray(branch_flux, dtype=float))
