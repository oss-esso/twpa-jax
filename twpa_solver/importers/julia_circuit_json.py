"""Importer for Harmonia/JosephsonCircuits-style exported circuit JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twpa_solver.model.graph import Branch, incidence_matrix
from twpa_solver.model.nonlinearities import JosephsonNonlinearity
from twpa_solver.model.ports import Port
from twpa_solver.model.topology import CircuitModel
from twpa_solver.model.units import CONSTANTS


GROUND_LABEL = "0"


@dataclass(frozen=True)
class ImportedJuliaCircuit:
    """Imported circuit plus bookkeeping preserved from the Julia netlist."""

    model: CircuitModel
    raw: dict[str, Any]
    node_labels: tuple[str, ...]
    node_index: dict[str, int]
    branch_index: dict[str, int]
    branch_names: tuple[str, ...]
    josephson_branch_names: tuple[str, ...]
    mutual_couplings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _LinearInductor:
    name: str
    node1: str
    node2: str
    inductance_h: float


def import_julia_circuit_json(path: str | Path) -> ImportedJuliaCircuit:
    """Load an exported old-IPM JSON file and assemble a node-flux model."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    elements = list(raw.get("circuit", []))
    if not elements:
        raise ValueError("exported Julia circuit JSON has no circuit elements")

    node_labels = _collect_node_labels(elements)
    node_index = {label: idx for idx, label in enumerate(node_labels)}
    n = len(node_labels)
    c = np.zeros((n, n), dtype=float)
    g = np.zeros((n, n), dtype=float)
    k_lin = np.zeros((n, n), dtype=float)

    linear_inductors: dict[str, _LinearInductor] = {}
    josephson_branches: list[Branch] = []
    josephson_ics: list[float] = []
    josephson_names: list[str] = []
    ports: list[Port] = []
    mutuals: list[dict[str, Any]] = []

    for elem in elements:
        kind = str(elem["kind"])
        name = str(elem["name"])
        a = str(elem["node1"])
        b = str(elem["node2"])
        value = float(elem["value"])
        if kind == "P":
            if b != GROUND_LABEL:
                raise ValueError(f"port {name} is not referenced to ground: {b}")
            ports.append(Port(f"P{int(round(value))}", node_index[a], 50.0))
        elif kind == "R":
            _stamp_branch(g, node_index, a, b, 1.0 / value)
        elif kind == "C":
            _stamp_branch(c, node_index, a, b, value)
        elif kind == "L":
            linear_inductors[name] = _LinearInductor(name, a, b, value)
        elif kind == "Lj":
            josephson_branches.append(Branch(_idx(node_index, a), _idx(node_index, b), name))
            josephson_ics.append(CONSTANTS.reduced_phi0 / value)
            josephson_names.append(name)
        elif kind == "K":
            mutuals.append(
                {
                    "name": name,
                    "branch_1": a,
                    "branch_2": b,
                    "k": value,
                }
            )
        else:
            raise ValueError(f"unsupported Julia element kind {kind!r} for {name}")

    coupled_branch_names = {m["branch_1"] for m in mutuals} | {m["branch_2"] for m in mutuals}
    for name, ind in linear_inductors.items():
        if name not in coupled_branch_names:
            _stamp_branch(k_lin, node_index, ind.node1, ind.node2, 1.0 / ind.inductance_h)

    if coupled_branch_names:
        _stamp_coupled_inductor_network(k_lin, node_index, linear_inductors, mutuals)

    d_j = incidence_matrix(n, josephson_branches) if josephson_branches else np.zeros((n, 0))
    josephson = JosephsonNonlinearity(np.asarray(josephson_ics)) if josephson_ics else None
    pump_nodes = tuple(port.node for port in ports if port.name == "P4")
    metadata = dict(raw.get("metadata", {}))
    metadata.update(
        {
            "importer": "julia_circuit_json",
            "source_schema": raw.get("schema", ""),
            "element_count": len(elements),
            "node_count": n,
            "linear_inductor_count": len(linear_inductors),
            "mutual_coupling_count": len(mutuals),
            "josephson_junction_count": len(josephson_branches),
            "source_convention": raw.get("source_convention", {}),
            "harmonics": raw.get("harmonics", {}),
            "surrogate_topology": False,
        }
    )
    model = CircuitModel(
        num_nodes=n,
        capacitance_f=c,
        conductance_s=g,
        linear_stiffness_h_inv=k_lin,
        josephson_incidence=d_j,
        josephson=josephson,
        ports=tuple(sorted(ports, key=lambda p: int(p.name[1:]))),
        pump_nodes=pump_nodes,
        metadata=metadata,
    )
    return ImportedJuliaCircuit(
        model=model,
        raw=raw,
        node_labels=tuple(node_labels),
        node_index=node_index,
        branch_index={name: idx for idx, name in enumerate(linear_inductors)},
        branch_names=tuple(linear_inductors),
        josephson_branch_names=tuple(josephson_names),
        mutual_couplings=tuple(mutuals),
    )


def _collect_node_labels(elements: list[dict[str, Any]]) -> tuple[str, ...]:
    labels: set[str] = set()
    branch_names: set[str] = {str(e["name"]) for e in elements}
    for elem in elements:
        kind = str(elem["kind"])
        a = str(elem["node1"])
        b = str(elem["node2"])
        if kind == "K":
            continue
        if a != GROUND_LABEL:
            labels.add(a)
        if b != GROUND_LABEL:
            labels.add(b)
        if kind != "K" and (a in branch_names or b in branch_names):
            raise ValueError(f"element {elem['name']} has branch name where node label expected")
    return tuple(sorted(labels, key=_node_sort_key))


def _node_sort_key(label: str) -> tuple[int, str]:
    try:
        return (0, f"{int(label):012d}")
    except ValueError:
        return (1, label)


def _idx(node_index: dict[str, int], label: str) -> int:
    if label == GROUND_LABEL:
        return -1
    return node_index[label]


def _stamp_branch(
    matrix: np.ndarray,
    node_index: dict[str, int],
    node1: str,
    node2: str,
    value: float,
) -> None:
    a = _idx(node_index, node1)
    b = _idx(node_index, node2)
    if a >= 0:
        matrix[a, a] += value
    if b >= 0:
        matrix[b, b] += value
    if a >= 0 and b >= 0:
        matrix[a, b] -= value
        matrix[b, a] -= value


def _stamp_coupled_inductor_network(
    stiffness: np.ndarray,
    node_index: dict[str, int],
    inductors: dict[str, _LinearInductor],
    mutuals: list[dict[str, Any]],
) -> None:
    names = sorted({m["branch_1"] for m in mutuals} | {m["branch_2"] for m in mutuals})
    missing = [name for name in names if name not in inductors]
    if missing:
        raise ValueError(f"K references unknown inductor branches: {missing[:5]}")
    pos = {name: idx for idx, name in enumerate(names)}
    lmat = np.zeros((len(names), len(names)), dtype=float)
    bmat = np.zeros((stiffness.shape[0], len(names)), dtype=float)
    for name in names:
        ind = inductors[name]
        i = pos[name]
        lmat[i, i] = ind.inductance_h
        a = _idx(node_index, ind.node1)
        b = _idx(node_index, ind.node2)
        if a >= 0:
            bmat[a, i] += 1.0
        if b >= 0:
            bmat[b, i] -= 1.0
    for mutual in mutuals:
        n1 = mutual["branch_1"]
        n2 = mutual["branch_2"]
        i = pos[n1]
        j = pos[n2]
        m_h = float(mutual["k"]) * np.sqrt(inductors[n1].inductance_h * inductors[n2].inductance_h)
        lmat[i, j] = m_h
        lmat[j, i] = m_h
    stiffness += bmat @ np.linalg.inv(lmat) @ bmat.T
