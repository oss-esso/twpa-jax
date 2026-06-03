from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from twpa.io.julia_bridge import read_status_json


def decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if hasattr(value, "decode"):
        return value.decode("utf-8")

    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)

    return str(value)


def read_json_dataset(h5: h5py.File, path: str, *, default: Any = None) -> Any:
    if path not in h5:
        return default

    raw = h5[path][()]
    text = decode_h5_scalar(raw)
    return json.loads(text)


@dataclass(frozen=True)
class TopologyArtifact:
    run_dir: Path
    h5_path: Path
    status: str
    simulation_type: str
    circuit_template: str
    backend: str | None
    topology_only: bool | None
    n_ports: int | None
    summary: dict[str, Any]
    topology: dict[str, Any]
    circuit_rows: list[dict[str, Any]]

    @property
    def n_elements(self) -> int | None:
        value = self.summary.get("n_elements")
        return None if value is None else int(value)

    @property
    def n_nodes(self) -> int | None:
        value = self.summary.get("n_nodes")
        return None if value is None else int(value)

    @property
    def element_kind_counts(self) -> dict[str, int]:
        raw = self.summary.get("element_kind_counts", {})
        return {str(k): int(v) for k, v in raw.items()}

    @property
    def circuit_names(self) -> set[str]:
        return {str(row.get("name")) for row in self.circuit_rows}

    @property
    def circuit_roles(self) -> set[str]:
        return {str(row.get("role")) for row in self.circuit_rows if row.get("role") is not None}


def load_topology_artifact(run_dir: str | Path) -> TopologyArtifact:
    run_dir = Path(run_dir)
    status = read_status_json(run_dir / "status.json")

    if status.h5_path is None:
        raise ValueError(f"Run has no h5_path: {run_dir}")

    h5_path = Path(status.h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 artifact: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        attrs = h5.attrs

        backend = decode_h5_scalar(attrs["backend"]) if "backend" in attrs else None
        topology_only = bool(attrs["topology_only"]) if "topology_only" in attrs else None
        n_ports = int(attrs["n_ports"]) if "n_ports" in attrs else None

        summary = read_json_dataset(h5, "topology/summary_json", default={})
        topology = read_json_dataset(h5, "topology/topology_json", default={})
        circuit_rows = read_json_dataset(h5, "topology/circuit_json", default=[])

    if not isinstance(summary, dict):
        raise ValueError(f"topology/summary_json is not a JSON object: {h5_path}")

    if not isinstance(topology, dict):
        raise ValueError(f"topology/topology_json is not a JSON object: {h5_path}")

    if not isinstance(circuit_rows, list):
        raise ValueError(f"topology/circuit_json is not a JSON array: {h5_path}")

    return TopologyArtifact(
        run_dir=run_dir,
        h5_path=h5_path,
        status=status.status,
        simulation_type=status.simulation_type,
        circuit_template=status.circuit_template,
        backend=backend,
        topology_only=topology_only,
        n_ports=n_ports,
        summary=summary,
        topology=topology,
        circuit_rows=circuit_rows,
    )


def require_topology_counts(
    artifact: TopologyArtifact,
    *,
    n_elements: int | None = None,
    element_kind_counts: dict[str, int] | None = None,
    required_names: set[str] | None = None,
    required_roles: set[str] | None = None,
) -> None:
    if n_elements is not None and artifact.n_elements != n_elements:
        raise AssertionError(
            f"Expected n_elements={n_elements}, got {artifact.n_elements}"
        )

    if element_kind_counts is not None:
        actual = artifact.element_kind_counts
        for kind, expected_count in element_kind_counts.items():
            actual_count = actual.get(kind, 0)
            if actual_count != expected_count:
                raise AssertionError(
                    f"Expected {expected_count} elements of kind {kind!r}, "
                    f"got {actual_count}. Actual counts: {actual}"
                )

    if required_names is not None:
        missing = set(required_names) - artifact.circuit_names
        if missing:
            raise AssertionError(f"Missing circuit names: {sorted(missing)}")

    if required_roles is not None:
        missing = set(required_roles) - artifact.circuit_roles
        if missing:
            raise AssertionError(f"Missing circuit roles: {sorted(missing)}")