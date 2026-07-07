from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import scipy.sparse as sp

from twpa_solver.core.constants import PHI0_REDUCED


@dataclass
class CircuitMatrices:
    """General node-flux circuit matrix model.

    Equation convention:

        C xddot + G xdot + K x + Bphi i_J(Bphi.T x) = i_src

    This is geometry-agnostic. It can represent IPM, JPA, JTWPA,
    FQJTWPA, FXJTWPA, or any compatible Josephson circuit.
    """

    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    Ic: np.ndarray
    phi0: float = PHI0_REDUCED
    nodes: np.ndarray | None = None
    port_to_index: dict[int, int] = field(default_factory=dict)
    Lj: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.C = self.C.tocsr()
        self.G = self.G.tocsr()
        self.K = self.K.tocsr()
        self.Bphi = self.Bphi.tocsr()
        self.Ic = np.asarray(self.Ic, dtype=float).reshape(-1)

        if self.Lj is not None:
            self.Lj = np.asarray(self.Lj, dtype=float).reshape(-1)

        n = self.C.shape[0]

        if self.C.shape != (n, n):
            raise ValueError("C must be square")
        if self.G.shape != self.C.shape:
            raise ValueError("G must match C")
        if self.K.shape != self.C.shape:
            raise ValueError("K must match C")
        if self.Bphi.shape[0] != n:
            raise ValueError("Bphi row count must match node count")
        if self.Bphi.shape[1] != self.Ic.size:
            raise ValueError("Bphi branch count must match Ic length")
        if self.Lj is not None and self.Lj.size not in (0, self.Ic.size):
            raise ValueError("Lj length must be zero or match Ic length")

        if self.nodes is None:
            self.nodes = np.arange(n, dtype=np.int64)
        else:
            self.nodes = np.asarray(self.nodes)
            if self.nodes.shape == ():
                self.nodes = np.arange(int(self.nodes), dtype=np.int64)

        self.port_to_index = {int(k): int(v) for k, v in self.port_to_index.items()}
        for port, idx in self.port_to_index.items():
            if idx < 0 or idx >= n:
                raise ValueError(f"port {port} has invalid node index {idx}")

    @property
    def node_count(self) -> int:
        return int(self.C.shape[0])

    @property
    def branch_count(self) -> int:
        return int(self.Bphi.shape[1])

    @property
    def summary(self) -> dict[str, Any]:
        """Backwards-compatible summary dict.

        Old experiment code used LoadedIPM.summary. New code should prefer
        CircuitMatrices.metadata, but this property keeps migrated workflows
        working while we finish the refactor.
        """
        summary = dict(self.metadata) if isinstance(self.metadata, dict) else {}
        summary.setdefault("nodes", self.node_count)
        summary.setdefault("node_count", self.node_count)
        summary.setdefault("jj_branches", self.branch_count)
        summary.setdefault("branch_count", self.branch_count)
        summary.setdefault("ports", {str(k): int(v) for k, v in self.port_to_index.items()})
        summary.setdefault("C_nnz", int(self.C.nnz))
        summary.setdefault("G_nnz", int(self.G.nnz))
        summary.setdefault("K_nnz", int(self.K.nnz))
        summary.setdefault("Bphi_nnz", int(self.Bphi.nnz))
        return summary


def load_circuit(circuit_dir: str | Path) -> CircuitMatrices:
    """Load a circuit from the matrix format used by the experiment scripts.

    The file is still called ipm_arrays.npz for backwards compatibility.
    Internally the object is generic.
    """
    d = Path(circuit_dir)

    for name in ("C.npz", "G.npz", "K.npz", "Bphi.npz", "ipm_arrays.npz"):
        path = d / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")

    C = sp.load_npz(d / "C.npz").tocsr()
    G = sp.load_npz(d / "G.npz").tocsr()
    K = sp.load_npz(d / "K.npz").tocsr()
    Bphi = sp.load_npz(d / "Bphi.npz").tocsr()

    arrays = np.load(d / "ipm_arrays.npz", allow_pickle=True)

    Ic = np.asarray(arrays["Ic"], dtype=float)
    Lj = np.asarray(arrays["Lj"], dtype=float) if "Lj" in arrays.files else None
    phi0 = float(np.asarray(arrays["phi0_reduced"]).reshape(-1)[0])

    nodes = np.asarray(arrays["nodes"]) if "nodes" in arrays.files else None
    if nodes is not None and nodes.shape == ():
        nodes = np.arange(int(nodes), dtype=np.int64)

    port_numbers = np.asarray(arrays["port_numbers"], dtype=int)
    port_indices = np.asarray(arrays["port_indices"], dtype=int)
    port_to_index = {int(p): int(i) for p, i in zip(port_numbers, port_indices)}

    metadata: dict[str, Any] = {}
    for name in ("circuit_summary.json", "ipm_summary.json", "summary.json"):
        path = d / name
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                metadata = loaded if isinstance(loaded, dict) else {}
            except Exception:
                metadata = {}
            break

    return CircuitMatrices(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        Ic=Ic,
        Lj=Lj,
        phi0=phi0,
        nodes=nodes,
        port_to_index=port_to_index,
        metadata=metadata,
    )


def save_circuit(circuit: CircuitMatrices, outdir: str | Path) -> None:
    """Save a CircuitMatrices object in the existing experiment-compatible format."""
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    sp.save_npz(d / "C.npz", circuit.C)
    sp.save_npz(d / "G.npz", circuit.G)
    sp.save_npz(d / "K.npz", circuit.K)
    sp.save_npz(d / "Bphi.npz", circuit.Bphi)

    port_numbers = np.asarray(sorted(circuit.port_to_index), dtype=np.int64)
    port_indices = np.asarray(
        [circuit.port_to_index[int(p)] for p in port_numbers],
        dtype=np.int64,
    )

    Lj = circuit.Lj
    if Lj is None:
        Lj = np.asarray([], dtype=np.float64)

    np.savez(
        d / "ipm_arrays.npz",
        nodes=np.asarray(circuit.nodes),
        node_count=np.asarray(circuit.node_count, dtype=np.int64),
        port_numbers=port_numbers,
        port_indices=port_indices,
        Ic=np.asarray(circuit.Ic, dtype=np.float64),
        Lj=np.asarray(Lj, dtype=np.float64),
        phi0_reduced=np.asarray([circuit.phi0], dtype=np.float64),
    )

    summary = {
        "nodes": circuit.node_count,
        "jj_branches": circuit.branch_count,
        "C_nnz": int(circuit.C.nnz),
        "G_nnz": int(circuit.G.nnz),
        "K_nnz": int(circuit.K.nnz),
        "Bphi_nnz": int(circuit.Bphi.nnz),
        "ports": {str(k): int(v) for k, v in circuit.port_to_index.items()},
        "metadata": circuit.metadata,
    }

    (d / "circuit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
