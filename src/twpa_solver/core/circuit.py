from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any
import json

import numpy as np
import scipy.sparse as sp

from twpa_solver.core.constants import PHI0_REDUCED
from twpa_solver.core.kinetic import KineticInductorBranchLaw
from twpa_solver.core.nonlinear import CompositeBranchLaw, EffectiveSnailBranchLaw, JosephsonBranchLaw

logger = logging.getLogger(__name__)


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
    branch_law: Any | None = None

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
        logger.debug(
            "circuit_matrices_validated nodes=%d branches=%d nnz=(C:%d,G:%d,K:%d,Bphi:%d) ports=%r",
            self.node_count, self.branch_count, self.C.nnz, self.G.nnz,
            self.K.nnz, self.Bphi.nnz, self.port_to_index,
        )
        if self.has_loss:
            real = np.abs(self.C.data.real)
            tangent = np.divide(
                -self.C.data.imag,
                real,
                out=np.zeros_like(self.C.data.imag, dtype=float),
                where=real != 0.0,
            )
            logger.debug(
                "circuit_loss_detected tan_delta_range=(%s,%s)",
                float(np.min(tangent)), float(np.max(tangent)),
            )

    @property
    def node_count(self) -> int:
        return int(self.C.shape[0])

    @property
    def branch_count(self) -> int:
        return int(self.Bphi.shape[1])

    @property
    def has_loss(self) -> bool:
        """Whether the stamped capacitance contains a non-zero loss term."""
        return bool(np.iscomplexobj(self.C.data) and np.any(self.C.data.imag != 0.0))

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
    logger.debug("circuit_load_start path=%s", d)

    for name in ("C.npz", "G.npz", "K.npz", "Bphi.npz"):
        path = d / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
    arrays_path = d / "ipm_arrays.npz"
    if not arrays_path.exists():
        arrays_path = d / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"missing {d / 'arrays.npz'} or {d / 'ipm_arrays.npz'}")

    C = sp.load_npz(d / "C.npz").tocsr()
    G = sp.load_npz(d / "G.npz").tocsr()
    K = sp.load_npz(d / "K.npz").tocsr()
    Bphi = sp.load_npz(d / "Bphi.npz").tocsr()

    arrays = np.load(arrays_path, allow_pickle=True)

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

    branch_law = None
    law_metadata = metadata.get("metadata", metadata)
    branch_info = metadata.get("branch_law") or law_metadata.get("branch_law", {})
    # Current persistence uses flat per-branch arrays.  This keeps mixed JJ/KI
    # circuits easy to inspect and avoids a recursive schema.  The nested
    # format below remains as a compatibility reader for circuits written by
    # the first implementation of KI persistence.
    if "branch_law_kind" in arrays.files:
        kinds = np.asarray(arrays["branch_law_kind"], dtype=np.int8).reshape(-1)
        if kinds.size != Ic.size or np.any(~np.isin(kinds, (0, 1))):
            raise ValueError("invalid branch_law_kind in persisted circuit")
        ki_columns = np.flatnonzero(kinds == 1)
        jj_columns = np.flatnonzero(kinds == 0)
        if ki_columns.size:
            required = {"ki_lk", "ki_istar2", "ki_istar4"}
            if not required.issubset(arrays.files):
                raise ValueError("kinetic branch kind requires flat KI arrays")
            istar4 = np.asarray(arrays["ki_istar4"], dtype=float)[ki_columns]
            ki_law = KineticInductorBranchLaw(
                np.asarray(arrays["ki_lk"], dtype=float)[ki_columns],
                Ic[ki_columns],
                np.asarray(arrays["ki_istar2"], dtype=float)[ki_columns],
                istar4 if np.all(np.isfinite(istar4)) else None,
                model=str(branch_info.get("model", "hung_2025")),
            )
            if jj_columns.size:
                branch_law = CompositeBranchLaw(
                    (JosephsonBranchLaw(Ic[jj_columns], phi0), ki_law),
                    (jj_columns, ki_columns),
                )
            else:
                branch_law = ki_law
    if branch_law is None and branch_info.get("type") == "effective_snail":
        if "snail_ratio" in arrays.files and "phi_ext" in arrays.files:
            branch_law = EffectiveSnailBranchLaw(
                Ic,
                np.asarray(arrays["snail_ratio"], dtype=float),
                np.asarray(arrays["phi_ext"], dtype=float),
                phi0,
                equilibrium_flux=(
                    np.asarray(arrays["equilibrium_flux"], dtype=float)
                    if "equilibrium_flux" in arrays.files
                    else None
                ),
                external_flux_on_small_junction=bool(
                    branch_info
                    .get("external_flux_on_small_junction", False)
                ),
            )
    elif branch_law is None and branch_info.get("type") == "kinetic_inductor":
        required = {"kinetic_inductance_h", "istar2_a"}
        if required.issubset(arrays.files):
            branch_law = KineticInductorBranchLaw(
                np.asarray(arrays["kinetic_inductance_h"], dtype=float),
                Ic,
                np.asarray(arrays["istar2_a"], dtype=float),
                np.asarray(arrays["istar4_a"], dtype=float) if "istar4_a" in arrays.files else None,
                model=str(branch_info.get("model", "hung_2025")),
                newton_max_iter=int(branch_info.get("newton_max_iter", 20)),
                newton_rtol=float(branch_info.get("newton_rtol", 1e-14)),
            )
    elif branch_law is None and branch_info.get("type") == "composite":
        laws = []
        columns = []
        for index, part_info in enumerate(branch_info.get("parts", [])):
            prefix = f"composite_{index}_"
            if part_info.get("type") == "josephson":
                columns.append(np.asarray(arrays[f"{prefix}columns"], dtype=int))
                laws.append(JosephsonBranchLaw(Ic[columns[-1]], phi0))
            elif part_info.get("type") == "kinetic_inductor":
                columns.append(np.asarray(arrays[f"{prefix}columns"], dtype=int))
                laws.append(KineticInductorBranchLaw(
                    np.asarray(arrays[f"{prefix}kinetic_inductance_h"], dtype=float),
                    Ic[columns[-1]],
                    np.asarray(arrays[f"{prefix}istar2_a"], dtype=float),
                    np.asarray(arrays[f"{prefix}istar4_a"], dtype=float) if f"{prefix}istar4_a" in arrays.files else None,
                    model=str(part_info.get("model", "hung_2025")),
                ))
            else:
                raise ValueError(f"unsupported persisted composite branch type: {part_info.get('type')!r}")
        branch_law = CompositeBranchLaw(tuple(laws), tuple(columns))
    circuit = CircuitMatrices(
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
        branch_law=branch_law,
    )
    logger.debug("circuit_load_complete path=%s summary=%r", d, circuit.summary)
    return circuit


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

    branch_arrays: dict[str, np.ndarray] = {}
    branch_metadata: dict[str, Any] = {}
    if isinstance(circuit.branch_law, EffectiveSnailBranchLaw):
        branch_arrays = {
            "snail_ratio": np.asarray(circuit.branch_law.ratio),
            "phi_ext": np.asarray(circuit.branch_law.phi_ext),
            "equilibrium_flux": np.asarray(
                circuit.branch_law.equilibrium_flux
                if circuit.branch_law.equilibrium_flux is not None
                else np.zeros(circuit.branch_law.ratio.size)
            ),
        }
        branch_metadata = circuit.branch_law.metadata
    elif isinstance(circuit.branch_law, KineticInductorBranchLaw):
        nbranch = circuit.branch_count
        branch_arrays = {
            "branch_law_kind": np.ones(nbranch, dtype=np.int8),
            "ki_lk": np.asarray(circuit.branch_law.kinetic_inductance_h, dtype=np.float64),
            "ki_istar2": np.asarray(circuit.branch_law.istar2_a, dtype=np.float64),
            "ki_istar4": np.asarray(
                circuit.branch_law.istar4_a
                if circuit.branch_law.istar4_a is not None
                else np.full(nbranch, np.nan),
                dtype=np.float64,
            ),
        }
        branch_metadata = {
            **circuit.branch_law.metadata,
            "newton_max_iter": circuit.branch_law.newton_max_iter,
            "newton_rtol": circuit.branch_law.newton_rtol,
        }
    elif isinstance(circuit.branch_law, CompositeBranchLaw):
        branch_metadata = circuit.branch_law.metadata
        nbranch = circuit.branch_count
        kinds = np.zeros(nbranch, dtype=np.int8)
        ki_lk = np.full(nbranch, np.nan, dtype=np.float64)
        ki_istar2 = np.full(nbranch, np.nan, dtype=np.float64)
        ki_istar4 = np.full(nbranch, np.nan, dtype=np.float64)
        for index, (law, columns) in enumerate(zip(circuit.branch_law.laws, circuit.branch_law.columns)):
            if isinstance(law, KineticInductorBranchLaw):
                columns = np.asarray(columns, dtype=np.int64)
                kinds[columns] = 1
                ki_lk[columns] = law.kinetic_inductance_h
                ki_istar2[columns] = law.istar2_a
                if law.istar4_a is not None:
                    ki_istar4[columns] = law.istar4_a
        branch_arrays.update({
            "branch_law_kind": kinds,
            "ki_lk": ki_lk,
            "ki_istar2": ki_istar2,
            "ki_istar4": ki_istar4,
        })
    np.savez(
        d / "ipm_arrays.npz",
        nodes=np.asarray(circuit.nodes),
        node_count=np.asarray(circuit.node_count, dtype=np.int64),
        port_numbers=port_numbers,
        port_indices=port_indices,
        Ic=np.asarray(circuit.Ic, dtype=np.float64),
        Lj=np.asarray(Lj, dtype=np.float64),
        phi0_reduced=np.asarray([circuit.phi0], dtype=np.float64),
        **branch_arrays,
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
        "branch_law": branch_metadata,
    }

    (d / "circuit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
