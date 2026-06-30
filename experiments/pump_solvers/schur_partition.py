"""Node partition and per-harmonic Schur elimination for the pump solve.

The pump linear operator is block-diagonal across harmonics:

    D_k = K - (k w_p)^2 C + i k w_p G     (one n x n block per pump mode k)

and the nonlinear Josephson term + source touch only the *retained* nodes
(Josephson-incident nodes plus the requested ports). All other nodes are
purely linear-internal and can be eliminated per harmonic:

    [ D_nn  D_ne ] [ X_n ]   [ S_n ]
    [ D_en  D_ee ] [ X_e ] = [ S_e ]

The eliminated rows are linear (S_e is zero on internal nodes and the nonlinear
current is zero there), so

    X_e,k = D_ee,k^{-1} (S_e,k - D_en,k X_n,k).

Because D_ee,k is *constant* in X, it is factored once per harmonic and reused
across every Newton and Krylov iteration -- the structural advantage over the
full real-coupled backend, which re-factors the X-dependent Jacobian each step.

No dense inverse is ever formed: ``D_ee,k^{-1}`` is applied as a sparse LU solve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class SchurPartition:
    """Retained/eliminated node partition with prefactored eliminated blocks.

    Attributes:
        retained: sorted retained node indices (length m).
        eliminated: sorted eliminated node indices (length p), m + p = n.
        n: total node count.
        dee_factors: one ``SuperLU`` per harmonic for D_ee,k (constant in X).
        dne, den, dnn: per-harmonic sparse partition blocks (csr/csc).
        factor_time_s: wall time to build all eliminated-block factorizations.
        retained_pos: map from full node index -> position within retained (or -1).
    """

    retained: np.ndarray
    eliminated: np.ndarray
    n: int
    dee_factors: list[spla.SuperLU]
    dnn: list[sp.csr_matrix]
    dne: list[sp.csr_matrix]
    den: list[sp.csr_matrix]
    factor_time_s: float
    retained_pos: np.ndarray = field(repr=False)
    # Optional assembled sparse Schur complements S_k = D_nn - D_ne D_ee^-1 D_en
    # (constant in X). For ladder topologies these stay banded (~3 nnz/row), so
    # assembling once per frequency beats the matrix-free eliminated back-sub.
    schur: list[sp.csc_matrix] | None = None
    schur_assemble_time_s: float = 0.0
    schur_nnz: int = 0

    @property
    def m(self) -> int:
        return int(self.retained.size)

    @property
    def p(self) -> int:
        return int(self.eliminated.size)

    @property
    def retained_fraction(self) -> float:
        return self.m / max(self.n, 1)


def build_partition(
    linear_blocks: list[sp.spmatrix],
    bphi: sp.spmatrix,
    port_indices: list[int],
) -> SchurPartition:
    """Partition nodes and factor each constant eliminated block once.

    Args:
        linear_blocks: per-harmonic D_k (n x n) sparse matrices.
        bphi: node-branch incidence (n x nb); rows with any nonzero are the
            Josephson-incident (retained) nodes.
        port_indices: node indices of the source/output/interface ports that
            must be retained even if not Josephson-incident.

    Returns:
        A ``SchurPartition`` with prefactored eliminated blocks.

    Raises:
        ValueError: if an eliminated block is singular (a purely linear internal
            node region that the partition cannot eliminate -- indicates the
            retained set is too small).
    """
    bphi = bphi.tocsr()
    n = bphi.shape[0]
    incident = np.unique(bphi.nonzero()[0])
    retained_mask = np.zeros(n, dtype=bool)
    retained_mask[incident] = True
    for p in port_indices:
        retained_mask[int(p)] = True

    retained = np.where(retained_mask)[0]
    eliminated = np.where(~retained_mask)[0]

    retained_pos = np.full(n, -1, dtype=np.int64)
    retained_pos[retained] = np.arange(retained.size)

    dnn: list[sp.csr_matrix] = []
    dne: list[sp.csr_matrix] = []
    den: list[sp.csr_matrix] = []
    dee_factors: list[spla.SuperLU] = []

    t0 = time.perf_counter()
    for Dk in linear_blocks:
        Dk = Dk.tocsr()
        dnn.append(Dk[retained][:, retained].tocsr())
        dne.append(Dk[retained][:, eliminated].tocsr())
        den.append(Dk[eliminated][:, retained].tocsr())
        Dee = Dk[eliminated][:, eliminated].tocsc()
        try:
            dee_factors.append(spla.splu(Dee))
        except RuntimeError as exc:  # singular eliminated block
            raise ValueError(
                "eliminated block D_ee is singular; the retained set is too "
                "small to eliminate the linear-internal nodes"
            ) from exc
    factor_time_s = time.perf_counter() - t0

    return SchurPartition(
        retained=retained,
        eliminated=eliminated,
        n=n,
        dee_factors=dee_factors,
        dnn=dnn,
        dne=dne,
        den=den,
        factor_time_s=factor_time_s,
        retained_pos=retained_pos,
    )


def assemble_schur_complements(
    part: SchurPartition, *, drop_tol: float = 1e-12
) -> SchurPartition:
    """Assemble the sparse Schur complements S_k in place (constant in X).

    S_k = D_nn,k - D_ne,k D_ee,k^{-1} D_en,k. Computed column-block-wise via the
    prefactored eliminated LU (no dense inverse of D_ee is formed). Entries below
    ``drop_tol`` times the per-block max magnitude are dropped to preserve the
    banded sparsity (numerical fill from the dense solve is at round-off level).
    """
    if part.schur is not None:
        return part
    t0 = time.perf_counter()
    sc: list[sp.csc_matrix] = []
    total_nnz = 0
    for h in range(len(part.dnn)):
        Den = part.den[h].tocsc()
        m = Den.shape[1]
        # Only retained columns adjacent to an eliminated node contribute to the
        # Schur correction; for a ladder that is a tiny interface set, so solve
        # just those columns instead of the full (p x m) dense system.
        active = np.unique(Den.nonzero()[1])
        if active.size:
            W_active = part.dee_factors[h].solve(Den[:, active].toarray())
            Wsp = sp.csc_matrix(
                (part.eliminated.size, m), dtype=np.complex128
            )
            Wsp = sp.csc_matrix(
                (W_active.ravel(order="F"),
                 (np.tile(np.arange(part.eliminated.size), active.size),
                  np.repeat(active, part.eliminated.size))),
                shape=(part.eliminated.size, m),
            )
            Sk = sp.csr_matrix(part.dnn[h] - part.dne[h] @ Wsp)
        else:
            Sk = part.dnn[h].tocsr().copy()
        Sk.eliminate_zeros()
        if Sk.nnz:
            thr = drop_tol * float(np.abs(Sk.data).max())
            Sk.data[np.abs(Sk.data) <= thr] = 0.0
            Sk.eliminate_zeros()
        sc.append(Sk.tocsc())
        total_nnz += Sk.nnz
    part.schur = sc
    part.schur_assemble_time_s = time.perf_counter() - t0
    part.schur_nnz = total_nnz
    return part


def restrict(x_full: np.ndarray, part: SchurPartition) -> np.ndarray:
    """Restrict a full (H, n) coefficient array to retained nodes (H, m)."""
    return x_full[:, part.retained]


def reduced_linear_apply(
    part: SchurPartition, h: int, vn: np.ndarray
) -> np.ndarray:
    """Apply the Schur-reduced linear operator for harmonic ``h``.

    S_k V_n = D_nn,k V_n - D_ne,k ( D_ee,k^{-1} ( D_en,k V_n ) ).

    The eliminated solve is a sparse LU back-substitution, never a dense inverse.
    """
    rhs = part.den[h] @ vn
    ye = part.dee_factors[h].solve(np.asarray(rhs, dtype=np.complex128))
    return part.dnn[h] @ vn - part.dne[h] @ ye


def back_substitute_full(
    part: SchurPartition,
    xn: np.ndarray,
    source_eliminated: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the full (H, n) solution from retained coefficients (H, m).

    X_e,k = D_ee,k^{-1} (S_e,k - D_en,k X_n,k). With no source on eliminated
    nodes (the default), this is -D_ee,k^{-1} D_en,k X_n,k.
    """
    H = xn.shape[0]
    x_full = np.zeros((H, part.n), dtype=np.complex128)
    x_full[:, part.retained] = xn
    for h in range(H):
        rhs = -(part.den[h] @ xn[h])
        if source_eliminated is not None:
            rhs = rhs + source_eliminated[h]
        xe = part.dee_factors[h].solve(np.asarray(rhs, dtype=np.complex128))
        x_full[h, part.eliminated] = xe
    return x_full
