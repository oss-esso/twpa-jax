"""Tuple-mode preconditioners for multitone Newton solves."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


MULTITONE_PRECONDITIONERS = (
    "real_coupled_fast",
    "floquet_sector",
    "spectral_coupled",
    "mean_tangent",
    "linear",
    "none",
)


def resolve_multitone_preconditioner(name: str | None) -> str:
    """Validate a multitone preconditioner, defaulting to the exact path."""
    resolved = "real_coupled_fast" if name is None else str(name)
    if resolved not in MULTITONE_PRECONDITIONERS:
        raise ValueError(
            f"unknown multitone preconditioner {resolved!r}; "
            f"choose from {MULTITONE_PRECONDITIONERS}"
        )
    return resolved


class FloquetSectorPreconditioner:
    """Block factorization by ``abs(q)`` using detuning-independent tangent terms."""

    def __init__(self, problem) -> None:
        self.problem = problem
        self.groups: dict[int, list[int]] = {}
        for index, tone in enumerate(problem.basis.tones):
            self.groups.setdefault(abs(tone.q), []).append(index)
        self.factors: dict[int, spla.SuperLU] = {}
        self.local_indices: dict[int, np.ndarray] = {}
        self.last_factor_backend = ""
        self.last_assembly_runtime_s = 0.0
        self.last_factor_runtime_s = 0.0

    def refactor(self, tangent) -> None:
        started = time.perf_counter()
        self.last_assembly_runtime_s = 0.0
        self.last_factor_runtime_s = 0.0
        spectral = self.problem.spectral_tangent_state(tangent)
        n = self.problem.n
        linear_blocks = getattr(self.problem, "part", None)
        if linear_blocks is not None:
            diagonal_blocks = linear_blocks.schur
        else:
            diagonal_blocks = self.problem._linear_blocks
        zero = sp.csr_matrix((n, n), dtype=np.complex128)
        for order, indices in self.groups.items():
            assembly_started = time.perf_counter()
            tones = [self.problem.basis.tones[i] for i in indices]
            # Build the four real quadrants separately so the local packed
            # solve has the same [Re(all tones); Im(all tones)] ordering.
            rr, ri, ir, ii = [], [], [], []
            for local_i, tone_i in enumerate(tones):
                rr_row, ri_row, ir_row, ii_row = [], [], [], []
                for local_j, tone_j in enumerate(tones):
                    diff = type(tone_i)(tone_i.h - tone_j.h, 0)
                    summ = type(tone_i)(tone_i.h + tone_j.h, tone_i.q + tone_j.q)
                    linear = spectral.khat.get(diff, zero)
                    if local_i == local_j:
                        linear = linear + diagonal_blocks[indices[local_i]]
                    conjugate = spectral.khat.get(summ, zero)
                    lr, li, pr, pi = linear.real, linear.imag, conjugate.real, conjugate.imag
                    rr_row.append((lr + pr).tocsr())
                    ri_row.append((pi - li).tocsr())
                    ir_row.append((li + pi).tocsr())
                    ii_row.append((lr - pr).tocsr())
                rr.append(rr_row); ri.append(ri_row); ir.append(ir_row); ii.append(ii_row)
            matrix = sp.bmat(
                [[sp.bmat(rr), sp.bmat(ri)], [sp.bmat(ir), sp.bmat(ii)]],
                format="csc",
            )
            self.last_assembly_runtime_s += time.perf_counter() - assembly_started
            factor_started = time.perf_counter()
            self.factors[order] = spla.splu(matrix)
            self.last_factor_runtime_s += time.perf_counter() - factor_started
            self.local_indices[order] = np.asarray(indices, dtype=int)
        self.last_factor_backend = "superlu"

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        result = np.zeros_like(rhs)
        n = self.problem.n
        for order, indices in self.local_indices.items():
            real_indices = np.concatenate(
                [index * n + np.arange(n) for index in indices]
            )
            packed_indices = np.concatenate([real_indices, real_indices + self.problem.H * n])
            result[packed_indices] = self.factors[order].solve(rhs[packed_indices])
        return result
