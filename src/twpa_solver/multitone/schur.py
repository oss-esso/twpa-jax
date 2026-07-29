"""Schur-reduced multitone harmonic-balance problem."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.pump.backends.schur_partition import (
    SchurPartition,
    assemble_schur_complements,
    back_substitute_full,
    build_partition,
    reduced_linear_apply,
)
from twpa_solver.pump.problem import SpectralTangentState, TangentState, pack_complex


class SchurMultiToneProblem:
    """Retained-node view of :class:`FullMultiToneProblem`."""

    def __init__(
        self,
        full: FullMultiToneProblem,
        partition: SchurPartition,
        *,
        linear_apply_mode: str = "assembled",
        preconditioner: str | None = None,
    ) -> None:
        if linear_apply_mode not in {"assembled", "matrix_free"}:
            raise ValueError(f"unknown linear_apply_mode {linear_apply_mode!r}")
        self.full = full
        self.part = partition
        assemble_schur_complements(self.part)
        self.partition = self.part
        self.linear_apply_mode = linear_apply_mode
        from twpa_solver.multitone.preconditioners import resolve_multitone_preconditioner

        self.preconditioner = resolve_multitone_preconditioner(preconditioner)
        self.basis = full.basis
        self.grid = full.grid
        self.branch = full.branch
        self.Bphi_r = full.Bphi[partition.retained].tocsr()
        self.BphiT_r = self.Bphi_r.T.tocsr()
        self.dc_branch_flux = full.dc_branch_flux
        self._linear_blocks = full._linear_blocks
        self.H = full.H
        self.n = partition.m
        self.nb = full.nb
        self.source_start = full.source_path.source_start[:, partition.retained]
        self.source_delta = full.source_path.source_delta[:, partition.retained]

    @property
    def mode_keys(self):
        return list(self.basis.tones)

    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.n), dtype=np.complex128)

    def source_coeffs(self, tau: float) -> np.ndarray:
        return self.source_start + float(tau) * self.source_delta

    def source_delta_coeffs(self) -> np.ndarray:
        return np.array(self.source_delta, copy=True)

    def _lin_apply(self, index: int, value: np.ndarray) -> np.ndarray:
        if self.linear_apply_mode == "assembled":
            return self.part.schur[index] @ value
        return reduced_linear_apply(self.part, index, value)

    def branch_flux_time(self, X: np.ndarray) -> np.ndarray:
        x_t = self.grid.synthesize(X)
        return (self.BphiT_r @ x_t.T).T

    def nonlinear_current_time(self, X: np.ndarray) -> np.ndarray:
        psi = self.branch_flux_time(X) + self.dc_branch_flux[None, :]
        current = self.branch.current(psi) - self.branch.current(self.dc_branch_flux[None, :])
        return (self.Bphi_r @ current.T).T

    def nonlinear_current_coeffs(self, X: np.ndarray) -> np.ndarray:
        return self.grid.project(self.nonlinear_current_time(X))

    def residual_coeffs(self, X: np.ndarray, tau: float) -> np.ndarray:
        nonlinear = self.nonlinear_current_coeffs(X)
        source = self.source_coeffs(tau)
        return np.asarray(
            [self._lin_apply(i, row) + nonlinear[i] - source[i] for i, row in enumerate(X)]
        )

    def tangent_state(self, X: np.ndarray) -> TangentState:
        psi = self.branch_flux_time(X) + self.dc_branch_flux[None, :]
        gamma_t = self.branch.gamma(psi)
        return TangentState(gamma_t=gamma_t, gamma_mean=np.mean(gamma_t, axis=0))

    def jvp_coeffs_with_tangent(self, V: np.ndarray, tangent: TangentState) -> np.ndarray:
        linear = np.asarray([self._lin_apply(i, row) for i, row in enumerate(V)])
        dpsi = self.branch_flux_time(V)
        nonlinear = (self.Bphi_r @ (tangent.gamma_t * dpsi).T).T
        return linear + self.grid.project(nonlinear)

    def jvp_coeffs(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        return self.jvp_coeffs_with_tangent(V, self.tangent_state(X))

    def spectral_tangent_state(self, tangent: TangentState) -> SpectralTangentState:
        needed = sorted(
            {k - q for k in self.basis.tones for q in self.basis.tones}
            | {k + q for k in self.basis.tones for q in self.basis.tones}
        )
        khat: dict[object, sp.csr_matrix] = {}
        for offset in needed:
            phase = self.grid.phase_rows([offset])[0]
            gamma_hat = np.sum(tangent.gamma_t * phase[:, None], axis=0)
            khat[offset] = (
                self.Bphi_r @ sp.diags(gamma_hat, 0) @ self.BphiT_r
            ).astype(np.complex128).tocsr()
        return SpectralTangentState(khat=khat)

    def jvp_coeffs_with_spectral_tangent(
        self, V: np.ndarray, spectral: SpectralTangentState
    ) -> np.ndarray:
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        output = np.empty_like(V)
        for i, k in enumerate(self.basis.tones):
            value = self._lin_apply(i, V[i])
            for j, q in enumerate(self.basis.tones):
                value += spectral.khat.get(k - q, zero) @ V[j]
                value += spectral.khat.get(k + q, zero) @ np.conj(V[j])
            output[i] = value
        return output

    def norms(self, X: np.ndarray, tau: float, compute_time_residual: bool):
        residual = pack_complex(self.residual_coeffs(X, tau))
        coeff_abs = float(np.linalg.norm(residual) / math.sqrt(residual.size))
        source = pack_complex(self.source_coeffs(tau))
        source_abs = float(np.linalg.norm(source) / max(math.sqrt(source.size), 1.0))
        return {
            "coeff_abs": coeff_abs,
            "coeff_rel": coeff_abs / max(source_abs, 1e-30),
            "time_abs": None,
            "time_rel": None,
        }

    def reconstruct_full(self, X: np.ndarray) -> np.ndarray:
        return back_substitute_full(self.part, X)

    def full_time_residual_rel(self, X: np.ndarray, tau: float) -> float:
        full_X = self.reconstruct_full(X)
        result = self.full.time_residual(full_X, tau)
        source = self.full.source_time(tau)
        return float(np.linalg.norm(result) / max(np.linalg.norm(source), 1e-30))

    def build_preconditioner_factors(self, X, mode, tangent=None):
        if mode == "none":
            return None
        if tangent is None:
            tangent = self.tangent_state(X)
        if mode == "linear":
            return [spla.splu(block.tocsc()) for block in self.part.schur]
        if mode == "mean_tangent":
            mean = self.Bphi_r @ sp.diags(tangent.gamma_mean, 0) @ self.BphiT_r
            return [spla.splu((block + mean).tocsc()) for block in self.part.schur]
        if mode == "floquet_sector":
            # The legacy controller applies list preconditioners one tone at a
            # time. Keep this opt-in fallback shape-safe; the coupled sector
            # implementation is selected through real_coupled_fast below.
            spectral = self.spectral_tangent_state(tangent)
            zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
            factors = []
            for i, tone in enumerate(self.basis.tones):
                diagonal = spectral.khat.get(type(tone)(0, 0), zero)
                factors.append(spla.splu((self.part.schur[i] + diagonal).tocsc()))
            return factors
        raise ValueError(f"unknown preconditioner mode {mode!r}")

    def assemble_coupled_preconditioner(self, spectral):
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        rows = []
        for i, k in enumerate(self.basis.tones):
            row = []
            for j, q in enumerate(self.basis.tones):
                block = spectral.khat.get(k - q, zero)
                if i == j:
                    block = block + self.part.schur[i]
                row.append(block)
            rows.append(row)
        return spla.splu(sp.bmat(rows, format="csc"))

    def assemble_real_coupled_preconditioner(self, spectral):
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        groups = [[], [], [], []]
        for i, k in enumerate(self.basis.tones):
            for j, q in enumerate(self.basis.tones):
                linear = spectral.khat.get(k - q, zero)
                if i == j:
                    linear = linear + self.part.schur[i]
                conjugate = spectral.khat.get(k + q, zero)
                lr, li, pr, pi = linear.real, linear.imag, conjugate.real, conjugate.imag
                groups[0].append((lr + pr).tocsr())
                groups[1].append((pi - li).tocsr())
                groups[2].append((li + pi).tocsr())
                groups[3].append((lr - pr).tocsr())
        ntones = self.H
        assembled = [
            sp.bmat([groups[i][r * ntones:(r + 1) * ntones] for r in range(ntones)])
            for i in range(4)
        ]
        return spla.splu(sp.bmat([[assembled[0], assembled[1]], [assembled[2], assembled[3]]], format="csc"))

    def assemble_real_coupled_fast(self, tangent):
        """Exact real-coupled preconditioner with assembly + symbolic reuse.

        ``FastCoupledPreconditioner`` is agnostic to the mode-key type: it only
        needs keys that are hashable, orderable, and support ``+``/``-`` (which
        ``ToneIndex`` does) plus a ``grid.phase_rows`` that accepts them. The
        scatter map and symbolic factorization depend only on the tone basis and
        the circuit graph, so they are built once per partition and cached on it;
        each Newton step then rebuilds ``M.data`` and runs a numeric-only factor.
        Produces the identical matrix to
        :meth:`assemble_real_coupled_preconditioner` (verified to 3.4e-12
        relative on the jtwpa S=10 solve).
        """
        from twpa_solver.multitone.preconditioners import (
            FloquetSectorPreconditioner,
            resolve_multitone_preconditioner,
        )
        from twpa_solver.pump.backends.fast_coupled import (
            FastCoupledPreconditioner,
        )

        selected = resolve_multitone_preconditioner(self.preconditioner)
        if selected == "floquet_sector":
            preconditioner = FloquetSectorPreconditioner(self)
            preconditioner.refactor(tangent)
            return preconditioner
        cached = getattr(self.part, "_fast_coupled_multitone", None)
        if cached is None:
            cached = FastCoupledPreconditioner(self)
            self.part._fast_coupled_multitone = cached
        cached.refactor(tangent)
        return cached


def build_multitone_schur_problem(
    full: FullMultiToneProblem,
    port_indices: list[int],
    *,
    linear_apply_mode: str = "assembled",
    preconditioner: str | None = None,
) -> SchurMultiToneProblem:
    """Build and factor the multitone Schur partition once."""
    part = build_partition(full._linear_blocks, full.Bphi, port_indices)
    return SchurMultiToneProblem(
        full,
        part,
        linear_apply_mode=linear_apply_mode,
        preconditioner=preconditioner,
    )
