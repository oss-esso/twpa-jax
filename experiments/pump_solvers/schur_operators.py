"""Schur-reduced pump problem: a drop-in for the full Newton-Krylov solver.

``SchurReducedProblem`` exposes the same method surface as
``FullIPMPumpProblem`` (residual_coeffs / tangent_state / jvp_coeffs_with_tangent
/ norms / preconditioner assembly), but every operator is reduced to the
retained nodes and the linear elimination is applied matrix-free through the
prefactored eliminated blocks (see ``schur_partition``). The unconverted full
solver -- including its line search, stall detection, and wall-time budget --
drives it without modification: it only ever sees retained-sized arrays.

The nonlinear Josephson term needs only retained (Josephson-incident) node
fluxes, so no eliminated-node reconstruction happens during the iteration. Full
(H, n) coefficients are reconstructed once after convergence for exp09.

Preconditioners on the retained system use ``D_nn`` as the linear part (the
sparse retained diagonal block), i.e. they drop the dense Schur correction
``D_ne D_ee^{-1} D_en``. That keeps them cheap and sparse; GMRES corrects the
approximation against the exact matrix-free operator. The full nonlinear
coupling -- including the conjugate ``K_{k+q} conj(V_q)`` term -- is retained in
the ``real_coupled`` reduced preconditioner exactly as in the full backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import exp08_full_ipm_pump_solve as exp08

from .schur_partition import (
    SchurPartition,
    assemble_schur_complements,
    back_substitute_full,
    build_partition,
    reduced_linear_apply,
)


@dataclass
class SchurReducedProblem:
    """Retained-node reduction of a :class:`FullIPMPumpProblem`.

    ``linear_apply_mode``:
        ``"assembled"`` -- apply the precomputed sparse S_k (fast for the banded
            ladder complement; the default).
        ``"matrix_free"`` -- apply S_k via the prefactored eliminated back-sub,
            never storing S_k (the GPU-friendly path; benchmarked against
            assembled).
    In both modes the *preconditioners* use the assembled sparse S_k, because the
    cheap D_nn-only approximation (dropping the linear coupling through eliminated
    nodes) does not precondition the reduced system at all.
    """

    full: exp08.FullIPMPumpProblem
    partition: SchurPartition
    linear_apply_mode: str = "assembled"
    # Harmonic-banding cutoffs for the real_coupled retained preconditioner.
    # gamma_hat decays fast with harmonic offset, so the coupled block can keep
    # only |k-q| <= precond_ell_diff_max and k+q <= precond_ell_sum_max and stay
    # an accurate preconditioner while being far cheaper to assemble and factor
    # (the "line"/banded preconditioner). None = keep all couplings (exact
    # real_coupled).
    precond_ell_diff_max: int | None = None
    precond_ell_sum_max: int | None = None

    def __post_init__(self) -> None:
        f = self.full
        self.grid = f.grid
        self.branch = f.branch
        self.H = f.H
        self.part = self.partition
        if self.linear_apply_mode not in ("assembled", "matrix_free"):
            raise ValueError(f"unknown linear_apply_mode {self.linear_apply_mode!r}")
        # The assembled sparse Schur complement is needed for preconditioning in
        # both modes (and for the operator apply in assembled mode).
        assemble_schur_complements(self.part)
        self.n = self.part.m  # retained DOF count (what the solver sees as "n")
        # Retained incidence: branch flux uses only retained node fluxes.
        self.Bphi_r = f.Bphi.tocsr()[self.part.retained].tocsr()
        self.BphiT_r = self.Bphi_r.T.tocsr()
        self.dc_branch_flux = f.dc_branch_flux
        # Source maps to the retained position of the pump node.
        self.source_row = f.source_row
        self.pump_current_a = f.pump_current_a
        pump_pos = int(self.part.retained_pos[f.pump_node_index])
        if pump_pos < 0:
            raise ValueError("pump node is not in the retained set")
        self.pump_pos = pump_pos
        self.nb = self.Bphi_r.shape[1]

    # -------------------------------------------------------- linear operator
    def _lin_apply(self, h: int, vn: np.ndarray) -> np.ndarray:
        """Apply S_k (assembled sparse matvec or matrix-free eliminated solve)."""
        if self.linear_apply_mode == "assembled":
            return self.part.schur[h] @ vn
        return reduced_linear_apply(self.part, h, vn)

    # ------------------------------------------------------------------ source
    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.n), dtype=np.complex128)

    def source_coeffs(self, source_scale: float) -> np.ndarray:
        S = np.zeros((self.H, self.n), dtype=np.complex128)
        S[self.source_row, self.pump_pos] = 0.5 * source_scale * self.pump_current_a
        return S

    # -------------------------------------------------------------- nonlinear
    def _branch_flux_time(self, Xn: np.ndarray) -> np.ndarray:
        x_t = self.grid.synthesize(Xn)  # (nt, m)
        return (self.BphiT_r @ x_t.T).T

    def nonlinear_current_coeffs(self, Xn: np.ndarray) -> np.ndarray:
        psi_t = self._branch_flux_time(Xn) + self.dc_branch_flux[None, :]
        i_t = self.branch.current(psi_t) - self.branch.current(
            self.dc_branch_flux[None, :]
        )
        dn_t = (self.Bphi_r @ i_t.T).T
        return self.grid.project_positive(dn_t)

    def residual_coeffs(self, Xn: np.ndarray, source_scale: float) -> np.ndarray:
        R = np.empty_like(Xn)
        N = self.nonlinear_current_coeffs(Xn)
        S = self.source_coeffs(source_scale)
        for h in range(self.H):
            R[h] = self._lin_apply(h, Xn[h]) + N[h] - S[h]
        return R

    # ---------------------------------------------------------------- tangent
    def tangent_state(self, Xn: np.ndarray) -> exp08.TangentState:
        psi_t = self._branch_flux_time(Xn) + self.dc_branch_flux[None, :]
        gamma_t = self.branch.gamma(psi_t)
        return exp08.TangentState(gamma_t=gamma_t, gamma_mean=np.mean(gamma_t, axis=0))

    def jvp_coeffs_with_tangent(
        self, Vn: np.ndarray, tangent: exp08.TangentState
    ) -> np.ndarray:
        JV = np.empty_like(Vn)
        for h in range(self.H):
            JV[h] = self._lin_apply(h, Vn[h])
        v_t = self.grid.synthesize(Vn)
        dpsi_t = (self.BphiT_r @ v_t.T).T
        di_t = tangent.gamma_t * dpsi_t
        dn_t = (self.Bphi_r @ di_t.T).T
        return JV + self.grid.project_positive(dn_t)

    def jvp_coeffs(self, Xn: np.ndarray, Vn: np.ndarray) -> np.ndarray:
        return self.jvp_coeffs_with_tangent(Vn, self.tangent_state(Xn))

    def spectral_tangent_state(
        self, tangent: exp08.TangentState
    ) -> exp08.SpectralTangentState:
        modes_int = [int(round(k)) for k in self.grid.k]
        needed = sorted(
            {k - q for k in modes_int for q in modes_int}
            | {k + q for k in modes_int for q in modes_int}
        )
        theta = self.grid.omega * self.grid.t
        khat: dict[int, sp.csr_matrix] = {}
        for ell in needed:
            phase = np.exp(-1j * ell * theta)
            gh = np.mean(tangent.gamma_t * phase[:, None], axis=0)
            khat[ell] = (
                self.Bphi_r @ sp.diags(gh, 0, format="csr") @ self.BphiT_r
            ).astype(np.complex128).tocsr()
        return exp08.SpectralTangentState(khat=khat)

    # ------------------------------------------------------------------ norms
    def norms(
        self, Xn: np.ndarray, source_scale: float, compute_time_residual: bool
    ) -> dict[str, float | None]:
        R = self.residual_coeffs(Xn, source_scale)
        R_flat = exp08.pack_complex(R)
        coeff_abs = float(np.linalg.norm(R_flat) / math.sqrt(R_flat.size))
        S_flat = exp08.pack_complex(self.source_coeffs(source_scale))
        src_abs = float(np.linalg.norm(S_flat) / max(math.sqrt(S_flat.size), 1.0))
        # Time residual is computed on the full reconstructed solution (it
        # couples eliminated nodes) -- skipped during the iteration and reported
        # once post-convergence via ``full_time_residual_rel``.
        return {
            "coeff_abs": coeff_abs,
            "coeff_rel": coeff_abs / max(src_abs, 1e-30),
            "time_abs": None,
            "time_rel": None,
        }

    def full_time_residual_rel(self, Xn: np.ndarray, source_scale: float) -> float:
        """Reduced -> full reconstruction, then the exact full time residual."""
        X_full = self.reconstruct_full(Xn)
        Rt = self.full.time_residual(X_full, source_scale)
        St = self.full.source_time(source_scale)
        time_abs = float(np.linalg.norm(Rt.ravel()) / math.sqrt(Rt.size))
        src_t = float(np.linalg.norm(St.ravel()) / max(math.sqrt(St.size), 1.0))
        return time_abs / max(src_t, 1e-30)

    # --------------------------------------------------------- reconstruction
    def reconstruct_full(self, Xn: np.ndarray) -> np.ndarray:
        return back_substitute_full(self.part, Xn)

    # ------------------------------------------------------- preconditioners
    def build_preconditioner_factors(
        self, Xn: np.ndarray, mode: str, tangent: exp08.TangentState | None = None
    ) -> list[spla.SuperLU] | None:
        if mode == "none":
            return None
        if mode == "linear":
            return [spla.splu(self.part.schur[h].tocsc()) for h in range(self.H)]
        if mode != "mean_tangent":
            raise ValueError(f"unknown preconditioner mode {mode!r}")
        if tangent is None:
            tangent = self.tangent_state(Xn)
        Ktan = (
            self.Bphi_r
            @ sp.diags(tangent.gamma_mean, 0, format="csr")
            @ self.BphiT_r
        ).astype(np.complex128).tocsc()
        return [
            spla.splu((self.part.schur[h] + Ktan).tocsc()) for h in range(self.H)
        ]

    def assemble_coupled_preconditioner(
        self, spectral: exp08.SpectralTangentState
    ) -> spla.SuperLU:
        modes_int = [int(round(k)) for k in self.grid.k]
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        rows = []
        for ki, k in enumerate(modes_int):
            row = []
            for qi, q in enumerate(modes_int):
                block = spectral.khat.get(k - q, zero)
                if ki == qi:
                    block = block + self.part.schur[ki]
                row.append(block.tocsr())
            rows.append(row)
        return spla.splu(sp.bmat(rows, format="csc"))

    def assemble_real_coupled_preconditioner(
        self, spectral: exp08.SpectralTangentState
    ) -> spla.SuperLU:
        """Reduced real-packed preconditioner with the conjugate term kept.

        Uses ``D_nn`` as the linear part (dropping the dense Schur correction,
        which only the matrix-free operator carries). The conjugate ``K_{k+q}``
        coupling is preserved exactly, as required.
        """
        modes_int = [int(round(k)) for k in self.grid.k]
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        ldiff = self.precond_ell_diff_max
        lsum = self.precond_ell_sum_max
        jrr, jri, jir, jii = [], [], [], []
        for ki, k in enumerate(modes_int):
            rrr, rri, rir, rii = [], [], [], []
            for qi, q in enumerate(modes_int):
                L = zero
                if ldiff is None or abs(k - q) <= ldiff:
                    L = spectral.khat.get(k - q, zero)
                if ki == qi:
                    L = L + self.part.schur[ki]
                P = zero
                if lsum is None or (k + q) <= lsum:
                    P = spectral.khat.get(k + q, zero)
                Lr, Li = L.real, L.imag
                Pr, Pi = P.real, P.imag
                rrr.append((Lr + Pr).tocsr())
                rri.append((Pi - Li).tocsr())
                rir.append((Li + Pi).tocsr())
                rii.append((Lr - Pr).tocsr())
            jrr.append(rrr)
            jri.append(rri)
            jir.append(rir)
            jii.append(rii)
        top = sp.bmat([[sp.bmat(jrr), sp.bmat(jri)]])
        bot = sp.bmat([[sp.bmat(jir), sp.bmat(jii)]])
        return spla.splu(sp.bmat([[top], [bot]], format="csc"))


def build_schur_problem(
    full: exp08.FullIPMPumpProblem,
    port_indices: list[int],
    *,
    linear_apply_mode: str = "assembled",
    precond_ell_diff_max: int | None = None,
    precond_ell_sum_max: int | None = None,
) -> SchurReducedProblem:
    """Construct a Schur-reduced problem from a full pump problem."""
    part = build_partition(full._linear_blocks, full.Bphi, port_indices)
    return SchurReducedProblem(
        full=full, partition=part, linear_apply_mode=linear_apply_mode,
        precond_ell_diff_max=precond_ell_diff_max,
        precond_ell_sum_max=precond_ell_sum_max,
    )
