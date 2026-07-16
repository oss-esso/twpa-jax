from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)

@dataclass
class HarmonicGrid:
    """Time/frequency grid for an arbitrary positive pump-mode list.

    `modes` is the list of positive integer pump-harmonic indices. The legacy
    dense behavior corresponds to modes = [1, 2, ..., H]; the JC odd basis is
    modes = [1, 3, 5, ..., 2K-1]. Synthesis uses the positive-phasor real
    reconstruction x(t) = 2 Re sum_k X_k exp(+i k omega t).
    """

    modes: np.ndarray
    nt: int
    omega: float

    def __post_init__(self) -> None:
        self.k = np.asarray(self.modes, dtype=float).reshape(-1)
        self.harmonics = int(self.k.size)
        if self.harmonics < 1:
            raise ValueError("pump basis must have >= 1 mode")
        max_mode = int(self.k.max())
        if self.nt < 2 * max_mode + 1:
            raise ValueError("--nt must be >= 2*max(mode)+1")
        if self.nt % 2 != 0:
            raise ValueError("--nt should be even")

        self.period = 2.0 * math.pi / self.omega
        self.t = np.arange(self.nt, dtype=float) * self.period / self.nt

        self.E = np.exp(1j * self.omega * self.t[:, None] * self.k[None, :])
        self.E_conj_T_over_nt = self.E.conj().T / self.nt

        logger.debug(
            "harmonic_grid_built nt=%s harmonics=%s max_mode=%s omega=%s modes=%s",
            self.nt, self.harmonics, max_mode, self.omega, [int(m) for m in self.modes],
        )

    def synthesize(self, X: np.ndarray) -> np.ndarray:
        logger.debug("grid_synthesize X_shape=%s", X.shape)
        return 2.0 * np.real(self.E @ X)

    def synthesize_derivative(self, X: np.ndarray, order: int) -> np.ndarray:
        if order == 0:
            return self.synthesize(X)
        multiplier = (1j * self.k * self.omega) ** order
        Xd = multiplier[:, None] * X
        return 2.0 * np.real(self.E @ Xd)

    def project_positive(self, y_t: np.ndarray) -> np.ndarray:
        return self.E_conj_T_over_nt @ y_t


@dataclass
class JosephsonBranchArray:
    Ic: np.ndarray
    phi0: float

    def current(self, psi_t: np.ndarray) -> np.ndarray:
        return self.Ic[None, :] * np.sin(psi_t / self.phi0)

    def gamma(self, psi_t: np.ndarray) -> np.ndarray:
        return (self.Ic[None, :] / self.phi0) * np.cos(psi_t / self.phi0)


def pack_complex(X: np.ndarray) -> np.ndarray:
    z = np.asarray(X, dtype=np.complex128).ravel()
    return np.concatenate([z.real, z.imag])


def unpack_complex(v: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    size = shape[0] * shape[1]
    return (v[:size] + 1j * v[size:2 * size]).reshape(shape)


@dataclass
class TangentState:
    gamma_t: np.ndarray
    gamma_mean: np.ndarray


@dataclass
class SpectralTangentState:
    khat: dict[int, sp.csr_matrix]


@dataclass
class FullPumpProblem:
    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    branch: JosephsonBranchArray
    grid: HarmonicGrid
    pump_node_index: int
    pump_current_a: float
    dc_branch_flux: np.ndarray | None = None
    source_mode: int = 1
    use_real_capacitance: bool = False
    def __post_init__(self) -> None:
        self.C = self.C.tocsr()
        if self.use_real_capacitance:
            self.C = self.C.real.astype(np.complex128).tocsr()
        self.G = self.G.tocsr()
        self.K = self.K.tocsr()
        self.Bphi = self.Bphi.tocsr()
        self.BphiT = self.Bphi.T.tocsr()

        self.n = self.C.shape[0]
        self.H = self.grid.harmonics
        self.nb = self.Bphi.shape[1]

        # Pump current source drives the fundamental pump mode (k == source_mode).
        modes_int = [int(round(k)) for k in self.grid.k]
        if self.source_mode not in modes_int:
            logger.debug(
                "pump_problem_source_mode_invalid source_mode=%s modes=%s",
                self.source_mode, modes_int,
            )
            raise ValueError(
                f"source_mode {self.source_mode} not in pump modes {modes_int}"
            )
        self.source_row = modes_int.index(self.source_mode)

        if self.dc_branch_flux is None:
            self.dc_branch_flux = np.zeros(self.nb, dtype=np.float64)
        else:
            self.dc_branch_flux = np.asarray(self.dc_branch_flux, dtype=np.float64).reshape(-1)
            if self.dc_branch_flux.size != self.nb:
                raise ValueError(f"dc_branch_flux length {self.dc_branch_flux.size} != branch count {self.nb}")

        logger.debug(
            "pump_problem_build n=%s H=%s nb=%s source_mode=%s source_row=%s "
            "pump_node_index=%s use_real_capacitance=%s",
            self.n, self.H, self.nb, self.source_mode, self.source_row,
            self.pump_node_index, self.use_real_capacitance,
        )

        self._linear_blocks = self._build_linear_blocks()

    def _build_linear_blocks(self) -> list[sp.csc_matrix]:
        blocks: list[sp.csc_matrix] = []
        Cc = self.C.astype(np.complex128)
        Gc = self.G.astype(np.complex128)
        Kc = self.K.astype(np.complex128)

        for k in self.grid.k:
            wk = float(k) * self.grid.omega
            Dk = Kc + (-wk * wk) * Cc + (1j * wk) * Gc
            blocks.append(Dk.tocsc())

        logger.debug("linear_block_built count=%s omega=%s", len(blocks), self.grid.omega)
        return blocks

    def zeros(self) -> np.ndarray:
        logger.debug("pump_problem_zeros shape=%s", (self.H, self.n))
        return np.zeros((self.H, self.n), dtype=np.complex128)

    def source_coeffs(self, source_scale: float) -> np.ndarray:
        S = np.zeros((self.H, self.n), dtype=np.complex128)
        source_value = 0.5 * source_scale * self.pump_current_a
        S[self.source_row, self.pump_node_index] = source_value
        logger.debug(
            "source_coeffs source_scale=%s source_row=%s pump_node_index=%s value=%s",
            source_scale, self.source_row, self.pump_node_index, source_value,
        )
        return S

    def source_time(self, source_scale: float) -> np.ndarray:
        src = np.zeros((self.grid.nt, self.n), dtype=float)
        src[:, self.pump_node_index] = (
            source_scale
            * self.pump_current_a
            * np.cos(self.grid.omega * self.grid.t)
        )
        return src

    def branch_flux_time(self, X: np.ndarray) -> np.ndarray:
        x_t = self.grid.synthesize(X)
        return (self.BphiT @ x_t.T).T

    def nonlinear_current_time(self, X: np.ndarray) -> np.ndarray:
        psi_t = self.branch_flux_time(X)
        psi_total_t = psi_t + self.dc_branch_flux[None, :]
        i_t = self.branch.current(psi_total_t) - self.branch.current(self.dc_branch_flux[None, :])
        return (self.Bphi @ i_t.T).T

    def nonlinear_current_coeffs(self, X: np.ndarray) -> np.ndarray:
        logger.debug("nonlinear_current_coeffs_synth X_shape=%s", X.shape)
        return self.grid.project_positive(self.nonlinear_current_time(X))

    def residual_coeffs(self, X: np.ndarray, source_scale: float) -> np.ndarray:
        R = np.empty_like(X)
        Ncoeff = self.nonlinear_current_coeffs(X)
        S = self.source_coeffs(source_scale)

        for h in range(self.H):
            R[h] = self._linear_blocks[h] @ X[h] + Ncoeff[h] - S[h]

        R_norm = float(np.linalg.norm(pack_complex(R)))
        logger.debug(
            "residual_coeffs source_scale=%s X_shape=%s residual_norm=%s",
            source_scale, X.shape, R_norm,
        )
        return R

    def tangent_state(self, X: np.ndarray) -> TangentState:
        x_t = self.grid.synthesize(X)
        psi_t = (self.BphiT @ x_t.T).T
        psi_total_t = psi_t + self.dc_branch_flux[None, :]
        gamma_t = self.branch.gamma(psi_total_t)
        gamma_mean = np.mean(gamma_t, axis=0)
        return TangentState(gamma_t=gamma_t, gamma_mean=gamma_mean)

    def jvp_coeffs_with_tangent(self, V: np.ndarray, tangent: TangentState) -> np.ndarray:
        logger.debug(
            "jvp_apply mode=analytic_aft V_shape=%s (matrix-free, not finite-difference/autodiff)",
            V.shape,
        )
        JV = np.empty_like(V)
        for h in range(self.H):
            JV[h] = self._linear_blocks[h] @ V[h]

        v_t = self.grid.synthesize(V)
        dpsi_t = (self.BphiT @ v_t.T).T

        di_t = tangent.gamma_t * dpsi_t
        dn_t = (self.Bphi @ di_t.T).T
        DN = self.grid.project_positive(dn_t)

        return JV + DN

    def spectral_tangent_state(self, tangent: TangentState) -> SpectralTangentState:
        # For positive unknown modes v_q, the real perturbation contains both
        # +q and -q components. Therefore J_k contains K_{k-q} v_q and
        # K_{k+q} conj(v_q). With an arbitrary positive mode list the needed
        # spectral offsets are exactly {k-q, k+q : k, q in modes}.
        modes_int = [int(round(k)) for k in self.grid.k]
        needed_ell = sorted(
            {k - q for k in modes_int for q in modes_int}
            | {k + q for k in modes_int for q in modes_int}
        )

        theta = self.grid.omega * self.grid.t
        khat: dict[int, sp.csr_matrix] = {}

        for ell in needed_ell:
            phase = np.exp(-1j * ell * theta)
            gamma_hat_ell = np.mean(tangent.gamma_t * phase[:, None], axis=0)

            Kh = (
                self.Bphi
                @ sp.diags(gamma_hat_ell, offsets=0, format="csr")
                @ self.BphiT
            ).astype(np.complex128).tocsr()

            khat[ell] = Kh

        logger.debug(
            "spectral_tangent_state_built n_modes=%s n_offsets=%s ell_min=%s ell_max=%s",
            len(modes_int), len(needed_ell),
            needed_ell[0] if needed_ell else None,
            needed_ell[-1] if needed_ell else None,
        )
        return SpectralTangentState(khat=khat)

    def jvp_coeffs_with_spectral_tangent(
        self,
        V: np.ndarray,
        spectral: SpectralTangentState,
    ) -> np.ndarray:
        if spectral is None:
            raise ValueError("spectral tangent state is required for spectral JVP")

        JV = np.empty_like(V)
        modes_int = [int(round(k)) for k in self.grid.k]

        for k_idx, k in enumerate(modes_int):
            acc = self._linear_blocks[k_idx] @ V[k_idx]

            for q_idx, q in enumerate(modes_int):
                K_k_minus_q = spectral.khat.get(k - q)
                if K_k_minus_q is not None:
                    acc = acc + K_k_minus_q @ V[q_idx]

                K_k_plus_q = spectral.khat.get(k + q)
                if K_k_plus_q is not None:
                    acc = acc + K_k_plus_q @ np.conj(V[q_idx])

            JV[k_idx] = acc

        return JV

    def jvp_coeffs(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        tangent = self.tangent_state(X)
        return self.jvp_coeffs_with_tangent(V, tangent)

    def time_residual(self, X: np.ndarray, source_scale: float) -> np.ndarray:
        x_t = self.grid.synthesize(X)
        dx_t = self.grid.synthesize_derivative(X, order=1)
        ddx_t = self.grid.synthesize_derivative(X, order=2)

        r = (self.C @ ddx_t.T).T
        r = r + (self.G @ dx_t.T).T
        r = r + (self.K @ x_t.T).T
        r = r + self.nonlinear_current_time(X)
        r = r - self.source_time(source_scale)
        return np.asarray(r, dtype=float)

    def norms(
        self,
        X: np.ndarray,
        source_scale: float,
        compute_time_residual: bool,
    ) -> dict[str, float | None]:
        R = self.residual_coeffs(X, source_scale)
        R_flat = pack_complex(R)
        coeff_abs = float(np.linalg.norm(R_flat) / math.sqrt(R_flat.size))

        S = self.source_coeffs(source_scale)
        S_flat = pack_complex(S)
        src_abs = float(np.linalg.norm(S_flat) / max(math.sqrt(S_flat.size), 1.0))
        coeff_rel = coeff_abs / max(src_abs, 1e-30)

        time_abs = None
        time_rel = None

        if compute_time_residual:
            Rt = self.time_residual(X, source_scale)
            time_abs = float(np.linalg.norm(Rt.ravel()) / math.sqrt(Rt.size))

            St = self.source_time(source_scale)
            src_t_abs = float(np.linalg.norm(St.ravel()) / max(math.sqrt(St.size), 1.0))
            time_rel = time_abs / max(src_t_abs, 1e-30)

        return {
            "coeff_abs": coeff_abs,
            "coeff_rel": coeff_rel,
            "time_abs": time_abs,
            "time_rel": time_rel,
        }

    def build_preconditioner_factors(self, X: np.ndarray, mode: str, tangent: TangentState | None = None) -> list[spla.SuperLU] | None:
        logger.debug("build_preconditioner_factors mode=%s H=%s", mode, self.H)
        if mode == "none":
            return None

        if mode == "linear":
            factors = [spla.splu(Dk) for Dk in self._linear_blocks]
            logger.debug("preconditioner_factors_built mode=linear n_factors=%s", len(factors))
            return factors

        if mode != "mean_tangent":
            raise ValueError(f"unknown preconditioner mode {mode!r}")

        if tangent is None:
            tangent = self.tangent_state(X)
        gamma_mean = tangent.gamma_mean

        Ktan = (
            self.Bphi
            @ sp.diags(gamma_mean, offsets=0, format="csr")
            @ self.Bphi.T
        ).astype(np.complex128).tocsc()

        factors = []
        for h in range(self.H):
            Pk = (self._linear_blocks[h] + Ktan).tocsc()
            factors.append(spla.splu(Pk))

        logger.debug("preconditioner_factors_built mode=mean_tangent n_factors=%s", len(factors))
        return factors

    def assemble_coupled_preconditioner(
        self, spectral: SpectralTangentState
    ) -> spla.SuperLU:
        """Full mode-coupled sparse LU preconditioner.

        Assembles the analytic part of the Jacobian as one (H*n) complex matrix:
        block[k][q] = D_k delta_{kq} + Khat_{mode_k - mode_q}. This captures the
        inter-mode (k-q) convolution coupling that the block-diagonal
        mean-tangent preconditioner omits, and is near-exact (it drops only the
        smaller conjugate K_{k+q} term). Required for stiff DC + mutual-inductor
        designs (e.g. FXJTWPA) where block-diagonal GMRES stalls.
        """
        modes_int = [int(round(k)) for k in self.grid.k]
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        rows: list[list[sp.csr_matrix]] = []
        for ki, k in enumerate(modes_int):
            row: list[sp.csr_matrix] = []
            for qi, q in enumerate(modes_int):
                block = spectral.khat.get(k - q, zero)
                if ki == qi:
                    block = block + self._linear_blocks[ki]
                row.append(block.tocsr())
            rows.append(row)
        full = sp.bmat(rows, format="csc")
        logger.debug("coupled_preconditioner_assembled size=%s (spectral_coupled)", full.shape)
        return spla.splu(full)

    def assemble_real_coupled_preconditioner(
        self, spectral: SpectralTangentState
    ) -> spla.SuperLU:
        """Exact real-packed Jacobian LU preconditioner (wraps the matrix)."""
        logger.debug("real_coupled_preconditioner_assemble H=%s n=%s", self.H, self.n)
        return spla.splu(self.real_coupled_jacobian(spectral))

    def _build_real_coupled_matrix(
        self, spectral: SpectralTangentState
    ) -> sp.csc_matrix:
        """Exact real-packed Jacobian as a sparse matrix.

        Builds the full (2*H*n) real Jacobian including BOTH coupling terms:
        per output mode k, the perturbation contribution from input mode q is
        L V_q + P conj(V_q) with L = Khat_{k-q} (+ D_k on the diagonal) and
        P = Khat_{k+q}. In real [Re; Im] coordinates the 2x2 super-block is

            [[ Lr+Pr , Pi-Li ],
             [ Li+Pi , Lr-Pr ]]

        Ordering matches pack_complex: [Re over (H,n); Im over (H,n)]. Because
        this captures the conjugate (k+q) coupling that the complex coupled
        preconditioner drops, it is exact and GMRES converges in ~1 iteration.
        Needed for FXJTWPA where the conjugate term is large.
        """
        modes_int = [int(round(k)) for k in self.grid.k]
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        jrr: list[list[sp.csr_matrix]] = []
        jri: list[list[sp.csr_matrix]] = []
        jir: list[list[sp.csr_matrix]] = []
        jii: list[list[sp.csr_matrix]] = []
        for ki, k in enumerate(modes_int):
            rrr, rri, rir, rii = [], [], [], []
            for qi, q in enumerate(modes_int):
                L = spectral.khat.get(k - q, zero)
                if ki == qi:
                    L = L + self._linear_blocks[ki]
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
        full = sp.bmat([[top], [bot]], format="csc")
        logger.debug(
            "real_coupled_matrix_built size=%s (2*H*n, conjugate k+q term included)",
            full.shape,
        )
        return full

    def real_coupled_jacobian(self, spectral: SpectralTangentState) -> sp.csc_matrix:
        """Public alias: the exact real-packed Jacobian as a sparse matrix."""
        return self._build_real_coupled_matrix(spectral)

# Backwards-compatible alias while old scripts/backends are migrated.
FullIPMPumpProblem = FullPumpProblem
