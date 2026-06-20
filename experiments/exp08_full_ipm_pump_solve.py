# experiments/exp08_full_ipm_pump_solve.py
"""
Experiment 08: full Python-built IPM pump solve.

No Julia. No JosephsonCircuits.

Input:
    outputs/ipm_python_design/C.npz
    outputs/ipm_python_design/G.npz
    outputs/ipm_python_design/K.npz
    outputs/ipm_python_design/Bphi.npz
    outputs/ipm_python_design/ipm_arrays.npz

Equation:
    C xddot + G xdot + K x + Bphi i_J(Bphi.T x) = i_src

Pump ansatz:
    x_p(t) = 2 Re sum_{k=1..H} X_k exp(i k omega_p t)

Unknowns:
    X_k, k=1..H

Nonlinear current:
    i_J(psi) = Ic sin(psi / phi0_reduced)

JVP:
    DF[X] V = D_k V_k + AFT projection of
        Bphi diag(Ic/phi0 cos(Bphi.T x / phi0)) Bphi.T v

Default:
    matrix-free Newton-Krylov with sparse LU preconditioner
    P_k = D_k + Bphi diag(mean_t gamma(t)) Bphi.T

Run smoke:
    python experiments/exp08_full_ipm_pump_solve.py --harmonics 1 --nt 16 --pump-current-ratio-ic 0.1

Run target-ish:
    python experiments/exp08_full_ipm_pump_solve.py --harmonics 3 --nt 32 --pump-current-ratio-ic 1.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from pump_basis import (  # noqa: E402
    POLICIES,
    PumpBasis,
    load_pump_basis_from_solution,
    promote_solution_to_basis,
    resolve_pump_basis,
)


# =============================================================================
# Harmonic grid
# =============================================================================



def load_dc_solution(dc_solution: str | Path | None, ipm: LoadedIPM) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a static DC operating point for shifted pump solves."""
    if dc_solution is None:
        return None, None

    p = Path(dc_solution)
    if p.is_dir():
        p = p / "dc_solution.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing dc solution: {p}")

    sol = np.load(p)
    x_dc = None
    psi_dc = None

    if "x_dc" in sol.files:
        x_dc = np.asarray(sol["x_dc"], dtype=np.float64).reshape(-1)
        if x_dc.size != ipm.C.shape[0]:
            raise ValueError(f"x_dc length {x_dc.size} != node count {ipm.C.shape[0]}")
        psi_dc = np.asarray(ipm.Bphi.T @ x_dc, dtype=np.float64).reshape(-1)

    if "psi_dc" in sol.files:
        psi_dc_file = np.asarray(sol["psi_dc"], dtype=np.float64).reshape(-1)
        if psi_dc_file.size != ipm.Bphi.shape[1]:
            raise ValueError(f"psi_dc length {psi_dc_file.size} != branch count {ipm.Bphi.shape[1]}")
        psi_dc = psi_dc_file

    if psi_dc is None:
        raise ValueError(f"dc solution {p} must contain x_dc or psi_dc")

    return x_dc, psi_dc


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

    def synthesize(self, X: np.ndarray) -> np.ndarray:
        return 2.0 * np.real(self.E @ X)

    def synthesize_derivative(self, X: np.ndarray, order: int) -> np.ndarray:
        if order == 0:
            return self.synthesize(X)
        multiplier = (1j * self.k * self.omega) ** order
        Xd = multiplier[:, None] * X
        return 2.0 * np.real(self.E @ Xd)

    def project_positive(self, y_t: np.ndarray) -> np.ndarray:
        return self.E_conj_T_over_nt @ y_t


# =============================================================================
# Branch law
# =============================================================================

@dataclass
class JosephsonBranchArray:
    Ic: np.ndarray
    phi0: float

    def current(self, psi_t: np.ndarray) -> np.ndarray:
        return self.Ic[None, :] * np.sin(psi_t / self.phi0)

    def gamma(self, psi_t: np.ndarray) -> np.ndarray:
        return (self.Ic[None, :] / self.phi0) * np.cos(psi_t / self.phi0)


# =============================================================================
# Packing
# =============================================================================

def pack_complex(X: np.ndarray) -> np.ndarray:
    z = np.asarray(X, dtype=np.complex128).ravel()
    return np.concatenate([z.real, z.imag])


def unpack_complex(v: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    size = shape[0] * shape[1]
    return (v[:size] + 1j * v[size:2 * size]).reshape(shape)


# =============================================================================
# Full IPM data loading
# =============================================================================

@dataclass
class LoadedIPM:
    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    Ic: np.ndarray
    Lj: np.ndarray
    phi0: float
    nodes: np.ndarray
    port_to_index: dict[int, int]
    summary: dict[str, Any]


def load_ipm(ipm_dir: str | Path) -> LoadedIPM:
    d = Path(ipm_dir)

    required = [
        "C.npz",
        "G.npz",
        "K.npz",
        "Bphi.npz",
        "ipm_arrays.npz",
    ]
    for name in required:
        path = d / name
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}. Run exp07_python_ipm_design_builder.py --write-matrices first."
            )

    C = sp.load_npz(d / "C.npz").tocsr()
    G = sp.load_npz(d / "G.npz").tocsr()
    K = sp.load_npz(d / "K.npz").tocsr()
    Bphi = sp.load_npz(d / "Bphi.npz").tocsr()

    arrays = np.load(d / "ipm_arrays.npz")
    nodes = arrays["nodes"]
    Ic = arrays["Ic"]
    Lj = arrays["Lj"]
    phi0 = float(arrays["phi0_reduced"][0])

    port_numbers = arrays["port_numbers"].astype(int)
    port_indices = arrays["port_indices"].astype(int)
    port_to_index = {int(p): int(i) for p, i in zip(port_numbers, port_indices)}

    summary_path = d / "ipm_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")
    if G.shape != C.shape or K.shape != C.shape:
        raise ValueError("G and K must match C")
    if Bphi.shape[0] != C.shape[0]:
        raise ValueError("Bphi row count must match node count")
    if Bphi.shape[1] != Ic.size:
        raise ValueError("Bphi branch count must match Ic length")

    return LoadedIPM(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        Ic=Ic,
        Lj=Lj,
        phi0=phi0,
        nodes=nodes,
        port_to_index=port_to_index,
        summary=summary,
    )


# =============================================================================
# Pump problem
# =============================================================================

@dataclass
class TangentState:
    gamma_t: np.ndarray
    gamma_mean: np.ndarray


@dataclass
class SpectralTangentState:
    khat: dict[int, sp.csr_matrix]


@dataclass
class FullIPMPumpProblem:
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

        return blocks

    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.n), dtype=np.complex128)

    def source_coeffs(self, source_scale: float) -> np.ndarray:
        S = np.zeros((self.H, self.n), dtype=np.complex128)
        S[self.source_row, self.pump_node_index] = (
            0.5 * source_scale * self.pump_current_a
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
        return self.grid.project_positive(self.nonlinear_current_time(X))

    def residual_coeffs(self, X: np.ndarray, source_scale: float) -> np.ndarray:
        R = np.empty_like(X)
        Ncoeff = self.nonlinear_current_coeffs(X)
        S = self.source_coeffs(source_scale)

        for h in range(self.H):
            R[h] = self._linear_blocks[h] @ X[h] + Ncoeff[h] - S[h]

        return R

    def tangent_state(self, X: np.ndarray) -> TangentState:
        x_t = self.grid.synthesize(X)
        psi_t = (self.BphiT @ x_t.T).T
        psi_total_t = psi_t + self.dc_branch_flux[None, :]
        gamma_t = self.branch.gamma(psi_total_t)
        gamma_mean = np.mean(gamma_t, axis=0)
        return TangentState(gamma_t=gamma_t, gamma_mean=gamma_mean)

    def jvp_coeffs_with_tangent(self, V: np.ndarray, tangent: TangentState) -> np.ndarray:
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
        if mode == "none":
            return None

        if mode == "linear":
            return [spla.splu(Dk) for Dk in self._linear_blocks]

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
        return spla.splu(full)

    def assemble_real_coupled_preconditioner(
        self, spectral: SpectralTangentState
    ) -> spla.SuperLU:
        """Exact real-packed Jacobian LU preconditioner (wraps the matrix)."""
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
        return full

    def real_coupled_jacobian(self, spectral: SpectralTangentState) -> sp.csc_matrix:
        """Public alias: the exact real-packed Jacobian as a sparse matrix."""
        return self._build_real_coupled_matrix(spectral)


# =============================================================================
# Solver
# =============================================================================

@dataclass
class NewtonKrylovSettings:
    newton_tol: float
    max_newton: int
    gmres_rtol: float
    gmres_atol: float
    gmres_restart: int
    gmres_maxiter: int
    min_alpha: float
    preconditioner: str
    compute_time_residual: bool
    verbose: bool
    continuation_predictor: str
    jvp_mode: str


@dataclass
class StepReport:
    converged: bool
    source_scale: float
    coeff_abs: float
    coeff_rel: float
    time_abs: float | None
    time_rel: float | None
    newton_iterations: int
    gmres_iterations_total: int
    factor_runtime_s: float
    runtime_s: float
    failure_reason: str


class HarmonicNewtonKrylovSolver:
    def __init__(self, settings: NewtonKrylovSettings):
        self.settings = settings

    def solve_one(
        self,
        problem: FullIPMPumpProblem,
        X0: np.ndarray,
        source_scale: float,
    ) -> tuple[np.ndarray, StepReport]:
        s = self.settings
        t0 = time.perf_counter()
        X = np.array(X0, dtype=np.complex128, copy=True)

        nrm = problem.norms(X, source_scale, s.compute_time_residual)

        if s.verbose:
            msg = (
                f"  init: lambda={source_scale:.6f} "
                f"coeff_rel={nrm['coeff_rel']:.3e}"
            )
            if nrm["time_rel"] is not None:
                msg += f" time_rel={nrm['time_rel']:.3e}"
            print(msg)

        if nrm["coeff_rel"] < s.newton_tol:
            return X, self._make_report(
                True, source_scale, nrm, 0, 0, 0.0, t0, ""
            )

        shape = X.shape
        dim_real = 2 * shape[0] * shape[1]
        gmres_total = 0
        factor_total = 0.0
        failure_reason = ""

        for it in range(1, s.max_newton + 1):
            R = problem.residual_coeffs(X, source_scale)
            rhs = -pack_complex(R)

            tangent = problem.tangent_state(X)
            spectral_tangent = None
            if s.jvp_mode == "spectral" or s.preconditioner in ("spectral_coupled", "real_coupled"):
                spectral_tangent = problem.spectral_tangent_state(tangent)

            tf = time.perf_counter()
            coupled_factor = None
            real_factor = None
            if s.preconditioner == "real_coupled":
                real_factor = problem.assemble_real_coupled_preconditioner(spectral_tangent)
                factors = None
            elif s.preconditioner == "spectral_coupled":
                coupled_factor = problem.assemble_coupled_preconditioner(spectral_tangent)
                factors = None
            else:
                factors = problem.build_preconditioner_factors(X, s.preconditioner, tangent=tangent)
            factor_s = time.perf_counter() - tf
            factor_total += factor_s

            def matvec(v_real: np.ndarray) -> np.ndarray:
                V = unpack_complex(v_real, shape)
                JV = problem.jvp_coeffs_with_tangent(V, tangent)
                return pack_complex(JV)

            Aop = spla.LinearOperator(
                shape=(dim_real, dim_real),
                matvec=matvec,
                dtype=np.float64,
            )

            Mop = None
            if real_factor is not None:
                def psolve_real(v_real: np.ndarray) -> np.ndarray:
                    return real_factor.solve(v_real)

                Mop = spla.LinearOperator(
                    shape=(dim_real, dim_real),
                    matvec=psolve_real,
                    dtype=np.float64,
                )
            elif coupled_factor is not None:
                def psolve_coupled(v_real: np.ndarray) -> np.ndarray:
                    V = unpack_complex(v_real, shape)
                    z = coupled_factor.solve(V.reshape(-1))
                    return pack_complex(z.reshape(shape))

                Mop = spla.LinearOperator(
                    shape=(dim_real, dim_real),
                    matvec=psolve_coupled,
                    dtype=np.float64,
                )
            elif factors is not None:
                def psolve(v_real: np.ndarray) -> np.ndarray:
                    V = unpack_complex(v_real, shape)
                    Z = np.empty_like(V)
                    for h in range(problem.H):
                        Z[h] = factors[h].solve(V[h])
                    return pack_complex(Z)

                Mop = spla.LinearOperator(
                    shape=(dim_real, dim_real),
                    matvec=psolve,
                    dtype=np.float64,
                )

            gmres_counter = {"n": 0}

            def cb(_pr_norm: float) -> None:
                gmres_counter["n"] += 1

            delta_real, info = gmres_call(
                A=Aop,
                b=rhs,
                M=Mop,
                rtol=s.gmres_rtol,
                atol=s.gmres_atol,
                restart=s.gmres_restart,
                maxiter=s.gmres_maxiter,
                callback=cb,
            )
            gmres_total += gmres_counter["n"]

            if info != 0:
                failure_reason = f"GMRES did not fully converge, info={info}"

            Delta = unpack_complex(delta_real, shape)

            accepted = False
            alpha = 1.0
            best_X = X
            best_nrm = nrm

            while alpha >= s.min_alpha:
                Xtrial = X + alpha * Delta
                trial_nrm = problem.norms(
                    Xtrial,
                    source_scale,
                    s.compute_time_residual,
                )

                if trial_nrm["coeff_rel"] < nrm["coeff_rel"]:
                    accepted = True
                    best_X = Xtrial
                    best_nrm = trial_nrm
                    break

                alpha *= 0.5

            if not accepted:
                failure_reason = failure_reason or f"line search failed at Newton {it}"
                return X, self._make_report(
                    False,
                    source_scale,
                    nrm,
                    it,
                    gmres_total,
                    factor_total,
                    t0,
                    failure_reason,
                )

            X = best_X
            nrm = best_nrm

            if s.verbose:
                msg = (
                    f"  newton={it:02d} alpha={alpha:.3e} "
                    f"gmres_it={gmres_counter['n']:04d} "
                    f"factor_s={factor_s:.3f} "
                    f"coeff_rel={nrm['coeff_rel']:.3e}"
                )
                if nrm["time_rel"] is not None:
                    msg += f" time_rel={nrm['time_rel']:.3e}"
                if info != 0:
                    msg += f" gmres_info={info}"
                print(msg)

            if nrm["coeff_rel"] < s.newton_tol:
                return X, self._make_report(
                    True,
                    source_scale,
                    nrm,
                    it,
                    gmres_total,
                    factor_total,
                    t0,
                    "",
                )

        failure_reason = failure_reason or "maximum Newton iterations reached"
        return X, self._make_report(
            False,
            source_scale,
            nrm,
            s.max_newton,
            gmres_total,
            factor_total,
            t0,
            failure_reason,
        )

    def solve_direct(
        self,
        problem: FullIPMPumpProblem,
        x_init: np.ndarray,
    ) -> tuple[np.ndarray, list[StepReport]]:
        """Single Newton-Krylov solve at full pump scale from a warm start."""
        print("=== warm-start direct solve at lambda=1.0 ===")
        X_new, report = self.solve_one(problem, x_init, 1.0)
        status = "VALID_CONVERGED" if report.converged else "FAIL"
        print(
            f"step_status={status} "
            f"coeff_rel={report.coeff_rel:.3e} "
            f"newton={report.newton_iterations} "
            f"gmres_total={report.gmres_iterations_total} "
            f"runtime_s={report.runtime_s:.3f} "
            f"reason={report.failure_reason}"
        )
        return X_new, [report]

    def solve_continuation(
        self,
        problem: FullIPMPumpProblem,
        continuation_steps: int,
        x_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[StepReport]]:
        reports: list[StepReport] = []

        lambdas = np.linspace(1.0 / continuation_steps, 1.0, continuation_steps)

        X_prevprev: np.ndarray | None = None
        X_prev = problem.zeros() if x_init is None else np.array(x_init, dtype=np.complex128, copy=True)
        lam_prevprev: float | None = None
        lam_prev: float | None = None

        for lam_raw in lambdas:
            lam = float(lam_raw)

            X_guess = X_prev

            if (
                self.settings.continuation_predictor == "secant"
                and X_prevprev is not None
                and lam_prev is not None
                and lam_prevprev is not None
                and abs(lam_prev - lam_prevprev) > 0.0
            ):
                beta = (lam - lam_prev) / (lam_prev - lam_prevprev)
                X_guess = X_prev + beta * (X_prev - X_prevprev)

            print(f"=== continuation lambda={lam:.6f} ===")
            X_new, report = self.solve_one(problem, X_guess, lam)
            reports.append(report)

            status = "VALID_CONVERGED" if report.converged else "FAIL"
            msg = (
                f"step_status={status} "
                f"coeff_rel={report.coeff_rel:.3e} "
                f"newton={report.newton_iterations} "
                f"gmres_total={report.gmres_iterations_total} "
                f"factor_s={report.factor_runtime_s:.3f} "
                f"runtime_s={report.runtime_s:.3f} "
                f"reason={report.failure_reason}"
            )
            if report.time_rel is not None:
                msg = msg.replace(
                    f"newton={report.newton_iterations}",
                    f"time_rel={report.time_rel:.3e} newton={report.newton_iterations}",
                )
            print(msg)

            if not report.converged:
                return X_new, reports

            X_prevprev = X_prev
            lam_prevprev = lam_prev
            X_prev = X_new
            lam_prev = lam

        return X_prev, reports

    @staticmethod
    def _make_report(
        converged: bool,
        source_scale: float,
        nrm: dict[str, float | None],
        newton_iterations: int,
        gmres_iterations_total: int,
        factor_runtime_s: float,
        t0: float,
        failure_reason: str,
    ) -> StepReport:
        return StepReport(
            converged=converged,
            source_scale=source_scale,
            coeff_abs=float(nrm["coeff_abs"]),
            coeff_rel=float(nrm["coeff_rel"]),
            time_abs=None if nrm["time_abs"] is None else float(nrm["time_abs"]),
            time_rel=None if nrm["time_rel"] is None else float(nrm["time_rel"]),
            newton_iterations=newton_iterations,
            gmres_iterations_total=gmres_iterations_total,
            factor_runtime_s=factor_runtime_s,
            runtime_s=time.perf_counter() - t0,
            failure_reason=failure_reason,
        )


def gmres_call(
    A: spla.LinearOperator,
    b: np.ndarray,
    M: spla.LinearOperator | None,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    callback,
) -> tuple[np.ndarray, int]:
    try:
        return spla.gmres(
            A,
            b,
            M=M,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            callback=callback,
            callback_type="pr_norm",
        )
    except TypeError:
        return spla.gmres(
            A,
            b,
            M=M,
            tol=rtol,
            restart=restart,
            maxiter=maxiter,
            callback=callback,
        )


# =============================================================================
# Diagnostics / output
# =============================================================================

def summarize_solution(problem: FullIPMPumpProblem, X: np.ndarray) -> dict[str, float]:
    x_t = problem.grid.synthesize(X)
    psi_t = problem.branch_flux_time(X)
    i_t = problem.branch.current(psi_t)

    out = {
        "x_rms": float(np.sqrt(np.mean(x_t * x_t))),
        "x_max_abs": float(np.max(np.abs(x_t))),
        "branch_psi_rms": float(np.sqrt(np.mean(psi_t * psi_t))),
        "branch_psi_max_abs": float(np.max(np.abs(psi_t))),
        "branch_i_rms": float(np.sqrt(np.mean(i_t * i_t))),
        "branch_i_max_abs": float(np.max(np.abs(i_t))),
    }

    for h in range(problem.H):
        out[f"X_h{h + 1}_norm"] = float(np.linalg.norm(X[h]))

    return out


def write_results(
    outdir: str | Path,
    X: np.ndarray,
    reports: list[StepReport],
    solution_summary: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    pump_modes = np.asarray(
        metadata.get("pump_modes", list(range(1, X.shape[0] + 1))),
        dtype=np.int64,
    )
    np.savez(
        d / "pump_solution.npz",
        X_real=X.real,
        X_imag=X.imag,
        harmonics=pump_modes,
        pump_modes=pump_modes,
    )

    report_json = {
        "metadata": metadata,
        "solution_summary": solution_summary,
        "reports": [asdict(r) for r in reports],
        "final_status": "VALID_CONVERGED"
        if reports and reports[-1].converged and abs(reports[-1].source_scale - 1.0) < 1e-12
        else "FAIL",
    }

    with open(d / "pump_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    print(f"wrote_solution={d / 'pump_solution.npz'}")
    print(f"wrote_report={d / 'pump_report.json'}")


def print_loaded_summary(ipm: LoadedIPM, pump_port: int, pump_idx: int, pump_current_a: float) -> None:
    print("=== loaded full IPM matrices ===")
    print(f"nodes={ipm.C.shape[0]}")
    print(f"C_nnz={ipm.C.nnz}")
    print(f"G_nnz={ipm.G.nnz}")
    print(f"K_nnz={ipm.K.nnz}")
    print(f"Bphi_shape={list(ipm.Bphi.shape)}")
    print(f"Bphi_nnz={ipm.Bphi.nnz}")
    print(f"Ic_min={float(ipm.Ic.min()):.12e}")
    print(f"Ic_max={float(ipm.Ic.max()):.12e}")
    print(f"Ic_median={float(np.median(ipm.Ic)):.12e}")
    print(f"phi0_reduced={ipm.phi0:.12e}")
    print(f"ports={ipm.port_to_index}")
    print(f"pump_port={pump_port}")
    print(f"pump_node_index={pump_idx}")
    print(f"pump_current_a={pump_current_a:.12e}")
    print(f"pump_current_over_Ic_median={pump_current_a / float(np.median(ipm.Ic)):.6f}")


def print_final_report(
    reports: list[StepReport],
    total_runtime_s: float,
    solution_summary: dict[str, float],
) -> None:
    final = reports[-1] if reports else None
    converged = bool(final and final.converged and abs(final.source_scale - 1.0) < 1e-12)

    print("\n=== final report ===")
    print(f"status={'VALID_CONVERGED' if converged else 'FAIL'}")
    print(f"continuation_steps_completed={len(reports)}")
    print(f"total_runtime_s={total_runtime_s:.6f}")

    if final is not None:
        print(f"final_lambda={final.source_scale:.12e}")
        print(f"final_coeff_abs={final.coeff_abs:.12e}")
        print(f"final_coeff_rel={final.coeff_rel:.12e}")
        if final.time_abs is not None:
            print(f"final_time_abs={final.time_abs:.12e}")
        if final.time_rel is not None:
            print(f"final_time_rel={final.time_rel:.12e}")
        print(f"final_newton_iterations_last_step={final.newton_iterations}")
        print(f"final_gmres_iterations_last_step={final.gmres_iterations_total}")
        print(f"final_factor_runtime_last_step_s={final.factor_runtime_s:.6f}")
        print(f"failure_reason={final.failure_reason}")

    print("\n=== solution summary ===")
    for k in sorted(solution_summary):
        print(f"{k}={solution_summary[k]:.12e}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")

    p.add_argument("--ipm-dir", default=os.path.join("outputs", "ipm_python_design"))
    p.add_argument("--dc-solution", default=None, help="Optional dc_solution.npz or folder containing it. Enables pump solve around a static DC operating point.")
    p.add_argument("--outdir", default=os.path.join("outputs", "exp08_full_ipm_pump"))

    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--pump-freq-ghz", type=float, default=7.9)

    p.add_argument("--pump-current-a", type=float, default=None)
    p.add_argument("--pump-current-ratio-ic", type=float, default=1.5)

    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--nt", type=int, default=32)

    p.add_argument(
        "--pump-mode-policy",
        choices=list(POLICIES),
        default="dense_real",
        help="Pump-mode basis policy. dense_real preserves legacy 1..H behavior.",
    )
    p.add_argument(
        "--pump-mode-count",
        type=int,
        default=None,
        help="K for positive_odd_jc -> modes [1,3,...,2K-1]. Defaults to --harmonics.",
    )
    p.add_argument(
        "--pump-modes",
        type=str,
        default=None,
        help="Explicit comma-separated positive modes, e.g. '1,3,5,7'.",
    )
    p.add_argument(
        "--promote-from-pump-dir",
        type=str,
        default=None,
        help="Warm-start a richer basis from an existing lower-basis pump solution dir.",
    )
    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--continuation-predictor", choices=["none", "secant"], default="none")

    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--max-newton", type=int, default=16)

    p.add_argument("--gmres-rtol", type=float, default=1e-7)
    p.add_argument("--gmres-atol", type=float, default=0.0)
    p.add_argument("--gmres-restart", type=int, default=60)
    p.add_argument("--gmres-maxiter", type=int, default=80)

    p.add_argument(
        "--preconditioner",
        choices=["mean_tangent", "linear", "none", "spectral_coupled", "real_coupled"],
        default="mean_tangent",
    )

    p.add_argument("--real-capacitance", action="store_true",
                   help="Drop the imaginary (lossy) part of C in the pump solve (loss-convention study).")
    p.add_argument("--jvp-mode", choices=["aft", "spectral"], default="aft")
    p.add_argument("--min-alpha", type=float, default=1.0 / 1024.0)
    p.add_argument("--skip-time-residual", action="store_true")
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ipm = load_ipm(args.ipm_dir)

    if args.pump_port not in ipm.port_to_index:
        raise ValueError(f"pump port {args.pump_port} not found in {ipm.port_to_index}")

    pump_idx = ipm.port_to_index[args.pump_port]

    if args.pump_current_a is None:
        pump_current_a = args.pump_current_ratio_ic * float(np.median(ipm.Ic))
    else:
        pump_current_a = args.pump_current_a

    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9

    print("=== experiment 08: full IPM harmonic pump solve ===")
    print_loaded_summary(ipm, args.pump_port, pump_idx, pump_current_a)

    basis = resolve_pump_basis(
        policy=args.pump_mode_policy,
        omega_p=omega_p,
        harmonics=args.harmonics,
        mode_count=args.pump_mode_count,
        explicit_modes=args.pump_modes,
        design_meta=ipm.summary,
    )
    print(f"pump_mode_policy={basis.policy}")
    print(f"pump_modes={basis.modes}")
    print(f"pump_basis={basis.basis}")

    grid = HarmonicGrid(
        modes=basis.k,
        nt=args.nt,
        omega=omega_p,
    )

    branch = JosephsonBranchArray(
        Ic=ipm.Ic,
        phi0=ipm.phi0,
    )

    dc_x, dc_branch_flux = load_dc_solution(args.dc_solution, ipm)
    if dc_branch_flux is not None:
        print(f"dc_solution={args.dc_solution}")
        print(f"dc_branch_flux_max_abs={float(np.max(np.abs(dc_branch_flux))):.12e}")
        print(f"dc_branch_flux_over_phi0_max_abs={float(np.max(np.abs(dc_branch_flux / ipm.phi0))):.12e}")

    problem = FullIPMPumpProblem(
        C=ipm.C,
        G=ipm.G,
        K=ipm.K,
        Bphi=ipm.Bphi,
        branch=branch,
        grid=grid,
        pump_node_index=pump_idx,
        pump_current_a=pump_current_a,
        dc_branch_flux=dc_branch_flux,
        source_mode=basis.source_mode,
        use_real_capacitance=args.real_capacitance,
    )

    settings = NewtonKrylovSettings(
        newton_tol=args.newton_tol,
        max_newton=args.max_newton,
        gmres_rtol=args.gmres_rtol,
        gmres_atol=args.gmres_atol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
        min_alpha=args.min_alpha,
        preconditioner=args.preconditioner,
        compute_time_residual=not args.skip_time_residual,
        verbose=not args.quiet,
        continuation_predictor=args.continuation_predictor,
        jvp_mode=args.jvp_mode,
    )

    print("\n=== solve settings ===")
    print(f"pump_freq_ghz={args.pump_freq_ghz}")
    print(f"omega_p={omega_p:.12e}")
    print(f"harmonics={args.harmonics}")
    print(f"nt={args.nt}")
    print(f"real_unknowns={2 * args.harmonics * ipm.C.shape[0]}")
    print(f"continuation_steps={args.continuation_steps}")
    print(f"continuation_predictor={args.continuation_predictor}")
    print(f"preconditioner={args.preconditioner}")
    print(f"jvp_mode={args.jvp_mode}")
    print(f"compute_time_residual={not args.skip_time_residual}")

    solver = HarmonicNewtonKrylovSolver(settings)

    x_init = None
    warm_started = False
    if args.promote_from_pump_dir:
        X_src, src_basis = load_pump_basis_from_solution(
            args.promote_from_pump_dir, fallback_omega_p=omega_p
        )
        if X_src.shape[1] != problem.n:
            raise ValueError(
                f"warm-start node count {X_src.shape[1]} != design node count {problem.n}"
            )
        x_init = promote_solution_to_basis(X_src, src_basis, basis)
        warm_started = True
        shared = sorted(set(src_basis.modes) & set(basis.modes))
        print(f"warm_start_from={args.promote_from_pump_dir}")
        print(f"warm_start_src_modes={src_basis.modes}")
        print(f"warm_start_shared_modes={shared}")

    t0 = time.perf_counter()
    if warm_started:
        X, reports = solver.solve_direct(problem, x_init)
    else:
        X, reports = solver.solve_continuation(
            problem,
            continuation_steps=args.continuation_steps,
        )
    total_runtime_s = time.perf_counter() - t0

    solution_summary = summarize_solution(problem, X)
    print_final_report(reports, total_runtime_s, solution_summary)

    metadata = {
        "ipm_dir": str(args.ipm_dir),
        "pump_port": args.pump_port,
        "pump_node_index": pump_idx,
        "pump_current_a": pump_current_a,
        "pump_current_ratio_ic_median": pump_current_a / float(np.median(ipm.Ic)),
        "pump_freq_ghz": args.pump_freq_ghz,
        "omega_p": omega_p,
        "harmonics": basis.n_modes,
        "nt": args.nt,
        **basis.to_metadata(),
        "warm_started": warm_started,
        "promote_from_pump_dir": args.promote_from_pump_dir,
        "continuation_steps": args.continuation_steps,
        "continuation_predictor": args.continuation_predictor,
        "real_unknowns": 2 * basis.n_modes * ipm.C.shape[0],
        "preconditioner": args.preconditioner,
        "jvp_mode": args.jvp_mode,
        "compute_time_residual": not args.skip_time_residual,
        "nodes": int(ipm.C.shape[0]),
        "jj_branches": int(ipm.Bphi.shape[1]),
    }

    metadata["dc_solution"] = args.dc_solution
    metadata["dc_enabled"] = args.dc_solution is not None

    write_results(args.outdir, X, reports, solution_summary, metadata)


if __name__ == "__main__":
    main()
