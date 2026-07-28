"""Full-node nonlinear multitone harmonic-balance problem."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.linear import dynamic_block
from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.grid import TorusGrid
from twpa_solver.multitone.source import AffineSourcePath
from twpa_solver.pump.problem import (
    JosephsonBranchArray,
    SpectralTangentState,
    TangentState,
    pack_complex,
)


@dataclass
class FullMultiToneProblem:
    """Duck-typed Newton problem for a finite positive-frequency tone basis."""

    circuit: CircuitMatrices
    basis: MultiToneBasis
    source_path: AffineSourcePath
    loss_model: str | object = "current_complex_c"
    input_power_dbm: float | None = None
    dc_branch_flux: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.C = self.circuit.C.tocsr()
        self.G = self.circuit.G.tocsr()
        self.K = self.circuit.K.tocsr()
        self.Bphi = self.circuit.Bphi.tocsr()
        self.BphiT = self.Bphi.T.tocsr()
        self.branch = JosephsonBranchArray(self.circuit.Ic, self.circuit.phi0)
        self.grid = TorusGrid(self.basis)
        self.n = self.C.shape[0]
        self.H = self.basis.n_tones
        self.nb = self.Bphi.shape[1]
        if self.source_path.source_start.shape != (self.H, self.n):
            raise ValueError("source path shape must be (basis.n_tones, circuit.node_count)")
        if self.dc_branch_flux is None:
            self.dc_branch_flux = np.zeros(self.nb, dtype=float)
        else:
            self.dc_branch_flux = np.asarray(self.dc_branch_flux, dtype=float).reshape(-1)
            if self.dc_branch_flux.size != self.nb:
                raise ValueError("dc_branch_flux length does not match branch count")
        self._linear_blocks = []
        for omega in self.basis.omegas:
            selected_loss = self.loss_model
            if hasattr(selected_loss, "evaluate"):
                selected_loss = selected_loss.evaluate(
                    float(omega / (2.0 * np.pi)),
                    "multitone",
                    self.input_power_dbm,
                )
            self._linear_blocks.append(
                dynamic_block(self.circuit, omega, loss_model=str(selected_loss)).tocsc()
            )

    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.n), dtype=np.complex128)

    def source_coeffs(self, tau: float) -> np.ndarray:
        return self.source_path.source(tau)

    def source_delta_coeffs(self) -> np.ndarray:
        return self.source_path.derivative()

    @property
    def mode_keys(self):
        return list(self.basis.tones)

    def assemble_real_coupled_fast(self, tangent: TangentState):
        return self.assemble_real_coupled_preconditioner(
            self.spectral_tangent_state(tangent)
        )

    def branch_flux_time(self, X: np.ndarray) -> np.ndarray:
        x_t = self.grid.synthesize(X)
        return (self.BphiT @ x_t.T).T

    def nonlinear_current_time(self, X: np.ndarray) -> np.ndarray:
        psi = self.branch_flux_time(X) + self.dc_branch_flux[None, :]
        current = self.branch.current(psi) - self.branch.current(self.dc_branch_flux[None, :])
        return (self.Bphi @ current.T).T

    def nonlinear_current_coeffs(self, X: np.ndarray) -> np.ndarray:
        return self.grid.project(self.nonlinear_current_time(X))

    def residual_coeffs(self, X: np.ndarray, tau: float) -> np.ndarray:
        nonlinear = self.nonlinear_current_coeffs(X)
        source = self.source_coeffs(tau)
        return np.asarray(
            [block @ row + nonlinear[i] - source[i] for i, (block, row) in enumerate(zip(self._linear_blocks, X))]
        )

    def tangent_state(self, X: np.ndarray) -> TangentState:
        psi = self.branch_flux_time(X) + self.dc_branch_flux[None, :]
        gamma_t = self.branch.gamma(psi)
        return TangentState(gamma_t=gamma_t, gamma_mean=np.mean(gamma_t, axis=0))

    def jvp_coeffs_with_tangent(self, V: np.ndarray, tangent: TangentState) -> np.ndarray:
        linear = np.asarray(
            [block @ row for block, row in zip(self._linear_blocks, V)]
        )
        dpsi = self.branch_flux_time(V)
        nonlinear = (self.Bphi @ (tangent.gamma_t * dpsi).T).T
        return linear + self.grid.project(nonlinear)

    def jvp_coeffs(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        return self.jvp_coeffs_with_tangent(V, self.tangent_state(X))

    def spectral_tangent_state(self, tangent: TangentState) -> SpectralTangentState:
        needed = sorted(
            {k - q for k in self.basis.tones for q in self.basis.tones}
            | {k + q for k in self.basis.tones for q in self.basis.tones}
        )
        khat: dict[ToneIndex, sp.csr_matrix] = {}
        for offset in needed:
            phase = self.grid.phase_rows([offset])[0]
            gamma_hat = np.sum(tangent.gamma_t * phase[:, None], axis=0)
            khat[offset] = (
                self.Bphi @ sp.diags(gamma_hat, 0) @ self.BphiT
            ).astype(np.complex128).tocsr()
        return SpectralTangentState(khat=khat)

    def jvp_coeffs_with_spectral_tangent(
        self, V: np.ndarray, spectral: SpectralTangentState
    ) -> np.ndarray:
        """Apply the tuple-lattice convolution represented by ``spectral``."""
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        output = np.empty_like(V)
        for i, k in enumerate(self.basis.tones):
            value = self._linear_blocks[i] @ V[i]
            for j, q in enumerate(self.basis.tones):
                value = value + spectral.khat.get(k - q, zero) @ V[j]
                value = value + spectral.khat.get(k + q, zero) @ np.conj(V[j])
            output[i] = value
        return output

    def norms(
        self, X: np.ndarray, tau: float, compute_time_residual: bool
    ) -> dict[str, float | None]:
        residual = pack_complex(self.residual_coeffs(X, tau))
        coeff_abs = float(np.linalg.norm(residual) / math.sqrt(residual.size))
        source = pack_complex(self.source_coeffs(tau))
        source_abs = float(np.linalg.norm(source) / max(math.sqrt(source.size), 1.0))
        time_abs = time_rel = None
        if compute_time_residual:
            time = self.time_residual(X, tau)
            time_abs = float(np.linalg.norm(time) / math.sqrt(time.size))
            source_time = self.source_time(tau)
            source_time_abs = float(np.linalg.norm(source_time) / math.sqrt(source_time.size))
            time_rel = time_abs / max(source_time_abs, 1e-30)
        return {"coeff_abs": coeff_abs, "coeff_rel": coeff_abs / max(source_abs, 1e-30), "time_abs": time_abs, "time_rel": time_rel}

    def time_residual(self, X: np.ndarray, tau: float) -> np.ndarray:
        x_t = self.grid.synthesize(X)
        dx = self._synthesize_derivative(X, 1)
        ddx = self._synthesize_derivative(X, 2)
        residual = (self.C @ ddx.T).T + (self.G @ dx.T).T + (self.K @ x_t.T).T
        return np.asarray(residual + self.nonlinear_current_time(X) - self.source_time(tau), dtype=float)

    def source_time(self, tau: float) -> np.ndarray:
        return self._source_to_time(self.source_coeffs(tau))

    def _source_to_time(self, source: np.ndarray) -> np.ndarray:
        return self.grid.synthesize(source)

    def _synthesize_derivative(self, X: np.ndarray, order: int) -> np.ndarray:
        values = self.grid.synthesize(X)
        derivative = np.zeros_like(values)
        for coefficient, tone in zip(X, self.basis.tones):
            multiplier = (1j * tone.omega(self.basis.omega_p, self.basis.delta)) ** order
            derivative += self.grid.synthesize(np.where(np.arange(self.H)[:, None] == self.basis.index_of(tone), coefficient * multiplier, 0.0))
        return derivative

    def build_preconditioner_factors(
        self, X: np.ndarray, mode: str, tangent: TangentState | None = None
    ) -> list[spla.SuperLU] | None:
        if mode == "none":
            return None
        if tangent is None:
            tangent = self.tangent_state(X)
        if mode == "linear":
            return [spla.splu(block) for block in self._linear_blocks]
        if mode == "mean_tangent":
            mean = self.Bphi @ sp.diags(tangent.gamma_mean, 0) @ self.BphiT
            return [spla.splu((block + mean).tocsc()) for block in self._linear_blocks]
        raise ValueError(f"unknown preconditioner mode {mode!r}")

    def assemble_coupled_preconditioner(self, spectral: SpectralTangentState) -> spla.SuperLU:
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        rows = []
        for i, k in enumerate(self.basis.tones):
            row = []
            for j, q in enumerate(self.basis.tones):
                block = spectral.khat.get(k - q, zero)
                if i == j:
                    block = block + self._linear_blocks[i]
                row.append(block)
            rows.append(row)
        return spla.splu(sp.bmat(rows, format="csc"))

    def assemble_real_coupled_preconditioner(self, spectral: SpectralTangentState) -> spla.SuperLU:
        zero = sp.csr_matrix((self.n, self.n), dtype=np.complex128)
        blocks = [[], [], [], []]
        for i, k in enumerate(self.basis.tones):
            for q_index, q in enumerate(self.basis.tones):
                linear = spectral.khat.get(k - q, zero)
                if i == q_index:
                    linear = linear + self._linear_blocks[i]
                conjugate = spectral.khat.get(k + q, zero)
                lr, li, pr, pi = linear.real, linear.imag, conjugate.real, conjugate.imag
                blocks[0].append((lr + pr).tocsr())
                blocks[1].append((pi - li).tocsr())
                blocks[2].append((li + pi).tocsr())
                blocks[3].append((lr - pr).tocsr())
        ntones = self.H
        groups = [sp.bmat([blocks[i][r * ntones:(r + 1) * ntones] for r in range(ntones)]) for i in range(4)]
        return spla.splu(sp.bmat([[groups[0], groups[1]], [groups[2], groups[3]]], format="csc"))
