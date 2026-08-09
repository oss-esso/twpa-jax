"""Full-node nonlinear multitone harmonic-balance problem."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.core.linear import default_loss_model_for, dynamic_block
from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.grid import TorusGrid
from twpa_solver.multitone.source import AffineSourcePath
from twpa_solver.pump.problem import (
    SpectralTangentState,
    TangentState,
    pack_complex,
)


@dataclass(frozen=True)
class _LinearBlockView:
    """Expose ``_linear_blocks`` under the attribute name Schur code expects."""

    schur: list[sp.csc_matrix]


@dataclass
class FullMultiToneProblem:
    """Duck-typed Newton problem for a finite positive-frequency tone basis."""

    circuit: CircuitMatrices
    basis: MultiToneBasis
    source_path: AffineSourcePath
    loss_model: str | object | None = None
    input_power_dbm: float | None = None
    dc_branch_flux: np.ndarray | None = None
    environment: object | None = None
    preconditioner: str | None = None
    # Survives ``dataclasses.replace``: the compression sweep rebuilds the
    # problem once per signal-power point with only ``source_path`` changed,
    # and everything expensive here (dynamic blocks, the fast preconditioner's
    # scatter map and symbolic factorization) depends on the circuit and basis
    # only. Sharing this dict keeps that work at once per sweep.
    cache: dict[object, object] | None = None

    def __post_init__(self) -> None:
        from twpa_solver.multitone.preconditioners import (
            resolve_multitone_preconditioner,
        )

        if self.loss_model is None:
            self.loss_model = default_loss_model_for(self.circuit)
        self.preconditioner = resolve_multitone_preconditioner(
            self.preconditioner
        )
        self.C = self.circuit.C.tocsr()
        self.G = self.circuit.G.tocsr()
        self.K = self.circuit.K.tocsr()
        self.Bphi = self.circuit.Bphi.tocsr()
        self.BphiT = self.Bphi.T.tocsr()
        self.branch = make_branch_law(self.circuit)
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
        if self.cache is None:
            self.cache = {}
        blocks_key = ("linear_blocks", id(self.loss_model), self.input_power_dbm)
        cached_blocks = self.cache.get(blocks_key)
        if cached_blocks is None:
            self._linear_blocks = []
            for omega in self.basis.omegas:
                selected_loss = self.loss_model
                if hasattr(selected_loss, "evaluate"):
                    selected_loss = selected_loss.evaluate(
                        float(omega / (2.0 * np.pi)),
                        "multitone",
                        self.input_power_dbm,
                    )
                block = dynamic_block(self.circuit, omega, loss_model=str(selected_loss))
                if self.environment is not None:
                    node = next(iter(self.circuit.port_to_index.values()))
                    correction = self.environment.admittance(float(omega))
                    block = block + sp.csr_matrix(
                        (np.asarray([correction]), ([node], [node])), shape=block.shape
                    )
                self._linear_blocks.append(block.tocsc())
            self.cache[blocks_key] = self._linear_blocks
        else:
            self._linear_blocks = cached_blocks
        # Attribute aliases so the shared fast-coupled preconditioner, written
        # against the Schur-reduced problem, works unmodified on the full node
        # set (where "retained" is every node and the Schur complement is just
        # the dynamic block).
        self.Bphi_r = self.Bphi
        self.BphiT_r = self.BphiT
        self.part = _LinearBlockView(self._linear_blocks)

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
        """Exact real-coupled preconditioner with assembly + symbolic reuse.

        Builds the scatter map and symbolic factorization once per basis and
        caches them on the shared ``cache``; each Newton step then only rebuilds
        ``M.data`` and runs a numeric-only factor. Produces the identical matrix
        to :meth:`assemble_real_coupled_preconditioner`.
        """
        if self.preconditioner == "floquet_sector":
            from twpa_solver.multitone.preconditioners import (
                FloquetSectorPreconditioner,
            )

            preconditioner = FloquetSectorPreconditioner(self)
            preconditioner.refactor(tangent)
            return preconditioner
        from twpa_solver.pump.backends.fast_coupled import (
            FastCoupledPreconditioner,
        )

        fast = self.cache.get("fast_coupled")
        if fast is None:
            fast = FastCoupledPreconditioner(self)
            self.cache["fast_coupled"] = fast
        fast.refactor(tangent)
        return fast

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
        linear = np.asarray([
            block @ row for block, row in zip(self._linear_blocks, X)
        ])
        residual = self.grid.synthesize(linear)
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

    def real_coupled_matrix(self, spectral: SpectralTangentState) -> sp.spmatrix:
        """Assemble the exact real-packed coupled Jacobian.

        Split out of :meth:`assemble_real_coupled_preconditioner` so the matrix
        itself is testable: the ``real_coupled_fast`` backend reproduces this
        assembly from a cached scatter map, and that equivalence is the gate on
        the fast path.
        """
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
        return sp.bmat([[groups[0], groups[1]], [groups[2], groups[3]]], format="csc")

    def assemble_real_coupled_preconditioner(self, spectral: SpectralTangentState) -> spla.SuperLU:
        return spla.splu(self.real_coupled_matrix(spectral))
