"""Fourier pseudo-spectral pump-only harmonic-balance residual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from twpa_solver_old.model.topology import CircuitModel
from twpa_solver_old.residuals.scaling import safe_scale


@dataclass(frozen=True)
class PumpAFTConfig:
    pump_frequency_hz: float
    harmonics: int = 2
    time_samples: int | None = None
    source_current_peak_a: float = 0.0
    source_phase_rad: float = 0.0
    residual_scale_a: float = 1e-9

    def __post_init__(self) -> None:
        if self.pump_frequency_hz <= 0.0:
            raise ValueError("pump_frequency_hz must be positive")
        if self.harmonics <= 0:
            raise ValueError("harmonics must be positive")
        if self.time_samples is not None and self.time_samples < 2 * self.harmonics + 1:
            raise ValueError("time_samples must be at least 2 * harmonics + 1")
        if self.residual_scale_a <= 0.0:
            raise ValueError("residual_scale_a must be positive")


class PumpAFTResidual:
    """Real cos/sin coefficient AFT residual for pump-only HB."""

    def __init__(self, model: CircuitModel, config: PumpAFTConfig) -> None:
        self.model = model
        self.config = config
        self.num_nodes = model.num_nodes
        self.harmonics = config.harmonics
        self.time_samples = config.time_samples or max(4 * config.harmonics + 2, 16)
        self.theta = np.linspace(0.0, 2.0 * np.pi, self.time_samples, endpoint=False)
        self.omega = 2.0 * np.pi * config.pump_frequency_hz
        self._cos = np.stack([np.cos((k + 1) * self.theta) for k in range(self.harmonics)])
        self._sin = np.stack([np.sin((k + 1) * self.theta) for k in range(self.harmonics)])
        self._c_sparse = sparse.csr_matrix(model.capacitance_f)
        self._g_sparse = sparse.csr_matrix(model.conductance_s)
        self._k_sparse = sparse.csr_matrix(model.linear_stiffness_h_inv)
        self._josephson_incidence_sparse = sparse.csr_matrix(model.josephson_incidence)

    @property
    def size(self) -> int:
        return 2 * self.harmonics * self.num_nodes

    def initial_guess(self, amplitude_wb: float = 0.0) -> np.ndarray:
        x = np.zeros(self.size, dtype=float)
        if amplitude_wb and self.model.pump_nodes:
            coeff = self.coefficients_view(x)
            for node in self.model.pump_nodes:
                coeff[0, 0, node] = amplitude_wb
        return x

    def coefficients_view(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.size != self.size:
            raise ValueError(f"expected vector length {self.size}, got {arr.size}")
        return arr.reshape(self.harmonics, 2, self.num_nodes)

    def coefficients_to_time(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coeff = self.coefficients_view(x)
        phi = np.einsum("kn,kt->tn", coeff[:, 0, :], self._cos)
        phi += np.einsum("kn,kt->tn", coeff[:, 1, :], self._sin)
        phidot = np.zeros_like(phi)
        phiddot = np.zeros_like(phi)
        for h in range(1, self.harmonics + 1):
            omega_h = h * self.omega
            cos_h = self._cos[h - 1]
            sin_h = self._sin[h - 1]
            a = coeff[h - 1, 0, :]
            b = coeff[h - 1, 1, :]
            phidot += np.outer(-omega_h * sin_h, a) + np.outer(omega_h * cos_h, b)
            phiddot += np.outer(-(omega_h**2) * cos_h, a)
            phiddot += np.outer(-(omega_h**2) * sin_h, b)
        return phi, phidot, phiddot

    def source_time(self) -> np.ndarray:
        source = np.zeros((self.time_samples, self.num_nodes), dtype=float)
        if self.config.source_current_peak_a == 0.0:
            return source
        waveform = self.config.source_current_peak_a * np.cos(
            self.theta + self.config.source_phase_rad
        )
        nodes = self.model.pump_nodes or (self.model.ports[0].node,)
        per_node = waveform / len(nodes)
        for node in nodes:
            source[:, node] += per_node
        return source

    def residual_time(self, x: np.ndarray) -> np.ndarray:
        phi, phidot, phiddot = self.coefficients_to_time(x)
        model = self.model
        residual = phiddot @ model.capacitance_f.T
        residual += phidot @ model.conductance_s.T
        residual += phi @ model.linear_stiffness_h_inv.T
        residual += model.nonlinear_current(phi)
        residual -= self.source_time()
        return residual

    def project_time_to_coefficients(self, values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        scale = 2.0 / self.time_samples
        cos_coeff = scale * np.einsum("tn,kt->kn", vals, self._cos)
        sin_coeff = scale * np.einsum("tn,kt->kn", vals, self._sin)
        return np.stack([cos_coeff, sin_coeff], axis=1).reshape(-1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return safe_scale(
            self.project_time_to_coefficients(self.residual_time(x)),
            self.config.residual_scale_a,
        )

    def jacobian_sparse(self, x: np.ndarray) -> sparse.csr_matrix:
        """Return the analytic sparse Jacobian of the scaled AFT residual.

        The unknowns are ordered as ``(harmonic, cos/sin, node)``.  The
        derivative is assembled by differentiating the time-domain residual and
        then applying the same Fourier projection used by ``__call__``.
        """
        phi, _, _ = self.coefficients_to_time(x)
        nonlinear_stiffness = self._nonlinear_stiffness_sparse_by_time(phi)
        scale = 2.0 / (self.time_samples * self.config.residual_scale_a)
        blocks: list[list[sparse.spmatrix]] = []
        for out_h in range(self.harmonics):
            out_cos = self._cos[out_h]
            out_sin = self._sin[out_h]
            row_cos: list[sparse.spmatrix] = []
            row_sin: list[sparse.spmatrix] = []
            for in_h in range(self.harmonics):
                omega_h = (in_h + 1) * self.omega
                in_cos = self._cos[in_h]
                in_sin = self._sin[in_h]
                cos_cos = self._projected_derivative_block(
                    nonlinear_stiffness,
                    out_cos,
                    in_cos,
                    -omega_h * in_sin,
                    -(omega_h**2) * in_cos,
                )
                cos_sin = self._projected_derivative_block(
                    nonlinear_stiffness,
                    out_cos,
                    in_sin,
                    omega_h * in_cos,
                    -(omega_h**2) * in_sin,
                )
                sin_cos = self._projected_derivative_block(
                    nonlinear_stiffness,
                    out_sin,
                    in_cos,
                    -omega_h * in_sin,
                    -(omega_h**2) * in_cos,
                )
                sin_sin = self._projected_derivative_block(
                    nonlinear_stiffness,
                    out_sin,
                    in_sin,
                    omega_h * in_cos,
                    -(omega_h**2) * in_sin,
                )
                row_cos.extend([scale * cos_cos, scale * cos_sin])
                row_sin.extend([scale * sin_cos, scale * sin_sin])
            blocks.extend([row_cos, row_sin])
        return sparse.bmat(blocks, format="csr")

    def jvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Apply the analytic sparse Jacobian to ``v``."""
        return self.jacobian_sparse(x) @ np.asarray(v, dtype=float)

    def _projected_derivative_block(
        self,
        nonlinear_stiffness: list[sparse.csr_matrix],
        output_basis: np.ndarray,
        phi_basis: np.ndarray,
        phidot_basis: np.ndarray,
        phiddot_basis: np.ndarray,
    ) -> sparse.csr_matrix:
        block = sparse.csr_matrix((self.num_nodes, self.num_nodes), dtype=float)
        for t in range(self.time_samples):
            weight_phi = output_basis[t] * phi_basis[t]
            if weight_phi:
                block = block + weight_phi * (self._k_sparse + nonlinear_stiffness[t])
            weight_ddot = output_basis[t] * phiddot_basis[t]
            if weight_ddot:
                block = block + weight_ddot * self._c_sparse
            weight_dot = output_basis[t] * phidot_basis[t]
            if weight_dot:
                block = block + weight_dot * self._g_sparse
        return block.tocsr()

    def _nonlinear_stiffness_sparse_by_time(self, phi_time: np.ndarray) -> list[sparse.csr_matrix]:
        d = self._josephson_incidence_sparse
        if self.model.josephson is None or d.shape[1] == 0:
            zero = sparse.csr_matrix((self.num_nodes, self.num_nodes), dtype=float)
            return [zero for _ in range(self.time_samples)]
        branch_flux = np.asarray(phi_time, dtype=float) @ self.model.josephson_incidence
        deriv = self.model.josephson.derivative(branch_flux)
        rows: list[sparse.csr_matrix] = []
        dt = d.transpose().tocsr()
        for t in range(self.time_samples):
            rows.append((d @ sparse.diags(deriv[t], format="csr") @ dt).tocsr())
        return rows

    def diagnostic_by_harmonic(self, x: np.ndarray) -> list[dict[str, float | int]]:
        coeff = self.project_time_to_coefficients(self.residual_time(x)).reshape(
            self.harmonics, 2, self.num_nodes
        )
        rows: list[dict[str, float | int]] = []
        for h in range(self.harmonics):
            block = coeff[h]
            rows.append(
                {
                    "harmonic": h + 1,
                    "residual_l2_a": float(np.linalg.norm(block)),
                    "residual_inf_a": float(np.max(np.abs(block))),
                }
            )
        return rows
