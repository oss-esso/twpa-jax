"""JAX-native AFT/HB residual for fixed-size TWPA models."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from twpa_solver.model.topology import CircuitModel
from twpa_solver.residuals.aft_hb import PumpAFTConfig


@dataclass(frozen=True)
class JaxPumpAFTResidual:
    """JIT/JVP-compatible real cos/sin pump HB residual."""

    model: CircuitModel
    config: PumpAFTConfig

    def __post_init__(self) -> None:
        num_nodes = self.model.num_nodes
        harmonics = self.config.harmonics
        time_samples = self.config.time_samples or max(4 * harmonics + 2, 16)
        theta = jnp.linspace(0.0, 2.0 * jnp.pi, time_samples, endpoint=False)
        cos_basis = jnp.stack([jnp.cos((k + 1) * theta) for k in range(harmonics)])
        sin_basis = jnp.stack([jnp.sin((k + 1) * theta) for k in range(harmonics)])
        source = jnp.zeros((time_samples, num_nodes), dtype=jnp.float64)
        nodes = self.model.pump_nodes or (self.model.ports[0].node,)
        waveform = self.config.source_current_peak_a * jnp.cos(
            theta + self.config.source_phase_rad
        )
        for node in nodes:
            source = source.at[:, node].add(waveform / len(nodes))
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "harmonics", harmonics)
        object.__setattr__(self, "time_samples", time_samples)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "omega", 2.0 * jnp.pi * self.config.pump_frequency_hz)
        object.__setattr__(self, "cos_basis", cos_basis)
        object.__setattr__(self, "sin_basis", sin_basis)
        object.__setattr__(
            self,
            "capacitance_f",
            jnp.asarray(self.model.capacitance_f, dtype=jnp.float64),
        )
        object.__setattr__(
            self,
            "conductance_s",
            jnp.asarray(self.model.conductance_s, dtype=jnp.float64),
        )
        object.__setattr__(
            self,
            "linear_stiffness_h_inv",
            jnp.asarray(self.model.linear_stiffness_h_inv, dtype=jnp.float64),
        )
        object.__setattr__(
            self,
            "josephson_incidence",
            jnp.asarray(self.model.josephson_incidence, dtype=jnp.float64),
        )
        if self.model.josephson is None:
            critical = jnp.zeros((0,), dtype=jnp.float64)
            reduced_phi0 = 1.0
        else:
            critical = jnp.asarray(self.model.josephson.critical_current_a, dtype=jnp.float64)
            reduced_phi0 = self.model.josephson.reduced_flux_quantum_wb
        object.__setattr__(self, "critical_current_a", critical)
        object.__setattr__(self, "reduced_phi0", float(reduced_phi0))
        object.__setattr__(self, "source", source)

    @property
    def size(self) -> int:
        return 2 * self.harmonics * self.num_nodes

    def initial_guess(self) -> jnp.ndarray:
        return jnp.zeros((self.size,), dtype=jnp.float64)

    def coefficients_to_time(
        self,
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        coeff = jnp.reshape(x, (self.harmonics, 2, self.num_nodes))
        phi = jnp.einsum("kn,kt->tn", coeff[:, 0, :], self.cos_basis)
        phi = phi + jnp.einsum("kn,kt->tn", coeff[:, 1, :], self.sin_basis)
        phidot = jnp.zeros_like(phi)
        phiddot = jnp.zeros_like(phi)
        for h in range(1, self.harmonics + 1):
            omega_h = h * self.omega
            cos_h = self.cos_basis[h - 1]
            sin_h = self.sin_basis[h - 1]
            a = coeff[h - 1, 0, :]
            b = coeff[h - 1, 1, :]
            phidot = phidot + jnp.outer(-omega_h * sin_h, a)
            phidot = phidot + jnp.outer(omega_h * cos_h, b)
            phiddot = phiddot + jnp.outer(-(omega_h**2) * cos_h, a)
            phiddot = phiddot + jnp.outer(-(omega_h**2) * sin_h, b)
        return phi, phidot, phiddot

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        phi, phidot, phiddot = self.coefficients_to_time(x)
        residual = phiddot @ self.capacitance_f.T
        residual = residual + phidot @ self.conductance_s.T
        residual = residual + phi @ self.linear_stiffness_h_inv.T
        branch_flux = phi @ self.josephson_incidence
        branch_current = self.critical_current_a * jnp.sin(branch_flux / self.reduced_phi0)
        residual = residual + branch_current @ self.josephson_incidence.T
        residual = residual - self.source
        scale = 2.0 / self.time_samples
        cos_coeff = scale * jnp.einsum("tn,kt->kn", residual, self.cos_basis)
        sin_coeff = scale * jnp.einsum("tn,kt->kn", residual, self.sin_basis)
        coeff = jnp.stack([cos_coeff, sin_coeff], axis=1)
        return jnp.reshape(coeff, (-1,)) / self.config.residual_scale_a
