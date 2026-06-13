"""Linearized pumped conversion-matrix residual assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twpa_solver.model.topology import CircuitModel
from twpa_solver.residuals.aft_hb import PumpAFTResidual
from twpa_solver.residuals.linear import y_to_s


@dataclass(frozen=True)
class ConversionResult:
    signal_frequency_hz: float
    sidebands: tuple[int, ...]
    frequencies_hz: tuple[float, ...]
    y_conversion_s: np.ndarray
    s_conversion: np.ndarray
    pump_success: bool
    pump_status: str
    metadata: dict[str, float | str | int]

    @property
    def signal_gain_db(self) -> float:
        return _db(abs(self.s_conversion[self._out_signal_index(), self._in_signal_index()]) ** 2)

    @property
    def idler_gain_db(self) -> float:
        idler_idx = self._sideband_index(1)
        return _db(abs(self.s_conversion[self._port_sb_index(1, idler_idx), self._in_signal_index()]) ** 2)

    def _sideband_index(self, sideband: int) -> int:
        return self.sidebands.index(sideband)

    def _port_sb_index(self, port_idx: int, sideband_idx: int) -> int:
        return port_idx * len(self.sidebands) + sideband_idx

    def _in_signal_index(self) -> int:
        return self._port_sb_index(0, self._sideband_index(0))

    def _out_signal_index(self) -> int:
        return self._port_sb_index(1, self._sideband_index(0))


def build_conversion_sparameters(
    model: CircuitModel,
    pump_residual: PumpAFTResidual,
    pump_solution: np.ndarray,
    signal_frequency_hz: float,
    sidebands: int,
    *,
    pump_success: bool,
    pump_status: str,
) -> ConversionResult:
    """Build sideband conversion S-parameters around a pumped periodic state."""
    if sidebands < 0:
        raise ValueError("sidebands must be non-negative")
    sb = tuple(range(-sidebands, sidebands + 1))
    phi_t, _, _ = pump_residual.coefficients_to_time(pump_solution)
    k_t = model.linear_stiffness_h_inv[None, :, :] + model.nonlinear_derivative_matrices(phi_t)
    k_coeff = _fourier_coefficients(k_t, pump_residual.harmonics)
    y_full = _sideband_port_admittance(
        model,
        k_coeff,
        pump_residual.config.pump_frequency_hz,
        signal_frequency_hz,
        sb,
    )
    z0 = []
    for port in model.ports:
        z0.extend([port.z0_ohm] * len(sb))
    s = y_to_s(y_full, np.asarray(z0))
    freqs = tuple(float(signal_frequency_hz + m * pump_residual.config.pump_frequency_hz) for m in sb)
    return ConversionResult(
        signal_frequency_hz=float(signal_frequency_hz),
        sidebands=sb,
        frequencies_hz=freqs,
        y_conversion_s=y_full,
        s_conversion=s,
        pump_success=bool(pump_success),
        pump_status=pump_status,
        metadata={
            "pump_frequency_hz": pump_residual.config.pump_frequency_hz,
            "pump_harmonics": pump_residual.harmonics,
            "sidebands": sidebands,
        },
    )


def _fourier_coefficients(k_t: np.ndarray, harmonics: int) -> dict[int, np.ndarray]:
    fft = np.fft.fft(k_t, axis=0) / k_t.shape[0]
    coeff: dict[int, np.ndarray] = {0: fft[0]}
    for h in range(1, harmonics + 1):
        coeff[h] = fft[h]
        coeff[-h] = fft[-h]
    return coeff


def _sideband_port_admittance(
    model: CircuitModel,
    k_coeff: dict[int, np.ndarray],
    pump_frequency_hz: float,
    signal_frequency_hz: float,
    sidebands: tuple[int, ...],
) -> np.ndarray:
    n = model.num_nodes
    ns = len(sidebands)
    ports = model.port_indices()
    pcount = len(ports)
    internal = [idx for idx in range(n) if idx not in ports]
    a = np.zeros((ns * n, ns * n), dtype=complex)
    for row_s, m in enumerate(sidebands):
        omega_m = 2.0 * np.pi * (signal_frequency_hz + m * pump_frequency_hz)
        dyn = -omega_m**2 * model.capacitance_f + 1j * omega_m * model.conductance_s
        for col_s, q in enumerate(sidebands):
            k = k_coeff.get(m - q, np.zeros((n, n), dtype=complex))
            block = dyn + k if row_s == col_s else k
            rs = slice(row_s * n, (row_s + 1) * n)
            cs = slice(col_s * n, (col_s + 1) * n)
            a[rs, cs] = block / (1j * omega_m)
    port_dofs = [s * n + node for node in ports for s in range(ns)]
    internal_dofs = [s * n + node for node in internal for s in range(ns)]
    ypp = a[np.ix_(port_dofs, port_dofs)]
    if not internal_dofs:
        return ypp
    ypi = a[np.ix_(port_dofs, internal_dofs)]
    yip = a[np.ix_(internal_dofs, port_dofs)]
    yii = a[np.ix_(internal_dofs, internal_dofs)]
    return ypp - ypi @ np.linalg.solve(yii, yip)


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), 1e-300)))
