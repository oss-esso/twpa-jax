"""Pump-off multi-port scattering utilities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.core.circuit import load_circuit
from twpa_solver.core.linear import dynamic_block, port_s_from_unit_current_response


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def passive_s_matrix(
    circuit_dir: str | Path,
    freqs_hz: np.ndarray,
    *,
    ports: tuple[int, ...] = (1, 2, 3, 4),
    z0_ohm: float = 50.0,
) -> np.ndarray:
    """Return ``S[frequency, output_port, source_port]`` with pump off."""
    circuit = load_circuit(circuit_dir)
    for port in ports:
        if port not in circuit.port_to_index:
            raise ValueError(f"port {port} not in design ports {circuit.port_to_index}")

    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    indices = [circuit.port_to_index[p] for p in ports]
    rhs = np.zeros((circuit.node_count, len(ports)), dtype=np.complex128)
    for column, index in enumerate(indices):
        rhs[index, column] = 1.0
    result = np.zeros((freqs.size, len(ports), len(ports)), dtype=np.complex128)

    gamma_off = circuit.Ic / circuit.phi0
    extra_k = (circuit.Bphi @ sp.diags(gamma_off) @ circuit.Bphi.T).astype(np.complex128).tocsr()
    for row, frequency_hz in enumerate(freqs):
        omega = 2.0 * math.pi * float(frequency_hz)
        solution = spla.spsolve(dynamic_block(circuit, omega, extra_K=extra_k), rhs)
        for source_column, source_port in enumerate(ports):
            for output_row, output_port in enumerate(ports):
                voltage = 1j * omega * solution[indices[output_row], source_column]
                result[row, output_row, source_column] = port_s_from_unit_current_response(
                    voltage, source_port=source_port, out_port=output_port, z0_ohm=z0_ohm
                )
    return result
