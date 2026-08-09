"""Pump-off multi-port scattering utilities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.core.circuit import load_circuit
from twpa_solver.core.linear import dynamic_block, port_s_from_unit_current_response
from twpa_solver.core.nonlinear import make_branch_law


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def passive_s_matrix(
    circuit_dir: str | Path,
    freqs_hz: np.ndarray,
    *,
    ports: tuple[int, ...] = (1, 2, 3, 4),
    z0_ohm: float = 50.0,
    dc_branch_flux: np.ndarray | None = None,
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

    dc = np.zeros(circuit.Bphi.shape[1]) if dc_branch_flux is None else np.asarray(dc_branch_flux, dtype=float)
    if dc.shape != (circuit.Bphi.shape[1],):
        raise ValueError("dc_branch_flux must have one value per branch")
    gamma_off = make_branch_law(circuit).tangent(dc[None, :])[0]
    extra_k = (circuit.Bphi @ sp.diags(gamma_off) @ circuit.Bphi.T).astype(np.complex128).tocsr()
    for row, frequency_hz in enumerate(freqs):
        omega = 2.0 * math.pi * float(frequency_hz)
        solution = spla.spsolve(dynamic_block(circuit, omega, extra_K=extra_k), rhs)
        if solution.ndim == 1:
            solution = solution[:, None]
        for source_column, source_port in enumerate(ports):
            for output_row, output_port in enumerate(ports):
                voltage = 1j * omega * solution[indices[output_row], source_column]
                result[row, output_row, source_column] = port_s_from_unit_current_response(
                    voltage, source_port=source_port, out_port=output_port, z0_ohm=z0_ohm
                )
    return result


def passive_network_matrices(
    circuit_dir: str | Path,
    freqs_hz: np.ndarray,
    *,
    ports: tuple[int, ...] | None = None,
    z0_ohm: float = 50.0,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return loaded-port ``Z``, ``Y`` and ``S`` matrices at every frequency.

    ``Z`` is the port voltage response to unit injected port currents using the
    same matched-port convention as :func:`passive_s_matrix`; ``Y`` is its
    numerical inverse and ``S = 2 Z / Z0 - I``.  The returned arrays have
    shape ``(frequency, output_port, input_port)`` and therefore work for any
    one-, two-, or four-port circuit.
    """
    circuit = load_circuit(circuit_dir)
    selected = tuple(sorted(circuit.port_to_index)) if ports is None else tuple(ports)
    for port in selected:
        if port not in circuit.port_to_index:
            raise ValueError(f"port {port} not in design ports {circuit.port_to_index}")
    if not selected:
        raise ValueError("at least one port is required")
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    indices = [circuit.port_to_index[p] for p in selected]
    rhs = np.zeros((circuit.node_count, len(selected)), dtype=np.complex128)
    for column, index in enumerate(indices):
        rhs[index, column] = 1.0
    z_matrix = np.zeros((freqs.size, len(selected), len(selected)), dtype=np.complex128)
    s_matrix = np.zeros_like(z_matrix)

    dc = np.zeros(circuit.Bphi.shape[1]) if dc_branch_flux is None else np.asarray(dc_branch_flux, dtype=float)
    if dc.shape != (circuit.Bphi.shape[1],):
        raise ValueError("dc_branch_flux must have one value per branch")
    gamma_off = make_branch_law(circuit).tangent(dc[None, :])[0]
    extra_k = (circuit.Bphi @ sp.diags(gamma_off) @ circuit.Bphi.T).astype(np.complex128).tocsr()
    identity = np.eye(len(selected), dtype=np.complex128)
    for row, frequency_hz in enumerate(freqs):
        omega = 2.0 * math.pi * float(frequency_hz)
        solution = spla.spsolve(dynamic_block(circuit, omega, extra_K=extra_k), rhs)
        if solution.ndim == 1:
            solution = solution[:, None]
        z = 1j * omega * solution[indices, :]
        z_matrix[row] = z
        s_matrix[row] = 2.0 * z / float(z0_ohm) - identity
    y_matrix = np.linalg.pinv(z_matrix)
    return {
        "ports": np.asarray(selected, dtype=int),
        "Z": z_matrix,
        "Y": y_matrix,
        "S": s_matrix,
    }
