"""Linear signal response about a converged autonomous torus.

The autonomous torus uses the two-generator lattice ``h*omega_p + q*omega_a``.
A weak signal is an independent third frequency.  This module keeps the
signal-sector lattice signed, ``omega_s + h*omega_p + q*omega_a``, so the
negative-frequency idler is retained explicitly and the linear response does
not get confused with a nonlinear pump-plus-signal solve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import CircuitMatrices
from twpa_solver.core.nonlinear import make_branch_law
from twpa_solver.core.linear import (
    dynamic_block,
    port_s_from_unit_current_response,
    solve_linear_scattering,
)
from twpa_solver.multitone.basis import MultiToneBasis, ToneIndex
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.pump.backends.schur_partition import (
    assemble_schur_complements,
    build_partition,
)
from twpa_solver.signal.floquet import solve_linear_system


@dataclass(frozen=True)
class TorusSignalResponseResult:
    """Result and telemetry for one torus small-signal solve."""

    signal_ghz: float
    gain_vs_off_db: float
    s_on: complex
    s_off: complex
    residual_rel: float
    response_unknowns: int
    matrix_nnz: int
    assemble_runtime_s: float
    factor_solve_runtime_s: float
    response_tones: tuple[ToneIndex, ...]


def response_tones(
    omega_p: float,
    omega_a: float,
    omega_s: float,
    *,
    h_min: int = -2,
    h_max: int = 2,
    q_min: int = -1,
    q_max: int = 1,
) -> tuple[ToneIndex, ...]:
    """Return a signed third-frequency response lattice.

    Signed tones are intentional.  For example, ``(h=-2, q=0)`` is the
    negative-frequency partner of the pump-generated idler when the signal is
    below the pump.  The source tone ``(0, 0)`` is always retained.
    """
    if omega_p <= 0.0 or omega_a <= 0.0:
        raise ValueError("omega_p and omega_a must be positive")
    if h_min > h_max or q_min > q_max:
        raise ValueError("invalid response lattice bounds")
    tones = [
        ToneIndex(h, q)
        for h in range(h_min, h_max + 1)
        for q in range(q_min, q_max + 1)
        if not (h == 0 and q == 0)
    ]
    tones.append(ToneIndex(0, 0))
    ordered = tuple(sorted(set(tones), key=lambda tone: (tone.h, tone.q)))
    frequencies = np.asarray(
        [omega_s + tone.omega(omega_p, omega_a) for tone in ordered],
        dtype=float,
    )
    if np.any(np.isclose(frequencies, 0.0, atol=1.0e3)):
        raise ValueError("response lattice contains a near-DC frequency")
    if np.unique(np.round(frequencies, decimals=3)).size != frequencies.size:
        raise ValueError("response lattice contains aliased frequencies")
    return ordered


def _response_khat(
    problem: FullMultiToneProblem,
    state: np.ndarray,
    offsets: set[ToneIndex],
) -> dict[ToneIndex, sp.csr_matrix]:
    """Build retained-node Fourier coefficients of the torus tangent."""
    tangent = problem.tangent_state(state)
    port_indices = list(problem.circuit.port_to_index.values())
    retained = np.unique(
        np.concatenate((
            np.asarray(problem.Bphi.nonzero()[0], dtype=int),
            np.asarray(port_indices, dtype=int),
        ))
    )
    bphi_r = problem.Bphi[retained].tocsr()
    bphi_t_r = bphi_r.T.tocsr()
    result: dict[ToneIndex, sp.csr_matrix] = {}
    for offset in offsets:
        phase = problem.grid.phase_rows([offset])[0]
        gamma_hat = np.sum(tangent.gamma_t * phase[:, None], axis=0)
        result[offset] = (
            bphi_r @ sp.diags(gamma_hat, 0) @ bphi_t_r
        ).astype(np.complex128).tocsr()
    return result


def _retained_partition(
    circuit: CircuitMatrices,
    frequencies: np.ndarray,
    loss_model: str,
) -> tuple[list[sp.csc_matrix], np.ndarray]:
    """Build Schur blocks for the signed response frequencies."""
    blocks = [
        dynamic_block(circuit, float(omega), loss_model=loss_model)
        for omega in frequencies
    ]
    partition = build_partition(
        blocks,
        circuit.Bphi,
        list(circuit.port_to_index.values()),
    )
    assemble_schur_complements(partition)
    if partition.schur is None:
        raise RuntimeError("Schur complements were not assembled")
    return partition.schur, partition.retained


def _port_response_s(
    coefficient: complex,
    omega: float,
    source_current_a: float,
    circuit: CircuitMatrices,
    *,
    source_port: int,
    out_port: int,
) -> complex:
    """Convert a positive-response flux coefficient into an S-like value."""
    voltage = 2.0 * 1j * omega * coefficient
    return port_s_from_unit_current_response(
        voltage / source_current_a,
        source_port=source_port,
        out_port=out_port,
    )


def solve_torus_signal_response(
    problem: FullMultiToneProblem,
    state: np.ndarray,
    *,
    signal_ghz: float,
    source_port: int,
    out_port: int,
    source_current_a: float = 1.0e-12,
    loss_model: str = "current_complex_c",
    linear_solver: str = "pardiso",
    h_min: int = -2,
    h_max: int = 2,
    q_min: int = -1,
    q_max: int = 1,
) -> TorusSignalResponseResult:
    """Solve the weak-signal linearization around an autonomous torus.

    The torus state is not modified.  The returned gain is normalized to the
    pump-off linear transmission at the same signal frequency, exactly as the
    existing ``gain_vs_off`` observable.
    """
    if source_current_a == 0.0:
        raise ValueError("source_current_a must be nonzero")
    state = np.asarray(state, dtype=np.complex128)
    if state.shape != (problem.basis.n_tones, problem.circuit.node_count):
        raise ValueError("state shape does not match the torus problem")
    omega_s = 2.0 * np.pi * float(signal_ghz) * 1.0e9
    tones = response_tones(
        problem.basis.omega_p,
        problem.basis.delta,
        omega_s,
        h_min=h_min,
        h_max=h_max,
        q_min=q_min,
        q_max=q_max,
    )
    frequencies = np.asarray(
        [omega_s + tone.omega(problem.basis.omega_p, problem.basis.delta) for tone in tones],
        dtype=float,
    )
    offsets = {
        left - right
        for left in tones
        for right in tones
    }
    khat = _response_khat(problem, state, offsets)
    schur_blocks, retained = _retained_partition(
        problem.circuit, frequencies, loss_model
    )
    zero = sp.csr_matrix((retained.size, retained.size), dtype=np.complex128)
    rows: list[list[sp.spmatrix]] = []
    for row_tone in tones:
        row: list[sp.spmatrix] = []
        for col_index, col_tone in enumerate(tones):
            block = khat.get(row_tone - col_tone, zero)
            if row_tone == col_tone:
                block = block + schur_blocks[tones.index(row_tone)]
            row.append(block)
        rows.append(row)
    started = time.perf_counter()
    matrix = sp.bmat(rows, format="csr")
    source_tone_index = tones.index(ToneIndex(0, 0))
    source_node = problem.circuit.port_to_index[int(source_port)]
    retained_position = np.flatnonzero(retained == source_node)
    if retained_position.size != 1:
        raise ValueError("source port is not retained by the Schur partition")
    output_node = problem.circuit.port_to_index[int(out_port)]
    output_position = np.flatnonzero(retained == output_node)
    if output_position.size != 1:
        raise ValueError("output port is not retained by the Schur partition")
    rhs = np.zeros(matrix.shape[0], dtype=np.complex128)
    rhs[source_tone_index * retained.size + retained_position[0]] = (
        0.5 * source_current_a
    )
    assemble_runtime = time.perf_counter() - started
    solve_started = time.perf_counter()
    solution = solve_linear_system(matrix, rhs, linear_solver=linear_solver)
    factor_solve_runtime = time.perf_counter() - solve_started
    residual = matrix @ solution - rhs
    residual_rel = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-300))
    signal_index = source_tone_index * retained.size + output_position[0]
    s_on = _port_response_s(
        solution[signal_index],
        omega_s,
        source_current_a,
        problem.circuit,
        source_port=source_port,
        out_port=out_port,
    )
    off = solve_linear_scattering(
        problem.circuit,
        frequency_hz=float(signal_ghz) * 1.0e9,
        source_port=source_port,
        out_port=out_port,
        source_current_a=source_current_a,
        loss_model=loss_model,
        extra_K=(
            problem.circuit.Bphi
            @ sp.diags(
                make_branch_law(problem.circuit).gamma(
                    np.zeros((1, problem.circuit.Bphi.shape[1]))
                )[0],
                0,
            )
            @ problem.circuit.Bphi.T
        ),
    )
    s_off = off.s
    gain = float(20.0 * np.log10(max(abs(s_on / s_off), 1e-300)))
    return TorusSignalResponseResult(
        signal_ghz=float(signal_ghz),
        gain_vs_off_db=gain,
        s_on=s_on,
        s_off=s_off,
        residual_rel=residual_rel,
        response_unknowns=int(matrix.shape[0]),
        matrix_nnz=int(matrix.nnz),
        assemble_runtime_s=float(assemble_runtime),
        factor_solve_runtime_s=float(factor_solve_runtime),
        response_tones=tones,
    )


def torus_junction_utilization(
    problem: FullMultiToneProblem,
    state: np.ndarray,
) -> float:
    """Return the existing ``max_abs(I_J/I_c)`` torus observable."""
    branch_flux = problem.branch_flux_time(state)
    total_flux = branch_flux + problem.dc_branch_flux[None, :]
    currents = np.asarray(problem.branch.current(total_flux), dtype=float)
    peak_current = np.max(np.abs(currents), axis=0)
    critical = np.asarray(problem.branch.critical_current, dtype=float).reshape(-1)
    return float(np.max(peak_current / np.maximum(np.abs(critical), 1e-300)))
