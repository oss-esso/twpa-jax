from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from twpa_solver.core.circuit import CircuitMatrices
from twpa_solver.core.linear import dynamic_block, port_s_from_unit_current_response
from twpa_solver.pump.backends.schur_partition import (
    assemble_schur_complements,
    build_partition,
)
from twpa_solver.signal.gain import GainResult, db10, gain_db_from_s

def sideband_list(sidebands: int) -> list[int]:
    return list(range(-sidebands, sidebands + 1))


def assemble_khat_conversion_base(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    ms: list[int],
) -> sp.csc_matrix:
    zero = sp.csr_matrix(circuit.C.shape, dtype=np.complex128)
    rows: list[list[sp.csr_matrix]] = []
    for m in ms:
        row: list[sp.csr_matrix] = []
        for q in ms:
            row.append(khat.get(m - q, zero).tocsr())
        rows.append(row)
    return sp.bmat(rows, format="csc")


def assemble_conversion_matrix_from_base(
    circuit: CircuitMatrices,
    khat_base: sp.spmatrix,
    omega_s: float,
    omega_p: float,
    ms: list[int],
    loss_model: str = "current_complex_c",
) -> sp.csc_matrix:
    dblocks = [dynamic_block(circuit, omega_s + m * omega_p, loss_model=loss_model) for m in ms]
    return (khat_base + sp.block_diag(dblocks, format="csc")).tocsc()


def assemble_conversion_matrix(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    omega_s: float,
    omega_p: float,
    ms: list[int],
    loss_model: str = "current_complex_c",
) -> sp.csc_matrix:
    zero = sp.csr_matrix(circuit.C.shape, dtype=np.complex128)
    rows: list[list[sp.csr_matrix]] = []

    D_cache: dict[int, sp.csr_matrix] = {}

    for m in ms:
        row: list[sp.csr_matrix] = []
        omega_m = omega_s + m * omega_p

        if m not in D_cache:
            D_cache[m] = dynamic_block(circuit, omega_m, loss_model=loss_model)

        for q in ms:
            ell = m - q
            block = khat.get(ell, zero)

            if m == q:
                block = block + D_cache[m]

            row.append(block.tocsr())

        rows.append(row)

    return sp.bmat(rows, format="csc")


def build_rhs(
    n: int,
    ms: list[int],
    signal_m: int,
    source_index: int,
    source_current_a: float,
) -> np.ndarray:
    b = np.zeros(len(ms) * n, dtype=np.complex128)
    row = ms.index(signal_m) * n + source_index
    b[row] = source_current_a
    return b


def extract_sideband_node(
    y: np.ndarray,
    n: int,
    ms: list[int],
    m: int,
    node_index: int,
) -> complex:
    return complex(y[ms.index(m) * n + node_index])


def voltage_from_flux(omega: float, phi: complex) -> complex:
    return 1j * omega * phi


def solve_single_block_transfer(
    circuit: CircuitMatrices,
    D_extra: sp.csr_matrix,
    omega_s: float,
    source_index: int,
    out_index: int,
    source_current_a: float,
    loss_model: str = "current_complex_c",
) -> tuple[complex, complex, float]:
    A = (dynamic_block(circuit, omega_s, loss_model=loss_model) + D_extra).tocsc()
    b = np.zeros(circuit.C.shape[0], dtype=np.complex128)
    b[source_index] = source_current_a

    t0 = time.perf_counter()
    y = spla.spsolve(A, b)
    runtime = time.perf_counter() - t0

    phi_out = complex(y[out_index])
    v_out = voltage_from_flux(omega_s, phi_out)

    return phi_out, v_out, runtime


def solve_linear_system(A: sp.spmatrix, b: np.ndarray, linear_solver: str = "superlu") -> np.ndarray:
    if linear_solver == "superlu":
        return spla.spsolve(A, b)
    if linear_solver == "pardiso":
        from pypardiso import spsolve as pardiso_spsolve
        if np.iscomplexobj(A.data) or np.iscomplexobj(b):
            A = A.tocsr()
            Ar = sp.bmat(
                [[A.real, -A.imag], [A.imag, A.real]],
                format="csr",
            )
            br = np.concatenate([np.asarray(b).real, np.asarray(b).imag])
            yr = pardiso_spsolve(Ar, br)
            n = b.size
            return np.asarray(yr[:n] + 1j * yr[n:], dtype=np.complex128)
        return pardiso_spsolve(A.tocsr(), b)
    raise ValueError(f"unknown linear_solver={linear_solver!r}")


def solve_gain_one(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    khat_off_0: sp.csr_matrix,
    omega_p: float,
    signal_ghz: float,
    sidebands: int,
    signal_m: int,
    idler_m: int,
    source_index: int,
    out_index: int,
    source_current_a: float,
    source_port: int,
    out_port: int,
    z0_ohm: float,
    loss_model: str = "current_complex_c",
    linear_solver: str = "superlu",
    khat_big_base: sp.spmatrix | None = None,
) -> GainResult:
    omega_s = 2.0 * math.pi * signal_ghz * 1e9
    ms = sideband_list(sidebands)
    n = circuit.C.shape[0]

    if signal_m not in ms:
        raise ValueError(f"signal_m={signal_m} not in sideband set {ms}")

    t0 = time.perf_counter()
    if khat_big_base is None:
        A = assemble_conversion_matrix(
            circuit=circuit,
            khat=khat,
            omega_s=omega_s,
            omega_p=omega_p,
            ms=ms,
            loss_model=loss_model,
        )
    else:
        A = assemble_conversion_matrix_from_base(
            circuit=circuit,
            khat_base=khat_big_base,
            omega_s=omega_s,
            omega_p=omega_p,
            ms=ms,
            loss_model=loss_model,
        )
    assemble_runtime_s = time.perf_counter() - t0

    b = build_rhs(
        n=n,
        ms=ms,
        signal_m=signal_m,
        source_index=source_index,
        source_current_a=source_current_a,
    )

    t0 = time.perf_counter()
    y = solve_linear_system(A, b, linear_solver=linear_solver)
    factor_solve_runtime_s = time.perf_counter() - t0

    r = A @ y - b
    linear_abs_residual = float(np.linalg.norm(r))
    linear_rel_residual = float(linear_abs_residual / max(np.linalg.norm(b), 1e-300))

    phi_out = extract_sideband_node(y, n, ms, signal_m, out_index)
    vout_on = voltage_from_flux(omega_s, phi_out)

    _, vout_off, off_runtime = solve_single_block_transfer(
        circuit=circuit,
        D_extra=khat_off_0,
        omega_s=omega_s,
        source_index=source_index,
        out_index=out_index,
        source_current_a=source_current_a,
        loss_model=loss_model,
    )

    # Pump-induced average stiffness only, no frequency-conversion sidebands.
    _, vout_pumpdiag, pumpdiag_runtime = solve_single_block_transfer(
        circuit=circuit,
        D_extra=khat.get(0, khat_off_0),
        omega_s=omega_s,
        source_index=source_index,
        out_index=out_index,
        source_current_a=source_current_a,
    )

    gain_vs_off = float(abs(vout_on / vout_off) ** 2)
    gain_vs_pumpdiag = float(abs(vout_on / vout_pumpdiag) ** 2)

    vout_idler = None
    idler_rel = None
    idler_rel_db = None

    if idler_m in ms:
        omega_i_signed = omega_s + idler_m * omega_p
        phi_i = extract_sideband_node(y, n, ms, idler_m, out_index)
        vout_idler = voltage_from_flux(omega_i_signed, phi_i)
        idler_rel = float(abs(vout_idler / vout_off) ** 2)
        idler_rel_db = db10(idler_rel)

    status = "VALID_SOLVED"
    if not np.isfinite(gain_vs_off) or linear_rel_residual > 1e-7:
        status = "CHECK"

    return GainResult(
        status=status,
        signal_ghz=signal_ghz,
        signal_m=signal_m,
        idler_m=idler_m,
        sidebands=sidebands,
        conversion_unknowns=A.shape[0],
        matrix_nnz=A.nnz,
        assemble_runtime_s=assemble_runtime_s,
        factor_solve_runtime_s=factor_solve_runtime_s,
        baseline_off_runtime_s=off_runtime,
        baseline_pumpdiag_runtime_s=pumpdiag_runtime,
        linear_abs_residual=linear_abs_residual,
        linear_rel_residual=linear_rel_residual,
        vout_on=vout_on,
        vout_off=vout_off,
        vout_pumpdiag=vout_pumpdiag,
        vout_idler=vout_idler,
        gain_vs_off=gain_vs_off,
        s_param_abs=abs(port_s_from_unit_current_response(vout_on / source_current_a, source_port=source_port, out_port=out_port, z0_ohm=z0_ohm)),
        gain_db=gain_db_from_s(port_s_from_unit_current_response(vout_on / source_current_a, source_port=source_port, out_port=out_port, z0_ohm=z0_ohm)),
        gain_vs_off_db=db10(gain_vs_off),
        gain_vs_pumpdiag=gain_vs_pumpdiag,
        gain_vs_pumpdiag_db=db10(gain_vs_pumpdiag),
        idler_power_rel_to_signal_off=idler_rel,
        idler_power_rel_to_signal_off_db=idler_rel_db,
    )


def _retained_khat(khat: dict[int, sp.csr_matrix], retained: np.ndarray, shape: tuple[int, int]) -> dict[int, sp.csr_matrix]:
    out: dict[int, sp.csr_matrix] = {}
    for ell, K in khat.items():
        out[ell] = K[retained][:, retained].tocsr()
    if 0 not in out:
        out[0] = sp.csr_matrix(shape, dtype=np.complex128)
    return out


def solve_gain_one_schur(
    circuit: CircuitMatrices,
    khat: dict[int, sp.csr_matrix],
    khat_off_0: sp.csr_matrix,
    omega_p: float,
    signal_ghz: float,
    sidebands: int,
    signal_m: int,
    idler_m: int,
    source_index: int,
    out_index: int,
    source_current_a: float,
    source_port: int,
    out_port: int,
    z0_ohm: float,
    loss_model: str = "current_complex_c",
    include_baselines: bool = True,
    linear_solver: str = "superlu",
    khat_big_base: sp.spmatrix | None = None,
) -> GainResult:
    """Schur-reduced signal solve; direct semantics, smaller coupled matrix."""
    omega_s = 2.0 * math.pi * signal_ghz * 1e9
    ms = sideband_list(sidebands)
    if signal_m not in ms:
        raise ValueError(f"signal_m={signal_m} not in sideband set {ms}")

    t0 = time.perf_counter()
    linear_blocks = [dynamic_block(circuit, omega_s + m * omega_p, loss_model=loss_model) for m in ms]
    part = build_partition(linear_blocks, circuit.Bphi, [source_index, out_index])
    assemble_schur_complements(part)
    mred = part.m
    khat_n = _retained_khat(khat, part.retained, (mred, mred))
    zero = sp.csr_matrix((mred, mred), dtype=np.complex128)
    rows: list[list[sp.csr_matrix]] = []
    for im, m in enumerate(ms):
        row: list[sp.csr_matrix] = []
        for q in ms:
            ell = m - q
            block = khat_n.get(ell, zero)
            if m == q:
                block = block + part.schur[im].tocsr()
            row.append(block.tocsr())
        rows.append(row)
    A = sp.bmat(rows, format="csc")
    assemble_runtime_s = time.perf_counter() - t0

    b = np.zeros(len(ms) * mred, dtype=np.complex128)
    src_pos = int(part.retained_pos[source_index])
    out_pos = int(part.retained_pos[out_index])
    b[ms.index(signal_m) * mred + src_pos] = source_current_a

    t0 = time.perf_counter()
    y = solve_linear_system(A, b, linear_solver=linear_solver)
    factor_solve_runtime_s = time.perf_counter() - t0

    r = A @ y - b
    linear_abs_residual = float(np.linalg.norm(r))
    linear_rel_residual = float(linear_abs_residual / max(np.linalg.norm(b), 1e-300))

    phi_out = complex(y[ms.index(signal_m) * mred + out_pos])
    vout_on = voltage_from_flux(omega_s, phi_out)

    off_runtime = 0.0
    pumpdiag_runtime = 0.0
    vout_off = complex(1.0)
    vout_pumpdiag = complex(1.0)
    if include_baselines:
        _, vout_off, off_runtime = solve_single_block_transfer(
            circuit=circuit,
            D_extra=khat_off_0,
            omega_s=omega_s,
            source_index=source_index,
            out_index=out_index,
            source_current_a=source_current_a,
            loss_model=loss_model,
        )
        _, vout_pumpdiag, pumpdiag_runtime = solve_single_block_transfer(
            circuit=circuit,
            D_extra=khat.get(0, khat_off_0),
            omega_s=omega_s,
            source_index=source_index,
            out_index=out_index,
            source_current_a=source_current_a,
        )

    gain_vs_off = float(abs(vout_on / vout_off) ** 2) if include_baselines else float("nan")
    gain_vs_pumpdiag = float(abs(vout_on / vout_pumpdiag) ** 2) if include_baselines else float("nan")

    vout_idler = None
    idler_rel = None
    idler_rel_db = None
    if idler_m in ms:
        omega_i_signed = omega_s + idler_m * omega_p
        phi_i = complex(y[ms.index(idler_m) * mred + out_pos])
        vout_idler = voltage_from_flux(omega_i_signed, phi_i)
        if include_baselines:
            idler_rel = float(abs(vout_idler / vout_off) ** 2)
            idler_rel_db = db10(idler_rel)

    s_abs = port_s_from_unit_current_response(
        vout_on / source_current_a,
        source_port=source_port,
        out_port=out_port,
        z0_ohm=z0_ohm,
    )
    status = "VALID_SOLVED"
    if not np.isfinite(abs(s_abs)) or linear_rel_residual > 1e-7:
        status = "CHECK"

    return GainResult(
        status=status,
        signal_ghz=signal_ghz,
        signal_m=signal_m,
        idler_m=idler_m,
        sidebands=sidebands,
        conversion_unknowns=A.shape[0],
        matrix_nnz=A.nnz,
        assemble_runtime_s=assemble_runtime_s,
        factor_solve_runtime_s=factor_solve_runtime_s,
        baseline_off_runtime_s=off_runtime,
        baseline_pumpdiag_runtime_s=pumpdiag_runtime,
        linear_abs_residual=linear_abs_residual,
        linear_rel_residual=linear_rel_residual,
        vout_on=vout_on,
        vout_off=vout_off,
        vout_pumpdiag=vout_pumpdiag,
        vout_idler=vout_idler,
        gain_vs_off=gain_vs_off,
        s_param_abs=abs(s_abs),
        gain_db=gain_db_from_s(s_abs),
        gain_vs_off_db=db10(gain_vs_off) if include_baselines else float("nan"),
        gain_vs_pumpdiag=gain_vs_pumpdiag,
        gain_vs_pumpdiag_db=db10(gain_vs_pumpdiag) if include_baselines else float("nan"),
        idler_power_rel_to_signal_off=idler_rel,
        idler_power_rel_to_signal_off_db=idler_rel_db,
    )
