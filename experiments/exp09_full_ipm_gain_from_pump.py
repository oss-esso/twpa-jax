# experiments/exp09_full_ipm_gain_from_pump.py
"""
Experiment 09: full IPM small-signal gain from solved pump.

No Julia. No JosephsonCircuits.

Inputs:
    outputs/ipm_python_design/C.npz
    outputs/ipm_python_design/G.npz
    outputs/ipm_python_design/K.npz
    outputs/ipm_python_design/Bphi.npz
    outputs/ipm_python_design/ipm_arrays.npz

    outputs/exp08_full_ipm_pump/pump_solution.npz
    outputs/exp08_full_ipm_pump/pump_report.json

Linearized conversion equation:
    A_mq y_q = b_m

with
    A_mq = D(Omega_m) delta_mq + Khat_{m-q}

    D(Omega) = K - Omega^2 C + i Omega G
    Khat_l  = Bphi diag(gamma_hat_l) Bphi.T
    gamma(t)= Ic/phi0 * cos(psi_pump(t)/phi0)

Sideband convention:
    delta x(t) = sum_m y_m exp(i (omega_s + m omega_p) t)

For 4WM conjugate-idler coupling, include negative sidebands.
Default sidebands=2 gives m = -2,-1,0,1,2.

Reported gain:
    pump_on output voltage at port 2, sideband m=0,
    normalized against no-pump linear transfer from port 1 to port 2.

This is a diagnostic gain, not final calibrated S-parameter gain.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from pump_basis import PumpBasis, load_pump_basis_from_solution  # noqa: E402


# =============================================================================
# Utilities
# =============================================================================

def db20(x: float) -> float:
    return 20.0 * math.log10(max(float(abs(x)), 1e-300))




def port_s_from_unit_current_response(
    response: complex,
    *,
    source_port: int,
    out_port: int,
    z0_ohm: float,
) -> complex:
    """Convert unit-current port voltage response into a JC-style S estimate.

    The current gain solver currently excites the source port with a unit current
    and reads the output port voltage-like response. Under the simple Z0
    Norton-port convention:

      transmission: S_ij = 2 V_i / (I_j Z0)
      reflection:   S_jj = 2 V_j / (I_j Z0) - 1

    This gives an absolute gain_db column comparable in form to JC.jl.
    """
    s = 2.0 * response / z0_ohm
    if int(source_port) == int(out_port):
        s -= 1.0
    return s


def gain_db_from_s(s: complex) -> float:
    return 20.0 * np.log10(max(abs(s), 1e-300))


def db10(x: float) -> float:
    return 10.0 * math.log10(max(float(x), 1e-300))


def complex_to_pair(z: complex) -> tuple[float, float]:
    return float(np.real(z)), float(np.imag(z))


def sideband_list(sidebands: int) -> list[int]:
    return list(range(-sidebands, sidebands + 1))


# =============================================================================
# Loading
# =============================================================================

@dataclass
class IPMMatrices:
    C: sp.csr_matrix
    G: sp.csr_matrix
    K: sp.csr_matrix
    Bphi: sp.csr_matrix
    Ic: np.ndarray
    phi0: float
    nodes: np.ndarray
    port_to_index: dict[int, int]


def infer_ipm_dir_from_pump_report(pump_dir: Path) -> str | None:
    report_path = pump_dir / "pump_report.json"
    if not report_path.exists():
        return None
    try:
        import json
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return None

    # Common report layouts.
    for container in (report, report.get("metadata", {}), report.get("settings", {})):
        if isinstance(container, dict):
            value = container.get("ipm_dir")
            if value:
                return str(value)

    return None


def load_ipm(ipm_dir: str | Path) -> IPMMatrices:
    d = Path(ipm_dir)

    C = sp.load_npz(d / "C.npz").tocsr()
    G = sp.load_npz(d / "G.npz").tocsr()
    K = sp.load_npz(d / "K.npz").tocsr()
    Bphi = sp.load_npz(d / "Bphi.npz").tocsr()

    arrays = np.load(d / "ipm_arrays.npz")
    Ic = arrays["Ic"].astype(float)
    phi0 = float(arrays["phi0_reduced"][0])
    nodes = arrays["nodes"].astype(int)

    port_numbers = arrays["port_numbers"].astype(int)
    port_indices = arrays["port_indices"].astype(int)
    port_to_index = {int(p): int(i) for p, i in zip(port_numbers, port_indices)}

    return IPMMatrices(
        C=C,
        G=G,
        K=K,
        Bphi=Bphi,
        Ic=Ic,
        phi0=phi0,
        nodes=nodes,
        port_to_index=port_to_index,
    )


@dataclass
class PumpSolution:
    X: np.ndarray
    omega_p: float
    pump_freq_ghz: float
    harmonics: int
    nt_original: int
    metadata: dict[str, Any]
    modes: list[int]
    basis: PumpBasis


def load_pump(pump_dir: str | Path, fallback_pump_freq_ghz: float) -> PumpSolution:
    d = Path(pump_dir)

    sol_path = d / "pump_solution.npz"
    if not sol_path.exists():
        raise FileNotFoundError(f"missing pump solution: {sol_path}")

    report_path = d / "pump_report.json"
    metadata: dict[str, Any] = {}

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        metadata = rep.get("metadata", {})
    else:
        rep = {}

    pump_freq_ghz = float(metadata.get("pump_freq_ghz", fallback_pump_freq_ghz))
    fallback_omega_p = 2.0 * math.pi * pump_freq_ghz * 1e9

    # Mode-aware load: reconstruct the exact pump basis (dense or odd-phasor).
    X, basis = load_pump_basis_from_solution(d, fallback_omega_p=fallback_omega_p)
    omega_p = basis.omega_p if basis.omega_p > 0.0 else fallback_omega_p
    nt_original = int(metadata.get("nt", 0))

    return PumpSolution(
        X=X,
        omega_p=omega_p,
        pump_freq_ghz=pump_freq_ghz,
        harmonics=X.shape[0],
        nt_original=nt_original,
        metadata=metadata,
        modes=list(basis.modes),
        basis=basis,
    )


# =============================================================================
# Pump synthesis and gamma Fourier coefficients
# =============================================================================

def synthesize_real_from_positive_harmonics(
    X: np.ndarray,
    omega: float,
    nt: int,
    modes: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the real pump waveform x(t) = 2 Re sum_k X_k exp(+i k omega t).

    `modes` are the positive integer pump-harmonic indices (one per row of X).
    Defaults to dense [1, 2, ..., H] for legacy solutions.
    """
    H = X.shape[0]
    if modes is None:
        k = np.arange(1, H + 1, dtype=float)
    else:
        k = np.asarray(modes, dtype=float).reshape(-1)
        if k.size != H:
            raise ValueError(f"modes length {k.size} != pump rows {H}")
    t = np.arange(nt, dtype=float) * (2.0 * math.pi / omega) / nt
    E = np.exp(1j * omega * t[:, None] * k[None, :])
    x_t = 2.0 * np.real(E @ X)
    return t, x_t




def load_dc_branch_flux(dc_solution: str | Path | None, ipm: IPMMatrices) -> np.ndarray | None:
    if dc_solution is None:
        return None

    p = Path(dc_solution)
    if p.is_dir():
        p = p / "dc_solution.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing dc solution: {p}")

    sol = np.load(p)
    if "psi_dc" in sol.files:
        psi_dc = np.asarray(sol["psi_dc"], dtype=np.float64).reshape(-1)
    elif "x_dc" in sol.files:
        x_dc = np.asarray(sol["x_dc"], dtype=np.float64).reshape(-1)
        if x_dc.size != ipm.C.shape[0]:
            raise ValueError(f"x_dc length {x_dc.size} != node count {ipm.C.shape[0]}")
        psi_dc = np.asarray(ipm.Bphi.T @ x_dc, dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"dc solution {p} must contain psi_dc or x_dc")

    if psi_dc.size != ipm.Bphi.shape[1]:
        raise ValueError(f"psi_dc length {psi_dc.size} != branch count {ipm.Bphi.shape[1]}")

    return psi_dc


def compute_gamma_hat(
    ipm: IPMMatrices,
    pump: PumpSolution,
    max_ell: int,
    gamma_nt: int,
    dc_branch_flux: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    t, x_t = synthesize_real_from_positive_harmonics(
        pump.X,
        pump.omega_p,
        gamma_nt,
        modes=pump.modes,
    )

    psi_t = (ipm.Bphi.T @ x_t.T).T
    if dc_branch_flux is not None:
        psi_t = psi_t + dc_branch_flux[None, :]
    gamma_t = (ipm.Ic[None, :] / ipm.phi0) * np.cos(psi_t / ipm.phi0)

    gamma_hat: dict[int, np.ndarray] = {}

    for ell in range(-max_ell, max_ell + 1):
        phase = np.exp(-1j * ell * pump.omega_p * t)
        gamma_hat[ell] = np.mean(gamma_t * phase[:, None], axis=0)

    return gamma_hat


def write_gamma_hat_summary(
    outdir: str | Path,
    gamma_hat: dict[int, np.ndarray],
) -> Path:
    """Diagnostic per-ell summary of the branch gamma_hat spectrum.

    Columns: ell, nbranches, l2_abs, l2_abs_over_zero_l2, max_abs, mean_abs,
    mean_real, mean_imag, conj_symmetry_rel_err.
    """
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "gamma_hat_summary.csv"

    zero = gamma_hat.get(0)
    zero_l2 = float(np.linalg.norm(zero)) if zero is not None else 0.0

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ell",
                "nbranches",
                "l2_abs",
                "l2_abs_over_zero_l2",
                "max_abs",
                "mean_abs",
                "mean_real",
                "mean_imag",
                "conj_symmetry_rel_err",
            ]
        )
        for ell in sorted(gamma_hat):
            gh = np.asarray(gamma_hat[ell])
            l2 = float(np.linalg.norm(gh))
            absgh = np.abs(gh)

            conj_err = 0.0
            mirror = gamma_hat.get(-ell)
            if mirror is not None and l2 > 0.0:
                conj_err = float(
                    np.linalg.norm(gh - np.conj(mirror)) / l2
                )

            w.writerow(
                [
                    ell,
                    int(gh.size),
                    l2,
                    l2 / zero_l2 if zero_l2 > 0.0 else 0.0,
                    float(np.max(absgh)) if gh.size else 0.0,
                    float(np.mean(absgh)) if gh.size else 0.0,
                    float(np.mean(gh.real)) if gh.size else 0.0,
                    float(np.mean(gh.imag)) if gh.size else 0.0,
                    conj_err,
                ]
            )

    print(f"wrote_gamma_hat_summary={path}")
    return path


def build_khat(
    Bphi: sp.csr_matrix,
    gamma_hat: dict[int, np.ndarray],
    drop_tol: float,
) -> dict[int, sp.csr_matrix]:
    khat: dict[int, sp.csr_matrix] = {}

    for ell, gh in gamma_hat.items():
        if np.max(np.abs(gh)) < drop_tol:
            khat[ell] = sp.csr_matrix((Bphi.shape[0], Bphi.shape[0]), dtype=np.complex128)
            continue

        Kh = Bphi @ sp.diags(gh, offsets=0, format="csr") @ Bphi.T
        khat[ell] = Kh.astype(np.complex128).tocsr()

    return khat


# =============================================================================
# Matrix assembly
# =============================================================================

def dynamic_block(
    ipm: IPMMatrices,
    omega: float,
) -> sp.csr_matrix:
    C = ipm.C.astype(np.complex128)
    G = ipm.G.astype(np.complex128)
    K = ipm.K.astype(np.complex128)
    return (K + (-omega * omega) * C + (1j * omega) * G).tocsr()


def assemble_conversion_matrix(
    ipm: IPMMatrices,
    khat: dict[int, sp.csr_matrix],
    omega_s: float,
    omega_p: float,
    ms: list[int],
) -> sp.csc_matrix:
    zero = sp.csr_matrix(ipm.C.shape, dtype=np.complex128)
    rows: list[list[sp.csr_matrix]] = []

    D_cache: dict[int, sp.csr_matrix] = {}

    for m in ms:
        row: list[sp.csr_matrix] = []
        omega_m = omega_s + m * omega_p

        if m not in D_cache:
            D_cache[m] = dynamic_block(ipm, omega_m)

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


# =============================================================================
# Baselines
# =============================================================================

def solve_single_block_transfer(
    ipm: IPMMatrices,
    D_extra: sp.csr_matrix,
    omega_s: float,
    source_index: int,
    out_index: int,
    source_current_a: float,
) -> tuple[complex, complex, float]:
    A = (dynamic_block(ipm, omega_s) + D_extra).tocsc()
    b = np.zeros(ipm.C.shape[0], dtype=np.complex128)
    b[source_index] = source_current_a

    t0 = time.perf_counter()
    y = spla.spsolve(A, b)
    runtime = time.perf_counter() - t0

    phi_out = complex(y[out_index])
    v_out = voltage_from_flux(omega_s, phi_out)

    return phi_out, v_out, runtime


# =============================================================================
# Solve one signal point
# =============================================================================

@dataclass
class GainResult:
    status: str
    signal_ghz: float
    signal_m: int
    idler_m: int
    sidebands: int
    conversion_unknowns: int
    matrix_nnz: int
    assemble_runtime_s: float
    factor_solve_runtime_s: float
    baseline_off_runtime_s: float
    baseline_pumpdiag_runtime_s: float
    linear_abs_residual: float
    linear_rel_residual: float

    vout_on: complex
    vout_off: complex
    vout_pumpdiag: complex
    vout_idler: complex | None

    gain_vs_off: float
    s_param_abs: float
    gain_db: float
    gain_vs_off_db: float
    gain_vs_pumpdiag: float
    gain_vs_pumpdiag_db: float
    idler_power_rel_to_signal_off: float | None
    idler_power_rel_to_signal_off_db: float | None


def solve_gain_one(
    ipm: IPMMatrices,
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
) -> GainResult:
    omega_s = 2.0 * math.pi * signal_ghz * 1e9
    ms = sideband_list(sidebands)
    n = ipm.C.shape[0]

    if signal_m not in ms:
        raise ValueError(f"signal_m={signal_m} not in sideband set {ms}")

    t0 = time.perf_counter()
    A = assemble_conversion_matrix(
        ipm=ipm,
        khat=khat,
        omega_s=omega_s,
        omega_p=omega_p,
        ms=ms,
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
    y = spla.spsolve(A, b)
    factor_solve_runtime_s = time.perf_counter() - t0

    r = A @ y - b
    linear_abs_residual = float(np.linalg.norm(r))
    linear_rel_residual = float(linear_abs_residual / max(np.linalg.norm(b), 1e-300))

    phi_out = extract_sideband_node(y, n, ms, signal_m, out_index)
    vout_on = voltage_from_flux(omega_s, phi_out)

    _, vout_off, off_runtime = solve_single_block_transfer(
        ipm=ipm,
        D_extra=khat_off_0,
        omega_s=omega_s,
        source_index=source_index,
        out_index=out_index,
        source_current_a=source_current_a,
    )

    # Pump-induced average stiffness only, no frequency-conversion sidebands.
    _, vout_pumpdiag, pumpdiag_runtime = solve_single_block_transfer(
        ipm=ipm,
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


# =============================================================================
# Printing / CSV
# =============================================================================

def print_single_result(r: GainResult) -> None:
    print("=== experiment 09: full IPM gain from pump ===")
    print(f"status={r.status}")
    print(f"signal_ghz={r.signal_ghz:.12g}")
    print(f"signal_m={r.signal_m}")
    print(f"idler_m={r.idler_m}")
    print(f"sidebands={r.sidebands}")
    print(f"conversion_unknowns={r.conversion_unknowns}")
    print(f"matrix_nnz={r.matrix_nnz}")
    print(f"assemble_runtime_s={r.assemble_runtime_s:.6f}")
    print(f"factor_solve_runtime_s={r.factor_solve_runtime_s:.6f}")
    print(f"baseline_off_runtime_s={r.baseline_off_runtime_s:.6f}")
    print(f"baseline_pumpdiag_runtime_s={r.baseline_pumpdiag_runtime_s:.6f}")
    print(f"linear_abs_residual={r.linear_abs_residual:.12e}")
    print(f"linear_rel_residual={r.linear_rel_residual:.12e}")

    print("\n=== output voltages ===")
    print(f"vout_on_real={r.vout_on.real:.12e}")
    print(f"vout_on_imag={r.vout_on.imag:.12e}")
    print(f"vout_off_real={r.vout_off.real:.12e}")
    print(f"vout_off_imag={r.vout_off.imag:.12e}")
    print(f"vout_pumpdiag_real={r.vout_pumpdiag.real:.12e}")
    print(f"vout_pumpdiag_imag={r.vout_pumpdiag.imag:.12e}")

    if r.vout_idler is not None:
        print(f"vout_idler_real={r.vout_idler.real:.12e}")
        print(f"vout_idler_imag={r.vout_idler.imag:.12e}")

    print("\n=== diagnostic gain ===")
    print(f"s_param_abs={r.s_param_abs:.12e}")
    print(f"gain_db={r.gain_db:.6f}")
    print(f"gain_vs_off={r.gain_vs_off:.12e}")
    print(f"gain_vs_off_db={r.gain_vs_off_db:.6f}")
    print(f"gain_vs_pumpdiag={r.gain_vs_pumpdiag:.12e}")
    print(f"gain_vs_pumpdiag_db={r.gain_vs_pumpdiag_db:.6f}")

    if r.idler_power_rel_to_signal_off is not None:
        print(f"idler_power_rel_to_signal_off={r.idler_power_rel_to_signal_off:.12e}")
        print(f"idler_power_rel_to_signal_off_db={r.idler_power_rel_to_signal_off_db:.6f}")

    print("\ninterpretation:")
    print("  gain_vs_off compares pump-on conversion to no-pump linear transfer.")
    print("  gain_vs_pumpdiag removes the pump-induced average stiffness shift,")
    print("  leaving mostly the effect of frequency-conversion sideband coupling.")
    print("  This is diagnostic transfer gain, not final calibrated S-parameter gain.")


def csv_header() -> list[str]:
    return [
        "status",
        "signal_ghz",
        "gain_db",
        "s_param_abs",
        "gain_vs_off_db",
        "gain_vs_pumpdiag_db",
        "idler_power_rel_to_signal_off_db",
        "linear_rel_residual",
        "conversion_unknowns",
        "matrix_nnz",
        "assemble_runtime_s",
        "factor_solve_runtime_s",
        "vout_on_real",
        "vout_on_imag",
        "vout_off_real",
        "vout_off_imag",
    ]


def result_to_csv_row(r: GainResult) -> list[Any]:
    return [
        r.status,
        r.signal_ghz,
        r.gain_db,
        r.s_param_abs,
        r.gain_vs_off_db,
        r.gain_vs_pumpdiag_db,
        r.idler_power_rel_to_signal_off_db,
        r.linear_rel_residual,
        r.conversion_unknowns,
        r.matrix_nnz,
        r.assemble_runtime_s,
        r.factor_solve_runtime_s,
        r.vout_on.real,
        r.vout_on.imag,
        r.vout_off.real,
        r.vout_off.imag,
    ]


def print_sweep_row(r: GainResult) -> None:
    idb = "" if r.idler_power_rel_to_signal_off_db is None else f"{r.idler_power_rel_to_signal_off_db:.9g}"
    print(
        f"{r.signal_ghz:.12g},"
        f"{r.status},"
        f"{r.gain_db:.9g},"
        f"{r.s_param_abs:.9g},"
        f"{r.gain_vs_off_db:.9g},"
        f"{r.gain_vs_pumpdiag_db:.9g},"
        f"{idb},"
        f"{r.linear_rel_residual:.3e},"
        f"{r.factor_solve_runtime_s:.6f}"
    )


def write_outputs(outdir: str | Path, rows: list[GainResult], metadata: dict[str, Any]) -> None:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    csv_path = d / "gain_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(csv_header())
        for r in rows:
            w.writerow(result_to_csv_row(r))

    js_path = d / "gain_report.json"
    payload = {
        "metadata": metadata,
        "results": [
            {
                **{
                    k: v
                    for k, v in r.__dict__.items()
                    if not isinstance(v, complex)
                },
                "vout_on": complex_to_pair(r.vout_on),
                "vout_off": complex_to_pair(r.vout_off),
                "vout_pumpdiag": complex_to_pair(r.vout_pumpdiag),
                "vout_idler": None if r.vout_idler is None else complex_to_pair(r.vout_idler),
            }
            for r in rows
        ],
    }

    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote_csv={csv_path}")
    print(f"wrote_report={js_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")

    p.add_argument("--ipm-dir", default=os.path.join("outputs", "ipm_python_design"))
    p.add_argument("--pump-dir", default=os.path.join("outputs", "exp08_full_ipm_pump"))
    p.add_argument("--dc-solution", default=None, help="Optional dc_solution.npz or folder containing it. Uses gamma around DC+pump state.")
    p.add_argument("--outdir", default=os.path.join("outputs", "exp09_full_ipm_gain"))

    p.add_argument("--z0-ohm", type=float, default=50.0, help="Port impedance for JC-style S-parameter gain extraction.")
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    p.add_argument("--source-current-a", type=float, default=1.0)

    p.add_argument("--signal-ghz", type=float, default=6.0)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--signal-start-ghz", type=float, default=4.0)
    p.add_argument("--signal-stop-ghz", type=float, default=8.0)
    p.add_argument("--points", type=int, default=21)

    p.add_argument("--sidebands", type=int, default=2)
    p.add_argument("--signal-m", type=int, default=0)
    p.add_argument("--idler-m", type=int, default=-2)

    p.add_argument("--real-capacitance", action="store_true",
                   help="Drop the imaginary (lossy) part of C in the linearized gain solve (loss-convention study).")
    p.add_argument("--gamma-nt", type=int, default=128)
    p.add_argument("--fallback-pump-freq-ghz", type=float, default=7.9)
    p.add_argument("--drop-gamma-tol", type=float, default=0.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    t_all = time.perf_counter()

    resolved_ipm_dir = args.ipm_dir or infer_ipm_dir_from_pump_report(Path(args.pump_dir)) or "outputs/ipm_python_design"
    ipm = load_ipm(resolved_ipm_dir)
    if args.real_capacitance:
        ipm.C = ipm.C.real.astype(np.complex128).tocsr()
        print("real_capacitance=True (loss dropped from linearized gain solve)")
    pump = load_pump(args.pump_dir, args.fallback_pump_freq_ghz)
    dc_branch_flux = load_dc_branch_flux(args.dc_solution, ipm)

    if args.source_port not in ipm.port_to_index:
        raise ValueError(f"source port {args.source_port} not in {ipm.port_to_index}")
    if args.out_port not in ipm.port_to_index:
        raise ValueError(f"out port {args.out_port} not in {ipm.port_to_index}")

    source_index = ipm.port_to_index[args.source_port]
    out_index = ipm.port_to_index[args.out_port]

    ms = sideband_list(args.sidebands)
    max_ell = max(abs(m - q) for m in ms for q in ms)

    print("=== experiment 09 setup ===")
    print(f"ipm_dir={resolved_ipm_dir}")
    print(f"nodes={ipm.C.shape[0]}")
    print(f"jj_branches={ipm.Bphi.shape[1]}")
    print(f"ports={ipm.port_to_index}")
    print(f"source_port={args.source_port}")
    print(f"out_port={args.out_port}")
    print(f"source_index={source_index}")
    print(f"out_index={out_index}")
    print(f"pump_harmonics={pump.harmonics}")
    print(f"pump_nt_original={pump.nt_original}")
    print(f"pump_freq_ghz={pump.pump_freq_ghz}")
    if dc_branch_flux is not None:
        print(f"dc_solution={args.dc_solution}")
        print(f"dc_branch_flux_max_abs={float(np.max(np.abs(dc_branch_flux))):.12e}")
        print(f"dc_branch_flux_over_phi0_max_abs={float(np.max(np.abs(dc_branch_flux / ipm.phi0))):.12e}")
    print(f"sideband_set={ms}")
    print(f"max_gamma_ell={max_ell}")
    print(f"gamma_nt={args.gamma_nt}")

    t0 = time.perf_counter()
    gamma_hat = compute_gamma_hat(
        ipm=ipm,
        pump=pump,
        max_ell=max_ell,
        gamma_nt=args.gamma_nt,
        dc_branch_flux=dc_branch_flux,
    )
    gamma_runtime = time.perf_counter() - t0

    write_gamma_hat_summary(args.outdir, gamma_hat)

    print(f"pump_modes={pump.modes}")
    print(f"pump_basis={pump.basis.basis}")
    print(f"pump_mode_policy={pump.basis.policy}")
    print(f"gamma_hat_runtime_s={gamma_runtime:.6f}")
    print(f"gamma_hat_0_abs_max={float(np.max(np.abs(gamma_hat[0]))):.12e}")
    if 2 in gamma_hat:
        print(f"gamma_hat_plus2_abs_max={float(np.max(np.abs(gamma_hat[2]))):.12e}")
    if -2 in gamma_hat:
        print(f"gamma_hat_minus2_abs_max={float(np.max(np.abs(gamma_hat[-2]))):.12e}")

    t0 = time.perf_counter()
    khat = build_khat(
        Bphi=ipm.Bphi,
        gamma_hat=gamma_hat,
        drop_tol=args.drop_gamma_tol,
    )
    khat_runtime = time.perf_counter() - t0

    if dc_branch_flux is None:
        gamma_off = ipm.Ic / ipm.phi0
    else:
        gamma_off = (ipm.Ic / ipm.phi0) * np.cos(dc_branch_flux / ipm.phi0)

    khat_off_0 = (
        ipm.Bphi
        @ sp.diags(gamma_off, offsets=0, format="csr")
        @ ipm.Bphi.T
    ).astype(np.complex128).tocsr()

    print(f"khat_build_runtime_s={khat_runtime:.6f}")
    print(f"khat_0_nnz={khat[0].nnz}")
    if 2 in khat:
        print(f"khat_plus2_nnz={khat[2].nnz}")
    if -2 in khat:
        print(f"khat_minus2_nnz={khat[-2].nnz}")

    if args.sweep:
        freqs = np.linspace(args.signal_start_ghz, args.signal_stop_ghz, args.points)
    else:
        freqs = np.array([args.signal_ghz], dtype=float)

    rows: list[GainResult] = []

    if args.sweep:
        print("\nsignal_ghz,status,gain_db,s_param_abs,gain_vs_off_db,gain_vs_pumpdiag_db,idler_rel_db,linear_rel_residual,factor_solve_runtime_s")

    for fghz in freqs:
        r = solve_gain_one(
            ipm=ipm,
            khat=khat,
            khat_off_0=khat_off_0,
            omega_p=pump.omega_p,
            signal_ghz=float(fghz),
            sidebands=args.sidebands,
            signal_m=args.signal_m,
            idler_m=args.idler_m,
            source_index=source_index,
            out_index=out_index,
            source_current_a=args.source_current_a,
            source_port=args.source_port,
            out_port=args.out_port,
            z0_ohm=args.z0_ohm,
        )
        rows.append(r)

        if args.sweep:
            print_sweep_row(r)
        else:
            print_single_result(r)

    metadata = {
        "ipm_dir": args.ipm_dir,
        "pump_dir": args.pump_dir,
        "source_port": args.source_port,
        "out_port": args.out_port,
        "source_index": source_index,
        "out_index": out_index,
        "source_current_a": args.source_current_a,
        "z0_ohm": args.z0_ohm,
        "pump_harmonics": pump.harmonics,
        "pump_modes": list(pump.modes),
        "pump_basis": pump.basis.basis,
        "pump_mode_policy": pump.basis.policy,
        "pump_nt_original": pump.nt_original,
        "pump_freq_ghz": pump.pump_freq_ghz,
        "omega_p": pump.omega_p,
        "sidebands": args.sidebands,
        "sideband_set": ms,
        "signal_m": args.signal_m,
        "idler_m": args.idler_m,
        "gamma_nt": args.gamma_nt,
        "gamma_hat_runtime_s": gamma_runtime,
        "khat_build_runtime_s": khat_runtime,
        "total_runtime_s": time.perf_counter() - t_all,
    }

    write_outputs(args.outdir, rows, metadata)

    print("\n=== final sweep summary ===")
    print(f"points={len(rows)}")
    print(f"all_status_valid={all(r.status == 'VALID_SOLVED' for r in rows)}")
    print(f"gain_db_max={max(r.gain_db for r in rows):.6f}")
    print(f"gain_db_mean={float(np.mean([r.gain_db for r in rows])):.6f}")
    print(f"gain_db_min={min(r.gain_db for r in rows):.6f}")
    print(f"peak_frequency_ghz={max(rows, key=lambda r: r.gain_db).signal_ghz}")
    print(f"nfrequencies={len(rows)}")
    print(f"max_gain_vs_off_db={max(r.gain_vs_off_db for r in rows):.6f}")
    print(f"min_gain_vs_off_db={min(r.gain_vs_off_db for r in rows):.6f}")
    print(f"max_gain_vs_pumpdiag_db={max(r.gain_vs_pumpdiag_db for r in rows):.6f}")
    print(f"total_runtime_s={time.perf_counter() - t_all:.6f}")


if __name__ == "__main__":
    main()
