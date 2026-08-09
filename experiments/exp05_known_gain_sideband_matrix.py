# experiments/exp05_known_gain_sideband_matrix.py
"""
Experiment 05: known-gain fixture in Exp03-style sideband matrix form.

Purpose:
    Exp04 proved that the conjugate-idler coupled-mode equations can produce
    gain and match their analytic formula.

    Exp03 proved that our pump -> gamma(t) -> sideband matrix -> linear solve
    pipeline runs.

    Exp05 bridges the two:
        build a sideband-indexed algebraic matrix A y = r
        with sideband labels m = -M ... M
        where the signal sideband m=0 couples to the negative-frequency
        conjugate-idler sideband m=-2.

    This tests the exact class of bug we care about:
        if we omit the negative/conjugate idler, gain disappears.

Sideband model:
    Let the signal sideband be m_s = 0.
    Let the conjugate-idler sideband be m_i = -2.

    Solve:
        A y = r

    with pair block:

        [ D_i    -rho* ] [ y_i ] = [ 0 ]
        [ -rho   D_s  ] [ y_s ]   [ 1 ]

    For D_s = D_i = 1:
        y_s = 1 / (1 - |rho|^2)

    Normalized power gain relative to no-pump baseline:
        G = | y_s / (1 / D_s) |^2

    If rho is chosen from Exp04 phase-matched transfer amplitude T:
        rho = sqrt(1 - 1 / |T|)
    then the algebraic sideband fixture has the same phase-matched gain.

Important:
    This is still a validation fixture, not the physical TWPA circuit.
    It validates sideband indexing, conjugate-idler inclusion, and sparse solve logic.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# Small utilities
# =============================================================================

def safe_db(power_gain: float) -> float:
    return 10.0 * math.log10(max(float(power_gain), 1e-300))


def complex_sqrt(z: complex) -> complex:
    return complex(np.sqrt(np.complex128(z)))


def exp04_transfer(kappa: float, delta: float, length: float) -> complex:
    """
    Same analytic transfer used in Exp04:

        d/dz [a_s, a_i^*]^T =
            [[-i delta/2, kappa],
             [ kappa,     i delta/2]] [a_s, a_i^*]^T

    Signal transfer:
        T = cosh(gL) + (-i delta/2)/g sinh(gL)
        g = sqrt(kappa^2 - (delta/2)^2)
    """
    g = complex_sqrt(kappa**2 - (0.5 * delta) ** 2)
    z = g * length

    if abs(g) < 1e-14:
        return complex(1.0 + (-0.5j * delta) * length)

    return complex(np.cosh(z) + ((-0.5j * delta) / g) * np.sinh(z))


def rho_from_target_transfer_amplitude(target_amp: float) -> float:
    """
    For D_s = D_i = 1 and y_s = 1 / (1 - rho^2),
    choose rho so y_s = target_amp.

        rho = sqrt(1 - 1/target_amp)

    Requires target_amp >= 1.
    """
    if target_amp < 1.0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - 1.0 / target_amp)))


def sideband_indices(sidebands: int) -> list[int]:
    return list(range(-sidebands, sidebands + 1))


# =============================================================================
# Sideband matrix result
# =============================================================================

@dataclass
class SidebandGainResult:
    status: str
    sidebands: int
    coupling_mode: str
    rho: complex
    delta: float
    signal_m: int
    idler_m: int
    conversion_unknowns: int
    matrix_nnz: int
    solve_runtime_s: float
    total_runtime_s: float
    linear_abs_residual: float
    linear_rel_residual: float
    y_signal: complex
    y_idler: complex
    baseline_positive_only: complex
    gain_sideband: float
    gain_sideband_db: float
    gain_positive_only: float
    gain_positive_only_db: float
    gain_analytic_pair: float | None
    gain_analytic_pair_db: float | None
    rel_transfer_error_pair: float | None
    exp04_phase_matched_gain_db: float
    exp04_phase_matched_transfer_abs: float


# =============================================================================
# Matrix construction
# =============================================================================

def dynamic_diagonal_for_sideband(
    m: int,
    signal_m: int,
    idler_m: int,
    delta: float,
    off_resonance_diag: complex,
) -> complex:
    """
    Normalized algebraic diagonal.

    Pair subspace:
        D_s = 1 - i delta/2
        D_i = 1 + i delta/2

    Other sidebands:
        off_resonance_diag, only used to keep the full sideband matrix
        well-conditioned when extra sidebands are present.
    """
    if m == signal_m:
        return 1.0 - 0.5j * delta
    if m == idler_m:
        return 1.0 + 0.5j * delta
    return off_resonance_diag


def assemble_sideband_matrix(
    sidebands: int,
    rho: complex,
    delta: float,
    signal_m: int,
    idler_m: int,
    coupling_mode: str,
    off_resonance_diag: complex,
) -> tuple[sp.csc_matrix, list[int], dict[int, int]]:
    """
    Build sideband-indexed A.

    coupling_mode='pair':
        Only couple signal_m <-> idler_m.
        This has an exact 2x2 analytic reference.

    coupling_mode='periodic_all':
        Couple every pair with m-q = +/-2, mimicking a scalar periodic
        stiffness harmonic. This is useful later, but the simple 2x2
        analytic reference no longer applies.
    """
    ms = sideband_indices(sidebands)
    idx = {m: i for i, m in enumerate(ms)}

    if signal_m not in idx:
        raise ValueError(f"signal_m={signal_m} not inside sideband range {ms}")
    if idler_m not in idx:
        raise ValueError(f"idler_m={idler_m} not inside sideband range {ms}")

    A = sp.lil_matrix((len(ms), len(ms)), dtype=np.complex128)

    for m in ms:
        A[idx[m], idx[m]] = dynamic_diagonal_for_sideband(
            m=m,
            signal_m=signal_m,
            idler_m=idler_m,
            delta=delta,
            off_resonance_diag=off_resonance_diag,
        )

    if coupling_mode == "pair":
        # Signal row receives idler contribution through harmonic +2.
        A[idx[signal_m], idx[idler_m]] = -rho

        # Idler-conjugate row receives signal contribution through harmonic -2.
        A[idx[idler_m], idx[signal_m]] = -np.conj(rho)

    elif coupling_mode == "periodic_all":
        # Exp03-like periodic coefficient pattern:
        # A_mq has coupling when m-q = +/-2.
        for m in ms:
            for q in ms:
                ell = m - q
                if ell == 2:
                    A[idx[m], idx[q]] += -rho
                elif ell == -2:
                    A[idx[m], idx[q]] += -np.conj(rho)

    else:
        raise ValueError(f"unknown coupling_mode={coupling_mode!r}")

    return A.tocsc(), ms, idx


def analytic_pair_transfer(
    rho: complex,
    delta: float,
) -> tuple[complex, complex, complex, float]:
    """
    Exact 2x2 formula for pair mode.

        [D_i    -rho*] [y_i] = [0]
        [-rho    D_s ] [y_s] = [1]

    D_s = 1 - i delta/2
    D_i = 1 + i delta/2

    y_s = D_i / (D_s D_i - |rho|^2)
    y_i = rho* / (D_s D_i - |rho|^2)

    Baseline:
        y_s0 = 1 / D_s
    """
    D_s = 1.0 - 0.5j * delta
    D_i = 1.0 + 0.5j * delta
    det = D_s * D_i - abs(rho) ** 2

    y_s = D_i / det
    y_i = np.conj(rho) / det
    y_baseline = 1.0 / D_s

    gain = float(abs(y_s / y_baseline) ** 2)
    return complex(y_s), complex(y_i), complex(y_baseline), gain


def solve_sideband_gain(
    sidebands: int,
    rho: complex,
    delta: float,
    signal_m: int,
    idler_m: int,
    coupling_mode: str,
    off_resonance_diag: complex,
    linear_tol: float,
    exp04_phase_matched_gain_db: float,
    exp04_phase_matched_transfer_abs: float,
) -> SidebandGainResult:
    t_total0 = time.perf_counter()

    A, ms, idx = assemble_sideband_matrix(
        sidebands=sidebands,
        rho=rho,
        delta=delta,
        signal_m=signal_m,
        idler_m=idler_m,
        coupling_mode=coupling_mode,
        off_resonance_diag=off_resonance_diag,
    )

    rhs = np.zeros(len(ms), dtype=np.complex128)
    rhs[idx[signal_m]] = 1.0

    t0 = time.perf_counter()
    y = spla.spsolve(A, rhs)
    solve_runtime_s = time.perf_counter() - t0

    residual = A @ y - rhs
    linear_abs_residual = float(np.linalg.norm(residual))
    linear_rel_residual = float(linear_abs_residual / max(np.linalg.norm(rhs), 1e-30))

    y_signal = complex(y[idx[signal_m]])
    y_idler = complex(y[idx[idler_m]])

    D_s = dynamic_diagonal_for_sideband(
        m=signal_m,
        signal_m=signal_m,
        idler_m=idler_m,
        delta=delta,
        off_resonance_diag=off_resonance_diag,
    )
    baseline_positive_only = complex(1.0 / D_s)

    gain_sideband = float(abs(y_signal / baseline_positive_only) ** 2)
    gain_positive_only = 1.0

    gain_analytic_pair = None
    gain_analytic_pair_db = None
    rel_transfer_error_pair = None

    if coupling_mode == "pair":
        y_s_ref, _y_i_ref, _base_ref, gain_ref = analytic_pair_transfer(
            rho=rho,
            delta=delta,
        )
        gain_analytic_pair = gain_ref
        gain_analytic_pair_db = safe_db(gain_ref)
        rel_transfer_error_pair = float(
            abs(y_signal - y_s_ref) / max(abs(y_s_ref), 1e-30)
        )

    status = "VALID_CONVERGED"
    if linear_rel_residual > linear_tol:
        status = "FAIL_LINEAR_RESIDUAL"
    if coupling_mode == "pair" and rel_transfer_error_pair is not None:
        if rel_transfer_error_pair > linear_tol:
            status = "FAIL_PAIR_ANALYTIC_MISMATCH"

    total_runtime_s = time.perf_counter() - t_total0

    return SidebandGainResult(
        status=status,
        sidebands=sidebands,
        coupling_mode=coupling_mode,
        rho=rho,
        delta=delta,
        signal_m=signal_m,
        idler_m=idler_m,
        conversion_unknowns=A.shape[0],
        matrix_nnz=A.nnz,
        solve_runtime_s=solve_runtime_s,
        total_runtime_s=total_runtime_s,
        linear_abs_residual=linear_abs_residual,
        linear_rel_residual=linear_rel_residual,
        y_signal=y_signal,
        y_idler=y_idler,
        baseline_positive_only=baseline_positive_only,
        gain_sideband=gain_sideband,
        gain_sideband_db=safe_db(gain_sideband),
        gain_positive_only=gain_positive_only,
        gain_positive_only_db=safe_db(gain_positive_only),
        gain_analytic_pair=gain_analytic_pair,
        gain_analytic_pair_db=gain_analytic_pair_db,
        rel_transfer_error_pair=rel_transfer_error_pair,
        exp04_phase_matched_gain_db=exp04_phase_matched_gain_db,
        exp04_phase_matched_transfer_abs=exp04_phase_matched_transfer_abs,
    )


# =============================================================================
# Sweeps and output
# =============================================================================

def print_single(r: SidebandGainResult) -> None:
    print("=== experiment 05: known gain in sideband matrix ===")
    print(f"status={r.status}")
    print(f"coupling_mode={r.coupling_mode}")
    print(f"sidebands={r.sidebands}")
    print(f"signal_m={r.signal_m}")
    print(f"idler_m={r.idler_m}")
    print(f"conversion_unknowns={r.conversion_unknowns}")
    print(f"matrix_nnz={r.matrix_nnz}")
    print(f"rho_real={r.rho.real:.12e}")
    print(f"rho_imag={r.rho.imag:.12e}")
    print(f"rho_abs={abs(r.rho):.12e}")
    print(f"delta={r.delta:.12e}")
    print(f"solve_runtime_s={r.solve_runtime_s:.6e}")
    print(f"total_runtime_s={r.total_runtime_s:.6e}")
    print(f"linear_abs_residual={r.linear_abs_residual:.12e}")
    print(f"linear_rel_residual={r.linear_rel_residual:.12e}")

    print("\n=== sideband solution ===")
    print(f"y_signal_real={r.y_signal.real:.12e}")
    print(f"y_signal_imag={r.y_signal.imag:.12e}")
    print(f"y_idler_conj_real={r.y_idler.real:.12e}")
    print(f"y_idler_conj_imag={r.y_idler.imag:.12e}")
    print(f"baseline_positive_only_real={r.baseline_positive_only.real:.12e}")
    print(f"baseline_positive_only_imag={r.baseline_positive_only.imag:.12e}")

    print("\n=== gain check ===")
    print(f"gain_sideband={r.gain_sideband:.12e}")
    print(f"gain_sideband_db={r.gain_sideband_db:.6f}")
    print(f"gain_positive_only={r.gain_positive_only:.12e}")
    print(f"gain_positive_only_db={r.gain_positive_only_db:.6f}")

    if r.gain_analytic_pair is not None:
        print(f"gain_analytic_pair={r.gain_analytic_pair:.12e}")
        print(f"gain_analytic_pair_db={r.gain_analytic_pair_db:.6f}")
        print(f"rel_transfer_error_pair={r.rel_transfer_error_pair:.12e}")

    print("\n=== exp04 phase-matched reference used for rho ===")
    print(f"exp04_phase_matched_transfer_abs={r.exp04_phase_matched_transfer_abs:.12e}")
    print(f"exp04_phase_matched_gain_db={r.exp04_phase_matched_gain_db:.6f}")

    print("\ninterpretation:")
    print("  gain_sideband > 0 dB means the sideband matrix can amplify.")
    print("  gain_positive_only = 0 dB means omitting the conjugate idler kills gain.")
    print("  pair mode should match the analytic 2x2 sideband formula to roundoff.")


def print_sweep_header() -> None:
    print("delta,status,gain_sideband_db,gain_positive_only_db,gain_analytic_pair_db,linear_rel_residual,rel_transfer_error_pair,rho_abs")


def print_sweep_row(r: SidebandGainResult) -> None:
    analytic_db = "" if r.gain_analytic_pair_db is None else f"{r.gain_analytic_pair_db:.12g}"
    rel_err = "" if r.rel_transfer_error_pair is None else f"{r.rel_transfer_error_pair:.6e}"

    print(
        f"{r.delta:.12g},"
        f"{r.status},"
        f"{r.gain_sideband_db:.12g},"
        f"{r.gain_positive_only_db:.12g},"
        f"{analytic_db},"
        f"{r.linear_rel_residual:.6e},"
        f"{rel_err},"
        f"{abs(r.rho):.12g}"
    )


def write_csv(path: str, rows: list[SidebandGainResult]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "delta",
                "status",
                "coupling_mode",
                "sidebands",
                "signal_m",
                "idler_m",
                "rho_abs",
                "rho_real",
                "rho_imag",
                "gain_sideband",
                "gain_sideband_db",
                "gain_positive_only",
                "gain_positive_only_db",
                "gain_analytic_pair",
                "gain_analytic_pair_db",
                "linear_rel_residual",
                "rel_transfer_error_pair",
                "solve_runtime_s",
                "total_runtime_s",
            ]
        )

        for r in rows:
            w.writerow(
                [
                    r.delta,
                    r.status,
                    r.coupling_mode,
                    r.sidebands,
                    r.signal_m,
                    r.idler_m,
                    abs(r.rho),
                    r.rho.real,
                    r.rho.imag,
                    r.gain_sideband,
                    r.gain_sideband_db,
                    r.gain_positive_only,
                    r.gain_positive_only_db,
                    r.gain_analytic_pair,
                    r.gain_analytic_pair_db,
                    r.linear_rel_residual,
                    r.rel_transfer_error_pair,
                    r.solve_runtime_s,
                    r.total_runtime_s,
                ]
            )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")

    p.add_argument("--sidebands", type=int, default=2)
    p.add_argument("--signal-m", type=int, default=0)
    p.add_argument("--idler-m", type=int, default=-2)

    p.add_argument("--coupling-mode", choices=["pair", "periodic_all"], default="pair")

    # If --rho is not provided, rho is derived from the Exp04 phase-matched gain.
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--rho-phase", type=float, default=0.0)

    # Exp04 reference parameters used only to choose default rho.
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--length", type=float, default=8.0)

    # Algebraic sideband detuning. This is dimensionless here.
    p.add_argument("--delta", type=float, default=0.0)

    # Other sideband diagonals. Keep >1 to avoid accidental resonances.
    p.add_argument("--off-resonance-diag-real", type=float, default=3.0)
    p.add_argument("--off-resonance-diag-imag", type=float, default=0.0)

    p.add_argument("--linear-tol", type=float, default=1e-12)

    p.add_argument("--sweep-delta", action="store_true")
    p.add_argument("--delta-max", type=float, default=2.0)
    p.add_argument("--points", type=int, default=41)
    p.add_argument("--csv", type=str, default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sidebands < 2:
        raise ValueError("Use --sidebands >= 2 so m=0 can couple to m=-2.")
    if args.points < 2 and args.sweep_delta:
        raise ValueError("--points must be >= 2 for --sweep-delta")

    T_exp04_pm = exp04_transfer(
        kappa=args.kappa,
        delta=0.0,
        length=args.length,
    )
    T_exp04_pm_abs = abs(T_exp04_pm)
    G_exp04_pm_db = safe_db(T_exp04_pm_abs**2)

    if args.rho is None:
        rho_abs = rho_from_target_transfer_amplitude(T_exp04_pm_abs)
    else:
        rho_abs = args.rho

    rho = rho_abs * np.exp(1j * args.rho_phase)

    off_diag = complex(args.off_resonance_diag_real, args.off_resonance_diag_imag)

    if args.sweep_delta:
        print("=== experiment 05: delta sweep ===")
        print(f"coupling_mode={args.coupling_mode}")
        print(f"rho_abs={abs(rho):.12e}")
        print(f"sidebands={args.sidebands}")
        print(f"exp04_phase_matched_gain_db={G_exp04_pm_db:.6f}")
        print_sweep_header()

        rows: list[SidebandGainResult] = []
        for delta in np.linspace(-args.delta_max, args.delta_max, args.points):
            r = solve_sideband_gain(
                sidebands=args.sidebands,
                rho=rho,
                delta=float(delta),
                signal_m=args.signal_m,
                idler_m=args.idler_m,
                coupling_mode=args.coupling_mode,
                off_resonance_diag=off_diag,
                linear_tol=args.linear_tol,
                exp04_phase_matched_gain_db=G_exp04_pm_db,
                exp04_phase_matched_transfer_abs=T_exp04_pm_abs,
            )
            rows.append(r)
            print_sweep_row(r)

        print(f"\nall_valid={all(r.status == 'VALID_CONVERGED' for r in rows)}")
        print(f"max_gain_sideband_db={max(r.gain_sideband_db for r in rows):.6f}")
        print(f"min_gain_sideband_db={min(r.gain_sideband_db for r in rows):.6f}")

        if args.csv:
            write_csv(args.csv, rows)
            print(f"wrote_csv={args.csv}")

    else:
        r = solve_sideband_gain(
            sidebands=args.sidebands,
            rho=rho,
            delta=args.delta,
            signal_m=args.signal_m,
            idler_m=args.idler_m,
            coupling_mode=args.coupling_mode,
            off_resonance_diag=off_diag,
            linear_tol=args.linear_tol,
            exp04_phase_matched_gain_db=G_exp04_pm_db,
            exp04_phase_matched_transfer_abs=T_exp04_pm_abs,
        )
        print_single(r)


if __name__ == "__main__":
    main()