# experiments/exp06_synthetic_gamma_circuit_matrix.py
"""
Experiment 06: synthetic Gamma(t) gain in Exp03-style circuit matrix.

Purpose:
    Validate that the Exp03-style circuit assembly

        A_mq = D_m delta_mq + Khat_{m-q}

    can produce known positive gain when Khat_l comes from a synthetic
    time-periodic differential stiffness Gamma(t).

This is the bridge between:
    Exp05: hand-built sideband pair matrix
    Exp03: circuit-style Khat_l = B diag(gamma_hat_l) B^T

Synthetic model:
    One scalar circuit coordinate, B = [1].

    We prescribe

        gamma(t) = 2 Re[ (-rho) exp(i 2 omega_p t) ]

    so that

        gamma_hat_{+2} = -rho
        gamma_hat_{-2} = -rho*

    Then the circuit assembly gives

        A_{m,q} = D_m delta_mq + gamma_hat_{m-q}

    For the minimal sideband set m in {-2, 0}, this becomes

        [ D_i    -rho* ] [ y_i ] = [ 0 ]
        [ -rho   D_s  ] [ y_s ]   [ 1 ]

    which is exactly the known positive-gain sideband fixture.

Success criteria:
    pair sideband set:
        gain_circuit_db matches analytic pair gain
        positive-only baseline is 0 dB
        residual is tiny

    range sideband set:
        gain remains finite and positive, but need not match pair formula
        because extra sidebands are included.
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
# Utilities
# =============================================================================

def safe_db(power_gain: float) -> float:
    return 10.0 * math.log10(max(float(power_gain), 1e-300))


def complex_sqrt(z: complex) -> complex:
    return complex(np.sqrt(np.complex128(z)))


def exp04_transfer(kappa: float, delta: float, length: float) -> complex:
    """
    Reference transfer from Exp04 coupled-mode model.
    Used only to choose default rho from the known phase-matched gain.
    """
    g = complex_sqrt(kappa**2 - (0.5 * delta) ** 2)
    z = g * length

    if abs(g) < 1e-14:
        return complex(1.0 + (-0.5j * delta) * length)

    return complex(np.cosh(z) + ((-0.5j * delta) / g) * np.sinh(z))


def rho_from_target_transfer_amplitude(target_amp: float) -> float:
    """
    In the minimal pair model with D_s = D_i = 1,

        y_s = 1 / (1 - rho^2)

    Choose rho so y_s equals target_amp:

        rho = sqrt(1 - 1/target_amp)
    """
    if target_amp < 1.0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - 1.0 / target_amp)))


# =============================================================================
# Gamma(t) synthetic AFT
# =============================================================================

@dataclass
class SyntheticGammaGrid:
    nt: int
    omega_p: float

    def __post_init__(self) -> None:
        if self.nt < 8:
            raise ValueError("--nt must be >= 8")
        self.period = 2.0 * np.pi / self.omega_p
        self.t = np.arange(self.nt, dtype=float) * self.period / self.nt

    def gamma_time(self, rho: complex) -> np.ndarray:
        """
        gamma(t) = 2 Re[ (-rho) exp(i 2 omega_p t) ]

        Therefore:
            gamma_hat_{+2} = -rho
            gamma_hat_{-2} = -rho*
        """
        coeff_plus_2 = -rho
        return 2.0 * np.real(coeff_plus_2 * np.exp(1j * 2.0 * self.omega_p * self.t))

    def gamma_hat(self, rho: complex, ell: int) -> complex:
        gamma_t = self.gamma_time(rho)
        phase = np.exp(-1j * ell * self.omega_p * self.t)
        return complex(np.mean(gamma_t * phase))


def build_khat_from_synthetic_gamma(
    grid: SyntheticGammaGrid,
    rho: complex,
    max_ell: int,
) -> dict[int, sp.csc_matrix]:
    """
    Scalar-node version of Exp03:

        Khat_ell = B diag(gamma_hat_ell) B^T

    with B = [1], so Khat_ell is a 1x1 sparse matrix.
    """
    khat: dict[int, sp.csc_matrix] = {}

    for ell in range(-max_ell, max_ell + 1):
        gh = grid.gamma_hat(rho, ell)
        khat[ell] = sp.csc_matrix([[gh]], dtype=np.complex128)

    return khat


# =============================================================================
# Circuit-style sideband assembly
# =============================================================================

def sideband_list(sideband_set: str, sidebands: int, signal_m: int, idler_m: int) -> list[int]:
    if sideband_set == "pair":
        return [idler_m, signal_m]

    if sideband_set == "range":
        return list(range(-sidebands, sidebands + 1))

    if sideband_set == "positive_only":
        return [signal_m]

    raise ValueError(f"unknown sideband_set={sideband_set!r}")


def dynamic_block_for_m(
    m: int,
    signal_m: int,
    idler_m: int,
    delta: float,
    off_resonance_diag: complex,
) -> sp.csc_matrix:
    """
    Normalized scalar D_m.

    For the target pair:
        D_s = 1 - i delta/2
        D_i = 1 + i delta/2

    Other sidebands get off_resonance_diag.
    """
    if m == signal_m:
        value = 1.0 - 0.5j * delta
    elif m == idler_m:
        value = 1.0 + 0.5j * delta
    else:
        value = off_resonance_diag

    return sp.csc_matrix([[value]], dtype=np.complex128)


def assemble_circuit_style_A(
    ms: list[int],
    khat: dict[int, sp.csc_matrix],
    signal_m: int,
    idler_m: int,
    delta: float,
    off_resonance_diag: complex,
) -> sp.csc_matrix:
    """
    Exact Exp03-style assembly:

        A_mq = D_m delta_mq + Khat_{m-q}

    Here each block is 1x1, but the block logic is identical.
    """
    blocks: list[list[sp.csc_matrix]] = []

    for m in ms:
        row: list[sp.csc_matrix] = []
        Dm = dynamic_block_for_m(
            m=m,
            signal_m=signal_m,
            idler_m=idler_m,
            delta=delta,
            off_resonance_diag=off_resonance_diag,
        )

        for q in ms:
            ell = m - q

            if ell in khat:
                block = khat[ell]
            else:
                block = sp.csc_matrix((1, 1), dtype=np.complex128)

            if m == q:
                block = block + Dm

            row.append(block.tocsc())

        blocks.append(row)

    return sp.bmat(blocks, format="csc")


def build_rhs(ms: list[int], signal_m: int) -> np.ndarray:
    rhs = np.zeros(len(ms), dtype=np.complex128)
    rhs[ms.index(signal_m)] = 1.0
    return rhs


def baseline_positive_only(signal_m: int, idler_m: int, delta: float, off_resonance_diag: complex) -> complex:
    D_s = dynamic_block_for_m(
        m=signal_m,
        signal_m=signal_m,
        idler_m=idler_m,
        delta=delta,
        off_resonance_diag=off_resonance_diag,
    )[0, 0]
    return complex(1.0 / D_s)


# =============================================================================
# Analytic pair reference
# =============================================================================

def analytic_pair_solution(rho: complex, delta: float) -> tuple[complex, complex, complex, float]:
    """
    Exact 2x2 reference:

        [D_i    -rho*] [y_i] = [0]
        [-rho    D_s ] [y_s] = [1]

    D_s = 1 - i delta/2
    D_i = 1 + i delta/2
    """
    D_s = 1.0 - 0.5j * delta
    D_i = 1.0 + 0.5j * delta

    det = D_s * D_i - abs(rho) ** 2

    y_s = D_i / det
    y_i = np.conj(rho) / det
    y_base = 1.0 / D_s
    gain = float(abs(y_s / y_base) ** 2)

    return complex(y_s), complex(y_i), complex(y_base), gain


# =============================================================================
# Solve/result
# =============================================================================

@dataclass
class Exp06Result:
    status: str
    sideband_set: str
    sidebands: int
    ms: list[int]
    signal_m: int
    idler_m: int
    rho: complex
    delta: float
    nt: int
    matrix_shape: tuple[int, int]
    matrix_nnz: int
    solve_runtime_s: float
    total_runtime_s: float
    linear_abs_residual: float
    linear_rel_residual: float
    gamma_hat_p2: complex
    gamma_hat_m2: complex
    gamma_hat_0: complex
    y_signal: complex
    y_idler: complex | None
    y_positive_only: complex
    gain_circuit: float
    gain_circuit_db: float
    gain_positive_only: float
    gain_positive_only_db: float
    gain_analytic_pair: float | None
    gain_analytic_pair_db: float | None
    rel_signal_error_pair: float | None
    exp04_reference_gain_db: float


def run_one(
    sideband_set: str,
    sidebands: int,
    signal_m: int,
    idler_m: int,
    rho: complex,
    delta: float,
    nt: int,
    omega_p: float,
    off_resonance_diag: complex,
    linear_tol: float,
    exp04_reference_gain_db: float,
) -> Exp06Result:
    t_total0 = time.perf_counter()

    ms = sideband_list(
        sideband_set=sideband_set,
        sidebands=sidebands,
        signal_m=signal_m,
        idler_m=idler_m,
    )

    if signal_m not in ms:
        raise ValueError(f"signal_m={signal_m} not in sideband set {ms}")
    if sideband_set != "positive_only" and idler_m not in ms:
        raise ValueError(f"idler_m={idler_m} not in sideband set {ms}")

    grid = SyntheticGammaGrid(nt=nt, omega_p=omega_p)
    max_ell = max(abs(m - q) for m in ms for q in ms)
    khat = build_khat_from_synthetic_gamma(grid=grid, rho=rho, max_ell=max_ell)

    A = assemble_circuit_style_A(
        ms=ms,
        khat=khat,
        signal_m=signal_m,
        idler_m=idler_m,
        delta=delta,
        off_resonance_diag=off_resonance_diag,
    )

    rhs = build_rhs(ms=ms, signal_m=signal_m)

    t0 = time.perf_counter()
    y = spla.spsolve(A, rhs)
    solve_runtime_s = time.perf_counter() - t0

    residual = A @ y - rhs
    linear_abs_residual = float(np.linalg.norm(residual))
    linear_rel_residual = float(linear_abs_residual / max(np.linalg.norm(rhs), 1e-30))

    y_signal = complex(y[ms.index(signal_m)])
    y_idler = complex(y[ms.index(idler_m)]) if idler_m in ms else None
    y_base = baseline_positive_only(
        signal_m=signal_m,
        idler_m=idler_m,
        delta=delta,
        off_resonance_diag=off_resonance_diag,
    )

    gain_circuit = float(abs(y_signal / y_base) ** 2)
    gain_positive_only = 1.0

    gain_analytic_pair = None
    gain_analytic_pair_db = None
    rel_signal_error_pair = None

    if sideband_set == "pair":
        y_s_ref, _y_i_ref, _y_base_ref, gain_ref = analytic_pair_solution(
            rho=rho,
            delta=delta,
        )
        gain_analytic_pair = gain_ref
        gain_analytic_pair_db = safe_db(gain_ref)
        rel_signal_error_pair = float(abs(y_signal - y_s_ref) / max(abs(y_s_ref), 1e-30))

    status = "VALID_CONVERGED"
    if linear_rel_residual > linear_tol:
        status = "FAIL_LINEAR_RESIDUAL"
    if sideband_set == "pair" and rel_signal_error_pair is not None:
        if rel_signal_error_pair > linear_tol:
            status = "FAIL_PAIR_REFERENCE"

    total_runtime_s = time.perf_counter() - t_total0

    return Exp06Result(
        status=status,
        sideband_set=sideband_set,
        sidebands=sidebands,
        ms=ms,
        signal_m=signal_m,
        idler_m=idler_m,
        rho=rho,
        delta=delta,
        nt=nt,
        matrix_shape=A.shape,
        matrix_nnz=A.nnz,
        solve_runtime_s=solve_runtime_s,
        total_runtime_s=total_runtime_s,
        linear_abs_residual=linear_abs_residual,
        linear_rel_residual=linear_rel_residual,
        gamma_hat_p2=grid.gamma_hat(rho, 2),
        gamma_hat_m2=grid.gamma_hat(rho, -2),
        gamma_hat_0=grid.gamma_hat(rho, 0),
        y_signal=y_signal,
        y_idler=y_idler,
        y_positive_only=y_base,
        gain_circuit=gain_circuit,
        gain_circuit_db=safe_db(gain_circuit),
        gain_positive_only=gain_positive_only,
        gain_positive_only_db=safe_db(gain_positive_only),
        gain_analytic_pair=gain_analytic_pair,
        gain_analytic_pair_db=gain_analytic_pair_db,
        rel_signal_error_pair=rel_signal_error_pair,
        exp04_reference_gain_db=exp04_reference_gain_db,
    )


# =============================================================================
# Printing / CSV
# =============================================================================

def print_single(r: Exp06Result) -> None:
    print("=== experiment 06: synthetic Gamma(t) in circuit-style sideband matrix ===")
    print(f"status={r.status}")
    print(f"sideband_set={r.sideband_set}")
    print(f"ms={r.ms}")
    print(f"signal_m={r.signal_m}")
    print(f"idler_m={r.idler_m}")
    print(f"rho_real={r.rho.real:.12e}")
    print(f"rho_imag={r.rho.imag:.12e}")
    print(f"rho_abs={abs(r.rho):.12e}")
    print(f"delta={r.delta:.12e}")
    print(f"nt={r.nt}")
    print(f"matrix_shape={r.matrix_shape}")
    print(f"matrix_nnz={r.matrix_nnz}")
    print(f"solve_runtime_s={r.solve_runtime_s:.6e}")
    print(f"total_runtime_s={r.total_runtime_s:.6e}")
    print(f"linear_abs_residual={r.linear_abs_residual:.12e}")
    print(f"linear_rel_residual={r.linear_rel_residual:.12e}")

    print("\n=== synthetic Gamma Fourier coefficients ===")
    print(f"gamma_hat_plus_2_real={r.gamma_hat_p2.real:.12e}")
    print(f"gamma_hat_plus_2_imag={r.gamma_hat_p2.imag:.12e}")
    print(f"gamma_hat_minus_2_real={r.gamma_hat_m2.real:.12e}")
    print(f"gamma_hat_minus_2_imag={r.gamma_hat_m2.imag:.12e}")
    print(f"gamma_hat_0_real={r.gamma_hat_0.real:.12e}")
    print(f"gamma_hat_0_imag={r.gamma_hat_0.imag:.12e}")
    print("expected_gamma_hat_plus_2=-rho")
    print("expected_gamma_hat_minus_2=-conj(rho)")

    print("\n=== solution ===")
    print(f"y_signal_real={r.y_signal.real:.12e}")
    print(f"y_signal_imag={r.y_signal.imag:.12e}")

    if r.y_idler is not None:
        print(f"y_idler_conj_real={r.y_idler.real:.12e}")
        print(f"y_idler_conj_imag={r.y_idler.imag:.12e}")

    print(f"y_positive_only_real={r.y_positive_only.real:.12e}")
    print(f"y_positive_only_imag={r.y_positive_only.imag:.12e}")

    print("\n=== gain ===")
    print(f"gain_circuit={r.gain_circuit:.12e}")
    print(f"gain_circuit_db={r.gain_circuit_db:.6f}")
    print(f"gain_positive_only={r.gain_positive_only:.12e}")
    print(f"gain_positive_only_db={r.gain_positive_only_db:.6f}")

    if r.gain_analytic_pair is not None:
        print(f"gain_analytic_pair={r.gain_analytic_pair:.12e}")
        print(f"gain_analytic_pair_db={r.gain_analytic_pair_db:.6f}")
        print(f"rel_signal_error_pair={r.rel_signal_error_pair:.12e}")

    print(f"exp04_reference_gain_db={r.exp04_reference_gain_db:.6f}")

    print("\ninterpretation:")
    print("  pair: validates Exp03-style A_mq = D_m delta + Khat_{m-q} against analytic gain.")
    print("  range: includes extra sideband couplings, so it need not match the pair formula.")
    print("  positive_only: removes conjugate idler and should give 0 dB normalized gain.")


def print_sweep_header() -> None:
    print("delta,status,gain_circuit_db,gain_positive_only_db,gain_analytic_pair_db,linear_rel_residual,rel_signal_error_pair")


def print_sweep_row(r: Exp06Result) -> None:
    analytic = "" if r.gain_analytic_pair_db is None else f"{r.gain_analytic_pair_db:.12g}"
    relerr = "" if r.rel_signal_error_pair is None else f"{r.rel_signal_error_pair:.6e}"
    print(
        f"{r.delta:.12g},"
        f"{r.status},"
        f"{r.gain_circuit_db:.12g},"
        f"{r.gain_positive_only_db:.12g},"
        f"{analytic},"
        f"{r.linear_rel_residual:.6e},"
        f"{relerr}"
    )


def write_csv(path: str, rows: list[Exp06Result]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "delta",
                "status",
                "sideband_set",
                "ms",
                "rho_abs",
                "rho_real",
                "rho_imag",
                "gamma_hat_plus_2",
                "gamma_hat_minus_2",
                "gain_circuit",
                "gain_circuit_db",
                "gain_positive_only",
                "gain_positive_only_db",
                "gain_analytic_pair",
                "gain_analytic_pair_db",
                "linear_rel_residual",
                "rel_signal_error_pair",
                "solve_runtime_s",
                "total_runtime_s",
            ]
        )

        for r in rows:
            w.writerow(
                [
                    r.delta,
                    r.status,
                    r.sideband_set,
                    " ".join(str(m) for m in r.ms),
                    abs(r.rho),
                    r.rho.real,
                    r.rho.imag,
                    r.gamma_hat_p2,
                    r.gamma_hat_m2,
                    r.gain_circuit,
                    r.gain_circuit_db,
                    r.gain_positive_only,
                    r.gain_positive_only_db,
                    r.gain_analytic_pair,
                    r.gain_analytic_pair_db,
                    r.linear_rel_residual,
                    r.rel_signal_error_pair,
                    r.solve_runtime_s,
                    r.total_runtime_s,
                ]
            )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")

    p.add_argument("--sideband-set", choices=["pair", "range", "positive_only"], default="pair")
    p.add_argument("--sidebands", type=int, default=4)
    p.add_argument("--signal-m", type=int, default=0)
    p.add_argument("--idler-m", type=int, default=-2)

    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--omega-p", type=float, default=1.0)

    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--rho-phase", type=float, default=0.0)

    # Exp04 reference used to choose rho if --rho is omitted.
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--length", type=float, default=8.0)

    p.add_argument("--delta", type=float, default=0.0)
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

    if args.sidebands < 2 and args.sideband_set == "range":
        raise ValueError("--sidebands must be >= 2 for range mode with idler_m=-2")
    if args.points < 2 and args.sweep_delta:
        raise ValueError("--points must be >= 2 for --sweep-delta")

    T_ref = exp04_transfer(kappa=args.kappa, delta=0.0, length=args.length)
    T_ref_abs = abs(T_ref)
    G_ref_db = safe_db(T_ref_abs**2)

    if args.rho is None:
        rho_abs = rho_from_target_transfer_amplitude(T_ref_abs)
    else:
        rho_abs = args.rho

    rho = rho_abs * np.exp(1j * args.rho_phase)
    off_diag = complex(args.off_resonance_diag_real, args.off_resonance_diag_imag)

    if args.sweep_delta:
        print("=== experiment 06: delta sweep ===")
        print(f"sideband_set={args.sideband_set}")
        print(f"rho_abs={abs(rho):.12e}")
        print(f"exp04_reference_gain_db={G_ref_db:.6f}")
        print_sweep_header()

        rows: list[Exp06Result] = []
        for delta in np.linspace(-args.delta_max, args.delta_max, args.points):
            r = run_one(
                sideband_set=args.sideband_set,
                sidebands=args.sidebands,
                signal_m=args.signal_m,
                idler_m=args.idler_m,
                rho=rho,
                delta=float(delta),
                nt=args.nt,
                omega_p=args.omega_p,
                off_resonance_diag=off_diag,
                linear_tol=args.linear_tol,
                exp04_reference_gain_db=G_ref_db,
            )
            rows.append(r)
            print_sweep_row(r)

        print(f"\nall_valid={all(r.status == 'VALID_CONVERGED' for r in rows)}")
        print(f"max_gain_circuit_db={max(r.gain_circuit_db for r in rows):.6f}")
        print(f"min_gain_circuit_db={min(r.gain_circuit_db for r in rows):.6f}")

        if args.csv:
            write_csv(args.csv, rows)
            print(f"wrote_csv={args.csv}")

    else:
        r = run_one(
            sideband_set=args.sideband_set,
            sidebands=args.sidebands,
            signal_m=args.signal_m,
            idler_m=args.idler_m,
            rho=rho,
            delta=args.delta,
            nt=args.nt,
            omega_p=args.omega_p,
            off_resonance_diag=off_diag,
            linear_tol=args.linear_tol,
            exp04_reference_gain_db=G_ref_db,
        )
        print_single(r)


if __name__ == "__main__":
    main()
