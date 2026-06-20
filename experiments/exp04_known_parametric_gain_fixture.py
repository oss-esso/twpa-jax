# experiments/exp04_known_parametric_gain_fixture.py
"""
Experiment 04: known positive-gain parametric fixture.

Purpose:
    Validate the sideband/conjugate-idler logic independently of the toy ladder.

    Exp03 proved the end-to-end machinery:
        pump -> gamma(t) -> conversion matrix -> linear solve -> transfer

    But Exp03 used a passive toy ladder, so loss/no gain is not diagnostic.

    This experiment uses the textbook two-mode undepleted-pump parametric amplifier:

        d/dz [ a_s      ] = [ -i Δ/2      κ      ] [ a_s      ]
             [ a_i^*    ]   [  κ^*       i Δ/2  ] [ a_i^*    ]

    For κ > |Δ|/2, the solution has exponential/hyperbolic gain:

        G_s = | cosh(gL) - i Δ/(2g) sinh(gL) |^2
        g   = sqrt(|κ|^2 - (Δ/2)^2)

    depending on sign convention; here the matrix convention gives:

        T = cosh(gL) + (-i Δ/2)/g sinh(gL)

    The point is not the sign of the phase term; the power gain is the same.

What this validates:
    1. A conjugate-idler basis produces gain.
    2. A positive-only/no-idler basis does not produce gain.
    3. Numerical matrix exponential matches the analytic formula.
    4. Gain decreases when phase mismatch Δ grows.

This is intentionally tiny and fast. It is a mathematical fixture, not a circuit fixture.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la


# =============================================================================
# Core math
# =============================================================================

@dataclass
class ParametricGainResult:
    kappa: complex
    delta: float
    length: float
    g: complex
    transfer_analytic: complex
    transfer_expm: complex
    gain_analytic: float
    gain_expm: float
    gain_analytic_db: float
    gain_expm_db: float
    gain_bad_positive_only: float
    gain_bad_positive_only_db: float
    abs_transfer_error: float
    rel_transfer_error: float
    status: str


def safe_db(power_gain: float) -> float:
    return 10.0 * math.log10(max(float(power_gain), 1e-300))


def complex_sqrt(z: complex) -> complex:
    return complex(np.sqrt(np.complex128(z)))


def generator_matrix(kappa: complex, delta: float) -> np.ndarray:
    """
    Coupled-mode generator:

        d/dz [a_s, a_i^*]^T = M [a_s, a_i^*]^T

    with

        M = [[-iΔ/2, κ],
             [κ*,    iΔ/2]]

    Then M^2 = (|κ|^2 - (Δ/2)^2) I.
    """
    return np.array(
        [
            [-0.5j * delta, kappa],
            [np.conj(kappa), 0.5j * delta],
        ],
        dtype=np.complex128,
    )


def analytic_transfer(kappa: complex, delta: float, length: float) -> tuple[complex, complex]:
    """
    Analytic signal transfer for input a_s(0)=1, a_i^*(0)=0.

        U = exp(M L)
        T = U_00

    Since M^2 = g^2 I:

        exp(M L) = I cosh(gL) + (M/g) sinh(gL)

    so

        T = cosh(gL) + M_00/g sinh(gL)
          = cosh(gL) + (-iΔ/2)/g sinh(gL)

    Handles g≈0 by the limit sinh(gL)/g -> L.
    """
    g = complex_sqrt(abs(kappa) ** 2 - (0.5 * delta) ** 2)
    z = g * length

    if abs(g) < 1e-14:
        transfer = 1.0 + (-0.5j * delta) * length
    else:
        transfer = np.cosh(z) + ((-0.5j * delta) / g) * np.sinh(z)

    return complex(transfer), g


def expm_transfer(kappa: complex, delta: float, length: float) -> complex:
    """
    Numerical transfer from matrix exponential.
    """
    M = generator_matrix(kappa, delta)
    U = la.expm(M * length)
    return complex(U[0, 0])


def bad_positive_only_transfer(delta: float, length: float) -> complex:
    """
    Deliberately wrong positive-only/no-idler model.

    If the conjugate idler is omitted, the signal equation is only:

        d a_s / dz = -i Δ/2 a_s

    so |a_s(L)|^2 = 1.

    This should never show parametric gain.
    """
    return complex(np.exp(-0.5j * delta * length))


def solve_known_gain_fixture(
    kappa: complex,
    delta: float,
    length: float,
    tolerance: float,
) -> ParametricGainResult:
    transfer_a, g = analytic_transfer(kappa, delta, length)
    transfer_e = expm_transfer(kappa, delta, length)
    transfer_bad = bad_positive_only_transfer(delta, length)

    gain_a = float(abs(transfer_a) ** 2)
    gain_e = float(abs(transfer_e) ** 2)
    gain_bad = float(abs(transfer_bad) ** 2)

    abs_err = abs(transfer_e - transfer_a)
    rel_err = abs_err / max(abs(transfer_a), 1e-30)

    status = "VALID_CONVERGED" if rel_err < tolerance else "FAIL_TRANSFER_MISMATCH"

    return ParametricGainResult(
        kappa=kappa,
        delta=delta,
        length=length,
        g=g,
        transfer_analytic=transfer_a,
        transfer_expm=transfer_e,
        gain_analytic=gain_a,
        gain_expm=gain_e,
        gain_analytic_db=safe_db(gain_a),
        gain_expm_db=safe_db(gain_e),
        gain_bad_positive_only=gain_bad,
        gain_bad_positive_only_db=safe_db(gain_bad),
        abs_transfer_error=float(abs_err),
        rel_transfer_error=float(rel_err),
        status=status,
    )


# =============================================================================
# Sweeps
# =============================================================================

def sweep_delta(
    kappa: complex,
    length: float,
    delta_max: float,
    points: int,
    tolerance: float,
) -> list[ParametricGainResult]:
    deltas = np.linspace(-delta_max, delta_max, points)
    return [
        solve_known_gain_fixture(
            kappa=kappa,
            delta=float(delta),
            length=length,
            tolerance=tolerance,
        )
        for delta in deltas
    ]


def write_csv(path: str, rows: list[ParametricGainResult]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "delta",
                "kappa_abs",
                "kappa_real",
                "kappa_imag",
                "length",
                "g_real",
                "g_imag",
                "gain_expm_db",
                "gain_analytic_db",
                "gain_bad_positive_only_db",
                "gain_expm",
                "gain_analytic",
                "gain_bad_positive_only",
                "rel_transfer_error",
                "status",
            ]
        )

        for r in rows:
            w.writerow(
                [
                    r.delta,
                    abs(r.kappa),
                    r.kappa.real,
                    r.kappa.imag,
                    r.length,
                    r.g.real,
                    r.g.imag,
                    r.gain_expm_db,
                    r.gain_analytic_db,
                    r.gain_bad_positive_only_db,
                    r.gain_expm,
                    r.gain_analytic,
                    r.gain_bad_positive_only,
                    r.rel_transfer_error,
                    r.status,
                ]
            )


# =============================================================================
# Printing
# =============================================================================

def print_single_result(r: ParametricGainResult, runtime_s: float) -> None:
    print("=== experiment 04: known parametric gain fixture ===")
    print(f"status={r.status}")
    print(f"kappa={r.kappa.real:.12e}+{r.kappa.imag:.12e}j")
    print(f"kappa_abs={abs(r.kappa):.12e}")
    print(f"delta={r.delta:.12e}")
    print(f"length={r.length:.12e}")
    print(f"g={r.g.real:.12e}+{r.g.imag:.12e}j")
    print(f"runtime_s={runtime_s:.6f}")

    print("\n=== transfer comparison ===")
    print(f"transfer_analytic_real={r.transfer_analytic.real:.12e}")
    print(f"transfer_analytic_imag={r.transfer_analytic.imag:.12e}")
    print(f"transfer_expm_real={r.transfer_expm.real:.12e}")
    print(f"transfer_expm_imag={r.transfer_expm.imag:.12e}")
    print(f"abs_transfer_error={r.abs_transfer_error:.12e}")
    print(f"rel_transfer_error={r.rel_transfer_error:.12e}")

    print("\n=== gain ===")
    print(f"gain_analytic={r.gain_analytic:.12e}")
    print(f"gain_analytic_db={r.gain_analytic_db:.6f}")
    print(f"gain_expm={r.gain_expm:.12e}")
    print(f"gain_expm_db={r.gain_expm_db:.6f}")

    print("\n=== deliberately wrong basis check ===")
    print(f"gain_bad_positive_only={r.gain_bad_positive_only:.12e}")
    print(f"gain_bad_positive_only_db={r.gain_bad_positive_only_db:.6f}")
    print("interpretation=positive-only/no-conjugate-idler model cannot amplify")


def print_sweep(rows: list[ParametricGainResult]) -> None:
    print("=== delta sweep ===")
    print(
        "delta,status,gain_expm_db,gain_analytic_db,"
        "gain_bad_positive_only_db,rel_transfer_error,g_real,g_imag"
    )
    for r in rows:
        print(
            f"{r.delta:.12g},"
            f"{r.status},"
            f"{r.gain_expm_db:.9g},"
            f"{r.gain_analytic_db:.9g},"
            f"{r.gain_bad_positive_only_db:.9g},"
            f"{r.rel_transfer_error:.6e},"
            f"{r.g.real:.12e},"
            f"{r.g.imag:.12e}"
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(conflict_handler="resolve")

    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--kappa-phase", type=float, default=0.0)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--length", type=float, default=8.0)
    p.add_argument("--tolerance", type=float, default=1e-12)

    p.add_argument("--sweep-delta", action="store_true")
    p.add_argument("--delta-max", type=float, default=1.0)
    p.add_argument("--points", type=int, default=41)
    p.add_argument("--csv", type=str, default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.points < 2 and args.sweep_delta:
        raise ValueError("--points must be >= 2 for --sweep-delta")

    kappa = args.kappa * np.exp(1j * args.kappa_phase)

    t0 = time.perf_counter()

    if args.sweep_delta:
        rows = sweep_delta(
            kappa=kappa,
            length=args.length,
            delta_max=args.delta_max,
            points=args.points,
            tolerance=args.tolerance,
        )
        runtime_s = time.perf_counter() - t0

        print_sweep(rows)
        print(f"\nruntime_s={runtime_s:.6f}")
        print(f"all_valid={all(r.status == 'VALID_CONVERGED' for r in rows)}")
        print(f"max_gain_db={max(r.gain_expm_db for r in rows):.6f}")
        print(f"min_gain_db={min(r.gain_expm_db for r in rows):.6f}")

        if args.csv:
            write_csv(args.csv, rows)
            print(f"wrote_csv={args.csv}")

    else:
        result = solve_known_gain_fixture(
            kappa=kappa,
            delta=args.delta,
            length=args.length,
            tolerance=args.tolerance,
        )
        runtime_s = time.perf_counter() - t0
        print_single_result(result, runtime_s)


if __name__ == "__main__":
    main()