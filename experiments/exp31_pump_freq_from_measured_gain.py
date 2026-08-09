"""Find the pump frequency that reproduces the measured Themis small-signal gain.

The Jan28 cube pumps at 7.256 GHz and, at its lowest signal power, shows

    G0(5.296 GHz) =  7.18 dB
    G0(6.540 GHz) = 12.18 dB
    G0(7.052 GHz) = 13.16 dB

The model does not amplify at that pump setting, so the operating point has to
be located rather than assumed.  For each candidate pump frequency this solves
the pump once and then evaluates the *linear* Floquet gain at the three measured
signal frequencies, which is cheap because the sidebands enter as a
factor-once linear system.  The reported figure of merit is the RMS mismatch
against the three measured values, so a match has to hold at all three
frequencies simultaneously rather than at one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from twpa_solver.core import load_circuit
from twpa_solver.pump import (
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import resolve_pump_basis
from twpa_solver.pump.problem import FullPumpProblem
import twpa_solver.signal as exp09

MEASURED: dict[float, float] = {
    5.296: 7.18,
    6.540: 12.18,
    7.052: 13.16,
}
CIRCUIT_DIR = "designs/ipm_2c_fixed"
PUMP_CURRENT_A = 7.231074707853736e-06
PUMP_MODE_COUNT = 10
PUMP_NT = 40
SIDEBANDS = 10
Z0_OHM = 50.0

SETTINGS = NewtonKrylovSettings(
    newton_tol=1e-10,
    max_newton=25,
    gmres_rtol=1e-8,
    gmres_atol=0.0,
    gmres_restart=20,
    gmres_maxiter=40,
    min_alpha=1.0 / 1024.0,
    preconditioner="real_coupled",
    compute_time_residual=False,
    verbose=False,
    continuation_predictor="none",
    jvp_mode="aft",
)


def gains_at(
    circuit,
    pump_state: np.ndarray,
    pump_basis,
    omega_p: float,
    signal_ghz: tuple[float, ...],
) -> dict[float, float]:
    """Linear Floquet gain-vs-off at each signal frequency for one pump."""
    pump = exp09.PumpSolution(
        X=pump_state,
        omega_p=omega_p,
        pump_freq_ghz=omega_p / (2.0 * math.pi * 1e9),
        harmonics=int(max(pump_basis.modes)),
        nt_original=PUMP_NT,
        metadata=pump_basis.to_metadata(),
        modes=list(pump_basis.modes),
        basis=pump_basis,
    )
    ms = exp09.sideband_list(SIDEBANDS)
    max_ell = max(abs(m - q) for m in ms for q in ms)
    gamma_hat = exp09.compute_gamma_hat(
        circuit=circuit, pump=pump, max_ell=max_ell, gamma_nt=PUMP_NT,
        dc_branch_flux=None,
    )
    khat = exp09.build_khat(Bphi=circuit.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
    gamma_off = circuit.Ic / circuit.phi0
    khat_off_0 = (
        circuit.Bphi @ sp.diags(gamma_off, offsets=0, format="csr") @ circuit.Bphi.T
    ).astype(np.complex128).tocsr()
    out: dict[float, float] = {}
    for fs in signal_ghz:
        result = exp09.solve_gain_one_schur(
            circuit=circuit, khat=khat, khat_off_0=khat_off_0,
            omega_p=omega_p, signal_ghz=fs, sidebands=SIDEBANDS,
            signal_m=0, idler_m=-2,
            source_index=circuit.port_to_index[1],
            out_index=circuit.port_to_index[2],
            source_current_a=1.0, source_port=1, out_port=2, z0_ohm=Z0_OHM,
            loss_model="current_complex_c", linear_solver="superlu",
        )
        out[fs] = float(result.gain_vs_off_db)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", default=CIRCUIT_DIR)
    parser.add_argument("--pump-freq-min-ghz", type=float, default=6.8)
    parser.add_argument("--pump-freq-max-ghz", type=float, default=8.4)
    parser.add_argument("--n-pump-freq", type=int, default=41)
    parser.add_argument(
        "--pump-current-a", type=float, action="append",
        help="repeatable; defaults to the production current",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp31_pump_freq_scan")
    )
    args = parser.parse_args()

    currents = args.pump_current_a or [PUMP_CURRENT_A]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circuit = load_circuit(args.circuit_dir)
    signal_ghz = tuple(sorted(MEASURED))
    targets = np.array([MEASURED[f] for f in signal_ghz])
    freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.n_pump_freq)

    rows: list[dict[str, object]] = []
    csv_path = args.output_dir / "pump_freq_scan.csv"
    fields = (
        ["pump_freq_ghz", "pump_current_a", "pump_converged", "rms_mismatch_db"]
        + [f"gain_{f:.3f}ghz_db" for f in signal_ghz]
        + [f"target_{f:.3f}ghz_db" for f in signal_ghz]
    )
    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()

    header = "  ".join(f"{f:.3f} GHz" for f in signal_ghz)
    print(f"targets: {'  '.join(f'{t:.2f}' for t in targets)} dB at {header}\n")
    print(f'{"fp GHz":>8} {"Ip A":>11} {"conv":>5} '
          + " ".join(f"{f'G({f:.3f})':>11}" for f in signal_ghz)
          + f'{"rms dB":>9}')

    for current in currents:
        for fp in freqs:
            omega_p = 2.0 * math.pi * float(fp) * 1e9
            basis = resolve_pump_basis(
                policy="positive_odd_jc", omega_p=omega_p, harmonics=None,
                mode_count=PUMP_MODE_COUNT, explicit_modes=None,
                design_meta=circuit.metadata,
            )
            problem = FullPumpProblem(
                C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
                branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
                grid=HarmonicGrid(np.asarray(basis.modes), nt=PUMP_NT, omega=omega_p),
                pump_node_index=circuit.port_to_index[4],
                pump_current_a=float(current),
            )
            t0 = time.perf_counter()
            try:
                state, reports = HarmonicNewtonKrylovSolver(
                    SETTINGS
                ).solve_continuation(problem, continuation_steps=4)
                converged = bool(reports and reports[-1].converged)
            except Exception as exc:  # noqa: BLE001 - scan must not abort
                print(f"{fp:>8.3f} {current:>11.4e} {'ERR':>5}  {type(exc).__name__}")
                continue
            gains = gains_at(circuit, state, basis, omega_p, signal_ghz)
            values = np.array([gains[f] for f in signal_ghz])
            rms = float(np.sqrt(np.mean((values - targets) ** 2)))
            row: dict[str, object] = {
                "pump_freq_ghz": float(fp),
                "pump_current_a": float(current),
                "pump_converged": converged,
                "rms_mismatch_db": rms,
            }
            for f in signal_ghz:
                row[f"gain_{f:.3f}ghz_db"] = gains[f]
                row[f"target_{f:.3f}ghz_db"] = MEASURED[f]
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(f"{fp:>8.3f} {current:>11.4e} {str(converged):>5} "
                  + " ".join(f"{gains[f]:>11.4f}" for f in signal_ghz)
                  + f"{rms:>9.3f}   [{time.perf_counter() - t0:.1f}s]")

    handle.close()
    ok = [r for r in rows if r["pump_converged"]]
    best = min(ok, key=lambda r: r["rms_mismatch_db"]) if ok else None
    (args.output_dir / "pump_freq_scan_summary.json").write_text(
        json.dumps(
            {
                "measured_pump_freq_ghz": 7.256,
                "targets_db": {f"{f:.3f}": MEASURED[f] for f in signal_ghz},
                "n_points": len(rows),
                "n_converged": len(ok),
                "best": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if best is not None:
        print(f"\nbest: fp={best['pump_freq_ghz']:.4f} GHz  "
              f"Ip={best['pump_current_a']:.4e} A  rms={best['rms_mismatch_db']:.3f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
