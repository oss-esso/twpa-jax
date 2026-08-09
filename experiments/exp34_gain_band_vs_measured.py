"""Compare the model's small-signal gain band against the measured G0(f).

The exp33 probe found the model running 1.8-2.6 dB low in G0 across 6.3-6.8 GHz
while matching at 7.052 GHz, i.e. its gain band is narrower than the device's.
That is a *linear* Floquet question -- no finite-signal solve is involved -- so
it is cheap: one pump solve at the exp31 operating point, then a factor-once
sideband solve per signal frequency.

The measured reference is the lowest-signal-power row of the Jan28 cube,
median-filtered the same way exp30/exp32 take G0.  It pumps at 7.256 GHz while
the model's matched operating point is 7.100 GHz, so the two bands are expected
to sit at slightly different centres; the question here is their *width* and
*depth*, not their alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from twpa_solver.core import load_circuit
from twpa_solver.pump import (
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import resolve_pump_basis
from twpa_solver.pump.problem import FullPumpProblem

from exp31_pump_freq_from_measured_gain import (  # noqa: E402 - sibling experiment
    PUMP_CURRENT_A,
    PUMP_MODE_COUNT,
    PUMP_NT,
    SETTINGS,
    gains_at,
)

CIRCUIT_DIR = "designs/ipm_2c_fixed"
PUMP_FREQ_GHZ = 7.100
CUBE = Path(
    "docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK"
    "/105C5_7.256GHz.npy"
)
MEAS_PUMP_GHZ = 7.256
PUMP_EXCLUSION_GHZ = 0.15


def measured_g0() -> tuple[np.ndarray, np.ndarray]:
    """Measured small-signal gain versus signal frequency, pump notch masked."""
    data = np.load(CUBE, allow_pickle=True).item()
    freq = np.asarray(data["Frequency"], dtype=float) / 1e9
    response = np.asarray(data["Response"], dtype=float)
    g0 = np.median(response[:10, :], axis=0)
    g0[np.abs(freq - MEAS_PUMP_GHZ) < PUMP_EXCLUSION_GHZ] = np.nan
    return freq, g0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", default=CIRCUIT_DIR)
    parser.add_argument("--pump-freq-ghz", type=float, default=PUMP_FREQ_GHZ)
    parser.add_argument("--pump-current-a", type=float, default=PUMP_CURRENT_A)
    parser.add_argument("--start-ghz", type=float, default=4.5)
    parser.add_argument("--stop-ghz", type=float, default=9.5)
    parser.add_argument("--points", type=int, default=81)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp34_gain_band")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    circuit = load_circuit(args.circuit_dir)
    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9
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
        pump_current_a=float(args.pump_current_a),
    )
    t0 = time.perf_counter()
    state, reports = HarmonicNewtonKrylovSolver(SETTINGS).solve_continuation(
        problem, continuation_steps=4
    )
    if not (reports and reports[-1].converged):
        raise SystemExit(
            f"pump solve did not converge at {args.pump_freq_ghz} GHz; "
            "gain sweep would be meaningless"
        )
    print(
        f"pump solved in {time.perf_counter() - t0:.1f} s "
        f"(coeff_rel={reports[-1].coeff_rel:.3e})",
        flush=True,
    )

    # The pump tone itself is not a valid signal frequency for this solve.
    sweep = np.linspace(args.start_ghz, args.stop_ghz, args.points)
    sweep = sweep[np.abs(sweep - args.pump_freq_ghz) > 1e-6]

    freq_meas, g0_meas = measured_g0()
    rows: list[dict[str, object]] = []
    csv_path = args.output_dir / "gain_band.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["signal_ghz", "model_gain_db", "measured_g0_db"]
        )
        writer.writeheader()
        print(f'{"fs GHz":>8} {"model dB":>10} {"meas dB":>9} {"delta":>8}')
        for fs in sweep:
            t1 = time.perf_counter()
            gain = gains_at(circuit, state, basis, omega_p, (float(fs),))[float(fs)]
            meas = float(np.interp(fs, freq_meas, g0_meas))
            row = {
                "signal_ghz": float(fs),
                "model_gain_db": gain,
                "measured_g0_db": meas,
            }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                f"{fs:>8.3f} {gain:>10.3f} {meas:>9.3f} {gain - meas:>8.3f}"
                f"   {time.perf_counter() - t1:.1f}s",
                flush=True,
            )

    model_ghz = np.array([r["signal_ghz"] for r in rows])
    model_db = np.array([r["model_gain_db"] for r in rows])

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.plot(freq_meas, g0_meas, lw=2.0, color="tab:orange",
            label=f"measured $G_0$ (pump {MEAS_PUMP_GHZ} GHz)")
    ax.plot(model_ghz, model_db, lw=1.8, color="tab:blue", marker="o", ms=3,
            label=f"model Floquet gain (pump {args.pump_freq_ghz} GHz)")
    ax.axvline(MEAS_PUMP_GHZ, color="tab:orange", ls=":", lw=1.2)
    ax.axvline(args.pump_freq_ghz, color="tab:blue", ls=":", lw=1.2)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("signal frequency (GHz)")
    ax.set_ylabel("small-signal gain (dB)")
    ax.set_title(
        f"Gain band: model ({Path(args.circuit_dir).name}, "
        f"{args.pump_current_a:.4e} A) vs Themis 105C5"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "gain_band.png", dpi=140)
    plt.close(fig)

    finite = np.isfinite(model_db)
    peak = int(np.argmax(np.where(finite, model_db, -np.inf)))
    meas_finite = np.isfinite(g0_meas)
    summary = {
        "circuit_dir": args.circuit_dir,
        "model_pump_ghz": args.pump_freq_ghz,
        "measured_pump_ghz": MEAS_PUMP_GHZ,
        "pump_current_a": args.pump_current_a,
        "model_peak_gain_db": float(model_db[peak]),
        "model_peak_ghz": float(model_ghz[peak]),
        "model_bandwidth_above_3db_ghz": float(
            np.count_nonzero(model_db > 3.0)
            * (model_ghz[1] - model_ghz[0])
        ),
        "measured_peak_gain_db": float(np.nanmax(g0_meas)),
        "measured_peak_ghz": float(freq_meas[np.nanargmax(g0_meas)]),
        "measured_bandwidth_above_3db_ghz": float(
            np.count_nonzero(g0_meas[meas_finite] > 3.0)
            * (freq_meas[1] - freq_meas[0])
        ),
    }
    (args.output_dir / "gain_band_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
