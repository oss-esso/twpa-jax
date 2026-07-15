"""Floquet stability sweep: Tier 1 (real-omega sigma_min) + Tier 2 (complex-omega refinement).

``A(omega_s)`` (twpa_solver.signal.floquet.assemble_conversion_matrix) is the
harmonic-balance Floquet/Hill determinant around a converged pump state: it is
singular exactly at the system's Floquet exponents. Existing gain-map code
only ever evaluates it at a real signal tone within ~1 GHz of the pump
detuning (run_gain_map.py's --signal-spectrum). This script widens that to a
dense real-frequency scan over the full first Brillouin zone
(0, pump_freq_ghz) and tracks sigma_min(A) as a resonance/near-singularity
proxy -- no new solver math, same per-point cost as one existing gain solve.

Tier 1 alone is a proxy, not a stability verdict: a dip in sigma_min flags a
candidate Floquet resonance near that real frequency, but only a complex-omega
root search can say whether it corresponds to Im(omega) < 0 (growth). Pass
``--refine-complex`` to run Tier 2 (twpa_solver.signal.stability.refine_resonances)
on each Tier 1 candidate: it tracks the eigenvalue of A(omega) nearest zero
into the complex omega plane via a secant search and reports
``growth_rate_per_s = -Im(omega)`` -- positive means an actual growing mode.

Caveats addressed here:
  - Convention/trust sanity check: if --baseline-pump-dir is given (a
    definitely-stable, low-power operating point), the same sweep (and, with
    --refine-complex, the same Tier 2 refinement) runs there too, so results
    can be compared against a known-stable baseline before trusting them.
  - Loss-model analyticity: conductance_abs_omega / conductance_abs_omega_
    opposite / complex_c_sign_omega are not analytic in omega (abs() or a
    sign() branch). Tier 1 is real-omega only, so it is unaffected and only
    warns; --refine-complex hard-refuses to run against these models (Tier 2
    needs analytic continuation into the complex plane to mean anything).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twpa_solver.core import load_circuit  # noqa: E402
import twpa_solver.signal as exp09  # noqa: E402
from twpa_solver.signal.stability import (  # noqa: E402
    NON_ANALYTIC_LOSS_MODELS,
    local_minima,
    refine_resonances,
    sweep_sigma_min,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit-dir", required=True)
    p.add_argument("--pump-dir", required=True,
                   help="Directory with pump_solution.npz + pump_report.json "
                        "for the operating point under investigation.")
    p.add_argument("--baseline-pump-dir", default=None,
                   help="Optional known-stable, low-power pump dir for a "
                        "sanity comparison (caveat: validates the sweep "
                        "against a point with no expected resonance).")
    p.add_argument("--pump-freq-ghz", type=float, default=None,
                   help="Fallback pump frequency if not recoverable from the "
                        "pump report metadata.")
    p.add_argument("--sidebands", type=int, default=6)
    p.add_argument("--gamma-nt", type=int, default=4096)
    p.add_argument("--loss-model", default="current_complex_c")
    p.add_argument("--span-start-ghz", type=float, default=0.05,
                   help="Low edge of the sweep (avoid the omega=0 boundary "
                        "of the Brillouin zone).")
    p.add_argument("--span-end-fraction", type=float, default=0.99,
                   help="High edge of the sweep as a fraction of "
                        "pump_freq_ghz (avoid the omega=omega_p boundary).")
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--iters", type=int, default=8,
                   help="Inverse-iteration steps per sweep point.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=8,
                   help="Number of deepest local minima (candidate "
                        "resonances) to report.")
    p.add_argument("--refine-complex", action="store_true",
                   help="Tier 2: refine each Tier 1 candidate resonance into "
                        "the complex-omega plane (eigenvalue-nearest-zero "
                        "secant search) to get an actual growth/decay "
                        "verdict instead of just a real-omega proxy. "
                        "Refuses to run against NON_ANALYTIC_LOSS_MODELS.")
    p.add_argument("--refine-max-iters", type=int, default=30)
    p.add_argument("--refine-tol", type=float, default=1e-9)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def _load_pump_and_khat(circuit, pump_dir: Path, fallback_freq_ghz: float, sidebands: int, gamma_nt: int):
    pump = exp09.load_pump(pump_dir, fallback_pump_freq_ghz=fallback_freq_ghz)
    ms = exp09.sideband_list(sidebands)
    max_ell = max(abs(m - q) for m in ms for q in ms)
    gamma_hat = exp09.compute_gamma_hat(
        circuit=circuit, pump=pump, max_ell=max_ell, gamma_nt=gamma_nt,
        dc_branch_flux=None,
    )
    khat = exp09.build_khat(Bphi=circuit.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
    return pump, ms, khat


def _run_sweep(circuit, pump_dir: Path, fallback_freq_ghz: float, args: argparse.Namespace) -> dict[str, Any]:
    pump, ms, khat = _load_pump_and_khat(
        circuit, pump_dir, fallback_freq_ghz, args.sidebands, args.gamma_nt,
    )
    pump_freq_ghz = float(pump.omega_p / (2.0 * math.pi * 1e9))
    span_end_ghz = pump_freq_ghz * args.span_end_fraction
    grid = np.linspace(args.span_start_ghz, span_end_ghz, args.n_points).tolist()

    t0 = time.perf_counter()
    estimates = sweep_sigma_min(
        circuit=circuit, khat=khat, omega_p=pump.omega_p, signal_ghz_grid=grid,
        ms=ms, loss_model=args.loss_model, iters=args.iters, seed=args.seed,
    )
    runtime_s = time.perf_counter() - t0

    sigma_min = [float(e.sigma_min) for e in estimates]
    ratio = [float(e.convergence_ratio) for e in estimates]
    min_idx = local_minima(sigma_min, k=args.top_k)
    resonances = [
        {"signal_ghz": grid[i], "sigma_min": sigma_min[i], "convergence_ratio": ratio[i]}
        for i in min_idx
    ]

    result = {
        "pump_dir": str(pump_dir),
        "pump_freq_ghz": pump_freq_ghz,
        "signal_ghz": grid,
        "sigma_min": sigma_min,
        "convergence_ratio": ratio,
        "resonances": resonances,
        "runtime_s": runtime_s,
    }

    if args.refine_complex and resonances:
        t1 = time.perf_counter()
        candidates = [r["signal_ghz"] for r in resonances]
        refined = refine_resonances(
            circuit=circuit, khat=khat, omega_p=pump.omega_p, ms=ms,
            candidates_ghz=candidates, loss_model=args.loss_model,
            max_iters=args.refine_max_iters, tol=args.refine_tol,
        )
        result["complex_resonances"] = [
            {
                "seed_signal_ghz": seed,
                "signal_ghz_real": r.signal_ghz.real,
                "signal_ghz_imag": r.signal_ghz.imag,
                "growth_rate_per_s": r.growth_rate_per_s,
                "unstable": r.growth_rate_per_s > 0.0,
                "converged": r.converged,
                "iterations": r.iterations,
                "residual": r.residual,
            }
            for seed, r in zip(candidates, refined)
        ]
        result["refine_runtime_s"] = time.perf_counter() - t1

    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    circuit = load_circuit(args.circuit_dir)

    if args.loss_model in NON_ANALYTIC_LOSS_MODELS:
        if args.refine_complex:
            raise SystemExit(
                f"--refine-complex requires an analytic D(omega); "
                f"loss_model={args.loss_model!r} is in NON_ANALYTIC_LOSS_MODELS. "
                f"Rerun with a different --loss-model (e.g. current_complex_c)."
            )
        print(f"WARNING: loss_model={args.loss_model!r} is not analytic in "
              f"omega. This real-omega sweep is unaffected, but a "
              f"complex-omega (Tier 2/3) extension must not be run against "
              f"this loss model without change.")

    def _print_complex_resonances(label: str, sweep: dict[str, Any]) -> None:
        for cr in sweep.get("complex_resonances", []):
            verdict = "UNSTABLE (growing)" if cr["unstable"] else "stable (decaying)"
            conv = "converged" if cr["converged"] else "NOT converged"
            print(f"  [{label}] seed={cr['seed_signal_ghz']:.6f} GHz -> "
                  f"omega/(2pi*1e9)={cr['signal_ghz_real']:.6f}"
                  f"{cr['signal_ghz_imag']:+.6f}j GHz "
                  f"growth_rate_per_s={cr['growth_rate_per_s']:.4e} "
                  f"{verdict} ({conv}, {cr['iterations']} iters, "
                  f"residual={cr['residual']:.2e})")

    target = _run_sweep(circuit, Path(args.pump_dir), args.pump_freq_ghz, args)
    print(f"[target] pump_freq_ghz={target['pump_freq_ghz']:.6f} "
          f"n_points={len(target['signal_ghz'])} runtime_s={target['runtime_s']:.2f}")
    for r in target["resonances"][:5]:
        print(f"  candidate resonance: signal_ghz={r['signal_ghz']:.6f} "
              f"sigma_min={r['sigma_min']:.6e} conv_ratio={r['convergence_ratio']:.4f}")
    _print_complex_resonances("target", target)

    baseline = None
    if args.baseline_pump_dir:
        baseline = _run_sweep(circuit, Path(args.baseline_pump_dir), args.pump_freq_ghz, args)
        print(f"[baseline] pump_freq_ghz={baseline['pump_freq_ghz']:.6f} "
              f"runtime_s={baseline['runtime_s']:.2f}")
        for r in baseline["resonances"][:5]:
            print(f"  candidate resonance: signal_ghz={r['signal_ghz']:.6f} "
                  f"sigma_min={r['sigma_min']:.6e} conv_ratio={r['convergence_ratio']:.4f}")
        _print_complex_resonances("baseline", baseline)
        # Compare genuine (interior, bracketed) local minima only -- the raw
        # array min can sit on an edge point (e.g. near omega=0), which is not
        # a resonance and is excluded from `resonances` for that reason.
        if target["resonances"] and baseline["resonances"]:
            target_min = target["resonances"][0]["sigma_min"]
            baseline_min = baseline["resonances"][0]["sigma_min"]
            print(f"  deepest interior resonance sigma_min: target={target_min:.6e} "
                  f"baseline={baseline_min:.6e} "
                  f"ratio(target/baseline)={target_min / baseline_min:.4e}")
        else:
            print("  no interior local minima found in one or both sweeps "
                  "(sigma_min may be monotonic over this span)")

    out = {
        "metadata": {
            "sidebands": args.sidebands,
            "loss_model": args.loss_model,
            "span_start_ghz": args.span_start_ghz,
            "span_end_fraction": args.span_end_fraction,
            "n_points": args.n_points,
            "iters": args.iters,
            "seed": args.seed,
            "non_analytic_loss_model_warning": args.loss_model in NON_ANALYTIC_LOSS_MODELS,
        },
        "target": target,
        "baseline": baseline,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
