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
    classify_floquet_resonance,
    local_minima,
    require_explicit_loss_model,
    refine_resonances,
    sweep_sigma_min,
)
from twpa_solver.stability import track_multiplier_branches  # noqa: E402
from twpa_solver.signal.floquet import assemble_khat_conversion_base  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit-dir", required=True)
    p.add_argument("--pump-dir", required=True, nargs="+",
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
    p.add_argument(
        "--loss-model", required=True,
        help="Explicit stability convention; never inferred from circuit metadata.",
    )
    p.add_argument("--span-start-ghz", type=float, default=0.05,
                   help="Low edge of the sweep (avoid the omega=0 boundary "
                        "of the Brillouin zone).")
    p.add_argument("--span-end-fraction", type=float, default=0.99,
                   help="High edge of the sweep as a fraction of "
                        "pump_freq_ghz (avoid the omega=omega_p boundary).")
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument(
        "--mode-spacing-mhz", type=float, default=241.7,
        help="Measured mode-comb spacing used by the scan-density guard.",
    )
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
    p.add_argument(
        "--refine-bifurcations",
        action="store_true",
        help="Also refine explicit Floquet-zone candidates and classify their "
        "multipliers. Fractions are relative to the pump frequency.",
    )
    p.add_argument(
        "--bifurcation-fractions",
        default="0.0,0.5",
        help="Comma-separated Floquet-zone frequency guesses; default checks "
        "+1 and -1 multiplier candidates.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def recommended_scan_points(pump_freq_ghz: float, mode_spacing_mhz: float) -> int:
    """Return the recommended full-zone point count at the measured spacing."""
    if pump_freq_ghz <= 0.0 or mode_spacing_mhz <= 0.0:
        raise ValueError("pump frequency and mode spacing must be positive")
    old_points_per_mode = 2000.0 / (7.9e3 / 85.0)
    return int(math.ceil(
        (pump_freq_ghz * 1.0e3 / mode_spacing_mhz) * old_points_per_mode
    ))


def enforce_scan_density_guard(args: argparse.Namespace, pump_freq_ghz: float) -> None:
    recommended = recommended_scan_points(pump_freq_ghz, args.mode_spacing_mhz)
    minimum = math.ceil(recommended / 4.0)
    if args.n_points < minimum:
        raise ValueError(
            f"--n-points={args.n_points} under-resolves the {args.mode_spacing_mhz:g} "
            f"MHz comb: use at least {minimum} points (recommended {recommended})"
        )


def track_complex_resonance_branches(
    sweeps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order refined roots continuously across the supplied power sequence."""
    previous: np.ndarray | None = None
    for sweep in sweeps:
        roots = sweep.get("complex_resonances", [])
        if not roots:
            continue
        values = np.asarray([
            complex(root["floquet"]["multiplier_real"], root["floquet"]["multiplier_imag"])
            for root in roots
        ])
        tracked = values if previous is None else track_multiplier_branches(previous, values)
        order = [int(np.argmin(np.abs(values - value))) for value in tracked]
        sweep["complex_resonances"] = [roots[index] for index in order]
        for branch_index, root in enumerate(sweep["complex_resonances"]):
            root["tracked_branch_index"] = branch_index
        sweep["max_abs_lambda"] = float(np.max(np.abs(tracked)))
        previous = tracked
    return sweeps


def _load_pump_and_khat(circuit, pump_dir: Path, fallback_freq_ghz: float, sidebands: int, gamma_nt: int):
    pump = exp09.load_pump(pump_dir, fallback_pump_freq_ghz=fallback_freq_ghz)
    ms = exp09.sideband_list(sidebands)
    max_ell = max(abs(m - q) for m in ms for q in ms)
    dc_value = pump.metadata.get("dc_branch_flux")
    if dc_value is None:
        dc_value = pump.metadata.get("dc_branch_flux_wb")
    if dc_value is None:
        dc_branch_flux = None
    else:
        dc_branch_flux = np.asarray(dc_value, dtype=float).reshape(-1)
        if dc_branch_flux.size == 1:
            dc_branch_flux = np.full(circuit.branch_count, float(dc_branch_flux[0]))
        if dc_branch_flux.size != circuit.branch_count:
            raise ValueError(
                "pump report DC flux length does not match the circuit branch count"
            )
    gamma_hat = exp09.compute_gamma_hat(
        circuit=circuit, pump=pump, max_ell=max_ell, gamma_nt=gamma_nt,
        dc_branch_flux=dc_branch_flux,
    )
    khat = exp09.build_khat(Bphi=circuit.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
    return pump, ms, khat


def _run_sweep(circuit, pump_dir: Path, fallback_freq_ghz: float, args: argparse.Namespace) -> dict[str, Any]:
    pump, ms, khat = _load_pump_and_khat(
        circuit, pump_dir, fallback_freq_ghz, args.sidebands, args.gamma_nt,
    )
    # This block is independent of signal frequency; reuse it across the scan.
    khat_base = assemble_khat_conversion_base(circuit, khat, ms)
    pump_freq_ghz = float(pump.omega_p / (2.0 * math.pi * 1e9))
    enforce_scan_density_guard(args, pump_freq_ghz)
    span_end_ghz = pump_freq_ghz * args.span_end_fraction
    grid = np.linspace(args.span_start_ghz, span_end_ghz, args.n_points).tolist()

    t0 = time.perf_counter()
    estimates = sweep_sigma_min(
        circuit=circuit, khat=khat, omega_p=pump.omega_p, signal_ghz_grid=grid,
        ms=ms, loss_model=args.loss_model, iters=args.iters, seed=args.seed,
        khat_base=khat_base,
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

    def serialize_resonance(seed: float, resonance: Any) -> dict[str, Any]:
        floquet = classify_floquet_resonance(resonance, pump.omega_p)
        return {
            "seed_signal_ghz": seed,
            "signal_ghz_real": resonance.signal_ghz.real,
            "signal_ghz_imag": resonance.signal_ghz.imag,
            "growth_rate_per_s": resonance.growth_rate_per_s,
            "unstable": resonance.growth_rate_per_s > 0.0,
            "converged": resonance.converged,
            "iterations": resonance.iterations,
            "residual": resonance.residual,
            "floquet": {
                "multiplier_real": floquet.multiplier.real,
                "multiplier_imag": floquet.multiplier.imag,
                "magnitude": floquet.magnitude,
                "phase_rad": floquet.phase_rad,
                "zone_frequency_ghz": floquet.zone_frequency_ghz,
                "kind": floquet.kind,
                "near_unit_circle": floquet.near_unit_circle,
            },
        }

    if args.refine_complex and resonances:
        t1 = time.perf_counter()
        candidates = [r["signal_ghz"] for r in resonances]
        refined = refine_resonances(
            circuit=circuit, khat=khat, omega_p=pump.omega_p, ms=ms,
            candidates_ghz=candidates, loss_model=args.loss_model,
            max_iters=args.refine_max_iters, tol=args.refine_tol,
            khat_base=khat_base,
        )
        result["complex_resonances"] = [
            serialize_resonance(seed, resonance)
            for seed, resonance in zip(candidates, refined)
        ]
        result["refine_runtime_s"] = time.perf_counter() - t1

    if args.refine_bifurcations:
        try:
            fractions = [
                float(token.strip())
                for token in args.bifurcation_fractions.split(",")
                if token.strip()
            ]
        except ValueError as exc:
            raise ValueError("invalid --bifurcation-fractions") from exc
        if not fractions or any(fraction < 0.0 or fraction >= 1.0 for fraction in fractions):
            raise ValueError("bifurcation fractions must lie in [0, 1)")
        candidates = [fraction * pump_freq_ghz for fraction in fractions]
        t2 = time.perf_counter()
        refined = refine_resonances(
            circuit=circuit,
            khat=khat,
            omega_p=pump.omega_p,
            ms=ms,
            candidates_ghz=candidates,
            loss_model=args.loss_model,
            max_iters=args.refine_max_iters,
            tol=args.refine_tol,
            khat_base=khat_base,
        )
        result["bifurcation_resonances"] = [
            serialize_resonance(seed, resonance)
            for seed, resonance in zip(candidates, refined)
        ]
        result["bifurcation_refine_runtime_s"] = time.perf_counter() - t2

    return result


def run_namespace(args: argparse.Namespace) -> dict[str, Any]:
    args.loss_model = require_explicit_loss_model(args.loss_model)
    circuit = load_circuit(args.circuit_dir)

    if args.loss_model in NON_ANALYTIC_LOSS_MODELS:
        if args.refine_complex or args.refine_bifurcations:
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
            floquet = cr.get("floquet", {})
            print(f"  [{label}] seed={cr['seed_signal_ghz']:.6f} GHz -> "
                  f"omega/(2pi*1e9)={cr['signal_ghz_real']:.6f}"
                  f"{cr['signal_ghz_imag']:+.6f}j GHz "
                  f"growth_rate_per_s={cr['growth_rate_per_s']:.4e} "
                  f"multiplier={floquet.get('magnitude', float('nan')):.6f}"
                  f"angle={floquet.get('phase_rad', float('nan')):.4f} "
                  f"kind={floquet.get('kind', 'UNKNOWN')} "
                  f"{verdict} ({conv}, {cr['iterations']} iters, "
                  f"residual={cr['residual']:.2e})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets: list[dict[str, Any]] = []
    for setting_index, pump_dir in enumerate(args.pump_dir):
        targets.append(
            _run_sweep(circuit, Path(pump_dir), args.pump_freq_ghz, args)
        )
        setting_path = out_path.with_name(
            f"{out_path.stem}.setting_{setting_index:02d}.json"
        )
        setting_path.write_text(
            json.dumps(
                {
                    "setting_index": setting_index,
                    "pump_dir": str(pump_dir),
                    "target": targets[-1],
                },
                indent=2,
            )
        )
    track_complex_resonance_branches(targets)

    for setting_index, target in enumerate(targets):
        setting_path = out_path.with_name(
            f"{out_path.stem}.setting_{setting_index:02d}.json"
        )
        setting_path.write_text(
            json.dumps(
                {
                    "setting_index": setting_index,
                    "pump_dir": str(args.pump_dir[setting_index]),
                    "target": target,
                },
                indent=2,
            )
        )

    baseline = None
    if args.baseline_pump_dir:
        baseline = _run_sweep(circuit, Path(args.baseline_pump_dir), args.pump_freq_ghz, args)

    out = {
        "metadata": {
            "sidebands": args.sidebands,
            "loss_model": args.loss_model,
            "span_start_ghz": args.span_start_ghz,
            "span_end_fraction": args.span_end_fraction,
            "n_points": args.n_points,
            "mode_spacing_mhz": args.mode_spacing_mhz,
            "recommended_n_points": recommended_scan_points(
                float(targets[0]["pump_freq_ghz"]), args.mode_spacing_mhz
            ),
            "iters": args.iters,
            "seed": args.seed,
            "non_analytic_loss_model_warning": args.loss_model in NON_ANALYTIC_LOSS_MODELS,
            "refine_bifurcations": args.refine_bifurcations,
            "bifurcation_fractions": args.bifurcation_fractions,
        },
        "target": targets[0] if len(targets) == 1 else targets,
        "baseline": baseline,
    }
    out_path.write_text(json.dumps(out, indent=2))
    for power_index, target in enumerate(targets):
        print(f"[target {power_index}] pump_freq_ghz={target['pump_freq_ghz']:.6f} "
              f"n_points={len(target['signal_ghz'])} runtime_s={target['runtime_s']:.2f}")
        for resonance in target["resonances"][:5]:
            print(f"  candidate resonance: signal_ghz={resonance['signal_ghz']:.6f} "
                  f"sigma_min={resonance['sigma_min']:.6e} "
                  f"conv_ratio={resonance['convergence_ratio']:.4f}")
        _print_complex_resonances(f"target {power_index}", target)
    if baseline is not None:
        print(f"[baseline] pump_freq_ghz={baseline['pump_freq_ghz']:.6f} "
              f"runtime_s={baseline['runtime_s']:.2f}")
        for resonance in baseline["resonances"][:5]:
            print(f"  candidate resonance: signal_ghz={resonance['signal_ghz']:.6f} "
                  f"sigma_min={resonance['sigma_min']:.6e} "
                  f"conv_ratio={resonance['convergence_ratio']:.4f}")
        _print_complex_resonances("baseline", baseline)
        if targets[0]["resonances"] and baseline["resonances"]:
            target_min = targets[0]["resonances"][0]["sigma_min"]
            baseline_min = baseline["resonances"][0]["sigma_min"]
            print(f"  deepest interior resonance sigma_min: target={target_min:.6e} "
                  f"baseline={baseline_min:.6e} "
                  f"ratio(target/baseline)={target_min / baseline_min:.4e}")
        else:
            print("  no interior local minima found in one or both sweeps "
                  "(sigma_min may be monotonic over this span)")
    print(f"wrote {out_path}")
    return out


def main(argv: list[str] | None = None) -> int:
    run_namespace(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
