"""Benchmark matrix-free / Schur pump-solver backends on the high-power fold.

Compares, on a warm-started power sweep at a fixed pump frequency, the legacy
full-Jacobian backends against the Schur-reduced backends, and -- crucially --
runs exp09 after every converged pump solve so each backend is judged on gain
drift, not just runtime. A backend is only acceptable if its exp09 gain matches
the full ``real_coupled`` baseline to < 0.01 dB.

Backends (``--backends``):
    full_mean_tangent     legacy block-diagonal preconditioner, full nodes.
    full_real_coupled     legacy exact full-Jacobian LU (the fold baseline).
    full_adaptive         power-adaptive: mean_tangent below the crossover,
                          fresh real_coupled at/above it (current best rule).
    schur_cpu_mt          Schur reduction, assembled sparse S_k, retained
                          mean_tangent preconditioner.
    schur_cpu_rc          Schur reduction, assembled sparse S_k, retained
                          real_coupled preconditioner.
    schur_mf_jfnk         Schur reduction, matrix-free S_k (eliminated back-sub),
                          retained mean_tangent preconditioner.

Outputs (under ``--outdir``, default outputs/pump_solver_backends):
    baseline_tail.csv             full_* backends.
    schur_matrixfree_tail.csv     schur_* backends.
    backends_tail_all.csv         every backend.
    *_summary.json                fold-point summaries + acceptance verdicts.

Example:
    python experiments/benchmark_exp08_pump_schur_matrixfree.py \
        --pump-freq-ghz 7.0 --power-min-dbm -34 --power-max-dbm -22 --n-power 25
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import exp08_full_ipm_pump_solve as exp08
import exp09_full_ipm_gain_from_pump as exp09
import pump_basis
from pump_solvers.schur_operators import SchurReducedProblem, build_schur_problem

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_CROSSOVER_DBM = -22.7  # mean_tangent below, fresh real_coupled at/above
BASELINE_BACKEND = "full_real_coupled"


def dbm_to_peak_current_a(power_dbm: float, *, attenuation_db: float, z0_ohm: float) -> float:
    source_dbm = float(power_dbm) - float(attenuation_db)
    return math.sqrt(2.0 * 1.0e-3 * 10.0 ** (source_dbm / 10.0) / float(z0_ohm))


@dataclass
class BackendSpec:
    name: str
    kind: str  # "full" or "schur"
    apply_mode: str  # "assembled" | "matrix_free" (schur only)
    preconditioner: str  # or "adaptive"


BACKENDS: dict[str, BackendSpec] = {
    "full_mean_tangent": BackendSpec("full_mean_tangent", "full", "", "mean_tangent"),
    "full_real_coupled": BackendSpec("full_real_coupled", "full", "", "real_coupled"),
    "full_adaptive": BackendSpec("full_adaptive", "full", "", "adaptive"),
    "schur_cpu_mt": BackendSpec("schur_cpu_mt", "schur", "assembled", "mean_tangent"),
    "schur_cpu_rc": BackendSpec("schur_cpu_rc", "schur", "assembled", "real_coupled"),
    "schur_mf_jfnk": BackendSpec("schur_mf_jfnk", "schur", "matrix_free", "mean_tangent"),
}


def make_settings(precond: str, args: argparse.Namespace) -> exp08.NewtonKrylovSettings:
    return exp08.NewtonKrylovSettings(
        newton_tol=args.newton_tol, max_newton=args.max_newton, gmres_rtol=1e-7,
        gmres_atol=0.0, gmres_restart=60, gmres_maxiter=args.gmres_maxiter,
        min_alpha=1.0 / 1024.0, preconditioner=precond, compute_time_residual=False,
        verbose=False, continuation_predictor="none", jvp_mode="aft",
        stall_ratio=0.8, stall_patience=4,
    )


class GainEvaluator:
    """exp09 gain pipeline, mirroring exp10's InProcessEngine._gain."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ipm09 = exp09.load_ipm(str(args.ipm_dir))
        self.source_idx = self.ipm09.port_to_index[args.source_port]
        self.out_idx = self.ipm09.port_to_index[args.out_port]

    def gain(self, pump_dir: Path, freq_ghz: float):
        a = self.args
        pump = exp09.load_pump(pump_dir, fallback_pump_freq_ghz=freq_ghz)
        ms = exp09.sideband_list(a.sidebands)
        max_ell = max(abs(m - q) for m in ms for q in ms)
        gamma_hat = exp09.compute_gamma_hat(
            ipm=self.ipm09, pump=pump, max_ell=max_ell, gamma_nt=a.gamma_nt,
            dc_branch_flux=None,
        )
        khat = exp09.build_khat(Bphi=self.ipm09.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
        gamma_off = self.ipm09.Ic / self.ipm09.phi0
        khat_off_0 = (
            self.ipm09.Bphi @ sp.diags(gamma_off, offsets=0, format="csr")
            @ self.ipm09.Bphi.T
        ).astype(np.complex128).tocsr()
        return exp09.solve_gain_one(
            ipm=self.ipm09, khat=khat, khat_off_0=khat_off_0, omega_p=pump.omega_p,
            signal_ghz=a.signal_ghz, sidebands=a.sidebands, signal_m=0, idler_m=-2,
            source_index=self.source_idx, out_index=self.out_idx,
            source_current_a=1.0, source_port=a.source_port, out_port=a.out_port,
            z0_ohm=a.z0_ohm, loss_model="current_complex_c",
        )


class ProblemFactory:
    """Builds full + (cached) Schur problems for a fixed pump frequency."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ipm = exp08.load_ipm(args.ipm_dir)
        self.branch = exp08.JosephsonBranchArray(Ic=self.ipm.Ic, phi0=self.ipm.phi0)
        self.omega = 2.0 * math.pi * args.pump_freq_ghz * 1e9
        self.basis = pump_basis.resolve_pump_basis(
            policy="positive_odd_jc", omega_p=self.omega, harmonics=3,
            mode_count=args.pump_mode_count, explicit_modes=None,
            design_meta=self.ipm.summary,
        )
        self.grid = exp08.HarmonicGrid(modes=self.basis.k, nt=args.nt, omega=self.omega)
        self.ports = list(self.ipm.port_to_index.values())
        self.pump_idx = self.ipm.port_to_index[args.pump_port]
        self._partition = None
        self._part_meta: dict[str, Any] = {}

    def full(self, current_a: float) -> exp08.FullIPMPumpProblem:
        return exp08.FullIPMPumpProblem(
            C=self.ipm.C, G=self.ipm.G, K=self.ipm.K, Bphi=self.ipm.Bphi,
            branch=self.branch, grid=self.grid, pump_node_index=self.pump_idx,
            pump_current_a=current_a, source_mode=self.basis.source_mode,
        )

    def schur(self, current_a: float, apply_mode: str) -> SchurReducedProblem:
        full = self.full(current_a)
        if self._partition is None:
            s = build_schur_problem(full, self.ports, linear_apply_mode=apply_mode)
            self._partition = s.part
            self._part_meta = {
                "retained_nodes": s.part.m, "eliminated_nodes": s.part.p,
                "total_nodes": s.part.n, "retained_fraction": s.part.retained_fraction,
                "schur_assemble_time_s": s.part.schur_assemble_time_s,
                "schur_nnz": s.part.schur_nnz,
                "dee_factor_time_s": s.part.factor_time_s,
            }
            return s
        return SchurReducedProblem(
            full=full, partition=self._partition, linear_apply_mode=apply_mode
        )

    def metadata(self) -> dict[str, Any]:
        return dict(self._part_meta)


def solve_backend_point(
    factory: ProblemFactory, spec: BackendSpec, power_dbm: float, current_a: float,
    warm: np.ndarray | None, args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray, list]:
    """Solve one pump point with a backend. Returns (metrics, warm_next, full_X, reports)."""
    precond = spec.preconditioner
    if precond == "adaptive":
        precond = "real_coupled" if power_dbm >= ADAPTIVE_CROSSOVER_DBM else "mean_tangent"
    settings = make_settings(precond, args)
    solver = exp08.HarmonicNewtonKrylovSolver(settings)

    t0 = time.perf_counter()
    if spec.kind == "full":
        prob = factory.full(current_a)
        if warm is None:
            X, reports = solver.solve_continuation(prob, continuation_steps=args.continuation_steps)
        else:
            X, reports = solver.solve_direct(prob, warm)
        full_X = X
        warm_next = X
    else:
        prob = factory.schur(current_a, spec.apply_mode)
        if warm is None:
            Xn, reports = solver.solve_continuation(prob, continuation_steps=args.continuation_steps)
        else:
            Xn, reports = solver.solve_direct(prob, warm)
        full_X = prob.reconstruct_full(Xn)
        warm_next = Xn
    pump_runtime = time.perf_counter() - t0

    converged = bool(reports and reports[-1].converged
                     and abs(reports[-1].source_scale - 1.0) < 1e-12)
    # Time residual on the full reconstructed solution (exact, reported once).
    full_prob = prob.full if spec.kind == "schur" else prob
    Rt = full_prob.time_residual(full_X, 1.0)
    St = full_prob.source_time(1.0)
    time_rel = float(np.linalg.norm(Rt.ravel()) / math.sqrt(Rt.size)) / max(
        float(np.linalg.norm(St.ravel()) / max(math.sqrt(St.size), 1.0)), 1e-30)
    summary = exp08.summarize_solution(full_prob, full_X)

    metrics = {
        "backend": spec.name, "preconditioner": precond, "power_dbm": power_dbm,
        "status": "VALID_CONVERGED" if converged else "FAIL",
        "pump_runtime_s": pump_runtime,
        "newton_total": int(sum(r.newton_iterations for r in reports)),
        "gmres_total": int(sum(r.gmres_iterations_total for r in reports)),
        "accepted_steps": int(sum(1 for r in reports if r.converged)),
        "factor_runtime_s": float(sum(r.factor_runtime_s for r in reports)),
        "coeff_rel": float(reports[-1].coeff_rel) if reports else None,
        "time_rel": time_rel,
        "branch_i_max": float(summary.get("branch_i_max_abs")),
        "branch_i_rms": float(summary.get("branch_i_rms")),
        "failure_reason": reports[-1].failure_reason if reports else "no reports",
    }
    return metrics, (warm_next if converged else None), full_X, reports


def run(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    powers = np.linspace(args.power_min_dbm, args.power_max_dbm, args.n_power)
    factory = ProblemFactory(args)
    gains = GainEvaluator(args)
    pump_scratch = args.outdir / "_pump_scratch"

    selected = [BACKENDS[b] for b in args.backends]
    all_rows: list[dict[str, Any]] = []
    # Baseline gains keyed by power, filled from BASELINE_BACKEND.
    baseline_gain: dict[float, dict[str, float]] = {}

    for spec in selected:
        print(f"\n=== backend {spec.name} (precond={spec.preconditioner}, "
              f"apply={spec.apply_mode or 'n/a'}) ===", flush=True)
        factory._partition = None  # rebuild Schur partition per backend run
        warm: np.ndarray | None = None
        for power_dbm in powers:
            p = float(power_dbm)
            current = dbm_to_peak_current_a(
                p, attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm)
            injected = current * args.pump_current_jc_scale
            metrics, warm, full_X, reports = solve_backend_point(
                factory, spec, p, injected, warm, args)

            # exp09 gain on the reconstructed full solution.
            metrics.update({k: None for k in (
                "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db",
                "linear_rel_residual", "gain_status")})
            if metrics["status"] == "VALID_CONVERGED":
                pdir = pump_scratch / spec.name / f"p_{p:+.2f}"
                exp08.write_results(
                    pdir, full_X, reports,
                    exp08.summarize_solution(factory.full(injected), full_X),
                    {**factory.basis.to_metadata(), "pump_freq_ghz": args.pump_freq_ghz,
                     "nt": args.nt, "omega_p": factory.omega, "pump_current_a": injected})
                g = gains.gain(pdir, args.pump_freq_ghz)
                if g is not None and g.status == "VALID_SOLVED":
                    metrics["gain_status"] = "VALID_SOLVED"
                    metrics["gain_db"] = float(g.gain_db)
                    metrics["gain_vs_off_db"] = float(g.gain_vs_off_db)
                    metrics["gain_vs_pumpdiag_db"] = float(g.gain_vs_pumpdiag_db)
                    metrics["linear_rel_residual"] = float(g.linear_rel_residual)
                else:
                    metrics["gain_status"] = "ERROR"

            if spec.name == BASELINE_BACKEND and metrics["gain_db"] is not None:
                baseline_gain[round(p, 4)] = {
                    "gain_db": metrics["gain_db"],
                    "gain_vs_off_db": metrics["gain_vs_off_db"],
                    "gain_vs_pumpdiag_db": metrics["gain_vs_pumpdiag_db"],
                }
            all_rows.append(metrics)
            print(f"  {p:7.2f} dBm  {metrics['status']:>16}  "
                  f"newton={metrics['newton_total']:3d} gmres={metrics['gmres_total']:4d} "
                  f"factor={metrics['factor_runtime_s']:5.2f}s pump={metrics['pump_runtime_s']:6.3f}s "
                  f"gain={_fmt(metrics['gain_db'])}", flush=True)

    # Gain drift vs baseline.
    for row in all_rows:
        base = baseline_gain.get(round(row["power_dbm"], 4))
        for key, dkey in (("gain_db", "gain_drift_db"),
                          ("gain_vs_off_db", "gain_vs_off_drift_db"),
                          ("gain_vs_pumpdiag_db", "gain_vs_pumpdiag_drift_db")):
            if base and row.get(key) is not None:
                row[dkey] = abs(row[key] - base[key])
            else:
                row[dkey] = None

    write_outputs(all_rows, factory.metadata(), args)


def _fmt(x: float | None) -> str:
    return f"{x:7.3f}" if x is not None else "     --"


FIELDS = [
    "backend", "preconditioner", "power_dbm", "status", "gain_status",
    "pump_runtime_s", "newton_total", "gmres_total", "accepted_steps",
    "factor_runtime_s", "coeff_rel", "time_rel", "branch_i_max", "branch_i_rms",
    "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db", "linear_rel_residual",
    "gain_drift_db", "gain_vs_off_drift_db", "gain_vs_pumpdiag_drift_db",
    "failure_reason",
]


def write_outputs(
    rows: list[dict[str, Any]], part_meta: dict[str, Any], args: argparse.Namespace
) -> None:
    def dump(path: Path, subset: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(subset)
        print(f"wrote {path}", flush=True)

    full_rows = [r for r in rows if r["backend"].startswith("full")]
    schur_rows = [r for r in rows if r["backend"].startswith("schur")]
    dump(args.outdir / "backends_tail_all.csv", rows)
    if full_rows:
        dump(args.outdir / "baseline_tail.csv", full_rows)
    if schur_rows:
        dump(args.outdir / "schur_matrixfree_tail.csv", schur_rows)

    fold_powers = [p for p in (-23.0, -22.5, -22.0)]
    summary = {
        "pump_freq_ghz": args.pump_freq_ghz, "signal_ghz": args.signal_ghz,
        "baseline_backend": BASELINE_BACKEND, "partition": part_meta,
        "fold_points": {},
    }
    for p in fold_powers:
        entry = {}
        for r in rows:
            if abs(r["power_dbm"] - p) < 1e-6:
                entry[r["backend"]] = {
                    "status": r["status"], "pump_runtime_s": r["pump_runtime_s"],
                    "newton_total": r["newton_total"], "gmres_total": r["gmres_total"],
                    "gain_db": r["gain_db"], "gain_drift_db": r["gain_drift_db"],
                }
        summary["fold_points"][f"{p:g}"] = entry

    # Acceptance verdicts: schur backends vs baseline at the fold.
    verdicts = {}
    for r in rows:
        if not r["backend"].startswith("schur"):
            continue
        ok = (r["status"] == "VALID_CONVERGED" and r["gain_status"] == "VALID_SOLVED"
              and (r["gain_drift_db"] or 9.9) < 0.01
              and (r["gain_vs_off_drift_db"] or 9.9) < 0.01
              and (r["gain_vs_pumpdiag_drift_db"] or 9.9) < 0.01)
        verdicts.setdefault(r["backend"], {"accepted_points": 0, "total_points": 0,
                                           "max_gain_drift_db": 0.0})
        v = verdicts[r["backend"]]
        v["total_points"] += 1
        v["accepted_points"] += int(ok)
        if r["gain_drift_db"] is not None:
            v["max_gain_drift_db"] = max(v["max_gain_drift_db"], r["gain_drift_db"])
    summary["acceptance"] = verdicts

    sjson = args.outdir / "schur_matrixfree_summary.json"
    bjson = args.outdir / "baseline_tail_summary.json"
    for path in (sjson, bjson):
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ipm-dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "pump_solver_backends")
    p.add_argument("--pump-freq-ghz", type=float, default=7.0)
    p.add_argument("--power-min-dbm", type=float, default=-34.0)
    p.add_argument("--power-max-dbm", type=float, default=-22.0)
    p.add_argument("--n-power", type=int, default=25)
    p.add_argument("--backends", nargs="+", default=list(BACKENDS),
                   choices=list(BACKENDS))
    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    p.add_argument("--pump-mode-count", type=int, default=10)
    p.add_argument("--nt", type=int, default=40)
    p.add_argument("--sidebands", type=int, default=10)
    p.add_argument("--gamma-nt", type=int, default=96)
    p.add_argument("--signal-ghz", type=float, default=6.0)
    p.add_argument("--attenuation-db", type=float, default=35.0)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    p.add_argument("--pump-current-jc-scale", type=float, default=2.0)
    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--max-newton", type=int, default=16)
    p.add_argument("--gmres-maxiter", type=int, default=80)
    p.add_argument("--continuation-steps", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
