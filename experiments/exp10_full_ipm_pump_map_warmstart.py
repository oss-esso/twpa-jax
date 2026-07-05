"""Experiment 10: warm-started IPM pump/gain map with a cold-vs-warm gate.

This orchestrates exp08 (pump harmonic-balance solve) and exp09 (linearized
gain) over a pump-power x pump-frequency grid, comparing two traversal
strategies:

``cold``
    Every point is solved from scratch with the legacy path (zero initial guess
    + fixed 20-step continuation). This is the trusted reference.

``warmstart``
    Each frequency column is traversed in increasing pump power. The first point
    of a column is seeded with the ``linear_phasor`` guess and solved with
    adaptive continuation; every subsequent (higher-power) point warm-starts from
    the previous converged pump solution via ``--promote-from-pump-dir`` (a single
    full-scale Newton solve, no continuation).

``both``
    Runs the cold pass then the warm pass and emits a PASS/FAIL **gate**: warm
    start is accepted only if every point converged, the per-point gain agrees
    with the cold reference within ``--gate-gain-db``, and the warm pass is
    faster in total pump runtime. This is the validation experiment.

For a large warm-only map, ``--gate-spotcheck N`` recomputes ``N`` points cold
after the warm pass and folds their gain drift into the gate, so the big run is
still guarded without paying for a full cold map.

Pump current is derived from delivered power with the JC-style convention
``I_peak = sqrt(2 * P_W / Z0)`` (after an optional ``--attenuation-db``).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
EXP08 = "experiments/exp08_full_ipm_pump_solve.py"
EXP09 = "experiments/exp09_full_ipm_gain_from_pump.py"

sys.path.insert(0, str(ROOT / "experiments"))
import exp08_full_ipm_pump_solve as exp08  # noqa: E402
import exp09_full_ipm_gain_from_pump as exp09  # noqa: E402
import pump_basis  # noqa: E402
from pump_solvers.schur_operators import SchurReducedProblem, build_schur_problem  # noqa: E402


# =============================================================================
# Units / helpers
# =============================================================================

def dbm_to_peak_current_a(power_dbm: float, *, attenuation_db: float, z0_ohm: float) -> float:
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    source_dbm = float(power_dbm) - float(attenuation_db)
    power_w = 1.0e-3 * 10.0 ** (source_dbm / 10.0)
    return math.sqrt(2.0 * power_w / float(z0_ohm))


def signal_ghz_for(pump_freq_ghz: float, args: argparse.Namespace) -> float:
    """Readout signal frequency for a map cell.

    Physically the map sweeps the pump frequency, so the signal must track it at a
    fixed detuning ws = wp - detuning (default 100 MHz). An explicit --signal-ghz
    overrides this with a fixed absolute signal.
    """
    if getattr(args, "signal_ghz", None) is not None:
        return float(args.signal_ghz)
    return float(pump_freq_ghz) - float(args.signal_detuning_mhz) / 1000.0


def spectrum_offsets_mhz(args: argparse.Namespace) -> list[float]:
    """Signal offsets (MHz, relative to fp) for the per-cell spectrum mode.

    Symmetric ladder around the pump: +/- start, +/- (start+step), ... e.g.
    start=100, step=250, count=5 -> +/-100, +/-350, +/-600, +/-850, +/-1100.
    The -detuning trailing point (default -100) is a member, so the spectrum
    contains the map's headline signal.
    """
    pos = [args.signal_offset_start_mhz + i * args.signal_offset_step_mhz
           for i in range(args.signal_offset_count_per_side)]
    return [float(-x) for x in reversed(pos)] + [float(x) for x in pos]


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slug_float(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def point_name(index: int, power_dbm: float, pump_freq_ghz: float) -> str:
    return (
        f"point_{index:04d}_p_{slug_float(power_dbm)}dbm_"
        f"fp_{slug_float(pump_freq_ghz)}ghz"
    )


def run_command(
    cmd: list[str], *, stdout_path: Path, stderr_path: Path, timeout_s: float
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True,
                timeout=timeout_s, check=False,
            )
        return int(proc.returncode), time.perf_counter() - start
    except subprocess.TimeoutExpired:
        with stderr_path.open("a", encoding="utf-8") as err:
            err.write(f"\nTIMEOUT after {timeout_s:.3f} s\n")
        return 124, time.perf_counter() - start


# =============================================================================
# Metric extraction
# =============================================================================

def pump_metrics(report: dict[str, Any] | None) -> dict[str, float | None]:
    if report is None:
        return {k: None for k in (
            "pump_runtime_s", "pump_factor_runtime_s",
            "pump_preconditioner_assembly_runtime_s",
            "pump_preconditioner_numeric_factor_runtime_s", "pump_coeff_rel",
            "pump_time_rel", "pump_newton_total", "pump_branch_current_max",
        )}
    reports = report.get("reports", [])
    final = reports[-1] if reports else {}
    summ = report.get("solution_summary", {})
    return {
        "pump_runtime_s": sum(finite_or_none(r.get("runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_factor_runtime_s": sum(finite_or_none(r.get("factor_runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_preconditioner_assembly_runtime_s": sum(finite_or_none(r.get("preconditioner_assembly_runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_preconditioner_numeric_factor_runtime_s": sum(finite_or_none(r.get("preconditioner_numeric_factor_runtime_s")) or 0.0 for r in reports) if reports else None,
        "pump_coeff_rel": finite_or_none(final.get("coeff_rel")),
        "pump_time_rel": finite_or_none(final.get("time_rel")),
        "pump_newton_total": int(sum(int(r.get("newton_iterations", 0)) for r in reports)),
        "pump_branch_current_max": finite_or_none(summ.get("branch_i_max_abs")),
    }


def gain_metrics(report: dict[str, Any] | None) -> dict[str, float | None]:
    if report is None:
        return {k: None for k in (
            "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db",
            "signal_ghz", "linear_rel_residual", "gain_total_runtime_s",
            "gain_gamma_hat_runtime_s", "gain_khat_build_runtime_s",
            "gain_khat_off_runtime_s", "gain_matrix_assemble_runtime_s", "gain_factor_solve_runtime_s",
            "gain_baseline_off_runtime_s", "gain_baseline_pumpdiag_runtime_s",
        )}
    results = report.get("results", [])
    valid = [r for r in results if finite_or_none(r.get("gain_db")) is not None]
    best = max(valid, key=lambda r: float(r["gain_db"])) if valid else {}
    return {
        "gain_db": finite_or_none(best.get("gain_db")),
        "gain_vs_off_db": finite_or_none(best.get("gain_vs_off_db")),
        "gain_vs_pumpdiag_db": finite_or_none(best.get("gain_vs_pumpdiag_db")),
        "signal_ghz": finite_or_none(best.get("signal_ghz")),
        "linear_rel_residual": finite_or_none(best.get("linear_rel_residual")),
        "gain_total_runtime_s": finite_or_none(report.get("metadata", {}).get("total_runtime_s")),
        "gain_gamma_hat_runtime_s": finite_or_none(report.get("metadata", {}).get("gamma_hat_runtime_s")),
        "gain_khat_build_runtime_s": finite_or_none(report.get("metadata", {}).get("khat_build_runtime_s")),
        "gain_khat_off_runtime_s": finite_or_none(report.get("metadata", {}).get("khat_off_build_runtime_s")),
        "gain_matrix_assemble_runtime_s": finite_or_none(best.get("assemble_runtime_s")),
        "gain_factor_solve_runtime_s": finite_or_none(best.get("factor_solve_runtime_s")),
        "gain_baseline_off_runtime_s": finite_or_none(best.get("baseline_off_runtime_s")),
        "gain_baseline_pumpdiag_runtime_s": finite_or_none(best.get("baseline_pumpdiag_runtime_s")),
    }


def pump_status(report: dict[str, Any] | None, returncode: int) -> str:
    if returncode != 0:
        return "ERROR"
    if report is None:
        return "MISSING"
    return str(report.get("final_status", "UNKNOWN"))


def gain_status(report: dict[str, Any] | None, returncode: int) -> str:
    if returncode != 0:
        return "ERROR"
    if report is None:
        return "MISSING"
    results = report.get("results", [])
    if results and all(r.get("status") == "VALID_SOLVED" for r in results):
        return "VALID_SOLVED"
    return "UNKNOWN"


# =============================================================================
# Single point execution
# =============================================================================

@dataclass
class GridPoint:
    index: int
    i_power: int
    j_freq: int
    power_dbm: float
    pump_freq_ghz: float
    current_a: float


def pump_flags_cold(args: argparse.Namespace) -> list[str]:
    return [
        "--initial-guess", "zero",
        "--continuation-mode", "fixed",
        "--continuation-steps", str(args.continuation_steps),
    ]


def pump_flags_warm_seed(args: argparse.Namespace) -> list[str]:
    return [
        "--initial-guess", "linear_phasor",
        "--linear-seed-maxiter", str(args.linear_seed_maxiter),
        "--continuation-mode", "adaptive",
        "--adaptive-initial-step", str(args.adaptive_initial_step),
        "--adaptive-min-step", str(args.adaptive_min_step),
    ]


def run_point(
    point: GridPoint,
    pass_dir: Path,
    args: argparse.Namespace,
    *,
    pump_flags: list[str],
    promote_from: Path | None,
) -> dict[str, Any]:
    pdir = pass_dir / "points" / point_name(point.index, point.power_dbm, point.pump_freq_ghz)
    pump_dir = pdir / "pump"
    gain_dir = pdir / "gain"
    pdir.mkdir(parents=True, exist_ok=True)

    point_start = time.perf_counter()

    # JC-source convention: JosephsonCircuits' frequency-domain port current maps
    # to a physical drive of 2*I*cos(wt) under the positive-phasor (x = 2 Re sum X)
    # reconstruction, so exp08's pump current must be 2x the physical port current
    # to match JC. This is the documented "pump scale 2" used by all exp14 parity
    # runs; without it the JTWPA is under-pumped ~2x and shows almost no gain.
    injected_current = point.current_a * args.pump_current_jc_scale
    pump_cmd = [
        args.python_executable, EXP08,
        "--ipm-dir", str(args.ipm_dir),
        "--outdir", str(pump_dir),
        "--pump-port", str(args.pump_port),
        "--pump-freq-ghz", f"{point.pump_freq_ghz:.12g}",
        "--pump-current-a", f"{injected_current:.17g}",
        "--pump-mode-policy", str(args.pump_mode_policy),
        "--nt", str(args.nt),
        "--newton-tol", str(args.newton_tol),
        "--quiet",
        *pump_flags,
    ]
    if args.pump_mode_count is not None:
        pump_cmd.extend(["--pump-mode-count", str(args.pump_mode_count)])
    else:
        pump_cmd.extend(["--harmonics", str(args.harmonics)])
    if promote_from is not None:
        pump_cmd.extend(["--promote-from-pump-dir", str(promote_from)])

    pump_rc, pump_wall_runtime_s = run_command(
        pump_cmd,
        stdout_path=pdir / "pump_stdout.txt",
        stderr_path=pdir / "pump_stderr.txt",
        timeout_s=args.pump_timeout_s,
    )
    pump_report = read_json(pump_dir / "pump_report.json")
    p_status = pump_status(pump_report, pump_rc)

    gain_rc = -1
    gain_report = None
    if p_status == "VALID_CONVERGED":
        gain_cmd = [
            args.python_executable, EXP09,
            "--ipm-dir", str(args.ipm_dir),
            "--pump-dir", str(pump_dir),
            "--outdir", str(gain_dir),
            "--z0-ohm", str(args.z0_ohm),
            "--source-port", str(args.source_port),
            "--out-port", str(args.out_port),
            "--signal-ghz", f"{signal_ghz_for(point.pump_freq_ghz, args):.12g}",
            "--sidebands", str(args.sidebands),
            "--gamma-nt", str(args.gamma_nt),
            "--fallback-pump-freq-ghz", f"{point.pump_freq_ghz:.12g}",
        ]
        gain_rc, gain_wall_runtime_s = run_command(
            gain_cmd,
            stdout_path=pdir / "gain_stdout.txt",
            stderr_path=pdir / "gain_stderr.txt",
            timeout_s=args.gain_timeout_s,
        )
        gain_report = read_json(gain_dir / "gain_report.json")
    else:
        gain_wall_runtime_s = None
    g_status = gain_status(gain_report, gain_rc)

    status = "PASS" if p_status == "VALID_CONVERGED" and g_status == "VALID_SOLVED" else "ERROR"

    row: dict[str, Any] = {
        "point_index": point.index,
        "i_power": point.i_power,
        "j_freq": point.j_freq,
        "pump_power_dbm": point.power_dbm,
        "pump_freq_ghz": point.pump_freq_ghz,
        "pump_current_peak_a": point.current_a,
        "status": status,
        "pump_status": p_status,
        "gain_status": g_status,
        "warm_started": promote_from is not None,
        "elapsed_s": time.perf_counter() - point_start,
        "pump_wall_runtime_s": pump_wall_runtime_s,
        "gain_wall_runtime_s": gain_wall_runtime_s,
        "pump_dir": str(pump_dir),
    }
    row.update(pump_metrics(pump_report))
    row.update(gain_metrics(gain_report))
    return row


# =============================================================================
# Passes
# =============================================================================

def run_cold_pass(
    points: list[GridPoint], pass_dir: Path, args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(points)
    for point in points:
        row = run_point(point, pass_dir, args, pump_flags=pump_flags_cold(args), promote_from=None)
        rows.append(row)
        print(
            f"[cold {point.index + 1}/{total}] P={point.power_dbm:.4g} dBm "
            f"fp={point.pump_freq_ghz:.4g} GHz status={row['status']} "
            f"gain={row.get('gain_db')} pump_s={row.get('pump_runtime_s')}",
            flush=True,
        )
    return rows


def run_warm_pass(
    points: list[GridPoint],
    pass_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Traverse each frequency column in increasing power, warm-starting."""
    by_col: dict[int, list[GridPoint]] = {}
    for point in points:
        by_col.setdefault(point.j_freq, []).append(point)

    rows: list[dict[str, Any]] = []
    total = len(points)
    done = 0
    for j in sorted(by_col):
        column = sorted(by_col[j], key=lambda p: p.power_dbm)
        previous_pump_dir: Path | None = None
        for point in column:
            promote = previous_pump_dir
            # First point of the column (or after a failure) is seeded with
            # linear_phasor + adaptive; the rest warm-start from the neighbour
            # via a single full-scale Newton solve. Pass zero-init flags on
            # warm-start points so exp08 skips the unused linear_phasor seed.
            flags = ["--initial-guess", "zero"] if promote is not None else pump_flags_warm_seed(args)
            row = run_point(
                point, pass_dir, args,
                pump_flags=flags,
                promote_from=promote,
            )
            # If a warm-start (promote) point diverged, retry once from a fresh
            # linear_phasor + adaptive seed (which has a fixed-continuation
            # fallback). This recovers stiff points where a single Newton solve
            # off the neighbour fails but a graded solve still converges.
            retried = False
            if row["status"] != "PASS" and promote is not None:
                retry = run_point(
                    point, pass_dir, args,
                    pump_flags=pump_flags_warm_seed(args),
                    promote_from=None,
                )
                if retry["status"] == "PASS":
                    row = retry
                    retried = True
            row["warm_retry_reseed"] = retried
            rows.append(row)
            done += 1
            print(
                f"[warm {done}/{total}] P={point.power_dbm:.4g} dBm "
                f"fp={point.pump_freq_ghz:.4g} GHz "
                f"{'WARM' if promote is not None else 'seed'}"
                f"{'+reseed' if retried else ''} "
                f"status={row['status']} gain={row.get('gain_db')} "
                f"pump_s={row.get('pump_runtime_s')}",
                flush=True,
            )
            # Only chain off a converged neighbour.
            if row["status"] == "PASS":
                previous_pump_dir = point_pump_dir(point, pass_dir)
            else:
                previous_pump_dir = None
    rows.sort(key=lambda r: r["point_index"])
    return rows


def point_pump_dir(point: GridPoint, pass_dir: Path) -> Path:
    return pass_dir / "points" / point_name(point.index, point.power_dbm, point.pump_freq_ghz) / "pump"


# =============================================================================
# In-process executor (no per-point subprocess imports; real_coupled precond)
# =============================================================================

class InProcessEngine:
    """Run the exp08 pump solve + exp09 gain in this process.

    The IPM matrices and the heavy numpy/scipy imports are paid once instead of
    per point. Numerics are identical to the subprocess path: the same exp08 and
    exp09 functions are called, and the pump solution is still round-tripped
    through ``pump_solution.npz`` so exp09's gamma/khat pipeline is byte-for-byte
    the same. ``real_coupled`` preconditioning gives a bit-identical pump
    solution while cutting GMRES iterations ~50x.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ipm08 = exp08.load_ipm(args.ipm_dir)
        self.ipm09 = exp09.load_ipm(str(args.ipm_dir))
        self.branch = exp08.JosephsonBranchArray(Ic=self.ipm08.Ic, phi0=self.ipm08.phi0)
        self.pump_idx = self.ipm08.port_to_index[args.pump_port]
        self.source_idx = self.ipm09.port_to_index[args.source_port]
        self.out_idx = self.ipm09.port_to_index[args.out_port]
        self.ic_median = float(np.median(self.ipm08.Ic))
        self.ports = list(self.ipm08.port_to_index.values())
        # Schur partition is constant in power -> cache per pump frequency, but
        # LRU-bounded: each partition holds a large factorized `_fast_coupled`
        # block, and the warm pass finishes one frequency column before moving
        # on, so only the current column's partition is live. Caching every
        # frequency unbounded is what OOMs large maps (50 partitions ~ 16 GB).
        # Keep the last few (insertion-ordered dict acts as the LRU).
        self._schur_part_cache: dict[float, Any] = {}
        self._schur_cache_max = max(1, int(getattr(args, "inproc_schur_cache_size", 2)))

    def _settings(self) -> exp08.NewtonKrylovSettings:
        return exp08.NewtonKrylovSettings(
            newton_tol=self.args.newton_tol, max_newton=self.args.inproc_max_newton, gmres_rtol=1e-7,
            gmres_atol=0.0, gmres_restart=60, gmres_maxiter=self.args.inproc_gmres_maxiter,
            min_alpha=1.0 / 1024.0,
            preconditioner=self.args.inproc_preconditioner, compute_time_residual=True, verbose=False,
            continuation_predictor="none", jvp_mode="aft", stall_ratio=0.8, stall_patience=4,
            solve_deadline_s=self.args.inproc_solve_deadline_s,
            precond_reuse=self.args.inproc_precond_reuse,
            precond_reuse_refresh_gmres=self.args.inproc_precond_refresh_gmres,
        )

    def _build_problem(self, freq_ghz: float, current_a: float):
        omega = 2.0 * math.pi * freq_ghz * 1e9
        basis = pump_basis.resolve_pump_basis(
            policy=self.args.pump_mode_policy, omega_p=omega,
            harmonics=self.args.harmonics, mode_count=self.args.pump_mode_count,
            explicit_modes=None, design_meta=self.ipm08.summary,
        )
        grid = exp08.HarmonicGrid(modes=basis.k, nt=self.args.nt, omega=omega)
        problem = exp08.FullIPMPumpProblem(
            C=self.ipm08.C, G=self.ipm08.G, K=self.ipm08.K, Bphi=self.ipm08.Bphi,
            branch=self.branch, grid=grid, pump_node_index=self.pump_idx,
            pump_current_a=current_a, source_mode=basis.source_mode,
        )
        return problem, basis, omega

    def solve_point(
        self, point: GridPoint, pass_dir: Path, *, mode: str, warm_X: np.ndarray | None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        a = self.args
        pdir = pass_dir / "points" / point_name(point.index, point.power_dbm, point.pump_freq_ghz)
        pump_dir = pdir / "pump"
        gain_dir = pdir / "gain"
        pdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        pump_wall_start = time.perf_counter()

        injected = point.current_a * a.pump_current_jc_scale
        t_setup = time.perf_counter()
        full_problem, basis, omega = self._build_problem(point.pump_freq_ghz, injected)
        pump_setup_runtime_s = time.perf_counter() - t_setup
        # Optional Schur-reduced backend: solve on retained nodes, reconstruct
        # the full solution for write_results/exp09 (which need full-node X).
        use_schur = a.inproc_pump_backend == "schur_cpu_mt"
        t_schur = time.perf_counter()
        if use_schur:
            cache = self._schur_part_cache
            # pop-then-reinsert so the used key becomes most-recent (LRU order).
            part = cache.pop(point.pump_freq_ghz, None)
            sprob = (SchurReducedProblem(full=full_problem, partition=part)
                     if part is not None
                     else build_schur_problem(full_problem, self.ports))
            cache[point.pump_freq_ghz] = sprob.part
            while len(cache) > self._schur_cache_max:
                del cache[next(iter(cache))]  # evict oldest -> frees its splu
                gc.collect()
            solve_problem = sprob
        else:
            solve_problem = full_problem
        pump_schur_setup_runtime_s = time.perf_counter() - t_schur
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())

        t_solve = time.perf_counter()
        if mode == "warm" and warm_X is not None:
            X, reports = solver.solve_direct(solve_problem, warm_X)
        elif mode == "seed" and not use_schur:
            X_seed, _ = exp08.build_linear_phasor_seed(
                full_problem, source_scale=1.0, method="gmres", rtol=1e-6,
                maxiter=a.linear_seed_maxiter, restart=60,
            )
            X, reports, _ = solver.solve_adaptive_continuation(
                full_problem, X_seed, initial_step=a.adaptive_initial_step,
                min_step=a.adaptive_min_step, growth=1.5, shrink=0.5,
                fallback_fixed_steps=20,
            )
        else:  # cold (and schur seed -> cold continuation on retained)
            X, reports = solver.solve_continuation(solve_problem, continuation_steps=a.continuation_steps)

        pump_solve_wall_runtime_s = time.perf_counter() - t_solve

        converged = bool(reports and reports[-1].converged
                         and abs(reports[-1].source_scale - 1.0) < 1e-12)

        # X is retained-sized for the Schur backend; reconstruct full nodes.
        chain_X = X
        X_full = solve_problem.reconstruct_full(X) if use_schur else X

        metadata = {
            **basis.to_metadata(),
            "pump_freq_ghz": point.pump_freq_ghz, "nt": a.nt, "omega_p": omega,
            "pump_current_a": injected,
            "pump_current_ratio_ic_median": injected / self.ic_median,
            "pump_backend": a.inproc_pump_backend,
        }
        t_write = time.perf_counter()
        summary = exp08.summarize_solution(full_problem, X_full)
        exp08.write_results(pump_dir, X_full, reports, summary, metadata)
        pump_write_runtime_s = time.perf_counter() - t_write
        pump_wall_runtime_s = time.perf_counter() - pump_wall_start

        row: dict[str, Any] = {
            "point_index": point.index, "i_power": point.i_power, "j_freq": point.j_freq,
            "pump_power_dbm": point.power_dbm, "pump_freq_ghz": point.pump_freq_ghz,
            "pump_current_peak_a": point.current_a, "warm_started": mode == "warm",
            "pump_status": "VALID_CONVERGED" if converged else "FAIL",
            "pump_runtime_s": float(sum(r.runtime_s for r in reports)),
            "pump_wall_runtime_s": pump_wall_runtime_s,
            "pump_setup_runtime_s": pump_setup_runtime_s,
            "pump_schur_setup_runtime_s": pump_schur_setup_runtime_s,
            "pump_solve_wall_runtime_s": pump_solve_wall_runtime_s,
            "pump_write_runtime_s": pump_write_runtime_s,
            "pump_factor_runtime_s": float(sum(r.factor_runtime_s for r in reports)),
            "pump_preconditioner_assembly_runtime_s": float(sum(getattr(r, "preconditioner_assembly_runtime_s", 0.0) for r in reports)),
            "pump_preconditioner_numeric_factor_runtime_s": float(sum(getattr(r, "preconditioner_numeric_factor_runtime_s", 0.0) for r in reports)),
            "pump_coeff_rel": float(reports[-1].coeff_rel) if reports else None,
            "pump_time_rel": (float(solve_problem.full_time_residual_rel(X, 1.0))
                              if use_schur and converged
                              else float(reports[-1].time_rel)
                              if reports and reports[-1].time_rel is not None else None),
            "pump_newton_total": int(sum(r.newton_iterations for r in reports)),
            "pump_gmres_total": int(sum(r.gmres_iterations_total for r in reports)),
            "pump_branch_current_max": finite_or_none(summary.get("branch_i_max_abs")),
        }
        row.update({k: None for k in (
            "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db", "signal_ghz",
            "linear_rel_residual", "gain_total_runtime_s", "gain_wall_runtime_s",
            "gain_gamma_hat_runtime_s", "gain_khat_build_runtime_s",
            "gain_khat_off_runtime_s", "gain_matrix_assemble_runtime_s",
            "gain_factor_solve_runtime_s", "gain_baseline_off_runtime_s",
            "gain_baseline_pumpdiag_runtime_s",
            "spectrum_peak_gain_db", "spectrum_peak_signal_ghz")})
        row["gain_status"] = "ERROR"

        if converged:
            g, gain_timing, spectrum = self._gain(pump_dir, gain_dir, point.pump_freq_ghz)
            row.update(gain_timing)
            if spectrum is not None:
                row["_spectrum"] = spectrum  # dropped from CSV; -> map_spectrum.npz
                gains = [gd for gd, st in zip(spectrum["gain_db"], spectrum["status"])
                         if st == "VALID_SOLVED"]
                if gains:
                    k = int(np.nanargmax(spectrum["gain_db"]))
                    row["spectrum_peak_gain_db"] = float(spectrum["gain_db"][k])
                    row["spectrum_peak_signal_ghz"] = float(spectrum["signal_ghz"][k])
            if g is not None and g.status == "VALID_SOLVED":
                row["gain_status"] = "VALID_SOLVED"
                row["gain_db"] = float(g.gain_db)
                row["gain_vs_off_db"] = float(g.gain_vs_off_db)
                row["gain_vs_pumpdiag_db"] = float(g.gain_vs_pumpdiag_db)
                row["signal_ghz"] = float(g.signal_ghz)
                row["linear_rel_residual"] = float(g.linear_rel_residual)

        row["status"] = "PASS" if (row["pump_status"] == "VALID_CONVERGED"
                                   and row["gain_status"] == "VALID_SOLVED") else "ERROR"
        row["elapsed_s"] = time.perf_counter() - t0
        row["pump_dir"] = str(pump_dir)
        return row, (X if converged else None)

    def _gain(self, pump_dir: Path, gain_dir: Path, freq_ghz: float):
        a = self.args
        gain_dir.mkdir(parents=True, exist_ok=True)
        t_all = time.perf_counter()
        pump = exp09.load_pump(pump_dir, fallback_pump_freq_ghz=freq_ghz)
        ms = exp09.sideband_list(a.sidebands)
        max_ell = max(abs(m - q) for m in ms for q in ms)
        t0 = time.perf_counter()
        gamma_hat = exp09.compute_gamma_hat(
            ipm=self.ipm09, pump=pump, max_ell=max_ell, gamma_nt=a.gamma_nt,
            dc_branch_flux=None,
        )
        gamma_runtime_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        khat = exp09.build_khat(Bphi=self.ipm09.Bphi, gamma_hat=gamma_hat, drop_tol=0.0)
        khat_runtime_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        gamma_off = self.ipm09.Ic / self.ipm09.phi0
        khat_off_0 = (
            self.ipm09.Bphi @ sp.diags(gamma_off, offsets=0, format="csr") @ self.ipm09.Bphi.T
        ).astype(np.complex128).tocsr()
        khat_off_runtime_s = time.perf_counter() - t0

        # Signal-frequency-independent Floquet conversion base: built once here,
        # reused by the trailing solve and every spectrum point (the dominant
        # speedup for multi-signal cells).
        khat_big_base = None
        khat_base_runtime_s = 0.0
        if a.signal_spectrum:
            t0 = time.perf_counter()
            khat_big_base = exp09.assemble_khat_conversion_base(self.ipm09, khat, ms)
            khat_base_runtime_s = time.perf_counter() - t0

        g = self._solve_signal(khat, khat_off_0, khat_big_base, pump.omega_p,
                               signal_ghz_for(freq_ghz, a))

        spectrum = None
        if a.signal_spectrum:
            offs = spectrum_offsets_mhz(a)

            def one(off: float) -> tuple[float, float, Any]:
                fs = float(freq_ghz) + off / 1000.0
                gg = self._solve_signal(khat, khat_off_0, khat_big_base,
                                        pump.omega_p, fs)
                return off, fs, gg

            if a.signal_workers > 1:
                with ThreadPoolExecutor(max_workers=a.signal_workers) as pool:
                    items = list(pool.map(one, offs))
            else:
                items = [one(off) for off in offs]
            spectrum = {
                "offsets_mhz": [it[0] for it in items],
                "signal_ghz": [it[1] for it in items],
                "gain_db": [float(it[2].gain_db) for it in items],
                "status": [it[2].status for it in items],
            }

        timing = {
            "gain_wall_runtime_s": time.perf_counter() - t_all,
            "gain_total_runtime_s": time.perf_counter() - t_all,
            "gain_gamma_hat_runtime_s": gamma_runtime_s,
            "gain_khat_build_runtime_s": khat_runtime_s + khat_base_runtime_s,
            "gain_khat_off_runtime_s": khat_off_runtime_s,
            "gain_matrix_assemble_runtime_s": float(g.assemble_runtime_s),
            "gain_factor_solve_runtime_s": float(g.factor_solve_runtime_s),
            "gain_baseline_off_runtime_s": float(g.baseline_off_runtime_s),
            "gain_baseline_pumpdiag_runtime_s": float(g.baseline_pumpdiag_runtime_s),
        }
        return g, timing, spectrum

    def _solve_signal(self, khat, khat_off_0, khat_big_base, omega_p, signal_ghz):
        a = self.args
        common = dict(
            ipm=self.ipm09, khat=khat, khat_off_0=khat_off_0,
            khat_big_base=khat_big_base, omega_p=omega_p, signal_ghz=signal_ghz,
            sidebands=a.sidebands, signal_m=0, idler_m=-2,
            source_index=self.source_idx, out_index=self.out_idx,
            source_current_a=1.0, source_port=a.source_port, out_port=a.out_port,
            z0_ohm=a.z0_ohm, loss_model="current_complex_c",
            linear_solver=a.signal_solver,
        )
        if a.signal_backend == "schur":
            return exp09.solve_gain_one_schur(
                **common, include_baselines=not a.skip_baselines)
        return exp09.solve_gain_one(**common)


def run_cold_pass_inprocess(
    points: list[GridPoint], pass_dir: Path, engine: InProcessEngine
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(points)
    for point in points:
        row, _ = engine.solve_point(point, pass_dir, mode="cold", warm_X=None)
        rows.append(row)
        print(f"[cold {point.index + 1}/{total}] P={point.power_dbm:.4g} dBm "
              f"fp={point.pump_freq_ghz:.4g} GHz status={row['status']} "
              f"gain={row.get('gain_db')} pump_s={row.get('pump_runtime_s'):.3f}", flush=True)
    return rows


def secant_guess(
    x_prevprev: np.ndarray, x_prev: np.ndarray,
    cur_prevprev: float, cur_prev: float, cur: float,
) -> np.ndarray:
    """Linear extrapolation of the pump state along the pump-current axis.

    Given the last two converged solutions ``x_prevprev`` (at ``cur_prevprev``)
    and ``x_prev`` (at ``cur_prev``), predict the solution at ``cur``:

        X_guess = x_prev + beta * (x_prev - x_prevprev),
        beta    = (cur - cur_prev) / (cur_prev - cur_prevprev).

    The current amplitude is the natural continuation parameter (the source term
    is linear in it). Only the initial guess changes -- physics is untouched.
    """
    denom = cur_prev - cur_prevprev
    if abs(denom) < 1e-30:
        return x_prev
    beta = (cur - cur_prev) / denom
    return x_prev + beta * (x_prev - x_prevprev)


SKIP_PAST_FOLD = "SKIP_PAST_FOLD"

# Row fields that a solved point fills but a skipped one leaves empty.
_SKIP_NONE_FIELDS = (
    "gain_db", "gain_vs_off_db", "gain_vs_pumpdiag_db", "signal_ghz",
    "linear_rel_residual", "gain_total_runtime_s", "gain_wall_runtime_s",
    "gain_gamma_hat_runtime_s", "gain_khat_build_runtime_s",
    "gain_khat_off_runtime_s", "gain_matrix_assemble_runtime_s",
    "gain_factor_solve_runtime_s", "gain_baseline_off_runtime_s",
    "gain_baseline_pumpdiag_runtime_s", "pump_runtime_s", "pump_wall_runtime_s",
    "pump_setup_runtime_s", "pump_schur_setup_runtime_s",
    "pump_solve_wall_runtime_s", "pump_write_runtime_s", "pump_factor_runtime_s",
    "pump_preconditioner_assembly_runtime_s",
    "pump_preconditioner_numeric_factor_runtime_s", "pump_coeff_rel",
    "pump_time_rel", "pump_newton_total", "pump_gmres_total",
    "pump_branch_current_max",
)


def past_fold_skip_row(point: GridPoint) -> dict[str, Any]:
    """Synthetic row for a cell skipped by the per-column fold short-circuit.

    Once a frequency column fails to reach full drive at some pump power, every
    higher-power cell is past the harmonic-balance fold -- a turning point with
    no re-convergence above it -- so it is marked past-fold without solving.
    Gain is NaN (a map hole), matching a genuine over-fold failure, and the row
    costs no solver time.
    """
    row: dict[str, Any] = {
        "point_index": point.index, "i_power": point.i_power,
        "j_freq": point.j_freq, "pump_power_dbm": point.power_dbm,
        "pump_freq_ghz": point.pump_freq_ghz,
        "pump_current_peak_a": point.current_a, "warm_started": False,
        "pump_status": SKIP_PAST_FOLD, "gain_status": SKIP_PAST_FOLD,
        "status": SKIP_PAST_FOLD, "warm_retry_reseed": False,
        "pump_predictor": "skip", "elapsed_s": 0.0, "pump_dir": "",
    }
    row.update({k: None for k in _SKIP_NONE_FIELDS})
    return row


def run_warm_pass_inprocess(
    points: list[GridPoint], pass_dir: Path, engine: InProcessEngine,
    *, fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Warm-start each frequency column in increasing power.

    fail_fast: do not pay the reseed/adaptive-fallback recovery on a failed
    point, and keep warm-starting subsequent points from the last *converged*
    neighbour. Within a column the harmonic-balance fold is a turning point (no
    re-convergence above it), so this leaves the convergent points unchanged
    while letting over-fold points fail in ~one stalled solve instead of
    thrashing through the full recovery chain.
    """
    by_col: dict[int, list[GridPoint]] = {}
    for point in points:
        by_col.setdefault(point.j_freq, []).append(point)
    rows: list[dict[str, Any]] = []
    total = len(points)
    done = 0
    predictor = getattr(engine.args, "inproc_fold_predictor", "none")
    scale = engine.args.pump_current_jc_scale
    patience = int(getattr(engine.args, "fold_skip_patience", 0))
    for j in sorted(by_col):
        column = sorted(by_col[j], key=lambda p: p.power_dbm)
        prev_X: np.ndarray | None = None
        # Last two converged (injected_current, X) for the secant predictor.
        last_good_X: np.ndarray | None = None
        last_good_cur: float | None = None
        prevprev_X: np.ndarray | None = None
        prevprev_cur: float | None = None
        consec_fail = 0  # consecutive non-converged points at increasing power
        for idx, point in enumerate(column):
            cur = point.current_a * scale
            base_X = prev_X if not fail_fast else last_good_X
            mode = "warm" if base_X is not None else "seed"

            # Predict the guess from the last two converged states. base_X is the
            # most recent converged solution (at last_good_cur) whenever it is set.
            use_secant = (
                predictor == "secant" and base_X is not None
                and prevprev_X is not None and prevprev_cur is not None
                and last_good_cur is not None and prevprev_X.shape == base_X.shape
            )
            guess = (secant_guess(prevprev_X, base_X, prevprev_cur, last_good_cur, cur)
                     if use_secant else base_X)
            pred_tag = "secant" if use_secant else "none"

            row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=guess)

            # Overshoot guard: a bad extrapolation past the fold -> retry once
            # from the plain warm start before paying the reseed.
            if row["status"] != "PASS" and use_secant:
                row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=base_X)
                pred_tag = "secant_fallback"

            retried = False
            if row["status"] != "PASS" and mode == "warm" and not fail_fast:
                row, X = engine.solve_point(point, pass_dir, mode="seed", warm_X=None)
                retried = row["status"] == "PASS"
            row["warm_retry_reseed"] = retried
            row["pump_predictor"] = pred_tag
            rows.append(row)
            done += 1
            print(f"[warm {done}/{total}] P={point.power_dbm:.4g} dBm "
                  f"fp={point.pump_freq_ghz:.4g} GHz {mode}"
                  f"{'+' + pred_tag if pred_tag != 'none' else ''}"
                  f"{'+reseed' if retried else ''} "
                  f"status={row['status']} gain={row.get('gain_db')} "
                  f"newton={row.get('pump_newton_total')} "
                  f"pump_s={row.get('pump_runtime_s'):.3f}", flush=True)
            if row["status"] == "PASS":
                prevprev_X, prevprev_cur = last_good_X, last_good_cur
                last_good_X, last_good_cur = X, cur
                prev_X = X
                consec_fail = 0
            else:
                prev_X = None  # non-fail-fast path re-seeds next point
                consec_fail += 1

            # Per-column fold short-circuit: after `patience` consecutive
            # non-converged points at increasing power, the column is past the
            # HB fold (a turning point -- no re-convergence above it), so mark
            # every remaining higher-power cell past-fold without solving.
            if patience > 0 and consec_fail >= patience and idx + 1 < len(column):
                skipped = column[idx + 1:]
                for rest in skipped:
                    rows.append(past_fold_skip_row(rest))
                done += len(skipped)
                print(f"[warm {done}/{total}] fp={point.pump_freq_ghz:.4g} GHz "
                      f"fold short-circuit: skipped {len(skipped)} past-fold "
                      f"cells above P={point.power_dbm:.4g} dBm", flush=True)
                break
    rows.sort(key=lambda r: r["point_index"])
    return rows


# =============================================================================
# Gate
# =============================================================================

@dataclass
class GateResult:
    evaluated: bool
    passed: bool
    reasons: list[str] = field(default_factory=list)
    max_gain_drift_db: float | None = None
    n_compared: int = 0
    warm_converged_frac: float | None = None
    n_warm_failed: int = 0
    cold_pump_runtime_s: float | None = None
    warm_pump_runtime_s: float | None = None
    cold_pump_mean_s: float | None = None
    warm_pump_mean_s: float | None = None
    pump_speedup: float | None = None


def total_pump_runtime(rows: list[dict[str, Any]]) -> float:
    return float(sum(finite_or_none(r.get("pump_runtime_s")) or 0.0 for r in rows))


def mean_pump_runtime(rows: list[dict[str, Any]]) -> float | None:
    vals = [
        finite_or_none(r.get("pump_runtime_s"))
        for r in rows
        if r.get("status") == "PASS" and finite_or_none(r.get("pump_runtime_s")) is not None
    ]
    return float(sum(vals) / len(vals)) if vals else None


def evaluate_gate(
    cold_rows: list[dict[str, Any]],
    warm_rows: list[dict[str, Any]],
    *,
    gate_gain_db: float,
    min_converged_frac: float,
) -> GateResult:
    reasons: list[str] = []

    n_warm = len(warm_rows)
    warm_failures = [r for r in warm_rows if r["status"] != "PASS"]
    converged_frac = (n_warm - len(warm_failures)) / n_warm if n_warm else None
    if converged_frac is not None and converged_frac < min_converged_frac:
        reasons.append(
            f"warm convergence {converged_frac:.4f} < {min_converged_frac:.4f} "
            f"({len(warm_failures)}/{n_warm} failed)"
        )

    cold_by_key = {(r["i_power"], r["j_freq"]): r for r in cold_rows}
    drifts: list[float] = []
    n_compared = 0
    for w in warm_rows:
        c = cold_by_key.get((w["i_power"], w["j_freq"]))
        if c is None or c["status"] != "PASS" or w["status"] != "PASS":
            continue
        gw = finite_or_none(w.get("gain_db"))
        gc = finite_or_none(c.get("gain_db"))
        if gw is None or gc is None:
            continue
        drifts.append(abs(gw - gc))
        n_compared += 1

    max_drift = max(drifts) if drifts else None
    if max_drift is None:
        reasons.append("no comparable cold/warm point pairs")
    elif max_drift > gate_gain_db:
        reasons.append(
            f"max gain drift {max_drift:.3e} dB > gate {gate_gain_db:.3e} dB"
        )

    # Per-point speedup. Comparing totals is invalid when the cold pass is a
    # sparse spot-check (5 points) against a full warm pass (e.g. 1225); use the
    # mean converged pump time per point instead.
    cold_mean = mean_pump_runtime(cold_rows)
    warm_mean = mean_pump_runtime(warm_rows)
    speedup = cold_mean / warm_mean if (cold_mean and warm_mean) else None
    if speedup is None or speedup <= 1.0:
        reasons.append("warm pass not faster than cold (per point)")

    return GateResult(
        evaluated=True,
        passed=not reasons,
        reasons=reasons,
        max_gain_drift_db=max_drift,
        n_compared=n_compared,
        warm_converged_frac=converged_frac,
        n_warm_failed=len(warm_failures),
        cold_pump_runtime_s=total_pump_runtime(cold_rows),
        warm_pump_runtime_s=total_pump_runtime(warm_rows),
        cold_pump_mean_s=cold_mean,
        warm_pump_mean_s=warm_mean,
        pump_speedup=speedup,
    )


# =============================================================================
# Output
# =============================================================================

def write_points_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "pass", "point_index", "i_power", "j_freq", "pump_power_dbm",
        "pump_freq_ghz", "pump_current_peak_a", "status", "pump_status",
        "gain_status", "warm_started", "warm_retry_reseed", "pump_predictor",
        "gain_db", "gain_vs_off_db",
        "gain_vs_pumpdiag_db", "signal_ghz", "linear_rel_residual",
        "pump_runtime_s", "pump_wall_runtime_s", "pump_setup_runtime_s",
        "pump_schur_setup_runtime_s", "pump_solve_wall_runtime_s",
        "pump_write_runtime_s", "pump_factor_runtime_s",
        "pump_preconditioner_assembly_runtime_s",
        "pump_preconditioner_numeric_factor_runtime_s", "pump_newton_total",
        "pump_gmres_total", "pump_coeff_rel", "pump_time_rel", "pump_branch_current_max",
        "gain_total_runtime_s", "gain_wall_runtime_s", "gain_gamma_hat_runtime_s",
        "gain_khat_build_runtime_s", "gain_khat_off_runtime_s",
        "gain_matrix_assemble_runtime_s", "gain_factor_solve_runtime_s",
        "gain_baseline_off_runtime_s", "gain_baseline_pumpdiag_runtime_s",
        "spectrum_peak_gain_db", "spectrum_peak_signal_ghz",
        "elapsed_s", "pump_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def gain_grid(rows: list[dict[str, Any]], n_power: int, n_freq: int) -> np.ndarray:
    grid = np.full((n_power, n_freq), np.nan, dtype=float)
    for r in rows:
        value = finite_or_none(r.get("gain_db"))
        if value is not None and r["status"] == "PASS":
            grid[int(r["i_power"]), int(r["j_freq"])] = value
    return grid


def write_spectrum(
    path: Path, rows: list[dict[str, Any]], powers: np.ndarray,
    freqs: np.ndarray, offsets: list[float],
) -> None:
    """Write the per-cell signal spectrum as a (n_power, n_freq, n_offset) cube.

    Reads the ``_spectrum`` payload each PASS row carries (offsets aligned to
    ``offsets``); non-solved cells stay NaN. ``signal_ghz`` is fp+offset, so the
    absolute signal axis is per (offset, j_freq) -- stored as a 2D helper too.
    """
    n_off = len(offsets)
    cube = np.full((powers.size, freqs.size, n_off), np.nan, dtype=float)
    off_arr = np.asarray(offsets, dtype=float)
    for r in rows:
        spec = r.get("_spectrum")
        if not spec or r["status"] != "PASS":
            continue
        i, j = int(r["i_power"]), int(r["j_freq"])
        for k, (gd, st) in enumerate(zip(spec["gain_db"], spec["status"])):
            if st == "VALID_SOLVED" and k < n_off:
                cube[i, j, k] = float(gd)
    signal_ghz = freqs[None, :] + off_arr[:, None] / 1000.0  # (n_off, n_freq)
    np.savez(path, pump_power_dbm=powers, pump_frequency_ghz=freqs,
             signal_offset_mhz=off_arr, gain_spectrum_db=cube,
             signal_ghz=signal_ghz)


def write_arrays(
    path: Path,
    powers: np.ndarray,
    freqs: np.ndarray,
    grids: dict[str, np.ndarray],
) -> None:
    np.savez(path, pump_power_dbm=powers, pump_frequency_ghz=freqs, **grids)


def total_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [finite_or_none(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return float(sum(vals)) if vals else None


def timing_totals(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = [
        "pump_wall_runtime_s",
        "pump_runtime_s",
        "pump_setup_runtime_s",
        "pump_schur_setup_runtime_s",
        "pump_solve_wall_runtime_s",
        "pump_write_runtime_s",
        "pump_factor_runtime_s",
        "pump_preconditioner_assembly_runtime_s",
        "pump_preconditioner_numeric_factor_runtime_s",
        "gain_wall_runtime_s",
        "gain_total_runtime_s",
        "gain_gamma_hat_runtime_s",
        "gain_khat_build_runtime_s",
        "gain_khat_off_runtime_s",
        "gain_matrix_assemble_runtime_s",
        "gain_factor_solve_runtime_s",
        "gain_baseline_off_runtime_s",
        "gain_baseline_pumpdiag_runtime_s",
        "elapsed_s",
    ]
    return {key: total_metric(rows, key) for key in keys}


def write_summary(
    outdir: Path,
    args: argparse.Namespace,
    cold_rows: list[dict[str, Any]],
    warm_rows: list[dict[str, Any]],
    gate: GateResult,
    elapsed_s: float,
) -> None:
    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out

    summary: dict[str, Any] = {
        "mode": args.mode,
        "output_dir": str(outdir),
        "grid": {"n_power": args.n_power, "n_frequency": args.n_frequency},
        "pump_power_dbm": [args.pump_power_min_dbm, args.pump_power_max_dbm],
        "pump_freq_ghz": [args.pump_freq_min_ghz, args.pump_freq_max_ghz],
        "attenuation_db": args.attenuation_db,
        "z0_ohm": args.z0_ohm,
        "signal_ghz": args.signal_ghz,
        "signal_detuning_mhz": args.signal_detuning_mhz,
        "signal_convention": ("fixed" if args.signal_ghz is not None
                              else f"ws = wp - {args.signal_detuning_mhz} MHz"),
        "current_convention": "I_peak = sqrt(2 * P_W / Z0), P = P_dbm - attenuation_db",
        "cold_status_counts": counts(cold_rows),
        "warm_status_counts": counts(warm_rows),
        "cold_pump_runtime_s": total_pump_runtime(cold_rows) if cold_rows else None,
        "warm_pump_runtime_s": total_pump_runtime(warm_rows) if warm_rows else None,
        "cold_gain_runtime_s": total_metric(cold_rows, "gain_total_runtime_s") if cold_rows else None,
        "warm_gain_runtime_s": total_metric(warm_rows, "gain_total_runtime_s") if warm_rows else None,
        "cold_timing_totals": timing_totals(cold_rows) if cold_rows else {},
        "warm_timing_totals": timing_totals(warm_rows) if warm_rows else {},
        "elapsed_s": elapsed_s,
        "gate": {
            "evaluated": gate.evaluated,
            "passed": gate.passed,
            "reasons": gate.reasons,
            "gate_gain_db": args.gate_gain_db,
            "max_gain_drift_db": gate.max_gain_drift_db,
            "n_compared": gate.n_compared,
            "min_converged_frac": args.gate_min_converged_frac,
            "warm_converged_frac": gate.warm_converged_frac,
            "n_warm_failed": gate.n_warm_failed,
            "pump_speedup_per_point": gate.pump_speedup,
            "cold_pump_mean_s": gate.cold_pump_mean_s,
            "warm_pump_mean_s": gate.warm_pump_mean_s,
            "cold_pump_runtime_s": gate.cold_pump_runtime_s,
            "warm_pump_runtime_s": gate.warm_pump_runtime_s,
        },
    }
    (outdir / "map_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    verdict = "n/a"
    if gate.evaluated:
        verdict = "PASS" if gate.passed else "FAIL"
    lines = [
        "# IPM Warm-Started Pump/Gain Map (exp10)",
        "",
        f"- mode: `{args.mode}`",
        f"- grid: `{args.n_power} x {args.n_frequency}` "
        f"(power `{args.pump_power_min_dbm}`..`{args.pump_power_max_dbm}` dBm, "
        f"freq `{args.pump_freq_min_ghz}`..`{args.pump_freq_max_ghz}` GHz)",
        f"- cold status: `{counts(cold_rows)}`" if cold_rows else "- cold pass: not run",
        f"- warm status: `{counts(warm_rows)}`" if warm_rows else "- warm pass: not run",
        f"- elapsed: `{elapsed_s:.3f}` s",
        f"- warm pump/gain total: `{total_metric(warm_rows, 'pump_runtime_s')}` / `{total_metric(warm_rows, 'gain_total_runtime_s')}` s" if warm_rows else "- warm timing: not run",
        f"- cold pump/gain total: `{total_metric(cold_rows, 'pump_runtime_s')}` / `{total_metric(cold_rows, 'gain_total_runtime_s')}` s" if cold_rows else "- cold timing: not run",
        "",
        "## Gate",
        "",
        f"- verdict: **{verdict}**",
    ]
    if gate.evaluated:
        if gate.reasons:
            lines.append(f"- reasons: `{'; '.join(gate.reasons)}`")
        lines.extend([
            f"- warm converged: `{gate.warm_converged_frac}` "
            f"({gate.n_warm_failed} failed; min `{args.gate_min_converged_frac}`)",
            f"- compared point pairs: `{gate.n_compared}`",
            f"- max gain drift: `{gate.max_gain_drift_db}` dB (gate `{args.gate_gain_db}` dB)",
            f"- pump mean per point: cold `{gate.cold_pump_mean_s}` s, warm `{gate.warm_pump_mean_s}` s",
            f"- pump speedup (per point): `{gate.pump_speedup}`x",
        ])
    lines.extend(["", "## Artifacts", "",
                  "- `map_points.csv`", "- `map_arrays.npz`", "- `map_summary.json`"])
    (outdir / "map_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["cold", "warmstart", "both"], default="both")
    p.add_argument("--executor", choices=["subprocess", "inprocess"], default="subprocess",
                   help="inprocess runs pump+gain in this process (no per-point import "
                   "tax); numerics are identical to the subprocess path.")
    p.add_argument("--inproc-gmres-maxiter", type=int, default=80,
                   help="GMRES outer restart cycles per Newton step (x gmres_restart=60 inner). "
                   "A low value (e.g. 4) bounds the per-Newton cost so over-fold solves don't "
                   "grind thousands of inner iterations; warm-start steps converge in <1 cycle.")
    p.add_argument("--inproc-solve-deadline-s", type=float, default=0.0,
                   help="Per-solve wall-time budget (s) for the in-process path; 0 disables. "
                   "Bounds stiff over-fold solves near the fold.")
    p.add_argument("--inproc-max-newton", type=int, default=16,
                   help="Max Newton iterations per in-process solve. A small cap (e.g. 10) "
                   "makes over-fold points fail fast; warm-start neighbours converge in few.")
    p.add_argument("--inproc-fail-fast", action="store_true",
                   help="In-process warm pass: skip reseed/fallback recovery on a failed "
                   "point and keep warm-starting from the last converged neighbour, so "
                   "over-fold points fail in ~one stalled solve. For high-power fold maps.")
    p.add_argument("--fold-skip-patience", type=int, default=0,
                   help="Per-column fold short-circuit (in-process warm pass): after "
                   "this many consecutive non-converged points at increasing power, "
                   "skip the remaining higher-power cells in the column (marked "
                   "SKIP_PAST_FOLD, gain NaN) without solving. The HB fold is a turning "
                   "point with no re-convergence above it, so those cells are guaranteed "
                   "past-fold. 2 preserves near-monotone columns; 0 disables (default). "
                   "This is the dominant runtime win on hot/over-fold maps.")
    p.add_argument("--inproc-preconditioner",
                   choices=["mean_tangent", "real_coupled", "real_coupled_fast",
                            "spectral_coupled", "linear"],
                   default="mean_tangent",
                   help="Preconditioner for the in-process pump solve. mean_tangent "
                   "(default) is cheapest for small warm-start steps; real_coupled "
                   "cuts GMRES iters but its full-Jacobian LU is costlier per Newton.")
    p.add_argument("--inproc-pump-backend", choices=["full", "schur_cpu_mt"],
                   default="full",
                   help="In-process pump backend. 'full' (default, legacy) solves all "
                   "nodes. 'schur_cpu_mt' eliminates linear-internal nodes via an "
                   "assembled sparse Schur complement (constant per frequency) and "
                   "solves the retained system -- 2.5-4.5x faster at the high-power "
                   "fold, gain identical. Pair with --inproc-preconditioner mean_tangent.")
    p.add_argument("--inproc-schur-cache-size", type=int, default=2,
                   help="Max per-frequency Schur partitions kept in memory (LRU). "
                   "Each partition holds a large factorized block; the warm pass "
                   "only needs the current frequency column's partition, so an "
                   "unbounded cache over 50 frequencies OOMs (~16 GB). 2 keeps a "
                   "small reuse window while bounding RAM regardless of map size.")
    p.add_argument("--inproc-precond-reuse", type=int, default=1,
                   help="Reuse the preconditioner factor for up to N consecutive Newton "
                   "steps (modified-Newton). 1 (default) refactors every step. N>1 "
                   "amortizes the LU across steps -- the big win for real_coupled near "
                   "the fold, where the exact LU barely changes between steps.")
    p.add_argument("--inproc-precond-refresh-gmres", type=int, default=0,
                   help="Force an early factor refresh when the previous Newton step's "
                   "GMRES iterations crossed this threshold (staleness guard for "
                   "--inproc-precond-reuse). 0 disables.")
    p.add_argument("--inproc-fold-predictor", choices=["none", "secant"],
                   default="none",
                   help="In-process warm pass: build the next power point's initial "
                   "guess by extrapolating along the pump-current axis from the last "
                   "two converged solutions (secant), instead of copying the previous "
                   "solution. Cuts Newton steps near the fold where the state moves "
                   "fast with power. Physics unchanged (initial guess only); a failed "
                   "predicted solve falls back to the plain warm start.")
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "exp10_pump_map_warmstart")
    p.add_argument("--ipm-dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")

    p.add_argument("--n-power", type=int, default=5)
    p.add_argument("--n-frequency", type=int, default=5)
    # External power window. With 35 dB attenuation + 50 ohm this spans physical
    # pump ~0.5..1.6 x median Ic; after the JC 2x scale the JTWPA gain ridge runs
    # from onset (~0 dB) up to ~12 dB near JC's 1.5 Ic operating point.
    p.add_argument("--pump-power-min-dbm", type=float, default=-30.0)
    p.add_argument("--pump-power-max-dbm", type=float, default=-20.0)
    p.add_argument("--pump-freq-min-ghz", type=float, default=6.0)
    p.add_argument("--pump-freq-max-ghz", type=float, default=8.0)
    p.add_argument("--attenuation-db", type=float, default=35.0)
    p.add_argument("--z0-ohm", type=float, default=50.0)
    # Signal readout frequency. Default: track the pump at a fixed detuning
    # ws = wp - 100 MHz per cell (the physically correct choice for a map that
    # sweeps pump frequency). Pass --signal-ghz to force a fixed absolute signal.
    p.add_argument("--signal-ghz", type=float, default=None,
                   help="Fixed absolute signal frequency (GHz). If omitted, the "
                   "signal tracks each cell's pump at ws = wp - "
                   "--signal-detuning-mhz.")
    p.add_argument("--signal-detuning-mhz", type=float, default=100.0,
                   help="Signal detuning below the pump when --signal-ghz is not "
                   "set: ws = wp - detuning (default 100 MHz).")

    # Per-cell signal spectrum: solve a ladder of signal frequencies around each
    # pump cell (reusing one Floquet conversion base), not just the single
    # trailing point. The trailing gain_db / map_arrays are unchanged; the
    # spectrum is an additive (n_power, n_freq, n_offset) cube in map_spectrum.npz.
    p.add_argument("--signal-spectrum", action="store_true",
                   help="Solve a spectrum of signal frequencies per cell (see below); "
                   "writes map_spectrum.npz. Reuses exp09's khat conversion base so "
                   "each extra signal point is cheap.")
    p.add_argument("--signal-offset-start-mhz", type=float, default=100.0,
                   help="First |offset| from fp for the spectrum ladder (MHz).")
    p.add_argument("--signal-offset-step-mhz", type=float, default=250.0,
                   help="Spacing between spectrum offsets (MHz).")
    p.add_argument("--signal-offset-count-per-side", type=int, default=5,
                   help="Offsets per side; 5 -> 10 points (+/-100,+/-350,... MHz).")
    p.add_argument("--signal-workers", type=int, default=1,
                   help="Threads over spectrum signal points (1 = serial).")
    p.add_argument("--signal-backend", choices=["direct", "schur"], default="direct",
                   help="Signal linear backend for the gain solve.")
    p.add_argument("--signal-solver", choices=["superlu", "pardiso"], default="superlu",
                   help="Sparse solver for the signal system.")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Skip off/pumpdiag baseline solves (schur backend); gain_db stays valid.")

    p.add_argument("--pump-port", type=int, default=4)
    p.add_argument("--source-port", type=int, default=1)
    p.add_argument("--out-port", type=int, default=2)
    # JTWPA (unbiased 4WM) pump basis: JC odd modes [1,3,...,2K-1], K=10 -> nt>=40.
    p.add_argument("--pump-mode-policy", default="positive_odd_jc")
    p.add_argument("--pump-mode-count", type=int, default=10,
                   help="K for positive_odd_jc -> modes [1,3,...,2K-1]. Set with the basis policy.")
    p.add_argument("--harmonics", type=int, default=3,
                   help="Dense [1..H] harmonics; only used when --pump-mode-count is unset.")
    p.add_argument("--nt", type=int, default=40)
    p.add_argument("--sidebands", type=int, default=10)
    p.add_argument("--gamma-nt", type=int, default=96)
    p.add_argument(
        "--pump-current-jc-scale",
        type=float,
        default=2.0,
        help="Multiply the physical port current by this before injecting (JC "
        "positive-phasor source convention; 2.0 matches JosephsonCircuits).",
    )

    p.add_argument("--continuation-steps", type=int, default=20)
    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--linear-seed-maxiter", type=int, default=5)
    p.add_argument("--adaptive-initial-step", type=float, default=1.0)
    p.add_argument("--adaptive-min-step", type=float, default=0.05)

    p.add_argument("--gate-gain-db", type=float, default=0.01)
    p.add_argument(
        "--gate-min-converged-frac",
        type=float,
        default=0.98,
        help="Gate passes only if at least this fraction of warm points converged "
        "(a few stiff points should not invalidate a large map).",
    )
    p.add_argument(
        "--gate-spotcheck",
        type=int,
        default=0,
        help="warmstart mode: recompute N points cold after the warm pass and "
        "fold their gain drift into the gate (corners+center first).",
    )

    p.add_argument("--pump-timeout-s", type=float, default=600.0)
    p.add_argument("--gain-timeout-s", type=float, default=300.0)
    p.add_argument("--python-executable", default=sys.executable)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_points(args: argparse.Namespace) -> tuple[list[GridPoint], np.ndarray, np.ndarray]:
    powers = np.linspace(args.pump_power_min_dbm, args.pump_power_max_dbm, args.n_power)
    freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.n_frequency)
    points: list[GridPoint] = []
    index = 0
    for i, power_dbm in enumerate(powers):
        for j, freq in enumerate(freqs):
            current = dbm_to_peak_current_a(
                float(power_dbm), attenuation_db=args.attenuation_db, z0_ohm=args.z0_ohm
            )
            points.append(GridPoint(index, i, j, float(power_dbm), float(freq), current))
            index += 1
    return points, powers, freqs


def select_spotcheck_points(points: list[GridPoint], n: int) -> list[GridPoint]:
    if n <= 0 or not points:
        return []
    n_power = max(p.i_power for p in points) + 1
    n_freq = max(p.j_freq for p in points) + 1
    by_ij = {(p.i_power, p.j_freq): p for p in points}
    priority = [
        (0, 0), (n_power - 1, 0), (0, n_freq - 1), (n_power - 1, n_freq - 1),
        (n_power // 2, n_freq // 2),
    ]
    chosen: list[GridPoint] = []
    seen: set[int] = set()
    for key in priority:
        pt = by_ij.get(key)
        if pt is not None and pt.index not in seen:
            chosen.append(pt)
            seen.add(pt.index)
    # Fill remaining slots with an even stride over the flattened grid.
    if len(chosen) < n:
        stride = max(1, len(points) // n)
        for pt in points[::stride]:
            if pt.index not in seen:
                chosen.append(pt)
                seen.add(pt.index)
            if len(chosen) >= n:
                break
    return chosen[:n]


def main() -> int:
    args = parse_args()
    outdir = args.outdir
    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    points, powers, freqs = build_points(args)
    start = time.perf_counter()

    cold_rows: list[dict[str, Any]] = []
    warm_rows: list[dict[str, Any]] = []

    engine = InProcessEngine(args) if args.executor == "inprocess" else None
    cold_pass = (lambda pts, d: run_cold_pass_inprocess(pts, d, engine)) if engine else \
        (lambda pts, d: run_cold_pass(pts, d, args))
    warm_pass = (lambda pts, d: run_warm_pass_inprocess(pts, d, engine, fail_fast=args.inproc_fail_fast)) if engine else \
        (lambda pts, d: run_warm_pass(pts, d, args))
    print(f"executor={args.executor}", flush=True)

    if args.mode in ("cold", "both"):
        cold_rows = cold_pass(points, outdir / "cold")
    if args.mode in ("warmstart", "both"):
        warm_rows = warm_pass(points, outdir / "warm")

    # Spot-check cold recompute for a warm-only run.
    if args.mode == "warmstart" and args.gate_spotcheck > 0:
        spot = select_spotcheck_points(points, args.gate_spotcheck)
        print(f"spot-checking {len(spot)} point(s) cold for the gate", flush=True)
        cold_rows = cold_pass(spot, outdir / "cold_spotcheck")

    if args.mode == "both" or (args.mode == "warmstart" and cold_rows):
        gate = evaluate_gate(
            cold_rows,
            warm_rows,
            gate_gain_db=args.gate_gain_db,
            min_converged_frac=args.gate_min_converged_frac,
        )
    else:
        gate = GateResult(evaluated=False, passed=False, reasons=["gate not applicable for this mode"])

    # Persist tagged rows and grids.
    tagged: list[dict[str, Any]] = []
    for r in cold_rows:
        tagged.append({"pass": "cold", **r})
    for r in warm_rows:
        tagged.append({"pass": "warm", **r})
    write_points_csv(outdir / "map_points.csv", tagged)

    grids: dict[str, np.ndarray] = {}
    if cold_rows and args.mode in ("cold", "both"):
        grids["gain_db_cold"] = gain_grid(cold_rows, args.n_power, args.n_frequency)
    if warm_rows:
        grids["gain_db_warm"] = gain_grid(warm_rows, args.n_power, args.n_frequency)
    if "gain_db_cold" in grids and "gain_db_warm" in grids:
        grids["gain_drift_db"] = np.abs(grids["gain_db_warm"] - grids["gain_db_cold"])
    write_arrays(outdir / "map_arrays.npz", powers, freqs, grids)

    if args.signal_spectrum and warm_rows:
        offsets = spectrum_offsets_mhz(args)
        write_spectrum(outdir / "map_spectrum.npz", warm_rows, powers, freqs, offsets)
        print(f"wrote {outdir / 'map_spectrum.npz'} "
              f"({len(offsets)} signal offsets/cell)", flush=True)

    elapsed = time.perf_counter() - start
    write_summary(outdir, args, cold_rows, warm_rows, gate, elapsed)

    print("", flush=True)
    if gate.evaluated:
        print(f"GATE={'PASS' if gate.passed else 'FAIL'}", flush=True)
        print(f"warm_converged_frac={gate.warm_converged_frac} (failed={gate.n_warm_failed})", flush=True)
        print(f"max_gain_drift_db={gate.max_gain_drift_db}", flush=True)
        print(f"pump_speedup_per_point={gate.pump_speedup}", flush=True)
        if gate.reasons:
            print(f"gate_reasons={'; '.join(gate.reasons)}", flush=True)
    print(f"wrote {outdir / 'map_summary.json'}", flush=True)
    print(f"elapsed_s={elapsed:.3f}", flush=True)

    if gate.evaluated and not gate.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
