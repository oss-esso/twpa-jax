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
``I_peak = sqrt(2 * P_W / Z0)``, after subtracting the line loss. The loss
defaults to the measured ``loss_A10`` model ``c + a*sqrt(f) + b*f`` (dB, f in
GHz); pass a flat ``--attenuation-db`` to override it.
"""

from __future__ import annotations

import argparse
import os
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

# Legacy subprocess paths are kept only for compatibility. The production
# default is the in-process package path below.
EXP08 = "experiments/exp08_full_ipm_pump_solve.py"
EXP09 = "experiments/exp09_full_ipm_gain_from_pump.py"

from twpa_solver import default_loss_model  # noqa: E402
from twpa_solver.core import load_circuit  # noqa: E402
import twpa_solver.pump.hb as exp08  # noqa: E402
import twpa_solver.signal as exp09  # noqa: E402
import twpa_solver.pump.basis as pump_basis  # noqa: E402
from twpa_solver.pump.backends.schur_operators import (  # noqa: E402
    SchurReducedProblem,
    build_schur_problem,
)


# =============================================================================
# Units / helpers
# =============================================================================

def dbm_to_peak_current_a(power_dbm: float, *, attenuation_db: float, z0_ohm: float) -> float:
    if z0_ohm <= 0.0:
        raise ValueError("z0_ohm must be positive")
    source_dbm = float(power_dbm) - float(attenuation_db)
    power_w = 1.0e-3 * 10.0 ** (source_dbm / 10.0)
    return math.sqrt(2.0 * power_w / float(z0_ohm))


def peak_current_to_power_dbm(current_a: float, freq_ghz: float, args: argparse.Namespace) -> float:
    """Inverse of ``dbm_to_peak_current_a``: on-chip peak current -> pump dBm.

    ``I = sqrt(2 P_W / Z0)`` with ``P_W = 1e-3 * 10^((dBm - att)/10)``, so
    ``dBm = 10*log10((I^2 Z0 / 2) / 1e-3) + att(freq)``.
    """
    if current_a <= 0.0:
        return float("-inf")
    power_w = current_a * current_a * float(args.z0_ohm) / 2.0
    source_dbm = 10.0 * math.log10(power_w / 1.0e-3)
    return source_dbm + attenuation_db_for(freq_ghz, args)


def attenuation_db_for(freq_ghz: float, args: argparse.Namespace) -> float:
    """Line attenuation (dB) at ``freq_ghz``.

    Default: the measured loss_A10 model ``c + a*sqrt(f) + b*f`` (frequency
    dependent, f in GHz). A numeric ``--attenuation-db`` overrides it with a flat
    value.
    """
    if args.attenuation_db is not None:
        return float(args.attenuation_db)
    return float(default_loss_model().attenuation_db(float(freq_ghz)))


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

def _final_failure_reason(report: dict[str, Any] | None) -> str | None:
    if report is None:
        return None
    reports = report.get("reports", [])
    final = reports[-1] if reports else {}
    reason = final.get("failure_reason")
    return str(reason) if reason else None


def pump_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {k: None for k in (
            "pump_runtime_s", "pump_factor_runtime_s",
            "pump_preconditioner_assembly_runtime_s",
            "pump_preconditioner_numeric_factor_runtime_s", "pump_coeff_rel",
            "pump_time_rel", "pump_newton_total", "pump_branch_current_max",
            "pump_failure_reason",
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
        "pump_failure_reason": _final_failure_reason(report),
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
        "--ipm-dir", str(args.circuit_dir),
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
            "--ipm-dir", str(args.circuit_dir),
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
        self.ipm08 = load_circuit(args.circuit_dir)
        self.ipm09 = load_circuit(args.circuit_dir)
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
        self._signal_schur_part_cache: dict[tuple[Any, ...], Any] = {}
        self._signal_schur_cache_max = max(1, int(getattr(args, "inproc_schur_cache_size", 2)))

    def _settings(self) -> exp08.NewtonKrylovSettings:
        # Intra-cell continuation predictor: tangent uses the exact lambda-tangent
        # Euler step; every other mode keeps the legacy (copy/secant-at-inter-cell)
        # behaviour, i.e. no intra-cell predictor on the seed path.
        continuation = getattr(
            self.args,
            "inproc_continuation",
            "adaptive_secant",
        )
        cont_pred = {
            "adaptive_secant": "secant",
            "adaptive_tangent": "tangent",
        }.get(continuation, "none")
        return exp08.NewtonKrylovSettings(
            newton_tol=self.args.newton_tol, max_newton=self.args.inproc_max_newton, gmres_rtol=1e-7,
            gmres_atol=0.0, gmres_restart=60, gmres_maxiter=self.args.inproc_gmres_maxiter,
            min_alpha=1.0 / 1024.0,
            preconditioner=self.args.inproc_preconditioner, compute_time_residual=True, verbose=False,
            continuation_predictor=cont_pred, jvp_mode="aft", stall_ratio=0.8, stall_patience=4,
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

    def _make_solve_problem(self, full_problem, freq_ghz: float):
        """The problem actually solved for a cell: Schur-reduced or full.

        For the Schur backend the per-frequency partition is cached (LRU); the
        retained solution vector has a constant (port-node) shape across all
        frequencies, so chained warm starts stay shape-compatible. The full
        backend returns the full problem unchanged.
        """
        if self.args.inproc_pump_backend != "schur_cpu_mt":
            return full_problem
        cache = self._schur_part_cache
        part = cache.pop(freq_ghz, None)  # pop-then-reinsert -> most-recent (LRU)
        sprob = (SchurReducedProblem(full=full_problem, partition=part)
                 if part is not None
                 else build_schur_problem(full_problem, self.ports))
        cache[freq_ghz] = sprob.part
        while len(cache) > self._schur_cache_max:
            del cache[next(iter(cache))]  # evict oldest -> frees its splu
            gc.collect()
        return sprob

    def build_problem_for(self, point: GridPoint):
        """Full pump problem bundle for a grid cell (no Schur reduction).

        Used by the traversal orchestrator to rank predictor candidates by
        residual before solving, and reused by ``solve_point`` via ``prebuilt``
        so the (cheap) problem build is not paid twice.
        """
        injected = point.current_a * self.args.pump_current_jc_scale
        full_problem, basis, omega = self._build_problem(point.pump_freq_ghz, injected)
        return full_problem, basis, omega, injected

    def residual_norm(self, full_problem, X: np.ndarray | None) -> float:
        """Relative coefficient residual of guess ``X`` at full drive (lambda=1).

        The ranking key for the residual-ranked predictor portfolio. Returns
        ``inf`` for a missing or shape-mismatched guess so it sorts last.
        """
        if X is None or X.shape != full_problem.zeros().shape:
            return float("inf")
        try:
            return float(full_problem.norms(X, 1.0, False)["coeff_rel"])
        except (ValueError, FloatingPointError):
            return float("inf")

    def solve_point(
        self, point: GridPoint, pass_dir: Path, *, mode: str, warm_X: np.ndarray | None,
        prebuilt: tuple | None = None, force_gain: bool = False,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        a = self.args
        pdir = pass_dir / "points" / point_name(point.index, point.power_dbm, point.pump_freq_ghz)
        pump_dir = pdir / "pump"
        gain_dir = pdir / "gain"
        pdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        pump_wall_start = time.perf_counter()

        t_setup = time.perf_counter()
        if prebuilt is not None:
            full_problem, basis, omega, injected = prebuilt
        else:
            full_problem, basis, omega, injected = self.build_problem_for(point)
        pump_setup_runtime_s = time.perf_counter() - t_setup
        # Optional Schur-reduced backend: solve on retained nodes, reconstruct
        # the full solution for write_results/exp09 (which need full-node X).
        use_schur = a.inproc_pump_backend == "schur_cpu_mt"
        t_schur = time.perf_counter()
        solve_problem = self._make_solve_problem(full_problem, point.pump_freq_ghz)
        pump_schur_setup_runtime_s = time.perf_counter() - t_schur
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())

        t_solve = time.perf_counter()
        continuation_info: dict[str, Any] = {
            "method": "direct" if mode == "warm" and warm_X is not None else None,
            "steps": None,
            "reached_target": None,
            "fold_lambda": None,
            "runtime_s": None,
        }
        if mode == "warm" and warm_X is not None:
            X, reports = solver.solve_direct(solve_problem, warm_X)
        else:
            cont = getattr(a, "inproc_continuation", "adaptive_secant")
            continuation_info["method"] = cont
            continuation_start = time.perf_counter()
            X_seed = solve_problem.zeros()
            if cont == "fixed":
                X, reports = solver.solve_continuation(
                    solve_problem,
                    continuation_steps=a.continuation_steps,
                )
                continuation_info["steps"] = len(reports)
                continuation_info["reached_target"] = bool(
                    reports
                    and reports[-1].converged
                    and abs(reports[-1].source_scale - 1.0) < 1e-12
                )
            elif cont == "ptc":
                X, reports = solver.solve_pseudo_transient(solve_problem, X_seed)
                continuation_info["steps"] = (
                    reports[-1].newton_iterations if reports else 0
                )
                continuation_info["reached_target"] = bool(
                    reports and reports[-1].converged
                )
            elif cont == "arclength":
                X_arc, _lam, arc_info = solver.solve_arclength(
                    solve_problem,
                    X_seed,
                    0.0,
                    ds=a.inproc_arclength_ds,
                    max_steps=a.inproc_arclength_max_steps,
                    target_lam=1.0,
                    max_wall_s=a.inproc_solve_deadline_s,
                )
                X, reports = solver.solve_direct(solve_problem, X_arc)
                continuation_info["steps"] = arc_info.get("steps")
                continuation_info["reached_target"] = arc_info.get("reached_target")
                continuation_info["fold_lambda"] = arc_info.get("fold_lambda")
            else:
                predictor = "none" if cont == "adaptive_copy" else cont
                # A zero continuation deadline historically meant unlimited
                # adaptive work. For map fail-fast runs, inherit the per-solve
                # deadline so a failed cold seed cannot spend the whole map on
                # repeated adaptive/fallback attempts.
                continuation_deadline = float(
                    getattr(a, "inproc_continuation_deadline_s", 0.0)
                )
                if continuation_deadline <= 0.0:
                    continuation_deadline = float(a.inproc_solve_deadline_s)
                if cont == "affine":
                    X, reports, trace = solver.solve_affine_continuation(
                        solve_problem,
                        X_seed,
                        initial_step=a.adaptive_initial_step,
                        min_step=a.adaptive_min_step,
                        fallback_fixed_steps=a.inproc_fallback_fixed_steps,
                        max_wall_s=continuation_deadline,
                    )
                else:
                    X, reports, trace = solver.solve_adaptive_continuation(
                        solve_problem,
                        X_seed,
                        initial_step=a.adaptive_initial_step,
                        min_step=a.adaptive_min_step,
                        growth=1.5,
                        shrink=0.5,
                        fallback_fixed_steps=a.inproc_fallback_fixed_steps,
                        max_wall_s=continuation_deadline,
                    )
                continuation_info["method"] = cont
                continuation_info["steps"] = len(trace.attempted_lambdas)
                continuation_info["reached_target"] = bool(
                    reports
                    and reports[-1].converged
                    and abs(reports[-1].source_scale - 1.0) < 1e-12
                )
                if predictor == "none" and cont != "affine":
                    continuation_info["method"] = "adaptive_copy"
            continuation_info["runtime_s"] = time.perf_counter() - continuation_start

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
            "pump_failure_reason": (reports[-1].failure_reason if reports else None),
            "pump_continuation_method": continuation_info["method"],
            "pump_continuation_steps": continuation_info["steps"],
            "pump_continuation_reached_target": continuation_info["reached_target"],
            "pump_continuation_fold_lambda": continuation_info["fold_lambda"],
            "pump_continuation_runtime_s": continuation_info["runtime_s"],
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

        # ``force_gain`` runs the gain solve on the last-iterate pump waveform
        # even when Newton did not converge (above-threshold / fold region), so
        # the diagnostic column resume can see what the gain does past the wall.
        if converged or force_gain:
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
        # In force_gain mode return the last-iterate X regardless of convergence
        # so the caller can keep warm-starting up the column past the wall.
        return row, (X if (converged or force_gain) else None)

    def solve_bridge(
        self, parent_X: np.ndarray, parent_current: float, parent_freq: float,
        target_current: float, target_freq: float, *, steps: int, mode: str,
    ) -> np.ndarray | None:
        """March a warm guess from a solved parent to the target along (P, f).

        Continuation in the physical parameters (not lambda): at each sub-step
        build the pump problem at the interpolated (current, frequency) and take
        one full-scale Newton solve warm-started from the previous sub-state.
        Returns the marched state near the target (a strong warm guess for the
        real target solve) or ``None`` if any sub-step fails.

        ``mode``: ``diagonal`` straight line; ``freq_first`` ramps frequency at
        parent power then power; ``power_first`` the reverse; ``adaptive`` walks
        the diagonal with a halving step on failure.
        """
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())

        def step_to(cur: float, frq: float, guess: np.ndarray) -> np.ndarray | None:
            prob, _basis, _omega = self._build_problem(frq, cur)
            solve_prob = self._make_solve_problem(prob, frq)
            X, report = solver.solve_one(solve_prob, guess, 1.0)
            return X if report.converged else None

        n = max(1, int(steps))
        if mode == "adaptive":
            guess = parent_X
            t, h = 0.0, 1.0 / n
            while t < 1.0 - 1e-9:
                nt = min(1.0, t + h)
                cur = parent_current + nt * (target_current - parent_current)
                frq = parent_freq + nt * (target_freq - parent_freq)
                nxt = step_to(cur, frq, guess)
                if nxt is None:
                    h *= 0.5
                    if h < 1.0 / 64.0:
                        return None
                    continue
                guess, t = nxt, nt
                h = min(1.0 / n, h * 1.5)
            return guess

        # Fixed paths. Build the (fraction-of-current, fraction-of-freq) schedule.
        fracs = [(k + 1) / n for k in range(n)]
        if mode == "freq_first":
            path = [(parent_current, parent_freq + fr * (target_freq - parent_freq)) for fr in fracs]
            path += [(parent_current + fr * (target_current - parent_current), target_freq) for fr in fracs]
        elif mode == "power_first":
            path = [(parent_current + fr * (target_current - parent_current), parent_freq) for fr in fracs]
            path += [(target_current, parent_freq + fr * (target_freq - parent_freq)) for fr in fracs]
        else:  # diagonal
            path = [(parent_current + fr * (target_current - parent_current),
                     parent_freq + fr * (target_freq - parent_freq)) for fr in fracs]

        guess = parent_X
        for cur, frq in path:
            nxt = step_to(cur, frq, guess)
            if nxt is None:
                return None
            guess = nxt
        return guess

    def solve_power_substep(
        self,
        freq_ghz: float,
        from_X: np.ndarray,
        from_current: float,
        to_current: float,
        *,
        init_db: float = 0.1,
        min_db: float = 0.005,
        deadline_s: float = 120.0,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Adaptive natural-parameter continuation along the map power axis.

        Walk the pump current from ``from_current`` (a converged state) to
        ``to_current`` (the failed target cell), warm-starting one full-scale
        Newton solve per micro-step. The step is measured in dBm (geometric in
        current: ``I *= 10**(step_db/20)``) so the schedule matches the physical
        gain-lobe spacing; it grows x1.5 on success and halves on failure. When
        the step must shrink below ``min_db`` the branch has a step-independent
        stall at that power (a numerical/fold boundary), and this returns
        ``None`` -- distinct from a coarse-grid miss, which recovers here.

        The returned ``X`` (retained-shape, like every chained warm state) is a
        strong guess for the real target solve, not the written solution: the
        caller re-runs ``solve_point`` from it so gain + files are produced by
        the normal path. Bounded by ``deadline_s`` wall time.
        """
        info: dict[str, Any] = {
            "reached_target": False, "substeps": 0, "min_step_db": init_db,
            "terminal_reason": "", "last_current": from_current,
        }
        if to_current <= from_current or from_X is None:
            info["terminal_reason"] = "noop"
            return None, info
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())
        # dBm distance is +20*log10(I2/I1); step geometrically in current.
        total_db = 20.0 * math.log10(to_current / from_current)
        t0 = time.perf_counter()
        guess = from_X
        cur = from_current
        done_db = 0.0            # dBm advanced from from_current
        step_db = min(init_db, total_db)
        while done_db < total_db - 1e-9:
            if time.perf_counter() - t0 > deadline_s:
                info["terminal_reason"] = "deadline"
                break
            trial_db = min(done_db + step_db, total_db)
            trial_cur = from_current * (10.0 ** (trial_db / 20.0))
            prob, _basis, _omega = self._build_problem(freq_ghz, trial_cur)
            solve_prob = self._make_solve_problem(prob, freq_ghz)
            X, report = solver.solve_one(solve_prob, guess, 1.0)
            info["substeps"] += 1
            if report.converged:
                guess, cur, done_db = X, trial_cur, trial_db
                info["last_current"] = cur
                step_db = min(init_db, step_db * 1.5)
            else:
                step_db *= 0.5
                info["min_step_db"] = min(info["min_step_db"], step_db)
                if step_db < min_db:
                    info["terminal_reason"] = "step_floor"
                    break
        if done_db >= total_db - 1e-9:
            info["reached_target"] = True
            info["terminal_reason"] = "reached"
            return guess, info
        return None, info

    def trace_column_arclength(
        self,
        freq_ghz: float,
        reference_current: float,
        X0: np.ndarray,
        current0: float,
        X1: np.ndarray,
        current1: float,
        targets: list[tuple[int, float]],
    ) -> tuple[dict[int, list[np.ndarray]], dict]:
        """Trace once from two map states and interpolate target-current crossings."""
        full_problem, _basis, _omega = self._build_problem(freq_ghz, reference_current)
        problem = self._make_solve_problem(full_problem, freq_ghz)
        solver = exp08.HarmonicNewtonKrylovSolver(self._settings())
        points, info = solver.trace_arclength_from_two_points(
            problem,
            X0,
            current0 / reference_current,
            X1,
            current1 / reference_current,
            ds=self.args.column_arclength_ds,
            max_steps=self.args.column_arclength_max_steps,
            max_wall_s=self.args.column_arclength_deadline_s,
        )
        guesses: dict[int, list[np.ndarray]] = {}
        for point_index, target_current in targets:
            target = target_current / reference_current
            for (Xa, la), (Xb, lb) in zip(points, points[1:]):
                if lb == la or (la - target) * (lb - target) > 0.0:
                    continue
                theta = (target - la) / (lb - la)
                if -1e-12 <= theta <= 1.0 + 1e-12:
                    guesses.setdefault(point_index, []).append(Xa + theta * (Xb - Xa))
        info["trace_points"] = len(points)
        info["target_crossings"] = sum(len(v) for v in guesses.values())
        return guesses, info

    def _gain(self, pump_dir: Path, gain_dir: Path, freq_ghz: float):
        a = self.args
        gain_dir.mkdir(parents=True, exist_ok=True)
        t_all = time.perf_counter()
        pump = exp09.load_pump(pump_dir, fallback_pump_freq_ghz=freq_ghz)
        ms = exp09.sideband_list(a.sidebands)
        max_ell = max(abs(m - q) for m in ms for q in ms)
        t0 = time.perf_counter()
        gamma_hat = exp09.compute_gamma_hat(
            circuit=self.ipm09, pump=pump, max_ell=max_ell, gamma_nt=a.gamma_nt,
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

        target_signal_ghz = signal_ghz_for(freq_ghz, a)
        g = None
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
            for _, fs, gg in items:
                if abs(float(fs) - target_signal_ghz) <= 1e-9:
                    g = gg
                    break

        if g is None:
            g = self._solve_signal(
                khat, khat_off_0, khat_big_base, pump.omega_p, target_signal_ghz
            )

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
        schur_part = None
        if a.signal_backend == "schur":
            key = (
                round(float(omega_p), 3),
                round(float(signal_ghz), 12),
                int(a.sidebands),
                int(self.source_idx),
                int(self.out_idx),
                "current_complex_c",
            )
            schur_part = self._signal_schur_part_cache.get(key)
            if schur_part is None:
                schur_part = exp09.build_signal_schur_partition(
                    self.ipm09, omega_p, signal_ghz, a.sidebands,
                    self.source_idx, self.out_idx,
                    loss_model="current_complex_c",
                )
                self._signal_schur_part_cache[key] = schur_part
                if len(self._signal_schur_part_cache) > self._signal_schur_cache_max:
                    self._signal_schur_part_cache.pop(next(iter(self._signal_schur_part_cache)))
        common = dict(
            circuit=self.ipm09, khat=khat, khat_off_0=khat_off_0,
            khat_big_base=khat_big_base, omega_p=omega_p, signal_ghz=signal_ghz,
            sidebands=a.sidebands, signal_m=0, idler_m=-2,
            source_index=self.source_idx, out_index=self.out_idx,
            source_current_a=1.0, source_port=a.source_port, out_port=a.out_port,
            z0_ohm=a.z0_ohm, loss_model="current_complex_c",
            linear_solver=a.signal_solver,
        )
        if a.signal_backend == "schur":
            return exp09.solve_gain_one_schur(
                **common, include_baselines=not a.skip_baselines,
                schur_part=schur_part)
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
        "pump_failure_reason": "skipped after consecutive pump failures in column",
    }
    row.update({k: None for k in _SKIP_NONE_FIELDS})
    return row


def continuation_failure_is_fold_evidence(row: dict[str, Any]) -> bool:
    """Recognize a failed seed that explored continuation far enough to count.

    A first-step Newton failure is ambiguous and may just be a poor seed. Once
    adaptive continuation has attempted multiple lambda values but still did
    not reach full drive, repeated fail-fast failures are stronger local
    evidence that the column crossed the accessible harmonic-balance branch.
    """
    if row.get("status") == "PASS":
        return False
    method = row.get("pump_continuation_method")
    steps = row.get("pump_continuation_steps")
    return (
        method in {"adaptive_secant", "adaptive_copy", "adaptive_tangent", "affine"}
        and row.get("pump_continuation_reached_target") is False
        and isinstance(steps, (int, float))
        and int(steps) >= 2
    )


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
    initial_seed = getattr(engine.args, "initial_pump_dir", None)
    if initial_seed and len(by_col) != 1:
        raise ValueError(
            "--initial-pump-dir currently supports exactly one frequency column"
        )
    for j in sorted(by_col):
        column = sorted(by_col[j], key=lambda p: p.power_dbm)
        prev_X: np.ndarray | None = None
        # Last two converged (injected_current, X) for the secant predictor.
        last_good_X: np.ndarray | None = None
        last_good_cur: float | None = None
        prevprev_X: np.ndarray | None = None
        prevprev_cur: float | None = None
        arclength_guesses: dict[int, list[np.ndarray]] = {}
        verified_fold = False
        consec_fail = 0  # consecutive non-converged points at increasing power
        if initial_seed and column:
            seed_path = Path(initial_seed)
            try:
                loaded_X, _ = pump_basis.load_pump_basis_from_solution(seed_path)
                prev_X = loaded_X
                last_good_X = loaded_X
                seed_power = getattr(engine.args, "initial_pump_power_dbm", None)
                if seed_power is None:
                    raise ValueError(
                        "--initial-pump-power-dbm is required with "
                        "--initial-pump-dir"
                    )
                last_good_cur = dbm_to_peak_current_a(
                    float(seed_power),
                    attenuation_db=attenuation_db_for(
                        column[0].pump_freq_ghz, engine.args
                    ),
                    z0_ohm=engine.args.z0_ohm,
                ) * scale
                print(
                    f"[warm] fp={column[0].pump_freq_ghz:.6g} GHz "
                    f"initial seed={seed_path}",
                    flush=True,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"invalid --initial-pump-dir {seed_path}: {exc}"
                ) from exc
        for idx, point in enumerate(column):
            cur = point.current_a * scale
            base_X = prev_X if not fail_fast else last_good_X
            if idx == 0 and last_good_cur is None:
                # The supplied seed is at the first target power. It is a
                # direct Newton initial guess, not a secant history point.
                base_X = last_good_X
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

            # Traced fresh on every failing cell that has a valid seed pair --
            # no per-column "once ever" lock, so one cell's crossing-less
            # trace never permanently strands later cells. Each attempt is
            # bounded by --inproc-solve-deadline-s, so retrying stays cheap.
            if (
                row["status"] != "PASS"
                and getattr(engine.args, "column_arclength_recovery", False)
                and prevprev_X is not None
                and prevprev_cur is not None
                and last_good_X is not None
                and last_good_cur is not None
            ):
                reference_current = column[-1].current_a * scale
                targets = [
                    (p.index, p.current_a * scale)
                    for p in column[idx:]
                ]
                arclength_guesses, arc_info = engine.trace_column_arclength(
                    point.pump_freq_ghz,
                    reference_current,
                    prevprev_X,
                    prevprev_cur,
                    last_good_X,
                    last_good_cur,
                    targets,
                )
                verified_fold = verified_fold or bool(arc_info.get("fold_lambdas"))
                if arc_info.get("fold_lambdas"):
                    row["pump_column_arclength_fold_lambda"] = float(
                        arc_info["fold_lambdas"][0]
                    )
                    row["pump_column_arclength_terminal_reason"] = arc_info.get(
                        "terminal_reason"
                    )
                print(
                    f"[arclength] fp={point.pump_freq_ghz:.6g} GHz "
                    f"steps={arc_info.get('steps')} points={arc_info.get('trace_points')} "
                    f"folds={arc_info.get('fold_lambdas')} "
                    f"crossings={arc_info.get('target_crossings')} "
                    f"reason={arc_info.get('terminal_reason')}",
                    flush=True,
                )

            if row["status"] != "PASS" and point.index in arclength_guesses:
                # Cap to the first 2 target-crossing guesses: each is a full
                # Newton solve, and a stiff branch can have several crossings.
                for arc_guess in arclength_guesses[point.index][:2]:
                    arc_row, arc_X = engine.solve_point(
                        point, pass_dir, mode="warm", warm_X=arc_guess,
                    )
                    if arc_row["status"] == "PASS":
                        row, X = arc_row, arc_X
                        pred_tag = f"{pred_tag}->arclength"
                        break

            # Overshoot guard: a bad extrapolation past the fold -> retry once
            # from the plain warm start before paying the reseed. Fail-fast
            # mode intentionally pays only one solve per cell.
            if row["status"] != "PASS" and use_secant and not fail_fast:
                row, X = engine.solve_point(point, pass_dir, mode="warm", warm_X=base_X)
                pred_tag = "secant_fallback"

            retried = False
            if row["status"] != "PASS" and mode == "warm" and not fail_fast:
                row, X = engine.solve_point(point, pass_dir, mode="seed", warm_X=None)
                retried = row["status"] == "PASS"

            # Adaptive power-substep recovery: the coarse power step can miss a
            # gain-lobe crest that a finer natural continuation crosses (see
            # diagnostics/2c_measurement_comparison). Walk from the last
            # converged state up to this target in adaptive dBm micro-steps; a
            # step-independent stall (min_db floor) is a real numerical/fold
            # boundary and leaves the cell FAILED so the fold short-circuit can
            # act on it.
            if (
                row["status"] != "PASS"
                and getattr(engine.args, "column_power_substep", False)
                and not fail_fast
                and last_good_X is not None
                and last_good_cur is not None
                and cur > last_good_cur
            ):
                X_sub, sub_info = engine.solve_power_substep(
                    point.pump_freq_ghz, last_good_X, last_good_cur, cur,
                    init_db=engine.args.column_power_substep_init_db,
                    min_db=engine.args.column_power_substep_min_db,
                    deadline_s=engine.args.column_power_substep_deadline_s,
                )
                row["pump_power_substep_substeps"] = sub_info["substeps"]
                row["pump_power_substep_terminal_reason"] = sub_info["terminal_reason"]
                if X_sub is not None:
                    sub_row, sub_X = engine.solve_point(
                        point, pass_dir, mode="warm", warm_X=X_sub)
                    if sub_row["status"] == "PASS":
                        row, X = sub_row, sub_X
                        pred_tag = f"{pred_tag}->substep"
                elif sub_info["terminal_reason"] == "step_floor":
                    # Step-independent stall -> treat as fold evidence so the
                    # per-column short-circuit can stop retrying above it.
                    row["pump_power_substep_stall_dbm"] = point.power_dbm
                    verified_fold = True

            row["warm_retry_reseed"] = retried
            row["pump_predictor"] = pred_tag
            verified_fold = verified_fold or continuation_failure_is_fold_evidence(row)
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
            if (
                patience > 0
                and verified_fold
                and consec_fail >= patience
                and idx + 1 < len(column)
            ):
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
# Traversal orchestrator (inter-cell method suite: Phases 1-3)
# =============================================================================

from twpa_solver.pump import predictors as _predictors  # noqa: E402


def _grid_dims(points: list[GridPoint]) -> tuple[int, int]:
    return max(p.i_power for p in points) + 1, max(p.j_freq for p in points) + 1


def _traversal_order(points: list[GridPoint], strategy: str, direction: str
                     ) -> list[GridPoint]:
    """Solve order for a traversal strategy (list of GridPoints).

    ``column``/``nearest`` sort column-major (low->high power within a column);
    ``backbone`` solves the lowest-power frequency row first then each column
    upward; ``serpentine`` alternates power direction per column; ``floodfill``
    is a Prim (cheapest-neighbour) order from a central low-power seed.
    """
    n_power, n_freq = _grid_dims(points)
    by_ij = {(p.i_power, p.j_freq): p for p in points}

    def col_order(js: list[int]) -> list[int]:
        if direction == "rtl":
            return sorted(js, reverse=True)
        if direction == "center_out":
            mid = (n_freq - 1) / 2.0
            return sorted(js, key=lambda j: abs(j - mid))
        if direction == "two_ended":
            lo, hi = sorted(js), sorted(js, reverse=True)
            out: list[int] = []
            for a, b in zip(lo, hi):
                out.append(a)
                if b != a and b not in out:
                    out.append(b)
            return [j for j in out if j in set(js)]
        return sorted(js)  # ltr

    all_js = sorted({p.j_freq for p in points})

    if strategy == "backbone":
        order: list[GridPoint] = []
        js = col_order(all_js)
        for j in js:  # backbone row (lowest power present in the column)
            col = sorted((p for p in points if p.j_freq == j), key=lambda p: p.i_power)
            if col:
                order.append(col[0])
        for j in js:  # each column upward from its backbone cell
            col = sorted((p for p in points if p.j_freq == j), key=lambda p: p.i_power)
            order.extend(col[1:])
        return order

    if strategy == "serpentine":
        order = []
        for k, j in enumerate(sorted(all_js)):
            col = sorted((p for p in points if p.j_freq == j), key=lambda p: p.i_power)
            order.extend(col if k % 2 == 0 else list(reversed(col)))
        return order

    if strategy == "floodfill":
        import heapq
        powers = sorted({p.i_power for p in points})
        rangeP = max(1, n_power - 1)
        rangeF = max(1, n_freq - 1)
        start = by_ij.get((powers[0], n_freq // 2)) or points[0]
        visited: set[tuple[int, int]] = set()
        order = []
        heap: list[tuple[float, int, int]] = [(0.0, start.i_power, start.j_freq)]
        while heap:
            _cost, i, j = heapq.heappop(heap)
            if (i, j) in visited or (i, j) not in by_ij:
                continue
            visited.add((i, j))
            order.append(by_ij[(i, j)])
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if (ni, nj) in by_ij and (ni, nj) not in visited:
                    cost = abs(di) / rangeP + abs(dj) / rangeF
                    heapq.heappush(heap, (cost, ni, nj))
        return order

    # column / nearest: column-major, ascending power.
    return sorted(points, key=lambda p: (p.j_freq, p.i_power))


def _nearest_solved(i: int, j: int, solved: dict, n_power: int, n_freq: int):
    """Nearest already-solved cell to (i,j) by normalised grid distance."""
    best = None
    best_d = float("inf")
    for (si, sj) in solved:
        d = abs(si - i) / max(1, n_power - 1) + abs(sj - j) / max(1, n_freq - 1)
        if d < best_d:
            best_d, best = d, (si, sj)
    return best


def _build_candidates(
    point: GridPoint, cur_t: float, solved: dict, n_power: int, n_freq: int,
) -> dict[str, np.ndarray | None]:
    """Predictor candidate guesses for a target cell from solved neighbours."""
    i, j = point.i_power, point.j_freq
    P, f = point.power_dbm, point.pump_freq_ghz

    def X(ii, jj):
        c = solved.get((ii, jj))
        return c["X"] if c else None

    def cur(ii, jj):
        c = solved.get((ii, jj))
        return c["current"] if c else None

    def frq(ii, jj):
        c = solved.get((ii, jj))
        return c["freq"] if c else None

    cands: dict[str, np.ndarray | None] = {}
    # copy: power parent, else nearest solved.
    parent = X(i - 1, j)
    if parent is None:
        nb = _nearest_solved(i, j, solved, n_power, n_freq)
        parent = solved[nb]["X"] if nb else None
    cands["copy"] = None if parent is None else _predictors.copy_predictor(parent)
    cands["power_secant"] = _predictors.axis_secant(
        X(i - 2, j), X(i - 1, j), cur(i - 2, j), cur(i - 1, j), cur_t)
    cands["freq_secant"] = _predictors.axis_secant(
        X(i, j - 2), X(i, j - 1), frq(i, j - 2), frq(i, j - 1), f)
    cands["corner"] = _predictors.corner_predictor(X(i, j - 1), X(i - 1, j), X(i - 1, j - 1))
    cands["diagonal"] = X(i - 1, j - 1)
    window = [(c["power"], c["freq"], c["X"]) for (si, sj), c in solved.items()
              if abs(si - i) <= 2 and abs(sj - j) <= 2]
    cands["plane"] = _predictors.plane_predictor(window, P, f)
    return cands


def _select_guess(
    point: GridPoint, cur_t: float, solved: dict, solve_problem, engine,
    n_power: int, n_freq: int, args: argparse.Namespace,
) -> tuple[np.ndarray | None, str, list[tuple[str, np.ndarray, float]]]:
    """Pick the initial guess for a cell per --predictor / --portfolio-policy.

    Returns (guess, tag, ranked) where ``ranked`` is the residual-sorted
    candidate list (non-empty only for the portfolio predictor; reused by the
    ranked recovery ladder). ``solve_problem`` is the Schur-reduced (or full)
    problem the cell is actually solved on, so candidate residuals match the
    chained warm-start state shape.
    """
    cands = _build_candidates(point, cur_t, solved, n_power, n_freq)
    predictor = args.predictor
    if predictor == "portfolio":
        ranked = _predictors.rank_candidates(
            cands, lambda X: engine.residual_norm(solve_problem, X))
        if not ranked:
            return None, "seed", []
        return ranked[0][1], f"portfolio:{ranked[0][0]}", ranked
    guess = cands.get(predictor)
    if guess is None:  # fall back to copy of best available parent
        guess = cands.get("copy")
        return guess, ("copy" if guess is not None else "seed"), []
    return guess, predictor, []


def _attempt(engine, point, pass_dir, prebuilt, *, mode, warm_X):
    row, X = engine.solve_point(point, pass_dir, mode=mode, warm_X=warm_X, prebuilt=prebuilt)
    return row, X, row["status"] == "PASS"


def _recover(
    engine, point, pass_dir, prebuilt, solve_problem, cur_t, solved,
    n_power, n_freq, ranked, args, failed_row, failed_X,
) -> tuple[dict, np.ndarray | None, bool, str]:
    """Recovery + fold-policy rescue ladder for a failed cell.

    Runs the --recovery ladder, then any extra --fold-policy attempt, and
    returns (row, X, converged, tag). ``converged`` False here means the cell is
    a genuine fold/skip candidate.
    """
    i, j = point.i_power, point.j_freq
    parent_i = solved.get((i - 1, j)) or solved.get((i, j - 1)) or solved.get((i - 1, j - 1))
    last_row, last_X = failed_row, failed_X

    def bridge_from(cell) -> tuple[dict, np.ndarray | None, bool] | None:
        if cell is None:
            return None
        guess = engine.solve_bridge(
            cell["X"], cell["current"], cell["freq"], cur_t, point.pump_freq_ghz,
            steps=args.bridge_steps, mode=args.bridge_mode)
        if guess is None:
            return None
        row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=guess)
        return (row, X, ok) if ok else None

    recovery = args.recovery
    if recovery == "alt_parent":
        for cell in (solved.get((i - 1, j)), solved.get((i, j - 1)), solved.get((i - 1, j - 1))):
            if cell is None:
                continue
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=cell["X"])
            if ok:
                return row, X, True, "alt_parent"
            last_row, last_X = row, X
    elif recovery == "bridge":
        res = bridge_from(parent_i)
        if res:
            return res[0], res[1], True, "bridge"
    elif recovery == "ladder":
        # ranked[0] was already attempted as the initial portfolio guess.
        for _name, guess, _rho in (ranked[1:] if ranked else []):
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=guess)
            if ok:
                return row, X, True, "ladder_predictor"
            last_row, last_X = row, X
        res = bridge_from(parent_i)
        if res:
            return res[0], res[1], True, "ladder_bridge"

    # Fold-policy extra rescue before counting toward the skip.
    fp = args.fold_policy
    if fp in ("cross_axis", "combined"):
        cell = solved.get((i, j - 1))
        if cell is not None:
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=cell["X"])
            if ok:
                return row, X, True, "cross_axis"
            last_row, last_X = row, X
    if fp in ("bridge_gate", "combined"):
        res = bridge_from(parent_i)
        if res:
            return res[0], res[1], True, "fold_bridge"
    if fp == "arclength":
        # Round the fold: pseudo-arclength from lambda=0 to full drive, then a
        # warm target solve from the arclength state.
        solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
        X_arc, _lam, info = solver.solve_arclength(
            solve_problem, solve_problem.zeros(), 0.0, ds=0.1, target_lam=1.0,
            max_wall_s=engine.args.inproc_solve_deadline_s)
        if info.get("reached_target"):
            row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="warm", warm_X=X_arc)
            if ok:
                return row, X, True, "arclength"
            last_row, last_X = row, X

    # Fail-fast still permits the explicitly selected cheap recovery policy,
    # but does not pay for a fresh continuation after those attempts fail.
    if args.inproc_fail_fast:
        return last_row, last_X, False, "fail_fast"

    # Final fallback: fresh linear_phasor + adaptive reseed.
    row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode="seed", warm_X=None)
    return row, X, ok, "reseed"


def run_fold_follow(engine: InProcessEngine, freqs: np.ndarray, outdir: Path,
                    args: argparse.Namespace) -> None:
    """Trace the fold power vs frequency with pseudo-arclength -> fold_curve.csv.

    At each frequency, build the pump problem at the maximum-power reference
    current and run the arclength fold locator; the fold ``lambda`` scales the
    reference current, giving a fold current and thus a fold power (dBm).
    """
    from twpa_solver.pump.solver import fold_power
    scale = args.pump_current_jc_scale
    solver = exp08.HarmonicNewtonKrylovSolver(engine._settings())
    rows: list[dict[str, Any]] = []
    for f in freqs:
        f = float(f)
        ref_phys = dbm_to_peak_current_a(
            args.pump_power_max_dbm, attenuation_db=attenuation_db_for(f, args),
            z0_ohm=args.z0_ohm)
        ref_injected = ref_phys * scale
        full_problem, _basis, _omega = engine._build_problem(f, ref_injected)
        # Solve on the Schur-reduced problem for speed (constant retained shape).
        problem = engine._make_solve_problem(full_problem, f)
        lam_fold = fold_power(solver, problem, max_steps=120)
        fold_dbm = (peak_current_to_power_dbm(lam_fold * ref_injected / scale, f, args)
                    if lam_fold is not None else None)
        rows.append({"pump_freq_ghz": f, "fold_lambda": lam_fold,
                     "fold_power_dbm": fold_dbm})
        print(f"[fold-follow] fp={f:.4f} GHz fold_lambda={lam_fold} "
              f"fold_power_dbm={fold_dbm}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "fold_curve.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pump_freq_ghz", "fold_lambda", "fold_power_dbm"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def run_map_traversal(
    points: list[GridPoint], pass_dir: Path, engine: InProcessEngine,
) -> list[dict[str, Any]]:
    """Warm pass with a pluggable traversal / predictor / recovery / fold policy.

    Keeps one in-process ``solved[(i,j)]`` state store shared across BOTH axes,
    so frequency-crossing methods (backbone, nearest, corner, plane, ...) can
    warm-start from converged neighbours in either direction. Requires a single
    process (enforced by the caller via --frequency-chunk-size 0).
    """
    a = engine.args
    n_power, n_freq = _grid_dims(points)
    scale = a.pump_current_jc_scale
    order = _traversal_order(points, a.traversal, a.backbone_direction)
    solved: dict[tuple[int, int], dict] = {}
    skip: set[tuple[int, int]] = set()
    col_fail: dict[int, int] = {}
    patience = int(getattr(a, "fold_skip_patience", 0))
    rows: list[dict[str, Any]] = []
    total = len(points)
    done = 0

    for point in order:
        i, j = point.i_power, point.j_freq
        if (i, j) in skip:
            rows.append(past_fold_skip_row(point))
            done += 1
            continue
        cur_t = point.current_a * scale
        prebuilt = engine.build_problem_for(point)
        solve_problem = engine._make_solve_problem(prebuilt[0], point.pump_freq_ghz)
        guess, tag, ranked = _select_guess(
            point, cur_t, solved, solve_problem, engine, n_power, n_freq, a)
        mode = "warm" if guess is not None else "seed"
        row, X, ok = _attempt(engine, point, pass_dir, prebuilt, mode=mode, warm_X=guess)

        if not ok and a.predictor == "portfolio" and a.portfolio_policy == "ranked":
            for name, candidate, _rho in ranked[1:]:
                row, X, ok = _attempt(
                    engine, point, pass_dir, prebuilt, mode="warm", warm_X=candidate,
                )
                tag = f"{tag}->ranked:{name}"
                if ok:
                    break

        if not ok:
            row, X, ok, rtag = _recover(
                engine, point, pass_dir, prebuilt, solve_problem, cur_t, solved,
                n_power, n_freq, ranked, a, row, X)
            tag = f"{tag}->{rtag}"

        row["warm_retry_reseed"] = "reseed" in tag
        row["pump_predictor"] = tag
        rows.append(row)
        done += 1
        print(f"[trav {done}/{total}] P={point.power_dbm:.4g} dBm "
              f"fp={point.pump_freq_ghz:.4g} GHz {a.traversal}:{tag} "
              f"status={row['status']} gain={row.get('gain_db')} "
              f"newton={row.get('pump_newton_total')}", flush=True)

        if ok and X is not None:
            solved[(i, j)] = {"X": X, "current": cur_t, "freq": point.pump_freq_ghz,
                              "power": point.power_dbm}
            col_fail[j] = 0
        else:
            col_fail[j] = col_fail.get(j, 0) + 1
            # Per-column fold short-circuit: skip higher-power cells in this
            # column once patience consecutive fails accrue at increasing power.
            if patience > 0 and col_fail[j] >= patience:
                for ii in range(i + 1, n_power):
                    skip.add((ii, j))

    rows.sort(key=lambda r: r["point_index"])
    return rows


def uses_traversal_orchestrator(args: argparse.Namespace) -> bool:
    """Return whether generic traversal/recovery semantics are requested."""
    return (
        args.traversal != "column"
        or args.recovery != "reseed"
        or args.fold_policy != "patience"
    )


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
        "pump_failure_reason", "gain_failure_reason",
        "gain_db", "gain_vs_off_db",
        "gain_vs_pumpdiag_db", "signal_ghz", "linear_rel_residual",
        "pump_runtime_s", "pump_wall_runtime_s", "pump_setup_runtime_s",
        "pump_schur_setup_runtime_s", "pump_solve_wall_runtime_s",
        "pump_write_runtime_s", "pump_factor_runtime_s",
        "pump_preconditioner_assembly_runtime_s",
        "pump_preconditioner_numeric_factor_runtime_s", "pump_newton_total",
        "pump_gmres_total", "pump_coeff_rel", "pump_time_rel", "pump_branch_current_max",
        "pump_continuation_method", "pump_continuation_steps",
        "pump_continuation_reached_target", "pump_continuation_fold_lambda",
        "pump_continuation_runtime_s",
        "pump_column_arclength_fold_lambda",
        "pump_column_arclength_terminal_reason",
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
        "attenuation_model": ("flat" if args.attenuation_db is not None
                              else "loss_A10 c + a*sqrt(f) + b*f"),
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

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["cold", "warmstart", "both"], default="warmstart")
    p.add_argument("--executor", choices=["subprocess", "inprocess"], default="inprocess",
                   help="inprocess runs pump+gain in this process (no per-point import "
                   "tax); numerics are identical to the subprocess path.")
    p.add_argument("--inproc-gmres-maxiter", type=int, default=80,
                   help="GMRES outer restart cycles per Newton step (x gmres_restart=60 inner). "
                   "A low value (e.g. 4) bounds the per-Newton cost so over-fold solves don't "
                   "grind thousands of inner iterations; warm-start steps converge in <1 cycle.")
    p.add_argument("--inproc-solve-deadline-s", "--inproc-solve-deadline",
                   dest="inproc_solve_deadline_s", type=float, default=0.0,
                   help="Per-solve wall-time budget (s) for the in-process path; 0 disables. "
                   "Bounds stiff over-fold solves near the fold.")
    p.add_argument("--inproc-max-newton", type=int, default=16,
                   help="Max Newton iterations per in-process solve. A small cap (e.g. 10) "
                   "makes over-fold points fail fast; warm-start neighbours converge in few.")
    p.add_argument(
        "--inproc-fallback-fixed-steps",
        type=int,
        default=20,
        help="Fixed ladder length after adaptive continuation gives up. Lower this "
        "for bounded recovery campaigns; the fixed method itself still uses "
        "--continuation-steps.",
    )
    p.add_argument(
        "--inproc-continuation-deadline-s",
        type=float,
        default=0.0,
        help="Total wall-time budget for adaptive/affine continuation, excluding "
        "the final target solve. 0 inherits --inproc-solve-deadline-s.",
    )
    p.add_argument("--inproc-fail-fast", action="store_true",
                   help="In-process warm pass: skip reseed/fallback recovery on a failed "
                   "point and keep warm-starting from the last converged neighbour, so "
                   "over-fold points fail in ~one stalled solve. For high-power fold maps.")
    p.add_argument(
        "--initial-pump-dir",
        type=Path,
        default=None,
        help="Optional verified pump directory containing pump_solution.npz. "
        "Use it as the first warm-start state of each frequency column.",
    )
    p.add_argument(
        "--initial-pump-power-dbm",
        type=float,
        default=None,
        help="Power coordinate of --initial-pump-dir, required for "
        "secant/pseudo-arclength continuation.",
    )
    p.add_argument("--fold-skip-patience", type=int, default=0,
                   help="Per-column fold short-circuit (in-process warm pass): after "
                   "skip the remaining higher-power cells only after the optional "
                   "pseudo-arclength recovery has reported a turning point (marked "
                   "SKIP_PAST_FOLD, gain NaN). A failed target solve alone is never "
                   "treated as a fold. 0 disables skipping (default).")
    p.add_argument(
        "--column-arclength-recovery",
        action="store_true",
        help="On every failed cell in each legacy power column (not just the "
        "first), trace one scaled pseudo-arclength branch from the last two "
        "converged states and use its target-power crossings (capped to 2 "
        "guesses) as Newton recovery guesses. No per-column lock: a cell "
        "whose own trace found no crossing does not block later cells from "
        "trying again once bounded by --inproc-solve-deadline-s.",
    )
    p.add_argument("--column-arclength-ds", type=float, default=0.02)
    p.add_argument("--column-arclength-max-steps", type=int, default=80)
    p.add_argument(
        "--column-arclength-deadline-s",
        type=float,
        default=180.0,
        help="Total wall-time budget for each column pseudo-arclength trace. "
        "Separate from the per-target Newton deadline; 0 disables the trace "
        "deadline.",
    )
    p.add_argument(
        "--column-power-substep",
        action="store_true",
        help="On a failed warm cell, recover by adaptive natural-parameter "
        "continuation along the power axis from the last converged state: "
        "walk up in adaptive dBm micro-steps, warm-starting each. Crosses "
        "gain-lobe crests the coarse power grid misses; a step-independent "
        "stall (min-db floor) is recorded as a numerical/fold boundary "
        "rather than retried. See diagnostics/2c_measurement_comparison.",
    )
    p.add_argument(
        "--column-power-substep-init-db",
        type=float,
        default=0.1,
        help="Initial (and maximum) dBm micro-step for --column-power-substep; "
        "grows x1.5 on success, halves on failure.",
    )
    p.add_argument(
        "--column-power-substep-min-db",
        type=float,
        default=0.005,
        help="Minimum dBm micro-step for --column-power-substep. Below this the "
        "branch is declared to have a step-independent stall (fold/numerical "
        "boundary) and the cell is left failed.",
    )
    p.add_argument(
        "--column-power-substep-deadline-s",
        type=float,
        default=120.0,
        help="Per-target wall-time budget for the --column-power-substep walk.",
    )
    p.add_argument("--inproc-preconditioner",
                   choices=["mean_tangent", "real_coupled", "real_coupled_fast",
                            "spectral_coupled", "linear"],
                   default="real_coupled_fast",
                   help="Preconditioner for the in-process pump solve. mean_tangent "
                   "(default) is cheapest for small warm-start steps; real_coupled "
                   "cuts GMRES iters but its full-Jacobian LU is costlier per Newton.")
    p.add_argument("--inproc-pump-backend", choices=["full", "schur_cpu_mt"],
                   default="schur_cpu_mt",
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
                   default="secant",
                   help="In-process warm pass: build the next power point's initial "
                   "guess by extrapolating along the pump-current axis from the last "
                   "two converged solutions (secant), instead of copying the previous "
                   "solution. Cuts Newton steps near the fold where the state moves "
                   "fast with power. Physics unchanged (initial guess only); a failed "
                   "predicted solve falls back to the plain warm start.")
    # --- Inter-cell method suite (opt-in; default reproduces the column pass) ---
    # See docs/reports/pump_map_continuation_methods.tex + the campaign matrix.
    # Frequency-crossing traversals require a single process, so they force
    # --frequency-chunk-size 0 (they share one in-process solved-state store).
    p.add_argument("--traversal",
                   choices=["column", "backbone", "nearest", "serpentine", "floodfill"],
                   default="column",
                   help="Map traversal / warm-start order. 'column' (default) is "
                   "the legacy per-frequency-column low->high power pass with no "
                   "cross-column warm state. The others reuse converged cells "
                   "across BOTH axes (single process only).")
    p.add_argument("--backbone-direction",
                   choices=["ltr", "rtl", "center_out", "two_ended"],
                   default="center_out",
                   help="For --traversal backbone: order in which the lowest-power "
                   "frequency backbone row is solved before launching each upward "
                   "power column from it.")
    p.add_argument("--predictor",
                   choices=["copy", "power_secant", "freq_secant", "corner",
                            "plane", "portfolio"],
                   default="power_secant",
                   help="Inter-cell initial-guess predictor for the traversal "
                   "orchestrator (ignored by --traversal column, which uses "
                   "--inproc-fold-predictor). 'portfolio' ranks several by target "
                   "residual.")
    p.add_argument("--portfolio-policy", choices=["best", "ranked"], default="best",
                   help="--predictor portfolio: 'best' tries only the lowest-residual "
                   "candidate; 'ranked' tries candidates in ascending-residual order "
                   "until one converges.")
    p.add_argument("--recovery",
                   choices=["none", "reseed", "alt_parent", "bridge", "ladder"],
                   default="reseed",
                   help="Failed-cell recovery for the traversal orchestrator. "
                   "'none' keeps the initial failure; 'reseed' (legacy) does a "
                   "fresh linear_phasor+adaptive solve; "
                   "'alt_parent' first retries from power/freq/diagonal parents; "
                   "'bridge' continues from the best parent along (P,f); 'ladder' "
                   "residual-ranks parents then bridges from the best.")
    p.add_argument("--bridge-steps", type=int, default=4,
                   help="Sub-steps for bridge continuation (recovery=bridge/ladder "
                   "and --fold-policy bridge_gate/combined).")
    p.add_argument("--bridge-mode",
                   choices=["diagonal", "freq_first", "power_first", "adaptive"],
                   default="adaptive",
                   help="Path from parent (P0,f0) to target (P1,f1) for bridge "
                   "continuation.")
    p.add_argument("--fold-policy",
                   choices=["patience", "cross_axis", "bridge_gate", "combined", "arclength"],
                   default="patience",
                   help="When a failed cell counts toward the per-column fold "
                   "short-circuit. 'patience' (legacy) counts every fail; the "
                   "others require cross-axis / bridge / full recovery to also fail "
                   "first; 'arclength' rounds the fold with pseudo-arclength.")
    p.add_argument("--inproc-continuation",
                   choices=["fixed", "adaptive_copy", "adaptive_secant",
                            "adaptive_tangent", "affine", "ptc", "arclength"],
                   default="adaptive_secant",
                   help="Intra-cell continuation for seed/cold cells (solver.py). "
                   "fixed is the 20-step reference; adaptive_copy is natural-parameter "
                   "continuation without prediction; adaptive_secant (default) uses "
                   "the previous two lambda states; adaptive_tangent uses the exact "
                   "lambda tangent; affine sizes steps from corrector contraction; "
                   "ptc is pseudo-transient; arclength augments state and lambda.")
    p.add_argument("--inproc-arclength-ds", type=float, default=0.1)
    p.add_argument("--inproc-arclength-max-steps", type=int, default=80)
    p.add_argument("--fold-follow", action="store_true",
                   help="Diagnostic: trace the fold power vs frequency with "
                   "pseudo-arclength and write fold_curve.csv; no gain map is run.")
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "exp10_pump_map_warmstart")
    p.add_argument("--circuit-dir", "--ipm-dir", dest="circuit_dir", type=Path, default=ROOT / "outputs" / "ipm_python_design")

    p.add_argument("--n-power", type=int, default=50)
    p.add_argument("--n-frequency", type=int, default=50)
    p.add_argument("--frequency-chunk-size", type=int, default=10,
                   help="Run frequency columns in separate worker processes of this "
                   "many columns each, then merge. 10 is the standard memory-safe "
                   "map behavior; 0 disables chunking.")
    p.add_argument(
        "--local-traversal-chunks",
        action="store_true",
        help="Allow non-column traversals to run as independent frequency-local "
        "chunks. This bounds native solver memory by restarting the process per "
        "chunk, at the cost of not sharing warm states across chunk boundaries.",
    )
    p.add_argument("--resume-chunks", action=argparse.BooleanOptionalAction, default=True,
                   help="With --frequency-chunk-size, skip chunk workers whose "
                   "map_points.csv/map_summary.json already look complete.")
    p.add_argument("--chunk-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--frequency-index-start", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--frequency-index-stop", type=int, default=None, help=argparse.SUPPRESS)
    # External power window. With the ~35 dB pump-band line loss + 50 ohm this
    # spans physical pump ~0.5..1.6 x median Ic; after the JC 2x scale the JTWPA
    # gain ridge runs from onset (~0 dB) up to ~12 dB near JC's 1.5 Ic point.
    p.add_argument("--pump-power-min-dbm", type=float, default=-30.0)
    p.add_argument("--pump-power-max-dbm", type=float, default=-20.0)
    p.add_argument("--pump-freq-min-ghz", type=float, default=7.0)
    p.add_argument("--pump-freq-max-ghz", type=float, default=8.0)
    p.add_argument("--attenuation-db", type=float, default=None,
                   help="Flat line attenuation (dB). If omitted, use the measured "
                   "loss_A10 frequency-dependent model c + a*sqrt(f) + b*f.")
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
    p.add_argument("--signal-spectrum", action=argparse.BooleanOptionalAction, default=True,
                   help="Solve a spectrum of signal frequencies per cell (see below); "
                   "writes map_spectrum.npz. Reuses exp09's khat conversion base so "
                   "each extra signal point is cheap.")
    p.add_argument("--signal-offset-start-mhz", type=float, default=100.0,
                   help="First |offset| from fp for the spectrum ladder (MHz).")
    p.add_argument("--signal-offset-step-mhz", type=float, default=500.0, #250
                   help="Spacing between spectrum offsets (MHz).")
    p.add_argument("--signal-offset-count-per-side", type=int, default=5,
                   help="Offsets per side; 5 -> 10 points (+/-100,+/-350,... MHz).")
    p.add_argument("--signal-workers", type=int, default=6,
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
    p.add_argument("--sidebands", type=int, default=6)
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
    p.add_argument(
        "--allow-superlu-fallback",
        action="store_true",
        help="Debug only: allow real_coupled_fast to fall back to SuperLU if PARDISO fails.",
    )
    p.add_argument(
        "--log-factor-backend",
        action="store_true",
        help="Print whether real_coupled_fast actually factors with PARDISO or SuperLU.",
    )

    # Production defaults for the standard gain-map workflow.
    p.set_defaults(
    )

    return p.parse_args(argv)


def build_points(args: argparse.Namespace) -> tuple[list[GridPoint], np.ndarray, np.ndarray]:
    powers = np.linspace(args.pump_power_min_dbm, args.pump_power_max_dbm, args.n_power)
    freqs = np.linspace(args.pump_freq_min_ghz, args.pump_freq_max_ghz, args.n_frequency)
    points: list[GridPoint] = []
    index = 0
    for i, power_dbm in enumerate(powers):
        for j, freq in enumerate(freqs):
            current = dbm_to_peak_current_a(
                float(power_dbm),
                attenuation_db=attenuation_db_for(float(freq), args),
                z0_ohm=args.z0_ohm,
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


_CHUNK_STRIP_VALUE_OPTS = {
    "--outdir",
    "--frequency-index-start",
    "--frequency-index-stop",
    "--frequency-chunk-size",
    "--n-frequency",
    "--pump-freq-min-ghz",
    "--pump-freq-max-ghz",
    "--gate-spotcheck",
}
_CHUNK_STRIP_FLAGS = {
    "--chunk-worker",
    "--overwrite",
    "--resume-chunks",
    "--no-resume-chunks",
}


def frequency_chunk_ranges(n_frequency: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return half-open frequency-column chunks."""
    n = int(n_frequency)
    size = int(chunk_size)
    if n <= 0:
        return []
    if size <= 0 or size >= n:
        return [(0, n)]
    return [(start, min(start + size, n)) for start in range(0, n, size)]


def _strip_chunk_driver_args(argv: list[str]) -> list[str]:
    """Remove parent/chunk-routing options before spawning a chunk worker."""
    cleaned: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in _CHUNK_STRIP_FLAGS:
            continue
        if token in _CHUNK_STRIP_VALUE_OPTS:
            skip_next = True
            continue
        if any(token.startswith(f"{opt}=") for opt in _CHUNK_STRIP_VALUE_OPTS):
            continue
        cleaned.append(token)
    return cleaned


def chunk_worker_command(
    base_argv: list[str],
    *,
    outdir: Path,
    n_frequency: int,
    pump_freq_min_ghz: float,
    pump_freq_max_ghz: float,
    overwrite: bool = False,
) -> list[str]:
    """Build the self-invocation used for one frequency-column chunk."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *_strip_chunk_driver_args(base_argv),
        "--chunk-worker",
        "--gate-spotcheck",
        "0",
        "--n-frequency",
        str(int(n_frequency)),
        "--pump-freq-min-ghz",
        f"{float(pump_freq_min_ghz):.12g}",
        "--pump-freq-max-ghz",
        f"{float(pump_freq_max_ghz):.12g}",
        "--outdir",
        str(outdir),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _expected_chunk_row_count(args: argparse.Namespace, start_col: int, stop_col: int) -> int:
    n_cols = max(0, int(stop_col) - int(start_col))
    pass_count = 2 if args.mode == "both" else 1
    return int(args.n_power) * n_cols * pass_count


def chunk_is_complete(
    chunk_dir: Path,
    args: argparse.Namespace,
    start_col: int,
    stop_col: int,
) -> bool:
    points_path = chunk_dir / "map_points.csv"
    summary_path = chunk_dir / "map_summary.json"
    if not points_path.exists() or not summary_path.exists():
        return False
    if args.signal_spectrum and args.mode in ("warmstart", "both") and not (chunk_dir / "map_spectrum.npz").exists():
        return False
    try:
        with points_path.open("r", encoding="utf-8", newline="") as f:
            n_rows = max(0, sum(1 for _ in f) - 1)
    except OSError:
        return False
    return n_rows == _expected_chunk_row_count(args, start_col, stop_col)


def read_chunk_rows(
    chunk_specs: list[tuple[Path, int, int]],
    *,
    global_n_frequency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cold_rows: list[dict[str, Any]] = []
    warm_rows: list[dict[str, Any]] = []
    for chunk_dir, start_col, _stop_col in chunk_specs:
        points_path = chunk_dir / "map_points.csv"
        with points_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cleaned = {k: _csv_value(v) for k, v in row.items()}
                pass_name = str(cleaned.pop("pass", ""))
                if "j_freq" in cleaned and "i_power" in cleaned:
                    global_j = int(start_col) + int(cleaned["j_freq"])
                    cleaned["j_freq"] = global_j
                    cleaned["point_index"] = int(cleaned["i_power"]) * int(global_n_frequency) + global_j
                if pass_name == "cold":
                    cold_rows.append(cleaned)
                elif pass_name == "warm":
                    warm_rows.append(cleaned)
    cold_rows.sort(key=lambda r: int(r["point_index"]))
    warm_rows.sort(key=lambda r: int(r["point_index"]))
    return cold_rows, warm_rows


def _csv_value(value: Any) -> Any:
    if value == "":
        return None
    if not isinstance(value, str):
        return value
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def merge_chunk_spectra(
    outpath: Path,
    chunk_specs: list[tuple[Path, int, int]],
    powers: np.ndarray,
    freqs: np.ndarray,
) -> bool:
    """Merge full-shape per-chunk spectrum cubes into one canonical NPZ."""
    merged: np.ndarray | None = None
    offsets: np.ndarray | None = None
    for chunk_dir, start_col, stop_col in chunk_specs:
        path = chunk_dir / "map_spectrum.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            cube = np.asarray(data["gain_spectrum_db"], dtype=float)
            if merged is None:
                merged = np.full((powers.size, freqs.size, cube.shape[2]), np.nan, dtype=float)
                offsets = np.asarray(data["signal_offset_mhz"], dtype=float)
            if cube.shape[1] == freqs.size:
                mask = np.isfinite(cube)
                merged[mask] = cube[mask]
            else:
                expected_cols = int(stop_col) - int(start_col)
                if cube.shape[1] != expected_cols:
                    raise ValueError(
                        f"chunk {chunk_dir} spectrum has {cube.shape[1]} frequency columns; "
                        f"expected {expected_cols}"
                    )
                merged[:, start_col:stop_col, :] = cube
    if merged is None or offsets is None:
        return False
    signal_ghz = freqs[None, :] + offsets[:, None] / 1000.0
    np.savez(
        outpath,
        pump_power_dbm=powers,
        pump_frequency_ghz=freqs,
        signal_offset_mhz=offsets,
        gain_spectrum_db=merged,
        signal_ghz=signal_ghz,
    )
    return True


def run_frequency_chunks(
    args: argparse.Namespace,
    raw_argv: list[str],
    outdir: Path,
    freqs: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[Path, int, int]]]:
    """Run the map in fresh 10-column worker processes and merge their rows."""
    ranges = frequency_chunk_ranges(args.n_frequency, args.frequency_chunk_size)
    chunk_root = outdir / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_specs: list[tuple[Path, int, int]] = []
    for chunk_index, (start_col, stop_col) in enumerate(ranges):
        chunk_dir = chunk_root / f"chunk_{chunk_index:03d}_cols_{start_col:03d}_{stop_col - 1:03d}"
        chunk_specs.append((chunk_dir, start_col, stop_col))
        fp0 = float(freqs[start_col])
        fp1 = float(freqs[stop_col - 1])
        if args.resume_chunks and chunk_is_complete(chunk_dir, args, start_col, stop_col):
            print(f"\n=== chunk {chunk_index} : already complete, skipping ===", flush=True)
            continue
        print(
            f"\n=== chunk {chunk_index} : cols [{start_col}..{stop_col - 1}] "
            f"fp {fp0:.4f}..{fp1:.4f} GHz ({stop_col - start_col} cols) ===",
            flush=True,
        )
        t0 = time.perf_counter()
        cmd = chunk_worker_command(
            raw_argv,
            outdir=chunk_dir,
            n_frequency=stop_col - start_col,
            pump_freq_min_ghz=fp0,
            pump_freq_max_ghz=fp1,
            overwrite="--overwrite" in raw_argv,
        )
        log_path = outdir / f"chunk_{chunk_index:03d}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = proc.wait()
        elapsed = time.perf_counter() - t0
        print(f"chunk {chunk_index} rc={rc} elapsed={elapsed:.1f}s log={log_path}", flush=True)
        if rc != 0:
            raise RuntimeError(f"frequency chunk {chunk_index} failed with return code {rc}")
    cold_rows, warm_rows = read_chunk_rows(chunk_specs, global_n_frequency=args.n_frequency)
    return cold_rows, warm_rows, chunk_specs


def main() -> int:
    raw_argv = sys.argv[1:]
    args = parse_args(raw_argv)

    if args.allow_superlu_fallback:
        os.environ.pop("TWPA_REQUIRE_PARDISO", None)
    else:
        os.environ["TWPA_REQUIRE_PARDISO"] = "1"

    if args.log_factor_backend:
        os.environ["TWPA_PARDISO_LOG"] = "1"
    outdir = args.outdir

    # Frequency-crossing traversals share one in-process solved-state store, so
    # they cannot be split across chunk worker processes. Force a single process
    # and widen the per-frequency Schur cache so a backbone row does not thrash.
    if (
        args.traversal != "column"
        and not args.chunk_worker
        and not args.local_traversal_chunks
    ):
        if int(args.frequency_chunk_size) > 0:
            print(f"traversal={args.traversal}: forcing --frequency-chunk-size 0 "
                  "(single process required for cross-column warm state)", flush=True)
            args.frequency_chunk_size = 0
        # NB: keep the Schur cache small (bounded RAM). A freq-crossing backbone
        # row rebuilds the per-frequency partition as it sweeps, but caching all
        # n_frequency partitions would OOM (~16 GB at 50 columns).

    use_chunk_driver = (
        not args.chunk_worker
        and args.executor == "inprocess"
        and int(args.frequency_chunk_size) > 0
        and int(args.n_frequency) > int(args.frequency_chunk_size)
    )
    if outdir.exists() and args.overwrite and not use_chunk_driver:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    points, powers, freqs = build_points(args)

    if args.fold_follow:
        if args.executor != "inprocess":
            raise SystemExit("--fold-follow requires --executor inprocess")
        engine = InProcessEngine(args)
        run_fold_follow(engine, freqs, outdir, args)
        return 0

    if args.frequency_index_start is not None or args.frequency_index_stop is not None:
        start_col = 0 if args.frequency_index_start is None else int(args.frequency_index_start)
        stop_col = args.n_frequency if args.frequency_index_stop is None else int(args.frequency_index_stop)
        if start_col < 0 or stop_col > args.n_frequency or start_col >= stop_col:
            raise ValueError(
                f"invalid frequency chunk [{start_col}, {stop_col}) for "
                f"n_frequency={args.n_frequency}"
            )
        points = [p for p in points if start_col <= p.j_freq < stop_col]
    start = time.perf_counter()

    cold_rows: list[dict[str, Any]] = []
    warm_rows: list[dict[str, Any]] = []
    chunk_specs: list[tuple[Path, int, int]] = []

    if use_chunk_driver:
        print(
            f"executor={args.executor} frequency_chunk_size={args.frequency_chunk_size}",
            flush=True,
        )
        cold_rows, warm_rows, chunk_specs = run_frequency_chunks(args, raw_argv, outdir, freqs)
    else:
        engine = InProcessEngine(args) if args.executor == "inprocess" else None
        cold_pass = (lambda pts, d: run_cold_pass_inprocess(pts, d, engine)) if engine else \
            (lambda pts, d: run_cold_pass(pts, d, args))
        use_traversal_orchestrator = uses_traversal_orchestrator(args)
        if engine and use_traversal_orchestrator:
            warm_pass = lambda pts, d: run_map_traversal(pts, d, engine)
        elif engine:
            warm_pass = lambda pts, d: run_warm_pass_inprocess(pts, d, engine, fail_fast=args.inproc_fail_fast)
        else:
            warm_pass = lambda pts, d: run_warm_pass(pts, d, args)
        print(f"executor={args.executor} traversal={args.traversal}", flush=True)

        if args.mode in ("cold", "both"):
            cold_rows = cold_pass(points, outdir / "cold")
        if args.mode in ("warmstart", "both"):
            warm_rows = warm_pass(points, outdir / "warm")

    # Spot-check cold recompute for a warm-only run.
    if args.mode == "warmstart" and args.gate_spotcheck > 0:
        if "cold_pass" not in locals():
            engine = InProcessEngine(args) if args.executor == "inprocess" else None
            cold_pass = (lambda pts, d: run_cold_pass_inprocess(pts, d, engine)) if engine else \
                (lambda pts, d: run_cold_pass(pts, d, args))
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
        merged = merge_chunk_spectra(outdir / "map_spectrum.npz", chunk_specs, powers, freqs) if chunk_specs else False
        if not merged:
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
