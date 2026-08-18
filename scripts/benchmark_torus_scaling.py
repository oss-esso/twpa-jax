"""Guarded performance ladder for the original pump HB and torus HB solvers.

The controller launches one fresh Python process per case and method.  The
worker writes a pump checkpoint for the torus worker only after a converged
period-1 solve.  Windows process APIs are used instead of psutil because this
repository's benchmark environment does not require psutil.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact with an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _filetime_seconds(value: wintypes.FILETIME) -> float:
    """Convert a Windows FILETIME value to seconds."""
    ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
    return ticks / 10_000_000.0


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_sample(pid: int) -> dict[str, float]:
    """Read current RSS, peak RSS, and CPU time for a Windows process."""
    process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not process:
        return {"rss_bytes": 0.0, "peak_rss_bytes": 0.0, "cpu_seconds": 0.0}
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        ctypes.windll.kernel32.GetProcessTimes(
            process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        return {
            "rss_bytes": float(counters.WorkingSetSize),
            "peak_rss_bytes": float(counters.PeakWorkingSetSize),
            "cpu_seconds": _filetime_seconds(kernel) + _filetime_seconds(user),
        }
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def run_child(
    command: list[str],
    *,
    log_path: Path,
    rss_limit_gb: float,
) -> dict[str, Any]:
    """Run one child, sampling resource usage without reading its log live."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "1"
    environment["TWPA_PARDISO_THREADS"] = "1"
    environment["TWPA_REQUIRE_PARDISO"] = "1"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        peak_rss = 0.0
        peak_cpu_percent = 0.0
        previous = process_sample(child.pid)
        previous_wall = time.perf_counter()
        memory_guard = False
        while child.poll() is None:
            time.sleep(0.5)
            current = process_sample(child.pid)
            now = time.perf_counter()
            delta_wall = max(now - previous_wall, 1e-9)
            delta_cpu = max(current["cpu_seconds"] - previous["cpu_seconds"], 0.0)
            peak_cpu_percent = max(
                peak_cpu_percent,
                100.0 * delta_cpu / delta_wall / max(os.cpu_count() or 1, 1),
            )
            peak_rss = max(
                peak_rss,
                current["rss_bytes"],
                current["peak_rss_bytes"],
            )
            if peak_rss > rss_limit_gb * 1024.0**3:
                memory_guard = True
                child.kill()
                break
            previous = current
            previous_wall = now
        return_code = child.wait()
        final = process_sample(child.pid)
    return {
        "command": command,
        "return_code": int(return_code),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": float(final["cpu_seconds"] or previous["cpu_seconds"]),
        "peak_rss_bytes": int(max(peak_rss, final["peak_rss_bytes"])),
        "peak_rss_gib": max(peak_rss, final["peak_rss_bytes"]) / 1024.0**3,
        "peak_cpu_percent_of_machine": peak_cpu_percent,
        "memory_guard_tripped": memory_guard,
        "log": str(log_path),
    }


def _case_configs(root: Path) -> list[dict[str, Any]]:
    """Return the agreed K=5 ladder, including the corrected 2c point."""
    return [
        {
            "case_id": "uniform_418jj",
            "kind": "uniform",
            "n_cells": 418,
            "frequency_ghz": 7.9,
            "pump_port": 1,
            "current_ratio": 0.05,
        },
        {
            "case_id": "uniform_4x418jj",
            "kind": "uniform",
            "n_cells": 1672,
            "frequency_ghz": 7.9,
            "pump_port": 1,
            "current_ratio": 0.05,
        },
        {
            "case_id": "jc_jtwpa",
            "kind": "circuit_dir",
            "circuit_dir": root / "outputs" / "jc_doc_python_designs" / "jc_jtwpa",
            "seed_dir": root / "outputs" / "chaos" / "onepoint_torus_jtwpa_m29p7" / "adaptive_pump",
            "frequency_ghz": 7.12,
            "pump_port": 1,
        },
        {
            "case_id": "ipm_2c_fixed_minus25dbm",
            "kind": "circuit_dir",
            "circuit_dir": root / "designs" / "ipm_2c_fixed",
            "frequency_ghz": 7.9,
            "pump_power_dbm": -25.0,
            "pump_port": 4,
            "attenuation_override_db": None,
            "power_convention": "legacy_traveling_wave",
        },
        {
            "case_id": "ipm_2c_fixed_minus23p8dbm",
            "kind": "circuit_dir",
            "circuit_dir": root / "designs" / "ipm_2c_fixed",
            "frequency_ghz": 7.9,
            "pump_power_dbm": -23.8,
            "pump_port": 4,
            "attenuation_override_db": None,
            "power_convention": "legacy_traveling_wave",
        },
    ]


def worker_baseline(args: argparse.Namespace) -> int:
    """Solve one period-1 pump point and persist a torus-compatible seed."""
    from twpa_solver.core import load_circuit
    from twpa_solver.core.nonlinear import make_branch_law
    from twpa_solver.core.linear import default_loss_model_for
    from twpa_solver.pump.basis import (
        PumpBasis,
        load_pump_basis_from_solution,
        positive_odd_modes,
        promote_solution_to_basis,
    )
    from twpa_solver.pump.problem import FullPumpProblem, HarmonicGrid
    from twpa_solver.pump.solver import HarmonicNewtonKrylovSolver, NewtonKrylovSettings

    print(f"twpa_solver={__import__('twpa_solver').__file__}", flush=True)
    if args.kind == "uniform":
        from twpa_solver.builders.complexity_ladder import (
            build_uniform_jtl,
            save_ladder_circuit,
        )

        circuit = build_uniform_jtl(args.n_cells)
        save_ladder_circuit(circuit, Path(args.out).parent / "circuit")
        frequency_ghz = args.frequency_ghz
        pump_current = float(args.current_ratio) * float(np.max(circuit.Ic))
    else:
        circuit = load_circuit(args.circuit_dir)
        frequency_ghz = args.frequency_ghz
        if args.seed_dir is not None:
            with (args.seed_dir / "pump_report.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle).get("metadata", {})
            pump_current = float(metadata["pump_current_a"])
        else:
            from twpa_solver.loss import pump_loss_model

            pump_current = pump_loss_model().dbm_to_peak_current_a(
                args.pump_power_dbm,
                frequency_ghz,
                convention=args.power_convention,
            )
    omega_p = 2.0 * math.pi * frequency_ghz * 1.0e9
    modes = positive_odd_modes(args.k)
    grid = HarmonicGrid(
        modes=np.asarray(modes, dtype=int),
        nt=max(40, 2 * max(modes) + 2),
        omega=omega_p,
    )
    problem = FullPumpProblem(
        circuit.C,
        circuit.G,
        circuit.K,
        circuit.Bphi,
        make_branch_law(circuit),
        grid,
        circuit.port_to_index[int(args.pump_port)],
        pump_current,
        source_mode=1,
        loss_model=default_loss_model_for(circuit),
    )
    settings = NewtonKrylovSettings(
        newton_tol=1.0e-9,
        max_newton=16,
        gmres_rtol=1.0e-7,
        gmres_atol=0.0,
        gmres_restart=60,
        gmres_maxiter=80,
        min_alpha=1.0 / 1024.0,
        preconditioner="mean_tangent",
        compute_time_residual=False,
        verbose=False,
        continuation_predictor="none",
        jvp_mode="aft",
        precond_reuse=1,
    )
    solver = HarmonicNewtonKrylovSolver(settings)
    initial_state = problem.zeros()
    if args.seed_dir is not None:
        seed_state, seed_basis = load_pump_basis_from_solution(
            args.seed_dir, fallback_omega_p=omega_p
        )
        target_basis = PumpBasis(
            modes=modes,
            policy="positive_odd_jc",
            omega_p=omega_p,
        )
        initial_state = promote_solution_to_basis(
            seed_state, seed_basis, target_basis
        )
    state, report = solver.solve_one(problem, initial_state, 1.0)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "pump_solution.npz"
    temporary = npz_path.with_suffix(npz_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            X_real=np.asarray(state.real, dtype=np.float64),
            X_imag=np.asarray(state.imag, dtype=np.float64),
            pump_modes=np.asarray(modes, dtype=np.int64),
        )
    temporary.replace(npz_path)
    metadata = {
        "pump_modes": modes,
        "pump_basis": "positive_phasor",
        "pump_mode_policy": "positive_odd_jc",
        "pump_source_mode": 1,
        "omega_p": omega_p,
        "pump_freq_ghz": frequency_ghz,
        "pump_current_a": pump_current,
        "pump_port": int(args.pump_port),
        "pump_power_dbm_requested": args.pump_power_dbm,
        "attenuation_override_db": args.attenuation_override_db,
        "power_convention": args.power_convention,
        "solver": "twpa_solver.pump.HarmonicNewtonKrylovSolver",
        "preconditioner": settings.preconditioner,
    }
    atomic_json(output / "pump_report.json", {"metadata": metadata, "report": report})
    result = {
        "method": "original_period1_hb",
        "case_id": args.case_id,
        "converged": bool(report.converged),
        "coeff_rel": float(report.coeff_rel),
        "newton_iterations": int(report.newton_iterations),
        "gmres_iterations_total": int(report.gmres_iterations_total),
        "factor_runtime_s": float(report.factor_runtime_s),
        "runtime_s": float(report.runtime_s),
        "failure_reason": report.failure_reason,
        "pump_solution_dir": str(output),
        "node_count": int(circuit.C.shape[0]),
        "branch_count": int(circuit.Bphi.shape[1]),
        "pump_current_a": pump_current,
        "pump_power_dbm_requested": args.pump_power_dbm,
        "attenuation_override_db": args.attenuation_override_db,
        "loss_model": default_loss_model_for(circuit),
    }
    atomic_json(output / "worker_result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if report.converged else 2


def controller(args: argparse.Namespace) -> int:
    """Run the ladder sequentially and add controller-side telemetry."""
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    python = sys.executable
    configs = _case_configs(ROOT)
    if args.only_case:
        selected = set(args.only_case)
        configs = [config for config in configs if config["case_id"] in selected]
    for config in configs:
        case_id = str(config["case_id"])
        case_root = output_root / case_id
        pump_dir = case_root / "pump"
        baseline_command = [
            python,
            str(Path(__file__).resolve()),
            "--worker-baseline",
            "--case-id",
            case_id,
            "--kind",
            str(config["kind"]),
            "--frequency-ghz",
            str(config["frequency_ghz"]),
            "--pump-port",
            str(config["pump_port"]),
            "--k",
            str(args.k),
            "--out",
            str(pump_dir),
        ]
        if config["kind"] == "uniform":
            baseline_command += ["--n-cells", str(config["n_cells"]), "--current-ratio", str(config["current_ratio"])]
        else:
            baseline_command += ["--circuit-dir", str(config["circuit_dir"])]
            if "seed_dir" in config:
                baseline_command += ["--seed-dir", str(config["seed_dir"])]
            if "pump_power_dbm" in config:
                baseline_command += ["--pump-power-dbm", str(config["pump_power_dbm"])]
            baseline_command += [
                "--attenuation-override-db",
                "" if config.get("attenuation_override_db") is None else str(config["attenuation_override_db"]),
                "--power-convention",
                str(config.get("power_convention", "legacy_traveling_wave")),
            ]
        baseline_telemetry = run_child(
            baseline_command,
            log_path=case_root / "original_hb.log",
            rss_limit_gb=args.rss_limit_gb,
        )
        baseline_result_path = pump_dir / "worker_result.json"
        baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8")) if baseline_result_path.exists() else {}
        row: dict[str, Any] = {
            "case": config,
            "baseline": {**baseline_result, "telemetry": baseline_telemetry},
        }
        if baseline_result.get("converged") and not baseline_telemetry["memory_guard_tripped"]:
            torus_command = [
                python,
                str(ROOT / "scripts" / "run_torus_branch.py"),
                "--device",
                case_id,
                "--circuit-dir",
                str(config.get("circuit_dir", "")) if config["kind"] != "uniform" else str(case_root / "circuit"),
                "--pump-dir",
                str(pump_dir),
                "--omega-a-ratio",
                "0.0917",
                "--q-max",
                "1",
                "--node-ref",
                "0",
                "--factor-backend",
                "pardiso",
                "--branch-step",
                "0.05",
                "--omitted-q-max",
                "3",
                "--min-off-comb-fraction",
                "1e-8" if case_id.startswith("ipm_2c") else "0.0",
                "--out",
                str(case_root / "torus.json"),
            ]
            if config["kind"] == "circuit_dir" and case_id.startswith("ipm_2c"):
                torus_command.append("--schur")
            if config.get("floquet_seed_npz") is not None:
                torus_command.extend(
                    ["--floquet-seed-npz", str(config["floquet_seed_npz"])]
                )
            torus_telemetry = run_child(
                torus_command,
                log_path=case_root / "torus.log",
                rss_limit_gb=args.rss_limit_gb,
            )
            torus_result_path = case_root / "torus.json"
            torus_result = json.loads(torus_result_path.read_text(encoding="utf-8")) if torus_result_path.exists() else {}
            row["torus"] = {**torus_result, "telemetry": torus_telemetry}
        else:
            row["torus"] = {"skipped": True, "reason": "baseline did not converge or memory guard tripped"}
        atomic_json(case_root / "summary.json", row)
        summaries.append(row)
        print(json.dumps(row, indent=2, default=str), flush=True)
    atomic_json(output_root / "ladder_summary.json", {"cases": summaries})
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-baseline", action="store_true")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--kind", choices=("uniform", "circuit_dir"), default="uniform")
    parser.add_argument("--n-cells", type=int, default=418)
    parser.add_argument("--circuit-dir", type=Path, default=None)
    parser.add_argument("--seed-dir", type=Path, default=None)
    parser.add_argument("--frequency-ghz", type=float, default=7.9)
    parser.add_argument("--pump-port", type=int, default=1)
    parser.add_argument("--current-ratio", type=float, default=0.05)
    parser.add_argument("--pump-power-dbm", type=float, default=None)
    parser.add_argument("--attenuation-override-db", default=None)
    parser.add_argument("--power-convention", default="legacy_traveling_wave")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "benchmarks" / "torus_scaling_20260817")
    parser.add_argument("--rss-limit-gb", type=float, default=6.0)
    parser.add_argument("--only-case", nargs="+", default=None)
    args = parser.parse_args(argv)
    if args.attenuation_override_db == "":
        args.attenuation_override_db = None
    if args.worker_baseline and (args.out is None or args.kind == "circuit_dir" and args.circuit_dir is None):
        parser.error("worker baseline requires --out and --circuit-dir for loaded circuits")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return worker_baseline(args) if args.worker_baseline else controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
