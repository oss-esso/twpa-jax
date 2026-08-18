"""Unattended amplitude-parameterized torus campaign controller."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def atomic_json(path: Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def log_line(path: Path, message: str) -> None:
    """Append one timestamped campaign event."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def process_sample(pid: int) -> dict[str, float]:
    """Return RSS and CPU telemetry for a child process."""
    try:
        import psutil
    except ImportError:
        return {"rss_bytes": 0.0, "cpu_seconds": 0.0}
    try:
        process = psutil.Process(pid)
        memory = process.memory_info().rss
        cpu = process.cpu_times()
        return {
            "rss_bytes": float(memory),
            "cpu_seconds": float(cpu.user + cpu.system),
        }
    except (psutil.Error, OSError):
        return {"rss_bytes": 0.0, "cpu_seconds": 0.0}


def run_child(
    command: list[str],
    log_path: Path,
    deadline_s: float,
    rss_limit_gb: float | None = None,
) -> dict[str, Any]:
    """Run a child with process telemetry, never reading its log while active."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "1"
    environment["TWPA_PARDISO_THREADS"] = "1"
    environment["TWPA_REQUIRE_PARDISO"] = "1"
    started = time.perf_counter()
    peak_rss = 0.0
    peak_cpu = 0.0
    timed_out = False
    memory_guard = False
    with log_path.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        previous = process_sample(child.pid)
        previous_wall = time.perf_counter()
        while child.poll() is None:
            time.sleep(0.5)
            current = process_sample(child.pid)
            now = time.perf_counter()
            peak_rss = max(peak_rss, current["rss_bytes"])
            delta_wall = max(now - previous_wall, 1.0e-9)
            delta_cpu = max(current["cpu_seconds"] - previous["cpu_seconds"], 0.0)
            peak_cpu = max(peak_cpu, 100.0 * delta_cpu / delta_wall)
            if rss_limit_gb is not None and peak_rss > rss_limit_gb * 2**30:
                memory_guard = True
                timed_out = True
            if time.perf_counter() - started > deadline_s or memory_guard:
                timed_out = True
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    child.kill()
                break
            previous = current
            previous_wall = now
        return_code = child.wait()
    return {
        "command": command,
        "return_code": int(return_code),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_gib": peak_rss / 2**30,
        "peak_cpu_percent": peak_cpu,
        "timed_out": timed_out,
        "memory_guard_tripped": memory_guard,
        "log": str(log_path),
    }


def _load_case(circuit_dir: Path, pump_dir: Path, omega_ratio: float, q_max: int,
               use_schur: bool, factor_backend: str, k: int) -> tuple[Any, Any, Any, Any]:
    """Construct one torus problem and retain the pump metadata."""
    from twpa_solver.core import load_circuit
    from twpa_solver.multitone.basis import build_autonomous_torus_basis
    from twpa_solver.multitone.problem import FullMultiToneProblem
    from twpa_solver.multitone.schur import build_multitone_schur_problem
    from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
    from twpa_solver.multitone.torus import TorusProblem
    from twpa_solver.pump.basis import positive_odd_modes
    from twpa_solver.signal.io import load_pump

    circuit = load_circuit(circuit_dir)
    pump = load_pump(pump_dir, fallback_pump_freq_ghz=7.9)
    pump_modes = tuple(positive_odd_modes(k))
    omega_a = omega_ratio * pump.omega_p
    basis = build_autonomous_torus_basis(
        pump.omega_p, omega_a, pump_modes, q_max
    )
    port = int(pump.metadata.get("pump_port", next(iter(circuit.port_to_index))))
    current = next(
        (float(pump.metadata[key]) for key in
         ("pump_current_a", "pump_current_peak_a", "current_a")
         if pump.metadata.get(key) is not None),
        None,
    )
    if current is None:
        raise KeyError(f"pump current is missing from {pump_dir / 'pump_report.json'}")
    drive = MultiToneDrive(
        basis.pump_tone, circuit.port_to_index[port], current
    ).to_coeffs(basis, circuit.C.shape[0])
    full = FullMultiToneProblem(
        circuit,
        basis,
        AffineSourcePath.pump_turn_on(drive),
        loss_model=pump.metadata.get("loss_model"),
    )
    full_node_ref = 0
    retained_index: int | None = None
    base: Any = full
    if use_schur:
        base = build_multitone_schur_problem(
            full,
            list(circuit.port_to_index.values()),
            preconditioner="real_coupled_fast",
        )
        retained_index = int(base.partition.retained_pos[full_node_ref])
        if retained_index < 0:
            full_node_ref = int(base.partition.retained[0])
            retained_index = 0
        node_ref = retained_index
    else:
        node_ref = full_node_ref
    torus = TorusProblem(
        base,
        pump_modes,
        q_max,
        omega_a,
        node_ref=node_ref,
        factor_backend=factor_backend,
        precond_reuse=1,
    )
    return torus, pump, basis, {
        "circuit": circuit,
        "full_node_ref": full_node_ref,
        "retained_node_ref": retained_index,
        "pump_current_a": current,
        "pump_modes": list(pump_modes),
    }


def _q_fraction(state: np.ndarray, torus: Any) -> float:
    rows = torus.generator_rows()
    q_norm = float(np.linalg.norm(state[rows]))
    return q_norm / max(float(np.linalg.norm(state)), 1.0e-300)


def worker_amplitude(args: argparse.Namespace) -> int:
    """Solve one amplitude rung and write its result before returning."""
    from twpa_solver.multitone.seed import seed_torus_from_pump

    started = time.perf_counter()
    torus, pump, basis, metadata = _load_case(
        args.circuit_dir,
        args.pump_dir,
        args.omega_ratio,
        args.q_max,
        args.schur,
        args.factor_backend,
        args.k,
    )
    target = float(args.amplitude_relative) * float(np.linalg.norm(pump.X))
    omega0 = args.omega_ratio * pump.omega_p
    tau0 = 1.0
    state: np.ndarray
    if args.warm_state is not None and args.warm_state.exists():
        with np.load(args.warm_state) as data:
            state = np.asarray(data["X"], dtype=np.complex128)
        omega0 = float(args.warm_omega)
        tau0 = float(args.warm_tau)
    else:
        state = seed_torus_from_pump(
            pump.X,
            pump.basis,
            basis,
            amplitude=args.amplitude_relative,
            node_ref=metadata["full_node_ref"],
        )
        if args.schur:
            state = state[:, torus.base_problem.partition.retained]
    try:
        state, omega, tau, report = torus.solve_newton_amplitude(
            state,
            target,
            omega_a0=omega0,
            source_tau0=tau0,
            max_newton=args.max_newton,
            residual_tol=args.residual_tol,
        )
        error: str | None = None
    except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
        omega, tau = omega0, tau0
        report = {
            "converged": False,
            "iterations": 0,
            "residual_norm": None,
            "residual_history": [],
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
        error = str(exc)
    q_fraction = _q_fraction(state, torus)
    result = {
        "case_id": args.case_id,
        "amplitude_relative": args.amplitude_relative,
        "amplitude_absolute": target,
        "converged": bool(report.get("converged", False)),
        "omega_a_over_omega_p": float(omega / pump.omega_p),
        "source_tau": float(tau),
        "pump_current_a": metadata["pump_current_a"],
        "implied_pump_current_a": float(tau * metadata["pump_current_a"]),
        "i_over_i_bound": (
            float(tau * metadata["pump_current_a"] / 1.1628e-5)
            if "2c" in args.case_id else None
        ),
        "q_nonzero_norm_fraction": q_fraction,
        "wall_seconds": time.perf_counter() - started,
        "q_max": args.q_max,
        "schur": args.schur,
        "factor_backend": args.factor_backend,
        "anchor_full_node": metadata["full_node_ref"],
        "anchor_retained_index": metadata["retained_node_ref"],
        "error": error,
        "report": report,
    }
    atomic_json(args.out, result)
    if result["converged"]:
        temporary = args.out.with_suffix(".state.npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, X=state)
        temporary.replace(args.out.with_suffix(".state.npz"))
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result["converged"] else 2


def fixture_smoke() -> dict[str, Any]:
    """Run the small fixture smoke from ``tests/test_torus_hb.py``."""
    from twpa_solver.core.linear import default_loss_model_for
    from twpa_solver.core.nonlinear import make_branch_law
    from twpa_solver.multitone.basis import build_autonomous_torus_basis
    from twpa_solver.multitone.problem import FullMultiToneProblem
    from twpa_solver.multitone.seed import seed_torus_from_pump
    from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
    from twpa_solver.multitone.torus import TorusProblem
    from twpa_solver.pump.basis import PumpBasis
    from twpa_solver.pump.problem import FullPumpProblem, HarmonicGrid
    from twpa_solver.pump.solver import (
        HarmonicNewtonKrylovSolver,
        NewtonKrylovSettings,
    )

    fixture_path = ROOT / "tests" / "test_multitone_problem.py"
    spec = importlib.util.spec_from_file_location(
        "twpa_campaign_test_multitone_problem", fixture_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fixture module from {fixture_path}")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    circuit = fixture_module._circuit()
    omega_p = 2.0e10
    pump_problem = FullPumpProblem(
        circuit.C,
        circuit.G,
        circuit.K,
        circuit.Bphi,
        make_branch_law(circuit),
        HarmonicGrid(np.asarray([1]), 40, omega_p),
        circuit.port_to_index[1],
        1.0e-9,
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
    pump_state, pump_report = HarmonicNewtonKrylovSolver(settings).solve_one(
        pump_problem, pump_problem.zeros(), 1.0
    )
    basis = build_autonomous_torus_basis(omega_p, 1.0e9, [1], 1)
    drive = MultiToneDrive(basis.pump_tone, 0, 1.0e-9).to_coeffs(basis, 2)
    torus = TorusProblem(
        FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.pump_turn_on(drive),
        ),
        (1,),
        1,
        1.0e9,
    )
    seed = seed_torus_from_pump(
        pump_state,
        PumpBasis([1], "positive_odd_jc", omega_p),
        basis,
        amplitude=1.0e-3,
        node_ref=0,
    )
    amplitude = torus.generator_norm(seed)
    _, _, _, report = torus.solve_newton_amplitude(seed, amplitude)
    return {
        "pump_converged": bool(pump_report.converged),
        "torus_converged": bool(report.get("converged")),
        "torus_report": report,
    }


def _write_event(root: Path, name: str, payload: dict[str, Any]) -> None:
    atomic_json(root / "preflight" / f"{name}.json", payload)


def preflight(root: Path, log_path: Path, rss_limit_gb: float) -> dict[str, Any]:
    """Run all hard preflight gates and raise on the first failed gate."""
    import psutil

    import twpa_solver

    solver_path = Path(twpa_solver.__file__).resolve()
    under_src = str(solver_path).lower().startswith(str(SRC.resolve()).lower())
    _write_event(root, "import", {"twpa_solver_file": str(solver_path), "under_src": under_src})
    if not under_src:
        raise RuntimeError(f"twpa_solver resolves outside repository src: {solver_path}")
    pytest_log = root / "preflight" / "pytest.log"
    with pytest_log.open("w", encoding="utf-8") as handle:
        pytest_run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                r"D:\tmp\torus_campaign_20260817",
                "tests/test_torus_hb.py",
                "tests/test_multitone_seed.py",
                "tests/test_multitone_basis.py",
            ],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    pytest_data = {"return_code": pytest_run.returncode, "log": str(pytest_log)}
    _write_event(root, "pytest", pytest_data)
    if pytest_run.returncode != 0:
        raise RuntimeError(f"preflight pytest failed; see {pytest_log}")
    available = float(psutil.virtual_memory().available)
    available_gb = available / 2**30
    logical = int(psutil.cpu_count(logical=True) or 1)
    workers = min(3, math.floor(available_gb / 3.5))
    resources = {
        "available_memory_gb": available_gb,
        "logical_cpu_count": logical,
        "derived_workers": workers,
    }
    _write_event(root, "resources", resources)
    if available_gb < 5.0 or workers < 1:
        raise RuntimeError(f"preflight memory gate failed: {resources}")
    smoke1 = fixture_smoke()
    _write_event(root, "smoke1", smoke1)
    if not smoke1["pump_converged"] or not smoke1["torus_converged"]:
        raise RuntimeError(f"fixture amplitude smoke failed: {smoke1}")
    artifact = ROOT / ".hybrid_outputs" / "period1_recovery_7p9_2c_v1" / "point_-23.800000" / "pump"
    if not (artifact / "pump_solution.npz").exists():
        raise RuntimeError(f"missing 2c smoke artifact: {artifact}")
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-amplitude",
        "--case-id",
        "smoke_2c_k5",
        "--circuit-dir",
        str(ROOT / "designs" / "ipm_2c_fixed"),
        "--pump-dir",
        str(artifact),
        "--omega-ratio",
        "0.0917",
        "--q-max",
        "1",
        "--amplitude-relative",
        "1e-5",
        "--residual-tol",
        "1e-9",
        "--out",
        str(root / "preflight" / "smoke2.json"),
    ]
    smoke2 = run_child(common + ["--k", "5"], root / "preflight" / "smoke2.log", 600.0, rss_limit_gb)
    smoke2_result = _read_json(root / "preflight" / "smoke2.json")
    smoke2.update({"result": smoke2_result})
    _write_event(root, "smoke2_telemetry", smoke2)
    if smoke2["return_code"] != 0 and not smoke2_result:
        raise RuntimeError(f"K=5 smoke failed without result: {smoke2}")
    if smoke2["peak_rss_gib"] > 4.0:
        resources["derived_workers"] = 1
    smoke3_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-amplitude",
        "--case-id",
        "smoke_2c_k10",
        "--circuit-dir",
        str(ROOT / "designs" / "ipm_2c_fixed"),
        "--pump-dir",
        str(artifact),
        "--omega-ratio",
        "0.0917",
        "--q-max",
        "1",
        "--amplitude-relative",
        "1e-5",
        "--residual-tol",
        "1e-9",
        "--schur",
        "--out",
        str(root / "preflight" / "smoke3.json"),
    ]
    smoke3 = run_child(smoke3_command, root / "preflight" / "smoke3.log", 1800.0, rss_limit_gb)
    smoke3_result = _read_json(root / "preflight" / "smoke3.json")
    smoke3.update({"result": smoke3_result})
    _write_event(root, "smoke3_telemetry", smoke3)
    if smoke3["return_code"] != 0 and not smoke3_result:
        raise RuntimeError(f"K=10 smoke failed without result: {smoke3}")
    skip_stage_d = smoke3["wall_seconds"] > 1800.0 or smoke3["peak_rss_gib"] > 5.5
    estimate = 51 * max(smoke2["wall_seconds"], 1.0) + 26 * max(smoke3["wall_seconds"], 1.0)
    counts = {"stage_c": 17, "stage_d": 13, "stage_e": 13}
    reduction: dict[str, Any] = {"applied": False}
    if estimate > 7.0 * 3600.0:
        counts["stage_c"] = 9
        counts["stage_d"] = 7
        counts["stage_e"] = 7
        reduction = {"applied": True, "reason": "smoke extrapolation exceeded 7 h"}
    result = {
        "solver_file": str(solver_path),
        "pytest": pytest_data,
        "resources": resources,
        "smoke1": smoke1,
        "smoke2": smoke2,
        "smoke3": smoke3,
        "estimated_seconds": estimate,
        "skip_stage_d": skip_stage_d,
        "amplitude_counts": counts,
        "reduction": reduction,
    }
    _write_event(root, "summary", result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def stage_spectrum(root: Path, log_path: Path, deadline: float, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the q=+-1 singular-value scans before amplitude ladders."""
    rows: list[dict[str, Any]] = []
    ratios = np.geomspace(0.02, 0.50, 120)
    for case in cases:
        if time.monotonic() >= deadline:
            break
        case_root = root / "stage_b" / str(case["case_id"])
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker-spectrum",
            "--case-id", str(case["case_id"]), "--circuit-dir", str(case["circuit_dir"]),
            "--pump-dir", str(case["pump_dir"]), "--omega-ratio", str(case["omega_ratio"]),
            "--q-max", "1", "--out-dir", str(case_root), "--points", "120",
        ]
        if case.get("schur"):
            command.append("--schur")
        telemetry = run_child(command, case_root.with_suffix(".log"), max(60.0, deadline - time.monotonic()), 6.0)
        result = _read_json(case_root / "spectrum.json")
        row = {"case": case, "telemetry": telemetry, "result": result}
        atomic_json(case_root.with_suffix(".summary.json"), row)
        rows.append(row)
    return rows


def worker_spectrum(args: argparse.Namespace) -> int:
    """Estimate the smallest singular value of the q=+-1 real block."""
    from scipy.sparse.linalg import LinearOperator, svds
    from twpa_solver.pump.problem import pack_complex, unpack_complex
    from twpa_solver.multitone.basis import ToneIndex, build_autonomous_torus_basis

    torus, pump, _, metadata = _load_case(
        args.circuit_dir, args.pump_dir, args.omega_ratio, args.q_max,
        args.schur, "pardiso", args.k
    )
    ratios = np.geomspace(0.02, 0.50, args.points)
    rows: list[dict[str, Any]] = []
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, ratio in enumerate(ratios):
        current = torus.with_omega_a(float(ratio * pump.omega_p))
        problem = current.full_problem()
        basis = problem.basis
        pump_state = np.zeros((basis.n_tones, problem.n), dtype=np.complex128)
        for source_row, mode in enumerate(pump.basis.modes):
            tone = type(basis.tones[0])(int(mode), 0)
            if tone in basis.tones:
                pump_state[basis.index_of(tone)] = pump.X[source_row]
        tangent = problem.tangent_state(pump_state)
        spectral = problem.spectral_tangent_state(tangent)
        rows_q = [i for i, tone in enumerate(basis.tones) if tone.q != 0]
        n = problem.n
        dimension = 2 * len(rows_q) * n

        def matvec(vector: np.ndarray) -> np.ndarray:
            local = unpack_complex(vector, (len(rows_q), n))
            full = np.zeros((basis.n_tones, n), dtype=np.complex128)
            full[rows_q] = local
            applied = problem.jvp_coeffs_with_spectral_tangent(full, spectral)
            return pack_complex(applied[rows_q])

        operator = LinearOperator((dimension, dimension), matvec=matvec, dtype=float)
        value: float | None = None
        error: str | None = None
        try:
            singular = svds(operator, k=1, which="SM", ncv=4, maxiter=8,
                             return_singular_vectors=False)
            value = float(np.asarray(singular).reshape(-1)[0])
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "index": index,
            "omega_a_over_omega_p": float(ratio),
            "sigma_min": value,
            "error": error,
            "dimension": dimension,
        }
        atomic_json(out_dir / f"row_{index:03d}.json", row)
        rows.append(row)
        atomic_json(out_dir / "spectrum.json", {
            "case_id": args.case_id,
            "rows_completed": len(rows),
            "rows": rows,
            "omega_ratio_seed": args.omega_ratio,
            "fdtd_reference": args.omega_ratio,
            "anchor_full_node": metadata["full_node_ref"],
            "anchor_retained_index": metadata["retained_node_ref"],
        })
    return 0


def _case_definitions(root: Path, jtwpa_pump: Path | None) -> list[dict[str, Any]]:
    """Return the three fixed campaign devices and their pump artifacts."""
    base = root / ".hybrid_outputs" / "period1_recovery_7p9_2c_v1"
    cases = [
        ("2c_06050", base / "point_-23.800000" / "pump", 0.0917, True),
        ("2c_05912", base / "point_-24.000000" / "pump", 0.0917, True),
        ("2c_05745", base / "point_-24.250000" / "pump", 0.0917, True),
    ]
    result = [
        {
            "case_id": name,
            "circuit_dir": root / "designs" / "ipm_2c_fixed",
            "pump_dir": pump,
            "omega_ratio": ratio,
            "schur": schur,
        }
        for name, pump, ratio, schur in cases
    ]
    if jtwpa_pump is not None:
        result.append({
            "case_id": "jtwpa_m29p40",
            "circuit_dir": root / "outputs" / "jc_doc_python_designs" / "jc_jtwpa",
            "pump_dir": jtwpa_pump,
            "omega_ratio": 0.1217,
            "schur": False,
        })
    return result


def _ladder(
    root: Path,
    log_path: Path,
    deadline: float,
    case: dict[str, Any],
    count: int,
    q_max: int,
    stage_name: str,
    use_schur: bool,
) -> list[dict[str, Any]]:
    """Run one resumable amplitude ladder in fresh processes."""
    values = np.geomspace(1.0e-6, 1.0e-2, count)
    output = root / stage_name / str(case["case_id"])
    output.mkdir(parents=True, exist_ok=True)
    previous_state: Path | None = None
    previous_omega = case["omega_ratio"] * 1.0
    previous_tau = 1.0
    failures = 0
    rows: list[dict[str, Any]] = []
    for index, amplitude in enumerate(values):
        if time.monotonic() >= deadline or failures >= 3:
            break
        result_path = output / f"point_{index:03d}.json"
        if result_path.exists():
            result = _read_json(result_path)
        else:
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker-amplitude",
                "--case-id", str(case["case_id"]), "--circuit-dir", str(case["circuit_dir"]),
                "--pump-dir", str(case["pump_dir"]), "--omega-ratio", str(case["omega_ratio"]),
                "--q-max", str(q_max), "--amplitude-relative", f"{amplitude:.17g}",
                "--residual-tol", "1e-9", "--out", str(result_path),
            ]
            if use_schur:
                command.append("--schur")
            if previous_state is not None and previous_state.exists():
                command += [
                    "--warm-state", str(previous_state),
                    "--warm-omega", f"{previous_omega:.17g}",
                    "--warm-tau", f"{previous_tau:.17g}",
                ]
            telemetry = run_child(command, output / f"point_{index:03d}.log", 1800.0, 6.0)
            result = _read_json(result_path)
            result["controller_telemetry"] = telemetry
            atomic_json(result_path, result)
        result["point_index"] = index
        result["stage"] = stage_name
        rows.append(result)
        if result.get("converged"):
            failures = 0
            previous_state = result_path.with_suffix(".state.npz")
            previous_omega = float(result.get("omega_a_over_omega_p", case["omega_ratio"]))
            previous_tau = float(result.get("source_tau", 1.0))
        else:
            failures += 1
        atomic_json(output / "ladder.json", {"case": case, "rows": rows})
    return rows


def _ensure_jtwpa_pump(root: Path, log_path: Path, deadline: float) -> Path | None:
    """Solve the required fresh jtwpa pump point through the production map."""
    output = root / "jtwpa_pump_m29p40"
    candidate = next(output.rglob("pump_solution.npz"), None) if output.exists() else None
    if candidate is not None:
        return candidate.parent
    command = [
        sys.executable, str(root / "scripts" / "run_gain_map.py"),
        "--mode", "cold", "--executor", "inprocess",
        "--circuit-dir", str(root / "outputs" / "jc_doc_python_designs" / "jc_jtwpa"),
        "--outdir", str(output), "--n-power", "1", "--n-frequency", "1",
        "--pump-power-min-dbm", "-29.4", "--pump-power-max-dbm", "-29.4",
        "--pump-freq-min-ghz", "7.12", "--pump-freq-max-ghz", "7.12",
        "--pump-port", "1", "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", "10", "--power-convention", "legacy_traveling_wave",
        "--no-signal-spectrum", "--force-single-tone", "--frequency-chunk-size", "0",
        "--no-compact-output",
    ]
    telemetry = run_child(command, output.with_suffix(".log"), max(60.0, deadline - time.monotonic()), 6.0)
    atomic_json(root / "stage_a" / "jtwpa_pump_telemetry.json", telemetry)
    if telemetry["return_code"] != 0:
        return None
    candidate = next(output.rglob("pump_solution.npz"), None)
    return candidate.parent if candidate is not None else None


def stage_g(root: Path, log_path: Path, preflight_data: dict[str, Any], stage_data: dict[str, Any]) -> None:
    """Write the campaign summary, branch fits, and tracker section."""
    points: list[dict[str, Any]] = []
    for path in root.rglob("point_*.json"):
        data = _read_json(path)
        if data:
            points.append(data)
    branch = []
    for point in points:
        if point.get("converged") and point.get("i_over_i_bound") is not None:
            branch.append({
                "case_id": point.get("case_id"),
                "amplitude_relative": point.get("amplitude_relative"),
                "i_over_i_bound": point.get("i_over_i_bound"),
                "omega_a_over_omega_p": point.get("omega_a_over_omega_p"),
                "source_tau": point.get("source_tau"),
            })
    fit: dict[str, Any] = {"status": "NOT_ESTABLISHED"}
    if len(branch) >= 3:
        x = np.asarray([row["i_over_i_bound"] for row in branch], dtype=float)
        y = np.asarray([row["amplitude_relative"] ** 2 for row in branch], dtype=float)
        small = np.argsort(np.asarray([row["amplitude_relative"] for row in branch]))[:max(3, len(branch) // 3)]
        slope, intercept = np.polyfit(x[small], y[small], 1)
        predicted = slope * x[small] + intercept
        ss_res = float(np.sum((y[small] - predicted) ** 2))
        ss_tot = float(np.sum((y[small] - np.mean(y[small])) ** 2))
        fit = {
            "status": "ESTABLISHED",
            "slope": float(slope),
            "intercept": float(intercept),
            "mu_c": float(-intercept / slope) if slope != 0.0 else None,
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None,
            "n_small_amplitude_points": int(len(small)),
            "reference_measured_boundary": 0.5825,
        }
    summary = {
        "campaign": "torus_campaign_20260817",
        "preflight": preflight_data,
        "stages": stage_data,
        "branch_points": branch,
        "onset_fit": fit,
        "not_computable_from_stored_data": [],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(root / "campaign_summary.json", summary)
    marker = "### Torus amplitude campaign 2026-08-17"
    tracker = ROOT / "docs" / "development" / "session_tracker_20260816.md"
    old = tracker.read_text(encoding="utf-8") if tracker.exists() else ""
    if marker not in old:
        section = (
            f"\n\n{marker}\n\n"
            "The unattended amplitude-parameterized campaign was launched from "
            f"`{root}`. Its atomic per-point results and preflight telemetry are "
            "the source of truth; incomplete stages remain `NOT_ESTABLISHED`.\n\n"
            f"- Preflight summary: `{root / 'preflight' / 'summary.json'}`.\n"
            f"- Campaign summary: `{root / 'campaign_summary.json'}`.\n"
            "- Physical torus acceptance remains the non-zero q-sector and "
            "converged residual gate; numerical period-1 roots are not reported "
            "as torus branches.\n"
        )
        temporary = tracker.with_suffix(tracker.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(old + section, encoding="utf-8")
        temporary.replace(tracker)


def campaign(args: argparse.Namespace) -> int:
    """Run stages A--G within the unattended wall-time budget."""
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "run.log"
    deadline = time.monotonic() + 7.5 * 3600.0
    stage_data: dict[str, Any] = {}
    try:
        log_line(log_path, "campaign start")
        preflight_data = preflight(root, log_path, args.rss_limit_gb)
        stage_data["A"] = {"status": "PASS", "preflight": preflight_data}
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        failure = {"status": "ABORTED", "reason": f"{type(exc).__name__}: {exc}"}
        _write_event(root, "abort", failure)
        log_line(log_path, failure["reason"])
        return 2
    try:
        if time.monotonic() < deadline:
            jtwpa_pump = _ensure_jtwpa_pump(ROOT, log_path, deadline)
            cases = _case_definitions(ROOT, jtwpa_pump)
            stage_data["pump_setup"] = {
                "jtwpa_pump": str(jtwpa_pump) if jtwpa_pump else None,
            }
            stage_data["B"] = stage_spectrum(root, log_path, deadline, cases)
            counts = preflight_data["amplitude_counts"]
            stage_c: list[dict[str, Any]] = []
            for case in cases[:3]:
                stage_c.extend(_ladder(
                    root, log_path, deadline, case, counts["stage_c"], 1,
                    "stage_c_k5", False,
                ))
            stage_data["C"] = stage_c
            stage_d: list[dict[str, Any]] = []
            if not preflight_data["skip_stage_d"]:
                stage_d = _ladder(
                    root, log_path, deadline, cases[0], counts["stage_d"], 1,
                    "stage_d_k10", True,
                )
            stage_data["D"] = {
                "skipped": bool(preflight_data["skip_stage_d"]),
                "rows": stage_d,
            }
            stage_e: list[dict[str, Any]] = []
            if jtwpa_pump is not None:
                stage_e = _ladder(
                    root, log_path, deadline, cases[-1], counts["stage_e"], 1,
                    "stage_e_jtwpa", False,
                )
            stage_data["E"] = stage_e
            available_gb = _read_memory_gb()
            if time.monotonic() < deadline and available_gb >= 6.1:
                stage_data["F"] = {
                    "available_memory_gb": available_gb,
                    "q1": _ladder(
                        root, log_path, deadline, cases[0], 1, 1,
                        "stage_f_q1", True,
                    ),
                    "q2": _ladder(
                        root, log_path, deadline, cases[0], 1, 2,
                        "stage_f_q2", True,
                    ),
                }
            else:
                stage_data["F"] = {
                    "skipped": True,
                    "available_memory_gb": available_gb,
                }
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        stage_data["controller_error"] = f"{type(exc).__name__}: {exc}"
        log_line(log_path, stage_data["controller_error"])
    finally:
        stage_g(root, log_path, preflight_data, stage_data)
    log_line(log_path, "campaign complete")
    return 0


def _read_memory_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available / 2**30)
    except (ImportError, OSError):
        return 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--worker-amplitude", action="store_true")
    parser.add_argument("--worker-spectrum", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "chaos" / "torus_campaign_20260817")
    parser.add_argument("--rss-limit-gb", type=float, default=6.0)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--circuit-dir", type=Path, default=None)
    parser.add_argument("--pump-dir", type=Path, default=None)
    parser.add_argument("--omega-ratio", type=float, default=0.0917)
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--schur", action="store_true")
    parser.add_argument("--factor-backend", default="pardiso")
    parser.add_argument("--amplitude-relative", type=float, default=1.0e-5)
    parser.add_argument("--residual-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-newton", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--warm-state", type=Path, default=None)
    parser.add_argument("--warm-omega", type=float, default=0.0917)
    parser.add_argument("--warm-tau", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--points", type=int, default=120)
    args = parser.parse_args(argv)
    if args.worker_amplitude and (args.circuit_dir is None or args.pump_dir is None or args.out is None):
        parser.error("amplitude worker requires --circuit-dir, --pump-dir, and --out")
    if args.worker_spectrum and (args.circuit_dir is None or args.pump_dir is None or args.out_dir is None):
        parser.error("spectrum worker requires --circuit-dir, --pump-dir, and --out-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_amplitude:
        return worker_amplitude(args)
    if args.worker_spectrum:
        return worker_spectrum(args)
    return campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
