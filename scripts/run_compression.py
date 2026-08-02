"""Run a finite-signal compression sweep at one pump operating point."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from twpa_solver import default_loss_model
from twpa_solver.builders.jc_doc import build_fqjtwpa, build_jpa, build_jtwpa
from twpa_solver.core import CircuitMatrices, load_circuit
from twpa_solver.multitone.basis import (
    MultiToneBasis,
    build_lattice_basis,
    build_sideband_matched_basis,
    build_three_tone_basis,
)
from twpa_solver.multitone.compression import solve_signal_power_point
from twpa_solver.multitone.compression_curve import (
    build_compression_curve,
    depletion_only_gain_db,
    depletion_only_model,
    refine_p1db,
)
from twpa_solver.multitone.observables import (
    power_balance,
    spatial_profile_summary,
    spatial_profiles,
    tone_s21,
)
from twpa_solver.multitone.preconditioners import (
    MULTITONE_PRECONDITIONERS,
    resolve_multitone_preconditioner,
)
from twpa_solver.multitone.problem import FullMultiToneProblem
from twpa_solver.multitone.schur import (
    SchurMultiToneProblem,
    build_multitone_schur_problem,
)
from twpa_solver.multitone.stability import assess_multitone_stability
from twpa_solver.multitone.source import AffineSourcePath, MultiToneDrive
from twpa_solver.multitone.io import write_compression_outputs
from twpa_solver.pump import (
    FullPumpProblem,
    HarmonicGrid,
    HarmonicNewtonKrylovSolver,
    JosephsonBranchArray,
    NewtonKrylovSettings,
)
from twpa_solver.pump.basis import (
    load_pump_basis_from_solution,
    resolve_pump_basis,
)
from twpa_solver.multitone.resources import (
    FastCoupledFootprint,
    ResourceLimitExceeded,
    available_memory_gb,
    fast_coupled_footprint,
)
from twpa_solver.multitone.seed import promote_pump_solution

logger = logging.getLogger(__name__)


class P1dbRefinementFailed(RuntimeError):
    """A bracketed P1dB refinement solve did not converge.

    Carries the failing point's status so the caller can record why refinement
    was skipped, instead of losing an otherwise complete sweep.
    """

    def __init__(self, status: str) -> None:
        super().__init__(f"P1dB refinement solve failed: {status}")
        self.status = status

# Physical memory deliberately left unallocated when sizing worker concurrency.
# Sized as a fraction of one worker rather than a flat figure: a flat 2.0 GB was
# calibrated when a worker peaked at 3.04 GB and machines had ~15 GB free, and
# it silently costs a whole worker once the per-worker peak drops or the machine
# has only ~7 GB to give. The floor still keeps a real reserve on tiny bases.
_MEMORY_HEADROOM_FLOOR_GB = 0.75
_MEMORY_HEADROOM_FRACTION = 0.30


def _memory_headroom_gb(peak_gb: float) -> float:
    """Physical memory left unallocated when sizing worker concurrency."""
    return max(_MEMORY_HEADROOM_FLOOR_GB, _MEMORY_HEADROOM_FRACTION * peak_gb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-signal-power", type=int, default=5)
    parser.add_argument(
        "--p1db-power-tol-db",
        type=float,
        default=0.1,
        help="Power bracket tolerance for real-solve P1dB refinement; 0 disables it.",
    )
    parser.add_argument(
        "--multitone-basis",
        choices=("matched", "three_tone", "lattice"),
        default="matched",
    )
    parser.add_argument("--multitone-sidebands", type=int, default=2)
    parser.add_argument("--resource-budget-gb", type=float, default=8.0)
    parser.add_argument(
        "--multitone-backend",
        choices=("auto", "full", "schur_cpu_mt"),
        default="auto",
        help="auto selects Schur reduction for --circuit-dir and full for fixtures.",
    )
    parser.add_argument("--save-states", choices=("none", "last", "selected", "all"), default="none")
    parser.add_argument("--signal-ghz-min", type=float)
    parser.add_argument("--signal-ghz-max", type=float)
    parser.add_argument("--n-signal-freq", type=int, default=1)
    parser.add_argument("--signal-workers", type=int, default=1)
    parser.add_argument("--summary-json", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixture", choices=("jpa", "jtwpa", "fqjtwpa"))
    source.add_argument("--circuit-dir", type=Path)
    parser.add_argument("--pump-solution-dir", type=Path)
    parser.add_argument("--pump-freq-ghz", type=float, default=4.75001)
    parser.add_argument(
        "--signal-ghz",
        type=float,
        help="Signal frequency in GHz; optional for a frequency-range sweep.",
    )
    parser.add_argument("--pump-current-a", type=float)
    parser.add_argument(
        "--pump-current-list",
        type=float,
        nargs="+",
        help=(
            "Run one complete signal-power sweep for each pump current. "
            "Each current is written to its own resumable output directory."
        ),
    )
    parser.add_argument(
        "--pump-current-jc-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the pump current (default: 1.0, the "
            "validated convention in docs/pump_current_conversions.tex)."
        ),
    )
    parser.add_argument("--pump-mode-policy", default="positive_odd_jc")
    parser.add_argument("--pump-mode-count", type=int, default=10)
    parser.add_argument("--pump-modes")
    parser.add_argument("--pump-harmonics", type=int, default=10)
    parser.add_argument("--pump-nt", type=int, default=40)
    parser.add_argument("--source-port", type=int)
    parser.add_argument(
        "--pump-port",
        type=int,
        help="Pump injection port; defaults to --source-port for fixture-style circuits.",
    )
    parser.add_argument("--out-port", type=int)
    parser.add_argument("--diagnostic-port", type=int)
    parser.add_argument("--attenuation-db", type=float, default=None)
    parser.add_argument("--z0-ohm", type=float, default=50.0)
    parser.add_argument("--signal-current-min-a", type=float, default=1e-12)
    parser.add_argument("--signal-current-max-a", type=float, default=1e-9)
    parser.add_argument("--recovery", choices=("plain", "ladder"), default="ladder")
    parser.add_argument("--signal-substep-init-db", type=float, default=0.5)
    parser.add_argument("--signal-substep-min-db", type=float, default=0.01)
    parser.add_argument("--signal-continuation-deadline-s", type=float, default=0.0)
    parser.add_argument("--signal-arclength-recovery", action="store_true")
    parser.add_argument(
        "--check-stability",
        action="store_true",
        help="Measure the finite-signal q=0 Floquet slice at selected states.",
    )
    parser.add_argument(
        "--spatial-profiles",
        action="store_true",
        help="Write branch profiles at small, mid, and near-P1dB signal powers.",
    )
    parser.add_argument(
        "--spatial-profiles-all",
        action="store_true",
        help="Write branch profiles at the four selected saturation powers; requires --save-states all.",
    )
    parser.add_argument(
        "--multitone-preconditioner",
        choices=MULTITONE_PRECONDITIONERS,
        default="real_coupled_fast",
    )
    parser.add_argument(
        "--factor-backend",
        choices=("auto", "pardiso", "banded"),
        default="pardiso",
        help=(
            "Sparse factorization for the real_coupled_fast preconditioner. "
            "'banded' reorders node-major and stores the factors as a LAPACK "
            "general band: measured on jtwpa S=10 it peaks at 1.84 GB against "
            "2.51 GB, at 1.18x the wall time. That trade is worth taking only "
            "when the smaller footprint buys another worker -- on a ~7 GB "
            "budget it turns 2 workers into 3, which more than repays the "
            "slower factor. For a single-frequency run it is a pure loss."
        ),
    )
    parser.add_argument(
        "--allow-memory-overcommit",
        action="store_true",
        help=(
            "Start the sweep even when a single worker does not fit in free "
            "physical memory. Off by default: overcommitting this solve swaps "
            "rather than degrading, and typically takes the machine down."
        ),
    )
    parser.add_argument(
        "--precond-reuse",
        type=int,
        default=1,
        help=(
            "Reuse one preconditioner factor for up to N consecutive Newton "
            "steps (modified-Newton preconditioning). The Newton update is "
            "always taken against the true Jacobian via the matvec, so this "
            "cannot change the converged solution -- it trades extra GMRES "
            "iterations for skipped factorizations. 1 = refactor every step."
        ),
    )
    parser.add_argument(
        "--precond-reuse-refresh-gmres",
        type=int,
        default=0,
        help=(
            "Force an early refactor whenever the previous step needed at "
            "least this many GMRES iterations (staleness guard). 0 disables."
        ),
    )
    return parser


def _fixture_circuit(name: str) -> tuple[CircuitMatrices, dict[str, object]]:
    builders = {"jpa": build_jpa, "jtwpa": build_jtwpa, "fqjtwpa": build_fqjtwpa}
    builder, metadata = builders[name]()
    arrays = builder.assemble()
    return CircuitMatrices(
        C=arrays["C"],
        G=arrays["G"],
        K=arrays["K"],
        Bphi=arrays["Bphi"],
        Ic=arrays["Ic"],
        port_to_index=arrays["ports"],
        metadata=metadata,
    ), metadata


def _load_source(args: argparse.Namespace) -> tuple[CircuitMatrices, dict[str, object], str]:
    if args.circuit_dir is not None:
        circuit = load_circuit(args.circuit_dir)
        return circuit, circuit.metadata, str(args.circuit_dir)
    fixture = args.fixture or "jpa"
    circuit, metadata = _fixture_circuit(fixture)
    return circuit, metadata, fixture


def _resolve_attenuation(args: argparse.Namespace) -> tuple[float, str]:
    if args.attenuation_db is not None:
        return float(args.attenuation_db), "explicit"
    if args.circuit_dir is None:
        return 0.0, "fixture_default_zero"
    return (
        float(default_loss_model().attenuation_db(args.pump_freq_ghz)),
        "themis_default_loss_model",
    )


def _build_multitone_basis(
    args: argparse.Namespace,
    pump_modes: list[int],
    omega_p: float,
    delta: float,
) -> MultiToneBasis:
    # Matched sidebands fold Floquet indices into h; the largest retained h
    # grows with the requested sideband count, even for a fundamental-only
    # pump.  The previous ``max(pump_modes)+1`` clipped S=10 production bases.
    omega_max = omega_p * (
        max(pump_modes) + float(args.multitone_sidebands) + 1.0
    )
    if args.multitone_basis == "matched":
        basis = build_sideband_matched_basis(
            pump_modes,
            args.multitone_sidebands,
            omega_p,
            delta,
            omega_max,
        )
    elif args.multitone_basis == "lattice":
        basis = build_lattice_basis(
            pump_modes,
            args.multitone_sidebands,
            omega_p,
            delta,
            omega_max,
        )
    else:
        basis = build_three_tone_basis(omega_p, delta)
    represented_modes = {tone.h for tone in basis.tones if tone.q == 0}
    missing_modes = sorted(set(pump_modes) - represented_modes)
    if missing_modes:
        raise ValueError(
            "multitone basis silently truncates pump harmonics: "
            f"pump_modes={list(pump_modes)}, "
            f"multitone_q0_modes={sorted(represented_modes)}, "
            f"missing={missing_modes}"
        )
    return basis


def _current_to_dbm(current_a: float, z0_ohm: float, attenuation_db: float) -> float:
    power_w = current_a * current_a * z0_ohm / 2.0
    return 10.0 * math.log10(power_w / 1.0e-3) + attenuation_db


def _interpolate_p1db_current(points: list[dict[str, object]]) -> float | None:
    adjacent_valid = (
        (left, right)
        for left, right in zip(points, points[1:])
        if left["status"] == "VALID_SOLVED"
        and right["status"] == "VALID_SOLVED"
        and np.isfinite(float(left["compression_db"]))
        and np.isfinite(float(right["compression_db"]))
    )
    for left, right in adjacent_valid:
        c_left = float(left["compression_db"])
        c_right = float(right["compression_db"])
        if c_left <= 1.0 <= c_right and c_right > c_left:
            fraction = (1.0 - c_left) / (c_right - c_left)
            log_left = math.log10(float(left["signal_current_a"]))
            log_right = math.log10(float(right["signal_current_a"]))
            return 10.0 ** (log_left + fraction * (log_right - log_left))
    return None


def _solve_compression(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, float]],
    dict[str, np.ndarray],
    dict[str, object],
    list[dict[str, object]],
]:
    circuit, metadata, circuit_source = _load_source(args)
    attenuation_db, attenuation_source = _resolve_attenuation(args)
    source_port = int(args.source_port or 1)
    pump_port = int(args.pump_port or source_port)
    out_port = int(args.out_port or (1 if circuit_source == "jpa" else 2))
    diagnostic_port = int(args.diagnostic_port or out_port)
    for label, port in (("pump", pump_port), ("source", source_port), ("output", out_port), ("diagnostic", diagnostic_port)):
        if port not in circuit.port_to_index:
            raise ValueError(f"{label} port {port} is absent from circuit ports {sorted(circuit.port_to_index)}")
    pump_sources = metadata.get("pump_sources", [])
    default_current = pump_sources[0].get("current_a") if pump_sources else None
    if args.pump_current_a is None and default_current is None:
        raise ValueError("--pump-current-a is required when circuit metadata has no pump source")
    pump_current = float(args.pump_current_a or default_current) * float(args.pump_current_jc_scale)
    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9
    selected_preconditioner = resolve_multitone_preconditioner(
        args.multitone_preconditioner
    )
    solver_preconditioner = (
        "real_coupled_fast"
        if selected_preconditioner == "floquet_sector"
        else selected_preconditioner
    )
    settings = NewtonKrylovSettings(
        newton_tol=1e-10, max_newton=20, gmres_rtol=1e-8, gmres_atol=0.0,
        gmres_restart=20, gmres_maxiter=40, min_alpha=1.0 / 1024.0,
        preconditioner=solver_preconditioner,
        compute_time_residual=False, verbose=False,
        continuation_predictor="none", jvp_mode="aft",
        precond_reuse=max(1, int(args.precond_reuse)),
        precond_reuse_refresh_gmres=max(0, int(args.precond_reuse_refresh_gmres)),
    )
    if args.pump_solution_dir is not None:
        pump_state, pump_basis = load_pump_basis_from_solution(
            args.pump_solution_dir,
            fallback_omega_p=omega_p,
        )
        pump_reports: list[object] = []
        pump_converged = True
    else:
        pump_basis = resolve_pump_basis(
            policy=args.pump_mode_policy,
            omega_p=omega_p,
            harmonics=args.pump_harmonics,
            mode_count=args.pump_mode_count,
            explicit_modes=args.pump_modes,
            design_meta=metadata,
        )
        pump_problem = FullPumpProblem(
            C=circuit.C, G=circuit.G, K=circuit.K, Bphi=circuit.Bphi,
            branch=JosephsonBranchArray(circuit.Ic, circuit.phi0),
            grid=HarmonicGrid(np.asarray(pump_basis.modes), nt=args.pump_nt, omega=omega_p),
            pump_node_index=circuit.port_to_index[pump_port], pump_current_a=pump_current,
        )
        # The pump solve keeps precond_reuse=1 so its Newton/GMRES iterate path
        # -- and therefore the pump state other artifacts are pinned against --
        # is unchanged by the multitone reuse setting.
        pump_settings = replace(
            settings,
            preconditioner="real_coupled",
            precond_reuse=1,
            precond_reuse_refresh_gmres=0,
        )
        pump_state, pump_reports = HarmonicNewtonKrylovSolver(
            pump_settings
        ).solve_continuation(pump_problem, continuation_steps=4)
        pump_converged = bool(pump_reports[-1].converged)
        if not pump_converged:
            final = pump_reports[-1]
            raise RuntimeError(
                "pump continuation failed before the multitone solve: "
                f"source_scale={final.source_scale}, "
                f"coeff_rel={final.coeff_rel:.6g}, "
                f"reason={final.failure_reason}"
            )
    delta = omega_p - 2.0 * math.pi * args.signal_ghz * 1e9
    basis = _build_multitone_basis(args, pump_basis.modes, omega_p, delta)
    pump_seed = promote_pump_solution(pump_state, pump_basis, basis)
    pump_source = MultiToneDrive(basis.pump_tone, circuit.port_to_index[pump_port], pump_current).to_coeffs(
        basis, circuit.node_count
    )
    signal_unit = MultiToneDrive(basis.signal_tone, circuit.port_to_index[source_port], 1.0).to_coeffs(
        basis, circuit.node_count
    )
    selected_backend = (
        "schur_cpu_mt"
        if args.multitone_backend == "auto" and args.circuit_dir is not None
        else "full"
        if args.multitone_backend == "auto"
        else args.multitone_backend
    )
    schur_partition = None
    # One cache for the whole run. Every problem built here shares the same
    # circuit and basis and differs only in source path, while the dynamic
    # blocks and the fast preconditioner (matrix, scatter map, factors) depend
    # on neither -- so they must be built once. Giving each problem its own
    # cache keeps one full preconditioner alive per problem, which at S=10 is
    # ~2.6 GB apiece and exhausts memory before the solve starts.
    problem_cache: dict[object, object] = {}

    def make_problem(path: AffineSourcePath):
        nonlocal schur_partition
        full = FullMultiToneProblem(
            circuit,
            basis,
            path,
            preconditioner=selected_preconditioner,
            cache=problem_cache,
        )
        if selected_backend == "full":
            return full
        if schur_partition is None:
            reduced = build_multitone_schur_problem(
                full,
                list(circuit.port_to_index.values()),
                preconditioner=selected_preconditioner,
            )
            schur_partition = reduced.partition
            return reduced
        return SchurMultiToneProblem(
            full,
            schur_partition,
            preconditioner=selected_preconditioner,
        )

    def solve_seed(full_state: np.ndarray) -> np.ndarray:
        if selected_backend == "full":
            return full_state
        if schur_partition is None:
            raise RuntimeError("Schur partition has not been initialized")
        return full_state[:, schur_partition.retained]

    def observable_state(problem, state: np.ndarray) -> np.ndarray:
        if isinstance(problem, SchurMultiToneProblem):
            return problem.reconstruct_full(state)
        return state

    currents = np.geomspace(args.signal_current_min_a, args.signal_current_max_a, max(args.n_signal_power, 1))
    pump_off_path = AffineSourcePath.signal_turn_on(
        np.zeros_like(pump_source),
        signal_unit * float(currents[0]),
    )
    pump_off_problem = make_problem(pump_off_path)
    pump_off_state, pump_off_report = HarmonicNewtonKrylovSolver(settings).solve_one(
        pump_off_problem,
        pump_off_problem.zeros(),
        1.0,
    )
    if not pump_off_report.converged:
        raise RuntimeError("pump-off small-signal reference did not converge")
    pump_off_state_full = observable_state(pump_off_problem, pump_off_state)
    pump_off_s21 = tone_s21(
        pump_off_state_full,
        basis,
        circuit,
        signal_tone=basis.signal_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=float(currents[0]),
        z0_ohm=args.z0_ohm,
    )
    pump_off_gain_db = float(20.0 * np.log10(max(abs(pump_off_s21), 1e-300)))
    signal_row = basis.index_of(basis.signal_tone)
    # ``output_node`` indexes the full node set, so the reference voltage has to
    # be read off the reconstructed state.  The Schur backend solves on the
    # retained ports only, and indexing it directly is out of bounds for every
    # ``--circuit-dir`` device.
    output_node = circuit.port_to_index[out_port]
    pump_off_signal_voltage = (
        1j * basis.omegas[signal_row] * pump_off_state_full[signal_row, output_node]
    )
    pump_only_problem = make_problem(AffineSourcePath.pump_turn_on(pump_source))
    pump_seed_solve = solve_seed(pump_seed)
    pump_only_state, pump_only_report = HarmonicNewtonKrylovSolver(settings).solve_one(
        pump_only_problem,
        pump_seed_solve,
        1.0,
    )
    if not pump_only_report.converged:
        pump_only_state, pump_only_reports, pump_only_trace = (
            HarmonicNewtonKrylovSolver(settings).solve_adaptive_continuation(
                pump_only_problem,
                None,
                initial_step=0.25,
                min_step=0.01,
                growth=1.5,
                shrink=0.5,
                fallback_fixed_steps=20,
                max_wall_s=args.signal_continuation_deadline_s,
            )
        )
        if (
            not pump_only_reports
            or not pump_only_reports[-1].converged
            or not pump_only_trace.accepted_lambdas
            or pump_only_trace.accepted_lambdas[-1] < 1.0 - 1e-12
        ):
            reason = (
                pump_only_reports[-1].failure_reason
                if pump_only_reports
                else pump_only_trace.failure_reason
            )
            raise RuntimeError(
                "pump-on signal-off adaptive reference did not converge: "
                f"{reason}"
            )
    pump_reference_s21 = tone_s21(
        observable_state(pump_only_problem, pump_only_state),
        basis,
        circuit,
        signal_tone=basis.pump_tone,
        source_port=source_port,
        out_port=out_port,
        source_current_a=pump_current,
        z0_ohm=args.z0_ohm,
    )

    pump_only_state_full = observable_state(pump_only_problem, pump_only_state)

    def gain_vs_off(state_full: np.ndarray, signal_current_a: float) -> float:
        signal_voltage = (
            1j
            * basis.omegas[signal_row]
            * state_full[signal_row, output_node]
        )
        return float(
            20.0
            * np.log10(
                max(
                    abs(
                        (signal_voltage / float(signal_current_a))
                        / (pump_off_signal_voltage / float(currents[0]))
                    ),
                    1e-300,
                )
            )
        )

    def measure_state(
        state_full: np.ndarray, signal_current_a: float
    ) -> dict[str, object]:
        """Extract all per-state signal, pump, residual, and balance observables."""
        signal_s21 = tone_s21(
            state_full,
            basis,
            circuit,
            signal_tone=basis.signal_tone,
            source_port=source_port,
            out_port=out_port,
            source_current_a=float(signal_current_a),
            z0_ohm=args.z0_ohm,
        )
        pump_s21 = tone_s21(
            state_full,
            basis,
            circuit,
            signal_tone=basis.pump_tone,
            source_port=source_port,
            out_port=out_port,
            source_current_a=pump_current,
            z0_ohm=args.z0_ohm,
        )
        idler_s21 = tone_s21(
            state_full,
            basis,
            circuit,
            signal_tone=basis.idler_tone,
            source_port=source_port,
            out_port=out_port,
            source_current_a=float(signal_current_a),
            z0_ohm=args.z0_ohm,
        )
        gain_db = float(20.0 * np.log10(max(abs(signal_s21), 1e-300)))
        gain_vs_off_db = gain_vs_off(state_full, signal_current_a)
        pump_depletion_db = float(
            20.0 * np.log10(max(abs(pump_s21), 1e-300))
            - 20.0 * np.log10(max(abs(pump_reference_s21), 1e-300))
        )
        gain_linear = float(10.0 ** (gain_vs_off_db / 10.0))
        signal_power = float(signal_current_a**2 * args.z0_ohm / 2.0)
        pump_power = float(pump_current**2 * args.z0_ohm / 2.0)
        depletion_model = depletion_only_gain_db(
            gain_linear, signal_power, pump_power
        )
        balance = power_balance(
            state_full,
            basis,
            circuit,
            reference_X_full=pump_only_state_full,
            z0_ohm=args.z0_ohm,
        )
        residual_problem = FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.signal_turn_on(
                pump_source, signal_unit * float(signal_current_a)
            ),
        )
        residual = residual_problem.residual_coeffs(state_full, 1.0)
        source_norm = np.linalg.norm(residual_problem.source_coeffs(1.0))
        hb_residual_rel = float(
            np.linalg.norm(residual) / max(source_norm, 1e-300)
        )
        return {
            "signal_s21": signal_s21,
            "pump_s21": pump_s21,
            "idler_s21": idler_s21,
            "gain_db": gain_db,
            "gain_vs_off_db": gain_vs_off_db,
            "pump_depletion_db": pump_depletion_db,
            "depletion_model": depletion_model,
            "balance": balance,
            "hb_residual_rel": hb_residual_rel,
        }
    states: dict[str, np.ndarray] = {}
    state_by_current: dict[float, np.ndarray] = {}
    full_state_by_current: dict[float, np.ndarray] = {}
    points: list[dict[str, float]] = []
    previous = pump_seed_solve
    previous_previous = None
    previous_current = 0.0
    previous_previous_current = 0.0
    reference_gain = None
    for index, current in enumerate(currents):
        base_problem = make_problem(AffineSourcePath.pump_turn_on(pump_source))
        solved = solve_signal_power_point(
            base_problem,
            previous,
            previous_previous,
            float(current),
            pump_source=pump_source,
            signal_source=signal_unit,
            solver=HarmonicNewtonKrylovSolver(settings),
            signal_current_prev_a=previous_current,
            signal_current_prevprev_a=previous_previous_current,
            recovery=args.recovery,
            pump_seed=pump_seed_solve,
            signal_substep_init_db=args.signal_substep_init_db,
            signal_substep_min_db=args.signal_substep_min_db,
            continuation_deadline_s=args.signal_continuation_deadline_s,
            arclength_recovery=args.signal_arclength_recovery,
        )
        state = solved.state
        state_full = observable_state(base_problem, state)
        if solved.status == "VALID_SOLVED":
            metrics = measure_state(state_full, float(current))
            signal_s21 = metrics["signal_s21"]
            pump_s21 = metrics["pump_s21"]
            idler_s21 = metrics["idler_s21"]
            gain_db = float(metrics["gain_db"])
            gain_vs_off_db = float(metrics["gain_vs_off_db"])
            pump_depletion_db = float(metrics["pump_depletion_db"])
            depletion_model = float(metrics["depletion_model"])
            balance = metrics["balance"]
            hb_residual_rel = float(metrics["hb_residual_rel"])
            reference_gain = (
                gain_vs_off_db if reference_gain is None else reference_gain
            )
            previous_previous = previous
            previous_previous_current = previous_current
            previous = state
            previous_current = float(current)
            state_by_current[float(current)] = state
            full_state_by_current[float(current)] = state_full
        else:
            gain_db = float("nan")
            gain_vs_off_db = float("nan")
            signal_s21 = pump_s21 = idler_s21 = complex(float("nan"), float("nan"))
            pump_depletion_db = float("nan")
            depletion_model = float("nan")
            balance = {
                "power_balance_rel_err": float("nan"),
                "external_power_balance_rel_err": float("nan"),
                "manley_rowe_photon_flux": float("nan"),
                "manley_rowe_photon_scale": float("nan"),
                "manley_rowe_evaluable": float("nan"),
                "manley_rowe_rel_err": float("nan"),
                "external_supplied_power": float("nan"),
                "external_dissipated_power": float("nan"),
                "external_manley_rowe_photon_flux": float("nan"),
                "external_manley_rowe_photon_scale": float("nan"),
                "external_manley_rowe_evaluable": float("nan"),
                "external_manley_rowe_rel_err": float("nan"),
                "pump_net_power_w": float("nan"),
                "pump_reference_net_power_w": float("nan"),
                "pump_net_power_delta_w": float("nan"),
                "pump_outgoing_power_w": float("nan"),
                "pump_reference_outgoing_power_w": float("nan"),
                "pump_depletion_all_port_db": float("nan"),
            }
            hb_residual_rel = float("nan")
        points.append({
            "signal_current_a": float(current),
            "signal_power_dbm": _current_to_dbm(
                float(current), args.z0_ohm, attenuation_db
            ),
            "gain_db": gain_db,
            "gain_vs_off_db": gain_vs_off_db,
            "pump_depletion_db": pump_depletion_db,
            "compression_model_depletion_only": depletion_model,
            "power_balance_rel_err": balance["power_balance_rel_err"],
            "external_power_balance_rel_err": balance[
                "external_power_balance_rel_err"
            ],
            "hb_residual_rel": hb_residual_rel,
            "manley_rowe_photon_flux": balance["manley_rowe_photon_flux"],
            "manley_rowe_photon_scale": balance["manley_rowe_photon_scale"],
            "manley_rowe_evaluable": balance["manley_rowe_evaluable"],
            "manley_rowe_rel_err": balance["manley_rowe_rel_err"],
            "external_supplied_power": balance["external_supplied_power"],
            "external_dissipated_power": balance["external_dissipated_power"],
            "external_manley_rowe_photon_flux": balance[
                "external_manley_rowe_photon_flux"
            ],
            "external_manley_rowe_photon_scale": balance[
                "external_manley_rowe_photon_scale"
            ],
            "external_manley_rowe_evaluable": balance[
                "external_manley_rowe_evaluable"
            ],
            "external_manley_rowe_rel_err": balance[
                "external_manley_rowe_rel_err"
            ],
            "pump_net_power_w": balance["pump_net_power_w"],
            "pump_reference_net_power_w": balance[
                "pump_reference_net_power_w"
            ],
            "pump_net_power_delta_w": balance["pump_net_power_delta_w"],
            "pump_outgoing_power_w": balance["pump_outgoing_power_w"],
            "pump_reference_outgoing_power_w": balance[
                "pump_reference_outgoing_power_w"
            ],
            "pump_depletion_all_port_db": balance[
                "pump_depletion_all_port_db"
            ],
            "signal_s21_real": float(np.real(signal_s21)),
            "signal_s21_imag": float(np.imag(signal_s21)),
            "pump_s21_real": float(np.real(pump_s21)),
            "pump_s21_imag": float(np.imag(pump_s21)),
            "idler_s21_real": float(np.real(idler_s21)),
            "idler_s21_imag": float(np.imag(idler_s21)),
            "compression_db": (
                float(reference_gain - gain_vs_off_db)
                if reference_gain is not None and np.isfinite(gain_vs_off_db)
                else float("nan")
            ),
            "status": solved.status,
            "recovery_rung": solved.used_recovery,
            "last_converged_signal_current_a": solved.last_converged_signal_current_a,
        })
        if index == len(currents) - 1:
            states["last"] = state_full
        if solved.status == "VALID_SOLVED":
            if index == 0:
                states["zero_signal"] = state_full
            if index == len(currents) // 2:
                states["mid"] = state_full
    small_signal_gain_db = (
        float(points[0]["gain_vs_off_db"]) if points else float("nan")
    )
    no_gain = not np.isfinite(small_signal_gain_db) or small_signal_gain_db < 3.0
    if no_gain:
        for point in points:
            point["compression_db"] = float("nan")
    curve = build_compression_curve(
        [float(point["signal_power_dbm"]) for point in points],
        [float(point["gain_vs_off_db"]) for point in points],
        small_signal_gain_db,
        [str(point["status"]) for point in points],
    )
    p1db_current_a = None if no_gain else _interpolate_p1db_current(points)
    # Kept even when refinement overwrites p1db_current_a: the refined-versus-
    # interpolated delta is the number that decides whether already-published
    # sweeps need re-running, and reading it off two separate runs would fold
    # run-to-run variation into a comparison that has none.
    p1db_interpolated_current_a = p1db_current_a
    p1db_method = "interpolated"
    p1db_refinement_failure: str | None = None
    adjacent_valid = (
        (left, right)
        for left, right in zip(points, points[1:])
        if left["status"] == "VALID_SOLVED"
        and right["status"] == "VALID_SOLVED"
        and np.isfinite(float(left["compression_db"]))
        and np.isfinite(float(right["compression_db"]))
    )
    for left, right in adjacent_valid:
        if (
            float(left["compression_db"]) < 1.0
            <= float(right["compression_db"])
        ):
            if not no_gain and args.p1db_power_tol_db > 0.0:
                bracket = (
                    float(left["signal_power_dbm"]),
                    float(right["signal_power_dbm"]),
                )

                def evaluate(power_dbm: float) -> float:
                    current_trial = math.sqrt(
                        2.0 * 1.0e-3
                        * 10.0 ** ((power_dbm - attenuation_db) / 10.0)
                        / args.z0_ohm
                    )
                    nearest = min(
                        state_by_current,
                        key=lambda value: abs(math.log(value / current_trial)),
                    )
                    trial_problem = make_problem(
                        AffineSourcePath.pump_turn_on(pump_source)
                    )
                    candidate = solve_signal_power_point(
                        trial_problem,
                        state_by_current[nearest],
                        None,
                        current_trial,
                        pump_source=pump_source,
                        signal_source=signal_unit,
                        solver=HarmonicNewtonKrylovSolver(settings),
                        signal_current_prev_a=nearest,
                        recovery="ladder",
                        pump_seed=pump_seed_solve,
                        signal_substep_init_db=args.signal_substep_init_db,
                        signal_substep_min_db=args.signal_substep_min_db,
                        continuation_deadline_s=(
                            args.signal_continuation_deadline_s
                        ),
                        arclength_recovery=args.signal_arclength_recovery,
                    )
                    if candidate.status != "VALID_SOLVED":
                        raise P1dbRefinementFailed(candidate.status)
                    trial_state = observable_state(trial_problem, candidate.state)
                    trial_gain_vs_off_db = gain_vs_off(
                        trial_state, current_trial
                    )
                    return float(
                        reference_gain - trial_gain_vs_off_db
                    )

                # A refinement solve that will not converge is a statement
                # about one mid-bracket power, not about the sweep that
                # bracketed it.  Degrading to the interpolated P1dB keeps every
                # solved point; raising here discarded the whole run's
                # artifacts after the sweep had already succeeded.
                try:
                    p1db_dbm_refined = refine_p1db(
                        evaluate,
                        bracket,
                        tolerance_db=args.p1db_power_tol_db,
                    )
                except (P1dbRefinementFailed, ValueError) as exc:
                    p1db_refinement_failure = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "P1dB refinement degraded to interpolation (%s)",
                        p1db_refinement_failure,
                    )
                else:
                    p1db_current_a = math.sqrt(
                        2.0 * 1.0e-3
                        * 10.0 ** ((p1db_dbm_refined - attenuation_db) / 10.0)
                        / args.z0_ohm
                    )
                    p1db_method = "refined"
            break
    p1db_dbm = (
        _current_to_dbm(p1db_current_a, args.z0_ohm, attenuation_db)
        if p1db_current_a is not None
        else None
    )
    refined_state_full: np.ndarray | None = None
    p1db_metrics: dict[str, object] | None = None
    p1db_state_status = "NOT_REQUESTED"
    if p1db_current_a is not None and state_by_current:
        nearest = min(
            state_by_current,
            key=lambda value: abs(math.log(value / p1db_current_a)),
        )
        p1db_problem = make_problem(
            AffineSourcePath.pump_turn_on(pump_source)
        )
        p1db_solution = solve_signal_power_point(
            p1db_problem,
            state_by_current[nearest],
            None,
            float(p1db_current_a),
            pump_source=pump_source,
            signal_source=signal_unit,
            solver=HarmonicNewtonKrylovSolver(settings),
            signal_current_prev_a=nearest,
            recovery="ladder",
            pump_seed=pump_seed_solve,
        )
        p1db_state_status = p1db_solution.status
        if p1db_solution.status == "VALID_SOLVED":
            refined_state_full = observable_state(
                p1db_problem, p1db_solution.state
            )
            p1db_metrics = measure_state(
                refined_state_full, float(p1db_current_a)
            )
            states["p1db"] = refined_state_full
    p1db_output_dbm = (
        p1db_dbm + small_signal_gain_db - 1.0
        if p1db_dbm is not None
        else None
    )
    summary_status = (
        "NO_GAIN_AT_OPERATING_POINT"
        if no_gain
        else "VALID_SOLVED"
        if all(point["status"] == "VALID_SOLVED" for point in points)
        else "CHECK"
    )
    if p1db_current_a is not None and p1db_state_status != "VALID_SOLVED":
        summary_status = "CHECK"
    stability_points = {}
    stability_status = "NOT_CHECKED"
    if args.check_stability:
        stability_problem = FullMultiToneProblem(
            circuit,
            basis,
            AffineSourcePath.pump_turn_on(pump_source),
            loss_model=getattr(circuit, "loss_model", "current_complex_c"),
        )
        for label in ("zero_signal", "p1db", "last"):
            candidate_state = states.get(label)
            if candidate_state is None:
                continue
            result = assess_multitone_stability(
                stability_problem,
                candidate_state,
                signal_ghz=args.signal_ghz,
            )
            stability_points[label] = {
                "status": result.status,
                "dominant_exponent_per_s": result.dominant_exponent_per_s,
                "sigma_min": result.sigma_min,
                "matrix_size": result.matrix_size,
                "torus_resolution": list(result.torus_resolution),
                "reason": result.reason,
            }
        statuses = [item["status"] for item in stability_points.values()]
        stability_status = (
            "STABLE" if statuses and all(status == "STABLE" for status in statuses)
            else "UNSTABLE" if "UNSTABLE" in statuses
            else "INCONCLUSIVE"
        )
    summary = {
        "status": summary_status,
        "stability_status": stability_status,
        "stability_points": stability_points,
        "circuit_source": circuit_source,
        "pump_freq_ghz": args.pump_freq_ghz,
        "signal_ghz": args.signal_ghz,
        "pump_current_a": pump_current,
        "pump_converged": pump_converged,
        "pump_current_jc_scale": args.pump_current_jc_scale,
        "pump_solution_dir": str(args.pump_solution_dir) if args.pump_solution_dir else None,
        "source_port": source_port,
        "pump_port": pump_port,
        "out_port": out_port,
        "diagnostic_port": diagnostic_port,
        "attenuation_db": attenuation_db,
        "attenuation_source": attenuation_source,
        "small_signal_gain_db": small_signal_gain_db,
        "small_signal_gain_vs_off_db": float(points[0]["gain_vs_off_db"]),
        "pump_off_gain_db": pump_off_gain_db,
        "pump_reference_s21_real": float(np.real(pump_reference_s21)),
        "pump_reference_s21_imag": float(np.imag(pump_reference_s21)),
        "p1db": p1db_dbm,
        "p1db_method": "none" if no_gain else p1db_method,
        "p1db_interpolated_dbm": (
            _current_to_dbm(
                p1db_interpolated_current_a, args.z0_ohm, attenuation_db
            )
            if p1db_interpolated_current_a is not None
            else None
        ),
        "first_1db_crossing_dbm": curve.first_1db_crossing_dbm,
        "number_of_crossings": curve.number_of_crossings,
        "n_requested_power_points": len(points),
        "n_failed_power_points": len(curve.failed_signal_power_dbm),
        "failed_signal_power_dbm": list(curve.failed_signal_power_dbm),
        "failed_power_point_statuses": [
            str(point["status"])
            for point in points
            if point["status"] != "VALID_SOLVED"
            or not np.isfinite(float(point["compression_db"]))
        ],
        "p1db_degraded": bool(curve.failed_signal_power_dbm),
        "p1db_refinement_failure": p1db_refinement_failure,
        "nonmonotonic_compression": curve.nonmonotonic_compression,
        "compression_model_depletion_only_description": (
            "dB gain from the linear-gain depletion trend; not an acceptance oracle"
        ),
        "max_power_balance_rel_err": max(
            (float(point["power_balance_rel_err"]) for point in points
             if np.isfinite(float(point["power_balance_rel_err"]))),
            default=None,
        ),
        "max_external_power_balance_rel_err": max(
            (
                float(point["external_power_balance_rel_err"])
                for point in points
                if np.isfinite(float(point["external_power_balance_rel_err"]))
            ),
            default=None,
        ),
        "max_manley_rowe_rel_err": max(
            (float(point["manley_rowe_rel_err"]) for point in points
             if np.isfinite(float(point["manley_rowe_rel_err"]))),
            default=None,
        ),
        "max_external_manley_rowe_rel_err": max(
            (
                float(point["external_manley_rowe_rel_err"])
                for point in points
                if np.isfinite(float(point["external_manley_rowe_rel_err"]))
            ),
            default=None,
        ),
        "p1db_signal_current_a": p1db_current_a,
        "p1db_input_dbm": p1db_dbm,
        "p1db_output_dbm": p1db_output_dbm,
        "p1db_state_status": p1db_state_status,
        "p1db_pump_depletion_db": (
            float(p1db_metrics["pump_depletion_db"])
            if p1db_metrics is not None
            else None
        ),
        "p1db_pump_depletion_all_port_db": (
            p1db_metrics["balance"]["pump_depletion_all_port_db"]
            if p1db_metrics is not None
            else None
        ),
        "p1db_pump_depletion_all_port_db_status": (
            "TRUSTED_POST_EXP29_TRACK1"
            if p1db_metrics is not None
            else "NOT_AVAILABLE"
        ),
        "p1db_pump_net_power_delta_w": (
            p1db_metrics["balance"]["pump_net_power_delta_w"]
            if p1db_metrics is not None
            else None
        ),
        "p1db_external_supplied_power": (
            p1db_metrics["balance"]["external_supplied_power"]
            if p1db_metrics is not None
            else None
        ),
        "p1db_external_power_balance_rel_err": (
            p1db_metrics["balance"]["external_power_balance_rel_err"]
            if p1db_metrics is not None
            else None
        ),
        "message": (
            f"Small-signal gain {small_signal_gain_db:.6g} dB is below the "
            "3 dB compression-study threshold; P1dB is not reported."
            if no_gain
            else None
        ),
        "multitone_preconditioner": selected_preconditioner,
        "multitone_backend": selected_backend,
        "factor_backend": args.factor_backend,
        "recovery": args.recovery,
        "basis": basis.to_metadata(),
    }
    spatial_rows: list[dict[str, object]] = []
    if args.spatial_profiles:
        for label in ("zero_signal", "mid", "p1db"):
            if label not in states:
                continue
            for row in spatial_profiles(states[label], basis, circuit):
                spatial_rows.append({"operating_point": label, **row})
    if args.save_states == "all":
        for index, current in enumerate(currents):
            state = state_by_current.get(float(current))
            if state is not None:
                states[f"signal_{index:04d}"] = state
    if args.spatial_profiles_all:
        if args.save_states != "all":
            raise ValueError("--spatial-profiles-all requires --save-states all")
        valid_currents = sorted(state_by_current)
        if not valid_currents:
            raise ValueError("no converged signal states available for spatial profiles")
        if p1db_current_a is None:
            raise ValueError("P1dB is required for --spatial-profiles-all")
        targets = {
            "smallest": valid_currents[0],
            "decade_below_p1db": p1db_current_a / math.sqrt(10.0),
            "p1db": p1db_current_a,
            "largest_converged": valid_currents[-1],
        }
        for label, target_current in targets.items():
            if label == "p1db" and refined_state_full is not None:
                for row in spatial_profiles(refined_state_full, basis, circuit):
                    spatial_rows.append(
                        {
                            "operating_point": label,
                            "selected_signal_current_a": float(p1db_current_a),
                            "target_signal_current_a": float(target_current),
                            **row,
                        }
                    )
                continue
            selected_current = min(
                valid_currents,
                key=lambda current: abs(math.log(current / target_current)),
            )
            for row in spatial_profiles(full_state_by_current[selected_current], basis, circuit):
                spatial_rows.append(
                    {
                        "operating_point": label,
                        "selected_signal_current_a": float(selected_current),
                        "target_signal_current_a": float(target_current),
                        **row,
                    }
                )
    if spatial_rows:
        summary["spatial_profile_summary"] = {}
        for label in sorted({str(row["operating_point"]) for row in spatial_rows}):
            label_rows = [row for row in spatial_rows if row["operating_point"] == label]
            unique_rows = {int(row["branch_index"]): row for row in label_rows}
            summary["spatial_profile_summary"][label] = spatial_profile_summary(
                list(unique_rows.values())
            )
    return points, states, summary, spatial_rows


def _write_one_result(args: argparse.Namespace) -> dict[str, object]:
    points, states, summary, spatial_rows = _solve_compression(args)
    summary.update({
        "multitone_basis": args.multitone_basis,
        "multitone_sidebands": args.multitone_sidebands,
        "n_signal_power": args.n_signal_power,
        "resource_budget_gb": args.resource_budget_gb,
    })
    write_compression_outputs(
        args.output_dir,
        points,
        summary=summary,
        states=states,
        save_states=args.save_states,
        spatial_rows=spatial_rows if args.spatial_profiles else None,
    )
    if summary.get("p1db_state_status") == "VALID_SOLVED":
        print(
            "[run_compression] pump depletion at P1dB: "
            f"single-port={summary['p1db_pump_depletion_db']:.6g} dB, "
            f"all-port={summary['p1db_pump_depletion_all_port_db']:.6g} dB, "
            f"net_delta={summary['p1db_pump_net_power_delta_w']:.6g} W",
            flush=True,
        )
    return summary


def _frequency_worker(payload: tuple[dict[str, object], float, str]) -> dict[str, object]:
    values, frequency_ghz, output_dir = payload
    args = argparse.Namespace(**values)
    args.signal_ghz = frequency_ghz
    args.output_dir = Path(output_dir)
    args.summary_json = None
    return _write_one_result(args)


def _pump_output_dir(root: Path, index: int, current_a: float) -> Path:
    """Return the deterministic output directory for one pump-current run."""
    return root / f"pump_{index:03d}_{current_a:.12g}a"


def _resumable_summary(
    path: Path, *, expected_pump_current_a: float
) -> dict[str, object] | None:
    """Load a summary only when it belongs to the requested pump current."""
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        actual = float(summary["pump_current_a"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not np.isclose(actual, expected_pump_current_a, rtol=1e-12, atol=1e-18):
        return None
    return summary


def _run_pump_current_sweep(args: argparse.Namespace) -> dict[str, object]:
    """Run or resume complete signal sweeps along the pump-current axis."""
    requested = args.pump_current_list
    if not requested:
        raise ValueError("--pump-current-list must contain at least one current")
    currents = [float(value) for value in requested]
    if any(not np.isfinite(value) or value <= 0.0 for value in currents):
        raise ValueError("pump currents must be finite and positive")
    if len(set(currents)) != len(currents):
        raise ValueError("--pump-current-list must not contain duplicate currents")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, current_a in enumerate(currents):
        run_dir = _pump_output_dir(args.output_dir, index, current_a)
        run_args = argparse.Namespace(**vars(args))
        run_args.output_dir = run_dir
        run_args.pump_current_a = current_a
        run_args.pump_current_list = None
        run_args.summary_json = None
        expected_current = current_a * float(args.pump_current_jc_scale)

        if int(args.n_signal_freq) > 1:
            frequency_summary_path = run_dir / "frequency_summary.json"
            existing = _resumable_summary(
                frequency_summary_path,
                expected_pump_current_a=expected_current,
            )
            if existing is None:
                frequency_summary = _run_frequency_sweep(run_args)
                frequency_summary["pump_current_a"] = expected_current
                frequency_summary_path.write_text(
                    json.dumps(frequency_summary, indent=2), encoding="utf-8"
                )
            else:
                frequency_summary = existing
            rows.append(
                {
                    "pump_current_a": expected_current,
                    "status": frequency_summary.get("status", "CHECK"),
                    "n_signal_freq": frequency_summary.get("n_signal_freq"),
                    "run_dir": str(run_dir),
                }
            )
            continue

        summary_path = run_dir / "compression_summary.json"
        summary = _resumable_summary(
            summary_path,
            expected_pump_current_a=expected_current,
        )
        if summary is None:
            summary = _write_one_result(run_args)
            summary.update(
                {
                    "signal_frequency_range_ghz": [
                        run_args.signal_ghz,
                        run_args.signal_ghz,
                    ],
                    "n_signal_freq": 1,
                    "signal_workers": 1,
                    "factor_backend": run_args.factor_backend,
                }
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        rows.append(
            {
                "pump_current_a": expected_current,
                "signal_ghz": summary.get("signal_ghz"),
                "status": summary.get("status", "CHECK"),
                "small_signal_gain_db": summary.get("small_signal_gain_db"),
                "p1db_input_dbm": summary.get("p1db_input_dbm"),
                "p1db_pump_depletion_all_port_db": summary.get(
                    "p1db_pump_depletion_all_port_db"
                ),
                "run_dir": str(run_dir),
            }
        )

    columns = list(rows[0]) if rows else ["pump_current_a", "status"]
    with (args.output_dir / "p1db_vs_pump_current.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": (
            "VALID_SOLVED"
            if rows and all(row["status"] == "VALID_SOLVED" for row in rows)
            else "CHECK"
        ),
        "pump_current_a": [float(value) for value in currents],
        "pump_current_jc_scale": float(args.pump_current_jc_scale),
        "n_pump_currents": len(rows),
        "signal_ghz": args.signal_ghz,
        "signal_frequency_range_ghz": (
            [float(args.signal_ghz_min), float(args.signal_ghz_max)]
            if args.n_signal_freq > 1
            else [float(args.signal_ghz), float(args.signal_ghz)]
        ),
        "results": rows,
    }
    (args.output_dir / "pump_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _frequency_worker_limit(args: argparse.Namespace, task_count: int) -> int:
    """Cap concurrent sparse-LU workers to measured per-worker peak memory.

    Each worker holds an independent ``real_coupled_fast`` preconditioner: the
    coupled matrix, its scatter map, and the PARDISO factors. That footprint
    scales as ``n_tones^2 * n_retained``, so a fixed per-worker constant either
    wastes cores on small bases or overcommits and thrashes on large ones. The
    cap is taken against BOTH the declared budget and the memory actually free,
    since the budget alone cannot know what else is running.
    """
    requested = max(1, int(args.signal_workers))
    try:
        footprint = _estimate_worker_footprint(args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        logger.warning(
            "worker_footprint_estimate_failed error=%s falling_back_to=1", exc
        )
        return 1

    peak_gb = footprint.peak_gb
    budget_limited = max(1, int(float(args.resource_budget_gb) // peak_gb))
    free_gb = available_memory_gb()
    if free_gb is None:
        free_limited = requested
    else:
        # Free memory is sampled once, at launch, but a long sweep competes with
        # whatever else the machine does for hours afterwards. Hold back a
        # reserve proportional to one worker so that drift does not turn a
        # just-fitting worker count into swap.
        headroom_gb = _memory_headroom_gb(peak_gb)
        if peak_gb > free_gb - headroom_gb and not args.allow_memory_overcommit:
            needed_gb = peak_gb + headroom_gb - free_gb
            raise ResourceLimitExceeded(
                f"a single worker needs ~{peak_gb:.2f} GB (plus {headroom_gb:.2f} GB "
                f"reserved headroom) but only {free_gb:.2f} GB is free. Free about "
                f"{needed_gb:.1f} GB more, reduce --multitone-sidebands "
                f"(memory scales as the square of the tone count), or pass "
                f"--allow-memory-overcommit to run anyway and risk swapping."
            )
        free_limited = max(1, int((free_gb - headroom_gb) // peak_gb))
    workers = max(1, min(requested, task_count, budget_limited, free_limited))
    logger.info(
        "worker_limit requested=%d task_count=%d peak_gb_per_worker=%.2f "
        "steady_gb=%.2f budget_gb=%.1f budget_limited=%d free_gb=%s "
        "free_limited=%d selected=%d",
        requested, task_count, peak_gb, footprint.steady_gb,
        float(args.resource_budget_gb), budget_limited,
        "unknown" if free_gb is None else f"{free_gb:.2f}",
        free_limited, workers,
    )
    if workers < requested:
        print(
            f"[run_compression] limiting --signal-workers {requested} -> {workers}: "
            f"each worker peaks at ~{peak_gb:.2f} GB "
            f"(budget {float(args.resource_budget_gb):.1f} GB, "
            f"free {'unknown' if free_gb is None else f'{free_gb:.2f} GB'})",
            flush=True,
        )
    return workers


def _select_factor_backend(args: argparse.Namespace, task_count: int) -> str:
    """Select the faster backend that fits the requested sweep budget."""
    requested = getattr(args, "factor_backend", "pardiso")
    if requested != "auto":
        return requested
    if int(getattr(args, "n_signal_freq", 1)) <= 1:
        return "pardiso"

    candidates: dict[str, int] = {}
    for backend in ("pardiso", "banded"):
        candidate_values = vars(args).copy()
        candidate_values["factor_backend"] = backend
        candidate = argparse.Namespace(**candidate_values)
        footprint = _estimate_worker_footprint(candidate)
        headroom = _memory_headroom_gb(footprint.peak_gb)
        budget_workers = int(
            max(0.0, float(args.resource_budget_gb) - headroom)
            // footprint.peak_gb
        )
        free_gb = available_memory_gb()
        free_workers = (
            budget_workers
            if free_gb is None
            else int(max(0.0, free_gb - headroom) // footprint.peak_gb)
        )
        candidates[backend] = max(
            1,
            min(int(args.signal_workers), task_count, budget_workers, free_workers),
        )
    return "banded" if candidates["banded"] > candidates["pardiso"] else "pardiso"


def _estimate_worker_footprint(
    args: argparse.Namespace,
) -> FastCoupledFootprint:
    """Estimate one worker's peak memory from the real circuit and basis."""
    circuit, metadata, _ = _load_source(args)
    omega_p = 2.0 * math.pi * args.pump_freq_ghz * 1e9
    if args.pump_solution_dir is not None:
        _, pump_basis = load_pump_basis_from_solution(
            args.pump_solution_dir, fallback_omega_p=omega_p
        )
    else:
        pump_basis = resolve_pump_basis(
            policy=args.pump_mode_policy,
            omega_p=omega_p,
            harmonics=args.pump_harmonics,
            mode_count=args.pump_mode_count,
            explicit_modes=args.pump_modes,
            design_meta=metadata,
        )
    # The tone count is frequency-independent; any in-band signal frequency
    # gives the same basis size, so the sweep midpoint is representative.
    signal_ghz = args.signal_ghz
    if signal_ghz is None:
        signal_ghz = 0.5 * (float(args.signal_ghz_min) + float(args.signal_ghz_max))
    delta = omega_p - 2.0 * math.pi * float(signal_ghz) * 1e9
    basis = _build_multitone_basis(args, pump_basis.modes, omega_p, delta)
    # The Schur backend retains the nonlinear nodes and ports; the full backend
    # keeps every node. Use the full node count, the conservative bound.
    return fast_coupled_footprint(
        basis.n_tones,
        circuit.node_count,
        factor_backend=getattr(args, "factor_backend", "pardiso"),
    )


def _run_frequency_sweep(args: argparse.Namespace) -> dict[str, object]:
    if args.signal_ghz_min is None or args.signal_ghz_max is None:
        raise ValueError(
            "--n-signal-freq > 1 requires --signal-ghz-min and --signal-ghz-max"
        )
    frequencies = np.linspace(
        args.signal_ghz_min,
        args.signal_ghz_max,
        args.n_signal_freq,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    summaries_by_index: dict[int, dict[str, object]] = {}
    values = vars(args).copy()
    for index, frequency in enumerate(frequencies):
        subdir = args.output_dir / f"frequency_{index:03d}_{frequency:.6f}ghz"
        summary_path = subdir / "compression_summary.json"
        if summary_path.exists():
            summaries_by_index[index] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            payloads.append((index, (values, float(frequency), str(subdir))))
    workers = _frequency_worker_limit(args, max(len(payloads), 1))
    if workers == 1:
        completed = [(index, _frequency_worker(payload)) for index, payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_frequency_worker, [payload for _, payload in payloads])
            completed = [(index, result) for (index, _), result in zip(payloads, results)]
    summaries_by_index.update(completed)
    summaries = [summaries_by_index[index] for index in range(len(frequencies))]
    rows = [
        {
            "signal_ghz": summary["signal_ghz"],
            "status": summary["status"],
            "small_signal_gain_db": summary["small_signal_gain_db"],
            "small_signal_gain_vs_off_db": summary["small_signal_gain_vs_off_db"],
            "p1db_input_dbm": summary["p1db_input_dbm"],
            "p1db_output_dbm": summary["p1db_output_dbm"],
            "p1db_pump_depletion_db": summary["p1db_pump_depletion_db"],
            "p1db_pump_depletion_all_port_db": summary.get(
                "p1db_pump_depletion_all_port_db"
            ),
            "p1db_external_power_balance_rel_err": summary.get(
                "p1db_external_power_balance_rel_err"
            ),
        }
        for summary in summaries
    ]
    with (args.output_dir / "p1db_vs_frequency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt

        figure, axis_p1db = plt.subplots(figsize=(7.0, 4.2))
        x = [float(row["signal_ghz"]) for row in rows]
        p1db = [
            float(row["p1db_input_dbm"])
            if row["p1db_input_dbm"] is not None
            else float("nan")
            for row in rows
        ]
        gain = [float(row["small_signal_gain_vs_off_db"]) for row in rows]
        axis_p1db.plot(x, p1db, "o-", color="tab:blue", label="P1dB input")
        axis_p1db.set_xlabel("Signal frequency (GHz)")
        axis_p1db.set_ylabel("P1dB input (dBm)", color="tab:blue")
        axis_gain = axis_p1db.twinx()
        axis_gain.plot(x, gain, "s--", color="tab:red", label="Small-signal gain")
        axis_gain.set_ylabel("Gain vs pump-off (dB)", color="tab:red")
        axis_p1db.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.output_dir / "p1db_vs_frequency.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    summary = {
        "status": (
            "VALID_SOLVED"
            if all(row["status"] == "VALID_SOLVED" for row in rows)
            else "CHECK"
        ),
        "stability_status": "NOT_CHECKED",
        "n_signal_freq": len(rows),
        "signal_workers": workers,
        "signal_workers_requested": int(args.signal_workers),
        "resource_budget_gb": float(args.resource_budget_gb),
        "factor_backend": args.factor_backend,
        "frequencies_ghz": [float(value) for value in frequencies],
    }
    (args.output_dir / "frequency_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pump_current_list and args.pump_current_a is not None:
        parser.error("--pump-current-a and --pump-current-list are mutually exclusive")
    if args.n_signal_freq == 1 and args.signal_ghz is None:
        parser.error("--signal-ghz is required for a single-frequency run")
    args.factor_backend = _select_factor_backend(args, max(args.n_signal_freq, 1))
    # The preconditioner is constructed deep inside the problem, and frequency
    # workers are separate processes, so the choice travels as an environment
    # variable the way the other factor-backend switches already do.
    os.environ["TWPA_BANDED_PRECOND"] = (
        "1" if args.factor_backend == "banded" else "0"
    )
    if args.pump_current_list:
        summary = _run_pump_current_sweep(args)
        if args.summary_json:
            args.summary_json.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        return 0
    if args.n_signal_freq > 1:
        summary = _run_frequency_sweep(args)
        if args.summary_json:
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0
    summary = _write_one_result(args)
    summary.update({
        "signal_frequency_range_ghz": [args.signal_ghz, args.signal_ghz],
        "n_signal_freq": 1,
        "signal_workers": 1,
        "factor_backend": args.factor_backend,
    })
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "compression_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
