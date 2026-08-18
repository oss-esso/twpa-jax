"""Continue a complex Hill root on a converged pump branch.

The pump source is re-solved at every accepted drive value.  The Hill operator
is never assembled from a rescaled nonlinear pump waveform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from twpa_solver.core import load_circuit  # noqa: E402
from twpa_solver.signal import (  # noqa: E402
    PumpSolution,
    assemble_khat_conversion_base,
    build_khat,
    compute_gamma_hat,
    load_pump,
    sideband_list,
)
from twpa_solver.signal.branch_tracking import (  # noqa: E402
    FloquetBranchPoint,
    track_floquet_point,
)
from twpa_solver.pump.backends.schur_partition import restrict  # noqa: E402


CSV_FIELDS = [
    "drive_amplitude",
    "requested_drive_dbm",
    "requested_source_current_a",
    "achieved_on_chip_current_a",
    "pump_converged",
    "pump_iterations",
    "pump_coeff_rel",
    "pump_time_rel",
    "pump_failure_reason",
    "omega_real_ghz",
    "omega_imag_ghz",
    "growth_rate_per_s",
    "multiplier_magnitude",
    "multiplier_phase_rad",
    "floquet_kind",
    "mode_overlap",
    "discontinuity",
    "stability_verdict",
    "converged",
    "iterations",
    "residual",
]


@dataclass
class PumpStep:
    """One source-drive solve and its converged pump state."""

    drive_dbm: float
    source_current_a: float
    achieved_current_a: float
    converged: bool
    iterations: int
    coeff_rel: float
    time_rel: float | None
    failure_reason: str
    pump: PumpSolution | None
    full_state: np.ndarray | None


def parse_float_list(value: str, *, name: str) -> list[float]:
    """Parse a strictly increasing finite list of real values."""
    try:
        values = [float(token.strip()) for token in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the continuation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", required=True)
    parser.add_argument("--pump-dir", required=True)
    parser.add_argument("--drive-dbms", default=None)
    parser.add_argument("--drive-amplitudes", default=None)
    parser.add_argument("--sidebands", type=int, required=True)
    parser.add_argument("--initial-signal-ghz", type=float, required=True)
    parser.add_argument("--loss-model", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--pump-port", type=int, default=None)
    parser.add_argument("--gamma-nt", type=int, default=4096)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--overlap-threshold", type=float, default=0.8)
    parser.add_argument("--discontinuity-threshold", type=float, default=0.25)
    parser.add_argument("--min-step", type=float, default=1.0e-4)
    parser.add_argument("--pump-initial-step", type=float, default=0.25)
    parser.add_argument("--pump-min-step", type=float, default=1.0e-3)
    parser.add_argument("--pump-solve-deadline-s", type=float, default=0.0)
    parser.add_argument("--pump-current-jc-scale", type=float, default=1.0)
    parser.add_argument("--pump-max-newton", type=int, default=16)
    parser.add_argument("--pump-stall-patience", type=int, default=4)
    parser.add_argument("--pump-stall-ratio", type=float, default=0.8)
    parser.add_argument("--pump-min-alpha", type=float, default=1.0 / 1024.0)
    parser.add_argument("--drive-min-step-dbm", type=float, default=1.0e-3)
    return parser.parse_args(argv)


def _dc_flux(circuit: Any, pump_dir: Path) -> np.ndarray | None:
    """Load the DC branch flux used by the saved pump, if present."""
    report_path = pump_dir / "pump_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    value = metadata.get("dc_branch_flux", metadata.get("dc_branch_flux_wb"))
    if value is None:
        return None
    flux = np.asarray(value, dtype=float).reshape(-1)
    if flux.size == 1:
        flux = np.full(circuit.branch_count, float(flux[0]))
    if flux.size != circuit.branch_count:
        raise ValueError("pump report DC flux does not match circuit branches")
    return flux


def _memory_guard(circuit: Any, sidebands: int) -> None:
    """Print and enforce a conservative conversion-matrix memory estimate."""
    dimension = circuit.node_count * (2 * sidebands + 1)
    estimate = max(64.0 * dimension * (2 * sidebands + 1), 64.0e6)
    try:
        import psutil

        available = float(psutil.virtual_memory().available)
    except ImportError:
        available = float("inf")
    print(
        "memory_guard dimension=%d estimated_process_bytes=%.0f "
        "free_bytes=%.0f" % (dimension, estimate, available)
    )
    if estimate > available:
        raise MemoryError("estimated process footprint exceeds available memory")


def _build_hill_operator(
    circuit: Any,
    pump: PumpSolution,
    ms: list[int],
    gamma_nt: int,
    dc_flux: np.ndarray | None,
) -> tuple[dict[int, Any], Any]:
    """Build the Hill operator from a converged pump solution."""
    max_ell = max(abs(left - right) for left in ms for right in ms)
    gamma_hat = compute_gamma_hat(
        circuit=circuit,
        pump=pump,
        max_ell=max_ell,
        gamma_nt=gamma_nt,
        dc_branch_flux=dc_flux,
    )
    khat = build_khat(circuit.Bphi, gamma_hat, drop_tol=0.0)
    return khat, assemble_khat_conversion_base(circuit, khat, ms)


class PumpContinuation:
    """Production in-process HB continuation over source power."""

    def __init__(
        self,
        args: argparse.Namespace,
        circuit: Any,
        saved_pump: PumpSolution,
    ) -> None:
        import run_gain_map

        self.run_gain_map = run_gain_map
        metadata = saved_pump.metadata
        self.frequency_ghz = float(saved_pump.pump_freq_ghz)
        self.scale = float(args.pump_current_jc_scale)
        self.engine_args = run_gain_map.parse_args(
            [
                "--circuit-dir",
                str(args.circuit_dir),
                "--outdir",
                str(ROOT / "outputs" / "track_critical_root_pump"),
                "--n-power",
                "1",
                "--n-frequency",
                "1",
                "--pump-power-min-dbm",
                str(metadata.get("pump_power_dbm_requested", -30.0)),
                "--pump-power-max-dbm",
                str(metadata.get("pump_power_dbm_requested", -30.0)),
                "--pump-freq-min-ghz",
                str(self.frequency_ghz),
                "--pump-freq-max-ghz",
                str(self.frequency_ghz),
                "--pump-port",
                str(args.pump_port or metadata.get("pump_port", 1)),
                "--source-port",
                str(metadata.get("source_port", 1)),
                "--out-port",
                str(metadata.get("out_port", 2)),
                "--pump-mode-policy",
                str(metadata.get("pump_mode_policy", "positive_odd_jc")),
                "--pump-mode-count",
                str(len(saved_pump.modes)),
                "--nt",
                str(metadata.get("nt", saved_pump.nt_original or 40)),
                "--frequency-chunk-size",
                "0",
                "--executor",
                "inprocess",
                "--no-signal-spectrum",
                "--force-single-tone",
            ]
        )
        self.engine_args.loss_model = args.loss_model
        self.engine_args.pump_current_jc_scale = self.scale
        self.engine_args.inproc_pump_backend = "schur_cpu_mt"
        self.engine_args.inproc_preconditioner = "real_coupled_fast"
        self.engine_args.inproc_schur_cache_size = 1
        self.engine_args.pump_solution_dtype = "float64"
        self.engine_args.hybrid_compact_storage = True
        self.engine_args.inproc_continuation = "adaptive_secant"
        self.engine_args.adaptive_initial_step = float(args.pump_initial_step)
        self.engine_args.adaptive_min_step = float(args.pump_min_step)
        self.engine_args.inproc_fallback_fixed_steps = 0
        self.engine_args.inproc_solve_deadline_s = float(args.pump_solve_deadline_s)
        self.engine_args.inproc_continuation_deadline_s = float(
            args.pump_solve_deadline_s
        )
        self.engine_args.inproc_fail_fast = True
        self.engine_args.inproc_max_newton = int(args.pump_max_newton)
        self.engine_args.inproc_stall_patience = int(args.pump_stall_patience)
        self.engine_args.inproc_stall_ratio = float(args.pump_stall_ratio)
        self.engine_args.inproc_min_alpha = float(args.pump_min_alpha)
        self.engine = run_gain_map.InProcessEngine(self.engine_args)
        self.circuit = circuit
        self._last_source_current_a: float | None = None

    def source_current_for_drive(self, drive_dbm: float) -> float:
        """Convert requested source power to the production port current."""
        run_gain_map = self.run_gain_map
        attenuation = run_gain_map.attenuation_db_for(
            self.frequency_ghz, self.engine_args
        )
        return float(
            run_gain_map.dbm_to_peak_current_a(
                drive_dbm,
                attenuation_db=attenuation,
                z0_ohm=self.engine_args.z0_ohm,
                convention=self.engine_args.power_convention,
            )
        )

    def solve(self, drive_dbm: float, warm_full: np.ndarray | None) -> PumpStep:
        """Solve one source drive and return no state when it fails."""
        run_gain_map = self.run_gain_map
        source_current = self.source_current_for_drive(drive_dbm)
        attenuation = run_gain_map.attenuation_db_for(
            self.frequency_ghz, self.engine_args
        )
        injected = source_current * self.scale
        full_problem, basis, omega = self.engine._build_problem(
            self.frequency_ghz, injected
        )
        solve_problem = self.engine._make_solve_problem(
            full_problem, self.frequency_ghz
        )
        x_init: np.ndarray | None = None
        if warm_full is not None:
            x_init = (
                restrict(warm_full, solve_problem.part)
                if solve_problem is not full_problem
                else np.asarray(warm_full)
            )
        solver = run_gain_map.exp08.HarmonicNewtonKrylovSolver(
            self.engine._settings()
        )
        fallback_used = False
        trace_failure_reason = ""
        if self._last_source_current_a is None and x_init is not None:
            X, report = solver.solve_one(solve_problem, x_init, 1.0)
            reports = [report]
        else:
            lambda_start = (
                0.0
                if self._last_source_current_a is None
                else self._last_source_current_a / source_current
            )
            X, reports, trace = solver.solve_adaptive_continuation(
                solve_problem,
                x_init,
                initial_step=float(self.engine_args.adaptive_initial_step),
                min_step=float(self.engine_args.adaptive_min_step),
                growth=1.5,
                shrink=0.5,
                fallback_fixed_steps=0,
                max_wall_s=float(self.engine_args.inproc_continuation_deadline_s),
                lambda_start=lambda_start,
            )
            fallback_used = trace.fallback_used
            trace_failure_reason = trace.failure_reason
        report = reports[-1] if reports else None
        converged = bool(
            report is not None
            and report.converged
            and abs(report.source_scale - 1.0) < 1.0e-12
            and not fallback_used
        )
        full_state = (
            solve_problem.reconstruct_full(X)
            if solve_problem is not full_problem
            else np.asarray(X)
        )
        norms = full_problem.norms(full_state, 1.0, True)
        reason = "" if converged else str(
            trace_failure_reason or (report.failure_reason if report else "no report")
        )
        iterations = int(report.newton_iterations) if report is not None else 0
        pump = None
        if converged:
            metadata = dict(getattr(self.engine, "ipm08", self.circuit).summary)
            metadata.update(
                {
                    "pump_freq_ghz": self.frequency_ghz,
                    "omega_p": omega,
                    "pump_current_a": injected,
                    "pump_power_dbm_requested": drive_dbm,
                    "pump_power_convention": self.engine_args.power_convention,
                    "attenuation_db": attenuation,
                    "pump_backend": "schur_cpu_mt",
                    "pump_continuation": "adaptive_natural_source_drive",
                }
            )
            pump = PumpSolution(
                X=full_state,
                omega_p=omega,
                pump_freq_ghz=self.frequency_ghz,
                harmonics=full_state.shape[0],
                nt_original=int(getattr(full_problem.grid, "nt", 0)),
                metadata=metadata,
                modes=list(basis.modes),
                basis=basis,
            )
            self._last_source_current_a = source_current
        return PumpStep(
            drive_dbm=float(drive_dbm),
            source_current_a=float(source_current),
            achieved_current_a=float(source_current) if converged else float("nan"),
            converged=converged,
            iterations=iterations,
            coeff_rel=float(norms["coeff_rel"]),
            time_rel=(
                None if norms["time_rel"] is None else float(norms["time_rel"])
            ),
            failure_reason=reason,
            pump=pump,
            full_state=full_state if converged else None,
        )


def _write_point(
    writer: csv.DictWriter,
    point: FloquetBranchPoint | None,
    drive_dbm: float,
    pump_step: PumpStep,
    base_current: float,
) -> None:
    """Write one pump/tracker row and flush is performed by the caller."""
    row: dict[str, Any] = {
        "drive_amplitude": pump_step.source_current_a / base_current,
        "requested_drive_dbm": drive_dbm,
        "requested_source_current_a": pump_step.source_current_a,
        "achieved_on_chip_current_a": pump_step.achieved_current_a,
        "pump_converged": pump_step.converged,
        "pump_iterations": pump_step.iterations,
        "pump_coeff_rel": pump_step.coeff_rel,
        "pump_time_rel": pump_step.time_rel,
        "pump_failure_reason": pump_step.failure_reason,
    }
    if point is not None:
        signal = complex(point.resonance.signal_ghz)
        classification = point.classification
        row.update(
            {
                "omega_real_ghz": signal.real,
                "omega_imag_ghz": signal.imag,
                "growth_rate_per_s": point.resonance.growth_rate_per_s,
                "multiplier_magnitude": classification.magnitude,
                "multiplier_phase_rad": classification.phase_rad,
                "floquet_kind": classification.kind,
                "mode_overlap": point.mode_overlap,
                "discontinuity": point.discontinuity,
                "stability_verdict": point.stability_verdict,
                "converged": point.resonance.converged,
                "iterations": point.resonance.iterations,
                "residual": point.resonance.residual,
            }
        )
    writer.writerow(row)


def _crossing(
    points: list[tuple[float, FloquetBranchPoint]],
    pump_freq_ghz: float,
) -> dict[str, float] | None:
    """Linearly interpolate the first growth-rate zero crossing."""
    for (left_amp, left), (right_amp, right) in zip(points, points[1:]):
        left_growth = left.resonance.growth_rate_per_s
        right_growth = right.resonance.growth_rate_per_s
        if left_growth == 0.0:
            fraction = 0.0
        elif left_growth * right_growth <= 0.0:
            fraction = -left_growth / (right_growth - left_growth)
        else:
            continue
        amplitude = left_amp + fraction * (right_amp - left_amp)
        left_signal = complex(left.resonance.signal_ghz)
        right_signal = complex(right.resonance.signal_ghz)
        generator = left_signal.imag + fraction * (right_signal.imag - left_signal.imag)
        return {
            "drive_amplitude": amplitude,
            "generator_frequency_ghz": generator,
            "omega_a_over_omega_p": generator / pump_freq_ghz,
        }
    return None


def _drive_dbms(args: argparse.Namespace, saved_pump: PumpSolution) -> list[float]:
    """Resolve source-drive dBm values, retaining the legacy amplitude input."""
    base_dbm = float(saved_pump.metadata.get("pump_power_dbm_requested", 0.0))
    if args.drive_dbms is not None and args.drive_amplitudes is not None:
        raise ValueError("provide only one of --drive-dbms and --drive-amplitudes")
    if args.drive_dbms is not None:
        return parse_float_list(args.drive_dbms, name="drive dBm")
    if args.drive_amplitudes is None:
        raise ValueError("one of --drive-dbms and --drive-amplitudes is required")
    amplitudes = parse_float_list(args.drive_amplitudes, name="drive amplitudes")
    if any(value <= 0.0 for value in amplitudes):
        raise ValueError("drive amplitudes must be positive")
    return [base_dbm + 20.0 * math.log10(value) for value in amplitudes]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run source-drive pump continuation and one deliberately seeded root."""
    if args.sidebands < 0:
        raise ValueError("sidebands must be non-negative")
    if not 0.0 < args.overlap_threshold <= 1.0:
        raise ValueError("overlap threshold must lie in (0, 1]")
    if args.pump_min_step > 1.0e-3:
        raise ValueError("--pump-min-step must be no greater than 1e-3")
    circuit = load_circuit(args.circuit_dir)
    pump_dir = Path(args.pump_dir)
    saved_pump = load_pump(pump_dir, fallback_pump_freq_ghz=1.0)
    drives = _drive_dbms(args, saved_pump)
    ms = sideband_list(args.sidebands)
    _memory_guard(circuit, args.sidebands)
    dc_flux = _dc_flux(circuit, pump_dir)
    pump_continuation = PumpContinuation(args, circuit, saved_pump)
    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_current = pump_continuation.source_current_for_drive(drives[0])
    saved_drive_dbm = float(
        saved_pump.metadata.get("pump_power_dbm_requested", drives[0])
    )
    points: list[tuple[float, FloquetBranchPoint]] = []
    seed = complex(args.initial_signal_ghz)
    previous_mode: np.ndarray | None = None
    previous_multiplier: complex | None = None
    warm_state: np.ndarray | None = (
        saved_pump.X if drives[0] >= saved_drive_dbm - 1.0e-12 else None
    )
    current_drive_dbm: float | None = None
    drive_step_dbm = 0.0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
        for target_drive_dbm in drives:
            if current_drive_dbm is None:
                candidate_drives = [target_drive_dbm]
            else:
                drive_step_dbm = max(
                    drive_step_dbm,
                    target_drive_dbm - current_drive_dbm,
                )
                candidate_drives = []
                while current_drive_dbm < target_drive_dbm - 1.0e-12:
                    candidate_drives.append(
                        min(target_drive_dbm, current_drive_dbm + drive_step_dbm)
                    )
                    break
            for drive_dbm in candidate_drives:
                previous_drive_dbm = current_drive_dbm
                previous_warm_state = warm_state
                previous_seed = seed
                previous_mode_state = previous_mode
                previous_multiplier_state = previous_multiplier
                pump_step = pump_continuation.solve(drive_dbm, warm_state)
                if not pump_step.converged or pump_step.pump is None:
                    _write_point(writer, None, drive_dbm, pump_step, base_current)
                    handle.flush()
                    print(
                        "pump_step drive_dbm=%.6f current_a=%.12e converged=False "
                        "coeff_rel=%.6e time_rel=%s iterations=%d reason=%s"
                        % (
                            drive_dbm,
                            pump_step.source_current_a,
                            pump_step.coeff_rel,
                            pump_step.time_rel,
                            pump_step.iterations,
                            pump_step.failure_reason,
                        )
                    )
                    return {
                        "pump_frequency_ghz": saved_pump.pump_freq_ghz,
                        "loss_model": args.loss_model,
                        "pump_failed": True,
                        "failed_drive_dbm": drive_dbm,
                        "accepted_steps": len(points),
                        "csv": str(output_path),
                    }
                khat, khat_base = _build_hill_operator(
                    circuit, pump_step.pump, ms, args.gamma_nt, dc_flux
                )
                current = pump_step.source_current_a / base_current
                point = track_floquet_point(
                    circuit=circuit,
                    khat=khat,
                    khat_base=khat_base,
                    parameter=current,
                    omega_p=pump_step.pump.omega_p,
                    ms=ms,
                    seed_signal_ghz=seed,
                    seed_mode_vector=previous_mode,
                    previous_multiplier=previous_multiplier,
                    loss_model=args.loss_model,
                    max_iters=args.max_iters,
                    tol=args.tol,
                    discontinuity_threshold=args.discontinuity_threshold,
                    mode_overlap_threshold=args.overlap_threshold,
                )
                low_overlap = bool(
                    point.mode_overlap is not None
                    and point.mode_overlap < args.overlap_threshold
                )
                if (
                    (low_overlap or not point.resonance.converged)
                    and previous_drive_dbm is not None
                ):
                    pump_continuation._last_source_current_a = (
                        pump_continuation.source_current_for_drive(previous_drive_dbm)
                    )
                    current_drive_dbm = previous_drive_dbm
                    warm_state = previous_warm_state
                    seed = previous_seed
                    previous_mode = previous_mode_state
                    previous_multiplier = previous_multiplier_state
                    drive_step_dbm *= 0.5
                    if drive_step_dbm < args.drive_min_step_dbm:
                        raise RuntimeError(
                            "mode overlap remained below threshold at "
                            "--drive-min-step-dbm"
                        )
                    continue
                _write_point(writer, point, drive_dbm, pump_step, base_current)
                handle.flush()
                print(
                    "pump_step drive_dbm=%.6f current_a=%.12e converged=True "
                    "coeff_rel=%.6e time_rel=%s iterations=%d"
                    % (
                        drive_dbm,
                        pump_step.source_current_a,
                        pump_step.coeff_rel,
                        pump_step.time_rel,
                        pump_step.iterations,
                    )
                )
                points.append((current, point))
                current_drive_dbm = drive_dbm
                seed = complex(point.resonance.signal_ghz)
                previous_mode = point.resonance.mode_vector
                previous_multiplier = point.classification.multiplier
                warm_state = pump_step.full_state
                if point.mode_overlap is None or not point.discontinuity:
                    drive_step_dbm = max(
                        args.drive_min_step_dbm,
                        drive_step_dbm * 1.5,
                    )
    crossing = _crossing(points, saved_pump.pump_freq_ghz)
    return {
        "pump_frequency_ghz": saved_pump.pump_freq_ghz,
        "loss_model": args.loss_model,
        "pump_port": args.pump_port,
        "pump_failed": False,
        "crossing": crossing,
        "accepted_steps": len(points),
        "csv": str(output_path),
    }


def main(argv: list[str] | None = None) -> None:
    """Run the command-line driver."""
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
