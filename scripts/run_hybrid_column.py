"""Run one bounded hybrid HB/TD production column.

This is a validation driver, not the overnight map.  It uses the production
``InProcessEngine`` for every HB attempt and delegates transient integration to
the validated H1 implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import h1_transient_branch_transfer as h1  # noqa: E402
from scripts import run_gain_map  # noqa: E402
from scripts.g0_column_recovery import (  # noqa: E402
    TIER2_DEADLINE_S,
    TIER3_DEADLINE_S,
    _build_args,
)
from scripts.g1_column_recovery import (  # noqa: E402
    _try_tier2,
    _try_tier3,
    _try_tier4,
)
from twpa_solver.pump import hb as exp08  # noqa: E402
from twpa_solver.hybrid_column import (  # noqa: E402
    ColumnBudget,
    ColumnController,
    HBResult,
    SolverRoute,
    TDClass,
    TDResult,
)


def prepare_output_dir(requested: Path | None, frequency_ghz: float) -> Path:
    """Create a writable run directory without requiring elevated privileges."""
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.append(ROOT / "outputs" / f"hybrid_{frequency_ghz:g}ghz")
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError as exc:
            last_error = exc
    raise OSError("no writable hybrid-column output directory found") from last_error


def _pump_success(row: dict[str, Any]) -> bool:
    return row.get("pump_status") == "VALID_CONVERGED"


_TIMING_FIELDS = (
    "pump_runtime_s", "pump_wall_runtime_s", "pump_setup_runtime_s",
    "pump_schur_setup_runtime_s", "pump_solve_wall_runtime_s",
    "pump_write_runtime_s", "pump_factor_runtime_s",
    "pump_preconditioner_assembly_runtime_s",
    "pump_preconditioner_numeric_factor_runtime_s",
    "gain_total_runtime_s", "gain_wall_runtime_s",
    "gain_gamma_hat_runtime_s", "gain_khat_build_runtime_s",
    "gain_khat_off_runtime_s", "gain_matrix_assemble_runtime_s",
    "gain_factor_solve_runtime_s", "gain_baseline_off_runtime_s",
    "gain_baseline_pumpdiag_runtime_s", "elapsed_s",
    "pump_continuation_method", "pump_continuation_steps",
    "pump_continuation_reached_target", "pump_continuation_runtime_s",
)


class ProductionPeriodicBackend:
    """Adapter around the same in-process production engine used by G1."""

    def __init__(self, args: argparse.Namespace, points: list[Any], outdir: Path) -> None:
        self.args = args
        self.points = {point.index: point for point in points}
        self.engine = run_gain_map.InProcessEngine(args)
        self.pass_dir = outdir / "pass"
        self.scale = args.pump_current_jc_scale
        self.residual_threshold = float(getattr(args, "hybrid_hb_residual_threshold", 1e-8))
        self.compact_storage = bool(getattr(args, "hybrid_compact_storage", False))
        self.retained_checkpoint: Path | None = None
        self.map_rows: dict[int, dict[str, Any]] = {}

    def _attach_homotopy_metadata(
        self, point: Any, row: dict[str, Any], info: dict[str, Any]
    ) -> None:
        """Persist TD-to-HB recovery telemetry beside the validated pump state."""
        row["td_residual_homotopy"] = info
        pump_dir = self.pass_dir / "points" / run_gain_map.point_name(
            point.index, point.power_dbm, point.pump_freq_ghz
        ) / "pump"
        report_path = pump_dir / "pump_report.json"
        if not report_path.exists():
            return
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload.setdefault("metadata", {})["td_residual_homotopy"] = info
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def solve_direct(self, target: Any, previous: HBResult | None) -> HBResult:
        point = self.points[target.index]
        started = time.perf_counter()
        row, state = self.engine.solve_point(
            point, self.pass_dir, mode="warm" if previous else "seed",
            warm_X=previous.state if previous else None,
        )
        return self._result(row, state, SolverRoute.DIRECT_HB, started, point)

    def recover(self, target: Any, previous: HBResult) -> HBResult:
        point = self.points[target.index]
        target_current = point.current_a * self.scale
        previous_current = previous.metadata["current_a"]
        started = time.perf_counter()
        attempts = (
            (SolverRoute.POWER_SUBSTEP, _try_tier2),
            (SolverRoute.PALC, _try_tier3),
            (SolverRoute.FREQUENCY_SUBSTEP, _try_tier4),
        )
        for route, attempt in attempts:
            if route == SolverRoute.POWER_SUBSTEP:
                result = attempt(
                    self.engine, point.pump_freq_ghz, previous.state,
                    previous_current, target_current, point, self.pass_dir,
                )
            elif route == SolverRoute.PALC:
                result = attempt(
                    self.engine, point.pump_freq_ghz, previous.state,
                    previous_current, target_current, point, self.pass_dir,
                )
            else:
                result = attempt(
                    self.engine, point.pump_freq_ghz, previous.state,
                    target_current, point, self.pass_dir,
                )
            row, state, telemetry = result
            if row is not None and _pump_success(row):
                return self._result(row, state, route, started, point, telemetry)
        return HBResult(False, reason="production recovery exhausted", runtime_s=time.perf_counter() - started)

    def restart_from_td(self, target: Any, seed: str) -> HBResult:
        point = self.points[target.index]
        data = np.load(seed)
        state = np.asarray(data["X_real"]) + 1j * np.asarray(data["X_imag"])
        td_state = np.array(state, dtype=np.complex128, copy=True)
        fallback_state = td_state
        started = time.perf_counter()

        # A Fourier projection is a physical TD orbit seed, not an HB root.
        # Follow a bounded residual homotopy from that seed to the unchanged
        # fixed-drive production residual before asking the normal map path to
        # write a checkpoint or evaluate gain.
        homotopy_info: dict[str, Any] = {"attempted": False}
        try:
            full_problem, basis, _omega, _injected = self.engine.build_problem_for(
                point
            )
            expected_shape = full_problem.zeros().shape
            modes = np.asarray(
                data.get("modes", data.get("pump_modes", basis.modes)),
                dtype=np.int64,
            ).reshape(-1)
            if state.shape != expected_shape or modes.tolist() != list(basis.modes):
                raise ValueError(
                    "TD projection basis does not match target production HB basis"
                )
            solve_problem = self.engine._make_solve_problem(
                full_problem, point.pump_freq_ghz
            )
            solve_state = state
            if hasattr(solve_problem, "part"):
                if state.shape != full_problem.zeros().shape:
                    raise ValueError(
                        "TD projection does not match the full production basis"
                    )
                solve_state = run_gain_map.restrict(state, solve_problem.part)
            elif state.shape != solve_problem.zeros().shape:
                raise ValueError(
                    "TD projection does not match the production HB basis"
                )
            fallback_state = solve_state
            solver = exp08.HarmonicNewtonKrylovSolver(self.engine._settings())
            homotopy_started = time.perf_counter()
            homotopy_state, homotopy_reports, homotopy_trace = (
                solver.solve_residual_homotopy(
                    solve_problem,
                    solve_state,
                    1.0,
                    initial_step=float(
                        getattr(self.args, "td_homotopy_initial_step", 0.1)
                    ),
                    min_step=float(
                        getattr(self.args, "td_homotopy_min_step", 1.0 / 128.0)
                    ),
                    max_step=float(
                        getattr(self.args, "td_homotopy_max_step", 0.25)
                    ),
                    max_steps=int(
                        getattr(self.args, "td_homotopy_max_steps", 64)
                    ),
                    max_wall_s=float(
                        getattr(self.args, "td_homotopy_deadline_s", 180.0)
                    ),
                )
            )
            homotopy_info = {
                "attempted": True,
                "reached_target": bool(homotopy_trace.reached_target),
                "final_eta": float(homotopy_trace.final_eta),
                "attempted_eta": homotopy_trace.attempted_eta,
                "accepted_eta": homotopy_trace.accepted_eta,
                "failed_attempts": int(homotopy_trace.failed_attempts),
                "failure_reason": homotopy_trace.failure_reason,
                "reports": len(homotopy_reports),
                "runtime_s": time.perf_counter() - homotopy_started,
            }
            if homotopy_trace.reached_target:
                state = homotopy_state
            else:
                state = None
        except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
            homotopy_info = {
                "attempted": True,
                "reached_target": False,
                "failure_reason": repr(exc),
                "runtime_s": time.perf_counter() - started,
            }
            state = None

        if state is not None:
            row, solved = self.engine.solve_point(
                point, self.pass_dir, mode="warm", warm_X=state,
            )
            self._attach_homotopy_metadata(point, row, homotopy_info)
            return self._result(
                row,
                solved,
                SolverRoute.TD_TO_HB_RESTART,
                started,
                point,
                {"td_seed": seed, "td_residual_homotopy": homotopy_info},
            )

        # Preserve the historical direct retry as a diagnostic fallback.  It
        # is still subject to the ordinary production residual/provenance gate
        # and therefore cannot turn an unpolished TD projection into a valid
        # gain point.
        row, solved = self.engine.solve_point(
            point, self.pass_dir, mode="warm", warm_X=fallback_state,
        )
        self._attach_homotopy_metadata(point, row, homotopy_info)
        return self._result(
            row, solved, SolverRoute.TD_TO_HB_RESTART, started, point,
            {"td_seed": seed, "td_residual_homotopy": homotopy_info},
        )

    def evaluate_td_period1(self, target: Any, seed: str) -> HBResult:
        """Reject an unpolished TD PERIOD_1 projection as a gain state.

        The method remains part of the adapter protocol so callers can record
        an explicit unresolved TD outcome.  Gain evaluation is intentionally
        not performed here: a Fourier projection is not an authoritative HB
        root until residual homotopy or periodic-orbit correction succeeds.
        """
        point = self.points[target.index]
        started = time.perf_counter()
        projected = np.load(seed)
        X = np.asarray(projected["X_real"], dtype=np.float64) + 1j * np.asarray(
            projected["X_imag"], dtype=np.float64
        )
        modes = np.asarray(
            projected.get("modes", projected.get("pump_modes")), dtype=np.int64
        ).reshape(-1)
        if modes.size != X.shape[0]:
            raise ValueError("TD projected state harmonic metadata does not match X")

        td_summary: dict[str, Any] = {}
        summary_path = Path(seed).parent / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            td_summary = payload.get("branch_transfer") or {}
        omega = 2.0 * np.pi * float(point.pump_freq_ghz) * 1e9
        pump_dir = self.pass_dir / "points" / run_gain_map.point_name(
            point.index, point.power_dbm, point.pump_freq_ghz
        ) / "pump"
        gain_dir = pump_dir.parent / "gain"
        pump_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pump_dir / "pump_solution.npz",
            X_real=X.real,
            X_imag=X.imag,
            harmonics=modes,
            pump_modes=modes,
        )
        projected_residual = td_summary.get("projected_hb_coeff_rel")
        report = {
            "metadata": {
                "hb_solver_family": "td_period1_projection",
                "pump_freq_ghz": float(point.pump_freq_ghz),
                "pump_frequency_hz": float(point.pump_freq_ghz) * 1e9,
                "omega_p": omega,
                "pump_modes": modes.tolist(),
                "nt": int(self.args.nt),
                "pump_current_a": float(point.current_a * self.scale),
                "pump_solution_dtype": "float64",
                "td_projection": True,
                "td_projection_residual_rel": projected_residual,
            },
            "final_status": "TD_PERIOD1_PROJECTION",
        }
        (pump_dir / "pump_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        # The projection may be a physically settled PERIOD_1 orbit, but it is
        # not an authoritative HB state until residual homotopy or a later
        # periodic-orbit correction proves it.  Do not count approximate TD
        # gain as ordinary solved map coverage.
        gain = None
        timing: dict[str, Any] = {}
        spectrum = None
        gain_ok = False
        row: dict[str, Any] = {
            "point_index": point.index,
            "i_power": point.i_power,
            "j_freq": point.j_freq,
            "pump_power_dbm": point.power_dbm,
            "pump_freq_ghz": point.pump_freq_ghz,
            "pump_current_peak_a": point.current_a,
            "pump_status": "TD_PERIOD1",
            "gain_status": gain.status if gain is not None else "ERROR",
            "status": "ERROR",
            "warm_started": True,
            "pump_predictor": "td_period1_projection",
            "pump_coeff_rel": projected_residual,
            "pump_time_rel": projected_residual,
            "gain_failure_reason": "TD PERIOD_1 projection is not an HB root",
            "pump_dir": str(pump_dir),
            "elapsed_s": time.perf_counter() - started,
            "td_period1_projection": True,
            "td_projected_state_path": str(seed),
        }
        if gain is not None:
            row.update({
                "gain_db": float(gain.gain_db) if gain_ok else None,
                "gain_vs_off_db": float(gain.gain_vs_off_db) if gain_ok else None,
                "gain_vs_pumpdiag_db": float(gain.gain_vs_pumpdiag_db) if gain_ok else None,
                "signal_ghz": float(gain.signal_ghz),
                "linear_rel_residual": float(gain.linear_rel_residual),
            })
        row.update(timing)
        if spectrum is not None:
            row["_spectrum"] = spectrum
        self.map_rows[point.index] = row
        return HBResult(
            success=gain_ok,
            state=X,
            residual_rel=float(projected_residual) if projected_residual is not None else None,
            route=SolverRoute.TD_PERIOD1_GAIN,
            reason="TD PERIOD_1 projection is not an HB root",
            checkpoint=str(pump_dir),
            runtime_s=time.perf_counter() - started,
            metadata={
                "td_period1_gain": True,
                "td_projected_state_path": str(seed),
                "projected_hb_residual_rel": projected_residual,
                "map_row": row,
            },
        )

    def _result(
        self, row: dict[str, Any], state: Any, route: SolverRoute,
        started: float, point: Any, metadata: dict[str, Any] | None = None,
    ) -> HBResult:
        values = dict(metadata or {})
        values.update({"current_a": point.current_a * self.scale, "point_index": point.index})
        values.update({name: row.get(name) for name in _TIMING_FIELDS})
        values["sidebands"] = row.get("sidebands")
        values["single_tone_forced"] = row.get("single_tone_forced", False)
        residual = row.get("pump_coeff_rel")
        valid_residual = residual is not None and np.isfinite(float(residual))
        checkpoint = self.pass_dir / "points" / run_gain_map.point_name(
            point.index, point.power_dbm, point.pump_freq_ghz
        ) / "pump"
        if self.compact_storage:
            if _pump_success(row):
                # Keep the previous successful checkpoint until this new one
                # has succeeded.  A failed target may immediately enter the TD
                # bridge, which needs the last valid HB report and state as its
                # physical anchor.
                if self.retained_checkpoint is not None and self.retained_checkpoint != checkpoint:
                    for name in (
                        "pump_solution.npz",
                        "pump_report.json",
                        "hybrid_fixture_config.json",
                    ):
                        (self.retained_checkpoint / name).unlink(missing_ok=True)
                (checkpoint / "hybrid_point_summary.json").write_text(
                    json.dumps(
                        {k: v for k, v in row.items() if not k.startswith("_")},
                        indent=2, default=str,
                    ), encoding="utf-8",
                )
                # Persist the exact production fixture metadata used by the
                # HB solve.  The TD bridge must not reconstruct DC bias or
                # source conventions independently from nominal CLI values.
                (checkpoint / "hybrid_fixture_config.json").write_text(
                    json.dumps(
                        {
                            "pump_port": int(self.args.pump_port),
                            "frequency_hz": float(point.pump_freq_ghz) * 1e9,
                            "pump_current_a": float(point.current_a * self.scale),
                            "dc_branch_flux": np.asarray(
                                self.engine.dc_branch_flux, dtype=float
                            ).tolist(),
                            "source_convention": "peak Norton current, cos(omega t)",
                            "source_mode": 1,
                        },
                        indent=2,
                    ), encoding="utf-8",
                )
                self.retained_checkpoint = checkpoint
            else:
                for name in (
                    "pump_solution.npz",
                    "pump_report.json",
                    "hybrid_fixture_config.json",
                ):
                    (checkpoint / name).unlink(missing_ok=True)
        if not hasattr(self, "map_rows"):
            self.map_rows = {}
        self.map_rows[point.index] = {
            k: v for k, v in row.items() if not k.startswith("_")
        }
        return HBResult(
            success=(
                _pump_success(row) and valid_residual
                and float(residual) <= self.residual_threshold
            ), state=state,
            residual_rel=row.get("pump_coeff_rel"), route=route,
            reason=row.get("pump_failure_reason"), runtime_s=time.perf_counter() - started,
            checkpoint=str(checkpoint), metadata=values,
        )


class H1DynamicBackend:
    """Adapter for the validated H1 transient/classification pipeline."""

    def __init__(
        self, circuit_dir: Path, freq_ghz: float, args: argparse.Namespace,
        pump_current_scale: float, ramp_periods: int, hold_periods: int,
        checkpoint_periods: int,
    ) -> None:
        self.circuit_dir = circuit_dir
        self.freq_ghz = freq_ghz
        self.args = args
        self.pump_current_scale = pump_current_scale
        self.ramp_periods = ramp_periods
        self.hold_periods = hold_periods
        self.checkpoint_periods = checkpoint_periods

    def _run(
        self, start_checkpoint: str, target: Any, output_dir: Path,
        transient_restart: str | None = None,
        ramp_periods: int | None = None,
        hold_periods: int | None = None,
    ) -> TDResult:
        options = argparse.Namespace(**vars(self.args))
        options.circuit_dir = self.circuit_dir
        options.checkpoint = Path(start_checkpoint)
        options.transient_restart = (
            Path(transient_restart) if transient_restart is not None else None
        )
        options.outdir = output_dir
        options.freq_ghz = self.freq_ghz
        options.target_current_a = target.current_a * self.pump_current_scale
        effective_ramp_periods = (
            self.ramp_periods if ramp_periods is None else int(ramp_periods)
        )
        effective_hold_periods = (
            self.hold_periods if hold_periods is None else int(hold_periods)
        )
        options.ramp_periods = effective_ramp_periods
        options.hold_periods = effective_hold_periods
        options.checkpoint_periods = self.checkpoint_periods
        options.compact_output = True
        result = h1.run_experiment(options)
        raw_classification = {
            "PERIOD_1": TDClass.PERIOD_1,
            "PERIOD_2": TDClass.PERSISTENT_PERIOD_N,
            "PERIOD_3": TDClass.PERSISTENT_PERIOD_N,
            "QUASIPERIODIC_OR_PERIOD_N": TDClass.PERSISTENT_NONPERIODIC,
            "BROADBAND_OR_CHAOTIC": TDClass.PERSISTENT_NONPERIODIC,
            "RUNNING_PHASE": TDClass.RUNNING_PHASE,
            "TRANSIENT_NUMERICAL_FAILURE": TDClass.TRANSIENT_NUMERICAL_FAILURE,
        }.get(result.get("classification"), TDClass.UNRESOLVED_SLOW_RELAXATION)
        decay_class = (result.get("decay_aware") or {}).get("class")
        if (
            result.get("classification") in {
                "QUASIPERIODIC_OR_PERIOD_N", "BROADBAND_OR_CHAOTIC"
            }
            and decay_class in {"UNRESOLVED_SLOW_RELAXATION", "RELAXING_TO_PERIOD1"}
        ):
            classification = TDClass.UNRESOLVED_SLOW_RELAXATION
        else:
            classification = raw_classification
        strobe = result.get("stroboscopic", {})
        d1 = strobe.get("d1", [])
        best = [strobe[key][-1] for key in ("d2", "d3") if strobe.get(key)]
        branch = result.get("branch_transfer") or {}
        integrator = result.get("integrator") or {}
        observable_path = output_dir / "transient_observables.npz"
        if not observable_path.exists():
            observable_path = output_dir / "td_compact.npz"
        r_j = None
        if observable_path.exists():
            observables = np.load(observable_path)
            if "max_abs_sin_phi" in observables:
                r_j = float(np.max(observables["max_abs_sin_phi"]))
        return TDResult(
            classification=classification,
            restart_seed=branch.get("projected_state_path") if classification == TDClass.PERIOD_1 else None,
            d1=float(d1[-1]) if d1 else None,
            best_low_order_dn=min(best) if best else None,
            periods=int(result.get("hold_periods", 0)),
            r_j=branch.get("transient_rj", r_j) if branch else r_j,
            phase_winding=result.get("mean_phase_winding_cycles"),
            runtime_s=float(integrator.get("runtime_s", 0.0)),
            metadata={
                "classification_raw": result.get("classification"),
                "decay_aware": result.get("decay_aware"),
                "integrator": integrator,
                "outdir": str(output_dir),
                "hb_checkpoint": str(options.checkpoint),
                "restart_checkpoint": str(
                    output_dir / "restart_checkpoints" / "transient_restart.npz"
                ),
                "ramp_periods": effective_ramp_periods,
                "hold_periods": effective_hold_periods,
            },
        )

    def bridge(self, start: HBResult, target: Any, output_dir: Path) -> TDResult:
        return self._run(start.checkpoint or "", target, output_dir)

    def continue_from_td(
        self, previous: TDResult, target: Any, output_dir: Path
    ) -> TDResult:
        metadata = previous.metadata
        restart = metadata.get("restart_checkpoint")
        checkpoint = metadata.get("hb_checkpoint")
        if not restart or not Path(restart).exists() or not checkpoint:
            return TDResult(
                TDClass.TRANSIENT_NUMERICAL_FAILURE,
                periods=0,
                metadata={
                    **metadata,
                    "failure_reason": "missing TD continuation checkpoint",
                },
            )
        return self._run(checkpoint, target, output_dir, restart)

    def extend_from_td(
        self, previous: TDResult, target: Any, output_dir: Path
    ) -> TDResult:
        """Hold the already-reached drive while slow PERIOD_1 decay settles."""
        metadata = previous.metadata
        restart = metadata.get("restart_checkpoint")
        checkpoint = metadata.get("hb_checkpoint")
        if not restart or not Path(restart).exists() or not checkpoint:
            return TDResult(
                TDClass.TRANSIENT_NUMERICAL_FAILURE,
                periods=0,
                metadata={
                    **metadata,
                    "failure_reason": "missing TD extension checkpoint",
                },
            )
        return self._run(
            checkpoint,
            target,
            output_dir,
            transient_restart=restart,
            ramp_periods=0,
            hold_periods=self.hold_periods,
        )


def build_targets(args: argparse.Namespace) -> tuple[list[Any], argparse.Namespace]:
    gain_args = _build_args(
        args.circuit_dir, args.outdir, args.freq_ghz, args.n_power,
        args.power_min_dbm, args.power_max_dbm,
    )
    gain_args.attenuation_db = args.attenuation_db
    gain_args.sidebands = args.sidebands
    gain_args.force_single_tone = args.force_single_tone
    points, _, _ = run_gain_map.build_points(gain_args)
    return sorted(points, key=lambda item: item.power_dbm), gain_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-dir", type=Path, default=ROOT / "designs" / "ipm_2c_fixed")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--freq-ghz", type=float, required=True)
    parser.add_argument("--n-power", type=int, default=20)
    parser.add_argument("--power-min-dbm", type=float, default=-26.0)
    parser.add_argument("--power-max-dbm", type=float, default=-16.0)
    parser.add_argument(
        "--attenuation-db", type=float, default=None,
        help="Optional flat pump-line attenuation override; omitted uses loss_A10.",
    )
    parser.add_argument(
        "--sidebands", type=int, default=6,
        help="Explicit small-signal sideband count for the delegated gain map.",
    )
    parser.add_argument(
        "--force-single-tone", action="store_true",
        help="Solve pump-only single-tone HB and skip delegated signal solves.",
    )
    parser.add_argument("--td-ramp-periods", type=int, default=10)
    parser.add_argument("--td-hold-periods", type=int, default=40)
    parser.add_argument("--td-checkpoint-periods", type=int, default=10)
    parser.add_argument(
        "--td-min-step-theta", type=float, default=1.0 / 32.0,
        help="Minimum implicit-trapezoid step used after Newton retries.",
    )
    parser.add_argument(
        "--td-max-newton", type=int, default=20,
        help="Maximum Newton corrections per adaptive TD step.",
    )
    parser.add_argument("--max-td-bridges", type=int, default=2)
    parser.add_argument(
        "--td-settle-extensions", type=int, default=0,
        help=(
            "Additional fixed-drive TD hold blocks for decay telemetry that "
            "explicitly indicates relaxation toward PERIOD_1."
        ),
    )
    parser.add_argument(
        "--td-junction-break-threshold", type=float, default=1.0 - 1e-6,
        help="Promote running-phase TD to a physical boundary only at this |I_J|/Ic threshold.",
    )
    parser.add_argument(
        "--td-continue-nonperiodic", action="store_true",
        help=(
            "Research mode: continue the TD ramp from non-periodic or "
            "slow-relaxing states. These states remain non-gain-valid and "
            "are never reported as a physical boundary by this mode."
        ),
    )
    parser.add_argument(
        "--td-homotopy-initial-step", type=float, default=0.1,
        help="Initial TD-to-HB residual-homotopy eta step.",
    )
    parser.add_argument(
        "--td-homotopy-min-step", type=float, default=1.0 / 128.0,
        help="Smallest accepted residual-homotopy eta step.",
    )
    parser.add_argument(
        "--td-homotopy-max-step", type=float, default=0.25,
        help="Largest residual-homotopy eta step.",
    )
    parser.add_argument(
        "--td-homotopy-max-steps", type=int, default=64,
        help="Maximum residual-homotopy corrector attempts.",
    )
    parser.add_argument(
        "--td-homotopy-deadline-s", type=float, default=180.0,
        help="Wall-time budget for one TD-to-HB residual homotopy.",
    )
    parser.add_argument("--hb-residual-threshold", dest="hybrid_hb_residual_threshold", type=float, default=1e-8)
    args = parser.parse_args(argv)
    args.outdir = prepare_output_dir(args.outdir, args.freq_ghz)
    targets, gain_args = build_targets(args)
    gain_args.attenuation_db = args.attenuation_db
    # The hybrid handoff validates the persisted full-node checkpoint after
    # Schur reconstruction.  Keep the nonlinear solve on the scalable Schur
    # backend: the full-node backend is not numerically equivalent near the
    # 2c high-power obstruction and can fail even at a known-valid anchor.
    gain_args.inproc_pump_backend = "schur_cpu_mt"
    gain_args.inproc_preconditioner = "real_coupled_fast"
    # A single live Schur partition is enough for one frequency column and
    # keeps native sparse factors bounded during high-power recovery.
    gain_args.inproc_schur_cache_size = 1
    gain_args.pump_solution_dtype = "float64"
    gain_args.hybrid_compact_storage = True
    gain_args.td_homotopy_initial_step = args.td_homotopy_initial_step
    gain_args.td_homotopy_min_step = args.td_homotopy_min_step
    gain_args.td_homotopy_max_step = args.td_homotopy_max_step
    gain_args.td_homotopy_max_steps = args.td_homotopy_max_steps
    gain_args.td_homotopy_deadline_s = args.td_homotopy_deadline_s
    backend = ProductionPeriodicBackend(gain_args, [
        point for point in run_gain_map.build_points(gain_args)[0]
    ], args.outdir)
    h1_args = h1.parse_args([])
    # Keep the transient bridge on the same physical port and external-flux
    # convention as the HB/gain column.  H1 has historically defaulted to the
    # four-port IPM pump (port 4), which is not present in compact RF-SQUID
    # validation designs.
    h1_args.pump_port = int(gain_args.pump_port)
    h1_args.min_step_theta = float(args.td_min_step_theta)
    h1_args.max_newton = int(args.td_max_newton)
    h1_args.dc_flux_over_phi0 = float(
        gain_args.dc_branch_flux_over_phi0
        if gain_args.dc_branch_flux_over_phi0 is not None else 0.0
    )
    dynamic = H1DynamicBackend(
        args.circuit_dir, args.freq_ghz, h1_args,
        gain_args.pump_current_jc_scale,
        args.td_ramp_periods, args.td_hold_periods, args.td_checkpoint_periods,
    )
    result = ColumnController(
        backend, dynamic, args.outdir,
        ColumnBudget(
            max_td_bridges=args.max_td_bridges,
            continue_nonperiodic=args.td_continue_nonperiodic,
            max_td_settle_extensions=args.td_settle_extensions,
            junction_break_threshold=args.td_junction_break_threshold,
        ),
    ).run(targets)
    result.write_json(args.outdir / "hybrid_column_summary.json")
    print(json.dumps({"status": result.status.value, "td_bridges": result.td_bridges}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
