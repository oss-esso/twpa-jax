"""CLI defaults for the gain-map runner."""

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

import scripts.run_gain_map as run_gain_map


def test_inproc_fail_fast_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])

    args = run_gain_map.parse_args()

    assert args.inproc_fail_fast is False
    assert args.fold_skip_patience == 0


def test_inproc_fail_fast_flag_enables_fast_failure(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py", "--inproc-fail-fast"])

    args = run_gain_map.parse_args()

    assert args.inproc_fail_fast is True


def test_all_intra_cell_continuation_methods_are_selectable(monkeypatch) -> None:
    methods = {
        "fixed",
        "adaptive_copy",
        "adaptive_secant",
        "adaptive_tangent",
        "affine",
        "ptc",
        "arclength",
    }
    for method in methods:
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_gain_map.py", "--inproc-continuation", method],
        )
        assert run_gain_map.parse_args().inproc_continuation == method


def test_solve_deadline_alias_matches_canonical_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_gain_map.py", "--inproc-solve-deadline", "14"],
    )
    assert run_gain_map.parse_args().inproc_solve_deadline_s == 14.0


def test_column_arclength_recovery_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])
    assert run_gain_map.parse_args().column_arclength_recovery is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_gain_map.py", "--column-arclength-recovery"],
    )
    assert run_gain_map.parse_args().column_arclength_recovery is True


def test_column_arclength_has_separate_trace_deadline(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])
    assert run_gain_map.parse_args().column_arclength_deadline_s == 180.0

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_gain_map.py", "--column-arclength-deadline-s", "12"],
    )
    assert run_gain_map.parse_args().column_arclength_deadline_s == 12.0


def test_column_power_substep_is_opt_in_with_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])
    args = run_gain_map.parse_args()
    assert args.column_power_substep is False
    assert args.column_power_substep_init_db == 0.1
    assert args.column_power_substep_min_db == 0.005
    assert args.column_power_substep_deadline_s == 120.0

    monkeypatch.setattr(
        sys, "argv",
        ["run_gain_map.py", "--column-power-substep",
         "--column-power-substep-min-db", "0.01"],
    )
    args = run_gain_map.parse_args()
    assert args.column_power_substep is True
    assert args.column_power_substep_min_db == 0.01


def _fake_substep_engine(monkeypatch, converge_rule):
    """Bind solve_power_substep to a fake engine with a scripted solver.

    ``converge_rule(dist_db, step_db) -> bool`` decides convergence, where
    ``dist_db`` is the dBm advanced from the start and ``step_db`` is the
    proposed micro-step. The fake problem carries only its current; the guess
    array encodes the last accepted current so the method's dBm bookkeeping is
    exercised for real.
    """
    class _Report:
        def __init__(self, converged: bool) -> None:
            self.converged = converged

    class _Solver:
        def __init__(self, _settings) -> None:
            pass

        def solve_one(self, prob, guess, _scale):
            trial = prob.cur
            prev = float(guess[0])
            step_db = 20.0 * math.log10(trial / prev)
            dist_db = 20.0 * math.log10(trial / prob.start)
            return np.array([trial]), _Report(converge_rule(dist_db, step_db))

    monkeypatch.setattr(run_gain_map.exp08, "HarmonicNewtonKrylovSolver", _Solver)
    start = 1.0

    def build(freq, cur):
        return SimpleNamespace(cur=cur, start=start), None, None

    eng = SimpleNamespace(
        _settings=lambda: None,
        _build_problem=build,
        _make_solve_problem=lambda prob, freq: prob,
    )
    method = run_gain_map.InProcessEngine.solve_power_substep.__get__(
        eng, run_gain_map.InProcessEngine
    )
    return method, start


def test_power_substep_crosses_a_crest_by_shrinking_step(monkeypatch) -> None:
    # Crest between 0.5 and 0.7 dB tolerates only <=0.05 dB steps; elsewhere
    # coarse 0.1 dB steps pass. The adaptive walk must shrink to cross it.
    def rule(dist_db, step_db):
        allowed = 0.05 if 0.5 <= dist_db <= 0.7 else 0.25
        return step_db <= allowed + 1e-9

    method, start = _fake_substep_engine(monkeypatch, rule)
    target = start * 10.0 ** (1.0 / 20.0)  # +1.0 dB

    X, info = method(8.0, np.array([start]), start, target,
                     init_db=0.1, min_db=0.005, deadline_s=30.0)

    assert X is not None
    assert info["reached_target"] is True
    assert info["terminal_reason"] == "reached"
    assert info["min_step_db"] <= 0.05 + 1e-9  # had to shrink at the crest


def test_power_substep_reports_step_floor_at_a_hard_fold(monkeypatch) -> None:
    # Nothing converges past 0.5 dB (a fold); the walk must give up at the
    # min-db floor and report a step-independent stall, not silently succeed.
    def rule(dist_db, step_db):
        return dist_db <= 0.5 + 1e-9

    method, start = _fake_substep_engine(monkeypatch, rule)
    target = start * 10.0 ** (1.0 / 20.0)  # +1.0 dB, past the fold

    X, info = method(8.0, np.array([start]), start, target,
                     init_db=0.1, min_db=0.05, deadline_s=30.0)

    assert X is None
    assert info["reached_target"] is False
    assert info["terminal_reason"] == "step_floor"
    # advanced to ~0.5 dB before stalling
    assert 20.0 * math.log10(info["last_current"] / start) <= 0.5 + 1e-6


def _fake_solve_point_engine(monkeypatch, *, converged: bool):
    """Bind solve_point to a fake engine whose pump solve reports `converged`.

    Stubs the heavy pump/gain internals so only the convergence-gate logic of
    solve_point (does gain run? is X returned for chaining?) is exercised.
    """
    class _Report:
        source_scale = 1.0
        time_rel = None
        newton_iterations = 4
        gmres_iterations_total = 10
        factor_runtime_s = 0.0
        runtime_s = 0.1
        preconditioner_assembly_runtime_s = 0.0
        preconditioner_numeric_factor_runtime_s = 0.0

        def __init__(self) -> None:
            self.converged = converged
            self.coeff_rel = 1e-12 if converged else 1e-3
            self.failure_reason = None if converged else "stalled"

    class _Solver:
        def __init__(self, _settings) -> None:
            pass

        def solve_direct(self, prob, warm_X):
            return np.array([1.0, 2.0, 3.0]), [_Report()]

    monkeypatch.setattr(run_gain_map.exp08, "HarmonicNewtonKrylovSolver", _Solver)
    monkeypatch.setattr(run_gain_map.exp08, "summarize_solution",
                        lambda prob, X: {"branch_i_max_abs": 1.0})
    monkeypatch.setattr(run_gain_map.exp08, "write_results",
                        lambda *a, **k: None)

    gain_calls: list[int] = []

    def fake_gain(pump_dir, gain_dir, freq_ghz):
        gain_calls.append(1)
        g = SimpleNamespace(
            status="VALID_SOLVED", gain_db=12.0, gain_vs_off_db=12.0,
            gain_vs_pumpdiag_db=12.0, signal_ghz=6.0, linear_rel_residual=1e-9,
        )
        return g, {}, None

    basis = SimpleNamespace(to_metadata=lambda: {"pump_basis": "x"})
    args = SimpleNamespace(
        inproc_pump_backend="full", nt=40, newton_tol=1e-9, inproc_max_newton=16,
        inproc_gmres_maxiter=80, inproc_preconditioner="real_coupled",
        inproc_solve_deadline_s=0.0, inproc_precond_reuse=0,
        inproc_precond_refresh_gmres=0, inproc_continuation="adaptive_secant",
    )
    eng = SimpleNamespace(
        args=args, ic_median=1.0, _gain=fake_gain,
        _settings=lambda: None,
        build_problem_for=lambda point: (object(), basis, 1.0, 1.0),
        _make_solve_problem=lambda full, freq: full,
    )
    method = run_gain_map.InProcessEngine.solve_point.__get__(
        eng, run_gain_map.InProcessEngine
    )
    return method, gain_calls


def _make_point():
    return run_gain_map.GridPoint(
        index=0, i_power=0, j_freq=0, power_dbm=-20.0, pump_freq_ghz=8.0,
        current_a=1e-6,
    )


def test_force_gain_computes_gain_and_chains_x_when_pump_not_converged(
    monkeypatch, tmp_path
) -> None:
    method, gain_calls = _fake_solve_point_engine(monkeypatch, converged=False)
    row, X = method(_make_point(), tmp_path, mode="warm",
                    warm_X=np.array([0.0, 0.0, 0.0]), force_gain=True)
    # gain ran on the non-converged pump waveform and X is returned for chaining
    assert len(gain_calls) == 1
    assert row["pump_status"] == "FAIL"
    assert row["gain_status"] == "VALID_SOLVED"
    assert row["gain_db"] == 12.0
    assert X is not None


def test_force_gain_off_skips_gain_and_drops_x_when_pump_not_converged(
    monkeypatch, tmp_path
) -> None:
    method, gain_calls = _fake_solve_point_engine(monkeypatch, converged=False)
    row, X = method(_make_point(), tmp_path, mode="warm",
                    warm_X=np.array([0.0, 0.0, 0.0]), force_gain=False)
    # default behaviour: no gain solve, X dropped so it can't seed the next cell
    assert gain_calls == []
    assert row["gain_status"] == "ERROR"
    assert X is None


def test_frequency_chunk_size_defaults_to_ten_columns(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])

    args = run_gain_map.parse_args()

    assert args.frequency_chunk_size == 10
    assert args.resume_chunks is True
    assert args.signal_spectrum is True
    assert args.local_traversal_chunks is False


def test_local_traversal_chunks_are_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_gain_map.py", "--local-traversal-chunks"],
    )

    args = run_gain_map.parse_args()

    assert args.local_traversal_chunks is True


def test_column_bridge_uses_generic_recovery_orchestrator(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_gain_map.py", "--traversal", "column", "--recovery", "bridge"],
    )

    args = run_gain_map.parse_args()

    assert run_gain_map.uses_traversal_orchestrator(args) is True


def test_legacy_column_control_keeps_legacy_orchestrator(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py"])

    args = run_gain_map.parse_args()

    assert run_gain_map.uses_traversal_orchestrator(args) is False


def test_signal_spectrum_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_gain_map.py", "--no-signal-spectrum"])

    args = run_gain_map.parse_args()

    assert args.signal_spectrum is False


def test_frequency_chunk_ranges_are_half_open() -> None:
    assert run_gain_map.frequency_chunk_ranges(25, 10) == [(0, 10), (10, 20), (20, 25)]
    assert run_gain_map.frequency_chunk_ranges(8, 10) == [(0, 8)]
    assert run_gain_map.frequency_chunk_ranges(8, 0) == [(0, 8)]


def test_chunk_worker_command_strips_parent_routing_options() -> None:
    cmd = run_gain_map.chunk_worker_command(
        [
            "--outdir",
            "outputs/full",
            "--frequency-chunk-size",
            "10",
            "--gate-spotcheck=5",
            "--overwrite",
            "--n-frequency",
            "50",
            "--pump-freq-min-ghz",
            "7.5",
            "--pump-freq-max-ghz",
            "8.5",
        ],
        outdir=Path("outputs/full/chunks/chunk_000"),
        n_frequency=10,
        pump_freq_min_ghz=7.5,
        pump_freq_max_ghz=7.683673469387755,
    )

    assert "--chunk-worker" in cmd
    assert "--n-frequency" in cmd
    assert "10" in cmd
    assert "--pump-freq-min-ghz" in cmd
    assert "--pump-freq-max-ghz" in cmd
    assert "7.5" in cmd
    assert "7.68367346939" in cmd
    assert "50" not in cmd
    assert "8.5" not in cmd
    assert "--frequency-chunk-size" not in cmd
    assert "--gate-spotcheck=5" not in cmd
    assert cmd.count("--outdir") == 1


def test_read_chunk_rows_globalizes_frequency_and_point_indices(tmp_path) -> None:
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir()
    (chunk_dir / "map_points.csv").write_text(
        "pass,point_index,i_power,j_freq,pump_power_dbm,pump_freq_ghz,status\n"
        "warm,0,0,0,-30,7.5,PASS\n"
        "warm,3,1,1,-29,7.6,PASS\n",
        encoding="utf-8",
    )

    _cold, warm = run_gain_map.read_chunk_rows(
        [(chunk_dir, 10, 12)],
        global_n_frequency=50,
    )

    assert warm[0]["j_freq"] == 10
    assert warm[0]["point_index"] == 10
    assert warm[1]["j_freq"] == 11
    assert warm[1]["point_index"] == 61


def test_fail_fast_does_not_retry_secant_fallback(tmp_path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                inproc_fold_predictor="secant",
                pump_current_jc_scale=1.0,
                fold_skip_patience=0,
            )
            self.calls = []

        def solve_point(self, point, pass_dir, *, mode, warm_X):
            self.calls.append((point.index, mode, None if warm_X is None else warm_X.copy()))
            status = "ERROR" if point.index == 2 else "PASS"
            row = {
                "point_index": point.index,
                "status": status,
                "gain_db": None,
                "pump_newton_total": 1,
                "pump_runtime_s": 0.1,
            }
            return row, np.array([float(point.index + 1)])

    points = [
        run_gain_map.GridPoint(0, 0, 0, -35.0, 7.5, 1.0),
        run_gain_map.GridPoint(1, 1, 0, -34.0, 7.5, 2.0),
        run_gain_map.GridPoint(2, 2, 0, -33.0, 7.5, 3.0),
    ]
    engine = FakeEngine()

    rows = run_gain_map.run_warm_pass_inprocess(
        points,
        tmp_path,
        engine,
        fail_fast=True,
    )

    assert [call[0] for call in engine.calls] == [0, 1, 2]
    assert rows[-1]["status"] == "ERROR"
    assert rows[-1]["pump_predictor"] == "secant"


def test_consecutive_failures_do_not_skip_without_verified_fold(tmp_path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                inproc_fold_predictor="none",
                pump_current_jc_scale=1.0,
                fold_skip_patience=1,
                column_arclength_recovery=False,
            )

        def solve_point(self, point, pass_dir, *, mode, warm_X):
            row = {
                "point_index": point.index,
                "status": "ERROR" if point.index >= 1 else "PASS",
                "gain_db": None,
                "pump_newton_total": 1,
                "pump_runtime_s": 0.1,
            }
            return row, np.array([float(point.index + 1)])

    points = [
        run_gain_map.GridPoint(0, 0, 0, -35.0, 7.5, 1.0),
        run_gain_map.GridPoint(1, 1, 0, -34.0, 7.5, 2.0),
        run_gain_map.GridPoint(2, 2, 0, -33.0, 7.5, 3.0),
    ]

    rows = run_gain_map.run_warm_pass_inprocess(
        points,
        tmp_path,
        FakeEngine(),
        fail_fast=True,
    )

    assert len(rows) == 3
    assert all(row["status"] == "ERROR" for row in rows[1:])


def test_fold_skip_requires_arclength_turning_point(tmp_path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                inproc_fold_predictor="none",
                pump_current_jc_scale=1.0,
                fold_skip_patience=1,
                column_arclength_recovery=True,
                column_arclength_deadline_s=1.0,
            )

        def solve_point(self, point, pass_dir, *, mode, warm_X):
            status = "ERROR" if point.index == 2 else "PASS"
            row = {
                "point_index": point.index,
                "status": status,
                "gain_db": None,
                "pump_newton_total": 1,
                "pump_runtime_s": 0.1,
            }
            return row, np.array([float(point.index + 1)])

        def trace_column_arclength(self, *args):
            return {}, {"fold_lambdas": [0.99], "steps": 1, "trace_points": 2}

    points = [
        run_gain_map.GridPoint(i, i, 0, -35.0 + i, 7.5, float(i + 1))
        for i in range(4)
    ]
    rows = run_gain_map.run_warm_pass_inprocess(
        points,
        tmp_path,
        FakeEngine(),
        fail_fast=True,
    )

    assert [row["status"] for row in rows] == ["PASS", "PASS", "ERROR", "SKIP_PAST_FOLD"]


def test_partial_continuation_failures_enable_fold_skip(tmp_path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.args = SimpleNamespace(
                inproc_fold_predictor="none",
                pump_current_jc_scale=1.0,
                fold_skip_patience=2,
                column_arclength_recovery=False,
            )

        def solve_point(self, point, pass_dir, *, mode, warm_X):
            failed = point.index >= 1
            row = {
                "point_index": point.index,
                "status": "ERROR" if failed else "PASS",
                "gain_db": None,
                "pump_newton_total": 1,
                "pump_runtime_s": 0.1,
                "pump_continuation_method": "adaptive_secant" if failed else "direct",
                "pump_continuation_steps": 3 if failed else 1,
                "pump_continuation_reached_target": False if failed else True,
            }
            return row, np.array([float(point.index + 1)])

    points = [
        run_gain_map.GridPoint(i, i, 0, -35.0 + i, 7.5, float(i + 1))
        for i in range(5)
    ]
    rows = run_gain_map.run_warm_pass_inprocess(
        points,
        tmp_path,
        FakeEngine(),
        fail_fast=True,
    )

    assert [row["status"] for row in rows] == [
        "PASS", "ERROR", "ERROR", "SKIP_PAST_FOLD", "SKIP_PAST_FOLD"
    ]
