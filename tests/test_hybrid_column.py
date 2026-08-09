from __future__ import annotations

from pathlib import Path

from twpa_solver.hybrid_column import (
    ColumnBudget,
    ColumnController,
    ColumnState,
    HBResult,
    SolverRoute,
    TDClass,
    TDResult,
)


class FakePeriodic:
    def __init__(self) -> None:
        self.direct_calls: list[int] = []
        self.recover_calls: list[int] = []
        self.restart_calls: list[int] = []

    def solve_direct(self, target: int, previous: HBResult | None) -> HBResult:
        self.direct_calls.append(target)
        if target == 2:
            return HBResult(False, reason="fold")
        return HBResult(True, state=target, residual_rel=1e-12)

    def recover(self, target: int, previous: HBResult) -> HBResult:
        self.recover_calls.append(target)
        return HBResult(False, reason="obstruction")

    def restart_from_td(self, target: int, seed: str) -> HBResult:
        self.restart_calls.append(target)
        return HBResult(True, state=target, residual_rel=2e-12)


class FakeDynamic:
    def __init__(self, result: TDResult, continuation: TDResult | None = None) -> None:
        self.result = result
        self.continuation = continuation or result
        self.calls: list[tuple[int, Path]] = []
        self.continue_calls: list[tuple[int, Path]] = []

    def bridge(self, start: HBResult, target: int, output_dir: Path) -> TDResult:
        self.calls.append((target, output_dir))
        return self.result

    def continue_from_td(self, previous: TDResult, target: int, output_dir: Path) -> TDResult:
        self.continue_calls.append((target, output_dir))
        return self.continuation


def test_period1_bridge_restarts_hb_and_continues() -> None:
    periodic = FakePeriodic()
    dynamic = FakeDynamic(
        TDResult(TDClass.PERIOD_1, restart_seed="seed.npz", periods=12, d1=1e-5)
    )
    result = ColumnController(
        periodic, dynamic, Path("outputs/test-hybrid"), ColumnBudget(max_td_bridges=1)
    ).run([1, 2, 3])

    assert result.status == ColumnState.COLUMN_COMPLETE_NO_BOUNDARY
    assert result.hb_restart_successes == 1
    assert periodic.restart_calls == [2]
    assert [record.route for record in result.records] == [
        SolverRoute.DIRECT_HB,
        SolverRoute.TD_TO_HB_RESTART,
        SolverRoute.DIRECT_HB,
    ]


def test_persistent_nonperiodic_state_stops_column() -> None:
    periodic = FakePeriodic()
    dynamic = FakeDynamic(
        TDResult(
            TDClass.PERSISTENT_NONPERIODIC,
            periods=40,
            d1=0.02,
            best_low_order_dn=0.018,
            r_j=0.88,
        )
    )
    result = ColumnController(
        periodic, dynamic, Path("outputs/test-hybrid"), ColumnBudget(max_td_bridges=1)
    ).run([1, 2, 3])

    assert result.status == ColumnState.PHYSICAL_BOUNDARY_FOUND
    assert result.first_outside is not None
    assert result.first_outside.classification == "PERSISTENT_NONPERIODIC"
    assert len(result.records) == 2


def test_unresolved_td_is_not_called_a_physical_boundary() -> None:
    periodic = FakePeriodic()
    dynamic = FakeDynamic(
        TDResult(TDClass.UNRESOLVED_SLOW_RELAXATION, periods=200, d1=2e-3)
    )
    result = ColumnController(
        periodic, dynamic, Path("outputs/test-hybrid"), ColumnBudget(max_td_bridges=1)
    ).run([1, 2])

    assert result.status == ColumnState.COLUMN_UNRESOLVED_BUDGET
    assert result.first_outside is None


def test_failed_td_hb_restart_falls_back_to_td_continuation() -> None:
    periodic = FakePeriodic()
    periodic.restart_from_td = lambda target, seed: HBResult(
        False, reason="HB representation unavailable"
    )
    dynamic = FakeDynamic(
        TDResult(TDClass.PERIOD_1, restart_seed="seed.npz", periods=8, d1=1e-5),
        continuation=TDResult(
            TDClass.PERSISTENT_NONPERIODIC, periods=10, d1=0.04, r_j=0.91
        ),
    )
    result = ColumnController(
        periodic, dynamic, Path("outputs/test-hybrid"), ColumnBudget(max_td_bridges=2)
    ).run([1, 2, 3])

    assert result.status == ColumnState.PHYSICAL_BOUNDARY_FOUND
    assert result.first_outside is not None
    assert result.first_outside.target == 3
    assert result.records[1].route == (
        SolverRoute.TD_PERIOD1_HB_RESTART_FAILED_FALLBACK_TD
    )
    assert result.records[1].classification == TDClass.PERIOD_1.value
    assert result.td_bridges == 2
    assert dynamic.continue_calls and dynamic.continue_calls[0][0] == 3


def test_td_period1_gain_evaluation_is_distinct_from_hb_restart() -> None:
    periodic = FakePeriodic()
    periodic.restart_from_td = lambda target, seed: HBResult(
        False, reason="HB representation unavailable"
    )
    periodic.evaluate_td_period1 = lambda target, seed: HBResult(
        True, state="projected", residual_rel=2e-2,
        metadata={"td_period1_gain": True},
    )
    dynamic = FakeDynamic(
        TDResult(TDClass.PERIOD_1, restart_seed="seed.npz", periods=8, d1=1e-5),
        continuation=TDResult(
            TDClass.PERSISTENT_NONPERIODIC, periods=10, d1=0.04, r_j=0.91
        ),
    )
    result = ColumnController(
        periodic, dynamic, Path("outputs/test-hybrid"), ColumnBudget(max_td_bridges=2)
    ).run([1, 2, 3])

    assert result.status == ColumnState.PHYSICAL_BOUNDARY_FOUND
    assert result.records[1].route == SolverRoute.TD_PERIOD1_GAIN
    assert result.records[1].metadata["td_period1_gain_evaluated"] is True
    assert result.records[1].hb_residual_rel == 2e-2
    assert dynamic.continue_calls and dynamic.continue_calls[0][0] == 3


def test_compact_storage_keeps_td_anchor_after_failed_target(tmp_path) -> None:
    from scripts.run_hybrid_column import ProductionPeriodicBackend

    backend = object.__new__(ProductionPeriodicBackend)
    backend.compact_storage = True
    backend.pass_dir = tmp_path / "pass"
    backend.scale = 1.0
    backend.residual_threshold = 1e-8
    previous = backend.pass_dir / "points" / "previous" / "pump"
    previous.mkdir(parents=True)
    (previous / "pump_solution.npz").write_bytes(b"state")
    (previous / "pump_report.json").write_text("{}", encoding="utf-8")
    backend.retained_checkpoint = previous

    point = type(
        "Point",
        (),
        {"index": 1, "power_dbm": -20.0, "pump_freq_ghz": 7.9, "current_a": 1.0},
    )()
    backend._result(
        {"pump_status": "FAIL", "pump_coeff_rel": 1.0},
        None,
        SolverRoute.DIRECT_HB,
        0.0,
        point,
    )

    assert (previous / "pump_solution.npz").exists()
    assert (previous / "pump_report.json").exists()
