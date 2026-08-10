"""Bounded HB-to-transient column orchestration.

This module deliberately contains policy only.  Production HB and transient
implementations are supplied through small adapters, which keeps map policy out
of both numerical backends and makes the state machine unit-testable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class ColumnState(str, Enum):
    HB_FAST = "HB_FAST"
    HB_RECOVERY = "HB_RECOVERY"
    TD_BRIDGE = "TD_BRIDGE"
    TD_SETTLING = "TD_SETTLING"
    TD_CONTINUE = "TD_CONTINUE"
    TD_TO_HB_RESTART = "TD_TO_HB_RESTART"
    PHYSICAL_BOUNDARY_FOUND = "PHYSICAL_BOUNDARY_FOUND"
    COLUMN_COMPLETE_NO_BOUNDARY = "COLUMN_COMPLETE_NO_BOUNDARY"
    COLUMN_UNRESOLVED_BUDGET = "COLUMN_UNRESOLVED_BUDGET"
    COLUMN_NUMERICAL_FAILURE = "COLUMN_NUMERICAL_FAILURE"


class SolverRoute(str, Enum):
    DIRECT_HB = "DIRECT_HB"
    POWER_SUBSTEP = "POWER_SUBSTEP"
    PALC = "PALC"
    FREQUENCY_SUBSTEP = "FREQUENCY_SUBSTEP"
    TD_BRIDGE = "TD_BRIDGE"
    TD_CONTINUE = "TD_CONTINUE"
    TD_TO_HB_RESTART = "TD_TO_HB_RESTART"
    TD_PERIOD1_GAIN = "TD_PERIOD1_GAIN"
    TD_PERIOD1_HB_RESTART_FAILED_FALLBACK_TD = (
        "TD_PERIOD1_HB_RESTART_FAILED_FALLBACK_TD"
    )
    PHYSICAL_BOUNDARY = "PHYSICAL_BOUNDARY"
    UNRESOLVED = "UNRESOLVED"


class TDClass(str, Enum):
    PERIOD_1 = "PERIOD_1"
    RELAXING_TO_PERIOD1 = "RELAXING_TO_PERIOD1"
    PERSISTENT_PERIOD_N = "PERSISTENT_PERIOD_N"
    PERSISTENT_NONPERIODIC = "PERSISTENT_NONPERIODIC"
    RUNNING_PHASE = "RUNNING_PHASE"
    UNRESOLVED_SLOW_RELAXATION = "UNRESOLVED_SLOW_RELAXATION"
    TRANSIENT_NUMERICAL_FAILURE = "TRANSIENT_NUMERICAL_FAILURE"


@dataclass(frozen=True)
class HBResult:
    success: bool
    state: Any = None
    residual_rel: float | None = None
    route: SolverRoute = SolverRoute.DIRECT_HB
    reason: str | None = None
    checkpoint: str | None = None
    runtime_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TDResult:
    classification: TDClass
    final_state: Any = None
    restart_seed: str | None = None
    d1: float | None = None
    best_low_order_dn: float | None = None
    periods: int = 0
    r_j: float | None = None
    phase_winding: float | None = None
    runtime_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PeriodicBackend(Protocol):
    def solve_direct(self, target: Any, previous: HBResult | None) -> HBResult: ...

    def recover(self, target: Any, previous: HBResult) -> HBResult: ...

    def restart_from_td(self, target: Any, seed: str) -> HBResult: ...

    def evaluate_td_period1(self, target: Any, seed: str) -> HBResult: ...


class DynamicBackend(Protocol):
    def bridge(self, start: HBResult, target: Any, output_dir: Path) -> TDResult: ...

    def continue_from_td(
        self, previous: TDResult, target: Any, output_dir: Path
    ) -> TDResult: ...


@dataclass(frozen=True)
class ColumnBudget:
    max_td_bridges: int = 2
    max_boundary_refinements: int = 3
    max_td_periods: int = 200
    # Research-only policy.  Non-periodic TD states are never gain-valid, but
    # an explicit single-column investigation may continue the physical ramp
    # from their restart checkpoint instead of declaring a boundary.
    continue_nonperiodic: bool = False
    # Optional extra fixed-drive TD holds for decay telemetry that explicitly
    # indicates relaxation toward PERIOD_1. These do not change the drive and
    # do not make the resulting state gain-valid by themselves.
    max_td_settle_extensions: int = 0
    # A running-phase trajectory is promoted to a physical junction break only
    # when its measured CPR utilization is effectively one.  This does not
    # impose a universal subcritical utilization limit.
    junction_break_threshold: float = 1.0 - 1e-6


@dataclass
class ColumnRecord:
    target: Any
    state: ColumnState
    route: SolverRoute
    classification: str
    hb_residual_rel: float | None = None
    r_j: float | None = None
    phase_winding: float | None = None
    d1: float | None = None
    best_low_order_dn: float | None = None
    td_periods: int = 0
    reason: str | None = None
    runtime_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnResult:
    status: ColumnState
    records: list[ColumnRecord]
    last_working: ColumnRecord | None
    first_outside: ColumnRecord | None
    td_bridges: int
    td_periods: int
    hb_restart_successes: int

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": self.status.value,
            "records": [asdict(item) for item in self.records],
            "last_working": asdict(self.last_working) if self.last_working else None,
            "first_outside": asdict(self.first_outside) if self.first_outside else None,
            "td_bridges": self.td_bridges,
            "td_periods": self.td_periods,
            "hb_restart_successes": self.hb_restart_successes,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class ColumnController:
    """Run one monotonically increasing frequency column with bounded TD use."""

    def __init__(
        self,
        periodic: PeriodicBackend,
        dynamic: DynamicBackend,
        output_dir: Path,
        budget: ColumnBudget | None = None,
    ) -> None:
        self.periodic = periodic
        self.dynamic = dynamic
        self.output_dir = output_dir
        self.budget = budget or ColumnBudget()

    def run(self, targets: list[Any]) -> ColumnResult:
        records: list[ColumnRecord] = []
        previous: HBResult | None = None
        last_working: ColumnRecord | None = None
        first_outside: ColumnRecord | None = None
        td_bridges = 0
        td_periods = 0
        restarts = 0
        td_anchor: TDResult | None = None
        td_restart_disabled = False
        nonperiodic_continued = False

        def try_td_period1_gain(target: Any, td: TDResult) -> HBResult | None:
            evaluator = getattr(self.periodic, "evaluate_td_period1", None)
            if evaluator is None or not td.restart_seed:
                return None
            try:
                return evaluator(target, td.restart_seed)
            except (OSError, ValueError, RuntimeError) as exc:
                # This is an optional gain-evaluation optimization.  A failed
                # evaluation must fall through to the authoritative TD path.
                return HBResult(
                    False, reason=f"TD PERIOD_1 gain evaluation failed: {exc}"
                )

        def reaches_junction_break(td: TDResult) -> bool:
            return (
                td.classification == TDClass.RUNNING_PHASE
                and td.r_j is not None
                and td.r_j >= self.budget.junction_break_threshold
            )

        def extend_slow_relaxation(
            td: TDResult, target: Any, bridge_index: int,
        ) -> TDResult:
            """Use bounded fixed-drive holds to resolve slow PERIOD_1 decay."""
            nonlocal td_periods
            extender = getattr(self.dynamic, "extend_from_td", None)
            limit = max(0, int(self.budget.max_td_settle_extensions))
            if extender is None or limit == 0:
                return td
            extensions = 0
            while extensions < limit:
                decay = td.metadata.get("decay_aware", {})
                if (
                    td.classification != TDClass.UNRESOLVED_SLOW_RELAXATION
                    or decay.get("class") != "RELAXING_TO_PERIOD1"
                    or reaches_junction_break(td)
                ):
                    break
                extensions += 1
                td = extender(
                    td, target,
                    self.output_dir
                    / f"td_bridge_{bridge_index:02d}_settle_{extensions:02d}",
                )
                td_periods += td.periods
                td.metadata["relaxation_extensions"] = extensions
            return td

        for index, target in enumerate(targets):
            started = time.perf_counter()

            # Once a trusted PERIOD_1 transient state could not be handed back
            # to HB, continue the physical ramp from that state.  Do not jump
            # back to the older HB checkpoint or reinterpret an HB optimization
            # failure as either numerical failure or a physical boundary.
            if td_anchor is not None:
                if td_bridges >= self.budget.max_td_bridges:
                    record = self._failure_record(
                        target, ColumnState.COLUMN_UNRESOLVED_BUDGET,
                        SolverRoute.UNRESOLVED, "TD bridge budget exhausted", started,
                    )
                    records.append(record)
                    return self._result(
                        ColumnState.COLUMN_UNRESOLVED_BUDGET, records,
                        last_working, first_outside, td_bridges, td_periods, restarts,
                    )
                td_bridges += 1
                td = self.dynamic.continue_from_td(
                    td_anchor, target, self.output_dir / f"td_bridge_{td_bridges:02d}",
                )
                td_periods += td.periods
                td = extend_slow_relaxation(td, target, td_bridges)
                if td.classification == TDClass.PERIOD_1:
                    if not td_restart_disabled and td.restart_seed:
                        restarted = self.periodic.restart_from_td(target, td.restart_seed)
                        if restarted.success:
                            restarts += 1
                            record = self._hb_record(
                                target, restarted, SolverRoute.TD_TO_HB_RESTART, started,
                            )
                            record.metadata["td_classification"] = td.classification.value
                            records.append(record)
                            previous = restarted
                            last_working = record
                            td_anchor = None
                            td_restart_disabled = False
                            continue
                    evaluated = try_td_period1_gain(target, td)
                    if evaluated is not None and evaluated.success:
                        record = self._hb_record(
                            target, evaluated, SolverRoute.TD_PERIOD1_GAIN, started,
                        )
                        record.metadata["td_classification"] = td.classification.value
                        record.metadata["hb_restart_available"] = False
                        record.metadata["td_period1_gain_evaluated"] = True
                        records.append(record)
                        last_working = record
                        # Continue the physical ramp from TD.  This gain point
                        # is not an HB root and must not become a continuation
                        # anchor merely because gain evaluation succeeded.
                        td_anchor = td
                        td_restart_disabled = True
                        continue
                    record = self._td_period1_fallback_record(target, td, started)
                    records.append(record)
                    last_working = record
                    td_anchor = td
                    td_restart_disabled = True
                    continue
                if td.classification in {
                    TDClass.PERSISTENT_PERIOD_N,
                    TDClass.PERSISTENT_NONPERIODIC,
                    TDClass.RUNNING_PHASE,
                    TDClass.UNRESOLVED_SLOW_RELAXATION,
                }:
                    if reaches_junction_break(td):
                        record = self._td_record(
                            target, td, ColumnState.PHYSICAL_BOUNDARY_FOUND, started,
                        )
                        record.metadata["junction_break_confirmed"] = True
                        record.metadata["junction_break_threshold"] = (
                            self.budget.junction_break_threshold
                        )
                        records.append(record)
                        first_outside = record
                        return self._result(
                            ColumnState.PHYSICAL_BOUNDARY_FOUND, records,
                            last_working, first_outside, td_bridges, td_periods, restarts,
                        )
                    if self.budget.continue_nonperiodic:
                        record = self._td_record(
                            target, td, ColumnState.TD_CONTINUE, started,
                            SolverRoute.TD_CONTINUE,
                        )
                        record.metadata["physical_boundary_not_declared"] = True
                        record.metadata["td_continuation_anchor"] = True
                        records.append(record)
                        td_anchor = td
                        td_restart_disabled = True
                        nonperiodic_continued = True
                        continue
                if td.classification in {
                    TDClass.PERSISTENT_PERIOD_N,
                    TDClass.PERSISTENT_NONPERIODIC,
                    TDClass.RUNNING_PHASE,
                }:
                    record = self._td_record(
                        target, td, ColumnState.PHYSICAL_BOUNDARY_FOUND, started,
                    )
                    records.append(record)
                    first_outside = record
                    return self._result(
                        ColumnState.PHYSICAL_BOUNDARY_FOUND, records,
                        last_working, first_outside, td_bridges, td_periods, restarts,
                    )
                state = (
                    ColumnState.COLUMN_NUMERICAL_FAILURE
                    if td.classification == TDClass.TRANSIENT_NUMERICAL_FAILURE
                    else ColumnState.COLUMN_UNRESOLVED_BUDGET
                )
                record = self._td_record(target, td, state, started)
                records.append(record)
                return self._result(
                    state, records, last_working, first_outside,
                    td_bridges, td_periods, restarts,
                )

            hb = self.periodic.solve_direct(target, previous)
            if hb.success:
                record = self._hb_record(target, hb, SolverRoute.DIRECT_HB, started)
                records.append(record)
                previous = hb
                last_working = record
                continue

            if previous is None:
                record = self._failure_record(
                    target, ColumnState.COLUMN_NUMERICAL_FAILURE,
                    SolverRoute.UNRESOLVED, hb.reason, started,
                )
                records.append(record)
                return self._result(
                    ColumnState.COLUMN_NUMERICAL_FAILURE, records,
                    last_working, first_outside, td_bridges, td_periods, restarts,
                )

            recovered = self.periodic.recover(target, previous)
            if recovered.success:
                record = self._hb_record(
                    target, recovered, recovered.route, started,
                )
                records.append(record)
                previous = recovered
                last_working = record
                continue

            if td_bridges >= self.budget.max_td_bridges:
                record = self._failure_record(
                    target, ColumnState.COLUMN_UNRESOLVED_BUDGET,
                    SolverRoute.UNRESOLVED, "TD bridge budget exhausted", started,
                )
                records.append(record)
                return self._result(
                    ColumnState.COLUMN_UNRESOLVED_BUDGET, records,
                    last_working, first_outside, td_bridges, td_periods, restarts,
                )

            td_bridges += 1
            td = self.dynamic.bridge(
                previous, target, self.output_dir / f"td_bridge_{td_bridges:02d}",
            )
            td_periods += td.periods
            td = extend_slow_relaxation(td, target, td_bridges)
            if td.classification == TDClass.PERIOD_1:
                if td.restart_seed:
                    restarted = self.periodic.restart_from_td(target, td.restart_seed)
                    if restarted.success:
                        restarts += 1
                        record = self._hb_record(
                            target, restarted, SolverRoute.TD_TO_HB_RESTART, started,
                        )
                        record.metadata["td_classification"] = td.classification.value
                        records.append(record)
                        previous = restarted
                        last_working = record
                        continue
                evaluated = try_td_period1_gain(target, td)
                if evaluated is not None and evaluated.success:
                    record = self._hb_record(
                        target, evaluated, SolverRoute.TD_PERIOD1_GAIN, started,
                    )
                    record.metadata["td_classification"] = td.classification.value
                    record.metadata["hb_restart_available"] = False
                    record.metadata["td_period1_gain_evaluated"] = True
                    records.append(record)
                    last_working = record
                    td_anchor = td
                    td_restart_disabled = True
                    continue
                record = self._td_period1_fallback_record(target, td, started)
                records.append(record)
                last_working = record
                td_anchor = td
                td_restart_disabled = True
                continue

            if td.classification in {
                TDClass.PERSISTENT_PERIOD_N,
                TDClass.PERSISTENT_NONPERIODIC,
                TDClass.RUNNING_PHASE,
                TDClass.UNRESOLVED_SLOW_RELAXATION,
            }:
                if reaches_junction_break(td):
                    record = self._td_record(
                        target, td, ColumnState.PHYSICAL_BOUNDARY_FOUND, started,
                    )
                    record.metadata["junction_break_confirmed"] = True
                    record.metadata["junction_break_threshold"] = (
                        self.budget.junction_break_threshold
                    )
                    records.append(record)
                    first_outside = record
                    return self._result(
                        ColumnState.PHYSICAL_BOUNDARY_FOUND, records,
                        last_working, first_outside, td_bridges, td_periods, restarts,
                    )
                if self.budget.continue_nonperiodic:
                    record = self._td_record(
                        target, td, ColumnState.TD_CONTINUE, started,
                        SolverRoute.TD_CONTINUE,
                    )
                    record.metadata["physical_boundary_not_declared"] = True
                    record.metadata["td_continuation_anchor"] = True
                    records.append(record)
                    td_anchor = td
                    td_restart_disabled = True
                    nonperiodic_continued = True
                    continue
            if td.classification in {
                TDClass.PERSISTENT_PERIOD_N,
                TDClass.PERSISTENT_NONPERIODIC,
                TDClass.RUNNING_PHASE,
            }:
                record = self._td_record(
                    target, td, ColumnState.PHYSICAL_BOUNDARY_FOUND, started,
                )
                records.append(record)
                first_outside = record
                return self._result(
                    ColumnState.PHYSICAL_BOUNDARY_FOUND, records,
                    last_working, first_outside, td_bridges, td_periods, restarts,
                )

            state = (
                ColumnState.COLUMN_NUMERICAL_FAILURE
                if td.classification == TDClass.TRANSIENT_NUMERICAL_FAILURE
                else ColumnState.COLUMN_UNRESOLVED_BUDGET
            )
            record = self._td_record(target, td, state, started)
            records.append(record)
            return self._result(
                state, records, last_working, first_outside,
                td_bridges, td_periods, restarts,
            )

        return self._result(
            (ColumnState.COLUMN_UNRESOLVED_BUDGET
             if nonperiodic_continued else ColumnState.COLUMN_COMPLETE_NO_BOUNDARY), records,
            last_working, first_outside, td_bridges, td_periods, restarts,
        )

    @staticmethod
    def _hb_record(
        target: Any, result: HBResult, route: SolverRoute, started: float,
    ) -> ColumnRecord:
        return ColumnRecord(
            target=target, state=ColumnState.HB_FAST, route=route,
            classification=TDClass.PERIOD_1.value,
            hb_residual_rel=result.residual_rel,
            runtime_s=time.perf_counter() - started,
            metadata=dict(result.metadata),
        )

    @staticmethod
    def _td_record(
        target: Any, result: TDResult, state: ColumnState, started: float,
        route: SolverRoute = SolverRoute.TD_BRIDGE,
    ) -> ColumnRecord:
        return ColumnRecord(
            target=target, state=state, route=route,
            classification=result.classification.value, d1=result.d1,
            best_low_order_dn=result.best_low_order_dn, td_periods=result.periods,
            r_j=result.r_j, phase_winding=result.phase_winding,
            runtime_s=time.perf_counter() - started, metadata=dict(result.metadata),
        )

    @classmethod
    def _td_period1_fallback_record(
        cls, target: Any, result: TDResult, started: float,
    ) -> ColumnRecord:
        record = cls._td_record(
            target, result, ColumnState.TD_CONTINUE, started,
            SolverRoute.TD_PERIOD1_HB_RESTART_FAILED_FALLBACK_TD,
        )
        record.metadata["td_classification"] = result.classification.value
        record.metadata["hb_restart_available"] = False
        record.metadata["physical_state_trusted"] = True
        return record

    @staticmethod
    def _failure_record(
        target: Any, state: ColumnState, route: SolverRoute,
        reason: str | None, started: float,
    ) -> ColumnRecord:
        return ColumnRecord(
            target=target, state=state, route=route,
            classification=state.value, reason=reason,
            runtime_s=time.perf_counter() - started,
        )

    @staticmethod
    def _result(
        status: ColumnState, records: list[ColumnRecord],
        last_working: ColumnRecord | None, first_outside: ColumnRecord | None,
        td_bridges: int, td_periods: int, restarts: int,
    ) -> ColumnResult:
        return ColumnResult(
            status=status, records=records, last_working=last_working,
            first_outside=first_outside, td_bridges=td_bridges,
            td_periods=td_periods, hb_restart_successes=restarts,
        )
