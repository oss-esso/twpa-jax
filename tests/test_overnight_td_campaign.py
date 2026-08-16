from __future__ import annotations

from scripts.run_overnight_7p9_dynamics import (
    HOLD_CHECKPOINTS,
    first_boundary,
    regime,
)


def _record(power: float, state: str) -> dict:
    return {"power_dbm": power, "final": {"regime": state}}


def test_campaign_uses_decay_aware_unresolved_state_before_raw_label() -> None:
    summary = {
        "integrator": {"success": True},
        "classification": "BROADBAND_OR_CHAOTIC",
        "decay_aware": {"class": "RELAXING_TO_PERIOD1"},
    }

    assert regime(summary) == "UNRESOLVED_LONG_TRANSIENT"


def test_campaign_boundary_finds_first_period1_loss() -> None:
    records = [
        _record(-25.0, "PERIOD1"),
        _record(-24.5, "PERIOD1"),
        _record(-24.0, "BROADBAND_NONPERIODIC"),
        _record(-23.0, "PERIOD1"),
    ]

    assert first_boundary(records) == (-24.5, -24.0)


def test_campaign_checkpoint_contract_matches_spec() -> None:
    assert HOLD_CHECKPOINTS == (40, 90, 140, 250, 440)
