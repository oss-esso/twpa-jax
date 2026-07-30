from __future__ import annotations

import csv

import pytest

import scripts.multitone_convergence_study as study
from scripts.multitone_convergence_study import (
    ConvergenceSetting,
    SettingBudgetExceeded,
    _settings,
    convergence_settings,
    solve_jpa_p1db,
)


def test_q2_to_q3_p1db_converges() -> None:
    q2 = solve_jpa_p1db(
        ConvergenceSetting("lattice", 2, "odd", 3, 1)
    )
    q3 = solve_jpa_p1db(
        ConvergenceSetting("lattice", 3, "odd", 3, 1)
    )
    assert abs(q3.p1db_dbm - q2.p1db_dbm) < 0.2
    assert q2.p1db_method == "refined"
    assert q3.p1db_method == "refined"


def test_convergence_matrix_includes_required_basis_comparisons() -> None:
    settings = convergence_settings()
    assert ConvergenceSetting("three_tone", 1, "odd", 1, 1) in settings
    assert ConvergenceSetting("lattice", 3, "dense", 5, 1) in settings
    assert {setting.signal_order_max for setting in settings[:3]} == {1, 2, 3}


def test_multitone_solves_use_the_cached_coupled_preconditioner() -> None:
    """The study must not refactor from scratch every Newton step.

    Q=3 was unaffordable because every multitone solve re-ran a full splu.
    The pump solve stays on the plain assembly so its iterate path matches
    production exactly.
    """
    assert _settings("real_coupled_fast").preconditioner == "real_coupled_fast"
    assert _settings().preconditioner == "real_coupled"


def test_a_setting_over_budget_raises_rather_than_running_on() -> None:
    """A budget that is already spent must stop before the sweep starts."""
    with pytest.raises(SettingBudgetExceeded):
        solve_jpa_p1db(
            ConvergenceSetting("lattice", 1, "odd", 1, 1), budget_s=1.0e-9
        )


def test_csv_survives_settings_that_never_complete(tmp_path, monkeypatch) -> None:
    """Rows already paid for must reach disk.

    The study previously buffered every result and wrote once at the end, so
    a matrix that ran out of budget produced no CSV at all -- which is exactly
    what happened on the first real-device attempt.
    """
    def always_over_budget(setting, **kwargs):
        raise SettingBudgetExceeded("synthetic budget stop")

    monkeypatch.setattr(study, "solve_jpa_p1db", always_over_budget)
    output = tmp_path / "convergence.csv"

    with pytest.raises(RuntimeError, match="no convergence setting completed"):
        study.main(["--output", str(output), "--device", "jtwpa"])

    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == len(convergence_settings())
    assert {row["status"] for row in rows} == {"TIMEOUT"}
    assert all(row["message"] == "synthetic budget stop" for row in rows)
