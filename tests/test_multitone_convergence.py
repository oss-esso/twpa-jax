from __future__ import annotations

from scripts.multitone_convergence_study import (
    ConvergenceSetting,
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
