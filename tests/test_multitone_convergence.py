from __future__ import annotations

from scripts.multitone_convergence_study import (
    ConvergenceSetting,
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
