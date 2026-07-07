from __future__ import annotations

from twpa_solver_old.experiments.solver_readiness import readiness_rows


def test_solver_readiness_has_single_allowed_class_per_solver() -> None:
    allowed = {
        "PRODUCTION_MAP_SOLVER",
        "VALIDATED_SMALL_TWPA",
        "TOY_ONLY",
        "SCAFFOLD_ONLY",
        "FAILED_OR_WEAK",
    }
    rows = readiness_rows()
    assert rows
    assert all(row.readiness_class in allowed for row in rows)
    scipy = next(row for row in rows if row.solver == "scipy-least-squares")
    assert scipy.readiness_class == "PRODUCTION_MAP_SOLVER"
    assert scipy.actual_twpa_tested
    assert scipy.map_used
