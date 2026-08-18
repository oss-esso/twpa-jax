from __future__ import annotations

from scripts.report_transition_boundary import summarize_transition


def test_transition_report_does_not_invent_missing_crossing() -> None:
    result = summarize_transition(
        "fixture",
        {"points": [{"multiplier": {"magnitude": 0.9}}]},
        {"points": []},
    )

    assert result["status"] == "NOT_ESTABLISHED"
    assert result["ns_crossing"] is None


def test_transition_report_records_solved_torus_skeleton() -> None:
    result = summarize_transition(
        "fixture",
        {"points": [{"multiplier": {"magnitude": 1.01}}]},
        {
            "points": [
                {"converged": True, "point_index": 2, "off_comb_norm_fraction": 0.1},
                {"converged": False, "point_index": 3, "off_comb_norm_fraction": 0.2},
            ]
        },
    )

    assert result["status"] == "ESTABLISHED"
    assert result["torus_solved_count"] == 1
    assert result["torus_last_parameter"] == 2
