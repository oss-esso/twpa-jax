"""Compact, mode-independent map coverage accounting."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def coverage_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return raw and physical-status counts without interpreting failures."""
    items = list(rows)
    valid = {"PASS", "WORKING_PERIOD1_HB", "WORKING_PERIOD1_TD_ONLY"}
    above = {"PHYSICAL_BOUNDARY", "SKIP_AFTER_PHYSICAL_BOUNDARY"}
    numerical = {
        "ERROR", "HB_NUMERICAL_FAILURE", "TD_NUMERICAL_FAILURE",
        "COLUMN_NUMERICAL_FAILURE",
    }
    unresolved = {
        "UNRESOLVED", "TD_UNRESOLVED", "COLUMN_UNRESOLVED_BUDGET",
        "SKIP_AFTER_COLUMN_FAILURE",
    }
    counts = Counter(str(row.get("status", "UNKNOWN")) for row in items)
    physical_boundary_rows = sum(counts[name] for name in above)
    numerical_rows = sum(counts[name] for name in numerical)
    unresolved_rows = sum(counts[name] for name in unresolved)
    eligible = len(items) - physical_boundary_rows
    eligible_valid = sum(counts[name] for name in valid)
    return {
        "requested_point_count": len(items),
        "gain_valid_point_count": sum(counts[name] for name in valid),
        "raw_coverage": (
            sum(counts[name] for name in valid) / len(items) if items else 0.0
        ),
        "confirmed_above_boundary_count": physical_boundary_rows,
        "physically_eligible_point_count": eligible,
        "physically_eligible_gain_valid_count": eligible_valid,
        "physically_eligible_coverage": eligible_valid / eligible if eligible else 0.0,
        "hb_numerical_failure_count": numerical_rows,
        "unresolved_count": unresolved_rows,
        "status_counts": dict(sorted(counts.items())),
    }
