"""Build the exp20 cross-device compression and basis-convergence table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def load_case(root: Path, name: str) -> dict[str, object]:
    summaries = {
        sidebands: json.loads((root / name / f"s{sidebands}" / "compression_summary.json").read_text())
        for sidebands in (2, 4)
    }
    points = list(csv.DictReader((root / name / "s2" / "compression_points.csv").open()))
    p2 = summaries[2].get("p1db_signal_current_a")
    p4 = summaries[4].get("p1db_signal_current_a")
    delta = (
        20.0 * math.log10(float(p4) / float(p2))
        if p2 is not None and p4 is not None
        else None
    )
    basis_flag = delta is None or abs(delta) > 0.2
    production = summaries[4] if basis_flag else summaries[2]
    rungs = Counter(point["recovery_rung"] for point in points)
    return {
        "device": name,
        "status": summaries[2]["status"],
        "small_signal_gain_db": summaries[2]["small_signal_gain_db"],
        "small_signal_gain_vs_off_db": summaries[2]["small_signal_gain_vs_off_db"],
        "p1db_input_dbm": summaries[2]["p1db_input_dbm"],
        "p1db_output_dbm": summaries[2]["p1db_output_dbm"],
        "p1db_pump_depletion_db": summaries[2]["p1db_pump_depletion_db"],
        "delta_p1db_s4_minus_s2_db": delta,
        "basis_spotcheck_flag": basis_flag,
        "recommended_sidebands": 4 if basis_flag else 2,
        "production_p1db_input_dbm": production["p1db_input_dbm"],
        "production_p1db_output_dbm": production["p1db_output_dbm"],
        "production_p1db_pump_depletion_db": production["p1db_pump_depletion_db"],
        "max_abs_pump_depletion_db": max(abs(float(point["pump_depletion_db"])) for point in points if point["pump_depletion_db"] != "nan"),
        "recovery_rungs": json.dumps(rungs, sort_keys=True),
        "stability_status": "NOT_CHECKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/exp20_multitone_compression"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp20_summary"))
    parser.add_argument(
        "--only",
        choices=("jpa", "jtwpa", "fqjtwpa", "2c"),
        action="append",
    )
    args = parser.parse_args()
    names = args.only or ["jpa", "jtwpa", "fqjtwpa", "2c"]
    rows = [load_case(args.input_dir, name) for name in names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
