"""Build the exp20 cross-device compression and basis-convergence table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from exp20_multitone_compression import CASES


def load_case(root: Path, name: str) -> dict[str, object]:
    case = next(case for case in CASES if case.name == name)
    sidebands = case.selected_sidebands
    production = json.loads(
        (root / name / f"s{sidebands}" / "compression_summary.json").read_text()
    )
    points = list(
        csv.DictReader(
            (root / name / f"s{sidebands}" / "compression_points.csv").open()
        )
    )
    gain = float(production["small_signal_gain_db"])
    floquet_gap = (
        gain - case.floquet_gain_db if case.floquet_gain_db is not None else None
    )
    jc_gap = gain - case.jc_gain_db if case.jc_gain_db is not None else None
    parity_pass = floquet_gap is not None and abs(floquet_gap) < 0.05
    jc_pass = jc_gap is not None and abs(jc_gap) < 0.2
    rungs = Counter(point["recovery_rung"] for point in points)
    return {
        "device": name,
        "status": production["status"],
        "selected_sidebands": sidebands,
        "selection_reason": (
            "same-S Floquet parity <0.05 dB and JC gap <0.2 dB"
            if case.jc_gain_db is not None
            else "NO_JC_REFERENCE; selected from the production map basis"
        ),
        "small_signal_gain_db": gain,
        "floquet_reference_db": case.floquet_gain_db,
        "multitone_minus_floquet_db": floquet_gap,
        "multitone_floquet_parity_pass": parity_pass,
        "jc_reference_db": case.jc_gain_db,
        "small_signal_minus_jc_db": jc_gap,
        "jc_reference_pass": jc_pass,
        "basis_selection_pass": parity_pass and jc_pass,
        "small_signal_gain_vs_off_db": production["small_signal_gain_vs_off_db"],
        "p1db_input_dbm": production["p1db_input_dbm"],
        "p1db_output_dbm": production["p1db_output_dbm"],
        "p1db_pump_depletion_db": production["p1db_pump_depletion_db"],
        "recommended_sidebands": sidebands,
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
