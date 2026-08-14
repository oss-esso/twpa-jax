"""Run the requested Guarcello transition gap scan and persist each point."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.chaos.run_phaseB_pump_only import _run_point, ROOT
from scripts.chaos.run_phaseB_pump_only import PAPER


def main() -> int:
    root = ROOT / "outputs" / "chaos" / "phaseB" / "guarcello"
    powers = np.linspace(-53.95, -53.40, 8)
    tmax_norm = 600.0 * PAPER.Device().omega_plasma / (7.0e9)
    rows = []
    for index, power in enumerate(powers):
        row = _run_point("guarcello", float(power), root / f"gap_{index:03d}", 0.01, tmax_norm)
        rows.append(row)
        keys = sorted({key for item in rows for key, value in item.items() if not isinstance(value, (list, dict))})
        with (root / "gap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({key: value for key, value in item.items() if not isinstance(value, (list, dict))} for item in rows)
        print(json.dumps({"index": index, "power_dbm": float(power), "verdict": row.get("verdict")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
