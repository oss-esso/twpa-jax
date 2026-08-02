"""Mark archived exp28 all-port depletion fields as pre-repair/untrusted."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = (
    ROOT / "outputs/exp28_controlled_pump_sweep",
    ROOT / "outputs/exp28_2c_termination_43ohm_pump_sweep",
)
STATUS = "UNTRUSTED_PRE_EXP29_TRACK1"


def main() -> None:
    for campaign in CAMPAIGNS:
        for path in campaign.rglob("compression_summary.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["p1db_pump_depletion_all_port_db_status"] = STATUS
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for path in campaign.rglob("compression_points.csv"):
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0]) if rows else []
            if "p1db_pump_depletion_all_port_db_status" not in fields:
                fields.append("p1db_pump_depletion_all_port_db_status")
            for row in rows:
                row["p1db_pump_depletion_all_port_db_status"] = STATUS
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    print(f"marked archived exp28 artifacts as {STATUS}")


if __name__ == "__main__":
    main()
