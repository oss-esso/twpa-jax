"""Plot aggregate diagnostics from HB-only column outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_column(root: Path, label: str) -> list[dict[str, float | str]]:
    rows = list(csv.DictReader((root / "hb_up_to_failure.csv").open(encoding="utf-8")))
    output: list[dict[str, float | str]] = []
    for row in rows:
        if row.get("status") != "PASS":
            continue
        report_path = Path(row["pump_dir"]) / "pump_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["solution_summary"]
        phi0 = float(report["metadata"].get("phi0", report["metadata"].get("phi0_reduced", 3.291059784754533e-16)))
        output.append({
            "device": label,
            "power_dbm": float(row["pump_power_dbm"]),
            "r_j": float(summary["branch_current_max_over_ic_all"]),
            "winding_cycles_per_period": 0.0,
            "min_cos_phi": float(summary["branch_min_cos_phase"]),
            "max_abs_phi_cycles": float(summary["branch_psi_max_abs"]) / phi0 / (2.0 * np.pi),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series", action="append", metavar="LABEL=PATH",
        help="HB column to include; repeat for multiple series.",
    )
    parser.add_argument("--jtwpa", type=Path)
    parser.add_argument("--fqjtwpa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    specs: list[tuple[str, Path]] = []
    if args.series:
        for item in args.series:
            label, separator, path = item.partition("=")
            if not separator or not label or not path:
                parser.error(f"--series must have LABEL=PATH form: {item!r}")
            specs.append((label, Path(path)))
    else:
        if args.jtwpa is None or args.fqjtwpa is None:
            parser.error("provide --series or both --jtwpa and --fqjtwpa")
        specs = [("JTWPA", args.jtwpa), ("FQJTWPA", args.fqjtwpa)]
    records = [record for label, path in specs for record in load_column(path, label)]
    if not records:
        parser.error("none of the requested columns contain converged HB points")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    markers = ("o", "s", "^", "D", "v", "P", "X", "<")
    for index, (label, _path) in enumerate(specs):
        data = [item for item in records if item["device"] == label]
        if not data:
            continue
        power = np.asarray([item["power_dbm"] for item in data], dtype=float)
        marker = markers[index % len(markers)]
        color = f"C{index % 10}"
        axes[0].plot(power, [item["r_j"] for item in data], marker + "-", color=color, label=label)
        axes[1].plot(power, [item["winding_cycles_per_period"] for item in data], marker + "-", color=color, label=label)
        axes[2].plot(power, [item["min_cos_phi"] for item in data], marker + "-", color=color, label=label)

    axes[0].set_ylabel(r"$r_J = \max |I_J/I_c|$")
    axes[0].set_title("HB-only Josephson current ratio")
    axes[1].set_ylabel("winding (cycles / pump period)")
    axes[1].set_title("HB phase winding: zero by construction")
    axes[1].text(0.01, 0.80, "Periodic HB orbit; no net phase winding", transform=axes[1].transAxes)
    axes[2].set_ylabel(r"$\min\cos(\phi_J)$")
    axes[2].set_xlabel("pump power (dBm)")
    axes[2].set_title("HB Josephson tangent margin")
    axes[2].axhline(0.0, color="k", linestyle="--", linewidth=0.8, label="zero margin")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("HB column diagnostics across pump-harmonic bases")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"wrote_plot={args.output}")
    print(f"wrote_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
