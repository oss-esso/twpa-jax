"""Plot tracked Floquet growth rates from a physical-column branch CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse plotting arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Write the growth-rate trajectory plot."""
    trajectories: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with args.branches_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trajectories[int(row["candidate_index"])].append(
                (float(row["drive_dbm"]), float(row["growth_rate_per_s"]) / 1.0e8)
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for index, points in sorted(trajectories.items()):
        points.sort()
        x_values, y_values = zip(*points)
        axis.plot(x_values, y_values, marker="o", label=f"branch {index}")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Requested pump drive (dBm)")
    axis.set_ylabel(r"Floquet growth rate ($10^8$ s$^{-1}$)")
    axis.set_title("Tracked Floquet branches")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Create the plot."""
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
