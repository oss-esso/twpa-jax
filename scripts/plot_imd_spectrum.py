"""Plot retained intermodulation products from a compression CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-dbc", type=float, default=None)
    return parser


def plot_imd(csv_path: Path, output: Path, floor_dbc: float | None = None) -> None:
    frame = pd.read_csv(csv_path)
    columns = [column for column in frame.columns if re.fullmatch(r"imd_o\d+_m\d+n\d+_dbc", column)]
    if not columns:
        raise ValueError(f"no IMD columns found in {csv_path}")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    colours = {3: "tab:blue", 5: "tab:orange", 7: "tab:green", 9: "tab:red"}
    styles = ("-", "--", ":", "-.")
    for index, column in enumerate(sorted(columns)):
        match = re.fullmatch(r"imd_o(\d+)_m(\d+)n(\d+)_dbc", column)
        assert match is not None
        order = int(match.group(1))
        axis.plot(
            frame["signal_power_dbm"], frame[column],
            label=f"IM{order} ({match.group(2)},{match.group(3)})",
            color=colours.get(order), linestyle=styles[index % len(styles)],
        )
    finite_values = frame[columns].to_numpy(dtype=float)
    if floor_dbc is None:
        floor_dbc = float(pd.DataFrame(finite_values).min().min())
    axis.axhspan(floor_dbc - 3.0, floor_dbc + 3.0, color="0.7", alpha=0.25, label="G3 floor band")
    axis.axhline(-30.0, color="0.35", linestyle="--", linewidth=0.8)
    axis.axhline(-40.0, color="0.35", linestyle=":", linewidth=0.8)
    axis.set_xlabel("Signal power (dBm)")
    axis.set_ylabel("Intermodulation power (dBc)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_imd(args.csv, args.output, args.floor_dbc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
