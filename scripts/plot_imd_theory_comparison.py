"""Plot finite-signal two-tone HB IMD against pump-dressed theory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_PRODUCT = re.compile(
    r"imd2_o(?P<order>\d+)_m(?P<m>\d+)n(?P<n>\d+)_(?P<ordering>w1w2|w2w1)_dbc"
)


def _label(column: str, summary: dict[str, object]) -> str:
    match = _PRODUCT.fullmatch(column)
    if match is None:
        return column
    order = int(match.group("order"))
    m = int(match.group("m"))
    n = int(match.group("n"))
    first, second = (
        ("ω₁", "ω₂") if match.group("ordering") == "w1w2" else ("ω₂", "ω₁")
    )
    first_term = first if m == 1 else f"{m}{first}"
    second_term = second if n == 1 else f"{n}{second}"
    return f"IM{order}: {first_term} − {second_term}"


def plot(csv_path: Path, summary_path: Path, output: Path) -> None:
    frame = pd.read_csv(csv_path).sort_values("tone1_power_dbm")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hb_columns = sorted(c for c in frame if _PRODUCT.fullmatch(c))
    if not hb_columns:
        raise ValueError(f"no two-tone IMD columns found in {csv_path}")
    theory_columns = {
        f"theory_{column}": column for column in hb_columns
        if f"theory_{column}" in frame
    }
    x = frame["tone1_power_dbm"]
    orders = sorted({int(_PRODUCT.fullmatch(c).group("order")) for c in hb_columns})
    figure, axes = plt.subplots(
        len(orders), 1, figsize=(10, 4.0 * len(orders)), squeeze=False, sharex=True
    )
    colours = {"w1w2": "tab:blue", "w2w1": "tab:orange"}
    for axis, order in zip(axes[:, 0], orders):
        order_columns = [
            c for c in hb_columns if int(_PRODUCT.fullmatch(c).group("order")) == order
        ]
        for column in order_columns:
            match = _PRODUCT.fullmatch(column)
            assert match is not None
            colour = colours[match.group("ordering")]
            axis.plot(
                x,
                frame[column],
                color=colour,
                marker="o",
                linewidth=1.5,
                label=f"HB {_label(column, summary)}",
            )
            theory_column = f"theory_{column}"
            if theory_column in theory_columns:
                axis.plot(
                    x,
                    frame[theory_column],
                    color=colour,
                    linestyle="--",
                    marker="x",
                    linewidth=1.2,
                    label=f"O({order}) theory {_label(column, summary)}",
                )
        axis.set_ylabel("IMD power (dBc)")
        axis.set_title(f"Two-tone IM{order}: finite-signal HB versus pump-dressed O({order})")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small", ncol=2)
    axes[-1, 0].set_xlabel("Tone-1 per-tone signal power (dBm)")
    residuals = summary.get("imd_theory_residuals", {})
    if residuals:
        figure.suptitle(
            "Exact discrete pump-dressed IMD theory versus finite-signal HB\n"
            + ", ".join(f"{key} residual={float(value):.2e}" for key, value in residuals.items()),
            y=0.995,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        figure.suptitle("Two-tone IMD: finite-signal HB versus pump-dressed theory")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.csv, args.summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
