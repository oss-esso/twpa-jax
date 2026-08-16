"""Plot retained intermodulation products from a compression CSV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-dbc", type=float, default=None)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Compression summary JSON; defaults to compression_summary.json beside the CSV.",
    )
    parser.add_argument(
        "--signal-attenuation-db",
        type=float,
        default=0.0,
        help="Signal-line attenuation to show the on-chip power on a top axis.",
    )
    return parser


def _omega_expression(m: int, n: int, conjugated: bool) -> str:
    """Format the positive-frequency expression for ``m*ws - n*wp``."""
    if conjugated:
        m, n = n, m
        first, second = "ωp", "ωs"
    else:
        first, second = "ωs", "ωp"
    first_term = first if m == 1 else f"{m}{first}"
    second_term = second if n == 1 else f"{n}{second}"
    return f"{first_term} − {second_term}"


def _two_tone_expression(m: int, n: int, ordering: str | None) -> str:
    """Format the two-tone expression for a retained IMD product."""
    first, second = (
        ("omega1", "omega2")
        if ordering == "w1_minus_w2"
        else ("omega2", "omega1")
    )
    first_term = first if m == 1 else f"{m}{first}"
    second_term = second if n == 1 else f"{n}{second}"
    return f"{first_term} - {second_term}"


def _product_labels(csv_path: Path, summary_path: Path | None) -> dict[str, str]:
    """Return legend labels keyed by IMD CSV column name."""
    if summary_path is None:
        candidate = csv_path.with_name("compression_summary.json")
        summary_path = candidate if candidate.exists() else None
    if summary_path is None:
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for product in summary.get("imd_products", []):
        label = str(product["label"])
        if product.get("family") == "two_tone":
            m = int(product["m"])
            n = int(product["n"])
            ordering = product.get("ordering")
            if product.get("conjugated"):
                m, n = n, m
                ordering = (
                    "w2_minus_w1"
                    if ordering == "w1_minus_w2"
                    else "w1_minus_w2"
                )
            expression = _two_tone_expression(m, n, ordering)
        else:
            expression = _omega_expression(
                int(product["m"]), int(product["n"]), bool(product["conjugated"])
            )
        labels[f"{label}_dbc"] = expression
    return labels


def plot_imd(
    csv_path: Path,
    output: Path,
    floor_dbc: float | None = None,
    signal_attenuation_db: float = 0.0,
    summary_path: Path | None = None,
) -> None:
    frame = pd.read_csv(csv_path)
    columns = [
        column
        for column in frame.columns
        if re.fullmatch(r"imd_o\d+_m\d+n\d+_dbc", column)
        or re.fullmatch(r"imd2_o\d+_m\d+n\d+_(?:w1w2|w2w1)_dbc", column)
    ]
    if not columns:
        raise ValueError(f"no IMD columns found in {csv_path}")
    fundamental_columns = [
        column
        for column in ("tone1_s21_db", "tone2_s21_db")
        if column in frame.columns and frame[column].notna().any()
    ]
    labels = _product_labels(csv_path, summary_path)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    colours = {3: "tab:blue", 5: "tab:orange", 7: "tab:green", 9: "tab:red"}
    styles = ("-", "--", ":", "-.")
    for index, column in enumerate(sorted(columns)):
        match = re.fullmatch(r"imd_o(\d+)_m(\d+)n(\d+)_dbc", column)
        two_match = re.fullmatch(
            r"imd2_o(\d+)_m(\d+)n(\d+)_(w1w2|w2w1)_dbc", column
        )
        match_data = match if match is not None else two_match
        assert match_data is not None
        order = int(match_data.group(1))
        axis.plot(
            frame["signal_power_dbm"], frame[column],
            label=labels.get(
                column,
                f"IM{order} ({match_data.group(2)},{match_data.group(3)})",
            ),
            color=colours.get(order),
            linestyle=styles[index % len(styles)],
            marker="o" if len(frame) == 1 else None,
        )
    fundamental_labels = {
        "tone1_s21_db": "w1 fundamental (dB rel. tone-1 input)",
        "tone2_s21_db": "w2 fundamental (dB rel. tone-2 input)",
    }
    fundamental_styles = {
        "tone1_s21_db": ("black", "-"),
        "tone2_s21_db": ("purple", "--"),
    }
    for column in fundamental_columns:
        colour, linestyle = fundamental_styles[column]
        axis.plot(
            frame["signal_power_dbm"],
            frame[column],
            color=colour,
            linestyle=linestyle,
            linewidth=1.4,
            label=fundamental_labels[column],
        )
    finite_values = frame[columns].to_numpy(dtype=float)
    if floor_dbc is None:
        floor_dbc = float(pd.DataFrame(finite_values).min().min())
    axis.axhspan(floor_dbc - 3.0, floor_dbc + 3.0, color="0.7", alpha=0.25, label="G3 floor band")
    axis.axhline(-30.0, color="0.35", linestyle="--", linewidth=0.8)
    axis.axhline(-40.0, color="0.35", linestyle=":", linewidth=0.8)
    axis.set_xlabel(
        "Tone-1 per-tone signal power (dBm)"
        if any(column.startswith("imd2_") for column in columns)
        else "Source / instrument signal power (dBm)"
    )
    if signal_attenuation_db:
        on_chip_axis = axis.secondary_xaxis(
            "top",
            functions=(
                lambda source_dbm: source_dbm - signal_attenuation_db,
                lambda on_chip_dbm: on_chip_dbm + signal_attenuation_db,
            ),
        )
        on_chip_axis.set_xlabel(
            f"On-chip signal power (dBm; source − {signal_attenuation_db:.3f} dB)"
        )
    axis.set_ylabel("Intermodulation power (dBc)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_imd(
        args.csv,
        args.output,
        args.floor_dbc,
        args.signal_attenuation_db,
        args.summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
