"""Plot single-tone regression differences and two-tone basis convergence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_SINGLE_RE = re.compile(r"imd_o(?P<order>\d+)_m(?P<m>\d+)n(?P<n>\d+)_dbc")
_TWO_RE = re.compile(r"imd2_o(?P<order>\d+)_m(?P<m>\d+)n(?P<n>\d+)_(?P<ordering>w1w2|w2w1)_dbc")


def _labels(summary_path: Path) -> dict[str, str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    for product in summary.get("imd_products", []):
        label = str(product["label"])
        if product.get("family") == "two_tone":
            ordering = str(product.get("ordering"))
            first, second = ("2", "1") if ordering == "w2_minus_w1" else ("1", "2")
            expression = f"{product['m']}ω{first} − {product['n']}ω{second}"
        else:
            expression = f"{product['m']}ωs − {product['n']}ωp"
        labels[f"{label}_dbc"] = expression
    return labels


def _product_columns(frame: pd.DataFrame, pattern: re.Pattern[str]) -> list[str]:
    return sorted(column for column in frame.columns if pattern.fullmatch(column))


def _save_single_comparison(reference: Path, current: Path, output_dir: Path) -> None:
    old = pd.read_csv(reference / "compression_points.csv")
    new = pd.read_csv(current / "compression_points.csv")
    columns = [column for column in _product_columns(old, _SINGLE_RE) if column in new]
    if not columns:
        raise ValueError("no common single-tone IMD products")
    x_old = old["signal_power_dbm"].to_numpy(float)
    x_new = new["signal_power_dbm"].to_numpy(float)
    if not np.array_equal(x_old, x_new):
        raise ValueError("single-tone comparison grids differ")
    labels = _labels(reference / "compression_summary.json")

    figure, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    for axis, column in zip(axes.flat, columns):
        axis.plot(x_old, old[column], color="0.35", linestyle="--", label="saved fixture")
        axis.plot(x_new, new[column], color="tab:blue", label="current path")
        delta = new[column].to_numpy(float) - old[column].to_numpy(float)
        finite = np.isfinite(delta)
        max_delta = float(np.max(np.abs(delta[finite]))) if np.any(finite) else float("nan")
        axis.set_title(f"{labels.get(column, column)}\nmax |Δ| = {max_delta:.3g} dB")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize="small")
    for axis in axes[-1, :]:
        axis.set_xlabel("Signal power (dBm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Product (dBc)")
    figure.suptitle("Single-tone IMD: saved fixture versus current path", y=0.995)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "single_tone_product_overlay.png", dpi=180)
    figure.savefig(output_dir / "single_tone_product_overlay.svg")
    plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    for axis, column in zip(axes.flat, columns):
        delta = new[column].to_numpy(float) - old[column].to_numpy(float)
        axis.plot(x_old, delta, color="tab:red")
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_title(labels.get(column, column))
        axis.grid(True, alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Signal power (dBm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Current − saved (dB)")
    figure.suptitle("Single-tone IMD divergence", y=0.995)
    figure.tight_layout()
    figure.savefig(output_dir / "single_tone_product_delta.png", dpi=180)
    figure.savefig(output_dir / "single_tone_product_delta.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x_old, old["gain_vs_off_db"], "--", color="0.35", label="saved fixture")
    axis.plot(x_new, new["gain_vs_off_db"], color="tab:blue", label="current path")
    axis.set_xlabel("Signal power (dBm)")
    axis.set_ylabel("Gain versus pump-off (dB)\n(signal tone)")
    axis.set_title("Single-tone signal-tone response")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "single_tone_signal_tone.png", dpi=180)
    figure.savefig(output_dir / "single_tone_signal_tone.svg")
    plt.close(figure)


def _save_two_tone_convergence(low: Path, high: Path, output_dir: Path) -> None:
    low_frame = pd.read_csv(low / "compression_points.csv")
    high_frame = pd.read_csv(high / "compression_points.csv")
    columns = [column for column in _product_columns(low_frame, _TWO_RE) if column in high_frame]
    if not columns:
        raise ValueError("no common two-tone IMD products")
    x = low_frame["tone1_power_dbm"].to_numpy(float)
    if not np.array_equal(x, high_frame["tone1_power_dbm"].to_numpy(float)):
        raise ValueError("two-tone convergence grids differ")
    labels = _labels(low / "compression_summary.json")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, column in zip(axes.flat, columns):
        axis.plot(x, low_frame[column], "--", color="tab:orange", label="lattice S=11")
        axis.plot(x, high_frame[column], color="tab:green", label="lattice S=13")
        delta = high_frame[column].to_numpy(float) - low_frame[column].to_numpy(float)
        finite = np.isfinite(delta)
        max_delta = float(np.max(np.abs(delta[finite]))) if np.any(finite) else float("nan")
        axis.set_title(f"{labels.get(column, column)}\nmax |Δ| = {max_delta:.3g} dB")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize="small")
    for axis in axes[-1, :]:
        axis.set_xlabel("Tone-1 per-tone signal power (dBm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Product (dBc)")
    figure.suptitle("Two-tone IMD basis convergence", y=0.995)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "two_tone_product_convergence.png", dpi=180)
    figure.savefig(output_dir / "two_tone_product_convergence.svg")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-reference", type=Path, required=True)
    parser.add_argument("--single-current", type=Path, required=True)
    parser.add_argument("--two-tone-low", type=Path, required=True)
    parser.add_argument("--two-tone-high", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _save_single_comparison(args.single_reference, args.single_current, args.output_dir)
    _save_two_tone_convergence(args.two_tone_low, args.two_tone_high, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
