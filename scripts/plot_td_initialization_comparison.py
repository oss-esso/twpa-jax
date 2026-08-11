"""Plot long-hold TD diagnostics for the three initialization histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "direct_hb": "direct_hb",
    "warm_hb": "warm_hb",
    "zero_pump": "zero_pump",
}


def plot_one(name: str, run_dir: Path, outdir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    compact = np.load(run_dir / "td_compact.npz")
    ramp_periods = int(summary["ramp_periods"])
    total_periods = ramp_periods + int(summary["hold_periods"])
    periods = compact["theta"] / (2.0 * np.pi)
    strobe = summary["stroboscopic"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(periods, compact["max_abs_sin_phi"], color="#264653", linewidth=1.2)
    axes[0].set_ylabel(r"sampled peak $|I_J|/I_c$")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.25)
    for key, color in (("d1", "#d1495b"), ("d2", "#00798c"), ("d3", "#edae49")):
        strobe_periods = ramp_periods + 1 + np.arange(len(strobe[key]), dtype=float)
        axes[int(key[1])].semilogy(strobe_periods, np.maximum(np.asarray(strobe[key]), 1e-12), color=color, linewidth=1.2)
        axes[int(key[1])].set_ylabel(key)
        axes[int(key[1])].grid(True, alpha=0.25, which="both")
    axes[-1].set_xlabel("total pump periods")
    axes[-1].set_xlim(0.0, float(total_periods))
    fig.suptitle(
        f"{name}: {summary['initialization_source']}\n"
        f"{summary['classification']} / decay-aware {summary['decay_aware']['class']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_peak_and_recurrence_vs_period.png", dpi=160)
    plt.close(fig)


def plot_combined(base: Path, outdir: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex="col")
    for row, (name, folder) in enumerate(RUNS.items()):
        run_dir = base / folder
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        compact = np.load(run_dir / "td_compact.npz")
        ramp_periods = int(summary["ramp_periods"])
        periods = compact["theta"] / (2.0 * np.pi)
        strobe = summary["stroboscopic"]
        axes[row, 0].plot(periods, compact["max_abs_sin_phi"], color="#264653")
        axes[row, 0].set_ylabel(name)
        axes[row, 0].set_ylim(bottom=0.0)
        for col, key, color in ((1, "d1", "#d1495b"), (2, "d2", "#00798c"), (3, "d3", "#edae49")):
            strobe_periods = ramp_periods + 1 + np.arange(len(strobe[key]), dtype=float)
            axes[row, col].semilogy(strobe_periods, np.maximum(np.asarray(strobe[key]), 1e-12), color=color)
            axes[row, col].set_title(key if row == 0 else "")
        for col in range(4):
            axes[row, col].grid(True, alpha=0.25, which="both")
    for col, title in enumerate((r"peak $|I_J|/I_c$", "d1", "d2", "d3")):
        axes[0, col].set_title(title)
        axes[-1, col].set_xlabel("total pump periods")
    fig.suptitle("2c initialization comparison at −24.473684 dBm; 40-period ramp + 250-period hold")
    fig.tight_layout()
    fig.savefig(outdir / "all_initializations_peak_and_recurrence_vs_period.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(".hybrid_outputs/td_compare_7p9_m24p473684_250p"))
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (args.base / "plots")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, folder in RUNS.items():
        plot_one(name, args.base / folder, outdir)
    plot_combined(args.base, outdir)
    print(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
