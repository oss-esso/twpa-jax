"""Run and plot distributed spatial phase-mismatch/depletion attribution."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from exp20_multitone_compression import CASES, Case


def run_command(case: Case, outdir: Path) -> list[str]:
    cmd = [
        sys.executable, "scripts/run_compression.py", "--output-dir", str(outdir), *case.source,
        "--pump-freq-ghz", str(case.pump_ghz), "--pump-current-a", str(case.pump_current_a),
        "--pump-current-jc-scale", "1.0", "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", "10", "--pump-nt", "40", "--signal-ghz", str(case.signal_ghz),
        "--source-port", "1", "--pump-port", str(case.pump_port),
        "--out-port", str(case.out_port), "--n-signal-power", "25",
        "--signal-current-min-a", "1e-12", "--signal-current-max-a", str(case.signal_max_a),
        "--attenuation-db", "0", "--multitone-basis", "matched", "--multitone-sidebands", str(case.selected_sidebands),
        "--recovery", "ladder", "--spatial-profiles", "--save-states", "selected",
    ]
    if case.pump_dir:
        cmd.extend(("--pump-solution-dir", case.pump_dir))
    if case.name == "2c":
        cmd.extend(("--signal-continuation-deadline-s", "600"))
    return cmd


def plot(outdir: Path) -> None:
    spatial = list(csv.DictReader((outdir / "spatial_profiles.csv").open()))
    points = list(csv.DictReader((outdir / "compression_points.csv").open()))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for label in ("zero_signal", "mid", "p1db"):
        rows = [row for row in spatial if row["operating_point"] == label]
        if not rows:
            continue
        x = [int(row["branch_index"]) for row in rows]
        axes[0, 0].plot(x, [float(row["theta_rad"]) for row in rows], label=label)
        axes[0, 1].plot(x, [float(row["delta_k_eff_rad_per_cell"]) for row in rows], label=label)
        axes[1, 0].plot(x, [float(row["signal_flux_abs"]) for row in rows], label=label)
    power = [float(row["signal_power_dbm"]) for row in points]
    gain = [float(row["gain_db"]) for row in points]
    depletion = [float(row["pump_depletion_db"]) for row in points]
    depletion_only = [gain[0] + 2.0 * value for value in depletion]
    axes[1, 1].plot(power, gain, "o-", label="multitone")
    axes[1, 1].plot(power, depletion_only, "--", label="depletion-only")
    axes[0, 0].set_ylabel("Theta (rad)")
    axes[0, 1].set_ylabel("Delta k eff (rad/cell)")
    axes[1, 0].set_ylabel("Signal branch flux magnitude")
    axes[1, 1].set_ylabel("Gain (dB)")
    axes[1, 1].set_xlabel("Signal input (dBm)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(outdir / "spatial_attribution.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp22_spatial_attribution"))
    parser.add_argument("--only", choices=("jtwpa", "2c"), action="append")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for case in CASES:
        if case.name not in {"jtwpa", "2c"} or (args.only and case.name not in args.only):
            continue
        outdir = args.output_dir / case.name
        cmd = run_command(case, outdir)
        if not args.plot_only:
            print(subprocess.list2cmdline(cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)
        plot(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
