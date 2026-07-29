"""Run frequency-resolved P1dB campaigns for distributed devices."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from exp20_multitone_compression import CASES, Case


BANDS = {"jtwpa": (5.8, 7.2), "fqjtwpa": (6.6, 8.0), "2c": (6.8, 7.5)}


def command(
    case: Case,
    outdir: Path,
    frequencies: int,
    workers: int,
    resource_budget_gb: float,
) -> list[str]:
    low, high = BANDS[case.name]
    cmd = [
        sys.executable, "scripts/run_compression.py",
        "--resource-budget-gb", str(resource_budget_gb),
        "--output-dir", str(outdir), *case.source,
        "--pump-freq-ghz", str(case.pump_ghz),
        "--pump-current-a", str(case.pump_current_a),
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "positive_odd_jc", "--pump-mode-count", "10", "--pump-nt", "40",
        "--signal-ghz-min", str(low), "--signal-ghz-max", str(high),
        "--n-signal-freq", str(frequencies), "--signal-workers", str(workers),
        "--source-port", "1", "--pump-port", str(case.pump_port),
        "--out-port", str(case.out_port),
        "--n-signal-power", "25", "--signal-current-min-a", "1e-12",
        "--signal-current-max-a", str(case.signal_max_a), "--attenuation-db", "0",
        "--multitone-basis", "matched", "--multitone-sidebands", str(case.selected_sidebands), "--recovery", "ladder",
    ]
    if case.pump_dir:
        cmd.extend(("--pump-solution-dir", case.pump_dir))
    if case.name == "2c":
        cmd.extend(("--signal-continuation-deadline-s", "600"))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp21_p1db_vs_frequency"))
    parser.add_argument("--only", choices=sorted(BANDS), action="append")
    parser.add_argument("--n-signal-freq", type=int, default=10)
    parser.add_argument("--signal-workers", type=int, default=4)
    parser.add_argument(
        "--resource-budget-gb",
        type=float,
        default=8.0,
        help=(
            "Memory budget the driver divides by the measured per-worker peak "
            "to cap concurrency. At S=10 a worker peaks near 3.1 GB, so the "
            "8.0 default allows 2; raise it only alongside actually-free RAM, "
            "which is capped against independently."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for case in CASES:
        if case.name not in BANDS or (args.only and case.name not in args.only):
            continue
        cmd = command(
            case,
            args.output_dir / case.name,
            args.n_signal_freq,
            args.signal_workers,
            args.resource_budget_gb,
        )
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
