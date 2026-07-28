"""Run corrected-basis compression and S=2/S=4 spot checks on four devices."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    source: tuple[str, str]
    pump_dir: str | None
    pump_ghz: float
    pump_current_a: float
    signal_ghz: float
    out_port: int
    signal_max_a: float


CASES = (
    Case("jpa", ("--fixture", "jpa"), "outputs/exp14_jpa_odd10_scale2/pump", 4.75001, 1.13e-8, 4.75, 1, 3e-8),
    Case("jtwpa", ("--fixture", "jtwpa"), "outputs/exp14_jtwpa_odd10_scale2/pump", 7.12, 3.7e-6, 6.6, 2, 3e-7),
    Case("fqjtwpa", ("--fixture", "fqjtwpa"), "outputs/exp14_fqjtwpa_odd10_scale2/pump", 7.9, 2.2e-6, 7.4, 2, 3e-7),
    Case("2c", ("--circuit-dir", "outputs/ipm_python_design"), None, 7.376811594202222, 6.349059495192881e-6, 7.1768115942022215, 2, 3e-7),
)


def command(case: Case, sidebands: int, outdir: Path, powers: int) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_compression.py",
        "--output-dir", str(outdir),
        *case.source,
        "--pump-freq-ghz", str(case.pump_ghz),
        "--pump-current-a", str(case.pump_current_a),
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", "10",
        "--pump-nt", "40",
        "--signal-ghz", str(case.signal_ghz),
        "--source-port", "1",
        "--out-port", str(case.out_port),
        "--n-signal-power", str(powers),
        "--signal-current-min-a", "1e-12",
        "--signal-current-max-a", str(case.signal_max_a),
        "--attenuation-db", "0",
        "--multitone-basis", "matched",
        "--multitone-sidebands", str(sidebands),
        "--recovery", "ladder",
        "--signal-substep-min-db", "0.01",
        "--save-states", "selected",
    ]
    if case.pump_dir:
        cmd.extend(("--pump-solution-dir", case.pump_dir))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp20_multitone_compression"))
    parser.add_argument("--only", choices=[case.name for case in CASES], action="append")
    parser.add_argument("--n-signal-power", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [case for case in CASES if not args.only or case.name in args.only]
    for case in selected:
        for sidebands in (2, 4):
            powers = args.n_signal_power if sidebands == 2 else max(15, args.n_signal_power)
            cmd = command(case, sidebands, args.output_dir / case.name / f"s{sidebands}", powers)
            print(subprocess.list2cmdline(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
