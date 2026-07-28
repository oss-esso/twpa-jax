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
    selected_sidebands: int
    floquet_gain_db: float | None
    jc_gain_db: float | None


CASES = (
    Case("jpa", ("--fixture", "jpa"), "outputs/exp14_jpa_odd10_scale2/pump", 4.75001, 1.13e-8, 4.75, 1, 3e-8, 2, 13.305111, 13.302331),
    Case("jtwpa", ("--fixture", "jtwpa"), "outputs/exp14_jtwpa_odd10_scale2/pump", 7.12, 3.7e-6, 6.6, 2, 3e-7, 10, 27.541036, 27.5402),
    Case("fqjtwpa", ("--fixture", "fqjtwpa"), "outputs/exp14_fqjtwpa_odd10_scale2/pump", 7.9, 2.2e-6, 7.4, 2, 3e-7, 6, 28.534877, 28.5367),
    Case(
        "2c",
        ("--circuit-dir", "outputs/ipm_python_design"),
        "outputs/solver_spectrum_2c_recover_m35_m23_7p5_8p5_50x50_s20_sb10/chunks/chunk_000_cols_000_009/warm/points/point_0212_p_m29p8571dbm_fp_7p54082ghz/pump",
        7.540816326531111,
        7.231074707853736e-6,
        7.440816326531111,
        2,
        3e-7,
        10,
        None,
        None,
    ),
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
    if case.name == "2c":
        cmd.extend(("--signal-continuation-deadline-s", "600"))
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
        sidebands = case.selected_sidebands
        cmd = command(
            case,
            sidebands,
            args.output_dir / case.name / f"s{sidebands}",
            args.n_signal_power,
        )
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
