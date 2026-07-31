"""Compare the 2c compression solver against the measured Themis saturation cube.

Reference data: ``docs/development/10.15.34_Themis_SetupJan28_VTS_transmission_15mK``
(fp = 7.256 GHz, pump -66.7 dBm at the device input, 121 signal powers x 2001
signal frequencies, ``Response`` already in dB of gain).

Two arms, because the model has no amplifying operating point at the measured
pump frequency:

``literal``
    The measurement's own pump setting, fp = 7.256 GHz at -66.7 dBm. A Floquet
    scan over 5.5-9.0 GHz peaks at 1.328 dB here and never reaches 3 dB at any
    convergent pump power, so this arm is expected to report
    ``NO_GAIN_AT_OPERATING_POINT``. It is run anyway so the negative result is
    an artifact rather than an assertion.

``matched``
    The published exp20 operating point, where the model does reach the measured
    gain (15.9 dB simulated against 15.87 dB measured). Sweeps the signal
    frequency across the full measured band so P1dB can be compared *at equal
    small-signal gain*, which is the only single-variable comparison available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CIRCUIT_DIR = "outputs/ipm_python_design"
PUMP_PORT, SOURCE_PORT, OUT_PORT = 4, 1, 2
SIDEBANDS = 10

# Pump solved fresh: the stored solutions all sit at 7.5408 GHz.
LITERAL_PUMP_GHZ = 7.256
LITERAL_PUMP_CURRENT_A = 2.9243522111088247e-06  # sqrt(2 P / Z0), P = -66.7 dBm

# The published exp20 2c point, reusing its stored pump solution so the pump
# state is bit-identical to the published numbers.
MATCHED_PUMP_GHZ = 7.540816326531111
MATCHED_PUMP_CURRENT_A = 7.231074707853736e-06
MATCHED_PUMP_DIR = (
    "outputs/solver_spectrum_2c_recover_m35_m23_7p5_8p5_50x50_s20_sb10/chunks/"
    "chunk_000_cols_000_009/warm/points/point_0212_p_m29p8571dbm_fp_7p54082ghz/pump"
)

# The measured band with G0 > 8 dB runs 5.34-8.95 GHz; the model shows gain over
# 5.8-9.0 GHz at the matched point. Sweep the overlap.
BAND_GHZ = (5.8, 9.0)


def command(
    arm: str,
    outdir: Path,
    frequencies: int,
    workers: int,
    resource_budget_gb: float,
    powers: int,
    factor_backend: str,
) -> list[str]:
    literal = arm == "literal"
    cmd = [
        sys.executable, "scripts/run_compression.py",
        "--output-dir", str(outdir),
        "--circuit-dir", CIRCUIT_DIR,
        "--resource-budget-gb", str(resource_budget_gb),
        # The banded backend is rejected on this Schur system: the retained 2c
        # nodes are not contiguous, so the reordered band spans essentially the
        # whole 156116-row matrix (measured 544.5 GB of band storage). PARDISO
        # is what the published exp20 2c point used.
        "--factor-backend", factor_backend,
        "--pump-freq-ghz",
        str(LITERAL_PUMP_GHZ if literal else MATCHED_PUMP_GHZ),
        "--pump-current-a",
        str(LITERAL_PUMP_CURRENT_A if literal else MATCHED_PUMP_CURRENT_A),
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", "10",
        "--pump-nt", "40",
        "--source-port", str(SOURCE_PORT),
        "--pump-port", str(PUMP_PORT),
        "--out-port", str(OUT_PORT),
        "--n-signal-power", str(powers),
        "--signal-current-min-a", "1e-12",
        "--signal-current-max-a", "3e-7",
        "--attenuation-db", "0",
        "--multitone-basis", "matched",
        "--multitone-sidebands", str(SIDEBANDS),
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", "600",
        "--signal-workers", str(workers),
        # ``fast_coupled_footprint`` bounds a worker with the full 6446-node
        # count and reports 6.04 GB, but this device runs the Schur backend,
        # which retains 2518 nodes; the measured S=10 peak is ~2.8 GB. The
        # guard is the conservative bound, not the actual requirement.
        "--allow-memory-overcommit",
    ]
    if literal:
        # One frequency at the measured gain peak; the arm only has to record
        # that there is nothing to compress.
        cmd.extend(("--signal-ghz", "7.080"))
    else:
        cmd.extend((
            "--signal-ghz-min", str(BAND_GHZ[0]),
            "--signal-ghz-max", str(BAND_GHZ[1]),
            "--n-signal-freq", str(frequencies),
            "--pump-solution-dir", MATCHED_PUMP_DIR,
        ))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp23_2c_measured_saturation")
    )
    parser.add_argument("--arm", choices=("literal", "matched"), action="append")
    parser.add_argument("--n-signal-freq", type=int, default=15)
    parser.add_argument("--n-signal-power", type=int, default=25)
    parser.add_argument("--signal-workers", type=int, default=2)
    parser.add_argument("--resource-budget-gb", type=float, default=8.0)
    parser.add_argument(
        "--factor-backend", choices=("pardiso", "banded"), default="pardiso"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = args.arm or ["literal", "matched"]
    for arm in arms:
        cmd = command(
            arm,
            args.output_dir / arm,
            args.n_signal_freq,
            args.signal_workers,
            args.resource_budget_gb,
            args.n_signal_power,
            args.factor_backend,
        )
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
