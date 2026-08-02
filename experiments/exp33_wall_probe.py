"""Probe the 6.540 GHz multitone continuation wall with short sweeps.

The exp32 campaign solved 5.296 and 7.052 GHz cleanly but walled at 6.540 GHz
while the gain was still flat at 10.30 dB.  The stall lives in the source-scale
lambda continuation inside a single power point, not in the power-axis substep,
so a finer power grid does not move it.

This runs deliberately small sweeps -- six power points from -110 to -85 dBm,
a 120 s continuation deadline, and a hard per-run timeout -- across neighbouring
signal frequencies and two sideband counts, to separate three hypotheses:

* an unlucky signal frequency (neighbours solve, 6.540 does not);
* an S=10 basis-conditioning defect (S=6 solves where S=10 walls);
* a genuine solution-branch boundary (everything walls at the same gain).

Every run is written to its own directory and skipped if already complete, so
the probe can be re-invoked to fill in the matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

CIRCUIT_DIR = "designs/ipm_2c_fixed"
PUMP_FREQ_GHZ = 7.100
PUMP_CURRENT_A = 7.231074707853736e-06
POWER_MIN_DBM = -110.0
POWER_MAX_DBM = -85.0
N_POWER = 6
Z0_OHM = 50.0
CONTINUATION_DEADLINE_S = 120.0
RUN_TIMEOUT_S = 1800.0

# (signal frequency GHz, sideband count).  6.540/S=10 reproduces the wall,
# 6.540/S=6 tests the basis, the neighbours test frequency sensitivity.
PROBES: tuple[tuple[float, int], ...] = (
    (6.540, 10),
    (6.540, 6),
    (6.300, 10),
    (6.800, 10),
    (6.440, 10),
    (6.640, 10),
)


def dbm_to_current(power_dbm: float) -> float:
    """Peak current for a given power into ``Z0_OHM``."""
    return math.sqrt(2.0 * 10.0 ** (power_dbm / 10.0) * 1e-3 / Z0_OHM)


def command(output_dir: Path, signal_ghz: float, sidebands: int) -> list[str]:
    """Short compression sweep for one (frequency, sideband count) probe."""
    return [
        sys.executable, "-u", "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--circuit-dir", CIRCUIT_DIR,
        "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
        "--pump-current-a", str(PUMP_CURRENT_A),
        "--pump-current-jc-scale", "1.0",
        "--pump-mode-policy", "positive_odd_jc",
        "--pump-mode-count", "10",
        "--pump-nt", "40",
        "--multitone-basis", "matched",
        "--multitone-sidebands", str(sidebands),
        "--source-port", "1", "--pump-port", "4", "--out-port", "2",
        "--attenuation-db", "0",
        "--factor-backend", "pardiso",
        "--n-signal-power", str(N_POWER),
        "--signal-current-min-a", repr(dbm_to_current(POWER_MIN_DBM)),
        "--signal-current-max-a", repr(dbm_to_current(POWER_MAX_DBM)),
        "--recovery", "ladder",
        "--signal-continuation-deadline-s", str(CONTINUATION_DEADLINE_S),
        "--signal-workers", "1",
        "--signal-ghz", str(signal_ghz),
        "--allow-memory-overcommit",
    ]


def read_points(path: Path) -> list[dict[str, str]]:
    """Rows of a ``compression_points.csv`` as plain string dicts."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def probe_row(run_dir: Path, signal_ghz: float, sidebands: int) -> dict[str, object]:
    """Reduce one completed run to the fields that separate the hypotheses."""
    summary = json.loads(
        (run_dir / "compression_summary.json").read_text(encoding="utf-8")
    )
    rows = read_points(run_dir / "compression_points.csv")
    solved = [r for r in rows if r["status"] == "VALID_SOLVED"]
    failed = [r for r in rows if r["status"] != "VALID_SOLVED"]
    last = solved[-1] if solved else None
    return {
        "signal_ghz": signal_ghz,
        "sidebands": sidebands,
        "small_signal_gain_db": summary.get("small_signal_gain_vs_off_db"),
        "n_solved": len(solved),
        "n_failed": len(failed),
        "last_solved_dbm": float(last["signal_power_dbm"]) if last else None,
        "last_solved_gain_db": float(last["gain_vs_off_db"]) if last else None,
        "first_failed_dbm": (
            float(failed[0]["signal_power_dbm"]) if failed else None
        ),
        "first_failed_status": failed[0]["status"] if failed else None,
        "p1db": summary.get("p1db"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/exp33_wall_probe")
    )
    parser.add_argument(
        "--only-ghz", type=float, default=None,
        help="Run a single signal frequency from the probe matrix.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for signal_ghz, sidebands in PROBES:
        if args.only_ghz is not None and abs(signal_ghz - args.only_ghz) > 1e-9:
            continue
        run_dir = args.output_dir / f"fs_{signal_ghz:.3f}ghz_s{sidebands}"
        summary_path = run_dir / "compression_summary.json"
        if not summary_path.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = command(run_dir, signal_ghz, sidebands)
            print("run " + subprocess.list2cmdline(cmd), flush=True)
            try:
                subprocess.run(cmd, check=False, timeout=RUN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(
                    f"TIMEOUT fs={signal_ghz} S={sidebands} "
                    f"after {RUN_TIMEOUT_S:.0f}s",
                    flush=True,
                )
        if not summary_path.exists():
            results.append({
                "signal_ghz": signal_ghz, "sidebands": sidebands,
                "small_signal_gain_db": None, "n_solved": 0, "n_failed": None,
                "last_solved_dbm": None, "last_solved_gain_db": None,
                "first_failed_dbm": None, "first_failed_status": "NO_ARTIFACT",
                "p1db": None,
            })
        else:
            results.append(probe_row(run_dir, signal_ghz, sidebands))
        print(json.dumps(results[-1], indent=2), flush=True)

    if not results:
        raise SystemExit(
            f"--only-ghz {args.only_ghz} matches no probe frequency; choose from "
            + ", ".join(f"{ghz:.3f}" for ghz, _ in PROBES)
        )

    csv_path = args.output_dir / "probe_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nwrote {csv_path}")
    header = (
        f"{'fs GHz':>8} {'S':>3} {'G0 dB':>8} {'solved':>7} {'failed':>7} "
        f"{'last dBm':>9} {'last G':>8} {'wall dBm':>9}  status"
    )
    print(header)
    for row in results:
        def fmt(key: str, width: int, spec: str) -> str:
            value = row[key]
            return (
                f"{'--':>{width}}" if value is None
                else f"{value:>{width}{spec}}"
            )
        print(
            f"{row['signal_ghz']:>8.3f} {row['sidebands']:>3} "
            f"{fmt('small_signal_gain_db', 8, '.3f')} "
            f"{row['n_solved']:>7} {fmt('n_failed', 7, 'd')} "
            f"{fmt('last_solved_dbm', 9, '.2f')} "
            f"{fmt('last_solved_gain_db', 8, '.3f')} "
            f"{fmt('first_failed_dbm', 9, '.2f')}  "
            f"{row['first_failed_status'] or 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
