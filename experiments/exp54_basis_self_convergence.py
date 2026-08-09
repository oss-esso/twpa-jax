"""Production-basis self-convergence: S=10 vs S=12 vs S=14 on the live 2c circuit.

Never run before this session -- CLAUDE.md has flagged S=10/S=12/S=14
self-convergence as unevaluated since the multitone-preconditioner work, and
[[2c-model-compresses-early-confirmed]] makes this the highest-value
remaining unknown in the production pipeline: is the ~18-22 dB
model-vs-hardware P1dB gap partly a basis-truncation artifact, or does it
survive at a converged basis?

Fixed operating point: fp=7.100 GHz, Ip=7.2311074707853736e-6 A (exp31-matched,
validated repeatedly this session: [[2c-live-pump-port-split-measured]],
[[2c-model-compresses-early-confirmed]]). Five signal frequencies spanning the
gain band (subset of exp45's grid, so results are directly comparable).
Production ``matched`` basis, standard flags, current bracket identical to
exp45's own S=10 sweep (2e-8 to 1e-6 A, 12 points, --stop-after-p1db) so G0
and P1dB are apples-to-apples across S.

Memory scales as (n_pump_modes + 2S + 1)^2 (CLAUDE.md); n_pump_modes=10 here
(positive_odd_jc, K=10), so S=14 needs roughly 1.58x the ~2.8 GB S=10 uses --
about 4.4 GB for a single worker. No ``--allow-memory-overcommit`` is passed,
so a too-tight machine gets a clean ``ResourceLimitExceeded`` instead of
swapping; that is an expected, not a failed, outcome if it happens.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "exp54_basis_self_convergence"

PUMP_FREQ_GHZ = 7.100
PUMP_CURRENT_A = 7.231074707853736e-06
FREQUENCIES_GHZ = [6.0, 7.2, 8.4]  # trimmed from 5 after measuring real per-point
# cost (~10-30 min at S=10 for a full --stop-after-p1db search): 3 freqs x 3
# S-values = 9 points fits an overnight budget alongside the Phase 5 sweep.
SIDEBAND_VALUES = [10, 12, 14]

BASE_ARGS = [
    "--circuit-dir", "designs/ipm_2c_fixed",
    "--pump-freq-ghz", str(PUMP_FREQ_GHZ),
    "--pump-current-a", str(PUMP_CURRENT_A),
    "--multitone-basis", "matched",
    "--signal-current-min-a", "2e-8",
    "--signal-current-max-a", "1e-6",
    "--n-signal-power", "12",
    "--stop-after-p1db",
    "--recovery", "ladder",
    "--factor-backend", "pardiso",
    "--signal-continuation-deadline-s", "180",
    "--power-convention", "legacy_traveling_wave",
]


def run_point(sidebands: int, freq_ghz: float, smoke: bool) -> dict[str, object]:
    label = f"S{sidebands}_f{freq_ghz:.3f}".replace(".", "p")
    output_dir = OUTPUT / ("smoke" if smoke else "full") / label
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", "scripts/run_compression.py",
        "--output-dir", str(output_dir),
        "--signal-ghz", f"{freq_ghz:.6f}",
        "--multitone-sidebands", str(sidebands),
        *BASE_ARGS,
    ]
    if smoke:
        # Tiny grid, no P1dB search, just confirm the (S, circuit, op) combo
        # solves cleanly before committing to the full overnight run.
        command = [
            arg for arg in command
            if arg not in ("--stop-after-p1db",)
        ]
        for flag, value in (
            ("--n-signal-power", "2"),
            ("--signal-current-max-a", "3e-8"),
        ):
            if flag in command:
                idx = command.index(flag)
                command[idx + 1] = value
    print(f"[exp54] running {label} ({'smoke' if smoke else 'full'}): {' '.join(command[2:4])} ...", flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (output_dir / "run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary_path = output_dir / "compression_summary.json"
    if not summary_path.exists():
        tail = completed.stderr.strip().splitlines()[-5:] if completed.stderr.strip() else ["(no stderr)"]
        print(f"[exp54]   FAILED (returncode={completed.returncode}): {' | '.join(tail)}", flush=True)
        return {
            "sidebands": sidebands, "signal_ghz": freq_ghz,
            "status": "SUBPROCESS_FAILED", "returncode": completed.returncode,
            "stderr_tail": tail, "path": str(output_dir),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {
        "sidebands": sidebands,
        "signal_ghz": freq_ghz,
        "status": summary.get("status"),
        "small_signal_gain_vs_off_db": summary.get("small_signal_gain_vs_off_db"),
        "p1db": summary.get("p1db"),
        "p1db_input_dbm": summary.get("p1db_input_dbm"),
        "power_convention": summary.get("power_convention"),
        "n_failed_power_points": summary.get("n_failed_power_points"),
        "path": str(output_dir),
    }
    print(f"[exp54]   status={result['status']} G0={result['small_signal_gain_vs_off_db']} "
          f"P1dB={result['p1db']}", flush=True)
    return result


def main() -> int:
    smoke = "--smoke" in sys.argv
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frequencies = FREQUENCIES_GHZ[:1] if smoke else FREQUENCIES_GHZ
    sidebands = SIDEBAND_VALUES[:1] if smoke else SIDEBAND_VALUES

    results = []
    for s in sidebands:
        for f in frequencies:
            results.append(run_point(s, f, smoke))

    tag = "smoke" if smoke else "full"
    report_path = OUTPUT / f"basis_convergence_{tag}.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'=' * 80}\nbasis convergence summary ({tag})\n{'=' * 80}")
    print(f"{'fs':>7} {'S':>4} {'status':>14} {'G0':>9} {'P1dB':>10}")
    for r in results:
        g0 = r.get("small_signal_gain_vs_off_db")
        p1 = r.get("p1db")
        g0_s = f"{g0:9.4f}" if isinstance(g0, (int, float)) else f"{'--':>9}"
        p1_s = f"{p1:10.3f}" if isinstance(p1, (int, float)) else f"{'--':>10}"
        print(f"{r['signal_ghz']:7.3f} {r['sidebands']:4d} {str(r['status']):>14} {g0_s} {p1_s}")

    if not smoke:
        print("\nconvergence deltas (S=12-S=10, S=14-S=12), per frequency:")
        by_freq: dict[float, dict[int, dict]] = {}
        for r in results:
            by_freq.setdefault(r["signal_ghz"], {})[r["sidebands"]] = r
        for f in frequencies:
            row = by_freq.get(f, {})
            g0_10 = row.get(10, {}).get("small_signal_gain_vs_off_db")
            g0_12 = row.get(12, {}).get("small_signal_gain_vs_off_db")
            g0_14 = row.get(14, {}).get("small_signal_gain_vs_off_db")
            p1_10 = row.get(10, {}).get("p1db")
            p1_12 = row.get(12, {}).get("p1db")
            p1_14 = row.get(14, {}).get("p1db")

            def fmt_delta(a, b):
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return f"{b - a:+7.4f}"
                return f"{'--':>7}"

            print(f"  fs={f:6.3f}  dG0(10->12)={fmt_delta(g0_10, g0_12)}  "
                  f"dG0(12->14)={fmt_delta(g0_12, g0_14)}  "
                  f"dP1dB(10->12)={fmt_delta(p1_10, p1_12)}  "
                  f"dP1dB(12->14)={fmt_delta(p1_12, p1_14)}")

    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
