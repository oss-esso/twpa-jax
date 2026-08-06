"""Overnight campaign: basis self-convergence (S=10/12/14) + a dense,
production-basis P1dB-vs-frequency sweep against measurement.

Two things this session established as the highest-value untested items
(see memory [[2c-model-compresses-early-confirmed]]):

  1. Production-basis self-convergence (S=10 vs 12 vs 14) -- flagged as
     unevaluated in CLAUDE.md since the multitone-preconditioner work, never
     run. ``experiments/exp54_basis_self_convergence.py``.
  2. A dense, properly-spaced (<=100 MHz) S=10 P1dB-vs-frequency sweep at the
     validated exp31 operating point (fp=7.100 GHz, Ip=7.2311e-6 A), compared
     against measurement with both 2026-08-05 calibration fixes applied
     (loss_B1, Norton) -- exp45's own sweep only sampled 18 points at ~200 MHz
     spacing, below the anti-aliasing spec.

Explicitly NOT included: the (fp, I_p) operating-point grid search
(``scripts/fit_model_operating_point.py``). Measured cost during prep:
~78 s per signal-frequency point at its own cheap S=2 probe resolution,
which means one (fp, I_p) candidate (full fit-window sweep) costs roughly
2 hours, and the default grid is 61 candidates (~5 days). Also low expected
value here: two independently-tuned operating points (op7.100, op7.379)
already give consistent ~18 dB gaps this session, so re-tuning the operating
point is unlikely to explain a gap that size. If a full operating-point
search is wanted later, run ``fit_model_operating_point.py`` directly with a
deliberately tiny grid and budget multiple nights for it.

Runs sequentially (not in parallel) since both phases compete for the same
machine's RAM; each phase's own subprocess crash or non-convergence is
logged and does NOT abort the remaining phases. Safe to Ctrl-C at any point
-- each phase resumes/re-reads what it can from disk, nothing is corrupted
by an interrupted run (compression_summary.json is only written after a
point's own subprocess exits).
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "overnight_campaign_logs"

PUMP_FREQ_GHZ = "7.100"
PUMP_CURRENT_A = "7.231074707853736e-06"
PHASE5_OUTPUT = ROOT / "outputs" / "phase5_dense_p1db_op7100"
PHASE6_OUTPUT = ROOT / "outputs" / "phase6_gain_matched_comparison_op7100"


def log_step(name: str) -> None:
    print(f"\n{'#' * 90}\n# {datetime.now().isoformat(timespec='seconds')}  {name}\n{'#' * 90}\n", flush=True)


def run(command: list[str], log_name: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    print(f"[overnight] running: {' '.join(command)}", flush=True)
    print(f"[overnight] logging to: {log_path}", flush=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False,
        )
    elapsed_min = (time.perf_counter() - t0) / 60.0
    print(f"[overnight] finished in {elapsed_min:.1f} min, returncode={completed.returncode}", flush=True)
    if completed.returncode != 0:
        print(f"[overnight] WARNING: {log_name} exited nonzero -- check {log_path}", flush=True)
    return completed.returncode


def main() -> int:
    smoke = "--smoke" in sys.argv
    overall_t0 = time.perf_counter()
    phase5_output = (ROOT / "outputs" / "phase5_smoke") if smoke else PHASE5_OUTPUT
    phase6_output = (ROOT / "outputs" / "phase6_smoke") if smoke else PHASE6_OUTPUT

    # --- Phase A: production-basis self-convergence, S=10/12/14 ---
    log_step("Phase A: basis self-convergence (S=10/12/14)")
    phase_a_cmd = [sys.executable, "-u", "experiments/exp54_basis_self_convergence.py"]
    if smoke:
        phase_a_cmd.append("--smoke")
    run(phase_a_cmd, "phaseA_basis_convergence.log")

    # --- Phase B.1: dense S=10 P1dB-vs-frequency sweep at op7.100 ---
    log_step("Phase B.1: dense S=10 P1dB sweep, 100 MHz spacing, 6.0-8.4 GHz")
    n_freq = "2" if smoke else "25"
    freq_max = "6.1" if smoke else "8.4"
    n_power = "2" if smoke else "12"
    stop_flags = [] if smoke else ["--stop-after-p1db"]
    run([
        sys.executable, "-u", "scripts/run_compression.py",
        "--output-dir", str(phase5_output),
        "--circuit-dir", "designs/ipm_2c_fixed",
        "--pump-freq-ghz", PUMP_FREQ_GHZ,
        "--pump-current-a", PUMP_CURRENT_A,
        "--signal-ghz-min", "6.0", "--signal-ghz-max", freq_max, "--n-signal-freq", n_freq,
        "--multitone-basis", "matched", "--multitone-sidebands", "10",
        "--signal-current-min-a", "2e-8", "--signal-current-max-a",
        ("3e-8" if smoke else "1e-6"),
        "--n-signal-power", n_power, *stop_flags,
        "--recovery", "ladder", "--factor-backend", "pardiso",
        "--power-convention", "norton",
        "--signal-workers", "1",
        "--signal-continuation-deadline-s", "300",
        # Safety valve: the worker-footprint pre-check aborts the WHOLE sweep
        # before it starts if free RAM is momentarily tight (measured during
        # prep: needed ~5.76 GB, had 3.43 GB free with other apps open).
        # Overcommitting risks swapping on a single point rather than losing
        # the whole night to a clean-but-total abort; if it swaps badly,
        # Ctrl-C and free more RAM before restarting.
        "--allow-memory-overcommit",
    ], "phaseB1_dense_p1db_sweep.log")

    # --- Phase B.2: gain-matched comparison against measurement ---
    # Skipped in smoke mode: the smoke sweep never crosses 1 dB compression
    # (2 tiny power points), so there is no p1db field for the comparison
    # script to read -- that is expected, not a bug.
    if not smoke:
        log_step("Phase B.2: gain-matched P1dB comparison (loss_B1 + Norton corrected)")
        run([
            sys.executable, "-u", "experiments/exp53_gain_matched_p1db_comparison.py",
            "--run-dir", str(phase5_output),
            "--output-dir", str(phase6_output),
        ], "phaseB2_gain_matched_comparison.log")

    elapsed_h = (time.perf_counter() - overall_t0) / 3600.0
    log_step(f"overnight campaign complete, total {elapsed_h:.2f} h")
    print(f"Results:\n"
          f"  Phase A (basis convergence): outputs/exp54_basis_self_convergence/basis_convergence_full.json\n"
          f"  Phase B.1 (dense sweep):     {phase5_output}\n"
          f"  Phase B.2 (final gap):       {phase6_output}/gain_matched_p1db_summary.json\n"
          f"  Logs:                        {LOG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
