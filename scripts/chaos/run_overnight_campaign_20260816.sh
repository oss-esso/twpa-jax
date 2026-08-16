#!/usr/bin/env bash
# Sequential chaos campaign, 2026-08-16.  Ordered by scientific value so that
# killing a later stage never costs an earlier result.  Every stage writes per
# point and skips completed points, so the whole script is resumable.
#
#   Stage 1  ~1.0 h  2c gap resolution, PUMP-ONLY, 2100 periods
#   Stage 2  ~1.8 h  transition refinement, 4 devices, signal-driven
#   Stage 3  ~1.5 h  timestep controls at dt/2
#
# Total approximately 4.3 h wall at 3 workers.

set -u
cd "$(dirname "$0")/../.." || exit 1

RUN=scripts/chaos/run_phaseB_overnight.py
LOG=outputs/chaos/campaign_20260816.log

stamp () { date "+%Y-%m-%d %H:%M:%S"; }
say () { echo "[$(stamp)] $*" | tee -a "$LOG"; }

say "=== campaign start ==="

# ---------------------------------------------------------------------------
# Stage 1.  Does a torus exist inside the 2c jump?
#
# The existing 2c points bracket the transition at 0.575 -> 0.600 with a 1595x
# step in stroboscopic spread, which excludes a supercritical Neimark-Sacker
# whose scaling region is comparable to the sampled range -- but NOT one
# narrower than the 0.025 grid step.  11 points at 0.005 spacing resolve that,
# and span both the pre-transition floor and the post-transition plateau at a
# single consistent setting.
#
# PUMP-ONLY: a signal tone would make the forcing quasi-periodic by
# construction and guarantee a 2-torus a priori, which is the thing under test.
# 2100 periods also clears the ~1050-period settling floor that the existing
# 1600-period runs open their analysis window below.
# ---------------------------------------------------------------------------
say "--- stage 1: 2c gap, pump-only, 11 points 0.575..0.625, 2100 periods"
python "$RUN" \
  --output outputs/chaos/phaseB_2c_gap \
  --devices ipm_2c_fixed \
  --control-linspace 0.575 0.625 11 \
  --periods 2100 --dt-norm 0.01 --workers 3 >> "$LOG" 2>&1
say "--- stage 1 done (exit $?)"

# ---------------------------------------------------------------------------
# Stage 2.  Transition refinement on the HB-validity observable.
#
# Signal-driven on purpose: this stage measures on_lattice, the accuracy
# ceiling for any HB result, which needs the signal present.  Brackets are the
# measured ones, not the wider proposed ranges -- guarcello's -55..-53.7 is
# already covered at 0.02-0.05 dB, and 2c's endpoints must be included or the
# bracket is not guaranteed.
#
# --periods 2100 is exact for all four devices and MUST NOT drift: the pump-off
# reference cache is keyed on device|dt|tmax|signal, and a miss silently
# re-measures the reference and changes the denominator of gain_vs_off_db.
# guarcello uses --signal-dbm -90 with zero current; the others 3e-08 A.
# ---------------------------------------------------------------------------
say "--- stage 2a: guarcello -53.7..-53.0"
python "$RUN" --output outputs/chaos/phaseB_signal \
  --devices guarcello --control-linspace -53.7 -53.0 9 \
  --periods 2100 --dt-norm 0.01 --signal-dbm -90 --workers 3 >> "$LOG" 2>&1
say "--- stage 2a done (exit $?)"

say "--- stage 2b: jc_jtwpa -28.2..-27.8"
python "$RUN" --output outputs/chaos/phaseB_signal \
  --devices jc_jtwpa --control-linspace -28.2 -27.8 9 \
  --periods 2100 --dt-norm 0.01 --signal-current-a 3e-08 --workers 3 >> "$LOG" 2>&1
say "--- stage 2b done (exit $?)"

say "--- stage 2c: jc_fqjtwpa -31.5..-31.2"
python "$RUN" --output outputs/chaos/phaseB_signal \
  --devices jc_fqjtwpa --control-linspace -31.5 -31.2 9 \
  --periods 2100 --dt-norm 0.01 --signal-current-a 3e-08 --workers 3 >> "$LOG" 2>&1
say "--- stage 2c done (exit $?)"

say "--- stage 2d: ipm_2c_fixed 0.575..0.625"
python "$RUN" --output outputs/chaos/phaseB_signal \
  --devices ipm_2c_fixed --control-linspace 0.575 0.625 9 \
  --periods 2100 --dt-norm 0.01 --signal-current-a 3e-08 --workers 3 >> "$LOG" 2>&1
say "--- stage 2d done (exit $?)"

# ---------------------------------------------------------------------------
# Stage 3.  Timestep controls.
#
# dt_norm = 0.01 has never been shown to leave the TRANSITION LOCATION fixed;
# the earlier screen only checked on_lattice at -27.8 dBm, already past the
# collapse.  Without this, every bracket above resolves a number that is not
# timestep-converged.
#
# Separate --output is mandatory: at the same root, _is_done finds the existing
# dt=0.01 trace and skips every point.
# ---------------------------------------------------------------------------
for spec in "jc_jtwpa -28.2 -27.8" "jc_fqjtwpa -31.5 -31.2" "ipm_2c_fixed 0.575 0.625"; do
  set -- $spec
  say "--- stage 3: $1 dt=0.005 endpoints $2 $3"
  python "$RUN" --output outputs/chaos/phaseB_dt005 \
    --devices "$1" --control-linspace "$2" "$3" 2 \
    --periods 2100 --dt-norm 0.005 --signal-current-a 3e-08 --workers 3 >> "$LOG" 2>&1
  say "--- stage 3 $1 done (exit $?)"
done

say "=== campaign complete ==="
