# Plan: Multistability Investigation + CLI Flag Trim

Companion to `docs/convergence_investigation_log.md` (terminology rule applies:
say "convergence failure," not "fold," until physics verifies) and
`docs/control_flow/{cli_inventory.csv,discrepancy_and_risk_report.md}`.

## Goal

**(A)** Determine whether the map's cold-reseed path can land pump solves on a
different, non-physical solution branch than the one a continuously-pumped
device sits on — and whether that explains the ~6.5 dB Themis peak-gain
undershoot (`docs/convergence_investigation_log.md`,
`themis-2c-reproduction-boundary-vs-fidelity` memory).
**(B)** Trim `run_gain_map.py`'s CLI surface to match what's actually run,
without deleting working experimental features.

## Current State Analysis

- `run_gain_map.py:821` — every cold reseed (fold-skip retry, warm-chain
  recovery `reseed`, force_gain fallback) calls `solve_problem.zeros()` then
  Newton-ladders `lambda: 0->1` at full target power. No branch-continuity
  check exists anywhere against the warm neighbor.
- `predictors.py` `rank_candidates` (portfolio) ranks purely by
  post-correction residual — no continuity check either.
- `docs/control_flow/cli_inventory.csv` (91 flags) +
  `discrepancy_and_risk_report.md` already audited every flag's
  reachability/status — reused directly here, not re-derived.
- Real production baseline = `run_campaign.ps1` `c04_baseline_prod`
  (`$Common` + `--traversal column`, `--inproc-fold-predictor secant`,
  default `--recovery reseed`, default `--fold-policy patience`).
- Confirmed drift: `--fold-skip-patience` default `0` vs real `4`;
  `--inproc-solve-deadline-s` default `0.0` vs real `14` (OOM guard,
  `run_campaign.ps1:67-70`).
- Confirmed dead: `--executor subprocess` + `--pump-timeout-s` /
  `--gain-timeout-s` / `--python-executable` — zero `.ps1` runners use it
  (docs-only references).
- Confirmed alive despite "experimental" tag: `--column-arclength-recovery`
  (`run_column_arclength_map.ps1`), `--fold-follow` (`run_campaign.ps1`),
  `--column-power-substep` (`run_continuation_diagnostics.ps1`).

## Decisions locked in (asked and answered before writing this plan)

1. Update the two drifted argparse defaults to match real practice
   (`--fold-skip-patience` -> `4`, `--inproc-solve-deadline-s` -> `14.0`).
2. `--executor subprocess` path: **soft-deprecate** (help text + one-time
   warning), not hard-deleted.
3. Experimental-but-alive flags (arclength recovery, fold-follow,
   column-power-substep): **keep, just label clearly**. Trim pass only
   touches truly dead/unreachable combos and genuinely unused flags — not the
   continuation-method research surface CLAUDE.md documents.

## What We're NOT Doing

- Not deleting traversal/predictor/recovery/fold-policy alternatives.
- Not hard-deleting the subprocess executor.
- Not wiring up `positive_phasor_explicit` into `run_gain_map`'s CLI (out of
  scope — feature work, not trim).
- Not touching solver numerics/algorithms in Track A — diagnosis only; any
  fix is a follow-up plan.

## Prerequisites

- [ ] None — both tracks work off existing outputs/code.

---

## Phase A1: Adjacent-cell discontinuity scanner

**Overview**: Read-only scan of a finished map's `map_points.csv` (start with
the Themis-recovery neighborhood used all session,
`outputs/power_sweep_col3_p7_p8`, plus one full campaign map e.g.
`c04_baseline_prod`) for cells adjacent in (P,f) where one side used a cold
reseed and gain_db/coeff_rel jumps discontinuously vs the smooth local trend.

**Changes**:
- New script `scripts/debug_branch_discontinuity_scan.py`. Load
  `map_points.csv` + per-point `pump_report.json` (needs
  `warm_started`/`warm_retry_reseed` columns, already written per row —
  `run_gain_map.py:938,2068`). For each column, compute local gain_db
  second-difference; flag cells where `warm_retry_reseed=True` AND the jump
  exceeds N-sigma of the column's smooth trend. Output CSV of flagged (P,f)
  pairs + a plot.

**Success criteria**:
- Automated: script runs clean on both target dirs, produces CSV.
- Manual: the known col3 fp=7.329 GHz wall region gets flagged (sanity check
  the detector fires where we already know reseeding happened).

**Status: DONE** (`scripts/debug_branch_discontinuity_scan.py`). Deviation
found during implementation: `outputs/measurement_match_debug_01/column_debug_col3_trim`
(the known col3 fp=7.329 GHz wall) has `warm_retry_reseed=False` on every row
-- that debug run used a bare `column` pass with no `--recovery reseed`
enabled, so it never actually cold-reseeded; the wall there is a plain
FAIL->SKIP_PAST_FOLD with no reseed to test against. The sanity check instead
validated against `campaign_continuation_methods/c04_baseline_prod` (real
recovery-enabled 50x50 run): its **only** `warm_retry_reseed=True` cell
(fp=7.704 GHz, i_power=27, P=-28.39 dBm) was correctly flagged -- second-diff
-6.28 dB vs column sigma 1.46 dB (>3sigma threshold 4.39 dB). Detector fires
on the one ground-truth reseed instance available; too small an n to say more
than "the detector works as specified." Phase A2 (direct seed-vs-warm probe)
is the real multistability test.

## Phase A2: Targeted alternate-seed probe

**Overview**: At a flagged (or already-known) wall point, empirically test
for multistability: solve the SAME target (P,f) from two different seeds —
(a) cold `X=0` ladder (current reseed behavior), (b) the last-converged
neighbor's `X` fed through `solve_direct` at lambda=1 directly (no ladder) —
and compare resulting `X`/gain_db/coeff_rel. Two genuinely different
converged fixed points at the same (P,f) = confirmed multistability.

**Changes**:
- New script `scripts/debug_alternate_seed_probe.py`. Reuse `InProcessEngine`
  (pattern from `debug_power_sweep_fine_continuation_methods.py`). At the
  target point, run `mode="seed"` (cold) and `mode="warm"` with `warm_X` from
  the nearest converged neighbor (both directions if available). Log `X`
  norm difference, gain_db difference, both `coeff_rel`.

**Success criteria**:
- Automated: both solves complete (converged or force_gain) and are logged.
- Manual: if both solves converge and gains agree within noise, no alternate
  branch was found at that target. If two converged results diverge
  meaningfully, branch switching is confirmed.

**Status: DONE** (`scripts/debug_alternate_seed_probe.py`, now CLI-parameterized
by freq/power/i_power/neighbor pump dirs rather than hardcoded, since the
original target cell turned out unusable -- see below). Result: **no
multistability confirmed in the tested cells.** Full detail in
`docs/convergence_investigation_log.md` 2026-07-17 entry; summary:
- The A1-flagged archived cell (`c04_baseline_prod`, fp=7.704 GHz, i27) does
  not reproduce under the current checkout and inputs (deterministic repro
  gives FAIL, archived record says PASS). The available provenance cannot
  attribute that mismatch to solver code alone; it is not a live
  discontinuity. Regenerating that column fresh gives zero reseed events.
- Regenerated a full fresh 50x50 map under current code, found 3 live reseed
  events, and probed all 3 (cold vs. both real-neighbor-X direct warm starts,
  restricted to Schur retained-port shape). Two cells have a converged warm
  result matching cold to x_norm_diff ~1e-9; the third has no converged warm
  result. The failing attempts fail outright and never settle on a distinct
  second branch.

## Phase A3: Cross-check against measurement

**Overview**: Only if A2 confirms multistability: rerun
`scripts/align_map_to_measurement.py` comparing the Themis measurement
against whichever branch (cold vs warm-continued) fits better, to see if
picking the "other" branch closes some of the 6.5 dB peak-gain gap.

**Success criteria**:
- Manual: read RMS/peak-gain-diff before/after branch choice; report the
  delta honestly either way.

**Status: SKIPPED.** A2 did not confirm multistability (no distinct
alternate branch found), so there is nothing to cross-check against
measurement.

## Phase A4: Write up

**Overview**: Append dated section to `docs/convergence_investigation_log.md`:
scanner results, probe results, measurement cross-check, and — only if
confirmed — a fix proposal (e.g. prefer warm-neighbor `X` over `X=0` as the
reseed seed) as a *proposal*, not implemented in this pass.

**Success criteria**:
- Manual: log entry has file paths + concrete numbers, follows "convergence
  failure not fold" terminology rule.

**Status: DONE.** `docs/convergence_investigation_log.md` 2026-07-17 entry.
No fix proposal written (A2 did not confirm the premise a fix would address).
**Track A conclusion: multistability via cold reseed is not supported by the
evidence; the ~6.5 dB Themis peak-gain undershoot needs a different
explanation.**

---

## Phase B1: Fix drifted defaults

**Changes**:
- `scripts/run_gain_map.py`: `--fold-skip-patience` default `0`->`4`;
  `--inproc-solve-deadline-s` default `0.0`->`14.0`. Update each flag's help
  text to note the old default and why it changed (OOM guard / fold
  short-circuit needed for unattended runs).
- `docs/control_flow/cli_inventory.csv`, `CLAUDE.md`: update `default`
  column / notes for both rows.

**Success criteria**:
- Automated: `python scripts/run_gain_map.py --help` shows new defaults.
- Manual: `run_campaign.ps1`'s `$Common` flags become redundant-but-harmless
  (fine to leave explicit for self-documentation).

## Phase B2: Soft-deprecate subprocess executor

**Changes**:
- `scripts/run_gain_map.py`: prefix `--executor subprocess` choice help text
  with `[DEPRECATED — no runner script uses this path; prefer inprocess]`.
  Same prefix on `--pump-timeout-s`, `--gain-timeout-s`,
  `--python-executable`. Add a one-time `logger.warning(...)` in `main()`
  when `args.executor == "subprocess"`.

**Success criteria**:
- Automated: `--executor subprocess --help` shows deprecation text; a smoke
  run with `--executor subprocess` still works and prints the warning once.

## Phase B3: Small correctness fixes flagged by the audit (not deletions)

**Changes**:
- `--pump-mode-policy` missing `choices=` (`run_gain_map.py:2394`, report
  item A1): add
  `choices=["dense_real", "positive_phasor_explicit", "positive_odd_jc", "auto_jc"]`
  so bad values fail at parse time instead of deep inside
  `resolve_pump_basis`. Append
  `[CLI-unreachable via this script; needs --pump-modes wiring, use exp08
  directly]` to `positive_phasor_explicit`'s help mention.
- `--linear-seed-maxiter`: help text prefix
  `[subprocess-only, no effect under default --executor inprocess]` (matches
  report item C1) — groups naturally with the B2 subprocess-only labeling.

**Success criteria**:
- Automated: `--help` confirms new choices reject a bad policy string
  immediately (`ValueError` at parse, not mid-solve).
- Manual: none else — label/validation only.

---

## Testing Strategy

**Project maturity level**: Active Development (research codebase; existing
`tests/` cover core solver only, no test suite for `run_gain_map.py`'s CLI
itself).

**Unit tests**: none new for Track A (diagnostic scripts, one-off
investigation tooling — matches existing `debug_*.py` pattern, no test
coverage). Track B: no existing test asserts argparse defaults; verify via
`--help` + a real invocation instead, consistent with current practice.

**Integration/manual tests**:
- A1/A2: run against real map data, eyeball results (this **is** the
  investigation).
- B1/B2/B3: `python scripts/run_gain_map.py --help | grep -A2
  "fold-skip-patience\|solve-deadline\|executor\|pump-mode-policy\|linear-seed-maxiter"`;
  one small smoke map (`--n-power 3 --n-frequency 3`) with bare defaults to
  confirm B1 doesn't change accepted behavior; one `--executor subprocess`
  smoke run to confirm B2 didn't break it.

## Rollback Plan

All changes are additive (help text, one new `choices=`, two default values,
two new scripts) — `git diff` / `git checkout -- scripts/run_gain_map.py`
reverts B1-B3 cleanly. A1/A2 scripts are new files, delete if unwanted. No
production map-generation behavior changes except the two default-value
flips (B1), which only affect callers who never explicitly override those
two flags — every existing `.ps1` runner already overrides both explicitly,
so zero behavior change there.

## Order

A1 -> A2 -> A3 -> A4 first (open physics question). B1-B3 after (mechanical,
low-risk, independent).

## Discriminator execution status (2026-07-17)

The post-A discriminator was executed at the representative Themis columns.
The Python no-skip/high-budget run and 5/10/15-mode basis ladder all place the
7.31122-GHz boundary between -27.606 and -27.237 dBm.  JosephsonCircuits.jl
continuation runs were completed at 6.1817, 7.31122, and 7.71462 GHz.  Their
artefacts are under `D:/Projects/Thesis/track_a_discriminator_20260717` and
the detailed interpretation is recorded in `docs/convergence_investigation_log.md`.
