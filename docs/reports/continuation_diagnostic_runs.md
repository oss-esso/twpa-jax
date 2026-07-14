# Continuation diagnostic runs

This is the follow-up campaign for the July continuation-method results. The
commands are implemented in `scripts/run_continuation_diagnostics.ps1`.

## Why the first campaign is not conclusive

The completed maps are useful traversal benchmarks, but they do not establish
that the missing region is physically past the fold:

1. `r10_baseline_bridge` used the legacy column runner, which ignored
   `--recovery bridge`. Its exact match to the baseline is therefore not a bridge
   result.
2. Every non-column failure ended with an unconditional fresh continuation.
   `--inproc-fail-fast` did not bound that recovery path. Completed backbone runs
   performed 211--351 reseeds, explaining much of their extra runtime.
3. `portfolio-policy=ranked` did not try the remaining ranked candidates. Ladder
   recovery retried the already-failed best candidate.
4. The baseline restarted Python every ten frequency columns. Cross-frequency
   methods ran all 2,500 cells in one process. The OOM comparison therefore mixed
   numerical method with native-memory lifetime.
5. `SKIP_PAST_FOLD` is not evidence of nonexistence: those cells were never
   solved. The campaign baseline contains four errors followed by later PASS
   cells, so failure monotonicity is already imperfect near the boundary.

These harness issues are corrected for the runs below. The legacy
`column + reseed + patience` control remains on its original runner.

## Domains

| phase | grid | power | frequency | purpose |
|---|---:|---:|---:|---|
| sentinel | 41x3 | -31..-23 dBm | 7.7857..8.1531 GHz | trough, middle, high-fold columns; cheap no-skip truth table |
| period | 41x17 | -31..-23 dBm | 7.9694..8.3571 GHz | one observed map period, peak-to-peak |
| fold | 1x3 | reference at -23 dBm | sentinel frequencies | pump-only arclength fold location |
| branch | 21x3 | -29..-23 dBm | sentinel frequencies | small upper/post-fold branch probe |

The 0.2 dB power spacing resolves isolated failures better than the original
0.245 dB spacing. All screening runs use one trailing gain solve per converged
pump cell. Full spectra are deferred until the pump branch is known.

## Run order

### 1. Sentinel controls

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_continuation_diagnostics.ps1 -Phase sentinel
```

Interpret the runs in this order:

| id | question |
|---|---|
| `s00_exhaustive` | Which cells converge when every cell is attempted with the full legacy retry? |
| `s01_failfast_noskip` | What does fail-fast lose before any skip policy is involved? |
| `s02_failfast_p2` / `s03_failfast_p4` | Which valid cells are hidden by patience 2 or 4? |
| `s04_altparent_ff` | Can a frequency/diagonal parent recover isolated direct failures cheaply? |
| `s05_bridge_ff` | Does a real adaptive bridge recover cells, now that column recovery is wired? |
| `s06_backbone_copy_ff` | Does cross-axis traversal help without reseed or predictor stacking? |
| `s07` / `s08` | Does residual ranking help, and is trying the portfolio worth its cost? |
| `s09_portfolio_bridge_ff` | Does the composed method add recovery beyond its components? |

Do not proceed based on coverage percentage alone. Compare cell identities. A
method is skip-safe only if every PASS in `s00_exhaustive` remains PASS or is
recovered to the same branch state and gain.

### 2. Fold locator

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_continuation_diagnostics.ps1 -Phase fold
```

Inspect `f00_fold_follow/fold_curve.csv`. The inferred fold powers should agree
with the last lower-branch PASS powers from `s00_exhaustive` within roughly one
power-grid step. A missing fold, a nonperiodic jump, or disagreement larger than
0.4 dB blocks the branch probe and indicates that the current arclength
implementation needs single-cell debugging.

### 3. One-period confirmation

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_continuation_diagnostics.ps1 -Phase period
```

`p00` is the no-skip reference. `p01` isolates fail-fast, and `p02` isolates the
skip rule. `p03` and `p04` test cheap cross-axis and bridge recovery. Compare
`p05` with `p06`: they use the same numerical method, but `p06` restarts Python
every four local frequency columns. If only `p05` shows rising memory or OOM,
the issue is process/native-factor lifetime rather than predictor state size.
Small differences at chunk boundaries are expected and must be reported
separately from interior differences.

### 4. Post-fold branch probe

Run this only after the fold locator passes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_continuation_diagnostics.ps1 -Phase branch
```

For every arclength-recovered cell, require a final ordinary Newton correction
at the exact target, a finite physical pump waveform, and a valid signal solve.
Compare adjacent branch states, not just gain. A post-fold solution with a large
state discontinuity may be a different coexisting branch and must not be merged
silently into the lower-branch map.

## Skip policy decision

Choose patience from the exhaustive runs, not by assumption:

1. Build the ordered PASS/ERROR sequence for every frequency column in `p00`.
2. For each candidate patience, simulate skipping without rerunning the solver.
3. Reject any patience that would hide a later `p00` PASS.
4. Before counting a failure, allow at most the recovery method shown useful by
   `s04`/`s05`; keep fresh continuation disabled in fail-fast production runs.
5. Record skipped cells as `UNTESTED_PAST_FAILURE`, not physical fold, unless the
   arclength fold curve independently places them past the lower-branch limit.

## Spectrum promotion

Only the winning lower-branch policy and validated post-fold branch should be
rerun with spectra. The gated one-period branch-spectrum run is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_continuation_diagnostics.ps1 -Phase spectrum
```

It uses a 21x9 grid, sidebands 6, one signal worker, and local chunks of three
frequencies. Increase workers only after measuring peak resident memory; workers
do not affect non-spectrum runs.

Required comparisons are PASS/ERROR/SKIP cell identities, recovered-cell source,
pump runtime, Newton/GMRES totals, predictor residual, final correction norm,
branch-state distance, gain drift, and peak resident memory. The current CSV does
not yet persist the final four diagnostics, so do not infer them from coverage.
