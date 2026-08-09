# Continuation-method campaign — results & why

> **Audit correction (2026-07-12):** the original fold-locked conclusion below
> is not established by these runs. The legacy column runner ignored
> `--recovery bridge`, so `r10` did not test bridge recovery. Non-column runs
> also ignored fail-fast for their final reseed, ranked portfolios did not try
> the remaining candidates, and only the baseline used process-level frequency
> chunks. See `continuation_diagnostic_runs.md` for the corrected controlled
> campaign. Historical results are retained below as measured data, not as proof
> that every skipped cell is beyond a physical fold.

Design: 2c (`outputs/ipm_python_design`), 50×50 grid, −35…−23 dBm × 7.5…8.5 GHz,
single-point gain at 100 MHz detuning. ~24 h wall. Data + figures:
`outputs/campaign_continuation_methods/comparison/` (`compare_campaign.py`,
snapshots preserved against `--overwrite`).

## TL;DR

Coverage is **fold-locked at ~69%**, not seed-locked. Every traversal, predictor,
and the one recovery method that completed land within ±3% of the column baseline
and share the **same peak gain (26.55 dB)** and the **same unsolved region**. The
proof: **bridge recovery (r10) recovered exactly 0 cells** vs baseline. The configs
that could in principle cross the fold (r11/r12 portfolio+bridge / ladder) **OOM'd**;
arclength (a15) **never ran**. Better warm starts cannot manufacture a solution that
does not exist on the lower branch.

## Coverage table (complete maps)

| id | list item | PASS | ERR | FOLD | cov% | Δcov | peak dB | pump_s |
|----|-----------|-----:|----:|-----:|-----:|-----:|--------:|-------:|
| **c04_baseline_prod** | 0.4/0.5 control | 1722 | 204 | 574 | **68.9** | — | 26.55 | 4691 |
| r10_baseline_bridge | 19/20 bridge | 1722 | 204 | 574 | 68.9 | **+0.0** | 26.55 | 4607 |
| t02_backbone_secant | 3 + 9 (center-out) | 1698 | 207 | 595 | 67.9 | −1.0 | 26.55 | 4703 |
| t03_backbone_ltr | 2 + 9 (ltr) | 1698 | 207 | 595 | 67.9 | −1.0 | 26.55 | 4230 |
| p07_backbone_portfolio | 16 best | 1696 | 206 | 598 | 67.8 | −1.0 | 26.55 | **4034** |
| p08_portfolio_ranked | 16 ranked | 1696 | 206 | 598 | 67.8 | −1.0 | 26.55 | 4841 |
| p05_backbone_corner | 12 corner | 1684 | 205 | 611 | 67.4 | −1.5 | 26.55 | 3904 |
| c03_warm_copy | 0.3 copy | 1679 | 204 | 617 | 67.2 | −1.7 | 26.55 | 4957 |
| p06_backbone_plane | 13 plane | 1677 | 215 | 608 | 67.1 | −1.8 | 26.55 | 4891 |
| p04_backbone_freqsec | 10 freq secant | 1668 | 208 | 624 | 66.7 | −2.2 | 26.12 | 5698 |
| t13_serpentine | 5 serpentine | 1646 | 554 | 300 | 65.8 | −3.0 | 26.12 | **8799** |

## What we actually tested vs the plan

### §0 Controls
| item | config | outcome |
|------|--------|---------|
| 0.3 warm copy | c03 | ✅ 67.2% |
| 0.4/0.5 warm secant + full recovery | c04 | ✅ 68.9% (reference) |
| 0.1 cold fixed-λ / 0.2 linear-phasor adaptive | — | not a standalone map (used only as anchor/fallback inside every run) |

### §1 Traversal
| item | config | outcome |
|------|--------|---------|
| 2 low-power backbone (ltr) | t03 | ✅ 67.9% |
| 3 central-anchor outward | t02 | ✅ 67.9% |
| 5 serpentine | t13 | ✅ but **worst**: 65.8%, 554 err, 2× cost |
| 4 nearest-neighbour | t01 | ❌ incomplete (machine reset mid-run; then skipped) |
| 6 flood-fill | t14 | ❌ **OOM** (204/2500 pts) |
| 1 adjacent-freq · 7 wavefront | — | not run |
| backbone dirs rtl / two-ended | — | not run (only ltr + center-out) |

### §2 Predictors (all on backbone)
| item | config | outcome |
|------|--------|---------|
| 9 power secant | t02/t03 | ✅ 67.9% |
| 10 freq secant | p04 | ✅ 66.7% (worst predictor) |
| 12 corner | p05 | ✅ 67.4% |
| 13 plane | p06 | ✅ 67.1% |
| 16 portfolio best / ranked | p07 / p08 | ✅ 67.8% (best runtime: p07 4034 s) |
| 11 tangent · 14 exact 2-param · 15 polynomial | — | not run as maps (11 exists intra-cell only) |

### §3–4 Recovery & combos
| item | config | outcome |
|------|--------|---------|
| 19/20 bridge (on column) | r10 | ✅ **68.9% = baseline exactly, 0 cells recovered** |
| 28 backbone+portfolio+bridge ("expected best") | r11 | ❌ **OOM** (682 pts) |
| 21 ladder + combined fold-policy (§7 #45/46) | r12 | ❌ **OOM** (370 pts) |
| 25/26/27 backbone + secant/corner/portfolio | t02/p05/p07 | ✅ (see above) |
| 18 alt-parent · 29/30 wavefront combos | — | not run |

### §5–8 Advanced (intra-cell / solver-reuse / fold-aware / branch)
Implemented in `solver.py` (`--inproc-continuation adaptive_tangent/affine/ptc`,
`solve_arclength`, `--fold-follow`) but **not benchmarked** this campaign.
- 48 pseudo-arclength through fold (a15_arclength_fold) — **never ran**.
- 31–37 intra-cell, 38–43 solver reuse, 44–49 fold-aware (except baseline patience),
  50–53 deflation/branch — **not run**.

## Why the outcome is what it is

1. **The wall is a fold, not a bad guess.** The unsolved 29% (204 ERROR + 574
   SKIP_PAST_FOLD) is the *same region* in every map — `delta_grid.png` is ≈0
   everywhere. That region sits past a turning point where the lower-branch HB
   solution **ceases to exist**. See [[twpa-fold-vs-impedance]],
   [[foldskip-culls-broadband-2c]].

2. **Bridge proved it.** r10 walks parameters gradually parent→target. If those
   cells held a hard-but-existing lower-branch solution, the bridge would land it.
   It recovered **0** cells and reproduced c04 bit-for-bit. Warm start, predictor,
   and bridge all live on the lower branch → none can cross a fold. This is why
   §1+§2+§3 are all coverage-neutral (±3%).

3. **±3% jitter = which near-fold cells happen to converge**, not moving the fold.
   Best (backbone/column) vs worst (serpentine) differ only in how good the seed is
   *just before* the fold. Serpentine loses because it drags a near-fold
   top-of-column state sideways into the next column — a predictor aimed the wrong
   way at the worst-conditioned point → 554 errors, 8799 s.

4. **The methods that *could* cross the fold OOM'd.** r11 (portfolio+bridge) and r12
   (ladder+fold-policy) stack many extra full in-process solves per failed cell
   (portfolio ~7 candidates × bridge N steps × ladder M parents), each holding the
   full SuperLU signal factorization, precisely at the stiffest near-fold cells. The
   baseline's 14 s deadline + single reseed was tuned to *bound* memory; the recovery
   ladders remove that bound. See [[campaign-signal-backend-schur]].

## Bottom line / recommendation

- **Stop grinding traversal + predictor variants.** They cannot beat ~69% here —
  it's a branch limit, not a warm-start limit. Among them, **p07 (portfolio, best)
  is the cheapest at ~baseline coverage** (4034 s); or just keep **c04**.
- **The only levers that can raise coverage** are branch-crossing methods —
  pseudo-arclength through the fold (48/52) and deflation (50) — which need no
  lower-branch seed. These are the biggest, unproven items and didn't run.
- **Before any of that, fix the OOM** so r11/r12/a15 can complete: cap concurrent
  recovery solves, put the 14 s deadline on *each* bridge/ladder sub-solve, and drop
  `--signal-workers` during recovery. Without a memory bound the advanced configs
  can't finish regardless of merit.
