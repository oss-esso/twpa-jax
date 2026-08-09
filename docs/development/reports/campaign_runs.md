# Continuation-method campaign — what each run does

> **Audit note (2026-07-12):** this document describes intended configurations.
> In the original implementation, column traversal did not route through generic
> recovery (`r10` therefore did not execute bridge), non-column fail-fast still
> performed a final reseed, and ranked portfolio policy did not try all ranked
> guesses. The corrected follow-up is in `continuation_diagnostic_runs.md`.

Driver: `scripts/run_campaign.ps1`. Design: 2c (`outputs/ipm_python_design`).
Each config runs one gain map, then plots it. This doc says exactly what every
run computes and what each configuration is testing.

## Per-config pipeline

For each config the script runs, in order:

1. **`run_gain_map.py`** — solve the pump + gain map over the power×frequency grid
   with that config's method flags. Writes `map_arrays.npz` (gain / status / fold
   maps), `map_points.csv`, `map_summary.json` into
   `outputs/campaign_continuation_methods/<id>/`.
2. **`plot_gain_map.py --top-k 3 --save-pdf`** — render the maps and, for the top-3
   PASS cells by gain, run real S21 candidate sweeps (via `--ipm-dir`, the
   no-spectrum candidate path). Skipped when the script is run with `-NoPlot`.
3. **prune** — commented out currently, so all point solutions are kept.

Runtime knobs: `-Only id1,id2` runs a subset, `-NPower/-NFreq` override the grid
(e.g. `-NPower 3 -NFreq 3` for a fast smoke), `-NoPlot` skips plotting,
`-SignalWorkers N` sets gain-solve concurrency (default 6).

## Shared configuration (`$Common`) — identical for every run

Band / grid:
- Grid **50×50** (`-NPower`/`-NFreq` override).
- Pump power **−35 … −23 dBm**, pump frequency **7.5 … 8.5 GHz**.

Pump solve (same physics every run):
- `--inproc-pump-backend schur_cpu_mt` — eliminate linear-internal nodes, solve the
  retained pump system (constant per frequency).
- `--inproc-preconditioner real_coupled_fast` — near-direct real-coupled Jacobian
  preconditioner (fast GMRES).
- `--pump-mode-count 10 --nt 40` — JC odd pump basis `[1,3,…,19]`, 40 time samples.
- `--inproc-fold-predictor secant`, `--fold-skip-patience 4` — column-pass fold
  short-circuit: after 4 consecutive fails in a frequency column, skip the rest of
  that column (past-fold).
- `--inproc-max-newton 16`, `--inproc-solve-deadline-s 14` — per-cell Newton cap and
  a **14 s wall deadline** that aborts a stiff cell before it can finish and OOM the
  gain solve. (Required — an earlier run without it OOM'd.)
- `--inproc-schur-cache-size 2` — keep at most 2 per-frequency Schur partitions.

Gain solve (same every run):
- `--signal-backend direct --signal-solver superlu` — sparse SuperLU factorization of
  the full Floquet signal matrix. **Not schur** — the signal Schur partition
  densifies and uses more memory.
- `--sidebands 10` — 21-harmonic Floquet ladder.
- `--signal-detuning-mhz 100` — the single measured gain point sits 100 MHz below the
  pump (ws = wp − 100 MHz).
- `--no-signal-spectrum` — **one** gain solve per cell at that detuning, not an
  11-point spectrum ladder → ~10× faster gain, so many more configs per campaign.
  (`--signal-offset-*` are inert while non-spectrum.)
- `--signal-workers 6` — 6 concurrent gain factorizations. Lower with
  `-SignalWorkers` if free RAM is tight.

So **every run produces the same thing**: a 50×50 map of single-point gain (dB) at
100 MHz detuning plus per-cell solve status and the fold boundary. Runs differ only
in **how the pump map is traversed and warm-started** — i.e. how robustly and
quickly each cell converges, which sets how much of the band is solved vs skipped.

## The four mechanisms configs vary

- **Traversal** (`--traversal`) — order cells are visited and which solved neighbour
  seeds the next.
  - `column` (legacy): per-frequency column, low→high power, no cross-column reuse.
  - `backbone`: solve the lowest-power frequency row first, then run each power
    column upward from it (`--backbone-direction` sets the row order).
  - `nearest`: visit any order, seed from the nearest solved cell in either axis.
  - `serpentine`: boustrophedon power sweep, alternating direction per column.
  - `floodfill`: grow outward from a central low-power seed.
  - All non-`column` traversals share one solved-state store across both axes and run
    single-process (force `--frequency-chunk-size 0`).
- **Predictor** (`--predictor`) — how the initial guess for a cell is built from
  solved neighbours: `copy` (0th order), `power_secant` / `freq_secant` (1st order
  along one axis), `corner` / `plane` (2-D extrapolation), `portfolio` (build several
  candidates, rank by target residual; `--portfolio-policy best|ranked`).
- **Recovery** (`--recovery`) — what happens when a cell fails: `reseed` (fresh cold
  solve), `alt_parent` (retry from other neighbours), `bridge` (physical-parameter
  continuation from the best parent along (P,f); `--bridge-mode`), `ladder`
  (residual-rank parents, then bridge from the best).
- **Fold policy** (`--fold-policy`) — when a failed cell counts toward the fold
  short-circuit: `patience` (legacy, every fail counts), `combined` (only after
  cross-axis + portfolio + bridge all fail), `arclength` (round the fold with
  pseudo-arclength continuation).

## The 16 runs

| id | traversal | predictor | recovery / fold | What it tests |
|----|-----------|-----------|-----------------|---------------|
| **c04_baseline_prod** | column | (fold secant) | patience | **Control** = current production run. The reference everything else is compared against. |
| **c03_warm_copy** | column | (fold none) | patience | Control with no fold predictor — plain copy warm start down each column. Isolates the value of the secant fold predictor. |
| **t01_nearest_copy** | nearest | copy | reseed | Cheapest cross-axis traversal: nearest solved neighbour, plain copy. Does 2-D neighbour reuse alone beat the column pass? |
| **t02_backbone_secant** | backbone (center_out) | power_secant | reseed | Backbone row then upward columns with a power-secant predictor, started from the middle frequency. Core proposed traversal. |
| **t03_backbone_ltr** | backbone (ltr) | power_secant | reseed | Same as t02 but backbone row solved left→right. Tests backbone start-direction sensitivity. |
| **t13_serpentine** | serpentine | power_secant | reseed | Boustrophedon power sweep — reuse the top-of-column state as the next column's seed. |
| **t14_floodfill_portfolio** | floodfill | portfolio (best) | reseed | Flood outward from a central low-power seed, portfolio-ranked guess. Tests region-growing order. |
| **p04_backbone_freqsec** | backbone | freq_secant | reseed | Backbone with a frequency-axis secant instead of power. Which secant direction predicts better. |
| **p05_backbone_corner** | backbone | corner | reseed | 2-D corner extrapolation from three solved neighbours. |
| **p06_backbone_plane** | backbone | plane | reseed | Least-squares plane through solved neighbours (higher-order 2-D guess). |
| **p07_backbone_portfolio** | backbone | portfolio (best) | reseed | Portfolio predictor, try only the lowest-residual candidate. |
| **p08_portfolio_ranked** | backbone | portfolio (ranked) | reseed | Portfolio, try candidates in ascending-residual order until one converges. Robustness vs p07's speed. |
| **r10_baseline_bridge** | column | (fold secant) | **bridge** (adaptive) | Production column pass but recover failed cells with bridge continuation. Does bridge recover fold-edge cells the baseline drops? |
| **r11_combined_best** | backbone | portfolio | **bridge** (adaptive) | Backbone + portfolio + bridge recovery. **Expected best** overall. |
| **r12_combined_foldpolicy** | backbone | portfolio | **ladder** + `--fold-policy combined` | As r11 but ladder recovery and the combined fold policy (only count a fail after full recovery also fails) — should push the fold boundary furthest. |
| **a15_arclength_fold** | backbone | portfolio | **ladder** + `--fold-policy arclength` | As r12 but round the fold with pseudo-arclength continuation. Experimental on the stiff 2c device. |

The trailing `# n,m` comments in the script map each id to the rows of the expanded
test matrix in `docs/reports/pump_map_continuation_methods.tex`.

## How to read the results

Because every run computes the same single-point gain map, compare configs on:
- **Fold-skip coverage** — how much of the −35…−23 dBm × 7.5…8.5 GHz band each config
  actually solves vs marks past-fold/skipped (`status` / fold maps). The methods win
  by solving more of the high-power operating region the column baseline skips.
- **Convergence cost** — Newton iterations / solve time per cell.
- **Single-point gain** in the solved region.

Then re-run the winning config with `--signal-spectrum` for full candidate S21
bandwidth plots.
