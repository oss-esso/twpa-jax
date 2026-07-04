# Harmonia → twpa_jax Port: Ripple Pump Placement

This documents the port of Harmonia.jl's `ripple_pump_placement` /
`ripple_map_crosscheck` workflow into `twpa_jax`, and the findings that came out
of it. For the workflow reference (CLI, knobs, outputs) see
[`ripple_pump_placement.md`](ripple_pump_placement.md); this file is the *why*
and the *what we learned*.

## What was ported

| Harmonia.jl (source) | twpa_jax (target) | Notes |
| -------------------- | ----------------- | ----- |
| `scripts/ripple_pump_placement.jl` | `experiments/exp17_ripple_pump_placement.py` | passive ripple → +120° placement → Ic ladder → gain sweep |
| `scripts/ripple_map_crosscheck.jl` | `experiments/exp17_ripple_map_crosscheck.py` | verify a pump map's top cells against the ripple |
| `scripts/plot_ripple_pump_placement.py` | `experiments/plot_ripple_pump_placement.py` | one PNG per operating point |
| (shared logic, inline in the `.jl`) | `experiments/ripple_common.py` | design build, passive 4-port S, peak/period, snap-to-120, pump + gain subprocess helpers |

The port reuses the existing twpa_jax stack unchanged: `exp07` (design build),
`exp08` (pump HB solve), `exp09` (linearised gain sweep). Both the 2-coupler
(cached coupler) and 3-coupler (re-optimised −20 dB @ 10 GHz) designs are
supported via `ripple_common.build_design`.

## Why port at all

The maps solve **much faster** in twpa_jax than in Harmonia and reach the
current fold, so the whole placement + cross-check loop runs in minutes rather
than hours. That made it worth reproducing the Harmonia workflow on the faster
stack rather than continuing to iterate in Julia.

## Key findings

### 1. Passive S42 needs no pump solve

The passive (pump-off) 4-port S-matrix is computed directly from the exp07
design matrices. The exp09 linear block `D(ω) + K̂₀` already carries the
Josephson inductance through `γ̂₀ = Ic/φ₀`, so **one sparse LU per frequency**
(solving every source column) yields a genuine passive S-parameter under the
same Norton-port normalisation exp09 uses for gain,
`S_ij = 2 V_i /(I_j Z0)` (`−1` on the diagonal). ~5 ms/freq; a 1401-point fine
grid is a few seconds. The 2c ripple reproduces Harmonia: ≈180–190 MHz period,
`S42` peaks at 8.02 / 8.17 / 8.32 / 8.55 GHz.

### 2. Accept the fold edge, not the strict convergence flag

This was the decisive tuning. The pump HB is stiff near the current fold, and
exp08's strict `final_status == VALID_CONVERGED` demands the Newton residual hit
`1e-9` at full drive. At the fold edge a **physically fine** solve (bounded node
flux, ~20 dB gain) plateaus slightly above that and gets flagged `FAIL`. Judging
by that flag alone throws away the best operating point.

The ladder therefore keeps the strongest **accepted** rung, where:

> **accepted** = the continuation reached the full requested current
> (`source_scale == 1`) **and** node flux is bounded (`ψ/φ₀ < 1e3`).

The full-scale test rejects rungs that stalled *below* the requested current
(genuinely past the fold), and the flux bound rejects runaways. This mirrors
Harmonia's flux-based acceptance, with the added `reached_full_scale` guard
specific to twpa_jax's continuation. Concretely, 2c at `fp = 7.86 GHz` gives
5 dB at the strict 3 Ic but **20.6 dB** at the fold-edge 4 Ic.

The pump solve uses the JC positive-odd phasor basis (`positive_odd_jc`, K=10,
Nt=40) with **adaptive continuation + secant predictor** from a `linear_phasor`
seed — the trusted pump-map reference settings. Fixed-step continuation stalls
at fp-specific fold tongues.

### 3. The map cross-check exposes warm-start-only cells

Run against the real 50×50 trailing-signal map
(`outputs/exp10_pump_map_trailing_50x50_m30_m20/`, headline cell 27.1 dB @
7.27 GHz, −20.4 dBm), the cross-check walks the top-gain pool, snaps each cell's
`fp` to the nearest +120° target, and **cold re-solves** at the cell's own pump
current. Result: every headline cell at **4.0–4.2 Ic is past the cold fold** —
the map reached those gains by *warm-starting* across a fold a cold solve cannot
cross, so they fail `reached_full_scale` even with adaptive+secant. Only the
lower ~3.68 Ic cell verifies:

| Cell | Map gain | Map `fp` / offset | Snapped `fp` | `Ic` | Re-solved `S21` | flux | +120° |
| ---- | -------- | ----------------- | ------------ | ---- | --------------- | ---- | ----- |
| pool 7 | 22.3 dB | 7.286 GHz (+57°) | 7.319 GHz | 3.68 | **21.1 dB** @ 7.40 | 0.77 | ✅ |
| pool 1 | 27.1 dB | 7.265 GHz (+19°) | 7.319 GHz | 4.04 | — (past cold fold) | 0.68 | ✗ |
| pool 2 | 26.1 dB | 7.490 GHz (+117°) | 7.491 GHz | 4.23 | — (past cold fold) | 0.72 | ✗ |

**Takeaway (matches Harmonia):** a map cell's high gain lives on a warm-started
branch past the cold current fold at its *specific* `fp`; the cold-reproducible,
+120°-coherent operating point is **~21 dB at 3.68 Ic**. Judge map candidates by
the re-solved fold-edge gain, never by the raw warm-started map cell.

### 4. 3c is a genuinely distinct design

`build_design("3c")` overrides `array_length=648`, `arrays_per_dc=2`,
`coupling_dB=−20`, `coupler_freq_hz=10e9`, `coupler_mode="optimize"`. The coupler
re-optimises to −18.45 dB and the circuit is 3888 JJs / 9662 nodes (vs 2c's
2508 / 6446) — a distinct topology, not a rescale. Its passive ripple produces
clean in-band +120° placements.

## Verification status

- **Passive path:** validated — design build ~0.1 s, passive S ~5 ms/freq; 2c
  ripple period and peak positions match Harmonia.
- **2c placement:** validated end-to-end — 20.6 dB fold-edge point.
- **2c cross-check:** validated against the real 50×50 map — 1 coherent
  survivor (21.1 dB), headline cells correctly identified as warm-start-only.
- **3c:** passive build validated (distinct coupler/topology). The full 3c pump
  ladder + cross-check has not yet been run end-to-end on this stack.
