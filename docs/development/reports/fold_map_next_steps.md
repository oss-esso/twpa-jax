# IPM fold gain map — result & next steps

Working note capturing the 50×50 trailing-signal gain map over the fold and the
concrete levers to push it further. Picked up again after a detour on
Harmonia.jl.

## What we have

50×50 IPM JTWPA gain map with the **trailing signal** convention
(`ws = wp − 100 MHz` per cell, verified Δ = −0.1 GHz), fastest solver
(`schur_cpu_mt` + `real_coupled_fast` + secant predictor), run **through the
fold**:

- Grid: pump **7.0–8.0 GHz × −30 → −20 dBm**.
- **Peak gain 27.1 dB @ 7.27 GHz, −20.4 dBm** (right at the fold edge).
- Four gain lobes = the IPM periodic-coupler phase-matching resonances
  (~7.05, 7.25, 7.5, 7.75 GHz), each rising with pump power toward the fold.
- Convergence **2118/2500 = 85 %**, and the un-converged cells are exactly the
  over-fold zone:

  | pump power band | solved |
  |---|--:|
  | −30 → −25 dBm (sub-fold) | 1250 / 1250 (100 %) |
  | −25 → −22 dBm | 666 / 750 (89 %) |
  | −22 → −20 dBm (over the fold) | 190 / 450 (42 %) |

The convergence frontier hugs the **top of each gain lobe** — the highest-gain
cells sit right at the edge of solvability, which is the expected signature of a
parametric fold (and is well past where JosephsonCircuits.jl diverges entirely,
so this is Python-only territory; see
[`runtime_vs_josephsoncircuits.md`](runtime_vs_josephsoncircuits.md)).

Artifacts (under the gitignored `outputs/`, regenerable):

- Map: `outputs/exp10_pump_map_trailing_50x50_m30_m20/` (`map_points.csv`,
  `map_summary.json`).
- Figure: `outputs/ipm_50x50_trailing_gain_map.png`.

Reproduce:

```bash
python experiments/exp10_full_ipm_pump_map_warmstart.py --executor inprocess \
    --mode warmstart --inproc-pump-backend schur_cpu_mt \
    --inproc-preconditioner real_coupled_fast --inproc-fold-predictor secant \
    --n-power 50 --n-frequency 50 \
    --pump-power-min-dbm -30 --pump-power-max-dbm -20 \
    --pump-freq-min-ghz 7.0 --pump-freq-max-ghz 8.0 \
    --signal-detuning-mhz 100 \
    --outdir outputs/exp10_pump_map_trailing_50x50_m30_m20
```

## Next steps (ranked)

### 1. Pseudo-arclength continuation to push the over-fold frontier

The secant power-axis predictor extrapolates *along the injected-current axis*,
so it **overshoots the turning point** at the fold — which is why the −22 → −20
band collapses to 42 %. A **pseudo-arclength predictor/corrector** parametrises
the branch by arclength instead of by power, so it can round the fold instead of
running into it. Expected payoff: recover a large fraction of the grey −22 → −20
cells and extend the gain ridge past 27 dB. This is the one remaining orthogonal
lever already flagged in the pump-solver catalog and the `schur-pump-backend`
notes.

Scope: add `--inproc-fold-predictor arclength` alongside `secant` in
`exp10_full_ipm_pump_map_warmstart.py`; the corrector reuses the existing
`real_coupled_fast` factorization, so per-step cost is unchanged. Validate on a
single frequency column (e.g. 7.27 GHz) before a full re-sweep.

### 2. Convergence-frontier contour + peak-gain ridge export

Cheap, no solver work:

- Overlay the **converged/failed boundary** as a contour on the heatmap so the
  fold frontier is explicit rather than implied by the grey mask.
- Export the **peak-gain ridge** (best pump power per frequency, and its gain +
  signal frequency) as a CSV — the practical "operating line" for the device.

### 3. (Optional) Finer power resolution at the fold only

If arclength lands, a targeted −22 → −20 dBm strip at finer power steps
(e.g. 0.1 dBm) would resolve the ridge shape near the turning point without
paying for a full 50×50 re-run.

## Status

- Map + figure generated and inspected; committed this note.
- Deferred while we work on Harmonia.jl. Resume at step 1 (arclength predictor).
