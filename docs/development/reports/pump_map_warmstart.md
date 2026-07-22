# Warm-Started Pump/Gain Map (exp10)

`experiments/exp10_full_ipm_pump_map_warmstart.py` runs an exp08 (pump
harmonic-balance) + exp09 (linearized gain) map over a pump-power × pump-frequency
grid and benchmarks two traversal strategies, gating the fast one against the
trusted one.

## Strategies

- **cold** — every point solved from scratch: zero initial guess + fixed
  20-step continuation (the legacy path). Trusted reference.
- **warmstart** — each frequency column is traversed in increasing pump power.
  The first point of a column is seeded with `--initial-guess linear_phasor`
  and solved with adaptive continuation; every later (higher-power) point
  warm-starts from the previous converged pump solution via
  `--promote-from-pump-dir` (one full-scale Newton solve, no continuation). A
  point only chains forward if it converged; otherwise the next point re-seeds.
- **both** — runs cold then warm and emits a PASS/FAIL gate (the validation run).

## Gate

Warm start is accepted only if:

1. every warm point converged (pump `VALID_CONVERGED` + gain `VALID_SOLVED`),
2. per-point gain agrees with the cold reference within `--gate-gain-db`
   (default `0.01` dB), and
3. the warm pass is faster in total pump runtime.

For a large warm-only map, `--gate-spotcheck N` recomputes `N` points cold
(corners + center first) after the warm pass and folds their gain drift into the
gate — so the big run stays guarded without a full cold map.

## Physics correctness: matching JosephsonCircuits

This is the IPM JTWPA (`Harmonia/IPM_JTWPA.jl`, 6×418 = 2508 JJs). To reproduce
the JC `hbsolve` gain the map **must** use the JTWPA settings, all now baked into
`exp10` defaults:

- **pump basis** `positive_odd_jc`, K=10 → modes [1,3,…,19], `nt 40`
  (dense [1,2,3] truncates the odd pump content and kills the gain),
- **`--sidebands 10`**, `--gamma-nt 96`,
- **`--pump-current-jc-scale 2.0`** — JC's frequency-domain port current maps to a
  physical `2 I cos(wt)` drive under the positive-phasor reconstruction, so the
  injected pump must be doubled (the documented "pump scale 2"; see
  `experiments/exp13_jtwpa_fast_scale2.py:27`). This is a convention factor, not
  over-driving: at JC's 1.5 Ic the JJ phase is 0.63 rad (JJ current 0.59 Ic, sub-Ic).

Validation at JC's operating point (external −25 dBm → physical 1.5 Ic, pump
7.9 GHz, signal 6 GHz):

| | Python (exp10) | JC `hbsolve` |
|---|---|---|
| gain at 6 GHz | **6.29 dB** | **6.28 dB** |

JC curve (I/Ic=1.5, pump 7.9 GHz): 6.28 / 8.71 / 10.52 / 11.12 / 10.92 dB at
6.0 / 6.5 / 7.0 / 7.5 / 8.0 GHz; peak 11.86 dB at 8.05 GHz.

> **Superseded:** the earlier `outputs/exp10_pump_map_warmstart_5x5/` run used the
> pre-fix defaults (dense H=3, 2 sidebands, no 2× scale) and reported ~0–1.4 dB
> gains — physically wrong. Re-run the maps with the corrected defaults below.

## Maps

Default external window `-35..-25` dBm (35 dB attenuation, 50 Ω) spans physical
pump ≈ 0.5–1.6 × median Ic, i.e. the JTWPA gain ridge from onset (~0 dB) up to
~7 dB at the 6 GHz readout (higher at the 7.5–8 GHz signal peak). The cold pump
solve is now ~6–33 s/point (10 odd modes, nt 40), so warm-start (the point of
`exp10`) matters: warm points reuse a neighbour and cost a single Newton solve.

### 5×5 cold-vs-warm validation (`outputs/exp10_pump_map_warmstart_5x5/`, corrected)

`mode=both`, default axes/settings, signal 6 GHz.

| metric | value |
|---|---|
| gate verdict | **PASS** |
| cold points converged | 24 / 25 (1 fixed-continuation failure at −27.5 dBm/6.5 GHz) |
| warm points converged | **25 / 25** (warm-start recovered the cold failure) |
| compared pairs | 24 |
| max gain drift | **9.5e-8 dB** (gate 0.01) |
| cold pump runtime | 306.0 s |
| warm pump runtime | 50.2 s |
| **pump speedup** | **6.1×** |
| gain range | −0.5 … 6.87 dB (onset → near JC's 1.5 Ic point) |
| wall time | 433 s |

Warm-start is both faster (6×) and **more robust** — the single point cold
diverged on is recovered because warm-start gives a far better initial guess than
zero. A full 35×35 cold map is infeasible (~hours); run it
`mode=warmstart --gate-spotcheck N` (warm-only, gated against N cold recomputes).

### 35×35 warm-only gated run (`outputs/exp10_pump_map_warmstart_35x35/`)

`mode=warmstart --gate-spotcheck 5`, 1225 points, default axes/settings.

| metric | value |
|---|---|
| gate verdict | **PASS** |
| warm converged | **1210 / 1225 = 98.78%** (threshold 0.98) |
| spot-check drift vs cold | **7.9e-10 dB** (4 comparable pairs) |
| **per-point pump speedup** | **10.3×** (cold mean 13.9 s, warm mean 1.35 s; warm median 0.72 s) |
| gain range | −0.59 … 7.91 dB |

The 15 non-converged points cluster at high power and pump freq ≈ 6.29–6.35 GHz
(a stiff coupler-resonance region) where even the graded seed solve struggles;
they appear as NaN holes in `gain_db_warm`.

**Gate notes (fixed after the first run):** the speedup is now **per point**
(mean converged pump time), since comparing a 5-point cold spot-check total to a
1225-point warm total is meaningless; and the convergence gate is a **fraction**
(`--gate-min-converged-frac`, default 0.98) so sparse stiff failures don't
invalidate the map. `run_warm_pass` now **retries a failed warm-start from a fresh
linear_phasor+adaptive seed** (with fixed-continuation fallback) to recover stiff
points. Gate tests: `tests/test_exp10_gate.py`.

## Reproduce

```bash
# 5x5 cold-vs-warm validation
python experiments/exp10_full_ipm_pump_map_warmstart.py \
    --mode both --n-power 5 --n-frequency 5 \
    --outdir outputs/exp10_pump_map_warmstart_5x5 --overwrite

# 35x35 warm-only gated run
python experiments/exp10_full_ipm_pump_map_warmstart.py \
    --mode warmstart --n-power 35 --n-frequency 35 --gate-spotcheck 5 \
    --outdir outputs/exp10_pump_map_warmstart_35x35 --overwrite
```

## Related

- Single-point speedup ablation: `docs/reports/pump_speedup_single_point.md`
  (recommended opt-in mode `linear_seed_adaptive`, ~19× single-point pump
  speedup).
- exp08 opt-in flags: `--initial-guess linear_phasor`, `--continuation-mode
  adaptive`, `--promote-from-pump-dir`.
- Tests: `tests/test_exp08_seed_adaptive_warmstart.py`.
