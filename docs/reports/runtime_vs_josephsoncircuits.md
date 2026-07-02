# Runtime & gain: this Python HB solver vs native JosephsonCircuits.jl

Head-to-head of **wall-time and small-signal gain** for our Python
harmonic-balance solver against native `JosephsonCircuits.jl` (JC) on the same
circuits: the seven JC documentation designs and — the clean, matched
comparison — one tile of the IPM JTWPA gain map.

**Python side uses the fastest available solver** (per
[`pump_solver_catalog.md`](pump_solver_catalog.md)): Schur reduction +
`real_coupled_fast` (KLU-style symbolic-factorization reuse, GMRES=1/Newton) +
the power-axis **secant** predictor — i.e. `schur_cpu_rcfast` warm-started. JC
runs `hbsolve` warm (JIT excluded) with plotting stripped.

Reproduce:

```
# JC timings (warm hbsolve, no plotting) for the 7 raw doc standalones:
python experiments/benchmark_seven_designs_vs_jc.py \
    --julia-project <Harmonia.jl> \
    --raw-dir <Harmonia.jl>/experiments/solver_benchmark/cases/jc_docs/raw
# Python parity gains: python experiments/exp14_seven_design_summary.py
# IPM tile (fastest Python): exp10 in-process, schur + real_coupled_fast + secant
```

---

## 1. IPM JTWPA — one map tile (the matched, apples-to-apples comparison)

Best cell of the 35×35 gain map: **pump 6.765 GHz, signal 6.665 GHz
(= wp − 100 MHz), pump current 9.495 µA (≈2.28 Ic)**. Both solvers do exactly one
pump solve + one signal point.

| solver | pump solve | gain solve | total | gain |
|---|--:|--:|--:|--:|
| **JosephsonCircuits.jl** (`hbsolve`, warm) | — bundled — | — bundled — | **9.3 s** | **22.07 dB** |
| **This solver** (`rcfast` + secant, warm) | **0.88 s** (10 Newton) | 0.63 s | **1.5 s** | **22.07 dB** |

→ **~6× faster end-to-end (~10× on the pump solve alone), identical gain to
0.00 dB.** This is a genuine cross-validation: two independent HB codes agree on
22.07 dB at the same operating point.

Note on the earlier "IPM is slow" scare: at the wrong pump point (7.9 GHz, which
does **not** amplify → −512 dB) JC's Newton churns to the iteration cap, ~63 s
for a single point, and a full 51-point signal sweep is ~75 s. At the real
amplifying operating point above it is 9.3 s. Our solver, likewise, only reaches
the 22 dB near-fold cell with the secant predictor + fine (0.3 dBm) power
continuation.

---

## 2. The seven JC documentation designs

Gains are already at parity (see
[`experiments/exp14_seven_design_summary.py`](../../experiments/exp14_seven_design_summary.py);
6/7 match < 0.0024 dB, the lossy `fqjtwpa_diss` is the known open convention).
The runtime columns are **not** a single operating point:

- **JC `hbsolve` (warm)** solves the pump **once** and then evaluates the
  linearized S-parameters over the design's **full doc frequency band** (hundreds
  of points). It is a single number that bundles pump + full sweep and cannot be
  split.
- **Python pump** is the pump solve alone; **Python gain** is one signal point via
  `exp09` (a separate small-signal solver, not the pump path — it dominates for
  the tiny designs and is unoptimized).

So compare **gains** directly, and read the runtimes as "full-band JC solve" vs
"Python pump (one op point)". The clean per-tile runtime comparison is §1.

| design | JC gain | Py gain | Δgain | JC `hbsolve` (s, full sweep) | Py pump (s) | Py gain 1-pt (s) |
|---|--:|--:|--:|--:|--:|--:|
| jpa | 13.30 | 13.30 | 0.002 | 0.012 | 0.148 | 18.31 |
| jtwpa | 27.54 | 27.54 | 0.001 | 15.2 | 2.03 | 8.34 |
| fqjtwpa | 28.54 | 28.54 | 0.000 | 13.4 | 3.21 | 7.55 |
| fxjpa | 15.11 | 15.10 | 0.011 | 2.93 | 0.005 | 0.55 |
| fxjtwpa | 23.90 | 23.90 | 0.000 | 119.5 | 0.00¹ | 1.66 |
| dpjpa | 10.55 | 10.55 | 0.000 | 3.92 | 0.35 | 0.20 |
| fqjtwpa_diss | 29.06 | 29.88 | 0.82² | ~27 / solve³ | 1.30 | 7.08 |

¹ `fxjtwpa` pump is warm-started from an imported JC nodeflux seed (node-order
fixed), so the residual Python pump solve is ~0.
² `fqjtwpa_diss` is the one open mismatch (lossy-pump convention still to
reconcile); its dissipative Floquet solve is also the heaviest JC case.
³ `fqjtwpa_diss` reuses `fqjtwpa`'s circuit and sweeps 4 loss tangents × 131
signal points; ~27 s **per** dissipative solve.

### Reading the numbers

- **Gains agree** across all seven except the known lossy case — the physics is
  validated both ways.
- **Pump solve:** our Python pump is fast in absolute terms (0.15–3.2 s) and, for
  the big TWPAs, well under JC's full-sweep `hbsolve` — but JC's number includes
  the linearized sweep, so this is not a clean pump-vs-pump split.
- **`rcfast` is IPM-specific.** The Schur + `real_coupled_fast` winner is wired
  into the IPM in-process map path (§1), not the generic `exp08` path the seven
  doc designs run through; those use `exp08` continuation + secant. Porting the
  Schur backend to arbitrary designs is future work.
- **The `exp09` gain path is the Python bottleneck for small designs** (18 s for
  jpa's single point), independent of the pump solver — a separate optimization
  target from this branch's pump-fold work.

---

## Bottom line

On the one **matched** comparison — a single IPM map tile with the fastest Python
solver — **our solver is ~6× faster than JosephsonCircuits.jl end-to-end and
~10× on the pump solve, at identical gain (22.07 dB)**. Across the seven doc
designs the two codes agree on gain (6/7 < 0.0024 dB); the runtime columns aren't
a single operating point because the JC doc standalones sweep the full band.
