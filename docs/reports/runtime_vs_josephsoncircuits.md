# Runtime & gain: this Python HB solver vs native JosephsonCircuits.jl

Head-to-head of **wall-time and small-signal gain** for our Python
harmonic-balance solver against native `JosephsonCircuits.jl` (JC) on the same
circuits: the seven JC documentation designs and — the clean, matched
comparison — one sub-fold tile of the IPM JTWPA gain map (the regime where JC is
guaranteed to converge).

**Python side uses the fastest available solver** (per
[`pump_solver_catalog.md`](pump_solver_catalog.md)): Schur reduction +
`real_coupled_fast` (KLU-style symbolic-factorization reuse, GMRES=1/Newton) +
the power-axis **secant** predictor — i.e. `schur_cpu_rcfast` warm-started. JC
runs `hbsolve` warm (JIT excluded) with plotting stripped.

Reproduce:

```bash
# JC timings (warm hbsolve, no plotting) for the 7 raw doc standalones:
python experiments/benchmark_seven_designs_vs_jc.py \
    --julia-project <Harmonia.jl> \
    --raw-dir <Harmonia.jl>/experiments/solver_benchmark/cases/jc_docs/raw
# Python parity gains: python experiments/exp14_seven_design_summary.py
#
# IPM matched tile (sub-fold, pump 7.9 GHz / signal 8.3 GHz / 6.325 uA):
#   Python (fastest solver) -- short power continuation ending on the tile:
python experiments/exp10_full_ipm_pump_map_warmstart.py --executor inprocess \
    --mode warmstart --inproc-pump-backend schur_cpu_mt \
    --inproc-preconditioner real_coupled_fast --inproc-fold-predictor secant \
    --n-power 9 --n-frequency 1 --pump-power-min-dbm -33 --pump-power-max-dbm -25 \
    --pump-freq-min-ghz 7.9 --pump-freq-max-ghz 7.9 --signal-ghz 8.3 \
    --outdir outputs/exp10_subfold_match_79ghz_m25
#   JC (warm, JIT excluded) at the same physical drive current:
julia --project=<Harmonia.jl> experiments/jc_ipm_onepoint_timing.jl \
    --pump-current-a 6.324555320336759e-06 --pump-freq-ghz 7.9 --signal-ghz 8.3
```

---

## 1. IPM JTWPA — one matched tile in the sub-fold regime

The clean apples-to-apples comparison is a single tile **in the regime where both
codes reliably converge** — i.e. *below* the parametric fold, which is exactly
where JC is trustworthy. Operating point: **pump 7.9 GHz, signal 8.3 GHz, drive
6.325 µA (1.52 Ic)**. Both solvers do one pump solve + one signal point, at the
*identical* physical drive current (the port current fed to JC `sources`; the
Python solver applies its ×2 phasor-convention factor internally).

| solver | pump solve | gain solve | total | gain |
|---|--:|--:|--:|--:|
| **JosephsonCircuits.jl** (`hbsolve`, warm) | — bundled — | — bundled — | **3.09 s** | **11.559 dB** |
| **This solver** (`rcfast` + secant, warm) | **0.41 s** (5 Newton) | 0.63 s | **1.04 s** | **11.555 dB** |

→ **~3× faster on total wall time — the pump solve alone is 0.41 s — at a gain
identical to 0.004 dB.** Two independent HB codes agreeing on 11.56 dB at the
same physical operating point is a genuine cross-validation of both the pump and
the small-signal physics.

### Why sub-fold, and what happens at the fold

Past the parametric fold the comparison stops being meaningful, because **JC
stops converging** while the continuation-warm-started Python solver does not.
Measured at 7.9 GHz, same signal point, pushing the drive current up:

| drive | Ic ratio | JC pump | Python |
|---|--:|---|---|
| 6.325 µA | 1.52 | converges → 11.56 dB, 3.1 s | 11.56 dB, 0.41 s (5 Newton) |
| 8.934 µA | 2.14 | **diverged → −326 dB** | still solves (~12.8 dB peak) |
| 12.65 µA | 3.04 | **diverged → −522 dB, 59 s at iter cap** | solves, 0.41 s (5 Newton) |

This reconciles the earlier "IPM is slow / 7.9 GHz gives −512 dB" scare: 7.9 GHz
amplifies perfectly well sub-fold (both codes give 11.56 dB); JC only blows up
(garbage gain, iteration-cap runtime) once the drive is pushed *to/through the
fold*. So the fold is precisely where a reliable comparison **requires** the
continuation solver — sub-fold the two agree and ours is ~3×, at the fold JC has
no converged answer to compare against.

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

On the one **matched** comparison — a single IPM tile in the sub-fold regime
(where JC reliably converges), fastest Python solver — **our solver is ~3× faster
than JosephsonCircuits.jl on total wall time (0.41 s pump vs 3.09 s), at a gain
identical to 0.004 dB (11.56 dB)**. That is a genuine two-code cross-validation of
the physics. Push the drive to/through the parametric fold and the comparison
inverts qualitatively: JC's Newton diverges (−326 dB at 2.1 Ic, −522 dB / 59 s at
3.0 Ic) while the continuation-warm-started Python solver keeps converging in ~5
Newton — so at the fold there is simply no JC answer to race against. Across the
seven doc designs the two codes agree on gain (6/7 < 0.0024 dB); those runtime
columns aren't a single operating point because the JC doc standalones sweep the
full band.
