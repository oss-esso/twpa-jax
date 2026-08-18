# Live tracker — high-power solver campaign (opened 2026-08-16)

Living status board. Update on every agent return and every verdict change.
Purpose: stop re-deriving settled facts and stop losing track of dispatched
work. Six hours were lost on 2026-08-15/16 to exactly that.

---

## 0. North star

> A solver that solves pump AND signal at high power, before chaos and during
> chaos. Ideally modifying HB, since HB is 1-20 s/point against FDTD's
> 300-900 s/point.

Before proposing or accepting any task, state which part of this it advances.
If none, say so.

---

## 1. Agents and dispatch status

| # | task | status | returns |
| --- | --- | --- | --- |
| A1 | 2c RCSJ stability column (G0-G6) | **DONE, verified** | `outputs/chaos/2c_rcsj_stability_column/` |
| A2 | Spurious-root filter + recompute G3/G4/G5 + correct G6 | **DONE, verified** | `hill_filtered_reduction.json`, `gate_report.json` |
| A3 | Task A jtwpa dt screen + Task B comb/null control | **DONE, verified** | `taskB_comb_test.csv`, `dense_m27p8000_dt005/` |
| A4 | Hill branch re-scan: repair branch hole, find crossing, predicted-rho mask | **prompt delivered, dispatch unconfirmed** | — |

Rule established by the record: **one agent per workstream, serialized.** Two
agents on the same workstream previously produced a contradictory
`-24.0000 dBm` duplicate differing by a factor of 560.

---

## 2. Established this session — do not re-derive

### 2c RCSJ column
- G0 fixture integrity PASS 12/12. The `INVALID_HB_FIXTURE` failure that killed
  Phase 3 three times is fixed by generating checkpoints on the RCSJ circuit.
- G1 PASS: HB tracks FDTD to max 0.586 dB below the transition.
- G2 PASS: RCSJ at `R/Rn = 1e5` shifts gain by **0.0012 dB** and `r_j` by
  **5.0e-5**. The regularizer is benign.
- **First decidable Hill spectrum on this circuit**: `max|lambda|` 0.954-0.999
  at `R/Rn` 1e4/1e5, against the lossless `1.000000000001` measured at a
  provably stable power.

### The Hill branch (R/Rn = 1e5, after DC/pump-harmonic filtering)

| I [A] | \|lambda\| | \|zone\| GHz | rho = \|zone\|/f_p | phase/pi |
| ---: | ---: | ---: | ---: | ---: |
| 5.2326e-06 | 0.974135 | 0.6721 | 0.08508 | +0.1702 |
| 5.5233e-06 | 0.978307 | 0.6819 | 0.08632 | -0.1727 |
| 6.1047e-06 | *hole* | (0.3955 off-branch) | — | — |
| 6.3954e-06 | 0.993518 | 0.7143 | 0.09042 | -0.1808 |
| 6.6861e-06 | **0.999230** | 0.7271 | 0.09204 | -0.1841 |

`phase_rad = 2*pi*zone_frequency/f_p` holds exactly, so rho and
`f_p - root_frequency` are the **same quantity**, not two measurements.

All tracked roots are complex pairs, `NEIMARK_SACKER_CANDIDATE`. Phase is
nowhere near 0 (fold) or pi (period doubling) and is stable to three digits
while `|lambda|` climbs. Peak `|lambda| = 0.999230` sits at exactly the last
TD-stable point. **`|lambda|` never reaches 1** — this is an approach, not a
confirmed crossing.

### TD on the same RCSJ circuit
Stable (envelope slope ~ -1e-6/period) at 6.1047 / 6.3954 / 6.6861e-06 A;
BLOWUP (83-167 rad) at 7.2675e-06 A. **Rung-independent** across 1e4/1e5/1e6 —
the regularizer does not move the transition. Bracket **(-24.242, -23.518] dBm**.

### Damping window for 2c
`Ic = 2.6562e-06 A` (uniform, 2508 junctions), `Cj = 145 fF`,
`Rn = pi*Delta/(2e*Ic) = 106.4456 ohm`. Damping/pump period at 7.9 GHz is
`T_p/(R_j*C_j)`:

| `R/Rn` | damping/period | verdict |
| ---: | ---: | --- |
| 1e6 | 8.2e-6 | too weak — spectrum stays pinned at 0.999996 |
| **1e5** | **8.2e-5** | **primary** |
| 1e4 | 8.2e-4 | usable |
| 1e2 | 8.2e-2 | **16x the 5.0e-3/period instability — would suppress it** |

Window is bounded below by the 1.5e-5/period that `tan_delta = 1e-5` needed to
lift the degeneracy, and above by the 5.0e-3/period growth being measured.

### Measured FDTD cost per point (69 points, this machine, `dt_norm = 0.01`)

| device | s/point (median) | steps/s | n_steps | pump periods |
| --- | ---: | ---: | ---: | ---: |
| guarcello | 81 | 64,426 | 5,229,400 | 2100 |
| jc_fqjtwpa | 809 | 15,009 | 12,140,121 | 2100 |
| jc_jtwpa | 870 | 14,711 | 12,782,904 | 2100 |
| ipm_2c_fixed | 985 | 6,368 | 6,271,516 | 2100 |

### Device split — Group A vs Group B

Discriminator is `best/median` prominence of the 400-trial generator scan:

| device | pre-transition | post-transition | reading |
| --- | ---: | ---: | --- |
| jc_jtwpa | 94.6, 10.7, 4.01 | **1.17, 1.14, 1.16** | no generator, continuum |
| jc_fqjtwpa | 241.6, 590.7, 861.0 | **1.16, 1.17, 1.13** | no generator, continuum |
| ipm_2c_fixed | (off_lattice ~ 0) | **3.12, 4.20, 4.92, 5.81, 3.46** | sharp peaks |
| guarcello | 7.27, 2.29, 2.31, 3.50 | 1.68, 1.32, 1.34, 1.23, 1.16 | 2 of 9 clear |

**jc_jtwpa's collapse is physical, not a timestep artifact.** At -27.8 dBm,
`dt_norm` 0.01 -> 0.005 moves `on_lattice` 0.0564 -> 0.0576 while the linear
gate error halves (0.2325 -> 0.1240). First-order Richardson extrapolates to
**0.0588** at `dt -> 0`, against Group B's 0.76-0.98 — a factor of 13.

---

## 3. Pre-existing, from the docs — also do not re-derive

Source: `high_power_investigation_full_record_20260812.md`,
`high_power_79ghz_instability_threshold_20260811.md`,
`2c_high_power_solver_investigation_2026-08-09.md`.

- HB period-1 orbit at -23.421053 dBm exists and **is dynamically unstable**,
  rate 0.005013/period, timestep-converged.
- **No fold near it.** `jacobian_min_eigenvalue` ~1e5 across the branch,
  minimum 7.28e4. The PALC fold at 1.1628e-05 A is ~4 dB higher with no
  approach.
- HB non-convergence at that power was a **power-step artifact** — 1.05 dB
  steps fail, 0.18-0.25 dB steps converge by plain Newton.
- 2c HB states are fully resolved: `production_hb_full_residual_rel` 2.4e-12 to
  5.9e-10 through -24.4737 dBm. Basis truncation is not the 2c problem.
- Phase 3 already ran `tan_delta` 0 / 1e-5 / 1e-4 / 1e-3. Loss lifts the
  degeneracy. It found no crossing, but that result is **uninformative**: lossy
  Hill was compared against lossless TD, i.e. two different circuits.
- Themis: orbit instability (-23.4..-24.2 dBm) is the closest candidate to the
  measured collapse (-21.4..-22.7); the PALC fold (-19.4) is on the wrong side.

---

## 4. Open threads

| # | thread | blocked on | cost |
| --- | --- | --- | --- |
| O1 | Does 2c's `\|lambda\|` cross 1? Is it Neimark-Sacker? | A4 | ~18 min |
| O2 | Predicted-rho mask vs p90 baseline on 2c spectra | A4 Task 3 | minutes |
| O3 | guarcello comb spacing never measured, so B3 never ran | needs group-delay measurement | ~1 h |
| O4 | jc_fqjtwpa timestep never screened (only jtwpa was) | — | ~1 h |
| O5 | 2.23 dB TD-vs-HB gain gap on 2c, survives timestep convergence | not blocking | unknown |
| O6 | G5 limit extrapolation (crossing vs `R/Rn -> inf`) | O1 | ~25 min |
| O7 | Group A validity-boundary write-up for the thesis | O1 not needed | writing only |
| O8 | FDTD transition refinement, 4 devices — **queued, see section 4b** | user launches 2026-08-16 evening | 3.3 h wall |
| O9 | Does the transition location move with `dt_norm`? | part of O8 | included above |

---

## 4b. QUEUED FOR TONIGHT — FDTD transition refinement (4 devices)

Approved 2026-08-16. Launch unattended. No new code required.

### Brackets (from data already on disk, not estimated)

| device | last clean | first collapsed | bracket | axis |
| --- | ---: | ---: | ---: | --- |
| guarcello | -53.70 (`cl=1`) | -53.00 (`cl=1051`) | **0.70 dB** | `pump_power_dbm` |
| jc_jtwpa | -28.20 (`on_lat` 0.9233) | -27.80 (0.0564) | **0.40 dB** | `pump_power_dbm` |
| jc_fqjtwpa | -31.50 (0.9346) | -31.20 (0.0509) | **0.30 dB** | `pump_power_dbm` |
| ipm_2c_fixed | 0.575 (1.0000) | 0.625 (0.8125) | 0.050 ratio = **0.72 dB** | `I_over_I_bound` |

Two corrections to the ranges proposed in conversation:
- guarcello **-55 to -52 wastes effort.** 14 points already exist between
  -55.00 and -53.70 at 0.02-0.05 dB spacing. The only virgin span is
  -53.70 to -53.00.
- ipm_2c **0.58 to 0.62 is too narrow.** It excludes both measured
  endpoints, so if the transition sits in (0.575, 0.58] or (0.62, 0.625]
  both ends return the same class and bisection has no bracket. Use the
  measured bracket itself; its endpoints are already on disk and cost
  nothing.

### Bisect on `on_lattice`, NOT on `poincare_clusters` or `verdict`

`poincare_clusters` is unusable on this data:

| device | clusters vs control |
| --- | --- |
| jc_fqjtwpa | 457, 390, 551, 567, 860, 488, **834** while clean, then **6, 3, 3** once collapsed — *inverted* |
| jc_jtwpa | 1, 1, **1011, 1036**, 1, 1, 1, then 4, 3 — *non-monotone*; 1011 lands where `on_lattice` = 0.9995 and gain = 23.3 dB |

`verdict` inherits the defect (jtwpa is labelled
`CHAOS_NO_CLEAN_BIFURCATION` at -29.7 dBm, `PERIOD_DOUBLING_ONSET` at
-28.2 dBm). `sigma_vprime_ps` — the actual section width — is clean on 2c
(4.90e4 -> 1.14e6, x23) and guarcello, but only x1.8 on fqjtwpa and
**non-monotone on jtwpa** (peaks 4.12e6 at -28.2, *before* the transition,
falls to 2.54e6 after).

Use **`on_lattice` crossing 0.5** as the primary criterion, with
**`residual_n1 > 1.35`** confirming. `residual_n1` rises monotonically then
pins at 1.40-1.42 (approximately sqrt(2), the uncorrelated-return value) on
both JC devices.

### Stage 1 — 9-point grid per device, one unattended batch

Nine points per bracket, endpoints included. Endpoints already exist, so
`_is_done` skips them and 7 are new per device.

| device | grid | step | new pts | serial |
| --- | --- | ---: | ---: | ---: |
| guarcello | linspace(-53.7, -53.0, 9) | 0.0875 dB | 7 | 9.5 min |
| jc_fqjtwpa | linspace(-31.5, -31.2, 9) | 0.0375 dB | 7 | 94 min |
| jc_jtwpa | linspace(-28.2, -27.8, 9) | 0.0500 dB | 7 | 102 min |
| ipm_2c_fixed | linspace(0.575, 0.625, 9) | 0.00625 (0.0905 dB) | 7 | 115 min |
| | | | **28** | **5.3 h** |

At `--workers 3`: approximately **1.8 h wall**.

`--control-linspace` overwrites the plan for *every* device in the run, so
this is one invocation per device:

```powershell
python scripts/chaos/run_phaseB_overnight.py --output outputs/chaos/phaseB_signal `
  --devices guarcello --control-linspace -53.7 -53.0 9 `
  --periods 2100 --dt-norm 0.01 --signal-dbm -90 --workers 3

python scripts/chaos/run_phaseB_overnight.py --output outputs/chaos/phaseB_signal `
  --devices jc_jtwpa --control-linspace -28.2 -27.8 9 `
  --periods 2100 --dt-norm 0.01 --signal-current-a 3e-08 --workers 3

python scripts/chaos/run_phaseB_overnight.py --output outputs/chaos/phaseB_signal `
  --devices jc_fqjtwpa --control-linspace -31.5 -31.2 9 `
  --periods 2100 --dt-norm 0.01 --signal-current-a 3e-08 --workers 3

python scripts/chaos/run_phaseB_overnight.py --output outputs/chaos/phaseB_signal `
  --devices ipm_2c_fixed --control-linspace 0.575 0.625 9 `
  --periods 2100 --dt-norm 0.01 --signal-current-a 3e-08 --workers 3
```

**`--periods 2100` is exact for all four devices**, verified by inverting
`_tmax_norm` against the four cached keys in
`outputs/chaos/phaseB_signal/pump_off_reference_cache.json`:

```text
guarcello      52294.181396   jc_jtwpa      127829.040505
jc_fqjtwpa    121401.209340   ipm_2c_fixed   62715.162101
```

The cache key is `device|dt|tmax|signal_current|signal_dbm`. **Any drift in
`--periods`, `--dt-norm` or the signal level misses the cache**, silently
re-measures the pump-off reference (13-16 min serial per device) and changes
the denominator of `gain_vs_off_db`, so the new points stop being comparable
to the 69 already on disk. guarcello uses `--signal-dbm -90` with
`signal_current_a = 0.0`; the other three use `--signal-current-a 3e-08`.

Dry-run each command first (`--dry-run`) to confirm argparse accepts the
negative `--control-linspace` values and the pending count is 7. Note that
`--dry-run` returns before the `tmax_norm` print, so it cannot verify
`--periods`; that is why the value was checked numerically above.

### Stage 2 — refinement, second launch

A second 9-point grid inside the surviving sub-bracket reaches 0.004-0.010
dB. Same cost, approximately 1.8 h wall. Not unattended — it needs stage 1
reduced first.

Rejected: a single dense grid to 0.01 dB in one pass. That is 216 points and
**37 h serial**, of which ipm_2c alone is 20 h. Two 9-point stages reach the
same resolution in 3.6 h.

### Timestep control — MUST run, gates the whole result

`dt_norm = 0.01` has never been shown to leave the *transition location*
fixed. Task A only checked `on_lattice` at -27.8 dBm, already past the
collapse. If the boundary moves with timestep, the bisection resolves a
number that is not converged.

Two points per device (the bracket endpoints) at `--dt-norm 0.005`, 2x cost
each: 178 min serial, plus 3 pump-off references re-measured at 2x
(approximately 90 min serial). Total approximately 4.5 h serial,
**+1.5 h wall**.

**Must use a separate `--output`.** At the same output root, `_is_done`
finds the existing `dt = 0.01` trace and skips every point:

```powershell
python scripts/chaos/run_phaseB_overnight.py --output outputs/chaos/phaseB_dt005 `
  --devices jc_jtwpa --control-linspace -28.2 -27.8 2 `
  --periods 2100 --dt-norm 0.005 --signal-current-a 3e-08 --workers 3
```
(and likewise for `jc_fqjtwpa` -31.5/-31.2 and `ipm_2c_fixed` 0.575/0.625).

**Combined tonight: approximately 9.8 h serial, 3.3 h wall at 3 workers.**

### Warm-start question — resolved, do not rebuild

`run_guarcello_jc_phase5.py:2239-2269` already warm-chains (`previous_state`,
`previous_current`, `initial_state=`, `start_current_a=`).
`run_phaseB_overnight.py` does **not** — it submits every point cold to a
`ThreadPoolExecutor`. Every point in `outputs/chaos/phaseB_signal/` was
started from zero.

Chaining is the wrong optimization for locating a boundary:
- Bisection buys a logarithmic factor in point count; warm start buys a
  linear factor in per-point cost, and serializes, giving back the 3x pool.
- Chaining from below **is** an up-ramp protocol. On a subcritical
  transition the up-ramp threshold differs from the cold-start one. That is
  a second measurement, not a cheaper version of the first. The campaign
  already shows protocol sensitivity: its own controls at -23.8 dBm gave
  `d1` = 2.41e-3 at ramp 20 against 1.03e-3 at ramp 80.
- The integrator is `implicit_trapezoid` — A-stable, not L-stable,
  `R(z) -> -1`, so no numerical damping at any frequency. Cold-started every
  point carries the same undecayed ringing; chained, each point *inherits*
  the previous point's on top of its own.
- Above the transition the previous state is a point on a strange attractor,
  not a fixed point, so it guarantees no transient reduction at all.

Separate follow-up if hysteresis is in question: up-ramp against cold-start
at 6 points around the jtwpa transition, approximately 1.7 h. That is a
physics result, not a speedup.

---

## 4c. Nonlinear-dynamics reduction of the existing campaigns (2026-08-16)

Instruments: `scripts/chaos/nonlinear_diagnostics.py`, driver
`scripts/chaos/run_nonlinear_diagnostics.py`. Output
`outputs/chaos/nonlinear/{device}.json` + `summary.json`, written per point.

### Instrument validation FIRST, against literature values

| system | D2 measured | literature | K | expected K |
| --- | --- | ---: | ---: | ---: |
| Henon | 1.184 @m=2 | 1.220 | +0.998 | 1 |
| logistic r=4 | 0.901 @m=2 | 1.000 | +0.999 | 1 |
| 2-torus | 2.232 @m=4 | 2.000 | -0.004 | 0 |
| periodic | 1.027 @m=2 | 1.000 | -0.002 | 0 |
| Lorenz | **no plateau** | 2.050 | +0.999 | 1 |

**K (Gottwald-Melbourne 0-1) is validated 5/5 with clean separation and is the
primary classifier. D2 (Grassberger-Procaccia) is NOT** -- it fails to plateau
on Lorenz at 3000 points and reads 12 percent high on the torus. D2 is
advisory only and must always be quoted with its full m-curve.

Two further limits found during validation:
- **K is meaningless on a dense flow.** Lorenz gives K = -0.000 unsampled and
  +0.927..+0.999 once subsampled 5x or more. The input must be a stroboscopic
  section, never the raw trace.
- **K is meaningless where the signal is at the numerical floor.** 2c below
  its transition has sigma ~ 5e-9 and K scatters -0.59..+0.91; that is the
  test running on roundoff. Use D2 there.

### Validity gating actually applied

`outputs/chaos/phaseB` holds 311 pump-only points (`signal_installed=False`,
`pump_amplitude` never zero -- no pump-off contamination). Of those:

| device | n | periods | verdict |
| --- | ---: | ---: | --- |
| guarcello | 87 | 12000 | usable |
| ipm_2c_fixed | 37 | 1600 | usable, settling caveat below |
| jc_jtwpa | 100 | 600 | **excluded** |
| jc_fqjtwpa | 87 | 600 | **excluded** |

The JC runs sit below the measured ~1050-period settling floor. Residual
ringing is non-recurrent and inflates BOTH D2 and K toward "chaotic", so
including them would manufacture false chaos.

`outputs/chaos/phaseB_signal` (69 points) is excluded from attractor
classification entirely, and not because of a parameter error: **a signal tone
makes the forcing quasi-periodic by construction**, so a 2-torus is guaranteed
a priori. Those points remain valid for `on_lattice` and gain, which is what
they were run for.

### ipm_2c_fixed: the transition is HARD, not a Neimark-Sacker

> **RETRACTED 2026-08-16 23:00. The whole of this subsection is a grid
> artifact.** The `1600x in one 0.025 step` jump was the transition falling
> between two samples. Re-measured pump-only at 0.005 spacing, 2100 periods
> (`outputs/chaos/phaseB_2c_gap` -> `outputs/chaos/nonlinear_2c_gap`), it
> resolves into a graded climb through four intermediate points, each about one
> order of magnitude, plus an intermediate **regular** window. See
> "ipm_2c_fixed re-measured at 0.005" below. The Neimark-Sacker exclusion and
> the instruction not to build the auxiliary-generator closure for 2c are both
> withdrawn -- they rested entirely on the single-step jump.
>
> The user asked directly whether too few points explained the hard transition.
> The answer was yes.

37 points, 0.300 to 1.200 in `I/I_bound`:

| control | sigma (strobe std) | D2 | K |
| ---: | ---: | --- | ---: |
| 0.300 | 4.558e-09 | 1.024 @m=2 | floor |
| 0.500 | 7.258e-09 | 1.024 @m=2 | floor |
| 0.575 | 8.638e-09 | 1.021 @m=2 | floor |
| **0.600** | **1.378e-05** | no plateau | -0.089 |
| 0.625 | 3.266e-05 | no plateau | +0.990 |
| 0.800 | 1.743e-05 | no plateau | +0.999 |
| 1.050 | 1.804e-05 | 2.739 @m=5 | +0.822 |

`sigma` jumps **1600x in one 0.025 step**. Below it D2 = 1.02 constant across
nine points (a period-1 orbit); above it K = 0.95..0.999 and D2 never
plateaus (high-dimensional). Normal-form fit: **`NO_SCALING_REGION`,
best R^2 = 0.744**.

**This excludes a supercritical Neimark-Sacker**, which would grow the
invariant circle as `(mu - mu_c)**0.5` -- smooth and fittable at R^2 > 0.99 on
the synthetic control. 2c jumps straight from a period-1 orbit to a
high-dimensional attractor with no intermediate torus.

**Consequence for the north star: do not build the auxiliary-generator /
quasi-periodic HB closure for 2c. There is no torus to represent.** This
settles from data what agents A1-A4 failed to settle from eigenvalues.

Cross-validation: the jump sits at 0.575 -> 0.600, and the independent
signal-driven `on_lattice` campaign brackets its transition at
(0.575, 0.625] -- different observable, different campaign, same boundary.

Caveat: 2c integrated 1600 periods with the analysed window opening at period
800, below the ~1050 floor. It is demonstrably not driving the result (the
pre-transition points read D2 = 1.02 and sigma = 5e-9, i.e. clean periodic),
but a 2100-period re-run should confirm before this is published.

### ipm_2c_fixed re-measured at 0.005 spacing (2026-08-16 23:00)

11 points, 0.575 to 0.625, PUMP-ONLY, 2100 periods (clears the ~1050 settling
floor the 1600-period run opened below). `outputs/chaos/phaseB_2c_gap`,
reduced to `outputs/chaos/nonlinear_2c_gap`:

| control | K | D2 | sigma |
| ---: | ---: | --- | ---: |
| 0.5750 | +0.3710 | 1.020 @m=2 | 6.8842e-09 |
| 0.5800 | +0.7144 | 1.021 @m=2 | 9.7983e-09 |
| 0.5850 | +0.1796 | 1.022 @m=2 | 5.1255e-08 |
| 0.5900 | -0.2815 | 1.057 @m=2 | 6.0807e-07 |
| 0.5950 | -0.1721 | none | 8.0722e-06 |
| 0.6000 | +0.0614 | none | 1.7228e-05 |
| 0.6050 | +0.1568 | none | 1.6609e-05 |
| 0.6100 | +0.9450 | none | 2.4835e-05 |
| 0.6150 | +0.9890 | none | 3.5016e-05 |
| 0.6200 | +0.9974 | none | 1.9999e-05 |
| 0.6250 | +0.9960 | none | 1.9853e-05 |

Three regimes:

| range | sigma | K | reading |
| --- | --- | --- | --- |
| 0.575-0.585 | 7e-9 - 5e-8 | +0.18..+0.71, erratic | floor, NOT classifiable |
| 0.590-0.605 | 6.1e-7 -> 1.7e-5 (x28) | -0.28, -0.17, +0.06, +0.16 | **regular while sigma climbs** |
| 0.610-0.625 | 2-3.5e-5, saturating | +0.945..+0.997 | chaotic |

The 0.590-0.605 window is the same qualitative structure jc_jtwpa shows at
-29.23..-29.09, but **much noisier**: `|K|` reaches 0.28 here against 0.008
there. Treat 2c's window as suggestive, jtwpa's as measured.

**Do not classify the bottom three points.** `sigma = 7e-9` is roundoff; K is
incoherent there (+0.71 at 0.580, where a clean periodic orbit must give ~0);
and `D2 ~ 1.02` would imply an invariant circle already present at the lowest
power, which is not credible. All three statistics are measuring numerical
noise. **K is only interpretable once sigma clears the floor** -- this applies
equally to the jc_jtwpa `D2 ~ 0.83-0.93` values below -29.66 dBm.

### Instrument defect: the normal-form reason string is unconditional

`fit_normal_form_exponent` emits `NO_SCALING_REGION -- best R^2 <x> below 0.90;
consistent with a hard transition` whenever the fit misses its gate, regardless
of the data. Both 2026-08-16 evening runs printed it (2c R^2 0.8973, jtwpa
0.7933) while their own sigma columns show clearly **graded** ramps -- jtwpa
over ~0.2 dB, 2c over 0.025. The string is backwards in both cases, and the 2c
value 0.8973 is a near-miss on a power law rather than a rejection of one.
Condition the wording on the sigma ratio between adjacent points, not on the
fit failing. Until fixed, ignore the phrase and read the sigma column.

### The normal-form fit must be LOCAL and post-onset -- first pass was wrong

The driver initially fitted every point in the sweep, including the flat
pre-transition floor. That gave guarcello `beta = 18.15` with `mu_c` pinned to
the scan edge (-95 dBm) and `R^2 = 0.41`, and 2c `R^2 = 0.744` -- both
artefacts of fitting the wrong subset, not measurements. A normal form is a
*local* statement, so the fit takes only points above onset, and the window is
shrunk toward onset to check the exponent is not a global-behaviour artefact.

Corrected, with the window measured as a fraction of the span above onset:

| device | window | pts | beta | R^2 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ipm_2c_fixed | 20% | 5 | 0.0275 +/- 0.0208 | 0.369 | NO_SCALING |
| ipm_2c_fixed | 35% | 9 | 0.0262 +/- 0.0150 | 0.302 | NO_SCALING |
| ipm_2c_fixed | 50% | 12 | 0.0225 +/- 0.0140 | 0.204 | NO_SCALING |
| ipm_2c_fixed | 100% | 25 | 0.0157 +/- 0.0131 | 0.059 | NO_SCALING |
| guarcello | 10% | 41 | **0.6659 +/- 0.0327** | 0.914 | scaling, not 1/2 |
| guarcello | 20% | 45 | 0.7014 +/- 0.0307 | 0.924 | scaling, not 1/2 |
| guarcello | 35% | 46 | 0.7630 +/- 0.0332 | 0.923 | scaling, not 1/2 |
| guarcello | 100% | 51 | 0.6789 +/- 0.0412 | 0.847 | NO_SCALING |

### The two devices are dynamically DIFFERENT

**ipm_2c_fixed -- hard transition, no scaling at any window.** `beta ~ 0.02`
with `R^2` 0.06 to 0.37 however narrow the window; sigma is flat before
(x1.9 over 12 points) and flat after. A supercritical Neimark-Sacker cannot
produce a 1595x step change between adjacent grid points.

**guarcello -- continuous scaling, exponent about 2/3, NOT 1/2.** As the
window narrows toward onset `beta` converges to **0.666 +/- 0.033** at
`R^2 = 0.914`, which excludes 1/2 at roughly 5 sigma. Its sigma also rises
x24 gradually over 36 points *before* onset, where 2c's rises only x1.9.

So the ansatz question splits by device:
- **2c: no torus exists to represent. Do not build the auxiliary-generator
  closure for it.**
- **guarcello: a growing invariant set does exist**, so a quasi-periodic
  extension is not excluded there -- but its exponent is not the textbook
  Neimark-Sacker 1/2, and which bifurcation gives 2/3 is NOT established here.
  Do not name one without deriving it.

This independently reproduces the split found from spectra (guarcello degrades
softly and recovers; 2c and the JC devices collapse hard), from completely
different mathematics.

Third independent agreement on a boundary: the `on_lattice` bracket for
guarcello is (-54, -53] and the sigma onset is -54.02.

**K is unavailable for guarcello** -- see the stride defect below; its strobe
yields 299 points, under the 500 where the test was validated. D2 and sigma
are unaffected because neither reads the time axis.

### jc_jtwpa pump-only, 2100 periods: TWO events, not one (2026-08-16)

12 points, -28.6 to -27.4 dBm, pump-only, 2100 periods, 1.06 h
(`outputs/chaos/phaseB_jtwpa_2100`, reduced to
`outputs/chaos/nonlinear_jtwpa`). Analysed window opens at period 1050,
exactly at the settling floor.

**K = 0.9942..0.9990 at every one of the 12 points**, sigma flat at
2.4e-05..4.4e-05, no jump anywhere in the window. Normal-form fit
`NO_SCALING_REGION`, R^2 = 0.481.

Three alternative explanations were tested and all excluded:

| alternative | test | result |
| --- | --- | --- |
| residual transient | recompute discarding 0/25/50/75% of the record | sigma 3.1163/3.1319/3.1306/3.1288e-05 -- **stable to 0.5%**, a decaying transient would shrink |
| misaligned strobe | FFT the trace for the true fundamental | 7.119991e9 vs 7.12e9 declared, drift **0.003 periods** over the whole record |
| sigma is just always this size | compare against -36 dBm control | 1.512e-06, i.e. **21x smaller** |

**So jc_jtwpa's pump-only period-1 orbit is already gone below -28.6 dBm**,
under the entire scanned window.

**This does NOT contradict the signal-driven `on_lattice` data -- it explains
it.** Off-lattice fraction there grows smoothly well before the collapse:

| P [dBm] | -29.7 | -29.3 | -28.9 | -28.6 | -28.2 | **-27.8** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off_lattice | 0.0005 | 0.0088 | 0.0272 | 0.0442 | 0.0767 | **0.9436** |
| gain [dB] | 23.3 | 27.4 | 31.4 | 34.1 | 36.1 | -6.0 |

So there are **two separate events**:
1. **Loss of the periodic orbit, at or below -28.6 dBm.** Off-lattice becomes
   non-zero and grows; pump-only K is already ~1.
2. **A hard collapse at (-28.2, -27.8]**, off-lattice 0.077 -> 0.944.

**Consequence for the north star: HB's accuracy ceiling starts falling well
before the visible collapse.** At -28.6 dBm the device has 34.1 dB gain and
`on_lattice` 0.9558 -- HB is already 4.4 percent wrong there. Published
operating points may sit inside the degraded region. Campaign stage 2b refines
event 2 only; **event 1 is unbracketed and below -28.6 dBm.**

The 600-period pump-only sweep cannot locate event 1: its sigma is
transient-dominated, non-monotone, with isolated single-point spikes at
-30.241, -29.717 and -29.194 dBm each followed by a drop back. This is
independent confirmation that excluding the 600-period data was correct.

Follow-up needed: pump-only, 2100 periods, about -31.0 to -28.6 dBm,
10 points, roughly 53 min.

### jc_jtwpa pump-only, full 21-point curve: THREE regimes (2026-08-16 19:20)

`outputs/chaos/phaseB_jtwpa_2100`, -30.5 to -27.4 dBm, all pump-only, all
2100 periods, all `dt_norm = 0.01`. Reduced to `outputs/chaos/nonlinear_jtwpa`.

| P [dBm] | sigma | D2 | K | regime |
| ---: | ---: | --- | ---: | --- |
| -30.5000 | 1.628e-08 | 0.933 @m=2 | +0.095 | periodic (floor) |
| -30.0778 | 1.745e-08 | 0.841 @m=2 | +0.060 | periodic |
| -29.8667 | 2.548e-08 | 0.887 @m=2 | -0.223 | periodic |
| -29.6556 | 3.216e-08 | 0.933 @m=2 | -0.272 | periodic |
| **-29.4444** | 1.162e-07 | 1.015 @m=2 | -0.219 | last regular |
| **-29.2333** | 7.433e-06 | none | **+0.001** | **sigma x64, still regular** |
| -29.0222 | 1.385e-05 | none | +0.823 | mixed |
| -28.8111 | 1.842e-05 | none | +0.800 | mixed |
| -28.6000 | 3.131e-05 | none | +0.994 | chaotic |
| -27.4000 | 2.414e-05 | none | +0.999 | chaotic |

**Event 1 = onset of chaos at approximately -29.4 dBm, and it is
PROGRESSIVE**: K sweeps 0 -> 1 over about 0.8 dB. Below -29.66, sigma is
pinned at the ~1.6e-08 floor and D2 is about 1 (period-1 orbit on the flow);
K is scattered there because the signal is at the numerical floor, the same
floor-noise regime already documented on 2c.

Independent corroboration: the signal-driven `off_lattice` begins growing at
the same place -- 0.0005 (-29.7), 0.0088 (-29.3), 0.0272 (-28.9), 0.0442
(-28.6). Two unrelated measurements, one pump-only and one signal-driven,
place the onset inside the same 0.2 dB.

**Event 2 is NOT a pump-only bifurcation.** Across -28.6 to -27.4, K stays
0.994-0.999 and sigma only wobbles 2.4e-05..4.4e-05. The hard `on_lattice`
collapse at -27.8 does not appear in the pump dynamics at all, so it is the
*signal* losing coherent response on top of an already-chaotic pump.

**Consequence for the north star: HB on jc_jtwpa is representing a chaotic
pump state from about -29.4 dBm upward, and the published operating points
sit above that.**

### CANDIDATE torus window at -29.2333 dBm -- ONE POINT, not established

At -29.2333 the section widens **x64** (1.162e-07 -> 7.433e-06) while
K = **+0.0013**, i.e. dynamically regular. A point attractor that widens 64x
yet stays regular is what an **invariant circle** looks like: the section is no
longer a point, but motion on it is quasi-periodic rather than chaotic. That is
the Neimark-Sacker -> torus -> chaos route, and jc_jtwpa would be the first
device here to show the intermediate state. It is consistent with the older
record that a torus appears on jtwpa "in a narrow window ... buys ~1.1 dB".

**If it holds it is the one case where the auxiliary-generator closure is
justified** -- the thing 2c ruled out for itself.

It rests on a single point. Resolution sweep launched 19:26:
-29.4444 to -29.0222, 13 points at 0.0352 dB, 10 new, 2 workers, ETA ~20:45.
Confirmation requires K to stay near 0 across several consecutive points while
sigma climbs. D2 returns `none` there, which is neither support nor refutation
given D2 failed the Lorenz control. **Lambda_1 would settle it** (torus 0,
chaos > 0) once the implicit-tangent fix below is applied.

### RESOLVED 2026-08-16 23:05 -- the window is REAL, five consecutive points

31-point pump-only curve, uniform 2100 periods / `dt_norm = 0.01`,
`outputs/chaos/phaseB_jtwpa_2100`, reduced to
`outputs/chaos/nonlinear_jtwpa_torus`:

| P [dBm] | K | sigma |
| ---: | ---: | ---: |
| -29.0574 | +0.7592 | 1.1539e-05 |
| **-29.0926** | **+0.0050** | 8.0129e-06 |
| **-29.1277** | **-0.0039** | 8.2064e-06 |
| **-29.1629** | **-0.0018** | 8.3115e-06 |
| **-29.1981** | **+0.0056** | 8.1026e-06 |
| **-29.2333** | **+0.0013** | 7.4331e-06 |
| -29.2685 | -0.0769 | 6.3084e-06 |

`|K| < 0.008` across five consecutive points spanning **0.14 dB**. The
numerical-floor points at -30.5..-29.66 scatter to `|K| = 0.53`, so the window
is about two orders tighter than this instrument's own noise, and its
`sigma ~ 8e-6` sits 500x above the floor -- a resolved orbit, not roundoff. The
upper edge is sharp: K goes 0.005 -> 0.759 inside one 0.035 dB step.

Full curve, five regimes:

| range [dBm] | sigma | K | reading |
| --- | --- | --- | --- |
| -30.50..-29.66 | 1.6-3.2e-8 | erratic, +-0.27 | floor, NOT classifiable |
| -29.44..-29.27 | 1.2e-7 -> 6.3e-6 (x54) | -0.08..-0.53 | regular, sigma climbing |
| -29.23..-29.09 | ~8e-6 flat | \|K\| < 0.008 | **regular window** |
| -29.06..-28.81 | 1.2-1.8e-5 | 0.76-0.82 | transitional |
| -28.60..-27.40 | 2.4-4.4e-5 | 0.994-0.999 | chaotic |

**Sigma is FLAT inside the window, not climbing** -- it climbs x54 on approach
(-29.44 -> -29.27) and then plateaus. The stated confirmation criterion
("K near 0 while sigma climbs") is therefore met on the approach and not
inside; the window itself is a plateau. This does not weaken the result but the
wording of the criterion was imprecise.

**D2 CORROBORATES the invariant circle, on the approach (2026-08-16 23:40).**
An earlier reading of this ("D2 found no plateau inside the window, so there is
no independent evidence of an invariant circle") was incomplete. The plateau is
absent *inside* the K~0 window but present and clean immediately below it:

| P [dBm] | sigma | D2 at m = 2..8 |
| ---: | ---: | --- |
| -29.6556 | 3.22e-08 | 0.933 0.936 0.934 0.937 0.936 0.938 0.939 |
| **-29.4444** | 1.16e-07 | **1.015 1.015 1.016 1.016 1.016 1.016 1.016** |
| **-29.4092** | 4.15e-07 | **1.026 1.031 1.032 1.032 1.032 1.032 1.032** |
| -29.3740 | 1.36e-06 | 1.075 1.100 1.105 1.106 1.104 1.105 1.106 |
| -29.3389 | 2.56e-06 | 1.129 1.173 1.183 1.185 1.182 1.185 1.187 |
| -28.0545 | 4.36e-05 | 1.844 2.307 (rising, no saturation) |
| -27.4000 | 2.41e-05 | 1.879 2.637 3.295 (rising, no saturation) |

`D2 = 1` **is** an invariant circle. At -29.4444 it is flat to four digits
across seven embedding dimensions. So the torus appears at about -29.44 dBm,
where sigma first clears the floor, and the K~0 window at -29.23..-29.09 is the
*developed* torus rather than its onset. **The torus branch is therefore wider
than the K window: roughly -29.44 to -29.09, about 0.35 dB.**

Below -29.66 dBm, D2 reads 0.83-0.94, i.e. *below* 1, on sigma ~ 2e-8. A
period-1 orbit must give D2 = 0, so those values are roundoff, consistent with
the floor rule.

Above -28.6 dBm, D2 climbs with m and never saturates (1.88 / 2.63 / 3.29 at
m = 2/3/4), which is what a high-dimensional chaotic set does.

**Instrument limitation, not a physical result:** the scaling-window search
returns nothing at all inside -29.2685..-29.0574, the very window where the
return map is a clean ellipse. A very thin, strongly elongated section appears
to defeat the correlation-integral fit. Do not read the empty D2 there as
absence of structure.

**What is still NOT established.** The 0-1 test returns K ~ 0 for periodic and
quasi-periodic orbits alike -- it separates regular from chaotic and nothing
finer, so K alone never proves a torus; the D2 = 1 saturation above is what
carries that claim, and it covers the approach rather than the window itself.
**Lambda_1 remains the decisive measurement**: negative for periodic, zero for
quasi-periodic, positive for chaos, at every point on one axis.

### guarcello has NO stroboscopic section (2026-08-17)

`outputs/chaos/phaseB/guarcello`, 87 pump-only points, -70..-45 dBm, reduced to
`outputs/chaos/nonlinear_guarcello`. Every point reports `strobe = 0`.

`record_stride = 20` at `dt_s = 1.1474e-12` and `f_p = 7 GHz` gives **6.23
stored samples per pump period**, below the eight-sample guard in
`nonlinear_diagnostics.py:567`. So `stroboscopic_section` returns empty and
every guarcello point silently takes the `poincare_branches['upward']`
fallback:

| | jc_jtwpa / ipm_2c_fixed | guarcello |
| --- | --- | --- |
| section | stroboscopic, 1 sample/pump period | Poincare upward crossings |
| observable | node voltage [V] | `dv/dt` at crossings [device units] |
| sigma range | 1.6e-8 .. 4.4e-5 | 9.5e2 .. 1.1e7 |
| samples | 1049 | 300 |

Consequences:

- **sigma is not comparable across devices** and `FLOOR_SIGMA = 1e-7` is
  meaningless for guarcello. The plot script classifies it on K alone and
  labels the axis and panel; do not put it on a shared sigma scale.
- **The beta = 0.666 +- 0.033 at R^2 = 0.914 result was fitted on this
  Poincare `dv/dt` spread**, not on the stroboscopic spread used for the other
  devices. It is a real quantity but a different observable, and must not be
  compared to jc_jtwpa's beta without that caveat.
- **Pre-transition K is unusable.** It swings between -0.69 and +0.70 between
  adjacent points across the whole -70..-54 dBm range -- the signature of a
  statistic computed on an unresolved section, not of physical structure. The
  regime shading below the transition in `regime_map.png` is noise. Read only
  the transition itself.

What survives: the transition is clear at about **-53.5 dBm**, where sigma
jumps roughly x1000 and K locks to 0.99+ and stays there to -45 dBm. And at
**-53.75 dBm the return map is a closed curve** (sparse, 300 points, K =
+0.0031) -- a third device showing an invariant circle immediately below its
chaos onset. That claim rests on the *shape*, not on K, since K is untrustworthy
in that band.

Fix requires a re-run, not re-analysis: `record_stride <= 15` clears the guard,
`stride = 4` gives 31 samples per period and a directly comparable sigma. Fold
it into any future guarcello campaign.

### Instrument defect: a reader can kill a campaign on Windows

`run_nonlinear_diagnostics.py` writes per point via `tmp.replace(out_path)`.
On Windows `os.replace` raises `PermissionError [WinError 5]` if any other
process holds the target open. Measured 2026-08-16: a 5 s polling loop reading
`guarcello.json` aborted the run at 71 of 87 points -- the exact loss the
per-point write exists to prevent. Fixed by `write_rows_atomically`, which
retries 20 times at 250 ms. Apply the same pattern to any other per-point
writer before tailing its output.

### Overnight campaign 2026-08-17: results

`scripts/chaos/run_overnight_campaign_20260817.ps1`, 4 h 25 min, all stages
exit 0. 112 new FDTD points. (A `.sh` version was written first and cost the
user eight hours of unattended time -- `bash` in PowerShell resolves to a
broken WSL. PowerShell only from now on; see [[no-bash-scripts-powershell-only]].)

**A serious instrument defect was found and fixed, exposed by the gap-4 fix
itself.** `run_phaseB_pump_only.py` wrote `dt_s` as
`np.mean(np.diff(t))`, but the two kernels write `t` differently: the paper
(guarcello) kernel's axis advances by the stored-sample spacing, the JC
kernels' by the integrator step. `stroboscopic_section` expects the JC
convention and multiplies by `record_stride`, so guarcello's value was
double-strided. At `--record-stride 4` the strobe therefore stepped a
**quarter pump period**, and a period-1 orbit produced four phases:
`sigma` equal to the full signal amplitude, `D2 = 1.02` and `K = 0` at every
power from -70 dBm up -- a torus reading everywhere, entirely spurious.
Before the stride fix the eight-sample guard had accidentally hidden this by
forcing the Poincare fallback. Fixed by `_integrator_dt_s`; the -70 dBm strobe
spread drops **3.45e-05 -> 3.60e-09**. The traces were always valid -- only the
metadata field was wrong -- so 81 existing points were repaired in place and
re-reduced with no re-integration.

**guarcello, with a real stroboscopic section (1049 points, was 300):**

| range [dBm] | sigma | K | reading |
| --- | --- | --- | --- |
| -70.00..-54.05 | 3.6e-9 -> 5e-8 | -0.721 constant | period-1; statistics on noise |
| **-54.00..-53.55** | 5.2e-6 -> 1.4e-5 | **+0.014..+0.080** | **regular window, 0.45 dB** |
| -53.50..-53.40 | 2.2-3.1e-5 | 0.25 -> 0.66 | transitional |
| -52.50..-45.00 | 3.4e-5 -> 1.5e-4 | 0.991-0.998 | chaotic |

`sigma` jumps **x100 in one 0.05 dB step** at -54.00. The constant `K = -0.721`
across the whole floor is the not-classifiable signature, consistent with the
standing rule -- it is a fixed artefact of running the test on a constant
series, not a measurement.

**jc_fqjtwpa, 2100 periods, 13 points:** transition between -32.0 (K = 0.104)
and -31.8 (K = 0.992). **Gap 2 is NOT closed for this device** -- the scan
range -32.0..-30.8 was chosen wrong and is 11/13 chaotic, so its regular
window lies below -32.0 and was never sampled.

### Gap 3 resolved, with a negative answer: there is no normal-form exponent

Dense onset data now exists on three devices (jc_jtwpa 0.025 dB, ipm_2c_fixed
0.0025, guarcello 0.05). Fitting `sigma = C (A - A_c)^beta` in **linear pump
amplitude** (a power law in dB is meaningless) is unstable across every
reasonable window:

| device | window | beta |
| --- | --- | ---: |
| jc_jtwpa | -29.50..-29.23, 16 pts | 1.23 +/- 0.13 |
| jc_jtwpa | -29.50..-29.33, 11 pts | 2.13 +/- 0.38 |
| ipm_2c_fixed | 0.5850..0.6000, 7 pts | 0.36 +/- 0.32 |
| ipm_2c_fixed | 0.5850..0.5950, 5 pts | 10.82 +/- 3.24 |

**beta spans 0.36 to 10.8 on data 10-20x denser than the previous grid.** The
cause is not sampling: `sigma` grows **210x across 0.27 dB** on jc_jtwpa (a 3.2%
change in amplitude), x100 in 0.05 dB on guarcello, x300 across 0.01 in
`I/I_bound` on 2c. A supercritical Neimark-Sacker grows as
`(A - A_c)^(1/2)`, so reproducing 210x inside a 3.2% amplitude span requires
`A_c` to sit within 1e-6 of the window's lower edge, which leaves no lever arm
to measure the exponent with.

So the transition is **hard on every device even at 10-20x finer sampling**,
and the invariant circle appears at finite size rather than growing from zero.
That is a physical statement -- a subcritical or otherwise discontinuous onset
-- not a fit failure, and it means **no supercritical normal form applies and
no beta should be quoted**. The earlier "the transition is graded, not a clean
supercritical bifurcation" reading was right about the conclusion and wrong
about the mechanism: it is graded in `K`, abrupt in `sigma`.

The summary `normal_form_fit` emitted by `run_nonlinear_diagnostics.py` is
still a GLOBAL fit over the whole sweep including the saturated region, so its
numbers (guarcello 3.88, jc_jtwpa 6.60, ipm_2c 4.96, jc_fqjtwpa 0.19) measure
saturation and must not be quoted. That defect is unchanged from 2026-08-16.

### WHY HB FAILS: it dies at the torus onset, and that is physical (2026-08-17)

The HB columns die exactly where the T-periodic solution stops existing. Two
devices, independent columns, measured against today's pump-only FDTD:

| device | HB last converged | HB first failure | FDTD torus onset | FDTD chaos onset |
| --- | ---: | ---: | ---: | ---: |
| jc_jtwpa | -29.6842 dBm | **-29.0526 dBm** | **-29.44 dBm** | -29.06 dBm |
| jc_fqjtwpa (fine) | -32.1356 dBm | **-31.9322 dBm** | transition bracket (-32.0, -31.8) | |

jc_jtwpa's HB column steps 0.632 dB, so its failure bracket is
`(-29.684, -29.053]` -- the torus onset at -29.44 sits **inside** it, and the
first failed point coincides with the end of the K~0 window and the start of
chaos. jc_fqjtwpa's fine column brackets the failure to `(-32.136, -31.932]`
against an FDTD regime change in `(-32.0, -31.8)`; the two overlap.

**Mechanism.** HB assumes `phi(t) = sum_k X_k exp(i k omega_p t)`, strictly
T-periodic. At the torus onset the attractor acquires a second incommensurate
frequency -- measured `f_a/f_p = 0.1217` (jc_jtwpa), `0.0917` (ipm_2c_fixed),
`0.3555` (guarcello) -- carrying 1.9%, 9.6% and 0.26% of in-band power. **The
pump-only basis has no function that can represent it.** The period-1 orbit
loses stability through a Neimark-Sacker bifurcation, and above the onset
Newton is chasing a solution that is at best unstable.

**The failure signature confirms this rather than a step-size problem.** All
columns report `stalled at Newton N (reduction ratio 0.81-0.88)`. A reduction
ratio approaching 1 means Newton is no longer reducing the residual, i.e. the
Jacobian is near-singular in one direction -- exactly what a complex
multiplier pair crossing the unit circle produces. It also explains the
2026-08-15 finding that fixed continuation ladders fail where solutions exist
and adaptive ones need `dlambda` down to `0.004454`: below onset the solution
survives but its basin shrinks as the multipliers approach the circle.

**Not a drive limit.** At jc_jtwpa's last converged point
`pump_branch_current_max_over_ic = 0.6168` and `branch_min_cos_phase = 0.787`.
The junctions are comfortably under-driven, consistent with
[[fold-plan-g1-5-fold-is-basis-converged]].

**RETRACTION: "the high-power wall is numerical" is withdrawn**
([[jtwpa-high-power-wall-also-numerical]]). That verdict rested on PALC
reporting zero `fold_lambda` events through -24 dBm. PALC searches for a
**fold** (a real multiplier through +1). A Neimark-Sacker is a **complex pair**
crossing, which a fold locator cannot see by construction. Finding no fold was
correct and meant the opposite of what was concluded: the wall is a genuine
loss of stability of the periodic orbit, not a solver artifact.

**What this buys the solver.** The auxiliary-generator closure is now
justified by direct measurement and its cost is known: one extra generator
explains **0.999 / 0.986 / 1.000** of the off-comb power in the window
(jc_jtwpa / ipm_2c_fixed / guarcello), and a lattice of `n <= 6, |m| <= 2`
reaches 99.9% of in-band power on all three (5 / 4 / 3 lines actually carry
power). That would extend HB from the current wall through the torus window,
about 0.35-0.40 dB on jc_jtwpa. Past chaos onset it does not help: one
generator explains only 0.19 / 0.21 / 0.17, and a full `n <= 12, |m| <= 6`
lattice (858 lines) still captures only 63.8% of in-band power on jc_jtwpa and
90.8% on guarcello. **There is no finite tone set for the chaotic regime.**

### Figures

`scripts/chaos/plot_nonlinear_diagnostics.py` -> `outputs/chaos/figures/`.
Three devices: jc_jtwpa (31), ipm_2c_fixed (11), guarcello (87). Representative
points are chosen by rule inside `DeviceSet.representatives` -- best-resolved
floor point, D2 plateau closest to 1, smallest |K|, largest K -- never named per
device, so adding a device changes no threshold.
Every measured panel is paired with a synthetic reference whose answer is known
in advance (periodic point, golden-mean rotation, Henon map):

- `regime_map.png` -- sigma and K against control for both devices, regimes
  shaded by thresholds fixed before plotting (floor 1e-7 V, regular |K| < 0.30,
  chaotic K > 0.90).
- `return_maps.png` -- stroboscopic return maps. The jc_jtwpa window panel at
  -29.1629 is a clean closed curve; this is the single most direct evidence for
  the torus and needs no statistic to read.
- `zero_one_translation.png` -- the 0-1 test's own (p, q) variables. Bounded
  annulus at -29.1629, unbounded random walk at -28.0545, matching the two
  reference systems.
- `d2_and_scaling.png` -- D2 against embedding dimension with the D2 = 1
  invariant-circle line, and sigma on approach against the supercritical
  Neimark-Sacker `(mu - mu_c)^(1/2)` law. **The measured approach is far
  steeper than the square-root law**, consistent with the separate finding that
  beta is NOT_ESTABLISHED and the curve is a sigmoid.

Also confirmed here: the `on_lattice` collapse at -27.8 dBm is **not** a
pump-only bifurcation. K is flat at 0.994-0.999 across -28.6..-27.4, straight
through it.

### jc_jtwpa beta is NOT_ESTABLISHED -- the curve is a sigmoid, not a power law

Local windowed fit above onset -29.6556:

| window | pts | beta | R^2 |
| ---: | ---: | ---: | ---: |
| 40% | 5 | 6.1323 +/- 1.1011 | 0.912 |
| 60% | 8 | 3.5702 +/- 0.4126 | 0.926 |
| 100% | 17 | 1.7277 +/- 0.1936 | 0.841 |

beta *increases* as the window narrows, the opposite of a converging
normal-form exponent, and beta = 6 is no normal form. sigma rises x64 in one
step then saturates at 2-4e-05: a saturating sigmoid. A power law fitted to a
sigmoid gives exactly this pattern. The onset is under-resolved -- which the
running sweep addresses.

### Agent results 2026-08-16 evening: 2 accepted, 1 rejected

| task | verdict | disposition |
| --- | --- | --- |
| A: Arnold tongues | `LOCKING_NOT_SUPPORTED` | **REJECTED** -- biased test |
| B: Lambda_1 | refs PASS, device `NOT_ESTABLISHED` | **ACCEPTED**, cause diagnosed |
| C: Z2 symmetry | measured | **ACCEPTED** |

**Task C accepted.** Bases were dense (`[1,2,3,4,5]`, `[1..7]`), so even
harmonics were free to be non-zero and came out at 2.19e-29 against a 4.46e-14
fundamental; biased RF-SQUID gives 2.10. The circularity trap was avoided.
Caveat not flagged by the agent: the 2c solution is from
`outputs/exp08_full_ipm_pump`, a **stale circuit** ([[real-designs-live-in-designs-not-outputs]]).
The conclusion is structural so it survives, but do not cite that file as current.
Corrected classification wording: **with Z2 present, `lambda = +1` admits a
pitchfork or transcritical branching, not only a saddle-node fold.**

**Task B accepted, and the failure is diagnosable.** Reference exponents are
within 1 percent (Lorenz 0.90363, Henon 0.42018, Rossler 0.07204, damped
-0.10006) so the instrument is real. The device numbers
(8.62e10..8.69e10 s^-1, identical at all three powers including one that must
be periodic) are numerical: `8.6211e10 * dt_s 1.1474e-12 = 0.0989 per step`,
i.e. the tangent grows ~10 percent *per timestep*. Growth proportional to
`1/dt` is the signature of **explicit integration of a stiff tangent**.
guarcello's fastest linear mode is ~1027 GHz while `1/dt` = 872 GHz, so that
mode is above the sampling rate and an explicit tangent step is
unconditionally unstable regardless of the physics. **Fix: step the tangent
with the same implicit scheme as the state, reusing the banded factorization
already computed for the state step.** This also explains why renormalization
interval made no difference -- it tests the wrong thing.

**Task A rejected.** `p = 0.9999998` means observed errors are *systematically
larger* than chance, which is a broken test, not a null result. Two causes:
30 of the 46 "surviving" points had `off_lattice < 0.01` (no off-lattice
content, so rho is a random argmax over noise, and the prominence gate alone
does not catch these); and 33 percent of surviving rho sit below 1/13, the
smallest rational in the q<=13 set, where relative error is structurally huge.
The agent did correctly find that the CSV holds **56 rows, not the 68** stated
in `CLAUDE.md` -- that figure is stale.

### What the correctly-gated tongue test actually found

Gating on `off_lattice >= 0.01` AND prominence >= 1.5 leaves **16 points**, and
the six JC post-collapse points drop out as predicted. rho is *pinned* across
pump power on three devices -- but it is not an internal generator:

| device | rho | lattice combination | rel. err |
| --- | ---: | --- | ---: |
| jc_fqjtwpa | 0.1264 | \|-2 + 2 f_s/f_p\| = 0.126582 | **0.14%** |
| jc_jtwpa | 0.2813 | \|-4 + 4 f_s/f_p\| = 0.280899 | **0.14%** |
| ipm_2c_fixed | 0.0656 | \|-1 + 1 f_s/f_p\| = 0.063291 | 3.5% |
| guarcello | 0.0276 | nearest is 0.0829 | **200%, genuinely not** |

`rho = q |1 - f_s/f_p|` for q = 1, 2, 4 -- the **pump-signal detuning and its
harmonics**, which are lattice points `(-q, +q)`.

### Instrument defect: `SIGNAL_ORDER = 4` makes `on_lattice` under-report

`measure_ansatz_validity.py:48`. A generator at 4x the detuning puts its own
harmonics at `(-8,+8)` and `(-12,+12)`, outside the `|m| <= 4` mask. Measured
by raising `SIGNAL_ORDER`:

| device | ctrl | m<=4 | m<=8 | m<=12 | m<=16 |
| --- | --- | ---: | ---: | ---: | ---: |
| jc_fqjtwpa | -31.5 | 0.9346 | 0.9978 | 0.9978 | 0.9978 |
| jc_jtwpa | -28.6 | 0.9558 | 0.9674 | 0.9717 | 0.9717 |
| jc_jtwpa | -28.2 | 0.9233 | 0.9416 | 0.9442 | 0.9444 |
| guarcello | -53.0 | 0.9526 | 0.9526 | 0.9528 | 0.9528 |
| ipm_2c_fixed | 0.575 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| jc_jtwpa | -27.8 | 0.0564 | 0.0925 | 0.1058 | 0.1081 |

**fqjtwpa's pre-collapse off-lattice was 97 percent mask truncation** -- true
value 0.9978, not 0.9346. jtwpa's is about 30 percent artifact. guarcello and
2c are unaffected, consistent with guarcello's generator not being a lattice
combination. Post-collapse residue is genuine on every device. Values saturate
by `m <= 12`; **raise `SIGNAL_ORDER` to 12.**

**Correction to an earlier claim in this session:** "HB is already 4.4 percent
wrong at -28.6 dBm" should read **2.8 percent**. Real, but smaller. The
pump-only K result is unaffected -- pump-only has no signal, so no IMD
confound exists there at all.

### Instrument defect: `trace.npz['t']` is not consistent across devices

| device | `median(diff(t))` | `dt_s` | ratio | `record_stride` |
| --- | ---: | ---: | ---: | ---: |
| guarcello | 1.147e-12 | 1.147e-12 | **1.0** | 20 |
| ipm_2c_fixed | 8.477e-13 | 4.239e-14 | 20.0 | 20 |

guarcello's stored time axis omits the record stride, so its span claims 600
pump periods where `n_steps * dt_s` says 12000. Any code that derives a rate
from `t` is wrong by 20x on that device. `stroboscopic_section` now derives the
sample spacing from `pump_hz`, `dt_s` and `record_stride` instead of reading
`t`. Guarcello's existing 299-point sections are a valid 20-fold decimation of
the true strobe (20 periods is an integer multiple of one), so D2 and sigma
stand; only K is lost, to sample count.

---

## 5. Closed / killed

- **Period-N, half-pump 2T basis.** `on_half` adds 0.5 pp on 2c; all Hill roots
  are complex pairs with phase far from pi. G6's original
  `lambda_to_minus_1_half_pump_2T_basis` inference is superseded.
- **Torus for Group A.** Post-collapse prominence 1.13-1.17 means the best
  generator barely beats a random one. The apparent "2-torus coverage 0.30" was
  coincidental masking.
- **`tan_delta` route to decidability.** Superseded by RCSJ, which keeps `C`
  real, `has_loss` False, Tier-2 alive, and the TD driver unchanged.
- **Continuation-ladder work, sideband-count studies, runtime studies.** None
  touch the north star.
- **jc_jtwpa / jc_fqjtwpa above collapse.** Physical continuum. No HB variant
  reaches it. Deliverable is a validity boundary, not a solver.

---

## 6. Instrument defects found

| defect | location | status |
| --- | --- | --- |
| Hill root selection had no spurious-root filter | stability sweep | **fixed** (A2), tolerance 0.12085 GHz = half the 241.7 MHz comb |
| Retained Hill candidate count 8 too small — branch hole at 6.1047e-06 | Hill scan | open, A4 |
| Generator scan capped at `0.5*f_p`; fqjtwpa returns 0.4975 two steps from the cap | `measure_ansatz_validity.py::_best_generator` | open — any pre-transition `f_a/f_p` near 0.5 is unreliable |
| Full-residual gate applies only to 3WM | `run_gain_map.py:1591` | open; harmless on 2c (`time_rel` 1e-12) but the gate is advisory for every 4WM run |
| `full_residual_threshold` defaults to `None` | `pump/validation.py:45` | open; only `run_period_doubled_branch.py` passes it |
| `top20` / `generator_share` degenerate where `off_lattice ~ 0` | same | understood — exclude points below `off_lattice = 0.01` |
| `poincare_clusters` inverted on fqjtwpa (834 clean -> 3 collapsed) and non-monotone on jtwpa (1 -> 1011 -> 1 -> 4) | phaseB `result.json` | **open — do not use as a transition indicator.** `verdict` inherits it |
| `sigma_vprime_ps` non-monotone on jtwpa (peaks 4.12e6 *before* the transition) | same | open — clean on 2c (x23) and guarcello, weak on fqjtwpa (x1.8) |
| `run_phaseB_overnight.py` `--dry-run` returns before the `tmax_norm` print | `run_phaseB_overnight.py:200` vs `:202` | open — cannot verify `--periods` from a dry run |

---

## 7. Retractions made this session

Listed so they are not re-derived.

| claim | why withdrawn |
| --- | --- |
| "HB is already 4.4% wrong at -28.6 dBm on jtwpa" | `SIGNAL_ORDER = 4` truncation; the real figure is 2.8% |
| "fqjtwpa degrades before collapse (on_lattice 0.9346)" | 97% of that was mask truncation; true value 0.9978 |
| "2c: no torus exists" (stated flatly) | only excluded on scales > 0.025 in `I/I_bound`; a narrower one is not excluded |
| "the ansatz campaign has 68 points" | the CSV holds 56; the `CLAUDE.md` figure is stale |
| jtwpa event-1 ETA "~18:45" | quoted the script's own ETA, which uses 600-period rates; actual was ~70 min |
| "2c dies at a genuine fold" | the fold is 4.81 dB above; the wall is a stability loss |
| "neither a 2T basis nor a torus reproduces the post-transition regime" | wrong for 2c — a torus covers 0.91-0.98 there |
| "three independent measurements agree on rho = 0.092" | two are algebraically the same quantity; the FDTD match is 1 of 5 points |
| "`R/Rn = 1e2` is the right rung" | 16x more damping than the instability it must resolve |
| "Group A 2-torus coverage 0.30" | prominence 1.15 — coincidental masking, real content is 0.05 + continuum |
| "the ansatz is exact where the device amplifies" (as a general claim) | true only below each device's transition; 2c amplifies to 12.9 dB at `on_lattice` 0.886 |
| "2c's transition is HARD, sigma jumps 1600x in one step" | grid artifact — at 0.005 spacing it is a graded climb through four points plus a regular window |
| "2c excludes a supercritical Neimark-Sacker" | rested entirely on the single-step jump above |
| "do not build the auxiliary-generator closure for 2c, there is no torus" | withdrawn with the exclusion; 2c has a regular window at 0.590-0.605 |
| "the candidate torus rests on a single point" | superseded — five consecutive points at `\|K\| < 0.008`, 0.14 dB wide |

---

## 8. Standing rules

- An HB non-convergence is not a physical boundary. Re-run at <= 0.25 dB steps
  from the last converged checkpoint first.
- Never quote diagnostics from a non-converged Newton iterate as physical state.
- `d1`, `tau_periods`, `decay_aware.trend_b` are retired. Use the `max_abs_phi`
  envelope slope, threshold 1e-5/period.
- Flag `max_abs_phi > 5 rad` as BLOWUP, not growth.
- Do not report a Floquet crossing on a circuit with `has_loss = False`.
- Write artifacts per setting and per checkpoint, atomically. Seven prior long
  runs were lost to end-buffered writes.
- State a threshold before looking at the data it judges.
- `K`, `D2` and `sigma` are uninterpretable while `sigma` sits at the numerical
  floor (~1e-8 on these devices). Floor points give `|K|` up to 0.53 and a
  spurious `D2 ~ 1`. Report them as NOT CLASSIFIABLE, never as periodic.
- `K ~ 0` means regular, not quasi-periodic. The 0-1 test cannot separate
  periodic from quasi-periodic; only `Lambda_1` can (negative vs zero).
- Before resuming a long job the harness reports as killed, check the real
  process table for the script name. The kill does not propagate to descendants,
  and relaunching gives two chains racing on one output directory.
- No `R/Rn` value is a device property. Physical `R/Rn` at 15 mK is ~1e60.
<!-- return-map section follows -->

## 9. Return-map characterization suite (2026-08-17)

Implemented in [return_map.py](/D:/Projects/Thesis/twpa_jax/scripts/chaos/return_map.py) and [plot_bifurcation.py](/D:/Projects/Thesis/twpa_jax/scripts/chaos/plot_bifurcation.py). The primary fixed coordinate is the within-period pair `(v(t_n), v(t_n + Delta))`; the strobe-to-strobe pair is retained as a secondary return map. The delay was selected with the existing `mutual_information_delay` helper on one median-control continuous trace per device, then frozen for that device.

### Validation gate

The gate passed before device analysis. The `validation` object in each [return-map JSON](/D:/Projects/Thesis/twpa_jax/outputs/chaos/return_map/) records:

| Reference | Result |
|---|---|
| Fixed point | PASS; period-1, rotation gate OFF |
| Exact period-2 | PASS; `q_min=2` |
| Exact period-5 | PASS; `q_min=5` |
| Golden-mean circle map | PASS; `rho=0.61803398`, torus |
| Locked circle map `rho=2/5` | PASS; `q_min=5`, locked |
| Henon map | PASS; no period, chaos |

### Per-device descriptor tables

The complete per-point tables, including every `mu`, are stored in the three JSON artifacts. The following table gives the measured ranges and representative transition points; `NOT_ESTABLISHED` is preserved in the JSON when the rotation gate is off.

| Device | `mu` | `q_min` | `d_1` | `r_RMS` | `r_std/r_mean` | `rho` | `f_a` (Hz) | locking |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| jc_jtwpa | -30.5 | 1 | 0.6364004 | 2.171984e-08 | 0.7970292 | 0.1148471 | 8.177113e+08 | UNLOCKED |
| jc_jtwpa | -29.198117 | 8 | 0.7548811 | 1.214506e-05 | 0.2432436 | 0.8784777 | 6.254761e+09 | LOCKED |
| jc_jtwpa | -27.4 | 27 | 1.177963 | 3.467150e-05 | 0.5258449 | 0.9869186 | 7.026861e+09 | LOCKED |
| ipm_2c_fixed | 0.575 | 31 | 1.027151 | 9.663039e-09 | 0.5951743 | 0.9132009 | 7.214287e+09 | LOCKED |
| ipm_2c_fixed | 0.6 | 11 | 0.4823064 | 2.091578e-05 | 0.4545475 | 0.9738007 | 7.693025e+09 | UNLOCKED |
| ipm_2c_fixed | 0.625 | 1 | 0.4731940 | 3.329257e-05 | 0.4421118 | 0.9916038 | 7.833670e+09 | LOCKED |
| guarcello | -70.0 | 2 | 0.0424614 | 5.111034e-09 | 0.6040251 | 0.0004774 | 3.341790e+06 | UNLOCKED |
| guarcello | -53.8 | 31 | 1.796125 | 1.509994e-05 | 0.4105747 | 0.3554821 | 2.488375e+09 | UNLOCKED |
| guarcello | -45.0 | 13 | 0.9362109 | 1.080090e-04 | 0.7538603 | 0.9842508 | 6.889755e+09 | LOCKED |

The exact per-point table is the `points` array in [jc_jtwpa.json](/D:/Projects/Thesis/twpa_jax/outputs/chaos/return_map/jc_jtwpa.json), [ipm_2c_fixed.json](/D:/Projects/Thesis/twpa_jax/outputs/chaos/return_map/ipm_2c_fixed.json), and [guarcello.json](/D:/Projects/Thesis/twpa_jax/outputs/chaos/return_map/guarcello.json).

### Geometric/spectral cross-check

The comparison uses `second_generator.verdict = TORUS` from [lyapunov_kantz JSON](/D:/Projects/Thesis/twpa_jax/outputs/chaos/lyapunov_kantz/) and compares modulo `rho`, `1-rho`, and integer shifts. Per-point rows are in each return-map JSON.

| Device | TORUS points | Best alias branch | Mean absolute difference | Maximum absolute difference |
|---|---:|---|---:|---:|
| jc_jtwpa | 19 | `1-rho` for all 19 | 0.0275881 | 0.0671108 |
| ipm_2c_fixed | 9 | `1-rho` for all 9 | 0.0420643 | 0.0766902 |
| guarcello | 19 | `rho` for 16; `1-rho` for 3 | 0.187758 | 0.496855 |

The first two devices show branch-consistent agreement at some torus points but not uniformly. Guarcello has a substantial disagreement; it is not smoothed over.

### Swept-figure conclusions

The primary figures are [jc_jtwpa](/D:/Projects/Thesis/twpa_jax/outputs/chaos/figures_20260817/jc_jtwpa_sweep.png), [ipm_2c_fixed](/D:/Projects/Thesis/twpa_jax/outputs/chaos/figures_20260817/ipm_2c_fixed_sweep.png), and [guarcello](/D:/Projects/Thesis/twpa_jax/outputs/chaos/figures_20260817/guarcello_sweep.png). The scalar strobe shows continuous thickening near the `jc_jtwpa` and `ipm_2c_fixed` onsets. Guarcello shows a direct finite-band jump in the sampled power interval. These are figure-level classifications and do not establish the local normal form.

Descriptor class counts from the final JSON are: `jc_jtwpa` = 4 torus, 36 chaos; `ipm_2c_fixed` = 16 chaos; `guarcello` = 77 chaos. No device point passed the empirically separated period-floor gate. The absence of period-1 and period-q labels is therefore a measured limitation of this scalar section, not an imputation from prior boundaries.

### CORRECTION (2026-08-17, same day): two instrument defects invalidated
### the section above; re-measured with `return-map-v3`

The class counts and the "route not established" reading in the two paragraphs
above are **withdrawn**. Both came from defects in the suite, not from the data.

**Defect 1 -- the radius floor was set at machine epsilon.**
`describe_section` used `radius_floor = eps * max|x|`, about `1e-20` here, while
the strobe interpolation's actual floor is `~2e-4` *relative* to the section
amplitude -- twelve orders of magnitude apart. Because `d_q` is normalised by
the section's own spread, floor noise is indistinguishable from a chaotic
scatter, so every quiescent point was classified `chaos`: **129 of 133 points
across the three devices**, including `jc_jtwpa` at `-30.5` dBm, `ipm_2c_fixed`
at `0.5750` and `guarcello` at `-70` dBm, all of which are period-1 to machine
precision by `on_comb`, `K` and `D2`. The claim that "no device point passed the
period-floor gate" was a property of the gate, not of the devices.

Fixed with a relative floor, `STROBE_FLOOR_RELATIVE = 1.0e-3`, placed in a gap
that is measured rather than assumed: the quiescent relative radius is
`2.9e-4..7.0e-4` (`jc_jtwpa`), `2.2e-4..3.2e-4` (`ipm_2c_fixed`) and
`1.8e-4..2.2e-4` (`guarcello`) -- a common floor across three physically
different devices, as a numerical floor should be -- while the first point past
each independently established onset jumps to `1.4e-3` and `1.9e-3`.

The period-1 boundary then reproduces the onsets established by the independent
instruments (0-1 test `K`, `D2`, spectral lattice coverage):

| device | period-1 ends | first non-period-1 | established torus onset |
| --- | ---: | ---: | ---: |
| `jc_jtwpa` | -29.475 | **-29.450** | -29.444 |
| `ipm_2c_fixed` | 0.5800 | **0.5850** | 0.5850 |
| `guarcello` | -54.05 | **-54.000** | `<= -53.55` (a bound, not a value) |

**Defect 2 -- the guarcello section coordinate was degenerate.** The AMI delay
returned 7 samples on a trace oversampled at 622.5 samples per pump period, i.e.
`Delta = 1.1%` of a period against `19.7%` and `28.1%` on the other two devices.
`corr(z1, z2) = 0.99985` median, 45 of 77 points above `0.999` -- a line, not a
plane. A degeneracy guard now rejects `|corr| > 0.99` and falls back to the
quarter period.

**The guard does not rescue guarcello, and that is itself the finding.** Scanned
across delays from 5% to 50% of a period, `|corr|` never falls below `0.987`.
Guarcello's stored `v_out` therefore admits **no** two-dimensional section at any
delay: its stroboscopic variation is essentially pure amplitude modulation with
no independent phase degree of freedom in this observable. Guarcello's `r_std /
r_mean`, circle test and locking verdict are consequently not measurable from
stored data, and the earlier "guarcello cross-check disagrees substantially"
reading was an instrument reporting on a line.

**The cross-check passes decisively, and the earlier aggregate hid it.** Averaging
the geometric-versus-spectral difference over all points mixes the torus window
with deep chaos, where `rho` is not defined. Taken at the points just past each
onset, geometric `rho` from the return-map angle and the spectral generator ratio
-- two fully independent routes -- agree to:

| device | control | geometric `rho` | spectral `f_a/f_p` | difference |
| --- | ---: | ---: | ---: | ---: |
| `jc_jtwpa` | -29.1981 | 0.878478 | 0.121446 | **7.6e-05** |
| `jc_jtwpa` | -29.0926 | 0.877935 | 0.121980 | 8.5e-05 |
| `ipm_2c_fixed` | 0.5900 | 0.910782 | 0.092614 | 3.4e-03 |
| `guarcello` | -53.9000 | 0.645786 | 0.354238 | **2.4e-05** |
| `guarcello` | -54.0000 | 0.648864 | 0.351034 | 1.0e-04 |

All match on the `1 - rho` branch, which is a sign and not an ambiguity: the
return map winds backwards, so `rho_geo = 1 - f_a/f_p`. Five consecutive
`jc_jtwpa` points hold `7.6e-5..2.4e-4` over a contiguous 0.14 dB band. Where the
difference degrades to `2e-2..6e-2` the *spectral* estimator has jumped to an
alias (`0.040289 ~= 0.121446 / 3`), not the geometry.

**Closedness criterion replaced (`return-map-v4`).** The `torus` label used
`r_std / r_mean <= 0.271`, calibrated on a round golden-mean circle map. A torus
section reconstructed from a scalar delay pair is a closed *curve* -- generally
an ellipse or Lissajous loop -- so a modest eccentricity failed a roundness test
even on a perfectly good invariant circle. `jc_jtwpa`'s ratio moves smoothly
`0.58 -> 1.16 -> 0.16` across its window; that is a shape changing, not a regime
changing.

`section_dimension` now tests closedness by D2 saturation on the strobe series
(`nonlinear_diagnostics.correlation_dimension`, m = 2, 3, 4, delay 1, Theiler 2),
which asks the right question: a fixed point gives 0, an invariant circle 1, a
chaotic set more, and saturation across `m` separates a low-dimensional set from
noise. Roundness is retained only as a fallback where D2 cannot be measured, so
a torus is never claimed on the criterion known to be too strict.

Thresholds set in measured gaps: golden-mean circle map gives `D2 = 1.032`,
spread `0.0077`; Henon gives `D2 = 1.208`, spread `0.0993` (literature 1.22, so
the estimator is sound). Saturation *spread* is the stronger discriminator -- a
factor of 13 against 6 -- because an invariant circle has an exact integer
dimension and must be `m`-independent, while a strange attractor's estimate
drifts upward with `m`. Hence `D2_SATURATION_SPREAD = 0.05`,
`D2_CIRCLE_TOLERANCE = 0.12`.

Adding this test initially mislabelled Henon as a torus, and the validation
harness passed anyway because the Henon case asserted only "no period found".
The case now asserts the classification itself.

A torus window is confirmed on two of three devices, in each case coinciding
with the band where the independent `rho` cross-check agrees to `1e-4`:

| device | torus points | D2 | matches `rho` band |
| --- | --- | ---: | --- |
| `jc_jtwpa` | -29.1981..-29.0926 | 1.113 | yes |
| `guarcello` | -53.9500..-53.8000 | 1.051 | yes |
| `ipm_2c_fixed` | none | `INSUFFICIENT_DATA` | not measurable |

`guarcello` is included despite its degenerate 2-D section because D2 is
computed on the scalar strobe series and does not need one. `ipm_2c_fixed`
returns insufficient data rather than a negative: with ~1049 strobes the
Eckmann-Ruelle bound `N > 10**D2` supports D2 only to about 3, and fewer than
two of its three embedding dimensions yielded an estimate.

**Net:** the period-1 -> torus route is corroborated, not refuted. The
period-1 boundary is now reproduced independently on all three devices, and the
auxiliary frequency is confirmed by two independent measurements to `1e-4`. What
this suite does **not** establish is the local normal form, or the torus label
itself on `ipm_2c_fixed` and `guarcello`.

### Successive-maxima map, movie, and common-scale overlay (`return-map-v6`)

Three additions, all from data already on disk.

**Successive-maxima return map (`successive_maxima`).** `a_{n+1}` against `a_n`
over local maxima of the continuous trace. This samples the trajectory wherever
the output peaks instead of once per pump period, so it touches **none** of the
strobe timebase -- and therefore none of the timebase defects that have bitten
this campaign twice. It reproduces the route independently on `jc_jtwpa`:

| regime | figure |
| --- | --- |
| period-1 (-30.5 dBm) | exactly 3 discrete points (the pump waveform has 3 local maxima per period, so a period-1 orbit is a 3-cycle in the maxima sequence) |
| torus (-29.1981) | those 3 points opened into 6 clean arcs |
| chaos (-29.0574) | the same arcs blurred into clouds with visible folds |

**Its scalar statistic is not a discriminant, and should not be quoted.**
`relative_spread` first normalised by the mean, which is near zero because these
are maxima of an oscillating waveform that include negative peaks; that gave
2.00 / 1.66 / 1.77 for period-1 / torus / chaos, i.e. a number dominated by how
close the mean came to zero. Renormalised to half the peak-to-peak range it is
at least bounded, but still reads 0.862 / 0.708 / 0.340 -- period-1 **above**
chaos, because a period-1 orbit's three maxima span the whole waveform by
construction. The figure discriminates; no single scalar tried here does.

**Common-scale overlay (`*_return_maps_common_scale.png`).** The per-regime
return-map panels each autoscale, which drew a floor-level point and a chaotic
cloud at the same apparent size. The overlay puts all three representatives on
one pair of linear axes plus a log-radius view. The regimes are separated by
three to four orders of magnitude in radius:

| device | period-1 | torus | chaos |
| --- | ---: | ---: | ---: |
| `jc_jtwpa` | ~1e-8 | ~1.3e-5 | ~1.7e-5 |
| `guarcello` | ~1e-8 | ~1e-5 | ~1e-4 |

The overlay also shows visually what `|corr| = 0.998` meant on `guarcello`: its
torus is a closed curve collapsed to a thin sliver in this observable. D2 = 1.05
still identified it correctly, because D2 is computed on the strobe series and
does not need the two-dimensional section.

**Section movie (`--animate`, `*_section_movie.gif`).** Frames sweep the control
axis with the axis limits computed once over every point and then held, so motion
is the attractor changing and never the view rescaling. Off by default: the 3-D
parameterized figure carries the same information statically, and this exists for
presentation.

### Quantities not computable from stored data

Each `trace.npz` contains only `t` and `v_out`. These were deliberately not estimated:

- Global state-space PCA and effective linear dimension `N_99`: rerun FDTD while recording the full state once per pump period, then perform one global PCA.
- Per-node sigma: the same full-state FDTD rerun is required.
- Participation ratio: the same full-state FDTD rerun is required.
- Tangent-space largest Lyapunov exponent: rerun with access to the integrator kernel and state, propagating the Jacobian tangent. The existing trace-based [lyapunov_kantz.py](/D:/Projects/Thesis/twpa_jax/scripts/chaos/lyapunov_kantz.py) remains the only lambda source used here.

### 2c K=5 HB/torus scaling ladder and Inosuisse cross-check (2026-08-17)

The guarded first-pass ladder is implemented in
[benchmark_torus_scaling.py](/D:/Projects/Thesis/twpa_jax/scripts/benchmark_torus_scaling.py).
Each method ran in a fresh child process with atomic artifacts and Windows
working-set/CPU telemetry. The first pass used K=5, not the production K=10
basis. The 2c case used `designs/ipm_2c_fixed`, 7.9 GHz, pump port 4,
requested pump power -25 dBm, `legacy_traveling_wave`, and a null attenuation
override, as required by
[high_power_2c_single_column_context.md](/D:/Projects/Thesis/twpa_jax/docs/development/high_power_2c_single_column_context.md).

For 2c, the original period-1 HB solve converged with coefficient residual
`4.684906446280008e-12`, 12 Newton iterations, and 242 GMRES iterations. Its
child wall time was `4.512435` s and peak working set was `0.153236` GiB.
The K=5 torus solve through Schur/PARDISO converged in one iteration, but the
`q != 0` norm fraction was `9.205732373031188e-19`; it therefore returned to
period-1 and did not establish a torus. Its child wall time was `8.016950` s
and peak working set was `0.983555` GiB. These values are from the atomic
[2c ladder summary](/D:/tmp/torus_scaling_20260817_r2/ipm_2c_fixed_minus25dbm/summary.json).

The closest retained Inosuisse map row at the same frequency region is
`point_index=296`: pump power `-24.89655172413793` dBm and pump frequency
`7.896551724136667` GHz. It is marked `PASS`, with pump current
`6.203032333817978e-06` A, pump coefficient residual
`1.7624281695016535e-13`, and maximum junction current
`1.4112140525324067e-06` A. The row is in
[map_points.csv](/D:/Projects/Thesis/twpa_jax/outputs/Inosuisse/2c_reg/map_points.csv);
the map-wide A10/null-override policy is recorded in
[map_summary.json](/D:/Projects/Thesis/twpa_jax/outputs/Inosuisse/2c_reg/map_summary.json).

The benchmark waveform reconstruction gives maximum junction current
`1.4013171511091051e-06` A at branch index 1255 and utilization
`0.527560136906335`; the reconstruction used the benchmark
[pump_solution.npz](/D:/tmp/torus_scaling_20260817_r2/ipm_2c_fixed_minus25dbm/pump/pump_solution.npz)
and the circuit matrices under `designs/ipm_2c_fixed`. The two current values
are close: the map row is only `0.7013` percent higher. This is a branch/current
cross-check, not a basis-equivalence result: the retained map pump solution
files are absent from the current checkout, so its harmonic-basis metadata
cannot be independently verified from `map_points.csv`.

The -25 dBm point is therefore a converged K=5 HB below-window
performance/control point, not the next torus target. It is not independently
TD-validated or gain-valid under the context document. Its junction-current
profile is numerically consistent with the map's smooth low-utilization
period-1 branch. No physical 2c torus has been established by this run.

#### Prior next step (executed below)

Run the production-basis 2c case at the documented mid-window control
`I/I_bound=0.6050`, using the existing converged pump seed at
`.hybrid_outputs/period1_recovery_7p9_2c_v1/point_-23.800000/pump`. Use K=10
(`1,3,...,19`), Q=1, Schur, `real_coupled_fast`, float64, PARDISO, and one
fresh process per point. Record the full residual, omitted-mode residual,
maximum junction utilization with index, and RSS/CPU telemetry.

Then run the torus solve at that mid-window point with seed amplitudes spanning
at least three decades. Apply the documented below-window control
`I/I_bound=0.5745` before accepting any positive off-comb result, followed by
the near-onset point `I/I_bound=0.5912`. The required gate remains: converged
production residual, positive generator frequency, and non-zero `q=+/-1`
content. If the mid-window K=10 result again collapses to the numerical floor,
the finding is `NOT_ESTABLISHED`, not evidence that the K=5 -25 dBm control
failed physically. Only after a valid K=10 torus is obtained should continuation
through the documented window begin.

### K=10 five-point solver ladder (2026-08-17)

The requested K=10 ladder was completed for five points, using one fresh child
process for the original period-1 HB solve and one for the autonomous torus
solve at each point. The complete atomic results, including wall time, CPU
seconds, peak RSS, residual histories, and q-sector fractions, are in
[ladder_summary.json](/D:/tmp/torus_scaling_20260817_k10/ladder_summary.json).
The original benchmark HB used the existing mean-tangent Newton/Krylov path;
the torus path used PARDISO, with Schur reduction for both 2c cases. The 2c
cases left attenuation unset, so the default A10 model was used.

| case | original HB wall / RSS | torus HB wall / RSS | torus residual | q=+/-1 fraction |
| --- | ---: | ---: | ---: | ---: |
| uniform 418 JJ | 2.009 s / 0.130 GiB | 2.009 s / 0.129 GiB | 2.41e-10 | 3.67e-6 |
| uniform 4x418 JJ | 2.009 s / 0.145 GiB | 2.008 s / 0.163 GiB | 4.81e-10 | 3.67e-6 |
| jc_jtwpa | 2.008 s / 0.161 GiB | 11.520 s / 2.526 GiB | 1.63e-12 | 1.66e-9 |
| ipm_2c_fixed, -25 dBm | 6.013 s / 0.173 GiB | 16.530 s / 3.131 GiB | 5.62e-20 | 2.63e-20 |
| ipm_2c_fixed, -23.8 dBm | 8.018 s / 0.180 GiB | 16.531 s / 3.132 GiB | 5.89e-20 | 1.12e-18 |

The residuals are numerical convergence results only. The jc_jtwpa and both
2c torus attempts collapsed to period-1, as shown by q=+/-1 fractions at the
floor. The uniform-circuit fractions are the retained initial perturbation,
not evidence of a nonlinear torus. The runner accepted the first numerical
convergence and therefore did not complete the intended three-decade seed
sweep. A physical torus is consequently **NOT_ESTABLISHED** by this ladder.

#### Corrected next step

Change the torus acceptance gate so that a small q=+/-1 fraction cannot be
accepted as a torus merely because the residual is small. For the two 2c K=10
points, force the full seed-amplitude sweep and record every attempt. Use the
documented mid-window point (-23.8 dBm) first, then apply -25 dBm as the
below-window control. The control must remain at the q-sector floor before any
positive sideband result is treated as physical. If all amplitudes return to
period-1, report **NOT_ESTABLISHED** and do not start continuation.

### K=10 q-sector gate rerun (2026-08-17)

The q-sector acceptance gate was implemented in
[run_torus_branch.py](/D:/Projects/Thesis/twpa_jax/scripts/run_torus_branch.py).
For the 2c cases, a torus is accepted only when Newton converges and the
q=+/-1 norm fraction is at least `1e-8`. The threshold is an operational gate,
chosen above the completed ladder's period-1 floors; it is not a claim about a
minimum physical torus amplitude.

The rerun used K=10, Q=1, Schur/PARDISO, default A10 attenuation, and recorded
all three seed amplitudes at each point. Results are in the atomic
[gate rerun summary](/D:/tmp/torus_scaling_20260817_gate_rerun/ladder_summary.json).

| point | seed amplitude | Newton converged | q=+/-1 fraction | gate |
| --- | ---: | ---: | ---: | --- |
| -25 dBm | 1e-6 | yes | 2.63e-20 | rejected |
| -25 dBm | 1e-5 | yes | 1.54e-17 | rejected |
| -25 dBm | 1e-4 | yes | 1.26e-16 | rejected |
| -23.8 dBm | 1e-6 | yes | 1.12e-18 | rejected |
| -23.8 dBm | 1e-5 | yes | 9.31e-19 | rejected |
| -23.8 dBm | 1e-4 | yes | 8.15e-18 | rejected |

The final residuals were `6.72e-20` at -25 dBm and `5.87e-20` at -23.8 dBm,
but both points had `physical_torus_gate_passed=false`. The torus is therefore
**NOT_ESTABLISHED** at either requested power. The corresponding torus child
peak RSS values were `3.25 GiB` and `3.25 GiB`; no memory guard fired.

The gate mutation check used a synthetic converged zero-sideband state. With
the `1e-8` threshold it rejected the state after all three attempts; with the
deliberate threshold mutation to `0`, it accepted the first attempt. This
demonstrates that the gate is active rather than merely reported.

#### Next step after the gate rerun

Do not continue the current K=10 branch: neither 2c point produced a physical
torus. The next useful solver experiment is a different seed source, such as
the measured/Floquet sideband direction, while retaining the same q-sector
acceptance gate and the -25 dBm negative control. If that seed also collapses
to period-1, the solver result remains **NOT_ESTABLISHED** rather than a basis
or residual failure.

Tracker status: the gate implementation, deliberate mutation check, complete
2c rerun, and focused verification are finished. No continuation was launched,
and no file under `src/twpa_solver/` was modified.


### Torus amplitude campaign 2026-08-17

The unattended amplitude-parameterized campaign was launched from `D:\Projects\Thesis\twpa_jax\outputs\chaos\torus_campaign_20260817`. Its atomic per-point results and preflight telemetry are the source of truth; incomplete stages remain `NOT_ESTABLISHED`.

- Preflight summary: `D:\Projects\Thesis\twpa_jax\outputs\chaos\torus_campaign_20260817\preflight\summary.json`.
- Campaign summary: `D:\Projects\Thesis\twpa_jax\outputs\chaos\torus_campaign_20260817\campaign_summary.json`.
- Physical torus acceptance remains the non-zero q-sector and converged residual gate; numerical period-1 roots are not reported as torus branches.

### Overnight torus campaign 2026-08-17: post-mortem, five faults, zero physics

The campaign ran 8 minutes and produced **no usable measurement**. Recorded so
none of these is repeated.

**1. The Stage A abort gate was not honoured.** Both device smokes returned
process code `2` (`smoke2.json`, `smoke3.json`: `converged: false`) and the
campaign proceeded to run stages B through G anyway. The specification made
Stage A a hard abort. Every later stage then failed in seconds, which is why 8
hours collapsed into 8 minutes.

**2. The amplitude ladder was 4 to 5 orders of magnitude too small.** The
ladder swept `A_rel` over `1e-6 .. 1e-2`. The measured torus at the primary
artifact (`I/I_bound = 0.6050`) has `relative_radius = 3.213e-1`
(`outputs/chaos/return_map/ipm_2c_fixed.json`). At `A_rel = 1e-5` the absolute
amplitude is `7.3e-19` against a pump of `~1e-13`, i.e. **below the residual
noise floor**, so `dR/domega_a` was pure roundoff. Any future ladder must be
sized from the measured `relative_radius` column, not guessed.

**3. The `(X, omega_a, source_tau)` amplitude formulation is structurally
ill-posed.** Measured at the 2c artifact, `A_rel = 0.15`:

| quantity | value |
| --- | ---: |
| `\|\|dR/domega_a\|\|` | 5.932e-16 |
| `\|\|dR/dtau\|\|` | 3.518e-06 |
| 2x2 closure entries | `1e-27 .. 1e-31` |
| `delta_tau` | 6.909e+15 |
| `\|\|step\|\| / \|\|state\|\|` | 2.254e+16 |

Two separate defects. The first is a **scaling** defect of exactly the kind
`solve_arclength` already had: the unknowns span twenty-two orders of magnitude
(node flux `~1e-13` Wb, `source_tau ~1`, `omega_a ~5e9` rad/s) and the closure
mixed them unscaled. Fixed in `multitone/torus.py` by nondimensionalizing the
2x2 (`omega_scale = omega_p`, `tau_scale`, `state_scale`) plus a trust region
(`max_step_over_state`, default 2.0).

The second defect survives that fix and is **structural**: `source_tau` has
almost no leverage on the two constraint rows. Both the phase anchor and the
amplitude normalization live in the `q != 0` sector, and the drive enters that
sector only by modulating the Hill operator, which is second order in the
perturbation -- measured, `tau`'s effect on the constraint rows is `~1e5`
smaller than `omega_a`'s. The 2x2 is therefore near-singular at any scaling.
**Do not pair the amplitude and anchor constraints with the drive.**

**4. The Stage B spectrum scan computed nothing.** All 120 jtwpa rows returned
`NotImplementedError: rmatvec is not defined` -- `scipy.sparse.linalg.svds` was
called on a `LinearOperator` with no adjoint. `pump/singularity.py::
jacobian_min_eigenvalue` already solves this problem with shift-invert Arnoldi
around the factored preconditioner and needs no `rmatvec`. Reuse it.

**5. Stage B crashed on all three 2c cases**: `could not broadcast (6136,)
into (2518,)` -- the full-node pump state was promoted into a Schur-reduced
basis without restricting to `partition.retained`.

### What the corrected scan then showed

Rerun through `jacobian_min_eigenvalue_with_estimator` (estimator
`shift_invert`, no failures), jtwpa at `-29.40` dBm, `omega_a/omega_p` from
0.06 to 0.16:

```text
0.06 -5.906e+04   0.09 -7.754e+04   0.12 -8.754e+04   0.15 -8.887e+04
0.07 -2.181e+04   0.10 -1.501e+04   0.13 +8.710e+02   0.16 +2.607e+04
0.08 +3.524e+04   0.11 +6.852e+04   0.14 +1.117e+05
```

Sign-flipping and unstructured, with **no dip at the measured 0.1217**. This is
Arnoldi returning a different member of a dense eigenvalue cluster at each
point, the same failure already recorded for the blind Hill scan on these
circuits. **A blind sweep over `omega_a` does not locate the Neimark-Sacker
condition.** The instrument needs branch tracking from a known-stable drive,
which is what `stability/tracking.py` exists for and what the plan already
specified for the Hill route; the campaign scan skipped it.

### Standing conclusion

The torus solver still has no device result. The credible remaining route is
branch switching: track the critical eigenvalue from a known-stable drive,
take its eigenvector at the crossing, seed `solve_newton` (fixed drive, phase
anchor) with it at an amplitude taken from the measured `relative_radius`, and
continue upward. The amplitude-parameterized closure is not a substitute for
that, because its extra parameter has no leverage on its extra constraints.


 ## Confirmed blockers

  1. The public torus driver still invokes the invalid phase-only formulation.

     /D:/Projects/Thesis/twpa_jax/scripts/run_torus_branch.py:165 calls solve_newton, even though /D:/Projects/Thesis/twpa_jax/src/twpa_solver/multitone/torus.py:489 explicitly documents
     that this formulation admits (X_pump, 0, any omega_a).

     Therefore run_torus_branch.py and the scaling benchmark built on it cannot currently compute a physical torus.

  2. The autonomous torus basis cannot represent a normal Hill eigenvector.

     /D:/Projects/Thesis/twpa_jax/src/twpa_solver/multitone/basis.py:236 retains q=±1 only at the odd pump harmonics plus (0,1). However, /D:/Projects/Thesis/twpa_jax/src/twpa_solver/
     multitone/seed.py:154 maps Hill sideband m to the torus tone (m,1), requiring all retained integer sidebands.

     Direct result:

     q rows [(0, 1), (1, -1), (1, 1), (3, -1), ..., (19, 1)]
     ValueError: Floquet sideband -4 maps to missing tone ToneIndex(h=4, q=-1)

     The inability to insert the tracked Hill eigenvector is confirmed. The stronger conclusion—that the odd-only basis selects the wrong parity family for unbiased four-wave mixing—is a     well-supported operator inference, but should be verified from the production khat sparsity.

  3. The amplitude solver’s convergence norm mixes amperes and webers.

     /D:/Projects/Thesis/twpa_jax/src/twpa_solver/multitone/torus.py:338 concatenates the HB coefficient residual with phase and amplitude constraints, then applies an absolute Euclidean     tolerance.

     Consequences:
      - The flux constraints are numerically negligible compared with current residuals.
      - The line search optimizes the same dimensionally inconsistent norm.
      - The smoke fixture declares convergence at iteration zero with a physically weak relative residual.

     Observed smoke result:

     residual = 3.4458e-12
     source coefficient = 5.0e-10
     relative residual ≈ 6.9e-3
     generator amplitude = 5.336e-22

     It passes only because 3.45e-12 < 1e-9 in absolute units.

  4. The proposed q-block singular-value instrument is broken.

     The campaign creates a LinearOperator with matvec only and passes it to svds. SciPy requires the adjoint action.

     Direct reproduction:

     NotImplementedError: rmatvec is not defined

     The JTWPA Stage B artifact contains this error in all 120 rows. The 2c version fails earlier:

     ValueError: could not broadcast input array from shape (6136,)
     into shape (2518,)

     This comes from copying a full-node pump state into a Schur-reduced state.

  5. Campaign gates do not stop failed stages.

     /D:/Projects/Thesis/twpa_jax/scripts/chaos/torus_campaign_20260817.py:450 accepts a smoke stage if it wrote result JSON, even when the worker returned failure and converged=false.

     Existing artifact summary:

     attempted torus points: 17
     converged torus points: 0
     controller_error: None

     The campaign then reports completion. This is the most direct reason the existing campaign output cannot be treated as HB evidence.

  6. Successful amplitude continuation would immediately encounter a frequency-unit defect.

     The ladder stores omega_a_over_omega_p, then supplies that dimensionless number as --warm-omega. The worker interprets it as radians per second.

     A value near 0.09 would therefore replace a physical angular frequency near 4e9 rad/s.

  7. Full and Schur amplitude constraints use different coordinate spaces.

     The campaign defines the target amplitude from the norm of the full 6,136-node pump, but evaluates the torus amplitude on the 2,518-node Schur state.

     Therefore:
      - A_rel changes meaning between full and Schur formulations.
      - It is not directly comparable with the FDTD relative_radius.
      - Full/Schur agreement cannot currently be assessed from this continuation coordinate.

  8. The nominal K=5 campaign actually runs at K=10.

     Stage C is labelled stage_c_k5, but its command does not pass --k; the parser default is 10. In addition, simply truncating a K=10 pump would not constitute a converged K=5 pump
     solution.

  9. The onset fit would pool incompatible points.

     Stage G combines all converged 2c results across stages, Q values, anchors, and full/Schur representations. It then marks the fit established from three points without branch-
     continuity, slope, or fit-quality gates.

     Even after convergence is repaired, that fit must not combine differently normalized branches.

  ## Important secondary gaps

  - run_torus_branch.py resets omega_a to its original guess at each step instead of carrying the solved value.
  - After a failed point, it cold-starts the next point, permitting branch replacement.
  - The exact-frequency problem cache grows without a bound and may retain many full Schur partitions and factorizations during a successful solve.
  - first_lyapunov_coefficient is not a complete Neimark–Sacker normal-form calculation. Its present test only mocks a sign.
  - Torus stability currently assesses the q=0 pump-periodic slice, not the full quasiperiodic torus. It cannot yet establish the unstable torus skeleton or chaos boundary.
  - The transition reporter accepts the first |lambda| >= 1 value without requiring a below-to-above bracket, convergence, or absence of a branch discontinuity.

  ## Test coverage result

  Focused torus/tracking suite:

  The six failures are the known design CLI/compact, KIMPA, and loss-model failures. No additional regression failures arose from the torus area.

  ## What remains valid

  The following work should be retained:

  - Mode-vector Hill root continuation and overlap checking.
  - Re-solving the period-1 pump at every source-drive step.
  - Stopping 2c natural continuation at its pump fold.
  - The equation count of the amplitude-normalized formulation:
    (X, omega_a, source_drive) with the HB residual, phase condition, and nonzero generator-amplitude condition.

  - The measured FDTD period-1 → torus → chaos route and the rejection of period doubling.

  This audit invalidates the existing HB torus campaign, not the FDTD evidence.

  ## Recommended implementation order

  1. Make the q-block singular-value gate work on converged jc_jtwpa and 2c pumps.
  2. Construct the torus q-sector from the actual Hill sideband support and prove that a production Hill eigenvector maps into it.
  3. Nondimensionalize every bordered residual and define a representation-consistent amplitude coordinate.
  4. Add a manufactured nontrivial torus test that converges from multiple phase-rotated seeds.
  5. Run full-versus-Schur equivalence on 418 and 4×418.
  6. Continue nontrivial branches on jc_jtwpa and 2c and compare the recovered onset and rotation number with FDTD.
  7. Implement full torus stability before claiming the unstable skeleton.

  The 418 and 4×418 designs are suitable for scaling, memory, and full/Schur equivalence. They do not have an independent torus reference, so physical validation must come from jc_jtwpa
  and ipm_2c_fixed.

---

## 5. Floquet confirmation and torus handoff (2026-08-18)

This work advances the north star by establishing a physically meaningful
period-1 instability before attempting a nonlinear torus solve. The pump is
solved at every source-drive point; no scaled pump waveform is used.

### Runner changes

`scripts/chaos/run_physical_torus_column.py` now provides:

- cold-start pump solution at the first requested source drive;
- adaptive warm-start pump continuation at subsequent drives;
- explicit loss-model selection and production Schur/PARDISO settings;
- candidate enumeration at the lowest solved drive, with an optional explicit
  frequency seed for a previously identified clustered root;
- mode-vector Floquet continuation with overlap and discontinuity diagnostics;
- `--floquet-only`, which disables the torus corrector for Floquet confirmation;
- incremental main and per-branch CSV output;
- optional incremental per-point mode-vector NPZ checkpoints.

`scripts/chaos/plot_floquet_column.py` produces the growth-rate plot from the
flat branch CSV. The torus corrector in `multitone/torus.py` was not changed or
invoked in the confirmation runs.

### Configuration distinction

The 2c runs use `current_complex_c` for the analytic Hill/Floquet circuit
operator. The external pump-line attenuation remains the measured A10 model
through the production `run_gain_map` path (35.275128996894026 dB at 7.9 GHz).
`current_complex_c` breaks conjugate symmetry and is a stability-analysis
convention only; it must not be used for published gain or compression values.

The production pump basis is `[1, 3, ..., 19]`. In `--floquet-only` mode,
`--k 10` records the later production torus basis but does not alter the Hill
operator; the Hill calculation in these runs uses `--sidebands 5`. The K=5 and
K=10 labels therefore refer to the autonomous torus basis, not to a change in
the Floquet sideband truncation.

### K=5 physical Floquet column

Device: `ipm_2c_fixed`, pump frequency 7.9 GHz, pump port 4,
`current_complex_c`, Schur/PARDISO. The source ladder was -24.30 to -23.70
dBm in 0.05 dB steps. All pump solves converged. The critical branch growth
rates were:

```text
-24.30  -1.3093121e+07
-24.25  -7.0049876e+06
-24.20  -5.896099e+05
-24.15  +6.2434447e+06
-24.10  +1.3547069e+07
-24.05  +2.1399892e+07
-24.00  +2.9867167e+07
-23.95  +3.8989666e+07
-23.90  +4.9002436e+07
-23.85  +6.0165773e+07
-23.80  +7.2195080e+07
-23.75  +8.4740508e+07
-23.70  +9.7915305e+07
```

Linear interpolation gave `P_NS = -24.1956856 dBm`,
`omega_a = 0.72909155 GHz`, and `omega_a/omega_p = 0.092290070`.

### K=10 focused confirmation

The first blind K=10 candidate enumeration did not include the K=5 critical
root. One selected clustered root also showed a mode overlap of 0.003862. That
run is retained as evidence of why candidate generation must be explicit; it
was not interpreted as a physical disagreement.

The corrected run included the lowest-drive seed
`0.72495699945944 GHz`, retained five blind candidates, and tracked the seeded
root through the bracket. Full critical-branch table:

| drive [dBm] | Re(omega) [GHz] | growth [s^-1] | multiplier magnitude | overlap | discontinuity |
| ---: | ---: | ---: | ---: | ---: | :---: |
| -24.30 | 0.7249569995 | -1.3093121e7 | 0.998344016 | -- | false |
| -24.25 | 0.7268596699 | -7.0049876e6 | 0.999113686 | 0.999602008 | false |
| -24.20 | 0.7289021930 | -5.8960994e5 | 0.999925369 | 0.999547073 | false |
| -24.15 | 0.7310966708 | +6.2434447e6 | 1.000790622 | 0.999480745 | false |
| -24.10 | 0.7334496730 | +1.3547069e7 | 1.001716290 | 0.999406936 | false |
| -24.05 | 0.7359803011 | +2.1399892e7 | 1.002712519 | 0.999318632 | false |

The other five tracked branches remained below zero. Scheduled untracked
probes returned best growth rates of `-3.790e6`, `-5.604e6`, and `-4.959e6`
s^-1 at -24.30, -24.20, and -24.10 dBm, respectively. No stronger untracked
root was found.

The K=5/K=10 mode-vector comparison used the same 67,496-component Hill basis:

```text
-24.20 dBm: overlap = 1.000000000000000
-24.15 dBm: overlap = 1.000000000000000
```

### NS refinement

A three-point K=10 refinement was run at -24.200, -24.195, and -24.190 dBm:

| drive [dBm] | growth [s^-1] | Re(omega) [GHz] | multiplier magnitude |
| ---: | ---: | ---: | ---: |
| -24.200 | -5.896099e5 | 0.7289021930 | 0.999925369 |
| -24.195 | +7.336233e4 | 0.7291149343 | 1.000009286 |
| -24.190 | +7.401915e5 | 0.7293291832 | 1.000093700 |

The refined linear estimate is:

```text
P_NS             = -24.1955501 dBm
omega_a          = 0.72909136 GHz
omega_a/omega_p  = 0.092290046
```

This is the same branch and the same generator frequency as the K=5 result.
The independent FDTD reference ratio is 0.0917; the absolute difference is
0.000590046, or +0.6435 percent.

### Validation and artifacts

Focused validation after the runner changes:

```text
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\physical_torus_k10_tests3 tests/test_branch_tracking.py tests/test_floquet_stability.py
19 passed in 0.82s
```

`git diff --check` completed cleanly. `graphify update .` completed with 8,011
nodes; only pre-existing ACL warnings for generated temporary directories were
reported.

Ignored run artifacts:

- `D:\tmp\physical_torus_k5_20260818\`
- `D:\tmp\physical_torus_k10_20260818\`
- `D:\tmp\physical_torus_k10_seeded_20260818\`
- `D:\tmp\physical_torus_k10_refine_20260818\`
- `D:\tmp\physical_torus_mode_compare_k5_20260818\`
- `D:\tmp\physical_torus_mode_compare_k10_20260818\`

The K=10 growth plot is
`D:\tmp\physical_torus_k10_seeded_20260818\growth_rates.png`.

### Current conclusion and next gate

O1 is resolved for the physical 2c period-1 branch: one tracked complex mode
crosses the unit multiplier circle, with no mode-vector discontinuity and no
stronger untracked root in the diagnostic probes. This is a Floquet NS result,
not yet a nonlinear torus result.

The next gate is the K=5 eigenvector branch switch and short torus PALC. It
must use the saved critical Floquet eigenvector, a nonzero branch-switch
arclength condition, and must not use the rejected source-drive amplitude
closure. The K=10/Schur torus continuation, omitted-sector residual test, and
FDTD comparison remain outstanding.

### Single K=5 branch-switch experiment

The manual convention audit passed before the run:

```text
loss_model=current_complex_c
analytic_in_omega=True
conjugate_symmetric=False
circuit_has_imaginary_capacitance_loss=False
```

The result is interpreted only as an analytic stability-analysis result. The
convention must not produce published gain or compression values.

At the refined drive `P_NS = -24.1955501 dBm`, the normal adaptive pump path
was run from a cold start. The authoritative checkpoint contains a converged
period-1 state:

```text
pump_current_a   = 6.721961811076382e-06
pump_coeff_rel   = 2.289350211742537e-12
pump_time_rel    = 2.289365787536506e-12
pump_iterations  = 15
```

The historical K=10 refinement did not persist its pump state or critical
eigenvector, so the requested comparison to that exact state cannot be made.
As a reproducibility substitute, a second independent cold-start solve at the
same drive gave:

```text
e_X_fresh_vs_replay       = 0.0000000000000000e+00
mode_overlap_fresh_replay = 0.9999999999999998
```

The fresh K=5 Floquet seed was:

```text
signal_real_ghz       = 0.7290914538559431
signal_imag_ghz       = -3.7247328545705696e-08
growth_rate_per_s     = 2.3403186745006883e+02
multiplier_magnitude  = 1.0000000296242872
floquet_residual      = 3.7762989580180845e-07
```

The first branch-switch corrector was attempted at `Delta s = 0.01`, then
retried at `0.005` and `0.0025`. The dedicated runner initially had a lower
hard-coded GMRES budget; it was extended to expose the production values
`gmres_maxiter = 240` and `gmres_restart = 80`. The Schur/PARDISO integrated
runner was also tested.

All correctors failed before the first Newton update with:

```text
failure_reason = augmented GMRES failed
gmres_info     = 240
```

At `Delta s = 0.0025`, the residual and nonzero-sector diagnostics were:

```text
residual_norm          = 8.94103086493591e-05
off_comb_norm_fraction = 2.493765586034913e-03
omitted_q_residual_rel = 1.2258318167386909e-06
```

The residual decreased with step halving, but no converged torus point was
obtained. The five-to-ten-point PALC branch was therefore not started. Newton
step components, finite displacement, and NS-mode overlap of a corrected torus
state are unavailable because GMRES failed before an update was accepted.

Artifacts:

- `D:\tmp\ns_branch_switch_20260818\checkpoint_replay\`
- `D:\tmp\ns_branch_switch_20260818\checkpoint_replay_b\`
- `D:\tmp\ns_branch_switch_20260818\modes_replay\`
- `D:\tmp\ns_branch_switch_20260818\floquet_seed.npz`
- `D:\tmp\ns_branch_switch_20260818\branch_switch_k5.json`
- `D:\tmp\ns_branch_switch_20260818\branch_switch_k5_ds005.json`
- `D:\tmp\ns_branch_switch_20260818\branch_switch_k5_ds0025.json`
- `D:\tmp\ns_branch_switch_20260818\branch_switch_k5_ds0025_gmres240.json`
- `D:\tmp\ns_branch_switch_20260818\integrated_k5_schur.json`

Status before the instrumentation below: Floquet confirmation and pump
reproducibility passed; the first nonlinear branch-switch gate failed at
augmented GMRES, and no physical column was run after that failure.

### Augmented linear-solver instrumentation and fix (2026-08-18)

The first diagnostic retry at K=5 and `Delta s = 0.01` exposed a state-coordinate
scaling defect before the bordered preconditioner could be assessed. The PALC
tangent and predictor use the packed state divided by `state_scale`, while the
augmented JVP had treated the GMRES state component as an unscaled physical
perturbation. The initial finite-difference errors were:

```text
random         = 2.615651587185491e-02
critical_mode  = 4.169778113103043e+01
phase_mode     = 4.171275618778140e+01
pure_omega     = 2.302633304881025e-06
pure_tau       = 3.576711204203705e-04
```

The augmented operator and Newton update now consistently map normalized GMRES
state coordinates through `state_scale`. The bordered preconditioner uses the
existing state factorization and a 2-by-2 Schur complement for the phase and
arclength rows. The source-column derivative also includes the source-dependent
residual normalization.

The focused regression suite passed after both changes:

```text
37 passed in 5.83s
```

The final single-point physical retry used the full-node K=5 basis, PARDISO,
`current_complex_c`, the saved Floquet seed, and `Delta s = 0.01`. It converged
before the continuation column was started:

```text
converged              = True
Newton iterations      = 3
residual_norm          = 3.43514407700318e-11
omega_a/omega_p        = 0.09228946690930089
source_tau             = 1.0000106403207871
off_comb_norm_fraction = 0.010110839525906136
border_schur_condition = 236.44636985008572
```

The complete augmented-JVP finite-difference audit at the first linear solve
was:

```text
random         = 1.33527822547307e-08
critical_mode  = 1.34664392148553e-06
phase_mode     = 1.34674044663072e-06
pure_omega     = 2.30263330488103e-06
pure_tau       = 1.00011734957538e-06
```

The true GMRES callback history was:

```text
5.833336891236922e-03, 4.5986799289438555e-03,
2.02926838691913e-12, 3.17309298529586e-13,
1.16914592717649e-13, 1.18523071995073e-13,
6.70943606429084e-14, 2.38852307030383e-23
```

The K=5 branch-switch linear gate therefore passes. No PALC ladder, K=10
solve, Q=2 solve, or production column was run in this step. The final JSON
artifact is
`D:\tmp\ns_branch_switch_20260818\branch_switch_k5_ds01_final_debug.json`.

### K=5 PALC smoke and production-basis gate (2026-08-18)

The successful branch-switch milestone was committed on `dev` before PALC
work:

```text
7f05c79 feat(chaos): validate augmented torus branch switch
```

The PALC runner was extended to retain the finite torus as its predictor after
the first branch switch. It now writes the signed effective source-drive
distance from the NS point, the explicit radius

```text
r2 = (||X[q=+1]||^2 + ||X[q=-1]||^2) / ||X[q=0]||^2
```

and per-point state/mode overlaps, tangent angle, phase and arclength
residuals, omitted-sector residuals, border-condition history, and total plus
per-Newton GMRES counts. State/tangent checkpoints can be promoted between
sideband bases by matching common `(h, q)` tones.

The five-point K=5 full-node PALC smoke used the same converged pump
checkpoint at `-24.1955501 dBm` for every row. The pump checkpoint was not
re-solved between rows because the continuation variable was the torus source
scale; the torus state was warm-started from the preceding PALC point.

Full per-point output:

```text
point  effective_P_dbm       dP_db       omega_a/omega_p       r2
0      -24.195528314711932   2.1785e-05  0.09228990266408323   2.6082281831283387e-05
1      -24.195459645208340   9.0455e-05  0.09228947903131639   1.0215116435701930e-04
2      -24.195345857804270   2.0424e-04  0.09228877719293296   2.2821137537875805e-04
3      -24.195186976588715   3.6312e-04  0.09228779749858572   4.0425342001614043e-04
4      -24.194983033760817   5.6707e-04  0.09228654043403287   6.3026480927754770e-04
```

```text
point  residual_norm          Newton  GMRES_total  critical_overlap  prev_overlap       tangent_angle  omitted_rel          phase_residual       arclength_residual
0      9.5316650818698e-13    3       17            0.999999999887359  --                 --              4.2217622441207e-06  -8.8054035365820e-20  8.6736173798840e-19
1      9.2508960297899e-13    2       12            0.999999998429502  0.999987501191223  5.7686266971e-03  1.6534641245412e-05  -2.6677542087632e-19  2.6020852105116e-18
2      9.3493486668690e-13    2       12            0.999999992307928  0.999987503417208  5.6310791094e-03  3.6939768226584e-05  -3.2110855726863e-19  0.0000000000000e+00
3      9.2612991843280e-13    2       12            0.999999976033434  0.999987506533339  5.6366243712e-03  6.5436215146344e-05  -1.1981119221705e-18  -8.6736173798840e-19
4      9.3147787348936e-13    2       12            0.999999941959792  0.999987510557541  5.6444327838e-03  1.0202281564071e-04   4.6825675041145e-19  0.0000000000000e+00
```

All five rows converged through the PALC route after the first row. The local
branch runs toward higher drive. A least-squares fit of `r2` against signed
`dP` gave:

```text
slope      = 1.1080216077421166 r2/dB
intercept  = 1.924723836095472e-06
R_squared  = 0.9999999939862152
```

The K=5 smoke gate passed. The artifact is
`D:\tmp\ns_branch_switch_20260818\k5_palc_smoke_5points.json`.

For production promotion, a full-node K=5 state/tangent checkpoint was
written to:

```text
D:\tmp\ns_branch_switch_20260818\k5_full_state\point_000.npz
```

A direct K=5 Schur checkpoint attempt timed out after `304` seconds without a
point artifact. The full-node checkpoint was therefore restricted to the
Schur retained coordinates before promotion; no full-node state was passed to
the Schur solver.

The one-point K=10, Q=1, Schur/PARDISO production gate was then launched with
the promoted state and `Delta s = 0.005`. The exact process limit was `300 s`
(`304.0 s` including command overhead). It timed out without writing either a
point JSON or a converged state:

```text
command = run_torus_branch.py --device ipm_2c_fixed --q-max 1
          --sideband-harmonics 10 --schur --factor-backend pardiso
          --initial-state-npz k5_full_state\point_000.npz
result  = process exit 124; no k10_schur_one_point.point_000.json
```

This is classified as a K=10 production scaling/backend blocker, not as
evidence that the torus branch is absent. The 10-point physical column was
not launched because the production-basis gate did not pass.

The final focused verification after the PALC/checkpoint changes was:

```text
40 passed in 4.82s
```

`graphify update .` completed and rebuilt the code graph. Existing permission
warnings for unreadable disposable output directories remain unchanged.
