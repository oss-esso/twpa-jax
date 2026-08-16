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

### Figures

`scripts/chaos/plot_nonlinear_diagnostics.py` -> `outputs/chaos/figures/`.
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
