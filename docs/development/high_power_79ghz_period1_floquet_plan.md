# 7.9 GHz high-power PERIOD1 and Floquet plan (2026-08-11)

Status: `EVIDENCE_REVISED_PLAN_PROPOSED`

## Goal

Establish, with controls that survive review, where the 7.9 GHz `ipm_2c_fixed`
PERIOD1 branch actually ends, and whether any Floquet instability is involved,
without adding a new harmonic-balance ansatz that the evidence does not
require.

---

## Executive summary of measured results

Measurements taken on 2026-08-11 change the working picture.

1. **The `-23.421053` dBm harmonic-balance failure is a power-step artifact, not
   a physical boundary.** `scripts/recover_period1_branch.py` reached every
   target from `-24.25` to `-23.421053` dBm by plain Newton
   (`method="natural"`); the pseudo-arclength fallback was never invoked.
2. **The transient-domain campaign's `UNRESOLVED_LONG_TRANSIENT` labels in that
   region are protocol artifacts.** The campaign's own control runs show the
   recurrence metric scaling with timestep and ramp length, and returning to the
   numerical floor when the trajectory is started on the harmonic-balance orbit.
3. **`|lambda|` is not a decidable stability discriminant for this circuit as
   built.** `designs/ipm_2c_fixed` resolves to `has_loss = False`: four 50-ohm
   port resistors are the only dissipation in a 6136-node, 16312-element
   network. The measured least-damped Floquet multiplier is within `6e-8` of the
   unit circle at a provably stable operating point.

4. **The Themis `14.18.08` cube measures the collapse boundary directly**, at 51
   pump frequencies spanning `5.980` to `7.997` GHz, and its `Response` array is
   a pump-on / pump-off ratio, hence free of the port-power convention and the
   line-loss model. Fitting `1/sqrt(G)` against pump amplitude on the measured
   trajectory predicts the observed collapse to within `0.3` dB at three of four
   frequencies near 7.9 GHz, so **the collapse is the parametric threshold**,
   established with no model input.
5. **The model and the measurement cannot yet be compared.** Only two model fold
   points lie inside the measured band against two unknown calibration offsets,
   giving zero degrees of freedom; and the two sides currently report different
   observables, with a measured single-tone versus peak bias reaching `11.90` dB
   that grows with pump power and inverts the apparent trend.

The 7.9 GHz boundary in **model** coordinates is approximately `1.16e-05 A`,
subject to a `0.61` dB spread between two existing fold runs. The point under
investigation in the current session sits `3.98 dB` below it. Whether that model
boundary coincides with the measured one is the open question Phase 2b answers.

---

## Phase dependency graph

```text
Phase 0  (caveat verification, blocking)
   |
   +--> Phase 1  (instrument repair)          -- independent of 2, 2b
   |
   +--> Phase 2  (locate the model boundary)
             |
             +--> Phase 2b  (external validation)   [needs Phase 1 item 1]
                       |
                       +--> Phase 3  (add dissipation)   [needs Phase 0.8]
                                 |
                                 +--> Phase 4  (instrument selection)
                                           |
                                           +--> Phase 5  (gated, may never run)
```

Phase 2b must precede Phase 3: dissipation moves the fold, so the lossless
baseline has to be fitted against the measurement first.

### Wall-time estimates

Order-of-magnitude only; measure before launching anything detached.

| item | estimate | basis |
| --- | ---: | --- |
| Phase 0 total | 2 to 4 h | 0.3 and 0.5 dominate; the dense Hill scan is ~185 s per power point per sideband setting |
| Phase 2 item 1 | 1 to 3 h | ~4.4 dB at `0.1` dB steps, `1` to `3` s per natural solve, PALC points far slower |
| Phase 2b item 3 | 4 to 12 h, run detached | 51 fold locations; scale from `outputs/phase4_fold_follow_reduced` per-point time first |
| Phase 2b item 5 | ~70 s per power point | `gain_total_runtime_s ~ 0.36` s per signal tone, 200 tones |
| Phase 3 item 1 | 2 to 4 h | four `tan_delta` values including the control, one dense Hill scan each |

---

## Current state analysis

### A. PERIOD1 recovery at the disputed power

Source: `.hybrid_outputs/period1_recovery_7p9_2c_v1/period1_recovery.json` and
the five written checkpoints.

| target [dBm] | I [A] | method | converged | PALC used |
| ---: | ---: | --- | --- | --- |
| -24.250000 | 6.679955e-06 | natural | true | no |
| -24.000000 | 6.875013e-06 | natural | true | no |
| -23.800000 | 7.035153e-06 | natural | true | no |
| -23.600000 | 7.199023e-06 | natural | true | no |
| -23.421053 | 7.348876e-06 | natural | true | no |

Basis `[1,3,5,...,19]`, `--pump-port 4`, `--freq-ghz 7.9`, circuit
`designs/ipm_2c_fixed` — identical to the failing column.

The original column
(`.hybrid_outputs/hb_up_7p9_m35_to_m21/hb_up_to_failure.csv`) stepped
`1.0526 dB` per point and failed at row 11 with
`pump_failure_reason = "stalled at Newton 6 (reduction ratio 0.991)"`,
`pump_coeff_rel = 0.0557`. The recovery used steps of `0.179` to `0.250` dB.

### B. The converged branch is benign at that power

`solution_summary` from the recovered checkpoints, compared with the last
converged column point:

| P [dBm] | I [A] | `branch_current_max_over_ic` | `branch_min_cos_phase` |
| ---: | ---: | ---: | ---: |
| -24.473684 (column) | 6.510125e-06 | 0.554490 | 0.832191 |
| -24.250000 | 6.679955e-06 | 0.558884 | 0.829246 |
| -24.000000 | 6.875013e-06 | 0.561562 | 0.827435 |
| -23.800000 | 7.035153e-06 | 0.577651 | 0.816284 |
| -23.600000 | 7.199023e-06 | 0.585522 | 0.810656 |
| -23.421053 | 7.348876e-06 | 0.602304 | 0.798267 |

Smooth and monotone. No stiffness cliff, no turning point.

The values `0.862025` and `0.506865` recorded in the failure row of
`hb_up_to_failure.csv` are diagnostics of the **last non-converged Newton
iterate**, not of a solution. They must not be quoted as physical state.

### C. Reconciliation with the established boundary

| source | quantity | value |
| --- | --- | ---: |
| PALC fold detection (`arclength_fold_resolution_plan.md`, memory `arclength-metric-bug-and-snaking-verdict`) | `I_bound` at 7.9 GHz | 1.1628e-05 A |
| TD ramp selection (`docs/development/h3_physical_boundary_79.md`) | `I_FIRST_NONPERIOD1` | <= 1.1600e-05 A |
| TD ramp selection, same document | `I_PHYSICAL_WORKING_MAX` | >= 1.1400e-05 A |
| this session's disputed point | I | 7.3489e-06 A |

Two independent methods put the boundary within 0.3 percent of each other. In
power, using the column's own calibration (`6.510125e-06 A = -24.473684 dBm`),
the boundary is approximately `-19.69` to `-19.44` dBm. The disputed point is
`3.98 dB` below it and carries `63 percent` of the boundary current.

### D. Transient-domain campaign: floor and controls

Source: `.hybrid_outputs/overnight_7p9_dynamics_v1/campaign_summary.json`.

Coarse sweep, `delta_theta = 0.05`, ramp 40 periods, hold 440 periods,
`method = implicit_trapezoid`:

| P [dBm] | regime | `tau_periods` | d1_late | d2_late | d3_late | winding |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| -35.0 | PERIOD1 | 410.3 | 4.270e-4 | 4.192e-4 | 4.143e-4 | 2.33e-08 |
| -33.0 | PERIOD1 | 406.5 | 4.257e-4 | 4.190e-4 | 4.142e-4 | 3.92e-08 |
| -31.0 | PERIOD1 | 403.9 | 4.256e-4 | 4.177e-4 | 4.136e-4 | 4.80e-08 |
| -29.0 | PERIOD1 | 406.9 | 4.254e-4 | 4.171e-4 | 4.129e-4 | 5.59e-08 |
| -27.0 | PERIOD1 | 409.6 | 4.253e-4 | 4.173e-4 | 4.132e-4 | 7.54e-08 |
| -26.0 | PERIOD1 | 405.3 | 4.257e-4 | 4.195e-4 | 4.137e-4 | 5.67e-08 |
| -25.0 | PERIOD1 | 396.3 | 4.251e-4 | 4.173e-4 | 4.132e-4 | 1.72e-07 |
| -24.473684 | PERIOD1 | 403.4 | 4.261e-4 | 4.213e-4 | 4.202e-4 | -5.81e-08 |
| -24.0 | PERIOD1 | 409.2 | 4.266e-4 | 4.230e-4 | 4.208e-4 | 4.97e-08 |
| -23.421053 | UNRESOLVED | - | 2.259e-2 | 4.347e-2 | 5.930e-2 | -6.43e-06 |
| -23.0 | UNRESOLVED | - | 2.922e-1 | 5.563e-1 | 7.610e-1 | 1.15e-04 |
| -22.5 | UNRESOLVED | 37.7 | 6.824e-1 | 9.559e-1 | 1.201e+0 | -9.74e-04 |
| -22.0 | UNRESOLVED | 38.1 | 5.461e-1 | 8.181e-1 | 1.044e+0 | -1.41e-06 |
| -21.0 | UNRESOLVED | 2842.7 | 8.151e-2 | 1.157e-1 | 1.305e-1 | -3.12e-05 |
| -20.0 | UNRESOLVED | 191.3 | 1.246e-1 | 1.073e-1 | 1.597e-1 | -1.74e-02 |
| -19.0 | UNRESOLVED | 100.4 | 1.705e-1 | 3.253e-1 | 4.509e-1 | -4.78e-05 |
| -18.0 | UNRESOLVED | - | 2.046e-1 | 3.483e-1 | 4.149e-1 | 4.65e-04 |
| -17.0 | UNRESOLVED | 269.6 | 1.247e-2 | 1.416e-2 | 2.124e-2 | -3.33e-02 |
| -16.0 | UNRESOLVED | 181.4 | 2.249e-2 | 3.571e-2 | 5.087e-2 | -9.62e-03 |
| -15.0 | UNRESOLVED | - | 1.973e-2 | 3.480e-2 | 5.104e-2 | -8.12e-02 |

Observations:

- `d1_late = 4.25e-4 +/- 1e-6` is constant across 11 dB of pump power. A
  physical recurrence residual would scale with amplitude. This is a numerical
  floor.
- `tau_periods` is `396` to `410` at every sub-boundary power. Fitting a decay
  slope on a flat floor produces a constant, meaningless timescale. The
  inference "1000+ periods to settle implies a Floquet multiplier near neutral"
  is not supported by this data.
- Above the transition the values are non-monotone and by `-17`/`-15` dBm they
  fall back to `1.2e-2` to `2.0e-2`. This is not the ordering of a
  period-doubling cascade.
- `mean_phase_winding_cycles` rises from `1e-8` below the transition to
  `-8.12e-2` at `-15` dBm. This is the one signal in the upper region that is
  clearly physical: junction phase drift.

Campaign control runs, same summary file:

| P [dBm] | `delta_theta` | ramp | hold | init | regime | d1_late |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| -23.8 | 0.05 | 20 | 250 | zero-pump | UNRESOLVED | 2.413e-3 |
| -23.8 | 0.05 | 80 | 250 | zero-pump | UNRESOLVED | 1.025e-3 |
| -23.8 | 0.025 | 40 | 250 | zero-pump | UNRESOLVED | 1.179e-3 |
| -23.8 | 0.05 | 0 | 440 | same-target TD restart | **PERIOD1** | 4.678e-4 |
| -23.8 | 0.05 | 40 | 250 | **HB orbit** | UNRESOLVED | **4.436e-4** |
| -23.6 | 0.025 | 40 | 250 | zero-pump | UNRESOLVED | 1.184e-3 |
| -23.6 | 0.05 | 40 | 250 | **HB orbit** | UNRESOLVED | **5.834e-4** |
| -23.421053 | 0.025 | 40 | 250 | zero-pump | UNRESOLVED | 1.367e-3 |
| -23.421053 | 0.05 | 40 | 250 | **HB orbit** | UNRESOLVED | **6.313e-4** |

Readings:

- The matched-protocol rerun does **not** reproduce that reading: at `-23.8`
  dBm, `delta_theta = 0.05` gives `d1_late = 1.075e-3` and
  `delta_theta = 0.025` gives `1.179e-3` at the same ramp and hold. The
  timestep-decrease criterion therefore fails; the earlier mixed-protocol
  comparison is withdrawn.
- Quadrupling the ramp length takes `d1` from `2.413e-3` to `1.025e-3`. The ramp
  injects the transient.
- Starting on the harmonic-balance orbit gives `4.436e-4` to `6.313e-4` at
  exactly the powers labelled unresolved. That is the same order as the
  `4.25e-4` value the classifier calls PERIOD1 at `-35` dBm. **The transient
  solver preserves the PERIOD1 orbit at the disputed powers.**
- The classifier keys on decay slope, not level, so it cannot distinguish a flat
  numerical floor from an unresolved transient. The audit also found a run
  classified `UNRESOLVED_SLOW_RELAXATION` while `max_abs_phi` grew monotonically
  by a factor of `2.7`. `d1` is therefore retired as a stability discriminant;
  the `max_abs_phi` envelope slope is the replacement diagnostic.

The integrator used by the campaign is hardcoded at
`scripts/run_overnight_7p9_dynamics.py:105`
(`"--method", "implicit_trapezoid"`). The implicit trapezoid rule is A-stable
but not L-stable: its amplification factor tends to `-1` as the eigenvalue
tends to negative infinity, so it applies no numerical damping at any
frequency. In a network whose only dissipation is four port resistors, content
excited by the ramp does not decay. `scripts/h1_transient_branch_transfer.py:1557`
already offers `BDF` and `Radau`, both L-stable.

### E. Time-domain monodromy Floquet path fails on this circuit

Source: `.hybrid_outputs/floquet_7p9_2c_v1_smoke2/floquet_results.json` and
`.../floquet_7p9_2c_v1_closure_smoke/floquet_results.json`.

```text
state_dimension     12271
steps_per_period    64
eigenvalues (k)     2
which               LM
arnoldi.message     "ARPACK error -1: No convergence (81 iterations, 0/2 eigenvectors converged)"
arnoldi.matvecs     651
arnoldi.runtime_s   121.04
closure.relative_error  3.921954e-03
spectral_radius     NaN
```

Two independent blockers:

- **Clustered spectrum.** With `has_loss = False`, most of the 12271 multipliers
  lie within `1e-8` of the unit circle. Requesting the two largest-magnitude
  eigenvalues asks Arnoldi to separate a two-dimensional invariant subspace out
  of a cluster with relative separation of order `1e-8`. This is a property of
  the operator, not a defect of `src/twpa_solver/stability/floquet.py`.
- **Discretization floor.** One-period closure error is `3.92e-3` at 64 steps
  per period, while the quantity of interest, `1 - |lambda|`, is of order
  `1e-8`. The trapezoid rule is second order, so reaching a `1e-9` closure
  requires of order `1e5` steps per period. Not reachable at this state
  dimension.

### F. Hill route works, but the discriminant is degenerate

Dense half-zone scan run on 2026-08-11 (700 points, `sidebands = 4`,
`gamma_nt = 1024`, top 20 `sigma_min` minima refined into the complex plane via
`twpa_solver.signal.stability.refine_complex_resonance`). Artifact:
`max_multiplier_scan_s4_n700.json` (scratchpad).

| P [dBm] | max `|lambda|` over refined port-coupled roots | `1 - |lambda|` | f [GHz] | label |
| ---: | ---: | ---: | ---: | --- |
| -35.000000 | 0.99999994 | 6e-8 | 3.90277 | PERIOD_DOUBLING_CANDIDATE |
| -24.473684 | 0.99999772 | 2.3e-6 | 2.69643 | NEIMARK_SACKER_CANDIDATE |
| -23.421053 | 1.00000000 | 2e-11 | 3.47645 | NEIMARK_SACKER_CANDIDATE |

Exactly-real roots (`Im(omega) = 0`, `|lambda| = 1` identically) were also found
at `2.97703` and `2.79742` GHz (`-24.473684` dBm) and `2.32048` and `2.33744`
GHz (`-23.421053` dBm). These are internal modes with no port coupling.

The `-35` dBm point, which is unquestionably stable, is closer to marginal by
this measure than `-24.47` dBm, and is labelled `PERIOD_DOUBLING_CANDIDATE`.
The classifier is not wrong; the question is not decidable at this loss level.

A 16-point power sweep of one seeded root
(`ns_root_vs_power_s4.json`, scratchpad) shows the port-coupled branch flat at
`|lambda| = 0.9550` to `0.9573` from `-35` to `-23.6` dBm with no approach to
the unit circle, while the secant intermittently converges onto a
power-independent neutral root at `3.5912` GHz. That is a branch-tracking
failure, not physics.

### G. Mode comb and scan aliasing

Measured on the unpumped linear circuit
(`solve_linear_scattering` with `extra_K = Bphi diag(gamma_off) Bphi^T` at zero
flux, 7.85 to 7.95 GHz, 201 points):

| path | `|S|` range [dB] | mean group delay | in pump periods |
| --- | --- | ---: | ---: |
| port 4 -> 3 | -0.05 to -0.67 | 0.121 ns | 1.0 |
| port 4 -> 2 | -8.46 to -23.15 | 5.885 ns | 46.5 |

One-way transit is about 46.5 pump periods, so the resonant mode comb spacing is
approximately `1 / (2 * 5.885 ns) = 85 MHz`. The default Tier-1 zone scan in
`scripts/floquet_stability_sweep.py` at `--n-points 200` over `0.05` to
`7.82` GHz has a `39 MHz` step; the 80-point scan used in the first
investigation had a `97 MHz` step and therefore aliased the comb.

### H. Defects found

| location | defect |
| --- | --- |
| `scripts/floquet_stability_sweep.py:262` | prints `U+2220` (angle sign); raises `UnicodeEncodeError` under the Windows `cp1252` console **before** the output JSON is written, so `--refine-complex` silently produces no artifact. Workaround `PYTHONIOENCODING=utf-8`. |
| `scripts/run_overnight_7p9_dynamics.py:105` | integrator hardcoded to `implicit_trapezoid`; not L-stable, and the campaign has no run with `BDF` or `Radau` for comparison. |
| decay-aware classifier (`h1_transient_branch_transfer.py`, consumed by the campaign) | classifies on trend slope, not level relative to a same-protocol reference; labels `d1 = 4.436e-4` as `UNRESOLVED` at `-23.8` dBm and `d1 = 4.270e-4` as `PERIOD1` at `-35` dBm. |
| Hill sweep CLI | no branch tracking in power; `src/twpa_solver/stability/tracking.py` exists but is wired only to the monodromy scan. |

### I. External measurement of the collapse boundary (Themis 14.18.08)

`docs/development/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK` contains
51 pump frequencies from `5.980` to `7.997` GHz at approximately `40` MHz
spacing, and therefore brackets the 7.9 GHz column. The companion dataset
`17.03.10_...` spans only `7.043` to `7.373` GHz and is not usable here.

Each `.npy` is a pickled dict:

```text
Frequency    (2001,)      4.0 to 12.0 GHz
Response     (31, 2001)   dB, calibrated on the unpumped device
PumpPower    (31,)        -29.0800 to -19.0267 dBm, 0.3351 dB step
SignalPower  scalar       -30 dBm
```

`logfile.txt` records `Calibration: cali on unpumped device`, so `Response` is a
pump-on / pump-off ratio. It is the **same observable** the solver reports as
`gain_vs_off_db`, and being a ratio it is independent of the source power
convention.

**The device collapses, abruptly and totally.** Across the whole 4 to 12 GHz
span the median response falls from approximately `0` dB to approximately
`-30` dB within one `0.3351` dB pump step. This is not gain rolling off; the
line stops transmitting. Example at `7.916` GHz:

| P [dBm] | peak gain [dB] | median [dB] |
| ---: | ---: | ---: |
| -23.0480 | 18.198 | -0.028 |
| -22.7129 | 20.694 | -0.052 |
| -22.3778 | 2.258 | -11.241 |
| -22.0427 | -3.488 | -15.810 |
| -21.7076 | -9.733 | -21.391 |

The collapse boundary is a **sawtooth comb in pump frequency**: median period
approximately `265` MHz (measured resets at `6.141`, `6.383`, `6.666`, `6.908`,
`7.150`, `7.432`, `7.714`, `7.997` GHz; the `40` MHz sampling quantizes the
period estimate), depth `5.36` dB, envelope from `-24.388` to `-19.027` dBm.
The upper value is censored by the instrument's maximum pump power, so at
`6.908`, `7.150`, `7.714`, `7.755` and `7.997` GHz the device did not collapse
within range.

Peak gain reached immediately before collapse ranges from `8.4` to `33.2` dB,
with a median near `20` dB. At `7.835` GHz the device sustains `33.15` dB.

#### Comparison with the model fold curve

An exhaustive search of `outputs/` and `.hybrid_outputs/` found exactly **three**
`fold_curve.csv` files. There is no 19-point sweep on disk.

| file | points | content |
| --- | ---: | --- |
| `outputs/phase4_fold_follow_reduced/fold_curve.csv` | 4 | 7.6: `-21.496451` (`lambda=0.5311`); 7.9: `-19.435096` (`lambda=0.6734`); 8.2: no fold; 8.5: `-19.825254` (`lambda=0.6438`) |
| `outputs/campaign_diss/2c_single_column_7p6_fold_trace/fold_curve.csv` | 1 | 7.6: `-22.109839` (`lambda=0.4949`) |
| `outputs/continuation_diagnostics/f00_fold_follow/fold_curve.csv` | 3 | 7.786, 7.969, 8.153: **all empty** (the retracted pre-metric-fix "zero folds everywhere" run) |

Comparison against the measured collapse bracket:

Comparison **at face value**, that is with the pump-frequency offset `df` and
the pump-power offset `dP` both set to zero. The next two subsections show that
this assumption is not admissible and that the "verdict" column therefore
carries no weight on its own. It is tabulated only to be dismantled.

| f_p [GHz] | model `fold_power_dbm` | source | measured bracket [dBm] | verdict at `df=dP=0` |
| ---: | ---: | --- | --- | --- |
| 7.6 | -21.496451 | phase4 | [-21.7076, -21.3724] at 7.593 GHz | inside the bracket |
| 7.6 | -22.109839 | campaign_diss | [-21.7076, -21.3724] at 7.593 GHz | 0.40 dB below the bracket |
| 7.9 | -19.435096 | phase4 | [-21.3724, -21.0373] at 7.876; [-22.7129, -22.3778] at 7.916 | model high by 1.6 to 3.3 dB |
| 8.2 | no fold found | phase4 | measurement ends at 7.997 GHz | no comparison |
| 8.5 | -19.825254 | phase4 | out of measured range | no comparison |

The two model runs also disagree with **each other** at `7.6` GHz by `0.61` dB
(`fold_lambda` `0.5311` versus `0.4949`, different continuation settings), which
already exceeds the measurement's `0.335` dB bracket width.

#### The table above assumes `df = 0` and `dP = 0`, which is not admissible

The measurement is a real device. Its pump-frequency axis and its pump-power
axis both carry offsets against an ideal model: fabrication tolerance on `Lj`
and `Cg` shifts the comb in frequency, and the pump line calibration shifts the
power. Neither is known here. Comparing model `7.6` GHz against measured
`7.593` GHz at face value silently sets both offsets to zero.

The offsets are small in their own units, but the comb converts a small `df`
into a large `dP`. Measured local slope of the collapse envelope:

| near | slope | per 10 MHz | per 40 MHz (the measurement's own grid step) |
| ---: | ---: | ---: | ---: |
| 7.6 GHz | -19.1 dB/GHz | 0.19 dB | **0.77 dB** |
| 7.9 GHz | -20.8 dB/GHz | 0.21 dB | **0.83 dB** |

The measurement's own frequency sampling therefore already carries about
`+/- 0.4` dB of irreducible power ambiguity, and a `df` the size of a normal
fabrication tolerance dominates every number in the comparison table.

#### Identifiability: the comparison currently has zero degrees of freedom

Of the four model fold values, `8.5` GHz lies outside the measured band
(`5.980` to `7.997` GHz) and `8.2` GHz has no fold. **Two** model points land
inside the band. There are **two** free calibration parameters. Degrees of
freedom: `0`.

A two-parameter fit to two points cannot validate anything, whatever residual it
reports. Measured, over physically plausible small-offset windows, using the
midpoint of each measured bracket and excluding the five censored frequencies
(46 usable measured points):

| window | best rms | at | `dP` range within 0.25 dB of best |
| --- | ---: | --- | --- |
| `df` within 50 MHz, `dP` within 1.5 dB | 1.006 dB | `df = -0.024` GHz, `dP = -0.76` dB | `-1.50` to `+0.40` dB (span **1.90 dB**) |
| `df` within 100 MHz, `dP` within 3.0 dB | 0.975 dB | `df = -0.100` GHz, `dP = +0.76` dB | `-3.00` to `+1.48` dB (span **4.48 dB**) |

Two conclusions, both correcting earlier drafts of this document.

1. **The `0.12` dB agreement at `7.6` GHz is a single-point coincidence.** Once
   the `7.9` GHz point is included, no small `(df, dP)` reconciles both to
   better than about `1.0` dB rms, which is three times the measurement's own
   `0.335` dB bracket. *Withdrawn*: "the model fold lands inside the measured
   bracket" as a statement about the model rather than about one arbitrary
   choice of coordinates.
2. **`dP` is not determined.** It is uncertain by `1.9` to `4.5` dB depending on
   the window, and the window edge rather than the data is doing the
   constraining. *Withdrawn*: the inference that `dP` is near zero rather than
   the `+2.5` to `+3.3` dB found by `scripts/align_map_to_measurement.py`. Both
   values remain admissible.

The `0.61` dB spread between the two model runs at `7.6` GHz is still real and
still blocking, but it is now the second-largest problem rather than the first.

What survives, because it does not depend on either offset:

- Both sides have frequencies with **no boundary in range**: model at `8.2` GHz,
  measurement at `6.908`, `7.150`, `7.714`, `7.755`, `7.997` GHz.
- Both known model folds and the whole measured envelope lie in the same
  `-19` to `-24` dBm window, so the model is not wrong by tens of dB.
- The measured boundary is a comb with a `265` MHz period and a `5.36` dB depth.
  Comparing the **shape** of the two curves is invariant under `dP` and only
  translates under `df`, so period and depth are testable today in principle and
  will be testable in practice as soon as the model curve has enough points.

That is the entire content of the comparison at present. It is a reason to run
Phase 2b, not a result.

*Also withdrawn*: an earlier draft cited a model fold range of `-18.6` to
`-23.0` dBm "over a 19-point sweep". That figure came from prior session notes
and **no file on disk supports it**. Do not cite it.

#### Observable mismatch: the two sides do not measure the same quantity

The production gain-map column reports `gain_vs_off_db` at a **single** signal
tone, `f_s = f_p - 500` MHz (the 7.9 GHz column used `f_s = 7.4` GHz). The
measurement reports the **maximum over the whole 4 to 12 GHz signal span**.
These are different observables and the gap between them is not a constant.

Measured directly from the cube, taking the same `|f_s - f_p| > 0.15` GHz
exclusion and comparing `G` at `f_p - 500` MHz against `max_f G`:

| f_p [GHz] | bias `G_peak - G_at_fp-500MHz` | change across the sweep |
| ---: | --- | ---: |
| 7.876 | 1.79 to **11.90** dB | +9.34 dB |
| 7.916 | 1.15 to **8.97** dB | +2.15 dB, non-monotone |

The bias grows with pump power because the gain lobes move in frequency as the
pump-induced phase accumulates. The peak wanders across lobes (at 7.876 GHz:
`4.824` to `6.800` to `6.408` to `8.256` to `8.212` GHz), while the probe tone
stays fixed.

Worse, at `7.876` GHz the single-tone trace is **non-monotone where the peak is
still rising**:

| P [dBm] | `G` at 7.376 GHz | `G_peak` |
| ---: | ---: | ---: |
| -22.0427 | **16.31** | 21.81 |
| -21.7076 | **15.55** | 23.48 |
| -21.3724 | **13.62** | 25.52 |

A fixed-offset probe therefore reports the device rolling over while it is in
fact still gaining. **No shape statistic, slope, or normalization removes this**,
because the bias is power dependent, non-monotone, and reaches `12` dB. The
observable must be matched at the source: run the model with a signal-frequency
spectrum and reduce it with the same rule as the measurement.

#### The measured device follows the parametric-threshold form

For a regenerative parametric amplifier near threshold, `G ~ (1 - I/I_th)^-2`,
so `1/sqrt(G_linear)` is linear in pump amplitude and reaches zero at threshold.
Fitted on the measured peak gain over the amplifying, pre-collapse points
(`3 dB < G < 40 dB`):

| f_p [GHz] | n | R^2 | extrapolated `P_th` [dBm] | observed collapse [dBm] | error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7.795 | 11 | 0.7630 | -18.459 | -19.697 | 1.24 |
| 7.835 | 7 | 0.9853 | -20.718 | -20.702 | **0.02** |
| 7.876 | 11 | 0.9009 | -21.634 | -21.372 | 0.26 |
| 7.916 | 16 | 0.9666 | -22.435 | -22.713 | 0.28 |

At three of four frequencies the extrapolation predicts the observed collapse to
within `0.3` dB, which is inside the measurement's own `0.335` dB bracket. The
`7.795` GHz point sits immediately after a comb reset and its `R^2 = 0.76`
indicates two lobes mixing; it should be excluded rather than trusted.

Two consequences:

- The collapse **is** the parametric threshold, established from the gain
  trajectory alone with no reference to the model. This is independent evidence
  that the model's fold and the measured collapse are the same object.
- It supplies the right comparison coordinates. Fitting `1/sqrt(G)` versus pump
  amplitude on both sides reduces each gain trajectory to **two physically
  meaningful scalars, a threshold and a slope**, from an over-determined fit.
  The vertical calibration `dG` enters as a multiplicative scale on `1/sqrt(G)`
  and the horizontal `dP` as a shift in `I_th`, so both are reported rather than
  assumed.

#### What this changes

- The `-23.421053` dBm point this session investigated is below the measured
  collapse at `7.9` GHz (approximately `-21.6` dBm interpolated) as well as
  below the model fold. It is not near any boundary, physical or numerical.
- Section D's `mean_phase_winding_cycles` rising to `-8.12e-2` at high power now
  has an external counterpart: total loss of transmission is what junction phase
  running looks like on a VNA.
- A calibration-free validation channel exists that the project has not used:
  **peak gain at the collapse point**. The measurement gives `8.4` to `33.2` dB.
  The 7.9 GHz column ran with `--no-signal-spectrum`, so the model's peak over
  signal frequency was never computed and the comparison cannot be made yet.

Artifact: `themis_collapse_envelope.csv` (scratchpad), regenerated by Phase 2b.

### J. What this does and does not retract

Retracted:

- The framing that ordinary harmonic balance "fails" at `-23.421053` dBm.
  It does not; it requires a step below approximately `0.25` dB.
- The inference in the prior session's section 8 that `1000+` period relaxation
  implies a Floquet multiplier near `0.9992`. The same relaxation timescale is
  present `11 dB` lower where nothing is marginal.
- Any use of the `-23.421053` dBm failure row's `0.862` junction utilization or
  `0.507` minimum cosine as physical state.

Not retracted, and still standing:

- `docs/development/h3_physical_boundary_79.md`'s bracket at
  `1.140e-05 A <= I_boundary <= 1.160e-05 A`. That work used
  `delta_theta = 0.01` and a different protocol; it is not invalidated by the
  controls above, but it has not been re-run with them either.
- The PALC fold at `I_bound = 1.1628e-05 A`.
- Everything in `docs/development/floquet_implementation.md` about the
  formulation of the tangent map. The formulation is sound; the eigensolver
  strategy is what fails.

---

## What we are NOT doing

- No new harmonic-balance ansatz. No `2T` / `omega_p/2` basis, no period-`N`
  basis, no two-frequency torus basis is to be enabled by this plan. The
  scaffolding in `src/twpa_solver/pump/floquet.py`,
  `src/twpa_solver/pump/periodic_branch.py`,
  `src/twpa_solver/signal/period_doubled.py` and
  `scripts/run_period_doubled_branch.py` stays dormant.
- No further blind transient campaigns. No transient run is added without a
  paired same-protocol control.
- No further work on the time-domain monodromy eigensolver for
  `ipm_2c_fixed` until Phase 3 has restored a usable spectral gap.
- No change to production gain-map defaults, port conventions, loss models used
  for published numbers, or the compression pipeline.
- No re-derivation of the 7.9 GHz boundary location from scratch; existing
  numbers are treated as the reference to be confirmed, not replaced.
- No changes to `scripts/align_map_to_measurement.py` or to the gain-map
  calibration offsets it reports. Phase 2b fits the **boundary**, which is a
  separate observable; the two results are allowed to disagree and that
  disagreement is itself a finding.
- No re-processing of the `17.03.10` Themis cube. It spans `7.043` to
  `7.373` GHz only and cannot see the 7.9 GHz region.
- No attempt to correct the single-tone versus peak-gain observable mismatch
  downstream. The model emits the matched observable or the comparison is not
  made.

## Prerequisites

- [ ] `dev` branch, working tree clean at start of each phase.
- [ ] `python -c "import twpa_solver, sys; print(twpa_solver.__file__)"` prints a
      path under this repository (guards the editable-install shadowing issue).
- [ ] `PYTHONIOENCODING=utf-8` exported for any script that prints non-ASCII.
- [ ] Scratchpad artifacts from 2026-08-11 preserved for comparison:
      `hill_sweep_m23p42_vs_m35.json`, `ns_root_vs_power_s4.json`,
      `max_multiplier_scan_s4_n700.json`, `themis_collapse_envelope.csv`.
- [ ] Measurement present and readable:
      `docs/development/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK`
      (51 `.npy` files plus `logfile.txt`). Load with
      `np.load(path, allow_pickle=True).item()`; the payload is a dict, not an
      array.
- [ ] No external input outstanding. Check 0.8 was answered on 2026-08-11:
      IPM circuits have no shunt conductance, RCSJ is not implemented, and
      Phase 3 uses `tan_delta` in `{1e-5, 1e-4, 1e-3}`.

---

## Phase 0: Double-check the caveats

### Overview

Every claim above is either confirmed, qualified, or withdrawn before any new
work is started. Phase 0 is blocking: no later phase begins until its checklist
is complete and its outcomes are recorded in this document.

### Checks required

#### 0.1 Recovered checkpoints are genuine solutions

**Files**: `.hybrid_outputs/period1_recovery_7p9_2c_v1/point_*/pump`

Re-solve each recovered state with a fresh
`HarmonicNewtonKrylovSolver.solve_one(problem, X, 1.0)` from a cold start
(no warm chain) and record `coeff_rel` and `time_rel`. Confirm the basis, pump
port, pump frequency and DC branch flux match the source column exactly.

**Pass**: `coeff_rel <= 1e-9` at all five points and metadata identical to the
column except `pump_power_dbm_requested` and `pump_current_a`.
**Fail action**: treat the whole recovery result as unconfirmed and stop.

#### 0.2 Recovery is reproducible and step-size dependent, not seed-luck

Re-run `scripts/recover_period1_branch.py` from the `-24.473684` dBm column
checkpoint to `-23.421053` dBm in one step, then in two, then in five. Record
which step counts converge.

**Pass**: a monotone relationship — coarse steps fail, fine steps converge.
**Fail action**: if a single `1.05` dB step now converges, the original failure
was nondeterministic and the diagnosis must be revised.

#### 0.3 Hill-scan basis truncation

Repeat the dense scan of section F at `sidebands = 4, 6, 8, 10` at the three
powers. `sidebands = 10` matches the production basis. Also raise `gamma_nt`
from `1024` to `4096` at `sidebands = 4` to isolate the aliasing of
`compute_gamma_hat` from the sideband truncation.

**Pass**: the qualitative conclusion (`1 - |lambda| < 1e-6` at all three powers,
non-monotone in power) is unchanged.
**Phase 0 result**: PASS with a resource gap. `S = 4, 6, 8` were completed at
all three powers and were flat; `S = 10` was completed at `-35` dBm only. The
two higher-power `S = 10` jobs terminated without artifacts. The loss of
per-setting artifacts is a driver defect to fix before Phase 3 reuses it.
**Fail action**: if `max |lambda|` becomes monotone and shows an approach at
higher `S`, section F is withdrawn and the Hill route becomes the primary
instrument immediately.

#### 0.4 Root-refinement accuracy supports the digits quoted

`refine_complex_resonance` reports `residual` of order `1e-6` on an eigenvalue
whose scale is `sigma_min ~ 1e4`. Establish the achievable precision in
`Im(omega)` by perturbing the seed and the tolerance and measuring the spread of
the converged root.

**Pass**: reproducible `Im(omega)` to at least one digit better than the
smallest value quoted in section F.
**Fail action**: quote `1 - |lambda|` only as an upper bound and state the
resolution limit explicitly.

#### 0.5 Timestep and ramp controls at matched protocol

The comparison in section D mixes `hold = 250` with `hold = 440`. Re-run, at
`-23.8` dBm only, the four-point matrix
(`delta_theta` in `{0.05, 0.025}`) x (`ramp` in `{40, 80}`) at fixed
`hold = 250` and fixed initialization `zero_pump_equilibrium_q0_p0`.

**Pass**: `d1` decreases with both smaller `delta_theta` and longer ramp, at
matched hold.
**Fail action**: if `d1` is flat in `delta_theta` at matched protocol, the
"discretization error" reading is withdrawn and only the ramp-injection and
HB-init readings stand.

#### 0.6 Protocol floor as a function of power

Run the standard protocol (`delta_theta = 0.05`, ramp 40, hold 440,
zero-pump init) at `-35` dBm and at `-24.0` dBm with `delta_theta = 0.025`.
This establishes whether the `4.25e-4` floor is set by the timestep.

**Pass**: the floor moves with `delta_theta` and is power-independent.
**Phase 0 result**: FAIL. The matched-protocol `-23.8` dBm controls give
`d1_late = 1.075e-3` at `delta_theta = 0.05` and `1.179e-3` at `0.025`, so
halving the step does not reduce the metric. The existing controls already
show `4.25e-4 +/- 1e-6` over 11 dB, including `-35` and `-24.0` dBm (the latter
with the older `delta_theta = 0.05` protocol). The fail action is triggered:
`d1` is a diagnostic artifact and is retired as a stability discriminant. The
audit found `UNRESOLVED_SLOW_RELAXATION` concurrent with a monotonic `2.7x`
growth in the `max_abs_phi` envelope; use its envelope slope instead.

#### 0.7 Group-delay measurement is representative

Section G measured the unpumped circuit at zero flux. Repeat with `extra_K`
evaluated at the `-23.421053` dBm pump's mean junction tangent, and over a wider
span (7.0 to 8.5 GHz, 1000 points), to confirm the `85 MHz` comb spacing under
pump-shifted inductance.

**Pass**: comb spacing within a factor of two of `85 MHz`.
**Phase 0 result**: FAIL. The pump-shifted `7.0--8.5 GHz`, 1000-point sweep
resolved six group-delay peaks with median spacing `241.7 MHz` (range
`231.2--255.3 MHz`), outside the factor-of-two criterion. Recompute the
required Tier-1 scan density from this measured spacing before Phase 4.

#### 0.8 Loss channels available to `ipm_2c_fixed` -- ANSWERED, no check needed

Resolved by the design owner on 2026-08-11. Recorded here so Phase 3 does not
re-open it.

The codebase has three physically distinct loss channels and they are not
interchangeable:

| channel | stamped in | analytic in `omega`? | available to `ipm_2c_fixed` |
| --- | --- | --- | --- |
| dielectric `tan_delta` | `Im(C)`, as `C*(1 - 1j*tan_delta)` | **no**, resolves to `conductance_abs_omega` | **yes**, the one to use |
| real shunt conductance | `G` | yes | **no** -- IPM circuits have no shunt conductance. `shunt_conductance_s` exists only as a parameter of `build_effective_snail_line` (`src/twpa_solver/builders/le_gal_2025.py:47`) and is stamped as a uniform node-to-ground conductance at line 113. It is a Le Gal benchmark knob, not an IPM one. |
| RCSJ / quasiparticle junction resistance | -- | -- | **not implemented anywhere in `src/`**. Junctions are purely reactive: ideal Josephson element plus `Cj`. `src/twpa_solver/pump/diagnostics.py:37` names quasiparticle switching as something the model does not capture. |

Note also that `CircuitMatrices.has_loss`
(`src/twpa_solver/core/circuit.py:106`) tests **only** `Im(C)`. It never reads
`G`. `has_loss = False` on 2c therefore means "no dielectric loss in `C`", not
"no dissipation" -- the four 50-ohm port resistors are in `G` and do dissipate.

**Decision**: Phase 3 uses `tan_delta` only, at `1e-5`, `1e-4` and `1e-3`.

#### 0.9 h3 boundary is not invalidated

`docs/development/h3_physical_boundary_79.md` used `delta_theta = 0.01` and a
10-period ramp, and reported `d1 = 3.0e-3` to `3.4e-3` at `1.16e-05 A`. Confirm
by running the 0.6 floor protocol at `delta_theta = 0.01` whether `3.0e-3` is
above that protocol's own floor by a margin that survives the section-D
critique.

**Pass**: `3.0e-3` is at least one order above the `delta_theta = 0.01` floor.
**Fail action**: the h3 bracket is downgraded to `UNCONFIRMED` and Phase 2
becomes the sole source of the boundary.

**Execution record**: the exact requested run reached the 100-period restart
checkpoint and then exited without a final summary. Its checkpoint is retained
under `.hybrid_outputs/phase0_0p9_floor_dt001_ramp10_m35/`, but it contains only
the endpoint state and cannot provide `d1_late`. Therefore this check is
**INCONCLUSIVE due to solver execution failure**, not treated as a physical
pass or fail. The existing evidence still gives only `3.0e-3 / 4.25e-4 = 7.1`
floor ratios, so the h3 bracket must not be upgraded on this basis.

### Success criteria

**Automated**: `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_phase0 tests/test_periodic_orbit_floquet.py tests/test_floquet_stability.py tests/test_h1_transient_branch_transfer.py tests/test_advanced_continuation.py`

**Manual**: every check in 0.1 to 0.9 has a recorded pass or fail and, where
failed, the corresponding claim in this document is edited to match.

### Phase 0 completion record (2026-08-11)

Checks `0.1`, `0.2`, and `0.4` passed. Check `0.3` passed with a stated
`S = 10` resource gap at the two higher powers. Check `0.5` failed the
timestep-decrease criterion; `d1` was retired as a stability discriminant and
the `max_abs_phi` envelope slope was adopted as the replacement diagnostic.
Check `0.6` failed for the same reason and its power-independence evidence was
already present in the campaign record. Check `0.7` failed because the
pump-shifted group-delay comb was `241.7 MHz`, not within a factor of two of
`85 MHz`. Check `0.9` was attempted with the exact h3 protocol but terminated
after 100 periods without a summary, so it remains execution-inconclusive.
The prescribed automated suite passed: `72 passed`.

---

## Phase 1: Repair the instruments

### Overview

Three small, independently testable fixes. None changes physics.

### Changes required

#### 1. Windows console crash in the Hill CLI

**File**: `scripts/floquet_stability_sweep.py:262`
**Changes**: replace the `U+2220` character with an ASCII token (`"angle="`).
Additionally, write the output JSON **before** the summary printing block so
that a formatting failure can never destroy a completed sweep.

#### 2. Envelope-slope transient classification

**File**: `scripts/h1_transient_branch_transfer.py` (decay-aware classifier),
`scripts/run_overnight_7p9_dynamics.py`
**Changes**: use the post-ramp `max_abs_phi` envelope slope as the primary
label. A fitted slope above `1e-5 / period` is `GROWING_MAX_ABS_PHI`; otherwise
the result is `NON_GROWING_MAX_ABS_PHI`. Record the fitted slope and threshold
as first-class artifact fields. `d1` and its trend fit are secondary
diagnostics only and never determine the label. No reference floor is needed:
the discriminant is a slope, not a level.

#### 3. Branch tracking in the Hill sweep

**File**: `scripts/floquet_stability_sweep.py`
**Changes**: accept multiple `--pump-dir` values in power order and apply
`twpa_solver.stability.tracking.track_multiplier_branches` across consecutive
powers so a root is followed rather than re-seeded. Report
`max |lambda|` over all refined roots per power as a first-class field.

#### 4. Scan density guard

**File**: `scripts/floquet_stability_sweep.py`
**Changes**: compute the required point count from the measured `241.7 MHz`
mode spacing and raise if `--n-points` under-resolves it by more than a factor
of four. The recommended full-zone count is approximately `700`; `200` is
thin but not aliased, and the guard threshold is approximately `175`.

### Success criteria

**Automated**: `pytest tests/test_floquet_stability.py`, plus a new test that
the sweep CLI writes its JSON when the print block raises, and a new test that
`track_multiplier_branches` is applied across a synthetic two-power sequence.
**Manual**: `scripts/floquet_stability_sweep.py --refine-complex` completes and
writes JSON on the Windows console with no `PYTHONIOENCODING` override.

### Phase 1 implementation record (2026-08-11)

All four instrument repairs are implemented. The transient label is now based
on the post-ramp `max_abs_phi` envelope slope, with `d1` retained as a
secondary diagnostic. The HB-periodic run at `-23.421053` dBm returned
`GROWING_MAX_ABS_PHI` with slope `6.0173e-4/period`; the zero-pump run with a
160-period ramp returned the same label with slope `8.7942e-4/period`. Both
exceeded the `1e-5/period` growth threshold and completed with successful
integrators. The focused Phase 1 and prescribed regression tests pass (`77
passed`).

---

## Phase 2: Locate the real boundary

### Overview

The boundary is at approximately `1.16e-05 A`, not `7.35e-06 A`. Establish it
with the recovered-branch machinery and record what actually stops the branch.

### Changes required

#### 1. Extend the PERIOD1 recovery upward

**File**: `scripts/recover_period1_branch.py` (driver invocation only)
**Changes**: run from the `-23.421053` dBm checkpoint upward in `0.1` dB steps
through `-19.0` dBm, PALC enabled, recording per step: `converged`, `method`,
`branch_current_max_over_ic`, `branch_min_cos_phase`, `coeff_rel`,
`palc.fold_lambda`, `palc.terminal_reason`.

Expected outcome, to be confirmed or refuted: natural continuation carries the
branch to approximately `-19.7` dBm; PALC then detects a fold near
`1.16e-05 A`; beyond that no `target_lam = 1.0` is reachable.

#### 2. Confirm the fold classification at the located point

**File**: `scripts/scan_branch_singularity.py`, `src/twpa_solver/pump/singularity.py`
**Changes**: driver invocation only. Run `jacobian_min_eigenvalue`,
`jacobian_det_signature` and `bordered_conditioning` at the located fold and
confirm the `SIMPLE_FOLD` classification recorded in memory
`fold-plan-f-bifurcation-classification` still holds on this column.

#### 3. Transient confirmation at the located boundary, with controls

**File**: `scripts/run_overnight_7p9_dynamics.py` (driver invocation only)
**Changes**: at three powers bracketing the located fold, run the full control
set: `{zero-pump ramp 40, zero-pump ramp 160, HB-orbit init}` x
`{delta_theta 0.05, 0.025}` x `{implicit_trapezoid, Radau}`, all at
`hold = 440`, with the Phase 1 envelope-slope classifier. The recurrence
floor is retained only as a secondary diagnostic.

### Success criteria

**Automated**: the recovery driver writes a complete `period1_recovery.json`
with no unhandled exception and at least one `converged = false` row.
**Manual**: the first genuinely non-recoverable power is identified and agrees
with `1.1628e-05 A` to within the `0.1` dB step, or the disagreement is
documented with its cause.

---

## Phase 2b: External validation against the Themis collapse envelope

### Overview

The measurement in section I is a direct observation of the boundary Phase 2
computes. It is a sharper validation target than gain-map peak alignment: the
collapse is a single-step feature, it is resolved at 51 pump frequencies, and
its shape is calibration-free. This phase turns it into a fit.

### Changes required

#### 1. Committed measurement reducer

**File**: new `scripts/themis_collapse_envelope.py`
**Changes**: read a Themis directory, and for each pump frequency emit
`p_last_working_dbm`, `p_first_collapsed_dbm`, `peak_gain_at_last_working_db`,
`max_peak_gain_db`, and a censoring flag for frequencies where the device did
not collapse within the instrument's power range. Collapse is defined on the
median response over `|f_signal - f_pump| > 0.15` GHz falling below a stated
threshold; the threshold and the excluded pump window are CLI arguments and are
recorded in the output. Emit CSV plus a PNG of the envelope.

Reproduces `themis_collapse_envelope.csv` from section I. Censored frequencies
must never be reported as a boundary value.

#### 2. Settings-convergence of the fold locator (blocks items 4 to 6)

**File**: `scripts/run_gain_map.py --fold-follow`, `src/twpa_solver/pump/solver.py::fold_power`
**Changes**: driver invocation only. At `7.6` GHz alone, locate the fold across
the settings grid `ds` x `rescale_every` x `max_steps` x `max_steps_after_fold`
spanning the two existing runs' configurations, and report the spread in
`fold_power_dbm` and `fold_lambda`.

Motivation: the two fold curves on disk disagree at `7.6` GHz by `0.61` dB
(`-21.496451` versus `-22.109839`) with `fold_lambda` `0.5311` versus `0.4949`.
The measured collapse bracket is `0.335` dB wide, so a `0.61` dB solver spread
is larger than the measurement's own resolution and no agreement claim can be
made until it is reduced or explained.

**Pass**: spread at or below `0.2` dB across the grid, or a documented reason why
one configuration is correct and the other is not converged.
**Fail action**: Phase 2b items 4 and 5 proceed with the spread carried as an
explicit error bar on every model fold value, and no "agreement" language is
used anywhere.

#### 3. Model fold curve on the measurement grid

**File**: `scripts/run_gain_map.py --fold-follow` (driver invocation only)
**Changes**: run the fold locator at the measurement's own 51 pump frequencies
(`5.980` to `7.997` GHz, approximately `40` MHz spacing), with
`--recovery-arclength-rescale-every 5` and a per-point budget matched to the
Phase 2 findings. Without `rescale_every` this run repeats the metric-mistuning
failure documented in `arclength_fold_resolution_plan.md`.

Record `fold_power_dbm`, `fold_lambda`, and an explicit `no_fold_in_range` flag
per frequency so the model's censored points can be compared with the
measurement's.

Estimated cost: the reduced 4-point run exists as
`outputs/phase4_fold_follow_reduced`; scale from its measured per-point time
before launching. Run detached.

#### 4. Fit the calibration offsets on the boundary, not the gain lobes

**File**: new `scripts/align_fold_to_collapse.py`
**Changes**: fit `P_collapse_meas(f) ~= P_fold_sim(f - df) + dP` over the 51
frequencies by least squares on `(df, dP)`, excluding censored points on either
side. Report the residual RMS and the fitted `df` and `dP` with their
identifiability, following the pattern in `scripts/align_map_to_measurement.py`.

**Degrees-of-freedom gate.** The fit reports `n_usable_model_folds` and
`DOF = n_usable - 2`, and **refuses to emit a `(df, dP)` estimate when
`DOF < 8`**. With the two model folds available today the DOF is `0` and the fit
is vacuous; the gate makes that a hard failure rather than a plausible-looking
number. A model fold curve at 51 frequencies with even half the points
converging gives `DOF ~ 23`.

**Shape first, offsets second.** Fit the two `dP`-invariant shape statistics
before fitting `(df, dP)`: the comb period (measured `265` MHz) and the envelope
depth (measured `5.36` dB). Period and depth do not move under `dP` and period
does not move under `df` either, so they are testable with no calibration
assumption at all. If the model curve does not reproduce them, no `(df, dP)`
value will make the boundaries agree and the discrepancy is physical, not
calibration.

**Censoring is two-sided.** The measurement is censored at `-19.0267` dBm by the
instrument; the model is censored where the fold locator finds no fold within
its continuation budget. These are different mechanisms and must be carried as
separate flags. A model point censored by solver budget must never be matched
against a measurement point censored by instrument range as if the two agreed.

**Expected outcome, to be confirmed or refuted**: `df` small and `dP` near zero,
consistent with the `0.12` dB agreement already observed at `7.6` GHz, rather
than the `+2.5` to `+3.3` dB found by gain-lobe alignment at `6.2` to
`7.45` GHz. If `dP` comes out near `+2.9` dB here as well, then the `7.6` GHz
agreement is coincidental and section I's conclusion is withdrawn.

#### 5. Calibration-free check: gain at the boundary

**File**: `scripts/run_gain_map.py` (driver invocation only) plus the Phase 2
recovery outputs
**Changes**: run the signal spectrum at each recovered checkpoint and at the
located fold, so that peak gain over signal frequency exists for the model.
Compare against the measurement's `peak_gain_at_last_working_db`
(`8.4` to `33.2` dB, median near `20` dB).

The 7.9 GHz column ran with `--no-signal-spectrum`, so only a single-frequency
`gain_vs_off_db` at `7.4` GHz exists (`13.124` dB at `-24.473684` dBm). That is
not the peak and must not be compared with the measurement's peak.

This comparison is independent of both the power convention and the line-loss
model, because both sides are pump-on / pump-off ratios. It is therefore the
strongest available test of whether the model reaches the boundary for the right
physical reason.

**The observable must be matched at the source.** The model must emit
`max_f gain_vs_off_db` over the same span and with the same pump-exclusion
window the reducer applies to the measurement, not `gain_vs_off_db` at
`f_p - 500` MHz. The measured bias between the two reaches `11.90` dB and is
power dependent and non-monotone; it cannot be corrected downstream. Cost from
the column CSV is `gain_total_runtime_s ~ 0.36` s per signal frequency, so a
200-point spectrum is roughly `70` s per power point.

#### 6. Curve-shape comparison protocol

Compare the gain trajectories in this order. Each step removes a specific
nuisance parameter by construction rather than by fitting it.

**5a. Landmark referencing (primary).** Re-reference both abscissae to each
curve's own collapse or fold power, `u = P - P_boundary`. Both curves then have
their boundary at `u = 0`, so the pump-power offset `dP` is eliminated by
construction and never enters as a free parameter. Overlay `G(u)` and report the
residual. This is landmark registration in the functional-data-analysis sense,
and it is the same normalization the parametric-oscillator and laser literature
uses when it plots against pump-over-threshold.

**5b. Threshold-form scalars (primary quantitative result).** On both sides fit
`1/sqrt(G_linear)` against pump amplitude over the amplifying, pre-collapse
range and report the **intercept** (threshold) and the **slope**. Two scalars
per pump frequency from an over-determined fit, both physically meaningful. Gate
each fit on `R^2 >= 0.9` and on at least `6` points, and exclude frequencies
immediately after a comb reset where two lobes mix.

**5c. Log-slope, as a secondary diagnostic only.** `dG/dP` in dB/dB removes an
additive vertical offset `dG` exactly, but does **not** remove `dP`, and it
amplifies noise on a `0.335` dB power grid. Compute it with a Savitzky-Golay
filter so smoothing and differentiation are one operation, and present it as an
approach-to-threshold diagnostic, never as the headline agreement metric.

**5d. Global `(df, dP)` registration, last.** Only after 5a to 5c, and only
subject to the degrees-of-freedom gate in item 4.

**Explicitly rejected.** Dynamic time warping absorbs a genuine physics
discrepancy into a nonlinear reparametrization of the pump axis and will report
a good match regardless; it is a time-series retrieval tool, not a validation
tool. Procrustes and other full shape statistics remove rotation and isotropic
scale, which are meaningless for a curve whose two axes are dBm and dB. Min-max
normalization of both axes discards the dB scale, which is information under
test. A correlation coefficient alone is insensitive to a uniform offset and so
cannot detect the failure mode that matters here.

### Success criteria

**Automated**: new tests for the reducer (a synthetic cube with a known collapse
index is recovered exactly; a cube with no collapse is reported as censored, not
as a boundary) and for the aligner (a synthetic shifted curve recovers its own
`df` and `dP`).

**Manual**:
- The model fold curve and the measured collapse envelope are plotted together
  on one axis over `5.98` to `8.00` GHz.
- `df` and `dP` are reported with a residual RMS, and section I's reading is
  either confirmed or withdrawn in this document.
- Model peak gain at the fold is reported against the measured
  `8.4` to `33.2` dB range.

### Why this is worth doing before Phase 3

Phase 3 adds dissipation to make `|lambda|` decidable. Dissipation will move the
fold power. Fitting the boundary against the measurement **first**, on the
lossless model, establishes the baseline that Phase 3 must not degrade.

---

## Phase 3: Make stability decidable

### Overview

`1 - |lambda| ~ 1e-8` at a stable operating point means the discriminant has no
resolving power. Introduce dielectric dissipation so the spectrum separates.
Per check 0.8 the only channel available to `ipm_2c_fixed` is `tan_delta`.

**This costs Tier 2.** Setting `tan_delta` makes `has_loss = True`, so
`default_loss_model_for` returns `conductance_abs_omega`, which is in
`NON_ANALYTIC_LOSS_MODELS` and hard-refuses complex-omega refinement. Phase 3
must therefore decide the loss model explicitly rather than accept the default;
item 2 does that.

### Changes required

#### 1. Loss sensitivity study

**File**: new `scripts/loss_sensitivity_floquet.py`
**Changes**: build `ipm_2c_fixed` variants with `tan_delta` in
`{0, 1e-5, 1e-4, 1e-3}` through the existing dielectric-dissipation path, and
measure `max |lambda|` at `-24.473684` dBm at each, using the Phase 1 dense
scan with branch tracking. Establishes the smallest `tan_delta` that separates
the Floquet spectrum from the unit circle by more than the Phase 0.4 resolution
limit.

`tan_delta = 0` is the control and must reproduce the section F numbers.

#### 2. Loss model for stability must be chosen, not defaulted

**File**: `src/twpa_solver/signal/stability.py`
**Changes**: add a stability entry point that takes an **explicit** loss model
and refuses to silently accept `default_loss_model_for`. Two admissible routes,
both to be run at the `tan_delta` selected in item 1:

- `conductance_abs_omega` (the production default for lossy circuits, preserves
  `D(-omega) = conj(D(omega))`): **Tier 1 only**. Real-omega `sigma_min` sweep
  is unaffected; complex-omega refinement is refused.
- `current_complex_c` (full complex `C`, polynomial in `omega`, therefore
  analytic): **Tier 1 and Tier 2**. It does not preserve the conjugate symmetry,
  so it is a stability-analysis convention and must never be used to produce a
  published gain or compression number.

Report both at Tier 1, where both are legal, and quantify the disagreement in
`sigma_min` at each candidate resonance. That disagreement is the price of
Tier 2 and must be stated, not assumed small.

#### 3. Re-run the Phase 0.3 matrix at the chosen loss level

Repeat the dense Hill scan across power at the `tan_delta` selected in item 1,
with branch tracking, and report `max |lambda|` versus power.

#### 4. Re-test the time-domain monodromy route

**File**: `scripts/run_floquet_2c.py` (driver invocation only)
**Changes**: rerun the Arnoldi solve at the selected `tan_delta`. Section E
attributed its `0/2 eigenvectors converged` failure to a spectrum clustered
within `1e-8` of the unit circle; dissipation is exactly what removes that
clustering, and the time-domain route needs no analyticity, so it is immune to
the Tier-2 loss above. If it converges here it becomes a genuine candidate in
Phase 4 rather than a discarded one. Use `k >= 40` and `ncv >= 120`, not the
`k = 2` that was tried.

### Success criteria

**Automated**: `pytest tests/test_loss_model.py tests/test_floquet_stability.py`
**Manual**:
- `max |lambda|` versus power is monotone and separated from `1` by at least two
  orders above the Phase 0.4 resolution limit at a stable power, at some
  `tan_delta` in the grid; or the study reports that none of `1e-5`, `1e-4`,
  `1e-3` achieves this.
- The `conductance_abs_omega` versus `current_complex_c` Tier-1 disagreement is
  tabulated at the selected `tan_delta`.
- The monodromy rerun is recorded as converged or not, with `k`, `ncv`,
  matvecs and runtime.

### Phase 3 execution record (2026-08-12)

Implemented in `scripts/loss_sensitivity_floquet.py` and the explicit loss-model
entry point in `src/twpa_solver/signal/stability.py`. The four-point study ran
with `tan_delta = 0, 1e-5, 1e-4, 1e-3`; the smallest resolved gap was at
`tan_delta = 1e-5`, with candidate stable roots at approximately
`|lambda| = 0.999985`. The selected-loss 700-point Hill scan covered the
available recovered power sequence and found no accepted crossing. The
`conductance_abs_omega` versus `current_complex_c` Tier-1 comparison was also
recorded at the selected tangent.

The selected-loss monodromy attempt (`k=40`, `ncv=120`) did not reach Arnoldi:
the existing real-time tangent implementation rejected the complex dielectric
capacitance during closure. This is recorded as unresolved instrumentation,
not as a physical result. No ansatz switching is permitted.

Detailed record: `docs/development/phase3_loss_sensitivity_7p9.md`.

The follow-up TD checks at `tan_delta = 1e-5`, `1e-4`, and `1e-3`, including
Radau/BDF at `1e-3`, were rejected as invalid HB fixtures before integration:
the current TD implementation has no validated real-time dielectric-loss
representation. The corrected lossless mode gate and the 64-step
`-24.2500 dBm` lower-side control completed; see the detailed Phase 3 report.

---

## Phase 4: Stability instrument selection

### Overview

Choose the production stability instrument on measured cost and accuracy, not
on availability.

### Changes required

#### 1. Benchmark the three candidates at one operating point

| candidate | where | expected cost |
| --- | --- | --- |
| Hill root refinement (current) | `src/twpa_solver/signal/stability.py` | one LU per omega evaluation; already paid by the gain solve |
| Koopman-Hill projection | new | one dense projection from the assembled Hill matrix; explicit error bound |
| time-domain monodromy + Arnoldi | `src/twpa_solver/stability/` | measured 121 s for a failed `k = 2` solve at `delta_theta = 0.098` |

Koopman-Hill projection returns the full monodromy from the Hill matrix with a
proven convergence bound and avoids both the Arnoldi clustering problem and
Hill spurious-root sorting. See the reference list below.

#### 2. Spurious-root filtering if Hill eigenvalue solves are adopted

If the projection route materializes the full Hill spectrum, apply
eigenvector-symmetry or real-part sorting to filter spurious exponents.

### Success criteria

**Automated**: a new test comparing the selected instrument against the analytic
damped-oscillator reference already used in
`tests/test_periodic_orbit_floquet.py`.
**Manual**: one instrument is selected, with a recorded cost and accuracy
measurement per operating point.

### Phase 4 execution record (2026-08-12)

The branch-tracked Hill route was selected: 700-point, four-sideband scan cost
152.2 s at the selected loss operating point. The analytic damped-oscillator
control for the existing monodromy implementation gave absolute spectral-radius
error `4.01e-5` at 96 steps per period. Koopman-Hill projection was not
implemented because the selected Hill route already met the cost/accuracy gate;
the lossy monodromy route remains a real-time representation blocker.

Detailed record: `docs/development/phase4_stability_instrument_selection_7p9.md`.

---

## Phase 5: Ansatz switching, gated

### Overview

Only reached if Phases 3 and 4 produce a resolved, timestep-converged,
branch-tracked crossing at the boundary located in Phase 2.

### Phase 5 execution record (2026-08-12)

The gate is **not met**. No accepted selected-loss Hill crossing was found in
the available power sequence, and the matching L-stable transient route remains
unresolved. No new harmonic-balance ansatz was enabled.

### Gate

All four must hold before any new basis is enabled:

1. A tracked multiplier branch crosses `|lambda| = 1` at a located power.
2. The crossing is stable under sideband count and under Hill scan density.
3. An independent transient run with the Phase 1 classifier and an L-stable
   integrator shows the matching signature.
4. The crossing power agrees with the Phase 2 boundary.

### Branch actions

| crossing | action | existing scaffolding |
| --- | --- | --- |
| `lambda -> +1` | stay with PERIOD1; PALC plus deflation across the fold | `pump/solver.py`, `pump/singularity.py` |
| `lambda -> -1` | enable the half-pump `2T` basis | `pump/floquet.py`, `pump/periodic_branch.py`, `signal/period_doubled.py`, `scripts/run_period_doubled_branch.py` |
| complex pair on the unit circle | **auxiliary-generator closure on the existing `multitone` two-frequency lattice** — two extra real unknowns `(A_a, omega_a)`, two extra real equations `Y_AG(A_a, omega_a) = 0`, solved in an outer loop. Do not build a torus basis from scratch; `MultiToneBasis`, `ToneIndex` and `torus_scale` already provide the lattice. Taking `A_a -> 0` gives the bifurcation locus directly. | `src/twpa_solver/multitone/` |
| no crossing found | the boundary is a fold or an accessibility limit; improve continuation, not the ansatz | Phase 2 output |

---

## Testing strategy

### Project maturity level

Active development. The solver package under `src/` is production; the drivers
under `scripts/` are campaign tooling.

### Unit tests

- Hill CLI writes its JSON artifact even when the summary print block raises.
- `track_multiplier_branches` applied across a synthetic power sequence returns
  a continuous branch.
- Scan-density guard raises below the required resolution.
- Envelope-slope classifier: the primary label follows the post-ramp
  `max_abs_phi` slope threshold, while identical `d1` values cannot change it.
- Loss sensitivity: `max |lambda|` decreases monotonically with `tan_delta` on a
  small fixture.
- Measurement reducer (Phase 2b item 1): a synthetic cube with a known collapse
  index recovers that index exactly; a cube that never collapses is reported as
  censored and **not** as a boundary; the pump-exclusion window and threshold
  are echoed into the output.
- Calibration fit (Phase 2b item 4): recovers its own `(df, dP)` from a
  synthetically shifted curve; **refuses to emit an estimate at `DOF < 8`** and
  says so; censored points on either side are excluded rather than matched.
- Threshold-form fit (Phase 2b item 6b): recovers the known threshold of a
  synthetic `G = (1 - I/I_th)^-2` trajectory; rejects a fit with `R^2 < 0.9` or
  fewer than six points.
- Landmark referencing (Phase 2b item 6a): two curves differing only by a known
  power shift collapse onto each other exactly once referenced to their own
  boundaries.
- Observable guard: the comparison entry point raises if the model side carries
  a single-tone gain rather than a peak over the matched spectral window.
- Existing coverage retained: `tests/test_periodic_orbit_floquet.py`,
  `tests/test_floquet_stability.py`, `tests/test_advanced_continuation.py`,
  `tests/test_h1_transient_branch_transfer.py`.

Coverage target: 70 percent on new modules, with every new gate verified by
mutation.

### Integration tests

- End to end: recovered checkpoint -> Hill sweep with tracking -> report, on a
  small fixture, completing under two minutes.
- Full suite before any promotion to `main`:
  `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_full_slow --run-slow`

### Manual verification

- Phase 0 checklist fully recorded in this document, each check marked pass or
  fail, and every failed check's claim edited in place.
- Phase 2 boundary compared against `1.1628e-05 A` and against
  `docs/development/h3_physical_boundary_79.md`.
- Phase 2b: model fold curve and measured collapse envelope plotted on one axis;
  comb period and envelope depth compared before any offset is fitted; `(df,
  dP)` reported with degrees of freedom and residual RMS, or explicitly withheld
  under the gate.
- Phase 2b: threshold and slope from the `1/sqrt(G)` fit tabulated per pump
  frequency for both sides, with `R^2` and point count for each.

---

## Rollback plan

- Phases 0 and 2 add no source changes; artifacts live under `.hybrid_outputs/`
  and `outputs/`, both ignored. Rollback is deletion.
- Phase 1 is four isolated edits in two scripts plus tests. Each is a separate
  commit; `git revert` per commit.
- Phase 2b adds three new scripts (`scripts/themis_collapse_envelope.py`,
  `scripts/align_fold_to_collapse.py`, and the comparison protocol driver) and
  changes nothing existing except the `--fold-follow` and signal-spectrum
  invocations, which are driver arguments. Reverting the commits removes the
  scripts and leaves behaviour identical. The measurement directories are
  read-only inputs and are never written to.
- Phase 3 adds one new script and one new entry point; defaults are unchanged,
  so reverting the commits restores current behaviour exactly.
- Phase 4 adds one instrument behind a selection flag; the current Hill route
  remains the default until the benchmark selects otherwise.
- Phase 5 is gated and touches only dormant scaffolding; if the gate is not met,
  nothing is enabled.
- No production gain-map default, port convention, loss model used for published
  numbers, or compression driver is modified by any phase.

---

## Reference list

Methods relevant to the problems identified above.

| problem | method | source |
| --- | --- | --- |
| multipliers of a large clustered monodromy | Newton-Picard, subspace iteration, Cayley transform | Lust, *Improved Numerical Floquet Multipliers*, Int. J. Bifurcation Chaos 11(9), 2001; Lust and Roose, Newton-Picard shooting |
| multipliers from the harmonic-balance Jacobian with an error bound | Koopman-Hill projection | *Explicit Error Bounds and Guaranteed Convergence of the Koopman-Hill Projection Stability Method for Linear Time-Periodic Dynamics*, J. Nonlinear Sci., 2026 |
| spurious Hill eigenvalues | eigenvector-symmetry and real-part sorting | Lazarus and Thomas, C. R. Mecanique, 2010; robust Floquet-Hill filtering, MSSP, 2022 |
| stability of a forced harmonic-balance solution in a microwave circuit | pole-zero identification and the auxiliary generator | Suarez and Quere, *Stability Analysis of Nonlinear Microwave Circuits*, Artech House, 2003; Anakabe et al., automatic pole-zero identification |
| quasi-periodic branch continuation | two-frequency harmonic balance with phase conditions | *Continuation of quasi-periodic solutions with two-frequency Harmonic Balance Method*, JSV, 2016 |
| slow envelope dynamics without full transient cost | envelope transient / circuit envelope | Ngoya and Larcheveque; Rizzoli, Neri, Masotti |
| device-specific instability route | period-doubling cascade to chaos in a JTWPA | arXiv:2406.01185; period tripling beyond the rotating-wave approximation, arXiv:2204.12210 |
