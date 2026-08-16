# High-power branch investigation: complete run record

Date range: 2026-08-11 to 2026-08-12
Primary device: `designs/ipm_2c_fixed` at `f_p = 7.9 GHz`, pump port 4, signal 7.4 GHz,
source port 1, output port 2.
Secondary device: `jc_jtwpa` (`outputs/jc_doc_python_designs/jc_jtwpa`), RCSJ study only.
Status at pause: instability confirmed, threshold location broken, no independent route.

---

## 1. Executive summary

### Confirmed

| claim | evidence |
| --- | --- |
| The HB period-1 orbit at `-23.421053 dBm` exists and is dynamically unstable | Two initialization protocols, two timestep resolutions, matching exponential rate `0.005013 /period` |
| The HB column failure at that power was a power-step artifact, not a boundary | `1.05 dB` step fails; `0.18`-`0.25 dB` steps converge by plain Newton |
| No fold at or near the instability | `jacobian_min_eigenvalue` stays at `~1e5` across the branch, minimum `7.28e4` |
| The PALC fold at `1.1628e-05 A` is a different, higher-power phenomenon | `~4 dB` above the instability; Jacobian shows no approach |
| TD gain is timestep-limited, not probe-limited | `10x` probe moves the answer `0.007 dB`; `32 -> 64 -> 128` steps/period moves it `4.5 dB` |
| The JTWPA high-power wall is numerical | Two non-physical knobs move it: power step size (`+0.84 dB`) and RCSJ damping (`+1.41 dB`) |

### Not confirmed

| item | state |
| --- | --- |
| Instability threshold power | **Broken.** `-24.2500 dBm` flips sign between 32 and 64 steps/period |
| Independent (non-time-domain) route | Hill contradicts TD; monodromy never ran on a lossy circuit |
| Mode identity (period doubling vs Neimark-Sacker) | No valid spectrum obtained in four attempts |
| L-stable integrator corroboration (C1) | Radau and BDF both failed numerically before measuring anything |
| Residual `2.23 dB` gain gap vs HB | Survives full timestep convergence. Unexplained |

### Bracket history

```
initial          (-25.5263, -23.4211]     2.10 dB
after B2         (-24.4737, -23.4211]     1.05 dB
after Task A     (-24.4737, -24.2500]     0.22 dB   <- reported as final
after 64/pd      (-24.2500, -23.4211]     0.83 dB   <- current, unverified interior
```

---

## 2. Run tree

```
ROOT  HB column 7.9 GHz on 2c dies at -23.421053 dBm
│     (.hybrid_outputs/hb_up_7p9_m35_to_m21, 1.0526 dB steps)
│
├── A. Is the failure physical?
│   ├── A1  recover_period1_branch.py, 0.18-0.25 dB steps
│   │       -> CONVERGED at -24.25, -24.00, -23.80, -23.60, -23.421053
│   │       -> plain Newton, PALC never invoked
│   │       -> VERDICT: power-step artifact
│   ├── A2  Check 0.2: 1-step fails, 2-step and 5-step converge
│   │       -> PASS, monotone in step count
│   ├── A3  Check 0.1: cold re-solve of all 5 recovered points
│   │       -> coeff_rel < 3e-11 (bar was 1e-9) -> PASS
│   └── A4  jacobian_min_eigenvalue along -25.53 .. -23.42
│           -> 4.10e5, 4.53e5, 7.28e4, 4.00e5, 3.30e5; no approach to zero
│           -> FOLD RULED OUT; eigenvalue identity not tracked, signs meaningless
│
├── B. Pre-existing TD campaign (~8 h, 125 sims)
│   ├── B1  d1_late = 4.25e-4 +/- 1e-6, CONSTANT across 11 dB of pump power
│   ├── B2  tau_periods = 396..410 at every power
│   ├── B3  controls at -23.8 dBm: d1 falls with ramp length AND (apparently) timestep
│   │       -> later split by Check 0.5, see D5
│   └── VERDICT: UNRESOLVED_LONG_TRANSIENT labels are protocol artifacts.
│       Retracted: "~1000-period relaxation implies multiplier near 0.9992"
│
├── C. Instrument survey (pre-Phase 0)
│   ├── C1  TD monodromy on 2c: ARPACK 0/2 converged, 651 matvecs, dim 12271
│   │       one-period closure error 3.922e-3 at 64 steps/period -> route abandoned
│   ├── C2  Hill dense scan, lossless: 1-|lambda| = 6e-8 / 2.3e-6 / 2e-11
│   │       at -35 / -24.47 / -23.42 dBm -> NON-MONOTONE, most marginal at the
│   │       provably stable power -> |lambda| not decidable on a lossless circuit
│   └── C3  Mode comb from group delay: port 4->2 delay 5.885 ns = 46.5 pump periods
│           -> comb ~85 MHz -> "needs >= 2000 scan points"  [LATER WRONG, see D7]
│
├── D. Phase 0 -- double-check the caveats
│   ├── D1  0.1 checkpoints genuine .................. PASS
│   ├── D2  0.2 recovery reproducible ................ PASS
│   ├── D3  0.3 Hill basis truncation ................ PASS with gap
│   │       S=4,6,8 flat at 3 powers; S=10 lost twice to resource termination
│   │       with no artifact written
│   ├── D4  0.4 root-refinement accuracy ............. SPLIT
│   │       precision reproducible inside a basin; nearby seeds select
│   │       DIFFERENT roots -> max|lambda| is not a well-defined function of power
│   ├── D5  0.5 timestep/ramp at matched protocol .... FAIL
│   │       d1 flat in delta_theta at matched hold
│   │       -> "discretization error" reading WITHDRAWN
│   │       -> only ramp-injection and HB-init readings stand
│   ├── D6  0.6 protocol floor vs power .............. FAIL (no new runs needed)
│   │       floor does not move with delta_theta; already known power-independent
│   │       -> d1 RETIRED as a stability discriminant
│   │       -> replaced by max_abs_phi envelope slope, threshold 1e-5 /period
│   ├── D7  0.7 group delay under pumped inductance ... FAIL
│   │       measured comb spacing 241.7 MHz, not 85 MHz
│   │       -> scan density 2000 -> ~700 recommended, guard rejects below ~175
│   ├── D8  0.8 loss channels ........................ ANSWERED, no check
│   │       dielectric tan_delta (Im C, non-analytic) -- the only one available
│   │       real shunt conductance (G, analytic) -- Le Gal builder only
│   │       RCSJ -- not implemented anywhere in src/   [LATER BUILT, see J]
│   └── D9  0.9 h3 boundary floor .................... UNCONFIRMED
│           exact-protocol run terminated at 100 periods with no artifact
│           -> h3 bracket 1.140e-05..1.160e-05 A DOWNGRADED
│           -> PALC fold becomes the only high-power boundary on record
│
├── E. Phase 1 -- instrument repair
│   ├── E1  envelope-slope classifier replaces d1 as primary label
│   ├── E2  Hill CLI: ASCII output, JSON written BEFORE the print block
│   ├── E3  Hill CLI: multi --pump-dir + track_multiplier_branches
│   ├── E4  Hill CLI: density guard from the measured 241.7 MHz comb
│   ├── E5  output-port voltage retention added
│   └── 77 tests pass
│
├── F. Instability hunt (all 2c, all trapezoid unless noted)
│   ├── F1  -23.4211, hb_periodic, start==target, no current ramp, 32/pd
│   │       -> +6.02e-4 then +1.077e-3 /period -- GROWING
│   │       -> excludes zero-start ramp injection
│   ├── F2  -23.4211, zero_pump_equilibrium, ramp 160 (4x), 32/pd
│   │       -> +8.79e-4 then +1.015e-3 /period -- GROWING
│   │       -> excludes ramp-length dependence
│   ├── F3  -24.4737, hb_periodic, 32/pd, hold 800 -> +3.63e-7 FLAT
│   │   └── F3b at 64/pd -> +4.64e-7 FLAT  (both resolutions agree)
│   ├── F4  -23.4211, 64/pd, hold 840
│   │       -> departure-segment rate 0.005013 /period vs ~0.005 at 32/pd
│   │       -> RATE IS TIMESTEP-CONVERGED. endpoint blew up to 107.8 rad
│   ├── F5  Task A ladder at 32/pd, hold 800
│   │       -25.5263  -1.06e-5  (200p)      decaying
│   │       -24.4737  +3.63e-7               flat
│   │       -24.2500  +2.23e-4               growing   <- LATER CONTRADICTED
│   │       -24.0000  +3.19e-4  run A        growing   } two runs of the
│   │       -24.0000  +1.78e-1  run B        BLOWUP    } same point, 560x apart
│   │       -23.8000  +6.04e-4               growing
│   │       -23.6000  +1.08e-3               growing
│   ├── F6  -24.2500 at 64/pd -> -2.66e-6 /period  NON-GROWING
│   │       -> DIRECT CONTRADICTION with F5 at the same power
│   │       -> BRACKET (-24.4737, -24.2500] WITHDRAWN
│   └── F7  C1 L-stable controls at -23.4211, hold 800
│           Radau: max_abs_phi = 5.92e32 already in the T=60-160 plateau window
│           BDF:   max_abs_phi = 536.79 in the same window
│           -> both are integration failures from the start, NOT growth
│           -> retried with state-scaled tolerances: failed within two hold periods
│           -> C1 UNRESOLVED. Non-damping-artifact hypothesis NOT excluded
│
├── G. Gain reconciliation
│   ├── G1  hardcoded "HB reference" 6.5559 dB not reproducible by any
│   │       sideband/loss-model setting -> RETIRED, provenance unknown
│   │       -> the earlier "TD validated to 0.006 dB" claim was agreement with
│   │          an unknown number. True low-power HB value is 6.224005 dB,
│   │          so TD's 6.5497 is +0.33 dB off
│   ├── G2  HB gain at -23.421053 dBm = 14.8290 dB (recovered checkpoint)
│   ├── G3  TD subtracted gain, 200-period plateau
│   │        32/pd, probe 1e-10 ->  8.1006 dB   cancellation ratio 254
│   │        32/pd, probe 1e-09 ->  8.1072 dB   ratio  25.4   (probe ruled out)
│   │        64/pd, probe 1e-10 -> 11.4732 dB   ratio 121
│   │       128/pd, probe 1e-10 -> 12.5977 dB   ratio   7.77
│   │       Richardson (2nd order) on 32/64 predicted 12.60 -> CONFIRMED
│   ├── G4  D3 window verified pre-departure (slope -1.33e-4 over periods 100-200)
│   └── VERDICT: TD gain converged at 12.5977. Residual gap vs HB = 2.23 dB.
│       NOT discretization. NOT the probe. UNEXPLAINED.
│
├── H. Phase 3 -- dielectric loss ladder (2c)
│   ├── H1  tan_delta ladder at -25.5263 dBm (a known-STABLE power)
│   │        0     -> unit-circle degeneracy (control reproduces C2)
│   │        1e-5  -> |lambda| = 0.999985     damping 1.5e-5 /period
│   │        1e-4  -> |lambda| = 0.954-0.968  damping 3.3e-2..4.7e-2 /period
│   │        1e-3  -> strongest damping
│   │       -> LOSS BREAKS THE DEGENERACY. First physically sensible Hill output
│   │       -> stable power correctly returns |lambda| < 1
│   ├── H2  Hill sweep at tan_delta=1e-5 across -24.4737 .. -23.4211
│   │       -> NO ACCEPTED CROSSING. roots 3.06-3.60 GHz, magnitudes ~0.999985
│   │       -> but TD is LOSSLESS, so this compares two different circuits
│   ├── H3  monodromy at selected loss -> BLOCKED
│   │       complex dielectric C cannot cast into the real tangent state
│   └── H4  lossy TD holds at -23.4211 (tan_delta 1e-5/1e-4/1e-3)
│           -> all three REJECTED as INVALID_HB_FIXTURE
│           -> driver cannot seed a complex-C circuit from a lossless checkpoint
│           -> fixture was never regenerated on the lossy circuit
│
├── I. Phase 4 / Phase 5
│   ├── I1  instrument selection -> branch-tracked Hill, 152.2 s for 700 points
│   ├── I2  monodromy analytic control, damped oscillator, 96 steps/period
│   │        exact spectral radius     0.8110386975
│   │        monodromy                 0.8110788441
│   │        error                     4.01e-5   -> implementation validated
│   │                                              on REAL states only
│   └── I3  Phase 5 gate CLOSED. No PERIOD2, period-N, torus, or
│           auxiliary-generator ansatz enabled. Correct call.
│
├── J. RCSJ on JTWPA (separate device)
│   ├── J1  motivating column step-dependence
│   │        0.632 dB steps -> last PASS -29.684210, gain 29.14, max/Ic 0.6168
│   │        0.203 dB steps -> last PASS -29.491525, gain 29.70, max/Ic 0.6300
│   │        0.101 dB steps -> last PASS -28.840336, gain 39.77, max/Ic 0.6765
│   │       -> wall moves +0.84 dB as step shrinks; utilization only 0.68
│   ├── J2  RCSJ implemented and gated
│   │        exact R/Rn=inf no-op, SPD stamp, has_loss stays False,
│   │        default_loss_model_for stays current_complex_c -> TIER-2 SURVIVES
│   │        real tangent state -> the H3 monodromy blocker is removed by
│   │        construction (not yet exercised)
│   ├── J3  ladder: Ic = 3.4 uA, Cj = 55 fF, Rn = 83.1598 ohm
│   │        R/Rn    damping/period    Qj        last PASS
│   │        inf     0                 inf       -28.840336   control
│   │        1e6     3.07e-5           1.98e6    -28.840336   no shift
│   │        1e4     3.07e-3           1.98e4    -28.840336   no shift
│   │        1e2     0.307             198       -27.428571   +1.41 dB
│   │        1       30.7              1.98      >= -24.0     censored
│   │       -> physical R/Rn at 15 mK is ~1e60. Every rung tested is at least
│   │          54 orders too damped, and the two mildest change NOTHING
│   ├── J4  TD + monodromy attempted at R/Rn = 1e4
│   │       -> the rung where damping demonstrably does nothing
│   │       -> |lambda| ~ e^-0.00307 = 0.9969, still unit-circle clustered
│   │       -> both lost to resource termination. WRONG RUNG.
│   └── J5  fixture problem: last-PASS 10-mode HB point has omitted-harmonic
│           residual ~1e-4 and fails the full-residual gate
│           -> highest valid transient fixture is -34.69 dBm against a
│              wall at -28.84 dBm = 5.8 dB BELOW the phenomenon
│           -> the planned TD measurement would have been uninformative
│              even if it had completed
│
└── K. Mode identity (Task D) -- four attempts, none usable
    ├── K1  attempt 1, non-compact -> terminated, no artifact
    ├── K2  attempt 2, compact -> stopped at ~41 min, no artifact, no stderr
    ├── K3  attempt 3 -> completed, but output voltage trace IDENTICALLY ZERO
    │       -> algebraic-node velocity reconstruction bug
    │       -> fixed at scripts/h1_transient_branch_transfer.py:501
    │       -> this is the SECOND identically-zero trace in the project
    │          (the first was signal_800, output_voltage_peak_v = 0.0)
    │       -> a "no f_p/2 component" conclusion drawn from it was retracted
    └── K4  attempt 4 -> completed 552 periods, voltage non-zero
            envelope slope +0.3121 /period -> BLOWUP, not linear growth
            dominant peaks 7.8846 / 7.9000 / 7.8691 GHz at 15.43 MHz resolution
            -> pump plus its +/-1 and +/-2 adjacent bins = window leakage
            f_p/2 = 3.95 GHz amplitude 0.00303, not dominant
            -> period doubling not observed; NO resolved incommensurate pair
            -> MODE IDENTITY STILL OPEN
```

---

## 3. External reference: Themis `14.18.08`

`docs/development/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK`, 51 pump
frequencies from `5.980` to `7.997 GHz`, so it brackets 7.9 GHz. Each `.npy` is a
pickled dict; `Response` is calibrated on the unpumped device, so it **is** a
pump-on/pump-off ratio and compares directly against `gain_vs_off_db` with no
power-convention or line-loss assumption.

| property | value |
| --- | --- |
| collapse behaviour | median response over 4-12 GHz falls `~0 dB -> ~-30 dB` in one `0.335 dB` pump step |
| collapse at 7.916 GHz | 20.694 dB peak at `-22.7129`, then 2.258 dB at `-22.3778` |
| boundary shape | sawtooth comb, period `~265 MHz`, depth `5.36 dB` |
| envelope | `-24.388` to `-19.027 dBm` |
| censored frequencies | 6.908, 7.150, 7.714, 7.755, 7.997 GHz -- never collapsed in range |
| peak gain before collapse | 8.4 to 33.2 dB, median `~20 dB` |

The measured device follows `G ~ (1 - I/I_th)^-2`: fitting `1/sqrt(G_lin)` against
pump amplitude and extrapolating to zero predicts the observed collapse power to
`0.02`, `0.26` and `0.28 dB` at 7.835, 7.876 and 7.916 GHz. **The collapse is the
parametric threshold**, established from the gain trajectory alone with no model
input.

### Candidate ranking against the measurement

| mechanism | model power | vs measured `-21.4..-22.7 dBm` |
| --- | ---: | --- |
| orbit instability | `-23.4` to `-24.2 dBm` | 0.7 to 2.8 dB below -- **closest** |
| PALC fold | `~-19.4 dBm` | 2 to 3 dB above, wrong side |

**Not a match claim.** The comparison carries two unknown calibration offsets
(`df`, `dP`), the envelope slope is `~-20 dB/GHz` so `10 MHz` of frequency offset
is worth `0.2 dB` of power, and a constrained fit left `dP` uncertain by `1.9` to
`4.5 dB`. A 1-2 dB gap is well inside the admissible range and cannot be called a
discrepancy in either direction.

---

## 4. Retractions ledger

Everything below was stated as a finding and later withdrawn. Listed so it is not
re-derived.

| claim | why withdrawn |
| --- | --- |
| "~1000-period relaxation implies a Floquet multiplier near 0.9992" | the same timescale is present 11 dB lower where nothing is marginal |
| "0.862 junction utilization / 0.507 min-cos at `-23.42`" | those are a diverging Newton iterate; converged values are 0.6023 and 0.7983 |
| "d1 halves with the timestep" | Check 0.5: flat in `delta_theta` at matched protocol |
| "the 4.25e-4 floor is set by the timestep" | Check 0.6: floor is timestep-independent -> metric artifact |
| "comb spacing ~85 MHz, scans need >= 2000 points" | Check 0.7: measured 241.7 MHz -> ~700 points |
| h3 boundary bracket `1.140e-05..1.160e-05 A` | Check 0.9 never produced its floor -> UNCONFIRMED |
| "PALC fold and TD ramp bracket agree to <0.3%, two independent routes" | h3 downgraded -> PALC alone |
| "TD gain method validated to 0.006 dB" | the 6.5559 reference is unreproducible; true offset is +0.33 dB |
| "period doubling ruled out" (twice) | first from a zero trace, then from a blown-up spectrum |
| "C1 passed, growth survives Radau and BDF" | both runs were at 5.9e32 and 537 rad **inside the plateau window** |
| threshold bracket `(-24.4737, -24.2500]` | `-24.2500` flips sign at 64 steps/period |
| "66.9 dB TD gain at high power" | subtraction assumes linear response; the base state had already destabilized |
| a 19-point fold sweep at `-18.6..-23.0 dBm` | exhaustive search found no such file; three fold curves exist, none 19-point |

---

## 5. Instrument defects found and fixed

| defect | location | consequence before fix |
| --- | --- | --- |
| `U+2220` printed before JSON write | `floquet_stability_sweep.py:262` | `--refine-complex` silently produced no artifact on the Windows console |
| `d1` used as primary stability label | decay-aware classifier | returned `UNRESOLVED` on a run whose envelope grew 2.7x monotonically |
| Hill roots re-seeded per power | `floquet_stability_sweep.py` | secant fell onto a power-independent neutral root at 3.5912 GHz |
| scan density derived from a wrong comb | same | 2000-point scans specified where ~700 suffice |
| algebraic output-port voltage | `h1_transient_branch_transfer.py:501` | two identically-zero traces, two retracted conclusions |
| end-buffered artifact writes | multiple drivers | seven measurements lost, see below |
| hardcoded HB gain reference | `analyze_td_signal_subtraction.py:48` | one constant emitted at every power; provenance unknown |
| no cancellation-ratio reporting | same | a 254:1 cancellation went unflagged |
| `implicit_trapezoid` hardcoded | `run_overnight_7p9_dynamics.py:105` | A-stable but not L-stable; no numerical damping in a near-lossless circuit |

---

## 6. Failure-mode tally

Seven long-running measurements terminated without producing any artifact:

1. Check 0.3 -- Hill S=10 at `-24.47 dBm`
2. Check 0.3 -- Hill S=10 at `-23.42 dBm`
3. Check 0.9 -- floor run, stopped at period 100
4. Task D attempt 1 -- non-compact voltage run
5. Task D attempt 2 -- compact rerun, ~41 min, no stderr
6. RCSJ -- BDF transient at `R/Rn = 1e4`, reached 1.5 GB RSS
7. RCSJ -- monodromy at `R/Rn = 1e4`, no `floquet_results.json`

Plus one 2-hour Hill sweep timeout. Common cause: end-buffered writes combined with
**multiple agents holding CPU-bound Python processes simultaneously**. Both drivers
now write per-setting and per-checkpoint with atomic replacement. Serializing the
agents is the remaining fix.

A second process failure: two agents ran the same 2c workstream in parallel and
produced the contradictory `-24.0000 dBm` duplicate (`+3.19e-4` versus `+1.78e-1`,
a factor of 560) plus two overlapping Hill sweeps. One agent was additionally
mislabelled as the "jtwpa/fqjtwpa" agent while producing exclusively 2c paths.

---

## 7. Key numbers

### 2c HB gain column, signal 7.4 GHz

| P (dBm) | I (A) | `gain_vs_off_db` | `max/Ic` |
| ---: | ---: | ---: | ---: |
| -27.631578 | 4.525792083e-06 | 6.224005 | 0.33898 |
| -26.578947 | 5.108885593e-06 | 8.251661 | 0.39650 |
| -25.526315 | 5.767103642e-06 | 10.971064 | 0.49092 |
| -24.473684 | 6.510125117e-06 | 13.123893 | 0.55448 |
| -23.421052 | 7.348875911e-06 | **ERROR** | 0.86202 (diverging iterate) |

Recovered by `recover_period1_branch.py`: `-23.421053 dBm` gain `14.8290 dB`,
converged `max/Ic = 0.6023`, `min cos phi = 0.7983`.

### 2c envelope slopes, `hb_periodic`, hold 800

| P (dBm) | 32 steps/period | 64 steps/period |
| ---: | ---: | ---: |
| -26.5789 | +6.0e-07 | -- |
| -25.5263 | +1.7e-07 | -- |
| -24.4737 | +3.63e-07 | +4.64e-07 |
| -24.2500 | +2.23e-04 | **-2.66e-06** |
| -24.0000 | +3.19e-04 / +1.78e-01 | -- |
| -23.8000 | +6.04e-04 | -- |
| -23.6000 | +1.08e-03 | -- |
| -23.4211 | +1.077e-03 | rate 0.005013 |

Discriminant threshold `1e-5 /period`. Departure-segment exponential fit gives
`|lambda| ~ 1.005`, doubling `~130` periods.

### Lossless Hill scan, 700 points, S=4

| P (dBm) | max `\|lambda\|` | tracked root (GHz) |
| ---: | ---: | ---: |
| -25.5263 | 1.000000000001 | 3.0628 |
| -24.4737 | 1.000000000012 | 2.9183 |
| -24.2500 | 0.999978802245 | 2.9627 |
| -24.0000 | **1.003787809741** | 3.1851 |
| -23.8000 | 1.000000000028 | 3.0850 |
| -23.6000 | 1.000000000000 | 3.3518 |
| -23.4211 | 1.000000004338 | 3.3629 |

The `-24.0000` row reports growth where the time domain decays at `-2.27e-5`. Root
frequency jumps non-smoothly. Not a physical crossing.

---

## 8. Open questions at pause

1. **Where is the threshold?** Bracket `(-24.2500, -23.4211]`. Interior points
   `-24.0000`, `-23.8000`, `-23.6000` have 32-step evidence only, which just proved
   unreliable at `-24.2500`. Three 64-step runs restore it.
2. **What mode is it?** Period doubling not observed; no resolved incommensurate
   pair. The spectrum must be taken **inside the growth window** (roughly periods
   300-450 at `-23.4211`), windowed against leakage -- not from a saturated
   late window.
3. **Is there an independent route?** The Hill instrument now works on a lossy
   circuit (H1 proves it separates roots and returns `|lambda| < 1` at a stable
   power) but has never been compared against a time-domain run on the **same**
   circuit. The cleanest path is RCSJ on 2c: real `G`, `C` stays real, the TD
   driver works unchanged, no fixture problem, and Tier-2 survives.
4. **Why the 2.23 dB gain gap?** TD gain is converged at 12.5977 against HB's
   14.8290. Not discretization, not the probe.
5. **Does monodromy converge under damping?** Never tested at a rung where damping
   actually does anything. `R/Rn = 1e2` gives `|lambda| ~ 0.736`, well separated,
   which is the regime ARPACK needs.
6. **The JTWPA fixture gap.** No valid transient fixture exists within 5.8 dB of
   the JTWPA wall. Raising the pump mode count until a near-wall point passes the
   full-residual gate is the prerequisite for any JTWPA dynamics work -- and is the
   one context where "add more harmonics" is the correct move, for accuracy of the
   initial condition rather than for convergence.

---

## 9. Standing rules established by this work

- A harmonic-balance non-convergence is not a physical boundary. Rerun at
  `<= 0.25 dB` steps from the last converged checkpoint before reporting one.
- Never quote diagnostics from a non-converged Newton iterate as physical state.
- `d1`, `decay_aware.trend_b`, `tau_periods` and `classification` are retired as
  stability discriminants. Use the `max_abs_phi` envelope slope.
- Flag any transient reaching `max_abs_phi > 5 rad` as `BLOWUP`, not growth. A run
  can report `success: true` with `1.0` Newton iterations per step and `0` step
  reductions while the state grows 500x.
- Do not report a Floquet multiplier crossing on a circuit with `has_loss = False`.
- Do not compare a model pump power against a measured one at face value; report
  degrees of freedom and refuse a `(df, dP)` estimate below DOF 8.
- Do not compare the model's single-tone `gain_vs_off_db` at `f_p - 500 MHz`
  against the measurement's peak over the signal span. The measured bias between
  those observables reaches `11.90 dB` and is non-monotone in power.
- Write artifacts per setting and per checkpoint, atomically. Never buffer to the
  end of a long run.
- One agent per workstream. State the device directory on line 1 of every report.
