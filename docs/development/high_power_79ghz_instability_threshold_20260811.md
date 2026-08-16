# 7.9 GHz PERIOD1 instability: threshold and independent-route audit

Date: 2026-08-11. Device: `designs/ipm_2c_fixed`; `f_p = 7.9 GHz`; pump port 4;
source port 1; output port 2.

## Current result

The completed 800-period extensions close the envelope threshold bracket as:

```
-24.4737 dBm < P_threshold <= -24.2500 dBm
```

The -24.4737 dBm run is stable (`3.63e-7 / period`); the completed 800-period
-24.2500 dBm run grows (`2.23e-4 / period`). The -24.0000 dBm results are retained as
contradictory duplicate runs and are not used as evidence for the threshold.

The instability is supported by the timestep test, the two initialization controls, and
the Jacobian/fold check. The L-stable controls are unresolved: both adaptive runs
reached nonphysical amplitudes before a meaningful departure could be measured. The
requested third independent route is not achieved.

## Task A: threshold sweep

All runs used `hb_periodic`, ramp 40 periods, `max-step = 0.19634954084936207`,
`min-step-theta = 0.03125`. Slopes are least-squares fits of `max_abs_phi` over `T=100`
to the end; ratios are slope divided by the window mean.

| P (dBm) | pump current (A) | hold | slope (/period) | window mean | slope/mean |
|---:|---:|---:|---:|---:|---:|
| -25.5263 | 5.767103642e-06 | 200 | -1.0623e-05 | 0.510404 | -2.0812e-05 |
| -24.4737 | 6.510125117e-06 | 800 | 3.6291e-07 | 0.569557 | 6.3719e-07 |
| -24.2500 | 6.679955138e-06 | 800 | 2.2331e-04 | 0.623182 | 3.5834e-04 |
| -24.0000 | 6.875013350e-06 | 800 | 3.19e-04 | 0.7836 | 4.07e-04 |
| -23.8000 | 7.035152983e-06 | 800 | 6.0355e-04 | 0.737399 | 8.1849e-04 |
| -23.6000 | 7.199022747e-06 | 800 | 1.0842e-03 | 0.877117 | 1.2361e-03 |
| -23.4211 | 7.348875600e-06 | 840 | 6.3572e-04 | 0.752151 | 8.4521e-04 |

The threshold rule is slope `> 1e-5 / period`. The initial 200-period values at
-24.2500 (`3.58e-06`) and -24.0000 (`-2.27e-05`) were inside the departure blind window
and were superseded by the 800-period extensions. The supported bracket is therefore
`(-24.4737, -24.2500] dBm`; the -24.0000 point is not used to set the bracket.

Two nominally identical -24.0000 dBm directories exist. Both metadata files identify
the same validated checkpoint and current (`6.87501334995057e-06 A`), `hb_periodic`
initialisation, 40-period ramp, 800-period hold, and 32 steps per period. They remain
contradictory: `phase1_taskA_m24p0_hold800` has slope `3.19e-4 / period` and bounded
endpoint `0.7836`, while `phase1_taskA_m-24p0_hold800` has slope `1.78e-1 / period`
and endpoint `164.15`. The latter is withdrawn pending a reproducible explanation.

## Task B: timestep confirmation

At -23.421053 dBm, current `7.348875600e-06 A`, `hb_periodic`, ramp 40, hold 840,
64 steps/period, implicit trapezoid, the departure-segment rate is `0.005013 / period`.
The 32-steps/period comparison is approximately `0.005 / period`; this is a clean pass.

Artifact: `.hybrid_outputs/phase1_taskB_dt64_hb_periodic`.

## Task C1: L-stable controls

At -23.421053 dBm, ramp 40, hold 800, both Radau and BDF showed envelope growth before
ending with `Required step size is less than spacing between numbers`.

This is unresolved, not a pass. BDF reached `|phi|=536.79` in the T=60--160 plateau
window and `4073` overall; Radau reached `5.92e32` in the same window and `2.14e35`
overall. These are adaptive integration failures at nonphysical amplitudes, not measured
departures. C1 does not exclude the implicit-trapezoid non-damping hypothesis.

The driver now exposes `--atol-mode state_relative` and `--atol-floor` for a controlled
retry with per-state absolute tolerances. Short controlled retries with
`rtol=atol=1e-8`, `atol-floor=1e-14`, and state-relative scaling still terminated within
two hold periods for both BDF and Radau. C1 is formally unresolved and is not used as
evidence.

Artifacts: `.hybrid_outputs/phase1_taskC1_radau/summary.json` and
`.hybrid_outputs/phase1_taskC1_bdf/summary.json`.

## Task C2: tracked Hill scan

The ascending seven-power scan used all pump directories in one invocation, 700 points,
sidebands 4, `gamma_nt=1024`, and complex refinement.

| P (dBm) | max `|lambda|` | first tracked root frequency (GHz) |
|---:|---:|---:|
| -25.5263 | 1.000000000001 | 3.0628 |
| -24.4737 | 1.000000000012 | 2.9183 |
| -24.2500 | 0.999978802245 | 2.9627 |
| -24.0000 | 1.003787809741 | 3.1851 |
| -23.8000 | 1.000000000028 | 3.0850 |
| -23.6000 | 1.000000000000 | 3.3518 |
| -23.4211 | 1.000000004338 | 3.3629 |

This contradicts the time-domain control at -24.0000 dBm, whose slope is decaying
(`-2.27e-05 / period`). It is positive evidence that the reported maximum is a neutral
root of the lossless clustered spectrum, not a physical multiplier. The branch frequency
also switches non-smoothly. C2 is therefore not a confirmation, and C3 was correctly
skipped.

Artifact: `.hybrid_outputs/phase1_taskC2_hill_tracked_700.json`, plus per-setting files.

## Task D: mode-frequency gate

The existing `td_mode_gate_7p9_2c/m23p421_zero512` artifact contains 552 periods and a
bounded envelope slope (`5.03e-4 / period`), but its retained output-voltage trace is
identically zero. This is an instrumentation defect: port 2 is the circuit's index-one
algebraic node (`4576`), and the reduced-state unpacker does not carry its velocity.
The artifact therefore cannot rule out `f_p/2` or identify a Neimark--Sacker sideband.

The compact driver now reconstructs algebraic-node velocity before recording voltage and
writes the trace incrementally at the existing 10-period cadence.
`scripts/analyze_td_mode_gate.py` reports the FFT resolution, nearest `f_p/2` bin, and
ranked sideband frequencies. A corrected Task D run is required before applying the
cross-route frequency gate.

## Verification

The compact transient hot loop was optimized without changing the equations or protocol:
step-invariant old-state quantities are computed once per step, bounded-path garbage
collection is amortized, and a conservative two-step chord-Newton LU reuse is used with
residual-triggered refactorization. The 10-period benchmark decreased from 7.79 s to
6.83 s. Both 800-period extensions completed with `26881` accepted steps.

The original prescribed suite passed: `77 passed in 27.86s`. The correction tests pass
separately: `14 passed` across the transient and mode-gate test files.

`graphify update .` was attempted after code edits but exceeded the 60-second command
limit during AST extraction; only existing access-denied warnings for disposable pytest
directories were emitted.
