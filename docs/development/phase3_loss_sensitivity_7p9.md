# Phase 3 — dielectric-loss stability sensitivity

## Result

Phase 3 was executed on the validated `-25.5263 dBm` HB checkpoint and on the
available recovered power sequence. The four loss variants were built through
the existing IPM dielectric-loss path:

| `tan_delta` | Tier-2 convention | result at `-25.5263 dBm` |
|---:|---|---|
| 0 | `current_complex_c` | unit-circle degeneracy remains |
| `1e-5` | `current_complex_c` | candidate roots separated to `|lambda| = 0.999985` |
| `1e-4` | `current_complex_c` | candidate roots separated to about `0.954`–`0.968` |
| `1e-3` | `current_complex_c` | strongest damping; one refined candidate remains numerically unstable and is not accepted as a physical crossing without branch and sideband confirmation |

The smallest tested tangent with a resolved stable spectral gap is therefore
`tan_delta = 1e-5`. The lossless control reproduces the known degenerate
near-unit-circle behavior. Tier-1 scans with the two admissible conventions at
`1e-5` agreed in frequency and differed only in the reported `sigma_min` values;
the largest difference among the eight tracked candidate minima was
`7.21e3` in the unnormalised matrix scale. This disagreement is recorded as a
model-convention uncertainty, not silently discarded.

The selected-loss Hill sweep covered the available sequence from
`-24.4737` through `-23.4211 dBm`, with 700 points over the Floquet zone,
four sidebands, and `gamma_nt=1024`. Its per-setting JSON files are under
`.hybrid_outputs/phase3_loss_sensitivity/`.

No accepted `|lambda|` crossing was found in this sequence. The refined roots
near 3.06–3.60 GHz at the first checkpoint have magnitudes approximately
`0.999985`, and the observed root frequencies are not consistent with a
validated time-domain mode gate. The lossless Hill route remains a diagnostic
control only.

## Monodromy result

The required `k=40`, `ncv=120` selected-loss monodromy invocation did not reach
Arnoldi. Closure failed before the solve because the lossy circuit stores
complex dielectric capacitance while the existing real-time tangent state
arrays are float-valued (`complex128` cannot be cast into the real state).
This is an instrumentation/model-representation limitation, not a physical
stability result. No new real-time dielectric-loss model was introduced.

## Implementation

- `scripts/loss_sensitivity_floquet.py` builds all four variants and writes
  each result immediately.
- `src/twpa_solver/signal/stability.py` now requires an explicit loss model for
  stability calculations.
- `scripts/floquet_stability_sweep.py` requires `--loss-model` and writes
  per-setting artifacts with serializable paths.

Targeted verification: 28 tests passed, including dissipation physics,
explicit loss-model validation, and sweep artifact behavior.

## Follow-up requested checks (2026-08-12)

### Lossy TD holds and adaptive controls

The three requested 800-period TD holds at `-23.4211 dBm`, using
`tan_delta = 1e-5`, `1e-4`, and `1e-3`, all stopped before integration with
`INVALID_HB_FIXTURE`. The same happened for the Radau and BDF retries at
`tan_delta = 1e-3`. The driver correctly detected that the supplied HB
checkpoint was generated for the lossless circuit while the TD circuit had
complex dielectric capacitance; its production-HB residual gate therefore
rejected the fixture. These are not stability results and do not close C1.

Completing these checks requires a validated HB checkpoint and a real-time
dielectric-loss representation for the lossy circuit. Bypassing the gate or
reusing the lossless checkpoint would invalidate the measurement.

### Corrected mode gate

The corrected `hb_periodic`, 40-ramp/552-hold run completed successfully with
the algebraic output-voltage reconstruction. Its max-`|phi|` envelope slope was
`+0.3121 /period`, so the growth control is present. The retained output
voltage is usable (`102.4 uV` peak) and has frequency resolution `15.43 MHz`.
The largest peaks were `7.8846`, `7.9000`, and `7.8691 GHz`; the exact
half-pump component at `3.9500 GHz` had amplitude `0.00303` and was not a
dominant peak. This rules out a dominant period-doubling signature in this
window, but does not by itself identify a resolved Hill multiplier.

### Final 64-step bracket check

The `-24.2500 dBm` `hb_periodic` run completed with 64 steps per period,
53,761 steps, no step reductions, and `NON_GROWING_MAX_ABS_PHI`. Its fitted
post-ramp envelope slope was `-2.65625e-6 /period`, below the `1e-5 /period`
growth threshold. This supplies the stable lower-side control for the current
bracket; the lossy C1 check remains unresolved.
