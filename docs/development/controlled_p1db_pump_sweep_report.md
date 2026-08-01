# Controlled fixed-frequency P1dB(G) campaign

**Status:** exp29 tracks in progress  
**Signal frequencies:** 6.4 GHz for `jtwpa`; 7.4 GHz for `2c`  
**Campaign output:** `outputs/exp28_controlled_pump_sweep`

## Purpose

The earlier model and hardware slopes were obtained by sweeping frequency and regressing `P1dB` against gain. This campaign holds signal frequency fixed and varies pump current, so the resulting slopes test actual gain dependence rather than frequency structure.

The hardware cannot supply this axis. Both pump-swept slopes are model-only diagnostics and are not direct comparisons with Themis.

## jtwpa result

Seven pump-current points completed at 6.4 GHz. Four were above the 3 dB gain threshold and were used for the fit:

| Pump current (A) | Gain (dB) | P1dB input (dBm) | All-port pump depletion (dB) | Status |
|---:|---:|---:|---:|---|
| 1.85e-6 | 0.35 | — | — | NO_GAIN |
| 2.20e-6 | 0.40 | — | — | NO_GAIN |
| 2.55e-6 | -0.57 | — | — | NO_GAIN |
| 2.90e-6 | 5.83 | -98.281 | -0.0366 | VALID |
| 3.25e-6 | 13.29 | -92.647 | -0.3311 | VALID |
| 3.60e-6 | 23.93 | -109.050 | -0.1690 | VALID |
| 3.70e-6 | 26.89 | -109.478 | -0.2901 | VALID |

The ordinary least-squares fit over the four valid points is:

```text
dP1dB/dG = -0.700 +/- 0.345 dB/dB
gain span = 21.06 dB
n = 4
```

The machine-readable result is `outputs/exp28_controlled_pump_sweep/jtwpa_6p4ghz/pump_swept_slope.json`.

This establishes that gain changes at fixed frequency and that the pump-swept slope is shallow over the converged range. The P1dB sequence is non-monotonic: the 3.60e-6 A point is substantially lower than the 3.25e-6 A point. The result should not be interpreted as a clean saturation law.

## Pump-current convergence boundary

The attempted 3.90e-6 A point failed during pump continuation at lambda=1.0 with `coeff_rel=0.127` and a stalled Newton reduction. The replacement 3.70e-6 A point subsequently completed and is included in the fit. Only 3.90e-6 A remains outside the converged range.

## Comparison to the frequency-swept result

The existing `exp21` comparison remains a frequency-swept comparison. Its reported model slopes and the hardware slope must be labelled as frequency-swept and fitted over the same gain window/estimator before any model-versus-Themis conclusion is made. The fixed-frequency numbers above are not direct hardware comparisons.

## 2c result

The 2c campaign completed at 7.4 GHz with pump currents 5.80, 6.50, 7.00, 7.2310747, and 7.40 microampere. The 5.80 microampere solve is valid but never reaches a 1 dB compression crossing; the four higher-current points supply the fit:

| Pump current (A) | Gain (dB) | P1dB input (dBm) | All-port pump depletion (dB) |
|---:|---:|---:|---:|
| 5.80e-6 | 7.84 | — | — |
| 6.50e-6 | 11.61 | -88.725 | -0.1051 |
| 7.00e-6 | 13.74 | -91.363 | -0.0841 |
| 7.231e-6 | 14.68 | -92.362 | -0.0780 |
| 7.40e-6 | 15.46 | -93.289 | -0.0734 |

The four-point fixed-frequency fit is:

```text
dP1dB/dG = -1.185 +/- 0.024 dB/dB
gain span = 3.85 dB
n = 4
R2 = 0.9992
```

The 2c slope is also a model-only diagnostic; it is not a direct comparison to Themis.

The all-port depletion fields in these exp28 summaries were generated before the
exp29 balance repair and are **UNTRUSTED** until the affected sweeps are rerun.
The repair is demonstrated by `experiments/exp29_track1_power_balance.py`.

## Phase 4: frequency-swept comparison label

For reference, the existing exp21 model artifacts use `small_signal_gain_vs_off_db` as the gain estimator and are frequency sweeps. Re-fitting every valid row gives:

| Device | Frequency-swept n | Gain span (dB) | Frequency-swept slope (dB/dB) | SE (dB/dB) | R2 |
|---|---:|---:|---:|---:|---:|
| jtwpa | 8 | 3.13 | -1.086 | 1.245 | 0.113 |
| 2c | 10 | 3.35 | -0.648 | 0.125 | 0.770 |
| fqjtwpa | 8 | 20.05 | -0.629 | 0.014 | 0.997 |

These are frequency-swept model numbers, not controlled pump-swept measurements. In particular, the jtwpa pump-swept fit spans 21.06 dB while exp21 has only a 3.13 dB frequency-swept gain span, so their slopes must not be interpreted as a like-for-like estimator comparison. A hardware/Themis re-fit still requires the corresponding raw frequency-sweep table and the same gain-window rule.

The archived Themis Aug25 files confirm the hardware-axis limitation: 51 MAT files have a fixed `SignalPower = -30 dBm`, a 31-point `PumpPower` axis, and a 2001-point frequency response. They contain no signal-power sweep from which a hardware P1dB can be recomputed. The controlled pump-swept slope therefore remains model-only.

## Phase 5 gate: 2c resonator interpretation

Both controlled slopes are shallow, so the conditional resonator investigation is warranted. Existing pump-off artifacts identify the feature: the approximately 0.1843 GHz period is in the passive 2c S21 ripple, and the standing-wave report finds a median `|V_-|/|V_+|` of 0.359 at 7.629 GHz. This is a reflection/resonant buildup feature, not evidence of a missing nonlinear branch law.

The termination correction is also explicit. The branch-node line impedance is `sqrt(Lj/Cg) = 43.33 ohm` for the exp07 default (`Lj=123.9 pH`, `Cg=66 fF`). The earlier exp27 matched-port attempt used 84.6 ohm, which came from the wrong capacitance and is not the line termination to test. The corrected check uses approximately 43 ohm and preserves the all-port observables.

The passive-only 43.33 ohm check is now complete: [`passive_termination_report.json`](D:/Projects/Thesis/twpa_jax/docs/development/exp28_2c_termination_43ohm/passive_termination_report.json). It gives S21 from -3.766 to -1.251 dB (2.515 dB peak-to-peak) while retaining the same 0.1843 GHz dominant period. This confirms that the corrected termination fixes the reference normalization but does not remove the resonant ripple.

That nonlinear check has now been extended to the full five-current sweep at 7.4 GHz: [`termination_slope_comparison.json`](D:/Projects/Thesis/twpa_jax/outputs/exp28_2c_termination_43ohm_pump_sweep/termination_slope_comparison.json). The corrected 43.33-ohm fit is `-1.843 +/- 0.093 dB/dB` over `G=5.65..12.80` (`n=5`, `R2=0.992`). The production 50-ohm fit is `-1.185 +/- 0.024 dB/dB` over `G=11.61..15.46` (`n=4`). These full-span slopes are not comparable because their gain windows do not overlap. Over the matched `G>11` window, 43.33 ohm gives `-1.328 +/- 0.043 dB/dB` (`n=3`, `G=11.81..12.80`) versus production `-1.185 +/- 0.024 dB/dB`; the difference is about `0.14 dB/dB`, not `0.66`. No claim that termination materially steepens the slope is supported without a wider matched window.

The production label is explicit: the `-1.185` result comes from the production circuit at `outputs\\ipm_python_design`, whose diagonal port conductances are 0.02 S. It is not an 84.6-ohm compression result; exp27's 84.6-ohm trial was a rejected impedance estimate.

## exp29 Track 1: external power balance

The defect is now isolated. `extract_port_waves` subtracts the explicit `V/Z0`
port-shunt current before forming power waves, while the old dissipation side
retained the same shunt loss. In addition, the wave-power convention is twice
the real-torus average used by the circuit residual. The repaired code removes
the shunt loss and applies the 1/2 wave-power normalization. The standalone
before/after demonstration prints a relative error of `9.52e-1` before repair
and `6.0e-14` after repair on the two-port fixture. Existing exp28 summaries
remain marked `UNTRUSTED` for the all-port depletion field until regenerated.

## exp29 Track 2 status

The production low-gain extension was attempted in
`outputs/exp29_track2_production_wide` with a raised signal-current ceiling, but
the first 2c solve stalled under the machine's memory pressure and was terminated
after approximately 17 minutes. No new production points were written. The
existing four-point result is therefore still a local slope, not a demonstrated
wide-span law.
