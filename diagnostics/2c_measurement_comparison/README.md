# 2c high-power continuation check

This diagnostic tests whether pump-solver failures identify a physical fold by
comparing the standard calibrated 2c model (`outputs/ipm_python_design`) with
the measured 2c data under
`docs/17.03.10_Themis_SetupAug25_noVTS_transmission_15mK`.

## Same-frequency continuation at 7.043 GHz

Successively smaller power steps crossed every apparent coarse boundary:

| Power step | Apparent result |
| ---: | --- |
| 0.5 dB | first failure at -39.5 dBm |
| 0.05 dB | all points from -40 to -39 dBm converge |
| 0.1 dB | first failure at -30.4 dBm |
| 0.02 dB | crosses -30.4 dBm; isolated failures at -30.30 and -30.20 dBm |
| 0.005 dB | all points from -30.24 to -30.10 dBm converge |
| 0.01 dB | all points from -30.10 to -29.50 dBm converge |

Model gain reaches 7.996 dB at -30.48 dBm, close to the apparent coarse wall,
then decreases smoothly to 6.198 dB at -29.50 dBm. The last point has residual
`4.45e-13` and needs two Newton iterations. The coarse failures therefore
occurred near the top of a gain lobe; they were not a fold of the periodic
solution branch.

Saved runs are `raw_7043_fine`, `raw_7043_boundary_fine`,
`raw_7043_boundary_ultrafine`, and `raw_7043_boundary_ultrafine2`.

## Measurement comparison

At 7.043 GHz, measured peak gain rises monotonically to 17.795 dB at
-21.754 dBm and collapses to -9.510 dB at the next sampled power,
-21.453 dBm. Across frequency, pre-collapse power forms a descending,
resetting envelope with a characteristic reset spacing of about 0.23 GHz.

After a frequency shift of about +0.91 GHz, the existing standard-2c model
solver-boundary envelope has correlation 0.96 with the clean measured wave
preceding a reset and 0.86 over the longer main wave. This is qualitative only:
the model has substantial frequency/power calibration offsets and larger wave
amplitude, and its passive pump transfer at 7.043 GHz differs from the newer
exact 2c build.

## Interpretation

Measurement supports a real abrupt high-power collapse, but the solver's first
failed cell is not a reliable estimator of it. Here, coarse failure landed near
the middle/top of the modeled gain lobe and finer natural continuation
recovered almost another 1 dB of branch. The 7-row stop should therefore remain
labelled a numerical continuation boundary until a step-size-independent
turning point is demonstrated, preferably by a clean pseudo-arclength trace.

## Matched-column fold localization

The high-frequency 2c column at `fp = 8.1530612245 GHz` is the closest
qualitative match to the measured 7.241 GHz wave after allowing the observed
frequency translation (`-0.91 GHz`) and a power translation of about `+4.3 dB`.
Its natural continuation was refined from the verified `-24.2 dBm` state in
`0.005 dB` steps.

The trace crosses isolated solver failures and reaches a sharp gain maximum of
`52.86 dB` at `-23.835 dBm`. It remains converged through `-23.635 dBm`, while
the next point (`-23.630 dBm`) fails. A two-point pseudo-arclength probe,
seeded at `-23.645` and `-23.640 dBm`, reports a fold at
`lambda = 0.99505896`, equivalent to approximately `-23.6330 dBm` for this
column. Target-power continuation above it has no crossing within the bounded
trace.

This is the first step-size-independent numerical fold found in the matched 2c
column. It is still a model-to-measurement comparison, not an absolute
calibration: the translated model peak is much higher than the measured peak.
The robust conclusion is the location and existence of the turning point, not
the absolute gain.

Saved runs:

- `matched_8153_seeded_step005` — natural `0.005 dB` refinement.
- `matched_8153_arclength_budget60` — seeded pseudo-arclength probe and fold
  record (`lambda = 0.99505896`).
