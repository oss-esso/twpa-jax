# Phase 3 P1dB refinement measurement

## Method

Both numbers come from **one** sweep per device, at the exact exp20 production
configuration (25 signal-power points, `--multitone-sidebands 10`, same pump
solution directory, same ports, zero line attenuation).  The driver now records
`p1db_interpolated_dbm` alongside the refined `p1db`, so the delta is a
single-variable comparison: identical coarse points, identical solver path,
differing only in whether the crossing is interpolated log-linearly between grid
points or located by nonlinear solves inside the same bracket.

The earlier version of this measurement used **nine** coarse points, which is not
the grid any published number came from, and read the two halves off two separate
runs.  Both of those numbers are superseded.

## Validation of the comparison

The interpolated value produced by each run reproduces the published exp20 value
to 5.1e-9 dB (jtwpa) and 6.0e-9 dB (2c), confirming that the coarse sweep is the
production one and that the delta below is attributable to refinement alone.

## Result

| device | published exp20 | interpolated (this run) | refined | refined - interpolated |
| --- | ---: | ---: | ---: | ---: |
| jtwpa, 6.6 GHz | -111.458017 dBm | -111.458017 dBm | -111.118089 dBm | **+0.339928 dB** |
| 2c, 7.440816 GHz | -95.083826 dBm | -95.083826 dBm | -94.857885 dBm | **+0.225941 dB** |

Both completed with `p1db_method="refined"` and `number_of_crossings=1`.  jtwpa is
monotone; 2c reports `nonmonotonic_compression=true`.

The coarse `first_1db_crossing_dbm` is a much weaker locator than either reported
number -- -109.299513 dBm for jtwpa (1.82 dB above the refined crossing) and
-91.042443 dBm for 2c (3.82 dB above).  The grid step at 25 points over this range
is 13.7 dB in power, so most of the accuracy comes from interpolating inside the
bracket, and refinement recovers the remainder.

## Interpretation

Both deltas exceed the plan's 0.2 dB acceptance scale, but only just: 0.34 dB and
0.23 dB.

This **supersedes the earlier claim of +0.461 dB and +2.920 dB**.  That
measurement used a nine-point coarse grid whose first crossing for 2c landed at
-86.478 dBm -- 8.4 dB away from the true crossing -- so its "+2.920 dB" was
almost entirely a grid artifact of a configuration no published number used.  The
real 2c error is 0.226 dB, roughly thirteen times smaller.

Consequently the conclusion drawn from that measurement -- "exp20/21 compression
numbers should be re-run with refinement enabled; a documentation-only correction
is not defensible" -- does not follow.  What the corrected numbers support:

- The bias is systematic and single-signed: the refined P1dB is **higher** on
  both devices, i.e. both compress slightly later than published.
- At 0.23-0.34 dB it is comparable to, not large against, the 0.2 dB scale the
  basis-convergence gate uses.
- A documented correction is defensible for analyses that use P1dB as a relative
  measure across devices or frequencies, since the offset is in the same
  direction and of similar size on both devices measured.
- Re-running is required only for claims that depend on the absolute P1dB to
  better than ~0.5 dB.

What this measurement does **not** establish: that the interpolation error is
0.23-0.34 dB everywhere.  It is one frequency on each of two devices.  The error
depends on the local curvature of the compression curve inside the bracket, which
varies across the exp21 frequency sweep and is not bounded by these two points.

Two things this measurement does **not** establish:

- It does not show the interpolation error is 0.34 dB everywhere.  It is one
  frequency on one device; the error depends on the local curvature of the
  compression curve inside the bracket, which varies across the exp21 frequency
  sweep.
- It does not by itself justify re-running exp20/21 wholesale.  The refined value
  is strictly the better estimate, and the delta is a systematic offset in a
  known direction (refined P1dB is *higher*, i.e. the devices compress slightly
  later than published), so a documented correction is defensible for any
  analysis that uses P1dB as a relative measure between devices or frequencies.
  Re-running is required for any claim that depends on the absolute value to
  better than ~0.5 dB.

## Conservation diagnostics observed alongside

Emitted by the same run, and worth recording because they bear on how much weight
the deep-saturation points can carry:

- Power balance closes well: `max_power_balance_rel_err` is 8.55e-7 (jtwpa) and
  5.49e-8 (2c), rising monotonically from ~1e-16 at the smallest signal power.
- `max_manley_rowe_rel_err` is 0.533 (jtwpa) and 0.500 (2c), but on jtwpa it is
  **largest at the smallest signal power** and falls to 0.034 in deep saturation.
  A photon-flux conservation residual that is worst in the linear limit is not
  measuring what its name says, and the near-identical ~0.5 value on two very
  different devices points at a fixed factor rather than device physics.  This
  diagnostic should not be quoted until that is explained.
- `compression_model_depletion_only` reads 567.68 at small signal on jtwpa, which
  is `10^(27.541/10)` -- the model's **linear power gain**, not a compression in
  dB.  The column is unconverted, so the exp22 baseline correction that depends
  on it is not yet usable.

The latter two are Phase 2 and Phase 4 defects, outside the scope of this
measurement, and are recorded here only so they are not mistaken for physics.
