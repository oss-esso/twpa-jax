# Exp22 saturation attribution re-analysis

This note uses the existing converged exp22 artifacts in
`outputs/exp22_spatial_attribution_converged`; no nonlinear solves were rerun.
The comparison point is the coarse point nearest the stored P1dB for each
device.  The closed-form null is

\[
G_\mathrm{dep}=G_0/(1+2G_0P_s/P_p),
\]

with powers computed from the recorded source and pump currents at 50 ohms.
The spatial null re-integrates the measured branch pump amplitudes while
using the zero-signal mismatch profile.  Its branch-flux gain is normalized to
the measured small-signal port gain before comparison, because branch flux
and port S21 have different endpoint normalization.

| device | actual compression | closed-form null | spatial null |
| --- | ---: | ---: | ---: |
| jtwpa | 1.352 dB | 0.166 dB | 0.385 dB |
| 2c | 0.848 dB | 0.071 dB | 1.205 dB |

The corrected conclusions are:

1. Depletion alone does not reproduce the multitone compression for either
   device; this survives both nulls.
2. Spatial phase evolution is required to explain the residual compression;
   this also survives.
3. The previous device-contrast claim does not survive.  The additive dB
   baseline made the devices appear to point in opposite directions; the
   defensible nulls do not support that quantitative contrast.
4. The broad statement that saturation combines depletion and
   power-dependent phase evolution survives, but the relative contribution is
   device- and null-dependent.  These coarse artifacts are not evidence for a
   universal depletion/phase partition.

The spatial zero-signal gate returns the branch-flux small-signal gain exactly
(34.479 dB for jtwpa and 15.298 dB for 2c); the normalized comparison above
then applies the same offset to the finite-signal null.
