# Phase 3 P1dB refinement measurement

The driver was run at the existing exp22 operating points with nine coarse
signal-current samples and `--p1db-power-tol-db 0.1`.  The interpolation value
was computed from the same coarse points with the driver's legacy logarithmic
interpolator; the refined value came from additional nonlinear solves in the
coarse bracket.

| device | interpolated P1dB | refined P1dB | refined minus interpolated |
| --- | ---: | ---: | ---: |
| jtwpa, 6.6 GHz | -111.624 dBm | -111.163 dBm | +0.461 dB |
| 2c, 7.4408 GHz | -97.769 dBm | -94.849 dBm | +2.920 dB |

The jtwpa run completed with `p1db_method="refined"`, one crossing, and a
monotone curve.  The 2c run also completed with `p1db_method="refined"`, but
reported `nonmonotonic_compression=true`; its coarse first crossing was
-86.478 dBm while the bracketed refined crossing was -94.849 dBm.  This is
precisely why the crossing metadata must travel with the result.

Both deltas exceed the plan's 0.2 dB acceptance scale.  Exp20/21 compression
numbers should therefore be re-run with refinement enabled; a documentation-only
correction is not defensible.
