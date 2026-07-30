# Le Gal phase-matching verdict

## Phase 3 gate

## Superseding Phase 5 calibration correction

Phase 5 measured the HB output pump node flux as `4.5057e-16 Wb` at
`-78.4 dBm`, while the old CME input convention gave `6.3719e-16 Wb`.
The ratio is `0.7071 = 1/sqrt(2)`, so the CME power-to-flux conversion now
uses the solver's peak-node convention with the explicit `sqrt(2)` denominator.
The original `+591.34 rad/m` nonlinear term overpredicted the measured pump
phase: `3.602300 rad` versus `+0.367634 rad`. The corrected effective
projection gives `dk_nl = +60.4 rad/m`.

The regenerated budget has symmetric phase-matched rows at `7.0` and `8.0`
GHz: `dk_lin = -127.968 rad/m`, `dk_total = -67.568 rad/m`, and
`|dk_total| L = 0.41164 rad`.

The prior Branch-A frequency statement at 6.4/8.6 GHz is superseded. The
corrected gate remains Branch A, but the lobe pair moves to 7.0/8.0 GHz.

Branch A: the signed budget contains phase-matched rows. At the published
parameters, the table's smallest measured value is at `f_s = 8.6 GHz` and
`f_i = 6.4 GHz`: `dk_lin = -621.558 rad/m`, `dk_nl = +591.340 rad/m`,
`dk_total = -30.218 rad/m`, and `|dk_total| L = 0.1840 rad` for
`L = 0.00609 m`. The scan therefore contains a row below 1 rad.

At the required 6.0 GHz check, the discrete ladder gives
`dk_lin = -1160.247 rad/m`; the nonlinear term is `+591.340 rad/m`, so the
terms partially cancel and `|dk_total| L = 3.464 rad`. The CME magnitude
cross-check is `591.34 rad/m`, between the recorded `591.3 rad/m` CME value
and the `707.8 rad/m` direct value.

The published `r = 0.062` has `g3/g1 = +0.015333515`. No reinterpretation of a
published parameter was made. The lobe scan is required next because the
gate is positive at 8.6 GHz but negative at 6.0 GHz; the four sampled Level-2
frequencies did not include 8.6 GHz.
