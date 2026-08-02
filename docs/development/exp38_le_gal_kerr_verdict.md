# exp38 Le Gal HB Kerr self-consistency verdict

## Task 1 - linear propagation

| method | wavenumber (rad/m) | residual |
| --- | ---: | ---: |
| stamped-cell Bloch eigenproblem | 79288.686655842 | analytic eigenproblem |
| driven recurrence | 79288.638218890 | 1.511e-07 |

The methods differ by 6.109e-07 relative
(0.000061%). The driven field has
backward/forward amplitude ratio 0.174765
and amplitude-envelope ripple 0.346799
peak-to-peak/mean; the explicit forward+backward fit residual is
1.158e-05.

`pump_modes=(1,3,5)` maps mode 1 to row 0.
The 45% low spatial result was caused by `build_effective_snail_line` stamping
the SNAIL small-signal tangent into `K` while `FullPumpProblem` also added the
full branch current, so the branch stiffness was counted twice.

## Task 2 - one primitive

The sole measurement implementation is
`src/twpa_solver/pump/wavenumber.py::measure_pump_nonlinear_wavenumber`. It fits
the standing-wave-safe interior recurrence and reports `k_linear - k_pumped`.
Both thin drivers use it:

| driver | dk_nl (rad/m) |
| --- | ---: |
| exp36 | 459.409835571 |
| exp37 | 459.409835571 |

Their relative difference is 0.000e+00 (0.000000000%).
At nominal power the pumped recurrence residual is
1.255e-03.

## Task 3 - branch-law comparison

Write the branch law as `I = g1 psi + g3 psi^3 + O(psi^5)` and the real pump
fundamental as `psi(t) = A cos(theta)`. Since
`cos^3(theta) = (3 cos(theta) + cos(3 theta))/4`, the fundamental current is
`[g1 A + (3/4) g3 A^3] cos(theta)`. Thus
`L_eff/L = 1 - (3/4)(g3/g1)A^2 + O(A^4)`. Because `k` is proportional to
`sqrt(L)` to leading order, the positive wavenumber reduction is
`k_linear-k_pumped = (3/8)(g3/g1)A^2 k_p + O(A^4)`.

The solver uses `x(t) = 2 Re sum_k X_k exp(+i k omega t)`, so the peak branch
amplitude is `A = 2|Psi_1|`. Synthesizing a unit mode-1 coefficient produced
peak 2.0,
confirming that convention.

| quantity | value (rad/m) |
| --- | ---: |
| recurrence measurement | 459.409835571 |
| independent cubic branch-law prediction | 439.977558565 |
| relative difference | 4.416652% |

**Verdict: YES - the HB solver's Kerr nonlinearity is consistent with its own
branch law (`4.42%` difference).**

In the CME input-envelope convention, HB implies `projection_factor =
0.201609224275804`. This is 1.612874 times `1/8 = 0.125` and
7.903082 times the superseded committed value
`0.025510204081632654`. No paper or measured gain entered this value.

## Task 4 - CME consequence

| signal frequency (GHz) | CME gain (dB) |
| ---: | ---: |
| 6.4 | 17.435331996 |
| 8.6 | 17.487040828 |

These values use the justified factor 0.201609224275804;
they were not tuned toward the paper's approximately 20 dB regime.

## Unverified

No direct-FFT branch-current localization was run because the measured and
analytic Kerr shifts agree within the requested 20% verdict threshold. No
pytest suite was run, as required.
