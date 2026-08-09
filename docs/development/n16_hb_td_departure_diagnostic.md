# N=16 HB versus transient departure diagnostic

Canonical fixture: `ladder_jtl16_constant025`, 7.9 GHz, 0.25 `Ic`, constant
drive, 10 retained pump harmonics.

The stored checkpoint was not an exact HB root. Evaluating its reconstructed
waveform in the exact time-domain equations gave a normalized projected/time
residual of about `4.21`, with essentially all of that residual in retained
harmonics. Re-solving the same checkpoint with the production HB Newton solver
corrected it in three iterations. The corrected root differs from the stale
checkpoint by relative norm `6.31` and has coefficient residual
`7.16e-15`, time residual `3.55e-12`.

For the corrected orbit, the exact pointwise residual on 4096 phase samples is:

| quantity | normalized value |
|---|---:|
| maximum pointwise | `1.56e-11` |
| RMS | `3.55e-12` |
| retained-harmonic RMS | `5.85e-15` |
| out-of-band RMS | `3.55e-12` |
| algebraic rows | `0` (no algebraic row in this fixture) |

One-period trapezoidal integration converges systematically:

| Δθ | closure | junction-observable error |
|---:|---:|---:|
| 0.020 | `6.80e-5` | `1.35e-5` |
| 0.010 | `1.70e-5` | `3.37e-6` |
| 0.005 | `4.26e-6` | `8.43e-7` |

The largest linearized frequency is `ω_max/ω_p=4.753`; at Δθ=0.01,
`ω_max Δt=0.0475`, so the tested step is not under-resolving this mode.
Ten exact-period integrations keep the corrected-orbit stroboscopic error near
`1.7e-5`, rather than growing exponentially.

A finite-difference shooting correction at Δθ=0.02 reduces the one-period
closure from `6.80e-5` to `1.25e-8` in one Newton update. The leading
monodromy values are one gauge-like multiplier at `|λ|≈1` and nontrivial
conjugate pairs below one, the largest being `|λ|≈0.99936`. This does not
support an unstable periodic orbit.

The previous N=16 departure was therefore caused by an invalid/stale ladder
checkpoint, not by a physical instability or a transient timestep failure.
The requested status remains `INCONCLUSIVE` because the original fixture was
not a trustworthy HB state; no complexity-ladder or RCSJ work should resume
until the ladder runner persists the corrected HB root and its actual solver
report.

Artifacts are in `n16_hb_td_debug_final/`, including the JSON report, residual
spectrum, one-period convergence plot, and stroboscopic drift plot.
