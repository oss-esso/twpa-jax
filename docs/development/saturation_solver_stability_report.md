# Saturation solver stability report

Phase 6 adds `--check-stability`, default off. It constructs the finite-signal
torus tangent, projects the existing Floquet machinery onto the pump-periodic
(`q=0`) slice, and reports the dominant complex exponent. Root failures are
`INCONCLUSIVE`; a numerical growth tolerance of `1e3 s^-1` prevents a residual
of a few microseconds inverse from being reported as physical growth.

The opt-in CLI smoke on the passive JPA fixture (`n_signal_power=2`, signal
4.5 GHz) returned `STABLE` at zero signal and deepest saturation. Both measured
dominant exponents were approximately `2.87e-6 s^-1`, with `sigma_min` about
`3.14e7` and matrix size 64; this is below the numerical-growth tolerance.

The checked-in exp22 artifacts contain branch profiles and observables, not the
full converged torus coefficient arrays needed to reconstruct the Floquet
tangent. Therefore the requested jtwpa and 2c zero/P1dB/deep-saturation
verdicts were not measured in this phase. No conclusion about dynamic
accessibility of those deep branches is claimed; the campaign must be rerun
with state checkpoints and `--check-stability` enabled.

