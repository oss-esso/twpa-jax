# Phase 6 stability measurement

## Why the previous result is void

The committed Phase 6 tests ran `assess_multitone_stability` on a fixture with
`Ic = np.array([0.0])` -- a linear circuit with no Josephson element -- evaluated
at `problem.zeros()`.  Both tests therefore pass against an implementation that
ignores its `state` argument entirely, which is very nearly what was shipped.

Measured on the jpa fixture (pump 4.75001 GHz, five odd pump modes, converged
multitone state), sigma_min was **bit-identical with the pump on and off**:

| pump scale | psi/phi0 | sigma_min | relative shift vs pump-off |
| ---: | ---: | ---: | ---: |
| 1 | 0.29 | 1.2566370615e+03 | 0.000e+00 |
| 1e2 | 5.72 | 1.2566370615e+03 | 0.000e+00 |
| 1e4 | 569.95 | 1.2566370615e+03 | 0.000e+00 |
| 1e6 | 57000.46 | 1.2566370615e+03 | 0.000e+00 |

A diagnostic that returns the same number at `psi/phi0 = 57000` as at zero drive
is not measuring the operating point.  Any verdict produced by that path -- in
particular the previously reported "stable, dominant exponent +2.87e-6 s^-1" --
carries no information about the nonlinear state and is withdrawn.

## Two defects

**1. Sideband ladder / Fourier index conflated.**  `assemble_conversion_matrix`
takes `ms` as a sideband ladder and looks up `khat[ell]` with `ell = m - q`; the
two are different axes.  `_q0_linearization` returned `ms = sorted(khat)`, i.e.
the pump-harmonic keys, giving the ragged asymmetric set
`[-8,-6,-4,-2,-1,0,1,2,3,4,6,8,10,12,14,16,18]` on the jpa fixture.  The
reference convention is `signal/floquet.py::sideband_list`, `range(-S, S+1)`.
Now built symmetric and contiguous with `S = max_ell // 2`, the inverse of the
reference relation `max_ell = max|m - q|`.

**2. Near-DC sideband owns the minimum.**  A dense SVD of the 38x38 conversion
matrix shows the minimal singular direction carries mass **1.000000** on sideband
`m = -1`, which for this fixture sits at `omega_s - omega_p = -10 kHz`.  Its
singular value is 1.2566e3 against 2.4e8 for the next one -- five orders of
magnitude of separation.  The circuit has no DC path, so its dynamic block
collapses as omega -> 0; that near-singularity exists with or without a pump.
Reporting it as STABLE reports the linear circuit.

`assess_multitone_stability` now returns `INCONCLUSIVE` with a stated reason when
any sideband falls within `1e-3 * omega_p` of DC, rather than a verdict it cannot
support.  Production operating points are unaffected: the closest sideband is
0.52 GHz off DC for jtwpa and 0.10 GHz for 2c, against thresholds of 7.1 MHz and
7.5 MHz.

Both defects are mutation-verified.  Reverting the ladder fails the ladder test;
disabling the near-DC guard returns `STABLE` (the old answer); making the
function read `problem.zeros()` instead of `state` reproduces `shift = 0.000e+00`
exactly and fails the sensitivity test.

## Measured verdicts at the exp22 operating points

25-point production sweeps with `--check-stability`, at the three states exp22
checkpoints.  Exponents are decay rates: negative means perturbations decay.

| device | state | status | dominant exponent (1/s) | sigma_min |
| --- | --- | --- | ---: | ---: |
| jtwpa | zero signal | STABLE | -3.3785e+08 | 88408.30 |
| jtwpa | P1dB | INCONCLUSIVE | -1.7638e+08 | 91553.33 |
| jtwpa | deepest saturation | INCONCLUSIVE | -1.1316e+08 | 89012.72 |
| 2c | zero signal | STABLE | -1.1034e-02 | 9927.47 |
| 2c | P1dB | STABLE | -1.6045e-02 | 9927.20 |
| 2c | deepest saturation | STABLE | -1.8755e-02 | 9926.26 |

Run-level `stability_status` is INCONCLUSIVE for jtwpa and STABLE for 2c.

### Reading these

- **No unstable point was found.**  Every exponent is negative on both devices at
  all three operating points, so nothing here says the deep-saturation branches
  are dynamically inaccessible.
- **The verdict now moves with the state**, which is the whole point: jtwpa's
  damping weakens monotonically as saturation deepens (-3.38e8 -> -1.76e8 ->
  -1.13e8), and 2c's strengthens monotonically (-0.011 -> -0.016 -> -0.019).
  The pre-fix implementation returned the same number at every state.
- **Scale matters when quoting these.**  jtwpa's exponents are ~1e8 1/s against
  omega_p = 4.47e10 rad/s, so |sigma|/omega_p ~ 8e-3 -- small but real damping.
  2c's are ~1e-2 1/s against omega_p = 4.74e10, i.e. ~4e-13 -- numerically
  indistinguishable from marginal.  A bare exponent with no reference scale is
  not interpretable, in either direction.
- **jtwpa's two INCONCLUSIVE points** come from `refine_complex_resonance` not
  converging, not from the near-DC guard (their `reason` is empty).  The exponent
  is still reported and is negative; the status reflects the refinement not
  reaching its tolerance, not evidence of instability.

## Scope

This is the q=0 pump-periodic slice of the two-frequency torus, matching the
existing signal Floquet implementation.  It is a measured slice diagnostic, not a
classifier for arbitrary incommensurate perturbations, and `stability_status`
stays `NOT_CHECKED` for any run that does not pass `--check-stability`.
