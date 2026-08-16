# Floquet bifurcation diagnostic

The implementation adds a diagnostic branch to the existing Hill-matrix
stability path. It does not alter the default pump or gain workflows.

## What is implemented

- A refined complex Hill root is converted to its one-pump-period multiplier
  with `exp(+i*omega*T_p)`.
- Roots near `+1`, `-1`, and other unit-circle phases are labelled as fold,
  period-doubling, and Neimark--Sacker candidates, respectively.
- `scripts/floquet_stability_sweep.py --refine-bifurcations` explicitly tests
  Floquet-zone guesses. The default guesses are `0.0,0.5` times the pump
  frequency.
- `period_doubled_basis` creates a dense basis with fundamental `f_p/2`, DC,
  and the physical pump at mode two.
- `build_period_doubled_seed` maps a refined Hill eigenvector into that basis.
- `scripts/run_period_doubled_branch.py` now performs the complete opt-in
  route: checkpoint validation, half-pump Floquet refinement, nonlinear HB
  correction, short PALC continuation, junction-utilization stopping, and
  optional half-pump gain evaluation.

## Acceptance rule

The multiplier label is not a physical-state result. The Hill truncation must
be converged, the nonlinear half-pump seed must be corrected by production HB,
and the resulting state must pass the independent full-residual and provenance
checks. A TD period-2 classification is still required before using the state
for gain.

The branch CLI refuses to seed unless the refined root is converged and is
classified as a period-doubling candidate. It writes no gain result for an
unvalidated HB state.

## Current verification

The focused stability, pump-basis, signal-import, half-pump, and multitone seed
tests pass. No claim is made yet about a period-doubling crossing in the real
2c circuit; that requires an explicit, memory-budgeted run on validated pump
checkpoints.
