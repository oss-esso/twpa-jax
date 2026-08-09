# Complexity-ladder production HB wiring audit

## Call graph

The accepted production map path is:

`run_gain_map.InProcessEngine` → `twpa_solver.pump.hb.FullIPMPumpProblem` →
`twpa_solver.pump.problem.HarmonicGrid` and `FullPumpProblem` →
`twpa_solver.pump.solver.HarmonicNewtonKrylovSolver`.

The problem residual and AFT are `FullPumpProblem.residual_coeffs`,
`nonlinear_current_coeffs`, and `HarmonicGrid.project_positive`. The exact JVP is
`FullPumpProblem.jvp_coeffs_with_tangent`; Newton line search and GMRES are in
`HarmonicNewtonKrylovSolver.solve_one`. Production preconditioners, fixed-current
continuation, adaptive continuation, and PALC are all methods in
`twpa_solver.pump.solver`; the map recovery code calls these methods through
`run_gain_map.InProcessEngine`.

Circuit loading is `twpa_solver.core.load_circuit`; production Josephson laws are
`twpa_solver.core.nonlinear.make_branch_law`; waveform reconstruction is
`HarmonicGrid.synthesize` using `2 Re Σ X_k exp(+i k ωt)`.

| ladder operation | current path | production equivalent | result |
|---|---|---|---|
| topology | `builders.complexity_ladder` | production `CircuitMatrices`/stamping | correct |
| HB solve | `run_complexity_ladder._checkpoint` | `HarmonicNewtonKrylovSolver.solve_one` | corrected |
| power recovery | `_checkpoint` fallback | `solve_continuation` | corrected |
| residual gate | `pump.validation.validate_production_hb_state` | `FullPumpProblem.norms` | added |
| transient | `h1_transient_branch_transfer.run_experiment` | validated H2 transient | gated |
| debugger | `debug_ladder_roundtrip`, `debug_n16...` | same validator/HB stack | gated |

The defect found was in the ladder client: it passed `frequency_hz` directly as
`HarmonicGrid.omega`, whereas the production convention requires
`omega=2π frequency_hz`. The transient used the correct angular frequency. This
has been fixed and covered by the production validation boundary.

## Checkpoint policy

`src/twpa_solver/pump/validation.py` is the single validation contract. It checks
state shape and finiteness, rebuilds the production `FullPumpProblem`, evaluates
the production coefficient residual, and records solver/module/entrypoint,
frequency, current, harmonics, dimensions, and residuals. H1 now aborts with
`INVALID_HB_FIXTURE` before integrating if this gate fails.

The audit output is [complexity_ladder_provenance.json](../../complexity_ladder_provenance.json).
All pre-reset ladder checkpoints are invalid legacy artifacts. Fresh outputs
`ladder_prod_n8_*`, `ladder_prod_n16_*`, and `ladder_prod_n32_*` are
`VALID_CURRENT_PRODUCTION`.

## Fresh production results

| N | drive | production residual | TD class | max `r_J` |
|---:|---:|---:|---|---:|
| 8 | 1.50 Ic | `4.44e-15` | PERIOD_1 | 0.811 |
| 8 | 1.625 Ic | `6.16e-13` | PERIOD_1 | 0.882 |
| 16 | 0.25 Ic | `7.42e-15` | PERIOD_1 | 0.148 |
| 16 | 1.50 Ic | `9.10e-14` | PERIOD_1 | 0.863 |
| 16 | 1.625 Ic | `6.98e-15` | PERIOD_1 | 0.929 |
| 32 | 1.50 Ic | `4.99e-10` | PERIOD_1 | 0.862 |
| 32 | 1.625 Ic | `1.15e-13` | PERIOD_1 | 0.913 |

At N=16, a fresh 2.0-Ic solve failed in the actual production stack: direct
Newton stalled, and the 20-step production power-substep route failed at
source scale 0.95. No transient was run for that point.

The old N=8 1.50/1.625 bracket and old N=16 nonperiodic interpretation are
retracted. These fresh points do not establish a period-1 loss bracket.
