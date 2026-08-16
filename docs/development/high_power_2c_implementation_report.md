# 2c high-power solver implementation report

## Scope

This report records the high-power investigation for the authoritative 2c
design generated from `designs/ipm_2c.yaml`, stored for production use as
`designs/ipm_2c_fixed`. The physical circuit, junction model, and measured
attenuation profile were kept unchanged.

The map used the measured `loss_A10` attenuation model. No flat attenuation
override was supplied.

## Implemented numerical path

The production path is now:

1. matrix-free production HB with the Schur backend;
2. bounded HB recovery and PALC;
3. adaptive implicit-trapezoid TD when HB recovery is obstructed;
4. TD continuation from the last accepted restart checkpoint;
5. junction-boundary declaration only when the TD trajectory reaches the
   configured `|I_J|/I_c >= 0.999999` gate;
6. optional TD-to-HB residual homotopy and strict production residual checks.

The adaptive TD integrator retries failed steps at smaller step size and keeps
the reduced step after recovery. The TD path does not add resistance,
capacitance, Gmin, or any other physical regularisation.

An optional `--td-settle-extensions N` mode performs bounded fixed-drive holds
only when the decay diagnostic explicitly reports relaxation toward PERIOD1.
The default is zero. A fixed-drive hold is never accepted as a gain state
without a validated HB restart.

## Full-map evidence

Output: `outputs/hybrid_20x20_adaptive_final`

The run completed all 400 points in 20 frequency columns.

| status | count |
|---|---:|
| `PASS` | 217 |
| `PHYSICAL_BOUNDARY` | 20 |
| `SKIP_AFTER_PHYSICAL_BOUNDARY` | 73 |
| `TD_CONTINUE` | 90 |
| numerical failure | 0 |

The map summary reports:

- physically eligible points: 307;
- physically eligible gain-valid points: 217;
- physically eligible gain coverage: 70.684%;
- raw gain coverage: 54.25%;
- peak recorded RSS: approximately 1.20 GB;
- elapsed time: approximately 4803 s.

The earlier 223/400 reference contained 6 direct numerical failures and 32
failure skips. Point-by-point comparison gives these relevant transitions:

- 217 points remain `PASS`;
- 5 old failure-skip points now identify the physical boundary, 2 continue in
  TD, and 25 correctly inherit a post-boundary skip;
- 5 of the 6 old direct numerical failures continue in TD, while the remaining
  one is covered by a confirmed post-boundary skip;
- 90 states are explicitly retained as `TD_CONTINUE`, not mislabeled as gain
  points.

Therefore the lower raw `PASS` count is not evidence of a new numerical hole:
the new run applies an explicit junction-current boundary and refuses gain
linearisation around unresolved non-periodic TD states.

Independent checks on the completed map found:

- maximum retained pump coefficient residual: `9.80e-10`;
- maximum retained pump time residual: `9.80e-10`;
- maximum linear gain residual: `1.67e-11`;
- no attenuation override rows;
- representative pump reports use `loss_A10 c + a*sqrt(f) + b*f`, Schur CPU
  backend, and float64 state storage.

## One-column high-power result

At 7.6 GHz, the adaptive TD column reaches `|I_J|/I_c` effectively equal to
one and returns `PHYSICAL_BOUNDARY_FOUND`. The former fixed-step transient
failure is therefore a numerical integration failure, not a sufficient reason
to terminate the physical ramp.

## Settling experiment

The exact 7.6 GHz 20-point grid was rerun with four optional fixed-drive
settling extensions. Three extensions were used. They did not produce a
validated PERIOD1 HB restart or an additional gain-valid point. The trajectory
still reached the explicit junction-current boundary without a numerical
failure.

This rejects unrestricted longer holding as the primary missing method. The
settling option remains available for diagnostic use, but adaptive TD
continuation is the production high-power fallback.

A second diagnostic allowed up to twelve extensions (480 possible fixed-drive
periods). The first extension was used, after which the decay telemetry no
longer reported monotonic PERIOD1 relaxation. No PERIOD1 HB restart occurred.
This confirms that the finite-hold ambiguity is not resolved reliably by an
unbounded hold loop.

## Decision

`CURRENT_ARCHITECTURE_NEEDS_TARGETED_CHANGES`.

The architecture is sufficient to traverse the modeled 2c ramp to the
junction-current boundary without treating HB or transient solver failure as
the boundary. The remaining unresolved `TD_CONTINUE` states require a valid
periodic-orbit or Floquet treatment before gain can be computed; they must not
be promoted to ordinary gain coverage by weakening residual or state gates.
