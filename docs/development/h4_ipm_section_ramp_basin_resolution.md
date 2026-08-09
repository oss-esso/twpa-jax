# H4: 418-cell IPM-section ramp/basin resolution

## Result

Primary status: `LONG_RAMP_TRANSIENT`.

The apparent broadband/quasiperiodic response of the 418-cell IPM nonlinear
section was not persistent at the tested 1.0-Ic endpoint. It relaxed to the
known PERIOD_1 state under a sufficiently long constant-drive hold.

## Production-HB provenance

Every run used the production pump solver and validation contract. The starting
0.5-Ic/0.8-Ic roots were freshly solved through
`HarmonicNewtonKrylovSolver.solve_one` and passed validation. The independently
initialized fixed 1.0-Ic root also passed with residual
`5.95e-11`.

## 418-cell controls

| experiment | endpoint | hold | result | stroboscopic tail | max `r_J` |
|---|---:|---:|---|---:|---:|
| 0.5→1.0 ramp | 1.0 Ic | 20 periods | broadband candidate | nonzero | ~0.58 |
| 0.5→1.0 ramp | 1.0 Ic | 80 periods | `PERIOD_1` | `4.58e-5` | 0.582 |
| 0.5→1.05 ramp | 1.05 Ic | 20 periods | broadband candidate | nonzero | measured in artifact |
| 0.5→1.10 ramp | 1.10 Ic | 20 periods | broadband candidate | nonzero | measured in artifact |
| 0.8→1.0 slower ramp | 1.0 Ic | 20 periods | `QUASIPERIODIC_OR_PERIOD_N` | `2.55e-3` | 0.578 |
| 0.8→1.0 slower ramp | 1.0 Ic | 80 periods | `PERIOD_1` | converged | measured in artifact |
| 0.8→1.0, max step 0.005 | 1.0 Ic | 20 periods | `QUASIPERIODIC_OR_PERIOD_N` | `2.56e-3` | 0.578 |
| fixed-drive HB initialization | 1.0 Ic | 10 periods | `PERIOD_1` | `1.29e-5` | 0.580 |

The 0.005-step repeat agrees with the 0.01-step run: the endpoint utilization,
stroboscopic distance, and winding are unchanged to the shown precision. Thus
the intermediate nonperiodic classification is not a timestep artifact.

The 20-period slower-ramp state differs from the fast-ramp state, but both
return to PERIOD_1 after 80 periods. This is best described as slow relaxation
and transient history dependence, not demonstrated coexisting persistent
attractors.

At the same 1.0-Ic drive a valid production PERIOD_1 HB root exists, so this is
not an existence-limit result. No Floquet escalation was needed: the fixed-drive
TD control preserves the root and the ramp departure decays during the extended
hold.

## 2508 conventional-line control

The separate `UNIFORM_JTWPA_2508` fixed 1.0-Ic test remains PERIOD_1 with a
validated HB residual of `2.64e-10` and successful HB→TD→HB round trip. The
requested fresh 80-period 2508 ramp hold was started but terminated because its
2509-node implicit transient was substantially more expensive than the 418-cell
control. Therefore it is reported as incomplete, not as a physical failure.

## Physical-principle classification

| circuit | classification | TWPA? | reason |
|---|---|---|---|
| single JJ | `SINGLE_JUNCTION` | no | one nonlinear resonant element |
| N=8,16,32,64 | `NONLINEAR_TRANSMISSION_LINE_SECTION` | partial | repeated line cells and 50-ohm ends, but no engineered phase matching or validated traveling-wave metric |
| `UNIFORM_JTWPA_2508` | `NONLINEAR_TRANSMISSION_LINE_SECTION` / `TWPA_LIKE` | partial | long propagation path, but uniform production-IPM parameters and no explicit phase-matching loading |
| repository `build_jtwpa()` | `TWPA_CORE / TWPA_LIKE` | yes, by design | 2048 cells, matched ports, periodic capacitive/resonator loading and documented pump operation |
| actual 418-cell IPM section | `NONLINEAR_TRANSMISSION_LINE_SECTION` | partial | one real IPM section with production cells and terminations, but couplers/embedding paths removed |
| complete IPM | `COMPLETE_TWPA` | architecture-dependent | full production parametric topology; its traveling-wave character must be assessed from the complete network rather than JJ count |

The short artificial ladders and the 418-cell isolated section should not be
called complete TWPAs solely because they contain many junctions. The repository
2048-cell fixture is the clearest conventional TWPA benchmark because its
periodic loading is explicitly intended to engineer dispersion. The present
milestone did not add new S-parameter or forward/backward-wave calculations;
those remain a separate low-drive characterization task if needed.

## Artifacts

Results are under the gitignored `outputs/` tree, notably:

- `h4_ipm_section_ramp100_hold20`
- `h4_ipm_section_ramp100_hold80`
- `h4_ipm_section_slowwindow_080_100`
- `h4_ipm_section_slowwindow_080_100_dt005`
- `h4_ipm_section_slowwindow_080_100_hold80`
- `h4_uniform_jtwpa_2508_ramp100_hold80` (incomplete run directory)
