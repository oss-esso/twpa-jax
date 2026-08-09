# Hybrid-column validation: 7.0 and 7.7 GHz

Validation used the production circuit, the production HB stack, full-node
production checkpoints (float64 for the TD handoff), the fixed −26 to −16 dBm
20-point grid, a 10-period upward ramp, and a 40-period hold. No overnight map
was launched.

| Frequency | Old direct HB wall | Last WORKING_PERIOD1 | First OUTSIDE_PERIOD1 | Bracket | TD bridges | TD periods | HB restart successes | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7.0 GHz | about −23.895 dBm | −22.842 dBm | −22.316 dBm | [−22.842, −22.316] dBm | 1 | 40 | 0 | PHYSICAL_BOUNDARY_FOUND |
| 7.7 GHz | about −20.737 dBm | −21.263 dBm | −20.737 dBm | [−21.263, −20.737] dBm | 1 | 40 | 0 | PHYSICAL_BOUNDARY_FOUND |

The old direct-HB walls are the first failed points in the accepted G1 direct
coverage (4/20 at 7.0 GHz and 10/20 at 7.7 GHz). The new 7.0 run extends
ramp-selected PERIOD_1 operation by two grid points; 7.7 reaches its direct-HB
wall and then transitions through TD.

## Routes

7.0 GHz:

```text
DIRECT_HB -> POWER_SUBSTEP -> POWER_SUBSTEP -> POWER_SUBSTEP
           -> DIRECT_HB -> TD_BRIDGE -> PHYSICAL_BOUNDARY
```

7.7 GHz:

```text
DIRECT_HB x 10 -> TD_BRIDGE -> PHYSICAL_BOUNDARY
```

All accepted HB residuals were below the 1e-8 gate; the largest listed
production residual was about 1.3e-10 at 7.7 GHz. TD started from a validated
PERIOD_1 checkpoint in both runs. Both held states were classified as
persistent non-PERIOD1; their low-order stroboscopic distances did not settle
to the PERIOD_1 floor. No TD-to-HB restart was naturally required, so no bridge
restart result is fabricated.

| Frequency | TD r_J | phi_max (rad) | min cos(phi) | phase winding (cycles) | HB time (s) | TD time (s) | output size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7.0 GHz | 0.764 | 0.869 | 0.646 | −3.01e−4 | 136.1 | 167.5 | 86.3 MB |
| 7.7 GHz | 0.913 | 1.150 | 0.409 | −1.78e−5 | 42.9 | 143.3 | 89.3 MB |

Peak memory was not exposed by the current process telemetry. The output size
is larger than the intended compact-map target because the production engine
still persists full pump/gain artifacts for each attempted point; this is a
remaining operational limitation.

## Gate

The controller executed the real columns correctly and kept numerical/unresolved
states distinct. However, neither column naturally exercised a PERIOD_1 TD
bridge back into HB, and the persisted validation artifacts are not yet compact
enough for an overnight map. Therefore the correct gate is:

```text
NOT_READY_FOR_OVERNIGHT_MAP
```

Blocking reason: the two validation columns did not exercise TD-to-HB restart,
and full per-point production artifacts make storage heavier than the requested
compact policy. No new physics investigation is indicated.
