# 7.9 GHz physical operating-boundary search

Status: `UNCONFIRMED`

This report uses the operational definition requested for the boundary: the
lowest drive at which the standardized upward-ramped full-IPM trajectory no
longer returns to the desired pump-period-1 state.  No claim is made here about
the detailed non-periodic attractor.

## Results

| drive | ramp/hold | final d1 | best low-order dN | trend | r_J | phase winding | decision |
|---:|---|---:|---:|---|---:|---:|---|
| 11.40 uA | 10/10 periods, corrected production HB start | 4.91e-4 | d1 (period-1 floor) | numerical-floor level | 0.799 | -1.82e-6 cycles | `WORKING_PERIOD1` |
| 11.60 uA | existing 70--120 record plus bounded continuation to period 190 and final 190--200 chunk | 0.171 in final chunk; prior late plateau 3.0--3.4e-3 | none for d2--d8 | no sustained decay | 0.881 (final chunk) | 5.50e-6 cycles | `OUTSIDE_PERIOD1` |

At 11.6 uA the saved 70--120 stroboscopic record was also checked directly for
d1 through d8.  In its last eight samples the medians were:

```text
d1  3.3876e-3   d2  6.7135e-3   d3  9.9444e-3   d4  1.3058e-2
d5  1.6035e-2   d6  1.8851e-2   d7  2.1482e-2   d8  2.3907e-2
```

None approached the validated periodic floor.  The late d1 trend over the last
20 saved samples was positive (`b = 7.61e-3` per period in log-distance), and
the final bounded continuation remained numerically healthy.  The 16 uA job was
not used for the boundary decision; its already-running checkpoint was stopped
once 11.6 uA was established as the first non-working candidate.

## Boundary

The result below is retained as an operational TD observation, but the physical
boundary claim is downgraded to `UNCONFIRMED`. The requested
`delta_theta = 0.01`, 10-period-ramp floor run terminated after 100 periods
without a final summary, and the available floor evidence is only
`3.0e-3 / 4.25e-4 = 7.1`, below the required tenfold margin. Phase 2 is the
sole source of the boundary until this protocol is completed successfully.

The operational current bracket is:

```text
I_PHYSICAL_WORKING_MAX >= 11.40 uA
I_FIRST_NONPERIOD1 <= 11.60 uA
```

Thus the useful 7.9-GHz operating boundary is bracketed by

```text
11.40 uA <= I_boundary(7.9 GHz) <= 11.60 uA
```

Using the accepted reference calibration (`11.299598687 uA = -19.6842 dBm`),
the corresponding source-power bracket is approximately:

```text
-19.607 dBm <= P_boundary(7.9 GHz) <= -19.456 dBm.
```

The validated HB fold/accessibility point is approximately 11.30 uA
(`-19.684 dBm`), so the ramp-selected physical working boundary lies above the
connected-fold point but below 11.6 uA.  The distinction is therefore:

```text
P_HB_ACCESS / fold       ~= 11.30 uA
P_PHYSICAL_WORKING       >= 11.40 uA
P_NONPERIOD1             <= 11.60 uA
```

No additional mechanism study, Floquet calculation, RCSJ model, frequency
campaign, or pump map was run.
