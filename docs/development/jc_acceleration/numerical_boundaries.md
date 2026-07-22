# Numerical Boundaries

The most important recorded numerical boundary is RF-JTL direct linear response at 10000 cells.

## Recorded Failure

In `outputs\jc_profiles\jc3m_extreme_direct_linear_showcase\extreme_direct_linear_showcase.csv`, both RF-JTL 10000-cell direct runs failed with:

```text
RF-JTL linear S-parameters contain non-finite values
```

The failing rows did not report elements, nodes, tuple counts, or timing because Harmonia rejected the non-finite S-parameter result.

## Interpretation

This is a numerical validity issue. It may reflect conditioning, frequency-window behavior, parameter sensitivity, impedance assumptions, or a topology-specific scaling issue. It is not evidence that direct `hblinsolve` is invalid for all RF-JTL cases; RF-JTL passed exact old-vs-direct comparisons up to 2393 cells and direct-only 5000-cell runs.

## Next Diagnosis

1. Map finite/non-finite status by RF-JTL cell count.
2. Sweep frequency windows.
3. Sweep `Lrf_H`, `Lp_H`, and port impedance.
4. Compare direct and old paths where old path remains tractable.
5. Record finite status, condition-related telemetry if available, and S-parameter ranges.

## Warning

Do not hide non-finite S-parameters by filtering them out of reports. Keep the failure row and attach the configuration that produced it.
