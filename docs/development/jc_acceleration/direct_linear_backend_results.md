# Direct Linear Backend Results

This page preserves the measured result summary from the direct linear backend phase. See [Equivalence Probes](equivalence_probes.md) and [Large Scale Showcase](large_scale_showcase.md) for the full evidence trail.

## Correctness

For tested zero-pump comparisons, S-parameters from the old path and direct path matched exactly:

```text
max_abs_diff_vs_hbsolve = 0.0
```

Largest exact old-vs-direct comparisons from `scaled_direct_linear_showcase.csv`:

| Family | Cells | Elements | Nodes | JC tuples |
|---|---:|---:|---:|---:|
| JTL | 3000 | 6004 | 3002 | 9004 |
| RF-JTL | 2393 | 9576 | 4788 | 11969 |
| ETHZ-JTL | 2048 | 6653 | 3326 | 8700 |

## Direct-Only Scale

| Family | Cells | Elements | Nodes | JC tuples | Time |
|---|---:|---:|---:|---:|---:|
| JTL | 10000 | 20004 | 10002 | 30004 | 2.67 s |
| JTL | 20000 | 40004 | 20002 | 60004 | 9.01 s |
| JTL | 30000 | 60004 | 30002 | 90004 | 40.23 s |
| RF-JTL | 5000 | 20004 | 10002 | 25004 | 12.00 s |
| ETHZ-JTL | 5000 | 16247 | 8123 | 21246 | 1.37 s |
| ETHZ-JTL | 10000 | 32497 | 16248 | 42496 | 6.48 s |

## Boundary

RF-JTL 10000 cells failed with:

```text
RF-JTL linear S-parameters contain non-finite values
```

This is a recorded numerical boundary.

## Interpretation

Direct backends clean up solver semantics, expose telemetry, and support large linear-response runs. They are not a universal warm large-geometry speedup and do not accelerate nonlinear pumped HB.
