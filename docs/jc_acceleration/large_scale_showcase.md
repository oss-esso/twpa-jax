# Large Scale Showcase

The showcase experiments exercise direct-only linear-response runs beyond the largest old-vs-direct comparisons. These runs are evidence of scale and numerical boundaries, not proof of nonlinear HB acceleration.

## Direct-Only Results

From `outputs\jc_profiles\jc3m_extreme_direct_linear_showcase\extreme_direct_linear_showcase.csv`:

| Family | Cells | Elements | Nodes | JC tuples | Warm/direct time | Status |
|---|---:|---:|---:|---:|---:|---|
| JTL | 10000 | 20004 | 10002 | 30004 | 2.67 s | PASS |
| JTL | 20000 | 40004 | 20002 | 60004 | 9.01 s | PASS |
| JTL | 30000 | 60004 | 30002 | 90004 | 40.23 s | PASS |
| RF-JTL | 5000 | 20004 | 10002 | 25004 | 12.00 s | PASS |
| ETHZ-JTL | 5000 | 16247 | 8123 | 21246 | 1.37 s | PASS |
| ETHZ-JTL | 10000 | 32497 | 16248 | 42496 | 6.48 s | PASS |

The table uses repetition 1 where available to avoid reporting first-run cold setup as the primary result.

## Failure Boundary

RF-JTL at 10000 cells failed in both repetitions:

```text
RF-JTL linear S-parameters contain non-finite values
```

The failure is recorded in the same CSV. It should be treated as a numerical validity boundary that needs diagnosis, not as a hidden benchmark failure.

## Reproduce

Run from `D:\Projects\Thesis\Harmonia.jl`:

```powershell
julia --project=. experiments\jc_setup_cache\run_extreme_direct_linear_showcase.jl
```

Expected output directory:

```text
D:\Projects\Thesis\outputs\jc_profiles\jc3m_extreme_direct_linear_showcase
```

## Interpretation

These runs support large linear-response workflows for topology and S-parameter checks. They do not cover nonlinear pump/gain behavior and should not be used as evidence for GPU readiness or nonlinear HB speedup.
