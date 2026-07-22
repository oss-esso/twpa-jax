# Benchmark Suite

`twpa_jax\scripts\run_harmonia_benchmark_suite.py` coordinates benchmark cases across Harmonia and JosephsonCircuits.

## Cases

The recent suite covers linear and nonlinear smoke cases including:

- `jtl_linear`
- `rf_jtl_linear`
- `ethz_jtl_linear`
- `lumped_jpa_linear`
- `tiny_nonlinear_hb`

Direct backend flags apply only to JTL, RF-JTL, and ETHZ-JTL linear cases.

## Batch Direct Backend Run

```powershell
cd D:\Projects\Thesis\twpa_jax

python scripts\run_harmonia_benchmark_suite.py `
  --repetitions 2 `
  --benchmark-dir D:\Projects\Thesis\outputs\benchmarks\example_direct_linear `
  --use-batch-runner `
  --jtl-linear-backend hblinsolve_direct `
  --rf-jtl-linear-backend hblinsolve_direct `
  --ethz-jtl-linear-backend hblinsolve_direct `
  --force
```

## Measured Batch Report

`outputs\benchmarks\jc3m_m7_direct_linear_backend_report\direct_linear_backend_report.md` records a two-repetition comparison of batch modes. All listed benchmark cases passed. Selected means:

| Mode | Case | Runs | Python mean |
|---|---|---:|---:|
| `batch_hbsolve` | `jtl_linear` | 2 | 3.630 s |
| `batch_jtl_rf_ethz_direct` | `jtl_linear` | 2 | 3.031 s |
| `batch_hbsolve` | `rf_jtl_linear` | 2 | 1.199 s |
| `batch_jtl_rf_ethz_direct` | `rf_jtl_linear` | 2 | 1.145 s |
| `batch_hbsolve` | `ethz_jtl_linear` | 2 | 0.564 s |
| `batch_jtl_rf_ethz_direct` | `ethz_jtl_linear` | 2 | 0.415 s |

This is a small benchmark suite result. Treat it as workflow evidence, not a statistically complete performance study.

## Output Files

- `benchmark_summary.json`: aggregate case statistics and selected backend flags.
- `benchmark_runs.csv`: per-case per-repetition status and time.
- `benchmark_cases.json`: resolved benchmark case definitions.
- `environment.json`: host and tool metadata where captured.
- `runs\*\status.json`: Harmonia/JosephsonCircuits status and solver telemetry.

## Tests

Focused tests:

- `tests\test_harmonia_benchmark_suite.py`
- `tests\test_harmonia_benchmark_suite_batch_runner.py`
- `tests\test_harmonia_benchmark_suite_jtl_hblinsolve_direct.py`
- `tests\test_harmonia_benchmark_suite_jtl_rf_hblinsolve_direct.py`
- `tests\test_harmonia_benchmark_suite_jtl_rf_ethz_hblinsolve_direct.py`
