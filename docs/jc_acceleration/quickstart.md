# Quickstart

This page shows the shortest supported path for running the benchmark suite with batch mode and direct zero-pump linear backends.

## Requirements

- Windows PowerShell.
- Local repositories at `D:\Projects\Thesis\JosephsonCircuits.jl`, `D:\Projects\Thesis\Harmonia.jl`, and `D:\Projects\Thesis\twpa_jax`.
- Julia available on `PATH`.
- `twpa_jax` Python environment with project test dependencies installed.

## Run Direct Linear Benchmark Suite

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

Expected output artifacts:

- `benchmark_summary.json`
- `benchmark_runs.csv`
- `benchmark_cases.json`
- `environment.json`
- per-run `status.json`, `stdout.log`, `stderr.log`, and `config_resolved.json`
- batch runner logs under `runs\_julia_batch_runner`

## Run Focused Tests

```powershell
cd D:\Projects\Thesis\twpa_jax

$baseTemp = "D:\Projects\Thesis\outputs\pytest_tmp_docs_check"
Remove-Item $baseTemp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $baseTemp | Out-Null

python -m pytest `
  tests\test_harmonia_benchmark_suite.py `
  tests\test_harmonia_benchmark_suite_batch_runner.py `
  tests\test_harmonia_benchmark_suite_jtl_hblinsolve_direct.py `
  tests\test_harmonia_benchmark_suite_jtl_rf_hblinsolve_direct.py `
  tests\test_harmonia_benchmark_suite_jtl_rf_ethz_hblinsolve_direct.py `
  -q -p no:cacheprovider `
  --basetemp $baseTemp
```

## When To Use This

Use this workflow for zero-pump linear response, topology validation, direct S-parameter checks, benchmark reporting, and dataset sanity checks.

## When Not To Use This

Do not use direct `hblinsolve` for pumped nonlinear gain workflows. Use full `hbsolve` for nonlinear HB, gain maps, pump sweeps, and cases with nonzero pump current.

## Troubleshooting

If the direct RF-JTL path reports `RF-JTL linear S-parameters contain non-finite values`, keep the failure. It is a real numerical boundary and should be diagnosed by frequency-window and parameter sensitivity maps.

If timings look inconsistent, separate cold start from warm run. First repetitions may include Julia compilation and setup effects. Batch mode reduces process startup overhead but does not change solver mathematics.
