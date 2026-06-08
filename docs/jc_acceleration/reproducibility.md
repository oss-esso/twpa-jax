# Reproducibility

This page lists commands to reproduce the main evidence.

## Repository State Checks

```powershell
cd D:\Projects\Thesis\JosephsonCircuits.jl
git status --short
git log --oneline --decorate -100
git log --stat --decorate -50
git diff --stat
git diff

cd D:\Projects\Thesis\Harmonia.jl
git status --short
git log --oneline --decorate -100
git log --stat --decorate -50
git diff --stat
git diff

cd D:\Projects\Thesis\twpa_jax
git status --short
git log --oneline --decorate -100
git log --stat --decorate -50
git diff --stat
git diff
```

## Benchmark Suite With Direct Backends

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

## Focused Pytest Check

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

## Harmonia Equivalence Probes

```powershell
cd D:\Projects\Thesis\Harmonia.jl

julia --project=. experiments\jc_setup_cache\probe_jtl_hbsolve_vs_hblinsolve.jl
julia --project=. experiments\jc_setup_cache\probe_rf_jtl_hbsolve_vs_hblinsolve.jl
julia --project=. experiments\jc_setup_cache\probe_ethz_jtl_hbsolve_vs_hblinsolve.jl
```

Expected report paths:

- `D:\Projects\Thesis\outputs\jc_profiles\jc3m_m1_jtl_hbsolve_vs_hblinsolve\jtl_hbsolve_vs_hblinsolve_report.json`
- `D:\Projects\Thesis\outputs\jc_profiles\jc3m_m5_rf_jtl_hbsolve_vs_hblinsolve\rf_jtl_hbsolve_vs_hblinsolve_report.json`
- `D:\Projects\Thesis\outputs\jc_profiles\jc3m_m6_ethz_jtl_hbsolve_vs_hblinsolve\ethz_jtl_hbsolve_vs_hblinsolve_report.json`

## Large Direct Showcase

```powershell
cd D:\Projects\Thesis\Harmonia.jl

julia --project=. experiments\jc_setup_cache\run_scaled_direct_linear_showcase.jl
julia --project=. experiments\jc_setup_cache\run_extreme_direct_linear_showcase.jl
```

Expected report paths:

- `D:\Projects\Thesis\outputs\jc_profiles\jc3m_scaled_direct_linear_showcase\scaled_direct_linear_showcase.csv`
- `D:\Projects\Thesis\outputs\jc_profiles\jc3m_extreme_direct_linear_showcase\extreme_direct_linear_showcase.csv`

## Missing Or Restricted Artifacts

During this documentation pass, `rg` reported access denied for several `outputs\pytest_tmp_jc3m_*` directories. These are temporary pytest directories and were not used as evidence. No expected benchmark JSON/CSV artifact listed above was missing.
