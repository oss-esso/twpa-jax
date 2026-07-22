# Evidence Manifest

This page lists the evidence used for these docs.

## Git History

Collected with `git status --short`, `git log --oneline --decorate -100`, `git log --stat --decorate -50`, `git diff --stat`, and `git diff` in all three repositories.

Key recent commits:

- `JosephsonCircuits.jl`: `15b4e82`, `c3f47b8`, `ae89e5b`, `f560af9`, `ff77c6b`, `b0b0d89`, `9d1caee`, `1997df3`, `810c5bb`, `024c905`, `a40b152`.
- `Harmonia.jl`: `259bde0`, `b365874`, `a9afbde`, `f112555`, `28098a1`, `ca299ec`, `60168e9`, `ea45d25`, `631fa39`.
- `twpa_jax`: `7054f09`, `c9837f6`, `aa29db6`, `46558d6`, `eaac53a`, `fdec801`, `1032d6c`, `5bb0f9d`, `28c8d0d`, `f9ab075`.

## Source Files

- `JosephsonCircuits.jl\experiments\thesis_gpu_parallel\run_hbsolve_profile.jl`
- `JosephsonCircuits.jl\experiments\thesis_gpu_parallel\analyze_five_block_timing.jl`
- `JosephsonCircuits.jl\experiments\thesis_gpu_parallel\reports\jc3_decision_note.md`
- `Harmonia.jl\scripts\run_simulation.jl`
- `Harmonia.jl\scripts\run_simulation_batch.jl`
- `Harmonia.jl\experiments\jc_setup_cache\probe_jtl_hbsolve_vs_hblinsolve.jl`
- `Harmonia.jl\experiments\jc_setup_cache\probe_rf_jtl_hbsolve_vs_hblinsolve.jl`
- `Harmonia.jl\experiments\jc_setup_cache\probe_ethz_jtl_hbsolve_vs_hblinsolve.jl`
- `Harmonia.jl\experiments\jc_setup_cache\run_scaled_direct_linear_showcase.jl`
- `Harmonia.jl\experiments\jc_setup_cache\run_extreme_direct_linear_showcase.jl`
- `twpa_jax\scripts\run_harmonia_benchmark_suite.py`
- `twpa_jax\twpa\io\julia_batch_runner.py`

## Tests

- `twpa_jax\tests\test_julia_batch_runner.py`
- `twpa_jax\tests\test_julia_batch_runner_cache_telemetry.py`
- `twpa_jax\tests\test_campaign_batch_runner.py`
- `twpa_jax\tests\test_harmonia_benchmark_suite.py`
- `twpa_jax\tests\test_harmonia_benchmark_suite_batch_runner.py`
- `twpa_jax\tests\test_harmonia_benchmark_suite_jtl_hblinsolve_direct.py`
- `twpa_jax\tests\test_harmonia_benchmark_suite_jtl_rf_hblinsolve_direct.py`
- `twpa_jax\tests\test_harmonia_benchmark_suite_jtl_rf_ethz_hblinsolve_direct.py`

## Output Artifacts

- `outputs\jc_profiles\jc3m_m1_jtl_hbsolve_vs_hblinsolve\jtl_hbsolve_vs_hblinsolve_report.json`
- `outputs\jc_profiles\jc3m_m5_rf_jtl_hbsolve_vs_hblinsolve\rf_jtl_hbsolve_vs_hblinsolve_report.json`
- `outputs\jc_profiles\jc3m_m6_ethz_jtl_hbsolve_vs_hblinsolve\ethz_jtl_hbsolve_vs_hblinsolve_report.json`
- `outputs\jc_profiles\jc3m_scaled_direct_linear_showcase\scaled_direct_linear_showcase.csv`
- `outputs\jc_profiles\jc3m_extreme_direct_linear_showcase\extreme_direct_linear_showcase.csv`
- `outputs\benchmarks\jc3m_m7_direct_linear_backend_report\direct_linear_backend_report.md`
- `outputs\benchmarks\jc3m_m7_direct_linear_backend_report\direct_linear_backend_report.json`
- `outputs\benchmarks\jc3m_m7_direct_linear_backend_report\direct_linear_backend_summary.csv`

## Missing Or Restricted

`rg` reported access denied for several temporary pytest directories under `outputs\pytest_tmp_jc3m_*`. These directories were not used as evidence.
