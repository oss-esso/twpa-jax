# Batch Runner

The batch runner reuses one Julia process across multiple simulation requests. This reduces repeated Julia startup and package load overhead in campaigns and benchmark suites.

## Components

| Layer | File | Role |
|---|---|---|
| Python | `twpa_jax\twpa\io\julia_batch_runner.py` | Builds manifest, launches Julia batch process, reads batch summary. |
| Julia | `Harmonia.jl\scripts\run_simulation_batch.jl` | Reads manifest and executes simulation items inside one process. |
| Benchmark CLI | `twpa_jax\scripts\run_harmonia_benchmark_suite.py` | Adds `--use-batch-runner`. |
| Campaign CLI | `twpa_jax\scripts\run_harmonia_jtl_linear_campaign.py` | Adds campaign-level `--use-batch-runner`. |

## Telemetry

Batch results expose `cache_telemetry`, including:

- `julia_process_reused`
- `batch_run_index`
- `setup_cache_integration`
- `hbcompiled_circuit_base_enabled`
- `hbnumeric_matrix_cache_enabled`

Current batch process reuse is wired. Full internal JC setup-cache reuse is not silently enabled in ordinary simulation execution.

## Evidence

- `twpa_jax` commit `7054f09`: Python batch runner wrapper.
- `twpa_jax` commit `46558d6`: batch cache telemetry.
- `Harmonia.jl` commit `259bde0`: Julia batch runner boundary.
- `tests\test_julia_batch_runner.py`
- `tests\test_julia_batch_runner_cache_telemetry.py`
- `tests\test_campaign_batch_runner.py`

## When To Use

Use batch mode for repeated campaign or benchmark runs where Julia process startup dominates part of the workflow cost.

## When Not To Use

Batch mode does not change solver semantics. If a single simulation is numerically invalid, batch mode will not make it valid. It should not be described as a solver acceleration.
