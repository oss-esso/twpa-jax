# API Reference

This reference covers orchestration flags and configuration keys added for the acceleration work.

## Benchmark Suite CLI

```powershell
python scripts\run_harmonia_benchmark_suite.py `
  --repetitions 2 `
  --benchmark-dir D:\Projects\Thesis\outputs\benchmarks\example `
  --use-batch-runner `
  --jtl-linear-backend hblinsolve_direct `
  --rf-jtl-linear-backend hblinsolve_direct `
  --ethz-jtl-linear-backend hblinsolve_direct `
  --force
```

| Flag | Values | Default | Meaning |
|---|---|---|---|
| `--use-batch-runner` | present/absent | absent | Reuse one Julia process for benchmark cases. |
| `--jtl-linear-backend` | `hbsolve`, `hblinsolve_direct` | `hbsolve` | Backend for `jtl_linear`. |
| `--rf-jtl-linear-backend` | `hbsolve`, `hblinsolve_direct` | `hbsolve` | Backend for `rf_jtl_linear`. |
| `--ethz-jtl-linear-backend` | `hbsolve`, `hblinsolve_direct` | `hbsolve` | Backend for `ethz_jtl_linear`. |
| `--force` | present/absent | absent | Overwrite an existing benchmark directory. |

## Harmonia Solver Configuration

```json
{
  "solver": {
    "jtl_linear_backend": "hbsolve",
    "rf_jtl_linear_backend": "hbsolve",
    "ethz_jtl_linear_backend": "hbsolve",
    "enable_jc_setup_cache": false
  }
}
```

Set one or more backend keys to `hblinsolve_direct` for zero-pump linear response.

## Python Batch Runner

Primary implementation:

```text
twpa\io\julia_batch_runner.py
```

The batch runner returns per-run status and optional `cache_telemetry`. Tests cover manifest creation, status parsing, and cache telemetry exposure.

## Julia Batch Runner

Primary implementation:

```text
Harmonia.jl\scripts\run_simulation_batch.jl
```

Expected usage:

```powershell
julia --project=. scripts\run_simulation_batch.jl --manifest <manifest.json> --summary <batch_summary.json>
```

Normally this is launched by Python rather than run manually.
