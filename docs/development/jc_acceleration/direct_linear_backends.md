# Direct Linear Backends

Direct linear backends add opt-in `hblinsolve` execution for selected zero-pump linear-response workflows.

## Solver Paths

Old path:

```julia
hbsolve(...).linearized.S(...)
```

New opt-in path:

```julia
hblinsolve(...)
```

The old path remains the default.

## Configuration

```json
{
  "solver": {
    "jtl_linear_backend": "hblinsolve_direct",
    "rf_jtl_linear_backend": "hblinsolve_direct",
    "ethz_jtl_linear_backend": "hblinsolve_direct",
    "enable_jc_setup_cache": false
  }
}
```

Supported keys:

| Family | Config key | Direct value | Default |
|---|---|---|---|
| JTL linear | `solver.jtl_linear_backend` | `hblinsolve_direct` | `hbsolve` |
| RF-JTL linear | `solver.rf_jtl_linear_backend` | `hblinsolve_direct` | `hbsolve` |
| ETHZ-JTL linear | `solver.ethz_jtl_linear_backend` | `hblinsolve_direct` | `hbsolve` |

## Guards

- Direct backend is opt-in.
- Direct backend requires `pump_current_a == 0.0`.
- Pumped and nonlinear workflows still require full `hbsolve`.
- `enable_jc_setup_cache` defaults to `false`.
- If setup cache is requested in these direct paths, telemetry records that it is requested but not wired.

The zero-pump guard is implemented in `Harmonia.jl\scripts\run_simulation.jl` for JTL, RF-JTL, and ETHZ-JTL direct branches.

## Status Telemetry

Direct runs report:

- `jc_backend = "Harmonia.CircuitIR + JosephsonCircuits.hblinsolve"`
- `cache_telemetry.setup_cache_integration = "jtl_hblinsolve_direct"`, `"rf_jtl_hblinsolve_direct"`, or `"ethz_jtl_hblinsolve_direct"`

The benchmark suite records these fields in per-run `status.json` and summary CSV/JSON outputs.

## Limitations

Direct `hblinsolve` is not a universal speedup. In warm large-geometry comparisons, it is close to the old `hbsolve(...).linearized.S(...)` path. Its confirmed value is exact-equivalent solver semantics for tested zero-pump cases, explicit telemetry, and direct-only large linear-response workflows.

Do not extend this to `lumped_jpa_linear` without a separate equivalence probe.
