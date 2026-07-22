# Old-IPM `hbsolve` Callsite Audit

Canonical script:

```text
Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map_gridn.jl
```

Canonical topology/design function:

```julia
build_old_ipm_circuit()
```

Backend entry function:

```julia
run_case(; pump_frequency_ghz, external_power_dbm, power_offset_db, iterations)
```

The JosephsonCircuits callsite is inside `run_case`, around the block beginning at the `hbsolve(...)` call in the canonical runner.

## Inputs To `hbsolve`

The call uses:

```julia
wp = (2 * pi * pump_frequency_ghz * 1e9,)
sources = [(mode=(1,), port=4, current=pump_current_a)]
Npumpharmonics = (10,)
Nmodulationharmonics = (5,)
circuit, circuitdefs, metadata = build_old_ipm_circuit()
```

Pump convention:

```julia
source_power_dbm = external_power_dbm - power_offset_db
pump_current_a = sqrt(2 * source_power_W / 50)
```

Call:

```julia
hbsolve(
    wp,
    wp,
    sources,
    Nmodulationharmonics,
    Npumpharmonics,
    circuit,
    circuitdefs;
    dc = false,
    iterations = iterations,
)
```

## Returned Fields Used Downstream

The returned object is assigned to `rpm`. The map runner uses:

```julia
s21_linear = rpm.linearized.S((0,), 2, (0,), 1, :)
gain_db = 10 .* log10.(abs2.(s21_linear))
gain_db_max = maximum(gain_db)
```

The solver log captured around `hbsolve` is parsed by:

```julia
parsed_log = classify_from_log_text(solver_log_text)
```

The status is then produced by:

```julia
classify_hb_row(...)
```

Inputs include finite S-parameters, finite gain values, parsed residual norm, parsed infinity norm, and whether solver warnings were seen.

## Backend Substitution Point

Only this callsite should be substituted. The recovery runner:

```text
Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map_backend_compare.jl
```

keeps `build_old_ipm_circuit()`, the map grid, source convention, row schema, and artifact writers. It switches only between:

- `run_case(...)` for `--backend josephsoncircuits`;
- `solve_old_ipm_backend_point` for independent backends using the exact exported circuit JSON.
