# IPM TWPA 35x35 All-Backends Report

## Entrypoint

The canonical user-facing entrypoint is:

```powershell
cd D:\Projects\Thesis\Harmonia.jl

julia --project=. experiments\jc_setup_cache\run_ipm_twpa_35x35_all_backends.jl `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\ipm_twpa_35x35_all_backends `
  --points 35 `
  --pump-freq-min-ghz 6.0 `
  --pump-freq-max-ghz 8.0 `
  --pump-power-min-dbm -28 `
  --pump-power-max-dbm -19 `
  --power-offset-db 32 `
  --backends josephsoncircuits,scipy-least-squares,scipy-root,scipy-newton-krylov,jax-dense-newton,jax-newton-krylov,pseudo-transient `
  --overwrite true
```

The script uses the Harmonia old-IPM frontend and calls `build_old_ipm_circuit()` from `run_report_old_ipm_power_map_gridn.jl`.  No reduced Python surrogate topology is used.

## Circuit Spec

- Circuit source: `build_old_ipm_circuit()`
- `Nj = 418 * 6 = 2508`
- Exported/imported model observed in smoke:
  - node count: `3134`
  - element count: `8788`
  - Josephson branch count: `2507`
  - mutual coupling count: `4`
- Ports:
  - input port: `1`
  - output port: `2`
  - pump/source port: `4`
- Pump convention:
  - `source_power_dbm = external_power_dbm - 32`
  - `pump_current_a = sqrt(2 * source_power_W / 50)`
- Harmonics:
  - JosephsonCircuits path: `Npumpharmonics = (10,)`, `Nmodulationharmonics = (5,)`
  - independent backend point adapter currently requests those harmonics and uses an effective one-pump-harmonic sparse AFT residual for the exact full old-IPM netlist.

## Backends

The script accepts and routes:

- `josephsoncircuits`
- `scipy-least-squares`
- `scipy-root`
- `scipy-newton-krylov`
- `jax-dense-newton`
- `jax-newton-krylov`
- `pseudo-transient`

For `josephsoncircuits`, the script calls the original `hbsolve(...)` path through `run_case(...)`.

For independent backends, the script exports the exact Julia old-IPM circuit and calls `twpa_solver.experiments.solve_old_ipm_backend_point`.  Placeholder statuses are forbidden; every backend name now performs a numeric attempt on the exact imported residual.  Only `scipy-least-squares` is a native full-size backend at this point; the other independent backend names are tagged in metadata as compatibility attempts using the shared exact old-IPM sparse AFT core.

## Output Layout

Each backend writes:

- `rows.csv`
- `gain_db_grid.csv`
- `convergence_mask_grid.csv`
- `finite_mask_grid.csv`
- `solver_warning_mask_grid.csv`
- `residual_norm_grid.csv`
- `infinity_norm_grid.csv`
- `point_runtime_grid.csv`
- `status_grid.csv`
- `map_timing.json`
- `report.md`
- `plots/gain_marked.png`
- `plots/gain_converged_only.png`
- `plots/residual_norm.png`
- `plots/infinity_norm.png`
- `plots/point_runtime.png`
- `plots/status_counts.png`

The root writes:

- `config.json`
- `run_summary.md`
- `all_backend_summary.csv`
- `all_backend_timing.csv`
- `comparison/backend_comparison_rows.csv`
- `comparison/backend_comparison_summary.md`
- `comparison/gain_difference_vs_jc_<backend>.csv`
- `comparison/status_difference_vs_jc_<backend>.csv`
- `comparison/timing_comparison.csv`
- comparison plots.

Rows are also appended incrementally to `rows.jsonl` per backend for resume support.

## Smoke Run Evidence

Smoke command that completed:

```powershell
cd D:\Projects\Thesis\Harmonia.jl

& 'C:\Users\Edoardo\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe' --project=. `
  experiments\jc_setup_cache\run_ipm_twpa_35x35_all_backends.jl `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\ipm_twpa_35x35_smoke_1x1 `
  --points 1 `
  --backends josephsoncircuits,scipy-least-squares
```

Artifacts:

- `D:\Projects\Thesis\outputs\new_twpa_solver\ipm_twpa_35x35_smoke_1x1`

Status summary:

| backend | points | valid converged | gain dB | median point runtime |
|---|---:|---:|---:|---:|
| `josephsoncircuits` | 1 | 1 | `8.260911886984063` | `3.7289998531341553` |
| `scipy-least-squares` | 1 | 1 | `-7.515974127150921` | `338.8785024999961` |

The gain difference at the first point is `-15.776886014134984 dB` for SciPy least-squares relative to JosephsonCircuits.  This means the independent backend is converging its current pump residual, but its conversion/gain path is not yet numerically equivalent to JosephsonCircuits.

## Full 35x35 Status

The full 35x35 all-backend command was not completed in this pass.  The exact old-IPM SciPy least-squares point currently takes about `339 s` for one point on this machine.  A 35x35 map for that backend alone is therefore approximately:

```text
1225 * 339 s = 415275 s = 115.35 h
```

Running all seven backends with the current independent backend compatibility core would exceed practical interactive runtime.  The script is implemented and resumable, but the independent backend needs a faster factorization/warm-start path before a complete all-backend 35x35 run is operationally reasonable.

## Verification

Python tests:

```powershell
cd D:\Projects\Thesis\twpa_jax
$files = Get-ChildItem tests -Filter 'test_*backend*.py' | ForEach-Object { $_.FullName }
$files += Get-ChildItem tests -Filter 'test_*old_ipm*.py' | ForEach-Object { $_.FullName }
$files = $files | Sort-Object -Unique
python -m pytest $files -q -p no:cacheprovider
```

Result:

```text
22 passed
```

Julia entrypoint syntax check:

```powershell
julia --project=. experiments\jc_setup_cache\run_ipm_twpa_35x35_all_backends.jl --help
```

Result: help text printed successfully.  For long-running commands, the WindowsApps `julia` alias intermittently failed to launch; the concrete Julia binary at `C:\Users\Edoardo\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe` worked.

## Current Numerical Limitation

The core remaining limitation is not topology import or workflow wiring.  The script uses the exact old-IPM netlist and the canonical frontend.  The blocker is numerical equivalence and runtime of the independent backend:

- SciPy least-squares converges the exact old-IPM pump residual at the first point.
- The independent gain is not yet equal to the JosephsonCircuits gain at that point.
- The sparse Newton path is still too slow for a complete 1225-point backend map.
- Non-least-squares backend names currently perform numeric exact-residual compatibility attempts through the shared sparse core, not distinct production implementations.

## Next Step

Before launching the full 35x35 all-backend run, optimize the exact old-IPM independent backend point solve:

1. Reuse symbolic sparse factorization structure across continuation stages and neighboring map points.
2. Add true point-to-point warm-start transfer through the Julia runner.
3. Validate the conversion matrix/gain extraction against JosephsonCircuits on the first converged point.
4. Only then run the resumable 35x35 map.
