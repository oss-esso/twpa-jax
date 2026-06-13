# SciPy Least-Squares Old-IPM Backend Milestone

This milestone uses the existing Julia backend-compare runner:

```text
Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map_backend_compare.jl
```

No reduced Python surrogate topology is used. The Python backend consumes the exact JSON exported from `build_old_ipm_circuit()`.

## One-Point Attempt

Command:

```powershell
cd D:\Projects\Thesis\Harmonia.jl
julia --project=. experiments\jc_setup_cache\run_report_old_ipm_power_map_backend_compare.jl `
  --backend scipy-least-squares `
  --points 1 `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\backend_compare_one_point\scipy_least_squares
```

Point:

```text
pump_frequency_ghz = 6.0
external_power_dbm = -28.0
source_power_dbm = -60.0
pump_current_a = 6.324555320336759e-6
```

Result:

```text
status = FAILED_MAX_NFEV
residual_norm = 6.32455532033676
infinity_norm = 6.32455532033676
runtime_s = 0.4440234999929089
gain_db_max = missing
```

The backend evaluated the full exact old-IPM pump residual with:

```text
node_count = 3134
element_count = 8788
josephson_junction_count = 2507
mutual_coupling_count = 4
requested_pump_harmonics = 10
effective_pump_harmonics = 1
residual_size = 6268
```

The harmonic downgrade is explicit in metadata. It is required for this first bounded attempt because a full finite-difference least-squares Jacobian at 10 harmonics is not yet practical.

Per-point artifacts:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\backend_compare_one_point\scipy_least_squares\points\fp6p000000_Pextm28p000000
```

Includes:

- `result.json`
- `residual_history.csv`
- `residual_block_norms.csv`
- `pump_solution_coefficients.npz`
- `runtime_summary.json`

## 5x5 Attempt

Command:

```powershell
cd D:\Projects\Thesis\Harmonia.jl
julia --project=. experiments\jc_setup_cache\run_report_old_ipm_power_map_backend_compare.jl `
  --backend scipy-least-squares `
  --points 5 `
  --reference-root D:\Projects\Thesis\outputs\jc_profiles\jc3m_report_old_ipm_power_map_5x5_marked `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\backend_compare_5x5\scipy_least_squares_real
```

Result:

```text
FAILED_MAX_NFEV = 25 cells
mean residual_norm = 11.3438206820157
max residual_norm = 17.8250187626749
mean runtime_s = 0.419890847999486
max runtime_s = 0.468996399999014
```

Comparison artifact:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\backend_compare_5x5\comparison_scipy_real
```

Classification:

```text
scipy-least-squares-real = FAILED_NUMERICALLY
```

## Conversion Gain

Conversion gain was not computed because no pump point converged. The implementation attempts conversion only after `VALID_CONVERGED` pump status. This avoids reporting finite gain from a failed pump solve.

## Current Backend Status

`scipy-least-squares` is now a real exact-netlist backend attempt, not a placeholder. It is not yet a useful production solver for the full old-IPM map because it stops at the bounded first evaluation with `FAILED_MAX_NFEV`.

## Next Step

Replace the placeholder zero Jacobian with a matrix-free or sparse analytic/JVP-based linearization strategy, or reduce the solve variable space through a physically justified pump-port/line-mode continuation. The next milestone should be one exact old-IPM point with residual reduction, not merely residual evaluation.
