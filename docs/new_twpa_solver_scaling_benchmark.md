# Scaling Benchmark

Benchmark artifact:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\scaling_benchmark_20260613_075629
```

Files:

- `scaling_rows.csv`
- `scaling_summary.md`
- `plots/runtime_vs_cells.png`
- `plots/success_rate_vs_cells.png`
- `plots/residual_vs_cells.png`

The benchmark ran both:

- `ipm_jtwpa_reduced_marker`
- `ipm_jtwpa_physical_coupler`

and tested `cells_per_line` 4, 8, 16, 32; pump harmonics 1, 3, 5;
sidebands 1, 3; SciPy least-squares for all rows; JAX dense Newton and
preconditioned JAX Newton-Krylov for tiny rows.

Key feasibility row:

| topology | cells | harmonics | sidebands | solver | status | runtime |
|---|---:|---:|---:|---|---|---:|
| `ipm_jtwpa_physical_coupler` | 32 | 5 | 3 | SciPy least-squares | converged | about 0.60 s |

This supported attempting the full physical-coupler 25x25 map with
`cells_per_line=32`, `pump_harmonics=5`, and `sidebands=3`.

The benchmark is a one-cell feasibility screen, not a production sweep. The
25x25 attempt below is the map-level validation.
