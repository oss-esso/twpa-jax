# Backend Substitution 5x5 Report

## Canonical Map Script

The canonical old-IPM gain map script is:

```text
Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map_gridn.jl
```

The backend-compare runner is a copied/extended runner:

```text
Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map_backend_compare.jl
```

It reuses `build_old_ipm_circuit()`, the same 5x5 axes, the same port/source convention, the same row schema, and the same artifact writers.

## JosephsonCircuits Callsite

The backend callsite is in `run_case(...)` in the canonical runner. It calls:

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

Downstream fields used:

```julia
s21_linear = rpm.linearized.S((0,), 2, (0,), 1, :)
gain_db = 10 .* log10.(abs2.(s21_linear))
gain_db_max = maximum(gain_db)
```

Convergence status comes from the captured solver log via `classify_from_log_text(...)` and `classify_hb_row(...)`.

## Backend Adapter

The adapter runner defines `BackendMapPointResult` with the fields needed by the old map writer:

```text
backend, status, success, gain_db_max, raw_gain_trace,
convergence_mask_value, residual_norm, infinity_norm,
solver_message, runtime_s, metadata
```

For `--backend josephsoncircuits`, it delegates to the original `run_case(...)`.

For independent backends, it exports the exact circuit from `build_old_ipm_circuit()` and calls:

```text
python -m twpa_solver.experiments.solve_old_ipm_backend_point
```

No reduced Python surrogate topology is used.

## 5x5 Runs

Output root:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\backend_compare_5x5
```

Reference root:

```text
D:\Projects\Thesis\outputs\jc_profiles\jc3m_report_old_ipm_power_map_5x5_marked
```

Comparison artifact:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\backend_compare_5x5\comparison\backend_comparison_summary.md
```

## Results

| Backend | rows | status | gain produced? | comparison |
|---|---:|---|---:|---|
| `josephsoncircuits` | 25 | `VALID_CONVERGED=10`, `FINITE_NONCONVERGED=15` | yes | `REFERENCE_REPRODUCED` on status and trusted converged gains |
| `scipy-least-squares` | 25 | `BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN=25` | no | `NOT_IMPLEMENTED` |
| `scipy-root` | 25 | `BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN=25` | no | `NOT_IMPLEMENTED` |
| `scipy-newton-krylov` | 25 | `BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN=25` | no | `NOT_IMPLEMENTED` |
| `jax-dense-newton` | 25 | `BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN=25` | no | `NOT_IMPLEMENTED` |
| `jax-newton-krylov` | 25 | `BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN=25` | no | `NOT_IMPLEMENTED` |
| `pseudo-transient` | 25 | `BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN=25` | no | `NOT_IMPLEMENTED` |

The JosephsonCircuits rerun matches all 25 reference status labels. On trusted `VALID_CONVERGED` gain cells, the maximum gain difference is `3.012168292571005e-11 dB`. The all-cell mean absolute gain difference is large because nonconverged diagnostic cells are not numerically stable and are not optimization-grade values.

## SciPy Least-Squares Real Attempt

A follow-up run used the same Julia backend-compare runner and the exact exported `build_old_ipm_circuit()` netlist:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\backend_compare_5x5\scipy_least_squares_real
```

This run no longer returns `BACKEND_NOT_IMPLEMENTED_FOR_OLD_IPM_GAIN`. It evaluates the exact full old-IPM pump residual and invokes `scipy.optimize.least_squares` with a strict bounded evaluation cap.

Result:

```text
FAILED_MAX_NFEV = 25 cells
gain produced = no
comparison classification = FAILED_NUMERICALLY
```

The first milestone point is documented in:

```text
twpa_jax/docs/scipy_least_squares_old_ipm_backend_milestone.md
```

## Next Implementation Target

Improve the `scipy-least-squares` exact-netlist backend so one old-IPM pump point reduces residual rather than stopping at `FAILED_MAX_NFEV`. The immediate blocker is scalable Jacobian/linearization for the 3134-node Fourier residual.
