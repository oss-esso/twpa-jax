# New TWPA Solver Comparison

The benchmark command was:

```powershell
python -m twpa_solver.experiments.benchmark_solvers `
  --out D:\Projects\Thesis\outputs\new_twpa_solver\solver_benchmark_summary.csv
```

The tested toy equation was `x^3 - 1 = 0` from initial guess `x0 = 0.2`.

| Solver | Status | Tested Problem | Notes |
|---|---|---|---|
| SciPy least_squares | implemented and passed | scalar cubic | CPU reference baseline. |
| SciPy root | implemented and passed | scalar cubic | Fast on the toy problem. |
| SciPy newton_krylov | implemented but failed | scalar cubic | Kept as matrix-free CPU baseline, but poor starts need globalization. |
| JAX dense Newton | implemented and passed | scalar cubic | Small-problem validator using dense Jacobian. |
| JAX matrix-free Newton-Krylov | implemented and passed | scalar cubic | Uses JVP plus SciPy GMRES on CPU. |
| Pseudo-transient + LS | implemented and passed | scalar cubic | Simple globalization wrapper, not yet optimized. |

The reduced 25x25 IPM attempt used SciPy least-squares and produced:

- rows: 625
- success rate: 0.9312
- invalid cells: 43
- median runtime per cell: about 0.0016 s
- artifact root: `D:\Projects\Thesis\outputs\new_twpa_solver\ipm_25x25_reduced_attempt`

Large-scale production status: the robust baseline path is SciPy least-squares
on reduced IPM models. JAX Newton paths are implemented and tested, but need a
JAX-native residual and GPU batching pass before they should be used for large
maps.
