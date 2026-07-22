# New TWPA Solver Architecture

JosephsonCircuits.jl is no longer the production solver backend. The new
package lives under `twpa_solver` and treats JosephsonCircuits/Harmonia outputs
as historical reference material only.

## Implementation Plan

1. Build an independent node-flux circuit model with explicit matrices,
   incidence graphs, port definitions, and nonlinear branch laws.
2. Implement harmonic balance as one residual/discretization, not as the model.
3. Put every nonlinear method behind a shared `SolverResult` with convergence
   metadata.
4. Compute S-parameters through linear admittance solves and pumped
   conversion-matrix analysis.
5. Save row-level artifacts, status masks, plots, config, and reports for each
   map run.

## Package Layout

- `twpa_solver/model`: graph, blocks, topology, ports, units, Josephson/RF-SQUID/KI nonlinear interfaces, IPM builder.
- `twpa_solver/residuals`: AFT pump HB residual, linear admittance, conversion matrix, scaling, time-domain helpers, two-tone grid scaffold.
- `twpa_solver/solvers`: SciPy least-squares/root/Newton-Krylov, JAX dense Newton, JAX JVP-GMRES Newton-Krylov, pseudo-transient wrapper, continuation, arclength, shooting, Anderson, multistart clustering, MOR hooks.
- `twpa_solver/sparams`: wave and conversion S-parameter helpers.
- `twpa_solver/experiments`: IPM gain-map runner, plotter, solver benchmark.

## Production-Ready vs Scaffolded

Production-ready first pass:

- Modular IPM JTWPA topology assembly.
- Distinct topology labels:
  `ipm_jtwpa_reduced_marker` and `ipm_jtwpa_physical_coupler`.
- Physical coupled-inductor directional-coupler block with optional
  shunt/mutual capacitance.
- Linear frequency-domain S-parameters.
- Pump-only AFT HB residual.
- SciPy least-squares pump solve for reduced IPM maps.
- Pumped conversion-matrix S-parameter calculation.
- Artifact-producing 3x3 smoke and reduced 25x25 map CLI.

Scaffolded or small-problem only:

- JAX dense Newton and JAX JVP-GMRES are validated on toy problems.
- JAX AFT residual is validated on tiny real TWPA residuals, but not yet used
  for production maps.
- SciPy Newton-Krylov is implemented but not robust on all toy starts.
- Pseudo-transient is a simple globalization wrapper.
- Arclength, shooting, Anderson, deflation, MOR, and two-tone HB are API-ready with focused toy tests.

## Commands

```powershell
cd D:\Projects\Thesis\twpa_jax

python -m twpa_solver.experiments.run_ipm_25x25_gain_map `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\ipm_smoke_converged `
  --points 3 `
  --cells-per-line 2 `
  --pump-harmonics 1 `
  --sidebands 1 `
  --solver scipy-least-squares
```

```powershell
python -m twpa_solver.experiments.run_ipm_25x25_gain_map `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\ipm_25x25_reduced_attempt `
  --points 25 `
  --cells-per-line 1 `
  --pump-harmonics 1 `
  --sidebands 1 `
  --solver scipy-least-squares
```

PowerShell in this environment did not expand `tests/test_twpa_*.py` for
pytest, so the verified test command was:

```powershell
$files = Get-ChildItem tests -Filter 'test_twpa_*.py' | ForEach-Object { $_.FullName }
python -m pytest $files -q -p no:cacheprovider
```
