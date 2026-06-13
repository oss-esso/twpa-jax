# New TWPA Solver IPM 25x25 Report

JosephsonCircuits.jl is no longer the production solver backend.
The new solver treats the TWPA as a modular nonlinear dynamical system.
Harmonic balance is implemented as a Fourier pseudo-spectral residual.
S-parameters are computed by pump-only HB followed by linearized
conversion-matrix analysis.
Alternative nonlinear solvers are implemented through a shared residual
interface.

## Runs

Smoke run:

- Path: `D:\Projects\Thesis\outputs\new_twpa_solver\ipm_smoke_converged`
- Grid: 3 x 3
- Topology: reduced IPM JTWPA, `cells_per_line=2`
- Pump harmonics: 1
- Sidebands: 1
- Solver: SciPy least-squares
- Success rate: 9 / 9 cells

Reduced 25x25 attempt:

- Path: `D:\Projects\Thesis\outputs\new_twpa_solver\ipm_25x25_reduced_attempt`
- Grid: 25 x 25
- Topology: reduced IPM JTWPA, `cells_per_line=1`
- Pump harmonics: 1
- Sidebands: 1
- Solver: SciPy least-squares
- Success rate: 582 / 625 cells

Physical-coupler 25x25 attempt:

- Path: `D:\Projects\Thesis\outputs\new_twpa_solver\ipm_25x25_physical_coupler_attempt_20260613_075737`
- Grid: 25 x 25
- Topology: `ipm_jtwpa_physical_coupler`
- Size: `cells_per_line=32`
- Pump harmonics: 5
- Sidebands: 3
- Solver: SciPy least-squares
- Success rate: 625 / 625 cells

The requested `cells_per_line=32`, `pump_harmonics=5`, `sidebands=3`
physical-coupler command was run to completion for the current reduced
physics model. This is still not a fully calibrated historical old-IPM twin.

## Artifact Files

Each run writes:

- `config.json`: exact CLI configuration.
- `rows.csv`: row-level raw values, gains, convergence, runtime, residuals.
- `pump_solution_status.csv`: compact convergence table.
- `gain_signal_db_grid.csv`: signal gain grid.
- `idler_gain_db_grid.csv`: idler conversion grid.
- `convergence_mask_grid.csv`: 1 for converged, 0 otherwise.
- `residual_norm_grid.csv`: scaled residual infinity norm.
- `runtime_grid.csv`: cell runtime.
- `solver_comparison_summary.csv`: per-run solver summary.
- `plots/signal_gain_unmarked.png`
- `plots/signal_gain_marked_by_convergence.png`
- `plots/signal_gain_converged_only.png`
- `plots/idler_gain_unmarked.png`
- `plots/residual_norm_heatmap.png`
- `plots/runtime_heatmap.png`
- `report.md`

## Limitations

- The reduced IPM source uses explicit `pump_current_coupling` instead of the
  old undocumented 32 dB offset convention.
- Conversion sideband ordering is implemented for two ports and integer pump
  sidebands; larger conversion systems need conditioning/preconditioning work.
- SciPy least-squares is the robust production baseline today.
- JAX solvers are small-problem validated, but not yet the default map engine.
- RF-SQUID, kinetic-inductance, MOR, deflation, and full two-tone HB are
  scaffolded rather than full production implementations.

## Next Step

Move the AFT residual to a JAX-native implementation, add a passive linear
preconditioner for JVP-GMRES, and benchmark `cells_per_line=8,16,32` with
increasing pump harmonics before returning to larger IPM gain maps.
