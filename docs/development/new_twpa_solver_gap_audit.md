# New TWPA Solver Gap Audit

Audit date: 2026-06-13.

| Question | Status | Finding |
|---|---|---|
| Is the current IPM coupler physical or only a pump marker? | PARTIAL | Two topologies now exist. `ipm_jtwpa_reduced_marker` is explicitly marker-based. `ipm_jtwpa_physical_coupler` stamps physical coupled inductors with optional capacitance. |
| Does conversion use the time-periodic Jacobian of the pump state? | OK | `build_conversion_sparameters` evaluates `model.nonlinear_derivative_matrices(phi_p(t))`, Fourier-expands it, and builds the sideband admittance. |
| Does zero-pump conversion reduce to ordinary linear S-parameters? | OK | Validated in tests and artifact `outputs/new_twpa_solver/conversion_validation_20260613_080228`. |
| Does pump solve status carry into gain rows? | OK | CLI masks `signal_gain_db` and `idler_gain_db` to NaN when pump status is not converged. |
| Is AFT/HB residual NumPy-only or JAX-native? | PARTIAL | NumPy residual remains the main SciPy path. `JaxPumpAFTResidual` is JIT/JVP-compatible for fixed-size reduced/physical models and is tested on a tiny TWPA residual. |
| Which solvers run on actual TWPA residuals? | PARTIAL | SciPy least-squares runs all maps. JAX dense Newton and JAX Newton-Krylov run tiny TWPA residual benchmark/tests. SciPy root/newton_krylov wrappers are still mostly toy/problem-interface coverage. |
| What exactly was simulated in the reduced 25x25 map? | OK | `outputs/new_twpa_solver/ipm_25x25_reduced_marker_validation_20260613_075706`: topology `ipm_jtwpa_reduced_marker`, cells_per_line=1, pump_harmonics=1, sidebands=1, pump range 6-8 GHz and -28 to -19 dBm, `pump_current_coupling=0.001`, solver SciPy least-squares. |

## Remaining Gaps

- The physical coupler is a compact coupled-inductor abstraction, not the full
  CPW geometry optimizer from Harmonia.
- The Josephson line is still a reduced lumped JTL; it is not calibrated to the
  old 2508-junction IPM.
- JAX paths are validated for small real TWPA residuals, but map production
  still uses SciPy least-squares.
- The conversion idler definition is still sideband-grid based; full two-tone
  compression/intermodulation is not implemented.
