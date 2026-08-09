# Physical-Coupler IPM Report

## What Is Physically Implemented

- `DirectionalCouplerBlock` stamps a two-branch coupled inductor:

```text
L = [[L1, M], [M, L2]], M = k sqrt(L1 L2)
i = L^-1 psi
i_node = B L^-1 B^T phi
```

- Optional shunt capacitances and mutual capacitance are stamped into the
  capacitance matrix.
- `ipm_jtwpa_physical_coupler` is a four-port model:
  input, output, pump_isolation, pump.
- Pump current is injected at the `pump` port node, matching the old port-4
  convention.

## What Remains Reduced

- The coupler is not yet the full distributed CPW geometry-optimized coupler.
- The JTL is still a reduced Josephson line, not the historical 2508-junction
  old-IPM.
- RF-SQUID, DC bias, and compression/two-tone HB are not production-ready.

## Map Artifacts

| Topology | cells | harmonics | sidebands | solver | points | converged | runtime | artifact |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `ipm_jtwpa_reduced_marker` | 1 | 1 | 1 | SciPy least-squares | 25x25 | 582/625 | median 0.0015 s/cell | `outputs/new_twpa_solver/ipm_25x25_reduced_marker_validation_20260613_075706` |
| `ipm_jtwpa_physical_coupler` | 2 | 1 | 1 | SciPy least-squares | 3x3 | 9/9 | median 0.0039 s/cell | `outputs/new_twpa_solver/ipm_physical_coupler_smoke_20260613_075721` |
| `ipm_jtwpa_physical_coupler` | 32 | 5 | 3 | SciPy least-squares | 25x25 | 625/625 | median 0.3019 s/cell | `outputs/new_twpa_solver/ipm_25x25_physical_coupler_attempt_20260613_075737` |

Each map folder includes:

- `config.json`
- `rows.csv`
- `pump_solution_status.csv`
- `gain_signal_db_grid.csv`
- `idler_gain_db_grid.csv`
- `convergence_mask_grid.csv`
- `residual_norm_grid.csv`
- `runtime_grid.csv`
- `solver_comparison_summary.csv`
- `plots/signal_gain_unmarked.png`
- `plots/signal_gain_marked_by_convergence.png`
- `plots/signal_gain_converged_only.png`
- `plots/idler_gain_unmarked.png`
- `plots/residual_norm_heatmap.png`
- `plots/runtime_heatmap.png`
- `report.md`

## Pump Power And Current Convention

Rows now include:

- `pump_power_dbm`: external/report power axis.
- `source_power_dbm`: internal equivalent source power after current coupling.
- `pump_current_a`: internal peak current injected into the pump source node.
- `pump_current_coupling`: explicit scaling from external Norton current.
- `pump_source_nodes`: actual node receiving the pump source.

For the physical-coupler 25x25 run:

```text
pump_power_dbm = -28 to -19 dBm
pump_current_coupling = 0.001
source_power_dbm = pump_power_dbm + 20 log10(0.001)
```

This differs from `run_report_old_ipm_power_map.jl`, which used:

```text
source_power_dbm = external_plot_power_dbm - 32 dB
```

The new convention is explicit and recorded in artifacts. To compare against
colleague maps, choose `pump_current_coupling` so the resulting
`source_power_dbm` reproduces the intended pump-chain calibration, rather than
hardcoding the old offset silently.

## Validation Status

- Conversion zero-pump consistency: validated.
- Pump status propagation: validated.
- Physical coupler current stamping: validated.
- JAX residual parity and JVP: validated on tiny TWPA residual.
- Full physical-coupler 25x25 artifact path: completed.

This is a physically improved reduced simulator, not a fully calibrated
experimental IPM twin.
