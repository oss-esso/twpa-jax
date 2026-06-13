# Julia Reference Map For New TWPA Solver

The old Julia/Harmonia/JosephsonCircuits files were inspected as topology and
convention references only. They are not used as the production solver backend.

## Old Map Script Findings

`Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map.jl`
and `run_report_old_ipm_power_map_gridn.jl` build the old report-style IPM
map. The important convention is:

```text
source_power_dbm = external_plot_power_dbm - power_offset_db
pump_current_a = sqrt(2 * source_power_W / 50 ohm)
sources = [(mode=(1,), port=4, current=pump_current_a)]
```

Defaults found:

- external/report pump-power axis: -28 dBm to -19 dBm.
- default grid points: 25.
- default power offset: 32 dB.
- pump frequency modes include smoke/map/power-slice variants.
- old port convention: input=1, output=2, pump/coupler source=4.
- old status artifacts distinguish finite data from converged data.

## Topology Reference Table

| Old Julia file | purpose | topology details extracted | new Python equivalent | open questions |
|---|---|---|---|---|
| `Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map.jl` | Old colleague-style IPM power map | Old-IPM netlist, external power axis, 32 dB offset, source current on port 4, convergence masks | `twpa_solver.experiments.run_ipm_25x25_gain_map` rows now include external power, source power, current coupling, topology, coupler model, and convergence status | Exact calibration from old 32 dB offset to measured lab pump chain still needs experimental traceability |
| `Harmonia.jl/experiments/jc_setup_cache/run_report_old_ipm_power_map_gridn.jl` | Patched arbitrary-grid version | Same topology and power convention as old map, with configurable `--points` | Python CLI exposes `--points` for both topology labels | None for grid mechanics |
| `Harmonia/IPM_JTWPA.jl` | Modern JTWPA IPM through `make_IPM(...)` | Ports 1/2 for signal path, ports 3/4 for coupler/pump line; source often on port 4; coupler target around -14 dB | `build_ipm_jtwpa_physical_coupler` uses four ports ordered input, output, pump_isolation, pump | Full geometric CPW optimizer is not ported |
| `Harmonia/directional_coupler.jl` | Standalone directional-coupler smoke | `generate_and_append_coupler!`, top and bottom lines, port 4 pump, target coupling dB | `DirectionalCouplerBlock` implements the requested coupled-inductor abstraction plus optional capacitances | The Python block is simpler than the CPW-derived distributed coupler |
| `Harmonia/Interferometer.jl` | Older manual/interferometric coupler | Coupled inductors `K`, shunt inductors to ground, separate normal and coupler lines | Python physical coupler uses `B L^-1 B^T` for two branches | Interferometric multi-section layout is not replicated |
| `Harmonia/IPM_rf_squid.jl` | RF-SQUID / RF-JTL IPM | Two physical couplers, RF-JTL sections, pump on port 4, optional DC bias | RF-SQUID nonlinearity remains interface-only; physical coupler topology now exists | RF-SQUID branch law and DC bias are not production-ready |
| `Harmonia/IPM_rf_squid_Cg.jl` | RF-SQUID variant | Same layout with different `Cg`, `Lrf`, pump current | Same as above | Parameter calibration remains open |
| `Harmonia/IPM_rf_squid_Lwp.jl` | RF-SQUID variant with extra inductive pieces | RF-JTL elements plus extra series inductance sections | Not yet represented except by generic linear inductors | Need explicit RF-SQUID/IPM reduced model before comparing |
| `Harmonia.jl/src/directional_coupler_block.jl` | Harmonia coupler generator | CPW-derived `L_cell`, `C_gnd_cell`, `Cc_cell`, `K_ind`, repeated coupled cells | Python block uses one or more simple coupled-inductor cells; current IPM uses two single-cell couplers | Port CPW geometry optimizer later if needed |
| `Harmonia.jl/src/templates/CouplerTemplates.jl` | CircuitIR coupler template | Capacitive coupling represented, K retained as metadata due export limits | Python model directly stamps K-style coupled inductors | CircuitIR export parity is not required for new solver |

## Python Convention Adopted

The new rows preserve both:

- `pump_power_dbm`: external/report axis.
- `source_power_dbm`: equivalent internal source power after
  `pump_current_coupling`.
- `pump_current_a`: peak internal current injected into `pump_source_nodes`.

The Python implementation does not inherit the old 32 dB offset. It uses an
explicit `pump_current_coupling`, recorded in `config.json` and `rows.csv`.
