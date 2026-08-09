# Old-IPM Backend Replacement Plan

The Julia design/map frontend remains canonical. JosephsonCircuits is now reference material, not the target production backend.

| Old workflow step | old JosephsonCircuits path | new backend replacement path | status |
|---|---|---|---|
| Build old-IPM circuit | `build_old_ipm_circuit()` returns `circuit, circuitdefs, metadata` | same Julia builder | complete |
| Resolve symbolic values | JosephsonCircuits parser during `hbsolve` | `export_old_ipm_circuit_json.jl` resolves numeric element values | complete |
| Preserve netlist | implicit Julia tuple list | `old_ipm_circuit.json` | complete |
| Import circuit | JosephsonCircuits internal netlist parser | `twpa_solver.importers.julia_circuit_json` | complete for `P/R/L/C/Lj/K` |
| Named mutual inductors | JosephsonCircuits `K` handling | Python coupled-inductor network assembled from named branches | complete |
| Josephson branches | JosephsonCircuits nonlinear branches | Python Josephson incidence and `Ic = phi0bar/Lj` | complete |
| Linear smoke solve | `rpm.linearized.S` after HB | Python unpumped linear S-parameters | smoke complete |
| Pump-only HB | `hbsolve(...)` | exported-netlist AFT/HB residual | not implemented for full old-IPM yet |
| Gain map rows | old CSV writer | `run_exported_julia_circuit_map.py` mirrored row schema | partial |
| Numerical old-map parity | JosephsonCircuits reference output | independent backend output on exact exported circuit | not established |

## Immediate Next Step

Build the AFT/HB residual directly on `ImportedJuliaCircuit.model` and solve a single pump point from the exported old-IPM JSON. The first target is not speed; it is correctness and convergence metadata on the exact circuit.

## Non-Goals For This Recovery Pass

Reduced Python topologies are not parity targets. They remain solver sandbox fixtures only.
