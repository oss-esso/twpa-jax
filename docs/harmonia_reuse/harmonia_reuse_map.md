# Harmonia Reuse Map

Purpose: classify reusable pieces from `Harmonia` and `Harmonia.jl` before migration.

## Classification labels

- `STABLE_PACKAGE_CODE`
- `REUSABLE_PROTOTYPE`
- `ONE_OFF_EXPERIMENT`
- `OBSOLETE`
- `UNKNOWN`

## Table

| source repo | file | classification | what it does | dependencies | target owner | target API | action |
|---|---|---|---|---|---|---|---|
| Harmonia.jl | src/Harmonia.jl | UNKNOWN | package entry point | unknown | Harmonia.jl | TBD | inspect |
| Harmonia.jl | src/CPW_Theory.jl | UNKNOWN | CPW/coupler theory | unknown | Harmonia.jl | coupler synthesis | inspect |
| Harmonia.jl | src/directional_coupler_block.jl | UNKNOWN | directional-coupler netlist block | unknown | Harmonia.jl | circuit template block | inspect |
| Harmonia.jl | src/IPM.jl | UNKNOWN | IPM topology builder | unknown | Harmonia.jl | IPM template | inspect |
| Harmonia.jl | src/Save_data.jl | UNKNOWN | HDF5/JLS export | HDF5/Serialization | Harmonia.jl + twpa_jax reader | versioned schema | inspect |
| Harmonia | core/fit_simulation_to_data.jl | UNKNOWN | fitting prototype | unknown | twpa_jax calibration docs | objective examples | inspect |
