# Harmonia API Gap Report

## Already clean in Harmonia.jl

`Harmonia.jl/src` cleanly exposes composable netlist builders:

- TL/JTL/RF-JTL and JJ blocks in `Transmission_line_block.jl`.
- RPM blocks in `JTWPA_standard_block.jl`.
- CPW geometry formulas and coupler optimization in `CPW_Theory.jl`.
- Discrete directional-coupler generation in `directional_coupler_block.jl`.
- IPM topology assembly in `IPM.jl`.
- HDF5 export prototype in `Save_data.jl`.

Boundary is appropriate: package builds Julia circuit descriptions; `JosephsonCircuits.jl` remains HB backend.

## Only in exploratory Harmonia

- Generic `run_circuit_simulation(...)->hbsolve(...)`.
- Richer TWPA PM/no-PM, RF-SQUID, DC-section and interferometer builders.
- Parameter mutation for fitting.
- Saturation, gain-length, DC-count, pump-frequency and pump-power sweeps.
- Measurement preprocessing, CSV/Sonnet import, objective functions and optimizer wrappers.
- Output-reading examples for HDF5/JLS/JLD2.
- Floquet, loss, taper, filter, JPA and IMPA research variants.

These are prototypes. Promote concepts selectively. Do not copy campaign orchestration into Julia.

## Missing stable Julia simulation CLI

1. `Harmonia.jl/scripts/run_simulation.jl`.
2. Versioned JSON config parser and validation.
3. Stable simulation request/result types.
4. Single supported baseline topology with deterministic builder.
5. Thin wrapper calling `JosephsonCircuits.hbsolve`.
6. Structured exception capture and nonzero exit code.
7. Atomic `status.json` state transitions.
8. Versioned HDF5 writer contract.
9. Provenance: config copy, package versions, timestamps, runtime.
10. Fast fixture test with tiny circuit; no expensive sweep.

## Missing HDF5/status compatibility for Python ML

Existing `Save_data.jl` is useful but not a stable interchange contract. Python needs:

- `schema_version`.
- Fixed group names and dataset shapes.
- Explicit SI units and axis names.
- Complex-array representation policy.
- Requested config and resolved/defaulted config.
- Stable success/failure status JSON.
- Run ID, timestamps, duration, Julia/Harmonia/JosephsonCircuits versions.
- Solver options and convergence diagnostics.
- Optional circuit summary without requiring Julia serialization.
- No Python dependency on `.jls` or `.jld2`.

## Missing calibration campaign features

Owned by `twpa_jax`, not Julia package:

- Run-directory allocator and campaign manifest.
- Parallel job launching, timeout, retries and resume.
- Parameter-space serialization and deterministic run IDs.
- HDF5/status validation and failed-run ingestion.
- Dataset builder and feature extraction.
- Objective functions, constraints and measured-data alignment.
- Bayesian optimization, SBI and ML loops.
- Provenance, report generation and artifact indexing.

## Promote first

1. Stable CLI shell plus schema.
2. Baseline standard JTWPA builder from `Harmonia.jl/src/Transmission_line_block.jl`.
3. Thin HB runner concept from `Harmonia/core/modules/circuit_simulators.jl`.
4. Schema-focused rewrite of `Harmonia.jl/src/Save_data.jl`.
5. Tiny integration fixture in `Harmonia.jl/test/runtests.jl`.
6. Later: IPM builder and directional coupler support.

## Do not touch first

- `Harmonia/Antoine_dev/**`: research archive.
- `Harmonia/core/User_scripts/**`: campaign references.
- Top-level `Harmonia/*.jl`: exploratory scripts.
- Binary `.jls` and `.jld2`: Julia-only legacy artifacts.
- Julia optimization wrappers: calibration ownership belongs in Python.
- JosephsonCircuits HB implementation: authoritative backend remains external.

## Confirmed vs inferred

Confirmed from source: package block builders, direct exploratory `hbsolve` calls, HDF5/JLS writer, optimizer wrappers and campaign scripts. Inferred: exact minimal production schema and promotion sequence. Binary artifact internal schemas were not deeply inspected.
