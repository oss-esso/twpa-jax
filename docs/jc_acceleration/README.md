# JosephsonCircuits / Harmonia Acceleration Notes

This folder documents the acceleration and workflow-cleanup work around Harmonia.jl,
JosephsonCircuits.jl, and the Python orchestration layer.

The goal is not to replace nonlinear HB. The goal is to make each workflow use the
right solver path:

- linear response -> direct `hblinsolve`
- nonlinear pumped/gain workflows -> full `hbsolve`
- Python -> orchestration, benchmark, dataset, reporting
- Julia -> authoritative circuit generation and simulation

## Documents

1. `direct_linear_backends.md`  
   What was changed: JTL, RF-JTL, and ETHZ-JTL direct `hblinsolve` backends.

2. `showcase_designs.md`  
   Which real and scaled designs should be used to demonstrate the impact.

3. `benchmark_methodology.md`  
   How to interpret cold-start, warm-run, batch-runner, and backend timings.

4. `decision_log.md`  
   Decisions made, rejected paths, and what must not be claimed.
