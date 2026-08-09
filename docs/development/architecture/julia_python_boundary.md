# Julia/Python Boundary

## Decision

Julia is the authoritative production simulation engine.

- `Harmonia.jl` owns stable Julia package code: topology generation, reusable circuit templates, JosephsonCircuits.jl calls, HDF5/JLS/status export.
- `Harmonia` is exploratory archaeology unless a piece is explicitly promoted.
- `twpa_jax` owns Python orchestration, run management, caching, HDF5 readers, datasets, calibration, Bayesian optimization, SBI, ML, and reporting.

## Workspace layout

```text
D:\Projects\Thesis
  Harmonia       # exploratory Julia workflows
  Harmonia.jl    # stable Julia package
  twpa_jax       # Python orchestration/calibration/ML repo
Non-goals
Do not reimplement JosephsonCircuits.jl in Python for production.
Do not treat Python HB fallback/surrogate outputs as production truth.
Do not start SBI/ML before schema-valid simulation data exists.
Do not run unbounded dense HB.
First milestone

M1 is not a full simulator. M1 is:

one tiny Julia simulation launched from a config;
structured output folder;
status.json;
simulation.h5;
Python can parse status.
Status semantics

PASS requires solver success, finite outputs, finite residuals where applicable, and physically meaningful output.

PARTIAL means fallback, surrogate, reduced diagnostic, or insufficient validation.

FAIL means solver failure, nonfinite arrays, resource failure, schema failure, or invalid physics.

UNKNOWN means not run or insufficient metadata.

Production boundary

Python can contain reference HB checks, dense tiny validation, and resource-safety experiments.

Production JTWPA/rf-SQUID/IPM simulation must go through Julia/Harmonia/JosephsonCircuits unless a future decision explicitly changes this boundary.
