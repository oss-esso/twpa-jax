# Exported Julia Netlist Backend

This backend preserves the Harmonia/JosephsonCircuits Julia design files as the canonical frontend and moves independent solving underneath an exported circuit netlist.

## Export

Command:

```powershell
cd D:\Projects\Thesis\Harmonia.jl
julia --project=. experiments\jc_setup_cache\export_old_ipm_circuit_json.jl `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\old_ipm_export
```

Output:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\old_ipm_export\old_ipm_circuit.json
```

The exporter calls `build_old_ipm_circuit()` and does not run `hbsolve`. It serializes:

- full circuit element list;
- resolved numeric `circuitdefs`;
- metadata from the builder;
- port convention;
- power/source convention;
- harmonic settings;
- old map axes.

Observed export:

```text
elements = 8788
metadata.Nj = 2508
n_final = 3134
ports = 1, 2, 3, 4
```

## Import

Python importer:

```text
twpa_solver/importers/julia_circuit_json.py
```

Supported elements:

- `P`: port metadata;
- `R`: branch conductance;
- `C`: branch capacitance;
- `L`: named linear inductor branch;
- `Lj`: Josephson branch with `Ic = reduced_phi0 / Lj`;
- `K`: named mutual coupling between prior inductor branch names.

The importer preserves Julia node labels, maps non-ground labels to reduced node indices, and assembles coupled-inductor groups from named `K` references.

## Backend Smoke

Command:

```powershell
cd D:\Projects\Thesis\twpa_jax
python -m twpa_solver.experiments.run_exported_julia_circuit_map `
  --circuit-json D:\Projects\Thesis\outputs\new_twpa_solver\old_ipm_export\old_ipm_circuit.json `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\old_ipm_python_backend_smoke `
  --points 3 `
  --pump-harmonics 10 `
  --modulation-harmonics 5 `
  --solver scipy-least-squares
```

Result:

```text
LINEAR_IMPORTED_SMOKE_OK_HB_NOT_IMPLEMENTED: 9
```

This is intentionally not old-map parity. It is an import/assembly and linear smoke proof for the exact exported old-IPM circuit.
