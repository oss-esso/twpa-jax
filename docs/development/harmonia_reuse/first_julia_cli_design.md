# First Julia CLI Design

Smallest supported command:

```powershell
julia --project=Harmonia.jl Harmonia.jl/scripts/run_simulation.jl --config config.json --output run_dir
```

## First simulation type

`standard_jtwpa_small_signal_hb`: build standard JTWPA transmission line/JJ netlist, run one pump with one signal-frequency vector through `JosephsonCircuits.hbsolve`, save raw HB-derived scattering output. No sweep orchestration, calibration, BO, SBI or ML in Julia.

## Config JSON

```json
{
  "schema_version": "1.0",
  "simulation_type": "standard_jtwpa_small_signal_hb",
  "run_id": "optional-python-generated-id",
  "circuit": {
    "n_cells": 50,
    "Cg_F": 7.1e-14,
    "Lj_H": 1.0e-10,
    "Cj_F": 2.0e-13,
    "sigma": 0.0,
    "source_port": 1,
    "output_port": 2,
    "port_impedance_ohm": 50.0
  },
  "drive": {
    "pump_frequency_Hz": 7.0e9,
    "pump_power_dBm": -70.0,
    "signal_frequencies_Hz": [4.0e9, 4.1e9],
    "signal_power_dBm": -120.0
  },
  "solver": {
    "Nmodulationharmonics": [3],
    "Npumpharmonics": [6],
    "dc": false,
    "threewavemixing": false,
    "fourwavemixing": true,
    "iterations": 50
  }
}
```

Validation: reject unknown `schema_version`, unsupported `simulation_type`, missing keys, non-finite numbers, non-positive physical values, empty frequency vectors and unsupported solver combinations.

## Output folder

```text
run_dir/
  config.input.json
  config.resolved.json
  status.json
  simulation.h5
  stderr.log
```

`simulation.h5` exists only after successful write. Writer uses temporary filename then rename.

## status.json

```json
{
  "schema_version": "1.0",
  "run_id": "id",
  "simulation_type": "standard_jtwpa_small_signal_hb",
  "state": "succeeded",
  "started_at_utc": "2026-06-01T19:00:00Z",
  "finished_at_utc": "2026-06-01T19:00:02Z",
  "duration_s": 2.0,
  "exit_code": 0,
  "error": null,
  "artifacts": ["config.input.json", "config.resolved.json", "simulation.h5"],
  "versions": {
    "julia": "x.y.z",
    "harmonia": "git-or-package-version",
    "josephson_circuits": "package-version"
  }
}
```

Allowed `state`: `running`, `succeeded`, `failed`. Failure adds:

```json
"error": {"type": "ExceptionType", "message": "short message", "stage": "validate|build|solve|write"}
```

## simulation.h5

```text
/metadata
  schema_version
  run_id
  simulation_type
  created_at_utc
  duration_s
/config
  input_json
  resolved_json
/axes
  signal_frequency_Hz             [n_signal]
/results
  s21_real                        [n_signal]
  s21_imag                        [n_signal]
  gain_dB                         [n_signal]
/solver
  converged                       scalar or [n_signal]
  iterations_requested
  diagnostics_json
/circuit
  component_count
  summary_json
```

Use split real/imag datasets for language-neutral complex data. Dataset names carry units.

## Failure behavior

1. Create output directory.
2. Write `config.input.json`.
3. Write `status.json` with `running`.
4. Validate, resolve defaults, build circuit, solve, write temporary HDF5, rename.
5. On success write `status.json` with `succeeded`, exit `0`.
6. On error append stack trace to `stderr.log`, write concise failed status, remove temporary HDF5, exit nonzero.

Python reads `status.json` first. Python never requires Julia serialization.

## Deliberately excluded

IPM, RF-SQUID, coupler geometry optimization, multi-dimensional sweeps, Julia fitting and report generation. Add after baseline contract works end-to-end.
