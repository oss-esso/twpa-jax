# Python Old-Julia Parity 25x25 Report

Python parity artifact:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\python_old_julia_parity_25x25
```

Run command:

```powershell
cd D:\Projects\Thesis\twpa_jax
python -m twpa_solver.experiments.run_ipm_25x25_gain_map `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\python_old_julia_parity_25x25 `
  --topology ipm_jtwpa_old_julia_parity `
  --points 25 `
  --pump-freq-min-ghz 6.0 `
  --pump-freq-max-ghz 8.0 `
  --pump-power-min-dbm -28 `
  --pump-power-max-dbm -19 `
  --use-old-julia-power-offset true `
  --power-offset-db 32.0 `
  --old-port-convention true `
  --cells-per-line 32 `
  --pump-harmonics 5 `
  --sidebands 3 `
  --solver scipy-least-squares `
  --continuation snake `
  --compute-conversion-sparams true
```

## Result

- rows: 625
- converged cells: 142 / 625
- success rate: 0.2272
- mean scaled residual infinity norm: 1.9124
- median runtime per cell: 0.592 s
- topology: `ipm_jtwpa_old_julia_parity`
- geometry profile: `old_julia_parity_compact_surrogate`
- coupler model: `compact_coupled_inductor`
- historical target junctions recorded: 2508

## What Parity Mode Reproduces

- external/report power axis: yes
- pump frequency axis: yes
- `source_power_dbm = external_power_dbm - 32`: yes
- old peak-current convention: yes
- port-equivalent metadata: input 1, output 2, pump 4
- convergence/status masking in rows: yes

## What Parity Mode Does Not Reproduce

- old JosephsonCircuits netlist: no
- old distributed/CPW Harmonia coupler: no
- 2508-junction old-IPM size: no, this run used `cells_per_line=32`
- old harmonic truncation exactly: no, Python run used pump harmonics 5 and sidebands 3
- old solver algorithm: no, Python uses SciPy least-squares on the independent residual

This is a compatibility/calibration mode, not proof of old-Julia output
equivalence.
