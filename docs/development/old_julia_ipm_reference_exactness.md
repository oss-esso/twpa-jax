# Old Julia IPM Reference Exactness

Reference run artifact:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\old_julia_reference_25x25
```

Run command:

```powershell
cd D:\Projects\Thesis\Harmonia.jl
julia --project=. experiments\jc_setup_cache\run_report_old_ipm_power_map_gridn.jl `
  --outdir D:\Projects\Thesis\outputs\new_twpa_solver\old_julia_reference_25x25 `
  --mode mapN `
  --points 25 `
  --power-offset-db 32.0
```

The command completed and wrote 625 rows.

## Exact Map Axes

- mode: `mapN`
- points: 25
- pump frequency axis: 6.0 GHz to 8.0 GHz inclusive
- external/report pump-power axis: -28.0 dBm to -19.0 dBm inclusive
- source-power convention: `source_power_dbm = external_power_dbm - 32.0`
- current convention: `Ip_peak = sqrt(2 * P_source_W / 50 ohm)`

`run_report_old_ipm_power_map_gridn.jl` is needed for arbitrary 25x25
`mapN`; the original `run_report_old_ipm_power_map.jl` exposes smoke, map35,
and power-slice modes.

## Port And Solver Conventions

- input port: 1
- output port: 2
- pump/coupler source port: 4
- pump source: `sources = [(mode=(1,), port=4, current=pump_current_a)]`
- pump harmonics: `(10,)`
- modulation harmonics: `(5,)`
- default iterations used here: 50
- S-parameter of interest: `linearized.S(outputport=2, inputport=1, mode=(0,))`

## Old Circuit Details

From `build_old_ipm_circuit()`:

- `Nj = 418 * 6 = 2508`
- `pmrpitch = 418`
- `ll = 105`
- `ll2 = 200`
- `Lj = 2 * 79e-12 H`
- `Cj = 145e-15 F`
- `Cg = 66e-15 F`
- `Cl = 10 * 1.73e-15 F`
- `Ll = 10 * 4.13e-12 H`
- `Lm = 2000 * 4.13e-12 H`
- `K = 0.999`
- `Rleft = Rright = Rm = 50 ohm`

The old script's simplified coupler uses explicit inductors and JosephsonCircuits
mutual-inductor `K` elements, not the compact Python coupled-inductor block.
Other Harmonia files (`directional_coupler_block.jl`) define a CPW-derived
distributed coupler with `L_cell`, `C_gnd_cell`, `Cc_cell`, and `K_ind`.

## Status Rules And Artifacts

Old reference status counts:

- `VALID_CONVERGED`: 246
- `FINITE_NONCONVERGED`: 379

Output files:

- `report_old_ipm_power_map_rows.csv`
- `raw_gain_max_db_grid.csv`
- `finite_mask_grid.csv`
- `convergence_mask_grid.csv`
- `convergence_masked_gain_max_db_grid.csv`
- `residual_norm_grid.csv`
- `infinity_norm_grid.csv`
- `solver_warning_mask_grid.csv`
- `status_grid.csv`
- `reproduction_status_summary.md`
- `report_old_ipm_power_map_summary.md`

Only `VALID_CONVERGED` rows should be treated as clean. Finite
nonconverged rows are useful for visual/debug comparison only.
