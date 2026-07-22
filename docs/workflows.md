# Workflows

The repository's end-to-end entry points live in [`workflows/`](../workflows/). They compose the reusable builders and solver/plotting backends, so a backend change should normally be made under `src/twpa_solver/` or in the existing backend module rather than copied into a workflow.

## 1. Build a design and generate passive plots

`build_design_and_passive.py` accepts the IPM builder options and adds passive S-parameter analysis. The design directory is both the circuit output directory and the destination for the passive data and figures.

```powershell
python workflows/build_design_and_passive.py `
  --design-dir designs/ipm_2c_fixed `
  --array-length 418 `
  --num-rows 6 `
  --cell-length-um 10
```

The workflow always enables matrix output because the passive solver requires `C.npz`, `G.npz`, `K.npz`, `Bphi.npz`, and `ipm_arrays.npz`.

Generated files include:

- the normal IPM design files and `ipm_summary.json`;
- `passive_sparameters.npz`;
- `passive_s21_s24.{png,pdf,svg}`;
- `passive_s11_s21_s31_s41.{png,pdf,svg}`.

The passive convention is `S[frequency, output_port, source_port]`. The four directional traces in the second figure are all excited from port 1.

## 2. Run a gain map and generate its catalogue of plots

`run_gain_map_and_plots.py` forwards unrecognised options to `scripts/run_gain_map.py`, then invokes the standard plotting backend and the two signal-frequency projections.

```powershell
python workflows/run_gain_map_and_plots.py `
  --design designs/ipm_2c_fixed `
  --run-dir outputs/ipm_2c_gain_map `
  --n-power 5 `
  --n-frequency 5 `
  --pump-power-min-dbm -30 `
  --pump-power-max-dbm -20 `
  --pump-freq-min-ghz 7 `
  --pump-freq-max-ghz 8
```

The workflow forces the in-process executor and writes the usual map artifacts under the run directory, followed by plots under `run-dir/plots/`:

- simple gain, status, and runtime maps;
- spectrum-fit maps when signal spectra are available: peak gain, GBP, ripple, smoothness, and selected candidates;
- candidate tables and candidate spectra;
- gain versus pump frequency/signal frequency;
- gain versus pump power/signal frequency.

Plot-specific controls are prefixed with `--plot-`, for example `--plot-top-k`, `--plot-min-gain-db`, `--plot-save-pdf`, and `--plot-save-svg`.

## 3. Run a one-shot fixed-pump signal spectrum

`run_signal_spectrum.py` fixes one pump power and frequency, runs the gain-map backend on that pump point, and creates the standard candidate plots. It also evaluates and plots the pump-off port-1 traces S11, S21, S31, and S41 over the requested signal range.

```powershell
python workflows/run_signal_spectrum.py `
  --design designs/ipm_2c_fixed `
  --pump-power-dbm -24 `
  --pump-frequency-ghz 8 `
  --signal-start-ghz 5 `
  --signal-stop-ghz 9 `
  --signal-points 801 `
  --run-dir outputs/ipm_2c_signal_spectrum
```

Generated S-parameter files are:

- `port1_sparameters.npz`;
- `port1_sparameters.{png,pdf,svg}`.

The gain-map portion additionally produces `map_points.csv`, `map_arrays.npz`, `map_spectrum.npz` when enabled, `map_summary.json`, and the standard plot tree.

## Shared conventions

- Frequencies on the command line are in GHz unless the option says otherwise.
- Pump powers are in dBm.
- Circuit directories are normally under `designs/`; computational runs are normally under `outputs/`.
- Use `--help` on each workflow for the workflow-specific options. IPM and gain-map options are intentionally forwarded to their existing parsers.

