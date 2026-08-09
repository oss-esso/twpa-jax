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

### Complete option reference

The workflow-specific options are:

| Option | Type/default | Meaning |
| --- | --- | --- |
| `--design-dir PATH` | required | Output directory for the generated design, matrices, passive data, and figures. It is created automatically. |
| `--passive-start-ghz FLOAT` | `4.0` | First passive-analysis frequency. |
| `--passive-stop-ghz FLOAT` | `11.0` | Final passive-analysis frequency. |
| `--passive-points INT` | `1401` | Number of passive-analysis frequency points; must be at least 2. |
| `--passive-z0-ohm FLOAT` | `50.0` | Reference impedance for the passive S-parameters. |

All other options are forwarded to the IPM builder. The complete forwarded
option set is:

```text
# Output and coupler mode
--outdir PATH                         # overridden by --design-dir
--coupler-mode {cached,optimize,ideal}
--write-matrices                      # always enabled by this workflow
--draw

# Topology and discretization
--start-node-top INT
--start-node-bot INT
--ground INT
--array-length INT
--num-rows INT
--arrays-per-dc INT
--length-of-long-tl INT
--length-of-short-tl INT
--coupler-section-length INT
--len1 INT
--len2 INT
--len3 INT
--len4 INT

# Electrical parameters
--coupling-db FLOAT
--z0-ohm FLOAT
--coupler-freq-ghz FLOAT
--lj-ph FLOAT
--cj-ff FLOAT
--cg-ff FLOAT
--cl-per-um-ff FLOAT
--ll-per-um-ph FLOAT
--rleft-ohm FLOAT
--rright-ohm FLOAT
--rm-ohm FLOAT
--cell-length-um FLOAT

# Cached-coupler geometry overrides
--cached-coupler-width-um FLOAT
--cached-coupler-gap-um FLOAT
--cached-coupler-gap-to-ground-um FLOAT
--cached-coupler-length-um FLOAT

# Component scatter
--lj-scatter-sigma FLOAT
--lj-scatter-seed INT
--scatter-seed INT
--cj-scatter-sigma FLOAT
--cg-scatter-sigma FLOAT
--scatter-distribution {normal,uniform}
--lj-scatter-clip-min FLOAT
--lj-scatter-clip-max FLOAT
--cj-scatter-clip-min FLOAT
--cj-scatter-clip-max FLOAT
--cg-scatter-clip-min FLOAT
--cg-scatter-clip-max FLOAT

# Spatial profiles
--profile-json PATH
--lj-profile TEXT                     # repeatable
--cg-profile TEXT                     # repeatable
```

Integer topology options are counts or node indices. Options whose names end
in `-ghz`, `-um`, `-ff`, `-ph`, `-ohm`, or `-db` use those units directly.
The scatter clip values are multiplicative bounds, and `--lj-profile` and
`--cg-profile` may each be supplied more than once. Cached-coupler geometry
overrides are ignored when `--coupler-mode optimize` is selected.

The workflow always enables matrix output because the passive solver requires `C.npz`, `G.npz`, `K.npz`, `Bphi.npz`, and `ipm_arrays.npz`.

Generated files include:

- the normal IPM design files and `ipm_summary.json`;
- `passive_sparameters.npz`;
- `passive_s21_s24.{png,pdf,svg}`;
- `passive_s11_s21_s31_s41.{png,pdf,svg}`.

The passive convention is `S[frequency, output_port, source_port]`. The four directional traces in the second figure are all excited from port 1.

## 2. Run a gain map and generate its catalogue of plots

`run_gain_map_and_plots.py` forwards unrecognised options to `scripts/run_gain_map.py`, then invokes the standard plotting backend and the two signal-frequency projections.

Port roles are resolved from the circuit when omitted: four-port devices use
pump=4 and signal 1→2, two-port devices use 1→2, and one-port devices use
1→1. Use `--pump-port`, `--source-port`, or `--out-port` only when a design
uses a nonstandard assignment. Likewise, `--mixing-order auto` is the default:
nonzero external DC current/flux selects 3WM and an unbiased circuit selects
4WM; explicit `--mixing-order 3` or `4` remains available.

```powershell
python workflows/run_gain_map_and_plots.py `
  --design designs/ipm_2c_fixed `
  --run-dir outputs/ipm_2c_gain_map `
  --n-power 5 `
  --n-frequency 5 `
  --pump-power-min-dbm -30 `
  --pump-power-max-dbm -20 `
  --pump-freq-min-ghz 7 `
  --pump-freq-max-ghz 8 `
  --log-level INFO
```

The workflow forces the in-process executor and writes the usual map artifacts under the run directory, followed by plots under `run-dir/plots/`:

- simple gain, status, and runtime maps;
- spectrum-fit maps when signal spectra are available: peak gain, GBP, ripple, smoothness, and selected candidates;
- candidate tables and candidate spectra;
- gain versus pump frequency/signal frequency;
- gain versus pump power/signal frequency.

Plot-specific controls are prefixed with `--plot-`, for example `--plot-top-k`, `--plot-min-gain-db`, `--plot-save-pdf`, and `--plot-save-svg`.

The KIMPA wrapper uses the same role and mixing-order resolution. Its standard
501-point best-spectrum output is `best_point_spectrum.npz`, containing the
signal-frequency `Z`, `Y`, and pump-off `S` matrices, the pumped signal matrix,
the idler conversion matrix, and `quantum_efficiency` / `quantum_efficiency_ideal`.
The CSV contains the corresponding per-frequency QE values and named `sij`
fields for every port pair present. A one-port KIMPA fixture therefore emits
reflection data; a two- or four-port fixture emits the applicable `S21` and
other matrix entries without changing the workflow.

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

## 4. Measure compression (P1dB)

`scripts/run_compression.py` holds the pump fixed and sweeps signal power until the gain drops 1 dB below its small-signal value. Unlike the gain map, the signal is not a perturbation here: the sidebands act back on the pump, so the pump and signal are solved together on a multitone basis.

Against a built-in reference device, which needs no circuit directory:

```powershell
python scripts/run_compression.py `
  --fixture jtwpa `
  --signal-ghz 6.6 `
  --output-dir outputs/compression_jtwpa
```

Against a real design:

```powershell
python scripts/run_compression.py `
  --circuit-dir designs/ipm_2c_fixed `
  --signal-ghz 7.44 `
  --pump-freq-ghz 7.1 `
  --pump-port 4 `
  --multitone-backend schur_cpu_mt `
  --output-dir outputs/compression_2c
```

Signal frequency is mandatory for a single run. Fixtures default to zero line attenuation; loaded circuits use the measured loss model unless `--attenuation-db` is given explicitly.

### Choosing a basis

`--multitone-basis` selects how the tone set is built:

- `matched` (default) retains the pump harmonics alongside the signal sidebands. Use this.
- `three_tone` is only valid with a fundamental-only pump basis. The driver raises rather than silently dropping a pump harmonic.
- `lattice` is the convergence-study basis.

`--multitone-sidebands` sets the basis size. Memory scales as `(n_pump_modes + 2S + 1)^2`, not as the packed dimension, because the coupled Jacobian is block-dense in tone index. Roughly 2.8 GB per worker at S=10, 1.6 GB at S=6, 0.9 GB at S=2. `--signal-workers` is capped automatically against both `--resource-budget-gb` and actual free memory.

### Reading the output

The driver emits both a refined `p1db` and an interpolated `p1db_interpolated_dbm` from the same sweep, so the difference between the two methods is a single-variable comparison. `--p1db-power-tol-db 0` falls back to interpolation only.

Two diagnostics need care. `manley_rowe_rel_err` is meaningful only in the conversion scope (pump/signal/idler); the all-tone variant is not a valid invariant and must never be used as a gate. `stability_status` stays `NOT_CHECKED` unless you pass `--check-stability` — a deep-saturation solution without it is not a stability claim, and any exponent it does report should be quoted against `omega_p` rather than in bare s^-1.

## 5. Compare a map against measurement

The Themis measurement cubes ship under `docs/development/`. Two scripts consume them.

`compare_map_to_measurement.py` overlays a simulated map on the measured peak-gain and collapse-power envelope, aligned by hand-supplied calibration offsets.

`align_map_to_measurement.py` instead *fits* those offsets as nuisance parameters. The model is `G_meas(f,P) ~= G_sim(f-df, P-dP) + dG`; for weighted least squares `dG` is analytic per `(df,dP)`, so only a two-dimensional grid search over frequency and power shift remains. It masks non-overlapping and failed cells, weights the amplified ridge above the flat background, and writes JSON plus a four-panel figure: measurement, aligned simulation, residual, and the loss surface itself.

```powershell
python scripts/align_map_to_measurement.py --help
```

Fit one band at a time with `--fit-freq-ghz` / `--fit-power-dbm`. A whole-map fit has to compromise between comb lobes and produces a frequency-elongated, weakly identified basin; per-section fits give a compact single minimum. `--min-overlap-frac` (default 0.25) rejects tiny-overlap corner fits, which would otherwise win on local residual alone.

## Shared conventions

- Frequencies on the command line are in GHz unless the option says otherwise.
- Pump powers are in dBm.
- Circuit directories are normally under `designs/`; computational runs are normally under `outputs/`.
- Use `--help` on each workflow for the workflow-specific options. IPM and gain-map options are intentionally forwarded to their existing parsers.
- Long campaigns should be pruned before archiving: `python scripts/prune_map_solutions.py <run-dir> --top-k 100 --purge-point-dirs --apply`.

