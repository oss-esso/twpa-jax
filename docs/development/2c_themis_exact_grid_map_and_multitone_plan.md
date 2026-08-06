# 2c Themis exact-grid map, shift fit, and multitone follow-up

This is the next discriminator for the early-compression investigation. It
runs `designs/ipm_2c_fixed` on the two archived Themis pump grids, estimates
the local frequency/power translation between simulation and measurement, and
then uses the finite-signal multitone worker on a small shifted section.

The `.npy` files contain 51 pump-frequency files and 31 pump-power points per
file. The frequency axes are not exact `linspace` grids, so the map runner has
an explicit `--grid-from-measurement-dir` option. It loads the numeric filename
frequencies and the `PumpPower` axis directly and stores those exact arrays in
`map_arrays.npz`.

## 1. Run the two exact-grid maps

Run these from the repository root. The first is the broad/coarse map; the
second is the localized/fine map. `--frequency-chunk-size 0` is intentional:
the nonuniform frequency grid must remain intact in the one map process.

### Coarse, broad grid: 14.18.08

```powershell
python workflows/run_gain_map_and_plots.py `
  --design designs/ipm_2c_fixed `
  --run-dir outputs/ipm_2c_fixed_themis_coarse_14.18.08 `
  --grid-from-measurement-dir docs/development/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK `
  --frequency-chunk-size 0 `
  --no-signal-spectrum `
  --log-level INFO
```

Expected pump axes: 51 frequencies from 5.980 to 7.997 GHz, and 31 powers
from -29.08 to -19.0266666667 dBm. The frequency steps alternate between
0.040 and 0.041 GHz.

### Fine, localized grid: 17.03.10

```powershell
python workflows/run_gain_map_and_plots.py `
  --design designs/ipm_2c_fixed `
  --run-dir outputs/ipm_2c_fixed_themis_fine_17.03.10 `
  --grid-from-measurement-dir docs/development/17.03.10_Themis_SetupAug25_noVTS_transmission_15mK `
  --frequency-chunk-size 0 `
  --no-signal-spectrum `
  --log-level INFO
```

Expected pump axes: 51 frequencies from 7.043 to 7.373 GHz, and 31 powers
from -25.96 to -16.9466666667 dBm. The frequency steps alternate between
0.006 and 0.007 GHz.

Do not replace `--grid-from-measurement-dir` with only min/max/count: that
would make a nearby interpolated grid, not the measured grid. Each run should
leave `map_arrays.npz`, `map_points.csv`, `map_summary.json`, and the plots in
its run directory. In particular, verify that
`map_arrays.npz[pump_frequency_ghz]` and `map_arrays.npz[pump_power_dbm]`
match the corresponding measurement axes before fitting.

## 2. Prompt to construct and run the local shift fit

After both map commands finish, give the following prompt to the coding agent:

```text
Construct and run a local simulation-to-Themis shift-fitting script for the
two completed maps:

  coarse map: outputs/ipm_2c_fixed_themis_coarse_14.18.08
  fine map:   outputs/ipm_2c_fixed_themis_fine_17.03.10

Measurements:

  docs/development/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK
  docs/development/17.03.10_Themis_SetupAug25_noVTS_transmission_15mK

First inspect scripts/run_gain_map.py, scripts/align_map_to_measurement.py,
and the map_arrays.npz files. Do not assume an endpoint/count linspace, and do
not silently compare different gain observables.

The map's default signal is signal = pump_frequency - 0.100 GHz. Build the
measurement observable consistently: for every Themis file and every
PumpPower row, interpolate Response along its 4 MHz signal-frequency axis at
that file's pump frequency minus 0.100 GHz. Also produce the existing
signal-band peak observable as a diagnostic, but do not use it as the primary
fit unless you demonstrate that it is the same observable as the simulation
map.

Fit each map locally, separately, with a two-dimensional coordinate shift and
an additive gain offset. Define the sign convention explicitly, for example:

  measurement(f, P) ~= simulation(f + df, P + dP) + dG.

Use interpolation on the saved simulation grid, restrict the fit to the
overlap after shifting, use a robust loss or a gain-ridge ROI so the large
low-gain background does not dominate, and report uncertainty/sensitivity by
repeating the fit with reasonable local ROIs. Start with bounds df = +/-1.5
GHz and dP = +/-6 dB, tightening them only if the overlap or the data require
it. Fit the fine map over its localized window and the coarse map over the
broad window; also report a common-window fit around their overlap if useful.

Implement the script under scripts/ with a focused test for the coordinate
sign, interpolation, and nonuniform axes. Run it on both map/measurement pairs
and write machine-readable JSON plus CSV/PNG diagnostics under:

  outputs/ipm_2c_shift_fit/

The report must include, for each map, df in GHz, dP in dBm, dG in dB, the
fit window/ROI, overlap fraction, RMSE or robust loss, peak-gain comparison,
and whether the inferred shifts are consistent between coarse and fine maps.
Do not launch any multitone runs yet. Finish by printing the exact fit-output
paths and a concise recommendation for the shifted multitone section.
```

The useful result is the fitted coordinate translation, not just the best
overlap score. Keep the sign convention in the output because it determines
which coordinates are passed to the multitone worker.

## 3. Prompt for the shifted multitone section

Use this after reviewing the fit JSON. The current
`scripts/run_compression.py` worker supports a pump-current list and a signal
frequency sweep, but the shifted section is a two-dimensional set of
pump-frequency/pump-power cells. Therefore ask the agent to decide whether a
small orchestrator is needed rather than manually inventing a large command.

```text
Using the completed local-fit outputs under outputs/ipm_2c_shift_fit/ and the
two exact-grid gain maps, prepare the finite-signal multitone validation.

Inspect scripts/run_compression.py and its tests first. Use the same circuit,
ports, attenuation model, Z0, Norton power convention, pump basis, and signal
detuning as the gain-map run. Do not apply the fitted shift twice.

For each dataset, select a small local section around the measured gain ridge
(start with at most a 3x3 or 5x5 section, and state the selected measurement
frequency/power cells). Map those measured coordinates into simulation
coordinates using the fitted df and dP, with the sign convention stored in the
fit JSON. Convert the shifted pump-power coordinates to the pump currents
expected by run_compression.py using the same frequency-dependent attenuation
and Norton relation as run_gain_map.py. Use signal = shifted pump frequency -
0.100 GHz unless the map metadata says otherwise.

If the existing worker can express the selected cells without ambiguity,
output copy-pastable PowerShell commands for the runs, with one resumable
output directory per map/section and a summary path. If it cannot express a
two-dimensional section, implement a small resumable orchestrator under
scripts/ that calls the existing multitone worker for each selected cell; add
a focused smoke test, then output the exact PowerShell commands that invoke
the orchestrator. Do not run the expensive multitone campaign in this step.

For every selected cell, compare the multitone small-signal result against the
linear gain-map value at the same shifted coordinate and against the
measurement observable. Report gain residuals, convergence status, and the
power/frequency coordinates actually used. Include a no-shift control for at
least one cell so that a falsely good result caused by a coordinate/sign error
is detectable. Keep the first campaign small; only expand it if the linear
regime agrees within the stated numerical tolerance.
```

The multitone comparison should be made at low signal power first. Its purpose
here is to validate the linearized gain calculation after the coordinate shift;
compression behavior should be tested only after that check passes.
