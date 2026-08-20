# Gain-map workflows

The canonical entry point is `workflows/run_gain_map_and_plots.py`.

`--fast` runs the ordinary production HB map and is the default when neither mode
flag is supplied. `--slow` runs the bounded HB-recovery plus transient physical
boundary controller. Slow mode does not integrate every point: it uses HB for the
bulk of a column and invokes TD only after the bounded HB path reaches an
obstruction.

`--design` takes a **generated circuit directory**, not a YAML source. Build
one first with `workflows/build_design_and_passive.py` (see
[`workflows.md`](workflows.md) §2); the examples below assume
`outputs/ipm_2c_passive`. `--outdir` is an accepted alias for `--run-dir`.

Examples:

```powershell
python workflows/run_gain_map_and_plots.py --fast `
  --design outputs/ipm_2c_passive --n-frequency 20 --n-power 20 `
  --frequency-workers 4 --outdir outputs/gain_maps/fast_2c

python workflows/run_gain_map_and_plots.py --slow `
  --design outputs/ipm_2c_passive --n-frequency 20 --n-power 20 `
  --frequency-workers 4 --outdir outputs/gain_maps/slow_2c
```

Multiple compiled design directories can be supplied to `--design`. The
workflow processes them sequentially and writes each result below the output
root using the input directory name:

```powershell
python workflows/run_gain_map_and_plots.py --fast `
  --design outputs/passive_batch/ipm_2c outputs/passive_batch/ipm_3c `
  --outdir outputs/gain_maps/batch
```

All generated files belong below the selected output root. Compact summaries use
the same raw and physically eligible coverage accounting in both modes. Physical
boundary, transient numerical failure, and unresolved statuses remain distinct.
Frequency workers are isolated processes; power points remain sequential within a
column. Completed column subtrees are reusable on restart. In fast mode, per-point
pump solutions are retained while the map and all candidate plots are generated,
then removed after plotting completes.

For a bounded high-power 2c column diagnostic, use the explicit recovery ladder
on the baseline-identical circuit and leave pump attenuation unspecified:

```powershell
python scripts/run_gain_map.py --mode warmstart --executor inprocess `
  --circuit-dir outputs/ipm_2c_passive --column-recovery-ladder `
  --inproc-fail-fast `
  --n-frequency 1 --pump-freq-min-ghz 7.9 --pump-freq-max-ghz 7.9
```

Omitting `--attenuation-db` selects the configured frequency-dependent pump loss model.
The output CSV records `DIRECT`, `POWER_SUBSTEP`, `ARCLENGTH_RECOVERY`,
`FREQUENCY_RECOVERY`, `FAILED_NUMERICAL`, and
`PAST_CONNECTED_BRANCH_BOUNDARY` routes separately.

For an exhaustive high-power 2c campaign, use `--high-power-recovery`:

```powershell
python scripts/run_gain_map.py --mode warmstart --executor inprocess `
  --circuit-dir outputs/ipm_2c_passive --high-power-recovery `
  --n-frequency 1 --pump-freq-min-ghz 7.9 --pump-freq-max-ghz 7.9 `
  --pump-power-min-dbm -21 --pump-power-max-dbm -16 `
  --outdir outputs/2c_high_power_79
```

Leave `--attenuation-db` absent. The measured frequency-dependent pump loss is
then used. The high-power path continues trying higher-power points after a
failed Newton solve and does not interpret that failure as a physical boundary.
Its bounded recovery sequence is direct Newton, adaptive power continuation,
local PALC, pseudo-transient continuation, and a nearby-frequency detour.

Every attempted state records the full reconstructed residual, the strongest
junction, `|I|/Ic`, the minimum Josephson tangent margin, and the dominant
omitted residual modes. The high-power default full-residual gate is `1e-7` and
can be changed explicitly with `--pump-full-residual-gate`.

The current ideal sine-junction model does not contain quasiparticle switching
or physical junction damage. Therefore `|I|/Ic` is a diagnostic of the modeled
branch, not a simulated breakdown mechanism. A failed point below the observed
Ic utilization remains a numerical or unresolved point until a validated HB or
TD state establishes its status.

## Measured 2c verification, and the build it was measured on

The verification below was run on a **6,136-node / 16,312-element 2c build**
with 2,508 junctions, using the measured attenuation profile. That is the
`--coupler-mode optimize` build. A 2c design built with `auto` — the
design-file default — has 6,096 nodes and 16,192 elements instead, because
`optimize` emits a 20-cell-longer directional coupler. The two agree to about
1% of input power in pump-off S-parameters (see
[`circuit_builders.md`](circuit_builders.md#two-2c-builds)), but the numbers
below have not been re-measured on the default build, so quote them with the
coupler mode attached.

That run reached 7.9 GHz at -19.625 dBm with a validated full residual of
`1.14e-11` and strongest-junction utilization `0.8274`. The next tested point,
-18.5625 dBm, remained unresolved after adaptive power continuation, PALC,
harmonic promotion through mode 35, nearby-frequency recovery, and a bounded
pseudo-transient step-size portfolio. Its best enriched iterate had full
residual `5.7319e-2` at utilization `0.99999956`; it is recorded as a
numerical failure, not a physical boundary.

The netlist provenance check for that run gave SHA-256
`7F1A4746001A15B6DE527902A23281F0C41D42804CE4A6D4C79FA8171C997BD6` for both
`outputs/2c_netlist_baseline.csv` and `designs/ipm_2c_fixed/elements.csv`
(1,014,288 bytes each). **Neither file is recoverable from a checkout** —
`outputs/` and `designs/ipm_2c_fixed/` are both gitignored — so this digest
identifies one particular local build rather than gating anything today. The
tracked parity reference for the default build is
`tests/data/ipm_2c_reference/`.

To inspect the current in every junction of a validated pump checkpoint:

```powershell
python scripts/plot_junction_current_profile.py `
  --circuit-dir outputs/ipm_2c_passive `
  --pump-dir <validated-pump-directory> `
  --outdir outputs/junction_profile_2c
```

The command writes `junction_current_profile.csv` and
`junction_current_profile.png`. The profile uses the maximum absolute
instantaneous current of each junction over the reconstructed HB waveform; the
map scalar is simply the maximum of the resulting per-junction `peak_ratio_ic`
column.
