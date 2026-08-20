# Fab-team workflow guide

This is the main operating document for the package. It describes what to
write, which command to run, which options to use, and where to find the
results.

The normal user flow is:

```text
1. Select or edit a Python Circuit design or a YAML design.
2. Build the design and check the pump-off passive response.
3. Run a fast gain map at moderate pump power.
4. Use the slow map only when a physical-boundary check is required.
5. Archive the YAML, generated circuit directory, command line, and outputs.
```

The README contains installation instructions and current limitations. The
YAML fields are explained in [`design_format.md`](design_format.md). Python
designs are assembled through `twpa_solver.circuit.Circuit`; both authoring
routes share `designs/technology/*.yaml` component defaults. See
[`circuit_builders.md`](circuit_builders.md) for the three-layer authoring
model and builder resolution order.

## 1. Files and directories

Keep source and generated files separate:

```text
designs/                         source YAML and technology presets
outputs/my_design/               generated circuit and passive response
outputs/my_gain_map/             gain-map data and plots
```

Use `designs/` for files that a person edits. Use `outputs/` for generated
files. A generated directory can be deleted and rebuilt from its YAML.

**No circuit directory is checked in.** `designs/` tracks YAML designs,
`designs/python/`, and `designs/technology/` only. Every command below that
takes a circuit directory takes one you built yourself, normally under
`outputs/`. A `designs/ipm_2c_fixed` present on a machine is a local, ignored
build artifact whose contents depend on the coupler mode it was built with;
see [`circuit_builders.md`](circuit_builders.md#two-2c-builds).

Do not edit `C.npz`, `G.npz`, `K.npz`, `Bphi.npz`, `arrays.npz`, or generated
CSV/JSON files by hand.

## 2. Build a design and passive response

This is the first command for a new or modified design. It compiles the YAML,
writes the solver matrices, calculates the pump-off response, and creates the
standard passive plots.

For a declarative YAML design:

```powershell
python workflows/build_design_and_passive.py `
  --design designs/ipm_2c.yaml `
  --output-dir outputs/ipm_2c_passive
```

Build several YAML designs sequentially by using one output root. Each design
is written to a subdirectory named after its YAML file:

```powershell
python workflows/build_design_and_passive.py `
  --design designs/ipm_2c.yaml designs/ipm_3c.yaml `
  --output-dir outputs/passive_batch
```

Omitting `--design` selects the **legacy IPM builder** instead of the
declarative compiler. This does not read an existing directory: it builds a
fresh IPM design from `IPMParams` defaults plus any legacy override flags, and
writes it into `--output-dir`.

```powershell
python workflows/build_design_and_passive.py `
  --output-dir outputs/ipm_2c_legacy `
  --coupler-mode auto
```

Which parser receives the unrecognised options depends entirely on this choice.
See [Forwarded options](#forwarded-options) below; the two flag sets are
disjoint and mixing them raises.

For a Python Circuit design, import the design module and compile the returned
`Circuit` directly. For example, the technology-driven IPM entry point is
`designs/python/ipm_2c.py:build_ipm_2c`; use
`build_ipm_2c().compile(node_numbering="legacy")` when published branch
indices must remain stable.

### Workflow options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--design PATH [PATH ...]` | none | One or more declarative YAML sources. Use together with `--output-dir`; with multiple inputs, that path is the output root. |
| `--output-dir PATH` (`--design-dir` alias) | none | Output root for generated circuits. Batched inputs are written to one subdirectory per YAML stem. |
| `--passive-start-ghz VALUE` | `4.0` | First passive-response frequency. |
| `--passive-stop-ghz VALUE` | `11.0` | Last passive-response frequency. |
| `--passive-points INT` | `1401` | Number of passive frequency points. Must be at least 2. |
| `--passive-z0-ohm VALUE` | `50.0` | Reference impedance for S-parameters. |

The passive frequency grid uses the old standard of 1,401 points unless you
change `--passive-points`.

<a id="forwarded-options"></a>
### Forwarded options

Unrecognised options are forwarded, but **not always to the same program**:

| Invocation | Options go to |
| --- | --- |
| with `--design` (YAML) | the declarative compiler, `python -m twpa_solver.design` |
| without `--design` | the legacy IPM builder, `twpa_solver.builders.ipm` |

The two flag sets are disjoint. Passing a builder flag on the YAML path, or a
compiler flag on the builder path, is an argument error rather than a silent
no-op. Put the forwarded options after the workflow options.

#### With `--design`: declarative compiler

This is the complete set the compiler accepts.

| Option | Meaning |
| --- | --- |
| `--write-matrices` | Enabled by this workflow. Writes solver matrices. |
| `--coupler-mode auto\|ideal\|optimize` | Override the YAML/technology coupler mode. `auto` is recommended. |
| `--overwrite` | Replace files in a non-empty generated directory. |
| `--strict` | Additionally require that every declared `parameters` entry is referenced. Recommended — it catches a typo in a parameter name. Schema validation itself always runs. |
| `--profile-json PATH` | Legacy profile file. Prefer YAML `profiles`. |
| `--lj-profile TEXT` | Legacy Lj profile override; repeatable. |
| `--cg-profile TEXT` | Legacy Cg profile override; repeatable. |
| `--lj-scatter-sigma VALUE` | Multiplicative Lj scatter. |
| `--cj-scatter-sigma VALUE` | Cj scatter. |
| `--cj-scatter-mode independent\|plasma_locked` | Cj scatter rule. |
| `--cg-scatter-sigma VALUE` | Cg scatter. |
| `--scatter-seed INT` | Reproducible scatter seed. |
| `--tan-delta VALUE` | Global dielectric loss tangent. |
| `--tan-delta-role ROLE=VALUE` | Role-specific loss tangent; repeatable. |

Geometry, node numbering, and nominal component values are **not** overridable
here. Set them in the YAML design or its technology preset. There is no
`--draw`, `--lj-ph`, `--ground`, or `--len1` on this path.

#### Without `--design`: legacy IPM builder

| Option | Meaning |
| --- | --- |
| `--write-matrices` | Enabled by this workflow. Writes solver matrices. |
| `--coupler-mode auto\|ideal\|optimize` | Coupler model selection. |
| `--draw` | Write the optional builder geometry drawing. |
| `--start-node-top INT`, `--start-node-bot INT` | Starting node numbers. |
| `--ground INT` | Ground node number. |
| `--jtl-cells-per-array INT`, `--jtl-row-count INT`, `--jtl-rows-per-coupler INT` | JTL cell, row, and coupler spacing values. `--array-length`, `--num-rows`, and `--arrays-per-dc` remain accepted as aliases. |
| `--inter-array-cpw-cells INT`, `--signal-inter-coupler-cpw-cells INT`, `--pump-inter-coupler-cpw-cells INT` | CPW routing lengths in cells. `--length-of-short-tl`, `--length-of-long-tl`, and `--coupler-section-length` remain accepted as aliases, in that order. |
| `--signal-input-cpw-cells INT`, `--signal-output-cpw-cells INT`, `--pump-input-cpw-cells INT`, `--pump-output-cpw-cells INT` | Signal and pump input/output CPW lengths. `--len1`–`--len4` remain accepted as aliases, in that order. |
| `--coupling-db VALUE`, `--z0-ohm VALUE`, `--coupler-freq-ghz VALUE` | Coupler target, impedance, and design frequency. |
| `--lj-ph VALUE`, `--cj-ff VALUE`, `--cg-ff VALUE` | Junction and ground values. |
| `--cl-per-um-ff VALUE`, `--ll-per-um-ph VALUE`, `--cell-length-um VALUE` | Transmission-line constants. |
| `--rleft-ohm VALUE`, `--rright-ohm VALUE`, `--rm-ohm VALUE` | Termination resistances. |
| `--profile-json PATH`, `--lj-profile TEXT`, `--cg-profile TEXT` | Profile inputs; the two shorthand flags are repeatable. |
| `--lj-scatter-sigma VALUE`, `--cj-scatter-sigma VALUE`, `--cg-scatter-sigma VALUE` | Per-component scatter. |
| `--cj-scatter-mode independent\|plasma_locked` | Cj scatter rule. |
| `--scatter-distribution normal\|uniform` | Scatter distribution. |
| `--scatter-seed INT` | Reproducible master scatter seed. |
| `--lj-scatter-seed INT` | Deprecated Lj-only alias for `--scatter-seed`. |
| `--lj-scatter-clip-min VALUE`, `--lj-scatter-clip-max VALUE` | Lj scatter limits. |
| `--cj-scatter-clip-min VALUE`, `--cj-scatter-clip-max VALUE` | Cj scatter limits. |
| `--cg-scatter-clip-min VALUE`, `--cg-scatter-clip-max VALUE` | Cg scatter limits. |
| `--tan-delta VALUE`, `--tan-delta-role ROLE=VALUE` | Dielectric loss tangent, global and per role. |

This path has no `--overwrite` and no `--strict`.

For normal fab work, put nominal values, profiles, and intentional scatter in
the YAML and use the `--design` path. Use command-line overrides only for a
temporary comparison.

### Passive output files

The output directory contains, depending on the available ports:

- `design_resolved.json`: fully expanded design and final values;
- `design_summary.json`: short build summary;
- `elements.csv`: generated element list;
- `ports.csv`: port numbers and node numbers;
- `C.npz`, `G.npz`, `K.npz`, `Bphi.npz`: solver matrices;
- `arrays.npz` and `ipm_arrays.npz`: solver arrays;
- `passive_sparameters.npz`: passive traces;
- `passive_s21_s24.*`: preferred signal/pump response plots;
- `passive_s11_s21_s31_s41.*`: response from the standard input ports.
- `coupler_passive_sparameters.npz`: isolated four-port coupler traces;
- `coupler_passive_s11_s21_s31_s41.*`: isolated coupler response from port 1.

The S-parameter array convention is:

```text
S[frequency_index, output_port_index, source_port_index]
```

Check the passive response before spending time on a gain map. A wrong port,
wrong design directory, or wrong frequency range is usually visible here.

## 3. Fast gain map

The fast map runs pump harmonic balance and small-signal gain over a pump-power
and pump-frequency grid. It is the normal production workflow for moderate
pump power.

```powershell
python workflows/run_gain_map_and_plots.py `
  --fast `
  --design outputs/ipm_2c_passive `
  --output-dir outputs/ipm_2c_gain `
  --n-power 20 `
  --n-frequency 20 `
  --pump-power-min-dbm -35 `
  --pump-power-max-dbm -23 `
  --pump-freq-min-ghz 7.5 `
  --pump-freq-max-ghz 8.5
```

Run several already-built design directories sequentially by using one run
root. Each map and its plots are written to a subdirectory named after the
input directory:

```powershell
python workflows/run_gain_map_and_plots.py `
  --fast `
  --design outputs/passive_batch/ipm_2c outputs/passive_batch/ipm_3c `
  --output-dir outputs/gain_batch
```

Your flags are forwarded to `scripts/run_gain_map.py`.

### What `--fast` sets for you

The wrapper prepends the production engine settings below to every run. Each
is applied **only if you did not pass that flag yourself**, so any single one
can still be overridden on the command line. These are not the
`run_gain_map.py` parser defaults; the table gives both so the difference is
visible.

| Injected flag | `run_gain_map.py` default | Effect |
| --- | --- | --- |
| `--inproc-fail-fast` | off | **changes behaviour** — stop a bad cell instead of grinding |
| `--fold-skip-patience 2` | `0` | **changes behaviour** — enables fold-skip counting |
| `--signal-detuning-mhz 150` | `100` | **changes results** — wider signal offset |
| `--no-signal-spectrum` | spectrum **on** | **changes cost** — the default sweeps signal frequency at *every* grid point and dominates wall time |
| `--log-level INFO` | `DEBUG` | **changes cost** — DEBUG logs are enormous on a 400-point map |
| `--mode warmstart` | `warmstart` | explicit only |
| `--inproc-pump-backend schur_cpu_mt` | `schur_cpu_mt` | explicit only |
| `--inproc-preconditioner real_coupled_fast` | `real_coupled_fast` | explicit only |
| `--inproc-fold-predictor secant` | `secant` | explicit only |
| `--pump-current-jc-scale 1.0` | `1.0` | explicit only; `2.0` would be a 6 dB error |

Only the first five change anything. The rest restate the parser default and
are pinned so a later default change cannot move a production map silently.
Note that the `--help` text for `--inproc-preconditioner` and
`--inproc-pump-backend` still names the *old* defaults (`mean_tangent`,
`full`); read the table above, not that help string.

The wrapper additionally forces `--no-compact-output` so per-point pump
solutions survive until plotting finishes, then deletes them itself.

`--frequency-workers` and `--frequency-chunk-size` are sized automatically
when you do not pass them: the worker count is derived from the measured
per-worker footprint of this circuit against half of currently free physical
RAM, then capped by CPU count and by `--n-frequency`. The chosen values are
printed as a `fast-map worker sizing:` line. Passing `--frequency-workers`
explicitly disables the estimate and pins the chunk size to it.

**`--fold-skip-patience 2` has previously culled the 2c broadband operating
region into a needle-only map when skip counting false-tripped.** Check the
PASS coverage of a sparse map before quoting it.

### Port selection

Port flags are optional. The workflow reads the ports in the circuit:

| Available circuit | Automatic roles |
| --- | --- |
| 1 port | pump=1, signal input=1, signal output=1 |
| 2 ports | pump=1, signal input=1, signal output=2 |
| 4 ports | pump=4, signal input=1, signal output=2 |

Use explicit flags only for a non-standard circuit:

```powershell
--pump-port 4 --source-port 1 --out-port 2
```

### 3WM and 4WM selection

The default is `--mixing-order auto`:

- external DC current, external flux, a DC solution, or design bias metadata
  selects 3WM;
- an unbiased circuit selects 4WM.

Use `--mixing-order 3` or `--mixing-order 4` to override automatic selection.
For a biased 3WM design, use a dense pump basis, for example:

```powershell
--mixing-order 3 --pump-mode-policy dense_real --harmonics 3
```

For an unbiased 4WM design, the normal basis is:

```powershell
--mixing-order 4 --pump-mode-policy positive_odd_jc --pump-mode-count 10
```

### Fast workflow options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--design PATH [PATH ...]` (`--design-dir` alias) | required | One or more generated circuit directories. With multiple inputs, `--output-dir` is the output root. |
| `--output-dir PATH` (`--run-dir`, `--outdir` aliases) | `outputs/gain_map_workflow` | Output root. Batched inputs are written to one subdirectory per design-directory name. |
| `--fast` | off | Run the standard fast map. This is the default if `--slow` is absent. |
| `--slow` | off | Use the HB-to-transient boundary workflow instead. |
| `--plot-top-k INT` | `5` | Number of candidate plots. |
| `--plot-min-gain-db VALUE` | `10` | Minimum gain for candidate selection. |
| `--plot-save-pdf` | off | Save PDF plots. |
| `--plot-save-svg` | off | Save SVG plots. |

### Gain-map grid and power options

| Option | Meaning |
| --- | --- |
| `--n-power INT` | Number of pump-power points. |
| `--n-frequency INT` | Number of pump-frequency points. |
| `--pump-power-min-dbm VALUE` | Lowest external pump-power coordinate. |
| `--pump-power-max-dbm VALUE` | Highest external pump-power coordinate. |
| `--pump-freq-min-ghz VALUE` | Lowest pump frequency. |
| `--pump-freq-max-ghz VALUE` | Highest pump frequency. |
| `--grid-from-measurement-dir PATH` | Use the exact measurement axes from a measurement directory. |
| `--attenuation-db VALUE` | Flat line attenuation. If omitted, use the configured measured-loss model. |
| `--signal-attenuation-db VALUE` | Flat signal-line attenuation for spectrum referral. |
| `--z0-ohm VALUE` | Reference impedance. |
| `--power-convention legacy_traveling_wave\|norton` | Converts dBm to injected peak current. Keep the default unless a calibration explicitly requires the alternate convention. |
| `--column-recovery-ladder` | Opt-in bounded column recovery: direct HB, adaptive power substeps, local pseudo-arclength, then nearby-frequency detour. It records the route and does not classify HB failure as physics. |

### Signal and pump options

| Option | Meaning |
| --- | --- |
| `--signal-ghz VALUE` | Fixed signal frequency. |
| `--signal-detuning-mhz VALUE` | Signal offset below the pump when `--signal-ghz` is not set. |
| `--dc-current-a VALUE` | Uniform external DC current. Nonzero value selects 3WM in `auto` mode. |
| `--dc-solution PATH` | Previously solved DC state. |
| `--dc-branch-flux-over-phi0 VALUE` | External reduced flux. Nonzero value selects 3WM in `auto` mode. |
| `--signal-spectrum` | Write per-map-cell signal spectra. |
| `--no-signal-spectrum` | Disable per-map-cell signal spectra. The fast wrapper uses this by default to reduce storage. |
| `--signal-offset-start-mhz VALUE` | First spectrum offset. |
| `--signal-offset-step-mhz VALUE` | Spectrum offset spacing. |
| `--signal-offset-count-per-side INT` | Number of offsets on each side. |
| `--signal-workers INT` | Parallel signal-spectrum workers. |
| `--signal-backend direct\|schur` | Signal linear solver backend. |
| `--signal-solver superlu\|pardiso` | Sparse factorisation backend. |
| `--skip-baselines` | Skip baseline solves when using the Schur signal backend. |
| `--pump-mode-policy VALUE` | Pump harmonic basis policy. Use `dense_real` for biased 3WM and `positive_odd_jc` for unbiased 4WM. |
| `--pump-mode-count INT` | Number of modes for `positive_odd_jc`. |
| `--harmonics INT` | Number of dense harmonics. |
| `--nt INT` | Pump time-grid size. Increase only for a controlled convergence check. |
| `--sidebands INT` | Small-signal sideband count. |
| `--gamma-nt INT` | Time-grid size used to calculate pump conversion coefficients. |
| `--pump-current-jc-scale VALUE` | Pump-current scale. Use `1.0` for the validated production current convention. |

### Execution and storage options

| Option | Meaning |
| --- | --- |
| `--executor inprocess\|subprocess` | Execution mode. `inprocess` is the normal fast path. |
| `--loss-model VALUE` | Loss interpretation. `auto` is recommended. |
| `--mode cold\|warmstart\|both` | Start every point cold, warm-start from neighbouring points, or run both. |
| `--frequency-chunk-size INT` | Number of frequency columns per worker chunk. In fast mode, the default equals the automatically selected worker count; `0` disables chunking. |
| `--frequency-workers INT` | Number of frequency chunks run in parallel. In fast mode, the default assigns at most half of currently free physical RAM to workers using the measured resource footprint. |
| `--local-traversal-chunks` | Allow independent chunks for non-column traversals. |
| `--resume-chunks` / `--no-resume-chunks` | Reuse complete chunk outputs or recalculate them. |
| `--compact-output` | Remove ordinary per-point pump arrays after the map is complete. |
| `--overwrite` | Replace the existing run directory. |
| `--log-level CRITICAL\|ERROR\|WARNING\|INFO\|DEBUG` | Amount of log output. |
| `--python-executable PATH` | Python executable for subprocess workers. |
| `--pump-timeout-s VALUE`, `--gain-timeout-s VALUE` | Subprocess time limits. |
| `--allow-superlu-fallback` | Allow the debug sparse-solver fallback. |
| `--log-factor-backend` | Report the sparse factorisation backend. |

### Advanced continuation options

Use these only when a normal warm-start map cannot reach the required pump
range:

| Option | Meaning |
| --- | --- |
| `--traversal column\|backbone\|nearest\|serpentine\|floodfill` | Grid traversal order. `column` is the normal order. |
| `--backbone-direction ltr\|rtl\|center_out\|two_ended` | Starting order for a backbone traversal. |
| `--predictor copy\|power_secant\|freq_secant\|corner\|plane\|portfolio` | Initial-state predictor. |
| `--portfolio-policy best\|ranked` | Candidate predictor policy. |
| `--recovery none\|reseed\|alt_parent\|bridge\|ladder` | Failed-point recovery policy. |
| `--bridge-steps INT` | Number of bridge steps. |
| `--bridge-mode diagonal\|freq_first\|power_first\|adaptive` | Bridge path. |
| `--fold-policy patience\|cross_axis\|bridge_gate\|combined\|arclength` | Fold handling policy. |
| `--recovery-arclength-rescale-every INT` | Recompute arclength state scaling periodically. |
| `--recovery-arclength-max-steps-after-fold INT` | Extend the arclength budget after a detected fold. |
| `--inproc-pump-backend full\|schur_cpu_mt` | Full or Schur pump solve. `schur_cpu_mt` is the normal large-map backend. |
| `--inproc-preconditioner VALUE` | Pump preconditioner. `real_coupled_fast` is the workflow default. |
| `--inproc-gmres-maxiter INT` | GMRES iteration limit per Newton step. |
| `--inproc-schur-cache-size INT` | Number of cached frequency Schur systems. |
| `--inproc-precond-reuse INT` | Number of Newton steps sharing a preconditioner factor. |
| `--inproc-precond-refresh-gmres INT` | Refresh threshold for a reused preconditioner. |
| `--inproc-max-newton INT` | Maximum Newton iterations per point. |
| `--inproc-solve-deadline-s VALUE` | Per-point solve time limit. |
| `--inproc-continuation-deadline-s VALUE` | Total continuation time limit. |
| `--inproc-continuation VALUE` | Continuation method for a cold or seed point. |
| `--inproc-arclength-ds VALUE`, `--inproc-arclength-max-steps INT` | Arclength continuation controls. |
| `--inproc-fallback-fixed-steps INT` | Fixed fallback continuation steps. |
| `--adaptive-initial-step VALUE`, `--adaptive-min-step VALUE` | Adaptive continuation step bounds. |
| `--continuation-steps INT` | Fixed continuation step count. |
| `--newton-tol VALUE` | Pump Newton tolerance. |
| `--linear-seed-maxiter INT` | Linear seed iteration limit. |
| `--initial-pump-dir PATH`, `--initial-pump-power-dbm VALUE` | Verified initial pump state and its power coordinate. |
| `--fold-skip-patience INT` | Number of failed points before fold skipping is allowed. |
| `--column-arclength-recovery` | Try bounded arclength recovery in each power column. |
| `--column-arclength-ds VALUE`, `--column-arclength-max-steps INT`, `--column-arclength-deadline-s VALUE` | Arclength recovery controls. |
| `--column-power-substep` | Try smaller pump-power steps after a failed point. |
| `--column-power-substep-init-db VALUE`, `--column-power-substep-min-db VALUE`, `--column-power-substep-deadline-s VALUE` | Power-substep recovery controls. |
| `--gate-gain-db VALUE` | Cold/warm gain agreement tolerance when `--mode both` is used. |
| `--gate-min-converged-frac VALUE` | Minimum converged fraction for the gate. |
| `--gate-spotcheck INT` | Number of cold spot checks after a warm map. |
| `--fold-follow` | Write a fold curve instead of running the normal map. |

### Gain-map output

Look first at:

- `map_points.csv`: one row per pump-power/frequency point;
- `map_arrays.npz`: numeric map arrays;
- `map_summary.json`: run configuration and status summary;
- `plots/`: standard gain, status, and runtime plots.

Failed points must be read together with their status and failure reason.
Do not replace a failed point with an interpolated value without recording that
the point was not solved.

## 4. Slow HB-to-transient map

The slow workflow adds a bounded transient check around failed or difficult HB
points. It is not a general high-power or saturation solver.

```powershell
python workflows/run_gain_map_and_plots.py `
  --slow `
  --design outputs/ipm_2c_passive `
  --output-dir outputs/ipm_2c_slow `
  --n-power 10 `
  --n-frequency 1 `
  --pump-power-min-dbm -61 `
  --pump-power-max-dbm -56 `
  --pump-freq-min-ghz 12.08 `
  --pump-freq-max-ghz 12.08
```

Slow-map options:

| Option | Meaning |
| --- | --- |
| `--n-power`, `--n-frequency` | Pump grid dimensions. |
| `--power-min-dbm`, `--power-max-dbm` | Pump-power range. The wrapper aliases `--pump-power-min-dbm` and `--pump-power-max-dbm`. |
| `--freq-min-ghz`, `--freq-max-ghz` | Pump-frequency range. |
| `--attenuation-db` | Flat pump-line attenuation. |
| `--pump-mode-count`, `--pump-mode-policy`, `--mixing-order`, `--harmonics` | Pump basis and mixing selection. |
| `--pump-port`, `--source-port`, `--out-port` | Explicit port roles. Omit for automatic selection. |
| `--dc-branch-flux-over-phi0` | External reduced flux for biased devices. |
| `--signal-ghz` | Fixed signal frequency. |
| `--nt` | Pump time-grid size. |
| `--signal-detuning-mhz` | Signal detuning when no fixed signal is supplied. |
| `--signal-offset-count-per-side`, `--signal-offset-step-mhz` | Signal sideband spectrum settings. |
| `--inproc-pump-backend` | `full` or `schur_cpu_mt`. Use `full` for the validated transient handoff. |
| `--inproc-preconditioner` | Pump preconditioner. |
| `--inproc-solve-deadline-s` | HB point time limit. |
| `--inproc-max-newton` | Maximum HB Newton iterations. |
| `--td-ramp-periods` | Transient ramp duration. |
| `--td-hold-periods` | Transient hold duration. |
| `--td-checkpoint-periods` | Checkpoint interval. |
| `--max-td-bridges` | Maximum transient recovery attempts per column. |
| `--frequency-workers` | Parallel frequency-column workers. |
| `--no-isolate-columns` | Keep all columns in one process. |
| `--log-level` | Log level. |

The slow output records whether a point was solved by HB or reached the
transient boundary. Treat a physical boundary as a boundary, not as a valid
high-power operating point.

## 5. KIMPA gain map

Use the KIMPA workflow for the built-in KIMPA fixtures:

```powershell
python workflows/run_kimpa_gain_map_and_plots.py `
  --run-dir outputs/kimpa_map `
  --fixture kimpa_fabricated_nominal `
  --pump-ghz 16.94 `
  --dc-current-a 550e-6
```

The KIMPA pump axis is reported as peak total branch current divided by
critical current. The internal dBm value is retained as metadata. This is
important when the external pump-line attenuation is unknown.

### KIMPA options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--run-dir PATH` | required | Output directory. |
| `--fixture NAME` | `kimpa_fabricated_nominal` | Built-in fixture: `kimpa_ideal_synthesis`, `kimpa_fabricated_nominal`, `kimpa_measured_seed`, or `kimpa_hung_2025`. |
| `--pump-dbm-start VALUE` | `-35` | First internal pump dBm point. |
| `--pump-dbm-stop VALUE` | `-8` | Last internal pump dBm point. |
| `--pump-points INT` | `15` | Number of pump points. |
| `--pump-attenuation-db VALUE` | `0` | Attenuation subtracted before circuit injection. |
| `--pump-ghz VALUE` | `16.94` | Pump frequency. |
| `--signal-start-ghz VALUE` | `7.8` | First map signal frequency. |
| `--signal-stop-ghz VALUE` | `9.1` | Last map signal frequency. |
| `--signal-points INT` | `27` | Number of map signal points. |
| `--dc-current-a VALUE` | `550e-6` | KIMPA DC bias current. Nonzero bias selects 3WM in `auto` mode. |
| `--sidebands INT` | `5` | Small-signal sideband count. |
| `--max-ell INT` | `6` | Maximum pump-conversion harmonic index. |
| `--pump-nt INT` | `32` | Pump time-grid size. |
| `--environment ideal\|paper_standing_wave` | `ideal` | Termination environment. |
| `--pump-port INT` | automatic | Pump port. |
| `--source-port INT` | automatic | Signal source port. |
| `--out-port INT` | automatic | Signal output port. |
| `--mixing-order auto\|3\|4` | `auto` | Select 3WM from external bias or override explicitly. |
| `--spectrum-start-ghz VALUE` | `7.5` | First best-point spectrum frequency. |
| `--spectrum-stop-ghz VALUE` | `9.5` | Last best-point spectrum frequency. |
| `--spectrum-points INT` | `501` | Best-point spectrum points. Keep `501` for the standard output. |
| `--no-spectrum` | off | Do not calculate the best-point spectrum. |
| `--no-plots` | off | Do not write plots. |

### KIMPA output files

The important files are:

- `kimpa_gain_map.csv`: map rows;
- `kimpa_gain_map.npz`: map arrays;
- `map_summary.json`: fixture, port, and mixing configuration;
- `best_point_spectrum.csv`: one row per signal frequency;
- `best_point_spectrum.npz`: 501-point matrices and quantum-efficiency arrays;
- `best_point_spectrum_summary.json`: best-spectrum metadata;
- `plots/`: map and best-spectrum plots.

`best_point_spectrum.npz` contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `signal_frequency_ghz` | `(N,)` | Signal-frequency grid. |
| `Z` | `(N, P, P)` | Pump-off input/output impedance matrix. |
| `Y` | `(N, P, P)` | Pump-off input/output admittance matrix. |
| `S` | `(N, P, P)` | Pump-off scattering matrix. |
| `S_pumped` | `(N, P, P)` | Pumped signal-frequency scattering matrix. |
| `S_idler` | `(N, P, P)` | Idler conversion matrix. |
| `quantum_efficiency` | `(N, P, P)` | Signal efficiency including the exported idler conversion contribution. |
| `quantum_efficiency_ideal` | `(N, P, P)` | Ideal reference efficiency array. |

The current fabricated KIMPA fixture is a one-port reflection fixture, so it
has `S11` rather than `S21`. The same workflow writes `S21` and all available
`Sij` entries automatically when a two- or four-port circuit is supplied.

## 6. One fixed-pump signal spectrum

Use this when the pump point is already known and only the signal-frequency
response is required:

```powershell
python workflows/run_signal_spectrum.py `
  --design outputs/ipm_2c_passive `
  --pump-power-dbm -24 `
  --pump-frequency-ghz 8.0 `
  --signal-start-ghz 5.0 `
  --signal-stop-ghz 9.0 `
  --signal-points 501 `
  --run-dir outputs/ipm_2c_spectrum
```

| Option | Meaning |
| --- | --- |
| `--design PATH` | Generated circuit directory. |
| `--pump-power-dbm VALUE` | Fixed pump power coordinate. |
| `--pump-frequency-ghz VALUE` | Fixed pump frequency. |
| `--signal-start-ghz VALUE` | First signal frequency. |
| `--signal-stop-ghz VALUE` | Last signal frequency. |
| `--signal-points INT` | Number of signal points. |
| `--run-dir PATH` | Output directory. |

The output contains the fixed-pump map point, spectrum CSV, S-parameter NPZ,
and plots.

## 7. Reading and archiving results

For every production run, keep:

1. the source YAML;
2. the generated design directory;
3. the exact command line;
4. `design_resolved.json` or `map_summary.json`;
5. the CSV/NPZ data and selected plots;
6. the status and failure files for any point that did not converge.

Use a new output directory for a new run. Use `--overwrite` only when the
previous output is disposable.

Common statuses include:

| Status | Meaning |
| --- | --- |
| `PASS` or `VALID_SOLVED` | The requested calculation completed and passed its numerical checks. |
| `ERROR` or `CHECK` | The point needs inspection. |
| `SKIP_PAST_FOLD` | The continuation crossed a detected operating boundary. |
| `PHYSICAL_BOUNDARY_FOUND` | The slow workflow found a physical/transient boundary. |
| `PERSISTENT_NONPERIODIC` | The transient did not settle into the required periodic state. |

Do not report a failed, skipped, or boundary point as a valid device result.

## 8. Current operating boundary

The supported fab-facing workflows are passive response, moderate-pump
harmonic balance, and small-signal gain. The following are outside the current
production scope:

- high pump power or high junction current near a fold;
- general kinetic-inductance models;
- saturation, compression, and the full high-signal regime;
- conclusions from a point that did not converge.

Stop the campaign at the reported boundary and retain the boundary bracket in
the output. Do not extend the pump range by silently changing solver limits.
