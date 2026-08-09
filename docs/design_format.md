# Declarative design format

Design YAML describes the fabricated circuit. Experiments, pump settings,
continuation, gain maps, and Monte Carlo scatter remain outside this schema.

Schema version 1 uses one ordered `topology` list with mixed granularity. The
normal IPM authoring surface uses `input_ports`, `output_ports`,
`directional_coupler`, `ipm_line`, and `ipm_tail`; the compiler expands these
into the detailed resolved representation.

- level 3 composites: `input_ports`, `output_ports`, `ipm_line`, and `ipm_tail`;
- level 2 reusable blocks: `jj_line`, transmission lines, and `coupler`;
- level 1 exact components: `capacitor`, `inductor`, `raw_element`, and inline
  `action: set/remove` edits.

Composite paths are stable and addressable. For example:

```yaml
- type: ipm_line
  name: line_1
  cursor: signal
  rows: 2
- type: directional_coupler
  name: c2
- type: capacitor
  name: local_cap
  nodes: [line_1.row[1].array[0].cell[205].right, ground]
  C: 98e-15
- action: set
  target: line_1.row[1].array[0].cell[300].Lj
  value: 140e-12
```

Generated paths include row, array, cell `left`/`right` nodes and `Lj`, `Cj`,
and `Cg` elements. Absolute integer nodes remain supported as an escape hatch.
`repeat` supports nesting through depth two.

Technology defaults are selected with `technology: ipm_default` and loaded
from `designs/technology/ipm_default.yaml`. Explicit design values override
the preset, and `design_resolved.json` records the final values. A design may
inherit from one parent with `extends: parent.yaml`; child mappings override
parent mappings deterministically.

Directional couplers use the design's `coupling_dB`, `coupler_freq_hz`, and
`Z0` values. With `coupler_mode: auto`, compilation optimizes the geometry and
selects the two-line model for stronger coupling (approximately -18 dB and
above) or the centre-ground three-line model for weaker coupling. The selected
geometry is retained in resolved metadata; `cached` and `ideal` remain
available for legacy and controlled-comparison builds.

Deterministic spatial profiles are part of the design:

```yaml
profiles:
  Lj:
    target: all
    type: half_sine       # constant, linear, half_sine
    start: 100e-12
    stop: 150e-12
    domain: per_row
```

`half_sine` is the named form of the existing safe `sin(pi*t/2)` profile
mathematics. `designs/ipm_2c_linear.yaml` and
`designs/ipm_2c_half_sine.yaml` are inherited variants and compile without
profile CLI options. The old profile CLI flags remain compatible overrides.

The `designs/` directory is intentionally source-only: it keeps the YAML
examples and technology presets, not matrices, plots, CSV files, or resolved
build artifacts. Put generated outputs under `outputs/` or another disposable
build directory.

The checked-in examples are the JJ-line ladders (`jtwpa_*jj.yaml`), the IPM
2c/3c/7c definitions, and the linear and half-sine 2c variants. The resolved
form is produced on demand by the compiler and is not checked in beside the
source design.

Build an artifact with:

```powershell
python -m twpa_solver.design --design designs/ipm_2c.yaml `
  --outdir outputs/design_test --write-matrices --strict
```

The compiler rejects unknown presets, bad inheritance, invalid profiles,
missing symbolic targets, cursor collisions, duplicate paths, and duplicate
element names. It emits the flat solver `Element[]` representation, so the
existing C/G/K/Bphi assembly is unchanged.

## RF-SQUID 3WM validation design

`designs/rf_squid_2393_3wm.yaml` is the first validation target for biased
three-wave mixing. Its `rf_squid_line` block expands each cell into series
`Lw`, a parallel `Lm` / (`Lpar` + Josephson junction) branch, junction `Cj`,
and split ground capacitance. The 24-cell loading pattern is
`[C1 x 6, C2 x 6, C1 x 6, C3 x 6]` and is truncated deterministically at
2,393 cells. Both capacitor halves are stamped, so one complete period sums to
`12*C1 + 6*C2 + 6*C3 = 837.6 fF`.

The design uses provisional `Cj = 20 fF`, because the experimental paper does
not provide a numerical junction-capacitance value. Run the staged workflow
with:

```powershell
python scripts/run_rf_squid_3wm.py `
  --design designs/rf_squid_2393_3wm.yaml `
  --outdir outputs/rf_squid_2393_3wm --no-pump
```

The workflow applies the `0.33 Phi0` DC branch-flux offset, generates pump-off
S-parameters, and can then run the dense-real pump basis with Floquet spacing
of one pump frequency (the 3WM idler is `m=-1`).

For the shared fast gain-map workflow, pass `--mixing-order 3` together with
the RF-SQUID ports and an ideal on-chip pump calibration, for example:

```powershell
python workflows/run_gain_map_and_plots.py --fast `
  --design outputs/rf_squid_2393_3wm `
  --pump-port 1 --source-port 1 --out-port 2 `
  --mixing-order 3 --pump-mode-policy dense_real --harmonics 3 `
  --dc-branch-flux-over-phi0 0.33 --attenuation-db 0
```

The default is now `--mixing-order auto`: an external DC current/flux selects
3WM, while an unbiased circuit selects 4WM. Explicit `--mixing-order 3` or
`4` remains available. Port flags are also optional; the workflow derives the
standard roles from the ports present in the resolved circuit.
