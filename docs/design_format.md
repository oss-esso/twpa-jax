# Design YAML reference

This document explains the design file that a fab user edits. It describes the
device definition only. Pump power, frequency sweeps, gain maps, and solver
settings belong to the workflow command.

The normal process is:

```text
design YAML -> compiler -> generated circuit directory -> workflow
```

Do not edit generated matrix files by hand. Change the YAML and rebuild the
output directory.

## 1. Smallest useful IPM design

```yaml
schema_version: 1
name: ipm_2c
technology: ipm_default

parameters:
  Lj: 123.9e-12
  Cj: 145e-15
  Cg: 66e-15
  array_length: 418

topology:
  - type: input_ports
    name: input

  - type: directional_coupler
    name: coupler_in

  - type: ipm_line
    name: line_1
    rows: 3

  - type: directional_coupler
    name: coupler_1

  - type: ipm_tail
    name: line_2
    rows: 3
    final_array: true

  - type: output_ports
    name: output
```

The source file is intentionally short. `input_ports`, `directional_coupler`,
`ipm_line`, `ipm_tail`, and `output_ports` are expanded by the compiler into
the detailed circuit representation.

## 2. Top-level fields

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | Must be the integer `1`. |
| `name` | yes | Human-readable design name. Use a unique name. |
| `technology` | recommended | Technology preset name, for example `ipm_default`. |
| `extends` | no | One parent YAML file. Child values replace parent values. |
| `ground` | yes after technology resolution | Integer ground node. Normally `0`. |
| `cursors` | yes after technology resolution | Starting node for each named physical line. Normally `signal` and `pump`. |
| `parameters` | no | Nominal electrical, geometry, and topology values. |
| `topology` | yes | Ordered list of physical blocks and local changes. |
| `profiles` | no | Deterministic cell-to-cell parameter changes. |
| `patches` | no | Optional separate list of local edits. |
| `coupler_mode` | no | Override the technology coupler mode with `auto`, `cached`, `ideal`, or `optimize`. |

The compiler rejects unknown top-level fields, missing required fields,
unknown blocks, duplicate names, invalid references, and unsupported schema
versions.

## 3. Technology presets

Set a preset with:

```yaml
technology: ipm_default
```

The preset is stored in `designs/technology/ipm_default.yaml`. It contains
values normally shared by a fabrication platform:

| Preset parameter | Unit | Meaning |
| --- | --- | --- |
| `Ll` | H | Inductance of one linear transmission-line cell. |
| `Cl` | F | Capacitance of one linear transmission-line cell. |
| `coupling_dB` | dB | Coupler target coupling. |
| `Z0` | ohm | Nominal line and port impedance. |
| `coupler_freq_hz` | Hz | Frequency used for coupler design. |
| `length_of_short_TL` | cells | Short signal-line section length. |
| `length_of_long_TL` | cells | Long signal-line section length. |
| `coupler_section_length` | cells | Pump-line section length through a coupler. |
| `len1`, `len2`, `len3`, `len4` | cells | Input/output line lengths. |
| `Rleft`, `Rright`, `Rm` | ohm | Signal and pump termination resistances. |
| `num_rows` | count | Default number of IPM rows. |
| `arrays_per_dc` | count | Rows between couplers. |
| `cached_coupler_*` | um | Cached coupler geometry values. |
| `cursors.signal` | node | Starting signal cursor. |
| `cursors.pump` | node | Starting pump cursor. |
| `ground` | node | Ground node number. |

A design value overrides the preset value:

```yaml
technology: ipm_default
parameters:
  coupling_dB: -16.0
  Cg: 70e-15
```

The resolved design records the final value used for every parameter.

## 4. Nominal component parameters

Use SI units in YAML. The most common values are:

| Parameter | Unit | Meaning |
| --- | --- | --- |
| `Lj` | H | Nominal Josephson inductance. `123.9e-12` is 123.9 pH. |
| `Cj` | F | Junction capacitance. `145e-15` is 145 fF. |
| `Cg` | F | Ground capacitance per nonlinear cell. |
| `array_length` | count | Number of cells in one JJ line. |
| `L` | H | Linear inductor value. |
| `C` | F | Capacitor or transmission-line capacitance. |
| `Ic` | A | Critical current for an RF-SQUID line. |
| `Lm`, `Lw`, `Lpar` | H | RF-SQUID inductances. |
| `R` or `value` | ohm | Resistance or raw element value. |

Do not write `pH`, `fF`, or `GHz` suffixes in numeric YAML values. Use the SI
conversion explicitly: `150e-12`, `66e-15`, and `8.0e9`.

## 5. Topology blocks

The topology list is ordered. Blocks may be mixed at any level in one file.
Use a block name once only.

### 5.1 Composite IPM blocks

| Type | Required fields | Optional fields | Meaning |
| --- | --- | --- | --- |
| `input_ports` | `name` | none | Standard signal and pump input ports, terminations, and input lines. |
| `output_ports` | `name` | none | Standard signal and pump output ports, terminations, and output lines. |
| `directional_coupler` | `name` | none | Coupler between the signal and pump cursors. |
| `ipm_line` | `name`, `cursor`, one of `rows`/`arrays` | `cells`, `Lj`, `Cj`, `Cg`, `between`, `end_coupler` | One or more nonlinear IPM rows. |
| `ipm_tail` | `name`, `rows` | `cursor`, `cells`, `Lj`, `Cj`, `Cg`, `final_array` | Final nonlinear section. Set `final_array: true`. |
| `ipm_row` | `name`, `cursor`, `cells`, `Lj`, `Cj`, `Cg` | none | One explicit nonlinear row. |

Example:

```yaml
topology:
  - type: input_ports
    name: input
  - type: directional_coupler
    name: c1
  - type: ipm_line
    name: line_1
    cursor: signal
    rows: 2
  - type: directional_coupler
    name: c2
  - type: ipm_tail
    name: line_2
    rows: 2
    final_array: true
  - type: output_ports
    name: output
```

`rows` and `arrays` are accepted aliases for the number of IPM rows. Use
`rows` in new files.

### 5.2 Reusable line and RF-SQUID blocks

| Type | Required fields | Meaning |
| --- | --- | --- |
| `jj_line` | `name`, `cursor`, `cells`, `Lj`, `Cj`, `Cg` | Series Josephson-junction line with ground capacitance. |
| `transmission_line` | `name`, `cursor`, `cells`, `L`, `C` | Linear transmission-line ladder. |
| `rf_squid_line` | `name`, `cursor`, `cells`, `Ic`, `Lm`, `Lw`, `Lpar`, `Cj` | Biased RF-SQUID line. |

An RF-SQUID line may use either one ground value or a repeating pattern:

```yaml
- type: rf_squid_line
  name: rf_line
  cursor: signal
  cells: 2393
  Ic: 0.93e-6
  Lm: 58.6e-12
  Lw: 37.0e-12
  Lpar: 8.9e-12
  Cj: 20e-15
  Cg_pattern: [10.5e-15, 68.2e-15, 10.5e-15, 50.4e-15]
  Cg_pattern_counts: [6, 6, 6, 6]
```

### 5.3 Ports and exact components

| Type | Required fields | Meaning |
| --- | --- | --- |
| `port` | `name`, `cursor`, `port` | Add a port at the current cursor position. |
| `signal_port` | `name`, `port` | Add a signal port using the signal cursor. |
| `pump_port` | `name`, `port` | Add a pump port using the pump cursor. |
| `resistor` | `name`, `cursor`, `value` | Add a resistor from the cursor to ground. |
| `capacitor` | `name`, `nodes`, `C` | Add a capacitor between two nodes. |
| `inductor` | `name`, `nodes`, `L` | Add a linear inductor between two nodes. |
| `raw_element` | `name`, `nodes`, `value`, `kind` | Add a supported low-level element. |

The supported raw `kind` values are `capacitor`, `coupling_capacitor`,
`linear_inductor`, `josephson_inductor`, `mutual_inductor_k`, `resistor`, and
`port`.

## 6. Hierarchical node and element references

Composite blocks expose stable paths in the resolved design. For an IPM line,
typical references are:

```text
line_1.row[0].array[0].cell[205].left
line_1.row[0].array[0].cell[205].right
line_1.row[0].array[0].cell[205].Lj
line_1.row[0].array[0].cell[205].Cj
line_1.row[0].array[0].cell[205].Cg
```

Use hierarchical references whenever possible. They remain meaningful when
the compiler changes internal node numbering. Absolute integer nodes are
still accepted for a controlled one-off change:

```yaml
nodes: [1278, 1279]
```

An unknown, ambiguous, or out-of-range reference is a build error.

## 7. Local modifications

Inline modifications keep the file in physical order:

```yaml
topology:
  - type: ipm_line
    name: line_1
    cursor: signal
    rows: 2

  - type: capacitor
    name: local_cap
    nodes:
      - line_1.row[0].array[0].cell[205].right
      - ground
    C: 98e-15

  - action: set
    target: line_1.row[0].array[0].cell[300].Lj
    value: 140e-12

  - action: remove
    target: line_1.row[0].array[0].cell[300].Cj
```

The alternative top-level `patches` list uses the same `set` and `remove`
forms. An `add` patch requires `name`, `nodes`, `value`, and `kind`.

## 8. Profiles

Profiles describe deterministic device variation. They are part of the design,
not an experiment setting. A profile in the YAML is the canonical way to
reproduce a tapered or intentionally varied device.

```yaml
profiles:
  Lj:
    target: all
    type: linear
    start: 100e-12
    stop: 150e-12
    domain: per_row

  Cg:
    target: all
    type: half_sine
    start: 53.2688e-15
    stop: 79.9031e-15
    domain: per_row
```

Supported profile types are:

| Type | Meaning |
| --- | --- |
| `constant` | One value over the selected cells. |
| `linear` | Linear change from `start` to `stop`. |
| `half_sine` | Smooth change using the standard half-sine shape. |
| `custom` | Advanced expression form; use only when a named type is not sufficient. |

Supported current domains are `selection` and `per_row`. The normal IPM
profile uses `per_row`. `target: all` selects all compatible rows. A named
line or row can be selected, for example `target: line_1`.

Profiles can also be placed under a parameter:

```yaml
parameters:
  Cj: 145e-15
  Lj:
    profile:
      type: linear
      start: 100e-12
      stop: 150e-12
      domain: per_row
```

The older `--lj-profile` and `--cg-profile` command-line options remain
available for compatibility. Use YAML profiles for new designs.

## 9. Inheritance

Inheritance is one-level-per-file and deterministic:

```yaml
# ipm_2c_linear.yaml
extends: ipm_2c.yaml
name: ipm_2c_linear

profiles:
  Lj:
    target: all
    type: linear
    start: 100e-12
    stop: 150e-12
    domain: per_row
```

The child replaces values with the same key and inherits all other values.
Circular or missing inheritance paths are build errors.

## 10. Compile a design

Compile a YAML design into a generated circuit directory:

```powershell
python -m twpa_solver.design `
  --design designs/ipm_2c.yaml `
  --outdir outputs/ipm_2c_compiled `
  --write-matrices `
  --strict
```

Compiler flags:

| Flag | Meaning |
| --- | --- |
| `--design PATH` | Source YAML file. Required. |
| `--outdir PATH` | Generated circuit directory. Required. |
| `--write-matrices` | Write `C.npz`, `G.npz`, `K.npz`, `Bphi.npz`, and arrays. Use this for workflows. |
| `--coupler-mode auto\|cached\|ideal\|optimize` | Override the YAML/technology coupler mode. |
| `--overwrite` | Allow writing into a non-empty output directory. |
| `--strict` | Enable strict compiler validation. |
| `--profile-json PATH` | Legacy profile input. Prefer YAML `profiles`. |
| `--lj-profile TEXT` | Legacy Lj profile override; repeatable. |
| `--cg-profile TEXT` | Legacy Cg profile override; repeatable. |
| `--lj-scatter-sigma VALUE` | Legacy multiplicative Lj scatter. |
| `--cj-scatter-sigma VALUE` | Legacy Cj scatter. |
| `--cj-scatter-mode independent\|plasma_locked` | Cj scatter mode. |
| `--cg-scatter-sigma VALUE` | Legacy Cg scatter. |
| `--scatter-seed INT` | Reproducible scatter seed. |
| `--tan-delta VALUE` | Global dielectric loss tangent. |
| `--tan-delta-role ROLE=VALUE` | Role-specific loss tangent; repeatable. |

Generated files include `design_resolved.json`, `design_summary.json`,
`elements.csv`, `ports.csv`, and, with `--write-matrices`, the solver matrices.

## 11. Example files

The most useful checked-in examples are:

| File | Use |
| --- | --- |
| `designs/ipm_2c.yaml` | Compact two-coupler IPM source. |
| `designs/ipm_3c.yaml` | Compact three-coupler IPM source. |
| `designs/ipm_7c_ideal_node205.yaml` | Seven-coupler IPM example with a local node-205 capacitor. |
| `designs/ipm_2c_linear.yaml` | IPM 2c with a YAML linear profile. |
| `designs/ipm_2c_half_sine.yaml` | IPM 2c with a YAML half-sine profile. |
| `designs/jtwpa_418jj.yaml` | Simple 418-junction line. |
| `designs/rf_squid_2393_3wm.yaml` | RF-SQUID 3WM validation design. |

Use the exact file names present in the checkout. Generated passive and gain
outputs should be written below `outputs/`, not beside the source YAML.
