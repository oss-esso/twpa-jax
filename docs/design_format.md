# Design format reference

The Python ``Circuit`` API is the authoritative design authoring interface.
This document describes the YAML adapter for users who need a declarative
device definition. Pump power, frequency sweeps, gain maps, and solver
settings belong to the workflow command.

The normal process is:

```text
Python Circuit design -> compile -> Element[] / matrices -> workflow
                         ^
                         |
                 optional YAML adapter
```

Use Python designs under ``designs/python/`` for new circuits — the checked-in
examples are ``ipm_2c.py``, ``ipm_v2.py``, and ``ipm_v3.py``, and the API
itself is documented in
[`development/circuit_api.md`](development/circuit_api.md). Existing YAML
designs remain supported through ``compile_design``, which translates the YAML
blocks into calls to the same public ``Circuit`` builders. The YAML compiler
is therefore an adapter, not a second circuit-generation implementation.

Do not edit generated matrix files by hand. Change the Python design or YAML
source and rebuild the output directory.

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
| `coupler_mode` | no | Override the technology coupler mode with `auto`, `ideal`, or `optimize`. |

The compiler rejects unknown top-level fields, missing required fields,
unknown blocks, duplicate names, invalid references, and unsupported schema
versions.

### 2.1 Line-scoped topology

For a circuit described naturally as ordered physical lines, each topology
item may declare one line and its ordered blocks.  `line: 1` and `line: 2` are
short forms for the `signal` and `pump` cursors.  Cursor names may be written
directly for other lines.

```yaml
parameters:
  config: ipm_default
  Lj: $Lj$ / 2
  Cg: 2 * $Cg$

topology:
  - line: signal
    port_in: 1
    port_out: 2
    blocks:
      - {type: input_ports, name: signal_input}
      - {type: cpw, name: signal_cpw, cells: 200}
      - type: directional_coupler
        name: coupler_1
        port_in_signal: 1
        port_in_pump: 3
        port_out_signal: 2
        port_out_pump: 4
      - {type: jtl, name: nonlinear_line, rows: 2, jj_number: 100}
      - {type: output_ports, name: signal_output}

  - line: pump
    port_in: 3
    port_out: 4
    blocks:
      - {type: input_ports, name: pump_input}
      - {type: cpw, name: pump_cpw, cells: 500}
      - {type: directional_coupler, name: coupler_1}
      - {type: jtl, name: nonlinear_line}
      - {type: output_ports, name: pump_output}
```

The first detailed occurrence of a shared name defines the block.  An
occurrence containing only `type` and `name` is a reference.  References add
ordering constraints but do not emit duplicate elements.  The compiler orders
all lines together, so both CPWs above are assembled before `coupler_1`.

`cpw` uses the existing `transmission_line` builder and inherits `Ll` and `Cl`.
`jtl` expands `rows` copies of the existing `jj_line` builder, each containing
`jj_number` cells.  `jj_number` and `jj number` are equivalent YAML spellings.

Input port numbers must be unique.  Two lines may declare the same output port
when they use the same final `output_ports` name and share a `jtl` block.  The
first shared JTL joins their current endpoints; subsequent shared JTL names
continue on the resulting common path.  A shared output without that explicit
JTL join is rejected.

## 3. Technology presets

Set a preset with:

```yaml
technology: ipm_default
```

Presets live in `designs/technology/`. The checked-in ones are
`ipm_default.yaml` and `qanova_ipm_v1.yaml`. A preset contains values normally
shared by a fabrication platform.

A preset file separates **electrical component values** from **architecture**,
and carries a few top-level keys:

```yaml
name: ipm_default
coupler_mode: auto
components:
  Ll: 4.24e-12
  Cl: 1.695e-15
  Lj: 123.9e-12
  Cj: 145.0e-15
  Cg: 66.0e-15
  coupling_dB: -14.0
  Z0: 50.0
  coupler_freq_hz: 8.0e9
  cell_length_um: 10.0
  Rleft: 50.0
  Rright: 50.0
  Rm: 50.0
architecture:
  jtl_cells_per_array: 418
  jtl_row_count: 6
  jtl_rows_per_coupler: 3
  inter_array_cpw_cells: 90
  signal_inter_coupler_cpw_cells: 900
  pump_inter_coupler_cpw_cells: 800
  signal_input_cpw_cells: 0
  signal_output_cpw_cells: 50
  pump_input_cpw_cells: 0
  pump_output_cpw_cells: 0
cursors:
  signal: 1
  pump: 10000
ground: 0
```

| Section | Key | Unit | Meaning |
| --- | --- | --- | --- |
| top level | `name` | — | Preset name. |
| top level | `coupler_mode` | — | Default coupler model: `auto`, `ideal`, or `optimize`. |
| `components` | `Ll` | H | Inductance of one linear transmission-line cell. |
| `components` | `Cl` | F | Capacitance of one linear transmission-line cell. |
| `components` | `Lj`, `Cj`, `Cg` | H, F, F | Nominal junction and ground values. |
| `components` | `coupling_dB` | dB | Coupler target coupling. |
| `components` | `Z0` | ohm | Nominal line and port impedance. |
| `components` | `coupler_freq_hz` | Hz | Frequency used for coupler design. |
| `components` | `cell_length_um` | µm | Physical length of one transmission-line cell. |
| `components` | `Rleft`, `Rright`, `Rm` | ohm | Signal and pump termination resistances. |
| `architecture` | `jtl_cells_per_array` | cells | Josephson cells in each JTL array. |
| `architecture` | `jtl_row_count` | count | Total number of JTL rows. |
| `architecture` | `jtl_rows_per_coupler` | count | JTL rows before each intermediate coupler. |
| `architecture` | `inter_array_cpw_cells` | cells | CPW cells between adjacent JTL arrays. |
| `architecture` | `signal_inter_coupler_cpw_cells` | cells | Signal CPW cells before the next coupler. |
| `architecture` | `pump_inter_coupler_cpw_cells` | cells | Pump CPW cells between couplers. |
| `architecture` | `signal_input_cpw_cells`, `signal_output_cpw_cells` | cells | Signal input/output CPW lengths. |
| `architecture` | `pump_input_cpw_cells`, `pump_output_cpw_cells` | cells | Pump input/output CPW lengths. |
| top level | `cursors.signal`, `cursors.pump` | node | Starting cursor for each physical line. |
| top level | `ground` | node | Ground node number. |

A **legacy flat `parameters:` mapping** is still accepted in place of
`components:`/`architecture:`, and `qanova_ipm_v1.yaml` still uses it. The two
are merged, with `components:` winning on a key present in both. Use the split
form in new presets.

Resolution order for any one parameter is deterministic:

```text
explicit call argument
    -> design-level override
    -> technology components:
    -> technology architecture:
    -> builder default
    -> error naming the parameter and path
```

A design value overrides the preset value:

```yaml
technology: ipm_default
parameters:
  coupling_dB: -16.0
  Cg: 70e-15
```

The line-scoped surface also accepts `parameters.config` as a compact
technology selection.  `$name$` and `${base.name}` refer to the unmodified
technology value, while `${name}` refers to the final design parameter.  Safe
scalar expressions support numeric literals, parentheses, and `+`, `-`, `*`,
and `/` only:

```yaml
parameters:
  config: ipm_default
  Lj: $Lj$ / 2
  Cg: 2 * ${base.Cg}
```

The resolved design records the final value used for every parameter.

## 4. Nominal component parameters

Use SI units in YAML. The most common values are:

| Parameter | Unit | Meaning |
| --- | --- | --- |
| `Lj` | H | Nominal Josephson inductance. `123.9e-12` is 123.9 pH. |
| `Cj` | F | Junction capacitance. `145e-15` is 145 fF. |
| `Cg` | F | Ground capacitance per nonlinear cell. |
| `jtl_cells_per_array` | count | Number of cells in one JJ line. |
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

`name` is required on every block and must be unique within its topology list.
An unknown field is rejected, so the optional column is exhaustive.

| Type | Required fields | Optional fields | Meaning |
| --- | --- | --- | --- |
| `input_ports` | `name` | `cursor`, `port`, `resistance` | Standard signal and pump input ports, terminations, and input lines. |
| `output_ports` | `name` | `cursor`, `port`, `resistance` | Standard signal and pump output ports, terminations, and output lines. |
| `directional_coupler` | `name` | `cursors` | Coupler between the signal and pump cursors. |
| `coupler` | `name`, `cursors` | none | Explicit-cursor form of the same block. |
| `ipm_line` | `name`, one of `rows`/`arrays` | `cursor`, `cells`, `Lj`, `Cj`, `Cg`, `between`, `trailing_signal_cpw_cells`, `trailing_pump_cpw_cells`, `end_coupler` | Nonlinear rows followed by the signal and pump routing to the next coupler. |
| `ipm_tail` | `name`, `rows` | `cursor`, `cells`, `Lj`, `Cj`, `Cg`, `between`, `final_array` | Final nonlinear rows without a trailing coupler section. `final_array` may only be `true`. |
| `ipm_row` | `name`, `cursor`, `cells`, `Lj`, `Cj`, `Cg` | none | One explicit nonlinear row. |

An omitted optional component value is resolved through the technology preset
chain in §3 rather than defaulting to zero. `ipm_row` is the explicit form and
takes all five values.

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

`ipm_line` and `ipm_tail` are composite blocks. They hide the following
operations:

1. Each requested row becomes one `jj_line` with `cells` Josephson cells.
2. `between` inserts a CPW transmission line between consecutive rows.
3. `ipm_line` then appends `trailing_signal_cpw_cells` on the signal path and
   `trailing_pump_cpw_cells` on the pump path. These are the routing sections
   that lead to the next directional coupler.
4. `ipm_tail` stops after the final JTL row. It does not add a long signal CPW,
   a pump CPW section, or a coupler.

For `ipm_2c_line_scoped`, the resulting paths are:

```text
Signal:
Port 1 -> termination -> coupler_in
  -> JTL row 0 (418 cells) -> CPW (90 cells)
  -> JTL row 1 (418 cells) -> CPW (90 cells)
  -> JTL row 2 (418 cells)
  -> signal inter-coupler CPW (900 cells) -> coupler_1
  -> JTL row 3 (418 cells) -> CPW (90 cells)
  -> JTL row 4 (418 cells) -> CPW (90 cells)
  -> JTL row 5 (418 cells)
  -> signal output CPW (50 cells) -> Port 2

Pump:
Port 3 -> termination -> coupler_in
  -> pump inter-coupler CPW (800 cells) -> coupler_1 -> Port 4
```

With `cell_length_um: 10`, these values correspond to 4.18 mm per JTL row,
0.90 mm per inter-array CPW, 9.00 mm of signal routing, 8.00 mm of pump
routing, and 0.50 mm at the signal output.

### 5.2 Reusable line and RF-SQUID blocks

| Type | Required fields | Optional fields | Meaning |
| --- | --- | --- | --- |
| `jj_line` | `name`, `cursor`, `cells` | `Lj`, `Cj`, `Cg` | Series Josephson-junction line with ground capacitance. |
| `transmission_line` | `name`, `cursor`, `cells` | `L`, `C` | Linear transmission-line ladder. |
| `rf_squid_line` | `name`, `cursor`, `cells`, `Ic`, `Lm`, `Lw`, `Lpar`, `Cj` | `Lj`, `Cg`, `Cg_pattern`, `Cg_pattern_counts` | Biased RF-SQUID line. |
| `jtl` | `name`, `cursor`, `rows`, `cells` | `Lj`, `Cj`, `Cg`, `join_cursors` | `rows` copies of `jj_line`. In the line-scoped surface (§2.1) `cursor` comes from `line`, and `cells` may be spelled `jj_number` or `jj number`. |

`jj_line` and `transmission_line` fall back to the technology preset for the
component values they omit; only the geometry (`cursor`, `cells`) is
mandatory. In the line-scoped surface, `cpw` is an accepted alias for
`transmission_line` and `coupler` for `directional_coupler`.

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
| `capacitor` | `name`, `nodes`, `C` | Add a capacitor between exactly two nodes. |
| `inductor` | `name`, `nodes`, `L` | Add a linear inductor between exactly two nodes. |
| `raw_element` | `name`, `nodes`, `value`, `kind` | Add a supported low-level element. |

The supported raw `kind` values are `capacitor`, `coupling_capacitor`,
`linear_inductor`, `josephson_inductor`, `mutual_inductor_k`, `resistor`, and
`port`.

### 5.4 Repeating a group of blocks

A topology item may be a `repeat` instead of a block. It emits its own
`topology` list `count` times, in order, at the position where it appears.

```yaml
topology:
  - type: signal_port
    name: p_in
    port: 1

  - repeat:
      name: stage
      count: 3
      topology:
        - {type: jj_line, name: jj, cursor: signal, cells: 5}
        - {type: transmission_line, name: cpw, cursor: signal, cells: 4}

  - type: signal_port
    name: p_out
    port: 2
```

| Key | Required | Meaning |
| --- | --- | --- |
| `count` | yes | Non-negative integer. `0` emits nothing. |
| `topology` | yes | Ordered list of blocks, actions, or one further `repeat`. |
| `name` | no | Occurrence label used in generated paths. Defaults to `repeat`. |

Each occurrence is path-qualified as `<name>[<occurrence>]`, so the example
above produces `stage[0].jj`, `stage[1].jj`, and `stage[2].jj`, and reaches
individual cells as `stage[0].cpw.cell[0].left`. Use those qualified paths in
hierarchical references (§6).

Block names must be unique **within one topology list**, and a `repeat` body
is its own list. Reusing `jj` across the three occurrences above is therefore
correct, not a duplicate-name error.

**Composite blocks cannot appear inside a `repeat`.** Composite expansion runs
over the top-level topology only, so `input_ports`, `output_ports`,
`ipm_line`, `ipm_tail`, `ipm_row`, and `jtl` reach the emitter unexpanded and
raise `unknown block type`. What a `repeat` body may contain is the primitive
set: `jj_line`, `transmission_line`, `rf_squid_line`, `directional_coupler`,
`coupler`, `port`, `signal_port`, `pump_port`, `resistor`, `capacitor`,
`inductor`, `raw_element`, and `set`/`remove` actions.

**Nesting is limited to two levels.** A `repeat` may contain a `repeat`; a
third level raises `repeat nesting deeper than 2 is unsupported`.

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

**`start` and `stop` are both required, for every type including `constant`.**
`end` is accepted as a spelling of `stop`. Omitting either is a build error.

Supported profile types are:

| Type | Shape `s(t)` on `t` in `[0,1]` | Shape parameter | Meaning |
| --- | --- | --- | --- |
| `constant` | — | — | One value over the selected cells. `const` is the same type. |
| `linear` | `t` | — | Linear change from `start` to `stop`. |
| `power` | `t**p` | `p` > 0, default 1 | Anchored power law. |
| `parabola` | `((t-v)²-v²)/((1-v)²-v²)` | `v` in `[0,1]`, not 0.5, default 0 | Anchored parabola; `v=0` is plain `t²`. |
| `half_cosine` | `(1-cos(pi t))/2` | — | Anchored and smooth at both ends. |
| `half_sine` | `sin(pi t / 2)` | — | Convenience alias compiled to `custom`. |
| `tanh` | normalized `tanh(k(2t-1))` | `k` > 0, default 1 | Anchored sigmoid. |
| `sine` | `(1+sin(2 pi f t + phi))/2` | `periods` default 1, `phase` default 0 | **Envelope**, see below. |
| `cosine` | `(1+cos(2 pi f t + phi))/2` | `periods` default 1, `phase` default 0 | **Envelope**, see below. |
| `custom` | user `expression` in `t` | `expression` | Use only when a named type is not sufficient. |

The emitted value is `start + (stop - start) * s(t)`, so every anchored shape
hits `start` at the first selected cell and `stop` at the last.

**Shape parameters are not reachable from the mapping form above.** The
mapping form sets only shape, endpoints, domain, target, and `expression`; a
`p`, `v`, `k`, `periods`, or `phase` key written there is ignored and the
default in the table applies. To set one, give the entry as a shorthand
string instead, which the list form also accepts:

```yaml
profiles:
  Lj:
    - "rows=3-5:tanh:120p->140p:k=4,domain=per_row"
    - "rows=0-2:power:120p->140p:p=2"
```

**`sine` and `cosine` do not.** Periodic and endpoint-anchored are mutually
exclusive, so for those two `start`/`stop` bound the oscillation envelope
(mean `(start+stop)/2`, amplitude `(stop-start)/2`) rather than giving the
first and last values.

`custom` expressions are restricted to an allowlist checked before evaluation.
The full shape catalogue, the allowlist, and the `Cg` boundary-halving rule
are in
[`component_profiles_and_scatter.md`](component_profiles_and_scatter.md).

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
extends: ipm_2c_line_scoped.yaml
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
  --design designs/ipm_2c_line_scoped.yaml `
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
| `--coupler-mode auto\|ideal\|optimize` | Override the YAML/technology coupler mode. |
| `--overwrite` | Allow writing into a non-empty output directory. |
| `--strict` | Additionally require that every declared `parameters` entry is actually referenced. Schema validation itself always runs. |
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
| `designs/ipm_2c_line_scoped.yaml` | Compact two-coupler IPM source. |
| `designs/ipm_3c.yaml` | Compact three-coupler IPM source. |
| `designs/ipm_7c_ideal_node205.yaml` | Seven-coupler IPM example with a local node-205 capacitor. |
| `designs/ipm_2c_linear.yaml` | IPM 2c with a YAML linear profile. |
| `designs/rf_squid_2393_3wm.yaml` | RF-SQUID 3WM validation design. |
| `designs/generic_blocks_showcase.yaml` | Line-scoped generic builder showcase. |
| `designs/three_port_cpw_jtl_example.yaml` | Three-port CPW/JTL example. |

Use the exact file names present in the checkout. Generated passive and gain
outputs should be written below `outputs/`, not beside the source YAML.
