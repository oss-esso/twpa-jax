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
