# Project working notes

- Dependency freedom: free Python packages may be installed when they materially
  improve the implementation or validation. The preinstalled environment is not
  a hard dependency limit; verify compatibility and record the package used.

## Branch and artifact policy

- `main` is the clean, installable package branch.
- `dev` is the development and cross-machine synchronization branch; it may
  contain experiments, generated artifacts, and incomplete work.
- Keep ignore rules compatible, but do not rely on `.gitignore` to separate
  tracked files. Ignored files are not synchronized; tracked files merge.
- Do not merge `dev` wholesale into `main`. Promote production work with
  focused commits and `git cherry-pick`, or use a manually curated merge.
- Keep generated outputs under ignored `outputs/` or another disposable path;
  source design YAML belongs under `designs/`.

## Canonical IPM coupler-count designs

- The canonical presentation sources are `designs/ipm_2c.yaml`,
  `designs/ipm_3c.yaml`, `designs/ipm_7c.yaml`, and `designs/ipm_20c.yaml`.
- `ipm_2c.yaml` is the renamed line-scoped 2c source. All four use the same
  IPM-line topology family and none sets `coupler_freq_hz` locally, so every
  coupler resolves the technology preset's value (`ipm_default` is 8 GHz).
  Only the historical `ipm_2c_coupler_edit.yaml` still pins 10 GHz.
- Couplers name the two lines they join by endpoint port number (`port in
  signal`, `port in pump`, `port out signal`, `port out pump`).
- For coupler-count comparisons and presentation runs, use only these four
  canonical files. Do not substitute `*_coupler_edit.yaml`,
  `*_no_coupler.yaml`, or unrelated legacy examples.
- Build and map artifacts belong below `outputs/`. For batched workflows,
  pass multiple design paths and one output root; each design receives a
  subdirectory named after its design directory or YAML stem.
- Repeated IPM sections use the declarative `repeat` group: `ipm_7c` and
  `ipm_20c` do. `ipm_2c` and `ipm_3c` write their sections out flat, because a
  `repeat` of count 1 or 2 costs more lines than it saves. Omitted `rows`,
  `cells`, `between`, and trailing inter-coupler fields resolve from the
  selected technology preset.
- `input_ports` and `output_ports` carry their own launch CPW from the preset.
  Do not add a separate `cpw` block beside a port block; that emits two CPWs
  in series.
- Dielectric loss and fabrication scatter are technology-backed design
  parameters. Global values belong under `parameters`; IPM/JTL blocks may also
  set local scatter fields and `tan_delta` for their shunt capacitors.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
