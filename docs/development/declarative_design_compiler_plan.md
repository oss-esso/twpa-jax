# Declarative design compiler — implementation plan

**Status:** planned, not started.
**Prerequisite:** the convergence/fold work on `dev` is finished and merged to `main`.
**Written:** 2026-08-07.

## Goal

Every TWPA device is described by one YAML file compiled by one generic
compiler, instead of by a bespoke Python builder function per architecture.
The refactor is complete when `designs/ipm.yaml` compiled through the new
path reproduces `make_ipm()` element-for-element and matrix-for-matrix, and
a second, structurally different device (`designs/uniform_jtwpa.yaml`)
compiles with no new Python.

Target boundary:

```text
design.yaml
     |
     v
generic design compiler        <- new
     |
     v
flat list[Element]             <- unchanged IR
     |
     v
build_matrices()               <- unchanged
     |
     v
C, G, K, Bphi, Ic, ports
```

## Scope boundary

This task covers **design definition and circuit generation only**. It does
not touch pump solving, gain solving, compression, gain maps, continuation,
solver backends, experiment/campaign definitions, or simulation result
formats. Stop once design generation is unified and the IPM is proven
equivalent; generalizing the experiment layer is a separate later task.

---

## Current state analysis

### The existing builder is already cursor-shaped

`make_ipm()` (`src/twpa_solver/builders/ipm.py:826-967`) is a pure forward
pass. It threads exactly three cursors and never looks ahead at the total
circuit size:

| cursor | variable | meaning |
| --- | --- | --- |
| signal rail node | `j_top` | next free node on the top/signal rail |
| pump rail node | `j_bottom` | next free node on the bottom/pump rail |
| cell index | `curr_mod_idx` | global junction-cell counter across all JTL rows |

The third one is easy to miss and is not optional: `add_jtl`
(`ipm.py:625`) both consumes `mod_start_idx` and returns the updated cell
index, because `cell_index` must be contiguous `0..n_cells-1` in element
order across every JTL row in the device. `tests/test_ipm_role_tags.py`
asserts exactly that. The compiler must therefore carry a cell-index cursor
alongside the two node cursors.

The atomic helpers already return next-free-node, which is the cursor
protocol the new schema wants:

| helper | line | returns |
| --- | ---: | --- |
| `add` | 549 | none (append only; infers `role` from `kind`) |
| `add_jj` | 562 | none (stamps `Lj`+`Cj` between two given nodes) |
| `add_tl_element` | 578 | none (one TL cell: shunt `C`, series `L`) |
| `add_jtl_element` | 589 | none (one JTL cell: shunt `Cg` + `add_jj`) |
| `add_tl` | 610 | `int` — next free node |
| `add_jtl` | 625 | `(int, int)` — next free node, next cell index |
| `add_coupling` | 656 | none |
| `add_edge_coupled_directional_coupler` | 674 | `(int, int)` — both rails |

### Finding 1 — the row loop is a modular conditional

```python
# ipm.py:865-916
for i in range(1, params.num_rows):
    <jtl row of array_length cells, with Cg/2 caps at both ends>
    if i % params.arrays_per_dc == 0:
        <long_TL on top> + <coupler_section_TL on bottom> + <coupler>
    else:
        <short_TL on top>
<final jtl row, outside the loop>
```

Schema v1 forbids conditionals in YAML. The only conditional-free encoding
is nested repetition with literal counts:

```text
repeat(n_periods) {
    repeat(arrays_per_dc - 1) { jtl_row; short_TL }
    jtl_row; long_TL; section_TL; coupler
}
repeat(remainder) { jtl_row; short_TL }
jtl_row                      # final row, no trailing TL
```

with `n_periods = (num_rows - 1) // arrays_per_dc` and
`remainder = (num_rows - 1) % arrays_per_dc`. Verified against every stored
design (coupler count = `1 + n_periods`, matching the `Nc` in each name):

| design | num_rows | arrays_per_dc | n_periods | remainder | couplers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ipm_2c_fixed` | 6 | 3 | 1 | 2 | 2 |
| `ipm_3c_fixed` | 6 | 2 | 2 | 1 | 3 |
| `ipm_7c_fixed` | 7 | 1 | 6 | 0 | 7 |

Two consequences, both of which override the original spec:

1. **Nested `repeat` must work.** The spec left "work cleanly or explicitly
   reject" as an option; rejecting it makes 2c and 3c inexpressible. Depth
   2 is required. Cap the implementation at depth 2 and reject deeper with a
   clear error, so path generation stays unambiguous.
2. **`designs/ipm.yaml` describes one concrete device, not a parametric IPM
   family.** The repeat counts derive from `num_rows`/`arrays_per_dc`, and
   §8 of the spec bans an expression language, so the YAML carries the
   literal counts `1`, `2`, `418`, `6`. Name the file `designs/ipm_2c.yaml`
   to make that honest. A different `(num_rows, arrays_per_dc)` shape is a
   different YAML file, which is consistent with "new topology -> YAML
   only". The legacy `IPMParams(num_rows=N)` route stays available for
   callers that need to sweep the shape numerically
   (`scripts/periodicity_campaign.py:45-47`,
   `scripts/tune_lj_to_themis.py:74-76`).

### Finding 2 — mutual inductors reference element names, not nodes

`add_coupling` (`ipm.py:656-671`) stamps:

```python
add(circuit, f"Cc{n_t}_{n_b}", n_t, n_b, Cc_cell, "coupling_capacitor")
add(circuit, f"K{n_t}_{n_b}",
    f"L{n_t}_{n_t + 1}",          # n1 is an ELEMENT NAME
    f"L{n_b}_{n_b + 1}",          # so is n2
    K_ind, "mutual_inductor_k")
```

`Element.n1`/`Element.n2` are typed `Any` (`ipm.py:92-100`) precisely because
a `mutual_inductor_k` stores two inductor *names* where every other kind
stores two integer nodes. `build_matrices` resolves them through a
`linear_L` name lookup (`ipm.py:1053-1059`).

Consequences for the compiler:

- `resolve_node(path)` is not sufficient. A parallel `resolve_element(path)`
  is needed so a `raw_element` or patch can target a mutual.
- The `raw_element` block and the patch system must both accept
  name-valued endpoints, and must validate that a `mutual_inductor_k`'s
  endpoints name elements that exist and are `linear_inductor`s.
- Element names are a pure function of node numbers
  (`f"{prefix}{n1}_{n2}"`). The compiler must not invent its own naming
  scheme; reusing the existing helpers gets this for free, and any block
  that stamps its own names breaks parity.

### Finding 3 — two stored designs have colliding rails

`build_matrices` derives node indices from the sorted set of integers seen
in `n1`/`n2` (`ipm.py:1001-1004`). If the signal rail counts up past
`start_node_bot`, the two rails silently merge and element names collide.
Measured across `designs/`:

| design | `start_node_bot` | `top_end_node` | duplicate element names |
| --- | ---: | ---: | ---: |
| `ipm_2c_fixed` | 10 000 | 4 577 | 0 |
| `ipm_3c_fixed` | 10 000 | 6 825 | 0 |
| `ipm_7c_old` | 10 000 | 7 230 | 0 |
| `ipm_7c_oldlen_newcl` | 10 000 | 7 230 | 0 |
| `ipm_7c_ideal_m25db_8ghz` | 100 000 | 10 960 | 0 |
| `ipm_7c_new` | 100 000 | 11 030 | 0 |
| **`ipm_7c_fixed`** | **10 000** | **11 030** | **1 642** |
| **`ipm_7c_lj158_cg66`** | **10 000** | **11 030** | **1 642** |

The two flagged directories short the signal and pump rails together over
roughly 1 030 nodes. `IPMParams.start_node_bot` already defaults to
`100_000` (`ipm.py:156-198`), so these are stale artifacts built before that
default landed. `ipm_7c_new` is the same topology at the corrected base and
supersedes `ipm_7c_fixed`.

No Python, PowerShell, JSON, or Markdown under `scripts/`, `tests/`, `src/`,
`experiments/`, `workflows/`, or `docs/` references either stale directory —
they are dead data.

Decision: the compiler treats a cursor collision as a **hard error**. The
stale directories are not repaired and not silently reproduced; deleting
them is a separate one-line cleanup (Phase 8) kept out of the refactor's
diff.

### Other inventory facts that constrain the design

- **`ipm_summary.json["params"]` is literally `dataclasses.asdict(IPMParams)`.**
  It is already a stored, complete params record — the natural seed for
  hand-writing `designs/ipm_2c.yaml`. What it does *not* record is
  `coupler_mode`, which `assert_source_topology` (`ipm.py:1430-1458`)
  currently recovers by brute-forcing all three of
  `("cached", "ideal", "optimize")` (`ipm.py:1404`). The YAML stating the
  mode explicitly is a free improvement.
- **Element order is construction order, grouped by block**, never sorted.
  `build_matrices` is order-independent for `C`/`G`/`K` (nodes come from a
  sorted set), but `Bphi` columns and the `Ic`/`Lj` arrays are filled in
  element-list order, so column `j` is the `j`-th `josephson_inductor`
  encountered. The parity test must compare the element list as an ordered
  sequence, not as a set.
- **The stored `ipm_elements.csv` files have six columns**
  (`idx,name,node1,node2,value,kind`) and predate the `role`/`cell_index`
  columns that `write_outputs` now emits (`ipm.py:1224-1228`). Role and
  cell-index parity therefore cannot be checked against disk; it must be
  checked in memory, legacy-builder output against compiler output.
- **Profiles and scatter are a caller-side concern.**
  `build_component_plan` (`ipm.py:752-796`) sizes its arrays from
  `n_cells = num_rows * array_length` *before* `make_ipm` runs, then feeds
  per-cell values in through `plan`. Schema v1 does not model this; the
  legacy `IPMParams` entry point keeps it unchanged.
- **`generate_and_append_coupler` (`ipm.py:741-749`) is a pass-through** to
  `add_edge_coupled_directional_coupler`. The coupler block adapter can call
  the latter directly.
- **`jc_doc.py` is a second, independent builder** with string node names and
  an insertion-order `node_map` (`jc_doc.py:103-107`, `184-313`). Its node
  index convention is incompatible with `ipm.py`'s sorted-integer one. It is
  out of scope; migrating the seven JC-parity fixtures is follow-up work.
- **`le_gal_2025.py` bypasses `Element` entirely**, assembling sparse
  matrices by hand (`le_gal_2025.py:102-106`). It cannot be expressed in a
  block language without first being rewritten onto the `Element` IR. Out of
  scope.

---

## Decisions locked before implementation

| # | Decision | Rationale |
| ---: | --- | --- |
| D1 | **Extract and reroute.** Move the atomic helpers and block builders into dedicated modules; `make_ipm` and the YAML compiler both call the *same* block builders. | One code path. Parity becomes a structural invariant rather than a coincidence that two parallel implementations happen to agree today and drift tomorrow. |
| D2 | Nested `repeat` is supported to **depth 2**; depth 3+ is a validation error. | Forced by Finding 1. Depth 2 covers every IPM shape; capping it keeps hierarchical paths unambiguous. |
| D3 | Cursor collision is a **hard error**, checked at compile time. Stale colliding designs are reported, not repaired. | Finding 3. The guard is cheap and obviously correct; repairing stored designs changes their matrices and is physics-scope creep. |
| D4 | `designs/ipm_2c.yaml` is one concrete device with literal repeat counts. `IPMParams` stays the parametric-shape entry point. | Forced by Finding 1 + the spec's ban on an expression language. |
| D5 | Schema v1 models the **resolved nominal topology only** — no profiles, no scatter, no Monte Carlo. | Spec §15. The legacy `ComponentPlan` path is preserved untouched behind `IPMParams`. |
| D6 | `Element` is not modified. | It is the solver IR; every downstream consumer depends on it. |

---

## What we are NOT doing

- Not rewriting `build_matrices`, CPW/coupler physics, `edge_coupled_cpw`,
  `optimize_coupler_geometry`, `make_ideal_coupler`, loss semantics, port
  conventions, or scatter/profile mathematics. Values must come out bitwise
  identical.
- Not modelling profiles, scatter, sweeps, Monte Carlo, optimization
  variables, or fabrication ensembles in the YAML schema.
- Not migrating `jc_doc.py`'s seven JC-parity fixtures, `kimpa.py`, or
  `le_gal_2025.py`.
- Not updating experiment scripts. The experiment layer must not need to
  know the design internals changed.
- Not putting solver settings (Newton/GMRES tolerances, continuation,
  pump/signal sweeps, gain-map or compression settings, plotting) into the
  design schema.
- Not building an expression language, conditionals, macros, inheritance,
  wildcard patch selectors, a GUI, or a SPICE parser.
- Not repairing or regenerating any stored design's matrices.

---

## Prerequisites

- [ ] Convergence/fold work on `dev` complete and merged to `main`.
- [ ] Working tree clean; branch from `main`.
- [ ] `python -c "import twpa_solver; print(twpa_solver.__file__)"` prints a
      path inside this repo. An editable install has previously shadowed the
      repo from `D:\tmp\finalclone`; verify before trusting any test run.
- [ ] Full suite green at baseline:
      `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_design_base`
- [ ] `pyyaml` available (add to project dependencies if absent).

---

## Phase 0 — Baseline capture

### Overview

Freeze what "identical" means before changing anything, so every later phase
compares against bytes on disk rather than against a re-derived expectation.

### Changes

Write a throwaway script under the scratchpad (not committed) that, for
`ipm_2c_fixed` and `ipm_3c_fixed`, rebuilds via
`assert_source_topology` -> `make_coupler_discrete` -> `make_ipm` and
pickles the resulting `list[Element]` plus the `ends` dict to
`D:\tmp\design_baseline\`. Also record `build_matrices` output hashes
(`C.data`, `C.indices`, `C.indptr` and the same for `G`/`K`/`Bphi`, plus
`Ic`).

### Success criteria

**Automated:** baseline pickles exist for both designs; re-running the
capture twice produces byte-identical pickles (determinism check).
**Manual:** element counts match `ipm_summary.json["counts"]` for both
designs (2c: 16 312 total).

---

## Phase 1 — Extract primitives, no behaviour change

### Overview

Pure code movement. Nothing about the generated circuit may change.

### Changes

#### 1. `src/twpa_solver/builders/primitives.py` (new)
Move, unmodified: `Element`, `LossSpec`, `add`, `add_jj`, `add_tl_element`,
`add_jtl_element`, `add_tl`, `add_jtl`, `add_coupling`.

#### 2. `src/twpa_solver/builders/coupler.py` (new)
Move, unmodified: `CouplerGeometry`, `CouplerDiscrete`, `edge_coupled_cpw`,
`optimize_coupler_geometry`, `calculate_discrete_params`,
`make_ideal_coupler`, `make_coupler_discrete`,
`add_edge_coupled_directional_coupler`, `generate_and_append_coupler`.

#### 3. `src/twpa_solver/builders/matrices.py` (new)
Move, unmodified: `build_matrices`.

#### 4. `src/twpa_solver/builders/ipm.py`
Re-export every moved name at module scope so
`from twpa_solver.builders.ipm import Element, build_matrices, ...`
keeps working. Retain `IPMParams`, `ComponentPlan`, `build_component_plan`,
`make_ipm`, `write_outputs`, `assert_source_topology`,
`build_variant_design`, and the CLI here.

### Success criteria

**Automated:**
`python -m pytest -q tests/test_ipm_role_tags.py tests/test_ipm_component_plan.py tests/test_variant_design.py tests/test_dissipation_unit.py tests/test_kinetic_persistence.py`
plus the full suite. Element pickles match Phase 0 byte-for-byte.
**Manual:** `python -m twpa_solver.builders.ipm --help` still works;
`grep -rn "from twpa_solver.builders.ipm import" src/ scripts/ tests/ experiments/`
shows no broken symbol.

### Rollback

Single revert; no semantic change to revert around.

---

## Phase 2 — Block builder layer, and reroute `make_ipm` through it

### Overview

Define the block interface, implement the block set on top of the Phase 1
primitives, then rewrite `make_ipm`'s body as a sequence of block calls.
This is the phase that makes D1 real: after it, there is one stamping path.

### Changes

#### 1. `src/twpa_solver/builders/blocks.py` (new)

A `BuildContext` carrying the mutable compile state:

```python
@dataclass
class BuildContext:
    circuit: list[Element]
    cursors: dict[str, int]        # rail name -> next free node
    cell_index: int                # global junction-cell counter
    ground: int
    named_nodes: dict[str, int]    # hierarchical path -> node
    named_elements: dict[str, str] # hierarchical path -> element name
    blocks: list[BlockRecord]      # hierarchy, for design_resolved.json
```

Each block builder has the signature:

```python
def build_<block>(ctx: BuildContext, cfg: Mapping[str, Any], path: str) -> None
```

and mutates `ctx` in place: appends to `ctx.circuit`, advances the cursors
it declares, advances `ctx.cell_index` if it stamps junctions, and registers
its own path plus any addressable sub-nodes.

Blocks for schema v1:

| block | cursors | wraps |
| --- | --- | --- |
| `port` | 1 | `add(kind="port")` |
| `resistor` | 1 | `add(kind="resistor")` |
| `transmission_line` | 1 | `add_tl` |
| `jj_line` | 1 | first-cell `add_jtl_element` + `add_jtl` + end cap |
| `directional_coupler` | 2 | `add_edge_coupled_directional_coupler` |
| `raw_element` | 0 | `add` |

`jj_line` must reproduce `make_ipm`'s row exactly: leading cell with
`Cg/2` via `add_jtl_element`, `array_length - 1` interior cells via
`add_jtl`, then the trailing `C{j}_{ground}_JTL_end` cap at `Cg/2` carrying
`role="jtl_cg"` and `cell_index = curr_mod_idx - 1` (`ipm.py:895-898`). It
advances both the node cursor and `ctx.cell_index`.

`directional_coupler` takes the `CouplerDiscrete` produced by
`make_coupler_discrete` and advances both named cursors.

#### 2. `src/twpa_solver/builders/registry.py` (new)

```python
BLOCK_BUILDERS: dict[str, BlockBuilder] = {...}
```

with a `register_block(name)` decorator. Nothing else in the compiler may
branch on block type.

#### 3. `src/twpa_solver/builders/ipm.py`
Rewrite `make_ipm`'s body as calls into `BLOCK_BUILDERS`, preserving the
signature `(params, coupler, mod_array=None, *, plan=None) -> (circuit, ends)`
and the `mod_array`/`plan` mutual exclusion. `plan`/`mod_array` are threaded
into `jj_line` as an optional explicit per-cell value source.

### Success criteria

**Automated:** full suite; element pickles still match Phase 0 byte-for-byte
for both 2c and 3c. This is a strict gate — a single reordered or renamed
element fails it.
**Manual:** `BLOCK_BUILDERS` has no IPM-specific entry; no CPW or Josephson
physics has moved into `blocks.py` beyond calling the Phase 1 primitives.

### Rollback

Revert Phase 2 only; Phase 1 stands alone and is independently useful.

---

## Phase 3 — Design model and compiler core

### Overview

The compiler, driven by an in-memory dict. No YAML parsing yet, so the
compiler and the file format can be tested separately.

### Changes

#### 1. `src/twpa_solver/design/model.py` (new)

```python
@dataclass(frozen=True)
class CompiledDesign:
    name: str
    elements: list[Element]
    cursors: dict[str, int]
    named_nodes: dict[str, int]
    named_elements: dict[str, str]
    ports: dict[int, PortRecord]
    blocks: list[BlockRecord]
    metadata: dict[str, Any]

    def resolve_node(self, path: str) -> int: ...
    def resolve_element(self, path: str) -> str: ...
```

`resolve_element` exists because of Finding 2. Both raise a
`DesignResolutionError` naming the closest matching paths on a miss.

#### 2. `src/twpa_solver/design/parameters.py` (new)
`${name}` substitution only. A whole scalar field is either a literal or
exactly one `${name}`; partial interpolation (`"${a}e-12"`) and arithmetic
are rejected. Unknown name -> `DesignParameterError`. Declared-but-unused
parameters are reported at `--strict` but are not an error by default.

#### 3. `src/twpa_solver/design/compiler.py` (new)
`compile_design(spec: Mapping, *, coupler_mode: str) -> CompiledDesign`.
Walks `topology` in order, expands `repeat` (depth <= 2), builds
hierarchical paths, dispatches through `BLOCK_BUILDERS`, then applies
`patches`.

**Cursor collision guard (D3):** after compilation, assert that no integer
node was reached by more than one cursor's monotone range, and that no
element name occurs twice. On failure raise `DesignCollisionError` naming
the two cursors, the overlapping node range, and a sample colliding element
name.

**Determinism (spec §23):** the walk is over ordered YAML sequences; the
registry is a plain dict, iterated only by key lookup, never by traversal.
Never iterate a `set` in a way that reaches circuit generation.

### Success criteria

**Automated:** new `tests/test_design_compiler.py` covers parameter
substitution, single-cursor advance (TL of 10 cells advances by exactly 10),
two-cursor advance through a coupler, `repeat count: 3` producing three
blocks with paths `period[0..2]`, nested repeat paths, cell-index continuity
across two `jj_line` blocks, hierarchical node resolution, and the collision
guard firing on a hand-built colliding spec.
**Manual:** compiling the same spec twice gives identical element lists.

---

## Phase 4 — YAML schema and validation

### Overview

The file format and its error messages. Validation quality is a deliverable,
not a nicety.

### Changes

#### 1. `src/twpa_solver/design/schema.py` (new)
Validate `schema_version` (must be `1`), `name`, `ground`, `parameters`,
`cursors`, `topology`, `patches`. Reject unknown top-level keys and unknown
block fields. No silent defaults for malformed topology.

#### 2. `src/twpa_solver/design/io.py` (new)
`load_design(path)` — `yaml.safe_load` only, never `yaml.load`. Preserve
mapping order (Python 3.7+ dicts do this natively).

#### 3. Errors
One exception hierarchy rooted at `DesignError`. Every message carries the
YAML path (`topology[3].repeat.topology[1].cells`), what was found, and what
was expected.

### Success criteria

**Automated:** `tests/test_design_schema.py` asserts a specific, actionable
error for each of: unknown block type, unknown parameter, unknown cursor,
duplicate names at one hierarchy level, missing required block field,
invalid node reference, patch target not found, patch target ambiguous,
negative repeat count, repeat nesting depth 3, unsupported schema version,
partial parameter interpolation, `yaml.load`-style tags.
**Manual:** each message identifies the offending line or path well enough
to fix without reading compiler source.

---

## Phase 5 — `designs/ipm_2c.yaml` and the golden parity gate

### Overview

The migration, and the test the whole refactor stands or falls on.

### Changes

#### 1. `designs/ipm_2c.yaml` (new)
Hand-written from `designs/ipm_2c_fixed/ipm_summary.json["params"]`, which
is already a complete `asdict(IPMParams)`. Structure per Finding 1, with
literal counts:

```yaml
schema_version: 1
name: ipm_2c
ground: 0

parameters:
  Lj: 123.9e-12
  Cj: 145.0e-15
  Cg: 66.0e-15
  array_length: 418
  short_tl: 90
  long_tl: 900
  coupler_section: 800
  # coupler geometry: cached mode, from the stored params record
  coupler_mode: cached
  coupler_width_um: 39.897
  coupler_gap_um: 44.762
  coupler_gap_to_ground_um: 10.5973385055
  coupler_length_um: 3787.7

cursors:
  signal: 1
  pump: 10000

topology:
  - {type: port,     name: in_signal,  cursor: signal, port: 1, ...}
  - {type: resistor, name: r_in,       cursor: signal, value: 50.0}
  # len1 = 0 -> no lead-in TL on the signal rail
  - {type: port,     name: in_pump,    cursor: pump,   port: 3, ...}
  - {type: resistor, name: rm_in,      cursor: pump,   value: 50.0}
  # len3 = 0 -> no lead-in TL on the pump rail
  - {type: directional_coupler, name: coupler_in, cursors: [signal, pump], ...}

  - repeat:
      count: 1                     # n_periods = (6-1)//3
      name: period
      topology:
        - repeat:
            count: 2               # arrays_per_dc - 1
            name: row
            topology:
              - {type: jj_line, name: array, cursor: signal, cells: ${array_length}, ...}
              - {type: transmission_line, name: link, cursor: signal, cells: ${short_tl}}
        - {type: jj_line, name: array, cursor: signal, cells: ${array_length}, ...}
        - {type: transmission_line, name: long,    cursor: signal, cells: ${long_tl}}
        - {type: transmission_line, name: section, cursor: pump,   cells: ${coupler_section}}
        - {type: directional_coupler, name: coupler, cursors: [signal, pump], ...}

  - repeat:
      count: 2                     # remainder = (6-1) % 3
      name: tail_row
      topology:
        - {type: jj_line, name: array, cursor: signal, cells: ${array_length}, ...}
        - {type: transmission_line, name: link, cursor: signal, cells: ${short_tl}}

  - {type: jj_line, name: final_array, cursor: signal, cells: ${array_length}, ...}
  - {type: transmission_line, name: out_signal_tl, cursor: signal, cells: 50}  # len2
  - {type: resistor, name: r_out,    cursor: signal, value: 50.0}
  - {type: port,     name: out_signal, cursor: signal, port: 2, ...}
  # len4 = 0 -> no trailing TL on the pump rail
  - {type: resistor, name: rm_out,   cursor: pump,   value: 50.0}
  - {type: port,     name: out_pump, cursor: pump,   port: 4, ...}
```

Note the port/resistor emission order at each end must match
`make_ipm` exactly: **port then resistor** at the inputs
(`ipm.py:841-842`, `845-846`), **resistor then port** at the outputs
(`ipm.py:947-948`, `951-952`). This asymmetry is real and is a parity trap.

`cursors.pump: 10000` reproduces the stored 2c design. New designs should
use `100000`.

#### 2. `tests/test_design_ipm_parity.py` (new) — the gate

```python
legacy_elements, legacy_ends = make_ipm(params_from_stored_summary, coupler)
compiled = compile_design(load_design("designs/ipm_2c.yaml"),
                          coupler_mode="cached")
```

Assert, in order:

- `len(compiled.elements) == len(legacy_elements) == 16312`
- element-by-element, at the same index: `name`, `n1`, `n2`, `kind`, `role`,
  `cell_index` all exactly equal; `value` equal exactly where it is copied
  through unchanged, and within `rel=1e-12` where it passes through the CPW
  float path
- `ports` identical; `compiled.cursors["signal"] == legacy_ends["top_end_node"] == 4577`,
  `compiled.cursors["pump"] == legacy_ends["bottom_end_node"] == 11558`,
  cell count `== legacy_ends["jj_mod_used"] == 2508`
- then `build_matrices` on both, asserting for each of `C`, `G`, `K`,
  `Bphi`: equal `shape`, equal `nnz`, equal `indices` and `indptr` arrays
  (sparsity structure), and `data` equal within `rel=1e-12`; plus `Ic` and
  every port vector

Both matrix sets are additionally compared against the stored
`designs/ipm_2c_fixed/{C,G,K,Bphi}.npz`, reusing the philosophy of
`tests/test_ipm_role_tags.py` rather than weakening it.

Add the same comparison for `designs/ipm_3c.yaml` if writing a second YAML
is cheap once the first exists — 3c exercises `remainder = 1` and
`n_periods = 2`, a different nesting arithmetic than 2c. Treat it as
optional; 2c is the gate.

### Success criteria

**Automated:** `tests/test_design_ipm_parity.py` passes; full suite passes.
**Manual:** the parity test has been *seen to fail* — perturb one YAML count
by 1, confirm a clear failure, revert. A gate never observed failing is not
a gate.

### Rollback

The YAML and its test are additive. Reverting Phase 5 leaves Phases 1-4
(dead but harmless) in place.

---

## Phase 6 — Raw elements, hierarchical addressing, patches

### Overview

The escape hatches. Deliberately after parity, so they cannot mask a
mismatch.

### Changes

#### 1. `raw_element` block
Accepts `nodes:` as either two integers or two hierarchical path strings
(`period[0].array.cell[120].left` / `.right`). For
`kind: mutual_inductor_k`, endpoints resolve through `resolve_element`
instead, and the compiler validates both targets exist and are
`linear_inductor`s. Both spellings must produce an identical `Element`.

#### 2. Sub-block path registration
`jj_line` and `transmission_line` register `cell[i].left` / `cell[i].right`;
`directional_coupler` registers cells on both rails. Registration is lazy
where possible — a 2 508-cell device must not pay to materialize every path
string on every compile. Store `(block_path, base_node, stride, count)` and
resolve on query.

#### 3. `src/twpa_solver/design/patches.py` (new)
Three actions, exact-target only, applied after topology expansion:

| action | target form | behaviour |
| --- | --- | --- |
| `add` | `nodes: [...]` | append one `Element` |
| `set` | `<block path>.<field>` | change exactly one element's value |
| `remove` | `<block path>.<field>` | delete exactly one element |

Zero matches or more than one match is a hard error naming the target and
the match count. No wildcards, no statistical modification, no profiles.

### Success criteria

**Automated:** `tests/test_design_patches.py` — integer-node and path-node
`raw_element` produce identical `Element`s; a mutual referencing a
non-inductor is rejected; `set` changes exactly one element and leaves
element count and every other value untouched; `remove` drops exactly one;
missing and ambiguous targets both raise. Parity test still passes
(a design with no `patches:` key is unaffected).
**Manual:** compiling 2c with path registration enabled stays within a
factor of ~2 of Phase 5's compile time.

---

## Phase 7 — CLI, artifacts, second device, documentation

### Overview

Make it usable and prove it is not IPM-specific.

### Changes

#### 1. `src/twpa_solver/design/__main__.py` (new)

```powershell
python -m twpa_solver.design --design designs/ipm_2c.yaml --outdir outputs/design_test --write-matrices
```

Flags: `--design`, `--outdir`, `--write-matrices`, `--coupler-mode`
(default `cached`), `--overwrite`, `--strict`, and `--draw` only if the
existing drawing code is reusable without modification. **No simulation
options.** This builds circuits; it does not run HB.

#### 2. Generic artifacts
Emit `elements.csv`, `ports.csv`, `design_summary.json`,
`design_resolved.json`, and with `--write-matrices` also
`C.npz`, `G.npz`, `K.npz`, `Bphi.npz`, `arrays.npz`.

`design_resolved.json` records schema version, design name, resolved
parameters, block hierarchy with final node ranges per block, resolved
cursors, `coupler_mode`, the resolved `CouplerDiscrete`, and the source YAML
path. It is an output artifact, not something users edit.

**`arrays.npz` must carry the same keys `load_circuit` requires**
(`src/twpa_solver/core/circuit.py:131-173`): `nodes`, `Ic`, `Lj`,
`phi0_reduced`, `port_numbers`, `port_indices`. Otherwise the solver cannot
load a generically-compiled design.

#### 3. Legacy IPM CLI preserved
`python -m twpa_solver.builders.ipm ...` keeps working and keeps writing
`ipm_elements.csv`, `ipm_ports.csv`, `ipm_summary.json`, `ipm_arrays.npz`.
`write_outputs` is unchanged. Both CLIs converge on the same block builders
underneath.

#### 4. `designs/uniform_jtwpa.yaml` (new)
A single `jj_line` between two ports, no coupler, no pump rail. Must compile
and validate; no experiment is added for it.

#### 5. `docs/design_format.md` (new)
Design vs experiment distinction; parameters; cursors; topology; the block
catalogue; `repeat` including the depth-2 limit and *why* it exists;
hierarchical names; `raw_element`; patches; how to build a design; how to
add a block. Full IPM and uniform-JTWPA examples. The abstraction boundary:

```text
expressible from existing blocks   -> edit YAML only
new reusable topology primitive    -> add a block builder
new constitutive/nonlinear law     -> extend solver/model code
```

Document the known gaps explicitly: no profiles/scatter in v1, no
conditionals, no expressions, `jc_doc`/`kimpa`/`le_gal_2025` not migrated.

#### 6. `CLAUDE.md`
Add a short section pointing at `docs/design_format.md`, recording the
depth-2 nested-repeat rule, the cursor-collision guard, and the fact that
`designs/*.yaml` are concrete devices rather than parametric templates.

### Success criteria

**Automated:** `tests/test_design_cli.py` — the generic CLI produces all
expected artifacts for both YAMLs; `load_circuit` successfully loads the
generically-written 2c output and its `Ic` matches the stored design;
`uniform_jtwpa.yaml` compiles with the expected element count; the legacy
IPM CLI still writes legacy filenames.
**Manual:** `docs/design_format.md` lets someone write a third design
without reading compiler source.

---

## Phase 8 — Stale design cleanup (separate commit)

Not part of the refactor diff.

`designs/ipm_7c_fixed` and `designs/ipm_7c_lj158_cg66` each carry 1 642
duplicate element names from the rail collision in Finding 3, and no code in
the repo references either. `designs/ipm_7c_new` is the same topology built
at `start_node_bot = 100000` and supersedes `ipm_7c_fixed`.

Proposed, pending confirmation at the time: delete both directories in their
own commit, with the measured collision numbers in the message. **Do not
bundle this with the refactor** — it is a data change, and mixing it into an
architecture diff makes both harder to review and to revert.

---

## Testing strategy

### Project maturity

**Established production.** The solver has published measurements resting on
these designs. The bar is exact reproduction, not "close enough".

### Layers

| layer | file | what it protects |
| --- | --- | --- |
| golden parity | `tests/test_design_ipm_parity.py` | the entire refactor; element-for-element and matrix-for-matrix |
| compiler unit | `tests/test_design_compiler.py` | cursors, repeat, nesting, paths, determinism, collision guard |
| schema/errors | `tests/test_design_schema.py` | every invalid config fails with an actionable message |
| patches | `tests/test_design_patches.py` | exact-target semantics; no collateral change |
| CLI/IO | `tests/test_design_cli.py` | artifacts, `load_circuit` compatibility, legacy CLI |
| existing | `test_ipm_role_tags`, `test_ipm_component_plan`, `test_variant_design`, `test_dissipation_unit`, `test_kinetic_persistence` | unchanged, must stay green at every phase |

### Determinism

Compile 2c twice in one process and once in a fresh subprocess with
`PYTHONHASHSEED` set to two different values; all three element lists must be
identical. This catches accidental set/dict-traversal order leaking into
circuit generation (spec §23).

### Mutation verification

Every new gate must be observed failing before it is trusted. At minimum:
perturb one YAML repeat count; swap the port/resistor emission order at one
end; drop one `Cg/2` half-cap; renumber `cursors.pump` to force a collision.
Each must fail loudly, then be reverted.

### Run command

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_design --run-slow
```

`--basetemp` off the repo avoids the Windows ACL issue on `.pytest_tmp`.

---

## Rollback plan

Each phase is a separate commit and independently revertible:

| revert | leaves | works? |
| --- | --- | --- |
| Phase 8 | designs restored | yes |
| Phase 7 | no generic CLI, no docs | yes — compiler still importable |
| Phase 6 | no patches/raw elements | yes — parity gate still passes |
| Phase 5 | no YAML IPM | yes — legacy path unchanged |
| Phases 3-4 | no compiler | yes — blocks used only by `make_ipm` |
| Phase 2 | `make_ipm` back to inline helper calls | yes |
| Phase 1 | original single-module `ipm.py` | yes |

Phases 1 and 2 are the only ones that touch code the solver depends on
today. Both are gated by byte-identical comparison against the Phase 0
pickles, so a regression surfaces at the phase that caused it rather than
downstream.

---

## Acceptance criteria

- [ ] Existing tests pass unchanged.
- [ ] `python -m twpa_solver.builders.ipm ...` still works and still writes
      `ipm_elements.csv` / `ipm_ports.csv` / `ipm_summary.json` /
      `ipm_arrays.npz`.
- [ ] No experiment script was redesigned.
- [ ] `designs/ipm_2c.yaml` compiles through the generic compiler.
- [ ] Compiled IPM is element-for-element identical to `make_ipm` output:
      count, order, names, `n1`, `n2`, `kind`, `role`, `cell_index`, values.
- [ ] `C`/`G`/`K`/`Bphi` parity tested against both the legacy build and the
      stored `.npz`, including sparsity structure, plus `Ic` and port
      vectors.
- [ ] `designs/uniform_jtwpa.yaml` compiles with no new Python design script.
- [ ] `repeat` works, including depth 2; depth 3 is rejected with a clear
      error.
- [ ] Named cursors work; two-cursor blocks advance both.
- [ ] Stable hierarchical paths exist and resolve to correct integer nodes.
- [ ] `resolve_element` addresses mutual-inductor endpoints by name.
- [ ] Raw elements can be inserted by integer node and by path, identically.
- [ ] `set` and `remove` patches each affect exactly one element.
- [ ] Cursor collisions are a hard error.
- [ ] Every listed invalid config produces an actionable error.
- [ ] Compilation is deterministic across processes and hash seeds.
- [ ] The generic CLI builds a design without invoking HB.
- [ ] `arrays.npz` from the generic writer loads through `load_circuit`.
- [ ] `docs/design_format.md` documents the format and the abstraction
      boundary.
- [ ] Every new gate has been observed failing under mutation.

---

## Follow-up work, explicitly out of scope

1. Migrate `jc_doc.py`'s seven JC-parity fixtures. Blocked on reconciling its
   string-node, insertion-order `node_map` with `ipm.py`'s sorted-integer
   convention.
2. Migrate `kimpa.py` — three parameterized Pi-ladders plus a resonator;
   straightforward once a `pi_ladder` block exists.
3. Migrate `le_gal_2025.py` — requires first rewriting it onto the `Element`
   IR, and modelling its per-cell `brentq` equilibrium solve as a
   value-derivation step distinct from wiring.
4. Model profiles and scatter declaratively (schema v2). Needs a value-layer
   concept orthogonal to topology; `builders/profiles.py` is already
   declarative and is the natural starting point.
5. Record `coupler_mode` in legacy `ipm_summary.json` so
   `assert_source_topology` stops brute-forcing all three modes
   (`ipm.py:1404`).
6. Generalize the experiment/campaign layer. Deliberately deferred until
   this compiler is stable.
