# Implementation Plan: Object-Oriented Circuit Design Framework

Status: planned, not started. Written 2026-08-16.

Source requirements: `docs/development/circuit_design_prd.md`.
Supporting research: the cloned fabrication repository at `../Prometheus`
(`Prometheus/Packages/`), inspected per PRD section 23.

---

## Goal

A `Circuit` class that lets a fabrication engineer build any TWPA topology in
Python with fabrication vocabulary and no node integers, compiling
deterministically to the exact `Element[]` the existing solver already
consumes — with `ipm_2c` proven bit-identical to today's netlist and matrices.

---

## Current state analysis

### Simulator side

| Item | Location | Note |
| --- | --- | --- |
| Netlist IR | `builders/ipm.py:94-102` | `Element(name, n1, n2, value, kind, role, cell_index)`. Unchanged by this work. |
| Matrix assembly | `builders/ipm.py:904` | `build_matrices` → `C/G/K/Bphi/Ic`. Unchanged. |
| Block layer | `builders/blocks.py` | Already has cursors, hierarchical node/element paths (`{path}.cell[{i}].left\|right\|Lj\|Cj\|Cg`, lines 104-127), and `BlockRecord` handles. |
| YAML compiler | `design/compiler.py:318` | `compile_design`. |
| Composite expansion | `design/compiler.py:208-305` | `_expand_composites` hardcodes `input_ports`/`output_ports`/`ipm_line`/`ipm_tail` as dict literals. |
| Profile math | `builders/profiles.py:122-172` | `half_cosine` = `(1-cos(pi*t))/2`; `half_sine` routes to `custom`/`sin(pi*t/2)` at `design/compiler.py:153-154`. |
| Coupler physics | `builders/cpw_coupler.py`, `builders/ipm.py:241-505` | Two- and three-conductor. Centre ground reduced out of the C matrix at `cpw_coupler.py:113-115`. |
| Parity oracle | `tests/test_design_compiler.py:13-36` | Triangulates `make_ipm` ↔ `compile_design` ↔ stored `designs/ipm_2c_fixed/*.npz`. |

Roughly 70 percent of PRD sections 13, 14 and 16 already exists in
`builders/blocks.py`. It is driven by dictionaries rather than Python objects,
which is the actual gap.

### Three real gaps

1. **Node allocation violates PRD section 7.** `IPMParams.start_node_top = 1`
   and `start_node_bot = 100_000` (`builders/ipm.py:161-163`) are verbatim the
   scheme the PRD forbids. Integers are assigned during construction:
   `add_tl` / `add_jtl` take `n_start` and return `n_curr`
   (`builders/ipm.py:638-681`).
2. **No Python object API.** The only authoring route is YAML.
3. **Composite blocks are dict literals**, so a fabrication-shaped composition
   layer has nowhere to live.

### YAML designs that must keep compiling (11)

`generic_blocks_showcase`, `ipm_2c`, `ipm_2c_half_sine`, `ipm_2c_linear`,
`ipm_2c_showcase`, `ipm_3c`, `ipm_7c_ideal_node205`,
`ipm_explicit_blocks_showcase`, `jtwpa_418jj`, `rf_squid_2393_3wm`,
`rf_squid_uniform_showcase`.

### Fabrication side (Prometheus)

Prometheus is a pure GDS geometry generator built on `gdspy`. It contains no
netlist and no lumped-element model. `sim_export=True`
(`Packages/JTWPAs.py:28`) only substitutes placeholder polygons for junctions,
and `Packages/Utilities.py::generate_simulation_files` exports DXF and STEP for
field solvers. Nothing there is reusable as implementation. What is reusable is
vocabulary, hierarchy, construction sequence, and the geometry-to-electrical
converters — several of which this repository has already ported.

`Packages/Devices.py:39` is already the architecture PRD section 4 requires:

```python
class Device(SNSPD, Resonator, ThreeDS, Filter, JosephsonJunction, AirBridge,
             Transmon, PA, JTWPA, KITWPA, DirectionalCoupler, Waveguide, IDC,
             Launchers):
```

One facade, fourteen mixin files, no monolith. The `Circuit` class mirrors this.

#### Fabrication variant decoding

Assembly loops: `Packages/JTWPAs.py:895-941` for v1 and v2,
`Packages/JTWPAs.py:1497-1621` for v3.

| | fab v1 `JTL_v1` | fab v2 `JTL_v2_JJA` | fab v3 `JTWPA_v3` |
| --- | --- | --- | --- |
| entry point | `makeIFMJTWPA` | `makeIFMJTWPA`, `JJArray=True` | `makeIFMJTWPA_`, `CouplerType='Meander'` |
| `CellNums` | `[20, 18, 3, 2]` | `[20, 15, 3, 2]` | `[36, 18, 6, 1]` |
| structure | 2 sections x 3 rows, coupler ends each section | same | 6 rows, coupler every 2 rows |
| couplers | 2 | 2 | 3 |
| coupler model | two-line edge-coupled, -14 dB at 8 GHz | same | three-conductor centre-ground, -25 dB at 10 GHz |
| per-coupler leakage | `getCouplerLeakage` applied (`:931`) | applied | present but commented out (`:1510-1512`) |
| junctions per cell | 1 | `JJArrayNumber = 3` in series | 1, `XJJ` style |

`CellNums` semantics, from the docstring at `Packages/JTWPAs.py:962-967`:

| Fab name | Meaning | Our name |
| --- | --- | --- |
| MicroCell | junction cells per subcell | cells per array |
| MacroCell | subcells per straight segment | arrays per row |
| CellLines | segments per section | rows |
| CouplersLines | sections | sections |

Our `ipm_2c` matches the v1 family structurally: two couplers, rows linked by
short transmission lines.

#### Fabrication conventions that confirm ours

- `Packages/JTWPAs.py:1099` comments `cellRect2` as the
  "Smaller (Cg/2) cell for beginning and end of line". Our `build_jj_line`
  already halves `Cg` at the first cell and at the terminator. Independently
  arrived at the same convention.
- The fabrication `Path` is a flat numpy polyline with an implicit cursor at
  `path[-2:]`, extended by `makeTurn` and `np.append`. Purely geometric, but
  the ergonomics match PRD section 8: create, extend, auto-advance, never
  manage indices.
- Ports are coordinate dictionaries returned by builders and chained by the
  caller (`Packages/DirectionalCouplers.py:351-363`). Numbering, from the
  docstring at `:265-269`: ports 1 and 2 are the top line (signal), ports 4 and
  3 the bottom line (pump); 1 and 4 are the left edge, 2 and 3 the right.

#### Ported converters

| Prometheus | twpa_solver |
| --- | --- |
| `MicrowaveFunctions.CPW_coupler` / `CL_matrices` | `builders/cpw_coupler.py::CPWConformalCoupler._cl_matrices` |
| `getEdgeCoupledDirectionalCouplerParameters` | `builders/ipm.py::estimate_edge_coupled_directional_coupler`, `optimize_coupler_geometry` |
| `edgeCoupledCPW` | `builders/ipm.py::edge_coupled_cpw` |

PRD section 34 ("do not rewrite proven physics") is therefore already satisfied
for the coupler. The object layer calls these unchanged.

#### Three-conductor coupler status

Supported and tested. `builders/cpw_coupler.py:113-115` deletes the centre
conductor's row and column, reducing the 3x3 capacitance matrix to 2x2, so
`parameters()` indexing at `:137-155` is correct when the mapping succeeds.
For the symmetric centre-ground layout used by v3, the endpoint formulation
is valid when the branch points are initialized from the first gap. The
three-conductor path therefore evaluates the fabrication dimensions directly;
it does not use the synthesized two-line fallback. The automatic selection
boundary is -20 dB (`:164`).

---

## Decisions taken

| Question | Decision |
| --- | --- |
| v2 junction arrays | Lumped effective junction: `Lj_eff = N * Lj`, `Cj_eff = Cj / N`, `Ic` unchanged. No extra nodes. |
| v3 coupler geometry | Explicit `gaps` / `widths` / `length` input, so the fabrication numbers go in directly. |
| v3 pump routing | Free parameters, as today. |
| Approach | New `twpa_solver/circuit/` over existing builders. YAML becomes an adapter. Legacy `make_ipm` retained as the parity oracle. |
| Ports 3 and 4 | Direction convention only. No change required. |
| Profiles | Both `HalfSine` (`sin(pi*t/2)`) and `Hann` (`(1-cos(pi*t))/2`, the existing `half_cosine`). |
| `Lk_sq` | Not in the API. Users supply `Lk` and perform their own multiplication. |

---

## Stated assumption: node numbering

`compile()` allocates integers in first-touch creation order, ground to 0. That
changes today's `signal: 1...` / `pump: 10000...` integers, which would permute
the stored `designs/ipm_2c_fixed/*.npz` and shift any downstream artifact keyed
on node index.

Therefore `compile()` takes `node_numbering: "creation" | "legacy"`, default
`"creation"`. The `"legacy"` mode reproduces the old integers exactly and is what
the parity gate and the YAML adapter use, so `ipm_2c_fixed` stays byte-identical
and no published branch index moves. The base offsets live only inside that
compatibility mode, never in the authoring API, which satisfies PRD section 7's
requirement that such schemes not appear in the public design API.

Element emission order is preserved in **both** modes, so `Bphi` column order
and `Ic` order never change under any policy.

---

## What we are not doing

- No changes to solver physics: harmonic balance, pump solve, continuation,
  gain maps, compression, transient, Floquet stability, backends, experiments,
  result schemas.
- No rewrite of `build_matrices`, the branch laws, coupler theory, the CPW
  conformal mapping, or the `builders/profiles.py` mathematics.
- No resonator builder. Prometheus has roughly twenty hanged-resonator variants
  (`Packages/Resonators.py:239-1670`); we have no simulation implementation and
  no design that needs one. Future work.
- No air bridges, ground-hole protection, launchers, pads, layers, or any other
  layout-only concept (PRD section 24).
- No change to `builders/kimpa.py`, `le_gal_2025.py`, `jc_doc.py`,
  `complexity_ladder.py` — they emit `Element[]` directly and bypass the design
  compiler.
- No stochastic-scatter redesign (PRD section 20). `builders/scatter.py` is
  wired through unchanged.
- The fabrication-geometry optimizer is ported as a bounded conformal fit,
  including Prometheus' deliberate `v_even` expression for `beta_odd`.
- No deprecation or deletion of YAML designs in this work.

---

## Prerequisites

- [ ] `git status --short` clean or changes understood. The `dev` branch
      currently carries 24 modified and 16 untracked files.
- [ ] `python -c "import twpa_solver; print(twpa_solver.__file__)"` resolves
      inside this repository, not `D:\tmp\finalclone`. The editable install has
      shadowed the repository before.
- [ ] Baseline suite green:
      `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_oop_base --run-slow`

---

## Phase 0: Baseline and API mapping document

### Overview

PRD section 38 forbids committing to public names before the GDS mapping
exists. Produce it, and freeze a golden baseline so any parity drift in later
phases is attributable.

### Changes required

#### 1. GDS-to-simulation API mapping

**File**: `docs/development/GDS_API_MAPPING.md` (new)

Full table: fabrication concept, fabrication function with `file:line`, key
fabrication parameters, simulation equivalent, proposed name, dropped
geometry-only parameters, required electrical parameters, notes. The
inventory below is the starting content.

| Fab concept | Fab function | Key fab params | Proposed sim name | Geometry-only (drop) | Electrical params needed |
| --- | --- | --- | --- | --- | --- |
| Device facade | `Device` (`Devices.py:39`) | `parameters`, `mask` | `Circuit(name)` | `mask`, `layer*`, `pos`, `maskSize`, `padSize` | — |
| Route / polyline | `np.array` + `makeTurn` + `np.append` | `center`, `radius`, `angle_i/f` | `c.path("signal")` | all | — |
| Port | `ports['port1']` dict | `[x, y]`, `PortsWidth`, `PortsGap` | `c.add_port(node, number=, impedance=)` | `PortsWidth`, `PortsGap`, `TaperLength`, `minPortsSpacing` | `impedance` |
| Launcher / pad | `makeLauncher`, `standardPADPosition` | `PADsize`, `PADlen`, `direction` | absorbed into `add_port` | all | `Z0` |
| Josephson junction | `makeETHlargeJJ`, `makeJunction` | `jjdimension`, `resistHeight`, `evaporationAngle` | `c.add_jj(n1, n2, Lj=, Cj=)` | `jjdimension`, `resistHeight`, `patchLen`, undercut, angles | `Lj`, `Cj` (or `Ic`) |
| Area-to-`Lj` | `getJunctionDimension` | `Rdensity`, `shape` | accept `Lj` directly (PRD section 24 option 1) | all | `Lj` |
| Junction array in cell | `makeETHlargeJJArray` | `JJnumber` | `c.add_jj_array(..., count=)` | spacing, undercut | `Lj`, `Cj`, `count` |
| Junction / IDF unit cell | `cellRect1` + `makeETHlargeJJ` (`JTWPAs.py:1092`, `:1157`) | `CellLen`, `CellWidth`, `IDF=[w,gap,len]` | `c.add_jj_cell(...)` | `CellLen`, `CellWidth`, `IDFwidth`, `IDFgap`, `IDFlen` | `Lj`, `Cj`, `Cg` |
| Half-`Cg` end cell | `cellRect2` (`JTWPAs.py:1099`) | — | internal to `add_jj_line` | — | `Cg/2` |
| Junction line | `edgeLcell` + `cellLine` + `edgeRcell` (`:1500-1502`) | `MicroCell`, `MacroCell` | `c.add_jj_line(path, cells=, Lj=, Cj=, Cg=)` | placement `xpos`, `ypos` | `Lj`, `Cj`, `Cg`, `cells` |
| Row of arrays | `CellLines` loop (`:1497`) | `CellLinePitch` | `rows=` argument | `CellLinePitch` | `arrays_per_row` |
| Coupler, two-line | `makeEdgeCoupledDirectionalCoupler` | `coupling_dB`, `central_frequency`, `Z0` | `c.add_directional_coupler(signal, pump, coupling_db=, frequency=)` | `taper_length`, bridges, `min_feature_size` | `coupling_db`, `frequency`, `Z0` |
| Coupler, three-conductor | `makeCoupler(gaps, widths, length, type=)` | `gaps[4]`, `widths[3]`, `length` | same, `geometry=ExplicitCouplerGeometry(...)` | `type`, `delta_y`, `turns`, `orientation`, air bridges | `gaps`, `widths`, `length` |
| Cascade coupling correction | `getCouplerLeakage` | coupler index | `coupler_leakage_db(...)` | — | per-coupler dB |
| Hanged resonator | `makeHangedResonatorCell` and siblings | `freq`, `Z`, `couplingDist` | out of scope, future | `couplingDist`, `turns`, `pitchFactor` | `freq`, `Z`, `C_coupling` |
| Composite transmission line | `makeCompositeWaveguide(Zs, Ls, freq)` | `Zs`, `Ls=['lambda/2', ...]` | `c.add_transmission_line(...)` | `turns`, `pitch`, `Zmethod` | `Z`, `length`, `freq` |
| Meander / thru | `makeWGMeander`, `makeWGthru` | `straightEnd`, `pitch` | `c.add_transmission_line(...)` | all shape params | `L`, `C`, `cells` |
| Hann profile | `IDFHannModulation` (`KITWPAs.py:338`) | `Nl`, `Nsc` | `Hann(...)` | — | `start`, `stop` |
| Periodic loading | `makeLoadedTWPA(Nu, Nl, IDFload)` | `Nu`, `Nl` | future `Periodic(...)` | `IDFlen`, `IDFwidth` | per-cell `Cg` |
| Sinusoidal loading | `makeSinusoidalTWPA` | `Nu` | `Sine(...)` | — | `start`, `stop` |
| Interdigitated capacitor | `makeIDC`, `capacitanceIDC` | finger geometry | `c.add_capacitor(n1, n2, C=)` | all finger geometry | `C` |
| Air bridges | `makeETHAirBridges` and siblings | everything | not represented | all | none |
| Ground-hole protection | `draw_protection` | — | not represented | all | none |

Plus an "Intentional deviations" section recording:

- `add_` prefix instead of the fabrication `make` prefix. In the fabrication
  code `make*` means "emit geometry"; `add_*` means "attach to this circuit".
- snake_case instead of CamelCase, per the project Python standard. Nouns are
  preserved: `cell_len`, `cell_width`, `idf_width`, `micro_cell`, `macro_cell`,
  `cell_lines`, `coupling_db`, `coupler_length`.
- Units: the fabrication code works in microns throughout
  (`Devices.py:76` converts nanometres to microns). We stay in SI, and any
  micron-valued argument carries an explicit `_um` suffix, as
  `builders/ipm.py` already does.
- `signal_in` / `signal_out` / `pump_in` / `pump_out` coupler terminals have no
  fabrication counterpart; `port(1)` through `port(4)` are provided as aliases
  matching the diagram at `DirectionalCouplers.py:265-269`.
- `HalfSine` is not the fabrication team's Hann. Both are provided.
- `Lk_sq` is deliberately absent. Every fabrication TWPA signature carries it
  and the JTWPA ones ignore it, documented as
  "Not used. Implemented for matching function signature".
- Ports 3 and 4 differ in direction convention only.

#### 2. Golden baseline snapshot

**File**: `tests/data/ipm_2c_baseline.json` (new)

SHA-256 of `designs/ipm_2c_fixed/elements.csv`; per-matrix
`(shape, nnz, sum, max|.|)` for `C`, `G`, `K`, `Bphi`; `Ic` checksum; element
count; port map.

**File**: `tests/test_ipm_2c_baseline.py` (new)

Asserts today's `compile_design(ipm_2c.yaml)` still reproduces the snapshot.
This is the tripwire for every later phase.

### Success criteria

**Automated**:
`pytest -q tests/test_ipm_2c_baseline.py tests/test_design_compiler.py tests/test_variant_design.py`

**Manual**: mapping table reviewed against `Prometheus/Packages/`. Every
`make*` method in `DirectionalCouplers.py`, `JTWPAs.py` and `Waveguides.py`
appears in a row or in the explicit "not represented" list.

---

## Phase 1: Symbolic graph core and primitives

### Overview

`Node`, `ElementRef`, `Port`, `CircuitGraph`, the `Circuit` facade, all Level-1
builders, and `compile()`. No paths, no cells, no blocks. After this phase an
arbitrary branched circuit is buildable and compilable.

### Changes required

#### 1. Package skeleton

**File**: `src/twpa_solver/circuit/__init__.py` (new)

Public surface: `Circuit`, `Node`, `Port`, `CompiledCircuit`, and re-exports
from `circuit.profiles`. Nothing else public.

#### 2. Symbolic node

**File**: `src/twpa_solver/circuit/nodes.py` (new)

```python
@dataclass(frozen=True)
class Node:
    uid: int
    owner_id: int       # identity of the owning Circuit; cross-circuit use is an error
    name: str | None = None
    path: str = ""
```

`uid` is a per-circuit monotonic counter, never a solver index. Ground is
`uid == 0` by construction.

#### 3. Element handle

**File**: `src/twpa_solver/circuit/elements.py` (new)

`ElementRef` with `n1`, `n2` (`Node`, or `ElementRef` for mutual inductors),
`value`, `kind`, `role`, `name`, `path`, `cell_index`, and a `removed` flag.
PRD section 17 requires `cap.n1`, `.n2`, `.value`, `.name`, `.path`.

#### 4. Graph store

**File**: `src/twpa_solver/circuit/graph.py` (new)

`CircuitGraph` owning ordered `nodes: list[Node]`, `elements: list[ElementRef]`,
`ports: dict[int, Port]`, the `named_nodes` and `named_elements` path tables,
and a hierarchy tree. Ordered lists only — no sets or unordered dictionary
traversal anywhere that affects output (PRD section 28).

#### 5. Ports

**File**: `src/twpa_solver/circuit/ports.py` (new)

`Port(number, node, impedance)`. Duplicate `number` is an error. Port numbers
never influence allocation (PRD section 9).

#### 6. Primitive builders

**File**: `src/twpa_solver/circuit/primitives.py` (new)

`PrimitiveBuilders` mixin: `add_resistor`, `add_capacitor`, `add_inductor`,
`add_jj`, `add_mutual_inductor`, `add_port`, plus `node(name=None)`, `ground`,
`set_value(ref, value)`, `remove(ref)`. Each returns an `ElementRef`. `add_jj`
emits the `Lj` and `Cj` pair with the same role tags `builders/ipm.py:590-603`
uses today, so `tests/test_ipm_role_tags.py` semantics carry over.

#### 7. Facade

**File**: `src/twpa_solver/circuit/circuit.py` (new)

```python
class Circuit(PrimitiveBuilders, CellBuilders, BlockBuilders, ArchitectureBuilders):
```

Mirrors `Prometheus/Packages/Devices.py:39`. In this phase only
`PrimitiveBuilders` exists; later mixins join the bases as they land. Holds the
`CircuitGraph`, a name allocator, and `compile()`.

#### 8. Compiler

**File**: `src/twpa_solver/circuit/compiler.py` (new)

`compile(node_numbering="creation" | "legacy") -> CompiledCircuit`. Walks
`graph.nodes` in creation order, assigns ground to 0 then 1 through N; emits
`list[Element]` in element creation order, skipping removed elements. The
`"legacy"` mode instead groups by owning path and applies the historical
per-path base offsets, reproducing today's integers.

`CompiledCircuit` carries `elements`, `node_map`, `reverse_node_map`, `ports`,
`hierarchy`, `metadata`, and a `matrices()` method delegating to the untouched
`build_matrices`.

#### 9. Validation

**File**: `src/twpa_solver/circuit/validation.py` (new)

PRD section 29 checks, each raising with the offending hierarchical path:
duplicate explicit names, node belonging to another `Circuit` (via `owner_id`),
duplicate port numbers, removing an already-removed element, dangling
references at compile time. Cell-index, profile-domain, and coupler/path checks
land with their own phases.

#### 10. Netlist export

**File**: `src/twpa_solver/circuit/netlist_export.py` (new)

`export_netlist(compiled, path=None) -> str` in the SPICE-like form of PRD
section 22, using existing element names so it cross-references `elements.csv`.

### Success criteria

**Automated**: `pytest -q tests/test_circuit_core.py`

- Each primitive: correct endpoints, kind, value, role, hierarchy path,
  resulting `Element`, and matrix stamp against a hand-built `Element[]` through
  `build_matrices`.
- An arbitrary branched circuit (PRD section 10 escape hatch) builds and
  compiles with no `Path`.
- Ground compiles to 0.
- Compiling the same construction twice gives identical `node_map`, element
  order, names, and matrix values.
- Port numbers `{1, 3}` versus `{7, 42}` produce identical node allocation.
- Every validation error fires and names the right path.
- `export_netlist` round-trips element count and endpoints.

---

## Phase 2: Paths

### Overview

`Path` as an ergonomic view over the graph — PRD section 5 is explicit that it
is not the fundamental representation — with the auto-advance protocol every
block builder will use.

### Changes required

#### 1. Path object

**File**: `src/twpa_solver/circuit/paths.py` (new)

`Path` with `name`, `start`, `end`, `node(i)`, `nodes`, `__len__`. Internally an
ordered `list[Node]`. `Path.extend(node)` appends and moves `end` — the single
mutation point block builders call.

#### 2. Facade wiring

**File**: `src/twpa_solver/circuit/circuit.py`

`c.path(name)` creates and registers a `Path` with a fresh start node.
Duplicate name is an error. Arbitrary path count (PRD section 9).

### Success criteria

**Automated**: `pytest -q tests/test_circuit_paths.py`

- Chaining several blocks advances `signal.end` correctly and leaves
  `signal.start` fixed.
- Creating a second and third path does not alter existing path node semantics.
- `signal.node(205)` is the node the 205th append produced.
- Duplicate path name raises.
- A circuit with 6 paths and 12 ports compiles — PRD section 9's
  forward-compatibility check.

---

## Phase 3: Cells, lines, and targeted edits

### Overview

Level 2 and Level 3 for everything today's YAML supports, plus the handle
system that makes local edits possible.

### Changes required

#### 1. Handles

**File**: `src/twpa_solver/circuit/handles.py` (new)

`CellHandle` (`left`, `right`, `Lj`, `Cj`, `Cg` resolving to `Node` or
`ElementRef`), `LineHandle` (`input`, `output`, `node(i)`, `cell(i)`, `cells`),
and a `BlockHandle` base carrying `path` and `children` for the PRD section 16
hierarchy.

#### 2. Cell builders

**Files**: `src/twpa_solver/circuit/cells/tl_cell.py`, `jj_cell.py`,
`rf_squid_cell.py` (new)

`add_tl_cell`, `add_jj_cell`, `add_rf_squid_cell` — each implemented only by
calling Phase-1 primitives (PRD sections 11 and 12). `add_jj_cell` reproduces
`add_jtl_element` (`builders/ipm.py:617-635`) element for element, including
names and roles.

#### 3. Line builders

**Files**: `src/twpa_solver/circuit/blocks/transmission_line.py`, `jj_line.py`,
`rf_squid_line.py` (new)

`add_transmission_line`, `add_jj_line`, `add_rf_squid_line` — loops over cell
builders, returning a `LineHandle`. `add_jj_line` preserves the half-`Cg`
boundary convention at the first cell and at the terminator, matching
`builders/blocks.py` and confirming the fabrication `cellRect2`
(`Prometheus/Packages/JTWPAs.py:1099`). `add_rf_squid_line` mirrors
`build_rf_squid_line` (`builders/blocks.py:135`) so `rf_squid_2393_3wm.yaml`
survives Phase 7.

#### 4. Targeted edits

**File**: `src/twpa_solver/circuit/primitives.py`

`set_value` and `remove` accept an `ElementRef` reached through a handle.
`add_capacitor(line.cell(205).right, c.ground, C=...)` works because
`CellHandle.right` is a `Node` (PRD section 14).

### Success criteria

**Automated**: `pytest -q tests/test_circuit_cells.py tests/test_circuit_lines.py`

- `add_jj_line(cells=418, ...)` produces an `Element[]` identical field for
  field to `add_jtl(...)` at the same parameters, including names, roles and
  `cell_index`.
- `add_transmission_line` matches `add_tl` likewise.
- `line.cell(205).left`, `.right`, `.Lj`, `.Cj`, `.Cg` reference the expected
  generated objects — PRD section 33's handle test.
- Adding one capacitor to an internal node changes exactly one element and
  nothing else, verified by diffing the full `Element[]`.
- `set_value` on `line.cell(205).Cg` changes only that element's value.
- `remove` on `line.cell(205).Cj` drops exactly one element; a second `remove`
  raises.
- `cells=0` on a junction line raises; `cells=-1` raises.
- The rf-SQUID line matches `build_rf_squid_line` output.

---

## Phase 4: Profile objects

### Overview

Deterministic spatial profiles as constructor arguments (PRD section 18),
wrapping the existing mathematics rather than reimplementing it (PRD section 19).

### Changes required

#### 1. Profile objects

**File**: `src/twpa_solver/circuit/profiles.py` (new)

Frozen dataclasses `Constant`, `Linear`, `HalfSine`, `Hann`, `Sine`, `Cosine`,
`Power`, `Parabola`, `Tanh`, `Custom` — each with `start`, `stop`,
`domain="selection" | "per_row"`, an optional selection, and
`to_segment() -> builders.profiles.Segment`.

Shape mapping:

| Object | `Segment.shape` | Note |
| --- | --- | --- |
| `Constant` | `const` | |
| `Linear` | `linear` | |
| `HalfSine` | `custom`, `expression="sin(pi*t/2)"` | PRD section 19 |
| `Hann` | `half_cosine` | the fabrication `IDFHannModulation` |
| `Sine`, `Cosine`, `Power`, `Parabola`, `Tanh`, `Custom` | one to one | |

Docstrings state plainly that `HalfSine` and `Hann` are different functions and
that `Hann` is the fabrication team's `IDFHannModulation`
(`Prometheus/Packages/KITWPAs.py:338`).

#### 2. Builder normalization

**File**: `src/twpa_solver/circuit/blocks/jj_line.py`

`Lj`, `Cj`, `Cg` accept `float | Profile`. Scalars remain scalars (PRD section
18). Profiles are evaluated through `builders.profiles.evaluate_profile` at
build time against the line's own cell count, per row when
`domain="per_row"`.

### Success criteria

**Automated**: `pytest -q tests/test_circuit_profiles.py`

- Every profile object's evaluated array equals
  `evaluate_profile([its_segment], ...)` exactly, by `np.array_equal`.
- `Linear(start, stop)` matches today's linear engine on the
  `designs/ipm_2c_linear.yaml` parameters.
- `HalfSine` matches `sin(pi*t/2)`; `Hann` matches `(1-cos(pi*t))/2`; and the
  two are asserted **unequal** on the same inputs.
- Scalar `Lj=123.9e-12` produces a constant array identical to
  `Constant(123.9e-12)`.
- An invalid domain raises, naming the block path.
- A profile producing a non-positive value raises, preserving the existing
  `builders/profiles.py:192-194` behaviour.

---

## Phase 5: Directional coupler

### Overview

The first two-path builder, and the explicit-geometry path v3 needs.

### Changes required

#### 1. Coupler builder

**File**: `src/twpa_solver/circuit/blocks/coupler.py` (new)

```python
add_directional_coupler(signal, pump, *, coupling_db=None, frequency=None,
                        z0=50.0, geometry=None, mode="auto",
                        cell_length_um=10.0)
```

Resolves a `CouplerDiscrete` through the untouched `make_coupler_discrete` and
`optimize_cpw_coupler`, then emits cells by calling Phase-1 and Phase-3
builders — never `add_edge_coupled_directional_coupler` directly, so PRD
section 12 holds. Advances **both** paths.

#### 2. Explicit geometry

**File**: `src/twpa_solver/circuit/blocks/coupler.py`

`geometry=ExplicitCouplerGeometry(gaps_um, widths_um, length_um)` bypasses
optimization and feeds `CPWConformalCoupler` directly, then
`calculate_conformal_discrete_params` (`builders/ipm.py:480`). Validates
`len(gaps) == len(widths) + 1` and that `widths` has length 2 or 3. Emits the
achieved `coupling_db` into metadata so a design can be checked against its
target.

#### 3. Coupler handle

**File**: `src/twpa_solver/circuit/handles.py`

`CouplerHandle` with `signal_in`, `signal_out`, `pump_in`, `pump_out`,
`port(n)` for n in 1 through 4 matching the diagram at
`Prometheus/Packages/DirectionalCouplers.py:265-269`, `cell(i)`, and `geometry`.

### Success criteria

**Automated**: `pytest -q tests/test_circuit_coupler.py`

- Both `signal.end` and `pump.end` advance by `N_coupled` — PRD section 33's
  coupler test.
- The emitted `Element[]` is identical to `add_edge_coupled_directional_coupler`
  at the same `CouplerDiscrete`.
- Handle terminals resolve to the expected nodes; `port(1) == signal_in`,
  `port(2) == signal_out`, and `port(3)` / `port(4)` sit on the pump path.
- Explicit geometry `gaps=[5.5, 5, 5, 5.5]`, `widths=[9.186, 15, 9.186]` builds,
  and its reported `coupling_db` is finite and negative.
- Passing the same path as both arguments raises.
- Passing a `Path` from another `Circuit` raises.

---

## Phase 6: IPM v1 parity — the gate

### Overview

Rebuild today's 2c through the public API only and prove it identical. **No
later phase begins until this is green.**

### Changes required

#### 1. Architecture builder

**File**: `src/twpa_solver/circuit/architectures/ipm.py` (new)

```python
add_ipm_section(signal, pump, *, rows, array_length, Lj, Cj, Cg,
                short_tl_cells, long_tl_cells, coupler_section_cells,
                coupler=..., name=...)
```

Calls `add_jj_line`, `add_transmission_line` and `add_directional_coupler` only
(PRD section 12). Returns a handle exposing `row[i].array[j].cell[k]` per PRD
section 16.

#### 2. Python design

**File**: `designs/python/ipm_2c.py` (new)

The current 2c device expressed entirely through the public API, in the PRD
section 26 style: `c.path("signal")`, `c.path("pump")`, `add_port`, coupler,
section, coupler, tail, ports, `compile()`. Reproduces the actual existing
topology, not the PRD's illustrative block counts — PRD section 26's closing
note is explicit about this.

#### 3. Parity test

**File**: `tests/test_circuit_ipm_parity.py` (new)

Extends the existing triangulation (`tests/test_design_compiler.py:13-36`) to a
fourth leg. Compares `designs/python/ipm_2c.py`, compiled with
`node_numbering="legacy"`, against **both** `make_ipm(params, coupler)` and
`designs/ipm_2c_fixed/*.npz`, field for field over: element count, element
order, `name`, `n1`, `n2`, `value`, `kind`, `role`, `cell_index`, ports, `C`,
`G`, `K`, `Bphi`, `Ic`, `Lj`. Exact equality on names, kinds, roles and
integers; `nnz == 0` on the sparse difference for matrices, as the existing test
does.

A second assertion covers `node_numbering="creation"`: the element sequence is
unchanged, and the matrices are related to the legacy ones by the permutation
`node_map` describes — verified by explicitly permuting and re-comparing, which
also proves the map is a bijection.

### Success criteria

**Automated**:

```
pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_oop \
  tests/test_circuit_ipm_parity.py tests/test_design_compiler.py \
  tests/test_ipm_2c_baseline.py tests/test_variant_design.py
```

**Manual**: `export_netlist` of the Python design diffs clean against
`designs/ipm_2c_fixed/elements.csv` on name, endpoints and value.

**Mutation verification**, per the standing project requirement that every gate
be shown failing first: perturb one `Cg` in `designs/python/ipm_2c.py` and
confirm the parity test fails; perturb one element's order and confirm it fails;
revert both.

---

## Phase 7: YAML becomes an adapter

### Overview

Eliminate the second circuit-generation implementation (PRD section 31). One
source of truth.

### Changes required

#### 1. Rewrite the compiler body

**File**: `src/twpa_solver/design/compiler.py`

`compile_design` builds a `Circuit` and calls its public builders instead of
populating a `BuildContext`. `_expand_composites` (`:208-305`) becomes a mapping
from YAML block types to `Circuit` method calls. `_apply_inline_action` (`:449`)
routes to `c.set_value` and `c.remove`. `_technology`, `validate_design`,
`resolve_parameters` and `apply_patches` are unchanged. Compiles with
`node_numbering="legacy"` so every stored artifact is preserved.

#### 2. Return type shim

**File**: `src/twpa_solver/design/model.py`

`CompiledDesign` is constructed from a `CompiledCircuit`. Its public surface
(`elements`, `cursors`, `named_nodes`, `named_elements`, `ports`, `blocks`,
`metadata`, `resolve_node`, `resolve_element`) is preserved exactly, so
`design/__main__.py`, `scripts/` and `tests/` need no changes.

#### 3. Retire the dictionary block layer

**Files**: `src/twpa_solver/builders/blocks.py`,
`src/twpa_solver/builders/registry.py`

`BLOCK_BUILDERS` entries become thin forwarders to `Circuit` methods, or are
deleted where the compiler no longer dispatches through them. `BuildContext` is
removed once nothing imports it.

### Success criteria

**Automated**: full suite —

```
pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_oop_yaml --run-slow
```

plus a new `tests/test_yaml_adapter_regression.py`: for each of the 11 YAML
designs, `compile_design` output is unchanged versus a snapshot captured at
Phase 0 — element list, and `C/G/K/Bphi` checksums where matrices are cheap
enough to build.

**Manual**:
`python -m twpa_solver.design --design designs/ipm_2c.yaml --outdir <tmp> --write-matrices`
produces `elements.csv` byte-identical to `designs/ipm_2c_fixed/elements.csv`.

---

## Phase 8: v2 and v3 architectures

### Overview

Add the two newer fabrication devices alongside the retained v1.

### Changes required

#### 1. Lumped junction array

**File**: `src/twpa_solver/circuit/primitives.py`

`add_jj_array(n1, n2, *, Lj, Cj, count)` emitting a single junction with
`Lj_eff = count * Lj`, `Cj_eff = Cj / count`, `Ic` unchanged. The docstring
states the approximation and its validity limit — each junction well below
`Ic` — and that it adds no nodes.

#### 2. Per-coupler leakage

**File**: `src/twpa_solver/circuit/architectures/ipm.py`

`coupler_leakage_db(coupling_db, coupler_number)` reproducing
`getCouplerLeakage` (`Prometheus/Packages/MicrowaveFunctions.py:901`). Applied
when a design passes a sequence of couplings or opts in, matching fabrication
v1's use at `JTWPAs.py:931`, and defaulting off, matching v3's commented-out
state.

#### 3. v2 design

**File**: `designs/python/ipm_v2.py` (new)

v1 topology with `add_jj_array(count=3)` cells and `CellNums = [20, 15, 3, 2]`:
2 sections x 3 rows x 15 arrays x 20 cells.

#### 4. v3 design

**File**: `designs/python/ipm_v3.py` (new)

`CellNums = [36, 18, 6, 1]`: 6 rows, 36 x 18 cells per row, a coupler every two
rows for 3 couplers, signal chaining row to row, pump chaining coupler to
coupler. The coupler uses
`ExplicitCouplerGeometry(gaps_um=[5.5, 5, 5, 5.5], widths_um=[9.186, 15, 9.186], length_um=...)`
with the length taken from the fabrication `getCouplerDimentions` target of
-25 dB at 10 GHz. Pump interconnect lengths are free parameters with documented
defaults.

#### 5. Architecture-level builder for the v3 shape

**File**: `src/twpa_solver/circuit/architectures/ipm.py`

Add `add_ipm_v3_section(...)` only if the v3 row-and-coupler interleave turns
out not to be expressible by composing `add_jj_line` and
`add_directional_coupler` in the design script. Prefer plain composition in
`ipm_v3.py`; PRD section 11 Level 4 says not to turn every complete design into
a special function.

### Success criteria

**Automated**: `pytest -q tests/test_circuit_ipm_variants.py`

- v2 compiles; node count equals the v1-shaped equivalent, proving the lumped
  array adds none; junction count is 20 x 15 x 3 x 2 = 1800; each junction's
  `Lj` is three times and `Cj` one third of the per-junction values.
- v3 compiles; 6 junction lines, 3 couplers, cell count 36 x 18 x 6 = 3888.
- v3's coupler reports `model == "three_line"` and a finite `coupling_db`.
- All three designs pass `export_netlist` and `build_matrices` without error;
  `C/G/K/Bphi` shapes are self-consistent and `Bphi.shape[1] == Ic.size`.
- A smoke linear-scattering solve on v3 returns finite S-parameters, reusing
  the `tests/test_linear_scattering_smoke.py` machinery.

**Manual**: v3's achieved `coupling_db` compared against the -25 dB target, with
the deviation recorded in the final report rather than silently accepted.

---

## Phase 9: Documentation and final report

### Changes required

#### 1. Design format document

**File**: `docs/design_format.md`

Reframed: the Python `Circuit` API is authoritative; YAML is documented as a
supported adapter over it.

#### 2. API guide

**File**: `docs/development/circuit_api.md` (new)

Worked examples at each abstraction level, the profile catalogue with the
`HalfSine` versus `Hann` distinction spelled out, the targeted-edit recipe, and
the `node_numbering` policy with its rationale.

#### 3. Mapping document finalization

**File**: `docs/development/GDS_API_MAPPING.md`

Reconciled against the names actually shipped; deviations section closed out.

#### 4. Agent notes

**File**: `CLAUDE.md`

New section recording the authoring API, the `node_numbering` default and why
`"legacy"` exists, the v2 lumped-array approximation, and that `designs/python/`
is the live design source.

#### 5. Final report

**File**: `docs/development/circuit_oop_migration_report.md` (new)

PRD section 39 contents: files added and modified, legacy code retained versus
deprecated, tests run, parity results with actual numbers, remaining migration
work, intentional differences from the GDS API.

### Success criteria

**Automated**:
`pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_oop_final --run-slow`

**Manual**: the PRD section 37 acceptance checklist walked item by item, each
ticked with the test or file that discharges it.

---

## Testing strategy

### Project maturity level

**Active development.** The existing suite is dense —
`tests/test_component_profiles.py` alone has 29 tests, and `CLAUDE.md` records
95 profile and scatter tests "each verified by mutation".

### Unit tests

- Every public `Circuit` method: endpoints, kind, value, role, hierarchy path,
  resulting `Element`, matrix stamp.
- Edge cases: zero and negative cell counts, duplicate names, duplicate port
  numbers, cross-circuit nodes, double removal, non-positive profile values, a
  coupler given one path, `len(gaps) != len(widths) + 1`.
- Determinism: every compile-affecting path asserted stable across repeat runs.
- Coverage target: 80 percent on `src/twpa_solver/circuit/`.

### Integration tests

- The Phase 6 parity gate is the primary integration test.
- Phase 7's 11-design YAML regression.
- Phase 8's linear-scattering smoke on v3.

### Mutation verification

Every parity and determinism gate must be shown failing first under a
deliberate perturbation, then restored. The project does not accept a gate that
has never failed.

### Manual tests

- `elements.csv` byte-diff against `designs/ipm_2c_fixed/`.
- Read `designs/python/ipm_v3.py` beside
  `Prometheus/Packages/JTWPAs.py:1497-1621` and confirm a fabrication engineer
  recognizes the same construction sequence.

---

## Rollback plan

- **Phases 1 through 6 are purely additive**: a new
  `src/twpa_solver/circuit/` package, new tests, new `designs/python/`. Nothing
  existing is modified. Rollback is deleting the package; the solver is
  untouched throughout.
- **Phase 7 is the only invasive commit.** Keep it as a single focused commit so
  `git revert` restores the dictionary-driven compiler wholesale. The Phase 0
  baseline snapshot and the 11-design regression detect any drift immediately.
  `builders/blocks.py` is not deleted until that regression is green, and its
  deletion is a separate commit.
- **Phases 8 and 9 are additive again.**
- Per `CLAUDE.md`, all work stays on `dev`. Nothing is merged wholesale to
  `main`; promotion is by curated `git cherry-pick` after Phase 9.
