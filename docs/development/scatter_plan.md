Full plan:

---
Implementation Plan: Per-Component Profiles and Multi-Component Scatter in the IPM Builder

Goal

python -m twpa_solver.builders.ipm can set Lj per junction from arbitrary shaped profiles over arbitrary index blocks, derive Cj to hold plasma frequency constant, profile Cg the same way, and apply independent percentage scatter to Lj, Cj, and Cg relative to each component's own nominal value.

Current State Analysis

┌─────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────┐
│                          Fact                           │                             Location                              │                          Consequence                          │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│                                                         │                                                                   │ Plasma frequency is already invariant under mod_factor.       │
│ add_jj computes Lj/mf, Cj*mf                            │ src/twpa_solver/builders/ipm.py:508-519                           │ Cj-from-Lj needs no new physics — set mod_array = Lj_base /   │
│                                                         │                                                                   │ Lj_profile.                                                   │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ make_ipm threads a contiguous JJ counter curr_mod_idx   │ ipm.py:712-790                                                    │ A global cell index 0..N-1 already exists. 2c: 6 rows × 418 = │
│                                                         │                                                                   │  2508, matching designs/ipm_2c_fixed exactly.                 │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Cg is a scalar, halved at row boundaries                │ ipm.py:721-745, 792                                               │ Per-row cap count is array_length+1 (2c: 2514), not           │
│                                                         │                                                                   │ array_length. Needs an explicit index mapping.                │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Lj is not stamped into K                                │ ipm.py:882-883, 951-962                                           │ Why scattered.py can rewrite only Ic/Lj in the npz. Cj and Cg │
│                                                         │                                                                   │  both stamp into C, so that shortcut cannot extend to them.   │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Stored matrices are the exact stamp of the CSV          │ verified: C/G/K/Bphi maxdiff 0.0 on designs/ipm_2c_fixed          │ Netlist is a faithful intermediate. Rebuild-and-restamp is    │
│                                                         │                                                                   │ exact, not approximate.                                       │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ designs/ipm_2c_fixed rebuilds bit-identically from      │ verified: 16312/16312 elements identical in name, nodes, kind,    │ The live design is reachable from parameters. --coupler-mode  │
│ ipm_summary.json["params"] with --coupler-mode cached   │ value                                                             │ ideal (16192) and optimize do not reproduce it — cached is    │
│                                                         │                                                                   │ mandatory.                                                    │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Cg is not structurally identifiable                     │ measured: "cap to ground on a JJ node" returns 2522 on 2c; 2514   │ Elements must carry explicit role tags. Name regex is not     │
│                                                         │ are Cg, 6 are TL Cl, 2 are coupler end caps                       │ acceptable.                                                   │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ No design in designs/ has scatter enabled               │ all 8 report lj_scatter_enabled: false                            │ No published artifact depends on the current RNG stream.      │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Only 4 .py files read ipm_elements.csv                  │ builders/ipm.py, builders/scattered.py, experiments/exp07_*,      │ The two experiments/ files are frozen provenance copies and   │
│                                                         │ experiments/build_scattered_ipm_design.py                         │ must not be edited.                                           │
├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ make_ipm is called with 2 positional args everywhere    │ scripts/tune_lj_to_themis.py:76,                                  │ New parameters must be keyword-only with defaults.            │
│

Decisions Fixed by the User

- Cj scatter is an independent draw with its own sigma and RNG stream. Nominal Cj is still derived from Lj to hold plasma frequency; scatter deliberately breaks that per junction.
- domain is a per-segment flag: "selection" (shape spans the whole selected range) or "per_row" (shape restarts in each JTL row).
- Scope is Lj, Cj, Cg only. TL and coupler components are untouched.
- Spec surface is JSON file + CLI shorthand, both parsing to the same dataclass.

What We're NOT Doing

- No profile or scatter on Ll, Cl, coupler L_cell/Cc_cell/C_gnd_cell/K_ind, or terminations. The profile module is written generically so this is a later wiring job, not a redesign.
- No --cj-profile. Nominal Cj is derived, never user-specified — that is what "plasma frequency must be constant" means.
- No wiring into builders/jc_doc.py or builders/le_gal_2025.py.
- No changes to any solver module, and no re-running of exp20–exp32 or any campaign.
- No edits to experiments/exp07_python_ipm_design_builder.py or experiments/build_scattered_ipm_design.py — frozen provenance.
- No regeneration of anything under designs/.

Prerequisites

- [ ] Confirm designs/ipm_3c_fixed and the six ipm_7c_* designs also rebuild element-identically from their stored params with --coupler-mode cached. Record which ones do. Any that do not are excluded from the Phase 5 rebuild path and must keep the legacy Lj-only route.
- [ ] Baseline: python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_profiles tests/ green before any edit.

Changes Required

1. New module

File: src/twpa_solver/builders/profiles.py

@dataclass(frozen=True)
class Selection:
    rows: tuple[int, int] | None = None       # inclusive row range
    index: tuple[int, int] | None = None      # inclusive global cell index range
    fraction: tuple[float, float] | None = None  # [0,1] closed range

@dataclass(frozen=True)
class Segment:
    shape: str
    start: float
    end: float | None = None
    select: Selection = Selection()
    domain: str = "selection"                 # "selection" | "per_row"
    params: Mapping[str, float] = field(default_fact

┌─────────────┬───────────────────────────────────────┬─────────────────────────────┬─────────────────────────────────────────────────┐
│    shape    │                 s(t)                  │           params            │                endpoint behavior                │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ const       │ —                                     │ —                           │ every cell = start; end must be absent or equal │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ linear      │ t                                     │ —                           │ anchored                                        │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ power       │ t**p                                  │ exponent p > 0              │ anchored                                        │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ parabola    │ ((t-v)**2 - v**2) / ((1-v)**2 - v**2) │ vertex v in [0,1], v != 0.5 │ anchored; v = 0.5 raises (zero denominator)     │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ half_cosine │ (1 - cos(pi*t))/2                     │ —                           │ anchored, smooth                                │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ tanh        │ normalized tanh(k(2t-1))              │ sharpness k > 0             │ anchored                                        │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ sine        │ 0.5 + 0.5*sin(2*pi*periods*t + phase) │ periods, phase              │ envelope: min = start, max = end                │
├─────────────┼───────────────────────────────────────┼─────────────────────────────┼─────────────────────────────────────────────────┤
│ cosine      │ 0.5 + 0.5*cos(2*pi*periods*t + phaseof raw text, no attribute access, no imports, no builtins. This satisfies the global rule against passing arbitrary input to eval/exec while still giving the arbitrary functions the user asked for.

Public API:
def evaluate_profile(
    segments: Sequence[Segment], *, n_cells: int, cells_per_row: int, base_value: float
) -> np.ndarray
def parse_profile_json(path: Path) -> dict[str, list[Segment]]   # keys: "Lj", "Cg"
def parse_profile_shorthand(text: str) -> Segment

Semantics: array starts filled with base_value; segments apply in order; overlap is last-writer-wins (so all:linear followed by rows=0:const is a legal override). A per-cell coverage map is returned in metadata. Any resulting value <= 0 raises ValueError naming the offending cell index.

Shorthand grammar (--lj-profile):
<select>:<shape>:<start>[-><end>][:k=v,k=v]
select := all | rows=A[-B] | index=A[-B] | frac=A[-B]
values accept SI suffixes: p (1e-12), f (1e-15), n, u
example: rows=0-2:sine:120p->140p:periods=2,phase=0

Success Criteria

Automated: pytest tests/test_component_profiles.py, mypy src/twpa_solver/builders/profiles.py, ruff check
- every anchored shape returns exactly start at index 0 and exactly end at index -1 (np.isclose at rtol=0, atol=0 for linear/const; rtol=1e-12 otherwise)
- sine/cosine with integer periods: min == start, max == end to rtol=1e-12
- N == 1 segment returns [start] without dividing by zero
- rows=(0,0), index=(0,417), frac=(0.0, 418/2508) resolve to the identical index set on a 6×418 layout
- domain="per_row" vs "selection" produce identical arrays for const, different for linear
- last-writer-wins on overlapping segments
- a segment producing a non-positive value raises ValueError
- custom evaluates 1 - (1-t)**2; __import__("os"), t.__class__, and open(...) are each rejected with ValueError
- JSON and shorthand paths produce equal Segment objects for the same spec

Manual: none — this phase is pure.

---
Phase 2: Scatter Engine

Overview

Multiplicative scatter relative to each component's local nominal, with independent, reproducible streams per component.

Changes Required

1. New module

File: src/twpa_solver/builders/scatter.py

@dataclass(frozen=True)
class ScatterSpec:
    sigma: float = 0.0                # fractional, relative to local nominal
    distribution: str = "normal"      # "normal" | "uniform"
    clip_min: float = 0.5
    clip_max: float = 1.5

def draw_factors(spec: ScatterSpec, n: int, rng: np.random.Generator) -> np.ndarray
def component_rng(master_seed: int, component: str) -> np.random.Generator
def apply_scatter(nominal: np.ndarray, spec: ScatterSpec, rng) -> tuple[np.ndarray, dict]

Stream assignment (contract — never reorder, only append):
- Lj -> np.random.default_rng(seed)
- Cj -> np.random.default_rng(np.random.SeedSequence([seed, 1]))
- Cg -> np.random.default_rng(np.random.SeedSequence([seed, 2]))

Lj deliberately uses the bare default_rng(seed) so it is bit-identical to today's apply_lj_scatter at the same seed and count. Cj/Cg draw from distinct SeedSequence entropy, so changing --cg-scatter-sigma cannot perturb the Lj realization.

Because nominal is the post-profile per-cell value, sigma is automatically "percentage of that cell's own nominal", which is the stated requirement.

Returned metadata per component: sigma, distribution, seed, stream, realized factor_{min,max,mean,std}, resulting value_{min,max}, clip_hits (count of clipped draws), and a blake2b digest of the factor array for cross-run reproducibility checks.

Success Criteria

Automated: pytest tests/test_component_scatter.py, mypy, ruff
- sigma == 0 returns factors exactly 1.0 and an array is-equal in value to the nominal
- same seed reproduces bitwise; different seed differs
- legacy parity: draw_factors for Lj at seed s, size n equals np.random.default_rng(s).normal(1.0, sigma, n) clipped — the same array today's ipm.apply_lj_scatter produces
- varying cg sigma leaves the Lj factor digest uncha
- relative semantics: with a linear 100p -> 200p Lj profile and sigma=0.02, std(scattered/nominal) ~= 0.02 while std(scattered) is far larger — proving sigma is local, not global

---
Phase 3: Role Tagging in ipm.py (No Behavior Change)

Overview

Give every element an explicit role and cell index so Cg can be addressed without name regex. Zero numerical change — this phase is gated on byte identity against the live 2c design.

Changes Required

1. Element gains tags

File: src/twpa_solver/builders/ipm.py:81-88
@dataclass
class Element:
    name: str
    n1: Any
    n2: Any
    value: float | int | str

2. CSV schema

File: ipm.py:1060-1064 (write_outputs)
Header becomes ["idx","name","node1","node2","value","kind","role","cell_index"]. Appending columns keeps csv.DictReader consumers working.

3. Reader tolerance

File: src/twpa_solver/builders/scattered.py:120
The hardcoded header = ["idx","name","node1","node2","value","kind"] must become "read the header from the file and write it back", or _scatter_elements_csv will silently drop the new columns.

Success Criteria

Automated: pytest tests/test_ipm_role_tags.py
- default build with IPMParams(**designs/ipm_2c_fixed params) and --coupler-mode cached is element-wise identical in (name, n1, n2, kind, value) to designs/ipm_2c_fixed/ipm_elements.csv — all 16312 rows
- build_matrices on that circuit gives C/G/K/Bphi with abs(A - stored).max() == 0.0
- role counts on 2c: jj_lj == 2508, jj_cj == 2508, jtl_cg == 2514, and jtl_cg count == num_rows * (array_length + 1)
- cell_index on jj_lj is exactly range(2508) in element order (this is what makes the Lj legacy stream parity of Phase 2 valid)
- every element has a non-empty role
Overview

Turn the arrays into an actual circuit and expose the knobs.

Changes Required

1. Component plan

File: src/twpa_solver/builders/ipm.py (new section)
@dataclass(frozen=True)
class ComponentPlan:
    lj: np.ndarray          # (n_cells,)
    cj: np.ndarray          # (n_cells,)
    cg: np.ndarray          # (n_cells,)
    metadata: dict[str, Any]

def build_component_plan(
    params: IPMParams, *,
    lj_segments: Sequence[Segment] = (),
    cg_segments: Sequence[Segment] = (),
    lj_scatter: ScatterSpec = ScatterSpec(),
    cj_scatter: ScatterSpec = ScatterSpec(),
    cg_scatter: ScatterSpec = ScatterSpec(),
    seed: int = 1,
) -> ComponentPlan

2. make_ipm accepts arrays

File: ipm.py:682-808
def make_ipm(params, coupler, mod_array=None, *, plan: ComponentPlan | None = None)
mod_array keeps its positional slot for the existing 2-arg callers (scripts/tune_lj_to_themis.py:76, scripts/periodicity_campaign.py:47). Passing both mod_array and plan raises.

Cg index mapping (the only non-obvious mapping):
- cell c of row r owns the ground cap at its left node, value cg[r*array_length + c], halved when c == 0
- the row's trailing cap (C{j_top}_{ground}_JTL_end, ipm.py:745/792) is cg[r*array_length + array_length - 1] / 2

For constant cg this reduces to today's Cg/2 + 417*Cg + Cg/2, which is exactly what the Phase 3 gate already pins.

Lj/Cj go through the existing mod_factor machinery: add_jj is called with the plan's explicit Lj_i, Cj_i rather than Lj/mf, Cj*mf — the mod_array path stays intact for back-compat but the plan path bypasses it, since independent Cj scatter can no longer be expressed as a single factor.

3. Artifacts

File: ipm.py:1049-1133 (write_outputs)
- ipm_arrays.npz gains Cj, Cg, cell_index, Lj_nominal, Cj_nominal, Cg_nominal (nominals let a consumer recover the scatter factors). load_circuit reads by name and ignores extras, so this is additive.
- ipm_summary.json gains a component_plan block: the full serialized segment specs, scatter specs, seed, stream digests, and realized statistics — enough to regenerate the design exactly.

4. CLI

File: ipm.py:1222-1272
- --profile-json PATH — {"Lj": [...], "Cg": [...]}
- --lj-profile STR / --cg-profile STR — repeatable (action="append"), appended after any JSON segments so CLI overrides the file
- --scatter-seed INT (master, default 1); --lj-scatter-seed kept as a deprecated alias that sets it, warning if both are given with different values
- --lj-scatter-sigma (exists), --cj-scatter-sigma, --cg-scatter-sigma
- --scatter-distribution {normal,uniform}, and --{l
- legacy scatter regression: --lj-scatter-sigma 0.01 --lj-scatter-seed 1 produces the same Lj array as the pre-change apply_lj_scatter on the same circuit
- constant-plasma: for an arbitrary Lj profile with no scatter, max|Lj_i*Cj_i - Lj_base*Cj_base| / (Lj_base*Cj_base) < 1e-15 over all 2508 cells
- half/half: rows=0-2:const:120p + rows=3-5:const:150p gives exactly 1254 cells at each value, and the boundary lands at index 1254
- "first line of junctions": rows=0:const:X changes exactly cells 0..417 and leaves 418..2507 at nominal
- Cg profile with a constant reproduces the stored 2c cap values including both halves; cap count stays 2514
- scatter changes C values but not its sparsity pattern ((C_new != 0) == (C_old != 0) structurally)
- Cj scatter with sigma > 0 makes Lj_i*Cj_i vary — the deliberate consequence of the independent-draw choice — asserted explicitly so nobody later "fixes" it
- make_ipm(params, coupler) two-arg call still works (import-and-call test mirroring scripts/tune_lj_to_themis.py)
- passing both mod_array and plan raises

Manual:
python -m twpa_solver.builders.ipm --outdir D:\tmp\prof_2c --write-matrices `
  --coupler-mode cached --lj-profile "rows=0-2:const:123.9p" `
  --lj-profile "rows=3-5:linear:123.9p->150p" --cg-profile "all:half_cosine:66f->72f" `
  --lj-scatter-sigma 0.01 --cj-scatter-sigma 0.005 --cg-scatter-sigma 0.02 --scatter-seed 7
python workflows\build_design_and_passive.py --design-dir D:\tmp\prof_2c
Confirm the passive S21 plot is continuous and shows the taper's dispersion shift — a discontinuity at a row boundary means the Cg boundary-halving mapping is wrong.

---
Phase 5: Variant Builder from Existing Designs, and Docs

Overview

Let an existing designs/* directory be re-emitted w
1. read source_dir/ipm_summary.json["params"] -> IPMParams
2. rebuild with --coupler-mode cached
3. hard gate: assert the rebuilt netlist is element-identical to source_dir/ipm_elements.csv; if not, raise naming the first differing element. Never silently proceed — a mismatch means the stored params do not describe the artifact.
4. apply the plan, restamp, write a full artifact set

scatter_existing_design keeps its exact current behavior and signature so scripts/build_scattered_design.py and workflows/ are unaffected.

2. Docs

Files: CLAUDE.md (new "Component profiles and scatter" section), docs/ note covering the shape catalogue, the sine/cosine envelope convention, the Cg boundary-halving rule, the RNG stream contract, and the fact that Cj scatter intentionally breaks constant plasma frequency.

Success Criteria

Automated: pytest tests/test_variant_design.py
- zero profile + zero scatter through build_variant_design reproduces designs/ipm_2c_fixed matrices with maxdiff 0.0
- the identity gate fires (raises) when params are perturbed before rebuild
- 3c and each 7c design that passed the prerequisite check also round-trip at zero settings
- scatter_existing_design output is unchanged from before this work at sigma=0.01, seed=1

Manual: build a scattered 2c variant, run workflows/build_design_and_passive.py, confirm S21 degrades with increasing sigma.

---
Testing Strategy

Project Maturity Level

Established Production — this repo gates physics claims on measured numbers and its notes explicitly distrust unverified agent claims.

Unit Tests

- tests/test_component_profiles.py — shape kernels, endpoint anchoring, envelope convention, selectors, domain flag, overlap, positivity, AST allowlist rejections, JSON/shorthand parity
- tests/test_component_scatter.py — determinism, stream independence, legacy Lj parity, clipping, relative-to-local-nominal semantics
- tests/test_ipm_role_tags.py — role counts, cell-index ordering, byte-identity against designs/ipm_2c_fixed
- tests/test_ipm_component_plan.py — constant plasma frequency, Cg mapping, block profiles, sparsity invariance, back-compat call signatures
- tests/test_variant_design.py — rebuild identity gate
- Coverage target: 80% on the two new modules; the identity gates matter more than the percentage.

Edge Cases

Single-cell segment; segment covering one row of a one-row device; sigma large enough to hit both clips; vertex == 0.5 on parabola; periods == 0 on sine; profile driving a value to zero or negative; unknown shape name; rows out of range; frac outside [0,1]; both --profile-json and --lj-profile supplied.

Integration

- Full suite with the project's documented slow invocation:
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_profiles_slow --run-slow
- workflows/build_design_and_passive.py on one profiled design, checked visually for continuity at row boundaries.

Mutation Checks (required before reporting done)

Per this repo's standing rule that every gate be shown failing first, the implementer must demonstrate each of these produces a red test, then revert:
1. flip the Cg boundary halving from cg[first]/2 to cg[first]
2. change cj_nom = Cj*(Lj_base/lj_nom) to cj_nom = Cj (constant)
3. give Cj and Cg the same RNG stream
4. change t_i = i/(N-1) to i/N (endpoint anchoring breaks)
5. drop the role tag from jtl_cg and fall back to the structural filter (must fail: it over-counts by 8 on 2c)

A gate that cannot be shown failing is not a gate.

---
Rollback Plan

Work on a branch off main (component-profiles). Phases 1–2 add only new files — deletable with no trace. Phases 3–5 touch ipm.py and scattered.py; all new parameters are keyword-only with defaults that reproduce current behavior, so git revert of any single phase commit leaves a working tree. Nothing under designs/ or outputs/ is written at any point; all manual verification targets D:\tmp\. The Phase 3 byte-identity gate is the tripwire: if it ever goes red, the phase is wrong and must be reverted rather than patched.