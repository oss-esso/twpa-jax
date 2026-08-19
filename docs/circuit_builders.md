# Circuit construction and builders

Circuit authoring is split into three layers. A design author normally works
at the assembly layer; the lower layers provide reusable electrical blocks and
shared defaults.

1. **Block library** — `src/twpa_solver/circuit/blocks/`, `cells/`, and
   `architectures/`. Add a new reusable block here, declare its
   `TECHNOLOGY_DEFAULTS`, and compose the existing public builders. The
   `parallel_lc.py` block is the worked example: it declares only `L -> Lk`
   and requires no central dispatch edit. It is demonstration code for the
   extensibility contract, not a required production device block.
2. **Component defaults** — `designs/technology/*.yaml`. The `components:`
   section contains electrical component values and physical component
   dimensions; `architecture:` contains counts and lengths in cells. These
   technology files are shared by Python and YAML designs. Legacy flat
   `parameters:` technology files remain accepted for compatibility.
3. **Assembly** — `designs/python/*.py` or `designs/*.yaml`. The author chooses
   the required granularity, from a high-level architecture down to cells and
   primitive elements. YAML is adapted through the same `Circuit` builders.

When a builder argument is omitted, resolution is deterministic:

```text
explicit call argument
    -> design-level override
    -> technology components:
    -> technology architecture:
    -> builder default
    -> error naming the parameter and path
```

`src/twpa_solver/builders/` is the layer beneath this authoring interface. It
contains the legacy `Element` IR, matrix assembly, and coupler physics that
`Circuit` composes; it is not the primary design-authoring surface.

The legacy builder layer remains documented below because it is still used by
solver-facing compatibility workflows. There are four builders and two
shared support modules.

| Module | Produces | Devices |
| --- | --- | --- |
| `ipm.py` | on-disk artifact directory | IPM / 2c / 3c / 7c coupled-line JTWPA |
| `jc_doc.py` | on-disk artifact directory | JPA, DPJPA, FXJPA, JTWPA, FQJTWPA, FQJTWPA_diss, FXJTWPA |
| `le_gal_2025.py` | in-memory `CircuitMatrices` | effective-SNAIL line (Le Gal 2025 benchmark) |
| `scattered.py` | on-disk artifact directory | copy of an existing 2c/3c design with Lj scatter |
| `profiles.py` | per-cell value arrays | support module, no circuit knowledge |
| `scatter.py` | per-cell multiplicative factors | support module, no circuit knowledge |

## The matrix convention every builder targets

```
C xddot + G xdot + K x + Bphi i_J(Bphi.T x) = i_src
```

`x` is the vector of non-ground node fluxes. The three square matrices are node
admittance stamps; `Bphi` is the node-to-branch incidence for the nonlinear
branches only.

The critical asymmetry: **Josephson inductances are not stamped into `K`.** They
become nonlinear `Bphi` branches carrying `Ic = phi0_reduced / Lj`, evaluated
through the branch law at solve time. Their `Cj` shunt capacitance *is* stamped
into `C`. Ordinary inductors stamp `1/L` into `K`. A mutually coupled pair
stamps `B Lpair^-1 B.T` instead of two independent `1/L` terms.

This is why the old `scattered.py` can rewrite `Lj` by editing one array in the
npz, but the same trick cannot work for `Cj` or `Cg` — those change `C`, so they
must be applied at netlist level before the stamp.

## Artifact layout

`ipm.py`, `jc_doc.py`, and `scattered.py` all write a directory that
`twpa_solver.core.load_circuit` can read:

| File | Contents |
| --- | --- |
| `C.npz`, `G.npz`, `K.npz`, `Bphi.npz` | scipy sparse CSR matrices |
| `ipm_arrays.npz` | `nodes`, `Ic`, `Lj`, `phi0_reduced`, `port_numbers`, `port_indices` |
| `ipm_summary.json` | parameters, element counts, matrix shapes, metadata |
| `ipm_elements.csv` | the netlist (`ipm.py` and `scattered.py` only) |
| `ipm_ports.csv` | port table (`ipm.py` only) |

`load_circuit` reads arrays by name and ignores unknown keys, so builders may
add fields without breaking existing consumers.

```python
from twpa_solver.core import load_circuit

circuit = load_circuit("designs/ipm_2c_fixed")
circuit.node_count      # 6136
circuit.branch_count    # 2508
circuit.port_to_index   # {1: ..., 2: ..., 3: ..., 4: ...}
```

---

# 1. `ipm.py` — the IPM / 2c coupled-line builder

The production builder. Ports the Julia topology generator
(`Transmission_line_block.jl`, `directional_coupler_block.jl`, `CPW_Theory.jl`,
`IPM.jl`, `IPM_JTWPA.jl`) with no Julia dependency.

## Topology

A top signal line and a bottom pump rail, joined by edge-coupled directional
couplers. The signal line alternates JTL rows (Josephson junction arrays) with
plain transmission-line sections; every `jtl_rows_per_coupler` rows the two lines meet
in another coupler.

Ports: `1` signal in, `2` signal out, `3` pump rail reference, `4` pump source.
The pump counter-propagates relative to the signal. **Pump injection is port 4,
not port 1** — using port 1 for both leaves a promoted-pump residual of sqrt(2).

## Parameters

`IPMParams` is a dataclass; every field has a CLI override. Lengths are cell
counts unless the name carries a unit.

| Group | Fields | Defaults (2c) |
| --- | --- | --- |
| Array | `jtl_cells_per_array`, `jtl_row_count`, `jtl_rows_per_coupler` | 418, 6, 3 |
| Junctions | `Lj`, `Cj`, `Cg` | 123.9 pH, 145 fF, 66 fF |
| Line | `Cl_per_um`, `Ll_per_um`, `cell_length_um` | 0.1695 fF, 0.424 pH, 10 µm |
| Coupler | `coupling_dB`, `Z0`, `coupler_freq_hz` | −14 dB, 50 Ω, 8 GHz |
| Sections | `inter_array_cpw_cells`, `signal_inter_coupler_cpw_cells`, `pump_inter_coupler_cpw_cells`, signal/pump input/output CPW fields | 90, 900, 800, 0/50/0/0 |
| Terminations | `Rleft`, `Rright`, `Rm` | 50 Ω each |

`Cl` and `Ll` are properties, so they always track `cell_length_um`.

## Coupler modes

`--coupler-mode` selects how the directional coupler is derived. **The modes are
not interchangeable — a design built with one is not reproduced by another.**

| Mode | Behavior | Cost |
| --- | --- | --- |
| `auto` (design-file default) | Optimizes the requested coupling/frequency and selects two-line or centre-ground three-line CPW geometry | slow |
| `optimize` | L-BFGS-B search over (width, gap, gap-to-ground) to hit `coupling_dB` and `Z0` | slow |
| `ideal` | No geometry at all: even/odd impedances set directly from the target coupling, discretized to a quarter wave at `coupler_freq_hz` | fast |

`auto` is the recommended design-file mode. It always optimizes from the
YAML `coupling_dB`, `coupler_freq_hz`, and `Z0` values. Targets at or above
-20 dB use the two-line CPW model; weaker-coupling targets select the
centre-ground three-line model. The selected model and optimized geometry
are retained in the resolved design metadata. `optimize` and `ideal` remain
available for compatibility and controlled comparisons.

`ideal` still produces a *distributed* coupler (many short cells), so it stays
broadband, unlike a single lumped cell that only matches at one frequency.

## Per-cell profiles

Nominal `Lj` and `Cg` can vary cell by cell. All arrays share one index space:
the JTL cell index, `0 .. jtl_row_count * jtl_cells_per_array - 1` (2508 on 2c).

**There is no `Cj` profile.** Nominal `Cj` is derived from the `Lj` profile as
`Cj_i = Cj * (Lj / Lj_i)`, so the plasma frequency `1/sqrt(Lj_i Cj_i)` is
constant everywhere by construction (verified to <1e-14 relative spread).

### Segments

A profile is an ordered list of segments. Later segments overwrite earlier ones
on overlap; untouched cells keep the scalar parameter. A segment that drives a
value to zero or below raises and names the offending cell.

Selection is `all`, `rows=A[-B]`, `index=A[-B]`, or `frac=A[-B]` — closed ranges,
at most one selector per segment.

`domain` is `selection` (shape spans the whole selected range) or `per_row`
(shape restarts inside each JTL row). For a 6-row device, `per_row` gives six
identical copies.

Every shape is a normalized kernel `s(t)` on `t in [0,1]`; the value is
`start + (end - start) * s(t)`, with `t_i = i/(N-1)` over `N` selected cells.
Anchored shapes therefore hit `start` at the first cell and `end` at the last by
construction.

| shape | `s(t)` | params | endpoints |
| --- | --- | --- | --- |
| `const` | — | — | all cells `start`; `end` must be absent or equal |
| `linear` | `t` | — | anchored, exact |
| `power` | `t**p` | `exponent` / `p` > 0 | anchored |
| `parabola` | `((t-v)²-v²)/((1-v)²-v²)` | `vertex` / `v` in [0,1], ≠ 0.5, default 0 | anchored |
| `half_cosine` | `(1-cos(pi t))/2` | — | anchored, smooth |
| `tanh` | normalized `tanh(k(2t-1))` | `sharpness` / `k` > 0 | anchored |
| `sine` | `(1+sin(2 pi f t + phi))/2` | `periods`, `phase` | **envelope** |
| `cosine` | `(1+cos(2 pi f t + phi))/2` | `periods`, `phase` | **envelope** |
| `custom` | user expression in `t` | — | caller's responsibility |

`vertex=0` is the plain `t²` parabola and is the default. `vertex=0.5` puts the
turning point at the midpoint, cannot be normalized, and raises.

**`sine` and `cosine` use an envelope convention.** Periodic and
endpoint-anchored are mutually exclusive, so for these two `start`/`end` are the
min/max of the oscillation (mean `(start+end)/2`, amplitude `(end-start)/2`),
not the first and last values. `linspace` does not land exactly on the extrema,
so the sampled min/max approach `start`/`end` as the cell count grows.

### Custom expressions

Parsed with `ast` and checked against an allowlist before evaluation:
arithmetic operators, `t`, `pi`, `e`, and direct calls to `sin`, `cos`, `tan`,
`tanh`, `sinh`, `cosh`, `arcsin`, `arccos`, `arctan`, `exp`, `log`, `log10`,
`sqrt`, `abs`, `sign`, `floor`, `ceil`, `minimum`, `maximum`. Attribute access,
subscripting, comprehensions, keyword arguments, and any other name are
rejected. Evaluation runs with empty builtins.

### Shorthand grammar

```
<select>:<shape>:<start>[-><end>][:k=v,k=v]
```

`domain` and `expression` are reserved keys in the trailing field, so both are
reachable without a JSON file. Values accept `p`, `n`, `u`, `f` SI suffixes.

```powershell
--lj-profile "rows=0:const:150p"
--lj-profile "all:linear:120p->140p"
--lj-profile "rows=3-5:sine:120p->140p:periods=2,domain=per_row"
--cg-profile "all:custom:66f->72f:expression=sin(pi*t)"
```

### JSON form

`--profile-json` takes `{"Lj": [...], "Cg": [...]}`. Each entry is either a
shorthand string or an object. CLI segments are appended after JSON ones, so
they win on overlap.

```json
{
  "Lj": [
    {"shape": "const",  "start": 1.5e-10, "select": {"rows": [0, 2]}},
    {"shape": "linear", "start": 1.239e-10, "end": 1.4e-10,
     "select": {"rows": [3, 5]}, "domain": "per_row"}
  ],
  "Cg": ["all:half_cosine:66f->72f"]
}
```

### Cg index mapping

The circuit has one more ground cap per JTL row than it has cells. Cell `c` of
row `r` owns the cap at its left node with value `cg[r*array_length + c]`,
halved when `c == 0`; the row's trailing cap is `cg[last cell of row] / 2`. For a
constant profile this reduces to the legacy `Cg/2 + (array_length-1)*Cg + Cg/2`.

## Scatter

Dielectric loss is selected with `--tan-delta` and repeatable
`--tan-delta-role ROLE=VALUE`. The `jj_cj` role is always lossless; the
default applies to other capacitor roles. Use
`--cj-scatter-mode plasma_locked` to derive Cj from Lj scatter and preserve
the plasma frequency (independent Cj sigma must be zero in that mode).

Multiplicative, against each cell's **own nominal**, so sigma is a fraction of
that cell's value rather than of a global mean. A cell at twice the nominal
absorbs twice the absolute deviation.

`Lj`, `Cj`, and `Cg` each get an independent sigma and an independent RNG
stream. Streams are fixed and must never be reordered — only appended to:

| component | generator |
| --- | --- |
| `Lj` | `default_rng(seed)` |
| `Cj` | `default_rng(SeedSequence([seed, 1]))` |
| `Cg` | `default_rng(SeedSequence([seed, 2]))` |

`Lj` deliberately uses the bare `default_rng(seed)` so it is bit-identical to
the pre-profile `apply_lj_scatter` at the same seed, and changing
`--cg-scatter-sigma` cannot perturb the `Lj` realization.

Independent `Cj` scatter retains the historical behavior and intentionally
breaks constant plasma frequency. `plasma_locked` is the opt-in mode for
preserving it exactly.

Factors are drawn `normal` or `uniform` (the uniform width is scaled so its
standard deviation equals sigma) and clipped to `[clip_min, clip_max]`, default
`[0.5, 1.5]`, per component.

`--scatter-seed` is the master seed. `--lj-scatter-seed` is a deprecated alias;
passing both with different values raises.

## Extra artifacts

`ipm_arrays.npz` gains `Cj`, `Cg`, `cell_index`, `Lj_nominal`, `Cj_nominal`,
`Cg_nominal`, all on the cell index space.

`ipm_summary.json` gains a `component_plan` block with the seed, the serialized
segments, the scatter specs, and realized statistics including a `blake2b`
factor digest — enough to regenerate the design, without embedding the per-cell
arrays that already live in the npz. The legacy `lj_scatter_*` keys are still
emitted. A full 2c summary is about 5 kB.

`ipm_elements.csv` gains `role` and `cell_index` columns. The roles that matter
are `jj_lj`, `jj_cj`, and `jtl_cg`; the rest restate `kind`. Tagging is required
because **`Cg` is not structurally identifiable** — filtering for "capacitor to
ground on a Josephson node" returns 2522 elements on 2c, of which only 2514 are
`Cg` (six are TL `Cl`, two are coupler end caps).

## Examples

Stock 2c, matrices written:

```powershell
python -m twpa_solver.builders.ipm --outdir outputs/ipm_2c --write-matrices --coupler-mode auto
```

Half the junctions at one value, half tapered, with a shaped `Cg` and scatter on
all three components:

```powershell
python -m twpa_solver.builders.ipm --outdir outputs/ipm_2c_taper --write-matrices `
  --coupler-mode auto `
  --lj-profile "rows=0-2:const:150p" `
  --lj-profile "rows=3-5:linear:123.9p->140p" `
  --cg-profile "all:half_cosine:66f->72f" `
  --lj-scatter-sigma 0.01 --cj-scatter-sigma 0.005 --cg-scatter-sigma 0.02 `
  --scatter-seed 7
```

Programmatic:

```python
from twpa_solver.builders.ipm import (
    IPMParams, build_component_plan, make_coupler_discrete, make_ipm,
    build_matrices, write_outputs,
)
from twpa_solver.builders.profiles import parse_profile_shorthand as shorthand
from twpa_solver.builders.scatter import ScatterSpec

params = IPMParams(jtl_cells_per_array=418, jtl_row_count=6)
coupler = make_coupler_discrete(params, "auto")
plan = build_component_plan(
    params,
    lj_segments=[shorthand("rows=0:const:150p")],
    cg_segments=[shorthand("all:linear:66f->72f")],
    lj_scatter=ScatterSpec(0.01),
    cj_scatter=ScatterSpec(0.005),
    seed=7,
)
circuit, ends = make_ipm(params, coupler, plan=plan)
write_outputs("outputs/my_design", circuit, params, coupler, ends,
              build_matrices(circuit), plan=plan)
```

`make_ipm(params, coupler)` with no plan still works and reproduces the legacy
scalar design exactly. Passing both `mod_array` and `plan` raises.

## Variants of an existing design

`build_variant_design` re-emits a stored design with a profile and scatter
applied, rebuilding from the design's own `ipm_summary.json["params"]`.

```python
from twpa_solver.builders.ipm import build_variant_design
from twpa_solver.builders.profiles import parse_profile_shorthand as shorthand
from twpa_solver.builders.scatter import ScatterSpec

build_variant_design(
    "designs/ipm_2c_fixed", "outputs/ipm_2c_variant",
    lj_segments=[shorthand("all:linear:123.9p->140p")],
    lj_scatter=ScatterSpec(0.01), seed=3, overwrite=True,
)
```

`assert_source_topology` gates this. It runs on a **nominal** rebuild —
comparing the profiled netlist against the source would reject every real
variant, since changing component values is the point. The gate is exact
(names, nodes, kinds, and values) and raises rather than proceeding, because a
mismatch means the stored parameters do not describe the artifact.

The coupler mode is not recorded in the summary, so `coupler_mode="auto"`
(default) tries `auto`, `ideal`, `optimize` in turn and records the winner as
`source_coupler_mode`. All eight directories under `designs/` currently pass;
`ipm_2c_fixed` matches at 16312/16312 elements with `C/G/K/Bphi` maxdiff 0.0,
under `auto`. Passing an explicit wrong mode still raises.

---

# 2. `jc_doc.py` — the seven JosephsonCircuits.jl documentation designs

Builds Python matrix models of the seven JC documentation examples. Used as
regression fixtures and as the compression-campaign devices.

> **These are not physical references.** JC is another simulator with no
> reference of its own, and `jc_jtwpa` / `jc_fqjtwpa` are JC's own documentation
> designs, so gating against them is circular. Agreement with JC measures
> numerical drift between two codes — a useful regression check and nothing
> more. Do not cite a "JC-reference gate".

Measured node, junction, and port counts:

| Case | Nodes | Junctions | Ports | Notes |
| --- | ---: | ---: | --- | --- |
| `jc_jpa` | 2 | 1 | 1 | single-pump reflection amplifier, 4.75 GHz |
| `jc_dpjpa` | 2 | 1 | 1 | same circuit, two pumps; needs true 2-D lattice HB |
| `jc_fxjpa` | 5 | 2 | 1, 2 | flux-biased, needs a DC operating point, mutual `k=0.999` |
| `jc_jtwpa` | 2560 | 2047 | 1, 2 | 2048 cells, resonator every 4th cell (`pmrpitch`) |
| `jc_fqjtwpa` | 2250 | 1999 | 1, 2 | flux-qubit weighted by a Gaussian of width 745 cells |
| `jc_fqjtwpa_diss` | 2250 | 1999 | 1, 2 | same topology, complex capacitors `C/(1+i tan_delta)` |
| `jc_fxjtwpa` | 1502 | 1000 | 1–4 | 500 cells, DC flux bias, mutually coupled pump line |

`CircuitBuilder` is a small element-collector with `port`, `resistor`,
`capacitor`, `linear_inductor`, `josephson_inductor`, and `mutual` methods, then
`assemble()` / `write()`. It infers a complex dtype when any element value is
complex, so the lossy case works without a separate code path. Overlapping
mutual couplings raise `NotImplementedError`; only single-ended ports to ground
are exported.

```powershell
python -m twpa_solver.builders.jc_doc --outdir outputs/jc_designs
python -m twpa_solver.builders.jc_doc --outdir outputs/jc_designs --cases jc_jpa jc_jtwpa
python -m twpa_solver.builders.jc_doc --outdir outputs/jc_lossy --cases jc_fqjtwpa_diss --tandelta 2e-3
```

```python
from twpa_solver.builders.jc_doc import build_jpa, build_jtwpa

builder, metadata = build_jtwpa()
assembled = builder.assemble()          # C, G, K, Bphi, Ic, ports
metadata["pump_freqs_ghz"]              # [7.12]
metadata["pump_sources"]                # [{"port": 1, "mode": [1], "current_a": 1.85e-6}]
```

Each case's metadata carries `pump_freqs_ghz`, `pump_sources`, the signal sweep,
`Npumpharmonics` / `Nmodulationharmonics`, a `features` dict, and a
`recommended_first_test` string. `write()` emits both `summary.json` and
`ipm_summary.json`, plus a `build_manifest.json` across all cases.

## Pump-mode policy

`Nmodulationharmonics` maps to the pump basis. JC's nonlinear pump for an
unbiased 4WM device uses the **odd** mode list `[1,3,...,2K-1]`, not dense
`[1..H]`. Using dense harmonics truncates the high odd pump content and left a
~0.89 dB JTWPA gain mismatch. See `twpa_solver.pump.basis`:

- unbiased 4WM (JPA, JTWPA, FQJTWPA): `positive_odd_jc`, `K = Nmodulationharmonics`
- biased / DC / 3WM (FXJPA): `dense_real` plus a DC solution
- complex/lossy (FQJTWPA_diss): `positive_odd_jc` with complex matrices
- multi-pump (DPJPA): needs 2-D lattice HB, not the scalar policy

---

# 3. `le_gal_2025.py` — effective-SNAIL line

Returns a `CircuitMatrices` directly; no artifact directory.

```python
from twpa_solver.builders.le_gal_2025 import build_effective_snail_line

circuit = build_effective_snail_line(
    cells=700,
    cell_length_m=8.7e-6,
    critical_current_a=1.4e-6,
    ratio=0.062,                    # small-junction to large-junction Ic ratio
    snail_capacitance_f=31e-15,
    ground_capacitance_f=223.5e-15,
    flux_over_flux0=0.5,
    port_impedance_ohm=62.4,
    shunt_conductance_s=0.0,
    external_flux_on_small_junction=False,
)
```

A serial SNAIL line with matched ports at both ends. The branch law is shifted
to its **solved static equilibrium**: for each cell, `brentq` finds the flux
where the branch current vanishes, and the law is rebuilt around that root. The
builder then asserts the residual is below `1e-16 * Ic` and that the tangent is
positive, so an unstable branch cannot slip through.

The 31 fF SNAIL capacitance is stamped across the branch; 223.5 fF is the ground
capacitance, giving `sqrt(L/Cg)` about 62.3765 Ω. `shunt_conductance_s` is an SI
conductance, not a loss tangent — convert with `conductance_from_loss_tangent`
at a chosen reference frequency.

**`K` is deliberately empty.** The effective-SNAIL stiffness enters through
`Bphi @ i_branch` in every nonlinear residual and tangent; stamping its
small-signal slope into `K` as well would count it twice. That double-count was
a real bug and it killed the model's gain.

`ladder_dispersion` gives the discrete-ladder wavenumber from the exact cell
relation `sin(k dx/2)² = omega² L Cg / (4(1 - omega² L C))`.

Benchmark gain must be read through `multitone.observables.tone_s21` as a
pump-on/pump-off ratio. Do **not** reconstruct gain from `|i omega X| / V_in` —
that path was biased by 12.041 dB.

---

# 4. `scattered.py` — legacy Lj-only artifact copy

Copies an existing design directory and applies multiplicative Gaussian scatter
to `Lj` only, editing `ipm_arrays.npz` and `ipm_elements.csv` in place. At
sigma 0 it is an exact artifact copy.

```powershell
python -m twpa_solver.builders.scattered --design 2c --outdir outputs/ipm_2c_s1pct `
  --lj-scatter-sigma 0.01 --lj-scatter-seed 1
```

This path works only because `Lj` is absent from `K` — it never restamps `C`,
so it **cannot** extend to `Cj` or `Cg`. For anything beyond `Lj`, use
`ipm.build_variant_design`. `scattered.py` is retained because its source
directories are treated as authoritative and some older maps depend on it.

It reads the CSV header from the file rather than assuming a fixed schema, so
designs carrying the newer `role` / `cell_index` columns round-trip intact.

---

# Support modules

## `profiles.py`

Pure index-to-value math and spec parsing. No circuit imports, no I/O.

```python
from twpa_solver.builders.profiles import (
    Segment, Selection, evaluate_profile,
    parse_profile_shorthand, parse_profile_json,
    segment_to_dict, segments_to_json,
)

values = evaluate_profile(
    [Segment("linear", 100e-12, 200e-12, select=Selection(rows=(0, 2)))],
    n_cells=2508, cells_per_row=418, base_value=123.9e-12,
)
```

Because it knows nothing about circuits, wiring these profiles into `jc_doc` or
`le_gal_2025` is a wiring job, not a redesign.

## `scatter.py`

```python
from twpa_solver.builders.scatter import (
    ScatterSpec, draw_factors, component_rng, apply_scatter,
)

values, meta = apply_scatter(nominal, ScatterSpec(0.01), component_rng(7, "Lj"))
meta["factor_std"], meta["clip_hits"], meta["factor_digest"]
```

---

# Choosing a builder

- Real IPM / 2c hardware, or anything needing per-cell control → `ipm.py`
- Regression fixtures and compression-campaign devices → `jc_doc.py`
- Le Gal benchmark → `le_gal_2025.py`
- Reproducing an old Lj-scatter map exactly → `scattered.py`

## Cautions

- **Live designs are `designs/*`.** Circuit directories under `outputs/` are
  stale; all of exp20–exp31 compression ran on a stale 2c circuit.
- **Coupler modes are not interchangeable.** `auto`, `ideal`, and `optimize`
  produce different coupler realizations when explicitly selected.
- **Pump injection is port 4** for the gain-map 2c wiring, while signal
  scattering is 1 → 2.
- **JC is not a physical reference** — see the `jc_doc.py` section.
- Profiled and scattered designs are verified to build, load, and converge
  through the nonlinear pump solve, but no physics validation has been done on
  what a given taper achieves.

# Tests

| File | Covers |
| --- | --- |
| `tests/test_component_profiles.py` | shape kernels, selectors, domain, AST allowlist, serialization |
| `tests/test_component_scatter.py` | determinism, stream independence, legacy parity, relative semantics |
| `tests/test_ipm_role_tags.py` | role counts, cell ordering, byte identity vs `designs/ipm_2c_fixed` |
| `tests/test_ipm_component_plan.py` | plasma frequency, Cg mapping, block profiles, back-compat |
| `tests/test_variant_design.py` | topology gate, coupler-mode detection, regenerability |
| `tests/test_builder_imports.py` | import surface |
| `tests/test_le_gal_2025_builder.py` | SNAIL equilibrium and stamps |
| `tests/test_fxjtwpa_node_order.py` | FXJTWPA node-ordering permutation |

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_builders `
  tests/test_component_profiles.py tests/test_component_scatter.py `
  tests/test_ipm_role_tags.py tests/test_ipm_component_plan.py tests/test_variant_design.py
```

Full suite, including the slow physics gates:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_full_slow --run-slow
```

Each profile/scatter gate was verified by mutation — removing the `Cg` halving,
making `Cj` constant, merging the `Cj`/`Cg` streams, breaking endpoint anchoring
to `i/N`, and running the variant gate on the variant netlist all turn the suite
red.
