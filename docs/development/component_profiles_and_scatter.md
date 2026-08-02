# Component profiles and scatter

The IPM builder accepts ordered per-cell profiles for `Lj` and `Cg`, plus
independent multiplicative scatter on `Lj`, `Cj`, and `Cg`. All three share one
index space: the JTL cell index, `0 .. num_rows * array_length - 1` (2508 on
2c). There is no `Cj` profile — nominal `Cj` is always derived from the `Lj`
profile so the plasma frequency stays constant.

## Segments

A profile is an ordered list of segments; later segments overwrite earlier ones
on overlap, and untouched cells keep the scalar `IPMParams` value. Any segment
driving a value to zero or below raises and names the offending cell.

Selection is `all`, `rows=A[-B]`, `index=A[-B]`, or `frac=A[-B]` (closed
ranges); at most one selector per segment. `domain` is `selection` (the shape
spans the whole selected range) or `per_row` (the shape restarts inside each
JTL row).

Every shape is a normalized kernel `s(t)` on `t in [0, 1]`, and the emitted
value is `start + (end - start) * s(t)`. With `N` cells, `t_i = i/(N-1)`, so
the anchored shapes hit `start` at the first cell and `end` at the last by
construction. A one-cell selection returns `start`.

| shape | `s(t)` | params | endpoints |
| --- | --- | --- | --- |
| `const` | — | — | every cell is `start`; `end` must be absent or equal |
| `linear` | `t` | — | anchored, exact |
| `power` | `t**p` | `exponent` (`p`), > 0 | anchored |
| `parabola` | `((t-v)**2 - v**2) / ((1-v)**2 - v**2)` | `vertex` (`v`) in `[0,1]`, not 0.5, default 0 | anchored |
| `half_cosine` | `(1 - cos(pi t)) / 2` | — | anchored, smooth |
| `tanh` | normalized `tanh(k(2t-1))` | `sharpness` (`k`), > 0 | anchored |
| `sine` | `(1 + sin(2 pi f t + phi)) / 2` | `periods`, `phase` | **envelope** |
| `cosine` | `(1 + cos(2 pi f t + phi)) / 2` | `periods`, `phase` | **envelope** |
| `custom` | user expression in `t` | — | caller's responsibility |

`vertex=0` is the plain `t**2` parabola and is the default; `vertex=0.5` puts
the turning point at the midpoint, cannot be normalized, and raises.

**Sine and cosine use an envelope convention.** Periodic and endpoint-anchored
are mutually exclusive, so for these two `start`/`end` are the min/max of the
oscillation (mean `(start+end)/2`, amplitude `(end-start)/2`), not the first
and last values. `linspace` does not land exactly on the extrema, so the
sampled min/max approach `start`/`end` as the cell count grows rather than
matching them exactly.

`custom` expressions are parsed with `ast` and checked against an allowlist
before evaluation: arithmetic operators, `t`, `pi`, `e`, and direct calls to a
fixed set of numpy functions (`sin`, `cos`, `tan`, `tanh`, `exp`, `log`,
`sqrt`, `abs`, `floor`, `ceil`, `sign`, `minimum`, `maximum`, and the inverse
trigonometric functions). Attribute access, subscripting, comprehensions,
keyword arguments, and any other name are rejected. Evaluation runs with empty
builtins.

## Specifying a profile

Shorthand, repeatable as `--lj-profile` / `--cg-profile`:

```
<select>:<shape>:<start>[-><end>][:k=v,k=v]
```

`domain` and `expression` are reserved keys in the trailing field, so both are
reachable without a JSON file. Values accept `p`, `n`, `u`, `f` SI suffixes.

```powershell
--lj-profile "rows=0-2:const:150p"
--lj-profile "rows=3-5:linear:123.9p->140p"
--lj-profile "all:sine:120p->140p:periods=2,domain=per_row"
--cg-profile "all:custom:66f->72f:expression=sin(pi*t)"
```

`--profile-json` takes `{"Lj": [...], "Cg": [...]}`, where each entry is either
a shorthand string or an object with `shape`, `start`, `end`, `select`,
`domain`, `params`, `expression`. CLI segments are appended after JSON ones, so
they win on overlap.

## Cg index mapping

The circuit has one more ground cap per JTL row than it has cells. Cell `c` of
row `r` owns the cap at its left node with value `cg[r*array_length + c]`,
halved when `c == 0`, and the row's trailing cap is `cg[last cell of row] / 2`.
For a constant profile this reduces to the legacy
`Cg/2 + (array_length-1)*Cg + Cg/2`, which is what pins the byte-identity gate.

## Scatter

Scatter is multiplicative against each cell's own nominal, so `sigma` is a
fraction of that cell's value, not of a global mean. A cell at twice the
nominal absorbs twice the absolute deviation.

Streams are fixed and must never be reordered — only appended to:

| component | generator |
| --- | --- |
| `Lj` | `default_rng(seed)` |
| `Cj` | `default_rng(SeedSequence([seed, 1]))` |
| `Cg` | `default_rng(SeedSequence([seed, 2]))` |

`Lj` deliberately uses the bare `default_rng(seed)` so it is bit-identical to
the pre-profile `apply_lj_scatter`, and changing `--cg-scatter-sigma` cannot
perturb the `Lj` realization.

**Independent `Cj` scatter intentionally breaks constant plasma frequency.**
Only nominal `Cj` is derived from the `Lj` profile; the scatter draw is
separate by design and must not be "corrected" to track `Lj`.

`--scatter-seed` is the master seed. `--lj-scatter-seed` is a deprecated alias;
passing both with different values raises.

## Artifacts

`ipm_arrays.npz` gains `Cj`, `Cg`, `cell_index`, and `Lj_nominal`,
`Cj_nominal`, `Cg_nominal`, all indexed by cell. `ipm_summary.json` carries a
`component_plan` block with the seed, the serialized segments, the scatter
specs, and realized statistics including a factor digest — enough to regenerate
the design, without embedding the per-cell arrays that already live in the npz.
The legacy `lj_scatter_*` summary keys are still emitted.

`ipm_elements.csv` gains `role` and `cell_index` columns. The roles that matter
are `jj_lj`, `jj_cj`, and `jtl_cg`; the rest restate `kind`. Tagging is
required because `Cg` is not structurally identifiable — filtering for
"capacitor to ground on a Josephson node" returns 2522 elements on 2c, of which
only 2514 are `Cg` (six are TL `Cl`, two are coupler end caps).

## Variants of an existing design

`build_variant_design(source_dir, outdir, ...)` rebuilds a stored design from
its own `ipm_summary.json["params"]` with `--coupler-mode cached`, gates the
result, then applies the profile and scatter.

`assert_source_topology` runs that gate on a **nominal** rebuild. Comparing the
profiled netlist against the source would reject every real variant, since
changing component values is the point. The gate is exact — names, nodes,
kinds, and values must all match — and a mismatch raises rather than
proceeding, because it means the stored params do not describe the artifact.

`designs/ipm_2c_fixed` passes at 16312/16312 elements with `C/G/K/Bphi`
maxdiff 0.0. Other design directories must be checked before being used as
variant sources; `--coupler-mode ideal` and `optimize` do **not** reproduce
`ipm_2c_fixed`.

## Tests

`tests/test_component_profiles.py`, `tests/test_component_scatter.py`,
`tests/test_ipm_role_tags.py`, `tests/test_ipm_component_plan.py`,
`tests/test_variant_design.py` (95 tests). Each gate was verified by mutation:
removing the `Cg` halving, making `Cj` constant, giving `Cj`/`Cg` one stream,
breaking endpoint anchoring to `i/N`, and running the variant gate on the
variant netlist all turn the suite red.
