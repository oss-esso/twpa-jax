# Circuit API guide

The Python ``Circuit`` API is the primary authoring interface for electrical
designs. It constructs a symbolic graph. Compilation assigns solver node
numbers and emits the existing ``Element[]`` representation; matrix assembly
continues to use ``builders.ipm.build_matrices``.

## 1. Create a circuit and paths

```python
from twpa_solver.circuit import Circuit

circuit = Circuit("small_example")
signal = circuit.path("signal")
pump = circuit.path("pump")
```

``ground`` is an explicit symbolic node and compiles to solver node ``0``.
Every other node is assigned an integer only by ``compile``. A ``Path`` keeps
its current endpoint and advances when a block is appended.

## 2. Primitive graph construction

```python
left = signal.start
right = circuit.node("junction.right")
circuit.add_capacitor(left, circuit.ground, 66e-15, name="input.Cg")
circuit.add_jj(left, right, Lj=123.9e-12, Cj=145e-15, name="input.JJ")
signal.extend(right)
circuit.add_port(signal.end, number=1, impedance=50.0)
```

Primitive builders return handles where a later edit is useful. For example,
``add_jj`` returns the Josephson-inductor handle and exposes its companion
junction capacitor through ``handle.companion``.

## 3. Cells and lines

Use cell and line builders for repeated structures. They compose the primitive
builders and return structured handles.

```python
line = circuit.add_jj_line(
    signal,
    cells=36,
    Lj=123.9e-12,
    Cj=145e-15,
    Cg=66e-15,
    name="row[0].array[0]",
)
first_junction = line.cell(0).Lj
circuit.set_value(first_junction, 124.1e-12)

tl = circuit.add_transmission_line(
    signal, cells=10, L=4.24e-12, C=1.695e-15, name="interconnect"
)
```

For a fabrication junction array, ``add_jj_array`` emits one lumped
equivalent and adds no nodes:

```python
array_junction = circuit.add_jj_array(
    signal.end,
    circuit.node("array.right"),
    Lj=123.9e-12,
    Cj=145e-15,
    count=3,
    name="array[0].JJ",
)
```

The approximation uses ``Lj_eff = count * Lj`` and
``Cj_eff = Cj / count``. It is a circuit approximation, not a claim that the
individual fabrication polygons or internal array nodes are represented.

## 4. Directional couplers

The two paths must belong to the same circuit and must be distinct. The
builder advances both paths by the number of coupled cells.

```python
circuit.add_directional_coupler(
    signal,
    pump,
    coupling_db=-14.0,
    frequency=8e9,
    z0=50.0,
    mode="cached",
    name="coupler[0]",
)
```

For the v3 cross-section, pass the fabrication dimensions explicitly. This
bypasses geometry optimization and selects the existing three-conductor CPW
path:

```python
from twpa_solver.circuit import ExplicitCouplerGeometry

geometry = ExplicitCouplerGeometry(
    gaps_um=[5.5, 5.0, 5.0, 5.5],
    widths_um=[9.186, 15.0, 9.186],
    length_um=2738.2160926784595,
)
circuit.add_directional_coupler(signal, pump, geometry=geometry)
```

The conformal path is a true three-conductor calculation for this symmetric
cross-section. With the starting dimensions it reports
``-28.2496555093 dB`` and ``Z_eff = 49.3393827325 ohm``. The v3 builder ports
Prometheus ``getCouplerDimentions`` as a bounded least-squares fit over the
three-conductor dimensions, then passes the optimized result through
``ExplicitCouplerGeometry``. At ``-25 dB`` and 10 GHz the resulting discrete
model reports approximately ``-25.0 dB`` and ``50.0 ohm``.

The fit deliberately reproduces Prometheus' ``beta_odd =
2*pi*frequency/v_even`` expression; this is a known deviation from the
theoretical ``v_odd`` expression. The Prometheus ``2738.216`` micrometre
starting value is an unrolled centre-line length. Its fab coupler contains one
straight section plus meanders; the discrete model represents only the
unrolled length and does not model bends, tapers, air bridges, or bend
discontinuities.

Prometheus' optional per-coupler leakage correction is exposed as
``coupler_leakage_db(coupling_db, coupler_number)``. v1-compatible designs
leave it disabled by default; a design must opt in explicitly.

## 5. Profiles

Profiles are deterministic frozen objects and can be passed directly to line
builders:

```python
from twpa_solver.circuit import HalfSine, Hann, Linear

line = circuit.add_jj_line(
    signal,
    cells=36,
    Lj=Linear(120e-12, 130e-12, domain="all"),
    Cj=145e-15,
    Cg=Hann(60e-15, 70e-15, domain="all"),
)
```

The names are intentionally not interchangeable:

| Profile | Existing shape | Formula on normalized `t` |
| --- | --- | --- |
| `HalfSine` | `custom` | `sin(pi*t/2)` |
| `Hann` | `half_cosine` | `(1 - cos(pi*t))/2` |

Both profiles are tested against their formulas and against each other on the
same inputs.

## 6. Targeted edits and removal

Element handles support local edits without rebuilding unrelated symbolic
objects:

```python
circuit.set_value(first_junction, 125e-12)
circuit.remove(first_junction.companion)
```

Removal is checked and repeated removal is rejected. Cross-circuit nodes and
element handles are rejected when they are passed to a builder.

## 7. Compilation and numbering

```python
compiled = circuit.compile()  # creation order is the default
netlist_text = circuit.export_netlist()
matrices = compiled.matrices()
```

``node_numbering="creation"`` assigns integers in first-touch creation order.
``node_numbering="legacy"`` is a compatibility policy only. It preserves the
historical IPM numbering so the YAML adapter and the stored v1 artifacts remain
byte-identical. Element emission order is unchanged by either policy, so
``Bphi`` columns and ``Ic`` order remain stable.

The API does not expose a squared kinetic-inductance parameter. Designs supply
the physical inductance value and perform any application-specific
multiplication before calling a builder.

## 8. v2 and v3 design sources

The fab-shaped examples are ordinary Python composition:

```python
from designs.python.ipm_v2 import build_ipm_v2
from designs.python.ipm_v3 import build_ipm_v3

v2 = build_ipm_v2()
v3 = build_ipm_v3()
```

v2 uses the lumped-array approximation for ``20 x 15 x 3 x 2 = 1800``
junction branches. v3 uses ``36 x 18 x 6 = 3888`` junction branches and three
explicit three-conductor couplers. These gates verify structure and numerical
lowering only; neither design has a stored electrical reference, measurement
validation, or GDS-equivalence claim.

## 9. YAML compatibility

Existing YAML remains supported:

```python
from twpa_solver.design import compile_design, load_design

design = compile_design(load_design("designs/ipm_2c.yaml"))
```

The compiler adapts YAML blocks to the public ``Circuit`` builders and compiles
with the legacy numbering policy. New authoring work should use Python under
``designs/python/`` so the electrical construction is visible in the source.
