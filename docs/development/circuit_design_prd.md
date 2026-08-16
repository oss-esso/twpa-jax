# PRD — Object-Oriented Circuit Design Framework Aligned with the Fabrication/GDS Codebase

## 1. Purpose

Replace the current YAML-centered design-definition workflow with a **Python object-oriented circuit construction API** that:

* closely mirrors the fabrication team's existing GDS design code;
* preserves the current SPICE-like circuit/netlist representation used by the simulator;
* supports arbitrary circuit topology;
* automatically manages nodes;
* supports arbitrary numbers of signal, pump, bias, auxiliary, and other lines;
* allows users to work at any desired level of abstraction;
* allows high-level premade blocks and precise local circuit modifications in the same design;
* supports deterministic spatial parameter profiles as part of the circuit design;
* compiles into the existing solver representation without changing solver physics.

The intended user experience is:

```text
GDS design code                  Simulation design code
      │                                  │
      │ similar vocabulary               │
      │ similar hierarchy                │
      ▼                                  ▼
physical layout                    Circuit object
                                         │
                                         ▼
                                symbolic netlist graph
                                         │
                                         ▼
                                  flat Element[]
                                         │
                                         ▼
                                   C / G / K / Bphi
                                         │
                                         ▼
                                      solver
```

The fabrication team should be able to look at the GDS script and the simulation script side by side and recognize the same physical design structure.

The simulator does **not** need to reproduce geometry parameters that have no meaning for the lumped/equivalent circuit model.

---

# 2. Scope

This task concerns **circuit/design construction only**.

It includes:

* circuit object model;
* node representation;
* path/line management;
* ports;
* primitive elements;
* unit cells;
* composite circuit blocks;
* deterministic spatial profiles;
* hierarchy/introspection;
* flattening to the existing solver representation;
* SPICE-like netlist export;
* validation;
* migration of existing designs.

It does **not** include redesigning:

* harmonic balance;
* pump solving;
* continuation;
* gain maps;
* compression;
* transient simulation;
* Floquet stability analysis;
* solver backends;
* experiments/campaigns;
* result schemas;
* optimization workflows.

The existing numerical and physics layers should remain untouched except where a compatibility adapter is required.

---

# 3. Fundamental architectural decision

The public API should present **one main `Circuit` class**.

Users should be able to do things such as:

```python
c = Circuit("ipm_2c")

signal = c.path("signal")
pump = c.path("pump")

c.add_port(signal.start, number=1, impedance=50)
c.add_port(pump.start, number=3, impedance=50)

coupler1 = c.add_directional_coupler(signal, pump, ...)

line1 = c.add_jj_line(
    signal,
    cells=418,
    Lj=123.9e-12,
    Cj=145e-15,
    Cg=66e-15,
)

c.add_capacitor(
    line1.cell(205).right,
    c.ground,
    C=98e-15,
)

coupler2 = c.add_directional_coupler(signal, pump, ...)

line2 = c.add_jj_line(signal, ...)

c.add_port(signal.end, number=2, impedance=50)
c.add_port(pump.end, number=4, impedance=50)

compiled = c.compile()
```

This is the target style.

The user should **not** manually manage node integers during normal circuit construction.

---

# 4. Important distinction: public facade vs implementation

The user should feel like `Circuit` contains everything:

```python
c.add_resistor(...)
c.add_capacitor(...)
c.add_inductor(...)
c.add_jj(...)

c.add_tl_cell(...)
c.add_jj_cell(...)
c.add_rf_squid_cell(...)

c.add_transmission_line(...)
c.add_jj_line(...)
c.add_rf_squid_line(...)

c.add_directional_coupler(...)
c.add_resonator(...)
c.add_ipm_section(...)
```

However, do **not** implement this as one enormous source file.

Internally use modular implementation code.

Suggested organization:

```text
twpa/
    circuit/
        circuit.py
        graph.py
        nodes.py
        paths.py
        ports.py
        elements.py

        primitives/
            resistor.py
            capacitor.py
            inductor.py
            josephson.py
            mutual.py

        cells/
            tl_cell.py
            jj_cell.py
            rf_squid_cell.py

        blocks/
            transmission_line.py
            jj_line.py
            coupler.py
            resonator.py
            rf_squid_line.py
            ipm.py

        profiles.py

        compiler.py
        validation.py
        netlist_export.py
```

Exact filenames are flexible.

The architectural requirement is not.

---

# 5. Core data model: arbitrary graph first

The **fundamental circuit representation must be a graph**:

```text
Nodes + Elements + Ports
```

Do not make the underlying model a chain or cascade.

A traveling-wave transmission path is common, but arbitrary circuits must remain representable.

Examples that must remain easy:

```text
                    resonator
                        |
signal ----o------------o------------o----
```

```text
signal ----o-------------------------o----
            |
          rf-SQUID
            |
          ground
```

```text
signal ----o==============o----
            coupled region
pump   ----o==============o----
```

Therefore:

```text
Circuit
    owns Nodes
    owns Elements
    owns Ports
    owns Paths
```

and:

```text
Path
```

is an ergonomic abstraction over part of the graph.

It is **not** the fundamental circuit representation.

---

# 6. Symbolic nodes

Nodes must become first-class objects.

Conceptually:

```python
@dataclass(frozen=True)
class Node:
    uid: int
    name: str | None
    path: str
```

The exact implementation may differ.

During circuit construction the user works with symbolic nodes such as:

```text
signal[0]
signal[1]
signal[2]

pump[0]
pump[1]

bias[0]
bias[1]
```

The user should not care what numerical node IDs the solver eventually receives.

---

# 7. Integer node IDs are assigned only during compilation

This is a firm requirement.

During circuit construction:

```text
signal[0]
signal[1]
pump[0]
...
```

are symbolic `Node` objects.

Only:

```python
compiled = circuit.compile()
```

assigns final integer solver IDs.

For example:

```text
ground      -> 0
signal[0]   -> 1
signal[1]   -> 2
signal[2]   -> 3
pump[0]     -> 4
pump[1]     -> 5
...
```

The allocation must be:

* deterministic;
* reproducible;
* independent of port numbers;
* independent of arbitrary hand-selected node-number ranges.

Do not use schemes such as:

```text
signal starts at 1
pump starts at 10000
bias starts at 20000
```

in the public design API.

---

# 8. Paths

Introduce a first-class `Path` abstraction.

Example:

```python
signal = c.path("signal")
pump = c.path("pump")
bias = c.path("bias")
```

A `Path` represents an ordered route through nodes and exists primarily to simplify traveling-wave circuit construction.

Conceptually it should support:

```python
signal.start
signal.end
signal.node(205)
len(signal.nodes)
```

Blocks that operate on a path should automatically advance it.

Example:

```python
line = c.add_jj_line(
    signal,
    cells=418,
    ...
)
```

should:

1. begin at `signal.end`;
2. append the required nodes;
3. create the appropriate elements;
4. update `signal.end`;
5. return a handle representing the created JJ line.

No explicit cursor management should be required.

---

# 9. Port numbers must not determine circuit topology or node allocation

Maintain strict separation:

```text
Node
    electrical connection point

Path
    ordered route through nodes

Port
    external source/measurement interface attached to a node
```

Example:

```python
signal = c.path("signal")

c.add_port(
    signal.start,
    number=1,
    impedance=50,
)

c.add_port(
    signal.end,
    number=2,
    impedance=50,
)
```

Similarly:

```python
pump = c.path("pump")

c.add_port(pump.start, number=3)
c.add_port(pump.end, number=4)
```

If a future device contains:

```text
6 paths
12 ports
3 DC-bias inputs
```

the node-management system must require no redesign.

---

# 10. Arbitrary graph construction without Paths

`Path` is optional.

Users must also be able to build arbitrary circuits directly.

For example:

```python
a = c.node("a")
b = c.node("b")
x = c.node("x")

c.add_capacitor(a, x, C=...)
c.add_inductor(x, b, L=...)
c.add_resistor(x, c.ground, R=...)
```

This is the low-level escape hatch.

A valid circuit must never be forced into the `Path` abstraction.

---

# 11. Builder hierarchy

The implementation must have multiple abstraction levels.

## Level 1 — primitive elements

At minimum:

```python
c.add_resistor(...)
c.add_capacitor(...)
c.add_inductor(...)
c.add_jj(...)
c.add_mutual_inductor(...)
c.add_port(...)
```

These are the foundational SPICE-like builders.

---

## Level 2 — physical unit cells

Examples:

```python
c.add_tl_cell(...)
c.add_jj_cell(...)
c.add_rf_squid_cell(...)
```

These must be implemented by calling Level-1 builders.

Example conceptually:

```python
def add_jj_cell(...):
    self.add_capacitor(...)
    self.add_jj(...)
    self.add_capacitor(...)
```

Do not stamp matrices directly.

---

## Level 3 — repeated/macroscopic blocks

Examples:

```python
c.add_transmission_line(...)
c.add_jj_line(...)
c.add_rf_squid_line(...)
c.add_directional_coupler(...)
c.add_resonator(...)
```

These must call Level-2 and/or Level-1 public builders.

Example:

```python
def add_jj_line(...):
    for i in range(cells):
        self.add_jj_cell(...)
```

---

## Level 4 — architecture-specific reusable structures

Examples:

```python
c.add_ipm_section(...)
c.add_rpm_section(...)
c.add_phase_matching_section(...)
```

These should only exist where the same structure is meaningfully reused.

Do not turn every complete design into a special function.

---

# 12. Non-negotiable composition rule

A high-level builder may **never bypass the netlist representation**.

Forbidden:

```text
add_ipm_section()
      ↓
directly modifies C/G/K/Bphi
```

Required:

```text
add_ipm_section()
      ↓
calls add_coupler()
calls add_jj_line()
calls add_transmission_line()
      ↓
calls lower-level builders
      ↓
creates Nodes + Elements
      ↓
compile()
      ↓
Element[]
      ↓
C/G/K/Bphi
```

There must always be one SPICE-like source of truth.

---

# 13. Return structured handles from builders

High-level builder methods must not simply return `None`.

They should return objects exposing useful internal structure.

Example:

```python
line = c.add_jj_line(...)
```

should support:

```python
line.input
line.output

line.node(205)

line.cell(205)
```

and:

```python
cell = line.cell(205)

cell.left
cell.right

cell.Lj
cell.Cj
cell.Cg
```

This is critical.

It gives the user high-level construction while preserving arbitrary targeted granularity.

---

# 14. Targeted granularity

The following must be easy:

```python
line = c.add_jj_line(
    signal,
    cells=418,
    ...
)
```

followed by:

```python
c.add_capacitor(
    line.cell(205).right,
    c.ground,
    C=98e-15,
)
```

or:

```python
c.set_value(
    line.cell(205).Cg,
    98e-15,
)
```

or:

```python
c.remove(
    line.cell(205).Cj,
)
```

The user must not have to recreate the line manually merely to modify one local component.

This applies recursively to all composite structures.

---

# 15. Coupler handle

The directional coupler is an important API test because it operates on two paths.

The following should be possible:

```python
coupler = c.add_directional_coupler(
    signal,
    pump,
    coupling_db=-14,
    frequency=8e9,
)
```

The operation should automatically advance both:

```python
signal.end
pump.end
```

The returned coupler handle should expose useful terminals and structure, for example:

```python
coupler.signal_in
coupler.signal_out

coupler.pump_in
coupler.pump_out

coupler.cell(30)
```

Exact names should preferably match the GDS/fab repository vocabulary.

---

# 16. Hierarchical naming

Every created block, cell, and relevant element should have a stable hierarchical path.

Example:

```python
section = c.add_ipm_section(
    signal,
    pump,
    name="section_1",
)
```

might create:

```text
section_1
    coupler
    row[0]
        array[0]
            cell[0]
            cell[1]
            ...
        array[1]
    row[1]
        ...
```

Individual elements should be addressable with stable paths such as:

```text
section_1.row[0].array[0].cell[205].Lj
section_1.row[0].array[0].cell[205].Cj
section_1.row[0].array[0].cell[205].Cg
```

This hierarchy is important for:

* local edits;
* debugging;
* fabrication variation;
* profiles;
* parameter extraction;
* comparison with GDS;
* future visualization.

Do not make solver node integers the primary identity of physical components.

---

# 17. Primitive element handles

Primitive builders should also return handles.

Example:

```python
cap = c.add_capacitor(
    node_a,
    node_b,
    C=66e-15,
)
```

should return a handle that provides at least:

```python
cap.n1
cap.n2
cap.value
cap.name
cap.path
```

Likewise:

```python
jj = c.add_jj(...)
```

should expose its connection and model parameters.

---

# 18. Profiles are part of design

Deterministic spatial variation is part of what the physical fabricated device is.

Therefore profiles belong directly in the OOP design API.

Do not require external CLI strings such as:

```text
--lj-profile "all:linear:100p->150p:domain=per_row"
```

Instead support profile objects.

For example:

```python
from twpa.profiles import Constant, Linear, HalfSine

line = c.add_jj_line(
    signal,
    cells=2400,
    Lj=Linear(
        start=100e-12,
        stop=150e-12,
        domain="per_row",
    ),
    Cg=HalfSine(
        start=53.2688e-15,
        stop=79.9031e-15,
        domain="per_row",
    ),
    Cj=145e-15,
)
```

A plain scalar must still work:

```python
Lj=123.9e-12
```

The builder should internally normalize:

```text
scalar
or
Profile
```

into the values used for each generated cell.

---

# 19. Required profile types

At minimum provide:

```python
Constant(...)
Linear(...)
HalfSine(...)
```

The named `HalfSine` profile must reproduce the existing behavior corresponding to:

```text
sin(pi*t/2)
```

where appropriate.

An advanced `CustomProfile` may be retained if existing code requires it, but common cases should not require users to type mathematical expression strings.

Reuse the existing profile mathematics rather than rewriting it unnecessarily.

---

# 20. Existing fabrication scatter

Do not confuse:

```text
deterministic design profile
```

with:

```text
stochastic fabrication variation
```

They may eventually share infrastructure, but they are conceptually distinct.

Examples:

```python
Lj=Linear(...)
```

means the designer intentionally fabricated a spatially varying Lj.

A future:

```python
variation=Gaussian(...)
```

would represent fabrication uncertainty.

Do not redesign stochastic campaign handling as part of this task unless required for compatibility.

---

# 21. SPICE-like flattened representation remains authoritative

The OOP design layer must compile to the existing solver-facing representation.

Required flow:

```text
Circuit object
    ↓
symbolic graph
    ↓
deterministic node numbering
    ↓
flat Element[]
    ↓
existing matrix assembler
    ↓
C / G / K / Bphi
```

Do not rewrite working matrix physics merely because the design API changes.

---

# 22. SPICE-like textual netlist export

Add or preserve:

```python
c.export_netlist(...)
```

The purpose is debugging and reproducibility.

Example conceptual output:

```text
R1 n1 0 50
C1 n1 0 66f
Lj1 n1 n2 123.9p
Cj1 n1 n2 145f
...
```

Exact syntax can follow existing project conventions.

The design pipeline should therefore be inspectable as:

```text
GDS design
    ↕
Python Circuit design
    ↓
SPICE-like flattened netlist
    ↓
Element[]
    ↓
matrices
```

---

# 23. GDS/fabrication repository alignment

Before finalizing public names or signatures, inspect the cloned GDS repository.

This is mandatory.

Inventory:

* directory structure;
* design classes;
* cell classes;
* helper functions;
* path/routing abstractions;
* port abstractions;
* naming conventions;
* directional coupler builders;
* JJ-array builders;
* rf-SQUID builders;
* resonator builders;
* repeat/supercell structures;
* technology/process parameters.

Produce a mapping table before major API edits.

Example:

```text
GDS/Fab concept             Simulation concept
---------------             ------------------
<their name>                <our equivalent>
JJ cell                     JJ equivalent cell
JJ array                    JJ line
directional coupler         lumped coupler model
route/path                  Path
port                        electrical Port
...
```

The simulation API should mirror the GDS vocabulary wherever the physical concept is the same.

---

# 24. Geometry-only parameters

Do not blindly reproduce every GDS method argument.

The fab code may contain:

```text
width
gap
corner radius
metal layer
GDS layer number
orientation
physical coordinate
meander radius
polygon offset
```

If those do not directly affect the equivalent circuit representation used by the simulator, they should not be required in the simulation API.

The goal is:

```text
same physical vocabulary
same hierarchy
similar construction sequence
```

not:

```text
identical function signatures regardless of relevance
```

Where geometry is converted into electrical parameters by a validated model, the simulation API may either:

1. accept the resulting electrical values directly, or
2. reuse the same geometry-to-electrical conversion if appropriate.

Do not duplicate unnecessary layout implementation.

---

# 25. Example desired side-by-side workflow

If the GDS code conceptually looks like:

```python
device.add_directional_coupler(...)
device.add_jj_array(...)
device.add_directional_coupler(...)
device.add_jj_array(...)
```

the simulation code should ideally look like:

```python
c.add_directional_coupler(signal, pump, ...)
c.add_jj_line(signal, ...)
c.add_directional_coupler(signal, pump, ...)
c.add_jj_line(signal, ...)
```

The fabrication user should not need to mentally translate between completely unrelated abstractions.

---

# 26. Complete representative example

A 2-coupler design should eventually be expressible at roughly this level:

```python
from twpa import Circuit
from twpa.profiles import HalfSine

c = Circuit("ipm_2c")

signal = c.path("signal")
pump = c.path("pump")

# Ports / terminations
c.add_port(signal.start, number=1, impedance=50)
c.add_port(pump.start, number=3, impedance=50)

# First coupler
coupler_in = c.add_directional_coupler(
    signal,
    pump,
    coupling_db=-14,
    frequency=8e9,
)

# First nonlinear section
line1 = c.add_jj_line(
    signal,
    cells=418,
    Lj=123.9e-12,
    Cj=145e-15,
    Cg=66e-15,
)

# Targeted local edit
c.add_capacitor(
    line1.cell(205).right,
    c.ground,
    C=98e-15,
)

# Second coupler
coupler_2 = c.add_directional_coupler(
    signal,
    pump,
    coupling_db=-14,
    frequency=8e9,
)

# Profiled nonlinear section
line2 = c.add_jj_line(
    signal,
    cells=418,
    Lj=HalfSine(
        start=100e-12,
        stop=150e-12,
    ),
    Cj=145e-15,
    Cg=66e-15,
)

# Output ports
c.add_port(signal.end, number=2, impedance=50)
c.add_port(pump.end, number=4, impedance=50)

compiled = c.compile()
```

Do not interpret these particular physical block counts as a specification of the existing 2C circuit. Reproduce the actual existing topology during migration.

This example defines the desired API style.

---

# 27. Compile result

`Circuit.compile()` should return a structured result object.

Conceptually:

```python
@dataclass
class CompiledCircuit:
    elements: list[Element]

    node_map: dict[Node, int]
    reverse_node_map: dict[int, Node]

    ports: ...
    hierarchy: ...
    metadata: ...

    C: ...
    G: ...
    K: ...
    Bphi: ...
```

If matrix construction should remain a separate stage for architectural reasons, this is also acceptable:

```python
compiled = c.compile()
matrices = assemble(compiled.elements)
```

Prefer whichever matches the existing solver architecture with the least disruption.

---

# 28. Determinism

Given the same Python design and parameters, compilation must always produce identical:

```text
symbolic hierarchy
integer node numbering
element ordering
element names
Element[]
matrix sparsity
matrix values
```

Do not rely on unordered traversal where it affects output.

---

# 29. Validation

The `Circuit` should detect invalid construction as early as practical.

At minimum validate:

* duplicate explicit names where uniqueness is required;
* invalid node handles;
* elements referencing nodes from another Circuit;
* invalid cell indices;
* invalid profile domains;
* invalid port numbers;
* duplicate port numbers;
* attempts to remove an already removed element;
* negative/zero cell counts where invalid;
* invalid coupler/path combinations;
* duplicate symbolic paths;
* compile with dangling invalid references.

Errors should identify the relevant hierarchical component.

---

# 30. Migration strategy

Do not rewrite all designs simultaneously.

Proceed in stages.

## Stage 1 — graph primitives

Implement and validate:

```text
Circuit
Node
Element
Port
Path
compile()
```

with:

```text
R
C
L
JJ
```

---

## Stage 2 — basic cells

Implement:

```text
TL cell
JJ cell
```

using primitives.

Verify parity with the existing circuit builder.

---

## Stage 3 — lines

Implement:

```text
transmission line
JJ line
```

using cells.

Verify:

```text
node count
element count
element values
matrix parity
```

---

## Stage 4 — directional coupler

Implement the existing electrical coupler model through the new API.

This is the first major multi-path test.

Verify both paths automatically advance correctly.

---

## Stage 5 — migrate existing IPM/2C circuit

Recreate the known existing design using only the public OOP API.

Compare against the current trusted implementation.

---

## Stage 6 — profiles

Move deterministic linear/half-sine design profiles into builder arguments.

Keep old CLI profile routes temporarily as compatibility wrappers if necessary.

---

## Stage 7 — other architectures

Only after IPM parity is proven, migrate:

```text
uniform JTWPA
Floquet/profiled JTWPA
rf-SQUID
RPM
KI-TWPA
other designs
```

as applicable.

---

# 31. Legacy YAML work

Do not make YAML the new primary design authoring interface.

The OOP Python API is now authoritative.

Existing YAML infrastructure may:

* remain temporarily for backward compatibility;
* be converted internally into calls to the new `Circuit` API;
* eventually be deprecated if it adds no value.

Do not maintain two independent circuit-generation implementations.

If YAML remains supported:

```text
YAML
  ↓
adapter
  ↓
Circuit API
```

not:

```text
YAML compiler       Python Circuit builder
     ↓                     ↓
independent physics implementations
```

There must be one circuit-construction source of truth.

---

# 32. Preserve experiments separately

A design Python file describes the physical device.

Experiment configuration remains a separate concern.

Do not put into `Circuit`:

```text
pump power sweep
pump-frequency map
signal-frequency sweep
compression sweep
Newton tolerances
GMRES tolerances
continuation options
plotting configuration
```

The desired separation remains:

```text
Circuit
    what device exists?

Experiment
    how do we excite/measure it?

Solver
    how do we numerically solve it?
```

This refactor concerns only the first.

---

# 33. Test requirements

## Primitive tests

For every primitive:

```text
correct endpoints
correct type
correct value
correct hierarchy
correct flattened Element
correct matrix stamp
```

---

## Path tests

Verify:

```python
signal = c.path("signal")
```

followed by several blocks advances `signal.end` correctly.

Verify adding a second, third, etc. path does not affect existing path node semantics.

---

## Arbitrary graph test

Construct a branched circuit without Paths and verify compilation.

---

## JJ-line handle test

Verify:

```python
line.cell(205).left
line.cell(205).right
line.cell(205).Lj
line.cell(205).Cj
line.cell(205).Cg
```

all reference the expected generated objects.

---

## Targeted edit test

Construct a high-level line and add one capacitor to an internal node.

Verify only the requested local addition occurs.

---

## Coupler test

Construct:

```python
signal = c.path(...)
pump = c.path(...)
c.add_directional_coupler(signal, pump, ...)
```

and verify:

```text
both paths advance correctly
coupler connectivity is correct
handle terminals are correct
compiled circuit matches legacy coupler
```

---

## Profile tests

Verify:

```python
Linear(...)
```

matches the existing linear-profile engine.

Verify:

```python
HalfSine(...)
```

matches the existing:

```text
sin(pi*t/2)
```

behavior.

---

## Full IPM parity test

For the current trusted IPM/2C configuration compare old vs new:

```text
element count
element ordering
n1/n2
values
kinds
roles
JJ metadata
ports
C
G
K
Bphi
Ic vectors
```

Use strict/tight numerical tolerance.

This is the most important regression test.

---

# 34. Do not rewrite proven physics

Reuse current:

```text
JJ equivalent model
TL cell equations
coupler calculations
CPW calculations
loss implementation
matrix stamps
Element representation
```

wherever practical.

Preferred approach:

```text
existing tested function
      ↓
adapt/extract
      ↓
call through OOP builder
```

Avoid:

```text
working tested function
      ↓
rewrite from scratch because API changed
```

This is primarily an architecture/API refactor.

---

# 35. Public API naming

Final names must be chosen only after inspecting the fab/GDS repository.

Prefer the fab terminology unless:

* the geometry operation has no simulation equivalent;
* the name would be misleading in the circuit domain;
* existing simulator terminology is already canonical and widely used.

Document intentional deviations.

Produce a mapping such as:

```text
Fab/GDS API                  Simulation API
-----------                  --------------
add_X(...)                   add_X(...)
add_Y(...)                   add_Y(...)
route(...)                   path(...)
...
```

before finalizing the public interface.

---

# 36. One additional API principle: explicit is still allowed

Convenience APIs should never remove low-level control.

For example, this is ergonomic:

```python
c.add_jj_line(signal, ...)
```

but the user must always retain the ability to write:

```python
n1 = c.node(...)
n2 = c.node(...)

c.add_jj(n1, n2, ...)
c.add_capacitor(n1, n2, ...)
```

Likewise automatic path advancement is convenient, but arbitrary explicit connectivity must remain possible.

---

# 37. Acceptance criteria

Do not declare the task complete unless:

* [ ] `Circuit` is the primary design API.
* [ ] Internally it is modular, not one giant implementation file.
* [ ] The fundamental representation is an arbitrary node/element graph.
* [ ] `Node` is a first-class symbolic object.
* [ ] Ground is represented explicitly and compiles to node 0.
* [ ] Integer solver nodes are assigned only at compile time.
* [ ] Port numbers do not control node allocation.
* [ ] Arbitrary numbers of paths can be created.
* [ ] `Path` automatically advances when blocks are appended.
* [ ] Arbitrary topology can also be constructed without `Path`.
* [ ] Primitive builders exist.
* [ ] Cell builders use primitive builders.
* [ ] Line/block builders use lower-level builders.
* [ ] No composite block stamps matrices directly.
* [ ] High-level block methods return structured handles.
* [ ] Internal cells/elements can be accessed through those handles.
* [ ] Targeted local additions/modifications are possible.
* [ ] Directional couplers correctly operate on and advance two paths.
* [ ] Deterministic profile objects are supported directly in design code.
* [ ] `Linear` matches the old linear profile implementation.
* [ ] `HalfSine` matches the old `sin(pi*t/2)` implementation.
* [ ] A SPICE-like flattened netlist can be exported.
* [ ] Compilation remains deterministic.
* [ ] Existing matrix assembly remains authoritative.
* [ ] Existing IPM/2C circuit has numerical/topological parity with the trusted implementation.
* [ ] Existing solver tests continue to pass.
* [ ] The GDS repository is inspected before final API naming.
* [ ] A GDS-to-simulation API mapping document is produced.
* [ ] Geometry-only GDS parameters are not unnecessarily copied into the circuit API.
* [ ] Existing YAML support, if retained, becomes an adapter to `Circuit`, not an independent implementation.
* [ ] Experiments remain separate from design construction.

---

# 38. Required agent workflow

Before editing:

```powershell
git status --short
git diff --stat
python --version
python -m pytest -q
```

Then inspect:

```text
simulation repo
GDS/fab repo
existing design builders
existing Element representation
existing matrix assembler
existing IPM implementation
existing profile implementation
existing coupler implementation
```

Before implementing public block names, produce:

```text
GDS_API_MAPPING.md
```

containing:

```text
Fab concept
Fab function/class
Important fab parameters
Simulation equivalent
Proposed simulation name
Ignored geometry-only parameters
Electrical parameters required
Notes
```

Do not commit to the public API until this mapping is complete.

---

# 39. Deliverables

Expected output includes:

```text
new Circuit/graph/path infrastructure
primitive builders
cell builders
JJ/TL line builders
coupler builder
profile objects
compiler
validation
netlist export
GDS API mapping
migrated IPM/2C design
tests
documentation
```

Also provide a final report stating:

```text
files added
files modified
legacy code retained
legacy code deprecated
tests run
parity results
remaining migration work
intentional differences from GDS API
```

---

# 40. Architectural north star

The final product should allow a fabrication engineer to think:

```python
c = Circuit(...)

signal = c.path("signal")
pump = c.path("pump")

c.add_coupler(signal, pump, ...)
line = c.add_jj_line(signal, ...)

# special local design change
c.add_capacitor(
    line.cell(205).right,
    c.ground,
    C=...,
)

c.add_coupler(signal, pump, ...)
c.add_jj_line(signal, ...)
```

without ever thinking about:

```text
global node integer ranges
manual cursor offsets
matrix indices
Bphi dimensions
internal solver numbering
verbose YAML expansion
```

Yet the system must still be capable of flattening that design deterministically into an inspectable SPICE-like netlist and the exact `Element[]` representation consumed by the existing solver.

The desired abstraction is:

```text
FAB / GDS vocabulary
        ↓
Object-oriented Circuit API
        ↓
hierarchical symbolic circuit graph
        ↓
SPICE-like flattened netlist
        ↓
existing Element[]
        ↓
existing matrix assembler
        ↓
existing solver
```

New topology assembled from known components should require only ordinary Python design code.

A local special modification should require only one local builder call.

A deterministic spatially engineered device should require only profile objects.

Only genuinely new circuit physics should require modification below the builder layer.
