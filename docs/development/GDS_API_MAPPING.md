# GDS-to-simulation API mapping

Status: Phase 9 finalized mapping for the shipped object-oriented circuit API.

Source repository: `../Prometheus/Packages/`. The Prometheus code is a GDS
geometry generator. It does not provide the lumped-element netlist or matrix
model used by `twpa_solver`. The simulation API therefore mirrors physical
vocabulary and construction hierarchy where there is an electrical equivalent,
while dropping layout-only arguments.

## Mapping

| Fab concept | Fab function or class | Important fab parameters | Simulation equivalent | Proposed simulation name | Ignored geometry-only parameters | Electrical parameters required | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Device facade | `Device` (`Packages/Devices.py:39`) | device mixins, technology parameters | circuit facade | `Circuit(name)` | mask, layer, position, mask size, pad size | none | The simulation facade follows the same composition pattern without inheriting geometry mixins. |
| Route / polyline | `makeTurn` and NumPy path appends used by the package builders | centre point, radius, start/end angle | symbolic electrical path | `c.path("signal")` | all coordinates and bend geometry | none | `Path` tracks ordered symbolic nodes; it is not the fundamental graph representation. |
| Electrical port | port dictionaries returned by `makeCoupler` (`Packages/DirectionalCouplers.py:260`) | coordinate, line width, gap, taper spacing | node-attached port | `c.add_port(node, number=, impedance=)` | coordinates, width, gap, taper length, minimum spacing | impedance, port number | Port numbers label external interfaces and never allocate solver nodes. |
| Launcher and pad | `makeLauncher` and `standardPADPosition` used by `makeWGthru` | pad size, pad length, direction, taper | port termination | absorbed into `add_port` | all pad and taper geometry | `Z0` or impedance | A launcher has no separate lumped element in the current solver model. |
| Josephson junction | `makeETHlargeJJ` / `makeJunction` (`Packages/JJs.py`) | junction dimensions, resist height, evaporation angle | junction inductance plus junction capacitance | `c.add_jj(n1, n2, Lj=, Cj=)` | junction polygon dimensions, resist, patch, undercut, evaporation angles | `Lj`, `Cj` or `Ic` | The existing `Element` representation and branch laws remain authoritative. |
| Junction-array cell | `makeETHlargeJJArray` (`Packages/JJs.py`) | junction count, spacing, process geometry | lumped junction array | `c.add_jj_array(..., count=)` | spacing and fabrication process parameters | `Lj`, `Cj`, count | Phase 8 uses the documented effective-junction approximation; it adds no nodes. |
| JJ/IDF unit cell | `cellRect1` and `makeETHlargeJJ` (`Packages/JTWPAs.py:1092`, `:1157`) | cell length, cell width, IDF width/gap/length | JJ unit cell | `c.add_jj_cell(...)` | placement and polygon dimensions | `Lj`, `Cj`, `Cg` | The cell is composed from primitive builders. |
| Half-ground-capacitance end cell | `cellRect2` (`Packages/JTWPAs.py:1099`) | end-cell geometry | internal boundary rule in JJ line | internal to `c.add_jj_line(...)` | all layout geometry | `Cg / 2` at the boundary | This reproduces the existing first-cell and terminator convention. |
| Junction transmission line | `edgeLcell`, `cellLine`, `edgeRcell` in `makeIFMJTWPA` (`Packages/JTWPAs.py:429`) | micro-cell count, macro-cell count, placement | repeated JJ line | `c.add_jj_line(path, cells=, Lj=, Cj=, Cg=)` | placement coordinates and routing geometry | `cells`, `Lj`, `Cj`, `Cg` | Profiles are supplied as Python objects in the design API. |
| Fab ETH JTWPA | `makeEthTWPA` (`Packages/JTWPAs.py:28`) | IDF, cell counts, coupler dimensions, chip type | composition of transmission lines and couplers | explicit `Circuit` composition | chip layout, air bridges, launcher placement | electrical line, junction, and coupler parameters | No separate architecture builder is introduced until a reusable electrical structure is identified. |
| Fab v1 JTWPA | `makeIFMJTWPA` (`Packages/JTWPAs.py:429`) | `CellNums`, IDF, coupler parameters, JJ style | composition of lines and couplers | `c.add_ipm_section(...)` or explicit composition | chip layout, air bridges, launcher placement | row counts, cell counts, junction and coupler parameters | The IPM v1 parity design follows this construction family. |
| Fab v2 JTWPA | `makeIFMJTWPA(..., JJArray=True)` (`Packages/JTWPAs.py:429`) | `JJArray`, `JJArrayNumber`, `CellNums` | v1 topology with lumped arrays | `c.add_jj_array(..., count=3)` | array spacing and process geometry | effective `Lj`, `Cj`, count | The effective model is an approximation, not a claim of geometric equivalence. |
| Fab v3 JTWPA | `makeIFMJTWPA_` (`Packages/JTWPAs.py:944`) | `CellNums`, `CouplerType`, `AirBridgeFab`, IDF | optimized explicit three-conductor coupler plus JJ lines | `c.add_directional_coupler(..., geometry=...)` | turns, orientation, air bridges, tapers, routing coordinates | starting gaps, widths, unrolled length, electrical line values | Prometheus dimensions are fitted through the conformal model; the discrete builder does not model meander bends. |
| Two-line directional coupler | `makeEdgeCoupledDirectionalCoupler` (`Packages/DirectionalCouplers.py:25`) | coupling dB, central frequency, `Z0`, width/gap, optimization bounds | discrete edge-coupled electrical block | `c.add_directional_coupler(signal, pump, coupling_db=, frequency=)` | position, bridges, taper geometry, optimization-only bounds | coupling dB, frequency, `Z0` | The builder calls the existing coupler and discrete-parameter functions. |
| Straight directional coupler geometry | `makeDirectionalCoupler` (`Packages/DirectionalCouplers.py:87`) | coupler width, length, gap, pitch, coupling, spacing | two-path coupler cells | internal to `add_directional_coupler` | polygon and bridge geometry | discrete coupler parameters | Geometry is not exported as a separate circuit object. |
| Meander grounded coupler | `makeMeanderGroundedDirectionalCoupler` (`Packages/DirectionalCouplers.py:112`) | offsets, width, gap, ground width, length, straight length | not represented as a distinct current electrical model | not exposed | all route and air-bridge parameters | none beyond a validated equivalent coupler, if later added | No matching lumped builder exists in the current simulator. |
| Grounded directional coupler | `makeGroundedDirectionalCoupler` (`Packages/DirectionalCouplers.py:197`) | pump offset, width, gap, ground width, length, straight length | not represented as a distinct current electrical model | not exposed | all route and air-bridge parameters | none beyond a validated equivalent coupler, if later added | Retained in the explicit out-of-scope inventory. |
| Three-conductor coupler | `makeCoupler` (`Packages/DirectionalCouplers.py:260`) | `gaps[4]`, `widths[3]`, length, type, orientation | conformal three-line coupler | `ExplicitCouplerGeometry(gaps_um=, widths_um=, length_um=)` | turns, delta-y, air bridges, tapers, orientation | gaps, widths, length | The centre conductor is reduced by the existing CPW model; matrix physics is not rewritten. |
| Composite waveguide | `makeCompositeWaveguide` (`Packages/Waveguides.py:133`) | impedances, frequency, electrical lengths, turns, `Zmethod` | repeated transmission-line block | `c.add_transmission_line(...)` | meander turns, pitch, layout width, bridge style | `Z`, length, frequency or cell parameters | The simulation builder accepts equivalent electrical values rather than geometry. |
| Waveguide meander | `makeWGMeander` (`Packages/Waveguides.py:27`) | straight end, bridge style, bridge distance | not represented separately | not exposed | all route and bridge geometry | none | Use `add_transmission_line` when an electrical equivalent is known. |
| Waveguide through and launcher route | `makeWGthru` (`Packages/Waveguides.py:79`) | pads, direction, bridge style, gap, pad dimensions | ports and optional line termination | absorbed into `add_port` | all pad, taper, route, and bridge geometry | impedance | No geometry-only route object is required by the circuit solver. |
| PAD test structure | `makePADtest` (`Packages/Waveguides.py:246`) | pad gap, pad dimensions, straight length, bridge style | not represented | not exposed | all geometry | none | This is a fabrication test structure, not a circuit topology. |
| Hanged resonator | `makeHangedResonatorCell` and related methods (`Packages/Resonators.py`) | resonant frequency, impedance, coupling distance, turns | future resonator block | future `c.add_resonator(...)` | layout turns, pitch, coupling placement | frequency, impedance, coupling capacitance | Deliberately outside the current migration scope because no equivalent builder is present. |
| Hann modulation | `IDFHannModulation` (`Packages/KITWPAs.py:338`) | modulation start/stop and cell selection | deterministic profile | `Hann(start=, stop=, ...)` | layout-specific IDF dimensions | profile start, stop, domain | `Hann` is the fabrication half-cosine profile. It is distinct from `HalfSine`. |
| Sinusoidal loading | `makeSinusoidalTWPA` (`Packages/KITWPAs.py`) | cell count and modulation parameters | deterministic profile | `Sine(start=, stop=, ...)` | placement and geometry | profile start, stop, domain | Uses the existing profile evaluation mathematics. |
| Periodic loading | `makeLoadedTWPA` (`Packages/KITWPAs.py`) | loaded/unloaded counts and IDF load geometry | future periodic profile/block | future `Periodic(...)` | IDF dimensions and layout | per-cell electrical loading | Not required by the current IPM migration. |
| Interdigitated capacitor | `makeIDC` / `capacitanceIDC` (`Packages/IDC.py`) | finger geometry and spacing | lumped capacitor primitive | `c.add_capacitor(n1, n2, C=)` | all finger geometry | capacitance | Geometry-to-capacitance conversion is not duplicated in the circuit layer. |
| Air bridges | `makeETHAirBridges` and related methods (`Packages/AirBridges.py`) | bridge geometry, layer, spacing | not represented | not exposed | all parameters | none | Layout-only. |

## Shipped simulation names

The electrical names listed in the mapping are implemented in the repository:

| Electrical responsibility | Shipped implementation |
| --- | --- |
| Circuit facade and graph | `twpa_solver.circuit.Circuit` |
| Symbolic paths | `Circuit.path` and `twpa_solver.circuit.Path` |
| Primitive elements | `add_capacitor`, `add_inductor`, `add_jj`, `add_jj_array`, `add_resistor`, `add_port` |
| Repeated lines | `add_jj_line`, `add_transmission_line`, `add_rf_squid_line` |
| Directional coupler | `add_directional_coupler` and `ExplicitCouplerGeometry` |
| IPM composition | `add_ipm_section` and explicit v1/v2/v3 Python designs |
| Deterministic profiles | `Linear`, `HalfSine`, `Hann`, and the other `circuit.profiles` classes |
| YAML compatibility | `twpa_solver.design.compile_design` as an adapter to `Circuit` |

The v2 and v3 sources are [designs/python/ipm_v2.py](../../designs/python/ipm_v2.py)
and [designs/python/ipm_v3.py](../../designs/python/ipm_v3.py). Their tests verify
electrical structure and lowering only. They are not validated against
measurement or against the fabrication GDS.

## Intentional naming and interface deviations

- Simulation builders use the `add_` prefix. In Prometheus, `make*` means
  emitting geometry; in the simulation API, `add_*` attaches an electrical
  block to a circuit or path.
- Public names use `snake_case` and retain physical nouns: `cell_len`,
  `cell_width`, `idf_width`, `micro_cell`, `macro_cell`, `cell_lines`,
  `coupling_db`, and `coupler_length`.
- Simulation values use SI units. Arguments that retain fabrication micrometre
  values use an explicit `_um` suffix.
- `signal_in`, `signal_out`, `pump_in`, and `pump_out` are electrical terminal
  names required by the two-path handle. `port(1)` through `port(4)` remain
  aliases matching the Prometheus port direction convention.
- `HalfSine` means `sin(pi*t/2)`. `Hann` means the fabrication
  `(1 - cos(pi*t))/2` profile. Both are exposed and are not interchangeable.
- `Lk_sq` is not part of the circuit API. Designs supply the electrical `Lk`
  value and perform any required multiplication explicitly; the fabrication
  JTWPA signatures carrying `Lk_sq` do not make it an electrical requirement
  for the current model.
- Geometry-only parameters are intentionally not copied into the circuit API.
  They remain relevant to Prometheus layout generation but have no effect on
  the current equivalent-circuit matrices.
