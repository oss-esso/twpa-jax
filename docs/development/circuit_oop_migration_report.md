# Circuit object-oriented migration report

## Scope

Phases 0–9 add a symbolic Python circuit authoring layer over the existing
solver element representation. The Python `Circuit` API is authoritative for
new electrical designs. YAML remains supported as an adapter for existing
designs and stored artifacts.

No solver physics was rewritten. `build_matrices`, the branch laws, the CPW
coupler implementation, and the existing profile shape mathematics remain the
authoritative implementations.

## Files added

- `src/twpa_solver/circuit/` symbolic graph, path, primitive, cell, line,
  coupler, architecture, profile, validation, compiler, and netlist modules.
- `designs/python/ipm_2c.py` migrated v1 IPM/2C source.
- `designs/python/ipm_v2.py` lumped-array v2 source.
- `designs/python/ipm_v3.py` explicit three-conductor v3 source.
- `tests/test_circuit_*.py` and `tests/test_circuit_ipm_variants.py` circuit
  and architecture gates.
- `tests/test_yaml_adapter_regression.py` and its Phase 0 snapshot.
- `docs/development/circuit_api.md`.
- `docs/development/circuit_oop_migration_report.md`.

## Files modified

- `src/twpa_solver/design/compiler.py`: YAML now calls public `Circuit`
  builders and compiles with legacy numbering for compatibility.
- `src/twpa_solver/circuit/__init__.py`: public exports, including `Path` and
  optional coupler leakage correction.
- `src/twpa_solver/circuit/primitives.py`: lumped junction-array builder.
- `src/twpa_solver/circuit/architectures/ipm.py`: explicit optional leakage
  correction.
- `docs/design_format.md`, `docs/development/GDS_API_MAPPING.md`, and
  `CLAUDE.md`.

Unrelated dirty-worktree files from concurrent work were not included in this
migration.

## Legacy code retained

- `builders/ipm.py` and `make_ipm` remain the parity oracle and matrix source.
- `builders/blocks.py` and `builders/registry.py` remain available for legacy
  callers. The YAML compiler no longer dispatches through them.
- Existing YAML designs remain supported through the adapter.

The old dictionary block implementation is therefore retained for compatibility
and rollback. Its deletion is intentionally separate from this migration.

## Architecture results

### v1

The Phase 6 parity gate compares element sequence, ports, matrices, `Ic`, and
`Lj` against both `make_ipm` and `designs/ipm_2c_fixed/*.npz`.

Result after later phases: `19 passed, 1 skipped`. The skip is the pre-existing
missing ideal-coupler reference design.

### v2

`IPMv2Config` describes `20 x 15 x 3 x 2 = 1800` effective junction branches.
`add_jj_array(count=3)` emits one branch per physical array and adds no nodes.
The tests compare node counts with a count-one control and verify the effective
inductance and capacitance scaling. This is a structural and numerical
lowering test only; it is not validation against measurement or GDS.

### v3

`IPMv3Config` describes `36 x 18 x 6 = 3888` junction branches and three
couplers. The couplers use the real three-conductor dimensions
`gaps=[5.5, 5, 5, 5.5]` and `widths=[9.186, 15, 9.186]` through
`ExplicitCouplerGeometry`.

The corrected conformal path evaluates all three conductors. The starting
dimensions report `coupling_db = -28.2496555093` and
`Z_eff = 49.3393827325 ohm`. The v3 builder ports the Prometheus bounded
least-squares optimization over the three-conductor dimensions and then emits
the optimized result through `ExplicitCouplerGeometry`. At 10 GHz, the
optimized v3 result is `coupling_db = -24.99999994` and
`Z_eff = 49.99999994 ohm` against the `-25 dB` target.

The implementation deliberately reproduces Prometheus' known
`beta_odd = 2*pi*frequency/v_even` expression rather than correcting it to
`v_odd`. The fab v3 coupler contains one straight section plus meanders. The
`2738.216` micrometre value is the fab starting unrolled centre-line length;
the optimized discrete model produces an electrical length of approximately
`2951.079` micrometres at 10 GHz. Neither the unrolled discrete model nor this
electrical length models bend discontinuities, tapers, or air bridges, and no
comparison to measurement or fab GDS is claimed.

### Authorized Phase 7 baseline regeneration

The corrected conformal/optimizer route changes only the two requested
auto-coupled snapshots. Before regeneration, both used the previous edge-CPW
resolution (`coupling_db` approximately `-14.000000` and
`Z_eff` approximately `50.00000048 ohm`). After regeneration:

| design | elements before -> after | element digest before -> after | coupling / `Z_eff` after |
| --- | --- | --- | --- |
| `ipm_3c` | 21,809 -> 21,629 | `c7885f0d54aabb2b72a99642ee9f156191783b30e5309296849e220bf943c31f` -> `e3d0abd12cd519ca2c0c5620bc59b6176801c9d06b018fa303174bee32437639` | `-14.0000005863 dB` / `50.0000001488 ohm` |
| `ipm_7c_ideal_node205` | 45,233 -> 44,813 | `20e8a861e7936d04abb17e3505b0f57f02b3e9dfb551de731b90ac44690c4910` -> `1379ae4d54310891db4ed3fa70a020e78d7d8c1a6c960dbb36e20f1d3ecc9084` | `-14.0000005863 dB` / `50.0000001488 ohm` |

The corresponding matrix digest transitions are recorded in
`tests/data/yaml_adapter_baseline.json`; `Ic` and `Lj` remain unchanged for
both designs. The other nine YAML designs remain byte-identical. The
`ipm_2c` YAML route remains `coupler_mode: cached` and is unchanged.

For auditability, the changed sparse-matrix digests were:

```text
ipm_3c
  Bphi 757d1cf7d4ccfba5ab5217b33a967b2165f0515ce5bdacfe705d07ef01336c27 -> c109c110ec536d72fbb37d3333a06abacf51286d8e60fb2985d34520c3ce6cc4
  C    d5999dd1787d56e0e862201efe12ebee42ebfac4fd670810d8ad9bf963ada0d1 -> 371fd963991c7bc579d95fb38e9a1b283dcf1be760568fed4a060471072cec04
  G    c52a76e8cd9073616ba7f32b72a436af18ca5e6c853f00c2d402236701a99aa5 -> 55bd7f0fce9626d8f4fa463c028eefbbd848c825fc8cf1fba93fe4b5b30f2274
  K    8aeffdec53a56dfff95112a03f9fd0ba7b653bc26c4529e30ab5d70d52b38483 -> bc4da78c605c319b4867bd212749251b702803a7443b9058b71d14eaa7a495b1
ipm_7c_ideal_node205
  Bphi 27158a21d40fac7110c1150b3eb1690ca79e4634411ca02f29d2050de284c5a2 -> 70214c3f94dba8d25b929c095a91f3282628d44f4940962d0d8081d219b77ed4
  C    6a56266e2c9959efabb6af4b17f99b89d7b95171c94bdc591cfd3d31baba7dcd -> 00d9ca045b9487156e57c7c262112e9704e96f7794c4df4a646824d3c2f5af6c
  G    085d8b22160385ddf4e364e2f7d03ca8e49464045f2083ddbd9e242670ae8369 -> 61e26b939d2bef395026ec62755e57e485f2858d5d308563919a74ffbbbbc12d
  K    c9eda0336ccac391f9f073e4bb1f43e03dbf8a2b3c606096879a789ab4d1df62 -> 05cba66a205d04ab2988084f773203b5f16be446d2ee0bd1213dad886e65c3ac
```

## Tests and mutation evidence

- Phase 8 architecture file: `11 passed`.
- Phase 9 documentation file: `4 passed` in `tests/test_circuit_phase9_docs.py`.
- Phase 7 YAML snapshot and determinism gate: `22 passed`.
- Phase 6 parity gate: `4 passed in 2.77 s`.
- Phase 7 targeted circuit/design regression: `98 passed`.
- Phase 5 directional-coupler file: `8` tests.

Parity and determinism gates were deliberately perturbed and shown failing
before restoration. Perturbations included changed `Cg`, reordered elements,
changed resistor values, wrong v2 array count, wrong v3 coupler placement,
optimizer use instead of explicit geometry, an altered leakage formula, and
alternating compile order.

For the current coupler work, restoring `y = 0.0` caused all three Fix A
conformal tests to fail: the v1/v3 values reverted to `-14.0018116609 dB` and
`-16.2760511037 dB`, and the v3 all-conductor root gate raised
`RuntimeError("CPW conformal-mapping root solve failed")`. Restoring
`y = gaps[0]` produced 3 passed tests. Setting the `ipm_2c` snapshot element
count to zero failed with `16312 != 0`; restoring it made the 22-test snapshot
file pass. Compiling the parity test with `node_numbering="creation"` failed
at the first element (`P1` versus legacy `P1_0`); restoring `"legacy"` made
the Phase 6 gate pass all 4 tests.

The required full slow-suite result was `873 passed, 5 failed, 1 skipped,
1 xfailed` in `500.43 s`.
The five failures are outside this migration: a missing
`designs/uniform_jtwpa.yaml`, two unrelated `run_kimpa_gain.py` failures, and
two unrelated loss-model convention failures.

## Intentional differences from the GDS API

- The circuit API represents electrical topology, not polygons, layers,
  coordinates, tapers, air bridges, or mask-processing parameters.
- `make*` geometry functions map to `add_*` electrical builders where an
  equivalent exists.
- v2 array internals are replaced by a documented lumped approximation.
- v3 retains the electrical three-conductor cross-section but does not model
  GDS routing or fabrication details.
- The circuit API uses SI units except for explicitly suffixed fabrication
  dimensions such as `_um`.

## Acceptance checklist

- [x] `Circuit` is the primary design API.
- [x] The implementation is modular.
- [x] The fundamental representation is an arbitrary node/element graph.
- [x] `Node` is a first-class symbolic object.
- [x] Ground is explicit and compiles to node 0.
- [x] Integer solver nodes are assigned at compile time.
- [x] Port numbers do not allocate nodes.
- [x] Multiple paths are supported.
- [x] Paths advance when blocks are appended.
- [x] Arbitrary topology can be built without paths.
- [x] Primitive builders exist.
- [x] Cell and line builders compose primitives.
- [x] Composite blocks do not stamp matrices directly.
- [x] High-level builders return structured handles.
- [x] Targeted local edits and removal are supported.
- [x] Directional couplers advance two paths.
- [x] Deterministic profile objects are supported.
- [x] `Linear`, `HalfSine`, and `Hann` preserve their specified formulas.
- [x] A flattened netlist can be exported.
- [x] Compilation is deterministic.
- [x] Existing matrix assembly remains authoritative.
- [x] IPM/2C parity is bit-identical.
- [ ] The existing full suite is completely green; five unrelated baseline
      failures remain documented above.
- [x] The GDS repository was inspected before final naming.
- [x] The GDS-to-simulation mapping is documented.
- [x] Geometry-only parameters are excluded from the circuit API.
- [x] YAML is an adapter to `Circuit`.
- [x] Experiments remain separate from design construction.

## Remaining work

The migration plan is complete. The documented full-suite failures require
separate maintenance work and are not part of the circuit migration. The
retained dictionary block layer may be removed in a future focused cleanup
after downstream legacy callers are audited.
