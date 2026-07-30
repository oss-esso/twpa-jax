# Implementation Plan: Le Gal 2025 Compression Benchmark — Correctness Fixes

Status: authored 2026-07-30 after review of the Phase 4-6 worktree.
Audience: the implementation agent. Every number below was measured on this
worktree, not estimated. Reproduce each one before and after your change.

## Goal

Make the effective-SNAIL line and its harmonic-balance readout physically
correct, so that a negative or positive benchmark result against the paper's
*simulated* curves is a statement about the solver rather than about four
independent normalization and stamping defects.

## Current State Analysis

`references/le_gal_2025_gain_compression/phase6_selected_observables.json`
contains 60 converged points with `hb_gain_db` in `[-19.181, -0.658]` — no
amplification anywhere. The prior agent reported this as an honest negative
result. It is not usable as one: four defects each independently suppress or
misreport gain.

Measured evidence (probe scripts under the session scratchpad, reproduce them):

```
# builders/le_gal_2025.py, cells=8
I(delta=0)      = -1.212435565298214e-06     <- must be 0
I(Phi*)         =  1.0629934216599025e-23
dI/dPhi at 0    -> L = 1.0280278375737243e-09
dI/dPhi at Phi* -> L = 8.663723545154484e-10  (paper 869.6e-12)
tangent(0)/tangent(Phi*) = 0.842751842751843
C diagonal = 2.545e-13 = Cg + Csnail
sqrt(L/Cdiag) = 58.34563885788355 ohm   vs paper sqrt(L/Cg) = 62.37649990575112 ohm
```

```
# scripts/run_le_gal_2025_hb.py, cells=20, fs=5.0 GHz, Ps=-115 dBm
driver hb_gain_db            = -17.49905712541861   (artifact row 0: -17.498733)
tone_s21 |S21| dB (pump on)  =  -5.457857298859363
tone_s21 |S21| dB (pump off) =  -5.693096890861156
gain_vs_off_db               =   0.23523959200179317
driver - tone_s21(on) [dB]   = -12.041199826559247
```

Baseline suite: `4 failed, 258 passed, 3 skipped`. The 4 failures are
pre-existing (`test_column_matrices_tracer` x2, `test_loss_model` x2, the
latter from `docs/loss_A10.csv not found`). **This count must be exactly 4 at
every phase boundary.**

## What We're NOT Doing

- Not touching `port_s_from_unit_current_response`, `port_waves`, or the
  Floquet path. Those are correct; only callers were wrong.
- Not changing the pump solve path. `FullPumpProblem` iterates must stay
  byte-identical for the existing designs (jpa/jtwpa/fqjtwpa/2c).
- Not digitizing paper figures in this plan. `digitized_full_theory_*.csv`
  stays empty until a separate, provenance-tracked task.
- Not chasing experimental traces. Target is the paper's simulated curves.
- Not running the 700-cell full-morphology (Level 3) sweep. Memory is
  constrained on this machine; Level 3 is a later, separately budgeted task.
- Not re-running exp20/21/22.

## Prerequisites

- [ ] `git stash list` clean; record `git status --short` before starting.
- [ ] Baseline suite reproduced at exactly 4 failures.
- [ ] Delete stray `phase6_observables.json` from the repo root and add
      `references/**/__pycache__/` to `.gitignore`.

---

## Phase 1: Device physics — make the line the paper's line

Nothing downstream is meaningful until the circuit is right. Do this first and
commit it alone.

### 1.1 Shift the SNAIL branch law to its static equilibrium

**File**: `src/twpa_solver/core/nonlinear.py`

`EffectiveSnailBranchLaw` is evaluated by the solver at the *dynamic* branch
flux `deltaPhi`, but is written as `I(Phi)`. The builder computes the linear
inductance from `tangent(Phi*)` while the solver sees `tangent(0)` — a 15.7%
disagreement — and every branch carries a spurious static `-0.866*Ic`.

Add an `equilibrium_flux: np.ndarray` field and evaluate at
`flux + equilibrium_flux[None, :]` in **both** `current` and `tangent`
(`gamma` delegates to `tangent`, so it follows). Keep the dataclass frozen and
keep the existing broadcast shape `(n_time, n_branch)`.

Analytic check for the default `Phi_ext = Phi0/2` case, `Phi* = pi*phi0`:

```
I(delta) = Ic * ( -r sin(delta/phi0) + sin(delta/(3 phi0)) )
```

The ratio term **sign-flips**. This is an odd function, so the device is pure
four-wave mixing at half flux — consistent with the `2*theta_p - theta_s -
theta_i` mismatch already used in `observables.spatial_profiles` and in the
CME oracle. Do not "fix" that mismatch convention.

**Report, do not gate on**: the cubic Taylor coefficient of the shifted law is
`Ic * (r/6 - 1/162) = +0.004160 * Ic` — *positive*, i.e. opposite in sign to a
bare Josephson junction's `-1/6`. State this number in the phase report. It
determines the sign of self-phase modulation and therefore whether the device
self-phase-matches. If it disagrees with the paper's stated nonlinearity sign,
stop and say so rather than proceeding.

### 1.2 Solve for the equilibrium instead of hardcoding it

**File**: `src/twpa_solver/builders/le_gal_2025.py`

`equilibrium = np.full((1, n), math.pi * PHI0_REDUCED)` is only correct at
`flux_over_flux0 = 0.5`. Solve `I(Phi*) = 0` numerically (`scipy.optimize.brentq`
on a bracket around `phi_ext`, or Newton seeded at `phi_ext`), assert the
residual is below `1e-18 * critical_current_a`, and assert the solution is a
*stable* branch (`tangent(Phi*) > 0`). Store `Phi*` in `metadata` and pass it
into the branch law.

At half flux the solved value must reproduce `pi * PHI0_REDUCED` to `1e-12`
relative — that is your regression tie to the current behaviour.

### 1.3 Stamp the SNAIL capacitance across the branch, not to ground

**File**: `src/twpa_solver/builders/le_gal_2025.py:43-44`

Current code:

```python
c = sp.eye(n, format="csr") * ground_capacitance_f
c = c + sp.eye(n, format="csr") * snail_capacitance_f
```

`C_snail = 31 fF` is the capacitance *across* each SNAIL, in parallel with the
SNAIL inductance, i.e. on the same branch as `Bphi`. It is not a shunt to
ground. Replace with:

```python
c = (bphi @ sp.diags(np.full(n, snail_capacitance_f)) @ bphi.T).tocsr()
c = c + sp.eye(n, format="csr") * ground_capacitance_f
```

Verify `sqrt(L / Cg) == 62.3765 ohm` to 4 significant figures against the
`port_impedance_ohm` default of 62.4 — the two must agree, and today they do
not (58.346 vs 62.4).

### 1.4 Make `loss_tangent` dimensionally meaningful

**File**: `src/twpa_solver/builders/le_gal_2025.py:46`

`g = sp.eye(n) * loss_tangent` stamps a dimensionless loss tangent as a
conductance in siemens: `tan(delta) = 2.19e-3` becomes a 457 ohm shunt on
every node. The default of `0.0` hides it, but Phase 5's staged-loss work
depends on this parameter.

A loss tangent on a capacitor is frequency-dependent (`G = omega * C * tan(delta)`),
which a static `G` matrix cannot express. Pick one and say which in the
docstring:

- **Preferred**: rename the parameter to `shunt_conductance_s` and require the
  caller to supply siemens, so the name matches the stamp. Add a module-level
  helper `conductance_from_loss_tangent(tan_delta, capacitance_f, omega)` for
  callers that want to evaluate at the pump frequency.
- Or: keep the name and stamp `omega_ref * C * tan(delta)`, taking a mandatory
  `omega_ref` argument, and document that loss is pinned at that reference.

Do not leave the current form.

### 1.5 Verify the port/branch topology

**File**: `src/twpa_solver/builders/le_gal_2025.py:37-42`

`bphi[0,0] = 1.0` makes branch 0 run from node 0 to ground, and there are `n`
branches over `n` nodes with `port_to_index = {1: 0, 2: n-1}`. Confirm against
the intent: a 700-cell line should present a SNAIL *between* consecutive
nodes, with the input port at node 0 and the output at node `n-1`. State in
the phase report whether branch 0's ground connection is deliberate (a
terminating half-cell) or an off-by-one. If it is an off-by-one, `n` cells
require `n+1` nodes.

### Phase 1 Success Criteria

**Automated** — new tests in `tests/test_nonlinear_branch.py` and
`tests/test_le_gal_2025_builder.py`. Each must be shown *failing* on the
pre-fix code before you claim it:

- `law.current(zeros) == 0` to `1e-20 * Ic` — currently `-1.212e-06`.
- `1.0 / law.tangent(zeros)` matches `metadata["linear_inductance_h"]` to
  `1e-12` relative — currently off by a factor `0.8428`.
- `1.0 / law.tangent(zeros)` is within 1% of the published `869.6e-12`.
- Solved `Phi*` at half flux equals `pi * PHI0_REDUCED` to `1e-12` relative.
- `tangent(Phi*) > 0` (stable branch).
- `sqrt(L_mean / ground_capacitance_f)` equals `port_impedance_ohm` to 0.5%.
- The shifted law is odd at half flux: `current(-x) == -current(x)` to `1e-18`.
- Existing `test_effective_snail_line_contract` and the save/load round-trip
  still pass, extended to round-trip the equilibrium flux.

**Manual**: report the measured cubic coefficient and its sign (1.1).

**Gate**: full suite still at exactly 4 failures.

---

## Phase 2: Power convention — the Level 1 check that was never written

This is the cheapest guard against the largest error and must land before any
campaign is re-run.

### 2.1 Create `tests/physics/`

New package directory with `__init__.py` (or rely on rootdir conftest, matching
whatever `tests/` already does — check before adding).

### 2.2 `tests/physics/test_compression_power_convention.py`

Build the effective-SNAIL line with the nonlinearity effectively switched off
(`critical_current_a` scaled so the branch is linear over the drive range, or a
dedicated linear line with the same `L`, `C`, `Cg`, `Z0`), matched at both
ports. Inject a signal at a requested dBm and assert:

- `tone_s21` magnitude is `0 dB` within `0.05 dB` for a matched lossless line
  at a frequency well inside the band.
- The incident power recovered from `extract_port_waves`'s `a_power` at the
  driven port equals the requested dBm to `0.01 dB` — this pins the
  peak-amplitude convention (`I_peak = sqrt(2 P / Z0)`, per
  `CLAUDE.md`'s pump-current section) end to end.
- `a_power == 0` at the undriven port and `b_power == 0` is *not* trivially
  true — the previous `current = voltage / z0_ohm` bug made `b` identically
  zero everywhere. Assert `b_power > 0` at the output port.

State the failing magnitude on the pre-fix code. On today's `run_le_gal_2025_hb`
normalization this check fails by **12.041 dB**.

### 2.3 `tests/physics/test_compression_low_signal_limit.py`

On `build_jpa()` at a pump/detuning point that actually produces gain, assert
the multitone finite-signal `gain_vs_off_db` converges to the Floquet
small-signal gain within `0.05 dB` as signal power goes to zero. If
`build_jpa()` cannot be driven above 3 dB `gain_vs_off_db`, say so explicitly
with the pump scales and detunings tried, and mark the test `xfail` with that
reason rather than shipping a `0 dB` vs `0 dB` comparison.

### 2.4 `tests/physics/test_compression_pump_depletion_limit.py`

Assert the paper's closed-form depletion model
`G = G_lin / (1 + 2 G_lin P_s / P_p)` gives
`P_1dB = P_p + 10*log10((10^0.1 - 1) / (2 G_lin))`, and that for
`G_lin = 20 dB`, `P_p = -78.4 dBm` this lands at `-107.3 dBm` within
`0.2 dB`. (`tests/test_le_gal_2025_cme.py` already has this; move or
re-export it here so the physics suite is self-contained, do not duplicate the
assertion in two places.)

### Phase 2 Success Criteria

**Automated**: the three files above pass post-fix. Show 2.2 failing by
12.041 dB pre-fix.

**Gate**: full suite at exactly 4 failures plus the new passes.

---

## Phase 3: Readout — stop hand-rolling normalization

### 3.1 Replace the driver's gain formula

**File**: `scripts/run_le_gal_2025_hb.py:76-89`

```python
output = abs(1j * basis.omegas[signal_row] * state[signal_row, output_node])
input_voltage = math.sqrt(2.0 * 10.0 ** ((power_dbm - 30.0) / 10.0) * 62.4)
"hb_gain_db": float(20.0 * np.log10(max(output / input_voltage, 1e-300))),
```

This drops `REAL_RECONSTRUCTION_FACTOR = 2` (the multitone convention is
`x(t) = 2 Re sum X_v e^{i theta_v}`, see `multitone/basis.py`) and treats
`sqrt(2 P Z0)` as the incident wave when it is `signal_current * Z0`. Together
that is a factor of 4 in amplitude, **-12.041 dB**, measured.

`observables.tone_s21` already does this correctly. Delete the hand-rolled
path and call it:

```python
s21 = tone_s21(state, basis, circuit, signal_tone=basis.signal_tone,
               source_port=1, out_port=2, source_current_a=signal_current,
               z0_ohm=Z0)
```

Do the same for the pump tone. Do not reimplement any part of
`port_s_from_unit_current_response`.

### 3.2 Report `gain_vs_off_db`, not absolute `|S21|`

Absolute `|S21|` folds in the line's insertion loss and port mismatch. Per
`CLAUDE.md`, the validated normalization is the pump-on / pump-off ratio of
voltage-per-input-current. Solve the extra pump-off state per point and emit
`gain_vs_off_db` alongside `s21_db`, and drive `compression_db`, the P1dB
crossing, and the reference gain from `gain_vs_off_db`.

Measured at the reference point (20 cells, 5.0 GHz, -115 dBm):
`|S21|` pump-on `-5.4579 dB`, pump-off `-5.6931 dB`, `gain_vs_off = +0.2352 dB`.

### 3.3 Fix or delete the fake comparison fields

**File**: `scripts/run_le_gal_2025_hb.py:92-93`

- `cme_endpoint_gain_dB` is the literal constant `0.172571879516` in all 60
  rows — it hardcodes `(1.0, 1e-3, 0.0)` and `coupling=0.2` regardless of
  cells, frequency, or power. Either derive the CME initial conditions and
  coefficients from the point's own parameters (see Phase 4) or remove the
  field. A constant column presented as a cross-check is worse than no column.
- `depletion_model_gain_dB` hardcodes `G_lin = 100.0` rather than the measured
  gain, so it varies only with signal power. Feed it the point's own
  `gain_vs_off_db` converted to linear power.

### 3.4 Clean up the driver

- Line 63: `circuit.port_to_index[0] if 0 in circuit.port_to_index else 0` —
  port `0` never exists; use `circuit.port_to_index[1]`.
- Lines 111-112: dead `for key in ("cells", "signal_GHz"): _ = key`. Delete.
- `_pump_current()` and the per-point signal current both inline
  `sqrt(2 * 10**((dBm-30)/10) / 62.4)` with a magic `62.4`. Take `Z0` from the
  circuit's `port_impedance_ohm` metadata and use one helper.
- `compression_db = reference_gain - hb_gain_db` uses the *lowest-power point*
  as the reference. Keep that, but assert the reference point is genuinely in
  the small-signal plateau (its gain must be within `0.05 dB` of the next
  power point); otherwise emit `status = "NO_PLATEAU"` for the group rather
  than a P1dB.

### 3.5 Plumb `z0_ohm` through `power_balance`

**File**: `src/twpa_solver/multitone/observables.py:158-173`

`power_balance` calls `extract_port_waves` with the default `z0_ohm = 50.0`
and exposes no parameter of its own. Every external power-wave and
Manley-Rowe quantity is therefore computed at 50 ohm even on the 62.4 ohm
line. Add `z0_ohm: float = 50.0` to `power_balance` and forward it; update
`scripts/run_compression.py` to pass `args.z0_ohm`.

While there: `extract_port_waves` is called twice per `power_balance` (once
for the state, once for the reference), each doing a full synthesize/project.
Hoist `_port_current_coefficients` if it is cheap to do so; do not restructure
otherwise.

### Phase 3 Success Criteria

**Automated**:
- New test asserting the driver's reported gain for one 20-cell point equals
  `tone_s21` to `1e-9 dB`. Show it failing by `12.041 dB` pre-fix.
- `power_balance(..., z0_ohm=62.4)` differs from the 50 ohm result on the
  effective-SNAIL line — a test that pins the parameter is actually wired.

**Manual**: no constant-valued columns remain in the emitted JSON. Verify with
`set(round(row[k], 12) for row in rows)` having more than one element for
every numeric field that claims to be point-dependent.

**Gate**: full suite at exactly 4 failures plus new passes.

---

## Phase 4: Restore the Manley-Rowe gate and make the CME a real oracle

### 4.1 Restore a lossless Manley-Rowe gate that can fail on physics

**File**: `tests/test_power_balance.py:61-68`

`test_power_balance_lossless_nonzero_state_and_manley_rowe` previously asserted
`result["manley_rowe_rel_err"] < 1e-12`. That assertion was **deleted** and
replaced with `assert result["external_manley_rowe_evaluable"] == 0.0` — the
test now asserts the invariant is unmeasurable, while its name still promises
it is checked. The plan item "Fix finite-signal lossless Manley-Rowe" is not
discharged by this.

Build a fixture where the invariant is genuinely evaluable: a lossless
two-node circuit driven with a pump *and* a finite signal strong enough that
`external_manley_rowe_photon_scale` clears the `1e-30` floor by several orders
of magnitude, then assert `external_manley_rowe_rel_err < 1e-10`. If no such
fixture can be constructed on the current formulation, that is a finding about
the formulation — report it with the achieved `photon_scale` values rather
than weakening the assertion.

Also rename or split the test so its name matches what it asserts.

### 4.2 Harden the harmonic-exclusion test

**File**: `tests/test_power_balance.py:80-105`

`test_manley_rowe_ignores_pump_harmonic_conversion` injects `1e-20 W` at
`(3, 0)`. Divided by `omega = 6e10`, that is `1.67e-31`, *below* the `1e-30`
evaluability floor — so the `evaluable == 0.0` assertion survives the mutation
it is meant to catch (broadening `conversion_tones` to all tones). Only the
`scale == 0.0` assertion bites. Raise the injected power so both assertions
are load-bearing, and verify by mutation: broaden `conversion_tones` in
`observables.py`, confirm the test fails, revert.

### 4.3 Document the three-tone restriction honestly

**File**: `src/twpa_solver/multitone/observables.py:215-227`

`external_manley_rowe_rel_err` sums only `(pump_tone, signal_tone,
idler_tone)`. That is a defensible modelling choice for the four-wave
conversion channel, but it is blind to photon leakage into any other retained
tone — which is exactly the truncation error the production-basis
self-convergence work is trying to bound. Rename the key to
`three_wave_manley_rowe_rel_err` (or `conversion_manley_rowe_rel_err`),
document it as a channel-restricted gate, and emit the all-tone quantity
alongside it so the difference between them is visible in the CSV. Update
`scripts/run_compression.py`'s field list and any consumer in
`experiments/exp22_spatial_attribution.py`.

### 4.4 Make the CME oracle the paper's CME

**File**: `references/le_gal_2025_gain_compression/cme.py`

The current module is a generic normalized three-wave integrator:
`coupling` is a bare scalar, `phase_mismatch` is a constant, and there are no
self-phase or cross-phase modulation terms, no `omega`, no `Z0`, and no
physical units. It cannot produce power-dependent phase matching, and
therefore cannot reproduce the two gain lobes, the offset between the
pump-depletion minima and the gain maxima, or the non-trivial `P1dB(f_s)`
shape — which are the five discriminating signatures the benchmark exists to
test. As an oracle for the paper it is not yet fit for purpose.

Extend it to the paper's Appendix C form, keeping the module free of any
`twpa_solver` import:

- Envelopes `A_p(x)`, `A_s(x)`, `A_i(x)` in physical units.
- Retain pump depletion, SPM on all three modes, all XPM cross terms,
  signal-idler conversion, and distributed loss per mode.
- Input convention `A_j(0) = sqrt(P_j * Z0) / omega_j`, `A_i(0) = 0`, with
  `Z0 = sqrt(L / Cg) = 62.3765 ohm` from the published `L` and `Cg`.
- Coefficients derived from the published device parameters, recorded in
  `parameters.json` under `inferred` with the derivation stated.

Keep `depletion_only_gain` as-is; it already matches the paper's
`G_lin / (1 + 2 G_lin P_s / P_p)`.

**Success criteria**: the extended CME must, with the published parameters,
produce a two-lobed `G(f_s)` at low signal power with the lobes on either side
of 7.5 GHz. If it does not, stop and report — that is a statement about the
coefficient derivation, and it must be resolved before any HB comparison is
meaningful.

### Phase 4 Success Criteria

**Automated**:
- Restored lossless Manley-Rowe gate passes with an evaluable photon scale;
  show the scale value.
- 4.2 mutation demonstrated: test fails when `conversion_tones` is broadened.
- Existing CME conservation and convergence tests still pass against the
  extended equations (photon invariant must still be conserved to `1e-7`).
- New test: CME with published parameters yields lobe peaks either side of the
  pump, reported with their frequencies.

**Gate**: full suite at exactly 4 failures plus new passes.

---

## Phase 5: Re-run the selected (Level 2) campaign

Only after Phases 1-4 are green. **Do not run Level 3.** Memory is
constrained on this machine — a previous full-matrix attempt OOMed.

### 5.1 Void the existing artifact

`references/le_gal_2025_gain_compression/phase6_selected_observables.json`
carries a 12.041 dB normalization bias on a device with the wrong impedance
and a mis-shifted branch law. Move it to
`phase6_selected_observables.void.json` with a header note stating why, or
delete it. It must not be cited as a negative result.

### 5.2 Run Level 2

Frequencies `5.0, 6.0, 8.0, 9.5 GHz`; powers `-115, -110, -105, -100, -94 dBm`.
Cells `20, 50, 700`, run in that order and **stop and report if 700 does not
fit in memory** rather than reducing scope silently.

Emit per point: `gain_vs_off_db`, `s21_db`, `pump_depletion_db`,
`compression_db`, `p1db_dBm`, `hb_residual_rel`, the restored Manley-Rowe
fields, and the CME comparison at the *same* parameters.

### 5.3 Check the Level 2 criteria from the benchmark spec

- Finite-signal gain approaches the Floquet/small-signal result at low `P_s`.
- Gain has a stable low-power plateau (Phase 3.4's plateau assertion).
- Compression begins smoothly; pump transmission falls as gain compresses.
- Basis self-convergence: `|P1dB(S + dS) - P1dB(S)| < 0.25 dB`. Run at least
  one frequency at two sideband counts. If it does not converge, report the
  delta; do not raise `S` until it does and then claim convergence.

### 5.4 Staged loss

Run version **A** (lossless, matched) only in this phase. Versions B
(constant loss) and C (power-dependent `tan(delta)` digitized from Fig. 6(d))
are out of scope until 1.4 lands a defensible loss stamp and the digitization
task has run.

### Phase 5 Success Criteria

**Automated**: a reduced (20-cell, two-frequency) campaign runs inside the
ordinary test suite as a regression fixture with stored reference metrics.

**Manual**: report, with numbers, whether the device now reaches an
amplifying regime. If low-power `gain_vs_off_db` is still at or below 0 dB
after all four fixes, that is a real finding — state it with the measured
values at every frequency and cell count, and do not tune parameters to force
gain.

---

## Phase 6: Documentation

Update `CLAUDE.md` with:

- The four defects and their measured magnitudes, so they are not reintroduced.
- The rule that HB gain readout goes through `tone_s21` / `reference_normalization`
  and never through a hand-rolled `|i omega X| / V_in` ratio.
- The status of the Le Gal benchmark: what was measured, against what, and
  which of the five paper signatures have and have not been reproduced.
- The `power_balance` `z0_ohm` parameter and the renamed Manley-Rowe keys.

Update `references/le_gal_2025_gain_compression/README.md` with the corrected
device contract (equilibrium flux, branch capacitance stamp, `Z0` consistency)
and note that Levels 1-2 are in scope and Level 3 is not yet run.

---

## Testing Strategy

### Project Maturity Level
Active Development — the solver is production, the benchmark is new.

### Ordinary suite (must stay fast)
- `tests/physics/test_compression_power_convention.py`
- `tests/physics/test_compression_low_signal_limit.py`
- `tests/physics/test_compression_pump_depletion_limit.py`
- Branch-law and builder contract tests (Phase 1)
- Driver-vs-`tone_s21` normalization test (Phase 3)
- Restored Manley-Rowe gate (Phase 4)
- One reduced 20-cell compression curve with stored metrics (Phase 5)

### Manual / nightly
- The full 700-cell Level 2 campaign.
- Level 3 morphology, when separately budgeted.

### Mutation discipline
Every gate in this plan must be demonstrated **failing on the pre-fix code**
before it is claimed. For gates with no pre-fix failure (4.2), mutate the
implementation, show the failure, revert. Report the measured pre-fix
magnitude for each, not the word "passes".

---

## Rollback Plan

Commit each phase separately and in order, so any phase can be reverted alone:

1. `fix(builders): shift effective-SNAIL law to its equilibrium` (1.1-1.2)
2. `fix(builders): stamp SNAIL capacitance across the branch` (1.3-1.5)
3. `test(physics): add the power-convention level-1 gates` (Phase 2)
4. `fix(benchmark): route HB gain through tone_s21` (Phase 3)
5. `fix(observables): restore the lossless Manley-Rowe gate` (4.1-4.3)
6. `feat(reference): extend the CME oracle to the paper equations` (4.4)
7. `feat(benchmark): re-run the level-2 campaign` (Phase 5)
8. `docs: record the benchmark defects and status` (Phase 6)

Phases 1 and 3 change reported numbers for the Le Gal device only; no existing
design's published results depend on them. Phase 3.5 and Phase 4.3 touch
`power_balance` and `run_compression.py`, which the exp20/21/22 campaigns
consume — those campaigns are **not** re-run here, so the field renames must
keep the old keys readable or the consumers must be updated in the same
commit. Verify `experiments/exp22_spatial_attribution.py` still imports
cleanly before committing Phase 4.
