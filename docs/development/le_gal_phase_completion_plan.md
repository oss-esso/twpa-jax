# Implementation Plan: Close the Le Gal 2025 Benchmark Phase

Authored 2026-07-30. Supersedes nothing; continues
`docs/development/le_gal_benchmark_fix_plan.md`, whose Phases 1-3 are
implemented and verified.

**Read this whole document before touching a file.** It contains explicit STOP
conditions. Hitting one is a successful outcome, not a failure — report and
halt. Do not work around a STOP.

---

## Goal

Determine whether the effective single-branch SNAIL reduction can reproduce
the paper's simulated gain morphology, and if it can, reproduce it. If it
cannot, produce the evidence that says so and name the reason.

## Ground Rules (violating any of these invalidates the work)

1. **Never tune a published parameter to make gain appear.** If gain only
   appears at a value the paper does not state, that is a finding to report,
   not a configuration to adopt.
2. **Report measured numbers, never the word "passes."** Every claim in your
   final report must carry the number that supports it.
3. **Show every new gate failing before you claim it.** If a gate cannot fail
   on the pre-change code, mutate the implementation, show the failure, revert
   the mutation, and say so.
4. **The full test suite must end at exactly 4 failures** at every phase
   boundary. Baseline, verified 2026-07-30:
   ```
   4 failed, 264 passed, 3 skipped, 1 xfailed
   ```
   The 4 are pre-existing: `test_column_matrices_tracer` x2,
   `test_loss_model` x2 (the latter from `docs/loss_A10.csv not found`).
   Command:
   ```
   python -m pytest -q -p no:cacheprovider --basetemp D:/tmp/twpa_close tests/
   ```
   `--basetemp` must be off the repo (Windows ACL issue).
5. **Memory is constrained** (~7 GB usable; a prior run OOMed). Never launch a
   700-cell sweep and a second heavy job at the same time. Phases are ordered
   cheapest-first for this reason.
6. **File edits go through the Edit/Write tools.** No heredocs, no shell
   redirection to create files.
7. **Do not commit unless the phase's success criteria are met.** Commit each
   phase separately, in the order given.
8. **If a measured number contradicts this document, trust the measurement**
   and say so in your report. The numbers here were measured on this worktree
   and should reproduce, but a disagreement is information.

## Current State (all measured, reproduce before changing anything)

Device builder `src/twpa_solver/builders/le_gal_2025.py`, corrected:

```
L (from tangent at equilibrium) = 866.372 pH        (paper 869.6 pH)
Z0 = sqrt(L / Cg)               = 62.3765 ohm       (= port_impedance_ohm 62.4)
equilibrium current residual    = 1.06e-23 A
node_count = cells + 1, ports {1: 0, 2: cells}
```

Pump at the published `-78.4 dBm`, `7.5 GHz`:

```
Ipump = 6.806e-07 A = 0.4862 * Ic
max|delta/phi0| = 0.847830 (20 cells) / 0.633732 (100) / 0.744390 (700)
```

Pump solver ceiling (this is a real limit, do not assume headroom):

```
-78.4 dBm  solved
-58.4 dBm  FAILED at continuation endpoint, residual 1.126e-1
-48.4 dBm  FAILED at continuation lambda=0.25, residual 1.805e-2
```

Solved multitone tone content, 20 cells, fs=6.0 GHz, Ps=-115 dBm, output node,
`|X|` and ratio to signal:

```
h= 1 q= 0  1.320829e-16   pump
h= 1 q=-1  2.696995e-18   signal      1.000
h= 1 q= 1  1.134462e-20   idler       4.21e-03
h= 3 q=-1  3.334294e-21               1.24e-03
h= 2 q= 1  4.301918e-33               1.60e-15   <- structurally zero
h= 0 q= 1  9.577423e-33               3.55e-15   <- structurally zero
h= 2 q=-1  1.533921e-33               5.69e-16   <- structurally zero
h= 4 q=-1  1.381919e-33               5.12e-16   <- structurally zero
```

Even-`h` tones are zero because the shifted half-flux current-phase relation
is an **odd** function of the dynamic flux. This is a selection rule. It means
the device is pure four-wave mixing, which matches the paper's morphology
(idler mirrored about the pump: `f_p=7.5`, `f_s=6.0`, `f_i=9.0`).

Level-2 campaign result (60 points, all converged):

```
gain_vs_off_db in [-0.684521, +0.353524] dB
no 3 dB gain point, no 1 dB P1dB crossing anywhere
```

Odd-ladder basis convergence S=3 -> S=5 (valid):

```
5 GHz: max difference 6.77e-8 dB
8 GHz: max difference 3.29e-9 dB
```

Phase-budget reconciliation over the 700-cell device (`L_dev = 6.090 mm`):

```
CME    gamma = 1.4569e+33 /m/Wb^2 ;  |A_p| = 6.3709e-16 Wb
       gamma*|A_p|^2 = 591.34 /m  ->  nonlinear phase = 3.6012 rad
CME    dk = 484.59 /m             ->  linear mismatch = 2.9512 rad
direct (g3/g1)*delta^2 = 6.1582e-3 rad/cell -> 4.3107 rad over 700 cells
```

Two independent routes to the nonlinear phase agree to ~20%. The nonlinear
phase and the linear mismatch are the **same magnitude**, which is the
self-phase-matching regime a four-wave-mixing TWPA requires. Whether they
cancel or add depends on their **relative sign**, which has **not** been
determined. That is the central open question of this plan.

## What We're NOT Doing

- Not modelling four explicit junctions per SNAIL. That is the fallback if
  Phase 3 says the reduction is wrong, and it is a separate, larger project.
- Not comparing against experimental traces. Target is the paper's *simulated*
  curves.
- Not comparing against JosephsonCircuits.jl or the Themis measurement. Both
  are retired as references for this work.
- Not re-running exp20/21/22.
- Not implementing loss stages B or C in this plan.
- Not touching the pump solve path, the Floquet path,
  `port_s_from_unit_current_response`, or `port_waves`.

## Prerequisites

- [ ] Record `git status --short` and `git log --oneline -3` before starting.
- [ ] Reproduce the baseline suite at exactly 4 failures.
- [ ] Reproduce `L = 866.372 pH` and `Z0 = 62.3765 ohm` from the builder.

---

## Phase 0: Commit the verified work

The worktree currently holds ~22 modified and ~15 untracked files spanning two
separate efforts, none committed. Land the verified part before adding more.

### 0.1 Create a branch

```
git checkout -b le-gal-benchmark
```

### 0.2 Commit in this order, verifying the suite between each

1. `feat(core): add branch-law protocol and effective-SNAIL law`
   — `src/twpa_solver/core/nonlinear.py`, `src/twpa_solver/core/circuit.py`,
   `src/twpa_solver/multitone/problem.py`, `src/twpa_solver/pump/problem.py`,
   `tests/test_nonlinear_branch.py`
2. `feat(builders): add the Le Gal 2025 effective-SNAIL line`
   — `src/twpa_solver/builders/le_gal_2025.py`,
   `tests/test_le_gal_2025_builder.py`
3. `test(physics): add level-1 power-convention gates`
   — `tests/physics/`
4. `feat(reference): freeze the Le Gal benchmark contract and CME oracle`
   — `references/le_gal_2025_gain_compression/`,
   `scripts/reproduce_le_gal_2025_cme.py`, `tests/test_le_gal_2025_cme.py`
5. `feat(benchmark): add the HB campaign driver`
   — `scripts/run_le_gal_2025_hb.py`
6. `fix(observables): plumb z0 and restore conservation diagnostics`
   — `src/twpa_solver/multitone/observables.py`,
   `scripts/run_compression.py`, `tests/test_power_balance.py`, and the other
   modified multitone/test files
7. `docs: record the benchmark defects, fixes, and measurements`
   — the `docs/development/*.md` and `.png` files, `CLAUDE.md`, `.gitignore`

If a file does not obviously belong to one of these, put it in (7).

### 0.3 Do not commit

- `tmptw_fullrun/` (test scratch)
- any `__pycache__/`
- `docs/development/loss_A10.csv` if its only change is the corrupted header
  (`\u200b whereFrequency_GHz`) — that is a separate pre-existing defect, leave
  it modified in the worktree and say so.

### Success Criteria

**Automated**: suite at exactly 4 failures after the last commit.
**Manual**: `git status --short` shows only the intentional exclusions above.

---

## Phase 1: Fix the wrong cubic coefficient and map the Kerr sign

Pure algebra. No circuit solves. This phase should take minutes.

### 1.1 Understand what is wrong

`references/le_gal_2025_gain_compression/parameters.json` currently contains:

```json
"alternate_cpr_cubic_coefficients": {
  "published_placement_Ic_units": 0.00416,
  "external_flux_on_small_junction_Ic_units": -0.016506
}
```

The second number is wrong. It equals `-(r/6 + 1/162) = -0.016506173`, which
is the cubic Taylor coefficient of `+r sin(u) + sin(u/3)` expanded at `u = 0`
— a law with **no external flux applied at all**. It does not describe the
alternate placement.

The two placements are *mathematically identical* at half flux once shifted to
their static equilibria:

- published: `I(Phi) = r Ic sin(Phi/phi0) + Ic sin((Phi - pi phi0)/(3 phi0))`,
  equilibrium `Phi* = pi phi0`, shifted law
  `I(d) = Ic(-r sin(d/phi0) + sin(d/(3 phi0)))`
- alternate: `I(Phi) = r Ic sin((Phi - pi phi0)/phi0) + Ic sin(Phi/(3 phi0))`,
  equilibrium `Phi* = 0`, shifted law
  `I(d) = Ic(-r sin(d/phi0) + sin(d/(3 phi0)))`

Identical. This is confirmed by measurement: at 100 cells the two conventions
give **bit-identical** `max|delta/phi0| = 0.633732` and a gain difference of
`-9.593021e-14 dB`. The `2e-13 dB` result previously reported as evidence that
the sign does not matter is instead a proof that the flag is a no-op at half
flux.

### 1.2 Add a coefficient function

**File**: `src/twpa_solver/core/nonlinear.py`

Add a module-level function (not a method — this must be callable without
building a circuit):

```python
def snail_taylor_coefficients(
    ratio: float, flux_over_flux0: float, *, phi0: float = PHI0_REDUCED
) -> dict[str, float]:
    """Return operating-point Taylor coefficients of the shifted SNAIL law.

    Returns ``g1`` and ``g3`` in units of ``Ic`` (that is, with ``Ic`` and
    ``phi0`` divided out), plus their ratio, plus the equilibrium flux.  The
    ratio ``g3/g1`` is the physically meaningful quantity: the sign of ``g1``
    depends on which of the two mirror equilibria the root finder lands on,
    and both coefficients flip together, leaving the ratio invariant.
    """
```

Requirements:

- Solve `I(Phi*) = 0` numerically. **Bracket properly**: the naive bracket
  `[phi_ext - 3 pi phi0, phi_ext + 3 pi phi0]` finds a *mirror* equilibrium at
  which `g1 < 0`. Select the root with `g1 > 0` (the stable branch). If no
  root with `g1 > 0` exists in the period, raise `ValueError` naming the
  `(ratio, flux_over_flux0)` pair.
- Compute `g1` and `g3` by **analytic** differentiation, not finite
  differences. Finite differences at `h = 1e-3 * phi0` on a third derivative
  lose ~9 digits and will not reproduce the reference values below.
- Return a dict with keys `g1`, `g3`, `g3_over_g1`, `equilibrium_flux_rad`.

**Analytic reference at half flux** (`flux_over_flux0 = 0.5`), where the
shifted law is `-r sin(u) + sin(u/3)`:

```
g1 = 1/3 - r
g3 = r/6 - 1/162
```

### 1.3 Pin the coefficients with a test

**File**: `tests/test_nonlinear_branch.py`

Add `test_snail_taylor_coefficients_at_half_flux`. Assert to `1e-9` relative:

| `r` | `g1` (Ic) | `g3` (Ic) | `g3/g1` |
| --- | ---: | ---: | ---: |
| 0.000000 | +0.333333333 | -0.006172840 | -0.018518519 |
| 0.020000 | +0.313333333 | -0.002839506 | -0.009062254 |
| 0.037037 | +0.296296333 | -0.000000006 | -0.000000021 |
| 0.050000 | +0.283333333 | +0.002160494 | +0.007625272 |
| **0.062000** | **+0.271333333** | **+0.004160494** | **+0.015333515** |
| 0.100000 | +0.233333333 | +0.010493827 | +0.044973545 |
| 0.200000 | +0.133333333 | +0.027160494 | +0.203703704 |

Also assert:

- The sign change of `g3` occurs at `r = 6/162 = 0.037037037` to `1e-9`.
- `g3` is **negative** for `r < 0.037037` and **positive** for `r > 0.037037`.
- Both flux placements give identical coefficients at half flux:
  `snail_taylor_coefficients` agrees with the shifted-law coefficients
  extracted from a circuit built with `external_flux_on_small_junction=True`
  to `1e-12` relative.

**Mutation check**: change `r/6 - 1/162` to `r/6 + 1/162` in the
implementation; the test must fail. Revert. Report that you did this.

### 1.4 Correct `parameters.json`

Replace `alternate_cpr_cubic_coefficients` with:

```json
"cpr_operating_point_coefficients": {
  "note": "Both external-flux placements reduce to the same shifted law at Phi0/2; the two are not independent conventions. Coefficients are in Ic units with phi0 divided out.",
  "half_flux_g1_over_Ic": 0.271333333,
  "half_flux_g3_over_Ic": 0.004160494,
  "half_flux_g3_over_g1": 0.015333515,
  "g3_sign_change_ratio": 0.037037037,
  "superseded": "The previously recorded -0.016506 was the cubic of the unshifted law at Phi=0 and described no solved circuit."
}
```

Add `"CPR operating-point Taylor coefficients"` to `provenance.inferred`.

### 1.5 Emit the sign map

**File**: `scripts/le_gal_kerr_sign_map.py` (new)

A standalone script, no circuit solves, that writes
`references/le_gal_2025_gain_compression/kerr_sign_map.csv` with columns
`ratio, flux_over_flux0, equilibrium_flux_rad, g1_over_Ic, g3_over_Ic,
g3_over_g1, status`.

Grid: `ratio` from 0.00 to 0.30 in steps of 0.005; `flux_over_flux0` from 0.20
to 0.50 in steps of 0.01. Where no stable equilibrium exists, write
`status = "NO_STABLE_EQUILIBRIUM"` and leave the numeric columns empty — do
not substitute a value.

**Important**: at `flux_over_flux0` strictly less than 0.5 the shifted law is
**not** odd, so it acquires a quadratic term and the device becomes
three-wave-mixing. Add a `g2_over_Ic` column. Report at which flux values
`|g2/g1|` becomes comparable to `|g3/g1| * delta^2` for `delta = 0.63` (the
measured pump phase) — that is where the device stops being a 4WM amplifier
and the entire `2 theta_p - theta_s - theta_i` basis convention in
`observables.spatial_profiles` and `build_sideband_matched_basis` becomes the
wrong model.

### Success Criteria

**Automated**: `tests/test_nonlinear_branch.py` passes with the table above;
mutation demonstrated. Suite at exactly 4 failures.

**Manual**: report in prose:
- the corrected coefficient values,
- the `r` at which `g3` changes sign,
- whether the paper's `r = 0.062` sits on the positive or negative side and by
  how much,
- the flux at which `g2` starts to matter.

**Commit**: `fix(reference): correct the SNAIL operating-point coefficients`

---

## Phase 2: Determine the sign of the linear phase mismatch

Phase 1 gives the sign of the nonlinear phase. Gain requires the two to
**cancel**. Nobody has measured the sign of the linear mismatch. Do that now.

### 2.1 Do not inherit the assumption

Earlier analysis in this repo asserted that a negative Kerr would cancel the
mismatch. That assumed `dk > 0` and **was never verified**. It may be that
`dk < 0` and the positive `g3` is correct, in which case the gain null has a
different cause entirely. Determine both signs independently.

### 2.2 Compute the discrete ladder dispersion

**File**: `src/twpa_solver/builders/le_gal_2025.py`

Add:

```python
def ladder_dispersion(
    omega_rad_per_s: float | np.ndarray,
    *,
    inductance_h: float,
    snail_capacitance_f: float,
    ground_capacitance_f: float,
    cell_length_m: float,
) -> np.ndarray:
    """Return the wavenumber k(omega) of the discrete LC ladder, in rad/m."""
```

Use the exact discrete-ladder relation for a series `L` in parallel with the
branch capacitance `C`, shunted by `Cg` — not a continuum approximation. State
the relation you used in the docstring, and assert that in the low-frequency
limit it reduces to `k -> omega sqrt(L Cg) / dx` to 1% at 100 MHz.

### 2.3 Emit the phase budget

**File**: `scripts/le_gal_phase_budget.py` (new)

For `f_p = 7.5 GHz` and `f_s` swept over 4.0-11.0 GHz in 0.1 GHz steps, with
`f_i = 2 f_p - f_s`, write
`references/le_gal_2025_gain_compression/phase_budget.csv` with:

```
f_s_GHz, f_i_GHz, k_p, k_s, k_i, dk_lin, dk_nl, dk_total, regime
```

where

```
dk_lin   = 2 k_p - k_s - k_i                  (rad/m, SIGNED)
dk_nl    = the nonlinear (Kerr) contribution   (rad/m, SIGNED)
dk_total = dk_lin + dk_nl
regime   = "PHASE_MATCHED" if |dk_total| * L_dev < 1.0 rad, else "MISMATCHED"
```

Derive `dk_nl` from `g3/g1` and the measured pump amplitude, and **state the
derivation explicitly in the script docstring** including how the sign is
carried. Cross-check the magnitude against the two independently measured
values already on record:

```
CME   :  3.6012 rad over 6.090 mm  ->  591.3 rad/m
direct:  4.3107 rad over 6.090 mm  ->  707.8 rad/m
```

Your `dk_nl` at `f_s = 6.0 GHz` must land between these two, or within 30% of
their mean. If it does not, your derivation is wrong — stop and report rather
than proceeding.

The already-recorded `dk_lin = 484.59 rad/m` at `f_s = 6.0 GHz` is a magnitude
only; the sign was never recorded. Your job is to establish it.

### 2.4 Report the phase-matching verdict

State plainly, with numbers:

- the sign of `dk_lin` at `f_s = 6 GHz`,
- the sign of `dk_nl`,
- whether they cancel or add,
- `|dk_total| * L_dev` in radians,
- the `g3/g1` value that would make `|dk_total| * L_dev < 1.0`, and the `r`
  that produces it via the Phase 1 map.

### Success Criteria

**Automated**: a test asserting `ladder_dispersion` reduces to the continuum
limit at low frequency to 1%, and a test pinning `dk_lin` at `f_s = 6.0 GHz`
to the value you measure (with its sign). Suite at exactly 4 failures.

**Manual**: the verdict above.

**Commit**: `feat(benchmark): measure the signed phase budget`

---

## Phase 3: DECISION GATE — stop and report

Do not proceed past this point without evaluating the gate. Write your
conclusion into
`docs/development/le_gal_phase_matching_verdict.md` before doing anything else.

### Gate

Using the Phase 1 map and the Phase 2 signed budget, answer:

**Q: At the paper's published parameters (`r = 0.062`, `flux = 0.5`,
`f_p = 7.5 GHz`, `P_p = -78.4 dBm`, 700 cells), is
`|dk_total| * L_dev < 1.0 rad` anywhere in the 4-11 GHz signal band?**

### Branch A — YES, phase-matched somewhere

The model can amplify and the earlier null was a measurement or operating-point
problem. Proceed to Phase 4. In your report, name the frequencies where
`|dk_total| * L_dev < 1.0` and explain why the Level-2 campaign (which sampled
5.0, 6.0, 8.0, 9.5 GHz) missed them.

### Branch B — NO, and a *published-parameter* variation fixes it

Example: the paper's `r` is defined as large/small rather than small/large, so
the effective ratio is `1/0.062 = 16.1` or the reciprocal reading gives
`r < 0.037`. Document the reading, cite where in the paper it is stated,
change it **once**, re-run Phases 1-2, and proceed to Phase 4.

**You may make exactly one such reinterpretation, and only if you can point to
the specific sentence in the paper that supports it.** If you cannot, take
Branch C.

### Branch C — NO, and no published-parameter reading fixes it

**STOP. Do not proceed to Phase 4 or beyond.** The conclusion is: *the
effective single-branch SNAIL reduction cannot reproduce this device; the
explicit four-junction topology is required.*

Write that up with:

- the `g3/g1` at published parameters (`+0.015333515`),
- the `r` needed for cancellation, from the Phase 1 map,
- the signed phase budget,
- the statement that the reduction, not the solver, is the limiting factor,
- an estimate of the four-junction cost: ~3 internal nodes per cell, ~2800
  nodes at 700 cells, real system dimension ~`2 * H * 2800`, versus ~43,400 for
  the reduction — and a note that memory scales as
  `(n_pump_modes + 2S + 1)^2`, so this is not a small increase.

Then report and halt. This is a legitimate, complete, valuable outcome.

### Success Criteria

**Manual**: the verdict document exists, names its branch, and carries every
number listed for that branch.

**Commit**: `docs: record the phase-matching verdict`

---

## Phase 4: Make the CME oracle produce lobes

**Only if Phase 3 took Branch A or B.**

### 4.1 Diagnose the instability

The calibrated frequency scan is reported as numerically unstable. Before
changing the integrator, determine *why*:

- Print `solve_ivp`'s `result.message` and `result.status` at the failing
  frequency. Do not swallow it.
- Check whether the failure is stiffness (integrator takes vanishing steps) or
  blow-up (envelope amplitudes diverge). Blow-up in a lossless three-wave
  system that conserves `sum |A|^2` means the conservation is broken, which is
  a coefficient-sign error, not an integrator problem.
- Report `photon_flux(envelopes)` at `z=0` and at the last successful `z`. If
  it is not conserved to `1e-7`, **the problem is the equations, not the
  solver.** Fix the equations.

### 4.2 Only then adjust the integration

If and only if 4.1 shows genuine stiffness: switch to `method="LSODA"` or
`"Radau"`, and add a `max_step` bounded by `0.05 / |dk_total|` so the phase
factor is resolved. Keep `rtol=1e-9`.

Do **not** silently clamp amplitudes, add damping, or shorten the device to
make it integrate. Any of those makes the oracle useless.

### 4.3 Emit the lobe scan

**File**: `scripts/reproduce_le_gal_2025_cme.py` (extend)

Sweep `f_s` over 4.0-11.0 GHz in 0.1 GHz steps at the published pump, low
signal power (`-115 dBm`), and write
`references/le_gal_2025_gain_compression/cme_gain_vs_frequency.csv` with
`f_s_GHz, gain_dB, photon_flux_rel_err, status`.

### Success Criteria

**Automated**:
- `photon_flux` conserved to `1e-7` at every scanned frequency (assert it).
- A test asserting the scan produces **two** local maxima, one below and one
  above 7.5 GHz, with their frequencies recorded.

**Manual**: report the two lobe centre frequencies and peak gains. If the scan
is conservative and stable but shows **one** lobe or none, that is a finding —
report it and return to Phase 3's gate with the new information rather than
proceeding.

**Commit**: `feat(reference): produce the CME lobe scan`

---

## Phase 5: Confirm the HB solver against the oracle

**Only if Phase 4 produced two lobes.**

### 5.1 Run the HB scan at the same frequencies

Use `scripts/run_le_gal_2025_hb.py` at 700 cells, low signal power, over the
same 4.0-11.0 GHz grid but at 0.25 GHz steps (29 points) to bound cost.
Sideband order 3. Serial. This is the heaviest job so far — run it alone.

If a point exceeds 600 s, record `status = "TIMEOUT"` and continue; do not
stall the sweep.

### 5.2 Compare

Write `references/le_gal_2025_gain_compression/hb_vs_cme.csv` and report:

- lobe centre agreement, HB vs CME, in GHz
- peak gain agreement, in dB
- the maximum `|HB - CME|` across the band

**Acceptance (engineering tolerances for this fixture, not paper claims)**:
lobe centres within 0.5 GHz, peak gain within 2 dB.

### 5.3 Basis convergence at the amplifying point

Re-run the S=3 vs S=5 comparison **at the peak-gain frequency**, not at the
0 dB points where the existing `6.77e-8 dB` and `3.29e-9 dB` were measured.
Those numbers say nothing about the basis at 20 dB. Gate:
`|P1dB(S=5) - P1dB(S=3)| < 0.25 dB`.

### Success Criteria

**Automated**: the comparison test with the tolerances above.
**Manual**: the three agreement numbers, plus the convergence delta.

**Commit**: `feat(benchmark): compare HB against the CME oracle`

---

## Phase 6: Digitize the paper reference curves

This phase is **independent of Phases 1-5** and may be started at any time. It
is a hard blocker for "matching the paper" regardless of the physics outcome:
`parameters.json` currently has `"digitized": []`, so there is nothing to
compare against numerically.

### 6.1 Digitize

Produce, in `references/le_gal_2025_gain_compression/`:

- `digitized_full_theory_p1db.csv` — Fig. 3(a), full-theory curve
- `digitized_full_theory_pump_depletion.csv` — Fig. 2(b)
- `digitized_gain_lobes.csv` — Fig. 2(a) low-power line cut
- `digitized_signal_loss_tangent.csv` — Fig. 6(d)

Each file must carry a header block or `.meta.json` sidecar stating: the
digitizer used, the figure and panel, the axis calibration points, the
estimated uncertainty, and whether the curve is the paper's *simulated* or
*experimental* trace. **Only simulated traces are acceptance targets.**

### 6.2 Register provenance

Add every file to `checksums.json` with its SHA-256. Move the entries from
`provenance.digitized` out of `[]`.

### 6.3 Never substitute

If a curve cannot be digitized reliably, leave the file absent and record
`"unavailable: <reason>"` in `provenance`. Do not synthesize points. Do not
interpolate from the text.

### Success Criteria

**Automated**: a test that every file listed in `provenance.digitized` exists
and matches its recorded SHA-256.

**Manual**: the uncertainty estimate per curve.

**Commit**: `feat(reference): digitize the paper simulated curves`

---

## Phase 7: Level 2 and Level 3 campaigns

**Only if Phase 5 passed and Phase 6 produced reference curves.**

### 7.1 Level 2 (selected compression curves)

Frequencies `5.0, 6.0, 8.0, 9.5 GHz` plus the two lobe centres from Phase 4.
Powers `-115` to `-94 dBm` in 3 dB steps. 700 cells. Lossless (version A).

Check, per the frozen benchmark spec:
- gain approaches the Floquet small-signal result as `P_s -> 0`
- a stable low-power plateau exists (assert first two points agree to 0.05 dB)
- compression begins smoothly
- pump transmission falls as gain compresses
- `|P1dB(S=5) - P1dB(S=3)| < 0.25 dB`

### 7.2 Level 3 (full morphology)

**Check free memory first and report it.** 31 frequencies over 4-11 GHz, 1 dB
power grid from -115 to -94 dBm, 700 cells. If the estimated footprint exceeds
free RAM, reduce the frequency count and **say so explicitly** rather than
running to an OOM.

Produce all six figures named in the benchmark spec:

```
gain_vs_signal_power_at_selected_frequencies.png
gain_frequency_power_map.png
pump_depletion_frequency_power_map.png
p1db_vs_frequency.png
pump_depletion_at_p1db.png
local_gain_vs_cell_and_frequency.png
```

### 7.3 Score against the acceptance criteria

| Criterion | Tolerance | Source |
| --- | --- | --- |
| Lobe locations | within 0.5 GHz | Fig 2(a) |
| Low-power gain magnitude | within 2 dB | Fig 2(a) |
| Central-band P1dB | within 3 dB | Fig 3(a) |
| Demeaned P1dB(f) shape correlation | > 0.8 | Fig 3(a) |
| Pump attenuation at P1dB | >= 1 dB over most of the band | Fig 2(b) |
| Central gain buildup | monotone | Fig 4 |
| Edge behaviour | oscillatory | Fig 4 |

Report all seven with their measured values, whether or not they pass. A
failing criterion is a result, not a reason to adjust a parameter.

### Success Criteria

**Automated**: a reduced (20-cell, two-frequency) regression fixture with
stored metrics, fast enough for the ordinary suite.

**Manual**: the seven-row scorecard.

**Commit**: `feat(benchmark): run the level-2 and level-3 campaigns`

---

## Phase 8: Documentation

Update, with measured numbers only:

- `CLAUDE.md` — the benchmark's status, which of the seven criteria are met,
  and the Phase 3 verdict.
- `references/le_gal_2025_gain_compression/README.md` — the device contract,
  the CPR coefficient correction, digitization provenance.
- `docs/development/saturation_solver_design_independent_validation.md` —
  replace the withdrawn `-0.016506` and the invalid `S=2 -> S=3` convergence
  claim if any trace remains.

**Commit**: `docs: record the benchmark outcome`

---

## Testing Strategy

### Project Maturity Level
Active Development.

### Ordinary suite (must stay under ~90 s total)
- `snail_taylor_coefficients` table (Phase 1.3)
- `ladder_dispersion` continuum limit + signed `dk_lin` (Phase 2)
- CME photon-flux conservation + two-lobe assertion (Phase 4)
- HB-vs-CME lobe agreement (Phase 5)
- digitized-file checksum registry (Phase 6)
- reduced 20-cell compression fixture (Phase 7)

### Manual / long-running
- 700-cell frequency scan (Phase 5.1)
- Level 2 and Level 3 campaigns (Phase 7)

### Mutation discipline
Every gate must be shown failing before it is claimed. Where the pre-change
code already passes, mutate the implementation, show the failure, revert, and
report that you did so. Specific mutation required in Phase 1.3.

---

## Rollback Plan

Each phase is one commit on branch `le-gal-benchmark`. Any phase reverts with
`git revert <sha>` without touching the others, because:

- Phases 1-2 add pure functions and data files; nothing existing calls them.
- Phase 3 is documentation only.
- Phases 4-5 touch only `references/` and `scripts/`.
- Phase 6 adds data files.
- Phase 7 adds outputs and one regression fixture.

The only shared-surface risk is Phase 0 commit 6, which touches
`multitone/observables.py` and `scripts/run_compression.py` — consumed by
`experiments/exp22_spatial_attribution.py`. Before committing it, run:

```
python -c "import experiments.exp22_spatial_attribution"
```

and confirm it imports.

---

## Final Report Format

Your closing message must contain, in this order:

1. Which Phase 3 branch was taken, and the numbers that decided it.
2. A table of every gate added, with the pre-change failing value and the
   post-change passing value.
3. The suite line, verbatim.
4. The seven-row acceptance scorecard if Phase 7 ran, or an explicit statement
   of which phase you stopped at and why.
5. Anything you could not do, and what blocked it.

Do not write "complete", "working", or "validated" without a number beside it.
