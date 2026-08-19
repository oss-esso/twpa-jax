# Implementation Plan: Solver Coverage Past the Neimark-Sacker Boundary

Written 2026-08-19. Scope decisions in this document are grounded in the
measurements recorded in `CLAUDE.md` and in the code references given below.
Every performance number quoted was measured on this machine, not estimated.

## Goal

Give the solver a defined, fast answer at every pump power on a column --
including above the Neimark-Sacker boundary -- by routing each operating point
to the cheapest method that is valid there, and by making the fallback method
affordable enough to use as an oracle.

## Current State Analysis

### Three regimes, two of them already solved

The measured ansatz-validity corpus (72 points, 4 devices: 56 rows in
`outputs/chaos/ansatz_validity/ansatz_validity.csv` plus 16 rows in
`run_tmp_phaseA_reduction_20260819/2c_gap_dense.csv`; `CLAUDE.md`'s "68 points"
predates the dense rerun) separates the pump-power axis into three regimes:

| regime | spectral content | solver | cost |
| --- | --- | --- | --- |
| period-1 | on-lattice fraction `1.0000 -> 0.9233` | single-tone HB (`pump/hb.py`) | 1-20 s |
| 2-torus | one extra incommensurate generator, `generator_share 0.999 -> 0.627` | two-frequency HB (`multitone/torus.py`) | minutes |
| broadband | on-lattice `0.056`, residue a continuum | **nothing** | FDTD only, 300-900 s/point |

**Regimes 1 and 2 are done.** `src/twpa_solver/multitone/torus.py` is 2465 lines
of autonomous two-frequency harmonic balance with branch-lock geometry
(`BranchLockGeometry:238`), bordered preconditioners
(`apply_border_aware_preconditioner:35`, `apply_one_border_preconditioner:66`)
and its own continuation; drivers live at `scripts/chaos/run_branch_locked_torus.py`,
`run_physical_torus_column.py`, `run_torus_signal_response.py`. It is the
operator that produced the torus onset `P_c = -24.2435 dBm`, which agrees with
the independent Hill crossing `-24.2334894 dBm` to **0.010 dB in power and
0.45% in frequency**. This plan does not rebuild any of that.

**Regime 3 has no method.** That is the entire remaining gap.

### The regime-3 measurement rules out the obvious extensions

At collapse, on-lattice power falls to `0.056`/`0.051` in one 0.4 dB step.
Admitting half-integer modes recovers only **4.8 / 4.5 percentage points**;
thirds recover 9.5 / 10.2. The top-20 off-lattice bins hold 2.4% of the power
and the single best extra generator explains 26%, with a floor 13-18 dB below
the pump. So the residue is a **continuum, not a few missed lines**.

Consequences, already recorded in `CLAUDE.md` and binding on this plan:
- No period-N ansatz. `build_half_pump_basis` (`multitone/basis.py:379`),
  `pump/floquet.py`, `pump/periodic_branch.py` and `signal/period_doubled.py`
  stay dormant. This measurement is the direct test of the hypothesis they
  serve and it fails.
- Adding more discrete generators does not converge on this residue. A third
  generator would chase 26% of 5% of the power.

### The FDTD oracle is affordable to fix, and currently sequential

`scripts/chaos/run_guarcello_jc_phase5.py` holds the time-domain kernel:

- `integrate_jc_banded:1111` -- readable NumPy/SciPy reference path.
- `_integrate_jc_banded_numba_stage_a:672` -- the production compiled loop, the
  one `_integrate_jc_compiled:1074` actually dispatches to.
- `_integrate_jc_banded_numba_stage_b:962` -- a compact-diagonal variant that
  **measures slower** and must not be called until that changes. Its own
  docstring records `rf_squid 5380 / 6885 / 4179` and `ipm_2c_fixed 4743 / 6900
  / 3977` steps/s for original / Stage A / Stage B.

Three properties of that loop drive this plan:

1. **The banded factor is drive-independent.** `A = C/dt^2 + G/(2 dt) (+ K)`
   (`integrate_jc_banded:1123-1127`) contains no pump or signal term. Every
   operating point on a column at fixed `dt` shares one LU. A batch of drives
   is therefore many right-hand sides against a single factorization.
2. **Points run strictly sequentially.** There is no process or thread pool
   anywhere in the file; `_run_point:1410` is called once per point. A campaign
   of `N` powers costs `N` times one point, on one core, on a 6-core part. The
   agent's three-point `dt=0.005` run took **0.54 h**.
3. **Small per-step waste exists but is not the lever.** Stage A divides by
   `dt_s*dt_s` and `2*dt_s` elementwise every step (`:736`, `:738`) where
   Stage B hoists reciprocals; and the recording block recomputes
   `math.sin(phase[j])` at `:752` over all `n_branches` after `:719` already
   computed it. Both are real, both are worth fixing while the file is open,
   and neither is worth a phase of its own -- the recording block only runs
   every `record_stride` steps (default 20).

### Hardware constraint that overrides the earlier sketch

Measured on this machine:

```text
jax    0.10.2   devices: [CpuDevice(id=0)]
numba  0.67.0
torch  2.10.0+cpu
cupy   MISSING
```

**There is no GPU.** A `jax.vmap` batching plan justified by GPU throughput
does not apply here. On CPU the same batch axis is served by `numba.prange`
over independent points, which parallelizes across the 6 cores the machine
actually has and requires no rewrite of a kernel that is already compiled and
measured. Phase 2 therefore specifies the batch *interface* and selects the
backend by measurement, with the JAX path kept as a drop-in for the day a GPU
is available.

## What We're NOT Doing

- **No period-N / half-pump / period-doubled ansatz.** Refuted by the 4.8 pp
  measurement above.
- **No UPO detection or cycle expansions.** Requires a dense set of unstable
  periodic orbits on a 6096-node system; the enumeration cost is not bounded
  and nothing in the corpus indicates a low-dimensional cycle structure.
- **No transfer-operator / Perron-Frobenius discretization.** State dimension
  6096; a discretized transfer operator is not representable.
- **No POD-DEIM or projection-based ROM.** The continuum residue is broadband
  and drive-dependent, so a basis trained at one drive does not transfer, and
  the training runs cost what the full runs cost.
- **No Koopman/EDMD surrogate.** Same training-transfer objection, plus it
  would produce a fitted model with no error bound, which fails the repository's
  standard for a published number.
- **No new gain or compression numbers from `current_complex_c`.** It is
  analytic but breaks conjugate symmetry; stability and response analysis only.
- **No rebuild of `multitone/torus.py`.** It works.
- **No production gain-map integration in this plan.** Routing lands as a
  library plus a diagnostic driver first.

## Prerequisites

- [ ] The outstanding FDTD control point is run: control `0.5975` at
      `dt_norm = 0.01` on the **live 6096 build**, compared against the stored
      `0.05205121`. The stored `dt=0.01` corpus carries
      `natural_bandwidth = 4578` (legacy 6136 build) while every new run uses
      `4558`. Until this reproduces, FDTD is not a trustworthy oracle for
      Phase 3 and the `-0.056 dB` timestep shift is not separable from the
      build change. ~10 min.
- [ ] `jc_jtwpa` / `jc_fqjtwpa` have no linear-limit validation at all -- the
      `Ic = 0` check is degenerate on both because it removes their only
      inductive path. A finite-linear-inductance variant of `_measure_linear_limit`
      is required before either device is used as a Phase 3 reference.
      `ipm_2c_fixed` passes at `dt_norm = 0.01` (0.0076 relative) and is the
      only device currently cleared.
- [ ] Baseline test state recorded before any edit: `6 failed, 939 passed,
      5 skipped, 1 xfailed`. One of those failures is the bandwidth expectation
      `4578` vs live `4558` and is the build difference, not a regression.

---

## Phase 1: Regime Router

### Overview

Decide, for one operating point, which of the three regimes it is in, and
dispatch to the method that is valid there. Every ingredient exists; none of
them is currently wired to a decision.

### Changes Required

#### 1. Regime classification library

**File**: `src/twpa_solver/chaos/routing.py` (new)

**Changes**: A pure-decision module with no solver imports at module scope, so
it is unit-testable without a circuit.

- `RegimeVerdict` dataclass: `regime` (`PERIOD_1 | TORUS | BROADBAND |
  UNDECIDED`), `evidence` (the scalar that decided it), `margin`, and
  `reason` (a string, always populated, including on `UNDECIDED`).
- `classify_from_multiplier(magnitude, *, tolerance)` -- `PERIOD_1` below
  `1 - tolerance`, `TORUS` above `1 + tolerance`, `UNDECIDED` inside the band.
  The measured slope is `0.019 per dB` in `|lambda|`, so a `tolerance` of
  `2e-3` is about `0.1 dB` of pump power; that is the default and it is a
  parameter, not a constant.
- `classify_from_spectrum(on_lattice, generator_share)` -- the FDTD-side
  classifier, thresholds taken from the corpus: `on_lattice >= 0.90` with
  `generator_share >= 0.60` is `TORUS`; `on_lattice < 0.30` is `BROADBAND`.
  The corpus has no points between `0.4620` and `0.8827` on-lattice at
  `ipm_2c_fixed`, so the gap is where `UNDECIDED` lives, honestly.
- `route(verdict) -> str` -- returns the method name only. It does not import
  or call a solver. Dispatch is the driver's job.

**Rationale for keeping the thresholds in one file**: they are the only
tunable physics in the router, and every one of them is a measured number that
will move when a new device is added.

#### 2. Cheap in-loop regime probe

**File**: `src/twpa_solver/chaos/routing.py`

**Changes**: `probe_multiplier(...)` wraps the existing Hill path -- the
conversion matrix already assembled by the gain solve is the Hill matrix, and
`refine_complex_resonance` already refines a root against it. The probe must
carry the two search properties that were established the hard way:

- search **both imaginary half-planes** (`--both-imaginary-half-planes`
  equivalent); the unstable pair sits in the negative-imaginary half-plane and
  a real-seeded shift-invert search never reaches it;
- track a **named branch** by mode overlap, never `max |lambda|` off a dense
  scan -- a dense maximum has no branch identity and returns whichever internal
  mode is highest. Gate on `mode_overlap >= 0.99`; the validated march held
  `>= 0.998979` throughout.

Reference implementations to lift from, not re-derive:
`scripts/chaos/enumerate_hill_candidates.py`, `scripts/chaos/track_critical_root.py`,
`src/twpa_solver/signal/branch_tracking.py`.

#### 3. Diagnostic driver

**File**: `scripts/chaos/route_column.py` (new)

**Changes**: Marches one pump-frequency column in power, calls the probe at
each point, writes `routing.csv` (`drive_dbm, regime, evidence, margin,
mode_overlap, reason, method`) and one PNG. It **classifies only** -- it does
not run the downstream solver. That keeps Phase 1 independently testable and
cheap enough to run on a whole column.

Follows the repository's one-column scope rule: `--column-freq-ghz`, everything
else pinned.

### Success Criteria

**Automated**:
- `python -m pytest tests/test_chaos_routing.py -q -p no:cacheprovider --basetemp D:\tmp\twpa_route`
- `ruff check src/twpa_solver/chaos/routing.py scripts/chaos/route_column.py`
- Classifier unit tests use table-driven fixtures built from the corpus rows,
  not from a live solve.

**Manual**:
- On `ipm_2c_fixed` at `f_p = 7.9 GHz`, the router must return `PERIOD_1` below
  `-24.30 dBm` and `TORUS` above `-24.15 dBm`, with the `UNDECIDED` band
  containing the known crossing `-24.2334894`. A router that reports a crisp
  regime *inside* its own tolerance band is wrong, not impressive.
- `mode_overlap` stays `>= 0.99` across the march. A drop means the tracker
  fell onto a different root and the verdict on that point is void.

---

## Phase 2: Batched FDTD

### Overview

Make the oracle affordable. One LU factorization serves an entire power column,
and independent points fill the machine's cores. This phase changes cost, never
physics: the converged trajectory for any single point must be **bit-identical**
to today's.

### Changes Required

#### 1. Batch interface

**File**: `scripts/chaos/run_guarcello_jc_phase5.py`

**Changes**: Add `integrate_jc_banded_batch(device, *, pump_currents_a, ...)`
taking an array of drives and returning stacked results. It builds `A` and its
banded factor **once** (`:1123-1127` is already drive-free) and threads a batch
axis through the right-hand side.

Signature and return shape mirror `integrate_jc_banded` per batch element, so
the existing single-point path becomes `batch of one` and every caller keeps
working.

#### 2. Backend selection, decided by measurement

**File**: `scripts/chaos/run_guarcello_jc_phase5.py`

**Changes**: Two implementations behind that one interface.

- `numba` backend: `prange` over the batch axis in a new
  `_integrate_jc_banded_numba_batch`. Each lane keeps its own `q_prev/q_cur/
  q_next` and shares the read-only factor arrays. This is the expected winner
  on this machine.
- `jax` backend: `vmap` over the drive axis of a `lax.scan` step. Written, kept
  behind the same interface, **not** made default.

**Decision rule, fixed now so this is not an open question**: run both on
`ipm_2c_fixed`, batch sizes 1/2/4/6, 200 pump periods, and select per measured
throughput in points/hour at equal peak RSS. The default backend is whichever
wins; on a machine reporting `[CpuDevice(id=0)]` the expectation is `numba`,
and the JAX path is retained only so a GPU makes it a one-line switch. Record
the table in this file when it is measured, the way the Stage A/B table is
recorded in the kernel docstring.

**Explicitly not done**: reviving Stage B. It measures slower on both devices
and both bandwidths, and its own docstring forbids calling it until that
changes.

#### 3. Per-step constant reduction

**File**: `scripts/chaos/run_guarcello_jc_phase5.py`

**Changes**: In the Stage A loop only:
- hoist `inv_dt_sq = 1.0/(dt_s*dt_s)` and `inv_two_dt = 1.0/(2.0*dt_s)` and
  multiply, replacing the elementwise divides at `:736` and `:738`;
- store `sin_phase[j]` in the loop at `:719` and reuse it in the recording
  block at `:752` instead of recomputing `math.sin` over all `n_branches`.

Expected gain is a few percent, not a multiple. It is included because the file
is open and the redundancy is unambiguous, and it is called out as small so
nobody later attributes a batching speedup to it.

### Success Criteria

**Automated**:
- `tests/test_chaos_batched_kernel.py`: a batch of `B` drives reproduces `B`
  sequential single-point runs to **bit-identity** on `times`, `voltage` and
  `branch_r`. Not `allclose` -- identity. The arithmetic is the same operations
  in the same order per lane; anything else means a lane is contaminated.
- A batch of one reproduces today's `_integrate_jc_compiled` output bit-identically.
- Existing `tests/test_chaos_phase5_jc_compare.py` still passes, except the
  known `4578`/`4558` bandwidth expectation.

**Manual**:
- Throughput table recorded for both backends at batch 1/2/4/6 with peak RSS.
- The `0.5975` prerequisite control point re-measured through the batched path
  gives the same off-lattice fraction as through the scalar path.

---

## Phase 3: Incoherent-Continuum Closure

### Overview

The only genuinely new physics in this plan. Above the second boundary, keep
solving the strong lines coherently -- pump harmonics and the retained torus
generator, which together still hold most of the power right up to collapse --
and represent the broadband residue by its **power spectral density** rather
than by more discrete unknowns. Close the two with an energy balance.

This is a research phase. It is planned to a first decision point, not to a
finished method, and its first deliverable is a falsification test.

### Why this shape and not another

The corpus dictates it. At collapse the residue is `2.4%` in the top-20 bins,
`26%` explained by the single best extra generator, and a floor `13-18 dB`
below the pump. A representation that adds discrete unknowns converges at the
rate that measurement implies, which is far too slow. A representation that
carries a *density* has the right number of degrees of freedom for a
continuum. Weak-turbulence closures are the standard tool for exactly this
situation and the repository has the coherent half already built.

### Changes Required

#### 1. Offline falsification test, before any solver work

**File**: `scripts/chaos/measure_continuum_closure.py` (new)

**Changes**: Operate entirely on the **stored** spectra -- 72 points, 4 devices,
no new integration. For each point:

- split the spectrum into the coherent set (pump lattice plus the fitted
  generator) and the residue;
- compute the residue's total power, its spectral centroid, its width, and its
  decay exponent against frequency offset from the pump;
- test whether the residue's *shape* is a function of a small number of
  coherent-side scalars (drive, on-lattice fraction, `branch_current_max_over_ic`,
  `min_cos_phi`) across devices.

**This is the go/no-go.** If the residue shape does not collapse onto a
low-parameter family across the corpus, a density closure has nothing to
close and the phase stops here having cost one script and no solver changes.
That outcome is a publishable negative result, not a failure.

#### 2. Closure formulation -- gated on 1

**File**: `docs/development/continuum_closure_formulation.md` (new)

**Changes**: Written only if the test in 1 passes. Specifies the residue's
parameterization, the energy-transfer term coupling it to the coherent lattice,
and how the coherent solve's Jacobian is augmented. Reviewed before any code.

#### 3. Prototype -- gated on 2

**File**: `src/twpa_solver/chaos/continuum.py` (new)

**Changes**: Smallest thing that produces a number: coherent lattice plus a
scalar residue power obeying the closure, solved on `ipm_2c_fixed` at
`f_p = 7.9 GHz` at one drive above collapse. Compared against the FDTD spectrum
at that point.

### Success Criteria

**Automated**:
- `tests/test_continuum_reduction.py` gates the spectral split on stored
  fixtures: coherent power plus residue power equals total power to `1e-12`
  relative, and the split is stable under FFT bin choice.

**Manual**:
- Phase 3 step 1 produces a written verdict with the fitted family and its
  residuals, or a written statement that no low-parameter family fits.
- If a prototype is built, its predicted residue power at one supercritical
  drive is compared against FDTD and the discrepancy is quoted, never suppressed.

---

## Testing Strategy

### Project Maturity Level

Active Development. The solver package under `src/` is production; the
`scripts/chaos/` tree is research tooling with real test coverage.

### Unit Tests

- `classify_from_multiplier` / `classify_from_spectrum`: table-driven from
  corpus rows, including the `UNDECIDED` band and the empty region between
  `on_lattice` `0.4620` and `0.8827`.
- `probe_multiplier`: asserts both half-planes are searched and that a verdict
  with `mode_overlap < 0.99` is reported as `UNDECIDED`, not as a regime.
- Batched kernel: bit-identity against sequential, batch-of-one identity
  against today's path, and one lane-contamination mutation (perturb one lane's
  initial state, assert the others are unchanged).
- Constant-reduction edits: covered by the bit-identity test above, which is
  what makes them safe to make.
- Coverage target: 70%, matching Active Development. Every new gate must be
  shown failing before it is shown passing.

### Integration / Manual Tests

- One full 7.9 GHz column through `route_column.py`, with the routing verdict
  overlaid on the known crossing at `-24.2334894 dBm`.
- Batched-kernel throughput table, both backends, recorded here.
- Prerequisite control point `0.5975` at `dt=0.01` on the live build.

---

## Rollback Plan

- **Phase 1** is additive: two new files plus one new test module. Delete them
  and nothing else changes; no existing code path imports the router.
- **Phase 2** touches one existing file. The batch entry point is new, the
  backend switch defaults to the measured winner, and the constant-reduction
  edits are guarded by a bit-identity test. Revert is the single commit;
  `_integrate_jc_compiled` keeps dispatching to Stage A either way.
- **Phase 3** step 1 is a read-only script over stored data and cannot affect
  any solver result. Steps 2 and 3 do not begin unless step 1 passes.

Each phase is a separate commit on `dev`. Nothing here is promoted to `main`;
`scripts/chaos/` is development tooling by policy.

---

## Sequencing

1. Prerequisite control point (~10 min). Without it Phase 3 has no trustworthy
   oracle and the timestep result stays confounded.
2. Phase 1 -- highest value per hour, no new physics, makes the existing
   period-1 and torus solvers dispatchable instead of manually selected.
3. Phase 2 -- pure cost. Required before Phase 3 can afford its reference runs.
4. Phase 3 step 1 -- one script, decides whether the rest of Phase 3 exists.

Phases 1 and 2 touch disjoint files and can proceed in either order.
