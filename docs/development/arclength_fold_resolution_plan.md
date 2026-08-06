# Arclength fold/wall resolution plan

## Goal

Reach `lambda=1` (or a defensible physical fold report) on both `designs/ipm_2c_fixed`
high-power boundaries currently mislabeled by
`arclength_metric_fix_and_fold_test_function_plan.md` Phase 5: fp=7.9 GHz
("confirmed numerical wall") and fp=7.0 GHz ("snaking"). Both verdicts rest on
a measurement-point defect in `scan_branch_singularity.py` and must be
re-derived before any remedy is chosen.

## Current state analysis

`src/twpa_solver/pump/solver.py::solve_arclength` (1110-1287) does pseudo-arclength
continuation with a state-scale-corrected metric (Phase 1 of the prior plan).
Confirmed behavior from `outputs/phase5_singularity_scan/{fp7p0,fp7p9}/singularity_scan.csv`:

- **fp=7.9 GHz**: `arclength_fold_lambda` populated on all 5 failing rows
  (0.977 -> 0.870 as power rises), `terminal_reason=max_steps` (never
  `minimum_step`), `peak_i_over_ic` 0.90-0.94. `I_bound = current_a * fold_lambda`
  is constant to 0.2% across 5 independent marches: **1.1929e-05 A**.
- **fp=7.0 GHz**: `arclength_fold_lambda` empty on all 20 failing rows,
  `terminal_reason=minimum_step` (corrector collapse, no `lam_dot` sign flip),
  `peak_i_over_ic` flat 0.44-0.48. `I_bound = current_a * lambda_reached` is
  constant to 0.2% across 20 independent marches spanning a 1.72x range of
  injected current: **8.640e-06 A**.

Both are one sharp, reproducible physical-current boundary, not evidence of
snaking (which would not reproduce one boundary current from 20 different
starting points).

**Why the existing verdicts are wrong**: `scan_branch_singularity.py:158-159`
measures `jacobian_min_eigenvalue`/`jacobian_det_signature` at `X`, which for
a failing row is `X_arc` at `lam_reached` (`scan_branch_singularity.py:143-150`)
-- a point *below* the boundary, on the healthy branch. Comparing that to the
converged baseline is comparing healthy points to healthy points. Separately,
`jacobian_min_eigenvalue` (`singularity.py:62-111`, inverse power iteration,
20-40 iters, nonsymmetric Rayleigh quotient) disagrees with itself by 2 orders
of magnitude between adjacent *converged* fp=7.0 points (+1.099e7, -1.072e5) --
not converged enough to anchor a verdict on its own.

`run_gain_map.py::_recover`'s `fold_policy=arclength` branch (2024-2088) only
consumes `info["reached_target"]`; it discards `fold_lambda`, never continues
past a detected fold, and never calls the singularity functions.

## What we're NOT doing

- Deflation (Farrell et al.). No code for it exists in `src/` (confirmed by
  grep). Only in scope if Phase 3 classifies fp=7.0 as a genuine branch point.
- JosephsonCircuits.jl comparison of any kind (`jc-is-not-a-reference`).
- Re-litigating Phase 1-4 of the prior plan (metric fix, bordered solve,
  corrector hardening) -- those are taken as correct; this plan only adds a
  correctly-sited measurement and acts on what it shows.
- Any change to `run_gain_map.py`'s non-arclength recovery paths (`alt_parent`,
  `bridge`, `ladder`, `cross_axis`, `bridge_gate`).

## Prerequisites

- [ ] `designs/ipm_2c_fixed` circuit unchanged since the Phase 5 CSVs were produced.
- [ ] `TWPA_REQUIRE_PARDISO=1` available for timed re-runs (matches existing convention).

---

## Phase 0: Correctly-sited singularity measurement

### Overview

Move the singularity test functions inside the continuation loop, sampled at
actually-visited accepted points, and replace the noisy eigenvalue estimator.
This is a hard prerequisite -- Phases 2 and 3 both read its output to decide
which remedy applies.

### Changes required

#### 1. `solve_arclength` per-step sampling

**File**: `src/twpa_solver/pump/solver.py`
**Changes**: Add an optional `on_step: Callable[[np.ndarray, float, float], None] | None = None`
parameter to `solve_arclength` (default `None`, so existing callers are
unaffected). Call it with `(Xc, lamc, step_size)` immediately after a corrector
step converges (after line 1241, before the tangent update at 1247), so a
caller can measure the Jacobian at the point the solver actually accepted --
not the final endpoint. Do not call it on rejected/halved attempts.

#### 2. Better eigenvalue estimator

**File**: `src/twpa_solver/pump/singularity.py`
**Changes**: Replace the inverse-power/Rayleigh loop in `jacobian_min_eigenvalue`
with `scipy.sparse.linalg.eigs(A_operator, k=1, sigma=0, which="LM", OPinv=factor_as_linop)`,
reusing the existing `factor.solve` as `OPinv` (shift-invert Arnoldi around
`sigma=0` finds the eigenvalue nearest zero directly, rather than hoping 20
power iterations converged). Fall back to the current inverse-power loop only
if `eigs` raises (e.g. `ArpackNoConvergence`), tagging the result so callers
can tell which estimator produced it. Keep the function signature identical.

#### 3. Bordered-conditioning diagnostic

**File**: `src/twpa_solver/pump/singularity.py`
**Changes**: New function `bordered_conditioning(problem, X, lam, Xdot, lam_dot, state_scale) -> float`
that assembles the same bordered system `solve_arclength`'s corrector uses
(`bordered_solve_refined`'s inputs) and returns its estimated condition number
(via the factor's own reciprocal-condition-number estimate if the backend
exposes one, else a cheap power-iteration ratio on the bordered operator). A
genuine fold has `J` singular but the *bordered* system well-conditioned
(bordering exists precisely to regularize a rank-1 degeneracy). A branch point
(two solution branches crossing) has the bordered system going singular too --
bordering cannot regularize a rank-2 degeneracy. This is the direct
fold-vs-branch-point discriminator Phase 3 needs.

#### 4. Rebuild `scan_branch_singularity.py` on the new hook

**File**: `scripts/scan_branch_singularity.py`
**Changes**: Pass `on_step` into `solve_arclength` calls (143-147) to collect
one singularity sample per accepted continuation point instead of one sample
at the endpoint (158-159). Write all per-step samples to a new
`singularity_scan_steps.csv` (columns: `point_index, step, lam, min_eigenvalue,
min_eigenvalue_estimator, bordered_condition, det_sign, log_abs_det`) alongside
the existing per-power-point CSV.

### Success criteria

**Automated**: `pytest tests/test_advanced_continuation.py -k arclength`,
new `tests/test_singularity_measurement.py` (unit test: `on_step` fires once
per accepted step on a small fixture problem with a known number of steps;
`eigs`-based estimate matches a dense `numpy.linalg.eigvals` reference on a
small dense fixture to 1e-8 relative; `bordered_conditioning` returns a finite
number on a converged fixture point).
**Manual**: Re-run `scan_branch_singularity.py` on fp=7.9 GHz, -19.5..-18.0
dBm (5 points, small range for a quick check), confirm
`singularity_scan_steps.csv` has multiple rows per failing point (not one) and
that `min_eigenvalue` trends toward zero near the fold_lambda row rather than
sitting flat at the failing-point endpoint value.

---

## Phase 1: Adaptive arclength metric

### Overview

`state_scale` is derived once from the initial tangent (`solver.py:1170-1179`)
and never revisited. As the branch approaches a fold the tangent rotates
toward the state direction; a metric balanced at the anchor point is not
necessarily balanced there. Needed before Phase 2 (fold rounding, which must
walk stably through the region where this matters most) and Phase 3 (a
mistuned metric is one of the two candidate explanations for fp=7.0's
corrector death).

### Changes required

#### 1. Periodic rescale

**File**: `src/twpa_solver/pump/solver.py`
**Changes**: In the main loop (1191-1287), after every `N` accepted steps
(new parameter `rescale_every: int = 10`, `0` disables -- preserves current
behavior exactly when unset), recompute `state_scale` from the current tangent
(`Xdot`, `lam_dot`) using the same formula as the initial derivation
(1170-1179), and rebuild the `metric_x` closure. Rescaling must not change
`X`, `lam`, or already-accepted history -- only the metric used for
subsequent corrector/tangent calculations. Record the number of rescales in
`info`.

#### 2. Relative step floor

**File**: `src/twpa_solver/pump/solver.py`
**Changes**: Replace the absolute `if step_size < 1e-4` (1238) with a relative
floor `if step_size < ds_initial * 1e-6`. An absolute `1e-4` against
`ds_initial=0.02` (the production `_recover` value, `run_gain_map.py:2066`)
allows only 8 halvings; a relative floor scales with whatever `ds` a caller
picks and matches the existing docstring note in `_recover` (2054-2063) that
0.02 was chosen empirically against a measured convergence radius as narrow
as 0.01.

### Success criteria

**Automated**: `tests/test_advanced_continuation.py` -- new case verifying
`rescale_every=0` reproduces existing byte-identical output on a fixture (no
behavior change when disabled), and a new case with `rescale_every=5` on a
fixture with an artificially stiff late-branch region converges where
`rescale_every=0` did not.
**Manual**: none (fixture-covered).

---

## Phase 2: Fold rounding (fp=7.9 GHz type)

### Overview

fp=7.9 GHz has a real fold (`arclength_fold_lambda` populated, `lam_dot` sign
flip, `peak_i_over_ic` up to 0.94). `solve_arclength` currently detects the
flip (1266-1267) but then keeps marching under the *same* `max_steps` budget
and exhausts it (`terminal_reason=max_steps` on every failing row) instead of
continuing onto the returning branch to look for a second crossing of
`target_lam`.

### Changes required

#### 1. Post-fold step budget

**File**: `src/twpa_solver/pump/solver.py`
**Changes**: New parameter `max_steps_after_fold: int | None = None` (default
`None` = no change from current `max_steps` behavior). When a fold is detected
(`info["fold_lambda"]` newly set), reset a local step counter and continue up
to `max_steps_after_fold` additional steps looking for the existing
target-crossing check (1273-1282), which already fires on any crossing in
either direction -- no change needed there.

#### 2. Wire into diagnostic driver first

**File**: `scripts/scan_branch_singularity.py`
**Changes**: Add `--arclength-max-steps-after-fold` (default matching
`--arclength-max-steps`, i.e. no behavior change until explicitly raised) so
Phase 2 can be measured standalone before touching production `_recover`.

### Success criteria

**Automated**: `tests/test_advanced_continuation.py` -- fixture with a known
fold-and-return branch (construct via a small polynomial-residual toy problem
with an analytic fold, if the existing fixture harness supports one; else a
2-DOF harmonic-balance fixture with `Ic` set low enough to fold within a few
dBm) confirms `reached_target=True` after fold when `max_steps_after_fold` is
large enough, and confirms `reached_target=False` with `max_steps_after_fold=0`
(pinned regression against current behavior).
**Manual**: Re-run `scan_branch_singularity.py --arclength-max-steps-after-fold 200`
on fp=7.9 GHz -19.0..-18.0 dBm. Report whether a returning branch reaches
`target_lam=1.0` (if yes: report the returning-branch pump waveform as the
high-power solution at this frequency) or exhausts the new budget too
(if so: report `I_bound=1.1929e-05 A` as the device's physical pump ceiling at
7.9 GHz, not a solver failure, and stop -- no further remedy exists for a
confirmed fold beyond going around it).

---

## Phase 3: Classify fp=7.0-type corrector death

### Overview

fp=7.0 GHz never reports a `lam_dot` sign flip; the corrector simply stops
converging (`minimum_step`) at a reproducible current. Two live hypotheses:
(a) mistuned metric/step-floor (Phase 1 fixes this class), or (b) a genuine
branch point where bordering itself degenerates (needs Phase 0's
`bordered_conditioning`, and is out of scope for this plan -- see "what we're
not doing").

### Changes required

#### 1. Re-run with Phase 0 + Phase 1 fixes

**File**: n/a (measurement only)
**Changes**: Re-run `scan_branch_singularity.py` on fp=7.0 GHz, full
-22.75..-18.0 dBm range, with Phase 1's `rescale_every` enabled and Phase 0's
`bordered_conditioning` sampled per accepted step. No code change beyond what
Phases 0-1 already added.

#### 2. Decision gate

**File**: `docs/development/arclength_fold_resolution_plan.md` (this doc,
append a "Phase 3 result" section after running)
**Changes**: Record the outcome:
- If Phase 1's relative step floor / periodic rescale lets the corrector push
  past 8.640e-06 A: mistuning confirmed, done, no further phase needed.
- If `bordered_conditioning` stays well-conditioned through the boundary but
  the corrector still dies: something else is wrong (candidate: Newton
  step-length control, not covered by this plan -- open a follow-up).
  the corrector still dies: something else is wrong (candidate: Newton
  step-length control, not covered by this plan -- open a follow-up).
- If `bordered_conditioning` also degenerates at 8.640e-06 A: genuine branch
  point confirmed. Scope a separate deflation plan; do not implement deflation
  under this plan.

### Success criteria

**Automated**: n/a (measurement/classification phase).
**Manual**: The decision-gate outcome above, written into this document with
the actual `bordered_conditioning` numbers at and near the boundary.

---

## Phase 4: Production wiring

### Overview

Only proceed with this phase for whichever of fp=7.9/fp=7.0 behavior Phases 2-3
resolved (rounded fold, or confirmed mistuning fix). Do not wire an unresolved
branch-point case into production recovery.

### Changes required

#### 1. `_recover` arclength branch

**File**: `scripts/run_gain_map.py`
**Changes**: In the `fp == "arclength"` block (2024-2088), pass through
`rescale_every` (Phase 1) and `max_steps_after_fold` (Phase 2) to the
`solve_arclength` call at 2065-2067, as new CLI flags
`--recovery-arclength-rescale-every` (default matching Phase 1's off-by-default,
`0`) and `--recovery-arclength-max-steps-after-fold` (default `0`, no change
from current behavior). If `info["fold_lambda"]` is set but `reached_target`
is `False` even after Phase 2's extension, record `info["fold_lambda"] *
injected_current` into the row (new column
`pump_arclength_fold_current_a`) so a map cell reports the physical boundary
instead of a bare failure.

#### 2. Re-run 19-frequency fold-follow sweep

**File**: n/a (campaign re-run)
**Changes**: Re-run the full `--fold-follow` sweep referenced in
`2c_convergence_arclength_and_port_convention_investigation_2026-08-06.md`
§2d under the fixed instrument (Phases 0-2). That section's "zero folds
everywhere" was measured on the pre-fix metric and has never been re-run.

#### 3. Update docs

**File**: `CLAUDE.md`
**Changes**: Replace the "Pseudo-arclength metric fix and fold-vs-numerical
measurement" section's fp=7.9/"confirmed numerical" and fp=7.0/"SNAKING"
verdicts with the corrected classification from Phases 2-3 of this plan, and
the new `I_bound` figures.

**File**: memory `arclength-metric-bug-and-snaking-verdict.md`
**Changes**: Supersede with corrected verdict per the memory-system's own
"update or remove memories that turn out to be wrong" rule.

### Success criteria

**Automated**: `tests/test_run_gain_map_cli.py` -- new case covering the two
new CLI flags default to current behavior; `pytest -q` full suite green.
**Manual**: A production 2c map re-run over the fp=7.0/7.9 GHz band showing
either wider PASS coverage (if fold rounding/mistuning-fix succeeded) or a
correctly-labeled fold boundary in place of a bare FAILED cell.

---

## Testing strategy

### Project maturity level

Active development (research solver, `--run-slow` gate already exists for the
full physics suite).

### Unit tests

- `tests/test_advanced_continuation.py`: `on_step` hook firing (Phase 0),
  `rescale_every` no-op-when-zero and stiff-fixture-converges-when-nonzero
  (Phase 1), fold-and-return fixture (Phase 2).
- New `tests/test_singularity_measurement.py`: `eigs`-based estimator vs dense
  reference, `bordered_conditioning` finiteness and fold-vs-branch-point
  fixture cases (construct two small fixtures: one with a known simple fold,
  one with a known branch point, if such fixtures can be built cheaply from
  the existing 2-DOF harmonic-balance test harness -- else flag as a manual
  check against the fp=7.9 vs fp=7.0 real data).
- `tests/test_run_gain_map_cli.py`: new flags default-preserving (Phase 4).

### Integration/manual tests

- `scan_branch_singularity.py` re-runs at fp=7.9 and fp=7.0 GHz per phase
  (Phases 0, 2, 3 manual steps above).
- Full `--run-slow` suite before any production wiring lands (Phase 4).

---

## Rollback plan

Every new parameter (`on_step`, `rescale_every`, `max_steps_after_fold`, new
CLI flags) defaults to current behavior. Phases 0-3 touch only
`singularity.py`, `solver.py` (additive parameters), and the diagnostic
driver `scan_branch_singularity.py` -- none of these are in the production
`run_gain_map.py` hot path until Phase 4 explicitly wires them in behind new
flags. Rollback is: revert the Phase 4 commit(s); Phases 0-3 can stay (they
change no default behavior) or revert independently per phase if a phase's
manual verification contradicts its hypothesis.
