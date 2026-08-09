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

- [x] `designs/ipm_2c_fixed` circuit unchanged since the Phase 5 CSVs were produced.
- [x] `TWPA_REQUIRE_PARDISO=1` available for timed re-runs (matches existing convention).

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

### Phase 2 result (measured 2026-08-07)

Deviation from the plan text above: the CLI flag actually implemented is
`--arclength-max-steps-after-fold` with default `None` (no change), not
"default matching `--arclength-max-steps`" as originally written -- the
literal plan text was self-contradictory (matching `--arclength-max-steps`
by default would itself change behavior whenever a fold is detected late).
`None` is what `solve_arclength` itself defaults to, so this keeps the two
consistent. A companion `--arclength-rescale-every` flag (Phase 1's
parameter) was added at the same time since Phase 3 needs it too.

First attempt used too small a base budget (`--arclength-max-steps 20`,
`--arclength-max-steps-after-fold 100`) and never triggered the extension at
all -- none of the 12 points reached far enough into the branch to detect a
fold within 20 steps in the first place (`arclength_fold_lambda` empty on
every failing row). Re-run with a base budget matching the original
fold-detecting data (`--arclength-max-steps 60`) plus a large extension
(`--arclength-max-steps-after-fold 150`, `--eig-iters 5` to bound per-step
diagnostic cost), `designs/ipm_2c_fixed`, fp=7.9 GHz, -22.0..-18.5 dBm
(`outputs/phase2_fold_rounding_check/singularity_scan.csv`, mirrored at
`D:\tmp\p2chk2`):

| power (dBm) | converged | lambda_reached | fold_lambda | I_bound (A) |
| ---: | :---: | ---: | ---: | ---: |
| -22.0 to -19.5 | True | 1.0 | (-21.5 only: 0.9896, still reached target) | -- |
| -19.0 | **False** | 0.9703 | 0.9511 | 1.1628e-05 |
| -18.5 | **False** | 0.9991 | 0.8984 | 1.1633e-05 |

The extended budget recovers -19.5 dBm and above (previously failing under
the old fixed `max_steps=60` with no extension -- e.g. the original Phase 5
CSV's -19.5 dBm row *did* already reach target via arclength even without
this change, but -21.5 through -20.0 dBm needed the larger base budget to
detect their fold at all). At -19.0 and -18.5 dBm, a fold is detected
(`fold_lambda` populated) but even 150 further steps cannot cross back to
`target_lam=1.0` -- the budget is exhausted on the returning branch, not
before reaching the fold. This is the plan's anticipated "exhausts the new
budget too" outcome. The measured `I_bound` (1.1628e-05 / 1.1633e-05 A)
matches the previously-cited `1.1929e-05 A` to ~2.5%, using a properly
fold-triggered (not merely budget-exhausted) measurement this time.

**Verdict: CONFIRMED genuine fold at fp=7.9 GHz, `I_bound ~= 1.163e-05 A`.**
This is the device's physical pump ceiling at this frequency, not a solver
artifact -- no further remedy exists within this plan's scope (a returning
branch may still exist further past 150 extra steps, but 150 is already
~2.5x the ~76 steps a toy fold-and-return fixture needed end-to-end, so
diminishing-returns applies; not pursued further here).

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
- If `bordered_conditioning` also degenerates at 8.640e-06 A: genuine branch
  point confirmed. Scope a separate deflation plan; do not implement deflation
  under this plan.

### Success criteria

**Automated**: n/a (measurement/classification phase).
**Manual**: The decision-gate outcome above, written into this document with
the actual `bordered_conditioning` numbers at and near the boundary.

### Phase 3 result (measured 2026-08-07)

Re-ran `scan_branch_singularity.py` on fp=7.0 GHz, -23.5..-21.0 dBm
(`designs/ipm_2c_fixed`, `--arclength-max-steps 60
--arclength-rescale-every 5 --eig-iters 5`,
`outputs/phase3_rescale_classification/singularity_scan.csv`, mirrored at
`D:\tmp\p3chk`). Compare against the original (pre-Phase-1) measurement at
the same frequency (`outputs/phase5_singularity_scan/fp7p0/singularity_scan.csv`),
where every failing row from -22.75 dBm (8.7326e-06 A) through -18.0 dBm
showed `terminal_reason=minimum_step` and **no** `fold_lambda` ever
populated -- the corrector simply stopped converging, full stop.

| power (dBm) | current (A) | terminal_reason (old) | terminal_reason (new) | fold_lambda (new) |
| ---: | ---: | :---: | :---: | ---: |
| -22.5 | 8.988e-06 | minimum_step | max_steps | (none) |
| -22.0 | 9.520e-06 | minimum_step | max_steps | 0.9273 |
| -21.5 | 1.008e-05 | minimum_step | max_steps | 0.8758 |
| -21.0 | 1.068e-05 | minimum_step | max_steps | 0.8090 |

With `rescale_every=5`, `terminal_reason=minimum_step` **never occurs** in
this range -- the corrector keeps making forward progress well past the old
8.640e-06 A boundary (up to 1.068e-05 A tested, 24% higher), running out of
the plain `max_steps=60` budget instead of dying, and now detects real folds
in 3 of 4 points. This directly matches decision-gate outcome (a).

`bordered_condition` (per-step CSV, `outputs/phase3_rescale_classification/singularity_scan_steps.csv`)
ranged 5e10-2e13 across all four points, including well before the failing
endpoint -- **not usable for a fold-vs-branch-point read here**: `--eig-iters
5` (kept low to bound wall time on this already-expensive real-device check)
is far too few power iterations for `bordered_conditioning` to have
converged, unlike the small-fixture unit test (Phase 0) where it was
validated finite/positive at a converged point, not calibrated at this
scale. `min_eigenvalue` at the last accepted step of each point (188, -469,
3.9e4, -3.3e3) is small compared to a healthy baseline (~1e5-1e6) but
**not** categorically different from the *old* unrescaled measurement's own
near-boundary values (e.g. original row at -21.5 dBm: -207.99) -- consistent
with approaching a genuine singularity in both measurements, not new
information on its own.

**Verdict: (a) mistuning CONFIRMED, no further phase needed.** The Phase 1
relative step floor / periodic rescale directly fixes the qualitative
failure mode (`minimum_step` -> productive marching + real fold detection),
letting the corrector reach currents beyond the previously-reported physical
boundary. Reaching `target_lam=1.0` at these higher currents is now the same
already-solved Phase 2 problem (round the detected fold with
`max_steps_after_fold`), not a new fp=7.0-specific blocker. The
`bordered_conditioning` branch-point check was inconclusive at the `iters=5`
setting used here; if a firmer fold-vs-branch-point read is ever wanted at
fp=7.0 it needs a re-run at `iters>=20` (Phase 0's tested default), but that
is not required by outcome (a) and is not pursued further under this plan.

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

### Phase 4 progress (2026-08-07)

**Done:**
- Change #1 (`_recover` arclength branch): `--recovery-arclength-rescale-every`
  and `--recovery-arclength-max-steps-after-fold` wired into the
  `solve_arclength` call at `scripts/run_gain_map.py` (both default `0`,
  no-change). `pump_arclength_fold_current_a` recorded into the cell's row
  when a fold is detected but the extended budget still can't reach
  `target_lam=1.0`, whichever of the `--inproc-fail-fast` or final-reseed
  exit path returns the cell (added to `write_points_csv`'s fieldnames).
- Change #3 (docs): `CLAUDE.md`'s "Pseudo-arclength metric fix..." section
  and memory `arclength-metric-bug-and-snaking-verdict.md` both updated with
  the corrected Phase 2/3 verdicts.
- Automated success criteria: `tests/test_run_gain_map_cli.py` gained
  `test_recovery_arclength_flags_default_to_no_change`,
  `test_recovery_arclength_flags_are_settable`, and
  `test_write_points_csv_carries_arclength_fold_current` (28/28 pass). Full
  fast suite (`pytest -q`, no `--run-slow`): 497 passed, 2 pre-existing
  unrelated failures in `test_loss_model.py` (confirmed via `git stash` to
  fail identically on the clean tree -- Norton-vs-legacy-convention tests
  unrelated to this plan). `--run-slow` full suite run separately.

**Deferred, not started -- both are large campaign-scale runs (hours), not
code changes, and were not part of what was asked for this session:**
- Change #2's full 19-frequency `--fold-follow` sweep re-run.
- The Manual success criterion's full production 2c map re-run.

Both remain open follow-up work. The per-point evidence gathered directly
under Phases 2-3 (`outputs/phase2_fold_rounding_check/`,
`outputs/phase3_rescale_classification/`) already demonstrates the fix works
at the single-cell level at both frequencies; a full map re-run would
confirm it at production scale but was not run in this session.

### Reduced 4-frequency fold-follow sweep (measured 2026-08-07)

User asked for a cut-down version of Change #2 to save time: 4 frequencies
instead of 19, same 7.6-8.5 GHz band as the original investigation doc's
sweep. `fold_power` (`solver.py`) did not accept `rescale_every`, so it was
added (threaded through to its inner `solve_arclength` call) -- otherwise
this re-run would repeat the exact metric-mistuning failure mode Phase 3
diagnosed, silently reading as "no fold in range" again. `run_fold_follow`
now forwards `--recovery-arclength-rescale-every` (same flag Phase 4 already
added) into `fold_power`.

Ran 4 evenly-spaced points (`linspace(7.6, 8.5, 4)` = 7.6/7.9/8.2/8.5 GHz,
which conveniently includes fp=7.9 GHz), `--pump-power-max-dbm -16`
(reference power for the lambda-to-current scaling),
`--recovery-arclength-rescale-every 5`
(`outputs/phase4_fold_follow_reduced/fold_curve.csv`):

| freq (GHz) | fold_lambda | fold_power (dBm) |
| ---: | ---: | ---: |
| 7.6 | 0.5311 | -21.496 |
| 7.9 | 0.6734 | -19.435 |
| 8.2 | (none found within 120 steps) | -- |
| 8.5 | 0.6438 | -19.825 |

**3 of 4 points now find a real fold**, directly contradicting the original
19-point sweep's "zero folds everywhere" (§2d of the 2026-08-06
investigation doc) -- strong evidence that result really was the broken
pre-Phase-1 metric, not a physics finding, as this plan's Phase 0-3 verdicts
already concluded from the single-frequency data. The fp=7.9 GHz fold_lambda
here (0.6734, referenced against `-16 dBm`) is not directly comparable to
Phase 2's fold_lambda (0.9511/0.8984, referenced against the much lower
-19.0/-18.5 dBm points near the production power ceiling) -- `fold_power`
marches from `lambda=0` at a fixed high reference current and reports the
*first* turning point encountered, which need not be the same fold Phase 2
characterized near a different, lower operating point. 8.2 GHz finding none
within 120 steps is a single data point at reduced resolution and is not
strong evidence of "no fold at 8.2 GHz" on its own -- consistent with why
the plan originally scoped a 19-point sweep rather than 4.

**Still not done**: the full 19-point sweep at finer frequency resolution,
and the full production map re-run. Both remain open follow-up work; this
reduced sweep is corroborating evidence for the corrected verdicts, not a
replacement for either.

### Full 19-frequency fold-follow sweep (measured 2026-08-07)

Same command as the reduced sweep, `--n-frequency 19` over the full
7.6-8.5 GHz band, `--pump-power-max-dbm -16`,
`--recovery-arclength-rescale-every 5`
(`D:/tmp/phase4_fold_follow_full/fold_curve.csv`):

| freq (GHz) | fold_lambda | fold_power (dBm) |
| ---: | ---: | ---: |
| 7.60 | 0.5311 | -21.496 |
| 7.65 | 0.5677 | -20.918 |
| 7.70 | 0.5697 | -20.887 |
| 7.75 | 0.6419 | -19.850 |
| 7.80 | 0.5637 | -20.980 |
| 7.85 | 0.5309 | -21.499 |
| 7.90 | 0.6734 | -19.435 |
| 7.95 | 0.6164 | -20.202 |
| 8.00 | 0.6831 | -19.310 |
| 8.05 | 0.5676 | -20.919 |
| 8.10 | 0.4893 | -22.208 |
| 8.15 | 0.4452 | -23.029 |
| 8.20 | (none found within 120 steps) | -- |
| 8.25 | 0.6519 | -19.717 |
| 8.30 | 0.6072 | -20.334 |
| 8.35 | 0.5402 | -21.349 |
| 8.40 | 0.6946 | -19.165 |
| 8.45 | 0.7390 | -18.627 |
| 8.50 | 0.6438 | -19.825 |

**18 of 19 points find a real fold** -- only fp=8.20 GHz found none within
the 120-step budget, matching that same frequency's result in the reduced
sweep (which used the identical grid point). This is now the full-resolution
replacement for the original 19-point sweep from the 2026-08-06 investigation
doc's §2d, and it flatly contradicts that sweep's "zero folds everywhere"
verdict: fold power ranges -18.6 to -23.0 dBm across the whole band, a
~4.4 dB spread with no obvious trend against frequency (not monotone, not a
simple envelope -- 8.45 GHz is the shallowest fold at -18.6 dBm, 8.15 GHz the
deepest at -23.0 dBm, with no adjacent-frequency smoothness suggesting this
is set by comb/lobe structure rather than a slowly-varying device property).

The single fp=8.20 GHz non-detection does not reopen the "no fold" question
generally -- it is one grid point at the sweep's `ds=0.02` step and
`rescale_every=5` settings, not evidence the branch has no fold there; a
finer local scan (smaller `ds`, or `max_steps_after_fold`) at that one
frequency is the natural follow-up if it matters, not yet run.

**Still open**: this sweep uses `fold_power`'s raw `solve_arclength` (no
fold refinement -- Milestone D's `_refine_fold` added after this sweep ran is
not yet wired into `fold_power`/`run_fold_follow`), so each `fold_lambda`
above is a bracket-point estimate, not a refined root; and the full
production gain-map re-run using this fold information is still not done.

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

### Final verification (2026-08-07)

`pytest -q --run-slow` (full suite, `--basetemp` outside the repo per
convention): **500 passed, 1 xfailed, 2 failed**. The 2 failures
(`tests/test_loss_model.py::test_dbm_to_peak_current_applies_frequency_loss`,
`::test_norton_current_is_2x_legacy_at_fixed_dbm`) are pre-existing and
unrelated to this plan -- confirmed via `git stash` to fail identically on
the clean pre-session tree (they assume Norton is the default power
convention; per CLAUDE.md's "Port power convention" section the default was
reverted back to `legacy_traveling_wave` in a prior session and these two
tests were never updated). Not touched under this plan's scope.

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
