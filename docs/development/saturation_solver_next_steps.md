# Implementation Plan: Saturation Solver — Next Steps

Continuation of `docs/development/saturation_solver_plan.md`. Phases 0–5 of that plan are
complete and validated; exp20/21/22 have all produced results. This plan closes the gap
between what that plan specified and what the driver actually emits.

### Goal

Every published compression number is produced by the machinery the original plan
specified — a real refined P1dB, the plan's depletion baseline, and closed power balance —
and the two largest caveats (basis convergence, dynamic stability) are either discharged or
quantified.

### Current State Analysis

**The central finding: most of the work below is already written and tested, but never
reaches the driver or the experiment scripts.** Four plan-specified components exist,
are exported from `src/twpa_solver/multitone/__init__.py`, and are dead code in
production:

| Component | Location | Plan ref | Status |
| --- | --- | --- | --- |
| `depletion_only_model` | `multitone/compression_curve.py:28` | Phase 6 §2 | imported by `compression.py:15`, never called by the driver, never written to CSV |
| `refine_p1db` | `multitone/compression_curve.py:35` | Phase 6 §2 | imported by `compression.py:16`, exercised only on a synthetic lambda in `tests/test_compression.py:30` |
| `power_balance` | `multitone/observables.py:121` | Phase 7 §2 | returns `power_balance_rel_err`, `manley_rowe_photon_flux`, `manley_rowe_rel_err`; not emitted per point |
| `scripts/multitone_convergence_study.py` | — | Phase 7 §1 | exists, never run; no output directory |
| `scripts/floquet_stability_sweep.py` | — | Phase 8 | exists, never run |

`compression_points.csv` currently carries 15 columns; none of `compression_model_depletion_only`,
`power_balance_rel_err`, or `manley_rowe_rel_err` are among them.

**Three consequences, in severity order.**

1. **exp22's depletion baseline is invented and contradicts the plan.**
   `experiments/exp22_spatial_attribution.py:50` computes
   `depletion_only = gain[0] + 2.0 * depletion` — additive in dB, no comment, no derivation.
   The plan (Phase 6 §2, line 437) specifies
   `depletion_only_model(G_lin, P_s, P_p) = G_lin / (1 + 2 G_lin P_s / P_p)` in **linear**
   gain, which is implemented. `pump_depletion_db` is an amplitude ratio
   (`scripts/run_compression.py:569-572`), so the additive dB form is dimensionally a
   different model.

   Measured on the exp22 artifacts, at the first sampled point above P1dB:

   | device | actual compression | exp22 additive baseline | exponential-gain baseline `g0·10^(d/10)` |
   | --- | ---: | ---: | ---: |
   | jtwpa | 1.352 dB | 0.704 (→ 0.649 "unexplained") | 2.143 (→ 0.79 **over**predicted) |
   | 2c | 2.172 dB | 2.778 (→ 0.606 "compensated") | 4.356 (→ 2.18 **over**predicted) |

   Under the additive baseline the two devices point in opposite directions, which is the
   basis of the current write-up's device-contrast conclusion. Under a defensible baseline
   they point the same way. **That conclusion is a baseline artifact and must not ship.**
   The claims that survive either choice are "depletion alone does not reproduce the
   multitone result" and "spatial phase evolution is required".

2. **No P1dB in exp20/21/22 was refined.** `scripts/run_compression.py` uses
   `_interpolate_p1db_current` (log-linear interpolation between grid points). With 25
   points over 5.5 decades the grid step is 1.69× in current = **4.6 dB in power**. The plan
   is explicit: "No spline-only answer." There is no `--p1db-power-tol-db` flag.

3. **Basis selection never met the plan's acceptance test.** Phase 7 §1 requires
   `|ΔP1dB| < 0.2 dB` between the top two settings of every knob, including an odd-vs-dense
   `h` comparison ("mandatory — even sectors are not assumed negligible"). Production bases
   (S=10/6/10) were instead chosen by JC-reference gating on *small-signal gain*, which does
   not bound P1dB error.

**Engineering state.** The scatter-map rewrite (7× memory: ~631 → ~94 MB at S=10) and the
banded backend are committed and gated by `tests/test_fast_coupled_assembly.py` (8 tests,
mutation-verified). `precond_reuse` was measured and is a dead end — keep the default at 1:

| reuse | factorizations | GMRES iterations | wall |
| ---: | ---: | ---: | ---: |
| 1 | 59 | 243 | 3313 ms |
| 2 | 33 | 1084 | 3574 ms |
| 3 | 30 | 1229 | 3545 ms |

Converged gain was bit-identical across all three. Banded is ~1.15× slower per solve but
its smaller footprint buys one more worker.

Not done: `multitone/resources.py::fast_coupled_footprint` is still calibrated to the
pre-rewrite peak, so the worker guard refuses runs that now fit.

### What We're NOT Doing

- Not re-running exp20/21/22 wholesale. Phase 2 recomputes baselines from **existing**
  artifacts; only Phase 3 re-runs solves, and only at the P1dB bracket.
- Not changing any solver numerics. Every converged root must stay bit-identical; the
  changes here are diagnostics, output plumbing, and study drivers.
- Not touching the pump path, `run_gain_map.py`, or the 7-design JC parity.
- Not building an automatic stability classifier. Phase 6 quantifies the caveat on selected
  operating points; `stability_status` stays `NOT_CHECKED` for anything not explicitly swept.
- Not revisiting `precond_reuse` — measured, closed.

### Prerequisites

- [ ] Baseline test state confirmed: full suite ends at exactly **4 pre-existing failures**
      (`test_column_matrices_tracer` ×2, `test_loss_model` ×2 — the latter from a missing
      `docs/loss_A10.csv`). Confirm this before starting so new failures are unambiguous.
- [ ] Run tests with `--basetemp` off the repo (Windows ACL issue), e.g.
      `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\tw_next`
- [ ] Free RAM check before any campaign: one jtwpa S=10 worker peaked at 3.04 GB before
      the scatter rewrite; re-measure in Phase 1 rather than assuming.

---

## Phase 1: Recalibrate the worker guard

### Overview

The memory work is done but unusable — the footprint estimator still predicts the old peak,
so `_frequency_worker_limit` refuses or under-provisions runs that now fit. Short phase;
unblocks every later campaign.

### Changes Required

#### 1. Re-measure the per-worker peak
**File**: none (measurement)
**Changes**: Measure actual peak RSS for one worker on jtwpa S=10, both `pardiso` and
`banded`, after the scatter rewrite. Record the numbers; they are the calibration target.

#### 2. Recalibrate the estimator
**File**: `src/twpa_solver/multitone/resources.py`
**Changes**: Update the documented constants (`_SCATTER_BUILD_OVERHEAD` in particular — the
scatter map is no longer a sparse matrix and the transient build cost it modelled is gone)
so `fast_coupled_footprint` tracks the new measurement. Add a `factor_backend` argument so
`banded` reports its smaller factor footprint instead of PARDISO's.

#### 3. Update the calibration gate
**File**: `tests/test_multitone_resources.py`
**Changes**: Update `MEASURED_STEADY_GB` / `MEASURED_PEAK_GB` to the Phase 1 numbers. Keep
`test_fast_coupled_estimate_is_conservative_against_measured_peak` — the estimate must never
read below measurement, since underestimating is the direction that OOMs.

#### 4. Banded selection policy
**File**: `scripts/run_compression.py`
**Changes**: `--factor-backend` should pick `banded` when it raises the affordable worker
count and `pardiso` otherwise. Single-frequency runs (exp22) get `pardiso`; multi-frequency
sweeps under a tight budget get `banded`. Make the choice explicit in the summary JSON.

### Success Criteria
**Automated**: `pytest tests/test_multitone_resources.py tests/test_run_compression_cli.py`
**Manual**: 3 concurrent jtwpa S=10 workers run to completion under a 7 GB ceiling without
swapping; record peak RSS. If 3 does not fit, report the measured shortfall rather than
forcing it — the guard refusing is the correct behaviour.

---

## Phase 2: Correct the compression baseline

### Overview

Replace exp22's invented baseline with the plan's model, emit it per point, and add the
spatial null that the exp22 data can already support. **This phase changes a physics
conclusion — do not skip the re-analysis step.**

### Changes Required

#### 1. Emit the plan's depletion model per point
**File**: `scripts/run_compression.py`
**Changes**: Call `depletion_only_model(gain_linear, signal_power, pump_power)` at each
converged point and write `compression_model_depletion_only` to `compression_points.csv`
alongside the existing `compression_db`. Pump power comes from the operating point; signal
power from `signal_power_dbm`. Document in the summary that it is a trend baseline, not an
acceptance oracle (the plan's wording).

#### 2. Replace the exp22 baseline
**File**: `experiments/exp22_spatial_attribution.py`
**Changes**: Delete the `gain[0] + 2.0 * value` line (`:50`). Plot the emitted
`compression_model_depletion_only` column instead. If the column is absent (older
artifacts), fail loudly rather than silently falling back.

#### 3. Spatial depletion null
**File**: `src/twpa_solver/multitone/observables.py`
**Changes**: Add `spatial_depletion_null(spatial_rows, small_signal_rows)` — re-integrate
gain along the chain using the **measured local pump amplitude** at the operating point
while holding `delta_k_eff_rad_per_cell` at its small-signal value. This isolates depletion
from phase mismatch properly instead of comparing against a lumped closed form. All required
inputs already exist in `spatial_profiles.csv` (`pump_flux_abs`, `delta_k_eff_rad_per_cell`,
`theta_rad`, `signal_flux_abs` per branch per operating point).

#### 4. Re-analysis
**File**: `docs/` (new short results note)
**Changes**: Recompute the exp22 attribution for jtwpa and 2c under both the closed-form and
spatial nulls from the **existing** artifacts. State plainly which conclusions survive.
Expect the device-contrast claim to change.

### Success Criteria
**Automated**: `pytest tests/test_compression.py tests/test_multitone_observables.py`
- `compression_model_depletion_only` matches `depletion_only_model` called directly on the
  same inputs, to 1e-12.
- `spatial_depletion_null` reproduces the small-signal gain to <0.05 dB when evaluated at
  the zero-signal operating point (it must be a no-op there — this is the gate that catches
  a wrong integration constant).
- A test asserts exp22's plotting path raises on artifacts lacking the new column.

**Manual**: Recomputed attribution table for jtwpa and 2c under both nulls, with an explicit
statement of which of the four write-up conclusions survive.

---

## Phase 3: True P1dB refinement

### Overview

Wire the existing bracketed refinement into the driver so P1dB stops being an interpolation
between points 4.6 dB apart.

### Changes Required

#### 1. Driver flag and wiring
**File**: `scripts/run_compression.py`
**Changes**: Add `--p1db-power-tol-db` (default 0.1). After the coarse sweep locates the
first `compression_db >= 1.0` crossing, call `refine_p1db` with an evaluator that runs a
**real nonlinear solve** at each trial power, warm-started from the nearest converged state,
until the bracket is under tolerance. Keep `_interpolate_p1db_current` as the fallback when
refinement is disabled (`--p1db-power-tol-db 0`) or when no bracket exists, and record which
path produced the number in the summary (`p1db_method`).

#### 2. Non-monotonic reporting
**File**: `scripts/run_compression.py`
**Changes**: Emit the plan's `first_1db_crossing_dbm`, `number_of_crossings`, and
`nonmonotonic_compression` (all already produced by `CompressionCurve`) into the summary.
Four exp21 frequency runs showed non-monotone compression curves; this is currently invisible
in the artifacts.

#### 3. Quantify the interpolation error
**File**: none (measurement)
**Changes**: On one jtwpa and one 2c frequency, compare refined vs interpolated P1dB. This
number determines whether exp20/21 need re-running or a documented correction suffices.

### Success Criteria
**Automated**: `pytest tests/test_compression.py tests/test_run_compression_cli.py`
- Refinement on a synthetic monotone curve returns the analytic crossing to <0.01 dB
  (already covered by `tests/test_compression.py:30`; extend to the driver path).
- `p1db_method` is recorded and the fallback path is reachable and tested.
- Non-monotone synthetic curve sets `nonmonotonic_compression=True` and reports the first
  crossing.

**Manual**: Refined-vs-interpolated delta on two real operating points, with a
recommendation on whether exp20/21 must be re-run.

---

## Phase 4: Conservation diagnostics

### Overview

`power_balance` is built and unused. Emitting it turns every existing point into a
self-check, and it is the cheapest evidence that deep-saturation solutions are physical.

### Changes Required

#### 1. Emit per point
**File**: `scripts/run_compression.py`
**Changes**: Call `power_balance` at each converged point; write `power_balance_rel_err`,
`manley_rowe_photon_flux`, and `manley_rowe_rel_err` to `compression_points.csv`.

#### 2. Threshold reporting, not gating
**File**: `scripts/run_compression.py`
**Changes**: Record `max_power_balance_rel_err` in the summary. Do **not** fail a run on it —
a lossy production circuit legitimately has larger residuals than a lossless fixture. Report
it and let the analysis decide.

### Success Criteria
**Automated**: `pytest tests/test_power_balance.py`
- Lossless fixture: `power_balance_rel_err < 1e-6`, Manley-Rowe photon flux conserved to 1e-6
  (the plan's Phase 7 numbers).
- Lossy fixture: balance closes to 1e-6 once dissipated power is included.

**Manual**: Report `max_power_balance_rel_err` across the exp20 operating points. A large
value at deep saturation is a finding worth stating, not a bug to hide.

---

## Phase 5: Basis convergence acceptance

### Overview

Run the study driver the original plan specified. Production bases were selected by
small-signal JC gating, which bounds gain error but says nothing about P1dB error.

### Changes Required

#### 1. Run the study
**File**: `scripts/multitone_convergence_study.py` (exists — verify it still runs, repair if
the multitone API has drifted since it was written)
**Changes**: Execute for jtwpa and fqjtwpa across Q ∈ {1,2,3} × pump-harmonic order ×
`(n_p, n_delta)`, plus `three_tone` vs lattice and **odd-only vs dense `h`** (the plan calls
the odd-vs-dense comparison mandatory). Use the Phase 3 refined P1dB as the metric, not the
interpolated one — otherwise grid noise swamps the 0.2 dB acceptance band.

#### 2. Record the outcome
**File**: `CLAUDE.md`
**Changes**: Write the converged basis per device and the measured `|ΔP1dB|` between the top
two settings of each knob. If a production basis fails the 0.2 dB gate, say so explicitly and
record the P1dB uncertainty it implies for exp20/21.

### Success Criteria
**Automated**: `pytest tests/test_multitone_convergence.py`
**Manual**: `|ΔP1dB| < 0.2 dB` between the top two settings of every knob, or a documented
statement of the residual uncertainty where it is not met. Memory budget: Q=2 at pump order
10 is ~2500 super-blocks against production's 900 — check `resources.guard` before launching
and expect to need the banded backend.

---

## Phase 6: Stability

### Overview

Discharge the caveat carried by every artifact since Phase 6 of the original plan. Scoped to
quantifying selected operating points, not to automatic classification.

### Changes Required

#### 1. Linearize about the finite-signal state
**File**: `src/twpa_solver/multitone/stability.py` (new)
**Changes**: Build the Floquet linearization about a converged multitone torus state and
compute the dominant exponents, reusing `signal/stability.py`
(`estimate_sigma_min`, `refine_complex_resonance`) and the pattern in
`scripts/floquet_stability_sweep.py`.

#### 2. Report per operating point
**File**: `scripts/run_compression.py`
**Changes**: Behind `--check-stability` (default off, so nothing existing changes), replace
`stability_status = "NOT_CHECKED"` with a measured verdict and the dominant exponent. Every
run without the flag keeps `NOT_CHECKED`.

#### 3. Apply at the points that matter
**Changes**: Run on the zero-signal, P1dB, and deepest-saturation states for jtwpa and 2c —
the three points exp22 already checkpoints.

### Success Criteria
**Automated**: `pytest tests/test_floquet_stability.py` plus new multitone stability tests
- A known-stable small-signal state returns a stable verdict.
- The verdict is invariant to torus resolution within tolerance.

**Manual**: Stability verdict for the three exp22 operating points per device. A finding that
deep-saturation branches are *not* dynamically accessible would be a significant result and
must be reported as such, not suppressed.

---

## Testing Strategy

### Project Maturity Level
**Established Production.** `twpa_solver` is pinned to JosephsonCircuits.jl by the 7-design
parity suite. Every phase here must leave the full suite at exactly the 4 pre-existing
failures, and must not move any converged number.

### Regression guard (every phase)
```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\tw_next --run-slow
```
Must end at 4 pre-existing failures. In particular `tests/test_fast_coupled_assembly.py`
(8 tests) gates the preconditioner assembly and `tests/test_multitone_distributed_physics.py`
gates the physics.

### Unit tests
Written alongside each change, per repo standard. New gates must be **mutation-verified**:
break the thing deliberately, confirm the test fails, restore. This is the standing bar in
this repo — a gate that has not been shown failing is not evidence.

### Numbers to report, not claim
Every phase's manual criterion asks for a measured number. Report measured values with
their measurement conditions; "passes" without the number is not acceptance.

---

## Rollback Plan

- Phases 2–4 are additive columns plus one deleted line in `exp22_spatial_attribution.py`.
  Rollback = revert the commit; existing artifacts remain readable since the new columns are
  additions.
- Phase 1 touches `resources.py` calibration constants only; the estimator's conservatism
  test is the guard against a bad recalibration.
- Phase 3 keeps the interpolation path behind `--p1db-power-tol-db 0`, so the old behaviour
  is always reachable.
- Phase 6 is entirely behind `--check-stability`; default-off means no existing artifact
  changes.
- Each phase is a separate atomic commit so one can be reverted without disturbing the rest.

## Sequencing note

Phase 1 first — it is short and every later campaign runs faster with the worker count the
memory work already bought. Phase 2 next: it is the only phase that changes a conclusion
already written down, and the longer that conclusion circulates the more expensive the
correction. Phases 3–5 are ordered by how much they affect published numbers. Phase 6 is
last because it is the largest and is additive rather than corrective.
