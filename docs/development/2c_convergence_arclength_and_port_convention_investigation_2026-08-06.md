# 2c gain-map convergence, arclength recovery, and port-convention investigation (2026-08-06)

Session summary. Two mostly-independent threads: (1) reverting the port
power convention, (2) understanding and improving `designs/ipm_2c_fixed`
gain-map convergence at high pump power, including a warm-started/leak-fixed
`--fold-policy arclength` recovery path.

## 1. Port power convention: reverted Norton -> matched travelling-wave

`designs/ipm_2c_fixed`'s four ports are each an ideal current source in
parallel with `G0 = 1/Z0` (the `G` matrix's only nonzeros). That topology is
ambiguous between two standard conventions:

- **Norton generator**: `I` is the source's own short-circuit current,
  splitting `I/2` across a separately-matched load -> `P = I^2 Z0/8`.
- **Matched wave port**: `I` *is* the incident wave's own current amplitude,
  `G0` only absorbs reflections -> `P = I^2 Z0/2`.

The 2026-08-05 fix had defaulted to Norton, reasoning that "the load sees
`I/2`". On review, the user's own derivation (confirmed with their
collaborator) is that this design's ports are matched wave ports, so `I` is
the full incident current, not a Norton short-circuit current -- the
`I^2Z0/8` reasoning does not apply here.

**Reverted 2026-08-06**: default is now `legacy_traveling_wave`
(`P = I^2 Z0/2`); `convention="norton"` remains selectable for comparison.
Changed defaults in:

- `src/twpa_solver/ports.py` (`port_available_power_w`,
  `port_current_from_power_a`) -- the single source of truth, docstring
  rewritten
- `src/twpa_solver/loss.py` (`InsertionLossModel.dbm_to_peak_current_a`)
- `scripts/run_gain_map.py` (`--power-convention` CLI default,
  `dbm_to_peak_current_a` default, module docstring)
- `scripts/run_compression.py` (`--power-convention` CLI default)
- `scripts/run_kimpa_gain.py`, `scripts/fit_model_operating_point.py` (x2),
  `scripts/measured_psat_pipeline.py`, `experiments/exp49_phase6_curves_two_way.py`,
  `experiments/exp51_spatial_depletion_profile.py`,
  `experiments/exp54_basis_self_convergence.py` -- hardcoded
  `convention="norton"` call sites flipped to `"legacy_traveling_wave"`
- `tests/test_port_power.py`, `tests/test_run_compression_cli.py` -- updated
  to assert the new default; `test_netlist_ports_are_norton_terminated_at_1_over_z0`
  renamed/reworded (the underlying `G[port,port]==1/Z0` fact is
  convention-neutral, only the docstring's interpretation changed)
- `CLAUDE.md` "Port power convention" section rewritten

**Practical effect**: for the same requested dBm, today's default injects
**half** the current of the interim (2026-08-05..06) Norton-era default --
a fixed 6.0206 dB offset (`LEGACY_TW_OFFSET_DB`). Any map/run produced during
that one-day window used `norton` and is not directly comparable to today's
numbers without accounting for this.

**Open, unresolved inconsistency** (do not "fix" unilaterally): 
`src/twpa_solver/multitone/observables.py::extract_port_waves` (~line 72-75)
independently re-derives the *outgoing* wave from the solved node state via
literal KCL: `current = injected_current - voltage/z0_ohm` -- i.e. a Norton
subtraction, computed from first principles on the actual solved state, not
from the `ports.py` convention flag. This was **not** touched by the revert.
The user's plan is to physically remove the port's separate shunt resistor
from the netlist later, which would make the topology genuinely a wave port
and resolve this on its own; until then, the source-side injection
(`ports.py`, now travelling-wave) and the solved-state wave extraction
(`observables.py`, still Norton-KCL) use two different physical pictures of
the same port. `pump_outgoing_power_w` (`power_balance`) still reads
`I^2Z0/8`-consistent values because it goes through this same KCL path.

## 2. 2c gain-map convergence and `--fold-policy arclength`

All numbers below use the corrected (post-revert) `legacy_traveling_wave`
convention. Test grid unless stated otherwise: `designs/ipm_2c_fixed`,
`-28..-18 dBm x 7.0..8.0 GHz`, `n_power=30`, production engine flags
(`schur_cpu_mt` + `real_coupled_fast` + `secant` predictor +
`--fold-skip-patience 2`).

### 2a. Fail-fast is a warm-chain artifact, not (mostly) physics

`--inproc-fail-fast` accepts the first Newton failure in a column and moves
on; the *next* point still warm-starts from the last **converged** state via
secant extrapolation (`run_warm_pass_inprocess`: `base_X = last_good_X if
fail_fast else prev_X`). Extrapolating that fixed anchor further and further
past the true fold makes the residual grow roughly monotonically with power
-- visually identical to "poisoning", but the actual mechanism is
extrapolation degradation from a stale (but correct) anchor.

Disabling fail-fast (`--recovery reseed`, the default; note this also keeps
`--fold-policy patience`, `--traversal column`, which together route through
the **legacy** `run_warm_pass_inprocess` path rather than the orchestrator)
recovers most of the resulting gap:

| column (fp, GHz) | fail-fast | reseed |
| --- | ---: | ---: |
| 7.1053 (worst tested) | 14/30 (46.7%) | 20/30 (66.7%) |
| 7.4737 (best tested) | 26/30 (86.7%) | 29/30 (96.7%) |
| 10-column chunk (300 pts) | 200/300 (66.7%) | 218/300 (72.7%) |

Cost: reseed pays the full adaptive-then-fixed continuation ladder even for
points that ultimately still fail -- chunk runtime went from ~350-400s to
2412s (~6-7x) *before* the memory-leak fix below.

### 2b. The map's high-power boundary is a genuine, frequency-dependent fold

Aggregate convergence vs. power, all 10 columns, reseed strategy: **100%
converged from -28 dBm up through -23.17 dBm**, then a clean monotonic
roll-off to **0% by -18.69 dBm**. Not a sharp single-power cliff -- individual
columns' folds are spread across a ~4 dB band (-23..-19 dBm), which averages
into the smooth aggregate transition. This is a real, global amplitude-limited
regime, confirmed physically at two points (below), not just a per-column
solver quirk.

### 2c. Memory leak: reseed and arclength recovery degrade the whole process

`pump_factor_runtime_s` (PARDISO factor time) was measured **exactly 4.00x**
slower for reseed-mode "both converged" points vs. the fail-fast baseline,
uniformly across every solve phase, despite identical predictor choice and
Newton iteration counts (i.e. not algorithmic -- pure overhead). Smoking gun:
comparing the *first point of each column* (fresh cold seed, no warm-start
baggage): columns 0-2 were comparable to fail-fast, but column 3's first
point was already 5x slower (14.8s vs 2.96s) *before that column had done any
recovery work itself* -- degradation accumulates process-wide from earlier
columns' expensive recovery attempts (PARDISO/GMRES temp-array churn,
10-25 MB per call, thousands of calls per stubborn cell, fragmenting the
Windows process heap over a long-lived worker).

**Fix**: `gc.collect()` after the two `mode="seed"` (full continuation from
scratch) call sites in `scripts/run_gain_map.py`:
`run_warm_pass_inprocess`'s reseed retry, and `_recover`'s final reseed
fallback. Verified on the same 4 columns: **identical** convergence outcome
(81 converged / 31 skip / 8 fail, bit-for-bit) with pump-solve time
**616.5s -> 267.4s (2.3x)**.

Same class of bug hit `--fold-policy arclength` harder: it originally
cold-started (`X=0, lambda=0`) with no wall-time cap, and crashed twice with
`numpy._core._exceptions._ArrayMemoryError` (11.5 MiB, then 23.4 MiB, at
different points ~170-470 cells into a chunk -- not a fixed-size bug, genuine
fragmentation). Fixed in `_recover`'s arclength block:

1. Warm-start from the best converged neighbour (`parent_i`) instead of
   `X=0, lambda=0` -- cuts total step/call volume by roughly the same factor
   as the distance reduction.
2. `gmres_maxiter=20` for the recovery-specific solver instance (via
   `dataclasses.replace` on a copy of `engine._settings()`) -- the exact
   preconditioner converges in ~1 iteration when healthy; more than that is
   grinding, not converging.
3. `gc.collect()` after every attempt (success or failure).
4. `max_steps=60` cap.

Result: no crashes across multiple subsequent full runs. On a 4-column test,
arclength+reseed converged 83/120 vs. reseed-alone's 81/120 (+2), but wall
time nearly doubled (323s -> 599s) -- `pump_wall_runtime_s` only records each
point's *final* reported solve, so it doesn't capture the cost of a failed
arclength attempt that then still pays full price for the reseed fallback
afterward. **Arclength earns its keep on individual stubborn cells (10/10
successful when it engages, in every test run), not as a blanket map-wide
setting layered on reseed.**

### 2d. Junction current vs. Ic: two different mechanisms found at two points

Peak instantaneous junction current relative to each cell's own critical
current (`Ic = phi0_reduced / Lj`, uniform ~2.656e-6 A across
`ipm_2c_fixed`'s 2508 junctions) was traced via pseudo-arclength continuation
(`solver.solve_arclength`), reconstructing the full-node state from the
Schur-reduced solution (`problem.reconstruct_full`) at every accepted step.

**fp=7.9 GHz (jc-scale 1.0): no real fold found.** Traced from lambda=0 to
1.03 (3% past nominal target); residual stayed converged throughout
(`coeff_rel` 1e-7..1e-9); peak junction current never exceeded **~77% of
Ic**; no `lam_dot` sign flip (turning point) was ever detected. A
`--fold-follow` sweep across 19 frequencies (7.6-8.5 GHz) found **zero**
folds (`fold_lambda=None` everywhere). The production "wall" here is
natural-parameter Newton continuation losing the solution in a region where
several spatially-distinct, nearly-degenerate solution branches compete (the
peak-current hot-spot migrates between neighbouring cells, observed hopping
across cell indices 1250-2500 as lambda approached 1) -- a **continuation
predictor artifact, not a physical amplitude limit**.

**fp=7.0 GHz, -20.069 dBm grid point: a genuine, tight fold.** This point
persistently fails in production with the "near-miss" signature
(`coeff_rel=1.9e-10`, `reached_target=False`, 109 Newton iterations). Seeded
from the actual close production neighbour (-20.414 dBm, lambda0=0.961, 96%
of the way to target, re-verified converged to `coeff_rel=1.4e-13`):
arclength could not reach lambda=1 at either `ds=0.1` (reached lambda=0.964)
or `ds=0.02` (reached lambda=0.972) -- both terminate via ds-underflow
("minimum_step"). True maximum sustainable amplitude at this frequency is
close to lambda~0.97 (~-20.2 to -20.3 dBm), a hair below what this grid point
requests.

**Methodology correction, worth recording:** an earlier attempt at the same
target, seeded from a much more *distant* neighbour (-23.172 dBm,
lambda0=0.551, only 45% of the way to target), showed the arclength
corrector unable to take even one step at `ds=0.1` -- `coeff_rel` diverged
from 30 to 4e31 within a handful of Newton sub-iterations. This was confirmed
against the real, unmodified `solve_arclength` (not a bug in the ad hoc
diagnostic script) and diagnosed as the step size exceeding the local Newton
convergence radius (measured ~0.01-0.05 in lambda at that specific point,
`+0.009` converged, `+0.049` did not). **That diagnosis was correct for that
test but did not explain the production failure**, because production's own
warm start uses the much closer neighbour, not the distant one. Lesson: when
diagnosing a specific persistently-failing grid point, seed the diagnostic
from the *actual* neighbour production would use, not an arbitrarily-chosen
earlier converged point -- the two can point to entirely different
mechanisms (predictor artifact vs. genuine fold).

**Fix shipped** (`scripts/run_gain_map.py`, `_recover`'s arclength block):
`ds` lowered from `0.1` to `0.02`. Confirmed safe and mildly beneficial in
aggregate (~6% faster on the 4-column test, same 83/120 converged, one fewer
point tagged specifically `arclength` but rescued via reseed instead) --
it does not rescue the fp=7.0 GHz point above, whose failure is a genuine
fold, not a step-size artifact.

## 3. Net conclusions

- **Disabling `--inproc-fail-fast` is the single highest-leverage change**
  for map completeness: a real, cheap-to-understand ~6-9 point-percent
  convergence gain map-wide, at a real (post-leak-fix) ~3-4x wall-time cost
  for a full map.
- **`--fold-policy arclength`** (now warm-started, leak-fixed, step-tuned) is
  a further small positive increment on top of reseed, not a replacement for
  it, and not worth using as a blanket full-map default given its cost
  relative to gain. Use it for targeted column/point-level investigation of
  specific stubborn cells.
- **The map's -23..-19 dBm convergence boundary is mostly a genuine,
  frequency-dependent amplitude fold** of the periodic pump solution --
  confirmed physically at fp=7.0 GHz -- though fail-fast's warm-chain
  extrapolation artifact exaggerates its apparent severity, and at least one
  individual frequency (7.9 GHz) turns out to have no real fold there at all.
  Different frequencies genuinely have different maximum sustainable pump
  amplitudes; do not assume every "wall" in this band is the same kind of
  thing without checking.

## Files touched

**Port convention**: `src/twpa_solver/ports.py`, `src/twpa_solver/loss.py`,
`scripts/run_gain_map.py`, `scripts/run_compression.py`,
`scripts/run_kimpa_gain.py`, `scripts/fit_model_operating_point.py`,
`scripts/measured_psat_pipeline.py`,
`experiments/exp49_phase6_curves_two_way.py`,
`experiments/exp51_spatial_depletion_profile.py`,
`experiments/exp54_basis_self_convergence.py`, `tests/test_port_power.py`,
`tests/test_run_compression_cli.py`, `CLAUDE.md`.

**Convergence/arclength**: `scripts/run_gain_map.py` only --
`run_warm_pass_inprocess` (reseed-retry `gc.collect()`), `_recover`
(warm-started arclength seed, `gmres_maxiter=20` recovery settings,
`gc.collect()`, `ds=0.1 -> 0.02`, final-fallback `gc.collect()`).

No test suite was added for the convergence/arclength changes in this
session -- verification was empirical (repeated map/column reruns compared
against prior results), not unit tests. Consider adding regression coverage
for the `gc.collect()` placement and the warm-started arclength seed
selection if this code is touched again.
