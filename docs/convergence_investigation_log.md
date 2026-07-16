# Convergence Investigation Log

Running notes on the pump harmonic-balance solver convergence investigation.
Chronological, most recent entry at the bottom. This is a working log, not a
polished report — read it as "what we tried, what we found, what's next."

**Terminology note (binding for this whole log):** do not use the word "fold"
for a solver stall unless it has been corroborated against physics (measured
device data, or an independent physical argument like a gain-compression
saturation curve matching the stall point). Until then, call it a
**convergence failure**. Some earlier entries below (and an earlier memory
file, `twpa-fold-vs-impedance.md`) used "fold" before this rule was set —
treat that language there as provisional, not settled.

---

## 2026-07-16 — Session start: control-flow investigation setup

Starting point: a detailed control-flow diagram of the pump solve was supplied
(Frequency loop -> Power loop -> Continuation loop -> Newton loop -> Residual
evaluation / Time-sample loop / Junction operations / Harmonic projection loop
-> Convergence check -> Tangent-state construction -> Preconditioner assembly
-> GMRES loop -> GMRES acceptance check -> Line-search loop), with the goal of
checking the solver's numerics "from the outside in" using a hand-picked debug
subset: `outputs/measurement_match_debug_01/column_debug_col3_trim` (2c
device, fp=7.329 GHz column, chosen for later comparison against measurement).

Heavy `logger.debug` instrumentation was already present across
`solver.py`, `run_gain_map.py`, `problem.py`, `circuit.py`,
`schur_operators.py`, `schur_partition.py`, `basis.py`, `predictors.py`,
`floquet.py`, `gamma.py` (user's own uncommitted additions, not authored in
this session).

### Bug found and fixed: adaptive-continuation fallback discarded progress

`solve_adaptive_continuation`'s fixed-step fallback (triggered when
lambda-bisection shrinks below `min_step`, i.e. a genuine stall in
source-scale space) was restarting the continuation ladder from `lambda=0`
using the *original* zero seed — discarding every state the adaptive phase
had already converged. On a real map column (fp=7.329 GHz, -28.25 dBm) this
burned the whole 14s per-point wall-time deadline re-deriving the cheap
low-lambda region from scratch, getting back only to lambda=0.75, when the
adaptive phase had already reached lambda=0.9375 converged before falling
back.

Fix (`src/twpa_solver/pump/solver.py`):
- `solve_continuation` gained a `lambda_start: float = 0.0` parameter; the
  lambda ladder is now `np.linspace(lambda_start, 1.0, continuation_steps +
  1)[1:]` (identical to the old formula when `lambda_start=0.0`).
- The fallback in `solve_adaptive_continuation` now resumes from
  `(X_current, lambda_current)` instead of the original seed/lambda=0, sized
  to the remaining span (`math.ceil(remaining / step_size)` steps, not the
  full `fallback_fixed_steps`).

Tests: `tests/test_adaptive_continuation_fallback.py` (2 tests, both pass) —
`test_solve_continuation_lambda_start_resumes_span` and
`test_adaptive_continuation_fallback_resumes_not_restarts` (tuned with
`pump_current=2.0, max_newton=3, min_step=0.1` on a one-node LC+Josephson
problem, found via a scratchpad probe sweep, to reliably trigger a partial
adaptive phase followed by fallback).

Documented in `CLAUDE.md` under "Continuation-method suite".

Confirmed already-implemented (not new this session, part of the user's own
prior debug work): mid-GMRES deadline abort (`solve_deadline_s` checked on
every GMRES iteration via `callback_type="pr_norm"`, raising
`GmresDeadlineExceeded`, not just checked between Newton iterations).

### Rerun after the fix: same column, same wall

Reran the fp=7.329 GHz column with the fix applied
(`outputs/measurement_match_debug_02_col3_trim_fixed/`, `--log-level DEBUG`).
Outcome unchanged (still 8 PASS / 2 ERROR / 10 SKIP_PAST_FOLD-equivalent) —
the fix works exactly as designed (log message confirms "resuming, not
restarting"), but the underlying convergence failure at that specific
operating point (P ~ -28.2..-28.0 dBm) is not a symptom of this bug. Two
independent signals corroborate a genuine solver-side wall at this point:

1. Newton reduction ratio climbs toward 1.0 across successive fine
   bisections as lambda -> 1 (0.897, 0.928, 0.957, 0.970, 0.998) — textbook
   stall saturation.
2. The power-axis substep mechanism (`--column-power-substep`) independently
   shrinks to its step floor (0.0035 dB) without crossing
   (`verified_fold=True` in the code's own naming, which predates this log's
   terminology rule above).

When both agree, the convergence failure is now treated as reproducible and
not an artifact of wall-time or fallback-restart bugs.

### Cold vs warm GMRES/Jacobian instrumentation

Wrote `scripts/debug_cell_gmres_matrix_analysis.py` to look at exactly one
failing cell (map point index=8, fp=7.32894736842 GHz, P=-28.253164556962027
dBm) two ways, reusing the production `InProcessEngine` path unmodified:

- **cold**: `mode="seed"`, full continuation from X=0 (adaptive_secant +
  fixed fallback).
- **warm**: `mode="warm"`, a single direct Newton solve at lambda=1, seeded
  with the previous cell's converged solution (point 7, P=-28.468... dBm,
  hardcoded, re-solved fresh in-script rather than reloaded from disk so the
  seed is in the exact Schur-retained shape the solver expects).

Instrumentation is monkeypatch-based (installed/restored around the run, no
production code touched):
- `FastCoupledPreconditioner.refactor` -> snapshots the exact real-packed
  Jacobian `M` (sparse CSR) every Newton iteration, both runs.
- `twpa_solver.pump.solver.gmres_call` -> snapshots the per-GMRES-iteration
  `pr_norm` history via the callback.

Output: `outputs/cell_gmres_analysis_col3_p8/` (`summary.csv`,
`gmres_convergence.png`, `coeff_rel_vs_iteration.png`, `spy_*.png`,
`condition_estimate.txt`).

**Finding 1 — GMRES/preconditioner layer is not the bottleneck.**
`gmres_iters == 1` on every single Newton iteration, cold or warm, at every
lambda, in every log examined this whole session. `real_coupled_fast`'s `M`
*is* the exact Jacobian (not an approximation — see
`fast_coupled.py`), so using it as its own preconditioner solves the linear
Newton-correction system to ~1e-19 in one GMRES shot, always. The GMRES loop
in the original control-flow diagram is exonerated for this failure mode.

**Finding 2 — raw matrix-norm statistics carry no signal.** `M`'s Frobenius
norm (~31.6-31.8e12) and asymmetry fraction (~0.006) are nearly constant
across the entire lambda path and both runs. The huge lambda-independent
linear part of the Jacobian dominates the norm and masks the comparatively
tiny nonlinear Josephson-coupling perturbation that actually governs
solvability.

**Finding 3 — `M` is intrinsically ill-conditioned but that never surfaces in
GMRES.** 1-norm condition estimate (Higham-Hager, `onenormest` + `splu`-based
`LinearOperator` for `M^{-1}`) on a randomly-picked snapshot: ~1.5e7-3.0e6
depending which snapshot the random pick lands on (reruns pick different
iterations; order of magnitude is consistent ~1e6-1e7). This never surfaces
in GMRES iteration count because `M` pre-factors itself out exactly.

**Conclusion at this point:** the convergence-failure signature lives
entirely in the **Newton / line-search layer**, not the linear-algebra layer.

### coeff_rel, precisely

`coeff_rel` (`problem.py:321-352`, and the Schur-backend equivalent in
`schur_operators.py:192-208`) is:

```
R = residual_coeffs(X, source_scale)        # R_k = D_k X_k + N_k(X) - S_k(lambda)
coeff_abs = ||R_flat|| / sqrt(len(R_flat))   # RMS over all harmonic coeffs, real+imag packed
S = source_coeffs(source_scale)
src_abs = ||S_flat|| / sqrt(len(S_flat))     # RMS of the source term at this lambda
coeff_rel = coeff_abs / max(src_abs, 1e-30)
```

i.e. the RMS harmonic-balance residual normalized by the RMS source magnitude
— exactly the `||R||coeff,rel` gate from the original control-flow diagram's
"Convergence check" box. `coeff_rel_vs_iteration.png` plots this per Newton
iteration for cold vs warm at point 8: cold zig-zags (deep drop at partial
lambda during continuation, hard stall ~1e-2..1e-3 every time it reaches
lambda=1), warm lands in that same stalled band on iteration 1 of its direct
lambda=1 attempt — same wall, reached two different ways.

### Per-harmonic residual breakdown

Extended `debug_cell_gmres_matrix_analysis.py` with a
`SchurReducedProblem.residual_coeffs` monkeypatch (`_capturing_residual_coeffs`)
that snapshots every call (X, source_scale, R, and the harmonic mode list),
tagged with the run label. The last call of a run is always its final state —
converged for `prev_seed` (point 7), stalled/non-converged for `cold` and
`warm` (both point 8, lambda=1).

Output: `outputs/cell_gmres_analysis_col3_p8/per_harmonic_residual.csv` /
`.png` — RMS `|R_k|` over retained nodes, one bar per harmonic k, for the
three picked snapshots.

**Finding:** smooth monotonic broadband decay across all 10 harmonics
(k=1,3,...,19), ~2 orders of magnitude drop per few harmonics, same shape
cold vs warm, both uniformly ~1e8-1e9x above the converged baseline at every
k. No single harmonic stands out — the residual is not concentrated in a
stiff high harmonic or a specific coupling term. This is consistent with a
broadband amplitude-domain effect (the sine nonlinearity saturating) rather
than one under-resolved mode.

### Per-node (spatial) residual breakdown

Extended the same snapshot with `retained_full_idx` (the Schur partition's
`part.retained` array, mapping retained position -> full circuit node index)
and `pump_pos`. Added `per_node_rms` (RMS of `R_n` over harmonics, per
retained node) and `dump_per_node_residual` / `plot_per_node_residual`.

Output: `outputs/cell_gmres_analysis_col3_p8/per_node_residual.png` +
`per_node_residual_top20.csv` (top 20 offending retained nodes by rank, for
each of the three picks).

**Finding:** broadband across essentially the whole retained set (~2518
nodes for this device) — converged baseline sits uniformly ~1e-21 across all
positions, stalled cold/warm sit uniformly ~1e-10, ~9 orders of magnitude
higher but with the *same flat shape*. There is a **mild but real upward
trend** toward the high-index end of the retained-node ordering: the top-20
offending nodes (both cold and warm) cluster at retained positions
~2400-2515, i.e. the tail end of the line (out of ~2518 total). This is a
weak but consistent signature — the last cells of the 2c line carry
somewhat more residual than the first, suggestive of an effect that
accumulates along propagation (e.g. pump depletion/saturation building up
cell-by-cell) rather than a single defective cell.

One artifact to note, not a finding: at the exact retained position of the
pump node itself, `R_n` drops to ~1e-24 for all three runs (visible as the
sharp single-sample dip at the dotted line in the plot). This is a
cancellation artifact of how the current source enters that node's residual
row, not a physically meaningful "this node is fine" result — do not read
anything into it.

### Power-axis bisection: the wall sits right at P7, not somewhere before P8

Question raised: prev_seed (point 7) converges to `coeff_rel=1.13e-13`; cold
and warm at point 8 both stall at `~6e-3` — a ~1e10 gap over just 0.215 dB.
Is that gap smooth or a cliff, and where exactly does it break?

Wrote `scripts/debug_power_sweep_col3_p7_p8.py`: splits the 0.215 dB gap
between P7 (-28.468354430379748 dBm) and P8 (-28.253164556962027 dBm) into
10 points, spaced with `t = 1-(1-x)**2` (denser toward P8, the end already
known to fail). Two passes, both `force_gain=True` (so `solve_point` always
returns the last-iterate X even on non-convergence, per the "Forced-gain
column resume" mechanism in CLAUDE.md, keeping any warm chain alive past a
failure instead of collapsing to `None`):

- **cold** — each of the 10 points solved independently from X=0 (full
  adaptive continuation each time).
- **warm_chain** — each point warm-started directly (lambda=1 one-shot
  Newton) from the previous point in this same sweep, regardless of whether
  that previous point converged.

Output: `outputs/power_sweep_col3_p7_p8/power_sweep_summary.csv` +
`power_sweep_coeff_rel.png`.

**Finding: the wall is not between P7 and P8 — it's immediately after P7.**
The very first sweep point above P7 (i=1, only **+0.045 dB** above P7, i.e.
~21% of the way to P8) already fails to converge in both cold and warm_chain:
`coeff_rel` jumps from `1.13e-13` (P7) straight to `~1.4e-4` (i=1) — an
**11-order-of-magnitude cliff in a 0.045 dB step**. From i=1 onward to P8
(the remaining 0.83 of the gap), `coeff_rel` only climbs gently and
smoothly, `1.4e-4 -> ~6e-3` (cold) — roughly two more orders of magnitude
spread across 5x the power range that produced the first 11-order jump. See
the plot: a near-vertical drop right after the P7 point, then a shallow,
smooth log-scale rise plateauing before P8.

This reframes the earlier picture: it is not "smooth convergence degrading
gradually until it breaks near P8." P7 sits essentially *at* the edge of
whatever this limit is; crossing it by any amount, however small, costs most
of the achievable precision immediately, and further power increase past
that point barely matters by comparison. That is the signature expected of
a true turning point / bifurcation in the algebraic system (the local
quadratic-convergence radius around the solution branch collapsing to ~0
right at that point), not a signal-quality or tolerance issue with the
solver's settings.

**Warm-chain artifact worth flagging (not a finding about the device):**
from i=2 onward, warm_chain's Newton fails on the very first iteration's
line search (`reason=line search failed at Newton 1`), which means
`solve_direct` never accepts a single step — the `force_gain`-returned X is
therefore *literally unchanged* from the input warm seed. This is visible in
the summary CSV as `gain_db=16.95052529912834` identically repeated for
i=2..9: the "solve" at each of those points did nothing, and the gain
reported is just the previous frozen state's gain reevaluated at a new
target current, not a real result for that power. `coeff_rel`'s smooth
climb from i=2..9 in the warm_chain series is consequently measuring "how
far a static, stale waveform is from satisfying HB at ever-higher target
power," not a sequence of genuine solve attempts — read the **cold** series
for the real per-point residual floor, not warm_chain, past i=1.

### Finer zoom + 5-continuation-method comparison, all cold

Two follow-ups requested together: (1) zoom in even further on where the
cliff actually starts (the first sweep only bracketed it to "somewhere in
the first 0.045 dB above P7"), and (2) check whether the cliff is a
continuation-method artifact by re-running the **cold** solve (mode="seed",
X=0 every time -- the warm_chain series is unreliable past its first
failure, see the artifact note above) under all 5 available intra-cell
continuation methods: `adaptive_secant` (the production default -- "whatever
we're doing now"), `adaptive_tangent`, `affine`, `ptc`, `arclength`.

Wrote `scripts/debug_power_sweep_fine_continuation_methods.py`. Zoom range:
P7 (distance 0) up to the old i=1 point (distance +0.0451633 dB, the first
point that failed in the previous sweep), with 10 points spaced
**geometrically in distance from P7** (`d_i = gap * 2**(i-9)`, i=0..9), so
resolution is finest right at P7 and coarsest at the already-known-bad end --
the opposite bias from the first sweep, chosen because the first sweep
already showed the cliff sits very close to P7 rather than near P8. Each of
the 11 distances (10 + P7 itself) solved under each of the 5 methods by
mutating `engine.args.inproc_continuation` between calls (`InProcessEngine.
_settings()` reads it fresh on every `solve_point` call, confirmed by
reading `run_gain_map.py:640-669`, so no need to rebuild the engine per
method) -- 55 solves total.

Output: `outputs/power_sweep_fine_continuation_methods/fine_methods_summary.csv`
+ `fine_methods_coeff_rel.png`.

**Finding 1 -- the cliff is bracketed, not smeared out.** All 5 methods
converge cleanly (coeff_rel between ~1e-10 and ~1e-14, indistinguishable from
P7's own quality) at every one of the 9 intermediate distances from
8.8e-5 dB up through 0.0226 dB above P7. It is **only** at the final,
largest zoom point (0.0452 dB above P7 -- the original old-i=1 point) that
every single method fails simultaneously. So the cliff is not present at
arbitrarily small offsets above P7 (ruling out "P7 itself is already
marginal and any noise pushes it over") -- it is bracketed to a specific
window, **(0.0226, 0.0452] dB above P7** (roughly -28.446 to -28.423 dBm at
this frequency), by two adjacent geometric samples 2x apart. That is as far
as this geometric ladder can localize it without another zoom pass.

**Finding 2 -- no continuation method avoids or delays the wall.** At the
failing distance (0.0452 dB), all 5 methods fail within the same order of
magnitude: `adaptive_secant` coeff_rel=1.36e-4, `affine` 1.40e-4, `ptc`
7.67e-5 (own failure mode: "ptc line search failed at iter 25"), `arclength`
1.26e-4. `adaptive_tangent` is a partial outlier worth flagging precisely:
it reports coeff_rel=2.83e-14 (converged-quality) yet `pump_status=FAIL` --
this is `run_gain_map.py`'s own `converged` gate (`reports[-1].converged and
abs(source_scale - 1.0) < 1e-12`), meaning its last accepted report was a
genuinely converged step at some lambda strictly less than 1, and it could
not push the remaining distance to lambda=1 at all -- a different failure
signature (stuck below target) from the other four (reaching lambda=1 and
stalling there), but still a failure to reach the physical target. No
method reaches the target at this power. All 5 approaches -- 3 different
predictor/step strategies, a pseudo-transient reformulation, and a genuinely
different algorithm (pseudo-arclength bordering) -- hit the identical wall
at the identical power. That is strong independent corroboration that this
is a property of the nonlinear map itself, not a weakness of any one
continuation strategy.

**Secondary sanity check:** every method that converges at a given distance
lands on matching `gain_db` to 5+ decimal places (e.g. all five give
~16.869175... at 0.0226 dB above P7) -- confirms all methods are tracking
the same physical solution branch, just via different paths, right up until
the branch itself becomes unreachable.

One data-collection nuance, not a finding: `ptc`'s `pump_gmres_total` column
reads 0 throughout, unlike every other method (which shows
`gmres_total == newton_total`, reconfirming the earlier "1 GMRES iteration
per Newton step" result for the other four). PTC just doesn't route through
the same instrumented `gmres_call` path -- 0 does not mean "no linear solve
happened," it means this particular counter doesn't apply to that code path.

### Open next step (not yet done)

Per-node/per-harmonic breakdowns both point toward a *broadband* nonlinear
effect rather than a localized numerical defect. Candidate next moves,
undecided as of this entry:
- Look at the actual pump waveform / Josephson phase amplitude at the
  stalled iterate (is `psi/phi0` approaching pi/2, i.e. genuinely saturating
  the sine?) — would be the first real physics check before ever
  reconsidering the "fold" terminology.
- Compare the per-node residual shape at a *converged* nearby lambda (e.g.
  cold's own lambda=0.9375 converged step) against the stalled lambda=1
  attempt, to see whether the spatial trend is already present pre-stall or
  only appears at the wall.
