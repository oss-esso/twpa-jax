# Implementation plan: scaled pseudo-arclength metric and a real fold test function (2026-08-06)

### Goal

Make `solve_arclength` an actual pseudo-arclength method (it is currently
natural-parameter continuation in disguise), and add a singularity test
function so "the solver stopped" can be told apart from "the branch turned"
by measurement rather than by inference.

---

## Current state analysis

### The defect

`src/twpa_solver/pump/solver.py:1063-1071` normalises the arclength tangent
with an **unscaled** Euclidean metric:

```python
Xdot = self._solve_linear(problem, X, S, deadline_s=max_wall_s, t0=t0)
lam_dot = 1.0
norm = math.sqrt(_real_dot(Xdot, Xdot) + lam_dot * lam_dot)
Xdot, lam_dot = Xdot / norm, lam_dot / norm
```

`X` is node flux in webers; `lambda` is dimensionless. Measured from a real
converged 2c pump state
(`outputs/a2_fresh_col_fp7p70408/warm/points/point_0020_p_m30p102dbm_fp_7p70408ghz/pump/pump_solution.npz`,
S=10, 6446 retained nodes):

```
||X|| = 9.386e-14 Wb      max|X| = 1.93e-15 Wb
```

so `||Xdot|| ~ 1e-13` and:

| quantity | value |
| --- | --- |
| `norm = sqrt(‖Xdot‖² + 1)` | `1 + 5e-27` |
| normalised `lam_dot` | `1 − 5e-27` |
| state term in the constraint, `<Xdot, Xc−X>` (solver.py:1095) | `~1e-27` |
| lambda term, `lam_dot·(lamc−lam)` | `~ds = 2e-2` |
| **ratio** | **~5e-26** |

Double-precision eps is 2.2e-16. The state contribution sits ten orders of
magnitude below the roundoff of the lambda contribution. The constraint at
`solver.py:1095` is bit-identically `lamc - lam = ds`, `denom` at
`solver.py:1100` is bit-identically `1.0`, and the method is
natural-parameter continuation.

Three consequences, each matching a symptom recorded in
`docs/development/2c_convergence_arclength_and_port_convention_investigation_2026-08-06.md`:

1. **Fold detection is structurally impossible.** `lam_dot` is
   `±1/sqrt(1+‖Xdot‖²)`; its sign can only flip through the
   direction-consistency test at `solver.py:1125`, which requires
   `|<Xdot_new, Xdot>| > 1`, i.e. `‖J⁻¹S‖ > 1` — thirteen orders from the
   measured value. The `--fold-follow` sweep reporting zero folds at 19
   frequencies (investigation doc §2d) is a tautology of this, not a result.
2. **The fold cannot be rounded.** Past a turning point natural-parameter
   Newton has no solution, the corrector fails at every `ds`, `ds` halves to
   the `1e-4` floor and returns `terminal_reason="minimum_step"` — the exact
   reported fp=7.0 GHz signature (ds=0.1 → lambda 0.964, ds=0.02 → 0.972).
   That string is returned for a genuine fold *and* for a merely sharp turn;
   it does not distinguish them.
3. The bordering algebra at `solver.py:1100-1104` reduces to `d_lam = -n`,
   `d_X = a + d_lam*b` — plain Newton with a lambda nudge.

### Why it survived the test suite

`tests/test_advanced_continuation.py:25-35` builds the toy problem with
`phi0=1.0`, `Ic=1.0`, `C=K=1.0`, `pump_current=0.6`, so `||X|| = O(1)` and the
unscaled metric is accidentally well-conditioned. The three arclength tests
(`:73`, `:86`, `:99`) all pass on a dimensionally unrepresentative fixture.

### The corrected implementation already exists

`trace_arclength_from_two_points` (`solver.py:1140-1256`) does it properly:

- `state_scale` from the seed secant, floored (`:1173-1177`)
- `metric_x(a,b) = _real_dot(a,b)/state_scale²` used in the tangent norm
  (`:1183`), constraint (`:1214`), `denom` (`:1227`) and direction test
  (`:1245`)
- full Newton — `lin` rebuilt at `Xc` every corrector iteration (`:1222`)
- step growth on cheap steps (`:1253`)

It is reached only from `--column-arclength-recovery`
(`run_gain_map.py:1632`). Every arclength result in the investigation doc came
from the **unscaled** function via one of:

| entry point | call site |
| --- | --- |
| `--inproc-continuation arclength` | `run_gain_map.py:896` |
| `--fold-policy arclength` (`_recover`) | `run_gain_map.py:2065` |
| `--fold-follow` → `fold_power` | `run_gain_map.py:2102`, `solver.py:1367` |

### Secondary defects in `solve_arclength`

| # | location | defect | consequence |
| --- | --- | --- | --- |
| S1 | `:1088-1099` | modified Newton: `lin` and `b` frozen at `X_pred` | a frozen near-singular Jacobian is worst exactly at the fold |
| S2 | `:1091,1099` | bordering by block elimination (`J a = −R`, `J b = S`) | Keller block elimination is unstable when `J` is singular (Chan; Govaerts). Both solves blow up at the fold and the regularising constraint is applied only afterwards |
| S3 | `:1111-1114` | `ds` only halves, floors at `1e-4`, never grows | one bad corrector permanently cripples the trace |
| S4 | `:1118` | new tangent from `lin(S)` at the **predictor** factor | tangent systematically O(ds) wrong |
| S5 | `:1073` | `tol = max(newton_tol*100, 1e-7)`, 100x looser than production Newton | corrector "converges" to points the outer solve then rejects |

### No singularity handling anywhere

`grep` over `src/twpa_solver/pump/backends/fast_coupled.py` finds no
`singular`, `isfinite`, or `nan` guard. At a fold the exact real-coupled
Jacobian **is** singular, so:

- PARDISO fails, the `except` at `fast_coupled.py:496-514` falls back to
  `spla.splu`, which on a singular matrix raises `RuntimeError` or produces
  inf/nan pivots silently;
- `solve_arclength` catches only `GmresDeadlineExceeded`
  (`solver.py:1107,1119`), and `_recover`'s block (`run_gain_map.py:2064-2071`)
  is `try/finally` with no `except`, so a singular-factor `RuntimeError`
  escapes into the map loop.

The preconditioner is exact for the Schur backend — `part.schur[h]` is the
true per-harmonic Schur complement (`schur_operators.py:264,322`) and
`precond_ell_diff_max`/`precond_ell_sum_max` default to `None`, i.e. no
truncation — which is what makes the `gmres_maxiter=20` recovery cap safe, and
also what makes the assembled `FastCoupledPreconditioner.M`
(`fast_coupled.py:222`) usable as the exact Jacobian for a test function.

### One technique worth borrowing from JosephsonCircuits.jl

JC exposes a `factorization` keyword on `hbsolve`/`hbnlsolve`:
`KLUfactorization()` (default, fastest), `LUfactorization()`, and
`QRfactorization()` — documented as "typically the slowest but can solve
systems which have singular matrices. If you get a SingularException error,
try this option." That is the direct analogue of the gap above: a
least-squares-capable fallback factorization for the one regime where LU is
undefined.

Scope note: this is a **numerical-technique borrow only**. Per
`jc-is-not-a-reference` and the CLAUDE.md entry retiring the 0.2 dB JC gate,
nothing in this plan compares physics against JC or treats JC output as a
reference. QR-vs-LU is a linear-algebra choice, not a validation.

Second scope note: QR at a singular point returns *a* least-squares solution;
it does not round a fold. It buys a clean, finite corrector step and a real
error message instead of an uncaught `RuntimeError`. Phase 1 is what rounds
the fold.

---

## What we're NOT doing

- No physics comparison against JosephsonCircuits.jl, and no re-opening of the
  retired JC gain gate.
- No port-convention work (`ports.py` / `observables.py` KCL inconsistency
  stays open and untouched — see CLAUDE.md "Port power convention").
- No changes to `trace_arclength_from_two_points`, which is already correct.
- No changes to the memory-leak `gc.collect()` placements or the arclength
  `ds=0.02` / `gmres_maxiter=20` / `max_steps=60` recovery settings shipped in
  the 2026-08-06 session, except where Phase 2 supersedes them.
- No re-running of exp20/exp21/exp22 or any published P1dB.
- No deflation implementation unless Phase 5 measurements demand it (Phase 6
  is conditional and explicitly gated).
- No change to production map defaults. Every new behaviour is opt-in until
  Phase 5 reports.

## Prerequisites

- [ ] Confirm the working tree imports the repo copy, not the editable
      shadow: `python -c "import twpa_solver; print(twpa_solver.__file__)"`
      must print under `D:\Projects\Thesis\twpa_jax\src`
      (see memory `editable-install-shadows-repo`).
- [ ] Baseline the three existing arclength tests green:
      `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\arc tests/test_advanced_continuation.py`

---

## Phase 1: Scale the arclength metric

### Overview

Make `solve_arclength` invariant under rescaling of the state units. This is
the fix; every later phase is hardening or measurement.

### Changes required

#### 1. Scaled metric in `solve_arclength`

**File**: `src/twpa_solver/pump/solver.py` (`:1028-1138`)

No secant is available at a cold start, so derive the scale from the initial
tangent, which equalises the two contributions by construction (standard
Rheinboldt / theta-weighted scaling):

```
state_scale = ||Xdot_raw|| / |lam_dot_raw|          # = ||J^-1 S||, lam_dot_raw = 1
state_scale = max(state_scale, sqrt(_real_dot(X, X)) * 1e-6, 1e-300)
metric_x(a, b) = _real_dot(a, b) / state_scale**2
```

matching the floors already used at `:1174`. Replace `_real_dot` with
`metric_x` at the tangent normalisation (`:1070`), the constraint (`:1095`),
`denom` (`:1100`), `d_lam` (`:1103`) and the direction test (`:1125`).
`state_scale` is computed once per call and reported in `info` (mirroring
`:1192`).

Degenerate case: if `Xdot_raw` is not finite or `state_scale` hits the
`1e-300` floor, return early with `terminal_reason="degenerate_tangent"`
rather than dividing by it.

#### 2. Regression gate: unit-rescaling invariance

**File**: `tests/test_advanced_continuation.py`

The existing fixture is O(1) in every quantity, which is why the bug survived.
Add a scaled builder alongside `_build_problem` (`:25`). The residual
`D(w)X + Bphi·Ic·sin(psi/phi0) − S` is exactly covariant under
`X → sX` when `phi0 → s·phi0`, `Ic → s·Ic`, `pump_current → s·pump_current`,
with `C`, `G`, `K` and `lambda` unchanged — so the solution scales by exactly
`s` and every `lambda` along the branch is identical.

New tests:

- `test_arclength_is_invariant_under_state_rescaling` — run
  `solve_arclength` at `s=1` and `s=1e-15`; assert the accepted `lambda`
  sequence and `info["reached_target"]` agree, and `X_scaled ≈ s * X_unit` to
  a relative tolerance.
- `test_arclength_reports_state_scale` — `info["state_scale"]` present,
  finite, and tracking `s`.

**Mutation discipline** (memory `verify-agent-claims-tree-and-mutation`):
record both new tests **failing** on the unpatched function before the fix
lands, and paste the failure output in the PR/commit body. The invariance test
must fail at `s=1e-15` and pass at `s=1`.

### Success criteria

**Automated**: `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\arc tests/test_advanced_continuation.py` — 5 pre-existing tests still green, 2 new tests green, and both new tests shown failing pre-fix.

**Manual**: on the toy problem at `s=1e-15`, `info["state_scale"]` is ~`1e-15`, not `1.0`.

---

## Phase 2: Corrector and step-control hardening

### Overview

Bring `solve_arclength` up to the standard `trace_arclength_from_two_points`
already meets. Independent of Phase 1 but pointless without it.

### Changes required

**File**: `src/twpa_solver/pump/solver.py` (`:1074-1138`)

- **S1/S4** — rebuild `lin` at `Xc` each corrector iteration (as `:1222`), and
  take the new tangent from the corrected point's factor, not the predictor's.
  Cost: one factorization per corrector iteration instead of per step. The
  `gmres_maxiter=20` recovery cap at `run_gain_map.py:~2050` stays.
- **S3** — grow the step on cheap correctors,
  `step_size = min(ds_initial, step_size * 1.25)` when the corrector took
  ≤3 iterations (as `:1253`); keep the halving and the `1e-4` floor.
- **S5** — tighten to `tol = max(newton_tol * 10.0, 1e-8)`, matching `:1197`.
  Flag in `info` when the returned point is above the production
  `newton_tol`, since the endpoint is consumed as a warm guess
  (`run_gain_map.py:2073`) and a loose endpoint silently wastes a solve.

### Success criteria

**Automated**: full `tests/test_advanced_continuation.py` green; `test_arclength_matches_direct_solution` (`:86`) tolerance tightened from `atol=1e-5` to at least `1e-7` and still passing.

**Manual**: on the fp=7.0 GHz, −20.069 dBm point seeded from its true production neighbour (−20.414 dBm, lambda0=0.961), the corrector iteration count per accepted step is recorded; compare against the Phase 1 run.

---

## Phase 3: Stable bordered solve and a singular-factor fallback

### Overview

Address S2 and the missing singularity handling. Block elimination is exactly
the operation that breaks where this method is supposed to work.

### Changes required

#### 1. One-step iterative refinement on the block elimination

**File**: `src/twpa_solver/pump/solver.py`

Govaerts & Pryce, *Block elimination with one refinement solves bordered
linear systems accurately* (BIT 30, 1990): one extra back-solve on the
existing factor restores accuracy through the singularity. Preferred over a
direct bordered factorization because it reuses `FastCoupledPreconditioner`'s
cached symbolic factorization and adds one solve per corrector iteration.

Fallback if refinement is insufficient: assemble the bordered
`[[M, c], [dᵀ, e]]` of size `2*H*n + 1` from `FastCoupledPreconditioner.M`
(`fast_coupled.py:222`) and factor it directly — nonsingular at a simple fold,
but pays a fresh symbolic factorization per step, so recovery path only, never
the hot loop.

#### 2. Least-squares fallback factorization

**File**: `src/twpa_solver/pump/backends/fast_coupled.py`

Add a `singular_fallback` path to `refactor`/`solve`: on
`RuntimeError`/singular pivots or a non-finite solve result, fall back to a
sparse least-squares solve (`scipy.sparse.linalg.lsqr` or a sparse QR if one
is available in the environment) and set an explicit
`last_factor_backend="lsq_singular_fallback"`. This is the analogue of JC's
`QRfactorization()` option and is a linear-algebra robustness measure only.

Report it: the fallback firing is itself the strongest available evidence that
the Jacobian went singular, so surface it in `info`/the map row rather than
swallowing it.

#### 3. Catch singular factorization at the call sites

**Files**: `src/twpa_solver/pump/solver.py:1087-1109`,
`scripts/run_gain_map.py:2064-2071`

Widen the `except GmresDeadlineExceeded` to also catch the singular-factor
`RuntimeError` and terminate with `terminal_reason="singular_jacobian"`;
convert `_recover`'s bare `try/finally` into `try/except/finally` so a
singular factor downgrades that cell instead of escaping into the map loop.

### Success criteria

**Automated**: new `tests/test_bordered_solve.py` — on a small problem with a deliberately near-singular Jacobian, the refined block elimination's `(d_X, d_lam)` matches a dense `numpy.linalg.solve` of the full bordered system to a stated tolerance where plain block elimination does not. A synthetic exactly-singular matrix drives `refactor` into the fallback and returns a finite solve with `last_factor_backend == "lsq_singular_fallback"`, no exception.

**Manual**: the fp=7.0 GHz point no longer terminates with an uncaught exception under any `ds`.

---

## Phase 4: Fold test function

### Overview

Nothing in `src/twpa_solver/pump/` tracks a singularity measure today
(`grep` for sigma_min / det / condition returns nothing). This is the
measurement that answers the actual question, and it is cheap because the
exact Jacobian is already assembled and factored every Newton step.

### Changes required

#### 1. New module

**File**: `src/twpa_solver/pump/singularity.py`

- `jacobian_min_eigenvalue(problem, X, lam, *, iters=20) -> float`
  Inverse power iteration `v ← M⁻¹v / ‖M⁻¹v‖` using
  `problem.assemble_real_coupled_fast(tangent).solve`. Converges to the
  smallest-magnitude **eigenvalue**, which is the right object: a fold is a
  zero eigenvalue, not merely a small singular value. Needs only `solve`, no
  transpose, so it works on both the PARDISO and SuperLU paths. ~20
  back-solves, seconds per point.
- `jacobian_det_signature(problem, X, lam) -> tuple[int, float]`
  `(sign, log|det|)` from `sum(log|diag(U)|)` plus the parity of `perm_r` and
  `perm_c`. Requires a `SuperLU`, so call
  `problem.assemble_real_coupled_preconditioner(spectral)`
  (`schur_operators.py:295`) — a separate diagnostic-only factorization that
  does not disturb the hot path's cached PARDISO factor.

Reading: a **sign change of det between adjacent converged points means an odd
number of eigenvalues crossed zero — a genuine simple singularity was
passed.** `min_eigenvalue → 0` smoothly as power → wall means a fold and gives
the fold power. It staying at its low-power magnitude while Newton fails means
the wall is numerical.

#### 2. Tests

**File**: `tests/test_singularity.py`

- Both functions on a small dense problem with a known spectrum, checked
  against `numpy.linalg.eigvals` / `numpy.linalg.slogdet`.
- Sign of `det` flips across a constructed turning point.
- `min_eigenvalue` is invariant under the Phase 1 state rescaling (`s=1` vs
  `s=1e-15`) up to the expected `s` power — the same trap as Phase 1, closed
  before it can bite.

### Success criteria

**Automated**: `pytest tests/test_singularity.py` green, every assertion mutation-verified.

**Manual**: both functions run on one real 2c Schur pump state without exceeding the per-point memory budget.

---

## Phase 5: Measure, then rewrite the conclusions

### Overview

Re-take the two decisive measurements with instruments that work, and correct
the investigation doc. **This is the phase the thesis actually needs**; Phases
1-4 exist to make it possible.

### Changes required

#### 1. Diagnostic driver

**File**: `scripts/scan_branch_singularity.py`

Marches one column via `InProcessEngine` (same pattern as
`scripts/resume_column_force_gain.py`), and at every converged point records
`pump_power_dbm, lambda, coeff_rel, min_eigenvalue, det_sign, log_abs_det,
peak_i_over_ic` to `singularity_scan.csv` plus a PNG. Takes the same engine
flags as `run_gain_map.py`. Standalone, not wired into the production map.

Windows note: use a short `--outdir` (MAX_PATH — same caveat as the matrix
tracer in CLAUDE.md).

#### 2. Re-run the two voided measurements

- fp=7.0 GHz, −20.069 dBm, seeded from its **true production neighbour**
  (−20.414 dBm, lambda0=0.961) — the methodology correction the investigation
  doc already records.
- the `--fold-follow` sweep, 7.6-8.5 GHz, 19 frequencies.

Both under the fixed `solve_arclength`, both with the singularity scan
alongside. Also scan fp=7.9 GHz, where the doc claims no fold and reports the
peak-current hot spot hopping across cell indices 1250-2500.

#### 3. Correct the record

**File**: `docs/development/2c_convergence_arclength_and_port_convention_investigation_2026-08-06.md`

Add a dated addendum (do not rewrite history — same convention as the
superseded port-convention entry in CLAUDE.md):

- §2b "genuine, frequency-dependent fold ... confirmed physically at two
  points" — the confirming instrument was degenerate; replace with the
  measured verdict.
- §2d "a `--fold-follow` sweep across 19 frequencies found zero folds" — that
  function could not return anything else; replace with the re-run.
- §3 net conclusions — re-derive.

Then CLAUDE.md and a memory entry recording what the wall actually is.

### Success criteria

**Automated**: none — this phase is measurement.

**Manual**: for each scanned frequency, a stated verdict of `FOLD` (min_eigenvalue → 0, det sign flips, fold power quoted), `NUMERICAL` (min_eigenvalue flat while Newton fails), or `SNAKING` (repeated near-zeros / multiple sign flips), each backed by the CSV.

---

## Phase 6 (conditional): deflation

### Gate

Run **only if** Phase 5 reports `SNAKING` — repeated `min_eigenvalue`
near-zeros or multiple `det` sign flips within the −23..−19 dBm band.

### Rationale

The hot-spot migration across cells 1250-2500 recorded in the investigation
doc is the signature of spatially localized states in a bistable discrete
lattice (homoclinic snaking: the branch folds repeatedly as the localized
region grows cell by cell, and can break into isolas). If that is what 2c
does, no step-size control reaches lambda=1 by continuation in lambda — the
branch genuinely turns back many times and the requested grid point may lie on
a **disconnected** branch. Deflation (Farrell et al.) seeded directly at the
target lambda finds the coexisting branch without traversing the snake.
Memory `continuation-campaign-fold-locked` already concluded "only
arclength/deflation can help"; arclength was never actually running, so that
conclusion is untested in both halves.

### Scope if triggered

Deflation operator on the existing residual/JVP, applied at the target lambda
with known-converged neighbours as deflation points. Separate plan; do not
start it inside this one.

---

## Testing strategy

### Project maturity level

Active development, with an established gate culture (95 mutation-verified
tests in the component-profile suite; CLAUDE.md requires the full slow suite
with a `--basetemp` outside the repo).

### Unit tests

- Phase 1: unit-rescaling invariance of `solve_arclength` (`s=1` vs `s=1e-15`);
  `state_scale` reported; degenerate-tangent early return.
- Phase 2: tightened `test_arclength_matches_direct_solution`; step growth
  observed on a cheap trace.
- Phase 3: refined block elimination vs a dense bordered solve near
  singularity; singular-matrix fallback returns finite with the right backend
  tag; singular `RuntimeError` is caught at both call sites.
- Phase 4: `min_eigenvalue` and `det_signature` against dense reference;
  det sign flips across a turning point; rescaling invariance.
- Every assertion mutation-verified — shown failing before the fix.

Coverage target: the four touched code paths, not a global percentage.

### Integration tests

- Full suite, per CLAUDE.md:
  `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_full_slow --run-slow`
- Regression: a short 2c column under production flags with `--fold-policy
  patience` (the default) must produce **bit-identical** PASS/FAIL/SKIP
  outcomes before and after Phases 1-3. None of these changes may touch the
  default map path.
- `--column-arclength-recovery` (the already-correct scaled tracer) unchanged
  in behaviour.

### Manual verification

The fp=7.0 GHz and fp=7.9 GHz scans of Phase 5, read against the criteria in
that phase.

---

## Rollback plan

Phases 1-4 are additive or confined to `solve_arclength`,
`fast_coupled.py`'s fallback path, two new modules and two new test files.

- Phase 1-2: revert `solve_arclength` to the current body; the new tests fail
  and are reverted with it. Nothing else imports the metric.
- Phase 3: the fallback is opt-in inside `refactor`'s exception path; reverting
  restores the current uncaught `RuntimeError`. The widened `except` clauses
  revert independently.
- Phase 4-5: new files, delete them.
- Production maps are unaffected at every phase because `--fold-policy` stays
  at `patience` and `--inproc-continuation` at `adaptive_secant`; the
  bit-identical column regression above is the guard.

Commit per phase so any single phase reverts cleanly. Per memory
`no-coauthor-trailer-in-commits`, no `Co-Authored-By` trailer.

---

## Sources

- Govaerts, *Stable Solvers and Block Elimination for Bordered Systems*,
  SIAM J. Matrix Anal. Appl. — https://epubs.siam.org/doi/10.1137/0612034
- Govaerts & Pryce, *Block elimination with one refinement solves bordered
  linear systems accurately*, BIT —
  https://link.springer.com/article/10.1007/BF01931663
- Chan, *Deflation Techniques and Block-Elimination Algorithms for Solving
  Bordered Singular Systems*, SIAM J. Sci. Stat. Comput. —
  https://epubs.siam.org/doi/10.1137/0905009
- *Snaking and isolas of localised states in bistable discrete lattices* —
  https://arxiv.org/pdf/1011.0307
- *Practical implementation of pseudo-arclength continuation to ensure
  consistent path direction* —
  https://www.sciencedirect.com/science/article/abs/pii/S0094576523006379
- JosephsonCircuits.jl factorization options (numerical technique only, not a
  physics reference) — https://github.com/kpobrien/JosephsonCircuits.jl
