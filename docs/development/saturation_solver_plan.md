## Implementation Plan: Nonlinear Multitone (2D Torus HB) Saturation Solver

### Goal
`python scripts/run_compression.py ...` returns a reproducible gain-compression curve,
pump-depletion curve, idler/spur powers, spatial phase-mismatch profiles and a refined
P1dB for one circuit at one pump operating point, from a fully nonlinear multitone
harmonic-balance solve in which the pump coefficients are unknowns — with a validated
small-signal limit `G_multitone(P_s->0) == G_Floquet` to <0.05 dB.

### Current State Analysis

**Reusable as-is (no changes needed)**
- `HarmonicNewtonKrylovSolver` (`src/twpa_solver/pump/solver.py:130`) is duck-typed on
  `problem`. Contract: state shape `(H, n)` complex, attribute `.H`, and methods
  `zeros`, `source_coeffs(scale)`, `residual_coeffs(X, scale)`,
  `norms(X, scale, compute_time_residual)`, `tangent_state(X)` -> object with
  `.gamma_t (nt, nb)` / `.gamma_mean (nb,)`, `jvp_coeffs_with_tangent(V, tangent)`,
  `spectral_tangent_state(tangent)`, `build_preconditioner_factors(X, mode, tangent=)`,
  `assemble_coupled_preconditioner`, `assemble_real_coupled_preconditioner`, optional
  `assemble_real_coupled_fast`. `pack_complex`/`unpack_complex`
  (`pump/problem.py:76,81`) are shape-generic. A `(n_tones, m)`-shaped multitone problem
  therefore inherits Newton, GMRES, line search, stall detection, the mid-GMRES deadline
  abort, `solve_continuation`, `solve_adaptive_continuation`, `solve_affine_continuation`,
  `solve_pseudo_transient` and `solve_arclength` **without touching the controller**.
- `build_partition(linear_blocks, bphi, port_indices)` (`pump/backends/schur_partition.py:82`)
  already accepts an arbitrary list of per-frequency linear blocks; `signal/floquet.py:427`
  already uses it for signal sidebands. Multitone Schur reduction = hand it `[D_v for v in B]`.
- `assemble_schur_complements` (`schur_partition.py:153`) produces sparse `S_v`, constant
  for fixed `(omega_p, omega_s, B)` — one factorization reused across the entire
  signal-power sweep, exactly as required.
- `dynamic_block(circuit, omega, loss_model=...)` (`core/linear.py:23`) is already per-omega
  with seven loss conventions -> per-tone `D_v` and tone-dependent loss drop straight in.
- `port_s_from_unit_current_response` (`core/linear.py:70`) and `GainResult.gain_vs_off_db`
  (`signal/gain.py:48`) — `gain_vs_off` in `solve_gain_one` (`signal/floquet.py:349`) is
  already `|V_on / V_pump_off|^2`, i.e. the paper's normalized signal gain
  `S21(pump on) - S21(pump off)`. Reference `vout_off` uses `khat_off_0`.
- `signal/stability.py` (`estimate_sigma_min`, `refine_complex_resonance`) exists as the
  hook for the later Floquet-stability phase.

**Constraints discovered**
- `HarmonicGrid` (`pump/problem.py:14`) synthesizes with a **dense exponent matrix**
  `E = exp(i*omega*t_r*k_j)` (`problem.py:41`), not an FFT. The 2D torus is genuinely new
  code; it must not reuse `HarmonicGrid`.
- `FastCoupledPreconditioner` (`pump/backends/fast_coupled.py:129`) is scalar-mode indexed
  (`self.modes = [int(round(k)) for k in problem.grid.k]`, `_needed_ells` = `{k-q} | {k+q}`,
  `_ell_index` a dict). Every index operation is set/dict arithmetic and
  `_phase_matrix` is `(n_ells, nt)` applied as `_phase_matrix @ gamma_t`. Replacing scalar
  modes with `(h,q)` tuples and `theta` with the flattened torus phase is **mechanical**;
  the khat data map (`_KhatDataMap`) and the `_W` scatter are already index-agnostic.
- `tangent_predictor` (`pump/solver.py:930`) hardcodes `dR/dlambda = -source_coeffs(1.0)`,
  valid only for `S(0) = 0`. Affine paths need `source_coeffs(1) - source_coeffs(0)`.
- Source convention: `FullPumpProblem.source_coeffs` (`problem.py:170`) writes
  `0.5 * scale * I_pump` because the real reconstruction is `2 Re sum`. A finite signal
  source must use the same `0.5 * I_s` factor at the source node.
- `scripts/compression_sweep.py` (2036 lines) is dead: it subprocesses
  `scripts/full_pump_hb_100mm.py` and `scripts/gain_from_pumped_solution.py`, neither of
  which exists in the repo.
- Real-size anchor: `fast_coupled.py` docstring records a 50360x50360 real-packed Jacobian
  for the production 2c device at H=10 -> **m ~= 2518 retained nodes**. Multitone sizes at
  that device (below) drive the resource guard and the sector-preconditioner phase.

  | basis | N_tones | N_real = 2*N*m | super-blocks (N^2) vs production |
  |---|---|---|---|
  | `three_tone` | 3 | 15,108 | 9 (0.09x) |
  | Q=1, pump order 10 | 30 | 151,080 | 900 (9x) |
  | Q=2, pump order 10 | 50 | 251,800 | 2500 (25x) |

  The exact global preconditioner is viable at `three_tone` and Q=1, and is the reason
  Q>=2 needs `floquet_sector`.

### What We're NOT Doing
- No modification to `HarmonicGrid`, `FullPumpProblem`, `SchurReducedProblem`,
  `signal/floquet.py` numerics, or `scripts/run_gain_map.py` behaviour. The 7-design JC
  parity and existing gain maps must stay byte-for-byte identical.
- No compression at every pump-map cell; no 3-D (pump freq x pump power x signal power)
  campaign driver.
- No arbitrary multifrequency input combs, no more than two independent fundamentals, no
  DC/biased-design multitone (`(0,0)` tone excluded), no thermal models, no chaotic-branch
  analysis.
- No automatic stability classification in this plan. Every output carries
  `stability_status = "NOT_CHECKED"`; Phase 8 is scoped but not planned in detail.
- No local-state-dependent loss (loss Stage 4). It would make the linear operator
  state-dependent and require a new JVP term.
- No GPU/JAX path. NumPy/SciPy only, per scope.

### Prerequisites
- [ ] Working tree clean or changes committed. Current diff touches
      `scripts/run_gain_map_column_matrices.py`, `src/twpa_solver/core/linear.py`,
      `src/twpa_solver/pump/backends/fast_coupled.py` — commit or stash before Phase 0
      so the baseline freeze captures a known state.
- [ ] `pypardiso` importable (`pardiso_available()` True) — otherwise the multitone
      preconditioner falls back to SuperLU and Q=1 runtimes will be several times worse.
- [ ] A tiny validation fixture: `twpa_solver.builders.jc_doc.build_jpa()` (single-junc
      for the final runs.
- [ ] Tests run with `--basetemp` off the repo (Windows ACL issue on `.pytest_tmp`, per
      CLAUDE.md).

---

## Phase 0: Freeze the baseline

### Overview
Pin the current pump and linear-signal numerics so any later regression is unambiguous,
and remove the dead compression script.

### Changes Required

#### 1. Baseline regression fixtures
**File**: `tests/test_baseline_freeze.py` (new)
**Changes**: Golden-value tests, tolerances `rtol=1e-12`:
- `build_jpa()` pump solve at a fixed `(f_p, I_p, modes, nt)` -> assert
  `report.coeff_rel`, `report.newton_iterations` and `X` checksum against literals
  captured on first run.
- `solve_gain_one` and `solve_gain_one_schur` on the same fixture -> assert `gain_db`,
  `gain_vs_off_db`, `linear_rel_residual` match, and that the two agree with each other.
- `port_s_from_unit_current_response` unit cases including the `source_port == out_port`
  branch (`core/linear.py:78`).
- `FullPumpProblem.source_coeffs(1.0)` normalization: assert the nonzero entry equals
  `0.5 * pump_current_a` and that `source_time` is `I_p cos(omega t)`, locking the
  factor-of-two convention the compression axis depends on.
- `assemble_real_coupled_fast` on the Schur JPA problem -> assert GMRES converges in
  <= 2 iterations and the resulting `X` matches the `real_coupled` path.

#### 2. Delete dead script
**File**: `scripts/compression_sweep.py`
**Changes**: Delete. Confirm nothing imports it
**Automated**: `pytest tests/ --basetemp=<off-repo>` green, including the new
`tests/test_baseline_freeze.py`.
**Manual**: `git grep compression_sweep` returns nothing outside history.

---

## Phase 1: Tone basis and torus transform

### Overview
The `(h,q)` lattice, its canonicalization, and an exact 2D FFT synthesis/projection pair.
No circuit physics yet.

### Changes Required

#### 1. Tone basis
**File**: `src/twpa_solver/multitone/basis.py` (new)
**Changes**:
- `ToneIndex` — frozen dataclass `(h: int, q: int)`, hashable, with
  `omega(omega_p, delta) -> float`, `conjugate() -> ToneIndex` (`(-h,-q)`),
  `__add__`/`__sub__` for the componentwise offset arithmetic the preconditioner needs.
- `canonicalize(tone, omega_p, delta) -> tuple[ToneIndex, bool]` — returns the
  positive-frequency partner and a `conjugated` flag. `(0,0)` raises: DC is out of scope.
- `MultiToneBasis` — ordered `list[ToneIndex]`, `index_of`, `omegas` array,
  `pump_tone = (1,0)`, `signal_tone = (1,-1)`, `idler_tone = (1,1)`,
  `signal_order(tone) -> abs(q)` for sector grouping, `to_metadata()` mirroring
  `PumpBasis.to_metadata` (`pump/basis.py:98`).
  Validation in `__post_init__`: all omegas strictly positive, no duplicates, no
  conjugate pairs both present, pump/signal/idler tones present.
- `build_three_tone_basis(omega_p, delta)` -> `[(1,0), (1,-1), (1,1)]`.
- `build_lattice_basis(pump_modes, signal_order_max, omega_p, delta, omega_max)` — the
  `B` of your §2.2. `pump_modes` accepts the JC odd list via
  `pump.basis.positive_odd_modes`, or a dense list for the even-sector convergence check.

- `__post_init__` aliasing guard: `n_p >= 2*max|h_eff| + 1`, `n_delta >= 2*max|q_eff| + 1`
  where the `_eff` bounds account for the second-order mixing offsets `h+h'`, `q+q'`
  actually generated by the sine (not just the retained basis); both even.
- `scatter_signed_spectrum(X) -> (n_p, n_delta, n_nodes) complex` — place `X_v` at
  `(h mod n_p, q mod n_delta)` and `conj(X_v)` at the conjugate partner, so the ifft2
  output is real to machine precision.
- `synthesize(X) -> (n_p*n_delta, n_nodes) float` — `ifft2` over the first two axes,
  scaled so the result equals `2 Re sum_v X_v exp(i(h*theta_p + q*theta_delta))`, then
  reshaped to the flat torus-point axis. Flat ordering is `r*n_delta + s` and is fixed
  for the whole codebase (the preconditioner's phase matrix depends on it).
- `project(y) -> (n_tones, n_nodes) complex` — `fft2`, normalize by `n_p*n_delta`, gather
  the retained tones. `gather_positive_modes` / gather-scatter are exposed separately for
  tests.
- `theta_flat -> (n_p*n_delta, 2) float` — the `(theta_p, theta_delta)` pairs, consumed by
  the preconditioner's phase matrix.
- No dense `(n_torus, n_tones)` exponent matrix anywhere.

#### 3. Resource estimator
**File**: `src/twpa_solver/multitone/resources.py` (new)
**Changes**: `estimate(basis, grid, n_retained, n_branches, preconditioner) -> ResourceEstimate`
with fields for coefficient state, torus node/junction waveform, JVP workspace, per-sector
or global matrix dimension, predicted factor nnz, and checkpoint size.
`guard(estimate, budget_gb)` raises `ResourceLimitExceeded` **before** any allocation.

### Success Criteria
**Automated**: `pytest tests/test_multitone_basis.py tests/test_torus_grid.py`
- Round trip `project(synthesize(X)) == X` to `1e-14` for random `X` on both
  `three_tone` and a Q=2 lattice.
- `synthesize` output has zero imaginary part to `1e-15` (conjugate symmetry exact).
- Analytic mixing: single tone `cos(theta_p)` cubed projects to `3/4` at `(1,0)` and
  `1/4` at `(3,0)`.
- Two-tone: `sin` of a pump+signal sum generates a nonzero coefficient at

---

## Phase 2: Full-node multitone problem

### Overview
The residual `R_v = D_v X_v + N_v(X) - S_v` and its analytic 2D AFT JVP, with the affine
source path held inside the problem so `solver.py` needs no structural change.

### Changes Required

#### 1. Affine source
**File**: `src/twpa_solver/multitone/source.py` (new)
**Changes**:
- `MultiToneDrive` — `(tone, node_index, current_a)` entries; `to_coeffs(basis, n_nodes)`
  writes `0.5 * current_a` at the tone/node, matching `FullPumpProblem.source_coeffs`.
- `AffineSourcePath(source_start, source_delta)` with `source(tau)`, `derivative(tau)`.
  Constructors `pump_turn_on(S_p)`, `signal_turn_on(S_p, S_s)`,
  `signal_substep(S_p, S_s_old, S_s_target)`.

#### 2. Multitone problem
**File**: `src/twpa_solver/multitone/problem.py` (new)
**Changes**: `FullMultiToneProblem` exposing exactly the solver's duck-typed surface, with
`self.H = basis.n_tones` and state shape `(n_tones, n_nodes)`:
- `_linear_blocks` built via `dynamic_block(circuit, omega_v, loss_model=...)` per tone —
  not the private `FullPumpProblem` builder — so tone-dependent loss is a later drop-in.
- `source_coeffs(tau)` returns `path.source(tau)`; `source_delta_coeffs()` returns
  `path.derivative(tau)` for the tangent predictor.
- `residual_coeffs`, `nonlinear_current_coeffs`, `branch_flux_time` — same shape as the
  1D versions but through `TorusGrid`.
- `tangent_state(X)` returns the existing `hb.TangentState` with `gamma_t` on the flat
  torus axis, so `mean_tangent` preconditioning and every `_finite_state` guard in
  `solver.py` work unchanged.
- `jvp_coeffs_with_tangent(V, tangent)` — 2D AFT: synthesize `V`, `BphiT @`, multiply by
  `gamma_t`, `Bphi @`, project. Identical structure to `problem.py:226`.
- `spectral_tangent_state(tangent)` — `khat` keyed by `ToneIndex` offsets
  `{k-q} | {k+q}` over the tuple lattice.
- `norms`, `time_residual` (torus-domain), `build_preconditioner_factors`
  (`none`/`linear`/`mean_tangent`), `assemble_coupled_preconditioner`,
  `assemble_real_coupled_preconditioner` — direct tuple-mode transcriptions of
  `problem.py:354-471`.

#### 3. Affine tangent predictor fix
**File**: `src/twpa_solver/pump/solver.py`
**Changes**: In `tangent_predictor` (line ~938), replace `S = problem.source_coeffs(1.0)`
with the affine-correct delta:
```python
S = problem.source_coeffs(1.0) - problem.source_coeffs(0.0)
For every existing pump problem source_coeffs(0.0) is exactly zero, so this is a no-op
on the validated path (asserted in Phase 0's freeze test).

Success Criteria

Automated: pytest tests/test_multitone_problem.py on build_jpa()
- Zero signal source: X_{h,0} equals the 1D pump solution to 1e-10 and
X_{h,q!=0} is zero to 1e-12.
- JVP vs central finite difference of residual_coeffs:
||JV - JV_fd|| / ||JV_fd|| < 1e-6 for random V, on three_tone and Q=2.
- spectral_tangent_state JVP agrees with the AFT JVP to 1e-10.
- Very small signal (I_s = 1e-6 * I_p): the (1,-1) output-node coefficient matches the
linear Floquet response from solve_gain_one scaled by I_s, to 1e-3 relative.
- AffineSourcePath cases: pump turn-on reproduces lambda * S_p.
- tangent_predictor regression: existing pump tests unchanged.
Manual: none.

---
Phase 3: Multitone Schur reduction

Overview

Reduce to retained nodes by reusing the existing partition machinery over the multitone
frequency list.

Changes Required

1. Schur multitone problem

File: src/twpa_solver/multitone/schur.py (new)
Changes: SchurMultiToneProblem, structurally the tuple-mode analogue of
SchurReducedProblem (pump/backends/schur_operators.py:48):
- build_multitone_schur_problem(full, port_indices) calls the existing
build_partition(full._linear_blocks, full.Bphi, port_indices) +
assemble_schur_complements. Retained set = Josephson-incident nodes + pump port +
signal source port + out port + any --diagnostic-port.
- Same method surface as the full problem; self.n = part.m.
- reconstruct_full(Xn) via back_substitute_full.
- The partition is built once per (omega_p, omega_s, basis) and cached on the driver, so
the D_ee,v factorizations are reused across every signal power, continuation step,
Newton iteration and GMRES call.

Success Criteria

Automated: pytest tests/test_multitone_schur.py on build_jpa() and a 20-cell
build_jtwpa()
- Full vs Schur nonlinear root: ||X_full[:, retained] - X_schur|| / ||X_full|| < 1e-9.
- reconstruct_full(X_schur) reproduces X_full on eliminated nodes to 1e-9.
- Residual of the reconstructed full state < 1e-9 relative.
- Output-node port quantities identical between backends to 1e-10.
Manual: none.

---
Phase 4: Preconditioners

Overview

Two preconditioners, both selectable: the exact global one (generalizing the production
fast path to tuple modes) as default, and the Floquet-sector one behind a flag.

Changes Required

1. Tuple-mode generalization of the fast coupled preconditioner

File: src/twpa_solver/pump/backends/fast_coupled.py
Changes: Make FastCoupledPreconditioner mode-key agnostic. It already only needs
modes to support -, +, hashing and equality:
- __init__: replace self.modes = [int(round(k)) for k in problem.grid.k] with
self.modes = list(problem.mode_keys), and add a mode_keys property to
SchurReducedProblem returning the existing scalar ints (backward compatible).
- _phase_matrix: build from problem.grid.phase_rows(self._ells) — a new grid method.
HarmonicGrid.phase_rows(ells) returns exp(-i*ell*theta)/nt (today's behaviour);
TorusGrid.phase_rows(ells) returns
exp(-i*(lh*theta_p + lq*theta_delta))/(n_p*n_delta) on the flat torus axis. Both are
(n_ells, n_grid) applied as _phase_matrix @ gamma_t — the existing contract.
- _needed_ells, _ell_index, src_seg: unchanged code; they already work on any
hashable key with -/+.
This is refactor-only for the pump path and is covered by Phase 0's freeze test.

2. Floquet-sector preconditioner

File: src/twpa_solver/multitone/preconditioners.py (new)
Changes:
- FloquetSectorPreconditioner — group tones by |q|; per sector assemble
`(M V){h,q} = S{h,q} V_{h,q} + sum_h' [ Khat_{(h-h',0)} V_{h',q}
  - Khat_{(h+h',0)} conj(V_{h',-q}) ], keeping only the detuning-independent tangent component gamma_hat_{l,0}. |q|and-|q|are solved in one block because the conjugate term couples them. Fixed sparse pattern per sector, symbolic factorization reused, numeric refresh per Newton step, GMRES-count-triggered refresh — same policy asfast_coupled, with PARDISO and SuperLU fallback telemetry (last_factor_backend, last_assembly_runtime_s, last_factor_runtime_s) so the existing solve_one timing hooks (solver.py:305`) light up unchanged.
- LinearSectorPreconditioner, MeanTangentSectorPreconditioner — reference/debu
{real_coupled_fast, floquet_sector, spectral_coupled, mean_tangent, linear, none}.
- The problem's assemble_real_coupled_fast(tangent) dispatches to whichever is selected,
so NewtonKrylovSettings.preconditioner = "real_coupled_fast" reaches both.

Success Criteria

Automated: pytest tests/test_multitone_precond.py tests/test_pump_solvers_schur.py
- Existing pump real_coupled_fast results bit-identical to Phase 0 goldens.
- On build_jtwpa() Q=1: real_coupled_fast gives GMRES <= 2 iterations per Newton step.
- floquet_sector reaches the same converged root; record its GMRES count.
- GMRES iterations with floquet_sector stay bounded (<= 3x the Q=1 count) going
Q=1 -> Q=2 -> Q=3 on the small fixture, with no global all-tone LU built
(assert via the resource estimator).
- resources.guard raises before allocation for a deliberately oversized config.
Manual: On the production 2c circuit, record wall time and peak RSS per Newton step
for both preconditioners at three_tone, Q=1 and Q=2; that measurement selects the
production default and is written into CLAUDE.md.

---
Phase 5: Seeding and signal-power continuation

Overview

Get from a converged pump state to a converged finite-signal state, and walk up in signal
power without cold-solving each point.

Changes Required

1. Seeds

File: src/twpa_solver/multitone/seed.py (new)
Changes:
- promote_pump_solution(X_pump, pump_basis, multitone_basis) — X_{h,0} = X_{p,
omega_j = omega_s + m*omega_p to (h,q) = (m+1, -1), canonicalize (conjugating when
omega_j < 0), scale by the actual signal source amplitude. Frequencies must match a
basis tone to 1e-9 relative or the mapping raises.
- X0 = promote_pump_solution(...) + seed_from_floquet(...).

2. Signal-power continuation

File: src/twpa_solver/multitone/compression.py (new, continuation half)
Changes:
- solve_signal_power_point(engine, X_prev, X_prevprev, I_s_target, ...) -> PointResult.
First point uses AffineSourcePath.signal_turn_on + solve_adaptive_continuation.
Later points: direct solve_one(tau=1) from the previous state, then the signal-current
secant predictor X_pred = X_j + (I_target - I_j)/(I_j - I_{j-1}) * (X_j - X_{j-1})
(current amplitude, not dBm).
- Recovery ladder, in order: (1) plain previous state, (2) pump + scaled Floquet seed,
(3) adaptive signal-amplitude substeps via AffineSourcePath.signal_substep down to
--signal-substep-min-db, (4) optional solve_arclength in signal amplitude.
- Distinct status strings — SIGNAL_CONTINUATION_FAILED, SIGNAL_SUBSTEP_STALL,
NONFINITE_STATE, DEADLINE — never reusing the pump-fold vocabulary. Per the repo's
terminology memory, a failure is reported as a convergence failure, not a fold,
unless physically verified.

Success Criteria

Automated: pytest tests/test_multitone_seed.py tests/test_signal_continuation.py
- On build_jpa(): the Floquet-seeded first point converges in <= 3 Newton iterations at
a signal power 60 dB below the pump.
- A 15-point log sweep converges with zero cold solves; assert every point after the
first used <= 5 Newton iterations.
- Seed frequency mapping: every Floquet sideband maps to a basis tone or raise

Overview

Everything read off a converged multitone state, in both absolute and paper-normalized
form, complex not just power.

Changes Required

1. Port waves and tone observables

File: src/twpa_solver/core/linear.py
Changes: Add (purely additive, existing functions untouched)
port_waves(v, i, z0) -> (a, b) with a = (V + Z0 I)/(2 sqrt(Z0)),
b = (V - Z0 I)/(2 sqrt(Z0)).

File: src/twpa_solver/multitone/observables.py (new)
Changes:
- extract_port_waves(X_full, basis, circuit, ports, z0) -> per tone per port
a, b, |a|^2, |b|^2.
- tone_s21(X_full, basis, ...) -> complex S21 per tone using
port_s_from_unit_current_response divided by the actual source current, so the
small-signal limit reproduces GainResult.gain_db exactly. The a/b waves are reported
alongside as an independent cross-check, never as the primary gain definition.
- reference_states — four solves per operating point:
pump_off_signal_on (multitone with S_p = 0), pump_on_signal_infinitesimal
(existing linear Floquet, G_0), pump_on_signal_finite, pump_on_signal_off
(the pump-only state). Normalized outputs:
G_s = 20log10|S21_s(pump on)| - 20log10|S21_s(pump off)|,
D_p = 20log10|S21_p(signal on)| - 20log10|S21_p(signal off)|.
The pump-off reference is computed at the same signal power (self-Kerr included);
the khat_off_0 linear reference from signal/floquet.py:329 is reported too, and the
two must agree at small signal.
- junction_diagnostics(X_full, ...) -> per junction
max|psi_b/phi0|, max|sin(psi_b/phi0)|, min cos(psi_b/phi0), evaluated on the full
torus. Documented note that |I_J/I_c| <= 1 by construction and is not a damage metric.

2. Compression curve and P1dB

File: src/twpa_solver/multitone/compression.py (sweep half)
Changes:
- run_compression_sweep(...) — coarse log signal-power sweep, C(P_s) = G_0 - G_NL(P_s).
- refine_p1db(...) — bracket the first C >= 1 dB crossing, then bracketed
secant/bisection with a real nonlinear solve at each refinement, warm-started from
the nearest converged state, until the bracket is under --p1db-power-tol-db. No
spline-only answer.
- Nonmonotonic handling: report first_1db_crossing_dbm, number_of_crossings,
nonmonotonic_compression.
- depletion_only_model(G_lin, P_s, P_p) = G_lin / (1 + 2 G_lin P_s / P_p) emitted as
compression_model_depletion_only next to compression_model_multitone_hb. Documented
as a trend baseline, not an acceptance oracle.

3. IO

File: src/twpa_solver/multitone/io.py (new)
Changes: write_compression_outputs producing exactly your §16 contract —
compression_points.csv (adding the complex-S21 columns *_s21_real/_imag/_mag_db/_phase_rad
for signal, pump and idler), compression_arrays.npz, compression_summary.json
(including basis, torus_grid, solver_config, resource_estimate,
stability_status="NOT_CHECKED"), and multitone_solution_*.npz at the requested
checkpoints. Solutions stored float32 savez_compressed and reloaded as complex128,
matching the pump-solution convention in CLAUDE.md.
--save-states {none,last,selected,all}; selected = zero-signal, P1dB bracket ends,
P1dB, highest converged power.
--resource-budget-gb, --diagnostic-port. Reuses load_circuit,
resolve_pump_basis, the pump solve, and default_loss_model /
--attenuation-db exactly as run_gain_map.py does — including
--pump-current-jc-scale 1.0 as the documented validated conversion. Accepts
--pump-solution-dir to skip the pump solve. Single operating point only; it does not
touch run_gain_map.py.

Success Criteria

Automated: pytest tests/test_multitone_observables.py tests/test_compression.py tests/test_run_compression_cli.py
- Small-signal parity: |G_NL(P_s -> 0) - G_Floquet| < 0.05 dB on build_jpa() and
build_jtwpa(). This is the gate for the whole plan.
- tone_s21 at tiny signal equals GainResult.gain_db to 1e-6 dB.
- Pump-off multitone reference agrees with the khat_off_0 linear reference to
1e-6 dB at tiny signal.
- port_waves cross-check: |b|^2 / |a|^2 at the output port is consistent with
|S21|^2 to 1%.
- refine_p1db on a synthetic monotone C(P) returns the analytic crossing to
< 0.01 dB; a synthetic nonmonotone curve sets nonmonotonic_compression=True and
reports the first crossing.
- depletion_only_model matches the closed form.
- CLI smoke: run_compression.py on the JPA fixture with --n-signal-power 5 --multitone-basis three_tone writes all four artifacts and exits 0.
Manual: On the production 2c circuit at a known-good (f_p, P_p, f_s), inspect the
compression curve and confirm G_NL starts at G_Floquet and decreases monotonically
into compression, and that D_p becomes measurably negative near P1dB.

---
Phase 7: Validation and basis convergence
1. Convergence study driver

File: scripts/multitone_convergence_study.py (new)
Changes: Sweep Q in {1,2,3} x pump-harmonic order x torus resolution
(n_p, n_delta) at one operating point; also three_tone vs lattice and odd-only vs
dense h. Writes a table plus a plot of P1dB vs each knob. Acceptance:
|Delta P1dB| < 0.2 dB between the top two settings of every knob. The odd-vs-dense
comparison is mandatory — even sectors are not assumed negligible.

2. Power balance

File: src/twpa_solver/multitone/observables.py
Changes: power_balance(X_full, basis, circuit) — sum incident vs outgoing power over
all ports and tones plus dissipated power in G (and Im C); Manley-Rowe photon-flux
check sum_v P_v / omega_v conservation. Reported per point in
compression_points.csv as power_balance_rel_err.

3. Tone-dependent loss

File: src/twpa_solver/multitone/problem.py
Changes: D_v = K - omega_v^2 C_v + i omega_v G_v with an optional
loss_model.evaluate(frequency_hz, tone_kind, input_power_dbm) hook evaluated once per
tone per signal-power point (never per Newton step, never state-dependent, so the JVP is
unchanged). Stage 1 (fixed circuit loss) is the default; Stage 2 (per-tone frequency
dependence, wired to twpa_solver.loss.InsertionLossModel) and Stage 3 (per-tone
input-power dependence) are flags. Stage 4 is explicitly out of scope.

4. Signal-frequency loop

File: scripts/run_compression.py
Changes: --signal-ghz-min/--signal-ghz-max/--n-signal-freq. Loop order is
pump point -> signal frequency -> signal power. Signal frequencies are independent given
the pump solution and may be parallelized (--signal-workers, process pool, per the
repo's memory that heavy runs need explicit worker sizing against free RAM); signal
powers within one frequency stay strictly sequential. Outputs
P1dB(f_s), P_s_out(P1dB, f_s), D_p(P1dB, f_s) and a summary plot.

Success Criteria

Automated: pytest tests/test_multitone_convergence.py tests/test_power_balance.py
- Lossless fixture: power_balance_rel_err < 1e-6; Manley-Rowe photon flux conserved to
1e-6.
- Lossy fixture: balance closes once dissipated power is included, to 1e-6.
- Q=2 vs Q=3 P1dB differ by < 0.2 dB on build_jtwpa().
- Stage 2 loss reduces to Stage 1 when the model is constant in frequency.
Manual:
- Run the full study on the production 2c device; record the converged basis in CLAUDE.md.
- Compare G_multitone(P_s) against G_depletion_only(P_s) and attribute the difference
using the spatial Theta(n) / dk_eff(n) profiles — the paper's core claim that
compression is depletion plus power-dependent phase mismatch.
- Reproduce the paper's Fig. 3 shape (P1dB vs f_s) and Fig. 4 shape
(position-dependent gain / power vs cell).

---
Phase 8 (scoped, not planned in detail): Stability

A converged root proves R(X)=0, not dynamic stability. Every artifact from Phase 6
onward carries stability_status="NOT_CHECKED". A later phase linearizes about the
finite-signal torus state and computes Floquet exponents, building on
src/twpa_solver/signal/stability.py (estimate_sigma_min, refine_complex_resonance)
and scripts/floquet_stability_sweep.py. Required before interpreting deep-saturation
branches, multistability or branch switching as physically accessible.

---
Testing Strategy

Project Maturity Level

Established Production — twpa_solver is pinned to JosephsonCircuits.jl by the
7-design parity suite and drives long production campaigns. New numerics must be proven
against an existing validated limit before being trusted.

Unit Tests

- basis.py: canonicalization, conjugate partners, (0,0) rejection, positive-frequency
invariant, omega_max truncation, three-tone vs lattice construction.
- grid.py: FFT round trip, real-output exactness, analytic cos^3 mixing, two-tone
2 omega_p - omega_s generation, aliasing guard.
- source.py: 0.5 * I factor, affine path endpoints, substep construction.
- problem.py: JVP vs finite difference, zero-signal pump reduction, spectral vs AFT JVP.
- schur.py: full/Schur parity of state, residual and port outputs.
- preconditioners.py: GMRES iteration bounds, sector vs exact same root, resource guard.
- seed.py: Floquet frequency mapping, seeded Newton iteration count.
- compression.py: P1dB refinement on synthetic curves, nonmonotonic reporting, recovery
ladder status strings.
- observables.py: S21 normalization parity, port-wave cross-check, power balance,
Manley-Rowe, phase unwrapping.
- Edge cases throughout: zero signal power, signal equal to pump frequency (Delta=0 must
raise), n_delta=1, single-tone basis, non-finite states.
- Coverage target: 80% on src/twpa_solver/multitone/.

Integration/Manual Tests

- End-to-end run_compression.py on build_jpa() (fast, in CI) and build_jtwpa()
(marked slow, --run-slow only), asserting the full artifact set and the small-signal
parity gate.
- Manual production runs per Phase 4 (preconditioner timing), Phase 6 (curve sanity) and
Phase 7 (convergence study, paper-figure reproduction).
- Regression guard on every phase: pytest tests/ including
tests/test_baseline_freeze.py, tests/test_pump_solvers_schur.py,
tests/test_run_gain_map_cli.py, tests/test_traversal.py — the multitone work must
never move an existing pump or gain number.

---
Rollback Plan

- Phases 1-3 and 5-7 add only new files under src/twpa_solver/multitone/,
scripts/run_compression.py and new tests. Rollback = delete the package and the script;
nothing else imports them.
- Three edits touch existing files and are individually revertable:
  a. pump/solver.py::tangent_predictor — one line; a no-op for all current callers,
asserted by the Phase 0 freeze test.
  b. pump/backends/fast_coupled.py — mode-key generalization plus a mode_keys property
and grid.phase_rows; refactor-only, guarded by the Phase 0 golden values and
tests/test_pump_solvers_schur.py.
  c. core/linear.py — additive port_waves only.
Each is a separate atomic commit so git revert of one does not disturb the others.
- scripts/compression_sweep.py deletion is recoverable from history; it is
non-functional today, so nothing depends on restoring it.
- Per-phase gate: if a phase's automated criteria fail, stop and fix before proceeding —
the small-signal parity gate in Phase 6 is the point of no return for the physics, and
Phases 1-5 are meaningless if it does not hold.