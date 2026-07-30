# Implementation prompt: design-independent validation of the multitone saturation solver

## Why this exists

Two external references were retired on 2026-07-29 and **must not be used**:

- **JosephsonCircuits.jl.** It is another simulator with no reference of its own,
  and `jc_jtwpa`/`jc_fqjtwpa` are JC's own documentation designs, so that half of
  the 7-design parity suite is circular. JC parity is still a valid *regression*
  check that our numerics have not drifted. It is **not** evidence of physical
  correctness and must never be reported as such.
- **The Themis measurement.** Its cube (`105C5_*GHz.npy`) stores `Response` with
  shape `(n_power, n_sig)` where the power axis is `d["PumpPower"]`. It sweeps
  *pump* power by signal frequency, with the signal small throughout. There is no
  signal-power axis anywhere in it, so it cannot validate compression or P1dB.
  It remains valid for the gain map and the pump solve.

That leaves the saturation solver with **zero external reference**. Everything
currently passing is either small-signal (validated only against our own Floquet
path) or internal numerical consistency — residual norms, resolution
independence, preconditioner invariance — all of which would pass equally well
for a solver converging to the *wrong equations*.

The work below fixes that by testing predictions that follow from perturbation
theory alone and hold for **any** circuit. No design-specific reference value is
needed anywhere in Part 1.

## Hard constraints

- **Memory: this machine has ~7 GB total and typically 4.5 GB free.** A run OOMed
  today. Never run two solver jobs concurrently. Check free memory before
  launching anything large. Prefer the `jpa` fixture for everything in Part 1 —
  these tests are design-independent, so the smallest device is the correct
  choice and makes the whole suite fast. Only Part 2's basis convergence needs a
  production device, and it needs the banded backend.
- **File editing: never use heredocs or shell redirection to create or edit
  files.** Use the Edit/Write tools directly. This is an inviolable rule.
- **Git: use the `git` CLI only. Do not commit unless explicitly asked.**
- Run tests with a basetemp outside the repo:
  `python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\tw_val`
- **The suite must end at exactly 4 pre-existing failures**
  (`test_column_matrices_tracer` x2, `test_loss_model` x2 — the latter from a
  missing `docs/loss_A10.csv`). Anything else is a regression you caused.
  Current baseline: **4 failed, 247 passed, 3 skipped.**
- **Every new gate must be mutation-verified**: break the thing deliberately,
  show the test failing, restore, show it passing. A gate never shown failing is
  not evidence. This is the standing bar in this repo and it exists because
  today's audit found three separate gates that could not fail — a stability
  check running on `Ic=0` at a zero state, a basis study never run on the
  production basis, and a P1dB error measured against a grid no publication used.
- **Report measured numbers with their conditions, never "passes".** If something
  does not work, say so with the number. A negative result reported honestly is
  worth more than a green check.

## Existing machinery — find it before writing anything new

- Driver: `scripts/run_compression.py` (sweeps signal power at a fixed pump point)
- Multitone: `src/twpa_solver/multitone/` — `problem.py` (`FullMultiToneProblem`),
  `basis.py` (`MultiToneBasis`, `ToneIndex`, `build_sideband_matched_basis`,
  `build_lattice_basis`, `build_three_tone_basis`), `source.py`
  (`MultiToneDrive`, `AffineSourcePath`), `observables.py` (`tone_s21`,
  `power_balance`, `extract_port_waves`), `compression.py`
  (`solve_signal_power_point`), `seed.py` (`promote_pump_solution`),
  `schur.py`, `stability.py`, `resources.py`
- Linear signal path: `src/twpa_solver/signal/floquet.py` (`solve_gain_one`,
  `solve_gain_one_schur`, `sideband_list`, `assemble_conversion_matrix`)
- Pump: `src/twpa_solver/pump/` (`FullPumpProblem`,
  `HarmonicNewtonKrylovSolver`, `NewtonKrylovSettings`, `basis.py`)

**Before Panel A, verify the reconstruction convention.** The multitone basis
declares `real_reconstruction_factor = 2` (`basis.py`, `to_metadata`), i.e.
`x(t) = 2 Re sum(X_v exp(i theta_v))`, while `signal/floquet.py` is single-sided.
Check whether `observables.py` applies that factor when extracting port
voltages — `grep -rn "REAL_RECONSTRUCTION_FACTOR" src/`. If it does not, the
multitone flux is exactly half the Floquet flux and Panel A cannot pass. Note
that this will **not** show up as a clean -6.02 dB, because
`core/linear.py::port_s_from_unit_current_response` is affine for a 1-port
(`s = 2V/Z0 - 1`), so `s(V/2) != s(V)/2`. Establish the convention by direct
measurement of the ratio before interpreting any gain discrepancy.

---

# Part 1 — Design-independent validation suite

Run in this acceptance order. Each step gates the next: do not interpret a
failure at step N before step N-1 is green.

Use one converged pump operating point comfortably away from any pump fold, one
signal frequency whose idler sits inside the modeled bandwidth, and a basis at
least Q=3. State the chosen point explicitly in the report.

## 1. JVP finite-difference convergence

Validate the analytic Jacobian-vector product against central differences.
Locate how the solver forms the JVP (`NewtonKrylovSettings(jvp_mode="aft")`;
find the actual method on the problem object rather than assuming a name).

For a random perturbation `V` at a converged `X`, compute

    E(eta) = ||J_analytic V - (R(X + eta V) - R(X - eta V)) / (2 eta)|| / ||J_analytic V||

and plot `E(eta)` versus `eta` on log-log axes over ~10 decades of `eta`.

**Expected:** a U-shaped curve — slope ~2 in the truncation-dominated region,
then rising as floating-point cancellation takes over. **Report the measured
slope of the descending branch and the minimum of `E`.** A slope of 1 means the
JVP is not the derivative of the residual actually being solved. A floor that
never gets below ~1e-6 means the two disagree at a level that matters.

This is first because everything downstream assumes the Newton direction is
correct.

## 2. Zero-signal pump parity

At zero signal source, the multitone solution must reduce to the 1D pump solve:

- `X[h, q=0]` equals the `FullPumpProblem` solution promoted through
  `promote_pump_solution`, to <1e-10 relative
- `X[h, q!=0]` is zero to <1e-12

Previously measured at 5.87e-15 and 5.25e-33 — this should pass immediately.
**Lock it in a test**; it is currently unguarded.

## 3. Multitone-to-Floquet small-signal limit — Panel A

Sweep signal power from the infinitesimal regime up into compression. At the
same pump state compute the existing linear Floquet gain.

Plot `G_multitone(P_s)` with `G_Floquet` as a horizontal line, and the more
sensitive difference `dG(P_s) = G_multitone(P_s) - G_Floquet`.

**Expected:**

    lim_{P_s -> 0} G_multitone(P_s) = G_Floquet

**Acceptance: |dG| < 0.05 dB in the asymptotic region, with identical
normalization on both sides.** Report the measured floor of `|dG|` and the
signal power at which it is reached.

Then plot small compression versus `P_s` on log-log axes. Since the leading
nonlinear correction to the signal amplitude is third order in `epsilon` and
`P_s ~ epsilon^2`, the gain correction in **linear** units is `O(P_s)`:

**Expected slope ~1** before numerical noise dominates. Report the fitted slope
and the fit window.

This single comparison tests source normalization, signal-mode indexing, idler
conjugation, port-wave extraction, 2D FFT normalization, pump-state promotion,
and agreement between the old and new formulations at once.

**Important — pick an operating point with real gain.** At the default jpa
fixture point the device shows no gain (`gain_vs_off_db` ~ -0.0041 dB), so a
parity gate there only ever compares 0 dB to 0 dB and is worthless. Find a pump
current and signal detuning where the linear solve reports
`gain_vs_off_db > 3 dB` and run the gate **there** as well as at the weak point.
If the chosen fixture cannot be driven into gain, say so explicitly with the
numbers rather than shipping the degenerate gate.

## 4. Sector scaling slopes — Panel B

Group the converged solution by detuning order `q` and sum over pump harmonic
`h`:

    P_q = sum_h P_{h,q}

Plot each generated sector power against input signal power on log-log axes.
Perturbation theory gives `X_{|q|} = O(epsilon^{|q|})` and `P_s ~ epsilon^2`,
hence

    P_{|q|} ~ P_s^{|q|}

**Expected slopes: 1, 2, 3 for |q| = 1, 2, 3.** Report the fitted slope per
sector with its fit window and residual.

Fit only in the clean asymptotic region — above the numerical noise floor and
below the onset of compression. Identify that window from the data and state it;
do not fit the whole sweep.

**Interpreting deviations:** an accidental symmetry can zero a leading
coefficient, so a sector may show a *higher* slope than predicted. That is
acceptable and should be reported as such. A sector appearing at an
inexplicably *lower* perturbative order is a defect — it means a tone is being
populated by a coupling path that should not exist at that order.

This validates the `(h, q)` tone-index mapping, convolution offsets, conjugate
frequency handling, the multi-diagonal coupling assembly, and the scatter/gather
of the 2D spectrum.

## 5. Pump depletion vanishes at zero signal — Panel C

Plot `D_p(P_s) = 10 log10(P_p_out(P_s) / P_p_out(0))`.

**Expected:** `D_p -> 0` as `P_s -> 0`. The pump correction is second order in
signal amplitude because depletion involves products like `A_s A_i A_p*`, so a
linear-unit depletion metric should scale as `O(P_s)`.

**Report the fitted slope.** This verifies that the pump responds to finite
signal power, that the signal does not spuriously modify the pump at zero
amplitude, and that the nonlinear coupling direction is correct.

## 6. Power balance and Manley-Rowe — Panel D

For every converged point, plot the normalized HB residual
`r_rel = ||R(X)|| / ||S||` (must stay below the Newton tolerance) **and** the
power-balance error

    eps_P = (P_in - P_out - P_diss) / P_in

Residual convergence alone is not sufficient — a solver converging to the wrong
equations converges tightly.

Build a **deliberately lossless copy** of the design (zero the conductance
matrix `G`) so that `P_in = P_out` exactly, and verify `eps_P -> 0` there. On
the lossless fixture the plan's target is `< 1e-6`.

**This step also carries the highest-priority defect in the repo — see Part 2.1.
Fix Manley-Rowe before reporting this panel.**

## 7. Preconditioner invariance

Run the same physical point across preconditioner variants. The converged root
**cannot** depend on the preconditioner; only iteration count and runtime may.

First check what exists: `MULTITONE_PRECONDITIONERS` in
`src/twpa_solver/multitone/preconditioners.py`, plus `--factor-backend
{pardiso,banded}` and the `mean_tangent` / `spectral_coupled` / `real_coupled` /
`real_coupled_fast` settings. A **bandwidth-truncated** family (block diagonal /
tri / penta / heptadiagonal, L = 0,1,2,3) may not exist. If it does not, either
implement a bandwidth-L truncation of the coupled Jacobian or run the invariance
test across the variants that do exist — both test the same invariant, so say
which you did.

Plot preconditioner bandwidth (or variant) on x, with gain difference and
pump-depletion difference from a reference on the left axis and GMRES iterations
plus factorization runtime on the right.

**Expected: physics curves flat at numerical precision; iteration count and
runtime vary.** Report the gain spread in dB — it should be at the 1e-10 level.
Prior measurements to compare against: banded vs pardiso agreed to 4.4e-10 dB,
and `precond_reuse` 1/2/3 gave bit-identical gain.

## 8. Signal-phase covariance

Hold signal power fixed and sweep the input phase, `S_s -> S_s exp(i phi)`.
Note `MultiToneDrive` currently takes a real `current_a`; a complex amplitude
may need to be threaded through.

**Expected, at fixed pump phase:**

- `P_s(phi)`, `P_i(phi)`, `P_p(phi)` all constant
- `arg X_s -> arg X_s + phi`
- `arg X_i -> arg X_i - phi`, since four-wave mixing gives `X_i ~ X_p^2 X_s*`

Report the measured power variation over the phase sweep and the fitted slope of
`arg X_s` and `arg X_i` versus `phi`. **The idler slope of -1 is the sharp
test** — it is an excellent detector of a missing conjugation or a wrong `(k+q)`
coupling term. A slope of +1 there means a conjugate is dropped.

---

# Part 2 — Two blocking defects

## 2.1 Manley-Rowe is broken (BLOCKING)

`power_balance` in `src/twpa_solver/multitone/observables.py` emits
`manley_rowe_rel_err`. Measured on production sweeps today:

| device | max_manley_rowe_rel_err | where it peaks |
| --- | ---: | --- |
| jtwpa | 0.533 | **smallest** signal power |
| 2c | 0.500 | smallest signal power |

Two things are wrong. It is **largest in the linear limit**, where photon-flux
conservation should be near-exact, and falls to 0.034 in deep saturation — the
opposite of the expected trend. And it reads ~0.5 on two very different devices,
which points at a fixed factor or a mis-scoped tone sum rather than physics.

This is now **the single most important check in the codebase**. With JC and
Themis both retired, lossless photon-flux conservation is the only rigorous,
reference-free statement available about a *saturated* solution. Everything else
either tests the linear limit or tests the numerics against themselves.

Find the factor. Candidates worth checking first: whether the sum runs over the
same tone set as the power balance; whether positive-phasor tones are being
double-counted or half-counted against the `real_reconstruction_factor = 2`
convention; whether the photon flux uses `P/omega` with each tone's own `omega`
(signal and idler sit at different frequencies, and the ladder-basis reweighting
`sqrt(freq[n]/freq[m])` is the same trap documented for `calc_qe` in CLAUDE.md).

**Acceptance:** on the lossless fixture, `manley_rowe_rel_err < 1e-6` at small
signal, rising smoothly and monotonically into saturation. Mutation-verify.

## 2.2 `compression_model_depletion_only` is unconverted

The column emits 567.68 at small signal on jtwpa, which is exactly
`10^(27.541/10)` — the model's **linear power gain**, not a compression in dB.
`depletion_only_model` in `multitone/compression_curve.py` returns a linear
gain; the driver writes it straight into a column named as a compression.

`experiments/exp22_spatial_attribution.py` depends on this for its baseline, so
the exp22 attribution conclusion cannot be recomputed until it is fixed. Convert
correctly, and add a gate asserting the column equals the plan's model called
directly on the same inputs to 1e-12.

## 2.3 Production-basis self-convergence (after Part 1)

The JC gate that selected the production bases is void, and this is what has to
replace it. Today's Phase 5 study measured the right *kind* of thing on the
wrong basis family — `build_lattice_basis` at 3.76-7.89 dB gain, where
production uses `build_sideband_matched_basis` at S=10 and 27.54 dB.

The unresolved problem: **jtwpa gain is non-monotone in S** — 30.7152, 24.2021,
26.5563, 27.5410 dB at S = 2, 4, 6, 10. Without a reference you cannot select
S=10 from that sequence by agreement, and non-monotonicity means it may still be
climbing.

Measure `|dP1dB|` between S=10 and S=12 (and S=14 if memory allows) on the
production `sideband_matched` basis at the real operating point, using refined
P1dB. **Use the banded factor backend** — it is what let `torus_scale=2` fit
today where pardiso OOMed. Expect this to be the most expensive item here;
budget it explicitly and use `--per-setting-budget-s`.

If S=10 is not converged, say so and state the P1dB uncertainty it implies for
every published exp20/21 number.

---

# Deliverable

1. **One four-panel validation figure**: (a) nonlinear and Floquet gain versus
   signal power; (b) q-sector power versus signal power, log-log; (c) pump
   depletion and gain compression versus signal power; (d) HB residual and
   normalized power-balance error. Plus the separate JVP convergence plot and
   the preconditioner-invariance plot.

2. **A results note** in `docs/development/` recording every measured number with
   its measurement conditions, and stating plainly which of the eight checks
   pass, which fail, and which could not be evaluated.

3. **Tests** for each check that can be made cheap enough to live in the suite,
   each mutation-verified. Checks too expensive for the suite stay as scripts
   with their numbers recorded in the note.

4. **CLAUDE.md updated** with the outcome.

The success criterion for the whole exercise:

    P_s -> 0:   G_MT -> G_Floquet
                P_|q| ~ P_s^|q|
                D_p -> 0
                eps_P -> 0
                r_rel < tolerance

If all five hold on an arbitrary design, that is strong evidence the multitone
physics, indexing, normalization, and nonlinear solver are correct — without
needing any external reference. That is the point of doing it this way, and it
is what we can establish while waiting on measurements from the rest of the team.
