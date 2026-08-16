# Solver plan after the 2026-08-15 ansatz-validity measurement

## Goal

Make the harmonic-balance solver report gain that is verified against an
independent time-domain reference, refuse to report unstable branches, and stop
mistaking its own continuation failures for device boundaries — without adding a
new frequency ansatz.

## What the measurement established

Full evidence: `CLAUDE.md` (sections "HB ansatz validity measured against the
FDTD kernel", "Fixed continuation ladders fail where solutions exist",
"`--sidebands` defaults to 6", "Open: 4.8-8.4 dB HB-vs-FDTD gain gap") and
thesis chapters 5 and 8. Canonical data:
`outputs/chaos/ansatz_validity/ansatz_validity.csv`, produced by
`scripts/chaos/measure_ansatz_validity.py`.

| Finding | Evidence |
| --- | --- |
| The ansatz is exact where the device amplifies | on-lattice fraction 1.0000 -> 0.9233, four devices, 68 points |
| The collapse is **not** a period doubling | half-integer nodes add 4.8 / 4.5 percentage points; the residue is a continuum (top-20 bins hold 2.4%) |
| A torus exists in a ~1.1 dB window | single-generator fit 0.999 -> 0.627 over -29.3 -> -28.2 dBm |
| Fixed continuation ladders fail where solutions exist | 4/8/16/32 steps all stall; adaptive reaches lambda=1 at 8.42e-12, needing steps down to 0.0045 |
| Same-S multitone/Floquet parity is intact | multitone S=6 29.141 dB vs Floquet column 29.140947 dB |
| The gain gap is real | 4.8-8.4 dB at matched pump current, with utilisation agreeing to <1% |

## Sequencing decision (2026-08-15)

Phases 1 and 2 proceed now; Phase 0 runs in parallel rather than blocking them.
Phase 3.1 does not commit to a loss model until both are measured at Tier 1.

## What this plan explicitly does not do

- No period-2, period-N or from-scratch torus ansatz. The measurement is the
  direct test of that hypothesis and it fails. `build_half_pump_basis`,
  `pump/floquet.py`, `pump/periodic_branch.py` and `signal/period_doubled.py`
  stay dormant.
- No replacement of harmonic balance. It is exact in the useful regime.
- No promotion of the FDTD kernel to a production workhorse. It stays the
  referee, and it is the only instrument that can see the broadband state.
- The auxiliary-generator closure is deferred to Phase 4, not cancelled.

---

## Phase 0 — Close the gain gap (runs in parallel)

If small-signal gain is wrong by 5-8 dB, every downstream number is wrong. This
phase does not block Phases 1-2, but no gain number is publishable until it
lands.

### 0.1 Timestep convergence for `jc_jtwpa` — ANSWERED, not yet converged

**The gap was the timestep.** Measured at a 3e-9 A probe:

| P_p [dBm] | dt=0.01 | dt=0.005 | model S=12 | residual |
| ---: | ---: | ---: | ---: | ---: |
| -30.1 | 19.996 | 24.916 | 27.743 | 2.83 dB |
| -29.7 | 25.891 | 29.585 | 30.440 | 0.86 dB |

Halving the step moves the FDTD +4.9 / +3.7 dB toward the model. Utilisation
moves <0.5% over the same refinement, so the pump was never in question — only
the small-signal response.

Remaining work: the `dt_norm = 0.0025` point. **Attempted and not obtained** —
the run was stopped during its pump-off reference integration (serial, about an
hour at 4x cost) and produced no output. Two points cannot fit a convergence
order, so the residual 0.9-2.8 dB must not be reported as physical until a
third lands. Precedent: `rf_squid_2393_3wm` needed 4x finer than default.

Cost note for whoever reruns it: the pump-off reference is charged once per
(device, dt_norm, probe level, budget) and is cached on disk at
`<output>/pump_off_reference_cache.json`. At `dt_norm = 0.0025` that single
serial integration dominates a two-point study.

### 0.2 Non-degenerate linear-limit gate — now the priority item

`scripts/chaos/run_guarcello_jc_phase5.py::_measure_linear_limit` sets `Ic = 0`,
which removes the only inductive path on both `jc_*` devices; both sides return
exactly 0.0 and `relative_error` is `None`. Add a variant that retains a finite
linear inductance instead of zeroing `Ic`.

This is the check that would have caught 0.1 without spending a campaign, and it
is the reason those two devices carry no independent validation at all. Now that
0.1 has confirmed a real timestep error on `jc_jtwpa`, this gate is the highest
priority item in the plan: every `jc_*` FDTD gain result at `dt_norm = 0.01` is
suspect until it exists.

**Success:** a finite relative error is reported for `jc_jtwpa` and
`jc_fqjtwpa`, and it flags `dt_norm = 0.01` as insufficient on `jc_jtwpa`
without needing a gain campaign to reveal it.

### 0.3 L-stable integrator option

`scripts/run_overnight_7p9_dynamics.py:105` hardcodes `implicit_trapezoid`,
which is A-stable but not L-stable (`R(z) -> -1`), so it applies no numerical
damping and ramp-injected content never decays. That is the origin of the
2.2-2.4e-7 V signal-tone floor which makes probes at or below 3e-10 A measure
the floor rather than the signal. `h1_transient_branch_transfer.py:1557` already
offers `BDF` and `Radau`; expose the choice.

**Success:** the floor drops by at least a decade, so a 3e-10 A probe becomes
usable and the compression correction stops needing a quadrature assumption.

### 0.4 If the kernel is exonerated, bisect the model side

Vary `pump_nt`, `gamma_nt` and the sideband-ladder convention one at a time
between the Floquet and multitone paths at the matched point. The pump states
already agree to better than 1% utilisation at a pump current matched to seven
figures, so the fault is confined to the small-signal response.

---

## Phase 1 — Boundary diagnostics

No solver mathematics, high diagnostic value.

### 1.1 Make `--sidebands` explicit and recorded

`scripts/run_gain_map.py:4379` defaults `--sidebands` to 6 and
`scripts/run_hb_column_until_failure.py` never overrides it, so every
`hb_up_to_failure.csv` gain is an unlabelled S=6 truncation, about 1.4 dB below
converged on `jc_jtwpa`. Pass it explicitly from the column driver and record it
on every row.

**Success:** the value appears in the emitted CSV and JSON; a re-run at S=10
reproduces 30.542 dB at `I_p = 3.8806468570637416e-06 A`.

### 1.2 Fix the FQJTWPA utilisation field

`pump_branch_current_max_over_ic` reads 0.086-0.151 across the whole converged
FQJTWPA column while the time domain reports 0.62-0.89 at the same points, and
the sequence is not monotone in pump power. The JTWPA column agrees with the
time domain to 11 significant figures, so this is device-specific — most likely
the branch set or the `Ic` normalisation for its flux-pumped junctions.

**Success:** FQJTWPA utilisation is monotone in pump power and agrees with the
time-domain `r_j` to the same ~1% the JTWPA column achieves.

### 1.3 Emit the boundary predictor

`I/Ic -> 1` together with `cos phi` turning negative marks the collapse on all
three real devices. Surface it as an explicit status field rather than a column
a reader has to know to look at.

**Success:** a run approaching the transition emits the warning at the last
converged point before collapse on `jc_jtwpa`, `jc_fqjtwpa` and `ipm_2c_fixed`.

---

## Phase 2 — Wire the continuation ladder into every driver

The clearest defect, with a reproducible failing case.

### 2.1 Adaptive pump continuation in the compression driver

`scripts/run_compression.py:877` calls
`solve_continuation(pump_problem, continuation_steps=4)` and raises at `:880`
when the last step stalls. Switch to `solve_adaptive_continuation` with the
fixed ladder as fallback, matching `run_gain_map.py`.

Refining the *fixed* ladder is not an alternative: 32 steps does no better than
8, because the difficulty is concentrated in a short interval that uniform
stepping cannot resolve.

### 2.2 Minimum step and discarded seeds

`scripts/run_compression.py:1118` and
`src/twpa_solver/multitone/compression.py:78` both use `min_step=0.01`, more
than twice the smallest step the `4.603781e-06 A` point needs, so they are
structurally unable to cross it regardless of time budget. Both also pass
`x_init=None`, discarding the promoted pump seed and restarting from zero.

Derive `min_step` from a floor rather than pinning it, and pass the seed.

### 2.3 A signal path that holds the pump fixed

`src/twpa_solver/multitone/source.py:57`, `AffineSourcePath.signal_turn_on`,
starts from zero and ramps pump and signal together, so every signal-power point
re-runs the pump ramp — it fails even at a 1e-11 A probe. Add a path that holds
the pump at its converged value and ramps only the signal, warm-started from the
pump-on state.

This is what makes a compression curve reachable at all. With the pump supplied
through `--pump-solution-dir`, the S=10 multitone state at -28.2 dBm converges
in **one Newton iteration** (`coeff_rel` 6.23e-12); the current code throws that
state away and re-derives it per power point.

**Phase 2 success:** `run_compression` reaches lambda=1 at
`I_p = 4.603781e-06 A`; a signal-power sweep at -29.7 dBm reaches 3e-8 A; every
existing compression result that already converged is unchanged.

---

## Phase 3 — Stability gate on the converged branch

The real defect behind "chaos slowly ensues" is not the ansatz. It is that
harmonic balance converges onto branches with no test of whether they are
stable, and then reports them.

### 3.1 Measure both loss models at Tier 1 before committing

Adding dielectric `tan_delta` makes Hill multipliers decidable on
`ipm_2c_fixed`, which currently resolves to `has_loss = False`. But it also
switches `default_loss_model_for` to `conductance_abs_omega`, which is
non-analytic in omega and therefore removes Tier-2 complex-omega refinement.

Quantify `conductance_abs_omega` against `current_complex_c` at Tier 1, where
both are legal, and only then choose. Do not assume the gap is small.

**Success:** a measured Tier-1 comparison exists, and the loss-model choice is
made against it rather than by assumption.

### 3.2 Promote the stability check to a gate

`assess_multitone_stability` already exists
(`src/twpa_solver/multitone/stability.py`) and is imported at
`scripts/run_compression.py:66`, behind `--check-stability`, default off. Runs
without it record `stability_status = "NOT_CHECKED"`. Make the status a stamped
field on every published gain and P1dB.

### 3.3 Wire branch tracking into the Hill sweep

`src/twpa_solver/stability/tracking.py` exists but serves only the monodromy
scan. Without it the Hill secant intermittently falls onto a power-independent
neutral root, which is a branch-tracking failure rather than a physical result.

---

## Phase 4 — Auxiliary-generator closure (deferred)

Two extra real unknowns `(A_a, omega_a)` and two extra real equations on the
**existing** `ToneIndex(h,q)` torus grid — not a new basis. Measured payoff:
about 1.1 dB of extra pump range, since the window in which off-lattice content
is discrete is that narrow.

Worth building for the Neimark-Sacker detection it provides, after Phases 0-3,
not instead of them.

---

## Testing strategy

Project maturity: active development.

Existing gates that constrain these changes:

| Gate | What it pins |
| --- | --- |
| `tests/test_multitone_physics.py:219` | same-S multitone/Floquet parity to 1e-4 dB |
| `tests/test_advanced_continuation.py` | continuation behaviour |
| `tests/test_run_compression_cli.py` | driver flags and defaults |
| `tests/test_adaptive_continuation_fallback.py` | the fallback resumes rather than restarts |

New gates required:

- A continuation test at `I_p = 4.603781e-06 A` on `jc_jtwpa` that **fails** on
  the fixed four-step ladder and **passes** on the adaptive one.
- The ansatz-validity CSV as a regression fixture, so the lattice measurement
  cannot silently change.
- A signal-ramp test asserting that the pump-fixed path converges from a
  supplied pump state in far fewer Newton iterations than the zero-start path.

Per this repository's own rule, every new gate must be shown failing before it
is accepted.

Full suite, with a temporary directory outside the repository:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_full_slow --run-slow
```

## Rollback

Phases 1 and 2 are localised edits behind existing flags and revert
individually. Phase 3.1 changes physics inputs and stays on a branch until its
Tier-1 comparison is measured. Phase 0 changes only diagnostic scripts under
`scripts/chaos/` and writes to ignored `outputs/`.
