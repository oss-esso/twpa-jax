# Solver plan: extend harmonic balance into the 2-torus

Status: execution plan. Rewritten 2026-08-17 after a first draft was judged to
have drifted off the actual goal.

## The goal, stated plainly

The FDTD campaign has settled the route on all three devices: as pump power
rises the state goes **period-1 -> 2-torus -> chaos**, with no period doubling
anywhere. Harmonic balance currently fails at the first transition.

**This plan makes the HB solver continue through that transition and report
where it genuinely stops.** Everything else in this document is subordinate to
that sentence. If a task does not move the solver further up in pump power, it
is not on the critical path.

Chaos scope, decided 2026-08-17: past the torus window the solver delivers the
**boundary plus the unstable-orbit skeleton**. It does not represent the
chaotic state. Trajectories in chaos stay with the FDTD kernel.

## Read this before starting

Two rules govern every change here.

**1. Extend, do not rewrite.** The production HB is fast because it was
optimized: `FastCoupledPreconditioner` caches the scatter map and the symbolic
factorization and re-runs only the numeric factor per Newton step; the pardiso
and banded factor backends are selected against measured footprints; GMRES
converges in about three iterations against the near-exact preconditioner. Any
new code path that calls a fresh `scipy.sparse.linalg.spsolve` or `splu` per
Newton step has thrown all of that away and is wrong regardless of whether it
converges. Phase 1 exists because exactly this happened.

**2. Run first, instrument second.** An earlier draft of this plan gated the
torus solve behind a Floquet stability instrument and two FDTD re-runs. That
was backwards: it put the fast tool behind the slow one. A torus seed needs two
numbers, the onset power and the generator frequency, and the campaign has
already measured both on all three devices. Stability instrumentation is now
**conditional** -- built only if the solver stops early and the reason is not
obvious.

## What already exists

| piece | location | state |
| --- | --- | --- |
| two-dimensional tone lattice `h*omega_p + q*delta`, exact torus grid | `multitone/basis.py:15,80`, `grid.py:12` | production |
| autonomous torus basis constructor | `multitone/basis.py`, `build_autonomous_torus_basis` | written, untested on a device |
| residual, JVP, spectral tangent, real-coupled Jacobian | `multitone/problem.py:33` | production |
| cached-symbolic fast preconditioner, pardiso and banded backends | `pump/backends/fast_coupled.py:138` (`refactor`, `solve`) | production |
| Govaerts-Pryce bordered solve taking a `linsolve` callable | `pump/solver.py:55` `bordered_solve_refined` | production |
| autonomous two-frequency HB problem: unknown `omega_a`, phase anchor, Newton | `multitone/torus.py` `TorusProblem` | written, **never run on a device**, uses the wrong linear solver |
| Hill root refinement and multiplier classification | `signal/stability.py:220,337` | production |
| multiplier branch matching | `stability/tracking.py:9` | built, wired only to the monodromy scan |
| tracked Floquet branch, NS coefficient, transition report | `signal/branch_tracking.py`, `pump/neimark_sacker.py`, `scripts/report_transition_boundary.py` | written, unvalidated, **conditional** |

The gap is narrow and specific: **an autonomous torus solver exists and has
never been run on a real circuit.**

## Measured inputs the solver needs

From the FDTD campaign. These are inputs, not things to re-derive.

| device | onset (period-1 boundary) | `f_a/f_p` | end of torus window |
| --- | ---: | ---: | ---: |
| `ipm_2c_fixed` | 0.5825 | 0.0917 | 0.6075 |
| `jc_jtwpa` | -29.4625 dBm | 0.1217 | -29.022 dBm |
| `guarcello` | -54.025 dBm | 0.3555 | -52.500 dBm |

Inside the window a single extra generator explains 0.986 / 0.999 / 1.000 of
the off-comb power. That is what makes a two-generator ansatz the right one and
is the number Phase 6 validates against.

## Problem size

Measured 2026-08-17 from the real constructors on `ipm_2c_fixed` (6136 nodes,
pump basis `[1,3,...,19]`).

| solve | tones | real unknowns | cost `~ tones^2` |
| --- | ---: | ---: | ---: |
| pump-only HB (gain map) | 10 | 122,720 | 1.0x |
| multitone compression S=6 | 23 | 282,256 | 5.3x |
| **multitone compression S=10 (production today)** | **31** | **380,432** | **9.6x** |
| **TORUS Q=1** | **31** | **380,432** | **9.6x** |
| TORUS Q=2 | 51 | 625,872 | 26.0x |
| TORUS Q=3 | 71 | 871,312 | 50.4x |

**The torus at `Q = 1` is exactly the size of the S=10 compression basis that
already runs in production.** It is not a new regime. The lattice is every pump
harmonic at `q = -1, 0, +1` plus the bare generator `(0, 1)`; the same tone
count as S=10 but a different set, since compression walks `q` outward at low
`h` while the torus puts `+-1` on every `h`.

**The bordering costs one real unknown and one real equation**, one part in
380,000. All cost is the lattice; none of it is the border. That is why
Phase 1 is about the linear solver and not about the bordering.

### Schur reduction is the lever

`build_partition` retains only nodes the nonlinearity touches, plus ports.
Measured on `ipm_2c_fixed`: `Bphi` is `6136 x 2508`, nodes touched by `Bphi`
number 2514, and with the four ports the retained set is **2518 of 6136, a
2.44x reduction**. Schur-reduced 2c is therefore only 1.23x `jc_jtwpa`'s full
2048 nodes, not 3.0x.

Scaling the measured jtwpa S=10 banded footprint (1.84 GB; pardiso 2.51 GB)
linearly in nodes and quadratically in tones:

| configuration | nodes | tones | banded estimate | fits ~7 GB |
| --- | ---: | ---: | ---: | --- |
| `jc_jtwpa` full, Q=1 | 2048 | 31 | 1.8 GB | yes |
| `jc_jtwpa` full, Q=2 | 2048 | 51 | 5.0 GB | yes |
| `ipm_2c_fixed` full, Q=1 | 6136 | 31 | 5.5 GB | yes, tight |
| `ipm_2c_fixed` full, Q=2 | 6136 | 51 | 14.9 GB | **no** |
| **`ipm_2c_fixed` Schur, Q=1** | 2518 | 31 | **2.3 GB** | yes |
| **`ipm_2c_fixed` Schur, Q=2** | 2518 | 51 | **6.1 GB** | yes, tight |
| `ipm_2c_fixed` Schur, Q=3 | 2518 | 71 | 11.9 GB | no |

### The banded backend does not apply to 2c. Measured, do not retry it.

An earlier draft of this plan asserted "`--factor-backend banded` is required
on 2c". **That was wrong**, derived by scaling jtwpa's footprint ratio without
checking the structural precondition banded depends on. A 2c torus run failed
on it: the backend measured a bandwidth of 283,836 over dimension 380,432 and
correctly refused, reporting a 2413.6 GB requirement.

Root cause, measured on the node coupling graphs of `designs/ipm_2c_fixed`:

| coupling graph | node bandwidth |
| --- | ---: |
| `Bphi Bphi^T` (nonlinear branches) | **1** |
| `C + G + K` (linear network) | **4578**, node 0 to node 4578 |

`4578 * (2 * 31 tones) = 283,836`, reproducing the reported figure exactly.
The nonlinear chain is a perfect 1-D band; the **linear** network destroys it.
4,552 of 22,956 linear nonzeros reach further than 100 nodes, and the ports sit
at 0, 4576, 4577 and 6135 -- 2c is a two-rail coupled device, not a single
chain. `jc_jtwpa` is a single chain, which is why banded works there and why
every banded measurement in this repository is a jtwpa measurement.

**Banded is a single-chain optimization. Do not run it on 2c, full-node or
Schur-reduced.** The long-range coupling lives precisely in the linear part
Schur eliminates, so the Schur complement is expected to inherit it as fill;
Schur buys dimension, not bandwidth.

### Schur is the production path on 2c, not an optimization

`scripts/run_hybrid_column.py:691-697` pins `inproc_pump_backend =
"schur_cpu_mt"` for the 2c column with the comment that **"the full-node
backend is not numerically equivalent near the 2c high-power obstruction and
can fail even at a known-valid anchor."** The torus target sits in exactly that
regime. A full-node 2c torus failure is therefore ambiguous: it could be the
ansatz, or it could be the documented full-node weakness.

Consequence: **Schur support is a prerequisite for the 2c production-basis run,
not a later phase.** It also carries the memory: Q=1 Schur is about 2.3 GB
against about 7.5 GB full-node pardiso, and it is the only route to Q=2 on 2c.

### Reduce the pump basis before reducing anything else

`K` is the cheapest lever and the only one that separates a physics failure
from an engineering failure. `[1,3,5,7,9]` (K=5) gives `5*3 + 1 = 16` tones,
196,352 unknowns, pardiso about 2.0 GB -- comfortable full-node. It is not the
production basis, so it cannot produce a final number, but it answers whether a
torus solution exists at all, in minutes and with no code. **Make this the
standing first move on any new device.**

---

## Phase 1: put the torus solver on the production linear algebra

**This is the only thing standing between us and a first result. Do it first.**

### The defect

`multitone/torus.py::jacobian` assembles a bordered sparse matrix with
`sp.hstack` / `sp.vstack`, and `solve_newton` calls `spla.spsolve` on it. Three
consequences:

- a fresh SuperLU factorization every Newton step, with no symbolic reuse;
- `FastCoupledPreconditioner` bypassed entirely, so the cached scatter map and
  cached symbolic factor are unused;
- the bordering breaks the node-major band, so the `banded` backend -- the one
  that makes large cases fit in memory -- cannot apply at all.

### The fix

Do not assemble a bordered matrix. Use the bordering *algorithm* against the
unbordered Jacobian, which is what `pump/solver.py:55` already implements.

For the bordered system

```text
[ J   b ] [dx]   [-r]
[ c^T d ] [dw] = [-a]
```

solve `J y1 = -r` and `J y2 = -b` through the **existing** preconditioned
solver, then combine:

```text
dw = (-a - c^T y1) / (d - c^T y2)
dx = y1 - dw * y2
```

Two solves sharing one numeric factorization: production cost per Newton step
plus two triangular solves.

### Changes required

1. `multitone/torus.py` -- delete `jacobian`'s bordered assembly and
   `solve_newton`'s `spsolve`. Per Newton step: build or `refactor` a
   `FastCoupledPreconditioner` on `problem.spectral_tangent_state(tangent)`,
   then call `bordered_solve_refined` with `linsolve = precond.solve`. Keep
   `b = dR/domega_a` (the existing central difference, which is fine and costs
   two residual evaluations) and `c^T` = the anchor row, `d = 0`.
2. `multitone/torus.py` -- expose `factor_backend` and `precond_reuse` so the
   torus solve takes the same knobs as the rest of the solver. Default
   `precond_reuse=1`, matching the measured production finding that reuse is a
   net loss here. **2c must run `pardiso`**; `banded` is excluded on 2c by
   measurement (see the sizing section).
3. Keep the line search. It is the one part of `solve_newton` worth keeping.

### Side effect: this is also what unblocks Schur

`TorusProblem.jacobian` currently calls `problem.real_coupled_matrix(...)`,
which exists **only** on `FullMultiToneProblem`. `SchurMultiToneProblem`
(`multitone/schur.py:22`) offers `assemble_real_coupled_fast` and
`assemble_real_coupled_preconditioner` instead, and exposes every other method
the torus problem needs (`residual_coeffs`, `tangent_state`,
`spectral_tangent_state`, `zeros`, `n`). So routing Phase 1 through
`FastCoupledPreconditioner` removes the one Full-only call and makes the torus
problem structurally Schur-compatible. Do not add a Schur code path in this
phase; just do not reintroduce a Full-only dependency.

### Success criteria

**Automated**: `pytest tests/test_torus_hb.py --basetemp D:\tmp\torus` --
must include a test that the bordered step from `bordered_solve_refined`
matches a dense reference solve on a small circuit to 1e-10. Without that,
this phase has no gate.

**Manual**: on the 2c circuit at one power, wall time per Newton step is within
2x of a same-size `FullMultiToneProblem` step. If it is 10x, the fast path is
still not being used.

---

## Phase 1B: K=5 feasibility probe on 2c

**Cheapest decisive move. No code. Run it before Phase 1C, in parallel with
writing it.**

Pump modes `[1,3,5,7,9]`, `Q = 1`, 16 tones, 196,352 unknowns, pardiso about
2.0 GB, full-node. Seed from the converged pump state at `I/I_bound = 0.6050`
(`.hybrid_outputs/period1_recovery_7p9_2c_v1/point_-23.800000/pump`), with
`omega_a = 0.0917 * omega_p`.

This is **not** the production basis and cannot produce a reportable `omega_a`.
Its only job is to separate a physics failure from an engineering failure: if a
torus solution does not exist here, no amount of memory work rescues the
production-basis run. Report converged / not, `omega_a`, and the `q = +-1` norm
fraction, explicitly labelled as a K=5 probe.

---

## Phase 1C: Schur support in `TorusProblem`

**Prerequisite for the 2c production-basis run.** Specification is under
"Phase 5: (moved)" below. `pardiso`, never `banded`.

---

## Phase 2: first convergence on 2c, production basis

`ipm_2c_fixed` is the priority device: it is ours. Run through **Schur**
(`schur_cpu_mt`, `real_coupled_fast`, `schur_cache_size = 1`, `float64`,
`pardiso`), matching the production engine settings table above.

### Changes required

`scripts/run_torus_branch.py` -- driver taking `--device`, `--circuit-dir`,
`--control`, `--omega-a-ratio`, `--q-max`, `--pump-solution-dir`.

Seed construction, in order of preference:

1. converged pump state at the seed power, promoted onto `q = 0` tones, plus a
   small amplitude on `q = +-1` at `omega_a = 0.0917 * omega_p`;
2. if that fails, sweep the seed amplitude over three decades before
   concluding anything;
3. if that fails, and only then, build the Hill eigenvector seed (Phase 6).

**Start mid-window, not at onset.** Above a supercritical Neimark-Sacker both
solutions exist: the period-1 orbit, now unstable, and the torus. Near onset
the torus amplitude goes to zero, so the two nearly coincide and Newton falls
back onto period-1 -- which is exactly what the below-onset `jc_jtwpa` control
run demonstrated (off-comb fraction 1.4e-9 at `-29.7` dBm, residual 1.6e-12:
the solver correctly reduces to period-1 when no torus is present). Deepest in
the window the torus is furthest from the trivial branch and easiest to land
on.

Converged 2c pump states already on disk, mapped onto the control axis by
**current** so no power convention enters (`I_bound = 1.1628e-05 A`):

| artifact | `I/I_bound` | use |
| --- | ---: | --- |
| `.hybrid_outputs/period1_recovery_7p9_2c_v1/point_-23.800000/pump` | 0.6050 | **try first**, mid-window |
| `.hybrid_outputs/period1_recovery_7p9_2c_v1/point_-24.000000/pump` | 0.5912 | second, near onset |
| `.hybrid_outputs/period1_recovery_7p9_2c_v1/point_-24.250000/pump` | 0.5745 | **negative control**, below onset |

The torus window is 0.5825 to 0.6075. The control must give off-comb content at
the `jc_jtwpa` floor; if it produces finite `q = +-1` content, the anchor is
manufacturing a solution and every result above it is void.

### Success criteria

**Manual**: a converged solve with residual at the production tolerance, a
positive `omega_a`, and non-zero `q = +-1` content. That is the whole gate.
Report the converged `omega_a / omega_p` but do not gate on it yet.

**If it does not converge**, the report must say which of the three seed routes
were tried, the residual history, and whether Newton stalled or the line search
collapsed. A bare "did not converge" is not a result.

---

## Phase 3: continue up in power and find the real wall

**This phase produces the deliverable.**

March the converged solution up in pump power, warm-starting `(X, omega_a)`
from the previous point, adaptive step, using `solve_adaptive_continuation`'s
construction. Continue until it genuinely stops.

Report, per device: the last converged power, the terminal residual behaviour,
and the failure mode. The headline number is

```text
HB previously failed at <old wall>; the torus branch reaches <new wall>.
```

on 2c at 7.9 GHz, against the documented pre-existing wall for that column.

### Success criteria

**Manual**: the branch passes the onset power. Anything beyond that is upside.
If the branch converges at onset but cannot be continued even one step, say so
plainly -- that is a real and reportable outcome, and it points at the
continuation scheme rather than the ansatz.

---

## Phase 4: repeat on jtwpa and guarcello

Same driver, same procedure, using each device's own `omega_a` ratio and onset
from the table above. Three devices, because one device cannot separate
device-specific behaviour from solver behaviour.

`guarcello` currently has **no HB circuit input** -- it is an analytic paper
device, not a circuit directory. Establishing one is part of this phase, and if
it turns out to be a larger job than the rest of the phase, report that and
finish the other two rather than blocking.

---

## Phase 5: (moved) Schur is now Phase 1C

Schur was originally scheduled here as an optional memory optimization. It is
now a **prerequisite for the 2c production-basis run** and has moved to
Phase 1C, because `run_hybrid_column.py:691-697` documents the full-node
backend as not numerically equivalent near the 2c high-power obstruction. The
content below is retained as the specification for Phase 1C.

### Changes required

1. `multitone/torus.py` -- accept `SchurMultiToneProblem` as `base_problem`.
   After Phase 1 the only remaining Full-only dependency is structural:
   `full_problem()` calls `dataclasses.replace(self.base_problem, basis=...,
   cache={})`, and `SchurMultiToneProblem` is a plain class, not a dataclass.
   Replace that with a rebuild helper that reconstructs the problem at the new
   `omega_a` for either type, via `build_multitone_schur_problem`
   (`multitone/schur.py:246`) when the partition is present.
2. The partition depends on the circuit graph, not on `omega_a`, so it is built
   and factored **once** and reused at every generator frequency. Do not
   rebuild it inside the Newton loop.
3. The anchor's node index is a **retained-set** index under Schur, not a
   full-node index. Anchor on a retained node and record which, so a Schur and
   a full run are comparable.

### Success criteria

**Manual**: a Schur torus solve at one 2c power reproduces the full-node
solution's `omega_a` to 1e-6 relative and its `q = +-1` norm fraction to 1e-6,
at the measured smaller footprint. If the two disagree, the Schur path is wrong
and the full-node number stands.

---

## Phase 6: validate against the measurement

Only now, with branches in hand.

- solved `omega_a / omega_p` against the **plateau** values 0.0917 / 0.1217 /
  0.3555, 2 percent. See the rho section below for why the plateau and not the
  onset;
- fraction of solution norm on `|q| >= 1` against the measured off-comb share
  0.986 / 0.999 / 1.000;
- predicted spectral peaks at `h*f_p + q*f_a` against the FDTD spectrum,
  positions and relative amplitudes;
- `Q`-convergence: `omega_a` and branch amplitude move less than 1 percent
  between `Q = 1` and `Q = 2`. Run `Q = 2` on `jc_jtwpa` full-node (5.0 GB,
  fits) and on `ipm_2c_fixed` **through Schur** (6.1 GB; full-node would be
  14.9 GB and does not fit). `Q = 3` is out of reach on 2c by either route and
  is not planned; if `Q = 1` and `Q = 2` disagree, that is the finding to
  report, not a reason to buy hardware.

---

## Phase 7: conditional instrumentation

**Build only if Phase 2 or Phase 3 fails and the reason is not evident from the
residual history.** These are already written but unvalidated:
`signal/branch_tracking.py`, `pump/neimark_sacker.py`,
`scripts/scan_hb_floquet_branch.py`, `scripts/report_transition_boundary.py`.

Their purpose is to answer "why did it stop", not to gate the attempt. If they
are needed, they need real reference systems first -- see the testing section;
the tests committed alongside them today are shape-only and validate nothing.

Two things worth knowing before trusting the Hill route on 2c: the earlier
dense scan was declared undecidable because thousands of multipliers sit within
1e-8 of the unit circle on a lossless circuit, and the fix is branch *tracking*
from a known-stable power, not a denser scan. If tracking also fails, the
fallback is a small `tan_delta`, which flips the loss model to
`conductance_abs_omega` and removes Tier-2 refinement -- measure that trade at
Tier-1, where both conventions are legal, rather than assuming it is small.

---

## Phase 8: wire into the production map

Opt-in only. `scripts/run_gain_map.py --stability-gate`, default off. Cells past
the transition are labelled rather than reported as a gain. Defaults must
reproduce current map output byte for byte
(`tests/test_run_gain_map_cli.py`, `tests/test_exp10_gate.py` unchanged with the
flag absent).

---

## Experimental scope

Scope is pinned before any run. Broad multi-knob campaigns on this project have
repeatedly produced conclusions that did not survive the next session. One
column per device.

### Fixed for the whole plan

| device | design source | `f_p` | pump port | signal | control axis |
| --- | --- | ---: | ---: | --- | --- |
| `ipm_2c_fixed` | `designs/ipm_2c_fixed` | 7.9 GHz | 4 | 1 -> 2 @ 7.4 GHz | `I/I_bound` |
| `jc_jtwpa` | `outputs/jc_doc_python_designs/jc_jtwpa` | 7.12 GHz | 1 | 1 -> 2 @ 6.62 GHz | pump dBm |
| `guarcello` | analytic paper device | 7.0 GHz | n/a | n/a | pump dBm |

`I/I_bound` on 2c refers to the PALC fold current `1.1628e-05 A`; `0.575`
is `I_p = 6.6861e-06 A`, instrument `-24.242 dBm`, on-chip `-59.517 dBm`.

Also fixed and not varied: pump basis (`positive_odd_jc`, `K = 10`, modes
`[1,3,...,19]`), loss model (`pump_line_loss_model_A10` on 2c and jtwpa), power
convention (`legacy_traveling_wave`).

### Engine settings must match production

Do not invent settings. The authorities are
`workflows/run_gain_map_and_plots.py::DEFAULT_ENGINE_FLAGS` and
`scripts/run_hybrid_column.py::main`. Any torus run must use the same solver
configuration, or a difference in result is not attributable to the ansatz.

| setting | value | source |
| --- | --- | --- |
| pump backend | `schur_cpu_mt` | both; `run_hybrid_column.py:693` |
| preconditioner | `real_coupled_fast` | both; `run_hybrid_column.py:694` |
| Schur cache size | `1` | `run_hybrid_column.py:697` |
| pump solution dtype | `float64` | `run_hybrid_column.py:698` |
| `--pump-current-jc-scale` | `1.0` | `DEFAULT_ENGINE_FLAGS` |
| `--fold-skip-patience` | `2` | `DEFAULT_ENGINE_FLAGS` |
| `--inproc-fold-predictor` | `secant` | `DEFAULT_ENGINE_FLAGS` |
| `--signal-detuning-mhz` | **150** | `DEFAULT_ENGINE_FLAGS` |
| attenuation | unset, i.e. the loss_A10 model | `run_hybrid_column.py:622` |
| `--pump-port` (2c) | `4` | design default |
| factor backend | **`pardiso`** | banded is excluded on 2c, see above |

`--signal-detuning-mhz 150` supersedes the `500` recorded in the CLAUDE.md
standard flag block; the workflow file is newer and is the authority. Signal
detuning does not enter a pump-only torus solve, but it must match whenever a
gain number is compared.

### Power windows

| phase | `ipm_2c_fixed` | `jc_jtwpa` | `guarcello` |
| --- | --- | --- | --- |
| 2, first convergence | 0.585, single point | -29.45 dBm, single point | -54.00 dBm, single point |
| 3, continuation | 0.585 -> as far as it goes, step 0.0025 | -29.45 -> onward, 0.02 dB | -54.00 -> onward, 0.05 dB |
| 6, `Q` check | `Q=1` full-node, `Q=2` **via Schur** | `Q=1,2` full-node | `Q=1,2` full-node |

### Explicitly deferred

Any second pump frequency; any signal-frequency sweep; `ipm_7c_new` or any
device beyond the three above; the Themis boundary comparison (it enters only
once the model side is internally consistent, and carries its own unresolved
`(df, dP)` calibration degeneracy).

---

## The rotation number is not an onset gate

Recorded so it is not re-derived. Measured on the stored return-map records:
rho spread across the torus window is 45 percent of its mean on `jc_jtwpa`,
186 percent on `ipm_2c_fixed`, 194 percent on `guarcello`. It is useless as an
onset marker, for two reasons. Structurally, an NS crosses in multiplier
*magnitude*; the angle passes through smoothly and takes no distinguished
value. Numerically, just above onset the invariant circle is barely above the
strobe floor, so the unwrapped `atan2` is measuring noise.

rho is good in one place, the plateau inside the torus classification:

| device | plateau | `1 - rho` | spread |
| --- | --- | --- | ---: |
| `jc_jtwpa` | -29.198 .. -29.093 | 0.12152, 0.12168, 0.12180, 0.12207 | 0.45% |
| `guarcello` | -53.95 .. -53.80 | 0.35274, 0.35421, 0.35507, 0.35545 | 0.76% |
| `ipm_2c_fixed` | none exists | - | - |

Alias branch: raw geometric rho is near 0.878 and 0.645; the quoted 0.1217 and
0.3555 are `1 - rho`. State the branch whenever either appears. `ipm_2c_fixed`
has no plateau and is validated on amplitude and spectrum only -- a declared
limitation, not a relaxed tolerance.

Where an onset number is needed, use the squared-radius threshold fit
`r^2 ~ (mu - mu_c)`, which locates the boundary to +0.0016 dB on jtwpa and
+0.0016 on 2c. `scripts/chaos/onset_threshold.py` implements it.

---

## What we are not doing

- **No period-N or half-pump ansatz.** Falsified: admitting half-integers at
  collapse adds 4.8 percentage points, thirds 10. `pump/floquet.py`,
  `pump/periodic_branch.py`, `signal/period_doubled.py` and
  `build_half_pump_basis` stay dormant.
  The `PERIOD_DOUBLING_ONSET` field in the FDTD campaign `result.json` is
  **not evidence against this**: it fires on known period-1 states (2c at
  `I/I_bound = 0.575`), its half-integer lines sit 110-124 dB below the pump,
  and it is emitted with `winning_n = 8` and "period test returned 0", meaning
  the period test never closed. Do not reopen the ansatz gate on that field.
- **No broadband or statistical HB for the chaotic state.** The residue at
  collapse is a continuum; the best single extra generator explains 19 percent.
- **No new linear algebra.** See rule 1 above.
- **No replacement of the FDTD kernel.**
- **No change to any published gain or P1dB number.**

## Prior runs that are not usable

Two FDTD runs were produced against the earlier draft's prerequisites. Both are
unusable, recorded here so they are not trusted later:

- `outputs/chaos/plan_prereq_jtwpa_dt005` -- the dt/2 run is **not settled**. A
  split-half radius test on its analysis window gives 7.27e-6 then 1.02e-5, a
  ratio of 1.407 and rising, at a power the stored campaign calls period-1
  (where the same test gives 0.833, settled and decaying). Its `r_RMS` reads
  146x the stored value at the same control point and is flat across the
  transition. Consistent with `implicit_trapezoid` not being L-stable, so
  halving the step adds no damping.
- `outputs/chaos/plan_prereq_guarcello` -- intended as a power-grid refinement,
  but it also halved `dt_s` (2.868e-14 against 5.737e-14) and shortened the run
  3.5x, leaving 299 strobes against 1049. It cannot be compared to the
  reference data. The four new points are internally consistent with each other,
  and among themselves the radius steps are 2.32 / 1.55 / 1.23 -- graded rather
  than a cliff, which is weak positive evidence for a supercritical onset.

Neither is a prerequisite any more. If a supercritical/subcritical answer is
ever needed, re-run guarcello at the **stored campaign's own timestep and
duration**, changing only the power grid.

---

## Testing strategy

Local precedent: every estimator written for this project was wrong on first
implementation and was caught only by a reference system with a known answer.
One read the Rossler exponent exactly 2x high. The return-map suite classified
129 of 133 points as chaos because a floor was set at machine epsilon. Device
data does not validate an estimator here.

The tests presently committed for the torus and branch-tracking modules are
**shape-only** -- they assert that a Jacobian is square, that a tone is in a
basis, and that an anchor returns `.imag`, all evaluated at a zero state.
Nothing in them would catch a wrong `omega_a` or a wrong multiplier. They must
not be described as validation.

Required before any device number is reported:

| reference | exercises | expected |
| --- | --- | --- |
| dense bordered solve on a small circuit | Phase 1 fix | matches `bordered_solve_refined` to 1e-10 |
| forced van der Pol inside its torus | `TorusProblem` | `omega_a` matches direct integration to 1e-6 |
| phase-rotated seeds of the same state | the anchor | both converge to the same physical state |
| circle map locked at 2/5 | anchor under locking | no spurious unlocked solution |
| linear time-periodic 2x2, analytic monodromy | Phase 6 only | multipliers match closed form to 1e-10 |
| forced van der Pol at a known NS | Phase 6 only | crossing power matches direct integration |

If a reference fails, fix the estimator. Never tune a device threshold to
compensate. Mutation-verify each gate: show it failing before it passes. The
committed stability tests once could not catch a real defect because they ran
on `Ic = 0`, a linear circuit, at a zero state.

Full suite:
`python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_full_slow --run-slow`.
Six failures currently pre-date this work (`test_design_cli`,
`test_design_compact`, two in `test_kimpa_gain`, two in `test_loss_model`); the
`test_loss_model` pair is the documented 2026-08-06 port-convention revert. Do
not report a bare pass/fail count -- name any new failure.

## Rollback

New files: `multitone/torus.py`, `signal/branch_tracking.py`,
`pump/neimark_sacker.py`, `scripts/chaos/onset_threshold.py`, three drivers.
Edits to existing modules are additive constructors and additive fields.
Phase 8 is a flag defaulting off. No production path changes behaviour until
Phase 8, and Phase 8 changes nothing unless its flag is passed.
