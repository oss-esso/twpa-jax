# A saturation solver for parametric amplifiers, from first principles

**Scope.** How to build, from nothing, a solver that answers one question: *how
does a Josephson parametric amplifier's gain fall as the signal power rises, and
why?* This document goes from the device physics to the software boundaries, the
numerics, the observables, and the validation ladder. It assumes no existing
code.

**Retained constraint.** One workstation: AMD Ryzen 5 5600G, 6 physical cores /
12 threads, 15.3 GB RAM (typically 3–7 GB genuinely free), Windows, no GPU worth
using, no cluster. Every design choice below is made under that budget, and the
budget is treated as a first-class input, not an afterthought. A formulation
that needs 60 GB is not "slow" here — it is unavailable.

---

## 1. What "saturation" means, precisely

A parametric amplifier is pumped at `ω_p` and amplifies a small signal at `ω_s`.
For weak signals the gain is a constant `G_lin` set by the pump. As the signal
grows, gain falls. The standard scalar summary is the **1 dB compression point**:

```
P1dB = the input signal power at which G(P_s) = G_lin - 1 dB
```

Three things must be said about this definition before any code is written,
because each is a place where a whole campaign can go wrong.

**(a) `P1dB` is not a property of the device alone.** It depends on `ω_s`, on the
pump operating point, and on `G_lin`. Comparing two `P1dB` numbers taken at
different gains is meaningless. The physically meaningful object is the
**function** `P1dB(G_lin)` — or, since it is usually close to linear over a
decade, its **slope** `dP1dB/dG_lin` in dB/dB. Design the campaign to produce
the slope, not a number.

**(b) `G_lin` appears on both sides.** The compression target is `G_lin - 1`, so
any error in `G_lin` moves `P1dB` in a correlated way. In a fit of `P1dB` against
`G_lin` this manufactures spurious negative slope. Any estimator must be checked
for this, on both simulated and measured data, by a split-sample construction
(estimate `G_lin` from one subset of power points, locate the crossing with a
disjoint subset).

**(c) The extraction rule is part of the measurement.** "Smooth, then take the
last upward crossing of `G_lin - 1`" and "bracket the crossing and refine it with
extra nonlinear solves" are different estimators and give different answers on
the same curve. Fix one rule, write it down, and apply the *identical* rule to
every dataset being compared, rescaling any smoothing window to the grid spacing
of each dataset. An estimator applied to a 0.5 dB/step measurement grid and a
5 dB/step simulation grid is not the same estimator.

### 1.1 What we actually want to learn

`P1dB` is a summary. The scientific question underneath is **which mechanism
limits the amplifier**, because different mechanisms have different signatures
and different cures:

| mechanism | signature | `dP1dB/dG` |
| --- | --- | ---: |
| pump depletion | pump power falls where gain compresses; conversion fraction hits `10^0.1 − 1` | **−1** |
| phase-mismatch drift (SPM/XPM) | `Δk` moves with signal power *before* compression | steeper than −1 |
| standing-wave / resonant saturation | spatial profile flattens; gain never was a traveling-wave buildup | ≈ 0, gain-independent |
| harmonic/sideband cascade | power leaves the 3-tone scope into `\|q\| ≥ 2` | between 0 and −1 |
| loss | compression weakly dependent on signal power | ≈ 0 |

The solver must be able to *distinguish* these, not just produce `P1dB`. That
requirement drives most of the observable design in §6.

---

## 2. The physics

### 2.1 The nonlinear element

A Josephson junction is a purely inductive, non-dissipative, nonlinear element
with current–phase relation

```
I(Φ) = Ic sin(Φ / φ0),        φ0 = ħ / 2e = 3.29106e-16 Wb
```

where `Φ` is the **branch flux** (time integral of the branch voltage) and `Ic`
the critical current. Its differential inductance is

```
L(Φ) = φ0 / (Ic cos(Φ/φ0)) = Lj / cos(Φ/φ0),     Lj = φ0/Ic
```

Everything about parametric amplification follows from expanding this: with a
large pump flux `Φ_p(t)`, `cos(Φ_p/φ0)` is time-periodic, so the line's
inductance is *modulated at the pump frequency and its harmonics*. That is the
parametric coupling. Nothing else is needed.

For an unbiased junction the expansion of `sin` is odd, so the leading
nonlinearity is cubic → **four-wave mixing**, `2ω_p = ω_s + ω_i`. Break the
symmetry (DC bias, flux-biased SQUID, SNAIL) and a quadratic term appears →
**three-wave mixing**, `ω_p = ω_s + ω_i`. The solver must not assume either; it
should evaluate the branch law as given and let the mixing order emerge.

**Composite elements.** A SQUID or SNAIL is several junctions and its own
internal nodes. Two representations are possible: resolve every junction
explicitly (more nodes, no approximation), or collapse the cell to one effective
branch law `I_eff(Φ)` obtained by solving the cell's internal static equilibrium.
The second is far cheaper and is exact for the static limit, but it discards the
cell's internal dynamics. **If the effective branch law is used, the static
equilibrium `Φ*` must be solved first and the dynamic flux measured from it**:
`I_branch(δΦ) = I_eff(Φ* + δΦ)`. Skipping the shift silently changes the mixing
order.

### 2.2 Circuit equations

Take node fluxes `Φ_n(t)` as the state (`V_n = dΦ_n/dt`). Kirchhoff's current law
at every non-ground node, with capacitances, conductances, linear inductances and
the nonlinear branches, gives one second-order ODE system:

```
C Φ̈  +  G Φ̇  +  K Φ  +  B_J · I_J(B_Jᵀ Φ)  =  I_src(t)
```

- `C` — capacitance matrix (ground caps, junction caps, coupling caps)
- `G` — conductance matrix (port terminations, dielectric loss)
- `K` — inverse-inductance matrix from linear inductors and mutual inductance
- `B_J` — node-to-branch incidence for the nonlinear branches
- `I_src` — current sources at the driven ports

This is the entire model. Every device (JPA, JTWPA, SNAIL line, multiport
coupler) is a different `(C, G, K, B_J, Ic)`, not different code.

**Sign and unit conventions must be declared once, in one place, and asserted by
a test.** Whether `Φ` is flux or reduced phase, whether the reconstruction of a
real signal from positive-frequency phasors carries a factor 2, whether currents
are peak or RMS — each of these is a factor that propagates into power as a
factor of 4 (6.02 dB) and will not announce itself. See §6.1.

### 2.3 Why the small-signal theory is not enough

Linearize about a periodic pump solution `Φ_p(t)`. The perturbation sees a
time-periodic inductance, so by Floquet's theorem its solutions are
`e^{iωt} × (periodic)`, i.e. the signal couples to a ladder of sidebands
`ω + m ω_p`. Solving that **linear** sideband system gives the small-signal gain
exactly, cheaply, and for free — the pump is an input, not an unknown.

That is also precisely why it cannot describe saturation: the signal does not act
back on the pump. Small-signal Floquet gain is *independent of signal power by
construction*. Saturation requires the pump and signal to be solved
**simultaneously**, as one coupled nonlinear problem. This is the fundamental
reason the saturation solver is expensive and the gain solver is not, and no
amount of engineering removes it.

The Floquet solver is still worth building — as the `P_s → 0` limit that the
nonlinear solver must reproduce (§7, L2).

### 2.4 The analytic bound

The crudest saturation model keeps only pump depletion: every amplified signal
photon costs pump photons, and

```
G(P_s) = G_lin / (1 + 2 G_lin P_s / P_p)
```

Setting `G(P_s) = G_lin/10^{0.1}` gives

```
P1dB = P_p − G_lin,dB + 10 log10[(10^{0.1} − 1)/2]
     = P_p − G_lin,dB − 8.8786 dB
```

The constant `−8.8786 dB` is universal — it contains no device physics. If a
boss's rule of thumb says `P_p − G + 9 dB`, that is this formula with a sign
convention on `G`. Check the sign before agreeing.

**The slope this predicts is exactly −1 dB/dB — and that is a reference line, not
a bound.** It is worth being precise about the direction of the inequality,
because getting it backwards inverts the diagnosis:

- Observed `P1dB` is set by whichever mechanism reaches its threshold *first*, so
  the composite threshold is the **minimum** over mechanisms.
- A **gain-independent** limiter (a fixed amplitude ceiling, a resonance
  flattening, a loss floor) has a `P1dB` that does not move with `G_lin` —
  slope 0. Composed with depletion it dominates at low gain and yields a
  composite slope **between −1 and 0**, i.e. *shallower*.
- A limiter whose threshold falls **faster** than 1 dB per dB of gain — e.g.
  phase-mismatch drift, where higher gain means higher circulating power means
  faster `Δk` walk-off — yields a slope **steeper than −1**.

So the depletion line partitions the diagnosis rather than bounding it:

| observed slope | reading |
| --- | --- |
| ≈ −1 | depletion-dominated |
| shallower than −1 | a **gain-independent** mechanism co-limits and dominates at low gain |
| steeper than −1 | a mechanism whose threshold is **super-linear in gain** dominates |

That is still a cheap, dimensionless test worth running before any comparison to
hardware — it just answers "which regime", not "pass/fail".

Same caution on the companion number. Within the depletion-only model, 1 dB of
compression requires converting `10^{0.1} − 1 = 0.2589` of the pump, i.e.
**1.302 dB of pump depletion**. That figure is **specific to that model**, not a
general acceptance criterion: a device compressing by another mechanism can and
should show less. Use it as a consistency probe — if a solver reports 1 dB of
compression with 0.05 dB of depletion, then either the depletion observable is
wrong or the compression is not depletion-driven, and both are worth knowing —
but never as a gate.

### 2.5 Mechanisms the model must be able to express

For a solver to be *capable* of the right answer, its basis must contain the
relevant tones:

- **Pump depletion** — signal growth feeding back on the pump: `|A_s|² A_p`.
- **SPM / XPM** — power-dependent phase, hence power-dependent phase matching.
- **Signal harmonics and secondary mixing** — `2ω_s − ω_p`, the signal acting as
  a weak second pump.
- **Pump harmonic generation** — `3ω_p`, `5ω_p`; strong in a driven Josephson
  line and it reshapes the pump waveform itself.
- **Loss**, which is frequency- and sometimes power-dependent.

The first two are representable in a minimal 3-tone basis. The third is not, and
that has consequences for the basis design in §4.1.

---

## 3. Choosing the formulation

Four candidates, evaluated against the hardware budget.

### 3.1 Time-domain transient to steady state

Integrate the ODEs from rest until transients die, then FFT the output.

- **Pros:** no tone truncation whatsoever; every mixing product is present;
  completely different error modes from every frequency-domain method.
- **Cons:** the Josephson plasma frequency sets a stiff timescale; reaching
  steady state takes many pump periods; and extracting a −1 dB change in a small
  signal riding on a large pump demands a very long, very clean record. Getting
  0.01 dB resolution on the signal tone is a dynamic-range problem, not a speed
  problem.
- **Verdict:** unusable as the production workhorse; **extremely valuable as an
  independent oracle at one or two operating points** (§7, L6). Build it. It is
  a few hundred lines and it is the only check that shares no assumptions with
  harmonic balance.

### 3.2 Harmonic balance (multitone)

Expand the steady state in a finite set of tones, require the residual to vanish
at each tone, solve the resulting algebraic system by Newton.

- **Pros:** directly computes the periodic/quasi-periodic steady state with no
  transient; resolution in tone amplitude is limited only by the linear solve;
  handles strong drive.
- **Cons:** requires choosing a tone set — and *the choice is a physical
  approximation, not a numerical parameter*; cost grows with the square of the
  tone count in the preconditioner.
- **Verdict:** **the production formulation.** Everything below assumes it.

### 3.3 Shooting / periodic boundary value problem

Solve for the initial condition that reproduces itself after one period. Elegant
for a single-frequency drive, awkward for the quasi-periodic pump+signal case
(the state is not periodic unless the frequencies are commensurate), and inherits
the stiffness of §3.1. Not recommended.

### 3.4 Coupled-mode / envelope equations

Reduce the line to three slowly-varying envelopes `A_p(x)`, `A_s(x)`, `A_i(x)`
along the propagation coordinate, with coefficients derived from the branch
expansion; integrate three ODEs in `x`.

- **Pros:** milliseconds per point. Contains depletion, SPM, XPM, conversion and
  loss explicitly and separably — you can switch terms off one at a time.
- **Cons:** assumes a traveling wave, a slowly-varying envelope, no reflections,
  no harmonics, and a well-defined propagation coordinate. All four assumptions
  fail for some real devices.
- **Verdict:** **build this first, as an independent reference**, not as the
  product. Its assumptions are exactly the ones the full solver must be checked
  against, and where the two disagree, the *reason* for the disagreement is the
  physics result.

### 3.5 The recommended stack

```
   coupled-mode envelope model   (fast oracle, transparent terms)
              ↕  agree in the weak, phase-matched, traveling-wave regime
   multitone harmonic balance    (production)
              ↕  agree at one point
   time-domain transient         (assumption-free oracle, expensive)
```

Two independent references bracketing the production solver. Neither is
"validation against reality" — that is §7, L7 — but disagreement between any two
of them localizes a defect in a way that no internal residual check can.

---

## 4. Discretization

### 4.1 The tone basis — the most consequential decision in the solver

With a pump at `ω_p` and a signal at `ω_s`, define the detuning

```text
δ = ω_p − ω_s
```

Every mixing product of the two lies on the 2-D lattice

```text
ω_{h,q} = h ω_p + q δ
```

- `h` — pump harmonic index. `h = 1,2,3,…` covers pump harmonic generation and
  the reshaping of the pump waveform.
- `q` — **signal-photon index**: how many net signal quanta the tone carries.
  Pump is `(1, 0)`, signal is `(1, −1)`, idler is `(1, +1)`.

**Declare the sign of `δ` once and assert it in a test.** Both conventions
(`ω_p − ω_s` and `ω_s − ω_p`) appear in the literature; they differ only by
`q → −q`, so mixing them swaps signal and idler silently, and every downstream
observable keeps working while reporting the wrong tone. The convention above is
the one used consistently for the rest of this document.

**`h` and `q` are two independent truncation axes and each needs its own
convergence study.** This deserves emphasis because there is a trap here that is
easy to walk into and hard to detect.

A natural-seeming alternative is the **Floquet sideband ladder**: index tones by
`m` and take `ω_s + m ω_p`. It is the obvious basis because it is what the
small-signal theory produces. But on the lattice above,
`ω_s + m ω_p = (m+1) ω_p − δ`, so sideband `m` maps to `(m+1, −1)` — **every
rung has `q = −1`**. With conjugates supplying `q = +1` and pump harmonics
filling `q = 0`, **a sideband ladder contains only `q ∈ {−1, 0, +1}` at every
ladder length.** Extending it adds `h`, never `q`.

Consequences:

- Pump depletion and SPM/XPM *are* representable (`|A_s|² A_p` has
  `q = −1+1+0 = 0`).
- `2ω_s − ω_p` and the signal's own harmonics are **structurally absent**.
- A convergence study that lengthens the ladder will converge beautifully and
  prove nothing about `q`.

So: **use a rectangular `(h,q)` basis, `|h| ≤ H`, `|q| ≤ Q`, and report
convergence in `H` and `Q` separately.** Expect `Q = 2` to matter and `Q = 3` to
be converged; expect the effect to be in the direction of *less* compression
(the extra channels return power to the signal), so a `Q = 1` solver
over-compresses and reports `P1dB` biased low.

Conjugate symmetry: the state is real, so `X_{−h,−q} = conj(X_{h,q})`. Store one
half-plane and reconstruct — a factor 2 in memory and cost, for free.

Practical sizes: a full rectangle has `(2H+1)(2Q+1)` lattice points, of which
`((2H+1)(2Q+1) − 1)/2` are retained after discarding DC and keeping one
half-plane. So `H = 5, Q = 3` gives **38** tones; `H = 10, Q = 1` gives **31**;
`H = 10, Q = 3` gives **73**. Since the dominant cost scales as `n_tones²`
(§5.2), 73 tones is roughly 5.5× the preconditioner cost of 31 — the point at
which `H` and `Q` start competing for the same budget. That tension is real and
should be resolved by measurement, not by assumption.

### 4.2 Evaluating the nonlinearity: alternating frequency–time

`sin()` has no closed form in the tone basis. The standard device is **AFT**:
inverse-transform the branch fluxes onto a time grid over the 2-torus
`(θ_p, θ_δ) ∈ [0,2π)²`, apply `Ic sin(·)` pointwise, forward-transform back,
keep the retained tones.

The time grid must satisfy the anti-aliasing condition, and **the classical
`3N+1` rule is the quadratic one — it is not sufficient here.** Derivation, per
axis, for retained modes `|h| ≤ H`: an order-`k` product reaches mode `kH`, which
on an `n_t` grid aliases to `kH − n_t`. Keeping that outside the retained band
requires `|kH − n_t| > H`, i.e.

```text
n_t > (k + 1) · H
```

So quadratic needs `n_t > 3H` (the familiar `3N+1`), and a **cubic** — the
leading term of an unbiased Josephson line — needs `n_t > 4H`. Since `sin()` has
all orders, no finite rule is exact; `4H` is the floor, not the answer.

A robust construction that gets this right automatically is to size the grid from
the **pairwise sums actually present in the basis**: `max_h = max|h_i + h_j|`
over retained tones, then `n_t ≥ 2·max_h + 1` rounded up to even. For a rectangle
this yields `n_t ≳ 4H` without hard-coding the nonlinearity's order. Do the same
independently on the `q` axis.

Above that floor the requirement is empirical — **increase `n_t` until the
residual stops moving, and record the value.** Under-resolving aliases high
harmonics back onto the retained tones and produces a converged, wrong answer.
This is a silent failure mode and warrants an explicit convergence test, not a
default.

Cost: one AFT is `O(n_branch · n_t,p · n_t,δ · log)`. It is not the bottleneck.

### 4.3 Size accounting

Real-packed unknowns:

```
N = 2 · n_tones · n_nodes
```

For a 2500-node line and 31 tones: `N = 155,000`. This is the number that
determines whether the machine can run the problem.

---

## 5. Numerics

### 5.1 Newton–Krylov, matrix-free residual and Jacobian

The harmonic-balance residual at tone `v`:

```
R_v(X) = [ −ω_v² C + i ω_v G + K ] X_v  +  B_J · Î_J[X]_v  −  S_v
```

The Jacobian's linear part is block-diagonal in tone index. The nonlinear part is
**not**: differentiating the AFT gives a multiplication in time by
`γ(t) = (Ic/φ0) cos(Φ_branch(t)/φ0)`, which is a **convolution in tone space**,
coupling every tone to every other through `γ̂`. So:

- the Jacobian-vector product is cheap and matrix-free: one inverse transform,
  one pointwise multiply by `γ(t)`, one forward transform;
- the Jacobian *as a matrix* is block-dense in tone index, which is what makes
  preconditioning expensive.

Use Newton with a Krylov (GMRES) inner solve on the JVP. Line search on the
residual norm; cap Newton iterations; converge on a **relative coefficient
residual** with an explicit tolerance, not on a step-size heuristic.

### 5.2 Preconditioning: where the memory goes

GMRES on the raw JVP converges badly — the tone coupling is strong. The effective
preconditioner is the **assembled coupled Jacobian, factorized**. Options, in
increasing fidelity:

1. **Block-diagonal in tone** (ignore `γ̂` off-diagonals): cheap, poor at strong
   pump.
2. **Mean-tangent**: replace `γ(t)` by its time average. Better, still poor when
   the pump is strong enough to matter — which is the entire regime of interest.
3. **Exact coupled Jacobian, factorized.** GMRES then converges in ~3 iterations.

Option 3 is right, and it sets the memory budget. **The assembled matrix's size
scales as `n_tones²`, not as `N`** — because the tone blocks are dense. Doubling
the tone count quadruples the factorization cost while only doubling the
dimension. This is the single most important cost fact in the design, and it is
why the small-signal solver is cheap (it treats sidebands as a *linear*
factor-once system) while the saturation solver is not.

**Ordering matters enormously.** Packed tone-major, the matrix couples
everything. Ordered **node-major** — index `node · 2n_tones + tone_block` — a 1-D
chain device collapses onto a band of width ~3 tone-blocks, because the *circuit*
is a chain. Then LAPACK general-band storage applies:

```
memory ≈ N · (3·bw + 1) · 8 bytes
```

With `N = 155,000` and `bw ≈ 190`: ≈ 0.7 GB. A general sparse LU of the same
matrix will typically cost 2–4× that. **Measure the bandwidth from the assembled
sparsity pattern; never assume it.** A device that is not a chain (multiport
couplers, ladders with rungs) will not band, and the fallback is a sparse direct
factorization.

Recommended: implement both a sparse-direct backend (PARDISO/MKL or SuperLU) and
a banded backend, select by a measured bandwidth threshold, and verify the two
give the same converged solution to ~1e-9 dB. They must, since this only changes
the preconditioner.

In exact arithmetic an exactly-inverted Jacobian converges GMRES in **one**
iteration; in practice, with restarts and an inexact factorization, expect 1–3.
Either way it is *few*, and that is the fact that matters for the next paragraph.

Modified-Newton — reusing one factor across `N` Newton steps — is a tempting
optimization whose value is **benchmark-dependent, not a design law**:

- Where the preconditioner is near-exact and GMRES already converges in a couple
  of iterations, reuse is typically a **loss**: one GMRES iteration (a JVP plus a
  triangular solve) costs about as much as one factorization, so there is no
  cheap-preconditioner regime to amortize, and a stale exact factor is worse than
  a fresh one.
- Near a fold or a stiff turning point, where the Jacobian is ill-conditioned and
  Newton is taking many small steps, reuse can dominate.

Default to refactoring every step, expose the reuse count as a flag, and record
the measurement per device class rather than asserting either outcome. Keep the
update itself against the **true** Jacobian so the converged solution cannot
depend on the reuse count.

### 5.3 Continuation in signal power

Newton from a cold start fails at high signal power. The natural parameter is the
signal source amplitude:

1. Solve the **pump-only** problem first (1-D harmonic balance, small, fast).
2. Promote that solution into the 2-D basis (shared tones copied, new tones zero)
   and solve at zero signal — this must reproduce the pump-only answer exactly
   (§7, L1).
3. Walk the signal amplitude up the sweep grid, seeding each point from the
   previous converged one.
4. On failure, subdivide the interval adaptively rather than abandoning the
   point.

**Design rule with teeth: a failed interior point must invalidate the bracket it
sits in.** If a solver interpolates `P1dB` across a hole in the power grid, it
will silently return a plausible number that is wrong by whatever the curvature
across the hole is. Record `n_failed_points`, the powers, and a `degraded` flag
in the summary; refuse to report a refined `P1dB` whose bracket contains a hole.

### 5.4 Parallelism on this machine

The natural parallel axis is **across signal frequencies** (independent problems,
no communication). Do not thread the linear algebra: measured on this class of
part, MKL PARDISO at 6 threads is ~4× *slower* than serial, and threads lose to
processes. Pin the factorization to one thread and run `k` worker processes.

Worker count is set by **peak RSS per worker against free RAM**, and peak RSS is
set by `n_tones²` (§5.2). Estimate it from the basis size before launching and
refuse to over-subscribe; a swapping run is worse than a serial one. Throughput
is memory-bandwidth-bound and plateaus around 3 workers on a 6-core part —
measure it once, then stop adding workers.

### 5.5 Realistic budget

For a 2500-node line on this machine. These are order-of-magnitude planning
figures and they are **strongly** superlinear in tone count (§5.2) — treat the
`~30` column as the budget and the `~70` column as the warning:

| item | ~30 tones | ~70 tones |
| --- | --- | --- |
| one factorization | ~0.3–1 s | ~2–6 s |
| one Newton step | ~1 s | ~3–8 s |
| one power point (10–20 Newton steps) | ~15–60 s | ~2–8 min |
| one 16-point compression curve | ~10–30 min | hours |
| 15-frequency campaign, 3 workers | ~2–4 h | overnight or worse |

Continuation failures widen these substantially: a point that needs adaptive
subdivision can cost several times the nominal. **Budget from the measured
worst point, not the median**, and always give the campaign a per-point deadline
that records a `TIMEOUT` row rather than stalling the matrix.

A campaign that does not fit should be reduced in *frequency count* first, never
in tone count — tone count is physics, frequency count is sampling.

---

## 6. Observables — where correctness is actually won or lost

The solver produces a state `X`. Everything reported to a human is a *derived*
quantity, and in practice derived quantities are where the errors live: the
nonlinear solve either converges or it does not, but an observable can be
confidently wrong forever. Design them defensively.

### 6.1 One reconstruction convention, declared once

If the state is stored as positive-frequency phasors and the real waveform is

```
x(t) = 2 · Re Σ_v X_v e^{iθ_v}
```

then the factor 2 must appear in **every** path from `X` to a physical quantity —
and it must be a single named constant, used everywhere, never a literal. A
missing factor 2 is a factor 4 in power = **6.02 dB**, and it will not present as
a clean 6.02 dB offset if it passes through any *affine* step (a one-port
reflection `s = 2V/Z0 − 1`, for instance, where halving `V` does not halve `s`).
It will present as an arbitrary number and be chased as physics.

Assert it: build a trivial linear circuit with a known analytic response and
check the reconstructed amplitude, in a test, in the suite.

### 6.2 Port waves and gain

Define, at each port and each tone, the incident and reflected waves from the
port voltage *and the actual port current*:

```
a = (V + Z0 I) / (2√Z0),      b = (V − Z0 I) / (2√Z0)
```

**Do not obtain the current by assuming a matched termination** (`I = V/Z0`). That
forces `b ≡ 0` identically and makes any power-wave cross-check vacuous while
appearing to work. Take the current from the branch/source currents actually
present at the node.

**Report two gains, and never let one silently stand in for the other.** They are
different physical quantities:

- **Absolute gain**, `|S21|²` (or `|S11|²` in reflection) with the port
  normalization done properly. This is what a calibrated measurement reports and
  it is the right number for a system budget.
- **Gain enhancement**, the pump-on / pump-off ratio at the same port pair. This
  divides out the passive line's own transmission — insertion loss, impedance
  ripple, coupler split — and is the right number for isolating the parametric
  process.

Two rules follow. First, **whichever is quoted must be named**; on a rippled or
mismatched line they can differ by many dB and the difference is not an error in
either. Second, **do not form a gain by subtracting absolute one-port `S`
values**: a one-port reflection `s = 2V/Z0 − 1` is *affine* in `V`, so scaling
`V` does not scale `s`, and the difference of two absolute values is not a ratio
of two responses. Compute the ratio from the underlying wave amplitudes, not from
the `S` values. Solve the pump-off problem once and keep it.

### 6.3 Conservation checks — and their scope

Two distinct invariants, with very different standing:

**Energy / power balance.** Sum net power over all ports plus dissipation in `G`.
This must close to solver tolerance (~1e-9 relative) at every point. It is
**scope-free** — it does not care which tones you retained — and is therefore the
primary numerical gate.

**Manley–Rowe photon-flux conservation.** For a 3-wave or 4-wave process,
`ΔN_s = ΔN_i` and the pump supplies them. This is a *physical* invariant with a
*scoped* validity:

- Restricted to `{pump, signal, idler}`, it holds up to the power that leaks to
  retained tones outside that scope — typically a few percent. That floor is
  physical, not noise; do not gate tighter than it, and do not chase it.
- Summed over **all** tones it is **not an invariant at all**, because harmonic
  generation destroys three photons and creates one. A solver that reports
  "all-tone Manley–Rowe error = 0.5" is reporting a meaningless quantity, not a
  bug.

State the scope in the field name. `manley_rowe_rel_err` with no scope is a trap.

Guard the evaluability: when the photon-flux scale approaches zero the ratio is
cancellation garbage. Emit the scale alongside the ratio and mark sub-threshold
points *not evaluable* rather than reporting a huge "error".

### 6.4 Multiport devices — where does the pump go?

A two-port amplifier has one obvious place to measure pump depletion. Real
devices often have four ports (separate pump injection, couplers, taps), and the
pump may leave predominantly through a port that is not the signal output.

**Rule: pump depletion must be a sum of net pump power over *all* ports, not a
transmission ratio at one port.** Measuring it at a port that carries 9% of the
pump gives a number that is not wrong so much as meaningless, and it will
quietly invalidate every depletion-based argument built on it. Make the all-port
sum the default and the single-port ratio an explicitly named diagnostic.

Corollary: before running any nonlinear campaign, print where the pump power
goes at the operating point. One line of output; saves months.

### 6.5 Spatial profiles

Per-branch pump / signal / idler amplitude and the local phase mismatch
`Δk = d/dx[2θ_p − θ_s − θ_i]`. These are what distinguish the mechanisms in §1.1:

- monotone signal buildup with a slowly falling pump → traveling-wave, depletion-
  limited;
- signal peaking mid-device with many cells losing power → standing wave;
- `Δk` drifting with signal power before compression → phase-mismatch-limited.

**Caveat with teeth:** the branch index is only a spatial coordinate if the
netlist is built monotonically along the device. For multi-row layouts, couplers,
or devices with node-number gaps, it is not, and a "peak at the midpoint" may be
a row boundary. Emit an explicit branch→position map, or emit nothing.

### 6.6 The linear characterization that must come first

Before any nonlinear result is interpreted, characterize the **pump-off linear**
device:

- `|S21|` on a fine frequency grid (few-MHz steps) — report peak-to-peak ripple
  and its period;
- forward/backward wave decomposition along the line, `|V₋|/|V₊|`;
- the line's characteristic impedance from the *actual* per-cell `L` and shunt
  `C` — being careful to exclude junction capacitance from the shunt term — and
  the port termination values.

A device with `|V₋|/|V₊| ~ 0.3` is a resonator, and a resonator saturates by
flattening its resonance, not by depleting its pump. Discovering that *after*
attributing a compression slope to missing nonlinear physics is the expensive
ordering.

---

## 7. The validation ladder

Ordered by increasing strength. Each level catches things the ones below cannot,
and — importantly — **each level has a stated blind spot**. Passing L0–L5 is not
validation of the physics; it is validation of the implementation.

**L0 — conventions and units.** Inject a known power into a matched linear line;
verify incident and absorbed power. Verify the reconstruction factor. Verify
dBm↔amplitude round-trips. *Blind spot:* everything physical.

**L1 — the pump-only limit.** In the 2-D basis at zero signal, the `q = 0` tones
must equal the 1-D pump solution to ~1e-10 and all `q ≠ 0` tones must be zero to
~1e-12. *Catches:* basis construction, seeding, source placement.

**L2 — the small-signal limit.** As `P_s → 0`, the nonlinear gain must converge to
the independent Floquet small-signal gain, **through identical observable code on
both sides**, to <0.05 dB. *Catches:* the linearization, the observable path.
*Blind spot:* everything about saturation — this test is, by construction,
insensitive to the entire regime of interest. **Run it at an operating point with
real gain (>3 dB), not at a degenerate 0 dB point where it compares 0 to 0.**

**L3 — conservation.** Power balance to solver tolerance at every point;
scoped Manley–Rowe within its floor. *Catches:* assembly and observable errors.
*Blind spot:* basis truncation — a truncated basis conserves energy perfectly
among the tones it kept.

**L4 — truncation convergence, per axis.** Independently vary `H`, `Q`, and
`n_t`; require `|ΔP1dB| < 0.25 dB` between the top two settings on each axis. Do
not vary two at once. Do not accept a study that varies only the axis that is
cheap. *Blind spot:* systematic errors present at every truncation.

**L5 — analytic reference lines.** The depletion model of §2.4. Where does the
measured slope sit relative to −1, and how does the all-port pump depletion at
`P1dB` compare with the 1.302 dB that the depletion-only model would require?
*Catches:* gross normalization errors and, more usefully, **tells you which
saturation regime the device is in** (§2.4 table), at zero cost and with no
reference data. **Highest value-per-hour in the ladder; run it first.**
*Blind spot — and this one is important:* it is **diagnostic, not a gate**.
Neither −1 nor 1.302 dB is a pass/fail threshold; both are properties of one
specific model, and a device limited by another mechanism legitimately misses
both. Do not wire either into an acceptance check.

**L6 — independent oracles.** (a) The coupled-mode model of §3.4, in the regime
where its assumptions hold. (b) A time-domain transient at one operating point.
*Catches:* errors common to the whole HB implementation, which nothing above can
see.

**L7 — measurement.** A hardware dataset with a genuine **signal-power axis**.
Note carefully: a gain-map measurement sweeping pump power and signal frequency
has *no compression information at all*. Check that the axis exists before
planning around a dataset.

### 7.1 Rules for using measurement

- **Reference, never target.** Measurement calibrates confidence; it does not
  tune parameters. A model tuned to match is no longer evidence of anything.
- **Match the operating point or match the gain, and say which.** If the model
  cannot reach the measured pump condition, an equal-gain comparison is legitimate
  but is a different claim.
- **Apply the identical `P1dB` estimator to both**, with smoothing windows scaled
  to each grid (§1c).
- **Report the slope over a stated gain window.** Real `P1dB(G)` relations are
  curved; a single slope quoted without its window is not reproducible.
- **Check the model's reachable gain range covers the comparison window.** If the
  model ceilings at 16 dB and the measured curvature lives above 20 dB, the
  comparison is an extrapolation and must be labelled one.

### 7.2 What another simulator is worth

Agreement with a second simulator is a **regression check**, not validation —
especially if the test designs originate from that simulator's own documentation,
which makes the argument circular. It is genuinely useful for detecting drift
after refactors. It must never be described as physical validation.

---

## 8. Campaign design for gain compression

**Sweep.** Log-spaced signal amplitude, ≥16 points spanning ~5 decades of power,
with the grid chosen so the knee is resolved by ≥4 points. Coarse-grid `P1dB` from
a 5 dB/step sweep carries a systematic error of a few tenths of a dB; refine
inside the bracket with extra nonlinear solves and report **both** the refined and
the interpolated value from the same sweep, so the difference is single-variable.

**Hold fixed.** Pump frequency, pump amplitude, basis, tolerances. One variable at
a time, always.

**Primary deliverable.** `P1dB(ω_s)` and the `P1dB` vs `G_lin` slope with its
standard error and its gain window — not a single `P1dB`.

**Alongside every point, record:** all-port pump depletion; required depletion
`10log10(1 + 2G_lin P_s/P_p)`; conversion fraction; `power_balance_rel_err`;
scoped Manley–Rowe and its photon scale; `Δk` statistics; number of failed points
and the degraded flag; and the full basis and tolerance metadata. The diagnosis in
§1.1 is only possible if these are all present at the same points.

**Diagnostics at `P1dB` must come from a solve at `P1dB`.** If the crossing is
located by refinement, the refined solve is a real state — read depletion,
`Δk`, conversion fraction and balance off *it*. Reporting the nearest coarse
grid point's diagnostics under a `p1db_*` name is a category error: on a 5 dB
grid the nearest point can be halfway to the next decade, and it silently
defeats any comparison against §2.4.

**Stability.** A converged deep-saturation solution is not automatically a stable
one. If stability is claimed, compute it explicitly, quote the dominant exponent
**normalized to `ω_p`** (a bare exponent in s⁻¹ is uninterpretable), and mark
non-converged cases INCONCLUSIVE rather than STABLE.

---

## 9. Software boundaries

Small, with narrow contracts and pure functions where possible:

| module | responsibility |
| --- | --- |
| `circuit` | build/load `(C, G, K, B_J, Ic, φ0, port_map)`; nothing else |
| `basis` | the `(h,q)` tone set, `ω` per tone, conjugate map, the **one** reconstruction constant |
| `nonlin` | branch law `I(Φ)` and analytic tangent `∂I/∂Φ`; static equilibrium solve |
| `aft` | tone ↔ time transforms on the 2-torus |
| `residual` | `R(X)`, the JVP, the assembled Jacobian |
| `precond` | banded and sparse-direct backends; bandwidth measurement |
| `newton` | Newton–Krylov, line search, convergence reporting |
| `continuation` | signal-power walk, adaptive subdivision, gap bookkeeping |
| `pump` | the 1-D pump solve and promotion into the 2-D basis |
| `floquet` | the linear small-signal solver (for L2) |
| `cme` | the envelope oracle (for L6a) |
| `transient` | the time-domain oracle (for L6b) |
| `observables` | port waves, gain, depletion, balance, Manley–Rowe, spatial |
| `estimator` | the `P1dB` extraction rule, shared by model and measurement |
| `campaign` | sweeps, parallelism, resumability, artifacts |

Two contracts worth enforcing in code:

- **`observables` is the only module that converts `X` into anything a human
  reads.** No script computes a gain itself. This is what keeps the factor-2 and
  wrong-port classes of error to one place.
- **`estimator` is applied to measurement and model by the same function.**
  Different call sites are how estimator mismatch enters unnoticed.

Artifacts: one directory per operating point, one row per power in a CSV, one JSON
summary with the full basis/tolerance/version metadata, written **incrementally**
so an overrun leaves partial results rather than nothing.

---

## 10. Failure modes to design against

Each of these is silent — the solver converges, the residual is tiny, the answer
is wrong.

| failure | detected by |
| --- | --- |
| missing reconstruction factor | L0 |
| `q` axis structurally absent from the basis | L4, if `q` is varied separately |
| AFT aliasing | `n_t` convergence |
| gap in the power grid bridged by interpolation | gap bookkeeping in `continuation` |
| depletion read at a minority port | all-port sum; §6.4 |
| gain from absolute one-port `S` | pump-on/pump-off ratio; §6.2 |
| `b ≡ 0` from an assumed matched termination | non-vacuous power-wave check |
| estimator mismatch model vs measurement | shared `estimator` module |
| shared-`G0` noise faking a slope | split-sample construction |
| branch index assumed to be position | explicit branch→position map |
| device is a resonator, not a traveling-wave line | §6.6, before anything nonlinear |
| comparison extrapolated past the model's gain ceiling | report the reachable gain range |

---

## 11. What this design deliberately does not do

- No GPU path. The bottleneck is sparse/banded factorization, which does not
  transfer well to a consumer GPU at these sizes.
- No automatic basis selection. The basis is a physical approximation and is
  chosen by a reported convergence study, not by a heuristic.
- No fitting to measurement, ever.
- No claim of stability without an explicit, normalized computation.
- No single-number `P1dB` as a headline result without its gain and its window.

---

## 12. Build order

1. `circuit`, `basis`, `nonlin`, `aft`, `residual` — plus **L0** and **L1**.
2. `floquet` — plus **L2** at a genuinely amplifying operating point.
3. `cme` oracle — cheap, and it makes every later result interpretable.
4. `newton`, `precond`, `continuation` — plus **L3**.
5. `observables` — including all-port depletion and §6.6 linear characterization.
   **Run §6.6 on the target device before proceeding.**
6. **L5**, the depletion bound. One afternoon. It is the cheapest possible test of
   whether the physics is right, and it is dimensionless.
7. `estimator`, `campaign` — plus **L4** convergence in `H`, `Q`, `n_t`.
8. `transient` oracle — **L6b** at one point.
9. **L7**, measurement — last, and only as a reference.

The ordering is deliberate: every level that can fail cheaply is placed before
every level that costs a machine-day. The single most common expensive mistake is
running L7-scale campaigns before L5 has been checked.
