# Solver architecture

How a circuit becomes a gain number. Three stages, each a separate subpackage,
each with a different mathematical character:

```
circuit  ->  pump solve        ->  signal solve       ->  compression
             (nonlinear,           (linear in the         (nonlinear,
              periodic steady      pump's periodic         pump and signal
              state)               background)             solved together)

             twpa_solver.pump      twpa_solver.signal     twpa_solver.multitone
```

The split is not cosmetic. It is why a gain map is cheap and a compression
sweep is not.

---

## Stage 0: the circuit

`twpa_solver.core.CircuitMatrices` holds one equation:

```
C xddot + G xdot + K x + Bphi i_J(Bphi.T x) = i_src
```

`x` is the vector of non-ground node fluxes. `C`, `G`, `K` are node-admittance
stamps. `Bphi` is the node-to-branch incidence for the nonlinear branches only,
and `Bphi.T x` is therefore the vector of branch fluxes handed to the branch
law.

`i_J` is the branch law. The default is the Josephson relation
`Ic sin(phi/phi0)`. `EffectiveSnailBranchLaw` in `core/nonlinear.py` implements
the SNAIL cell, shifted to its solved static equilibrium.

**Josephson inductances never enter `K`.** They are nonlinear branches with
`Ic = phi0_reduced / Lj`. Only their shunt `Cj` is stamped, into `C`. Getting
this wrong by also stamping the small-signal slope into `K` double-counts the
stiffness — that was a real bug in the SNAIL builder and it destroyed the
model's gain.

`load_circuit` / `save_circuit` move these to and from a directory. Builders
are described in [`circuit_builders.md`](circuit_builders.md).

---

## Stage 1: the pump solve (`twpa_solver.pump`)

Find the periodic steady state of the circuit driven by a strong pump at
`omega_p`. This is the only genuinely nonlinear step in a gain map.

### Representation

The unknown is a set of complex phasors `X_k`, one per retained pump mode, and
the real waveform is reconstructed as

```
psi(t) = 2 * Re sum_k X_k exp(+i k omega_p t)
```

The factor of 2 and the `+i` sign are the positive-phasor convention. They
propagate everywhere — into the source scaling, into the current conversion, and
into how a pump solution from another code must be interpreted.

`HarmonicGrid` (`pump/hb.py`) carries the mode list, the pump frequency, and
`nt`, the number of time samples used for the alternating-frequency-time
evaluation of the nonlinearity. `nt` must be at least `2*max(mode)+1`.

### Mode basis

`pump/basis.py` is the single source of truth. `resolve_pump_basis` takes a
policy:

| Policy | Modes | Use for |
| --- | --- | --- |
| `positive_odd_jc` | `[1,3,...,2K-1]` | unbiased 4-wave-mixing (JPA, JTWPA, FQJTWPA) |
| `dense_real` | `[1,2,...,H]` | biased / DC / 3-wave-mixing devices |
| `positive_phasor_explicit` | user list | anything else |
| `auto_jc` | inferred | single-pump cases |

The odd list is not an optimization. An unbiased 4WM device has no even pump
content, so a dense basis spends modes on zeros while truncating the high odd
harmonics that carry real amplitude.

`PumpBasis.to_metadata` writes the basis, the convention, and the
reconstruction factor into the solution, and
`load_pump_basis_from_solution` reads them back — so a gain solve can never
silently disagree with the pump solve about what the phasors mean.

### Solving

`HarmonicNewtonKrylovSolver` (`pump/solver.py`) is Newton–Krylov: GMRES on the
Jacobian-vector product, no dense Jacobian. Entry points:

| Method | Behavior |
| --- | --- |
| `solve_one` | one Newton solve at a given source scale |
| `solve_continuation` | fixed ladder in source scale, `lambda_start` resumable |
| `solve_adaptive_continuation` | adaptive lambda stepping with bisection on failure |
| `solve_arclength` | pseudo-arclength, for turning points |
| `solve_pseudo_transient` | damped pseudo-time, for very stiff starts |

Continuation exists because a cold Newton solve at full pump power usually
diverges. The solver walks the source amplitude up from zero, warm-starting
each step.

`solve_arclength` normalizes its tangent with a state-scale-derived metric.
An unscaled Euclidean metric mixing node flux (order `1e-13` Wb on a real
device) with the dimensionless source scale makes the state's contribution
negligible, which silently degrades the method to natural-parameter
continuation and makes fold detection structurally impossible. Pass
`rescale_every` on a stiff device so the metric tracks the branch.

`pump/singularity.py` holds the fold instrumentation that reads out of a
continuation run: `jacobian_min_eigenvalue` (shift-invert Arnoldi around the
exact real-packed Jacobian), `jacobian_det_signature`, and
`bordered_conditioning`, which discriminates a fold from a branch point —
a fold leaves the bordered system well conditioned even though `J` is
singular, a rank-2 branch point does not. `solve_arclength`'s `on_step` hook
feeds all three at every accepted step; a single endpoint measurement cannot
detect a fold and must not be used.

Related modules not covered here: `pump/bifurcation.py`,
`pump/neimark_sacker.py`, and `pump/periodic_branch.py` carry the
bifurcation-tracking and period-N scaffolding, and `stability/` holds the
time-domain monodromy route. `core/rcsj.py` and `core/kinetic.py` hold
alternative branch physics.

### Preconditioners

Set through `NewtonKrylovSettings.preconditioner`. The block-diagonal modes are
built in `pump/problem.py::build_preconditioner_factors`; the coupled ones are
assembled there too and driven from `pump/solver.py`:

| Name | Structure | Cost |
| --- | --- | --- |
| `linear`, `none` | trivial | cheapest, weakest |
| `mean_tangent` | block-diagonal in mode | default |
| `spectral_coupled` | mode-coupled `(k-q)` complex Jacobian, one LU | strong |
| `real_coupled` | exact real-packed Jacobian including the conjugate `(k+q)` term | GMRES converges in ~1 iteration |

`real_coupled` is the right choice for stiff DC and mutual-inductor designs and
is what the in-process gain-map engine uses. The backend under
`pump/backends/fast_coupled.py` caches the scatter map and symbolic
factorization and redoes only the numeric factor per Newton step.

Factorization goes through PARDISO when `pypardiso` is installed, SuperLU
otherwise. PARDISO is pinned to one thread by default
(`TWPA_PARDISO_THREADS`) because MKL intermittently fails reordering at higher
counts on some AMD parts; with several workers, extra workers beat extra
threads anyway.

---

## Stage 2: the signal solve (`twpa_solver.signal`)

With the pump fixed, a small signal sees a **linear, time-periodic** circuit.
That is the whole reason gain maps are affordable.

### The Floquet picture

The pumped junction has a time-varying inverse inductance

```
gamma(t) = cos(psi_p(t)/phi0) * Ic/phi0
```

Its Fourier coefficients `gamma_hat[l]` (`signal/gamma.py`) couple a signal at
`omega_s` to sidebands at `omega_s + l*omega_p`. Truncating `l` to `+-S`
sidebands gives a finite linear system, solved once per frequency by
`signal/floquet.py::solve_gain_one`.

For a correct real pump, `gamma_hat[-l] == conj(gamma_hat[l])` exactly. The
`gamma_hat_summary.csv` diagnostic reports `conj_symmetry_rel_err` per
sideband; anything nonzero means the pump waveform is wrong, not that the gain
is slightly off.

### Backends

`--signal-backend direct` factors the full sideband system. `schur` eliminates
the interior and keeps the retained ports, which is faster per solve but
densifies — on large maps it can exhaust memory, so `direct` is the safe
default for campaigns.

### Passive response

`signal/passive.py::passive_s_matrix` is the same machinery with the pump off.
It still carries the Josephson inductance through `gamma_hat[0] = Ic/phi0`, so
it is a genuine linearized S-parameter, not a lumped approximation. Convention
is `S[frequency, output_port, source_port]`.

### Quantum efficiency

`signal/quantum_efficiency.py` ports `calcqe` / `calcqeideal`. These expect `S`
in the **photon ladder-operator basis**, not the classical voltage-ratio `S`
that `solve_gain_one` returns. Signal and idler sit at different frequencies,
so converting needs the Manley–Rowe reweighting

```
S_ladder[m,n] = S_classical[m,n] * sqrt(freq[n]/freq[m])
```

A 2x2 `[signal, idler]` truncation will not satisfy unitarity on a device that
genuinely uses ten sidebands. That is truncation, not a bug.

---

## Stage 3: compression (`twpa_solver.multitone`)

At large signal power the sidebands act back on the pump. The linear-in-signal
assumption fails, and pump and signal must be solved together.

### Why it costs so much more

The multitone problem carries every retained tone as an unknown, and the
coupled Jacobian is **block-dense in tone index**. Memory scales as
`(n_pump_modes + 2S + 1)^2`, not as the packed dimension. On the same circuit,
going from a gain-map pump basis (H=10) to a multitone basis at S=10 (H=31)
took the Jacobian from 2.5M to 23.6M nonzeros — the `H^2` ratio — while the
dimension grew only 3.1x.

This is exactly the cost the Floquet split avoids in stage 2, where sidebands
are a separate linear factor-once system.

### Basis

`multitone/basis.py` builds the tone set:

- `build_sideband_matched_basis` retains the pump harmonics alongside the
  signal sidebands. This is the production basis.
- `build_lattice_basis` is used by the convergence study.
- a three-tone basis is valid only with a fundamental-only pump.

### Backends and preconditioners

`multitone/problem.py` holds the full backend; `multitone/schur.py` the
Schur-reduced one, cached on the partition. Both route the exact coupled
preconditioner through `pump/backends/fast_coupled.py`, which caches the
symbolic factorization across Newton steps.

`--factor-backend banded` reorders the coupled Jacobian **node-major** so the
factors fit a LAPACK general band. That works because the device is a 1-D
chain: packed tone-major the matrix spans everything, node-major it collapses
to a band a few tone-blocks wide. Bandwidth is measured from the assembled
pattern, never assumed. It trades ~18% more wall time for ~27% less memory,
which is only worth it when the smaller footprint buys another worker.

Modified-Newton preconditioner reuse (`--precond-reuse N`) is available and
**measured to be a net loss** at N>1: the exact preconditioner converges GMRES
in about three iterations, and one GMRES iteration costs about as much as one
factorization, so there is no cheap-preconditioner regime to amortize.

### Observables

`multitone/observables.py` is the readout contract.

- Gain must be read through `tone_s21` and reported as the pump-on/pump-off
  ratio. Reconstructing it as `|i omega X| / V_in` is biased — that path was
  off by 12.041 dB on the SNAIL benchmark.
- `power_balance` takes `z0_ohm`.
- Manley–Rowe comes in two scopes. `conversion_manley_rowe_*` is restricted to
  pump/signal/idler and is a valid invariant with a real floor of a couple of
  percent. `all_tone_manley_rowe_*` reports the retained-tone scope and is
  **not** a valid invariant — never gate on it.

### Stability

`multitone/stability.py` linearizes about the converged large-signal state and
looks for growing Floquet exponents. It is opt-in (`--check-stability`);
without it `stability_status` is `NOT_CHECKED`.

Two failure modes it guards against, both of which were once live bugs: passing
pump-harmonic keys where a sideband ladder is expected, and letting a near-DC
sideband own `sigma_min`, which made the answer identical with the pump on and
off. Near-DC cases now return `INCONCLUSIVE` with a reason rather than a
confident `STABLE`.

Always quote an exponent against `omega_p`. The same numerical value can be
real damping on one device and numerically marginal on another.

### Pump-orbit bifurcation diagnostics

`signal/stability.py::classify_floquet_resonance` converts a refined Hill root
`omega` into the one-pump-period multiplier
`mu = exp(+i*omega*2*pi/omega_p)`. It labels roots near `+1` as fold
candidates, roots near `-1` as period-doubling candidates, and other
near-unit-circle roots as Neimark--Sacker candidates. These labels are
diagnostic only: the Hill truncation and nonlinear branch must be validated
independently.

The explicit CLI path is
`scripts/floquet_stability_sweep.py --refine-bifurcations`, which checks the
requested fractions of the pump frequency (default `0.0,0.5`). A confirmed
`-1` candidate can be represented with
`pump/floquet.py::period_doubled_basis`: the fundamental becomes
`omega_p/2`, the physical pump is mode two, and odd half-pump modes are
retained. `build_period_doubled_seed` maps the refined Hill eigenvector into
that basis. The seed is never accepted directly; it must converge under the
production HB residual and full-residual/provenance gates.

---

## Where to make a change

| Change | Place |
| --- | --- |
| New device topology | `builders/` — emit `CircuitMatrices` or an artifact directory |
| New branch physics | `core/nonlinear.py` — implement the branch-law protocol |
| Pump convergence trouble | `pump/solver.py` continuation, `pump/predictors.py` |
| Faster linear algebra | `pump/backends/` |
| New readout quantity | `signal/` for small-signal, `multitone/observables.py` for large-signal |
| New campaign | `scripts/`, composing the above — not by copying solver internals |

A backend change belongs under `src/twpa_solver/`, not copied into a workflow
script. The workflows are thin on purpose.

---

## Related documents

- [`circuit_builders.md`](circuit_builders.md) — every builder in detail
- [`component_profiles_and_scatter.md`](component_profiles_and_scatter.md) — per-cell parameter control
- [`workflows.md`](workflows.md) — the end-to-end entry points
- [`design_format.md`](design_format.md) — the declarative YAML adapter
- [`development/circuit_api.md`](development/circuit_api.md) — the Python `Circuit` authoring API
- [`development/pump_current_conversions.tex`](development/pump_current_conversions.tex) — the two pump-current conventions
