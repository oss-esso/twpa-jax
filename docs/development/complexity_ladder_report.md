# Controlled circuit-complexity ladder: first results

This is the first lossless ladder tranche.  It deliberately stops before the
full-length/coupler ablations and before RCSJ, because the short-array rung
already changes the observed mechanism relative to a single junction.

## Topology and parameters

All values are taken from `IPMParams` in the production builder:

| quantity | value | provenance |
|---|---:|---|
| `Lj` | 123.9 pH | production |
| `Cj` | 145 fF | production |
| `Cg` | 66 fF | production |
| `Ic` | 2.65622 µA | derived `phi0/Lj` |
| terminations | 50 Ω | production |
| `Rj` | infinity | deliberate lossless ladder setting |

Rung 0 uses a fixed linear `Lj` only as a consistency fixture. Rung 1 and the
JTL use the actual `Ic*sin(phi)` branch law and stamp `Cj`; no fixed `Lj` is
substituted for the nonlinear junction.

The implementation is in `src/twpa_solver/builders/complexity_ladder.py`, and
the experiment runner is `scripts/run_complexity_ladder.py`.

## Low-drive consistency

The linear fixture stiffness matrix equals the zero-phase tangent of the single
nonlinear JJ to numerical precision:

```text
K_linear = Bphi diag(Ic/phi0) Bphiᵀ
```

The builder/save/load tests pass for Rung 0, Rung 1, and an N=8 JTL.

## First high-drive results

| topology | drive | state | max `|sin(phi)|` | `phi_max` | phase winding |
|---|---:|---|---:|---:|---:|
| single JJ | 1 Ic | PERIOD_1 | 0.533 | 0.562 | −3.7e−6 cycles |
| single JJ | 2 Ic | PERIOD_1 | ~1.000 | 1.941 | −3.3e−3 cycles |
| single JJ | 3 Ic | RUNNING_PHASE | ~1.000 | 34.76 | +0.493 cycles |
| uniform JTL N=8 | 1.5 Ic | PERIOD_1 | 0.812 | 0.948 | −6.6e−5 cycles |
| uniform JTL N=8 | 2 Ic | RUNNING_PHASE | ~1.000 | 26.92 | −0.490 cycles |
| uniform JTL N=16 | 1.5 Ic | QUASIPERIODIC_OR_PERIOD_N | 0.865 | 1.044 | −4.2e−4 cycles |

The transient integrator succeeded for every listed run. PERIOD_1 cases with a
successful projection also restarted the HB solver successfully.

## Interpretation so far

The isolated JJ does reproduce a high-drive transition, but only near local
utilization one: it remains periodic at `r_J≈0.95` and runs at `r_J≈0.98–1`.
The full IPM loses PERIOD_1 around `r_J≈0.87–0.89`.

Adding only a short distributed JJ line moves the transition earlier and makes
the behavior length-dependent: N=8 is periodic at 1.5 Ic but running at 2 Ic,
while N=16 is already non-periodic at 1.5 Ic. The current evidence therefore
supports `MIXED_MECHANISM`, with a clear collective/distributed contribution;
it does not yet distinguish finite-length propagation from coupler embedding.

The next justified rung is the N=8/16/32/64 controlled length comparison,
followed by one full-length uniform line. Rung 4 and RCSJ remain intentionally
deferred.

## Literature motivation

- Single-junction RF bifurcation and embedding: Manucharyan et al.,
  [RF bifurcation of a Josephson junction](https://arxiv.org/abs/cond-mat/0612576).
- Discrete JTL topology and ground-capacitance models: Kogan,
  [Josephson Transmission Line Revisited](https://onlinelibrary.wiley.com/doi/full/10.1002/pssb.202200475).
- Distributed JTWPA nonperiodicity/chaos diagnostics: Guarcello et al.,
  [Driving a Josephson Traveling Wave Parametric Amplifier into chaos](https://arxiv.org/abs/2406.01185).
