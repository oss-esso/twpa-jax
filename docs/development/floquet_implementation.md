# PERIOD1 HB Floquet stability layer

The production implementation is under `src/twpa_solver/stability/` and the
experiment wrapper is `scripts/run_floquet_2c.py`.

## Mathematical formulation

For a validated pump-periodic HB waveform `q*(theta)`, with
`theta = omega_p t`, the tangent equation is

```text
C omega_p^2 d2(delta q)/dtheta2
 + G omega_p d(delta q)/dtheta
 + [K + Bphi diag(I'(phi_J*(theta))) Bphi^T] delta q = 0.
```

The exact branch-law tangent is evaluated at the reconstructed HB waveform.
No effective-inductance approximation is used.

For one implicit-trapezoid step of size `h`, the tangent update is obtained
from the same endpoint Jacobian as the nonlinear transient step. Eliminating
the kinematic equation gives

```text
S dq_(n+1) = Rq dq_n + Rp dp_n
dp_(n+1) = 2/h (dq_(n+1) - dq_n) - dp_n,
```

where `S`, `Rq`, and `Rp` are sparse matrices assembled from the endpoint
Josephson tangent stiffness and the production `C`, `G`, and `K` matrices.
One sparse factorization of `S` is cached per phase step and reused for every
Arnoldi application.

## DAE state space

The multipliers are computed on the physically consistent reduced state

```text
(delta q_d, delta p_d, delta q_a),
```

where `d` denotes differential nodes and `a` denotes algebraic nodes. The
algebraic flux perturbation is projected with

```text
[K_J]_(aa) delta q_a = -[K_J]_(ad) delta q_d.
```

The algebraic velocity perturbation is reconstructed from the differentiated
algebraic constraint using the existing `G_aa` sparse factor. Consequently,
Arnoldi never injects arbitrary perturbations into the algebraic node.

The current 2c fixture has one algebraic node and a factorable `G_aa` block.
The legacy full-state fallback used when `G_aa` is not factorable is rejected
explicitly by the Floquet builder instead of being silently approximated.

## Numerical policy

`scipy.sparse.linalg.eigs` receives a `LinearOperator`; the monodromy matrix is
never assembled or stored. The default target is dominant magnitude (`which=
LM`) because multipliers near the unit circle determine stability. Arnoldi
iteration count, convergence status, requested eigenpair count, and matvec
count are persisted in the scan JSON. Partial or failed convergence is marked
unresolved.

Floquet exponents use the principal logarithm,
`mu = log(lambda) / T_p`, with the documented positive-exponent phase
convention. A single multiplier is never treated as proof of a bifurcation;
the scan tracks branches by complex-plane proximity over pump power.

## Validation and branch gates

The unit tests in `tests/test_periodic_orbit_floquet.py` cover:

- analytic damped linear oscillator monodromy action and multipliers;
- finite-difference monodromy action parity;
- DAE tangent constraint consistency;
- timestep convergence;
- complex multiplier branch tracking and conservative crossing labels.

The 2c CLI also performs a one-period nonlinear HB-to-TD closure check unless
`--skip-closure` is supplied. Multipliers are scientifically interpretable
only when that check is successful and the result is converged and timestep
stable.

No PERIOD2, period-N, or quasiperiodic HB ansatz is implemented. A new ansatz
is gated on a continuous, timestep-converged Floquet crossing plus the
corresponding independent TD evidence.
