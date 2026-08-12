D:\Projects\Thesis\twpa_jax\outputs\jc_doc_python_designs\jc_jtwpa

# RCSJ junction damping on the JTWPA high-power wall

Date: 2026-08-12

This is a numerical-regularization study of `jc_jtwpa`.  A finite RCSJ
resistance is not a physical property of this aluminium junction model.  The
physical limit is `R -> infinity`; finite-R thresholds and gains are reported
only as dependence on the regularizer.

## Implementation and gates

`twpa_solver.core.stamp_rcsj_shunt` stamps

```text
G_rcsj = Bphi @ diag(1 / R_j) @ Bphi.T
```

with `Ic * Rn = pi * Delta / (2e)`, `Delta = 180 ueV`, and
`R_j = (R/Rn) * Rn`.  `stamp_rcsj_shunt(..., inf)` returns the original
`CircuitMatrices` object without arithmetic.  The finite-R stamp is real,
symmetric, and positive semidefinite.

The focused gate result is `23 passed`, including:

- exact `R/Rn = inf` matrix no-op;
- `has_loss == False` and `default_loss_model_for(...) == current_complex_c`
  for the control and finite-R circuits;
- symmetric positive-semidefinite shunt stamp;
- Ambegaokar--Baratoff and damping scaling checks;
- transient output-voltage nonzero guard and `max_abs_phi > 5 rad` BLOWUP
  handling in the H1 path.

The explicit control column rerun reached the same 72 PASS rows and the same
last-PASS power as the stored ultrafine control.  Its solver diagnostics differ
in the final floating-point digits because it is a separate iterative solve;
the strict implementation gate is the exact matrix/object no-op above, not a
claim that two independent sparse/Newton executions are byte-identical.

## RCSJ ladder

The JTWPA has `Ic = 3.4 uA` and `Cj = 55 fF`.  Values below are medians over
the 2,047 junctions; this device is uniform, so the medians are also the
per-junction values.

| `R/Rn` | `Rn` [ohm] | `Rj` [ohm] | `beta_c` | `Qj` | damping / pump period |
|---:|---:|---:|---:|---:|---:|
| `inf` | 83.1598 | `inf` | `inf` | `inf` | 0 |
| `1e6` | 83.1598 | 8.31598e7 | 3.92946e12 | 1.98229e6 | 3.07075e-5 |
| `1e4` | 83.1598 | 8.31598e5 | 3.92946e8 | 1.98229e4 | 3.07075e-3 |
| `1e2` | 83.1598 | 8.31598e3 | 3.92946e4 | 198.229 | 0.307075 |
| `1` | 83.1598 | 83.1598 | 3.92946 | 1.98229 | 30.7075 |

The damping number is `T_p / (R Cj)` at 7.12 GHz.  The generated ladder is
also available in [rcsj_ladder.csv](../../outputs/rcsj_jtwpa_campaign/rcsj_ladder.csv)
and the per-setting matrices and metadata are under
`outputs/rcsj_jtwpa_campaign/variants/`.

## HB wall

The 0.101 dB column used 120 points from -36 to -24 dBm, 10 positive odd
pump modes, and no PERIOD2 or period-N ansatz.  These are last-PASS values;
`R/Rn = 1` is censored by the requested -24 dBm upper limit rather than a
failure.

| `R/Rn` | last PASS [dBm] | interpretation |
|---:|---:|---|
| `inf` | -28.840336 | control |
| `1e6` | -28.840336 | no resolved shift at this grid |
| `1e4` | -28.840336 | no resolved shift at this grid |
| `1e2` | -27.428571 | finite-R numerical wall moved upward |
| `1` | >= -24.000000 | no failure within the campaign ceiling |

The finite-R wall therefore moves in the predicted direction.  This is a
statement about numerical continuation under the RCSJ ladder, not a JTWPA
boundary at finite resistance.

## Transient and shunt-power result

The requested BDF runs were attempted from the highest checkpoint that passed
the production full-residual gate.  The raw last-PASS 10-mode HB point is not
always a valid transient fixture: its omitted-harmonic residual reaches about
`1e-4`.  The highest full-residual-gated checkpoints were approximately
`-34.6891 dBm` for `1e6` and `1e4`, `-34.2857 dBm` for `1e2`, and `-28.6387 dBm`
for `1`.

No 200-period BDF run completed under the available resource contention, and
the 800-period run was stopped before producing a summary.  Consequently the
following required measured fields are **not answered** in this run:

- `max_abs_phi` envelope slope and slope divided by `max_abs_sin_phi`;
- `min_cos_phi`, `mean_phase_winding_cycles`, `state_norm`, and strongest branch;
- `steps`, Newton iterations, step reductions, and final BLOWUP status;
- shunt-dissipated power and its pump-power fraction.

The 200-period attempt was made at `R/Rn = 1e4` with BDF, 40 ramp periods,
32 samples per period, and `max_step = 2*pi/32`.  It reached approximately
1.5 GB resident memory without writing a summary.  A replacement segmented
run used one-period segments and atomically published restart, state, and
observable progress artifacts after each completed segment; it remained in
the first stiff period at approximately 1.1 GB and was stopped before an
artifact was produced.  These are resource/convergence blockers, not physical
failure rows.  The code records `shunt_power_w` and its drive-referenced
fraction when a run completes, writes compact transient observables atomically,
and rejects an identically-zero output voltage trace.  No finite-R transient
value is quoted as a physical result.

## Tier-2 and monodromy

Tier-2's complex-omega route survives the RCSJ implementation: `C` remains
real, `has_loss` remains false, and the default analytic model remains
`current_complex_c`.  The time-domain tangent state is also real in the
finite-R circuit.

A finite-R monodromy attempt was launched for `R/Rn = 1e4` at the validated
checkpoint with 256 steps per period, 40 requested eigenvalues, and `ncv=120`.
It produced no `floquet_results.json` before the bounded attempt was stopped
under CPU and memory contention.  Therefore monodromy convergence is
**unresolved**, not evidence of a physical instability or stability result;
the route was not declared to have survived operationally.

No 2c circuit or `designs/ipm_2c_fixed` file was modified.

The optional `jc_fqjtwpa` extension was not implemented or measured in this
run.  Its design directory is therefore an unanswered scope item.
