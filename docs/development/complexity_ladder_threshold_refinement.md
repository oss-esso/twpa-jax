# Short-array threshold refinement: blocked before RCSJ

## Results at the validated transient step

The N=8 bracket is usable:

| N | last PERIOD_1 | first non-PERIOD_1 | local `r_J` bracket |
|---:|---:|---:|---:|
| 8 | 1.50 Ic, PERIOD_1, `r_J=0.809` | 1.625 Ic, QUASIPERIODIC/PERIOD_N, `r_J=0.880` | 0.809–0.880 |

For N=16 and N=32, the intended bracket cannot yet be reported. At Δθ=0.01,
even low-drive runs do not preserve the converged HB orbit:

| N | drive | class | `r_J` | d₁ tail | phase winding |
|---:|---:|---|---:|---:|---:|
| 16 | 0.25 Ic | QUASIPERIODIC/PERIOD_N diagnostic | 0.287 | ~not settled | ~0 |
| 16 | 1.00 Ic | QUASIPERIODIC/PERIOD_N diagnostic | 0.569 | ~1.6e−3 | ~0 |
| 32 | 0.25 Ic | QUASIPERIODIC/PERIOD_N diagnostic | 0.287 | ~not settled | ~0 |
| 32 | 1.25 Ic | QUASIPERIODIC/PERIOD_N diagnostic | 0.701 | ~4e−4, but d₂/d₃ large | ~0 |

The constant-drive control (`start_fraction=target_fraction=0.25`) produces the
same departure from the HB orbit. Therefore these are not physical threshold
measurements. They are a ladder transient/HB round-trip failure.

## Decision

The N=32 threshold-saturation gate is **not passed**. No RCSJ experiment was
run. This prevents conflating an unvalidated short-array transient with a
physical full-IPM damping result.

The isolated JJ and N=8 results still support an array contribution, but the
N=16/N=32 attribution remains unresolved. The next allowed action is a focused
diagnosis of this constant-drive round-trip mismatch; no full-length ladder or
RCSJ parameter choice is justified yet.
