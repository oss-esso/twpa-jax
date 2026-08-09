# Full-line origin milestone

## H3 provenance

`H3_PROVENANCE = VERIFIED_UNAFFECTED`.

The H3 source report recorded `omega_p=49.6371639267 GHz`, exactly
`2π·7.9 GHz`. H3 calls `scripts.h1_transient_branch_transfer.build_system`,
which computes `2π·freq_ghz·1e9`; it does not import the complexity-ladder
builder. The persisted G1 state was float32-quantized and failed the current
strict residual gate at `2.90e-4`, so it was corrected with the production HB
solver at the same drive. The replacement checkpoint passes with residual
`1.01e-12` and is in `h3_provenance_79/`.

## Topology

The full-length uniform line uses `build_uniform_jtl(2508)`, where
`2508 = IPMParams.num_rows * IPMParams.array_length = 6·418`. It contains one
uniform nonlinear Josephson cell per production JJ, production `Lj`, `Cj`, `Cg`,
50-ohm source/load terminations, and no couplers or second line. It is built by
the same production `add_jtl_element`/`build_matrices` path as the short-line
fixtures. The source and load are ports 1 and 2.

## Completed results

| topology | drive | production starting residual | class | max `r_J` | strobe tail | winding |
|---|---:|---:|---|---:|---:|---:|
| N=64 uniform line | 1.625 Ic | `1.12e-12` | PERIOD_1 | 0.934 | `6.33e-5` | `-2.0e-6` |
| full uniform line, ramp | 1.00 Ic | `2.11e-11` | BROADBAND_OR_CHAOTIC | 0.567 | 0.0727 | `-8.7e-6` |
| full uniform line, ramp | 1.25 Ic | `2.11e-11` | BROADBAND_OR_CHAOTIC | 0.701 | 0.1259 | `-1.1e-5` |
| full uniform line, ramp | 1.50 Ic | `2.11e-11` | BROADBAND_OR_CHAOTIC | 0.830 | 0.1924 | `4.4e-6` |

All completed integrations succeeded; no sustained phase winding was observed.
The 1.5-Ic direct HB target could not be reached by production continuation,
but each transient used a valid 0.5-Ic production HB seed as required.

A slower 1.0-Ic ramp was started as a ramp-rate control and exceeded the 300 s
bounded runtime before producing a completed late-time result. It is therefore
not classified physically. No half-IPM or RCSJ experiment was run.

The full uniform line is evidence that distributed topology can produce strong
non-periodic behavior, but because the completed fast-ramp transition occurs at
much lower local utilization than the full IPM and the slower confirmation did
not complete, the attribution is not yet decisive.

Final status: `NUMERICAL_BLOCKER`.
