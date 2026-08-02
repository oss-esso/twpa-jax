# exp35 Le Gal dispersion validation

The Bloch eigenvalue is extracted from the assembled residual linearization:
`K` plus the effective-SNAIL branch tangent. The builder ladder relation agrees
with it; the older CME ground-capacitance relation does not.

| candidate | max relative deviation | RMS relative deviation |
| --- | ---: | ---: |
| builder ladder | 1.094e-15 | 3.823e-16 |
| old CME ground-capacitance form | 0.060675 | 0.039734 |

The phase-budget and HB/CME sections formerly in this report are superseded by
`docs/development/exp38_le_gal_kerr_verdict.md`. exp38 found that the Le Gal
builder had also stamped the SNAIL tangent into `K`, double-counting the branch
stiffness in HB; therefore the old exp35 HB gain comparison is historical.
