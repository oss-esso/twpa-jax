# Conversion S-Parameter Validation

Validation artifact:

```text
D:\Projects\Thesis\outputs\new_twpa_solver\conversion_validation_20260613_080228
```

Rows are in `conversion_validation_rows.csv`.

| Validation | result | tolerance | artifact |
|---|---|---:|---|
| zero-pump conversion admittance is sideband block diagonal | PASS | `1e-18` | `conversion_validation_rows.csv` |
| zero-pump signal S21 equals ordinary linear S21 | PASS | `1e-8` | `conversion_validation_rows.csv` |
| small-pump idler conversion tends to zero | PASS | ratio `< 1e-3` | `conversion_validation_rows.csv` |
| Josephson derivative matches finite difference | PASS | relative error `< 1e-5` | `conversion_validation_rows.csv` |
| diagnostic pump status masks gain row | PASS | gains are NaN | `conversion_validation_rows.csv` |

The zero-pump S21 test is the key correctness check: it verifies that the
conversion-matrix path reduces to the same unpumped linear S-parameter solver
when the pump state is zero.

The small-pump idler check uses the even Josephson derivative response on the
`m=+2` sideband. This is appropriate for the current Josephson-only reduced
model. RF-SQUID three-wave mixing will need a separate sideband convention once
that nonlinearity is implemented.
