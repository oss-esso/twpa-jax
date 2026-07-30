# Le Gal Tasks B–D reconciliation

The Task B ridge used 700 cells, S=5, and pump modes `[1, 3, 5]`. The measured
row maxima were:

| Pump dBm | Measured peak dB | Frequency GHz |
|---:|---:|---:|
| -78.4 | 0.8405 | 8.9 |
| -74.0 | 2.1876 | 8.2 |
| -71.0 | 4.0630 | 7.8 |
| -69.3 | 5.9373 | 8.9 |
| -66.0 | 1.2216 | 8.9 |

The requested pump-ceiling search found `-64.0 dBm` converged with residual
`1.2180e-20`. The first failure was `-63.0 dBm` at the continuation endpoint,
with residual `7.938e-2` and a stalled Newton reduction. The diagnostic also
recorded failures at `-62.0` through `-58.4 dBm`.

The dBm-to-current path is `I = sqrt(2 P / Z0)`. The corrected CME input flux
at `-78.4 dBm` is `4.505618e-16 Wb`, compared with the HB output reference
`4.5057e-16 Wb`, ratio `0.9982`. This rules out a missing factor in the local
power-to-node-flux conversion.

The frozen benchmark contract labels power as dBm at the device input and
does not apply a generator attenuation. No generator-versus-device offset is
therefore supported by the local contract. The remaining discrepancy is a
model discrepancy: the reduced single-branch model reproduces the measured
small-pump phase but not the paper-scale gain morphology at the published
`-78.4 dBm`. The benchmark configuration remains `-78.4 dBm`; diagnostic
higher pump powers are not adopted.

The explicit one-cell three-large/one-small comparison at half flux gives
`g1 = +0.271333333 Ic`, `g3 = +0.004160494 Ic`, and
`g3/g1 = +0.015333515`. The ratio to the reduced value is `1.00000002`, so
the effective single-branch reduction is not the source of an 8x or 22x
nonlinearity error. The earlier attribution to that reduction is superseded;
the remaining open defect is the linear/reference-path zigzag identified in
Task 1.
