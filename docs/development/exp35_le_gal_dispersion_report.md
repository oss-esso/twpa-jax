# exp35 Le Gal dispersion validation

## Task 1 — assembled dispersion

The single-cell Bloch eigenvalue was extracted from the interior stamped `C` and
`K` entries of the 700-cell circuit built with published parameters. This is method
B: it directly measures the matrices used by HB and avoids end-reflection phase
contamination.

| candidate | max relative deviation | RMS relative deviation | verdict |
|---|---:|---:|---|
| builder ladder dispersion | 1.094e-15 | 3.823e-16 | matches |
| CME ground-capacitance form | 0.060675 | 0.039734 | wrong for assembled circuit |

The builder form is therefore the circuit dispersion; the CME form had `Cs` added
to ground instead of stamped across each branch.

## Task 2 — mismatch convention

The convention adopted everywhere is `dk_total = 2*k_p - k_s - k_i + dk_nl`;
small `|dk_total|` is the degenerate-4WM gain condition. At 6.0 GHz with a 7.5 GHz
pump, the corrected linear value is `-1168.145 rad/m`; the opposite expression is
`+1168.145 rad/m`.

| file | expression | value at fs=6.0 GHz | action |
|---|---|---:|---|
| `references/.../cme.py` | `ks + ki - 2*kp` | +1168.145 rad/m | changed to `2*kp - ks - ki`; wave number corrected |
| `scripts/reproduce_le_gal_2025_cme.py` | `2*kp - ks - ki` | -1168.145 rad/m | retained; uses measured assembled `L=866.372 pH` |
| `scripts/le_gal_phase_budget.py` | `2*kp - ks - ki` | -1168.145 rad/m | retained; uses measured assembled `L=866.372 pH` |
| `scripts/run_le_gal_2025_hb.py` | `2*phase[pump] - phase[signal] - phase[idler]` | signed spatial diagnostic | retained |

## Task 3 — corrected phase budget

Zero crossings occur at approximately 6.4273 and 8.5727 GHz.

| fs (GHz) | dk_total (rad/m) | |dk_total| L (rad) |
|---:|---:|---:|
| 5.0 | -2680.311 | 16.323 |
| 6.0 | -568.907 | 3.465 |
| 8.0 | 463.372 | 2.822 |
| 8.6 | -30.218 | 0.184 |

The corrected line can phase-match at the published pump, but only in narrow
frequency neighborhoods; 5–8 GHz examples are strongly incoherent.

## Task 4 — corrected HB/CME comparison

| fs (GHz) | HB gain vs off (dB) | CME gain (dB) | absolute difference (dB) |
|---:|---:|---:|---:|
| 6.4 | 0.120502 | 0.232548 | 0.112047 |
| 8.6 | 0.078911 | 0.232563 | 0.153652 |

The old 24.8 dB CME result is not reproduced after making the oracle use the
assembled circuit dispersion and sign convention. No published parameter was
changed and no measurement was used as a tuning target.

## Artifacts

- `experiments/exp35_le_gal_dispersion.py`
- `references/le_gal_2025_gain_compression/exp35_dispersion.csv`
- `references/le_gal_2025_gain_compression/exp35_dispersion.json`
- `references/le_gal_2025_gain_compression/hb_vs_cme_corrected.csv`
