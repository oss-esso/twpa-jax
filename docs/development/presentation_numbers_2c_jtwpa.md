# Presentation numbers — 2c and JTWPA

Figures in `outputs/presentation/`. Every number below is reproducible from the
artifact named beside it.

Two devices, two kinds of evidence. **2c** is a fabricated chip with a measured
Themis dataset, so it carries a model-versus-measurement comparison. **JTWPA**
is a JosephsonCircuits.jl documentation design with no measurement, so it
carries internal physics instead.

---

## 1. 2c — model versus measurement

Circuit `designs/ipm_2c_fixed`: 6136 nodes, 2508 branches, ports
`{1:0, 2:4576, 3:4577, 4:6135}`.

### Operating point (`outputs/exp31_pump_freq_scan_2cfixed`)

The model does not amplify at the measured pump setting, so the operating point
was located rather than assumed: scan pump frequency and current, and match the
measured small-signal gain at three frequencies **simultaneously**.

| | model | measured |
| --- | ---: | ---: |
| pump frequency | 7.100 GHz | 7.256 GHz |
| pump power on chip | −58.84 dBm | −66.7 dBm |

Best match **fp = 7.100 GHz, Ip = 7.2311 µA, rms 1.247 dB**, confirmed at
0.02 GHz resolution. The pump frequency lands **0.156 GHz** from the measured
one, at the production current.

> **Do not present the pump-power gap.** The often-quoted "~7.9 dB" is not
> established. Two separate conventions are unresolved:
>
> 1. **Norton factor.** All four ports carry a 50 ohm shunt, so the drive is a
>    Norton source and the available power is `I²Z₀/8`, not the `I²Z₀/2` the
>    conversion uses. The solver's own observable agrees:
>    `pump_outgoing_power_w` = 3.26795e-10 W = **−64.86 dBm**, exactly `I²Z₀/8`
>    and **6.02 dB** below the quoted −58.84 dBm. Against the depletion-inferred
>    −65.98 dBm the like-for-like gap would be **~1.1 dB**.
> 2. **Line loss.** The Themis pump-line figure (45.7 dB) is undocumented and
>    sits 11 dB from the repo's own measured `loss_A10` model (34.54 dB at
>    7.1 GHz). It does not enter the key result — the depletion inversion
>    derives the on-chip pump from the compression data itself — but it should
>    not be quoted as a measurement.
>
> The measured on-chip pump of **−65.98 dBm** (depletion inversion) is solid and
> depends only on the 72.5 dB signal-line loss.

### Gain band — `2c_gain_band.png` (`outputs/exp34_gain_band`)

81-point linear Floquet sweep against the measured G0(f).

| | model | measured |
| --- | ---: | ---: |
| peak gain | 12.84 dB | 15.76 dB |
| peak frequency | 7.062 GHz | 7.080 GHz |
| **bandwidth > 3 dB** | **4.375 GHz** | **4.268 GHz** |

Same width, same centre. Systematic envelope deficit **0.92 ± 0.61 dB**.

**Ripple dominates any few-frequency comparison.** Model 3.8 dB peak-to-peak,
measured 3.2 dB, so adjacent frequencies flip the sign of the comparison:
6.500 GHz has the model +0.735 dB **above** measurement, 6.562 GHz has it
−2.423 dB **below**. A 22 MHz grid offset moves model gain 0.5 dB. Always use
the ripple-averaged envelope.

### Compression — `2c_compression_vs_themis.png`

Seven frequencies at the matched operating point, S=10, against the Themis
105C5 cuts. Long sweeps at 5.296 / 6.540 / 7.052 GHz (exp32), short probes at
6.300 / 6.440 / 6.640 / 6.800 GHz (exp33), and the exp33 deep run that carried
6.540 GHz past the continuation wall to −85 dBm. Solid = measurement,
dashed with open circles = model, one colour per frequency.

| fs GHz | model G0 | meas G0 | model P1dB | meas P1dB | ΔP1dB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5.296 | 6.16 | 7.18 | −83.18 | −84.33 | +1.15 |
| 6.300 | 9.64 | 12.26 | −85.57 | −88.94 | +3.37 |
| 6.440 | 10.03 | 11.79 | −85.43 | −87.19 | +1.76 |
| 6.540 | 10.29 | 12.18 | −86.74 | −89.79 | +3.05 |
| 6.800 | 9.83 | 11.62 | −84.53 | −86.84 | +2.31 |
| 7.052 | 13.43 | 13.15 | −94.21 | −99.72 | +5.51 |

**Model compresses late at every frequency, mean +2.86 dB.** Curve *shape*
agrees well — at 7.052 GHz model and measurement overlay from −90 to −78 dBm,
and both collapse at ≈ −77 dBm regardless of frequency.

Two caveats to state if asked:
- The **spread** (+1.15 to +5.51) is not trustworthy. Ripple moves local gain
  ±2 dB and gain drives local P1dB, so only the mean survives.
- A real chip has fabrication defects an ideal model lacks (loss, junction
  disorder, impedance ripple), and every one of them makes hardware saturate
  **earlier**. The model being late is the expected direction.

---

## 2. JTWPA — internal physics

`outputs/exp20_multitone_compression_converged/jtwpa/s10`, pump 7.12 GHz at
3.7 µA, signal 6.6 GHz, S=10.

### Compression — `jtwpa_compression.png`

| | value |
| --- | ---: |
| small-signal gain | 27.543 dB |
| **P1dB** | **−111.458 dBm** |
| pump depletion at P1dB | 0.263 dB |
| max pump depletion | 2.125 dB |

Gain roll-off and pump depletion bend together — that is what distinguishes
real saturation from an amplitude limiter.

### P1dB across frequency — `jtwpa_p1db_vs_frequency.png`

8 frequencies, `outputs/exp21_p1db_vs_frequency_converged/jtwpa`.

| | value |
| --- | ---: |
| gain span across the band | **3.13 dB** |
| P1dB span across the band | **8.70 dB** |
| ratio | **2.78×** |

**This is the headline physics result.** Under pure pump depletion, P1dB is a
function of gain alone — `P1dB = Pp + 10log10[(10^0.1−1)/(2 G_lin)]` — so the
P1dB spread can never exceed the gain spread. Measured, it is **2.78× larger**.

Saturation in this solver is therefore *not* pump depletion alone. The test
needs no reference data, no digitized curve and no absolute calibration, so it
holds independently of every calibration question above.

### Spatial attribution — `jtwpa_spatial.png`

`outputs/exp22_spatial_attribution_converged/jtwpa`, 6141 rows = 2047 branches
at three operating points (zero signal, P1dB, deep saturation). Shows where
along the line the compression is generated.

---

## 3. What is honestly still open

- **Pump-power gap on 2c — size unknown.** Somewhere between ~1.1 and ~7.9 dB
  depending on the Norton factor above. Needs settling before it is quoted.
- **Model runs used no pump-line loss at all** (`--attenuation-db 0`). New runs
  must use `default_loss_model()` (the measured `loss_A10` fit, 34.54 dB at
  7.1 GHz), since all future measurements use the current cabling.
- **Absolute gain 0.92 dB low on 2c**, band shape otherwise correct.
- **Mean +2.86 dB late compression on 2c**, sign consistent with fabrication
  defects the ideal model does not carry.
- **No external saturation reference.** JosephsonCircuits.jl is another
  simulator with no reference of its own, and the Themis gain-map cubes have no
  signal-power axis. The Jan28 105C5 cube used above is the one exception and
  is the only external saturation data in the project.
- **Production-basis truncation.** Sideband count S=10 was chosen against a JC
  comparison that has since been retired. A partial self-convergence check
  (S=6 vs S=10 on 2c at 6.540 GHz) agrees to 0.013 dB, but the full check has
  not been run.

---

## 4. Reproduce

```powershell
python experiments\exp43_presentation_figures.py
python experiments\exp34_gain_band_vs_measured.py
python experiments\exp32_overlay_cuts.py
```
