# exp29 controlled-pump-sweep consolidation

This report records the five independent tracks. No pytest suite was run.

## Track 1 — external power balance (blocking defect repaired)

The old observable mixed boundaries: `extract_port_waves` removed the explicit
`V/Z0` port-shunt current, but `power_balance` retained that shunt's loss in the
dissipation term. The wave-power convention also carries a factor of two relative
to the real-torus average. The repair removes the shunt loss and applies the 1/2
normalization to external wave powers. The standalone mutation demonstration is
[`exp29_track1_power_balance.py`](D:/Projects/Thesis/twpa_jax/experiments/exp29_track1_power_balance.py): the fixture error is `9.52e-1` before and `6.0e-14` after.

Historical exp28 summaries remain marked untrusted for their all-port depletion
fields because they were written before this repair. A fresh jtwpa/2c campaign
rerun is still required to regenerate those fields.

The corrected-termination Q-convergence check completed Q=2 at the representative
7.4 GHz point: gain `12.4555 dB`, P1dB `-89.5883 dBm`, versus Q=1 P1dB
`-89.7446 dBm` (a `0.16 dB` shift). Q=3 was not completed after the machine
restart and is explicitly open.

## Track 2 — production 2c wide gain span

The isolated low-gain extension was attempted in
`outputs/exp29_track2_production_wide`. It stalled in the first heavy solve and
was terminated after approximately 17 minutes to release memory; no new summary
was written. The existing four-point production fit remains `-1.185 +/- 0.024`
over `G=11.61..15.46`. Its low/high two-point slopes are `-1.239` and `-1.191`
dB/dB, respectively, which is not evidence of curvature but is far too narrow a
span to decide the question. Track 2 is therefore open, not silently accepted.

## Track 3 — jtwpa fixed-frequency control

The completed exp28 jtwpa control has four valid points over `G=5.83..26.89 dB`:
`dP1dB/dG = -0.700 +/- 0.345 dB/dB`. This is shallow and non-monotonic, but its
21.06 dB gain window does not match exp21's frequency-swept `G=...` subset. The
frequency-swept values (`-1.086` all and `-3.847` for `G>25`) remain confounded
by frequency structure and are not a controlled comparison.

## Track 4 — termination comparison

The production label is **50 ohm**, from `outputs/ipm_python_design`; no 84.6-ohm
compression slope exists. The corrected 43.33-ohm run is `-1.843 +/- 0.093`
over `G=5.65..12.80` (`n=5`). Production is `-1.185 +/- 0.024` over
`G=11.61..15.46` (`n=4`). Over the matched `G>11` window, corrected is
`-1.328 +/- 0.043` (`n=3`) versus production `-1.185 +/- 0.024`: a `0.14`
dB/dB difference. The full-span slopes are not comparable, and no material
steepening claim is retained. The passive 43.33-ohm sweep worsens the ripple to
`2.515 dB p-p` while preserving the `0.1843 GHz` period; it is not independent
evidence that 43.33 ohm is the true line impedance.

## Track 5 — honest Themis comparison

| Quantity | Axis | Gain window | Slope |
|---|---|---:|---:|
| Model, exp24b/exp21 2c | frequency sweep | reported subsets | -0.387 +/- 0.031; -0.648 |
| Hardware, Themis | frequency sweep | `G>4`, `G>8`, `G>12`, split | -2.165; -3.049; -6.780; -3.027 |
| Model, exp29 2c | fixed-frequency pump sweep | `G=11.61..15.46` | -1.185 +/- 0.024 |
| Model, exp28 jtwpa | fixed-frequency pump sweep | `G=5.83..26.89` | -0.700 +/- 0.345 |

Themis has no signal-power axis at fixed frequency, so no hardware pump-swept
P1dB slope exists. Model/hardware agreement is therefore established only as a
confounded frequency-swept comparison; the fixed-frequency model rows quantify
the confound-free model response and are not hardware matches.

## Ranked conclusion

1. Supported: fixed-frequency pump sweeps produce shallow, device-dependent
   model slopes; production 2c is `-1.185 +/- 0.024` over four points.
2. Supported: the old external balance was invalid; its exact missing terms are
   now repaired and demonstrated.
3. Excluded: the 84.6-ohm compression label and the claim that full-span
   termination slopes differ by `0.66 dB/dB`.
4. Open: regenerated post-repair depletion fields, the production wide-span
   curvature fit (Track 2), and Q=3 convergence after the crash.

### Thesis headline

At fixed 7.4 GHz signal frequency, the production 2c model gives
`dP1dB/dG = -1.185 +/- 0.024 dB/dB` across `G=11.61..15.46 dB` from four
converged pump currents. This is a model-only, fixed-frequency result; Themis
cannot provide the corresponding controlled axis, so it must not be presented as
a direct model-to-hardware comparison.
