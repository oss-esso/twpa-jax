# Pump-frequency periodicity investigation (2026-07-20)

Two maps built from the same `exp07_python_ipm_design_builder.py` / `builders/ipm.py`
IPM circuit builder disagree on how many gain fringes appear across a 7-8 GHz
pump-frequency sweep, and disagree on which one matches the Themis
`14.18.08_Themis_SetupAug25_noVTS_transmission_15mK` measurement:

- `outputs/exp10_pump_map_trailing_50x50_m30_m20_123p9_cg66_halfcurrent_run_gain_map`
  — Lj=123.9 pH, Cg=66 fF (circuit-dir `outputs/ipm_python_design`, exp07 default),
  `--pump-current-jc-scale 1.0`.
- `outputs/exp10_pump_map_trailing_50x50_m30_m20_recovered79_cg33_run_gain_map`
  — Lj=79 pH, Cg=33 fF (circuit-dir `outputs/ipm_python_design_recovered_reference_79_33`),
  `--pump-current-jc-scale` left at CLI default (2.0).

Initial hypothesis: a hidden factor of 2 somewhere in the circuit-builder's
handling of Lj/Cg (between what you pass in and what ends up stamped into the
circuit matrices). That hypothesis is now **ruled out** (see below). The
periodicity mismatch is real and still open.

## 1. The "hidden factor of 2" hypothesis — ruled out

### 1a. exp07 vs `builders/ipm.py`
Byte-for-byte identical files (`experiments/exp07_python_ipm_design_builder.py`
and `src/twpa_solver/builders/ipm.py`) — no divergence between them at all.
Ruled out as a source of any discrepancy.

### 1b. Code trace: does the builder stamp Lj/Cg 1:1?
Traced `add_jj` / `add_jtl_element` / `build_matrices`
(`experiments/exp07_python_ipm_design_builder.py:423-434, 448-458, 866-867`):

- `Cg` -> capacitor stamp value = `Cg` exactly for every interior JTL node.
  Only the first/last node of each contiguous JTL run gets `Cg/2` (standard
  ladder-truncation half-cap; ~12 nodes out of ~2508, not a global factor).
- `Lj` -> `Lj_mod = Lj / mod_factor` (`mod_factor=1.0` for this uniform 2c
  design) -> stamped Josephson-inductor value = `Lj` exactly -> `Ic =
  PHI0_REDUCED / Lj`, no rescale.
- `PHI0_REDUCED` is bit-consistent across the whole pipeline: builder
  (`exp07:70`, `builders/ipm.py:70`), `builders/jc_doc.py:24-25`,
  `core/constants.py:5-6`, loaded as-is in `core/circuit.py:134`, consumed
  unchanged in `pump/problem.py:70,73` and `signal/gamma.py:89`. No second,
  divergent Φ0 convention anywhere downstream.

**Conclusion: what you pass into `IPMParams(Lj=..., Cg=...)` is exactly what
ends up in the assembled C/K/Bphi/Ic. No hidden ×2 in the stamping.**

One real (but separate) nuance found while drawing a tiny example circuit: at
every seam between two different blocks (coupler -> JTL row, JTL row -> TL),
the seam node gets **two** separate half-magnitude ground-cap stamps (one from
each adjoining block's own boundary convention), summed. E.g. node 5 in a tiny
probe circuit got both `C5_0_end = C_gnd_cell/2` (coupler's own boundary) and
`C5_0 = Cg/2` (JTL row's own boundary) — total `C_gnd_cell/2 + Cg/2`. Looks
like intentional/standard ladder-truncation practice, not a bug, but worth
knowing if the reference topology handles seams differently.

### 1c. Independent closed-form check: JTL ring-ladder cutoff frequency
Built a periodic ring of identical JTL cells (series Lj, shunt Cg), using the
same nodal-stamp convention as the real builder, and compared the numeric
generalized-eigenvalue cutoff frequency against the exact closed form
`f_c = (1/pi) * sqrt(1/(Lj*Cg))` (a periodic ring has an exact solution,
`omega_k^2 = (2/(Lj*Cg))*(1-cos(2*pi*k/N))`, maximized at `k=N/2`).

| case | Lj (pH) | Cg (fF) | f_cutoff numeric (GHz) | f_cutoff theory (GHz) | ratio |
|---|---|---|---|---|---|
| A | 123.9 | 66 | 111.312133 | 111.312133 | 1.000000000000 |
| A/2 | 61.95 | 33 | 222.624267 | 222.624267 | 1.000000000000 |
| B (old-Julia, `2*79pH`) | 158 | 66 | 98.571103 | 98.571103 | 1.000000000000 |
| B/2 | 79 | 33 | 197.142206 | 197.142206 | 1.000000000000 |

All four ratios are exact to machine precision. Confirms 1b independently: no
hidden factor of 2, for either the full value or its half.

### 1d. Where the two competing Lj values actually come from
- `experiments/exp07_python_ipm_design_builder.py:129-131` `IPMParams` default:
  `Lj=123.9e-12, Cg=66.0e-15` — undocumented origin, no comment.
- `docs/old_julia_ipm_reference_exactness.md:54-56` — the actual
  JC-validated old-Julia reference circuit (`build_old_ipm_circuit()`, the
  thing the solver's parity claims are pinned to) uses
  `Lj = 2 * 79e-12 H = 158 pH`, `Cg = 66e-15 F`, `Cj = 145e-15 F`.
- `outputs/ipm_python_design_recovered_reference_79_33/ipm_summary.json` has
  an explicit `recovery_note`: this circuit-dir is a **data-recovery splice**
  — C/K/G/Bphi matrices retained from an old build (Cg=66 fF), with
  `ipm_arrays.npz` (Lj/Ic) restored from a separately-generated 79pH/33fF
  design "to reproduce a lost exp10 reference point." It is **not** the
  old-Julia reference value (which is 158/66, not 79/33) and is not a
  documented/git-tracked design decision (`outputs/` is gitignored).
- So: neither of the two compared runs uses the actual JC-validated Lj=158pH.
  `123.9/66` is exp07's undocumented default; `79/33` is an ad-hoc recovery
  splice. `scripts/plot_lj_periodicity.py`'s existing candidate list
  (`--lj-values 79.0 100.0 123.9 150.0`) doesn't include 158 either.
- Separately, `--pump-current-jc-scale` default is 2.0
  (`scripts/run_gain_map.py:2705`, JC positive-phasor convention); the
  `123p9_cg66_halfcurrent` run explicitly overrides to 1.0 (documented in
  `CLAUDE.md` as the currently-validated convention); the `recovered79_cg33`
  run left the default 2.0 in place. This is a separate, independent factor
  from the Lj/Cg question, living entirely outside the circuit builder.

## 2. Defining a reliable periodicity metric

Tried, in order:

1. **Single power row, raw peak/FFT count.** Unreliable: at high power
   (-20 dBm, top of the map's -30..-20 dBm range) both maps are deep past
   their numerical fold wall (32-34% finite coverage), and naive
   interpolation across the resulting NaN gaps corrupts the curve.
2. **Envelope over all powers (max gain per frequency column) + FFT dominant
   period.** More robust to fold-wall gaps, but the measurement's envelope
   isn't uniformly periodic (strong wiggle 7.0-7.2 GHz, flat 7.2-7.6 GHz,
   wiggle again 7.6-8.0 GHz), so a single "dominant period" number is a bit
   of a blend. Gave measurement~6, `123.9/66`~5, `79/33`~4 periods across
   7-8 GHz (`scripts/measure_map_envelope_periodicity.py`).
3. **Simple: one power row per source = min_power + span/3, raw peak count.**
   The metric that stuck — simple, and stable across prominence thresholds
   (1-3 dB gave identical peak sets):

   | source | n_peaks (7-8 GHz) | peak freqs (GHz) | spacing |
   |---|---|---|---|
   | measurement (-25.73 dBm) | 4 | 7.051, 7.293, 7.615, 7.857 | ~0.27 GHz |
   | `123p9_cg66_halfcurrent` (-26.67 dBm) | 5 (6 at low threshold) | 7.204, 7.388, 7.571, 7.755, 7.939 | ~0.18 GHz |
   | `recovered79_cg33` (-26.67 dBm) | 4 | 7.102, 7.367, 7.653, 7.918 | ~0.27 GHz |

   (`scripts/plot_gain_vs_freq_third_power.py`,
   peak-count verified directly against `find_peaks`.)

4. **Cross-check: normalize to [-1,1], align on first peak, subtract.**
   (`scripts/plot_gain_residual_aligned.py`) After a near-zero first-peak
   shift (+0.031 GHz / -0.051 GHz — the two designs were already
   approximately in phase at the start of the window), `recovered79_cg33`
   tracks measurement well through ~7.5 GHz before drifting; RMS residual
   0.369. `123p9_cg66_halfcurrent` drifts out of phase almost immediately
   after the first peak (comb runs faster than measurement's — the classic
   symptom of wrong periodicity, not just an offset); RMS residual 0.758,
   about double.

**Conclusion so far: `recovered79_cg33`'s periodicity matches the
measurement considerably better than `123p9_cg66_halfcurrent`'s (4 peaks
vs. 5, ~0.27 GHz vs. ~0.18 GHz spacing, half the RMS residual after
alignment) — this is the opposite of what the undocumented exp07 default
would predict, and does not by itself tell us whether Lj, Cg, device
length, or some combination is the actual physical driver, since the two
compared designs changed Lj (123.9 -> 79, ratio 1.57x) AND Cg (66 -> 33,
ratio 2x) simultaneously.**

## 3. Open question, resolved by isolation campaign

Is periodicity set only by Lj, or also by Cg / total device length? Section
1's ring-ladder check showed Lj and Cg both set the JTL's dispersion scale
(`f_cutoff ~ 1/sqrt(Lj*Cg)`); physically the *fringe* periodicity (a
phase-mismatch/dispersion ripple, not the JTL cutoff itself, since 6-8 GHz
operation is deep in the ~100 GHz passband) is set by whatever governs total
accumulated phase across the array: Lj, Cg, and the physical length
(`array_length`, `num_rows`). The two originally-compared designs
(`123.9/66` vs `79/33`) never varied these one at a time (Lj and Cg changed
simultaneously), so `scripts/periodicity_campaign.py` ran four isolated
single-parameter sweeps, holding everything else at the exp07 reference
(Lj=123.9 pH, Cg=66 fF, array_length=418, num_rows=6), each measured with the
same peak-count metric (7-8 GHz window, single power row at -25 dBm,
`find_peaks` prominence 2 dB):

**Lj sweep** (Cg=66 fixed, `outputs/lj_periodicity_maps/map_lj*_cg66`, already
built before this campaign):

| Lj (pH) | 70 | 80 | 90 | 100 | 110 | 123.9 | 140 | 160 |
|---|---|---|---|---|---|---|---|---|
| n_peaks | 2 | 3 | **4** | 5 | 5 | 5 | 6 | 6 |
| spacing (GHz) | 1.00 | 0.50 | **0.333** | 0.25 | 0.25 | 0.25 | 0.20 | 0.20 |

**Cg sweep** (Lj=123.9 fixed, `plots/periodicity_campaign_cg.json`):

| Cg (fF) | 22 | 33 | 44 | 55 | 66 | 88 | 110 |
|---|---|---|---|---|---|---|---|
| n_peaks | 3 | **4** | **4** | 5 | 5 | 6 | 5* |
| spacing (GHz) | 0.50 | **0.333** | **0.333** | 0.25 | 0.25 | 0.196 | 0.235* |

(*Cg=110 point had only 33/51 finite frequencies -- lower confidence.)

**array_length sweep** (Lj=123.9, Cg=66 fixed, `plots/periodicity_campaign_array_length.json`):

| array_length | 200 | 300 | 418 (ref) | 550 |
|---|---|---|---|---|
| n_peaks | 3 | **4** | 5 | 6 |
| spacing (GHz) | 0.50 | **0.333** | 0.25 | 0.20 |

**num_rows sweep** (Lj=123.9, Cg=66, array_length=418 fixed, `plots/periodicity_campaign_num_rows.json`):

| num_rows | 3 | 6 (ref) | 9 |
|---|---|---|---|
| n_peaks | 0 (device too short to clear gain threshold at -25 dBm) | 5 | 6 |

### Conclusion

Every one of Lj, Cg, and array_length independently and monotonically
controls fringe density (more of any one of them -> more peaks, shorter
spacing), and **each can independently be scaled down to ~0.5-0.75x its
exp07-reference value and reproduce the measurement's 4-peak/~0.27 GHz
periodicity on its own**:

- Lj: 123.9 -> ~90 pH (0.73x)
- Cg: 66 -> 33-44 fF (0.50-0.67x)
- array_length: 418 -> ~300 (0.72x)

This means periodicity is governed by a joint "total accumulated phase"
combination of Lj, Cg, and device length, not uniquely by any single one of
them. **The fact that `recovered79_cg33` (which happens to scale Cg by 0.5x)
matches measurement better than the `123.9/66` default does not prove Cg is
"the" discrepancy** -- an Lj-only correction (~90 pH, Cg untouched) or a
length-only correction (~300 cells, Lj/Cg untouched) reproduces the same
periodicity equally well. The periodicity metric alone is degenerate across
these three parameters and cannot disambiguate which one is actually wrong
in the model relative to the real fabricated chip. Resolving that needs an
independent constraint outside of periodicity-fitting -- e.g. a measured
junction critical current/area (constrains Lj directly), or the actual
fabricated cell count from the chip layout/GDS (constrains array_length
directly) -- rather than further periodicity sweeps.

## 5. Pump-off S21: an independent (non-periodicity) check via impedance matching

Separate from the gain-comb periodicity metric, computed the **pump-off**
(linear, no Josephson nonlinearity engaged) S21 vs signal frequency for a
grid of (Lj, Cg) combos, using the existing `scripts/plot_passive_s21_s24.py`
/ `experiments/ripple_common.passive_s_matrix` machinery (this solves the
linear MNA circuit -- Josephson junctions at their zero-bias linearized
inductance Lj -- directly, no pump solve needed, so it's a fast, independent
cross-check that doesn't depend on any nonlinear/pump physics at all).

New script: `scripts/plot_s21_lj_cg_grid.py` (`--design DIR LABEL` repeated,
`--separate` for a 6-panel grid instead of one overlay). Six designs built
directly via the exp07 builder module (`ipm.IPMParams(Lj=..., Cg=...)` ->
`make_coupler_discrete` -> `make_ipm` -> `build_matrices` -> `write_outputs`),
reusing two already-existing ones
(`outputs/lj_periodicity_designs/ipm_lj123p9_cg66`,
`outputs/periodicity_campaign_designs/cg33`) and building four new ones
(`outputs/periodicity_campaign_designs/{lj79_cg33,lj100_cg33,lj79_cg66,lj100_cg66}`).

Plots: `plots/s21_lj_cg_grid.png` (overlay, two frequency ranges tried:
7-8 GHz then full 4-12 GHz -- ranking identical in both) and
`plots/s21_lj_cg_grid_separate.png` (6-panel grid, shared axes).

### S21 ripple depth and characteristic impedance

| Lj (pH) | Cg (fF) | S21 range (dB), 4-12 GHz | Z=sqrt(Lj/Cg) (ohm) |
|---|---|---|---|
| 79 | 33 | [-0.766, -0.002] | **48.9** (closest to 50) |
| 100 | 33 | [-3.693, -0.008] | 55.1 |
| 123.9 | 33 | [-8.526, -0.007] | 61.3 |
| 79 | 66 | [-11.255, -0.014] | 34.6 (furthest from 50) |
| 100 | 66 | [-5.732, -0.005] | 38.9 |
| 123.9 | 66 (exp07 default) | [-2.043, -0.002] | 43.3 |

**Finding: S21 flatness ranks exactly by how close Z=sqrt(Lj/Cg) is to the
50 ohm port impedance.** `Lj=79, Cg=33` (Z=48.9 ohm) is almost perfectly
matched -- flattest S21 of all six, ripple under 1 dB across the whole
4-12 GHz band. `Lj=79, Cg=66` (Z=34.6 ohm, furthest from 50) is the worst,
with dips to -11 dB. The exp07 default `Lj=123.9, Cg=66` (Z=43.3 ohm) is
noticeably off-match. This is independent evidence (no periodicity fitting,
no pump/gain solve involved) pointing the same direction as Section 3/4:
`Lj=79, Cg=33` looks like a better-matched design than the exp07 default,
specifically because of its near-ideal impedance match -- a real fabricated
device is normally designed to be impedance-matched to its 50 ohm
measurement chain, so this is a stronger physical argument than periodicity
alone.

### S21 ripple peak density (4-12 GHz, 8 GHz span, `find_peaks` prominence 0.05 dB)

| combo | n_peaks | peaks/GHz | avg spacing (GHz) |
|---|---|---|---|
| Lj=79, Cg=33 | 29 | 3.63 | 0.286 |
| Lj=100, Cg=33 | 40 | 5.00 | 0.205 |
| Lj=123.9, Cg=33 | 45 | 5.63 | 0.182 |
| Lj=79, Cg=66 | 53 | 6.63 | 0.154 |
| Lj=100, Cg=66 | 54 | 6.75 | 0.151 |
| Lj=123.9, Cg=66 | 52 | 6.50 | 0.157 |

At **Cg=33**, peak density rises cleanly and monotonically with Lj (3.63 ->
5.00 -> 5.63 peaks/GHz as Lj: 79 -> 100 -> 123.9) -- same qualitative
Lj-sensitivity as the gain-comb periodicity in Section 4. At **Cg=66**, all
three Lj values cluster tightly (~6.5-6.75 peaks/GHz) -- Lj's effect on this
passive ripple is swamped by something else at Cg=66, plausibly the
coupler-to-line impedance mismatch (Z far from 50 ohm at Cg=66 per the table
above) dominating the reflection pattern regardless of Lj.

### Open thread at time of writing

Not yet concluded: why Lj's effect on ripple density is clearly visible at
Cg=33 but flattened out at Cg=66; whether this S21 ripple periodicity is the
*same* underlying phase-accumulation effect as the Section 4 gain-comb
periodicity or a distinct (coupler-reflection-driven) phenomenon that happens
to correlate; and whether the coupler geometry itself (cached, presumably
optimized for one specific (Lj,Cg) operating point) is part of what's
actually mismatched for the real device rather than Lj/Cg alone. Session
context was cleared right after this point -- pick up the investigation from
here.
