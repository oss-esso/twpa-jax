# 2c early-compression investigation — everything checked, with numbers

Status: the model-vs-hardware P1dB gap is **confirmed real, ~18 dB, direction:
model compresses at less signal power than hardware**. This doc is a single
index of every check performed against that finding, in roughly chronological
order, each with its measured number and verdict. It does not propose fixes —
see individual referenced docs/experiments for methodology detail.

Headline number, most current: **mean same-gain ΔP1dB = -18.20 dB, median
-18.75 dB** (model minus measured), `outputs/phase6_gain_matched_comparison_op7100/gain_matched_p1db_summary.json`,
13 frequencies at 100 MHz spacing, 6.0-7.6 GHz, operating point fp=7.100 GHz /
Ip=7.2311074707853736e-6 A.

## 1. Establishing the headline number

| # | Check | Result | Verdict |
|---|---|---|---|
| 1.1 | Stale-circuit slope disagreement (`outputs/ipm_python_design`, exp23) | measured `-2.231*G - 62.324` vs simulated `-0.400*G - 86.811` (rms 3.31/0.87 dB); crosses at G=13.4 dB | **SUPERSEDED** — wrong circuit ([[real-designs-live-in-designs-not-outputs]]) |
| 1.2 | Live-circuit frequency-matched comparison (exp45, pre-fix) | mean **+2.86 dB** ("model late") | **RETRACTED** — used fabricated flat 72.5 dB signal loss + legacy (non-Norton) port convention |
| 1.3 | Corrected gain-matched comparison, op7.100 (loss_B1 + Norton applied) | mean **-18.25 dB**, median -18.39 dB, rms 18.41 dB, n=18 | direction flips: model compresses **early** |
| 1.4 | Corrected gain-matched comparison, op7.379 (independent operating point) | mean **-17.95 dB**, median -21.96 dB, rms 20.33 dB, n=16 | confirms 1.3 at a second, independently-tuned operating point |
| 1.5 | Dense re-sweep, 100 MHz spacing (vs. the ~200 MHz/18-pt grid above), op7.100 | mean **-18.20 dB**, median -18.75 dB, n=13 | reproduces 1.3 almost exactly — not a frequency-grid-resolution artifact |
| 1.6 | First attempt at 1.5 — units bug | spurious **+41.37 dB**, sign-flipped | caused by comparing model's external/instrument-referred `p1db` field directly against measured on-chip power; fixed by subtracting the run's own `attenuation_db` before comparing ([[exp53-model-p1db-is-external-referred]]) |

Per-frequency deltas at op7.100 (check 1.3) range **-14.45 to -22.85 dB, every
single one negative** — a consistent offset, not sign-scattered noise.

## 2. Ruling out calibration as the cause

| # | Check | Number | Verdict |
|---|---|---|---|
| 2.1 | Port power convention | Norton `P=I²Z0/8` vs legacy traveling-wave `I²Z0/2` — exactly **6.0206 dB** overstatement in every pre-2026-08-05 published number | Fixed; `norton` is now default ([[pump-power-norton-6db]]) |
| 2.2 | Signal-line loss model | fabricated flat 72.5 dB vs fitted loss_B1 (`50.0+3.3√f+0.14f`, RMS 2.80e-5 dB) — off by **+9.4 to +15.3 dB**, erasing a real 5.95 dB band tilt | Fixed ([[loss-b1-fabricated-72p5-fixed]]) |
| 2.3 | `run_compression.py` signal-power labeling | used loss_A10 (pump line) at pump freq instead of loss_B1 (signal line) at signal freq — **~25.1-25.3 dB** too little attenuation | Fixed ([[run-compression-signal-attenuation-fixed]]) |
| 2.4 | `run_compression.py` pump-port default | silently fell back to `source_port` instead of port 4 — signature: √2 residual | Fixed ([[run-compression-pump-port-default]]) |
| 2.5 | Themis on-chip pump calibration | two independent routes (line-loss-derived -66.7 dBm vs depletion-inversion -65.98 dBm) agree to **0.72 dB** | Calibration excluded as explanation for the (separate) ~7.9 dB pump-power gap ([[themis-pump-calibration-confirmed]]) |
| 2.6 | Operating-point tuning as the cause (direct user challenge) | -18.25 dB (op7.100) vs -17.95 dB (op7.379) — same magnitude at two independently-tuned points | **Ruled out** — not an operating-point artifact |

With 2.1-2.4 applied on both sides, the -18 dB gap in section 1 is what
remains. **Do not re-litigate calibration without new evidence.**

## 3. Basis-truncation checks

| # | Check | Number | Verdict |
|---|---|---|---|
| 3.1 | Sideband-count (S) self-convergence, S=10 vs 12 vs 14, live 2c circuit, op7.100, 3 frequencies (`experiments/exp54_basis_self_convergence.py`) | agree to **<1e-6 dB** at 6.0/7.2 GHz; **5e-9 dB** at 8.4 GHz (via the smooth-interpolated P1dB, since the refined nonlinear solve itself stalled at S=12/14 there — a solver hiccup, not a basis-size effect) | **CONVERGED** — S-axis truncation ruled out ([[2c-basis-self-convergence-passed]]) |
| 3.2 | JTWPA Q-axis (`signal_order_max`) convergence, `lattice` basis, low-gain regime (3.756-7.894 dB vs production 27.541 dB) | Q=2 → -110.500076 dBm, Q=3 → -110.440913 dBm, Δ=**0.059163 dB** | PASS, but **different device, different basis family, wrong gain regime** — does not validate 2c/matched |
| 3.3 | 2c Q-axis (`\|q\|`) truncation, production 50Ω circuit, 7.4 GHz, Ip=7.2311e-6 A (same pump current as op7.100), 16 power points | q≤1: P1dB=-92.3820 dBm; q≤2: -90.0640 dBm (**Δ+2.318 dB**); q≤3: -89.9915 dBm (Δ+0.072 dB, converged) | Real, converged at q=2, but **only ~13% of the -18 dB gap**; direction reduces (doesn't explain) the early-compression finding. **Not reproducible on disk** — output dirs pruned, no surviving script; documented only in memory, not in `CLAUDE.md` |
| 3.4 | `reference_gain` (Glin) estimator — is it biased by using the first finite-power point instead of an independent/extrapolated small-signal solve? | floor-flatness delta (`small_signal_floor_delta_db`) across all 13 op7.100 frequencies: **0.00031-0.02710 dB** | Structurally not an independent estimator, but **empirically negligible** at this operating point — 3+ orders of magnitude below the 18 dB gap |

**Net: basis truncation (S-axis and Q-axis) and the Glin estimator together
account for at most ~2.3 dB, in the direction that shrinks the gap slightly.
They do not explain the -18 dB finding.**

## 4. Pump-power / port-reference checks

| # | Check | Number | Verdict |
|---|---|---|---|
| 4.1 | Live-circuit pump port split, pump-only solve, op7.100 (`experiments/exp50_live_pump_port_split.py`) | port1 0.068%, port2 (signal out) **4.935%**, port3 (pump rail ref) **94.986%**, port4 (pump source) 0.011%; cross-checked to 1.6e-16 rel. vs `power_balance` | Confirms `pump_depletion_all_port_db` is dominated by port 3, not a minority-port artifact ([[2c-live-pump-port-split-measured]]) |
| 4.2 | Reference-plane correction: use port-2-coupled fraction (-13.067 dB) instead of rail-injected power as the depletion-bound reference | rail-injected: model early-by **+7.34 dB**; port-2-coupled: **-5.73 dB** (overshoots to "late") | **Ruled out** — a single lumped port fraction is the wrong model for this two-line coupled-directional-coupler geometry ([[2c-reference-plane-hypothesis-overshoots]]) |
| 4.3 | Energy-accounting: measured vs modeled pump-referred P_sat and depletion at compression | measured `P_sat - P_pump = -8.89 dB` vs modeled **-22.67 dB** (13.78 dB pump-referred gap); measured depletion at compression **~26%** vs modeled **~1.3%** (~20× ratio) | Both real; plan explicitly flags the *measured*-side relationship (26% depletion → only ~1 dB compression) as itself internally questionable, not just the model (`psat_comparison_fix_plan.md:100-105`) |
| 4.4 | `P_sat` masking/amplification of the ΔP1dB defect (provisional OP, fp=7.725 GHz) | ΔG0=-2.73 dB, ΔP1dB=**-15.35 dB**, ΔP_sat=-19.14 dB, masked fraction **-19%** (negative = amplified) | Sign relationship is **operating-point-dependent**, not a fixed law — opposite of the sign seen during planning (ΔG0/ΔP1dB then had opposite signs, partially cancelling) ([[psat-decomposition-masking-flips-sign]]) |

## 5. Spatial / distributed depletion checks

| # | Check | Number | Verdict |
|---|---|---|---|
| 5.1 | Standing-wave characterization, pump-off linear solution, 7.629 GHz | `\|V-\|/\|V+\|` median **0.359**, mean 0.357, max 0.533 | Device is resonant/standing-wave, **not** a clean traveling-wave line ([[2c-standing-wave-not-traveling-wave]]) — but confirmed NOT a simple port-impedance mismatch (forcing 84.6Ω made ripple worse, 2.040→3.090 dB p-p) |
| 5.2 | Spatial depletion profile, op7.100, fs=7.2 GHz, 2508 branches | lumped (all-port) depletion **-0.317 dB**; local median -0.407 dB; **worst branch 2502: -8.651 dB (~86%)**; best branch +2.185 dB (local pump gain) | Deep local depletion is real and highly localized near the output-coupler end of the chain |
| 5.3 | Is local depletion caused by pump-intensity hot-spotting? | `pump_above_10pct_fraction = 1.0` at both small-signal and P1dB states — every branch within 10× of peak pump intensity | **Falsified** — pump amplitude is spatially uniform; depletion localization is not an intensity-concentration effect |
| 5.4 | Mechanism re-check: standing-wave interference null vs ordinary distributed depletion | pump_p1db/pump_small ratio: smooth ~2.5% modulation (0.379→0.369→0.370) across ~20 branches, not a sharp null; `corr(signal_flux_abs, local_depletion_db)` = **-0.82** (row 5), -0.74 (whole chain) | **CORRECTED** — this is ordinary distributed 4WM depletion tracking local signal amplitude (depletion accumulates where signal has already grown largest, near the amplifier output), **not** a standing-wave interference null. Earlier "razor-sensitive null" framing retracted |
| 5.5 | `spatial_depletion_null` (distributed estimator) vs lumped port-referred bound, same test point | distributed estimator predicted 3.53 dB (vs crude local reference 4.08 dB, ~0.55 dB implied) against actual solved **0.857 dB**; lumped bound off by 12-16 dB | Distributed/local accounting tracks the device far better than any single scalar pump reference, though still not tight |

## 6. Disorder / fabrication-realism checks

| # | Check | Number | Verdict |
|---|---|---|---|
| 6.1 | 1% Lj scatter vs nominal (`designs/campaign_diss/2c_sc1pct`, seed=1, `plasma_locked` Cj), same op point and signal point | worst branch 2502→2501; local worst depletion -8.65→**-8.09 dB**; lumped -0.32→-0.30 dB; actual compression 0.857→0.710 dB | Null barely moves — **1% disorder does not wash out the localized depletion pattern** |
| 6.2 | 5% Lj scatter | pump solve converges (adaptive fallback), signal solve `SIGNAL_CONTINUATION_FAILED` | **Untestable at this operating point** — disorder moves the resonance/fold structure enough that (7.100 GHz, 7.231e-6 A) stops being valid |
| 6.3 | 10% Lj scatter | pump solve itself fails (`line search failed at Newton 8`) | Same as 6.2, worse |

**Verdict: neither confirms nor refutes "disorder explains the hardware/model
gap."** The one clean data point (1%) argues against it as a smearing
mechanism; 5-10% (where it might plausibly matter more) was never reachable
without re-tuning the operating point per disorder realization, which was not
done. Single seed only — not an ensemble statement.

## 7. Slope / methodology checks (P1dB vs gain)

| # | Check | Number | Verdict |
|---|---|---|---|
| 7.1 | "Model slope -0.39 vs hardware -3.03" (early framing) | model subsets: all-jtwpa -1.086 (R²=0.11), **jtwpa G>25: -3.847** (R²=0.99, steeper than hardware); fqjtwpa all -0.629 (R²=0.997, but a **two-cluster artifact** — 7 pts at G=27.4-28.7, 1 at G=8.6); 2c all -0.648 (R²=0.77) | **Withdrawn as stated** — the model is not uniformly flat; it spans -0.63 to -3.85 depending on device/subset, same subset-sensitivity as hardware |
| 7.2 | Root cause of slope disagreements | every slope on both sides comes from a **frequency** sweep (P1dB varies with frequency independent of gain — ripple, band edges, phase matching), not a controlled fixed-frequency pump-power sweep | Confounded; the controlled experiment (fixed frequency, vary pump power) has never been run on either side. Hardware datasets on hand cannot supply it either (Jan28 cube: signal-power × signal-freq at one pump power; Aug25 cubes: pump-power × signal-freq, no signal-power axis) |
| 7.3 | Is `-1 dB/dB` (depletion-only) or `1.302 dB` (depletion at 1dB compression) a valid bound/gate? | depletion-only model values are internal to that model, not general | **No** — they are diagnostics only; shallower-than-(-1) means a gain-independent limiter dominates, steeper means a super-linear-in-gain limiter dominates. Never gate on either number |
| 7.4 | Themis band-resolved pump inference vs G0 | correlation `Pp` vs `G0`: r=**-0.633**; band-resolved deviation from true pump symmetric about the amplifying core (+4.7 dB at 5.5-6.0 GHz, +6.8 dB at 8.5-9.0 GHz); where G0≥10 dB, hardware sits within ~1 dB of the depletion bound with the independently-known pump | Substantially withdraws the earlier "hardware compresses ~3× faster than depletion" claim — that slope was dominated by low-gain band-edge columns diluted by non-parametric transmission |

## 8. What survives as open

Not ruled out, not confirmed causal for the ~18 dB magnitude:

- **Distributed/spatial depletion mechanism** (§5) — real, localized, tracks
  local signal amplitude, gets closer than any lumped bound, but not yet shown
  to explain 18 dB specifically; only measured at one (fs, Ip) point.
- **Genuine missing physics in the lumped-element model** — the model is
  fully deterministic (no thermal/shot noise, no self-heating, no
  quasiparticle/TLS/wire-bond/package-mode physics anywhere in
  `src/twpa_solver`); whether any of these matter on the real chip is
  unknown from this codebase alone.
- **Standing-wave/resonant content as a contributor** (§5.1) — real and
  measured, but its *cause* is unidentified (ripple period matches neither
  `1/τ` nor `1/2τ` for the chain) and its causal link to the depletion
  pattern was explicitly asserted-then-retracted (§5.4) — open, not a dead
  end, just unproven.
- **Disorder/scatter at the 5-10% level** (§6) — plausible, untested, would
  require re-tuning the operating point per realization first.

## Reference operating points used throughout

| label | pump freq | pump current | note |
|---|---|---|---|
| op7.100 | 7.100 GHz | 7.231074707853736e-6 A | exp31-matched, primary operating point for sections 1, 3.1, 3.3, 4.1-4.2, 5.2-5.5, 6 |
| op7.379 | 7.379 GHz | 7.381e-6 A | exp45-refined, independent cross-check (§1.4) |
| provisional (fp=7.725) | 7.725 GHz | 1.077e-5 A | superseded/provisional Phase 4 point, used only in §4.4 |

## Source memory / doc index

`[[2c-model-compresses-early-confirmed]]`, `[[2c-basis-self-convergence-passed]]`,
`[[exp53-model-p1db-is-external-referred]]`, `[[production-basis-caps-q-at-one]]`,
`[[2c-live-pump-port-split-measured]]`, `[[2c-reference-plane-hypothesis-overshoots]]`,
`[[2c-spatial-depletion-localized-not-uniform]]`, `[[2c-disorder-does-not-wash-out-null-at-1pct]]`,
`[[p1db-slope-confounded-by-frequency]]`, `[[depletion-slope-is-diagnostic-not-bound]]`,
`[[psat-decomposition-masking-flips-sign]]`, `[[themis-pump-calibration-confirmed]]`,
`[[2c-standing-wave-not-traveling-wave]]`, `[[2c-p1db-gain-slope-disagreement]]` (superseded, method-only),
`[[live-2c-model-compresses-late]]` (retracted), `[[pump-power-norton-6db]]`,
`[[loss-b1-fabricated-72p5-fixed]]`, `[[run-compression-signal-attenuation-fixed]]`,
`[[run-compression-pump-port-default]]`; `docs/development/psat_comparison_fix_plan.md`,
`CLAUDE.md` ("2c basis self-convergence, measured 2026-08-06").
