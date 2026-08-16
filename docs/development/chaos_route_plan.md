# Implementation Plan: chaos-route reproduction and transfer

Date: 2026-08-12
Corpus: `docs/development/chaos_papers/` (8 PDFs, full extraction, plus a
working FDTD implementation)

## Execution record (2026-08-12)

The prerequisite and plumbing gates are complete: `numba` imports, the FDTD
quick smoke and 50,000-step benchmark pass (`14.399704 us/step`), the
rf-SQUID design compiles with non-empty `Bphi` and non-zero transmission, and
the focused diagnostic tests pass.  The Phase-1 runner now accepts the paper's
explicit bifurcation-current list and reports both raw drive-cycle samples and
the estimated number of distinct Poincare points.

The published Shukrinov window probe preserves the damping sum rule to about
`1e-8` and finds both positive and non-positive largest Lyapunov exponents.
The universal `D = 0.868 +/- 0.012` gate is not yet established: the current
box-counting result is a Poincare-section dimension, not the paper's
high-resolution `(I,V)` staircase dimension, and the seven bifurcation points
have not been independently re-localized from a fine current scan.

That measurement distinction is now implemented.  A downward 401-point CVC
continuation from `I=1.2` to `0.2`, with `1000+1000` normalized settling and
averaging units, is recorded at
`outputs/chaos/phase1/staircase_401.json`; its staircase-specific dimension is
`1.0582`.  Changing the fitted box scales moves the estimate from about `1.06`
to `0.74`, demonstrating that this grid is under-resolved and that selecting a
scale range to force `0.868` would be invalid.  The paper's `1e-7`--`1e-8`
current resolution still requires a denser continuation before the universal
dimension gate can be accepted.

A local `4,001`-point probe at exactly `1e-7` current spacing around the
`5x/3` window is recorded at
`outputs/chaos/phase1/fine_window_1e7.json`.  It is accelerated with numba but
uses the same RK4 equations and carry-forward state; the final states were
regression-tested against the generic implementation.  Its CVC dimension is
`1.5815`, with a voltage range of only `5.6e-4` but alternating values at the
sampling scale.  The longer generic runs at the paper's listed currents give a
period-3 section at `I=0.25756981` and period-6 sections at the subsequent
listed currents.  `extract_period_doubling_sequence` now consumes those
measured period counts, orders samples by current, and emits transition
brackets plus Feigenbaum ratios; on this sparse window it finds only the
`3 -> 6` bracket `[0.25756981, 0.26025001]`, midpoint `0.25890991`, and no
ratio because only one candidate is resolved.  This validates the extraction
plumbing without claiming a cascade: the paper's seven bifurcation points,
dimension, and universal `4.6692` ratio still require an independently
resolved fine scan.

The accelerated continuation now also records scalar drive-phase voltage
sections, so the fine scan can extract periods rather than only CVC means.  The
recorded `1e-6` scan over `I=0.25750..0.26080`, with `2000+2000` normalized
settling/averaging, is `outputs/chaos/phase1/period_scan_1e6.json`.  It has
3,301 points: 717 period-3 and 2,584 period-6 samples, with a narrow
non-monotonic 3/6 region near `I=0.25822`; no period-12, period-24, or
accumulation sequence is resolved.  The extracted candidates are
`0.2582155` and `0.2582245`, so no Feigenbaum ratio is accepted.  This is a
stronger resolution check, but it remains a mismatch to the paper's reported
seven-point cascade rather than a universal-constant pass.

A focused `1e-7` scan of the reported accumulation neighborhood
`I=0.26070..0.26080` is recorded at
`outputs/chaos/phase1/period_scan_accumulation_1e7.json`.  All 1,001 samples
remain period-6, with no period-12 transition or candidate.  Together with
the full-window `1e-6` scan, this makes the missing cascade an observed
reproduction mismatch at the tested protocol, not merely an untested coarse
grid.

The paper-scale Guarcello runs are recorded under `outputs/chaos/phase2/`.
The 51-point pump sweep (`fig2a_full`) does not reproduce the stated threshold:
gain remains between approximately `-10` and `+5 dB` near `-54` to `-53.5
dBm`, with no `~10 dB` jump.  The 51-point signal negative control
(`fig2b_full`) spans `-11.8` to `39.0 dB` and has a sharp resonance at `7 GHz`,
rather than a flat `~8 dB` response.  The full convention audit records
`-6.265 dB` for paper/stable, `+3.184 dB` for 50-ohm/stable, and a failed
(`NaN`) paper-centered run at the same point.  These are explicit corpus
reproduction mismatches, not acceptance passes.

The RF-SQUID transfer is now exercised on both required axes.  The power
campaigns are in `outputs/chaos/phase3/rf_hold10/`; the new nine-point,
20-period DC-flux campaign is in
`outputs/chaos/phase3/rf_bias_9/bias_campaign_summary.json`.  All 18 bias
direction/point records pass the decay gate and include the Eq. (4) screening
predictions for both `L_m` and `L_m + L_par`.  Interior broad-scatter labels
are not promoted to a device verdict because `sigma_V'PS` remains below the
paper's `0.1` cut and the spectral corroboration is not decisive.

The RF-bias driver now also emits the sampled-period largest Lyapunov estimate
while restoring the carried state after each perturbation map.  The bounded
end-to-end validation is
`outputs/chaos/phase3/rf_bias_lyap_smoke/bias_campaign_summary.json`: four
records cover both directions and all four contain finite LE estimates.  Its
one-period hold intentionally fails the decay gate, so those values validate
the diagnostic plumbing only; the longer nine-point campaign remains the
authoritative bias-axis geometry/decay result.

The 2c campaign has now been extended through the requested wall bracket.  The
corrected geometry-priority classifier report is
`outputs/chaos/phase3/ladder_bracket20/campaign_summary.json`: 11 powers from
`-24.5` through `-22 dBm`, four resistance settings (`inf`, `1e4`, `1e2`, and
`1`), both directions, and 20-period holds produced 88/88 decay-gate passes.
The result is predominantly `NO_BIFURCATION_FOUND`; the only broad/scattered
labels are isolated upward points (`-24.25 dBm` for the three weak-damping
settings and `-24.5 dBm` for `R/Rn=1`), with no matching down-sweep structure
and `sigma_V'PS < 0.1`.  The earlier scalar-spectrum `NEIMARK_SACKER`
over-call was removed because it lacked two-dimensional Poincare geometry.
This is strong bracket evidence against a robust physical bifurcation, but a
final solver decision still requires timestep/sideband convergence and the
L-stable corroboration specified by the ansatz gate.

The timestep audit is recorded in
`outputs/chaos/phase3/timestep_ms05/`, `timestep_ms025/`, and
`timestep_ms0125/`.  For the lossless route, reducing the implicit-trapezoid
maximum phase step from `0.5` to `0.125` moves `sigma_V'PS` from roughly
`0.056` to `0.066` while retaining compact two-point geometry.  For `R/Rn=1`,
the upward sweep remains broad (`14--16` clusters) while the downward sweep
remains compact; this is reproducible hysteresis, but not a converged chaotic
classification because the scalar spread is far below `0.1`.

The short L-stable corroboration is also explicit.  Both BDF and Radau runs at
the isolated `-24.25 dBm` point are in
`outputs/chaos/phase3/bdf_m24p25/summary.json` and
`radau_m24p25/summary.json`; both terminate with
`Required step size is less than spacing between numbers`.  The RF-SQUID
formulation rejects BDF/Radau by construction because its lossless algebraic
constraints require implicit trapezoid.  Thus the ansatz gate's L-stable
requirement is not satisfied, and no solver change is authorized.

For a like-for-like dissipative check, an `R/Rn=1` circuit variant was solved
independently with the full pump HB runner at the same current; its checkpoint
is `outputs/chaos/phase3/r1e00_hb_m24p47/` and the final coefficient residual is
`9.4e-13`.  BDF still fails after both the raw HB handoff and a successful
implicit-trapezoid restart (`r1e00_bdf_authoritative/` and
`r1e00_bdf_restart2/`), including a relaxed-tolerance diagnostic.  The
zero-equilibrium BDF path produces an identically zero output trace and is
therefore not a valid corroboration.  This establishes a reproducible
adaptive-integrator limitation, not an L-stable acceptance pass.

The failure is not caused by the segmented restart wrapper: a direct one-period
`solve_ivp(method="BDF")` call on the dissipative `R/Rn=1` reduced system
(`full_state=false`) reaches only `theta=5.43e-6` before returning the same
minimum-step message (`nfev=288`, `njev=16`, `nlu=129`).  The lossless RF-SQUID
variant remains correctly restricted to implicit trapezoid because its
`full_state=true` algebraic representation is not an admissible BDF/Radau
state.  Thus the L-stable gate is a measured limitation of the available
state-space formulation, not an untested wrapper configuration.

Accordingly, the Phase-4 result remains explicitly gated: the Themis corpus is
`UNDETERMINED`, and no harmonic-balance ansatz or production solver change is
authorized by this route until the unresolved Phase-1/Phase-2 acceptance
mismatches and the device-side convergence evidence are resolved.

### Next-stage execution record (2026-08-12)

The Phase-3 `R/Rn=1`, upward `-24.50 dBm` anomaly was tested with the same
20-period holds and 0.25 dB spacing after shifting the starting power. The
original first point had 15 Poincare clusters and
`sigma_V'PS = 0.0036`. Starting at `-25.00 dBm` produced a compact
two-cluster `-24.50 dBm` point with `sigma_V'PS = 9.4e-5`; the independent
`-25.50` prefix likewise remained compact at `-24.50 dBm`. The verdict is
**initialization transient**, not a bifurcation pinned to that power. The
complete `-25.00` run is in
`outputs/chaos/phase3/ladder_init_m25p00/campaign_summary.json`; the clean
five-point `-25.50` prefix is in
`outputs/chaos/phase3/ladder_init_m25p50_prefix/campaign_summary.json`.

The classifier now retains absolute `sigma_V'PS` but exposes the dimensionless
within-sweep statistic `sigma_max / sigma_deep_stable`, with a default
40x threshold. A smooth monotone sigma rise is explicitly classified as
`NO_BIFURCATION_FOUND`; the existing 2c-style smooth 2.2--3.2x regression is
covered by `tests/test_chaos_attractor_classify.py`. A missing checkpoint power
can now be supplied explicitly to continuation with
`--reference-power-dbm`; this is metadata plumbing only.

Levinsen is now implemented in `scripts/chaos/levinsen_paramp.py` as a
four-state normalized RK4 model with phasor Gamma readout and a persisted
signal-plus-white-noise control. Its analytic-limit and linear readout tests
pass. The Levinsen pump sweep completed at 29 points from `i_D=0.20` to
`0.27`. Measured gain stayed at approximately `-1.329 dB` throughout, so the
paper's `>30 dB` half-harmonic threshold gate **failed**. The white-noise
negative control did run: its noise-temperature ratio stayed in
`[34.659, 34.666]`, with no noise rise, but this does not rescue the missing
positive gain. The likely cause is an incomplete circuit normalization or
coupling law in the first four-state reproduction, not a paper reproduction
pass; the artifact is
`outputs/chaos/phase1/levinsen_pump_sweep.json`. No parameter was tuned to
force the published gain.

The corrected Guarcello Fig. 2(a) and Fig. 2(b) sweeps have been launched at
the existing 51-point density with `--power-convention 50ohm`. Their outputs
are being written under `outputs/chaos/phase2/`; the earlier `paper`
convention artifacts remain the overdrive control. Fig. 2(b) uses `-54.5 dBm`
as specified by the body text; the caption's approximately `-55 dBm`
discrepancy remains recorded above.

### Directional Poincare re-reduction (2026-08-12)

The Poincare statistic was corrected in both the corpus FDTD implementation
and the attractor classifier. Every analysis now emits the upward branch, the
downward branch, and the legacy both-sign spread; verdicts use the upward
branch. A clean sinusoid now has directional `sigma` near zero while retaining
the expected derivative amplitude. The regression and chaos-focused tests pass.

The existing Phase-2 artifacts were re-reduced from their saved
`poincare_map.npz` values without integration:

- Fig. 2(a) has the corrected stable directional
  sigma is `2.6486e-4`; the maximum ratio is `250.3`, and the first listed
  transition point at `-53.5 dBm` has ratio `74.6`, so the requested `40--50x`
  absolute-from-deepest-stable magnitude is not the relevant local gate. The
  requested local gate **passes**: `sigma_up` rises from `0.000820` at
  `-54.5 dBm` to `0.019769` at `-53.5 dBm`, a `24.1x` jump within `1 dB`.
  The transition location is reproduced. The corrected gain sequence is
  evaluated below; the earlier paper-convention low-power dip is withdrawn
  and must not be used as evidence.
- Fig. 2(b) masks exactly `7.00 GHz` as pump-frequency contamination. After
  masking, the 6--8 GHz gain mean is `2.76 dB` with a `13.69 dB` range, not the
  requested approximately `8 dB` flat response. The endpoint gains are
  `-3.18` and `-4.78 dB`; the dense-Poincare negative control remains present.
  The frequency-response gates therefore **fail**.

The reduction artifacts are
`outputs/chaos/phase2/fig2a_50ohm_full/directional_sigma_reduction.json` and
`outputs/chaos/phase2/fig2b_50ohm_full/directional_sigma_reduction.json`.

### Guarcello gain-reference correction (2026-08-12)

The pump-off reference is recorded at
`outputs/chaos/phase2/pump_off_reference_50ohm.json`. The unpumped absolute
signal response at `6.42 GHz` is `-6.278 dB`, close to the equal-resistor
source/load divider prediction `-6.021 dB`; the divider is the dominant
reference error. Pump-off normalization removes the divider and the remaining
distributed line loss by construction, while retaining the old absolute
`gain_db` field.

The corrected Fig. 2(a) gain values at the digitized checkpoints are:
`-1.095, -1.047, 0.206, 1.984, 4.302, 5.697, 7.098, 8.420, 9.462,
11.232 dB` for `-70, -65, -62, -60, -58, -57, -56, -55, -54, -53.5 dBm`.
Against the digitized curve the mean absolute residual is `0.930 dB` and the
maximum is `1.647 dB`, consistent with figure-reading uncertainty. The
`<=8 dB` gate through `-54 dBm` **fails narrowly** because the corrected
`-54 dBm` value is `9.462 dB`; the jump and sigma-transition gates pass, and
the post-threshold curve reaches `11.232 dB` at `-53.5 dBm` but does not
reproduce the paper's clipped `~20 dB` lower bound. Therefore Fig. 2(a) is
**not fully reproduced**, although its shape, transition location, and local
directional-sigma gate are reproduced.

The previously reported non-monotonic `-10.476 dB` dip at `-66.5 dBm` came
from the withdrawn `paper`-convention dataset, not the authoritative `50ohm`
curve. It is withdrawn from this record; the 50-ohm curve is smooth and
monotone through the low-power region.

### Fig. 2(b) stable-regime check and port-update bound (2026-08-12)

The requested independent signal sweep at `Ppump = -57 dBm` completed in
`outputs/chaos/phase2/fig2b_50ohm_pump_m57_full/`. After masking exactly
`7.00 GHz`, the corrected-reference gain remains scattered: the 6--8 GHz
mean is `5.998 dB`, the range is `11.462 dB`, and the endpoints are `2.604`
and `2.092 dB`. The dense-Poincare negative control remains true. This was the
former **Outcome B**, now **WITHDRAWN**: the single-tone gain estimator was
contaminated by finite-record pump leakage. The corrected multi-tone sweep and
its smoothness gate are recorded below.

The initial stable versus paper-centered audit is complete at `-60`, `-57`,
and `-55 dBm` in
`outputs/chaos/phase2/port_update_boundary_audit.json`. The stable absolute
gains are `-4.294`, `-0.582`, and `2.142 dB`; the literal centered update is
`NaN` at all three powers. Smaller-timestep retries are being recorded in
`outputs/chaos/phase2/paper_centered_timestep_audit.json`; no centered result
survived at `dt_norm = 0.01`, `0.005`, or `0.0025` at any of the three powers.
The Task-4 bound is therefore **persistent divergence through the tested
timestep range**, not a measured alternative gain curve. The centered scheme
is not a viable physical reference for this route.

The saved Phase-3 traces were also re-reduced. In `ladder_bracket20`, the
upward branch is one compact cluster for most records; the remaining broad
points are isolated upward records with no matched down-sweep controls. The
old two-sign two-cluster interpretation is therefore invalid. The corrected
2c verdict remains **NO_BIFURCATION_FOUND**, supported only by compact
directional geometry over most of the sweep, lack of matched up/down structure,
and disappearance of the first-point anomaly when initialization is shifted.
The prior smooth 2.2--3.2x both-sign sigma argument is void. The reduction is
in `outputs/chaos/phase3/ladder_bracket20/directional_sigma_reduction.json`.

The RF-bias and RF-hold artifacts were re-reduced with the same directional
statistic under their respective output directories. Their old both-sign
numbers are retained in each reduction; no new integrations were run.

### Levinsen topology and normalization correction (2026-08-12)

The Levinsen model now feeds tuned-circuit current back into the junction,
uses the derived resonator coupling `omega_0/(R_L Q)`, and requires an explicit
`--tuned-frequency-hz` assumption. The McCumber normalization is plasma-
frequency based with damping `1/sqrt(beta_c)`. A no-pump ring measurement with
the explicit 240 Hz tuned-circuit assumption measures `299.7 Hz` against the
paper's stated 265 Hz biased resonance. This is recorded as a normalization /
parameter mismatch, not tuned to the target; see
`outputs/chaos/phase1/levinsen_ring_measurement.json`.

The corrected empirical pump sweep completed at 29 points in
`outputs/chaos/phase1/levinsen_pump_sweep_corrected.json`. The strongest gain
was `-0.345 dB` at `i_D=0.270`; no point exceeded 0 dB, despite a resolved
half-harmonic FFT bin near `249.7 Hz`. The paper's `>30 dB` gain gate therefore
**fails** again, now for the feedback-corrected model. Every noise record is
`not_evaluable_no_gain`; the earlier constant-noise claim is withdrawn.

## Current gate status

| phase | implementation status | authoritative evidence | acceptance status |
| --- | --- | --- | --- |
| 0 prerequisites | complete | `outputs/chaos/phase0/`, dependency and RF-SQUID smoke records | PASS |
| 1 Levinsen single-junction parametric amplifier | feedback topology and plasma normalization implemented; empirical sweep complete | `outputs/chaos/phase1/levinsen_pump_sweep_corrected.json`, `levinsen_ring_measurement.json` | `>30 dB` gain gate FAILED; noise NOT_EVALUABLE because gain never exists |
| Shukrinov diagnostic record | dropped as a target; failure diagnosis retained | `outputs/chaos/phase1/` | dimension/Feigenbaum gates NOT PASSED and out of scope |
| 2 Guarcello reproduction | multi-tone re-runs and dual gain reductions complete; centered timestep audit complete | `outputs/chaos/phase2/` | G1-G5/B3-B4 PASS; G6 wideband agreement FAILS; dense negative control PASS |
| 3 device transfer | complete for both devices, directions, damping/timestep axes | `outputs/chaos/phase3/` | provisional geometry; L-stable gate NOT PASSED; `sigma = 0.1` cut inherited from an unvalidated Phase 2 |
| 4 Themis classifier and solver decision | complete as a gated diagnostic/decision record | `outputs/chaos/phase4/themis_wm_classification.json`, `docs/development/chaos_route_solver_plan.md` | `UNDETERMINED`; NO solver change authorized |

This table separates unfinished code from measured acceptance failures.  The
route is implementation-complete for its permitted scope, but it is not a
scientific acceptance pass and therefore does not authorize the dormant
period-doubled or quasi-periodic ansatz paths.

### Figure readings and scope change (2026-08-12, second revision)

PDF page rendering is available through the already-installed PyMuPDF
(`fitz`), so figures in `chaos_papers/sources/` can be read directly. Poppler
is not required.

**Scope change, user direction:** Shukrinov is dropped as a Phase 1 target.
Guarcello (device-level) and Levinsen (single junction) are the priorities. The
Shukrinov defect diagnosis below is retained as a record of why its gates
failed, not as work to resume.

#### The power convention is settled by the paper's own internal cross-check

Guarcello page 4 states a convention-independent number: "the range of values
within the Josephson phase oscillates enlarges with `Ppump`, so that
`|phi| <~ 1.5` just before the onset of a chaotic regime".

Measured at `-54 dBm`:

| convention | gain | drive vs `Ic` | `max abs phi` | vs published `<~1.5` |
| --- | ---: | ---: | ---: | --- |
| `50ohm` | `+3.18 dB` | `6.3x` | `1.544 rad` | matches |
| `paper` | `-6.27 dB` | `63.8x` | `35.255 rad` | `23x` too large |

The `50ohm` convention matches the paper on three independent axes at once:
transition power `(-54, -53.5)`, phase threshold `<~1.5`, and gain magnitude
(Fig. 2(a) reads `3`-`5 dB` at `-54 dBm`). No impedance rescues `0.032 mV`:
`V_pk^2/2P` requires `5120 ohm` and `V_rms^2/P` requires `10240 ohm`. The
50-ohm peak value is `0.0032 mV`. The published figure is a misplaced decimal
and the simulations used the standard convention.

#### Fig. 2 readings

| panel | axis | reading |
| --- | --- | --- |
| 2(a) gain | `0`-`20 dB` | smooth `0 -> 5 dB` below `-55`; knee at `-55`..`-54`; `15`-`20 dB` and visibly jagged above `-53.5` |
| 2(a) `V'_PS` | `0.0`-`0.45` | thin line below `-55`, small precursor wiggles `-57`..`-54`, abrupt scatter to `0.05`-`0.45` at `-53.5` |
| 2(b) gain | `2`-`8 dB` | peak `~8.2 dB` at `7 GHz`, falling to `~2 dB` at `4` and `10 GHz`; inset ripple `7.8`-`8.3 dB` |
| 2(b) `V'_PS` | **`0.116`-`0.126`** | band at `0.120`-`0.123` with a beating envelope; total spread `~0.003` |
| 2(c) `V'_PS` | `0.00`-`0.25` | structured bands versus `I_bias`, the DC axis relevant to `rf_squid_2393_3wm` |

**Consequence for the `sigma_V'PS` criterion.** A stable cluster has
`sigma ~ 0.002`; the chaotic state has `sigma ~ 0.1`. The paper's transferable
statistic is therefore the **ratio**, about `40`-`50x`, crossed abruptly inside
`0.5 dB` -- not the absolute `0.1`, which carries Guarcello's own voltage scale
and `omega_plasma = 27.74 GHz` and is dimensionally meaningless on another
device. All device work must quote `sigma_max / sigma_deep_stable` within its
own sweep.

This also corrects an earlier reading in this document: our `50ohm` point at
`-54 dBm` with `sigma = 0.0946` is not "comfortably below the chaos cut". It is
about `40x` a stable cluster, i.e. the transition itself, which is exactly where
the paper places `-54 dBm`.

#### Two further findings from the figure page

- **The paper is internally inconsistent about the Fig. 2(b) pump power.** The
  caption says `Ppump = -55 dBm`; the body text on pages 3 and 5 says
  `-54.5 dBm` twice. Our runs used `-54.5`, following the body text. Record the
  discrepancy rather than resolving it.
- **Guarcello attribute their own route to chaos to period doubling** (page 4):
  "for `Ppump >~ -53.5 dBm`, there is a sudden proliferation of new spectral
  lines and a noticeable broadening of the existing ones. This is typically
  indicative of a period-doubling cascade". This is corroborating context for
  the dormant period-doubled ansatz, not evidence for it.

#### Levinsen replaces Shukrinov as the Phase 1 target

Fig. 5 and Fig. 6 supply a complete, closed parameter set for a single-junction
parametric amplifier, and Fig. 6 measures **gain** -- the observable this
project actually needs, which Shukrinov never computes.

Model (Fig. 5): a circulator isolates the signal source, so it enters as a
current source `i_in` with source resistance equal to the load resistance. A
series tuned circuit DC-isolates `R_L` from the junction. The load current is
the sum of the tuned-circuit current and `i_in`, and amplification is
`Gamma = (i_out / i_in)^2`.

Parameters (Fig. 6 caption plus "junction parameters as Fig. 2"):

| quantity | value |
| --- | --- |
| McCumber parameter `beta_c` | `25` |
| junction characteristic frequency `f_J` | `318 Hz` |
| dc bias `i_dc` | `0.4` |
| pump frequency `f_D` | `480 Hz` |
| signal frequency `f_s` | `265 Hz` |
| half-harmonic `f_D/2` | `240 Hz` |
| implied idler `f_D - f_s` | `215 Hz` |
| load resistor | `4 x R_J` |
| tuned-circuit `Q` (including load) | `10` |
| pump amplitudes | `i_D = 0.230` (lower curve), `0.245` (upper) |
| plot scales | `10 dB` vertical division, `10 Hz` horizontal |

Acceptance target, from the page-6 text: "gains above `30 dB` were observed on
the threshold of the half-harmonic. Small gain was obtained above threshold but
as soon as higher order subharmonics were created, **the gain disappeared
completely** and we in fact never obtained coexistent gain and chaotic noise."

This is four ODE states (junction phase and velocity, tuned-circuit charge and
current), so runs are seconds. The existing
`scripts/chaos/rcsj_single_junction.py` RK4 and spectral machinery is reused;
only the tuned circuit and the `Gamma` readout are new. It requires no box
counting, no fractal dimension, no `1e-8` period detection, and no universal
constants -- every failure mode that stopped Shukrinov is absent.

Levinsen's negative result must be reproduced along with the positive one:
applied white noise and signal are amplified together, giving constant noise
temperature and **no noise rise**. A reproduction that shows gain but not this
is incomplete.

### Diagnosis of the Phase 1 and Phase 2 failures (2026-08-12)

None of the three failed gates is a physical result. Each has an identified
instrument or protocol cause, recorded in the corrected success-criteria
sections of Phases 1 and 2:

| failed gate | cause | fix cost |
| --- | --- | --- |
| `D = 0.868` | box counting applied to the two-dimensional `(I,<V>)` curve instead of the one-dimensional complement of the locked steps; the measured object has `D` in `(1,2)` by construction | post-processing of existing CVC data |
| Feigenbaum `4.6692` | fixed point-clustering tolerance saturates at period `6` because branch splitting shrinks by `alpha ~ 2.503` per doubling; and `2000` settling units is `159` drive cycles, against a required `1e3`-`1e5` | re-scan at `1e-8` with adaptive clustering and longer settling |
| Guarcello Fig. 2(a)/(b) | run on `--power-convention paper`, which drives `63.8x Ic` and winds the junction phase to `35.3 rad`; the script's own docstring had already measured this as overdriven and defaulted to `50ohm` | re-run two sweeps on `50ohm` |

The Phase 3 device conclusions do not depend on defects 1 or 2, but their
`sigma_V'PS < 0.1` criterion is inherited from Phase 2 and is unvalidated until
Phase 2 passes.

### Phase 3 re-reduced on the ratio statistic (2026-08-12)

Re-reading `outputs/chaos/phase3/ladder_bracket20/campaign_summary.json`
against `sigma_max / sigma_min` within each sweep rather than against the
absolute `0.1`:

| damping | direction | `sigma` min | `sigma` max | ratio | Poincare clusters |
| --- | --- | ---: | ---: | ---: | --- |
| `inf` | up | `0.0199` | `0.0561` | `2.83` | 2 everywhere except `-24.25` (12) |
| `inf` | down | `0.0264` | `0.0578` | `2.19` | 2 at all 11 powers |
| `r1e04` | up | `0.0198` | `0.0561` | `2.83` | 2 except `-24.25` (12) |
| `r1e04` | down | `0.0263` | `0.0577` | `2.19` | 2 at all 11 powers |
| `r1e02` | up | `0.0167` | `0.0541` | `3.24` | 2 except `-24.25` (9) |
| `r1e02` | down | `0.0221` | `0.0497` | `2.25` | 2 at all 11 powers |
| `r1e00` | up | `0.00010` | `0.00360` | `36.40` | 2 except `-24.50` (15) and `-24.25` (3) |
| `r1e00` | down | `0.00010` | `0.00013` | `1.29` | 2 at all 11 powers |

**The `NO_BIFURCATION_FOUND` verdict is strengthened, not weakened.** For the
three weakly damped settings the ratio is `2.2`-`3.2x` against Guarcello's
`40`-`50x`, and `sigma` rises **smoothly and monotonically** with power over
`2.5 dB` rather than jumping inside `0.5 dB`. A smooth monotone rise in `sigma`
with drive is the same signature identified as an artifact in the
paper-convention Phase 2 sweep: `sigma` tracking drive amplitude, not attractor
structure.

**The damping ladder behaves as intended.** Baseline `sigma` falls
`0.020 -> 0.020 -> 0.017 -> 0.0001` across `inf`, `1e4`, `1e2`, `1`. Real
dissipation collapses the Poincare spread by about `200x` and the damped device
is period-2 across the whole bracket. That is the RCSJ shunt doing its job.

**One loose end.** `r1e00` up shows `36.4x`, but the entire ratio comes from a
single point at `-24.50 dBm`, which is the **first point of the up sweep**, with
`15` clusters and no down-sweep counterpart. Three reasons this reads as an
initialization transient rather than a bifurcation: it sits at the bottom of the
sweep and everything above it is clean, whereas Guarcello's jump sits at the top
and persists; the cluster counts at the isolated points do not reproduce across
damping (`12`, `12`, `9`, `3`), while a genuine period-`N` bifurcation would give
the same `N`; and the envelope-slope decay gate can pass on a marginally
decaying transient that still carries residual structure.

**Decisive cheap test:** restart the up sweep one or two powers lower. If the
anomalous point moves with the starting power it is initialization; if it stays
at `-24.50 dBm` it is a property of the state. This is one short run and it
closes the last open geometry question on 2c.

## Goal

Reproduce the chaos/bifurcation diagnostics of Shukrinov (2014) and Guarcello
(2024) against their own published acceptance numbers, then apply the same
diagnostics to `designs/rf_squid_2393_3wm` and `designs/ipm_2c_fixed` to
determine whether the high-power harmonic-balance wall is a bifurcation of the
periodic orbit, and if so which class.

## Current state analysis

### What the corpus supplies

`docs/development/chaos_papers/guarcello_jtwpa_fdtd.py` is a complete 798-line
numba implementation of Guarcello 2024 Appendix A, Eqs. (A14)-(A60): implicit
centered finite differences with a prefactored tridiagonal solve, both boundary
cells, the transparency-dependent CPR of Eq. (6), and `point` / `sweep-pump` /
`sweep-signal` / `sweep-bias` / `map-pump-bias` / `benchmark` subcommands. It
already computes gain, Poincare sections, and FFT spectra. Phase 2 runs this
script; it does not rewrite it.

### Why the corpus is relevant to the open question

The investigation recorded in
`docs/development/high_power_investigation_full_record_20260812.md` confirmed
that the period-1 orbit at `-23.421053 dBm` on 2c is dynamically unstable but
failed to identify the mode in four attempts, failed to fix a threshold power,
and found no route independent of time-domain integration.

Four specific alignments motivate this plan.

1. **Guarcello Fig. 2(a) describes the same phenomenon.** At `nu_p = 7 GHz`,
   `nu_s = 6.42 GHz`, `P_sign = -100 dBm`, `I_bias = 0`:

   | `P_pump` | gain | Poincare section |
   | --- | --- | --- |
   | `<= -54 dBm` | `<= 8 dB`, moderate | concentrated clusters |
   | `(-54, -53.5)` | jump to `~10 dB` | spreading |
   | `>= -53.5 dBm` | "quite high values" | chaotic |

   The paper states these high-gain conditions "do not actually give signal
   amplification". Our own retracted `66.9 dB` time-domain gain at high power
   was withdrawn for the same reason: the base state had already destabilized,
   so the linear-response subtraction was invalid.

2. **Levinsen (1982)** reports gain above `30 dB` near the half-harmonic
   instability, followed by collapse of coherent gain after further subharmonic
   bifurcation. The Themis `14.18.08` cube measures peak gain `8.4`-`33.2 dB`
   immediately before an abrupt total collapse in one `0.335 dB` pump step.

3. **Wiesenfeld & McNamara (1986)** supplies a classifier requiring no
   eigenvalue: gain diverges on approach to a codimension-one bifurcation, and
   the *resonance frequency* identifies the class. Period doubling gives peaks
   at odd half-harmonics of the drive; Hopf/Neimark-Sacker gives sidebands set
   by the imaginary part of the critical exponent; saddle-node and pitchfork
   give integer harmonics. The Themis cube is `51` pump frequencies x `31`
   powers x `2001` signal frequencies, and its `Response` field is a
   pump-on/pump-off ratio, so the test is calibration-free.

4. **The protocol itself is the fix for the four failed mode-identity runs.**
   Those runs perturbed the harmonic-balance orbit and analysed the *departure*,
   which is a diverging transient. Shukrinov and Guarcello both do the opposite:
   continue the control parameter carrying the final state forward as the next
   initial state, discard the transient, and classify the *attractor* reached.
   Period doubling turns one Poincare point into two; a torus produces a closed
   curve; chaos produces a scattered cloud. This is geometric and does not
   require an eigenvalue or a converged growth rate.

A fifth point is structural rather than physical. Guarcello's junctions carry
`RJ = 20 kOhm`, i.e. real per-junction dissipation, which is why their
second-order non-L-stable centered scheme behaves. `designs/ipm_2c_fixed` has
no dissipation except four `50 Ohm` port resistors
(`CircuitMatrices.has_loss` tests only `Im(C)`, `src/twpa_solver/core/circuit.py:106`),
which is why the Hill route was undecidable and the trapezoid integrator never
damped ramp-injected content. `src/twpa_solver/core/rcsj.py` is the enabling
piece for the transfer phase and is required, not optional.

### Existing infrastructure

| component | location | state |
| --- | --- | --- |
| two-frequency lattice basis | `src/twpa_solver/multitone/basis.py` | `ToneIndex(h,q)` for `h*omega_p + q*delta`, torus grid `n_p x n_delta`, `build_half_pump_basis` present. `delta` is a prescribed drive, not an unknown |
| period-doubled pump basis | `src/twpa_solver/pump/floquet.py::period_doubled_basis` | written, dormant |
| period-doubled continuation | `src/twpa_solver/pump/periodic_branch.py` | written, dormant |
| period-doubled gain | `src/twpa_solver/signal/period_doubled.py` | written, dormant |
| RCSJ shunt | `src/twpa_solver/core/rcsj.py::stamp_rcsj_shunt` | built 2026-08-12, characterized on `jc_jtwpa` only |
| transient integrator | `scripts/h1_transient_branch_transfer.py` | validated implicit-trapezoid path; BDF/Radau probes and failure records are retained |
| stroboscopic Poincare | `scripts/run_overnight_7p9_dynamics.py:373`, `scripts/chaos/rcsj_single_junction.py` | pump/state projection plus fixed-drive scalar section periods |
| Lyapunov exponent | `scripts/chaos/attractor_classify.py::largest_lyapunov_map` | implemented and exercised by both-direction power and RF-bias campaigns |
| Guarcello-form Poincare (`V'out` at `Vout=0`) | `scripts/chaos/attractor_classify.py::poincare_crossings` | implemented, unit-tested, and stored in campaign traces |

The dormant period-doubled path is gated by the standing rule in `CLAUDE.md`:
no new harmonic-balance ansatz until a tracked multiplier crossing is resolved,
timestep-converged, sideband-converged, and corroborated by an L-stable
transient run. This plan does not lift that gate.

### Device comparison

| | Guarcello 2024/2025 | `rf_squid_2393_3wm` | `ipm_2c_fixed` |
| --- | --- | --- | --- |
| cell type | rf-SQUID | rf-SQUID | JJ + `Cg` ladder |
| cells | 990 | 2393 | 2508 |
| `Ic` | `2 uA` | `0.93 uA` | - |
| `Cj` | `200 fF` | `20 fF` | - |
| loop inductance | `Lg = 120 pH` | `Lm = 58.6 pH` (`+ Lpar = 8.9 pH`) | - |
| `beta_L` | `0.729` (paper states `~0.74`) | `0.166` (`Lm`) / `0.191` (`Lm+Lpar`) | - |
| `f_plasma` | `27.74 GHz` | `59.82 GHz` | - |
| DC-bias scan | yes, Figs. 2(c), 3 | yes, via `Bphi` | no |
| dispersion engineering | none | `Cg_pattern` `[10.5, 68.2, 10.5, 50.4] fF`, counts `[6,6,6,6]` | - |

`beta_L` is quoted two ways because the loop contains `Lm` in parallel with
`Lpar + JJ` (`src/twpa_solver/builders/blocks.py:135-215`); which inductance
enters the screening parameter is a modelling choice and is recorded rather
than asserted.

`designs/rf_squid_2393_3wm.yaml` is referenced only by
`src/twpa_solver/design/schema.py:63,84`, `builders/blocks.py`,
`builders/registry.py:22` and `docs/design_format.md:390`. **No script has ever
solved it.** Phase 0 smoke-checks it before any later phase depends on it.

## What we're NOT doing

- Not implementing any new harmonic-balance ansatz. Phase 4 writes the gated
  plan for the class that Phase 3 measures; it does not enable it.
- Not reproducing Dixon (2019) CME-1 through CME-5 convergence. That is a
  basis-truncation question, partly answered already by `exp54` for 2c, and it
  needs WRspice or an equivalent unrestricted reference.
- Not reproducing Guarcello 2025 (second-harmonic CPR) or Guarcello 2026 (RPM)
  figures. The `tau` CPR knob exists in the supplied script and may be exercised
  opportunistically, but neither is a deliverable here.
- Not changing any production gain, compression, or P1dB number. No published
  result depends on anything in this plan.
- Not modifying `src/twpa_solver/` except for one optional additive CLI flag in
  Phase 3A, defaulting to off.
- Not re-opening the threshold-power bracket from the previous investigation.
  Classification does not require it.

## Prerequisites

- [x] `pip install numba` (dry run confirms `numba-0.67.0` + `llvmlite-0.49.0`,
      cp313 wheels, no numpy downgrade from the installed `2.5.1`)
- [x] All generated output under `outputs/chaos/` (ignored path). Nothing in
      this plan writes to `designs/`.
- [x] New code under `scripts/chaos/`.

---

## Phase 0: prerequisites and smoke checks

**Budget: ~15 min.**

### Overview

Install the dependency, measure the real integration rate so later sweeps are
sized from measurement rather than estimate, and prove the untested rf-SQUID
design builds and solves.

### Changes required

#### 1. Dependency

Install `numba`. Confirm `python -c "import numba"`.

#### 2. FDTD smoke test

**File**: `docs/development/chaos_papers/guarcello_jtwpa_fdtd.py` (run in place,
unmodified)

```
python docs/development/chaos_papers/guarcello_jtwpa_fdtd.py point --quick
python docs/development/chaos_papers/guarcello_jtwpa_fdtd.py benchmark --bench-steps 50000
```

Record `us_per_step` and `estimated_full_runtime_min`. **Every sweep point count
in Phase 2 is derived from this measured number.** The paper-scale point is
`2,000,000` steps at `dt_norm = 0.01`, `tmax_norm = 20000`.

#### 3. rf-SQUID design smoke test

**File**: `scripts/chaos/smoke_rf_squid.py` (new, ~60 lines)

Build `designs/rf_squid_2393_3wm.yaml`, report element and node counts, confirm
`Bphi` is present and non-empty (DC flux path), and run one
`solve_linear_scattering` at zero flux over `4`-`12 GHz` to confirm the line
transmits and to locate the dispersion features created by the `Cg_pattern`.

### Success criteria

**Automated**: `numba` imports; `point --quick` writes `summary.json`;
smoke script exits 0 and writes `outputs/chaos/phase0/rf_squid_smoke.json`.

**Manual**: `us_per_step` recorded; rf-SQUID `|S21|` is not identically zero and
shows the periodic stopband structure expected from a 24-cell `Cg` pattern.

---

## Phase 1: Shukrinov 2014 single-junction reproduction

**Budget: ~45 min.**

### Overview

Reproduce the driven-RCSJ chaos toolchain against **universal constants**, which
is the reason this phase comes first under the chosen approach. If the Lyapunov
exponent and fractal dimension come out right here, every downstream chaos claim
rests on a tested instrument rather than a new one.

### Changes required

#### 1. Single-junction integrator and diagnostics

**File**: `scripts/chaos/rcsj_single_junction.py` (new, ~350 lines)

**Model**: normalized RCSJ, Shukrinov Eqs. (1)-(2), the driven damped pendulum
in first-order voltage/phase form. Currents normalized by `Ic`, time by inverse
plasma frequency.

**Parameters** (paper's principal example): `beta = 0.3`, normalized radiation
frequency `omega = 0.5`, amplitude `A = 0.8`.

**Integrator**: fourth-order Runge-Kutta, fixed step `h = 1/32` (the paper's
value; do not substitute an adaptive integrator without recording the
difference).

**Continuation** — this is the step the dossier singles out as most likely to be
missed: sweep `I_dc` **downward**, starting above the critical-current region,
using the converged final state at each bias as the initial state for the next.
Transients `1e3`-`1e5` normalized units, averaging `1e4`-`1e5`, both CLI knobs.

**Diagnostics**:

| quantity | method |
| --- | --- |
| current-voltage characteristic | mean normalized voltage vs `I_dc` |
| largest Lyapunov exponent | tangent-vector propagation with periodic Gram-Schmidt renormalization |
| second Lyapunov exponent | second tangent vector, same scheme |
| Poincare section | phase-space sample at fixed drive phase |
| box-counting dimension | on the staircase/chaotic set |
| Feigenbaum ratios | Eqs. (6)-(7) from successive bifurcation spacings |

#### 2. Tests

**File**: `tests/test_chaos_rcsj_single.py` (new)

Pin the integrator on an analytically known limit (undriven overdamped
relaxation), and pin the Lyapunov estimator on a linear system with known
exponents before trusting it on the pendulum.

### Success criteria

**Automated**: `pytest tests/test_chaos_rcsj_single.py`

**Manual — absolute gates from the paper, no free parameters**:

| gate | target | source |
| --- | --- | --- |
| damping sum rule | `lambda_1 + lambda_2 = -beta` | stated in text |
| fractal dimension | `D = 0.868 +/- 0.012` | reported value |
| Feigenbaum delta | `-> 4.6692` | Eqs. (6)-(7) |
| chaos identification | `lambda_1 > 0` inside chaotic windows, `<= 0` on locked steps | Figs. |
| Poincare point count | equals the denominator of the rational locking ratio on a phase-locked step | Figs. |

#### Correction 2026-08-12: which set carries `D = 0.868`

The first implementation box-counted the two-dimensional `(I, <V>)` staircase
curve and returned `1.0582`, `1.5815`, `1.286` at successively finer current
spacing. That object has `D` in `(1, 2)` by construction and can never equal
`0.868` at any resolution.

The paper measures a different set (p. 5): "the staircase is complete if **the
set not covered by the steps** is of measure zero.  For the complete devil's
staircase, this corresponds to a universal fractal dimension of close to 0.87."
`D = 0.868` is therefore the box dimension of the **complement of the locked
steps, on the one-dimensional current axis**, and must be below `1`.

Required computation:

1. From the CVC, mark the locked intervals: contiguous `I` ranges where `<V>`
   is constant at a rational `m*omega/n`.
2. Form the complement of those intervals within `[I_start, I_end]`.
3. Box-count that one-dimensional set: `N(r)` = number of length-`r` intervals
   needed to cover it.
4. Fit the slope at the small-`r` end only. The paper uses "only the six points
   to the right of the figure".

This is post-processing of CVC data already on disk; it does not require a new
integration. Do not tune the fit range to reach `0.868` -- the earlier refusal
to do so was correct, and on the wrong set it would not have helped anyway.

#### Correction 2026-08-12: period detection and settling time

Table I lists the seven bifurcation currents, measured by the paper at
`Delta I = 1e-8` (the step is stated explicitly in Sec. V A):

| n | `I_n` | `delta_n` | `d_n` | `alpha_n` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.25756981` | - | `0.20707` | `2.956` |
| 2 | `0.26025001` | `6.3363` | `0.07004` | `2.505` |
| 3 | `0.26067300` | `4.7500` | `0.02796` | `2.503` |
| 4 | `0.26076205` | `4.6943` | `0.01117` | - |
| 5 | `0.26078102` | `4.6840` | - | - |
| 6 | `0.26078507` | `4.6713` | - | - |
| 7 | `0.26078594` | - | - | - |
| inf | `0.26078606` | - | - | - |

The orbit period below `I_1` is `3 tau`, doubling at each `I_n`: `6` on
`(I_1, I_2)`, `12` on `(I_2, I_3)`, `24` on `(I_3, I_4)`, `48` on `(I_4, I_5)`.

The recorded accumulation scan over `0.26070`-`0.26080` **contains `I_4`
through `I_7`**, so periods `48`, `96`, `192` and `384` fall inside it. All
1,001 samples returned period `6`. That is detector saturation, not absence of
a cascade. Two independent causes, each sufficient:

- **Fixed clustering tolerance.** The branch splitting `d_n` shrinks by
  `alpha ~ 2.503` per doubling (`0.207`, `0.070`, `0.028`, `0.011`), so an
  absolute point-merging tolerance loses every doubling past `n ~ 2`. Use
  scale-adaptive clustering, or detect doublings from subharmonic spectral
  lines (`f/2`, `f/4`, `f/8`), or test orbit closure directly at `2^n T`.
- **Settling and averaging far too short.** The scans used `2000 + 2000`
  normalized units. The paper specifies `1e3`-`1e5` "depending on the fine
  structure being resolved", and the cascade is the fine end. At `omega = 0.5`
  the drive period is `12.57`, so `2000` units is `159` drive cycles -- about
  `3.3` periods of a `48 tau` orbit. Critical slowing down near `I_inf` makes
  this worse.

Current resolution must also reach `1e-8` from `I_5` onward: the `I_6 -> I_7`
gap is `8.7e-7` and `I_7 -> I_inf` is `1.2e-7`.

---

## Phase 2: Guarcello 2024 990-cell FDTD reproduction

**Budget: ~60 min.**

### Overview

Run the supplied FDTD script against the paper's stated Fig. 2 and Fig. 3
acceptance numbers. Do not rewrite the solver.

### Changes required

#### 1. Convention pinning

**File**: `scripts/chaos/run_guarcello_repro.py` (new, ~200 lines) — a thin
driver and comparison wrapper around the supplied script.

Two ambiguities in the supplied implementation must be *measured*, not assumed,
because this project has twice been damaged by silently-chosen conventions
(Norton vs travelling-wave port power; the fabricated flat `72.5 dB` signal
loss).

| ambiguity | options | policy |
| --- | --- | --- |
| power convention | `paper` (anchors `-100 dBm -> 0.032 mV`) vs `50ohm` (`Vpk=sqrt(2 Z P)`) — the script's docstring notes these differ by `20 dB` in power | Run the reproduction gate on `--power-convention paper`, since the target is the paper's own figure. Record the `50ohm` number for the same point alongside. |
| port update | `stable` (integrating-factor, the script's default) vs `paper-centered` (literal Eqs. A59/A60) | Run **one** point both ways and report the difference. The default deviates from Appendix A to suppress a parasitic leapfrog mode; its effect must be quantified, not assumed negligible. |

#### 2. Figure 2(a) — pump sweep

```
sweep-pump --start -70 --stop -45 --num <N from Phase 0 rate> --workers <cores>
  --signal-ghz 6.42 --pump-ghz 7.0 --signal-dbm -100 --bias-ua 0
```

Then a refined second pass over `(-56, -52)` at higher density to resolve the
transition.

#### 3. Figure 2(b) — signal sweep (negative control)

```
sweep-signal --start 4 --stop 10 --pump-dbm -54.5 --num <N>
```

#### 4. Figure 3 — pump/bias map (optional, run only if Phase 2 is under budget)

```
map-pump-bias --pump-start -61 --pump-stop -53 --bias-start 0 --bias-stop 6
```

### Success criteria

**Automated**: each sweep writes `summary.csv`, `spectra_map.npz`,
`poincare_map.npz`, `overview.png`.

**Manual — gates from the paper text**:

| figure | gate |
| --- | --- |
| 2(a) | gain `<= 8 dB` for `P_pump <= -54 dBm` |
| 2(a) | gain jumps to `~10 dB` within `(-54, -53.5) dBm` |
| 2(a) | Poincare points concentrated below `-54 dBm`, spread above `-53.5 dBm` |
| 2(a) | `sigma_V'PS << 0.1` stable, `>= 0.1` chaotic (Fig. 3 criterion) |
| 2(b) | `~8 dB` across `nu_s` in `[6, 8] GHz` |
| 2(b) | ripple amplitude `~0.2 dB` |
| 2(b) | **Poincare stays dense across the whole sweep** — frequency alone does not induce chaos |
| 3 | re-entrant stable, high-gain window at `I_bias` in `(4, 5) uA` |

2(b) is the load-bearing negative control. A pipeline that reports chaos when
only the signal frequency changes is broken regardless of what 2(a) shows.

If the pump thresholds land within roughly `1 dB` of `-54`/`-53.5`, treat the
reproduction as successful and record the offset. An exact match is not
expected: the paper does not state its sweep point counts or its
transient-discard fraction, and both are exposed as CLI options in the supplied
script precisely because they were not published.

#### Correction 2026-08-12: the gate runs on `50ohm`, not `paper`

**The instruction above to gate on `--power-convention paper` was wrong and is
withdrawn.** The supplied script's own docstring already records the reason and
sets its default accordingly: the paper's stated `-100 dBm -> 0.032 mV`
anchoring "is 20 dB larger in power than the standard 50-ohm convention and a
literal use strongly overdrives the translated model". That was a measurement by
the script's author, overridden here by an argument from principle.

Measured at `P_pump = -54 dBm`, `nu_s = 6.42 GHz`, `tmax_norm = 20000`:

| convention | `V_pk` | drive current | vs `Ic = 2 uA` | gain | `max abs phi` | `sigma_V'PS` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper` | `6.385 mV` | `127.70 uA` | `63.8x` | `-6.265 dB` | `35.255 rad` | `0.7696` |
| `50ohm` | `0.631 mV` | `12.62 uA` | `6.3x` | `+3.184 dB` | `1.544 rad` | `0.0946` |

`35.255 rad` is `5.6` full phase windings: the junctions are slipping
continuously, and no parametric-amplifier interpretation applies. The spectrum
at that point is a `7 GHz` carrier with a `+/-17.5 MHz` sideband comb, with no
`6.42 GHz` signal line and no `7.58 GHz` idler among the twelve strongest
components.

The `50ohm` point is physical and lands `sigma_V'PS = 0.0946`, just below the
paper's `0.1` cut, at the power where the paper places the transition.

Two further indicators confirm the `paper`-convention sweep carried no
dynamical information: `poincare_count` was `803` at all 51 powers (equal to
`401.6` cycles of `7 GHz` in the analysis window, i.e. it counted pump zero
crossings), and `sigma_V'PS` rose smoothly and monotonically from `0.2036` to
`2.1844` across `25 dB` with no jump. It was tracking pump amplitude, not
attractor structure.

Re-run `fig2a_full` and `fig2b_full` on `--power-convention 50ohm`. Retain the
`paper` numbers as the overdrive control. Record the `paper`/`50ohm` pair at one
point in the convention audit, as originally specified -- that part stands.

This correction also matters to Phase 3: Phase 2 is what validates carrying the
paper's `sigma_V'PS = 0.1` threshold onto our own devices. Until Phase 2 passes
on `50ohm`, the `0.1` cut used in the 2c and rf-SQUID campaigns is unvalidated.

---

## Phase 3: transfer to `rf_squid_2393_3wm` and `ipm_2c_fixed`

**Budget: ~60 min interactive, plus an optional overnight extension.**
This phase is past the "couple of hours" reproduction budget and is where the
open question is actually addressed.

### Overview

Port the validated diagnostics onto our own transient solver, switch to the
attractor-continuation protocol, and classify.

### Changes required

#### 1. Diagnostic port

**File**: `scripts/chaos/attractor_classify.py` (new, ~400 lines)

Reuse `scripts/h1_transient_branch_transfer.py` for integration. Add:

- **Guarcello-form Poincare**: `V'out` sampled at every `Vout = 0` crossing,
  with linear interpolation of the derivative at the crossing (the supplied
  script's `poincare_crossings` is the reference implementation). This is not
  the same object as the existing stroboscopic pump-flux projection at
  `run_overnight_7p9_dynamics.py:373`, and both should be emitted.
- **`sigma_V'PS`** as the scalar chaos indicator, with the paper's `0.1`
  threshold carried over as the initial cut and re-derived per device.
- **Largest Lyapunov exponent**, reusing the Phase 1 estimator.
- **FT map** versus the control parameter.

RCSJ damping via `src/twpa_solver/core/rcsj.py::stamp_rcsj_shunt` is **required**
for 2c. Sweep the resistance ratio and report every classification as a function
of it; a verdict that only holds at one damping value is not a verdict.

**Optional additive flag** in `h1_transient_branch_transfer.py`: expose the raw
output-voltage trace at full sample rate for external analysis, defaulting to
off. This is the only permitted touch to existing code in this plan.

#### 2. Protocol change

**Attractor continuation**, replacing "perturb the HB orbit and measure the
departure":

1. Start at a power well below the wall, from the converged HB orbit.
2. Integrate until the transient has decayed (envelope-slope criterion, the
   discriminant already adopted in the previous investigation, `1e-5 /period`).
3. Record the attractor's Poincare section, spectrum, and Lyapunov exponent.
4. Step the control parameter up by a small increment, **carrying the final
   state forward as the initial state**.
5. Repeat through and past the wall.

Both continuation directions must be run. Hysteresis between up and down sweeps
is itself diagnostic and distinguishes a subcritical bifurcation from a
supercritical one.

#### 3. Device campaigns

**`designs/rf_squid_2393_3wm`** — the structural twin. Runs Guarcello's own two
scan axes directly:

- pump-power scan at fixed `nu_s` and zero bias (Guarcello Fig. 2(a) analogue)
- **DC-bias scan** at fixed pump (Guarcello Fig. 2(c) analogue) — 2c cannot do
  this at all

Before running, predict the operating point from Guarcello Eq. (4),
`beta = (beta_L/2) sin(phi_dc)`, `gamma = (beta_L/6) cos(phi_dc)`, using
`beta_L` in `[0.166, 0.191]` against their `0.729`. Record the prediction first,
then the measurement. A `4.4x` weaker screening parameter against `2.4x` more
cells is a specific, falsifiable expectation about where the transition should
move.

**`designs/ipm_2c_fixed`** at `f_p = 7.9 GHz`, pump port 4, signal `7.4 GHz`,
source port 1, output port 2 — the device with the open question and the
hardware. Power range bracketing `-23.421053 dBm`.

### Success criteria

**Automated**: `pytest tests/test_chaos_attractor_classify.py` — pin the
Poincare crossing detector and `sigma` against the supplied script's
implementation on a synthetic trace with a known period.

**Manual**: each device and each continuation direction returns exactly one of:

| verdict | Poincare evidence | spectral evidence |
| --- | --- | --- |
| `PERIOD_DOUBLING` | one point becomes two | line appears at `f_p/2` |
| `NEIMARK_SACKER` | points fill a closed curve | incommensurate line, not `f_p/2` |
| `CHAOS_NO_CLEAN_BIFURCATION` | scattered cloud, no intermediate structure | broadband |
| `NO_BIFURCATION_FOUND` | stays compact through the wall | stays discrete |

Poincare geometry is primary, spectrum is corroborating. The previous
investigation's four failures were all spectral-only on a diverging trace; a
spectral claim without matching Poincare geometry is not a verdict here.

`NO_BIFURCATION_FOUND` on 2c is a real possible outcome and would mean the wall
is numerical after all, matching the `jc_jtwpa` finding.

---

## Phase 4: Themis Wiesenfeld-McNamara test and gated solver plan

**Budget: ~30 min.**

### Overview

Test the bifurcation class against hardware, calibration-free, then write the
solver change for whichever class Phases 3 and 4A agree on. Per the chosen
scope, the change is specified but not implemented.

### Changes required

#### 1. Wiesenfeld-McNamara frequency classifier

**File**: `scripts/chaos/themis_wm_classifier.py` (new, ~250 lines)

**Input**: `docs/development/14.18.08_Themis_SetupAug25_noVTS_transmission_15mK/*.npy`
(51 files, `np.load(..., allow_pickle=True).item()`, keys `Frequency (2001,)`,
`Response (31, 2001)`, `PumpPower (31,)`, `SignalPower` scalar).

**Method**: for each pump frequency, take the amplifying pre-collapse power
points; track the peak-response signal frequency as power approaches collapse;
test which Wiesenfeld-McNamara prediction the peak location follows.

| bifurcation class | predicted peak location |
| --- | --- |
| period doubling | odd half-harmonics of `f_p` |
| Hopf / Neimark-Sacker | sidebands at `Im(critical exponent)`, drifting with power |
| saddle-node / pitchfork | integer harmonics of `f_p` |

**Band caveat, stated up front**: at `f_p = 7.9 GHz`, `f_p/2 = 3.95 GHz` lies
*below* the measured `4`-`12 GHz` span. The third half-harmonic
`3f_p/2 = 11.85 GHz` is inside it. A period-doubling signature is therefore
testable, but only at the third half-harmonic, and absence at `f_p/2` is not
evidence of anything.

**Secondary, already partly measured**: the previous investigation established
that the measured device follows `G ~ (1 - I/I_th)^-2`, with `1/sqrt(G_lin)`
extrapolating to the observed collapse power within `0.02`-`0.28 dB` at three of
four frequencies. Wiesenfeld-McNamara predicts the *exponent* of the divergence
per class; compare the fitted exponent against the prediction, not just the
threshold location.

This phase is free of the port-power convention and the line-loss model, because
`Response` is a pump-on/pump-off ratio. That is the whole reason it is worth
doing.

#### 2. Gated solver plan

**File**: `docs/development/chaos_route_solver_plan.md` (new)

Written after Phases 3 and 4A report. Branches on the agreed class:

| class | solver change | build cost |
| --- | --- | --- |
| `PERIOD_DOUBLING` | lift the `CLAUDE.md` ansatz gate; use existing `pump/floquet.py::period_doubled_basis`, `pump/periodic_branch.py`, `signal/period_doubled.py`, `scripts/run_period_doubled_branch.py` | code exists, gate flip |
| `NEIMARK_SACKER` | auxiliary-generator closure on the existing `multitone` `(h,q)` lattice: promote `delta` to an unknown autonomous frequency, adding two real unknowns `(A_a, omega_a)` and two real equations `Y_AG = 0` in an outer loop. This is two-frequency harmonic balance for quasi-periodic continuation | one new module |
| `CHAOS_NO_CLEAN_BIFURCATION` | no ansatz represents a chaotic attractor. The correct change is a validity-boundary detector that halts HB and reports the boundary as physics rather than as a convergence failure | one new diagnostic |
| `NO_BIFURCATION_FOUND` | no ansatz change. Wall is numerical; the existing continuation-recovery ladder is the right tool | none |

If Phase 3 and Phase 4A **disagree**, that disagreement is the result and is
recorded as such. Do not average them or pick the convenient one.

### Success criteria

**Automated**: `pytest tests/test_themis_wm_classifier.py` — pin the cube loader
and the peak-tracking on a synthetic response with an injected known peak.

**Manual**: classifier returns a class or an explicit `UNDETERMINED` with the
reason (most likely: the `0.335 dB` power grid is too coarse to resolve the
approach scaling, which is a real and expected possibility). The solver plan
document exists and names exactly one change.

---

## Testing strategy

### Project maturity level

**Active Development.** Coverage target ~70% on new modules; every new
diagnostic gated by a test that has been shown failing first.

### Unit tests

- `tests/test_chaos_rcsj_single.py` — RK4 integrator against an analytic
  overdamped limit; Lyapunov estimator against a linear system with known
  exponents; Poincare point count on a synthetic period-3 orbit; period-
  doubling bracket extraction from a synthetic measured sequence.
- `tests/test_chaos_attractor_classify.py` — crossing detector and `sigma`
  against the supplied script's implementation on a synthetic trace of known
  period; classification verdicts on synthetic period-1, period-2, quasi-periodic
  and chaotic traces.
- `tests/test_themis_wm_classifier.py` — cube loader on a fixture; peak tracking
  with an injected known peak location.

- `tests/test_chaos_rf_bias.py` - sampled-period Lyapunov map and branch-state
  restoration for the RF-SQUID bias driver.

Every estimator is pinned on a signal whose answer is known analytically before
it is applied to any circuit. The previous investigation's central failure mode
was instruments that produced numbers without ever having been shown correct.

### Integration and manual tests

- Phase 1 universal-constant gates (`lambda_1 + lambda_2 = -beta`,
  `D = 0.868 +/- 0.012`, Feigenbaum `4.6692`) are the integration test for the
  whole chaos toolchain.
- Phase 2 Fig. 2(b) is the negative control: no chaos from a frequency sweep.
- Phase 3 requires both continuation directions on both devices.
- `jc_jtwpa` is available as a further negative control if wanted; its wall was
  already shown numerical, so it should return `NO_BIFURCATION_FOUND`.

### Guarcello multi-tone estimator correction (2026-08-12)

The earlier Fig. 2(b) **Outcome B** conclusion is **WITHDRAWN**. It was caused
by the gain estimator, not by the device response. The old estimator projected
the output onto the signal frequency alone, allowing finite-record leakage from
the approximately 40 dB larger pump to enter the signal phasor. On the same
`6.760 GHz`, `-57 dBm` waveform, the old gain changes by `10.075 dB` across
half, nominal, and double steady-state windows, whereas the simultaneous
multi-tone estimator changes by `0.00499 dB`. The corresponding
`sigma_upward` and `max_abs_phi_last_recorded` observables remain smooth and
flat. This three-column evidence identifies estimator leakage as the root cause.

`exact_tone_amplitude` remains available as the legacy regression function.
`multitone_amplitude` uses one `numpy.linalg.lstsq` fit containing DC, the
signal, pump harmonics through `5 f_p`, and `f_p +/- f_s` and
`2 f_p +/- f_s`, dropping out-of-Nyquist and one-record-resolution collisions.
Each run records retained and dropped basis frequencies. The synthetic 43 dB
pump-to-signal test fails with the old estimator by `23.26%` relative and
recovers the signal with the new estimator at `1.35e-13` relative error.

Each new run emits both narrowband `gain_db` and `gain_wideband_db`. The latter
sums `spectrum_dbm()` power in `f_s +/- 0.5 GHz`, with a pump-harmonic notch
half-width of `2/T`; both widths are recorded in the analysis output. The
pump-off reference remains
`outputs/chaos/phase2/pump_off_reference_50ohm.json`.

The former `gain_le_8_db_to_minus54` gate is **RETIRED**. The digitized paper
value at `-54 dBm` is `9.0 dB`, so that gate encoded an eyeball threshold rather
than a valid acceptance condition. The replacement gate is the below-transition
mean residual against the digitized curve (`<= 1 dB`), plus transition location,
directional broadening, phase amplitude, narrowband/wideband agreement, and
the smooth Fig. 2(b) response.

The Guarcello PDF was rendered with PyMuPDF before replotting. Fig. 2(a) has
gain in dB, `V'_PS` over approximately `0--0.4`, and a Fourier-spectrum
frequency axis in GHz. The `0.116--0.126` range belongs to Fig. 4's mean-gamma
axis, not Fig. 2(b)'s Poincare panel. Fig. 4 defines
`beta = beta_L/2 sin(phi_dc)` and `gamma = beta_L/6 cos(phi_dc)` and plots
their means and standard deviations; its pump axis spans `-70` to `-45 dBm`
and its bias axis spans `0` to approximately `34 microampere`.

The literal paper-centered port update remains closed as an unstable scheme:
it produced NaN at `-60`, `-57`, and `-55 dBm`, and also diverged at
`dt_norm = 0.01`, `0.005`, and `0.0025`. The stable update is therefore
required; this is a measured numerical-stability result, not a fitting choice.

The corrected 51-point sweeps are in
`outputs/chaos/phase2/fig2a_50ohm_mtls/run/` and
`outputs/chaos/phase2/fig2b_50ohm_pump_m57_mtls/run/`. Their original gate results are:

| gate | measured result | status |
| --- | --- | --- |
| G1 | mean residual `0.323 dB`, maximum `0.733 dB` over `-70..-54 dBm` | PASS |
| G2 | first upward cluster rise at `-53.5 dBm` | PASS |
| G3 | `sigma_up = 0.000820` at `-54.5 dBm` to `0.019769` at `-53.5 dBm`, ratio `24.10x` | PASS |
| G4 | `max abs(phi) = 1.54395` at `-54 dBm` | PASS |
| G5/B3 | maximum adjacent Fig. 2(b) narrowband gain step `0.276 dB` | PASS |
| B4 | peak Fig. 2(b) narrowband gain `4.697 dB` | PASS |
| G6 | maximum wideband/narrowband difference below transition `1.885 dB` | FAIL |

G6 fails with the explicitly fixed `f_s +/- 0.5 GHz` band and `2/T` pump
notches; the band definition is therefore not yet validated and was not tuned.
The Fig. 2(b) smoothness gate passes, directly resolving the former estimator
failure. In the chaotic region, the wideband observation is approximately
`17.7--30.5 dB`: some points overlap the paper's `14--20 dB` band, but the
upper excursions do not, so the definition mismatch explains part, not all,
of the prior deficit.

The new paper-layout plots are
`outputs/chaos/phase2/plots_mtls/fig2a_mtls_three_panel.png` and
`outputs/chaos/phase2/plots_mtls/fig2b_mtls.png`. The Fig. 2(a) middle panel
now plots retained upward-branch values, not sigma; sigma remains available
only in the separate reduction JSON and diagnostic plots.

The amended bifurcation checks give `V'_PS` branch means of `0.012405` at
`-70 dBm` and `0.094559` at `-54 dBm`, so the Fig. 2(a) stable branch lies in
the paper's approximately `0--0.4` panel range. The `f_p/2` spectral bins at
`-56.0`, `-54.5`, `-53.5`, and `-52.0 dBm` are respectively `-211.03`,
`-209.41`, `-153.91`, and `-112.66 dBm`. Relative to the local spectral
medians, the measured separations are approximately `0.00`, `0.00`, `2.36`,
and `0.15 dB`. That earlier B2 **FAIL** is withdrawn: it used an incorrect
floor-relative denominator. The corrected pump-referred values rise by about
`66 dB` for `f_p/2` and `75 dB` for `3f_p/2` across the transition, so B2 is
**PASS** and the period-doubling attribution is reproduced.

### Addendum 4 execution results

The required Fig. 2(b) pump power is `-55 dBm`. The new sweep is in
`outputs/chaos/phase2/fig2b_50ohm_pump_m55_mtls/run/`; the `-57 dBm` control is
retained. After masking `7.00 GHz`, the `-55 dBm` sweep has peak narrowband
gain `8.381 dB`, 6--8 GHz mean `7.337 dB`, and maximum adjacent step
`0.684 dB`. B3 and B4 **PASS**. No signal-frequency point has an upward
cluster count above one. The prior Fig. 2(b) branch-magnitude B1 failure is
**VOID**, because `0.116--0.126` is a gamma axis, not a Poincare axis.

The Fig. 4 zero-flux gamma check **PASSES**: analytic `gamma_0 = 0.1215433`
and measured pump-off `gamma_mean = 0.1215412677`. The pump-side gamma falls
from this verified value to `0.00645` by `-45 dBm`; the paper curve also shows
the same qualitative collapse. B5 **PASS qualitative collapse**. The completed
0--34 microampere bias sweep at `-55 dBm` is in
`outputs/chaos/phase2/fig4_bias_0_34_m55/`; beta changes sign repeatedly and
the beta/gamma structures repeat with period about `17.1--17.2 microampere`.
The 17.1 microampere-shift RMSE is `0.00449` for beta and `0.00141` for gamma.
B6 **PASS** and B7 **PASS**.

The bias derivation confirms the existing audited core. Rearranged A42 gives
`I_i + I_bias = qdot_0 + I_1`; the left phase row therefore receives
`+(C_1/C_0) I_bias`, matching `+cminus[0]*bias_a`. Rearranged A55 gives
`C_N R_l I_dot_l = -(1+C_N/C_l) I_l - I_bias + I_N`; the right phase row
therefore receives `+I_bias`, matching the code. The two port current updates
retain `-bias_a`. No sign or coefficient change is warranted.

Poincare sampling Gate B9 **PASSES**. At `-70 dBm`, sigma values for
stride-20 nearest, stride-1 nearest, and stride-20 linear are `0.000265776`,
`0.000265097`, and `0.000264860`; the extrema ratio is `1.0035`. At
`-54.5 dBm` they are `0.000695825`, `0.000819412`, and `0.000820377`.

The fine scan in `outputs/chaos/phase2/transition_scan_m54_to_m53p4.json`
finds a clean period-2 window at `-53.75 dBm`: both half-harmonic lines are
strong while the broadband floor remains within about 2 dB of its pre-transition
pump-referred level. At `-53.70 dBm` the broadband floor has already risen by
about 25 dB. B8 **PASS**, with the clean window bounded at the scan resolution
between `-53.75` and `-53.70 dBm`.

Bandwidth identification over 25, 50, 100, 200, 300, and 500 MHz found no
Fig. 2(a) band satisfying G6's `0.3 dB` below-transition constraint. The
smallest deviation was `1.739 dB` at 25 MHz. No chaotic-region bandwidth
prediction is accepted; the bandwidth was not tuned to the paper.

The FDTD integrator retains `fastmath=True` for the 2e6-step numba kernel.
This prevents bitwise chaotic-trajectory reproduction and can shift a measured
transition slightly. It is a known limitation and was not disabled.

## Rollback plan

Everything new is additive: `scripts/chaos/`, `tests/test_chaos_*.py`,
`outputs/chaos/`, and two new documents under `docs/development/`. No production
module under `src/twpa_solver/` is modified except one optional CLI flag in
`scripts/h1_transient_branch_transfer.py` defaulting to off.

Rollback is `git revert` of the phase commits plus deletion of
`outputs/chaos/`. No published gain, compression, or P1dB number depends on any
of it, and the harmonic-balance ansatz gate in `CLAUDE.md` remains in force
throughout.

The one externally visible change is the `numba` dependency. It is used only by
the corpus FDTD script and nothing in `src/twpa_solver/` imports it.

## Phase 5 JC sparse-transient-versus-HB comparison (2026-08-12)

The phase-5 entry point is implemented at
`scripts/chaos/run_guarcello_jc_phase5.py`. The method attribution is
**our sparse transient engine versus our harmonic-balance solver**, not
Guarcello's FDTD algorithm. It reads the untouched JC
documentation exports and HB CSVs and writes only under `outputs/chaos/phase5/`.
The JC circuits are not the paper's rf-SQUID ladder: `jc_jtwpa` has 2047
uniform Josephson branches with an every-fourth-cell resonator profile, while
`jc_fqjtwpa` has 1999 non-uniform Josephson branches with an every-eighth-cell
resonator profile and Gaussian junction weighting. Their port matrices contain
50 ohm source and load resistors. This is a numerical comparison against
JosephsonCircuits.jl documentation designs, not external physical validation,
and it must not be transferred to `designs/`.

The runner therefore uses the exact sparse transient engine for these explicit
JC topologies rather
than silently projecting them onto the paper ladder. It records requested and
achieved pump current, pump-off-referenced narrowband gain, the identified
25 MHz minimum admissible wideband half-bandwidth, all-cell `r_j`, minimum
`cos(phi)`, strongest-branch index, and late-time recurrence diagnostics.

The time-budget check found that the paper's `tmax_norm=20000` does not retain
300 pump periods for either JC device. The effective holds were raised to
`36522.583` for JTWPA and `34686.060` for FQJTWPA, with 6087 and 5781
integrated steps per pump period respectively. The estimates are about 216
s/point and 171 s/point before sparse Newton overhead, recorded in
`outputs/chaos/phase5/jc_jtwpa_cost_estimate.json` and
`outputs/chaos/phase5/jc_fqjtwpa_cost_estimate.json`.

The JTWPA campaign was started after the cost check but stopped after more
than five minutes while still computing its pump-off reference; no point
result had been written. The full JC comparison is therefore **UNRESOLVED**:
M-G1 through M-G4 have no measured verdicts, and no claim is made about the
HB wall. The implementation and descriptor tests pass (`3 passed`), but the
current sparse transient cost must be reduced or explicitly scheduled as a
long-running measurement before reporting FDTD-vs-HB physics.

### Phase 5 bounded-integrator diagnosis and rescope (2026-08-13)

The previous phase-5 launch used the unbounded
`implicit_trapezoid_ramp`, which retains every full Newton state. That caused
the multi-gigabyte memory growth. The runner now uses the existing
`implicit_trapezoid_ramp_bounded` path with bounded samples and history; no
topology or device parameter was changed.

The invalid linear extrapolation from the 25-second Guarcello tridiagonal
benchmark was removed. A measured 200-step JTWPA benchmark is recorded in
`outputs/chaos/phase5/jc_jtwpa/measured_rate_200_steps.json`: the cold run
measured `280.45` accepted steps/s and the repeat after compilation measured
`348.50` accepted steps/s. A separate process monitor measured peak working
set `175050752` bytes (`166.99 MiB`). At the current 6087 steps per pump
period, the required 600-period hold is about `3652258` steps, or roughly
2.9--3.6 hours per point at those measured rates. This is a measured runtime
limit, not an analytic estimate.

The campaign was stopped before any long point. The three-point JTWPA rescope
and the bounded P1 comparison of implicit trapezoid, BDF, and Radau remain
pending because even the measured bounded rate makes the required hold exceed
the available runtime. No FQJTWPA point was started. Phase-5 scientific gates
M-G1 through M-G4 remain **UNRESOLVED**.

### Phase 5 Guarcello-algorithm correction (2026-08-13)

The sparse implicit-trapezoid interpretation was withdrawn. Phase 5 now uses
Guarcello's defining known-time-level scheme: the Josephson current is
evaluated from `phi^m`, the linear node matrix is constant, and it is factored
once in natural-order banded storage. No Newton iteration, BDF/Radau
comparison, or sparse-engine attribution is part of this deliverable.

The measured JC matrices are both naturally banded with bandwidth 2: JTWPA
has 2560 nodes and FQJTWPA has 2250 nodes. FQJTWPA's non-uniform `Ic` profile
is retained from `ipm_arrays.npz`; JTWPA is confirmed uniform. The 200-step
Guarcello benchmark measured `2559.38 steps/s` for JTWPA and `2919.07 steps/s`
for FQJTWPA. The results are in
`outputs/chaos/phase5/jc_jtwpa/measured_rate_200_steps.json` and
`jc_fqjtwpa/measured_rate_200_steps.json`.

The previous sparse-engine runtime and memory findings remain a documented
withdrawn approach, not a Phase-5 device result. The JTWPA-only P1/P2/P3
campaign is now launched using the Guarcello banded algorithm; FQJTWPA is
deferred until all three JTWPA rows exist.

The campaign was then explicitly stopped at PID `29684` before a complete
point row was available. The runner now persists `trace.npz` per completed
point with `t` and `v_out`, and records `record_stride`, `dt_s`, `n_steps`,
and `steady_state_start_index` in the row. No JTWPA point has a trace yet;
the stopped run therefore has no reduction-side gain data to reprocess.
