OVERNIGHT 7.9 GHz 2c DYNAMICAL-REGIME CAMPAIGN
==============================================

You have approximately 8 hours of unattended runtime.

Execute the experiment autonomously. Do not stop to ask questions unless continuing would risk modifying trusted production code or destroying existing artifacts.

The goal is NOT merely to generate another power sweep.

The goal is to obtain, by morning, the strongest practical answer to:

1. Over -35 to -15 dBm at 7.9 GHz, what long-time state is selected when the device is independently turned on from zero pump?

2. At what pump power does the normal ramp-selected PERIOD1 state cease to be the operating state?

3. What dynamical state replaces PERIOD1:
       PERIOD2 / other PERIOD_N
       quasiperiodic
       persistent broadband/chaotic candidate
       running phase
       unresolved long transient

4. Is there sufficient evidence to justify extending ordinary pump-period HB to a specific nonlinear HB ansatz such as:
       2T / fp/2 HB,
       NT / fp/N HB,
       or two-frequency/quasiperiodic HB?

5. Are the conclusions robust to initialization, ramp rate, hold length, and TD timestep?

This is an experiment and classification campaign, NOT a solver-development campaign.


TRUSTED CONTEXT
===============

Circuit:
    designs/ipm_2c_fixed

Pump frequency:
    7.9 GHz

Pump port:
    4

Power range allowed:
    -35 to -15 dBm

Correct upward HB result currently trusted:

    HB valid from -35 dBm through -24.4736842105 dBm.

    First ordinary HB failure:
        -23.4210526316 dBm

    At failure:
        coefficient residual ~5.57e-2
        stalled Newton iteration 6

At -24.473684 dBm, three 250-period matched TD tests at Delta-theta = 0.05 gave:

DIRECT HB INITIALIZATION
    raw: PERIOD_1
    decay-aware: PERIOD_1
    late d1 ~1.92e-4
    peak |IJ|/Ic ~0.553

LOWER-POWER HB WARM INITIALIZATION
    raw: QUASIPERIODIC_OR_PERIOD_N
    decay-aware: PERIOD_1
    late d1 ~2.25e-4
    peak |IJ|/Ic ~0.564

ZERO-PUMP UPWARD RAMP
    raw: QUASIPERIODIC_OR_PERIOD_N
    decay-aware: PERIOD_1
    late d1 ~5.00e-4
    peak |IJ|/Ic ~0.552

All had negligible phase winding and successful integration.

The plot shows that warm-HB and especially zero-pump trajectories are still decaying toward the direct-HB recurrence floor.

Therefore:

    DO NOT use a finite-tail raw d1 threshold alone.

Decay trend matters.


VERY IMPORTANT: SUPERSEDED RESULTS
==================================

A previous campaign carried a nonperiodic high-power TD state DOWN through successively lower powers.

That experiment is a hysteresis / attractor-persistence experiment.

It must NOT be used to define the normal upward-turn-on operating boundary.

Preserve those files, but do not use their classifications for this primary campaign.

Also ignore previously superseded synthetic high-power checkpoints/results.


PRIMARY PHYSICAL PROTOCOL
=========================

The primary classification at EVERY pump power must be an independent experiment:

    zero-pump equilibrium
        ->
    pump ramped UP from zero to target power
        ->
    fixed target power
        ->
    long-time classification

Each target must restart independently.

NEVER use:

    final TD state at P_i
        ->
    P_(i+1)

for the primary map.

After each target is classified, discard its final state as an initializer for the next primary target.

Record explicit provenance proving this.


NORMAL TURN-ON RAMP
===================

Use the presently validated zero-pump upward-ramp protocol as baseline.

Use:

    ramp = 40 pump periods

unless inspection of the existing validated implementation shows a different exact protocol was used in the successful -24.473684 comparison.

Match that established protocol exactly.

Do not silently change ramp shape, phase, source normalization, timestep, or fixture.


BASE TRANSIENT SETTINGS
=======================

Use:

    Delta-theta = 0.05

because it is the setting of the latest successful matched comparison.

Use compact storage.

Use existing validated implicit transient implementation.

Do not introduce new physics.

Do not add Rj/damping.

Do not modify junction model.

Do not change circuit parameters.

Do not modify HB/preconditioner internals.

Do not "fix" convergence issues during this campaign.

Peak RSS per child has been ~240 MB; keep memory bounded and monitor it.


TIME BUDGET
===========

Hard campaign wall-clock budget:

    ~7.5 hours active work

Reserve the final ~30 minutes for:

    aggregation
    plots
    consistency checks
    final report

Do not let one pathological point consume the whole night.

Use adaptive stopping and move on when a classification has reached the campaign confidence limit.


PHASE A — WHOLE-RANGE INDEPENDENT TURN-ON MAP
==============================================

First establish the broad dynamical topology from -35 to -15 dBm.

Do NOT spend maximum hold time everywhere.

Use a coarse set such as:

    -35
    -33
    -31
    -29
    -27
    -26
    -25
    -24.5
    -24.0
    -23.5
    -23.0
    -22.5
    -22.0
    -21.0
    -20.0
    -19.0
    -18.0
    -17.0
    -16.0
    -15.0 dBm

You may modify exact powers slightly to reuse meaningful existing HB grid values, but cover the whole range.

Every point:

    zero pump
    -> 40-period upward ramp
    -> adaptive fixed-drive hold

At checkpoints approximately:

    40
    90
    140
    250
    440 periods of hold

evaluate classification.

If a trustworthy PERIOD1 recurrence floor is reached early and remains stable:
    terminate early.

If sustained RUNNING_PHASE is robustly detected:
    terminate early.

If still nonperiodic but valid:
    continue toward 440.

If trajectory remains clearly evolving at 440:
    mark UNRESOLVED_LONG_TRANSIENT rather than forcing a label.

For every point save:

    power dBm
    actual source current
    initialization provenance
    ramp information
    hold length
    integrator success
    timestep statistics
    d1/d2/d3 histories
    peak and late-window |IJ|/Ic
    max-utilization JJ identity
    phase winding
    runtime
    peak RSS
    decay-aware classification
    raw classification


PHASE B — REFINE ALL REGIME TRANSITIONS
=======================================

After the coarse sweep, automatically identify every adjacent pair whose classifications differ materially.

Examples:

    PERIOD1 -> non-PERIOD1
    PERIOD_N -> quasiperiodic
    quasiperiodic -> broadband
    bounded nonperiodic -> RUNNING_PHASE

Refine each transition.

For the first loss of PERIOD1, priority is highest.

Use ~0.25 dB spacing first and then ~0.1 dB or bisection where monotonic.

Do NOT assume the entire power response is monotonic.

If:

    PERIOD1
    nonperiodic
    PERIOD1

appears as power increases, preserve it and investigate as a possible island/window rather than forcing bisection.

Aim for:

    <= 0.1-0.2 dB bracket

on the first ramp-selected loss of PERIOD1 if computationally possible.


PHASE C — SPECIFIC NONPERIODIC CLASSIFICATION
=============================================

This phase is crucial.

"BROADBAND_OR_CHAOTIC" and "QUASIPERIODIC_OR_PERIOD_N" are not sufficient final scientific classifications for deciding the HB extension.

Choose representative points:

    A. last robust PERIOD1 point

    B. first robust non-PERIOD1 point

    C. one point somewhat deeper into the first nonperiodic regime

    D. first running-phase point if one exists

    E. any later regular window / re-entrant periodic state if observed

For B and C perform deeper diagnostics.

Compute stroboscopic recurrence:

    d_N(n) = distance between state separated by N pump periods

for at least:

    N = 1, 2, 3, 4, 5, 6, 8, 12, 16

Prefer normalized metrics compatible with existing d1 implementation.

Do not infer PERIOD_N merely because d_N is smaller once.

Require persistent late-time closure and consistency.


PERIOD-N TEST
=============

A PERIOD-N candidate requires all of:

    d1 does not close to PERIOD1 floor

    some modest dN does close robustly

    Poincare/stroboscopic points form N repeatable clusters/states

    Fourier spectrum contains corresponding subharmonic structure

For PERIOD2 specifically expect evidence consistent with:

    fundamental response period = 2 Tp

and spectral structure at:

    fp/2

with the pump at the second harmonic of the expanded basis.

If this is convincingly observed, report:

    PERIOD_2_CONFIRMED

and explicitly state:

    2T / omega_p/2 HB is physically justified by TD.

Likewise for PERIOD3, PERIOD4, etc.


QUASIPERIODIC TEST
==================

A quasiperiodic candidate should show:

    no low-order dN closure

    bounded trajectory

    negligible sustained phase winding

    stroboscopic trajectory forming a curve/ring/torus-like structure rather than finite N points

    Fourier spectrum dominated by sharp discrete incommensurate lines / sideband combs rather than broadband continuum

If strongly supported, classify:

    QUASIPERIODIC_CANDIDATE

or CONFIRMED if current diagnostics justify that word.

Estimate the dominant secondary modulation frequency Omega if possible.

Record whether a two-frequency HB ansatz:

    sum X_(k,m) exp[i(k omega_p + m Omega)t]

would be the natural next representation.


BROADBAND / CHAOS TEST
======================

Do not equate "not PERIOD_N" with chaos.

Use the same type of evidence used in the JTWPA chaos literature:

    late-time Fourier spectrum
    pump-stroboscopic/Poincare structure

Generate these explicitly at representative nonperiodic points.

Look for:

    spectral-line proliferation
    broad/dense spectral content
    diffuse stroboscopic cloud
    lack of low-order recurrence

If the evidence only establishes broadband irregular motion, call it:

    BROADBAND_NONPERIODIC

or:

    CHAOTIC_CANDIDATE

Do NOT claim CHAOTIC_CONFIRMED solely from dN.

If an existing trustworthy Lyapunov / tangent-dynamics diagnostic already exists in the repo and can be run WITHOUT implementing a new major solver, it may be used.

Do not spend the night building a Lyapunov solver from scratch.


RUNNING PHASE
=============

Treat separately.

Require sustained nonzero phase winding / mean phase advance.

Do not classify running based only on:

    |IJ|/Ic -> 1.

Save:

    winding versus time
    responsible junction
    late mean winding rate
    peak |IJ|/Ic

A running-phase threshold is distinct from the PERIOD1-loss threshold.


PHASE D — HISTORY / BASIN CONTROLS
==================================

At only the most informative powers, compare initialization histories.

Do not do this across the whole map.

Choose approximately:

    last PERIOD1
    first non-PERIOD1
    one deeper non-PERIOD1 point

For each, where HB solutions exist, compare:

1. zero-pump independent upward ramp

2. warm start from the nearest LOWER-POWER validated HB solution,
   followed by physical ramp upward to target

3. direct exact HB waveform initialization,
   only as a mathematical/dynamical-stability control

Interpret carefully.

Direct HB is NOT evidence of normal basin selection.

It only answers:

    if placed exactly on/near the HB orbit, does TD remain there?

Cases:

ZERO -> PERIOD1
DIRECT HB -> PERIOD1
    ordinary attracting operation likely.

ZERO -> nonperiodic
DIRECT HB -> PERIOD1
    coexistence / basin dependence is possible.

ZERO -> nonperiodic
DIRECT HB -> departs from PERIOD1
    HB periodic orbit may be dynamically unstable.

Do not make the last conclusion unless the HB->TD handoff/fixture is validated.


PHASE E — RAMP-RATE SENSITIVITY
===============================

At the first transition only, test whether turn-on history materially shifts classification.

Use approximately:

    20-period ramp
    40-period ramp
    80-period ramp

at:

    one point just below the boundary
    one point just above the boundary

Keep fixed-drive hold sufficiently long.

This tells us whether the "physical boundary" is strongly protocol/basin dependent.

Do not expand this into a full ramp-rate map unless results show a large effect.


PHASE F — NUMERICAL ROBUSTNESS
==============================

Do not rerun everything at smaller timestep.

At a few critical representative cases:

    stable PERIOD1
    first non-PERIOD1
    running phase or broadband state

repeat enough of the trajectory at a smaller Delta-theta, preferably:

    0.025

to verify classification-level consistency.

Compare:

    recurrence behavior
    spectral peaks
    phase winding
    peak |IJ|/Ic
    qualitative Poincare structure

We care about classification robustness, not bitwise trajectory equality in chaotic dynamics.

If classification changes under timestep refinement:
    mark NUMERICALLY_UNRESOLVED.


DECAY-AWARE CLASSIFIER
======================

The latest -24.473684 plot is a mandatory regression example.

The classifier must recognize:

DIRECT HB:
    immediately near recurrence floor.

WARM HB:
    raw finite-tail threshold may say nonperiodic,
    but d1/d2/d3 clearly decay toward PERIOD1.

ZERO PUMP:
    much larger initial recurrence,
    including transient growth/structure,
    followed by long decay toward PERIOD1.

Therefore classify using:

    late magnitude
    AND
    trend / envelope decay
    AND
    consistency across d1/d2/d3

Do not classify a trajectory as quasiperiodic simply because its d1 at 250 periods is 5e-4 while still trending downward.

Likewise, do not assume every downward trend eventually reaches PERIOD1.

If decay is too slow to extrapolate reliably:
    UNRESOLVED_LONG_TRANSIENT.


SPECTRAL / POINCARE ARTIFACTS
=============================

For the key representative points save plots/data for:

1. late-time Fourier spectrum

2. zoom around pump and subharmonics:
       fp
       fp/2
       fp/3
       fp/4
       low-frequency modulation components

3. pump-stroboscopic/Poincare projection

4. d1...dN versus period

5. peak |IJ|/Ic versus period

6. phase winding versus period

Choose informative observables/projections.

Do not save enormous full-state trajectories unnecessarily.

Compact storage is mandatory.


LITERATURE-ALIGNED HB-EXTENSION DECISION
========================================

The final result must answer which of these is supported:

A.
    PERIOD1 remains valid throughout useful operation.
    No nonlinear HB extension currently justified.

B.
    PERIOD2 appears first.
    Recommend implementing 2T / omega_p/2 HB.

C.
    PERIOD_N with N>2 appears first.
    Recommend generalized subharmonic period-N HB.

D.
    Quasiperiodic state appears first.
    Recommend investigating two-frequency / quasiperiodic HB rather than period-N HB.

E.
    Transition goes essentially directly to broadband/chaotic dynamics.
    A finite period-N HB ladder is unlikely to recover the physical attractor.

F.
    Running phase occurs before a clearly identifiable bounded nonperiodic attractor.
    Ordinary periodic HB extension is not the primary solution.

G.
    Evidence insufficient because transients remain unresolved.

Do NOT choose an extension merely because it is easy to code.

Choose it from the observed dynamics.


DO WE NEED EVERY POWER POINT?
=============================

No.

The campaign should be exhaustive in DYNAMICAL POSSIBILITIES, not brute-force uniform sampling.

Spend compute preferentially on:

    transitions
    ambiguous points
    first non-PERIOD1 state
    representative nonperiodic regimes
    robustness controls

Do not waste several hours proving that -35, -34, -33, etc. are all trivially PERIOD1 if coarse evidence already establishes a broad stable region.

However, ensure the whole -35 to -15 range has enough coarse coverage to detect re-entrant windows or additional transitions.


PARALLELISM
===========

Parallelize independent target-power runs if the current implementation allows it safely.

Each zero-pump target is independent.

Respect CPU-library thread counts to avoid oversubscription.

Given current memory ~240 MB/child, choose a worker count based on actual machine RAM and CPU.

Do a short 2-worker/4-worker sanity check if needed.

Do not risk OOM for marginal throughput.

Never parallelize sequential integration inside a single trajectory in a way that changes numerical behavior.


ARTIFACT LOCATION
=================

Use a NEW directory.

Suggested:

    .hybrid_outputs/overnight_7p9_dynamics_v1/

Subdirectories:

    coarse/
    boundary_refinement/
    deep_classification/
    initialization_controls/
    ramp_rate/
    timestep_validation/
    plots/

Do not overwrite prior trusted results.


CAMPAIGN SUMMARY
================

Maintain a machine-readable campaign_summary.json.

Each run should record:

    experiment type
    power
    pump current
    initialization
    ramp periods
    hold periods
    Delta-theta
    classification raw
    classification decay-aware
    d1/d2/d3 late statistics
    best closing dN
    inferred candidate period N
    spectral diagnostics
    Poincare diagnostics
    winding
    peak |IJ|/Ic
    responsible junction
    integrator success
    steps
    Newton iterations
    step reductions
    runtime
    peak RSS
    artifact paths


FINAL MORNING REPORT
====================

Produce a concise but complete report with:

1. Total campaign runtime and number of simulations.

2. Coarse state map from -35 to -15 dBm.

3. Highest independently ramp-selected PERIOD1 point.

4. Lowest independently ramp-selected persistent non-PERIOD1 point.

5. Best bracket for first PERIOD1-loss boundary.

6. Whether boundary depends materially on ramp rate.

7. First non-PERIOD1 state's detailed classification.

8. Table of dN for representative states.

9. Fourier-spectrum interpretation.

10. Poincare/stroboscopic interpretation.

11. Whether any clear:
        PERIOD2
        PERIOD3/4/etc.
        quasiperiodic
        broadband/chaotic candidate
        running-phase
    regions exist.

12. If running phase exists, give its threshold separately from PERIOD1 loss.

13. Whether direct HB orbit remains dynamically stable above the zero-pump ramp-selected transition, where testable.

14. Evidence for or against multistability.

15. Numerical timestep robustness.

16. The most defensible HB extension, choosing exactly one:

        NO_EXTENSION_YET
        PERIOD_N_HB
        TWO_FREQUENCY_HB
        FINITE_PERIOD_HB_NOT_APPROPRIATE
        INSUFFICIENT_EVIDENCE

If PERIOD_N_HB:
    state N and explain the evidence.

17. Separate:
        facts directly demonstrated,
        plausible interpretation,
        unresolved questions.

18. Include the key plots.

19. Explicitly identify every run whose classification remained unresolved.


HARD RULES
==========

Do not modify production solver internals.

Do not alter circuit physics.

Do not add artificial damping.

Do not use the old descending campaign as the normal-operation map.

Do not carry TD state between independent primary target powers.

Do not call HB failure a physical transition.

Do not call a transient numerical failure a physical transition.

Do not call a valid nonperiodic TD state a solver failure.

Do not call high |IJ|/Ic alone running phase.

Do not call every nonperiodic state chaotic.

Do not trust only the raw finite-tail classifier.

Do not use a direct HB initialization as evidence of normal turn-on basin selection.

Do not spend all 8 hours on one unresolved point.

If a run is ambiguous:
    record it,
    move on,
    return later only if it is scientifically important and budget remains.


PRIORITY ORDER IF TIME RUNS OUT
===============================

1. Independent zero-pump coarse map over entire -35 to -15 dBm range.

2. Refine first PERIOD1-loss boundary.

3. Deep dN + FFT + Poincare classification of first non-PERIOD1 state.

4. Identify first running-phase state.

5. Timestep validation at critical points.

6. Initialization/basin controls.

7. Ramp-rate controls.

8. Additional refinement / secondary transitions.


THE MOST IMPORTANT QUESTION TO ANSWER BY MORNING
================================================

Do not merely tell me:

    "HB fails around X dBm."

Tell me:

    Starting from zero pump and physically ramping the 7.9 GHz 2c device upward,
    what is the first dynamical state that replaces the stable PERIOD1 solution,
    and what harmonic-balance representation, if any, is mathematically justified
    by that observed state?



ADDENDUM — VISUAL-FIRST PLOT PACKAGE
====================================

Visual inspection is substantially faster for the user than reading classification JSON.

Therefore plots are a PRIMARY campaign output, not an optional post-processing step.

The campaign must continuously generate compact, standardized plots so that by morning the dynamical regimes can be inspected visually without opening individual JSON files.

Do not rely on the automatic classifier to communicate the result. The classifier and plots must be independently inspectable.


1. ONE STANDARD SUMMARY FIGURE PER TARGET POWER
===============================================

For every primary zero-pump -> upward-ramp target, generate one standardized figure with a fixed layout and identical axis conventions wherever possible.

Suggested filename:

    plots/by_power/p_m23p5000dbm_summary.png

The figure should contain, at minimum:

    A. peak |IJ|/Ic versus total pump periods

    B. d1 versus total pump periods

    C. d2 versus total pump periods

    D. d3 versus total pump periods

    E. selected dN curves for:
           N = 1, 2, 3, 4, 6, 8
       preferably together when visually readable

    F. phase winding versus total pump periods

Use logarithmic y axes for recurrence metrics where appropriate.

Mark clearly:

    end of pump ramp
    start of fixed-drive hold
    adaptive-classification checkpoints
    final classification window

Include in the title:

    frequency
    target power
    initialization = zero-pump upward ramp
    ramp periods
    hold periods
    Delta-theta
    final decay-aware classification

The existing -24.473684 dBm comparison figure is a good visual reference for the recurrence-history style.


2. OVERVIEW POWER-LADDER FIGURE
===============================

Produce one compact overview figure spanning the entire -35 to -15 dBm campaign.

Rows should correspond to target pump powers, ordered LOW -> HIGH.

Columns should show compact traces or summary information for:

    peak |IJ|/Ic
    d1
    best dN / inferred periodicity
    phase winding

The purpose is to make regime transitions visible at a glance.

If putting full traces for every target is visually overcrowded, generate separate overview figures:

    overview_d1.png
    overview_utilization.png
    overview_winding.png
    overview_best_dN.png

Use exactly the same x coordinate:

    total pump periods

for all rows where possible.

Highlight or annotate powers classified as:

    PERIOD1
    PERIOD2
    PERIOD_N
    QUASIPERIODIC
    BROADBAND_NONPERIODIC
    RUNNING_PHASE
    UNRESOLVED

Do not encode the only distinction by color. Include labels/text/markers so plots remain readable if printed.


3. CLASSIFICATION MAP VERSUS POWER
==================================

Generate a very simple regime-versus-power plot.

x axis:
    pump power [dBm]

y axis:
    categorical dynamical regime

For example:

    PERIOD1
    PERIOD2
    PERIOD_N
    QUASIPERIODIC
    BROADBAND
    RUNNING_PHASE
    UNRESOLVED

Plot every independently tested zero-pump upward-ramp point.

This should immediately show:

    first loss of PERIOD1
    periodic windows
    re-entrant behavior
    broadband regions
    running-phase onset

Overlay/mark separately:

    successful ordinary HB points
    first ordinary HB failure

so the user can visually compare:

    HB existence/convergence boundary
        versus
    actual TD dynamical boundary.


4. LATE dN "PERIOD FINDER" PLOT
===============================

For every important non-PERIOD1 target, generate a bar/point plot of late-time recurrence versus N:

    N = 1,2,3,4,5,6,8,12,16

x:
    N

y:
    late dN

log y scale.

Suggested filename:

    deep_classification/p_m23p4_dN.png

This is one of the most useful visual classifiers.

Examples:

PERIOD2:

    d1 large
    d2 extremely small
    d4/d6/... also small according to multiples

PERIOD3:

    d3 closes

PERIOD4:

    d4 closes while d1/d2 do not

QUASIPERIODIC/BROADBAND:

    no modest N closes convincingly.

Do not just show the final dN sample if it is noisy.

Prefer:

    late median dN
    with a visual range/error indication from the late window

when straightforward.


5. dN VERSUS TIME FOR KEY POINTS
================================

For:

    last PERIOD1 point
    first non-PERIOD1 point
    one deeper non-PERIOD1 point
    any clear PERIOD_N point

generate a figure containing:

    d1(t)
    d2(t)
    d3(t)
    d4(t)
    d6(t)
    d8(t)

over the entire hold.

This makes it visually obvious whether a candidate recurrence is:

    actually converging
    oscillating around a floor
    slowly decaying
    slowly growing
    only temporarily crossing a threshold.

Use log scale.

This plot is more important than a single final classifier label.


6. POINCARE / STROBOSCOPIC PLOTS
================================

For each representative non-PERIOD1 regime generate pump-stroboscopic plots.

Do not attempt to visualize the full ~6000-dimensional state.

Choose a few physically meaningful scalar/projection pairs, for example:

    junction phase vs phase velocity
    selected node flux vs derivative
    two representative mode/amplitude coordinates
    dominant JJ phase at stroboscopic times

Prefer observables that already exist without introducing expensive new machinery.

Sample once per pump period after discarding the transient portion.

Generate at least one 2D scatter plot per representative point.

Interpret visually:

PERIOD1:
    one compact point/cluster

PERIOD2:
    two compact clusters

PERIOD3:
    three compact clusters

PERIOD4:
    four compact clusters

QUASIPERIODIC:
    smooth closed curve / ring-like structure

CHAOTIC/BROADBAND CANDIDATE:
    diffuse irregular cloud / complex filled structure

Also save the underlying compact stroboscopic data.


7. FOURIER SPECTRUM PLOTS
=========================

For every representative regime, compute a late-time spectrum from a meaningful observable.

Generate:

A. broad spectrum

B. low-frequency/subharmonic zoom covering at least:

       0
       fp/4
       fp/3
       fp/2
       fp
       2fp

C. zoom around fp showing sidebands/modulation

Prefer normalized dB spectra when useful.

Mark vertical guide lines at:

    fp
    fp/2
    fp/3
    fp/4

If a secondary incommensurate frequency Omega is detected, mark:

    Omega
    fp - Omega
    fp + Omega

The plots should make the following visually distinguishable:

PERIOD1:
    integer pump harmonics / expected discrete structure

PERIOD2:
    strong half-frequency/subharmonic structure

PERIOD_N:
    corresponding fp/N structure

QUASIPERIODIC:
    discrete sharp incommensurate sidebands/combinations

BROADBAND/CHAOTIC CANDIDATE:
    dense/broadened spectral structure rather than only discrete lines.


8. SPECTROGRAM FOR TRANSITIONAL POINTS
======================================

For the first non-PERIOD1 point and any slowly evolving/unresolved point, also generate a time-frequency spectrogram if inexpensive.

x:
    pump periods / time

y:
    frequency

This is especially useful for distinguishing:

    slow transient decay toward PERIOD1

from:

    growth of a subharmonic instability

from:

    progressive spectral broadening.

Keep this limited to a few important targets; do not generate expensive spectrograms for the full coarse sweep.


9. JUNCTION-UTILIZATION HEATMAP
===============================

For representative points near each major transition, generate a compact spatial visualization of junction utilization.

x:
    junction index / position along device

y:
    late-time max or RMS |IJ|/Ic

If the device topology makes a simple line misleading, group/index according to the existing physical circuit ordering.

Compare at least:

    last PERIOD1
    first non-PERIOD1
    deeper nonperiodic
    first RUNNING_PHASE

This can visually expose whether the transition originates:

    locally at one JJ
    in a particular section
    or broadly across the traveling-wave structure.


10. PEAK UTILIZATION VERSUS POWER
=================================

Generate:

    max late-window |IJ|/Ic versus pump power

for all independently ramped targets.

Mark each point by dynamical regime.

This will visually answer whether:

    PERIOD1 loss happens near Ic

or, as current evidence suggests, occurs while junction utilization is still moderate.

Do not interpret |IJ|/Ic = 1 by itself as running phase.


11. RECURRENCE VERSUS POWER
===========================

Generate summary curves versus pump power for:

    late d1
    late d2
    late d3
    minimum late dN for N <= 16

Use log y scale.

Also plot:

    N_best(P)

where N_best is the period offset with the smallest trustworthy late recurrence.

This can make a period-doubling cascade visually obvious if something like:

    N_best = 1 -> 2 -> 4 -> 8

emerges with increasing power.


12. PHASE-WINDING VERSUS POWER
==============================

Generate:

    late mean winding
    maximum absolute winding

versus pump power.

Make the first robust nonzero-winding point visually obvious.

This boundary must be displayed separately from the PERIOD1-loss boundary.


13. BOUNDARY-REFINEMENT PLOT
============================

As the first PERIOD1 transition is refined, maintain a dedicated plot showing every tested power in that local interval.

Display each target as one of:

    validated PERIOD1
    validated non-PERIOD1
    unresolved

Also mark:

    ordinary HB succeeds
    ordinary HB fails

Update this plot after every refinement run.

The user should be able to open one image and immediately see the current physical bracket.


14. INITIALIZATION-COMPARISON PLOTS
===================================

For the few history-control points, reproduce the same style as the existing three-row comparison figure:

rows:

    direct HB
    lower-power HB warm start
    zero-pump upward ramp

columns:

    peak |IJ|/Ic
    d1
    d2
    d3

Where useful add:

    d4
    winding

Keep axes matched across rows.

These figures are extremely useful for visually distinguishing:

    different transient histories converging to the same attractor

from:

    genuine basin dependence / multistability.


15. RAMP-RATE COMPARISON PLOTS
==============================

At the selected boundary controls, generate a similar matched figure for:

    20-period ramp
    40-period ramp
    80-period ramp

Plot:

    utilization
    d1
    relevant dN
    winding

This should visually show whether the selected dynamical regime depends materially on pump turn-on rate.


16. TIMESTEP-ROBUSTNESS PLOTS
=============================

For Delta-theta = 0.05 versus 0.025 validation points, overlay or stack:

    d1/dN
    utilization
    winding

Do not require trajectories to remain phase-identical if dynamics are chaotic.

We want visual agreement in:

    regime
    envelopes
    dominant frequencies
    recurrence/nonrecurrence
    winding behavior.


17. MASTER MORNING DASHBOARD
============================

At the end of the campaign, produce ONE master image:

    plots/MASTER_7p9_DYNAMICS_SUMMARY.png

It should summarize the entire experiment.

Suggested panels:

    A. dynamical regime vs pump power

    B. late d1 vs power

    C. best dN / inferred N vs power

    D. peak |IJ|/Ic vs power

    E. phase winding vs power

    F. zoom of first PERIOD1-loss boundary

    G. spectrum of last PERIOD1

    H. spectrum of first non-PERIOD1

    I. Poincare of last PERIOD1

    J. Poincare of first non-PERIOD1

The purpose is that the user can inspect this ONE figure first and then drill down into per-point plots only if needed.


18. BUILD A SIMPLE HTML INDEX IF EASY
=====================================

If straightforward and it does not consume meaningful campaign time, generate:

    plots/index.html

containing:

    master dashboard
    overview plots
    regime map
    boundary refinement
    thumbnails/links for each target's summary figure
    deep-classification spectra
    Poincare plots

Static local HTML only.

No server is necessary.

Do not spend substantial solver time building a UI.


19. PLOTTING PRIORITY IF TIME IS SHORT
======================================

If plotting work itself becomes nontrivial, prioritize:

1. per-target utilization + d1/d2/d3 plots
2. master regime-vs-power plot
3. first-boundary refinement plot
4. dN period-finder at first non-PERIOD1
5. FFT at first non-PERIOD1
6. Poincare at first non-PERIOD1
7. winding/utilization versus power
8. everything else


20. IMPORTANT VISUAL INTERPRETATION RULE
========================================

Plots are diagnostic evidence, but do not let the automatic labels override an obvious long-term trend.

The existing -24.473684 dBm comparison demonstrates this:

    direct-HB starts close to the recurrence floor,

while:

    warm-HB
    and
    zero-pump upward-ramp

begin substantially farther away but visibly decay toward the same PERIOD1 state.

A raw threshold applied at an arbitrary finite time can therefore produce a misleading label.

Always retain enough trajectory history in the plots to visually distinguish:

    persistent nonperiodicity

from:

    slow convergence to PERIOD1.

The user will use these plots as a rapid independent classification check in the morning.