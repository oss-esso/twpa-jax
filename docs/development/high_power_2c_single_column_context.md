_2c# High-Power 2c Single-Column Investigation Context

## Standing objective

Determine how far the lossless production 2c TWPA model can be solved before a
Josephson junction reaches its critical-current utilization. Solver failure is
not a physical boundary. The investigation must continue through numerical
obstructions using valid numerical methods until either a validated junction
break state is reached or the physical/model boundary is established with
independent evidence.

## Fixed experiment

- Circuit: `designs/ipm_2c_fixed`.
- Frequency: 7.9 GHz.
- Pump powers: 20 points from -35 dBm through -15 dBm.
- Attenuation: standard A10 attenuation profile.
- Attenuation override: empty/null. Never substitute flat attenuation.
- Physical model: unchanged production lossless 2c model.
- No artificial resistance, capacitance, Gmin, damping, parameter changes, or
  relaxed validation thresholds.
- Work on the `dev` branch only.
- Do not interrupt processes belonging to other workspaces.

## Baseline policy

Use whichever existing 7.9 GHz single-column result is authoritative between
the previous baseline and `campaign_diss_2c_base`, after independently
recomputing the current production residual, circuit provenance, mode metadata,
port metadata, and attenuation metadata. A filename or historical status is
not sufficient evidence.

## Required success criterion

For every reported state, distinguish:

1. validated HB state;
2. validated TD state;
3. numerical failure;
4. unresolved physical classification; and
5. confirmed physical boundary.

Junction utilization means the maximum, over all Josephson junctions and over
the reconstructed waveform, of

`max_t |I_j(t)| / Ic_j`.

The current profile must identify the junction index and location of the
maximum. Reaching or approaching utilization one must be demonstrated from an
independently validated state. Do not infer it from Newton failure, a map edge,
or a coarse last-pass point.

## Investigation order

### Current mandatory TD campaign protocol

The active experiment is a descending-power TD campaign. Start at the first
high-power point near `-21.3158 dBm`, then move downward one point at a time
through the fixed 20-point 7.9 GHz column. Do not switch to preconditioner or
other solver-scope work during this campaign.

Before selecting the production TD method, run the first point with every
available DAE integrator supported by the existing workflow and compare:

- successful integration and numerical-failure status;
- peak `max|I_j|/Ic_j`;
- cumulative-period evolution;
- final `d1`, `d2`, and `d3`;
- phase winding;
- runtime and peak memory.

Select the best method only from validated evidence on that first point. Then
use that method consistently while descending point by point. At every point,
extend the same fixed-drive state through the high-period checkpoints and
record the classification and utilization evolution. Existing TD artifacts
must be reused where they already cover a checkpoint; do not silently relabel
or replace them.

This protocol is an explicit scope boundary: the 2c circuit is unbiased 4WM,
so no dynamic DC representation or 3WM-specific solver change is to be
introduced during this campaign.

First test the proposals in `docs/development/solver_plan_post_td.md` on this
exact single-column experiment, recording quantitative outcomes:

- harmonic enrichment and omitted-harmonic residuals;
- full time-domain residual acceptance;
- continuation and source/power homotopy;
- PALC and fold diagnostics;
- exact JVP versus preconditioner behavior;
- alternative sparse/Krylov/preconditioner paths;
- TD bridge, settling, and HB projection;
- shooting or periodic-orbit alternatives where applicable;
- branch switching, Floquet, and subharmonic representations where indicated.

Then test additional credible numerical methods systematically. Each method
must state its hypothesis, use the same physical problem, preserve provenance,
and have an explicit acceptance or rejection result. Do not spend extended
runtime on a method after it has failed the representative single-column gate
unless new evidence changes the hypothesis.

## Reporting requirements

Every campaign must record:

- exact circuit directory and validation result;
- attenuation override and resolved attenuation profile;
- frequency, power, source convention, ports, and harmonic modes;
- retained and full residuals, including omitted-mode spectrum where available;
- Newton, GMRES, factorization, and continuation telemetry;
- peak junction current, `max|I|/Ic`, junction index, phase, and tangent margin;
- TD classification, period count, recurrence metrics, and phase winding;
- runtime and peak memory;
- whether the state is suitable for gain calculation.

No state may be called solved, physical, gain-valid, or junction-breaking
without the corresponding independent validation evidence.

## First integrator screen and descending campaign record (2026-08-10)

The first-point screen used the same validated fixed-drive restart at
`-21.3158 dBm`, 7.9 GHz, with a 100-period hold. Implicit trapezoid and
implicit Euler completed; BDF and Radau terminated with the SciPy error
`Required step size is less than spacing between numbers`. Implicit trapezoid
was selected because it was successful, required 1,257 steps and 1,257 Newton
iterations, and took 24.0 s, versus 62.2 s and 3,772 Newton iterations for
implicit Euler. Both successful methods remained unresolved after this finite
hold. The screen did not establish a physical boundary.

The subsequent serial descending campaign used implicit trapezoid, 40 ramp
periods between adjacent fixed-drive targets, and a 440-period hold. Point 12
(`-22.3684 dBm`) used the validated HB fixture. The lower targets had empty
checkpoint directories; their currents were generated from the documented
20-point power grid and are explicitly recorded as TD fixed-drive targets,
not HB checkpoints. The bounded path completed 6,032 accepted steps per
point with peak child RSS between approximately 229 and 305 MB. No numerical
TD failure or memory-limit termination occurred after explicit per-step
temporary release and periodic garbage collection were added.

The tested descending segment was point 13 (`-21.3158 dBm`, existing
440-period artifact) through point 0 (`-35 dBm`). Point 13 was
`RUNNING_PHASE`, with peak sampled `max|sin(phi)|/Ic = 0.9999999973` and
0.494 cycles of late phase winding. Point 12 was
`UNRESOLVED_SLOW_RELAXATION`. Points 11 through 2 were reproducible
non-period-1 trajectories classified by the current classifier as
`BROADBAND_OR_CHAOTIC`; point 1 was unresolved; point 0 was again classified
as broadband/non-period-1. These carried-state classifications are not gain
validations and are not evidence that the lower-power physical ramp-selected
state is absent. No points above `-21.3158 dBm` were run in this descending
campaign.

## Corrected HB-gated campaign record

The generated-target campaign above is superseded and must not be used as an
HB-column result. A production HB-up run was then repeated from `-35 dBm` on
the fixed 14-point grid and stopped at the first failure:

- validated HB points: `-35` through `-24.4737 dBm`;
- first HB failure: `-23.4211 dBm`;
- the separately available `-22.3684 dBm` checkpoint was not used because it
  was not reached by this upward run.

A 440-period implicit-trapezoid TD bridge from the last valid HB state to the
first failed target completed successfully (`6032` steps), but was classified
`BROADBAND_OR_CHAOTIC` / persistent non-periodic and was not used for gain.
The subsequent descent from that bridge through `-24.4737` to `-35 dBm` used
only the newly produced validated HB checkpoints. All points completed 440
periods successfully, with no TD numerical failure and approximately
`244--307 MB` peak child RSS. The carried-state classifications are recorded
in the corrected campaign artifact; they are not evidence that the low-power
HB PERIOD1 states are absent.
