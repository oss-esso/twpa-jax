# Saturation solver: first-principles design vs current implementation

**Status:** implementation comparison and failure analysis  
**Compared against:** `docs/development/saturation_solver_from_first_principles.md`  
**Scope:** the production saturation path rooted at `scripts/run_compression.py`, its multitone HB implementation, tests, and the saturation-validation artifacts currently in the repository.

## Executive conclusion

The current solver is not a wholesale implementation failure. Its circuit equations, nonlinear branch-law boundary, two-frequency tone convention, AFT reconstruction, residual/JVP machinery, and continuation solver are substantially aligned with the updated first-principles design. Focused low-power and toy-model tests support those pieces.

The saturation result went wrong farther up the evidence chain. The production campaign treated a minimally truncated, incompletely characterized model as though it were already a converged traveling-wave saturation calculation. It then attached mechanism diagnostics to the wrong state or to incomplete port observables. The dominant confirmed divergences are:

1. The device was not shown to be in the intended traveling-wave regime before nonlinear interpretation. Later artifacts show substantial passive ripple and backward-wave content.
2. The default production basis structurally excludes the `|q| >= 2` mixing products whose convergence is required for saturation. Existing Q-axis results show material movement in both P1dB and slope and do not establish Q convergence.
3. The production small-signal reference is the first finite-power HB sample, not an independent linearized/Floquet gain or a deliberately shared estimator.
4. Refined P1dB is computed, but depletion, spatial, and stability diagnostics are taken from a nearby coarse-grid state.
5. Pump depletion is inferred from one output port even though the modeled network has multiple pump exits. A later experiment found that this port carried only about 9.1% of the measured outgoing pump power in that setup.
6. The reported power-balance closure is an internal residual/dissipation identity, not the all-port wave-energy balance required to validate the observable normalization.
7. Production validation has not completed the independent-oracle ladder: the generic CME oracle is absent, the reference CME calibration is unresolved, no transient oracle exists, and production Q/stability checks remain incomplete.
8. Campaign defaults and persistence are too weak for an expensive convergence study: five signal-power samples by default, no default deadline, optional diagnostics, and output written only after the full solve returns.

Consequently, the repository does not yet contain evidence that the shallow simulated P1dB-versus-gain slope is a property of the intended TWPA physics. It is currently a property of a particular resonant network, basis truncation, estimator, and diagnostic path. There is also no confirmed evidence that the core Newton/AFT algebra is the source of the discrepancy.

## Implementation status after this comparison

The follow-up implementation has addressed the observable and campaign defects that can be fixed without launching new HB campaigns:

- `power_balance` now exposes the all-port net pump-power change, all-port outgoing-pump dB ratio, per-tone net-power map, and an external-vs-dissipation balance alongside the legacy internal balance.
- P1dB is solved once more at the reported crossing current. Its refined state now supplies P1dB depletion, external balance, spatial profiles, and stability diagnostics; a failed final solve is reported as `CHECK` rather than silently falling back to the nearest coarse row.
- `run_compression.py` accepts `--pump-current-list`, writes deterministic per-current result directories, resumes only summaries matching the requested effective current, and emits `p1db_vs_pump_current.csv` plus an aggregate JSON.
- The existing single-port pump ratio remains available and is printed beside the all-port result so prior artifacts remain interpretable.

The controlled pump-swept HB campaign has now run at fixed frequency: jtwpa at 6.4 GHz and 2c at 7.4 GHz. It establishes shallow but non-identical model-only slopes (`-0.700 +/- 0.345 dB/dB` over 21.06 dB for jtwpa; `-1.185 +/- 0.024 dB/dB` over 3.85 dB for 2c). These results answer the fixed-frequency question for the converged points, but do not remove the remaining Q-convergence, passive-regime, or frequency-swept model-versus-Themis confounds.

## Status vocabulary

- **Aligned:** the implementation follows the updated design and has direct code/test support.
- **Partial:** the mechanism exists but the production path does not satisfy the design contract completely.
- **Divergent:** production behavior directly conflicts with the updated design.
- **Missing:** no production implementation was found.
- **Inconclusive:** evidence exists, but it is insufficient to assign causality.

## Current production execution path

The actual saturation calculation follows this path:

1. Build the circuit and solve a pump-only harmonic-balance problem.
2. Define `delta = omega_p - omega_s` and construct either a matched Floquet basis or a rectangular lattice.
3. Promote the pump solution into the multitone basis.
4. Solve one pump-off signal point and then the pump-on signal-power continuation.
5. Use the first successful finite-power pump-on result as the small-signal gain reference.
6. Detect a 1 dB crossing on the coarse samples and optionally refine only the crossing current.
7. Associate stability, spatial, and depletion diagnostics with saved coarse states or with the nearest coarse row.
8. Write CSV, NPZ, and JSON artifacts only after `_solve_compression` returns.

The central implementation is `scripts/run_compression.py:259-1039`. The residual and JVP are implemented in `src/twpa_solver/multitone/problem.py`; basis construction is in `src/twpa_solver/multitone/basis.py`; wave and balance observables are in `src/twpa_solver/multitone/observables.py`; final artifact writing is in `src/twpa_solver/multitone/io.py:13-61`.

## Design-to-code comparison

### 1. Physical model and circuit boundary

| First-principles contract | Current implementation | Status |
|---|---|---|
| A single physical circuit model should provide `C`, `G`, `K`, ports, branch incidence, and nonlinear branch laws. | `src/twpa_solver/core/circuit.py` exposes the circuit matrices and branch mapping. `src/twpa_solver/core/nonlinear.py` provides Josephson and effective SNAIL branch laws. | Aligned |
| The branch law must be expanded about the correct equilibrium rather than an arbitrary zero phase. | `src/twpa_solver/builders/le_gal_2025.py:67-98` solves the SNAIL equilibrium, and `EffectiveSnailBranchLaw` incorporates the equilibrium shift. | Aligned |
| Pump-only, small-signal, and multitone solvers should use the same circuit model. | The production pump and multitone problems share the constructed circuit and branch law. | Aligned |
| The passive device must be characterized before nonlinear saturation is interpreted. | Passive tools exist, but `run_compression.py` does not perform or require pump-off S-parameter, ripple, standing-wave, or impedance checks before running saturation. | Divergent |

The important distinction is that the solver has an appropriate physical-model boundary, but the campaign does not prove that the instantiated circuit is operating in the physical regime assumed by the interpretation.

### 2. Frequency convention, basis, and AFT

| First-principles contract | Current implementation | Status |
|---|---|---|
| Use `omega_(p,q) = p omega_p + q delta`, with `delta = omega_p - omega_s`; signal `(1,-1)`, idler `(1,+1)`. | `run_compression.py:423` and `basis.py:15-27,141-146` use the same sign convention and tone identities. | Aligned |
| Reconstruct real waveforms with the explicit factor of two. | `basis.py:11,177-216` uses `REAL_RECONSTRUCTION_FACTOR = 2.0` consistently in synthesis/projection. | Aligned |
| Choose an anti-aliased AFT grid for the retained nonlinear products. | `basis.py:112-121` builds the projection grid from pair sums, producing the required cubic margin for the present branch laws. | Aligned |
| Saturation must be converged in both pump harmonic order H and mixing sideband order Q. | A rectangular lattice is available through `build_lattice_basis`, but the default `matched` basis retains only `q = -1, 0, +1`; `--multitone-sidebands` does not enlarge that default basis. | Divergent |
| A minimum saturation campaign should not rely on a three-tone or small-signal-matched basis. | `run_compression.py:97-101` defaults to `matched`; `three_tone` is also exposed. The default matched basis is structurally unable to represent `|q| >= 2` cascades. | Divergent |

This is not merely a conservative recommendation. `outputs/exp24b_q_axis_slope/slope_verdict.json` reports a slope change from approximately `-0.444` at Q1 to `-0.683` at Q2, and its Q3 spot check shifts P1dB by about `0.5 dB`; the Q-convergence gate is false. The broader Q-axis summaries show several multi-dB Q1-to-Q2 P1dB changes. These artifacts demonstrate basis sensitivity, not convergence.

### 3. Nonlinear residual, JVP, and linear solves

| First-principles contract | Current implementation | Status |
|---|---|---|
| Residual and Jacobian-vector products should be derived from the same nonlinear branch law. | `FullMultiToneProblem.residual` and its AFT/spectral tangent JVPs use the same branch-law interface. | Aligned |
| JVPs should be verified against finite differences. | `tests/test_multitone_problem.py` and `tests/test_multitone_physics.py` exercise central-difference JVP agreement and physical low-power limits. | Aligned |
| Matrix-free Newton-Krylov should use a reusable linear preconditioner, with exact or approximate factorization selected by benchmark. | The pump linear operator is reused as the preconditioner; exact and Schur paths exist, and factor backends are selectable in `run_compression.py:173-220,432-483`. | Aligned |
| Time-domain residual evaluation is an independent diagnostic, not the primary solve formulation. | The problem implements it, but production hard-codes `compute_time_residual=False` at `run_compression.py:371-379`. | Partial |

The low-power JPA regression reaches approximately `15.59 dB` gain (`tests/test_multitone_physics.py:204-216`), and the same test module covers the pump-only limit, full-sideband reference behavior, and tone-S21 parity. These results make a basic sign, reconstruction-factor, or residual/JVP failure less likely. One older test remains marked `xfail` with the stale claim that the default JPA fixture has not exceeded 3 dB (`tests/physics/test_compression_low_signal_limit.py:6-8`), which should not be treated as the current validation status.

### 4. Continuation and P1dB extraction

| First-principles contract | Current implementation | Status |
|---|---|---|
| Use continuation and do not bridge failed interior points when finding P1dB. | Adaptive continuation is implemented, and `tests/test_compression.py:95` verifies that a failed gap is not bridged. | Aligned |
| Use a sufficiently resolved low-power plateau and compression curve. | `--n-signal-powers` defaults to 5 at `run_compression.py:86-101`, versus the updated design's recommended minimum of 16 for a production sweep. | Divergent |
| Define Glin independently or with an explicitly shared estimator. | `small_signal_gain_db` is assigned from `points[0]["gain_vs_off_db"]` at `run_compression.py:753-764`. It is neither an independent Floquet solve nor a fitted/split-sample plateau. | Divergent |
| Refine the 1 dB crossing and compute all P1dB diagnostics from that refined nonlinear state. | The current evaluator refines the scalar crossing (`run_compression.py:823-864`) but does not retain the refined state. `p1db_point` is the nearest coarse row, and saved `states["p1db"]` is the first coarse state at or beyond the threshold. | Divergent |
| Record uncertainty or local resolution at the reported crossing. | The summary reports the refined current/P1dB but no bracketing width, local slope uncertainty, or estimator uncertainty. | Partial |

This state mismatch is large enough to matter. `docs/development/saturation_solver_p1db_measurement.md` records refined-versus-interpolated shifts of roughly `0.23-0.34 dB` in earlier cases. More importantly, a nearest coarse row may already be compressed by substantially more than 1 dB, so its pump depletion, spatial profile, or stability cannot be labeled as the refined P1dB mechanism state.

### 5. Gain and port-wave normalization

| First-principles contract | Current implementation | Status |
|---|---|---|
| A single observable layer should convert port voltages/currents into incident and outgoing waves. | `observables.py:42-101` provides `extract_port_waves` and `tone_s21` using KCL-derived network currents and the reconstruction convention. | Aligned |
| Absolute gain and pump-on/pump-off gain enhancement must be kept distinct. | The script computes both `gain_db` and `gain_vs_off_db` (`run_compression.py:508,615-635`). | Aligned |
| The production driver should use the central observable/normalization layer. | `reference_states` and `reference_normalization` exist at `observables.py:418-576`, but the driver hand-computes gain enhancement and refined gain in `run_compression.py:616-635,823-845`. | Divergent |
| Pump depletion must include all outgoing pump ports. | `pump_depletion_db` is derived from one pump `S21` ratio at `run_compression.py:636-639`. | Divergent |

The all-port requirement is decisive for the existing depletion argument. `experiments/exp27_track5_depletion_bound.py` and its report explicitly note that the selected port-2 wave accounted for only about 9.1% of the outgoing pump power in that experiment. Therefore the existing `p1db_pump_depletion_db` is not total pump depletion and cannot establish or exclude pump depletion as the saturation mechanism.

### 6. Energy and photon-flux diagnostics

| First-principles contract | Current implementation | Status |
|---|---|---|
| Verify all-port incoming minus outgoing wave power against physical dissipation. | `power_balance` does calculate per-port/per-tone wave power, but the reported `power_balance_rel_err` uses `internal_supplied_power` from the residual-source term (`observables.py:149-202`), not `external_supplied_power`. | Divergent |
| A passing balance must validate wave orientation and normalization. | Because the reported balance closes an internal identity, it can be near machine precision even if external wave normalization or port selection is wrong. | Divergent |
| Separate scoped conversion Manley-Rowe diagnostics from an all-tone diagnostic. | Both explicit aliases now exist at `observables.py:249-262`, but legacy ambiguous fields remain at `observables.py:237-247`, and the driver serializes the legacy/external names rather than making the scoped contract primary. | Partial |
| A saturated production state must pass a meaningful external conservation gate. | Current unit tests deliberately leave the lossless external diagnostic unevaluable (`tests/test_power_balance.py:61-69,78-105`), and the independent validation report records the production saturated external gate as unevaluated. | Missing |

This explains an apparent contradiction in prior results: an internal `power_balance_rel_err` around machine precision does not prove that the port-derived gain, pump depletion, or conversion photon flux is correctly normalized. The metric and the observable under test are not independent.

For three-wave mixing, the present simple pump/signal/idler photon-flux sum may also need a mixing-order-aware invariant. That is a technical hypothesis, not a confirmed defect in this comparison; it requires a derivation tied to the selected nonlinear process before changing the gate.

### 7. Spatial and stability diagnostics

| First-principles contract | Current implementation | Status |
|---|---|---|
| Spatial diagnostics should be evaluated at zero signal, low signal, refined P1dB, and deep compression. | Spatial output is optional and uses saved coarse states; deep-compression coverage depends on save settings. | Partial |
| Position must map to the actual physical propagation coordinate. | `observables.py:354-391` uses branch indices/start-stop nodes. The current multi-row topology requires care because branch order is not automatically a single propagation coordinate. | Inconclusive |
| Stability should be reported in a dimensionless normalized form and should state the perturbation space. | `stability.py:77-82` explicitly evaluates only the pump-periodic `q=0` slice and returns a dimensional `dominant_exponent_per_s` (`stability.py:19-25`). The driver does not normalize it by `omega_p`. | Partial |
| Mechanism diagnostics at P1dB must correspond to the refined state. | Both spatial and stability P1dB labels refer to coarse saved states (`run_compression.py:877-908,994-1033`). | Divergent |

Existing spatial artifacts show strong internal peaking and many gradient sign changes, but those findings characterize the current resonant topology and coarse operating states. They are not yet a clean traveling-wave P1dB mechanism diagnosis.

### 8. Validation oracles

| Validation layer from the updated design | Repository status | Assessment |
|---|---|---|
| L0 linear circuit/passive checks | Passive analysis tools and reports exist, but they are not a prerequisite of the saturation driver. | Partial |
| L1 pump-only limit | Covered in `tests/test_multitone_physics.py:291-313`. | Aligned |
| L2 low-power Floquet agreement | A physical JPA low-power regression and sideband reference exist. | Aligned for the tested fixture |
| L3 weakly nonlinear CME agreement | CME code exists only under `references/le_gal_2025_gain_compression/`; calibration/benchmarking is unresolved and it is not a generic production oracle. | Missing for production |
| L4 basis convergence | H/Q campaign machinery and artifacts exist, but Q3 is incomplete and the current verdict is false. | Not passed |
| L5 depletion-slope relation | Diagnostic experiments exist. The updated design correctly treats this only as a diagnostic, not a bound or acceptance gate. | Inconclusive |
| L6 nonlinear time-domain oracle | No implementation was found. | Missing |
| L7 stability | A `q=0` diagnostic exists, but production zero/P1dB/deep-compression coverage is incomplete. | Partial |

The validation ladder therefore stops before the two checks most capable of distinguishing a correct nonlinear HB solution from a plausible but basis- or topology-dependent result: production saturation-basis convergence and an independent nonlinear oracle.

### 9. Campaign durability and reproducibility

| First-principles contract | Current implementation | Status |
|---|---|---|
| Long campaigns should save incremental per-power artifacts and explicit failure status. | `write_compression_outputs` writes after `_solve_compression` returns (`run_compression.py:1045-1061`; `io.py:13-61`). A process failure can leave no partial curve. | Divergent |
| Each continuation solve should have a bounded deadline. | `--signal-continuation-deadline-s` defaults to `0.0`, meaning disabled (`run_compression.py:150-156`). | Divergent |
| Production defaults should favor a defensible saturation run. | Five signal points, matched Q1-like basis, no stability, no spatial profiles, and no saved states are the defaults. | Divergent |
| Memory/factorization choices should be explicit and guarded. | The driver estimates memory, limits workers, and exposes exact/Schur and PARDISO/banded choices (`run_compression.py:1073-1198`). | Aligned |

The artifact history reflects this weakness: some long production attempts timed out or failed without a usable saturation artifact. A convergence-study script later added more incremental status handling, but the main production driver still has the all-or-nothing write path.

## Where the result most likely departed from the intended physics

### Confirmed divergence chain 1: physical regime was assumed before it was established

The updated design requires passive characterization before saturation interpretation. The production driver does not enforce it. Later reports show:

- `outputs/exp26_track3_linear_ripple/linear_ripple_report.json`: approximately `2.04 dB` pump-off S21 ripple with a roughly `0.184 GHz` period.
- `outputs/exp27_track1_standing_wave/standing_wave_report.json`: median backward/forward wave ratio about `0.36`, maximum about `0.53`, at the inspected frequency.
- `outputs/exp25_track4_spatial_profile/spatial_metrics.json`: internal peaking factors of about `2.1-5.7` and many spatial-gradient sign changes.

These are not the signatures of a clean, weak-reflection traveling-wave line. They mean that the computed compression curve is entangled with resonant buildup, standing waves, and the chosen terminations. This is a confirmed regime mismatch. Whether it quantitatively explains the entire P1dB-slope discrepancy remains unproven.

The subsequent matched-port experiment is not a clean resolution. Its passive comparison used the rejected 84.6-ohm impedance estimate rather than the branch-node line impedance `sqrt(Lj/Cg)=43.33 ohm` for the exp07 default. The production compression baseline is the unmodified 50-ohm circuit at `outputs\\ipm_python_design`; no 84.6-ohm compression slope exists. A corrected passive rerun gives S21 from -3.766 to -1.251 dB and preserves the 0.1843 GHz period; the termination normalization is fixed, but the resonant ripple remains. The corrected five-current nonlinear sweep gives `-1.843 +/- 0.093 dB/dB` over `G=5.65..12.80`, versus production 50-ohm `-1.185 +/- 0.024 dB/dB` over `G=11.61..15.46`. Those full-span fits are not comparable. Over `G>11`, the corrected slope is `-1.328 +/- 0.043 dB/dB` versus production `-1.185 +/- 0.024 dB/dB`, a 0.14 dB/dB difference. The feature is passive S21 ripple, not a new nonlinear branch-law effect.

### Confirmed divergence chain 2: a low-order signal basis was promoted to a saturation model

The default matched basis is appropriate for a small-signal/Floquet-style response around the pump, but it retains only the signal/idler pair and pump harmonics. Saturation redistributes energy into additional Q orders. Because those orders do not exist in the default basis, Newton convergence only proves convergence of the truncated equations.

Existing Q-axis experiments show large enough P1dB and slope movement to reject the assumption that Q1 is already converged. The production saturation slope was therefore interpreted before the numerical model had passed its most relevant basis-convergence test.

### Confirmed divergence chain 3: the reference gain and P1dB state are internally inconsistent

The current driver defines the low-power reference from the first finite input-power row. That makes P1dB depend on the power-grid start, early continuation behavior, and the estimator used in each comparison dataset. Separate estimator experiments support this sensitivity:

- `outputs/exp26_track1_estimator_matched/estimator_comparison.json` moves the modeled slope from about `-0.387` to about `-0.291` under one measurement-style smoothing choice.
- `outputs/exp25_track1_measured_slope/slope_sensitivity.json` shows the measured slope itself changes materially with fit window, gain floor, and first-downward-crossing rules.

The estimator mismatch does not explain the full measured-versus-modeled gap, but it means the compared slopes were not initially defined by one shared measurement functional.

After refinement, the driver reports the refined P1dB scalar while attaching the nearest coarse row's depletion and a coarse saved state's spatial/stability data. Thus a single summary can combine observables from different nonlinear states. Any mechanism conclusion based on those fields is not a conclusion about the reported P1dB point.

### Confirmed divergence chain 4: the diagnostics could not validate the claimed mechanism

The two most prominent diagnostic fields—pump depletion and power balance—do not implement the all-port contracts now stated in the design:

- Pump depletion observes one output transfer ratio, while the device has multiple outgoing pump paths.
- `power_balance_rel_err` closes internal supplied power against conductance loss rather than closing incoming/outgoing external waves.

Therefore a small reported depletion and an excellent reported balance can coexist even when most pump power leaves elsewhere or the external wave normalization is incomplete. Those fields cannot support the conclusion that pump depletion is negligible or that the port-level observables are fully validated.

### Confirmed divergence chain 5: independent validation stopped too early

The repository has credible L1/L2 checks, but the production result lacks passed L4 convergence and an L3/L6 independent nonlinear oracle. The Le Gal reference benchmark has also not reproduced a complete gain/P1dB/morphology target in the current topology. The shallow slope was consequently compared to experiment before the code path was independently shown to predict saturated behavior in a known case.

## What is not currently shown to be wrong

The evidence does **not** currently identify the following as the primary failure:

- the sign of `delta` or the signal/idler tone labels;
- the real-waveform reconstruction factor;
- the AFT projection-grid construction;
- the basic nonlinear residual or its JVP;
- the pump-only limit;
- the low-power JPA gain fixture;
- the failed-gap handling in P1dB interpolation;
- the use of Newton-Krylov with pump-linear preconditioning.

These areas have direct implementation/test support. They may still contain untested edge cases, but blaming them for the saturation-slope mismatch would go beyond the current evidence.

## Confirmed findings vs causal hypotheses

### Confirmed

- The default saturation basis excludes `|q| >= 2` products.
- Existing Q-axis results do not pass convergence and show material output movement.
- Production Glin is the first finite-power point.
- Refined P1dB diagnostics do not use the refined state.
- Pump depletion is single-port, not all-port.
- The primary power-balance error is based on internal supplied power, not external port waves.
- The modeled device exhibits substantial ripple/backward-wave content in current artifacts.
- The generic production CME and transient oracles are absent.
- Production output is not persisted incrementally.

### Plausible but not yet proven causal

- Standing waves and topology/termination mismatch are likely major contributors to the shallow simulated slope.
- Higher-Q mixing products may change the final slope enough to alter the model/measurement conclusion.
- External port normalization errors may explain some apparently contradictory balance/depletion results.
- A mixing-order-aware Manley-Rowe invariant may be required for the selected three-wave process.
- Pump phase mismatch may contribute at compression, but the current phase proxy did not show phase drift leading the onset of compression.

### Evidence against some simple explanations

- `outputs/exp25_track5_loss/loss_slopes.json` does not show ordinary added dielectric loss producing the steep measured slope; larger loss instead removes the usable gain/P1dB regime.
- `outputs/exp26_track1_estimator_matched/estimator_comparison.json` shows estimator matching changes the modeled slope but does not bridge the full gap to the measured value near `-3 dB/dB`.
- `outputs/exp26_track2_shared_noise/` indicates shared correlated G0 noise changes the fitted measurement slope modestly, not enough to explain the model/measurement discrepancy.
- `outputs/exp25_track3_pump_phase/` does not show the chosen phase-shift proxy leading 0.1 dB compression.

## Bottom line

The first-principles document now describes a defensible solver and validation program, but the current production path implements only part of that program. The most important mismatch is epistemic rather than algebraic: a converged solution of a truncated HB problem was treated as a converged prediction of traveling-wave saturation, and coarse or incomplete observables were then used to explain its mechanism.

The current artifacts support restarting the diagnosis from the passive regime, the full saturation basis, and a single all-port observable contract. They do not support discarding the HB core or attributing the discrepancy to one confirmed nonlinear-solver bug.
