• ## Executive conclusion

  The current -58…-56 dBm failure is primarily a solver-integration problem, not evidence of a physical boundary.

  I reproduced the actual flat/on-chip drive on the 2393-cell circuit:

   Pump basis               Result at -56 dBm
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   [1,2,3]                  fails at continuation scale 0.75
  ───────────────────────  ────────────────────────────────────────────────────
   [0,1,2,3]                reaches full drive, but full time residual 4.24e-2
  ───────────────────────  ────────────────────────────────────────────────────
   [0..5]                   reaches full drive, full time residual 3.04e-3
  ───────────────────────  ────────────────────────────────────────────────────
   [0..9], warm-enriched    converges, full time residual 7.42e-5

  The main production path still constructs [1,2,3] for 3WM and omits dynamic DC: /D:/Projects/Thesis/twpa_jax/scripts/run_gain_map.py:780. The standalone rf-SQUID
  script already prepends zero: /D:/Projects/Thesis/twpa_jax/scripts/run_rf_squid_3wm.py:102.

  The default-attenuation map is misleading here: it passes through -56 dBm because the configured loss model reduces the injected current. The flat/on-chip map
  fails at -59, -58, -57, -56 dBm, exactly matching the reported problem.

  ## A. Ranked root causes

  ### 1. Missing dynamic DC in the production 3WM basis — confirmed primary cause

  The production gain-map path resolves positive modes only. The rf-SQUID DC bias is passed into the nonlinear branch law, but the pump waveform has no dynamic k=0
  unknown.

  This causes the production solver to solve the wrong finite-dimensional problem. Adding zero changes the result decisively:

  - [1,2,3]: failure at λ=0.75
  - [0,1,2,3]: successful continuation to λ=1

  The harmonic grid itself supports zero correctly: /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/problem.py:17. The integration into the production map is
  incomplete.

  ### 2. The production acceptance gate ignores omitted-harmonic residual — confirmed

  The solver accepts coeff_rel, while time_rel is diagnostic only: /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/validation.py:75.

  The existing H=3 checkpoint had approximately:

  - coefficient residual: 1.87e-10
  - full reconstructed time residual: 1.29e-2

  Thus “HB converged” currently means “the retained modes balance,” not “the full nonlinear DAE is balanced.”

  The residual tail moves to the next omitted harmonic as the basis is enriched. This requires adaptive harmonic enrichment and a full-residual gate.

  ### 3. The real-coupled preconditioner is mathematically wrong when k=0 is present — confirmed new bug

  The real-coupled Jacobian assumes every mode has an independent positive-frequency conjugate pair:

  [
  L V_q + P\overline{V_q}.
  ]

  That is correct for q>0, but not for q=0. The DC coefficient is real and has reconstruction weight one; it must not receive the positive-mode conjugate treatment.

  Affected code:

  - /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/problem.py:305
  - /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/problem.py:446
  - /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/backends/schur_operators.py:295
  - /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/backends/fast_coupled.py:210

  Direct comparison against the exact AFT JVP gave:

  - no DC mode: relative mismatch ~1e-16
  - with DC mode: relative mismatch ~2.7e-3

  This explains why higher-H real-coupled solves can become extremely GMRES-sensitive. The AFT residual/JVP path is still correct; the coupled preconditioner is not.

  ### 4. Harmonic enrichment interacts badly with continuation/preconditioning — confirmed numerical issue

  A cold H7 continuation became a GMRES wall, while an H5-promoted warm H9 solve succeeded. This means the next basis must be reached by continuation from the
  previous converged/enriched state, not by restarting from zero.

  The current Schur reduction and PARDISO-backed fast coupled preconditioner are valuable and should be retained, but they need the DC-special Jacobian correction.

  ### 5. Period doubling is possible but not currently demonstrated at -56 dBm

  The paper explicitly reports pump subharmonic generation at fp/2, so period doubling must be tested rather than dismissed. Gaydamachenko et al.
  (https://arxiv.org/abs/2503.02489)

  My preliminary Tier-1 Floquet/Hill test around fp/2 = 6.04 GHz, with DC included, showed no deep half-tone singularity for the provisional H5 state. This is only
  negative evidence:

  - the H5 state still had full residual 3e-3
  - only five sideband blocks were used
  - sigma_min is a candidate-resonance diagnostic, not a stability proof

  The existing stability script also omits the DC branch flux when constructing gamma_hat: /D:/Projects/Thesis/twpa_jax/scripts/floquet_stability_sweep.py:101. Its
  rf-SQUID results are therefore currently invalid.

  ### 6. No evidence yet of a true fold before -56 dBm

  The corrected DC-inclusive H3/H5/H9 paths reach the target drive. Therefore the old failure cannot be interpreted as a physical fold.

  A fold remains possible in the fully converged high-H branch, but it must be established using the existing singularity diagnostics after harmonic and residual
  convergence. A Newton failure alone is insufficient.

  ### 7. Power and port provenance remain hazardous

  The rf-SQUID artifact has only:

  - port 1: pump/source
  - port 2: load/output

  /D:/Projects/Thesis/twpa_jax/outputs/rf_squid_3wm_fig3a_pumpcheck/ports.csv

  But run_gain_map.py defaults to pump port 4: /D:/Projects/Thesis/twpa_jax/scripts/run_gain_map.py:3120. Every workflow should require or validate ports against the
  circuit rather than relying on that default.

  Pump reports also do not persist dc_branch_flux, attenuation, source convention, and physical pump mode sufficiently. That makes detached gain/stability analysis
  unsafe.

  ## B. Recommended architecture

  ### 1. Period-1 production pump

  Use:

  - base frequency fp
  - explicit modes [0,1,…,H]
  - solved RF-SQUID DC branch flux
  - Schur reduction
  - exact AFT JVP
  - corrected real-coupled preconditioner
  - adaptive H enrichment

  The preferred sequence is:

  H=3 + DC
    -> full residual spectrum
    -> H=5
    -> H=7
    -> H=9 ...
    -> production gate

  Each enrichment should warm-start from the previous state.

  Do not replace HB with transient integration.

  ### 2. DC handling

  The cleanest implementation is to represent DC as a real-only unknown. If retaining the current complex packing, add special q=0 handling to:

  - spectral JVP
  - full real-coupled Jacobian
  - fast coupled Schur preconditioner
  - any mode-coupled derivative implementation

  Add a unit test comparing AFT JVP with spectral/real-coupled JVP for bases both with and without zero.

  ### 3. Harmonic convergence

  Use three independent checks:

  1. retained coefficient residual;
  2. full reconstructed time-domain residual;
  3. residual spectrum outside the retained basis.

  Also repeat with doubled nt. The current nt >= 2*max(mode)+1 check is only a Nyquist condition; it is not an anti-aliasing/convergence check.

  ### 4. Period-doubling branch

  For a period-doubling candidate, use:

  [
  f_0 = f_p/2.
  ]

  Represent the pump with:

  - k=1: half-tone
  - k=2: physical pump
  - source_mode=2
  - explicit DC mode k=0

  This requires metadata distinguishing:

  - base frequency,
  - physical pump frequency,
  - physical pump mode,
  - source mode.

  The paper and prior Josephson-circuit literature support this approach. Wiesenfeld et al. (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.29.2102) and
  period-doubling HBM continuation study (https://www.sciencedirect.com/science/article/pii/S0960077905003929)

  ### 5. Gain

  For a stable Period-1 state, the existing Floquet gain solver is appropriate after:

  - DC-aware pump loading,
  - full pump residual validation,
  - sideband convergence checks.

  For a Period-2 state, the gain solver must use the half-frequency lattice. For the example fs=6.83 GHz, fi=5.25 GHz, the idler is reached through the m=-2 sideband
  relative to f0=6.04 GHz, not the current Period-1 m=-1 convention.

  Do not change the gain solver until the pump orbit classification is known.

  ## C. Minimal implementation plan

  ### Phase 1 — correctness fixes

  1. Add dynamic DC to run_gain_map.py 3WM bases.
  2. Add DC branch flux and source/port/power metadata to every pump report.
  3. Make rf-SQUID port validation explicit.
  4. Fix zero-mode handling in:
      - spectral JVP,
      - real-coupled Jacobian,
      - fast coupled preconditioner.

  5. Add tests:
      - zero-mode synthesis;
      - DC residual balance;
      - AFT-vs-spectral JVP;
      - AFT-vs-real-coupled Jacobian;
      - persisted checkpoint round-trip.

  ### Phase 2 — adaptive HB

  1. Solve [0,1,2,3].
  2. Compute oversampled residual spectrum.
  3. Promote to H5/H7/H9 as required.
  4. Require full residual and nt convergence.
  5. Persist all enriched states with float64 for validation.

  Suggested gates:

  - exploratory state: time_rel < 1e-4;
  - production state: time_rel <= 1e-8, plus H/nt convergence;
  - gain state: pump state passes production gate and gain linear residual independently passes.

  ### Phase 3 — stability and branches

  1. Fix the stability script’s DC handoff.
  2. Scan the first Floquet zone, especially fp/2.
  3. Refine candidate complex resonances.
  4. Compute the pump Floquet multiplier closest to -1.
  5. If a flip crossing exists, seed the fp/2 HB branch from the Floquet eigenvector.
  6. Use TD only to confirm the physical attractor and ramp-selected branch.

  ### Phase 4 — gain and maps

  1. Run gain around the validated Period-1 state.
  2. If Period-2 is selected, use the generalized half-frequency gain lattice.
  3. Only then integrate the route into fast/slow gain maps.

  PALC, shooting, and deflation should be secondary tools:

  - PALC: true fold or branch geometry;
  - shooting/TD: branch transfer and physical attractor validation;
  - deflation: discovering additional coexisting branches after the primary branch is understood. Deflation is established for finding distinct nonlinear solutions,
    but is not the first fix here. Farrell, Birkisson & Funke (https://doi.org/10.1137/140984798)

  ## D. Scalability

  The existing architecture is suitable for 2k–10k nonlinear elements if the retained space and harmonic count remain controlled.

  Keep:

  - Schur elimination of linear internal nodes;
  - matrix-free AFT residual/JVP;
  - cached sparse symbolic factorizations;
  - PARDISO fast coupled factors;
  - HB for maps;
  - TD only at transitions.

  Avoid:

  - full real-coupled SuperLU on every Newton step;
  - brute-force TD maps;
  - dense multitone Jacobians for ordinary gain maps.

  At the current circuit:

  - full nodes: 7180
  - Schur retained nodes: 4787
  - H5 real-packed reduced system: approximately 57k unknowns
  - H9 real-packed reduced system: approximately 96k unknowns

  The H5 Schur/PARDISO solve completed in roughly six seconds in the targeted test. The main scaling risk is harmonic coupling and factor memory, not the nonlinear
  branch count alone.

  Period-2 HB roughly doubles the required base-period resolution and can increase retained harmonic count, so it should be activated only near detected transitions.

  ## E. Hard decision tree

  1. Does the checkpoint carry valid DC flux, ports, mode list, source convention, and current?
      - No → reject checkpoint.
      - Yes → continue.

  2. Does Period-1 HB with [0,1,2,3] converge?
      - No → fix zero-mode preconditioner and retry Schur/PARDISO.
      - Yes → continue.

  3. Is the full residual or omitted-harmonic tail above the production gate?
      - Yes → enrich H and warm-start.
      - No → continue.

  4. Does H/nt refinement converge to the same state and gain?
      - No → basis or aliasing problem; do not classify physically.
      - Yes → continue.

  5. Does Floquet analysis show a multiplier approaching/crossing -1, or does TD show d2 small while d1 remains large?
      - Yes → construct Period-2 HB at f0=fp/2.
      - No → retain Period-1 representation.

  6. Does the enriched Period-1 Jacobian become singular?
      - Minimum eigenvalue approaches zero and bordering/determinant diagnostics agree → true fold; use PALC.
      - No singular trend, GMRES worsens → numerical/preconditioner issue.
      - Branch continues → no physical boundary.

  7. Does TD settle to Period-1 or Period-2?
      - Period-1 → project back to Period-1 HB.
      - Period-2 → use Period-2 HB/gain formulation.
      - Persistent nonperiodic/running phase → physical state result.
      - Unresolved → extend hold/adaptive classification; never call it chaos automatically.

  ## F. Existing bugs and duplicated logic

  Most important:

  - production 3WM map omits k=0;
  - real-coupled/spectral Jacobians mishandle q=0;
  - production validation gates only retained coefficient residual;
  - Floquet stability loader omits RF-SQUID DC flux;
  - pump reports do not persist complete DC/provenance metadata;
  - summarize_solution() reports dynamic branch current without the absolute DC offset: /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/io.py:13;
  - transient observables similarly omit DC offset: /D:/Projects/Thesis/twpa_jax/scripts/h1_transient_branch_transfer.py:396;
  - Periodic TD projection currently projects one pump period only: /D:/Projects/Thesis/twpa_jax/scripts/h1_transient_branch_transfer.py:487;
  - standalone rf-SQUID pump logic and production map logic are duplicated;
  - standalone defaults to compute_time_residual=False;
  - production checkpoints default to float32 storage;
  - rf-SQUID map defaults to invalid pump port 4;
  - MultiToneBasis forbids DC, which will matter if 3WM compression is later extended to include dynamic DC.

  The parts to retain are the physical circuit formulation, exact sine law, DC branch solve, AFT residual/JVP, Schur reduction, fast sparse preconditioning, PALC,
  and DAE transient solver.

  The immediate fix is not RCSJ damping, looser tolerances, or more Newton iterations. It is:

  integrated DC mode
  + correct DC Jacobian/preconditioner treatment
  + adaptive harmonic enrichment
  + full-residual acceptance
  + explicit Floquet period-doubling gate

  Baseline visible tests passed (9 passed), and no production source files were modified during this investigation. A generated untracked research_runtime/ directory
  remains from the diagnostics.