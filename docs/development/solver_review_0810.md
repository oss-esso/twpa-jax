The current high-power limitation is primarily a branch-access and state-handoff problem. It is not a Mode35 limitation, a junction-current limit, or a Krylov/preconditioner
  failure.

  No processes were started, stopped, reprioritized, or inspected during this investigation. No files were changed.

  ## Evidence from the 7.6 GHz column

  Using the correct ipm_2c circuit and measured attenuation profile:

  - Last validated point: −20.736842 dBm
  - Full residual: (3.62\times10^{-12})
  - Peak junction utilization: (I_\mathrm{JJ}/I_c=0.7971)
  - First failed target: −20.210526 dBm
  - Final residual after all recovery: approximately (3.42\times10^{-2})

  At the failed point:

  - Increasing the basis from Mode19 through Mode35 barely changed the residual.
  - GMRES normally required approximately one iteration per Newton step.
  - Pseudo-transient continuation stalled at the same residual.
  - Direct Newton either reached its iteration limit or failed its line search.
  - The exact coupled factorization remained effective.

  The evidence is in the validated /D:/Projects/Thesis/twpa_jax/outputs/campaign_diss/2c_high_power_cleanup_validation_first4_4workers/chunks/chunk_000_cols_000_000/warm/points/
  point_0010_p_m20p7368dbm_fp_7p6ghz/pump/pump_report.json and /D:/Projects/Thesis/twpa_jax/outputs/campaign_diss/2c_high_power_cleanup_validation_first4_4workers/chunks/
  chunk_000_cols_000_000/warm/points/point_0011_p_m20p2105dbm_fp_7p6ghz/pump/pump_report.json.

  Therefore:

  1. Mode35 is not resolving the obstruction.
  2. The linear solver is not the limiting component.
  3. The solver is converging toward the wrong local branch or cannot enter the basin of the higher-power physical branch.
  4. The present globalization and branch-transfer machinery are inadequate for this transition.

  The 7.9 GHz work independently supports this diagnosis. PALC crossed a simple fold, then entered a dense multifold shelf without reaching the physical ramp-selected state. Every
  accepted point was numerically valid, but the traced branch was not the useful high-power branch. See /D:/Projects/Thesis/twpa_jax/docs/development/fold_plan.md:4254 and /D:/
  Projects/Thesis/twpa_jax/docs/development/h3_physical_boundary_79.md:53.

  ## Recommended solver change

  The best targeted architecture is:

  ordinary HB continuation
          ↓
  bounded local PALC/recovery
          ↓
  TD ramp selects the physical high-power state
          ↓
  accurate periodic-orbit projection
          ↓
  residual homotopy to an exact HB root
          ↓
  HB continuation on the recovered high-power branch
          ↓
  gain calculation

  The important new component is a TD-seeded residual homotopy.

  Given the physical HB residual (F(X)) and a Fourier projection (X_\mathrm{TD}) of the transient state, solve

  [
  H(X,\eta)=F(X)-(1-\eta)F(X_\mathrm{TD})=0.
  ]

  At (\eta=0), (X=X_\mathrm{TD}) is an exact solution of the modified residual. At (\eta=1), the equation becomes the original, unchanged physical HB problem:

  [
  H(X,1)=F(X)=0.
  ]

  Advantages:

  - It starts exactly at the TD-selected state rather than asking Newton to cross a large residual gap.
  - It reuses the current AFT residual, exact JVP, Schur reduction, and coupled factorization.
  - Memory remains approximately that of one HB solve.
  - PALC can be used in (\eta) if the homotopy path itself folds.
  - The final result must pass the existing full production residual, so the method does not falsify the physical equations.

  Residual homotopy is not guaranteed to follow the desired root. Final waveform agreement with TD, junction-current continuity, stability, and the full residual gate remain
  mandatory. Homotopy continuation is established for difficult nonlinear systems and circuit problems, but branch validation is still required (Watson’s homotopy overview
  (https://citeseerx.ist.psu.edu/document?doi=6d2791401fde6d810f67e5894c42657fe68c66f3&repid=rep1&type=pdf)).

  ## Required supporting changes

  Before another map run:

  1. Improve the TD projection.

     /D:/Projects/Thesis/twpa_jax/scripts/h1_transient_branch_transfer.py:493 currently uses nt=40, reuses the checkpoint’s harmonic basis, and averages five periods. That is
     insufficient for an HB restart when the projected residual is around (10^{-2}).

     Projection should use:
      - phase alignment to the pump source;
      - one converged final period;
      - timestep refinement;
      - adaptive Fourier order;
      - a full DC/even/odd spectral diagnostic before choosing the HB basis.

  2. Add robust nonlinear globalization.

     The current fixed-drive line search accepts any strict decrease and repeatedly halves the step /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/solver.py:550. The PALC
     corrector is effectively undamped /D:/Projects/Thesis/twpa_jax/src/twpa_solver/pump/solver.py:1194.

     Add:
      - Armijo or cubic backtracking;
      - a trust-region Newton option;
      - the same globalization for the augmented PALC/homotopy corrector.

     Trust-region Newton is an established nonlinear-system fallback, including in PETSc SNES (PETSc nonlinear solver documentation (https://petsc.org/main/manual/snes/)).

  3. Fix validation semantics.

     /D:/Projects/Thesis/twpa_jax/scripts/run_hybrid_column.py:135 can mark a TD-projected state as PASS when gain succeeds, without requiring an exact HB/full-residual gate. An
     approximate TD projection must not count as an ordinary gain-valid HB point.

  4. Fix duplicate arclength seeds.

     When an initial seed is also the first grid point, the map workflow can retain two identical source scales and later call two-point arclength continuation. The continuation
     routine correctly rejects these identical scales. The map seed-history update needs a distinct-scale guard.

  5. Reuse complete branch objects.

     /D:/Projects/Thesis/twpa_jax/scripts/run_gain_map.py:2534 currently starts a new two-seed branch trace for individual failures. Once a high-power root is recovered, trace and
     cache that branch, then resolve multiple requested powers from it.

  ## Shooting fallback

  If residual homotopy cannot polish the TD state into an exact HB root, the next method should be matrix-free shooting:

  [
  R(y_0)=\Phi_T(y_0)-y_0,
  ]

  where (\Phi_T) is one period of the existing implicit trapezoidal transient integrator.

  Use discrete variational equations for the JVP. Do not construct a dense finite-difference monodromy matrix. Matrix-free Krylov shooting is established for large periodic
  steady-state circuit simulation (Telichevesky, Kundert and White (https://kenkundert.com/docs/dac95.pdf)).

  If single shooting is ill-conditioned because many distributed modes have multipliers near one, use 4–8 segment multiple shooting. Shooting should remain a branch bridge, not the
  routine map engine.

  ## Alternatives assessment

   Method                               Assessment
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   More odd harmonics through Mode35    Tested; no material residual reduction. Reject as default.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Dense DC/even/odd HB basis           Use only when refined TD spectra show resolved symmetry breaking.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   More PALC on the low branch          Already enters a dense multifold shelf. Retain for local folds, not physical branch discovery.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Trust-region Newton                  Useful targeted improvement, but cannot alone find a disconnected or remote branch.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   TD-seeded residual homotopy          Highest-priority implementation. Low memory and reuses current HB infrastructure.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Matrix-free shooting                 Strong fallback and periodic-orbit polisher; more expensive than homotopy.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Multiple shooting                    Conditional fallback if single-shooting conditioning fails.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Deflation                            Useful later for discovering additional HB roots, but cannot identify the ramp-selected physical state by itself (Farrell et al. (https://
                                        epubs.siam.org/doi/10.1137/140984798)).
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Full TD gain-map engine              Too expensive; retain only for transitions and physical classification.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   More PTC                             Already stalled at the same representative residual.
  ───────────────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Replacing current sparse HB core     Not justified by the evidence.

  ## Hard acceptance gates

  Use one 7.6 GHz column first.

  1. Run TD from −20.736842 to −20.210526 dBm with timestep refinement.
  2. Establish PERIOD1 closure or classify a physical transition.
  3. Measure DC, even, odd, and subharmonic content relative to the timestep error floor.
  4. Apply residual homotopy.
  5. Accept the recovered state only if:
      - retained and full residuals pass the authoritative production thresholds;
      - the circuit and attenuation provenance match;
  The correct target is:

  > No numerical holes below the modeled physical operating boundary.

  Unstable mathematical periodic orbits beyond that boundary may still be computed, but they must be labelled unstable and must not be presented as usable operating gain.

  Final decision: CURRENT_ARCHITECTURE_NEEDS_TARGETED_CHANGES.

  Keep the HB/AFT residual, exact JVP, Schur/coupled sparse linear algebra, and PERIOD1 gain formulation. Add TD-seeded residual homotopy, scalable shooting fallback, stronger
  globalization, corrected projection, and strict validation semantics.