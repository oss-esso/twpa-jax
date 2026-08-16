# Gated solver plan for the chaos route

This document is intentionally a decision record, not an implementation of a
new harmonic-balance ansatz.  The ansatz gate in `CLAUDE.md` remains closed
until the transient campaign is timestep-converged, sideband-converged, and
corroborated by an L-stable run.

The diagnostic route reports one of the following classes for each device and
continuation direction.  The solver action is then selected from the matching
row; a disagreement between transient geometry and the Themis frequency test
is recorded as `UNDETERMINED` rather than averaged.

| measured class | gated solver action |
| --- | --- |
| `PERIOD_DOUBLING` | Lift the gate only after review. Reuse `pump/floquet.py::period_doubled_basis`, `pump/periodic_branch.py`, `signal/period_doubled.py`, and `scripts/run_period_doubled_branch.py`. Add no second ansatz. |
| `NEIMARK_SACKER` | Add an auxiliary-generator closure on the existing multitone `(h,q)` lattice. Promote `delta` to an autonomous unknown, solve for `(A_a, omega_a)`, and impose the two real `Y_AG = 0` equations in an outer continuation loop. |
| `CHAOS_NO_CLEAN_BIFURCATION` | Add a validity-boundary diagnostic that halts harmonic balance and reports the chaotic-attractor boundary as physics, not as a convergence failure. No chaotic ansatz is introduced. |
| `NO_BIFURCATION_FOUND` | Make no ansatz change. Re-run the existing continuation-recovery ladder because the wall is numerical or a transient/integration boundary. |

The production change is therefore gated by evidence. Until both the
Poincare geometry and the hardware-side classifier agree, this file is the
complete solver deliverable.

## Current evidence gate (2026-08-12)

The real Themis corpus was processed by
`scripts/chaos/themis_wm_classifier.py`. It contains 51 response cubes on a
`4--12 GHz` signal grid and a `0.335 dB` pump-power grid. The result is
explicitly `UNDETERMINED: cube verdicts disagree`: 49 cubes are unresolved and
2 are provisionally tagged `NEIMARK_SACKER` by the frequency heuristic. The
secondary inverse-power fits are available in
`outputs/chaos/phase4/themis_wm_classification.json`; 46 cubes have enough
amplifying points for a fit and 5 correctly report that the grid is too short.
The spread of fitted exponents is not evidence for a single class.

The Phase-3 device gate is now partially open: a validated 2c checkpoint is
available under `.hybrid_outputs/hb_up_7p9_m35_to_m21/...point_0010.../pump`,
and a bounded rf-SQUID HB smoke checkpoint is available under
`outputs/chaos/phase3/rf_hb_coarse/...point_0000.../pump`. The full final
device verdict remains gated on broader power resolution and longer
convergence checks. Consequently the selected solver action is still:

> `NO_SOLVER_CHANGE_UNTIL_PHASE3`: preserve the existing ansatz gate and run
> both-direction, timestep-converged attractor continuation before changing the
> solver ansatz.

This is a measured, gated outcome—not a classification of either device.

The Phase-3 executable has since been smoke-tested against the validated 2c
checkpoint at `.hybrid_outputs/hb_up_7p9_m35_to_m21/...point_0010.../pump`.
The short ladder report is
`outputs/chaos/phase3/ladder_smoke/campaign_summary.json`: all five resistance
settings ran in both directions. The lossless through `R/Rn=10^2` settings
remain compact over this short hold; `R/Rn=1` produces at least one
`CHAOS_NO_CLEAN_BIFURCATION` point. Because the hold is only two periods, this
is an integration/plumbing smoke, not the required final device verdict.

The longer 10-period 2c report is
`outputs/chaos/phase3/ladder_hold10/campaign_summary.json`. All 30
direction/ratio/point records pass the envelope-decay gate; the measured
`sigma` range is recorded per point and the provisional geometric verdicts
remain mixed between `NO_BIFURCATION_FOUND` and
`CHAOS_NO_CLEAN_BIFURCATION`. This does not establish a bifurcation class
because the three-point power bracket does not resolve intermediate attractor
structure.

The rf-SQUID also passed a bounded production-HB smoke at `-60`, `-50`, and
`-40 dBm` (residuals below `6e-11`) and a two-ratio, 10-period continuation in
both directions. Its report is
`outputs/chaos/phase3/rf_hold10/campaign_summary.json`; all 12 records pass the
decay gate and remain provisional for the same resolution reason.
