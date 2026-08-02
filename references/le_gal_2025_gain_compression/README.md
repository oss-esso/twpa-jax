# Le Gal et al. gain-compression benchmark

This directory freezes the paper benchmark contract for the simulated curves
in *Gain compression in Josephson Traveling-Wave Parametric Amplifiers*,
arXiv:2502.03022v2, DOI:10.1103/tq8k-m3dr.

`parameters.json` contains published parameters and the unit conventions used
by the independent coupled-mode implementation. `digitized_full_theory_*`
files are reserved for values digitized from simulated curves in the paper;
experimental traces are deliberately kept separate and are not acceptance
targets. Values marked `inferred` are calculated from published parameters
(for example `sqrt(L/Cg)`) and are never treated as paper data.

The source paper and figure numbers are recorded in `parameters.json`. Every
imported curve must be accompanied by a SHA-256 entry in `checksums.json` and
must state the digitizer, figure, axis calibration, and uncertainty in its
header or sidecar metadata. No experimental ripple or exact minimum is part
of this contract.

The implemented line solves the SNAIL static equilibrium and evaluates the
dynamic law at `Phi* + deltaPhi`. The SNAIL capacitance is a branch-parallel
stamp, not an additional ground capacitor. With the published values the
linearized inductance is 866.372 pH and `sqrt(L/Cg)=62.3765 ohm`, consistent
with the idealized 62.4 ohm port contract. Levels 1 and 2 are in scope; the
700-cell Level-3 morphology sweep is intentionally not run. The corrected
campaign is a solver measurement only and makes no claim of reproducing the
paper's gain lobes or P1dB.

The line topology is `n` nonlinear branches between `n+1` nodes, with ports
at nodes 0 and n. The former grounded-first-branch construction is retained
only in superseded artifacts. The independent CME coefficient derivation and
the alternate external-flux CPR convention are recorded in `parameters.json`;
the calibrated CME scan currently becomes numerically unstable before a
two-lobed curve can be established.
