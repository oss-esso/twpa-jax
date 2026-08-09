# Design-independent saturation validation

Status: partial implementation, 2026-07-29.

This note deliberately does not use JosephsonCircuits.jl or the Themis cube as
physical references. JC remains regression-only, and Themis has no
signal-power axis.

## Implemented and measured

The multitone reconstruction convention was checked in
`multitone/observables.py`: port voltage and current coefficients use the
declared factor 2, while `MultiToneDrive` uses the positive-phasor source
coefficient 0.5. This is a direct convention check, not a gain-reference
claim.

The existing JVP test uses `FullMultiToneProblem.jvp_coeffs_with_tangent` and
central differences. It passes at the repository's current test tolerance;
the requested ten-decade curve and fitted slope were not run in this change.

The depletion baseline defect is fixed. `depletion_only_model` remains a
linear power-gain function. The CSV field
`compression_model_depletion_only` now contains the model power gain converted
to dB, `10 log10(G_depletion)`, rather than the raw linear gain.
The focused unit gate agrees with the closed form to `1e-12`. Mutation check:
replacing the logarithm with the raw linear value made the gate fail, then
restoration made it pass.

The existing lossless observable test passes, but it is degenerate: its
single-pump state has zero nonlinear photon-flux scale. It is therefore not
evidence that saturated Manley--Rowe conservation is fixed.

## Eight requested checks

The full acceptance-order campaign was not completed in this implementation.
The existing JVP, zero-signal parity, Panels A--C, sector scaling, phase
covariance, and preconditioner checks remain the previously recorded JPA
measurements; they were not rerun as a single acceptance campaign. Production
basis convergence and a real-gain production Manley--Rowe sweep remain not
evaluated. No JC or Themis value is used as physical evidence.

The finite-signal observable has now been scoped to the primary conversion
channel `(h,q)=(1,0),(1,-1),(1,1)`, with each port-power term divided by its
own angular frequency and with the pump-only reference subtracted. This avoids
counting pump-harmonic generation as signal/idler conversion. The passive toy
fixture is intentionally marked unevaluable for this diagnostic because it has
no conversion channel; its ordinary energy balance still closes. A real-gain
lossless JPA measurement is still required before declaring the blocking
Manley--Rowe gate complete. Mutation verification was performed: replacing the
three-tone selector with all retained tones made the harmonic-only test fail
(`3.33e-31` photon scale became evaluable); restoring the selector returned
`3 passed`.

The depletion-only CSV defect is fixed and covered by a unit gate: the stored
`compression_model_depletion_only` value is now `10 log10(G_depletion)` in dB.
The validation plotting script's Panel C now plots measured pump depletion,
not the separate depletion-only gain baseline.

## Paper benchmark phases

Phase 2 is frozen under `references/le_gal_2025_gain_compression/`. The
contract records published parameters and equations separately from generated
CME arrays; no experimental trace or synthetic digitization is used as an
acceptance target. The independent CME oracle is implemented in `cme.py` and
the map generator is `scripts/reproduce_le_gal_2025_cme.py`. Its focused gates
measure passive propagation, lossless invariant closure below `1e-7`, tighter
integration agreement, and the simple-model estimate near `-107.3 dBm`.
Mutation verification of the CME pump-coupling factor changed the invariant
error to `0.2982` and failed the gate; restoration passed.

The effective branch-law interface and SNAIL line builder are implemented and
covered by unit tests, including save/load preservation of the effective law.

Phase 6 serial HB campaign results are stored in
`references/le_gal_2025_gain_compression/phase6_selected.json`. The reduced
20-cell, 50-cell, and 700-cell devices all converged for 5.0, 6.0, 8.0, and
9.5 GHz at -115, -110, -105, -100, and -94 dBm with sidebands=2. The 700-cell
single-point smoke at 6 GHz/-110 dBm had residual norm `4.95e-20` and runtime
`2.14 s`; the 20-cell serial smoke had residual `2.09e-17`. These are solver
measurements, not paper agreement: the current builder is an effective
linearized ladder and the campaign driver does not yet compute refined P1dB,
spatial phase mismatch, or the paper's calibrated CME coefficients. Therefore
the selected HB results establish convergence and output production only; no
gain, depletion, P1dB, or morphology agreement claim is made.

## Verification

Focused commands:

```text
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\tw_val tests/test_power_balance.py tests/test_compression.py
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\tw_val tests/test_multitone_physics.py tests/test_run_compression_cli.py
```

The focused power/compression command returned `8 passed, 1 skipped`; the
multitone/CLI command returned `33 passed, 1 skipped`. The prior full
repository result was `249 passed, 3 skipped, 4 failed`: exactly the
pre-existing two column-matrix tracer failures and two missing-
`docs/loss_A10.csv` failures. A final full-suite rerun is still required after
the latest observable and plotting changes.

The generated deliverables are
`docs/development/saturation_validation_four_panel.png` and
`docs/development/saturation_validation_jvp.png`. The four-panel figure uses
the lattice-Q=3 sweep so its sector panel contains q=1,2,3.

## Measurement update

The serial JPA validation point used for the measured figures was
`pump_frequency=4.75001 GHz`, `pump_current=11.3 nA`, `signal_frequency=4.75
GHz`, ports `1 -> 1`, positive pump modes `[1,3,5]`, banded factors, and a
signal-current sweep from `1e-12` to `3e-8 A`. The linear Floquet gain-vs-off
value was `7.402467 dB`; the first multitone point was `7.402510 dB`, a gap of
`4.28e-5 dB`.

The JVP curve at the first converged state has descending-branch slope
`1.9414` and minimum relative error `1.566e-12`.

The lattice-Q=3 sweep gives clean low-power sector slopes, fitted over its
first five points: `|q|=1: 1.009/1.012`, `|q|=2: 2.024/2.025`, and
`|q|=3: 3.034/3.035` for negative/positive q. These satisfy perturbative
scaling in that window. The gain curve is nevertheless non-monotone: it rises
from `7.4004 dB` to `13.9505 dB`, then falls below `0 dB` before recovering;
therefore no conventional monotone P1dB claim is made. The maximum HB residual
in the matched sweep was `3.43e-11`; maximum internal power-balance relative
error was `7.33e-10`.

The finite-signal phase-covariance gate passes with measured slopes `+1.000`
for the signal and `-1.000` for the idler; its mutation (expecting `+1` for
the idler) failed before restoration.

The production JTWPA S=10 run was attempted at the documented point
`7.12 GHz` pump, `3.7 µA`, `6.6 GHz` signal, ports `1 -> 2`, odd pump count
10, and banded factors. It exceeded two serial execution budgets (120 s and
300 s) without writing an artifact. S=12 and the resulting `|dP1dB|` are
therefore not evaluable on this machine; no production-basis convergence claim
is made.

The Manley--Rowe diagnostic now exposes its photon-flux scale and an
`evaluable` flag. Below `1e-28` photon-flux units the quotient is dominated by
roundoff and reports zero relative error with `evaluable=false`, rather than
the former artificial `0.49256` floor peak. On the lossless finite-signal JPA
lattice fixture the internal metric was approximately `2.46e-11` at
`1e-12 A` and `4.03e-11` at `1e-10 A`; the external port-wave diagnostic was
not conserved because the fixture's port normalization is not yet closed.
The saturated production conservation gate therefore remains unevaluated.

Mutation evidence for this diagnostic: with the old `1e-30` floor the first
`1e-12 A` point reported `manley_rowe_rel_err=0.4925566`; with the explicit
`1e-28` evaluability threshold it reports `evaluable=false` and relative error
`0.0`. The next `1e-10 A` point is evaluable and reports `0.0031244`.

## Le Gal benchmark fix-plan update (2026-07-30)

The benchmark fix plan corrected four independent artifacts: the
effective-SNAIL law was evaluated away from equilibrium, the 31 fF branch
capacitance was incorrectly added to ground, the loss-tangent argument was
stamped as a dimensionless conductance, and HB gain was formed with a
hand-rolled voltage ratio. The corrected builder reports `L=866.372 pH`,
`Z0=sqrt(L/Cg)=62.3765 ohm`, and equilibrium current residual `1.06e-23 A`
for the 1.4 uA half-flux device. The shifted half-flux law has cubic
coefficient `Ic*(r/6 - 1/162)=0.004160*Ic` (positive).

The driver now uses `tone_s21` and pump-on/pump-off normalization. At the
8-cell, 5.0 GHz, -115 dBm, 7.5 GHz pump point with sideband order 2, the
measurement is `s21=-7.659984 dB`, `gain_vs_off=-0.054994 dB`, pump depletion
`-8.07e-6 dB`, and HB residual `4.69e-21`. This remains a negative result:
the sampled effective line has not reached amplification, no P1dB crossing is
present, and the paper's two-lobed morphology is not reproduced. The earlier
12.041 dB-biased JSON is void.

The corrected Level-2 run covered 20, 50, and 700 cells, 5.0/6.0/8.0/9.5
GHz, and -115/-110/-105/-100/-94 dBm, serially with sideband order 2. All
60 points converged. Across the grid, `gain_vs_off_db` ranged from -0.684521
to +0.353524 dB; there were no 3 dB gain points and no 1 dB P1dB crossings.
At 700 cells the four low-to-high frequency ranges were [0.353174, 0.353524],
[-0.066649, -0.064755], [-0.281919, -0.271932], and [-0.684521, -0.682817]
dB respectively. The earlier order-2 to order-3 comparison is void: the
shifted half-flux CPR populates only odd pump harmonics, so order 3 adds
symmetry-forbidden even-harmonic entries at about 1e-33 flux. The valid
20-cell S=3 to S=5 comparison at 5 and 8 GHz changed gain by at most
`6.77e-8 dB` and `3.29e-9 dB`, respectively. These are self-convergence
results, not paper agreement; the effective line still does not show the
paper's two gain lobes or a finite P1dB.

### CME calibration and CPR convention audit

The independent CME now has nonzero coefficients derived from the published
Taylor coefficient and discrete ladder dispersion. At 6 GHz the inferred
values are `gamma=1.4568982e33 1/(m Wb^2)` and phase mismatch `484.5916 1/m`,
with `Z0=62.3765 ohm`; the derivation and inputs are in `parameters.json`.
A scan with these coefficients became numerically unstable before producing a
trustworthy gain curve, so it does not establish the required two-lobed
morphology. This is a calibration failure to resolve, not evidence that the
HB result is correct. The old zero-coefficient oracle is no longer used as a
comparison.

The alternate CPR placement, with external flux on the ratio/small-junction
term, has equilibrium `Phi*=0` at half flux and cubic coefficient
`-0.016506 Ic`, versus `+0.004160 Ic` for the published placement. A
representative alternate HB probe remains to be rerun after the topology
correction.

The builder topology audit found the old `n`-branch/`n`-node matrix grounded
the first branch. It now uses `n` branches between `n+1` nodes and ports
`{1:0, 2:n}`. This is an intentional topology correction, so all earlier
Level-2 artifacts are superseded pending representative reruns.

Pump-ceiling probes on the topology-corrected 20-cell line at 5 GHz,
signal -115 dBm, sideband order 3 gave: pump -78.4 dBm `SOLVED` in 0.311 s;
pump -58.4 dBm `PUMP_FAILED` after 0.290 s, stalled at continuation lambda
1 with relative coefficient residual `1.126e-1`; and pump -48.4 dBm
`PUMP_FAILED` after 0.080 s, failing at lambda 0.25 with residual `1.805e-2`.
These are pump-solver ceiling measurements, not saturation results.

The alternate-CPR probe at 20 cells and 5/6/8/9.5 GHz remained effectively
identical to the published-placement probe at the sampled -115 and -94 dBm
levels; the largest observed gain difference was below `2e-13 dB`. This
low-signal equivalence is expected because the two conventions share the
same linearized slope at half flux; it does not resolve the nonlinear
convention question.
