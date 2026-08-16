# Phase 4 — stability instrument selection

## Selection

The selected production instrument is the branch-tracked Hill route:

| instrument | measured result | decision |
|---|---|---|
| branch-tracked Hill scan | 152.2 s, 700 points, four sidebands; explicit loss convention | selected |
| Koopman-Hill projection | not implemented in this repository | not selected |
| time-domain monodromy | analytic reference accurate, but lossy 2c closure rejects complex `C` | control only |

The analytic damped-oscillator control at 96 steps per period gave:

```text
exact spectral radius       0.8110386975
monodromy spectral radius   0.8110788441
absolute error              4.01e-05
```

This validates the existing monodromy implementation on its supported real
state representation. It does not validate applying it to dielectric-loss
circuits, which require a real-time loss model rather than complex frequency-
domain capacitance.

The benchmark artifact is
`.hybrid_outputs/phase4_instrument_benchmark/benchmark.json`.

The corrected TD mode gate independently retained a usable voltage trace. Its
dominant frequencies were `7.8846`, `7.9000`, and `7.8691 GHz` at 15.43 MHz
resolution; the exact `3.9500 GHz` component was not dominant. This is the
current mode-frequency evidence available for cross-route comparison.

## Phase 5 gate

The gate is not satisfied: the selected-loss Hill sequence has no accepted,
sideband- and density-converged multiplier crossing at the Phase 2 boundary,
and the matching L-stable transient evidence is unresolved. No PERIOD2,
period-N, torus, or auxiliary-generator ansatz was enabled.
