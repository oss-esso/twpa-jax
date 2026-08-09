# Supervisor Summary

Recent work built a documented acceleration and validation layer across JosephsonCircuits, Harmonia, and twpa_jax.

The main implemented change is not a replacement for nonlinear harmonic balance. It is an opt-in direct `hblinsolve` path for zero-pump linear-response cases in JTL, RF-JTL, and ETHZ-JTL workflows. Each direct backend was added only after an old-vs-new probe compared `hbsolve(...).linearized.S(...)` against `hblinsolve(...)`; the tested S-parameters matched exactly with `max_abs_diff = 0.0`.

The batch runner is the clearest workflow speed improvement. It reuses a Julia process across benchmark or campaign runs, reducing repeated process startup overhead while preserving Julia as the authoritative simulator.

Large direct-only checks reached JTL 30000 cells, RF-JTL 5000 cells, and ETHZ-JTL 10000 cells. RF-JTL 10000 cells produced non-finite S-parameters, which is now recorded as the next numerical boundary to investigate.

The correct conclusion is conservative: direct linear backends clean up solver semantics, expose telemetry, and support large linear-response workflows. They are not a universal warm large-geometry speedup and do not accelerate nonlinear pumped gain workflows. Next work should map RF-JTL numerical validity before expanding to frequency/pump maps or GPU reassessment.
