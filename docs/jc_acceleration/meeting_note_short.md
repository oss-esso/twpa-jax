# Meeting Note

Recent work made the linear-response path across Harmonia and JosephsonCircuits explicit and testable.

JTL, RF-JTL, and ETHZ-JTL linear workflows now have opt-in direct `hblinsolve` backends. The default remains `hbsolve`, and direct mode is guarded to zero pump current. Nonlinear pumped gain workflows still use full HB.

Equivalence probes compared `hbsolve(...).linearized.S(...)` against direct `hblinsolve(...)`; tested S-parameters matched with `max_abs_diff = 0.0`. Largest exact comparisons were JTL 3000 cells, RF-JTL 2393 cells, and ETHZ-JTL 2048 cells.

Large direct-only checks reached JTL 30000, RF-JTL 5000, and ETHZ-JTL 10000 cells. RF-JTL 10000 cells produced non-finite S-parameters and is the next numerical boundary to map.

Main interpretation: batch runner/process reuse is the practical workflow speedup; direct `hblinsolve` is a semantics, telemetry, and large linear-response cleanup. It is not a universal nonlinear HB speedup claim.
