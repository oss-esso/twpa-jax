# Decision Log

## Profiling First

Decision: profile HB internals before choosing acceleration work.

Reason: five-block timing and setup-cache probes showed where time was spent and prevented premature GPU work.

Evidence: `JosephsonCircuits.jl` commits `15b4e82`, `b0b0d89`, `9d1caee`, `1997df3`; `outputs\jc_profiles\jc2f_five_block_report`.

## Keep Batch Runner

Decision: keep the Julia batch runner and expose it through benchmark and campaign CLIs.

Reason: repeated workflows benefit from Julia process reuse. This is a real workflow speed improvement independent of solver backend changes.

Evidence: `twpa_jax` commits `7054f09`, `c9837f6`, `aa29db6`; `Harmonia.jl` commit `259bde0`; tests for `julia_batch_runner`.

## Direct Linear Backends Are Opt-In

Decision: add direct `hblinsolve` backends for JTL, RF-JTL, and ETHZ-JTL linear response only.

Reason: equivalence probes showed exact S-parameter agreement in tested zero-pump cases. Direct backends clarify semantics and telemetry.

Evidence: `Harmonia.jl` commits `a9afbde`, `28098a1`, `60168e9`, `f112555`, `ca299ec`, `ea45d25`; output reports with `max_abs_diff = 0.0`.

## No Universal Speedup Claim

Decision: do not claim major warm large-geometry speedup over `hbsolve(...).linearized.S(...)`.

Reason: largest warm comparisons are close:

- JTL 3000: direct 0.207 s vs old path 0.211 s.
- RF-JTL 2393: direct 2.470 s vs old path 2.466 s.
- ETHZ-JTL 2048: direct 0.225 s vs old path 0.297 s.

Evidence: `outputs\jc_profiles\jc3m_scaled_direct_linear_showcase\scaled_direct_linear_showcase.csv`.

## Do Not Patch lumped_jpa_linear Yet

Decision: stop rollout after JTL, RF-JTL, and ETHZ-JTL. Do not expand to `lumped_jpa_linear` without a separate equivalence probe.

Reason: lumped JPA linear reflection is a distinct workflow and was not covered by the direct backend equivalence probes.

Evidence: `outputs\benchmarks\jc3m_m7_direct_linear_backend_report\direct_linear_backend_report.md`.

## Do Not Jump To GPU Yet

Decision: defer GPU/accelerator reassessment.

Reason: CPU numerical validity and bottlenecks are not fully understood, especially RF-JTL non-finite boundaries.

Evidence: `JosephsonCircuits.jl\experiments\thesis_gpu_parallel\reports\jc3_decision_note.md`; RF-JTL 10000-cell failure in extreme showcase CSV.

## Next Step

Decision: map RF-JTL S-parameter finite/non-finite boundary first.

Reason: RF-JTL 5000 direct-only passed, but RF-JTL 10000 failed with non-finite S-parameters. Numerical validity must be mapped before frequency/pump maps or accelerator work.
