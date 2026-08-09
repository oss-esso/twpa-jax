# JosephsonCircuits Acceleration Work

This documentation records recent acceleration, profiling, and orchestration work across three repositories:

- `D:\Projects\Thesis\JosephsonCircuits.jl`: solver internals, HB timing instrumentation, setup-cache probes.
- `D:\Projects\Thesis\Harmonia.jl`: Julia simulation backend, CircuitIR templates, direct linear backend integration.
- `D:\Projects\Thesis\twpa_jax`: Python orchestration, benchmark campaigns, datasets, reports, and tests.

The work did not replace nonlinear harmonic balance. Julia remains the authoritative simulation backend. Python orchestrates runs, batches, datasets, benchmark summaries, and reports.

## Main Outcomes

| Area | Outcome | Evidence |
|---|---|---|
| HB profiling | Five-block timing and setup-cache probes identified where time is spent before backend changes. | `JosephsonCircuits.jl` commits `15b4e82` through `a40b152`; `outputs\jc_profiles\...` reports |
| Batch runner | `twpa_jax` can reuse one Julia process for benchmark/campaign runs. | `twpa_jax` commits `7054f09`, `c9837f6`, `aa29db6`; `twpa\io\julia_batch_runner.py` |
| Direct linear backends | JTL, RF-JTL, and ETHZ-JTL support opt-in direct `hblinsolve` for zero-pump linear response. | `Harmonia.jl` commits `f112555`, `ca299ec`, `ea45d25`; `scripts\run_simulation.jl` |
| Equivalence | Old `hbsolve(...).linearized.S(...)` and direct `hblinsolve(...)` produced `max_abs_diff = 0.0` in tested zero-pump cases. | `outputs\jc_profiles\jc3m_m*_hbsolve_vs_hblinsolve\*.json`; `scaled_direct_linear_showcase.csv` |
| Large linear showcase | Direct-only runs reached JTL 30000, RF-JTL 5000, and ETHZ-JTL 10000 cells; RF-JTL 10000 exposed a non-finite S-parameter boundary. | `outputs\jc_profiles\jc3m_extreme_direct_linear_showcase\extreme_direct_linear_showcase.csv` |

## Correct Interpretation

Use direct `hblinsolve` as an explicit zero-pump linear-response backend. It is exact-equivalent for tested cases, improves semantics and telemetry, and supports large direct-only linear-response workflows.

Do not describe it as a universal speedup. Warm large-geometry timings are close to `hbsolve(...).linearized.S(...)`, because the old zero-pump path already spends much of its work in linearized solve machinery. The stronger workflow speed improvement is the Julia batch runner and process reuse.

## Documents

- [Quickstart](quickstart.md)
- [Architecture](architecture.md)
- [Direct Linear Backends](direct_linear_backends.md)
- [Benchmark Suite](benchmark_suite.md)
- [Batch Runner](batch_runner.md)
- [Profiling Methodology](profiling_methodology.md)
- [Equivalence Probes](equivalence_probes.md)
- [Large Scale Showcase](large_scale_showcase.md)
- [Numerical Boundaries](numerical_boundaries.md)
- [API Reference](api_reference.md)
- [Reproducibility](reproducibility.md)
- [Decision Log](decision_log.md)
- [Roadmap](roadmap.md)
- [Supervisor Summary](supervisor_summary.md)
- [Evidence Manifest](evidence_manifest.md)
