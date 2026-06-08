# Profiling Methodology

Profiling preceded backend changes. The intent was to identify solver bottlenecks before choosing acceleration work.

## HB Timing Phases

JosephsonCircuits profiling split HB execution into coarse blocks:

1. Frequency bookkeeping.
2. Circuit matrix construction.
3. Nonlinear pump HB solve.
4. Linearized signal/idler solve.
5. Port/output conversion.

Evidence:

- `JosephsonCircuits.jl\experiments\thesis_gpu_parallel\run_hbsolve_profile.jl`
- `JosephsonCircuits.jl\experiments\thesis_gpu_parallel\analyze_five_block_timing.jl`
- `outputs\jc_profiles\jc2f_five_block_report\five_block_summary.csv`
- `outputs\jc_profiles\jc2f_five_block_report\five_block_timing_report.md`

## Decision-Safe Timing

`JosephsonCircuits.jl\experiments\thesis_gpu_parallel\reports\jc3_decision_note.md` records that:

- linearized signal/idler solve was often a large block,
- circuit/matrix construction was a strong secondary bottleneck,
- setup-cache work should be integrated at repeated workflow/campaign level first,
- one-shot `hbsolve` speedup should not be assumed.

## Setup Cache Probes

Setup-cache probes investigated repeated setup reuse without changing public solver semantics:

- `probe_compiled_setup_cache.jl`
- `probe_hbsetupcache_vs_rebuild.jl`
- `probe_repeated_setup_reuse.jl`
- `run_cached_setup_workload.jl`

The documented conclusion is workload-shaped: cache value matters for repeated workflows, not as a guaranteed one-shot speedup.

## Timing Rules

- Separate cold-start and warm-run timings.
- Keep public API timings and staged internal timings separate.
- Report failures and non-finite outputs.
- Avoid using smoke timings as broad performance claims.
- Treat process reuse, direct solver path, and numerical validity as separate concerns.
