# Architecture

The stack has three layers. Each layer owns a different responsibility.

## JosephsonCircuits.jl

`JosephsonCircuits.jl` is the solver layer. Recent work added internal timing instrumentation, five-block HB analysis scripts, and setup-cache probes. The authoritative solver APIs remain Julia functions such as `hbsolve` and `hblinsolve`.

Relevant evidence:

- `JosephsonCircuits.jl` commit `15b4e82`: thesis HB profiling experiment.
- `JosephsonCircuits.jl` commit `9d1caee`: five-block HB timing balance.
- `JosephsonCircuits.jl` commit `810c5bb`: internal HB setup cache boundary.
- `JosephsonCircuits.jl` commit `024c905`: cached setup workload runner.
- `experiments\thesis_gpu_parallel\run_hbsolve_profile.jl`
- `experiments\thesis_gpu_parallel\reports\jc3_decision_note.md`

## Harmonia.jl

`Harmonia.jl` is the Julia simulation backend and CircuitIR layer. It builds JTL, RF-JTL, ETHZ-JTL, lumped JPA, and other templates, exports circuits into JosephsonCircuits-compatible form, runs the selected solver path, and writes simulation artifacts.

Recent work added:

- CircuitIR templates and topology smokes for JTL, RF-JTL, ETHZ-JTL, and lumped JPA.
- `scripts\run_simulation_batch.jl` for long-lived Julia batch execution.
- Opt-in direct `hblinsolve` backends for JTL, RF-JTL, and ETHZ-JTL linear cases.

Relevant evidence:

- `Harmonia.jl` commit `259bde0`: batch simulation runner boundary.
- `Harmonia.jl` commits `a9afbde`, `28098a1`, `60168e9`: equivalence probes.
- `Harmonia.jl` commits `f112555`, `ca299ec`, `ea45d25`: direct backend integration.
- `Harmonia.jl` commit `631fa39`: large direct linear showcase experiments.

## twpa_jax

`twpa_jax` is the Python orchestration layer. It launches Julia simulations, reads outputs, batches campaigns, builds datasets, evaluates objectives, runs benchmark suites, and writes reports. It does not replace the Julia simulation backend.

Relevant evidence:

- `twpa_jax` commit `7054f09`: Julia batch runner wrapper.
- `twpa_jax` commit `aa29db6`: benchmark-suite batch mode.
- `twpa_jax` commits `eaac53a`, `fdec801`, `1032d6c`: direct backend flags in benchmark suite.
- `twpa\io\julia_runner.py`
- `twpa\io\julia_batch_runner.py`
- `scripts\run_harmonia_benchmark_suite.py`

## Data Flow

1. Python builds a benchmark or campaign manifest.
2. Python launches one Julia process per run, or one long-lived Julia batch process.
3. Harmonia resolves CircuitIR templates and solver configuration.
4. JosephsonCircuits runs `hbsolve` or `hblinsolve`.
5. Harmonia writes HDF5/status/log artifacts.
6. Python reads artifacts and writes CSV/JSON summaries.

Confirmed behavior comes from source files and tests listed in [Evidence Manifest](evidence_manifest.md). Claims about runtime or exact equivalence are limited to measured artifacts under `D:\Projects\Thesis\outputs`.
