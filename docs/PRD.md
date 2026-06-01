# PRD: Agentic Workflows for the Julia–Python TWPA Simulation and Calibration Platform

**Document status:** Draft v0.1  
**Owner:** Edoardo / Thesis repo  
**Workspace context:** VS Code workspace with three sibling repositories:

```text
<workspace>/
  Thesis/       # Python orchestration, ML, calibration, dataset tooling, reference checks
  Harmonia/     # Julia exploratory/development workflows and reusable prototype code
  Harmonia.jl/  # Julia package/core circuit-generation layer
```

This PRD defines how coding agents should work across the three repositories to build an industrial-grade simulation and calibration platform for TWPA/JTWPA/KITWPA/rf-SQUID/IPM designs.

---

## 1. Executive Summary

The platform must use **Julia as the authoritative physics simulation engine** and **Python as the orchestration, calibration, Bayesian inference, SBI, ML, and reporting layer**.

The target architecture is:

```text
Harmonia / Harmonia.jl
  -> circuit/topology generation
  -> JosephsonCircuits.jl harmonic-balance simulation
  -> S/QE/CM/noise/gain/compression outputs
  -> HDF5/JLS/status export

Thesis Python repo
  -> launch Julia simulations
  -> validate status and resources
  -> read HDF5 outputs
  -> build ML-ready datasets
  -> run Bayesian optimization / SBI / surrogate models
  -> produce campaign reports
```

The Python repo already contains a package-native TWPA/HB stack and resource-safety infrastructure. This code is valuable as a reference implementation, structured-HB sandbox, regression-test suite, and fallback diagnostics, but **production JTWPA/rf-SQUID/IPM physics should rely on JosephsonCircuits.jl through Harmonia-style Julia workflows**.

The product goal is not “more scripts.” It is a **status-aware, reproducible, schema-driven simulation and calibration factory** where every run is traceable, cacheable, validated, and usable by ML without guessing what happened.

---

## 2. Problem Statement

Current simulation/calibration work is split across:

1. a Python research package with strong lower-layer tests and resource-bounded HB experiments;
2. a Julia package/prototype ecosystem that can generate complex Josephson/IPM/rf-SQUID circuits and call JosephsonCircuits.jl;
3. exploratory notebooks/scripts and partial workflows that are hard to compose, hard to validate, and hard to feed into ML.

The missing industrial layer is the connective tissue:

- one stable Julia simulation CLI;
- one versioned simulation-result schema;
- one Python run manager and cache;
- one HDF5 reader/dataset builder;
- one campaign framework for calibration and ML;
- strict status semantics that prevent fallback or failed simulations from being interpreted as valid results.

---

## 3. Product Goals

### G1. Reproducible Julia simulation engine

Agents must create or stabilize a Julia entry point that can run a simulation from a machine-readable config and always produce a structured output folder, even on failure.

### G2. Python orchestration layer

The Python repo must launch Julia simulations, track run status, cache by config hash, read outputs, and build datasets.

### G3. ML-ready data contract

Every simulation must produce data that can become:

```text
x = physical/design parameters
y = S-parameters, gain, noise/QE/CM, compression metrics, pump solution, etc.
status = PASS/PARTIAL/FAIL/UNKNOWN
mask = valid frequency/sweep points
metadata = solver/config/environment provenance
```

### G4. Calibration loop

The system must support a closed loop:

```text
Python proposes parameters
  -> Julia simulates
  -> Python reads HDF5/status
  -> Python computes loss vs measurement/synthetic target
  -> optimizer proposes next batch
```

### G5. Agent-safe development

Every agent run must leave logs, command traces, changed-file summaries, and validation results. No agent may silently hide failures or run unsafe dense HB jobs.

---

## 4. Non-Goals

The following are explicitly out of scope for the first industrial workflows:

- reimplementing JosephsonCircuits.jl in Python;
- making the Python HB implementation the production simulator;
- running unbounded 20,000-cell dense HB;
- building SBI before the simulator/data schema is stable;
- training neural networks before there is a validated dataset builder;
- marking coupled-mode, analytic-surrogate, or fallback outputs as `PASS`;
- enabling GPU/CUDA backends by default.

---

## 5. Current State Inputs

### 5.1 Python Thesis repo

The Python repo is a research package for TWPA/KITWPA linear modeling, HB experiments, gain-map workflows, synthetic calibration, and parameter recovery. The documented developer path is:

```powershell
python -m pip install -e .
python -m pytest
```

Direct script execution is expected from the repo root. Pump, gain-map, compression, and nonlinear recovery workflows default to package-native HB paths, while legacy gain-script, coupled-mode, and analytic-surrogate routes must remain explicit `PARTIAL` diagnostics.

### 5.2 Resource safety state

The nonlinear distributed HB path has:

- dense reference backend for tiny validation only;
- matrix-free Newton-Krylov backend;
- resource estimation;
- bounded subprocess runner;
- refusal gates for unsafe dense pump/gain/finite-signal allocations.

Dense real Newton dimension is:

```text
2 * K * (2 * N + 1)
```

for `N` cells and `K` frequency tones. A dense float64 Jacobian scales as the square of that dimension times eight bytes, plus solver/JAX overhead.

A bounded 512-cell matrix-free pump run has been recorded with:

```text
preconditioner: linearized_mixed_ladder
elapsed:        ~18.14 s
peak RSS:       ~692 MiB
peak disk:      ~0.179 MiB
residual:       5.55e-7 -> 2.40e-13
```

This is evidence for one bounded configuration, not a universal convergence claim.

### 5.3 Industrialization TODO state

Completed or partially completed Python-side capabilities include:

- unsafe dense allocation refusal;
- bounded runner;
- memory estimation;
- Newton-Krylov integration into pump HB;
- matrix-free characteristic-impedance scaling;
- tiny validated block-Jacobi preconditioning;
- structured gain and finite-signal compression paths;
- package-native pump-grid gain-map orchestration;
- linear and nonlinear recovery datasets;
- identifiability diagnostics;
- optional accelerator capability probes.

Open items include:

- cell-local block Jacobian extraction without dense materialization;
- scalable linear-ladder/block-tridiagonal preconditioner;
- continuation telemetry;
- dense-vs-structured agreement on 1, 2, 4, 8 cells;
- resource-bounded benchmarks for 16–256 cells;
- physical gain-map fixtures;
- measurement schema validation and deterministic fixtures;
- backend adapters for accelerators.

### 5.4 Accelerator policy

CPU is the supported baseline. Optional WSL2/NVIDIA tooling remains opt-in. JAX CPU is the reference for tiny dense-vs-structured checks. GPU parity tests are required before any accelerator becomes default.

---

## 6. User Personas

### P1. Thesis developer

Needs fast iteration, strict scientific validation, clean campaign reports, and clear distinction between real HB results and fallback diagnostics.

### P2. Simulation specialist

Needs topology correctness, solver status, physical parameter provenance, resource limits, and reproducible sweeps.

### P3. ML/calibration developer

Needs reliable HDF5/dataframe/tensor outputs, status masks, failure metadata, priors/bounds, and campaign history.

### P4. Coding agent

Needs explicit repo boundaries, commands, acceptance criteria, safety rules, and deliverable paths.

---

## 7. Repo Boundaries

### 7.1 `Harmonia.jl`

Authoritative location for stable Julia package code:

- circuit generation;
- reusable topology builders;
- CPW/directional-coupler synthesis;
- rf-SQUID/IPM/JTWPA templates;
- HDF5/JLS export helpers;
- stable Julia API.

Agents may modify this repo only when the task explicitly targets package stabilization.

### 7.2 `Harmonia`

Exploratory Julia workspace:

- old examples;
- development scripts;
- fitting prototypes;
- IPM/rf-SQUID experiments;
- reference scripts to mine for reusable patterns.

Agents should treat this repo primarily as read-only archaeology unless explicitly asked to clean it.

### 7.3 `Thesis`

Authoritative Python repo for:

- orchestration;
- Julia process runner;
- status/cache registry;
- HDF5 reader;
- dataset builder;
- calibration objectives;
- Bayesian optimization;
- SBI;
- ML surrogates;
- reporting;
- Python reference HB checks and resource-bounded experiments.

Agents should usually make production workflow code here.

---

## 8. Product Requirements

## R1. Agent run protocol

Every agentic workflow must create a run folder:

```text
Thesis/outputs/agent_runs/<YYYYMMDD_HHMMSS>_<short_task_name>/
  commands.md
  edit_log.md
  test_progress.md
  changed_files.md
  final_report.md
```

Required command preamble:

```powershell
git status --short
git diff --stat
python --version
python -m pytest -q
```

If Julia is involved:

```powershell
julia --version
julia --project=<path> -e "using Pkg; Pkg.status()"
```

The final report must state:

- commands run;
- files changed;
- tests before/after;
- scripts executed;
- artifacts generated;
- failures and why they remain.

---

## R2. Simulation status semantics

All scripts and agents must use the same statuses:

```text
PASS
  Solver/workflow completed, numerical outputs are finite, residual/status checks pass,
  and output is physically meaningful for the claimed workflow.

PARTIAL
  Workflow ran but used fallback/surrogate/compatibility mode, lacks enough validation,
  or only covers a reduced diagnostic case.

FAIL
  Solver failed, residual is NaN/Inf, arrays are nonfinite, validation mismatch occurred,
  resource limits were exceeded, or required inputs were missing.

UNKNOWN
  Not run, not inspected, or insufficient metadata exists.
```

Forbidden state:

```text
status = PASS
residual_norm = NaN or Inf
```

Fallback outputs must never be `PASS`.

---

## R3. Julia simulation CLI

A stable Julia CLI must exist or be created:

```text
Harmonia.jl/scripts/run_simulation.jl
```

Required invocation:

```powershell
julia --project=Harmonia.jl Harmonia.jl/scripts/run_simulation.jl `
  --config <config.json> `
  --output <run_dir>
```

Required behavior:

- parse JSON config;
- resolve circuit template;
- call Harmonia/JosephsonCircuits simulation;
- write structured output folder;
- write status even on failure;
- never silently drop failed sweep points.

Supported initial simulation types:

```text
linear_sparams
pump_hb
small_signal_gain
gain_map
compression_sweep
disorder_sweep
defect_sweep
calibration_point
```

MVP may implement only `linear_sparams` and one tiny `small_signal_gain` case, but schema must allow the full list.

---

## R4. Versioned output schema

Every Julia run must produce:

```text
run_manifest.json
status.json
config_resolved.json
simulation.h5
stdout.log
stderr.log
```

Minimum `status.json`:

```json
{
  "schema_version": "0.1.0",
  "run_id": "string",
  "status": "PASS | PARTIAL | FAIL | UNKNOWN",
  "simulation_type": "string",
  "circuit_template": "string",
  "solver_success": true,
  "residual_norm": 1e-10,
  "relative_residual_norm": 1e-8,
  "failure_reason": null,
  "runtime_s": 0.0,
  "random_seed": 1234,
  "julia_version": "string",
  "josephsoncircuits_version": "string",
  "harmonia_commit": "string",
  "python_commit": "string"
}
```

Minimum HDF5 groups:

```text
/config
/parameters
/axes
/results/S
/results/gain
/results/QE
/results/CM
/results/noise
/results/pump_solution
/status
/metadata
```

For each sweep point, store status and failure reason.

---

## R5. Python Julia runner

Create or stabilize:

```text
Thesis/twpa_calibration/engine/
  julia_runner.py
  run_config.py
  run_status.py
  run_registry.py
  cache.py
```

Required public functions/classes:

```python
@dataclass
class SimulationConfig:
    simulation_type: str
    circuit_template: str
    parameters: dict
    sweep: dict
    solver: dict
    output: dict
    seed: int

@dataclass
class SimulationResult:
    run_id: str
    status: str
    output_dir: Path
    h5_path: Path | None
    residual_norm: float | None
    failure_reason: str | None
    runtime_s: float | None

def run_julia_simulation(config: SimulationConfig, force: bool = False) -> SimulationResult: ...
def read_status(run_dir: Path) -> SimulationResult: ...
def hash_config(config: SimulationConfig) -> str: ...
```

Requirements:

- no manual `PYTHONPATH`;
- subprocess stdout/stderr captured;
- timeout support;
- config-hash cache;
- failures retained, not hidden;
- resource-bounded mode available.

---

## R6. HDF5 reader and dataset builder

Create or stabilize:

```text
Thesis/twpa_calibration/data/
  hdf5_reader.py
  dataset_schema.py
  dataset_builder.py
  validators.py
```

Required behavior:

```python
read_simulation_h5(path: Path) -> SimulationSample
build_dataset(run_dirs: list[Path]) -> SimulationDataset
validate_dataset(dataset) -> ValidationReport
```

Dataset must preserve:

- physical parameters;
- sweep axes;
- complex S matrices;
- gain/noise/QE/CM outputs when present;
- per-point masks;
- run status;
- failure reasons;
- config hash;
- random seed;
- circuit/template identity.

Acceptable storage:

- scalar metadata: CSV/Parquet;
- tensor data: HDF5/Zarr/xarray;
- campaign state: JSON/CSV plus tensor store.

---

## R7. Circuit template configs

Create stable template schemas:

```text
Thesis/configs/templates/
  standard_jtwpa.yaml
  rpm_jtwpa.yaml
  rf_squid_twpa.yaml
  ipm_jtwpa.yaml
  rf_ipm_twpa.yaml
```

Each template must define:

- nominal parameters;
- units;
- bounds;
- calibratable/fixed flags;
- ports;
- pump convention;
- default frequency axes;
- default solver settings;
- expected output quantities.

Example parameter field:

```yaml
Lj:
  value: 45e-12
  unit: H
  bounds: [30e-12, 70e-12]
  calibratable: true
```

No calibration script may hardcode physical bounds that should live in the template.

---

## R8. Golden validation tests

Add cross-repo integration tests under:

```text
Thesis/tests/golden/
```

Required test classes:

```text
test_julia_cli_smoke.py
test_hdf5_schema.py
test_status_semantics.py
test_linear_sparams_reciprocity.py
test_dataset_builder.py
test_calibration_loss.py
```

Golden simulations must be tiny and fast:

1. passive TL/JTL linear S-parameters;
2. one tiny Josephson chain or rf-SQUID cell;
3. one tiny directional coupler linear S-parameter run;
4. one tiny gain point if feasible.

Acceptance:

```powershell
python -m pytest tests/golden -q
```

must pass locally without large memory usage.

---

## R9. Calibration objectives

Create:

```text
Thesis/twpa_calibration/calibration/
  objectives.py
  metrics.py
  priors.py
  transforms.py
```

Required loss components:

- complex S-parameter loss;
- gain curve loss;
- return-loss penalty;
- band-edge error;
- peak-gain error;
- ripple penalty;
- pump-frequency/pump-power nuisance terms;
- failed-simulation penalty.

Outputs must include scalar loss and diagnostics:

```python
@dataclass
class LossResult:
    total: float
    components: dict[str, float]
    status_penalty: float
    valid: bool
```

---

## R10. Bayesian optimization campaign

Create:

```text
Thesis/twpa_calibration/optim/
  bayesopt.py
  acquisition.py
  campaign.py
```

Campaign folder:

```text
Thesis/campaigns/<campaign_id>/
  campaign_config.yaml
  trials.csv
  runs/
  reports/
  cache/
```

Trial table fields:

```text
trial_id
config_hash
run_id
parameter values
status
loss
runtime_s
failure_reason
output_dir
```

MVP acceptance:

```text
A 10-trial synthetic or tiny-Julia calibration campaign runs end-to-end.
```

---

## R11. SBI and ML layers

SBI and ML are second-order after runner/schema/dataset/campaign are stable.

Create only after R1–R10 are functional:

```text
Thesis/twpa_calibration/sbi/
  simulators.py
  summaries.py
  posterior.py

Thesis/twpa_calibration/ml/
  datasets.py
  models.py
  train.py
  evaluate.py
```

Required initial summary statistics:

- peak gain;
- center frequency;
- bandwidth above threshold;
- gain ripple;
- mean return loss;
- selected S21 samples;
- PCA/embedding of gain curve.

Acceptance for SBI:

```text
Synthetic posterior recovery works on known generated data.
```

Acceptance for ML:

```text
Surrogate predicts held-out gain summaries better than a naive mean/nearest baseline.
```

---

## R12. Campaign reports

Create:

```text
Thesis/twpa_calibration/reports/
  campaign_report.py
  dataset_quality_report.py
  failure_report.py
  best_fit_report.py
```

Required one-command report:

```powershell
python scripts/make_campaign_report.py --campaign campaigns/<id>
```

Report must answer:

- what was simulated;
- what failed;
- best-fit parameters;
- identifiability/correlation diagnostics;
- uncertainty;
- dataset coverage;
- next recommended simulation points.

---

## R13. Resource-bounded execution

Any potentially expensive simulation must run through bounded execution.

Python-side required CLI:

```powershell
python scripts/run_resource_bounded.py --timeout-s <seconds> --max-memory-mib <MiB> `
  --max-disk-mib <MiB> --disk-root <dir> --log-json <path> -- <command...>
```

Julia runner integration must support:

- timeout;
- memory limit when launched through Python wrapper;
- disk budget;
- process-tree termination;
- JSON resource log;
- status `FAIL` or `PARTIAL` when bounded run exceeds limits.

---

## 9. Agent Roles

## A1. Orchestrator Agent

Responsibilities:

- read PRD;
- create run folder;
- inspect all three repos;
- decompose work;
- assign subagents or phases;
- enforce acceptance criteria;
- produce final report.

May edit:

- primarily `Thesis`;
- other repos only when explicitly needed.

Must not:

- hide failing subprocesses;
- claim research-grade status without tests and artifacts.

---

## A2. Julia Engine Agent

Responsibilities:

- stabilize `Harmonia.jl` simulation CLI;
- mine `Harmonia` exploratory scripts for reusable code;
- define template-to-circuit path;
- write HDF5/status outputs.

May edit:

- `Harmonia.jl`;
- selected copied/adapted code from `Harmonia` only with attribution/comments.

Must produce:

- Julia tests;
- example config;
- output folder from one tiny run.

---

## A3. Python Bridge Agent

Responsibilities:

- subprocess launcher;
- config hashing;
- cache;
- status parsing;
- resource-bounded launch;
- logs.

May edit:

- `Thesis/twpa_calibration/engine`;
- scripts.

---

## A4. Data Schema Agent

Responsibilities:

- HDF5 reader;
- dataset builder;
- validators;
- schema docs.

Must produce:

- schema document;
- tests using fixture HDF5 files.

---

## A5. Calibration Agent

Responsibilities:

- measurement/synthetic target loaders;
- objectives;
- priors/bounds;
- Bayesian optimization campaign.

Must not:

- start SBI before runner/data layer is working.

---

## A6. ML/SBI Agent

Responsibilities:

- summary statistics;
- surrogate datasets;
- baseline models;
- SBI posterior recovery.

Must use:

- validated dataset builder outputs only.

---

## A7. Resource Safety Agent

Responsibilities:

- resource estimation;
- bounded runs;
- dense allocation refusal tests;
- CPU/GPU backend capability boundaries.

Must enforce:

- no unbounded dense 100 mm HB;
- GPU opt-in only;
- CPU remains baseline.

---

## A8. Documentation Agent

Responsibilities:

- architecture docs;
- run reports;
- campaign summaries;
- API examples;
- “known limitations” sections.

Must not:

- overwrite technical caveats with marketing language.

---

## 10. Workflow Blueprints

## W1. Cross-repo archaeology workflow

Purpose: extract reusable Julia pieces from `Harmonia` into `Harmonia.jl` or interface docs.

Steps:

1. inventory files;
2. classify as stable package code / reusable prototype / one-off experiment / obsolete;
3. identify dependencies;
4. map functions to target API;
5. write migration table;
6. do not edit until table is reviewed.

Deliverable:

```text
Thesis/docs/harmonia_reuse_map.md
```

---

## W2. Julia CLI workflow

Purpose: make one simulation runnable from config.

Steps:

1. create example config;
2. resolve template;
3. build circuit;
4. run tiny simulation;
5. write HDF5/status;
6. run from Python subprocess;
7. validate schema.

Acceptance:

```text
Python can launch Julia and read PASS/PARTIAL/FAIL status.
```

---

## W3. Dataset workflow

Purpose: convert run folders into ML samples.

Steps:

1. collect run dirs;
2. parse statuses;
3. read HDF5 tensors;
4. create scalar table;
5. create tensor store;
6. validate masks and axes;
7. export dataset manifest.

Acceptance:

```text
20 mixed-status runs become one dataset with failures retained.
```

---

## W4. Calibration campaign workflow

Purpose: run optimization loop.

Steps:

1. load template and measurement/synthetic target;
2. sample initial parameters;
3. run Julia simulations;
4. compute loss;
5. update surrogate/acquisition;
6. propose next batch;
7. save trials;
8. generate report.

Acceptance:

```text
10-trial tiny campaign completes and writes report.
```

---

## W5. Resource-bound scaling workflow

Purpose: gather honest scaling evidence.

Steps:

1. estimate memory;
2. choose bounded configs;
3. run through bounded subprocess;
4. record RSS/disk/runtime/residual;
5. classify PASS/PARTIAL/FAIL;
6. update scaling table.

Acceptance:

```text
No scaling claim exists without resource logs.
```

---

## W6. Measurement ingestion workflow

Purpose: prepare real data for calibration.

Steps:

1. define measurement schema;
2. load CSV/HDF5/VNA data;
3. apply calibration metadata;
4. unwrap phase;
5. align frequency axes;
6. store processed target;
7. validate units and masks.

Acceptance:

```text
One deterministic fixture target passes schema validation.
```

---

## 11. File and Directory Conventions

### 11.1 Python package

```text
Thesis/twpa_calibration/
  engine/
  data/
  calibration/
  optim/
  sbi/
  ml/
  reports/
```

### 11.2 Configs

```text
Thesis/configs/
  templates/
  campaigns/
  measurements/
```

### 11.3 Outputs

```text
Thesis/outputs/
  agent_runs/
  julia_runs/
  datasets/
  campaigns/
  resource_smoke/
```

### 11.4 Docs

```text
Thesis/docs/
  architecture/
  schemas/
  workflows/
  harmonia_reuse/
```

---

## 12. Acceptance Criteria by Milestone

## M0. Architecture freeze

- `docs/architecture/julia_python_boundary.md` exists.
- PRD committed or copied into `Thesis/PRD.md`.
- Agent run protocol documented.

## M1. Julia simulation smoke

- one Julia CLI exists;
- one config runs;
- output folder contains required files;
- Python can parse `status.json`.

## M2. Python runner

- config hashing works;
- cache works;
- stdout/stderr captured;
- failure writes status;
- tests pass.

## M3. HDF5 dataset builder

- HDF5 fixture read;
- scalar/tensor dataset created;
- statuses retained;
- invalid runs masked.

## M4. Calibration MVP

- one target loaded;
- loss computed;
- 10-trial campaign runs;
- report generated.

## M5. Industrial skeleton

- golden tests pass;
- resource-bounded smoke exists;
- schema docs exist;
- CI or local validation command documented;
- no fallback PASS outputs.

---

## 13. Safety and Quality Gates

Before merge or acceptance, agents must run:

```powershell
python -m pytest -q
```

and any focused tests they touched.

For Julia changes:

```powershell
julia --project=Harmonia.jl -e "using Pkg; Pkg.test()"
```

For cross-repo workflows:

```powershell
python -m pytest -q tests/golden
```

Resource-sensitive jobs must use:

```powershell
python scripts/run_resource_bounded.py ...
```

Generated outputs must not be committed unless they are tiny fixtures intentionally placed under `tests/fixtures`.

---

## 14. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Agents build another Python HB simulator | Wastes time and duplicates JosephsonCircuits | Keep production boundary explicit |
| Fallback outputs treated as valid | Invalid ML dataset | Strict status semantics and tests |
| Failed sweep points disappear | Biased calibration | Store per-point status/failure reason |
| Huge HB run kills machine | Lost time/resources | Use resource estimation and bounded runner |
| HDF5 schema drifts | Python reader breaks | Version schema and validate |
| Harmonia exploratory code copied blindly | Fragile package | Migration table and tests |
| ML starts before data is trustworthy | False confidence | Gate SBI/ML behind dataset validation |
| GPU dependency breaks Windows CPU | Unusable baseline | Accelerator optional, CPU reference |
| No measurement fixtures | Calibration untestable | Create deterministic synthetic fixtures first |

---

## 15. Open Questions

1. Which Julia repo should own the stable CLI: `Harmonia.jl` only, or a thin wrapper package inside `Thesis`?
2. What is the minimum JosephsonCircuits.jl output needed for first calibration: `S21` only, or full S/QE/CM?
3. Which first physical template should be stabilized: standard JTWPA, rf-SQUID TWPA, or IPM JTWPA?
4. What real measurement formats are expected: VNA Touchstone, CSV, HDF5, lab-specific exports?
5. Should Python use `pandas + h5py`, `xarray + zarr`, or both for dataset storage?
6. What optimizer is preferred for MVP: BoTorch, scikit-optimize, Optuna, or custom Bayesian optimization?
7. Which parameters are first calibratable: `Lj`, `Cj`, `Cg`, `sigma`, pump attenuation, flux offset, or loss?
8. How large can local bounded runs be on the target machine?
9. Which artifacts are allowed in git as fixtures?
10. How much of `Harmonia` exploratory code should be promoted into `Harmonia.jl` before Python integration begins?

---

## 16. Ruthless Implementation Order

Do this in order:

1. Write/copy this PRD to `Thesis/PRD.md`.
2. Freeze Julia/Python boundary document.
3. Create Julia simulation CLI with one tiny linear simulation.
4. Define status/HDF5 schema.
5. Implement Python runner/cache/status parser.
6. Implement HDF5 reader and schema validator.
7. Create golden tiny Julia fixtures.
8. Build dataset builder.
9. Implement calibration losses.
10. Run 10-trial Bayesian optimization MVP.
11. Generate campaign report.
12. Add measurement schema and deterministic fixtures.
13. Add surrogate ML baseline.
14. Add SBI only after hundreds/thousands of valid samples exist.
15. Scale to IPM/rf-SQUID and larger sweeps under resource bounds.

---

## 17. Definition of Industrial Grade

The platform is industrial-grade only when:

- every simulation is config-driven;
- every run has a status file;
- every failure is preserved;
- every dataset is schema-validated;
- every calibration campaign is reproducible;
- every expensive run is resource-bounded;
- every result can be traced to git commit, config hash, environment, and solver settings;
- no fallback result is indistinguishable from a real HB result;
- Python can consume Julia outputs without manual interpretation;
- reports are generated automatically from artifacts, not hand-written after the fact.

Industrial-grade does **not** mean the largest possible simulation. It means **trustworthy automation under explicit physical, numerical, and resource constraints**.
