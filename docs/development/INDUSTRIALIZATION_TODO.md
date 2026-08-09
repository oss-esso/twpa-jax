# Industrialization TODO

## P0 Resource Safety

- [x] Refuse unsafe dense pump HB allocations before Jacobian construction.
- [x] Refuse unsafe dense gain linearization allocations before matrix construction.
- [x] Reduce full pump `--quick` mode to an 8-cell reference smoke.
- [x] Add bounded subprocess monitoring for time, RSS, and artifact size.
- [x] Add formula-only HB memory estimation.
- [x] Add CI jobs that run bounded smoke commands on Windows and Linux CPU.

## P1 Structured Classical HB

- [x] Wire `twpa.solvers.newton_krylov` into distributed pump HB.
- [ ] Build cell-local block Jacobian extraction without dense materialization.
- [x] Add matrix-free characteristic-impedance scaling preconditioning.
- [x] Wire cell-local block-Jacobi preconditioning for tiny validated ladders.
- [ ] Add linear-ladder or block-tridiagonal preconditioner for scalable GMRES convergence.
- [ ] Add continuation telemetry: residual, GMRES iterations, RSS, disk, and checkpoint size.
- [ ] Prove dense versus structured agreement on 1, 2, 4, and 8 cells.
- [ ] Benchmark 16, 32, 64, 128, and 256 cells only under resource limits.
  Current evidence: `32/64/128` cells are memory-safe under auto-selected preconditioning, but remain `PARTIAL` because first-step GMRES hits `max_iter`.
- [x] Record an honest bounded 512-cell evidence run.
  Result: `PASS` with `linearized_mixed_ladder`, `18.14 s`, `692.39 MiB` peak RSS, `0.179 MiB` peak disk, and residual reduction from `5.55e-7` to `2.40e-13`.

## P2 Gain And Compression

- [x] Route gain solves through structured operators.
- [x] Add guarded matrix-free finite-signal HB for commensurate reduced problems.
- [x] Add generic multi-fundamental torus projection and wire the 2D DP4WM finite-signal path.
- [x] Route target-frequency and wideband compression CLI modes to package-native finite-signal sweeps.
- [x] Keep coupled-mode and gain-script fallbacks labelled `PARTIAL`.
- [x] Add package-native in-process pump-grid gain-map orchestration.
- [ ] Add physical gain-map fixtures with expected phase matching and passivity checks.

## P3 Calibration

- [x] Add package-native linear S-parameter recovery with explicit identifiability reporting.
- [x] Add package-native HB-backed nonlinear recovery datasets.
- [x] Add finite-difference rank, condition-number, and parameter-correlation diagnostics.
- [ ] Add measurement schema validation and deterministic fixture datasets.

## P4 Workflow Hygiene

- [x] Exclude generated report summaries and the active output directory from recursive report scans by default.
- [x] Deduplicate bridge sources by resolved path and kind.
- [x] Keep bridge source JSON compact by default with an explicit full-embed debugging flag.

## P5 Optional WSL2 Acceleration

- [x] Add optional accelerator capability protocol without import-time CUDA dependencies.
- [ ] Add backend adapters for CPU and accelerator operators.
- [ ] Add pinned WSL2 CUDA environment after a GPU runner exists.
- [ ] Add optional JAX CUDA backend.
- [ ] Evaluate cuQuantum for batched GPU linear algebra after structured HB exists.
- [ ] Evaluate CUDA-Q only for later quantum-circuit experiments.
- [ ] Require CPU/GPU numerical parity tests before enabling accelerator defaults.

## Read-Only Reuse Review

`../twpa_learning_pack` is documentation and a small dense reference scaffold.
Reuse its explanations and tiny cross-check fixtures only. Do not copy its
dense normal-equation solver into the production nonlinear path.
