# Resource Safety And Acceleration

## Current CPU Backend

The nonlinear distributed HB path has a dense reference backend and an explicit
matrix-free Newton-Krylov backend. Dense mode is for tiny validation problems
only. For `N` cells and `K` frequency tones, the
real Newton dimension is:

```text
2 * K * (2 * N + 1)
```

A single dense float64 Jacobian uses the square of that dimension times eight
bytes. Linear solvers and JAX compilation add copies and runtime overhead.

The bounded runner aggregates RSS and CPU across the launched process tree and
terminates descendants first when a limit is exceeded. Use:

```powershell
python scripts/estimate_hb_resources.py --n-cells 8 16 32 64 2000 20000 --n-tones 6
python scripts/run_resource_bounded.py --timeout-s 60 --max-memory-mib 700 `
  --max-disk-mib 64 --disk-root outputs/resource_smoke `
  --log-json outputs/resource_smoke/run.json -- `
  python scripts/pump_hb_small_ladder.py --quick --output-dir outputs/resource_smoke/pump
```

The package refuses unsafe dense pump, gain, and finite-signal HB problems
before Jacobian allocation. Raising the caps is an explicit local-validation
override, not a production scaling strategy.

## Bounded CPU Evidence

The 512-cell matrix-free pump run below was recorded under the process-tree
monitor. This is bounded evidence for the tested configuration, not a general
convergence guarantee.

```powershell
python scripts/run_resource_bounded.py --timeout-s 120 --max-memory-mib 1024 `
  --max-disk-mib 256 --disk-root industrial_completion_outputs/resource_512 `
  --log-json industrial_completion_outputs/resource_512/resource_log.json -- `
  python scripts/pump_hb_small_ladder.py --no-plots --no-checkpoint `
  --numerical-backend newton_krylov --n-cells 512 --n-time 32 --max-iter 20 `
  --continuation-steps 4 --harmonic-orders -1 1 --pump-current-ratio 0.08 `
  --output-dir industrial_completion_outputs/resource_512/run
```

| cells | result | preconditioner | elapsed s | peak RSS MiB | peak disk MiB | residual |
|---:|---|---|---:|---:|---:|---|
| 512 | `PASS` | `linearized_mixed_ladder` | 18.14 | 692.39 | 0.179 | `5.55e-7` to `2.40e-13` |

## Required Structured Backend

Production nonlinear scaling needs:

1. Cell-local residual blocks and neighbor coupling blocks.
2. Matrix-free JVP/VJP operators.
3. Block-Jacobi or linear-ladder preconditioning.
4. Newton-Krylov convergence reports with memory and iteration telemetry.
5. Dense-reference equivalence tests on tiny ladders.

`twpa.solvers.newton_krylov` is wired into pump HB and small-signal gain.
Distributed solves use a matrix-free characteristic-impedance scaling
preconditioner. The next numerical step is a cell-local block-tridiagonal
preconditioner and bounded scaling evidence beyond tiny ladders.

## WSL2 GPU Roadmap

Windows CPU remains the supported baseline. Optional accelerator work belongs
behind backend adapters and separate dependency extras.

Use WSL2 for NVIDIA tooling. CUDA-Q and cuQuantum are optional future backends,
not import-time dependencies:

1. Extend the optional capability protocol in `twpa.accelerators` into backend
   adapters for HB linear operators and batched small-signal solves.
2. Keep NumPy/JAX CPU dense-reference tests authoritative for tiny problems.
3. Add a WSL2 environment file with pinned CUDA, JAX CUDA, CUDA-Q, and
   cuQuantum versions after an NVIDIA GPU runner exists.
4. Add capability detection and explicit `UNKNOWN`/`SKIP` reports when GPU
   libraries are unavailable.
5. Validate CPU/GPU numerical parity before enabling accelerator defaults.

CUDA-Q is relevant only for later quantum-circuit simulation experiments.
cuQuantum may help batched GPU linear algebra, but it does not replace the
required structured classical HB formulation.
