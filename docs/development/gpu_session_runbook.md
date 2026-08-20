# GPU session runbook

> **PAUSED 2026-08-20 — never executed.** The toolchain below is built,
> CPU-smoke-tested and committed, but no GPU session was ever run, so **no
> measurement in this repository depends on it**. The route was paused in
> favour of other alternatives, not because anything here failed. See
> "Why this is paused" before reviving it.

This runbook targets the available GPU machine, an **RTX 3060 laptop**
(Ampere GA106; an earlier draft said RTX 4060 and that was wrong -- see the
throughput note below, the two parts differ by a factor of two in FP64). The
repository must be on `dev` at the commit that contains the committed fixture
and session scripts.

## Why this is paused

The kernel runs `jax_enable_x64=True` throughout, and consumer NVIDIA parts
cripple FP64. Spec-based peak, against this development machine's CPU:

| part | FP32 | FP64 ratio | FP64 | vs CPU |
| --- | ---: | ---: | ---: | ---: |
| dev CPU, 6c @4GHz AVX2 FMA | -- | -- | 0.384 TFLOPS | 1.00x |
| RTX 3060 laptop (Ampere) | 10.94 | 1/32 | 0.342 | 0.89x |
| RTX 3060 desktop (Ampere) | 12.74 | 1/32 | 0.398 | 1.04x |
| RTX 4060 laptop (Ada) | 15.10 | **1/64** | 0.236 | 0.61x |

So in float64 a 3060 is **at parity** with the CPU already in use -- the trip
only pays if float32 is adequate, which would buy 28-33x. That is what
`measure_kernel_precision.py` decides, and it has never been run on hardware.

The second unmeasured question is whether `solve_kind="scan"` wins at all. The
associative-scan banded solve trades `O(n)` depth for `O(log n)` at roughly
100x the arithmetic (bandwidth 5 gives 10x10 transfer matrices), and it
measures **29x slower than sequential on CPU**. Whether the depth reduction
pays on a GPU is exactly what `benchmark_batched_fdtd.py` exists to answer.

**Nothing here is known to be broken.** The paused state is "built and
untested on target hardware", not "tried and failed".

## Reviving this

1. Run the commands below on the GPU box. `run_gpu_session.ps1` is
   self-validating: preflight hard-fails rather than degrading, and the
   precision study prints its own GO/NO-GO.
2. Read `precision.json` first. A `NO_GO` means float32 misses the `1e-4`
   observable tolerance, and the honest conclusion is that a consumer GPU does
   not help this problem -- record it and stop.
3. `nvidia-smi` before the CUDA install, to confirm the driver is visible and
   to read the actual VRAM (3060 laptop is commonly 6 GB, desktop 12 GB).
   After the chunked-scan fix outputs are 5 MB/lane, so batch 64 needs only
   0.32 GB and VRAM is not expected to bind.
4. On a laptop part, watch for thermal throttling: the benchmark excludes
   warm-up from timing, so throttling shows as declining throughput at a fixed
   configuration rather than as an error.

## Install and run

Paste these commands into PowerShell on the GPU machine:

```powershell
Set-Location D:\Projects\Thesis\twpa_jax
git switch dev
git pull --ff-only origin dev
python -m pip install --upgrade "jax[cuda12]"
python -c "import twpa_solver; print(twpa_solver.__file__)"
powershell.exe -ExecutionPolicy Bypass -File .\scripts\chaos\run_gpu_session.ps1
```

The script creates one timestamped directory under
`outputs\chaos\gpu_sessions`. It runs, in order, preflight, the precision
study, and the benchmark. A stage with an existing output is skipped, so an
interrupted session can be resumed by invoking the same script with its output
directory:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\chaos\run_gpu_session.ps1 `
    -OutputDir .\outputs\chaos\gpu_sessions\gpu_YYYYMMDD_HHMMSS
```

Use `-Force` to rerun existing stages:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\chaos\run_gpu_session.ps1 `
    -OutputDir .\outputs\chaos\gpu_sessions\gpu_YYYYMMDD_HHMMSS -Force
```

## What a pass looks like

`preflight.json` has `status: "PASS"`; its float64/sequential comparison is at
most `1e-11` relative to the committed Numba fixture. `precision.json` has
`verdict: "GO"`, meaning the float32 pump-only off-lattice fraction differs
from float64 by less than `1e-4` absolute. `benchmark.csv` contains completed
rows plus any explicitly recorded OOM rows, and `summary.txt` names the
highest-throughput completed configuration together with its peak VRAM.

The precision study is the physics gate. The local CPU smoke tests do not
decide it.

## Failure handling

- `twpa_solver.__file__` points outside the checked-out `src` tree: stop,
  correct the editable install, and rerun. Do not work around a shadowed
  package.
- `jax_device='gpu' requested but jax reports no accelerator`: the CUDA JAX
  installation or driver is not visible. Check `nvidia-smi`, reinstall the
  CUDA 12 JAX extra, and rerun preflight.
- Fixture/live `natural_bandwidth` mismatch: stop. The fixture is tied to the
  live 6096-node build (`4558`); do not regenerate it on a legacy 6136-node
  checkout.
- Float64/sequential equivalence exceeds `1e-11`: stop the session. The GPU
  result is not validated against the CPU oracle, so downstream measurements
  are void.
- Precision returns `NO_GO`: stop. The RTX 4060 float32 path does not meet the
  `1e-4` observable tolerance; do not reinterpret it as a GPU speedup.
- A benchmark row is `OOM`: the matrix continues and records the failure.
  Keep the completed rows, and use the winning completed configuration in
  `summary.txt`; reduce batch size only for a separately requested rerun.
- The benchmark has no completed rows or the driver exits non-zero: preserve
  the timestamped directory, inspect the stage JSON/CSV error, and resume with
  the same `-OutputDir` after fixing the environment.

Bring back the complete timestamped directory and the output of
`git rev-parse HEAD` with the measured results.
