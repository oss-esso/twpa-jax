# GPU session runbook

This runbook is for the RTX 4060 laptop. The repository must be on `dev` at
the commit that contains the committed fixture and session scripts.

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
