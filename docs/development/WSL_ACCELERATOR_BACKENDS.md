# Optional WSL2 Accelerator Backends

The supported baseline is Windows or Linux CPU with JAX x64 enabled. Optional
NVIDIA experiments belong in WSL2 or Linux and must remain opt-in.

Use:

```powershell
python -c "from twpa.accelerators import detect_accelerator_capabilities; print(detect_accelerator_capabilities().to_dict())"
```

The capability probe does not import CUDA-Q or cuQuantum. It only reports
whether they are installed.

## Backend Boundaries

- JAX CPU remains the reference implementation for tiny dense-versus-structured checks.
- JAX CUDA may accelerate matrix-free JVP and batched linear algebra after a GPU runner exists.
- cuQuantum is an optional experiment for batched GPU workloads. It does not replace the classical structured HB formulation.
- CUDA-Q is reserved for later quantum-circuit experiments and is not a TWPA HB dependency.
- CPU/GPU parity tests are required before enabling any accelerator by default.

## WSL2 Status

Create and pin a WSL2 environment only on a host with an NVIDIA GPU and a GPU
CI runner. Record CUDA toolkit, driver, JAX CUDA wheel, CUDA-Q, and cuQuantum
versions together. Windows CPU installs must continue to work without them.

