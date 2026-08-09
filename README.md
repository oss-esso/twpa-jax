# twpa-solver

Python tools for building and simulating Josephson-junction travelling-wave
amplifiers, IPM devices, and related circuits.

This repository is intended to be used through the commands in
[`docs/workflows.md`](docs/workflows.md). Fab users normally provide a design
YAML file and select a workflow. The workflow creates the circuit files,
passive response, gain data, and plots.

## Installation

Requirements:

- Windows, Linux, or macOS;
- Python 3.10 or newer;
- a writable project directory and output directory.

From the repository root, create or activate a virtual environment and install
the package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For faster sparse matrix factorisation, install the optional accelerator:

```powershell
python -m pip install -e ".[fast]"
```

The accelerator is optional. The standard installation is sufficient for
small designs and functional checks.

After installation, use [`docs/workflows.md`](docs/workflows.md) as the fab
team operating guide and [`docs/design_format.md`](docs/design_format.md) as
the YAML reference.

## Current limitations

The following cases are not production-ready on the `main` package surface:

- high-pump-power or high-current operation close to a fold or instability;
- general kinetic-inductance device models;
- the saturation and compression regime;
- production claims based on an unconverged or solver-boundary point;
- automatic conversion of YAML into a foundry mask or fabrication file.

Use the passive response and low-to-moderate pump gain workflows first. A
failed high-power point must be reported as a solver or operating-boundary
result, not as a fabricated-device prediction.
