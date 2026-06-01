# TWPA Full

Research code for TWPA/KITWPA linear modeling, harmonic-balance experiments,
gain-map workflows, synthetic calibration, and parameter recovery.

## Developer Quickstart

Use the active Python interpreter, not the global `pytest` launcher:

```powershell
python -m pip install -e .
python -m pytest
```

Direct script execution is supported from the repository root:

```powershell
python scripts/run_linear_validation.py --help
```

Pump, gain-map, wideband compression, and nonlinear recovery workflows default
to package-native harmonic-balance paths. Legacy gain-script, coupled-mode, and
analytic-surrogate routes remain explicit `PARTIAL` compatibility diagnostics.

Dense nonlinear HB is restricted to small reference problems. Read
[`docs/RESOURCE_SAFETY_AND_ACCELERATION.md`](docs/RESOURCE_SAFETY_AND_ACCELERATION.md)
before increasing ladder sizes.
