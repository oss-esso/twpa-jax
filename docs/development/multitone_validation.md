# Multitone validation status

The finite-signal multitone path uses positive-frequency coefficients with the
same real reconstruction convention as the pump solver:

```text
x(t) = 2 Re sum_v X_v exp(i theta_v)
```

`REAL_RECONSTRUCTION_FACTOR` in `src/twpa_solver/multitone/basis.py` is the
single source of truth for this factor. `tone_s21` uses the voltage reconstructed
with that factor and retains the paper's current-response normalization. Power
waves are only a cross-check: their port current is the current entering the
network, `I_network = I_Norton - V/Z0`.

## Physics gates

The non-degenerate JPA gain gate uses the measured operating point:

- `build_jpa()` with pump modes `[1, 3, 5]`;
- pump frequency `4.75001 GHz` and default pump current;
- signal frequency `4.80 GHz`;
- Floquet `sidebands=2`, `idler_m=-2`;
- pump-off stiffness `Bphi @ diag(Ic / phi0) @ Bphi.T`.

The measured Floquet gain-vs-off is `15.59126731 dB`; the multitone result is
`15.59126494 dB`, a `2.37e-06 dB` difference. The separate 4.5 GHz test is
explicitly a weak/no-gain limit, not the gain validation point.

Run the focused gates with:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp <outside-repo> \
  tests/test_multitone_physics.py tests/test_multitone_observables.py
```

## Current scope and remaining gaps

The validated implementation covers the JPA reference fixture, finite-signal
continuation, Schur and full-node solves, resource guarding, power balance, and
Floquet-fed seeding. Compression artifacts are written by
`scripts/run_compression.py` and carry `stability_status="NOT_CHECKED"`.

The CLI currently executes the JPA three-tone path. Its `lattice`, signal-frequency
range, worker, and resource-budget options are recorded in the summary but are not
yet connected to independent solve loops or allocation guards. Multi-frequency
campaigning, basis-convergence studies, and dynamic-stability classification remain
follow-up work from `docs/development/saturation_solver_plan.md`.
