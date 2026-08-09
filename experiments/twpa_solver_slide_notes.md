# TWPA Solver Map Flow

These notes describe the practical solver path used by
`experiments/exp10_full_ipm_pump_map_warmstart.py` and the core implementation in
`src/twpa_solver`.

## Big Picture

For a pump-power by pump-frequency map, every grid cell is:

1. Convert external pump power to a peak current.
2. Solve the nonlinear pump harmonic-balance problem.
3. If the pump converged, solve the small-signal Floquet gain problem.
4. Store one row in `map_points.csv`.
5. Store map arrays in `map_arrays.npz`.
6. Optionally store a per-cell signal spectrum in `map_spectrum.npz`.
7. Optionally fit spectra and select best candidates.

The warm-start map does the same physics as the cold map. It only changes the
initial guess strategy and some failure handling.

## Map Traversal

### Grid construction

- `--n-power`, `--pump-power-min-dbm`, `--pump-power-max-dbm` define the power axis.
- `--n-frequency`, `--pump-freq-min-ghz`, `--pump-freq-max-ghz` define the pump frequency axis.
- `--attenuation-db` and `--z0-ohm` convert external dBm into physical peak pump current:
  `I_peak = sqrt(2 * P_W / Z0)`.
- `--pump-current-jc-scale` multiplies that current before injection. Default is `2.0`,
  matching the JosephsonCircuits positive-phasor source convention.

### Cold pass

Each cell is solved independently:

1. Start from zero or a generated seed.
2. Use fixed continuation in pump source scale.
3. Run signal gain after pump convergence.

Cold is slow but useful as a reference.

### Warm-start pass

The map is grouped by pump-frequency column. Within each column, power increases.

1. First point in a frequency column gets a seed solve.
2. Later points reuse the previous converged pump state.
3. A warm-started point usually runs one direct full-scale Newton solve.
4. If enabled, failed points can be retried from a fresh seed or skipped past the fold.

This works because adjacent power points in the same frequency column normally
have nearby pump solutions until the harmonic-balance fold is reached.

## Per-Cell Pump Solve

The pump solve lives mostly in:

- `twpa_solver.pump.problem`
- `twpa_solver.pump.solver`
- `twpa_solver.pump.basis`
- `twpa_solver.pump.backends.*`

### Pump problem setup

For one map point:

1. Load circuit matrices: `C`, `G`, `K`, `Bphi`, ports, branch data.
2. Resolve pump basis:
   - Default JTWPA map basis is `positive_odd_jc`.
   - With `--pump-mode-count 10`, modes are `[1, 3, ..., 19]`.
   - `--nt` is the time grid used for AFT/nonlinear evaluation.
3. Build a harmonic grid at pump angular frequency `omega_p`.
4. Build the nonlinear branch model using Josephson critical currents and `phi0`.
5. Inject pump current at `--pump-port`.

The real pump waveform follows the JC-compatible convention:

```text
psi_pump(t) = 2 * Re sum_k X_k exp(+i k omega_p t)
```

### Newton-Krylov solve

The unknown is the complex pump phasor array `X`, indexed by pump mode and node.

For each Newton step:

1. Evaluate the harmonic-balance residual.
2. Build the current tangent state.
3. Assemble or reuse a preconditioner.
4. Solve the Newton linear system with GMRES using a Jacobian-vector product.
5. Apply a backtracking line search.
6. Accept the step only if the residual decreases.
7. Stop when `coeff_rel < --newton-tol`.

Failure exits include:

- GMRES not converging.
- Line search failure.
- Max Newton iterations.
- Stall detection near the fold.
- Optional wall-time deadline.

### Pump continuation modes

Cold or seeded points use continuation in source scale `lambda`:

- Fixed continuation: solve `lambda = 1/N, 2/N, ..., 1`.
- Adaptive continuation: try larger source steps, shrink after failures, and fall back to fixed continuation if needed.

Warm-started points usually skip continuation and run a direct solve at
`lambda = 1` from the previous cell's solution.

## Pump Backends And Preconditioners

### `--inproc-pump-backend full`

Solves the full node system directly. This is the legacy in-process path.

### `--inproc-pump-backend schur_cpu_mt`

Eliminates linear internal nodes with a Schur complement and solves only the
retained nonlinear/port system. The full pump state is reconstructed before
writing `pump_solution.npz`, so the signal solve sees the normal full-node data.

Use this for large maps near the fold. The Schur partition is cached per pump
frequency, bounded by `--inproc-schur-cache-size`.

### `--inproc-preconditioner mean_tangent`

Cheap block-diagonal preconditioner based on the mean tangent. Usually best for
small warm-start steps.

### `--inproc-preconditioner linear`

Uses the linearized circuit preconditioner. Simpler, less nonlinear information.

### `--inproc-preconditioner spectral_coupled`

Builds a mode-coupled complex preconditioner. More accurate than block-diagonal,
but more expensive.

### `--inproc-preconditioner real_coupled`

Builds the exact real-packed coupled preconditioner, including conjugate coupling.
It can cut GMRES iterations strongly, but the LU factorization is expensive.

### `--inproc-preconditioner real_coupled_fast`

Fast exact coupled preconditioner for the Schur-reduced backend. Reuses assembly
and symbolic factorization where possible.

## Per-Cell Signal Solve

The signal solve lives mostly in:

- `twpa_solver.signal.gamma`
- `twpa_solver.signal.floquet`
- `twpa_solver.signal.gain`

After a pump converges:

1. Load `pump_solution.npz`.
2. Reconstruct pump branch flux versus time.
3. Compute the time-periodic tangent:
   `gamma(t) = cos(psi_pump / phi0) * Ic / phi0`.
4. Fourier-transform it into `gamma_hat[ell]`.
5. Build sparse stiffness harmonics `khat[ell] = Bphi diag(gamma_hat[ell]) Bphi.T`.
6. Assemble the Floquet conversion matrix over sidebands `m = -M ... +M`.
7. Inject a small signal at `--source-port`.
8. Solve the linear system.
9. Extract output voltage at `--out-port`.
10. Convert to S-parameter gain in dB.

The main signal frequency is either:

- fixed by `--signal-ghz`, or
- trailing the pump: `signal = pump_frequency - --signal-detuning-mhz`.

## Signal Spectrum

With `--signal-spectrum`, each map cell solves multiple signal frequencies around
the pump frequency instead of only the trailing point.

Important flags:

- `--signal-offset-start-mhz`: first absolute offset from pump.
- `--signal-offset-step-mhz`: spacing between offsets.
- `--signal-offset-count-per-side`: number of positive and negative offsets.
- `--signal-workers`: thread count over spectrum points.

The solver reuses a signal-frequency-independent `khat` conversion base, so each
extra spectrum point is cheaper than rebuilding the full Floquet structure.

The output is `map_spectrum.npz`, containing the per-cell gain spectrum.

## Candidate Extraction

The plotting/candidate helpers load:

- `map_points.csv`
- `map_arrays.npz`
- `map_spectrum.npz`

For each valid point, the spectrum is:

1. Sorted and filtered to finite samples.
2. Smoothed with Savitzky-Golay.
3. Fit with a cubic spline.
4. Evaluated on a dense frequency grid.
5. Reduced to metrics:
   - fitted peak gain,
   - fitted peak signal frequency,
   - operation bandwidth around peak minus `operation_drop_db`,
   - gain-bandwidth product,
   - ripple inside the operation band,
   - smoothness/curvature,
   - combined score.

Candidate labels include:

- `best_peak_gain`
- `best_gbp`
- `best_ripple`
- `best_smoothness`
- `best_score`
- `rank_001`, `rank_002`, ...

Tables are written under `tables/` as `point_fit_metrics.*` and
`selected_candidates.*`.

## Important Map Flags

### Execution

- `--mode cold`: run only independent cold solves.
- `--mode warmstart`: run only warm-start map.
- `--mode both`: run cold and warm, then compare them with the gate.
- `--executor subprocess`: each pump/gain point calls `exp08` and `exp09` as subprocesses.
- `--executor inprocess`: load matrices once and run pump/gain in one Python process.

### Warm-start and fold handling

- `--inproc-fail-fast`: on a failed warm point, do not run the expensive reseed/adaptive recovery. Continue using the last converged solution as the reference. Useful for high-power maps that cross the pump fold.
- `--fold-skip-patience N`: after `N` consecutive failures in one increasing-power column, mark the rest of that column as `SKIP_PAST_FOLD` without solving.
- `--inproc-fold-predictor secant`: predict the next pump state from the last two converged power points. If it fails, retry from the plain previous-state warm start.

### Pump solve limits

- `--inproc-gmres-maxiter`: bounds GMRES work per Newton step in the in-process path.
- `--inproc-max-newton`: max Newton iterations per pump solve.
- `--inproc-solve-deadline-s`: optional wall-time limit per pump solve.
- `--newton-tol`: convergence tolerance on harmonic-balance residual.
- `--continuation-steps`: fixed continuation steps for cold/seeded solves.
- `--adaptive-initial-step`: first adaptive source-scale step.
- `--adaptive-min-step`: smallest adaptive source-scale step before fallback.

### Pump basis

- `--pump-mode-policy positive_odd_jc`: default JC-compatible odd positive phasor modes for unbiased 4WM JTWPA-style designs.
- `--pump-mode-count K`: for `positive_odd_jc`, use `[1, 3, ..., 2K-1]`.
- `--harmonics H`: dense `[1..H]` fallback when no mode count is supplied.
- `--nt`: AFT time grid size. Must be large enough for the highest pump mode.

### Signal solve

- `--sidebands M`: solve Floquet sidebands `[-M, ..., +M]`.
- `--gamma-nt`: time grid used for Fourier coefficients of the pump-induced tangent.
- `--signal-ghz`: fixed signal frequency.
- `--signal-detuning-mhz`: if `--signal-ghz` is absent, use `signal = pump - detuning`.
- `--signal-backend direct`: full Floquet system.
- `--signal-backend schur`: Schur-reduced signal system.
- `--signal-solver superlu|pardiso`: sparse linear solver.
- `--skip-baselines`: skip off/pumpdiag baseline solves for the Schur signal backend. `gain_db` remains valid.

### Gate and validation

- `--gate-gain-db`: allowed gain drift between cold and warm for compared points.
- `--gate-min-converged-frac`: required warm convergence fraction.
- `--gate-spotcheck N`: in warmstart mode, recompute selected points cold and include their drift in the gate.

## Output Artifacts

Per map:

- `map_points.csv`: one row per point/pass with statuses, gains, residuals, timings, and pump directory.
- `map_arrays.npz`: power/frequency axes and gain grids.
- `map_spectrum.npz`: optional spectrum cube from `--signal-spectrum`.
- `map_summary.json`: machine-readable run summary and gate result.
- `map_summary.md`: short human-readable run summary.

Per point:

- `pump/pump_solution.npz`: solved pump phasors and basis metadata.
- `pump/pump_report.json`: pump residuals, timings, convergence status.
- `gain/gain_report.json`: signal gain result and linear residuals.

## Slide-Sized Summary

```text
map grid
  -> for each pump frequency column
    -> for each increasing pump power
      -> pump solve
        -> choose pump basis
        -> build HB residual
        -> Newton iterations
          -> GMRES JVP solve
          -> preconditioner
          -> line search
        -> write pump_solution
      -> signal solve
        -> gamma(t) from pump
        -> gamma_hat and khat harmonics
        -> Floquet sideband matrix
        -> sparse linear solve
        -> S-parameter gain
      -> optional signal spectrum
      -> append point row
  -> write map arrays and summary
  -> fit spectra
  -> select best candidates
```
