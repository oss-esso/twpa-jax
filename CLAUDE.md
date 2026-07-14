# twpa_jax — agent notes

Package `twpa_solver` (under `src/`) is the production solver; `scripts/run_gain_map.py`
is the pump/gain-map orchestrator. The solver was extracted from the `experiments/`
research scripts (exp08 pump solve, exp09 gain, exp10 maps, exp14 parity), which
now serve as validation provenance — most notes below apply to the solver modules.

Module map: pump solve `twpa_solver.pump.hb` + `twpa_solver.pump`
(`HarmonicNewtonKrylovSolver`, `NewtonKrylovSettings`, `FullPumpProblem`); pump
basis `twpa_solver.pump.basis`; gain `twpa_solver.signal`; circuits
`twpa_solver.core`; loss `twpa_solver.loss`.

## Line loss model (`src/twpa_solver/loss.py`)

`run_gain_map.py` converts pump dBm → on-chip peak current after subtracting
line loss. Loss defaults to the measured `docs/loss_A10.csv` fit, not a flat
35 dB:

    att_dB(f) = 27.3882 + 0.4579*sqrt(f) + 0.8354*f    (f in GHz)

`InsertionLossModel` / `default_loss_model()` expose it; `InsertionLossModel.fit_csv`
re-fits the CSV. C = fixed coupling loss, A*sqrt(f) = skin effect, B*f = dielectric
(RMS 0.37 dB). The constant C is required — the CSV has ~26 dB loss at f=0, so a
pure `A*sqrt(f)+B*f` fits terribly (RMS 4.6 dB, B<0). Sanity: model at 8 GHz ≈
35.4 dB, matching the old band-calibrated flat 35 dB.

`--attenuation-db` defaults to `None` (= use the model); pass a float to force a
flat value. Only `run_gain_map.py` is wired to the model; the `experiments/exp10_*`
scripts still use their local flat `dbm_to_peak_current_a`. Tests:
`tests/test_loss_model.py`.

## Pump-mode-policy layer (`twpa_solver.pump.basis`)

The harmonic-balance pump solve (`twpa_solver.pump.hb`, solver
`twpa_solver.pump.HarmonicNewtonKrylovSolver`) reconstructs the **real** pump
waveform with the JosephsonCircuits.jl (JC) positive-phasor convention:

    psi_pump(t) = 2 * Re sum_{k in modes} X_k * exp(+i k omega_p t)

`twpa_solver.pump.basis` (`resolve_pump_basis`, `PumpBasis`) is the single source
of truth for the pump-mode basis. The pump solve and the gain solve
(`twpa_solver.signal`) both consume it. `scripts/run_gain_map.py` drives both.

### Why this exists
JC's nonlinear pump for an unbiased 4WM device uses the **odd** mode list
`[1,3,5,...,2K-1]` (K = `Nmodulationharmonics`), e.g. `[1,3,...,19]` for the
JTWPA. The legacy code hardcoded dense harmonics `[1,2,...,H]`, which truncated
the high odd pump content and left a ~0.89 dB JTWPA gain mismatch vs JC. Using
the JC odd basis fixes it: **JTWPA gain RMS dropped to ~0.0006 dB.**

### Pump-solve knobs (`resolve_pump_basis` / pump-solve CLI)
- `policy`: `dense_real | positive_odd_jc | positive_phasor_explicit | auto_jc`
  (default `dense_real` preserves the legacy `[1..H]` behavior).
- `mode_count K` — for `positive_odd_jc` -> `[1,3,...,2K-1]`
  (`positive_odd_modes`).
- explicit modes `1,3,5,...` — for `positive_phasor_explicit`
  (`parse_explicit_modes`).
- promote-from an existing lower-basis solution
  (`promote_solution_to_basis`): shared modes copied, new modes zero-filled,
  then a single full-scale Newton solve (no continuation).
- `nt` must be `>= 2*max(mode)+1` (JC uses Nt=40 for max mode 19).

### Metadata persisted (pump_report.json metadata + pump_solution.npz)
`pump_modes`, `pump_basis="positive_phasor"`, `real_reconstruction_factor=2`,
`omega_p`, `phase_convention="exp_plus_i_k_omega_t"`, `pump_mode_policy`,
`pump_source_mode` (via `PumpBasis.to_metadata`). `pump_solution.npz` stores
`X_real`/`X_imag` as **float32, `savez_compressed`** (~2.1x smaller than the old
float64/uncompressed 1.5 MB/point — matters at 10k points/map), plus `pump_modes`
(and legacy `harmonics`). The gain solve reloads these via
`twpa_solver.pump.basis.load_pump_basis_from_solution`, which upcasts back to
complex128 (float32 would otherwise leak complex64 into scipy). float32's ~1e-7
relative precision is far below the ~1e-3 dB gain-map tolerance. Recompress legacy
maps in place with `scripts/recompress_pump_solutions.py <dir> --apply` (dry-run by
default, idempotent).

### Gain diagnostics (`twpa_solver.signal`)
`gamma_hat_summary.csv` — per-ell branch spectrum of
`gamma(t)=cos(psi_p/phi0)*Ic/phi0` (`compute_gamma_hat`):
`ell,nbranches,l2_abs,l2_abs_over_zero_l2,max_abs,mean_abs,mean_real,mean_imag,conj_symmetry_rel_err`.
For a correct real pump, `conj_symmetry_rel_err == 0` (gamma_hat[-ell] =
conj(gamma_hat[ell])).

### Policy selection per design family
- Unbiased 4WM (JPA, JTWPA, FQJTWPA): `positive_odd_jc`, K = `Nmodulationharmonics`.
- Biased / DC / 3WM (FXJPA): symmetry broken -> use **`dense_real`** (all-mode
  phasor basis) + a DC solution.
- Complex/lossy (FQJTWPA_diss): complex C **just works** — physical node fluxes
  stay real, loss only makes D(omega) complex. Use `positive_odd_jc` + complex
  matrices (loads automatically). (Gain currently ~0.9 dB off near threshold; JC
  lossy-pump convention still to reconcile.)
- Multi-pump (DPJPA): needs true 2D-lattice HB -> use the standalone
  `exp14_dpjpa_multitone.py` (modes are (k1,k2) tuples). `auto_jc` in exp08 still
  raises for multi-pump (scalar policy can't represent it).
- DC + mutual-inductor distributed (FXJTWPA): **MATCHED (RMS 0.0 dB)** via an
  imported JC pump nodeflux seed. The blocker was never the fold or the stiff
  mutual K (exp10's mutual stamp is algebraically identical to JC's `calcinvLn`,
  doctest in `capindmat.jl`). It was **node ordering**: exp10 inserts nodes per
  cell as (node, node+3, node+2, node+1, node+4) -- unsorted -- while JC orders
  by sorted node number. The identity seed left a real ~45 pump residual on the
  SQUID nodes; the sorted-rank permutation drops it to ~5e-9. Pipeline:
  `exp14_build_jc_warmstart.py` (raw seed) -> `exp14_fxjtwpa_fix_seed.py`
  (applies the node-order permutation to pump X **and** DC node fluxes) ->
  `exp09 --pump-dir outputs/exp14_fxjtwpa_seed_fixed/pump --dc-solution .../dc
  --source-port 1 --out-port 2 --sidebands 4 --signal-m 0 --idler-m -2`.
  Test: `tests/test_fxjtwpa_node_order.py`.

### Preconditioners (`NewtonKrylovSettings.preconditioner`)
- `mean_tangent` (default), `linear`, `none` — block-diagonal.
- `spectral_coupled` — assembles the mode-coupled (k-q) complex Jacobian, one LU.
- `real_coupled` — exact full real-packed Jacobian incl. the conjugate (k+q) term;
  GMRES converges in ~1 iteration. Use for stiff DC/mutual designs.
  `run_gain_map.py`'s in-process engine defaults to `real_coupled`.

## Continuation-method suite (`run_gain_map.py` + `solver.py`)

Opt-in inter-cell traversal / predictor / recovery / fold-policy layers plus
advanced intra-cell continuation, from `docs/reports/pump_map_continuation_methods.tex`
and its expanded test matrix. **Defaults reproduce the legacy `column` pass
byte-for-byte** (regression: `tests/test_traversal.py::test_column_order...`,
existing gate/CLI tests). Everything below is off unless a flag selects it.

- **Traversal** `--traversal {column,backbone,nearest,serpentine,floodfill}`
  (`+ --backbone-direction {ltr,rtl,center_out,two_ended}`). `column` is the
  legacy per-frequency-column pass. The others share one in-process
  `solved[(i,j)]->X` store across BOTH axes, so they **force
  `--frequency-chunk-size 0`** (single process; the Schur cache stays small to
  bound RAM, so a backbone row rebuilds the per-frequency partition as it
  sweeps). Orchestrator: `run_map_traversal` (not `run_warm_pass_inprocess`,
  which stays the `column` path).
- **Predictors** `--predictor {copy,power_secant,freq_secant,corner,plane,portfolio}`
  (`+ --portfolio-policy {best,ranked}`). Pure math in
  `src/twpa_solver/pump/predictors.py`; `portfolio` ranks candidates by
  `problem.norms(X,1)` residual (`engine.residual_norm`). Tests:
  `tests/test_predictors.py`.
- **Recovery** `--recovery {reseed,alt_parent,bridge,ladder}`
  (`+ --bridge-steps`, `--bridge-mode {diagonal,freq_first,power_first,adaptive}`).
  Bridge = physical-parameter continuation from a solved parent to the target
  along (P,f), `InProcessEngine.solve_bridge`.
- **Fold policy** `--fold-policy {patience,cross_axis,bridge_gate,combined,arclength}`
  — when a failed cell counts toward the per-column fold short-circuit; `combined`
  is the report's recommended ladder (power/freq parent + portfolio + bridge before
  counting); `arclength` rounds the fold.
- **Intra-cell** (`solver.py`) `--inproc-continuation {adaptive_secant,adaptive_tangent,affine,ptc}`:
  tangent/Euler predictor (`dR/dlambda=-S`, `source_coeffs(1)`),
  affine-ish step control, and pseudo-transient (`solve_pseudo_transient`).
  Pseudo-arclength (`solve_arclength`, bordering algorithm, modified-Newton) and
  the `fold_power` locator drive `--fold-policy arclength` and `--fold-follow`
  (writes `fold_curve.csv`, no gain map). **Key perf detail:** the advanced linear
  solves use `problem.assemble_real_coupled_preconditioner` (near-direct) via
  `_linear_solver`; the mean-tangent block factors leave GMRES grinding on the
  coupled system. Tests: `tests/test_advanced_continuation.py`.
  arclength/fold-follow are functional but **experimental** on the stiff 2c device
  (fold-follow may report no fold in range; the arclength target endpoint is
  linearly interpolated, so it is used as a warm guess, not a polished root).

The engine's `X` is Schur-reduced (retained-port shape, constant across
frequencies), so chained warm starts and residual ranking all use the same
`engine._make_solve_problem(...)` representation — never the full-node problem.

### Campaign (`scripts/run_campaign.ps1`)
Sequential 2c campaign (`outputs/ipm_python_design`) mirroring the current
production run (`outputs/solver_spectrum_2c_recover_m35_m23_7p5_8p5_50x50_s20_sb10`:
50x50, -35..-23 dBm x 7.5..8.5 GHz, spectrum, sb10). Each config runs
`run_gain_map -> plot_gain_map (--top-k 3, maps + candidate S21 sweeps) ->
prune_map_solutions (--top-k 100 --purge-point-dirs --apply)`. `-DryRun` prints
commands; `-Only id1,id2` runs a subset. ~16 configs, est. ~20-26 h; pruned to
~0.1-0.2 GB/run.

## Validation provenance (experiments/)

The solver's numerics are pinned to JosephsonCircuits.jl by the exp13/exp14
parity runs below. These live in `experiments/` + `outputs/`; they are the
reference the solver was tuned against, not part of the production path.

### 7-design parity status (outputs/exp14_seven_design_summary/)
6/7 MATCHED < 0.0024 dB: jpa, jtwpa, fqjtwpa, fxjpa, dpjpa, **fxjtwpa (RMS 0.0)**.
fqjtwpa_diss SOLVED ~0.89 dB (lossy convention) -- the only remaining mismatch.
JC reference
curves: `outputs/exp14_jc_refs/` via `exp14_jc_doc_curve_dump.jl` (generic; splats
each case `hbsolve_kwargs` so DC/3WM/4WM work).

### Reproduce parity
JC reference curves: `outputs/exp13_compare/jc_jpa_curve.csv`,
`outputs/exp13_jtwpa_fast_scale2/jc_jtwpa_curve_21pt.csv`. "Pump scale 2" means
pump source current = 2 x the design's AC pump current. Runs land under
`outputs/exp14_*`; build the table with
`python experiments/exp14_seven_design_summary.py`.

## Tests
`tests/` covers the solver: `test_loss_model.py` (loss fit),
`test_pump_basis.py` (pump-mode basis), `test_fxjtwpa_node_order.py`,
`test_exp10_gate.py` (map gate). Run with `--basetemp` off the repo to dodge a
Windows ACL issue on `.pytest_tmp`.
