# twpa_jax experiments — agent notes

## Pump-mode-policy layer (exp08/exp09/pump_basis.py)

The Python harmonic-balance pump solve reconstructs the **real** pump waveform
with the JosephsonCircuits.jl (JC) positive-phasor convention:

    psi_pump(t) = 2 * Re sum_{k in modes} X_k * exp(+i k omega_p t)

`experiments/pump_basis.py` is the single source of truth for the pump-mode
basis. Both `exp08_full_ipm_pump_solve.py` (pump solve) and
`exp09_full_ipm_gain_from_pump.py` (gain) import it.

### Why this exists
JC's nonlinear pump for an unbiased 4WM device uses the **odd** mode list
`[1,3,5,...,2K-1]` (K = `Nmodulationharmonics`), e.g. `[1,3,...,19]` for the
JTWPA. The legacy Python code hardcoded dense harmonics `[1,2,...,H]`, which
truncated the high odd pump content and left a ~0.89 dB JTWPA gain mismatch vs
JC. Using the JC odd basis fixes it: **JTWPA gain RMS dropped to ~0.0006 dB.**

### CLI (exp08)
- `--pump-mode-policy dense_real|positive_odd_jc|positive_phasor_explicit|auto_jc`
  (default `dense_real`, which preserves legacy `[1..H]` behavior).
- `--pump-mode-count K` — for `positive_odd_jc` -> `[1,3,...,2K-1]`.
- `--pump-modes 1,3,5,...` — explicit modes for `positive_phasor_explicit`.
- `--promote-from-pump-dir PATH` — warm-start a richer basis from an existing
  lower-basis pump solution (shared modes copied, new modes zero-filled).
  Warm-start runs a single full-scale Newton solve (no continuation).
- `--nt` must be `>= 2*max(mode)+1` (JC uses Nt=40 for max mode 19).

### Metadata persisted (pump_report.json metadata + pump_solution.npz)
`pump_modes`, `pump_basis="positive_phasor"`, `real_reconstruction_factor=2`,
`omega_p`, `phase_convention="exp_plus_i_k_omega_t"`, `pump_mode_policy`,
`pump_source_mode`. `pump_solution.npz` stores `pump_modes` (and legacy
`harmonics`). exp09 reads these via `pump_basis.load_pump_basis_from_solution`.

### exp09 diagnostics
`gamma_hat_summary.csv` — per-ell branch spectrum of
`gamma(t)=cos(psi_p/phi0)*Ic/phi0`:
`ell,nbranches,l2_abs,l2_abs_over_zero_l2,max_abs,mean_abs,mean_real,mean_imag,conj_symmetry_rel_err`.
For a correct real pump, `conj_symmetry_rel_err == 0` (gamma_hat[-ell] =
conj(gamma_hat[ell])).

### Policy selection per design family
- Unbiased 4WM (JPA, JTWPA, FQJTWPA): `positive_odd_jc`, K = `Nmodulationharmonics`.
- Biased / DC / 3WM (FXJPA): symmetry broken -> use **`dense_real`** (all-mode
  phasor basis) + `--dc-solution`.
- Complex/lossy (FQJTWPA_diss): complex C **just works** — physical node fluxes
  stay real, loss only makes D(omega) complex. Use `positive_odd_jc` + complex
  matrices (loads automatically). (Gain currently ~0.9 dB off near threshold; JC
  lossy-pump convention still to reconcile.)
- Multi-pump (DPJPA): needs true 2D-lattice HB -> use the standalone
  `exp14_dpjpa_multitone.py` (modes are (k1,k2) tuples). `auto_jc` in exp08 still
  raises for multi-pump (scalar policy can't represent it).
- DC + mutual-inductor distributed (FXJTWPA): pump branch **folds**; plain
  continuation can't cross it. Use `--preconditioner real_coupled` (exact). Full
  scale-2 needs a JC frequency-domain warm-start seed (open).

### Preconditioners (exp08 `--preconditioner`)
- `mean_tangent` (default), `linear`, `none` — block-diagonal.
- `spectral_coupled` — assembles the mode-coupled (k-q) complex Jacobian, one LU.
- `real_coupled` — exact full real-packed Jacobian incl. the conjugate (k+q) term;
  GMRES converges in ~1 iteration. Use for stiff DC/mutual designs.

### 7-design parity status (outputs/exp14_seven_design_summary/)
5/7 MATCHED < 0.0024 dB: jpa, jtwpa, fqjtwpa, fxjpa, dpjpa. fqjtwpa_diss SOLVED
~0.89 dB (lossy convention). fxjtwpa FINITE_NONCONVERGED (fold). JC reference
curves: `outputs/exp14_jc_refs/` via `exp14_jc_doc_curve_dump.jl` (generic; splats
each case `hbsolve_kwargs` so DC/3WM/4WM work).

### Reproduce parity
JC reference curves: `outputs/exp13_compare/jc_jpa_curve.csv`,
`outputs/exp13_jtwpa_fast_scale2/jc_jtwpa_curve_21pt.csv`. "Pump scale 2" means
pump source current = 2 x the design's AC pump current. Runs land under
`outputs/exp14_*`; build the table with
`python experiments/exp14_seven_design_summary.py`.

### Tests
`tests/test_pump_basis.py` (run with `--basetemp` off the repo to dodge a
Windows ACL issue on `.pytest_tmp`).
