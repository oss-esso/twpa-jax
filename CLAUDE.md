# twpa_jax — agent notes

## Design-independent saturation validation status (2026-07-29)

Do not use JosephsonCircuits.jl or the Themis measurement cube as physical
references for finite-signal saturation. JC is regression-only; Themis has no
signal-power axis. The CSV field `compression_model_depletion_only` is the
depletion model's dB power gain. The underlying
`depletion_only_model` API still returns linear gain.

The design-independent campaign is incomplete. The existing JVP and lossless
observable tests are numerical/unit checks, not external physical validation;
the current lossless Manley–Rowe test is degenerate because it has no generated
signal/idler photon-flux scale. Do not report the eight-check campaign as
passed until the finite-signal lossless fixture and the requested measured
slopes/figures have been run.

The driver now uses voltage-per-input-current ratios for `gain_vs_off_db`,
matching Floquet normalization; absolute one-port `S` values are affine and
must not be subtracted to form gain-vs-off. The sideband basis builder also
allocates `omega_max` as a function of requested sideband count, so production
S=10 bases are not clipped. A serial JPA lattice-Q=3 measurement found sector
slopes near `1.01, 2.02, 3.03`, while its gain curve was non-monotone. JPA JVP
finite differences gave slope `1.9414` and minimum error `1.566e-12`. The
production JTWPA S=10 run exceeded 300 s without an artifact; S=10/S=12 basis
self-convergence remains unevaluated. Manley–Rowe remains unresolved.
The Manley–Rowe output now includes its photon-flux scale and marks
sub-`1e-28` cancellation-dominated points as not evaluable instead of turning
the denominator floor into a physical error.

Package `twpa_solver` (under `src/`) is the production solver; `scripts/run_gain_map.py`
is the pump/gain-map orchestrator. The solver was extracted from the `experiments/`
research scripts (exp08 pump solve, exp09 gain, exp10 maps, exp14 parity), which
now serve as validation provenance — most notes below apply to the solver modules.

Run the complete test suite, including compression and distributed multitone
physics gates, with a temporary directory outside the repository:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp D:\tmp\twpa_full_slow --run-slow
```

## Production multitone compression campaigns (exp20-exp22)

### Phase 5 basis-convergence status (2026-07-29)

`scripts/multitone_convergence_study.py` now accepts `--device jtwpa|fqjtwpa` and
uses the Phase 3 bracketed nonlinear P1dB refinement for every setting. The
required matrix includes Q=1/2/3, pump-order, torus-scale, three-tone versus
lattice, and odd-only versus dense pump modes.

Two earlier blockers are fixed: the study hardcoded the plain `real_coupled`
preconditioner (a fresh `splu` every Newton step) and buffered every result to
write the CSV once at the end, so an overrun produced no output at all. The
multitone solves now use `real_coupled_fast` (pump solve stays `real_coupled`,
production-identical): measured 8.53 s -> 3.51 s on one jpa setting, P1dB
bit-identical at -116.179752. The CSV is written per setting, and
`--per-setting-budget-s` records `TIMEOUT`/`FAILED`/`OOM` rows instead of
stalling or losing the matrix.

jtwpa matrix, 8 of 9 settings (`outputs/phase5_convergence/jtwpa_full.csv`,
banded backend, 15 coarse points, refined P1dB):

| knob | top two settings | \|ΔP1dB\| | verdict |
| --- | --- | ---: | --- |
| signal_order_max Q | Q=2 -110.500076, Q=3 -110.440913 | **0.059163 dB** | PASS |
| torus_scale | sc=1 -110.440913, sc=2 -110.440913 | **0.000000 dB** | PASS |
| pump_order | order=1 -108.784341, order=3 -110.440913 | 1.656572 dB | **not evaluable** |
| odd vs dense | dense order-5 did not run | — | **not evaluated** |

`torus_scale` is converged exactly -- doubling `n_p`/`n_delta` moves nothing.
The `pump_order` number spans a **gap**: `odd` order 2 (modes `[1,3]`) fails its
pump solve reproducibly in ~1.4 s across two independent runs, while `[1]` and
`[1,3,5]` both converge, so 1-vs-3 are not adjacent settings and 1.66 dB is not
a convergence measurement. The dense order-5 point (the plan's **mandatory**
odd-vs-dense comparison) exhausted memory on this 7 GB machine. Consistency
check that did pass: `three_tone` and `lattice` Q=1/order=1 agree to all six
reported decimals (-112.748281), as they must, being the same basis.

**This does not discharge the production-basis caveat.** Every setting in this
matrix solves at 3.756-7.894 dB small-signal gain against production's 27.541 dB,
and the study builds `build_lattice_basis` where production uses
`build_sideband_matched_basis` at S=10. It converges the lattice family at a much
weaker operating point. The exp20/21 production basis still must not be described
as having passed the 0.2 dB gate.

`scripts/run_compression.py` defaults to the pump-harmonic-retaining
`--multitone-basis matched --multitone-sidebands 2`. A three-tone basis is only
valid with a fundamental-only pump basis; the driver raises if any pump `(h,0)`
mode would be silently dropped. Signal frequency is mandatory for a single run.
Fixture runs default to zero line attenuation; loaded production circuits use
the Themis loss model unless `--attenuation-db` is explicit.

Reproduce the four-device compression curves and S=2/S=4 P1dB check:

```powershell
python experiments/exp20_multitone_compression.py --output-dir outputs/exp20_multitone_compression_converged
python experiments/exp20_summary.py --input-dir outputs/exp20_multitone_compression_converged --output-dir outputs/exp20_summary_converged
```

Run frequency-resolved P1dB and spatial attribution campaigns:

```powershell
python experiments/exp21_p1db_vs_frequency.py --output-dir outputs/exp21_p1db_vs_frequency_converged --signal-workers 4
python experiments/exp22_spatial_attribution.py --output-dir outputs/exp22_spatial_attribution_converged
```

### P1dB refinement versus interpolation (measured 2026-07-29)

`--p1db-power-tol-db` (default 0.1) locates P1dB by nonlinear solves inside the
coarse bracket; 0 falls back to log-linear interpolation. The driver now emits
**both** numbers from one sweep (`p1db` and `p1db_interpolated_dbm`), so the
delta is single-variable. Validation: each run's interpolated value reproduces
the published exp20 number to 5.1e-9 dB (jtwpa) and 6.0e-9 dB (2c).

| device | published exp20 | refined | delta |
| --- | ---: | ---: | ---: |
| jtwpa 6.6 GHz | -111.458017 | -111.118089 | +0.339928 dB |
| 2c 7.440816 GHz | -95.083826 | -94.857885 | +0.225941 dB |

This **supersedes an earlier +0.461 / +2.920 dB claim**, which was measured on a
nine-point coarse grid that no published number used; that grid put 2c's first
crossing 8.4 dB off, making its "+2.920" a grid artifact. The real errors are
0.23-0.34 dB: above the 0.2 dB scale, single-signed (refined is always higher,
so devices compress later than published), and *not* grounds for re-running
exp20/21 wholesale. See `docs/development/saturation_solver_p1db_measurement.md`.

Two diagnostics emitted alongside are **not yet trustworthy**:
`manley_rowe_rel_err` peaks at the *smallest* signal power (0.533 jtwpa, 0.500
2c -- near-identical across very different devices, so a fixed factor rather
than physics). The depletion-model CSV unit defect is fixed: it now emits the
plan model's power gain in dB and exp22 consumes that value without a second
conversion.

### Finite-signal stability (measured 2026-07-29)

Runs without `--check-stability` keep `stability_status = "NOT_CHECKED"`;
deep-saturation solutions are not stability claims.

The previously reported "stable, dominant exponent +2.87e-6 s^-1" is
**withdrawn**. `multitone/stability.py` was passing pump-harmonic keys where
`assemble_conversion_matrix` expects a sideband ladder, and its sigma_min was
owned by a near-DC sideband: measured bit-identical with the pump on and off up
to psi/phi0 = 57000. The committed tests could not catch it because they ran on
`Ic=0` -- a linear circuit -- at a zero state. Both defects are fixed and
mutation-verified; the near-DC case now returns INCONCLUSIVE with a reason
rather than STABLE. Production points are clear of the guard (closest sideband
0.52 GHz for jtwpa, 0.10 GHz for 2c, against ~7 MHz thresholds).

Measured at the three exp22 operating points, all exponents negative (decaying),
now state-dependent as they must be:

| device | zero signal | P1dB | deepest saturation |
| --- | ---: | ---: | ---: |
| jtwpa | -3.3785e+08 (STABLE) | -1.7638e+08 (INCONCL) | -1.1316e+08 (INCONCL) |
| 2c | -1.1034e-02 (STABLE) | -1.6045e-02 (STABLE) | -1.8755e-02 (STABLE) |

**No unstable point was found.** Always quote these against omega_p: jtwpa's
|sigma|/omega_p ~ 8e-3 is real damping, 2c's ~4e-13 is numerically marginal. A
bare exponent is not interpretable. jtwpa's INCONCLUSIVE points are
`refine_complex_resonance` not converging, not the near-DC guard. Details in
`docs/development/saturation_solver_stability_measurement.md`.

Final exp20 evidence (2026-07-29):

| device | S | gain dB | gain - JC dB | input P1dB dBm | depletion at P1dB dB | recovery rungs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| jpa | 2 | 13.305144 | +0.002813 | -132.732343 | -3.368085 | adaptive 1, previous 24 |
| jtwpa | 10 | 27.541036 | +0.000836 | -111.458017 | -0.351776 | adaptive 1, previous 20, rescaled 2, substep 2 |
| fqjtwpa | 6 | 28.534877 | -0.001823 | -107.386000 | -0.598642 | adaptive 1, previous 23, substep 1 |
| 2c | 10 | 15.916611 | no JC reference | -95.083826 | -0.622737 | adaptive 1, previous 24 |

Final exp21 directories are under
`outputs/exp21_p1db_vs_frequency_converged/{jtwpa,fqjtwpa,2c}`. JTWPA reached
P1dB at 8/10 frequencies (minimum -112.280936 dBm at 6.888889 GHz), FQJTWPA at
8/10 (minimum -107.881062 dBm at 7.533333 GHz), and 2c at 10/10 (minimum
-94.011690 dBm at 7.422222 GHz). `NO_GAIN_AT_OPERATING_POINT` is retained at
the dead-band frequencies rather than reporting a P1dB.

Final exp22 spatial artifacts are under
`outputs/exp22_spatial_attribution_converged`: JTWPA contains 6,141 rows
(2,047 branches x three operating points) and 2c contains 7,524 rows (2,508 x
three). The 2c nonlinear branches are monotone but segmented by five intentional
node-number gaps; spatial mapping validates order, not literal node adjacency.

### Multitone preconditioner backend (measured 2026-07-28)

`--multitone-preconditioner real_coupled_fast` (the default) now really is fast:
it routes to `pump.backends.fast_coupled.FastCoupledPreconditioner`, which caches
the scatter map + symbolic factorization and reruns only the numeric factor per
Newton step. It previously forwarded to the plain
`assemble_real_coupled_preconditioner`, i.e. a fresh `spla.splu` every step.
Both the Schur backend (`multitone/schur.py`, cached on the partition) and the
full backend (`multitone/problem.py`, cached on the `cache` field that survives
`dataclasses.replace`) use it. Identical matrix, verified to 1.0e-11 relative.

jtwpa S=10, per Newton step, single process:

| path | before | after |
| --- | --- | --- |
| full backend (`--fixture`) | 6.148 s | 0.93 s (6.6x) |
| schur backend (`--circuit-dir`) | 5.004 s | 1.00 s (5.0x) |

`pypardiso` is present (0.4.7) and is the default factor backend. SuperLU still
works via the automatic fallback but is a bad trade at this size — measured on
jtwpa S=10: refactor 3.94 s vs 0.92 s (4.3x slower, because `spla.splu` has no
symbolic-reuse API and this problem refactors every Newton step) for only 7% less
peak RSS (2.81 vs 3.01 GB). Since peak sets the worker count and both give the
same count, PARDISO wins ~3.8x at equal concurrency. Set `TWPA_REQUIRE_PARDISO=1`
to make a silent fallback a hard error during timed campaigns.

The assembly scatter used to be a sparse matrix `W` of shape `(M.nnz, 2*n_ells*
nnzk)` so assembly was one spmv. It carried no information beyond target
indices -- every stored value is +-1 and the source indices are contiguous
per-ell ranges -- and cost ~630 MB at S=10 to hold 94 MB of indices. It is now
four int32 target arrays per (mode, mode) block (`_index_contributions`), which
cut peak RSS 3.04 -> 2.51 GB for +87 ms of assembly (scattered writes replace a
sequential-write spmv). Gate: `tests/test_fast_coupled_assembly.py` compares the
assembled matrix against `problem.real_coupled_matrix`, per quadrant, since a
dropped sign would only show as a slower preconditioner and no physics gate
would catch it.

**Memory scales as (n_pump_modes + 2S + 1)^2**, not as the packed dimension: the
coupled Jacobian is block-dense in tone index. Same jtwpa circuit, same n=2048,
gain-map pump (H=10) vs multitone S=10 (H=31): M.nnz 2.46M -> 23.6M, exactly the
9.6x H^2 ratio, while the dimension grew only 3.1x. The gain map is cheap because
`signal/floquet.py::solve_gain_one` treats the sidebands as a separate *linear*
factor-once system; saturation cannot, since the sidebands act back on the pump.
Per-worker peak: 2.80 GB at S=10, ~1.6 GB at S=6, ~0.9 GB at S=2.

`scripts/run_compression.py --signal-workers` is capped by
`multitone.resources.fast_coupled_footprint` against BOTH `--resource-budget-gb`
and actual free RAM. The old cap was a hardcoded 3.0 GB/worker independent of
basis size, which underestimated S=10 by 38% and OOMed a 15.3 GB machine at
2 workers. Tests: `tests/test_multitone_resources.py` pins the estimate to the
measurement and asserts it never reads below it.

`--precond-reuse N` / `--precond-reuse-refresh-gmres` expose modified-Newton
preconditioning (reuse one factor across N Newton steps; the update is always
taken against the true Jacobian, so the converged solution cannot change).
Default 1 = refactor every step. **Measured 2026-07-29: N>1 is a large net
loss, leave it at 1.** On the jpa compression fixture, N=2 saved 26 of 59
factorizations but cost 841 extra GMRES iterations (243 -> 1084); N=3 saved 29
and cost 986. The exact preconditioner converges GMRES in ~3 iterations, and one
GMRES iteration (jvp + triangular solve, 303 ms at S=6) costs about as much as
one factorization (346 ms) -- so there is no cheap-preconditioner regime to
amortize, and a stale exact factor is far worse than a fresh one. Wall time on
jpa: 3313 ms (N=1) vs 3574 (N=2) vs 3545 (N=3); gain identical to all digits.
The pump solve is pinned to `precond_reuse=1` so its iterate path is unchanged.

### Factor backends and worker count (measured 2026-07-29)

`--factor-backend {pardiso,banded}` (default `pardiso`). `banded` reorders the
coupled Jacobian **node-major** (`node * 2*n_tones + super_block`) and stores the
factors as a LAPACK general band instead of a general sparse LU. Packed
tone-major the matrix spans everything; node-major it collapses onto a band
~3 tone-blocks wide, because the device is a 1-D chain. That is a property of
the circuit, not of an ordering search -- RCM does worse (max bandwidth 147 vs
89 on jtwpa S=2). Bandwidth is measured from the assembled pattern, never
assumed. jtwpa S=10, 3 power points, single process:

| backend | peak RSS | wall | workers in 7 GB | net throughput |
| --- | ---: | ---: | ---: | ---: |
| pardiso | 2.51 GB | 147.7 s | 2 | 1.55x |
| banded | 1.84 GB | 173.7 s | 3 | **1.95x** |

Converged gain agrees to 4.4e-10 dB (27.541036124719 vs 27.541036124280), as it
must -- this only changes the preconditioner. `banded` is worth it only when the
smaller footprint buys another worker; for a single-frequency run (exp22) it is
a pure 1.18x loss, so the default stays `pardiso` and exp21 passes it
explicitly. Worker throughput does not scale linearly -- measured on jtwpa S=2,
1/2/3/4/6 workers give 1.00/1.55/2.30/2.00/2.41x (+-15% run-to-run), so it is
bandwidth-bound and plateaus around 3 workers on this 6-core part.

**PARDISO is pinned to one thread** (`TWPA_PARDISO_THREADS`, default 1) because
MKL intermittently fails reordering with error -3. That pin costs single-run
latency: at S=6 the factor is 340 ms at 1 thread, 228 ms at 2 (1.49x), and
**1374 ms at 6** -- MKL PARDISO at high thread counts on this AMD part is 4x
slower than serial. Raising it only helps when running one worker; with several
workers, extra threads lose to extra workers (3 workers x 1 thread = 5.00 cyc/s
vs 3 x 2 threads = 3.66).

**The sideband-selection justification is void as of 2026-07-29.** Production
bases (jtwpa S=10, fqjtwpa S=6, 2c S=10) were picked by gating small-signal gain
against a JosephsonCircuits.jl reference. JC is another simulator with no
reference of its own, and `jc_jtwpa`/`jc_fqjtwpa` are JC's own documentation
designs, so that gate was circular; the user has retired it. Do not cite the
"0.2 dB JC-reference gate", and do not present JC agreement as validation --
it measures numerical drift between two codes, which is a useful regression
check and nothing more.

What this leaves unresolved: JTWPA gain is **non-monotone in S** (30.7152,
24.2021, 26.5563, 27.5410 dB at S=2,4,6,10). Without an external reference,
S=10 cannot be selected from that sequence by agreement -- it requires a
self-convergence argument (S=10 against S=12/14) that has never been run, and
non-monotonicity means S=10 may still be climbing. 2c never had a JC reference
at all and inherited S=10 by analogy. Same-S multitone/Floquet parity remains a
valid internal check and should still pass.

Until production-basis self-convergence is measured, treat every published P1dB
as carrying an unquantified basis-truncation uncertainty.

The earlier 2c blocker was a settings/wiring mismatch, not a fold or intrinsic
continuation wall. The gain-map pump is injected at port 4 while signal
scattering is 1 -> 2; using port 1 for both gave a promoted-pump residual of
sqrt(2). With `--pump-port 4`, the saved pump residual is 7.97e-12. The S=10
Schur multitone point converged adaptively at lambda=0.25/0.625/1.0 in
82.1/108.7/224.7 s with final coefficient residual 1.45e-14, producing
15.9166 dB small-signal gain at 7.440816 GHz. Production circuit runs default
to `--multitone-backend schur_cpu_mt`; fixtures retain the full backend.

Running without `--run-slow` is not complete validation.

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

`default_loss_model()`/this A10 fit is now named `pump_line_loss_model()` — it
is the PUMP feedline's loss, not a generic loss. The SIGNAL feedline is a
separate, physically distinct line: `signal_line_loss_model()` fits
`docs/development/loss_B1.csv` (`att_dB(f) = 50.0 + 3.3*sqrt(f) + 0.14*f`, RMS
2.80e-5 dB). Before 2026-08-05, `scripts/measured_psat_pipeline.py` subtracted
a fabricated flat `SIGNAL_LINE_LOSS_DB = 72.5` from the Themis cube's signal
axis instead — off by +9.4 to +15.3 dB across the band and erasing a real
5.95 dB tilt (still present in `experiments/exp30_themis_map_and_pump_inference.py`
and `exp45_curves_two_way.py`, both unfixed legacy scripts —
`outputs/presentation/2c_themis_map.png` is NOT on the corrected calibration).
Which physical line gets which model is forced by energy conservation, not a
free fit choice: signal+idler output at compression cannot exceed the pump
(`P_sat + 3 dB <= P_pump`), and only `pump_line_loss_model()` on the pump
feedline satisfies it — `signal_line_loss_model()` there gives P_sat 16.5 dB
*above* the pump, which is impossible. See
`docs/development/psat_comparison_fix_plan.md` Phase 2 and
`scripts/measured_psat_pipeline.py::energy_conservation_gate` (Phase 7), which
now asserts this on every run instead of trusting it.

## Port power convention: Norton, not travelling-wave (resolved 2026-08-05)

`src/twpa_solver/ports.py` is the single source of truth for current<->dBm.
`designs/ipm_2c_fixed`'s `G` matrix has **exactly four nonzeros**, all
`0.02 S = 50 Ω`, one per port — every drive is an ideal current source in
parallel with `G0 = 1/Z0`, i.e. a **Norton** source, not a matched travelling
wave. The load sees `I/2` (peak), so available power is `P = I^2 Z0 / 8`, not
the travelling-wave `I^2 Z0 / 2` used everywhere before this date — an
overstatement of exactly `10*log10(4) = 6.0206 dB`. Confirmed independently by
the solver's own `pump_outgoing_power_w` observable
(`multitone/observables.py::power_balance`): at `I = 7.2311e-6 A`,
`pump_outgoing_power_w` reads -64.857 dBm, matching `I^2 Z0/8` to the last
digit and off by 6.02 dB from `I^2 Z0/2`.

`port_available_power_w(current_a, z0_ohm, convention="norton")` /
`port_current_from_power_a(...)` are the conversion functions;
`convention="legacy_traveling_wave"` reproduces every pre-2026-08-05 published
number bit-for-bit (`LEGACY_TW_OFFSET_DB = 10*log10(4)`). `--power-convention`
(default `norton`) is wired through `scripts/run_compression.py`,
`scripts/run_gain_map.py`, and `src/twpa_solver/loss.py::dbm_to_peak_current_a`;
gain maps lacking a `power_convention` metadata key are `legacy_traveling_wave`
and get a `-6.0206` dB relabel at read time, never a re-solve — gain is a
pump-on/pump-off ratio (`gain_vs_off_db`), invariant under the source-scale
convention, so the fix is a pure relabel of absolute powers.

**This supersedes "Pump-current conversion (validated 2026-07-18)"** — that
entry validated `--pump-current-jc-scale` against JosephsonCircuits.jl, a
factor-of-two check in *current* against another simulator
([[jc-is-not-a-reference]]); it never addressed whether the port termination
itself was Norton or travelling-wave, and `--pump-current-jc-scale`
(`docs/development/pump_current_conversions.tex`) remains a **separate,
orthogonal** knob layered on top of whichever power convention is selected —
do not conflate the two when reading that doc, which now carries a dated
addendum for this fix rather than being rewritten.

**Depletion cross-check (Phase 7,** `measured_psat_pipeline.py::model_depletion_cross_check`**):**
an energy-accounted depletion estimate (`(P_sat+3dB)/P_pump`, assuming
signal+idler output ~ P_sat + 3 dB) agrees with the solver's own
`pump_depletion_all_port_db` field to within 1.2-1.5x across a 7-point
provisional-operating-point sweep — see [[pump-power-norton-6db]] for the
resolved memory entry and current numbers.

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

### Quantum efficiency (`twpa_solver.signal.quantum_efficiency`)
`calc_qe(S, S_noise=None)` / `calc_qe_ideal(S)` are direct ports of
JosephsonCircuits.jl's `calcqe`/`calcqeideal`, vectorized (no separate `!`
in-place variant — numpy's row-sum already gives the cache-efficient behavior
the Julia loop hand-rolled). Both expect `S` in the **photon ladder-operator
basis**, not the classical voltage-ratio S this repo's `solve_gain_one(_schur)`
returns: signal and idler sidebands sit at different frequencies, so converting
requires the Manley-Rowe reweighting `S_ladder[m,n] = S_classical[m,n] *
sqrt(freq[n]/freq[m])` before calling calc_qe (see
`experiments/exp19_calcqe_validation.py::ladder_basis_weights`). Validated on
ipm_2c_fixed/jc_jtwpa/jc_fqjtwpa by building the 2x2 [signal,idler]x[signal,idler]
sub-matrix via two `solve_gain_one_schur` calls (excite signal_m, excite
idler_m); the resulting unitarity check `|S_ss|^2-|S_is|^2==1` (lossless
non-degenerate amp) holds to ~2%/~0.4% for ipm_2c_fixed/jc_fqjtwpa but is off
~47% for jc_jtwpa — expected, not a bug: a real unitarity check needs the full
multi-sideband S (jc_jtwpa solves 10 sidebands), and the 2x2 truncation used
for this compact demo drops power leaking to other sidebands. Tests:
`tests/test_quantum_efficiency.py` (ports the Julia docstring examples exactly).

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
- **Power substep** `--column-power-substep` (`+ --column-power-substep-init-db`
  0.1, `--column-power-substep-min-db` 0.005, `--column-power-substep-deadline-s`
  120). On a failed warm cell, adaptive natural-parameter continuation **along the
  map power axis** from the last converged state: walk up in adaptive dBm
  micro-steps (geometric in current, grow x1.5 / halve on fail), warm-starting
  each; `InProcessEngine.solve_power_substep`. This is the diagnostics'
  "0.005-0.01 dB steps cross the wall" finding operationalized inside the map:
  the coarse 0.30 dB power grid overshoots gain-lobe crests that finer stepping
  crosses. A step-independent stall (step < min_db) is recorded
  (`pump_power_substep_stall_dbm`, sets `verified_fold`) as a real
  numerical/fold boundary rather than retried. **Demonstrated:** 2c themis
  column fp=8.099 GHz, coarse map died at -29.36 dBm (3-4 PASS); substep
  recovered to ~-25.75 dBm (16 PASS, +3.9 dB, gains through the multi-lobe
  ripple) before a genuine boundary. The intra-loop `last_good_X` is
  retained-shape (Schur), so substep solves are shape-compatible with no disk
  round-trip (the disk-load seed path in `--initial-pump-dir` is full-shape and
  is a separate concern). Compare a recovered map to the Themis measurement with
  `scripts/compare_map_to_measurement.py` (peak-gain + collapse-power envelope,
  aligned by the ~+0.99 GHz / few-dB calibration offsets;
  `docs/17.03.10_Themis_SetupAug25_noVTS_transmission_15mK`).
- **Calibration-shift map fit** (`scripts/align_map_to_measurement.py`): instead of
  hand-tuning the calibration offsets, fit them as nuisance parameters on the 2-D
  peak-gain maps. Model `G_meas(f,P) ~= G_sim(f-df, P-dP) + dG`; for weighted LSQ
  `dG` is analytic per `(df,dP)`, so a coarse+fine 2-D grid search over `(df,dP)`
  remains (`align_maps`). Reduces the Themis cube (`105C5_*GHz.npy`: transmission
  over power x signal-freq per pump freq) to a peak-gain map and plots it (the raw
  `docs/14.18.08_Themis_...` ships data only, no plot); resamples the sim with
  `RegularGridInterpolator` at shifted coords, masks non-overlap + NaN (failed) sim
  cells, ROI-weights so the amplified ridge (not the flat background) drives the fit,
  `--loss {l2,huber}`. Writes JSON + a measurement-map PNG + a 4-panel comparison
  (meas / aligned sim / residual / **loss surface**, clipped color so the min basin
  is visible). Fit one section only with `--fit-freq-ghz LO HI` / `--fit-power-dbm
  LO HI` (hard-mask the measurement grid outside the window); `--min-overlap-frac`
  (default 0.25) rejects tiny-overlap corner fits (a soft 1/overlap penalty alone
  lets a ~9-cell corner with near-zero local residual win -- the guard is relative
  to the floor-weighted window, whose max achievable overlap is only ~0.3-0.4
  because the sim fails at high power, so 0.25 is the practical ceiling not 0.5).
  **Demonstrated (14.18.08 vs `map_2c_scan_6p0_8p5_100x70`):** full-map best
  df=-0.30 GHz, dP=+2.55 dB, dG=-1.30 dB, RMS 6.0 dB, df-elongated (weakly
  identified) basin. **Per-section fits are far better identified:** restricting to
  one comb branch gives **df ~= 0 GHz** (the two datasets' combs are already
  frequency-aligned -- no ~+1 GHz offset like 17.03.10; the full-map df=-0.30 was a
  comb-alias compromise since a single df can't align all lobes when the comb phase
  drifts) and **dP ~= +2.5..+3.3 dB robustly across 6.2-7.45 GHz** (the one real
  calibration offset), with a compact single-minimum loss surface. RMS stays ~2-4 dB
  per lobe because the sim still (a) does not reach the measured high-power lobes
  (numerical-boundary cap) and (b) has a slightly different comb periodicity -- a
  model-fidelity/coverage gap, not a fit bug. Band edge 7.45+ GHz has ~no sim
  coverage (junk dG). Tests: `tests/test_align_map.py`.
- **Forced-gain column resume** (`scripts/resume_column_force_gain.py` +
  `InProcessEngine.solve_point(force_gain=True)`): diagnostic to test whether a
  column's high-power wall is the real device fold or a numerical boundary. Marches
  one column (`--column-freq-ghz`, nearest grid col; omit = all columns) up in
  power, warm-starting each cell from the previous, and runs the gain solve on the
  **last Newton iterate regardless of convergence** — the normal path only gains
  converged pumps (`solve_point` gate is `converged or force_gain`; it also returns
  the last-iterate `X` so the warm chain continues past the wall). Never skips;
  stops a column after `--force-max-nonfinite` (default 3) consecutive non-finite
  pump states (warm chain diverged). Writes per-cell dirs, a per-column CSV
  (`pump_converged`, `forced_gain`, `gain_db` columns) and a PNG (gain vs power,
  converged pts vs forced pts). Takes the SAME engine/grid flags as
  `run_gain_map.py`. **Demonstrated:** 2c fp=8.099 GHz — converged branch 9.2→12.3→
  15.7 dB (-32..-30 dBm), then -29 dBm pump FAILS (coeff_rel 0.089) and the forced
  gain **collapses to 1.5 dB**, i.e. the non-converged waveform does not sustain the
  gain (evidence the wall is near a genuine transition, not a mere solver miss). The
  in-loop warm state is retained-shape (Schur), so force-marching through a
  non-converged iterate is shape-compatible with no disk round-trip. Tests:
  `tests/test_run_gain_map_cli.py::test_force_gain_*`.
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
- **Mid-GMRES deadline abort** (`solver.py` `gmres_call`): `solve_deadline_s`
  checks elapsed wall time on every GMRES iteration via `callback_type="pr_norm"`,
  raising `GmresDeadlineExceeded` from inside the callback instead of only
  checking between Newton iterations (the old scheme let one pathological GMRES
  call run ~200s past a 14s budget with `gmres_total` in the thousands).
- **Adaptive-continuation fallback resumes, not restarts** (`solve_adaptive_continuation`):
  when lambda-bisection shrinks below `min_step` (near a genuine fold in source
  scale, reduction ratio stuck near 1.0 at every lambda=1 attempt), it falls back
  to a fixed-step ladder (`solve_continuation`). This used to pass the *original*
  seed and lambda=0, discarding every state the adaptive phase had already
  converged and re-deriving the cheap low-lambda region from scratch -- on a real
  map column (fp=7.329 GHz, -28.25 dBm,
  `outputs/measurement_match_debug_01/column_debug_col3_trim`, debug-logged) this
  burned the whole 14s per-point deadline getting back only to lambda=0.75 after
  the adaptive phase had already reached lambda=0.9375 converged. Fixed:
  `solve_continuation` takes a `lambda_start` (default 0.0, so all other callers
  are unaffected) and the fallback resumes from `(X_current, lambda_current)`,
  sized to the remaining span at the original `1/fallback_fixed_steps`
  granularity (`math.ceil(remaining / step_size)` steps, not the full
  `fallback_fixed_steps`). Column-level `--column-arclength-recovery` (separate
  from this intra-solve fallback) already retries fresh on every failing cell,
  not once per column. Tests: `tests/test_adaptive_continuation_fallback.py`.

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

### Standard gain-map flag set (2026-08-03)

Every production gain map uses this block. Pass it to `scripts/run_gain_map.py`,
or forward it through `workflows/run_gain_map_and_plots.py` (which injects
`--circuit-dir`, `--outdir`, `--executor inprocess` and adds the plot
catalogue). Only the grid bounds and the directories change per run.

```
--executor inprocess --mode warmstart
--inproc-pump-backend schur_cpu_mt --inproc-preconditioner real_coupled_fast
--inproc-fold-predictor secant --inproc-fail-fast --fold-skip-patience 2
--pump-current-jc-scale 1.0
--n-power 20 --n-frequency 20 --frequency-chunk-size 10
--pump-power-min-dbm -26 --pump-power-max-dbm -16
--pump-freq-min-ghz 7.6 --pump-freq-max-ghz 7.85
--signal-detuning-mhz 500 --no-signal-spectrum
--log-level INFO --overwrite
```

Most of these already match the parser default, but **six do not** and omitting
them silently changes results or cost:

| flag | default | reason it is set |
| --- | --- | --- |
| `--inproc-fail-fast` | `False` | stop a bad cell instead of grinding |
| `--fold-skip-patience 2` | `0` | enable skip counting |
| `--signal-detuning-mhz 500` | `100` | wider signal offset |
| `--no-signal-spectrum` | spectrum **on** | the default runs a full signal sweep at *every* grid point and dominates wall time |
| `--log-level INFO` | `DEBUG` | DEBUG logs are enormous on a 400-point map |
| `--overwrite` | `False` | reruns land in the same directory |

Leave `--attenuation-db` unset (measured loss_A10 model applies) and
`--loss-model` at `auto`. `--pump-port` defaults to 4, correct for 2c.

`--pump-current-jc-scale` is kept in the block for explicitness only: 1.0 is now
both the parser default and the validated conversion. Passing `2.0` would be a
6 dB error.

Caveat: `--fold-skip-patience 2` has previously culled the 2c broadband
operating region into needle-only maps when skip counting false-tripped. Check
the PASS coverage of a sparse map before quoting it.

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

## Matrix-tracing driver (`scripts/run_gain_map_column_matrices.py`)

Diagnostic one-column driver that runs `run_gain_map` in-process (forces
`--n-frequency 1`, single `--column-frequency-ghz`) and archives the sparse
matrices flowing through solver frames (`src/twpa_solver/`, `experiments/`,
`run_gain_map.py`), plus the static design matrices. The production engine flags
are baked in (`PRODUCTION_ENGINE_FLAGS`: schur_cpu_mt + real_coupled_fast +
secant + fail-fast + patience 2 + cache 1 + detuning 100 MHz + direct/superlu +
mode-count 10 + nt 40 + no-spectrum); pass extra run_gain_map flags (e.g.
`--log-level INFO`) after `--extra-map-args` (REMAINDER, must be last).

**Capture uses `sys.setprofile` (call/return), NOT `sys.settrace` (per-line) —
do not switch it back.** Line tracing the stiff assembly loops slowed one pump
solve **~40x** (factor 2.2s -> 92s; the hot `assemble_real_coupled_fast`/Schur
assembly lives *inside* a traced module, so scoping settrace to solver files did
not help). `setprofile` scans a target frame's `f_locals` only on `return`
(args + assembled results both present) -> ~4x over untraced (92.8s -> 8.95s for
3 points), archiving matrices that cross function boundaries. Trade-off: a matrix
built and discarded within one function without surviving to its return is not
archived (per-line count 1103 -> boundary count ~42). The callback is **fail-safe**
(catches `Exception` per matrix, logs + skips, reports `matrix_save_errors`) so a
save error never aborts the solve; only `KeyboardInterrupt`/`SystemExit` propagate.
Because the hook is hot, a **Ctrl-C tends to surface inside `dispatch`** -- that is a
real interrupt, not a tracer bug. On Windows the archive paths are deep: use a
**short `--outdir`** to stay under MAX_PATH (a deep default outdir can hit an OSError
that is now skipped rather than fatal). Tests: `tests/test_column_matrices_tracer.py`.

## Tests
`tests/` covers the solver: `test_loss_model.py` (loss fit),
`test_pump_basis.py` (pump-mode basis), `test_fxjtwpa_node_order.py`,
`test_exp10_gate.py` (map gate), `test_column_matrices_tracer.py` (setprofile
matrix tracer). Run with `--basetemp` off the repo to dodge a Windows ACL issue
on `.pytest_tmp`.

## Le Gal benchmark readout contract

The effective-SNAIL benchmark uses a branch law shifted to its solved static
equilibrium. Its 31 fF SNAIL capacitance is stamped across the branch, while
223.5 fF is the ground capacitance; `sqrt(L/Cg)` is about 62.3765 ohm.
`shunt_conductance_s` is an SI conductance, not a loss tangent.

Benchmark HB gain must be read through `multitone.observables.tone_s21` and
reported as the pump-on/pump-off ratio when compression is discussed. Do not
recreate gain with `|i omega X|/V_in`; that old path was biased by 12.041 dB.
`power_balance` accepts `z0_ohm`. The `conversion_manley_rowe_*` fields are
restricted to pump/signal/idler; `all_tone_manley_rowe_*` exposes the retained
tone scope. Legacy names remain for compatibility.

The corrected Level-2 campaign currently measures a passive or weakly
deamplifying effective line at the sampled points; it has not reproduced the
paper's two gain lobes or a finite P1dB. This is a measured negative result,
not evidence against the paper.

## Component profiles and scatter

The IPM builder supports ordered per-cell `Lj` and `Cg` profiles and
independent multiplicative `Lj`, `Cj`, and `Cg` scatter, all on the JTL cell
index (2508 on 2c). See `docs/development/component_profiles_and_scatter.md`
for the shape catalogue, selection rules, Cg boundary-halving mapping, and the
stable RNG stream contract.

There is **no** `Cj` profile: nominal `Cj` is derived from the `Lj` profile so
the plasma frequency stays constant. Independent `Cj` *scatter* intentionally
breaks that, and must not be "corrected" to track `Lj`. Scatter sigma is a
fraction of each cell's own nominal, and the `Lj` stream is bit-identical to
the pre-profile `apply_lj_scatter` at the same seed.

`sine`/`cosine` are the one place `start`/`end` are not the first and last
values — for those two they bound the oscillation envelope, since periodic and
endpoint-anchored are mutually exclusive.

`build_variant_design` re-emits a stored design with a profile applied. Its
topology gate (`assert_source_topology`) must run on a **nominal** rebuild;
gating the profiled netlist against the source would reject every real variant.
`designs/ipm_2c_fixed` passes at 16312/16312 elements, `C/G/K/Bphi` maxdiff
0.0, with `--coupler-mode cached` (`ideal`/`optimize` do not reproduce it).
Other design dirs must be checked before use as variant sources.

Gates: `tests/test_component_profiles.py`, `test_component_scatter.py`,
`test_ipm_role_tags.py`, `test_ipm_component_plan.py`, `test_variant_design.py`
(95 tests, each verified by mutation).
# Dielectric dissipation

On-chip dielectric loss is stored as `C * (1 - 1j*tan_delta)`, so
`Im(C) = -C*tan_delta`. Lossy circuits resolve to `conductance_abs_omega`,
which uses `G + |omega| C tan_delta` and preserves
`D(-omega) = conj(D(omega))`. The `jj_cj` junction-capacitance role is always
lossless; the tangent applies to substrate and coupling capacitors.

`plasma_locked` Cj scatter derives Cj from the Lj factor, preserving each
cell's plasma frequency. Because `conductance_abs_omega` is non-analytic in
complex omega, Tier-1 stability remains available but Tier-2 complex-omega
resonance refinement is unavailable for lossy circuits.
